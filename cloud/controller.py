"""Run one experiment batch on spot Cloud TPU VMs, from this machine.

Usage: python cloud/controller.py experiments/<batch> [experiments/<batch2> ...] [--accel v6e-8,v6e-4] [--max-chips 16]
                                  [--max-price 0.75] [--tag NAME] [--drain-other-sizes] [--on-demand-chips 8 --on-demand-budget 90]

The batch folder's jobs.py defines JOBS (name, args, expected_seconds), as for the Kaggle batches. Several batches can share one
pool of VMs: their jobs run in the order the batches are given, each batch's results still go to results/<batch>/, and the log
and VM names belong to the first batch (one controller per chip family avoids controllers competing for the same quota, and a VM
that runs out of one batch's jobs takes the next batch's instead of being deleted). With --on-demand-chips N it also keeps up to N
chips of on-demand (never preempted) VMs, until their spend reaches --on-demand-budget; they take the long jobs first
(expected_seconds >= LONG_JOB), which a spot VM takes only when no short job is waiting. The controller:
- creates TPU VMs named mhc-<tag>-<n> (spot; zones tried cheapest first, up to --max-price per chip-hour), sets them up
  (cloud/setup_vm.sh: the Kaggle image's torch / torch_xla / libtpu, the 20 data shards) and gives each chip one job at a time
  (cloud/vm_job.sh, detached; one process per chip);
- polls every VM once a minute (cloud/vm_status.py over ssh), copies each finished job's run.json, metrics.jsonl, summary.json and
  log.txt to results/<batch>/runs/<job>/, and copies running jobs' metrics to results/<batch>/live/ every ~10 minutes;
- when a VM is preempted (or stops answering), deletes it and puts its running jobs back in the queue. They restart from step 0:
  checkpoints live on the VM's own disk (this project's VMs have no Cloud Storage access), so they only help when a job is
  restarted on the same VM;
- deletes VMs as soon as nothing is left for them; records every create/delete in cloud/ledger.jsonl (cloud/cost.py adds it up);
- is stateless: the VMs and results/<batch>/runs are the state. Restarted, it adopts the batch's VMs that are still up (and
  sets up any that an earlier run created but had not set up, logging their creation in the ledger if it is missing).
VM creation is staggered (at most CREATE_PER_MINUTE requests a minute: a burst of 31 hit the TPU API's request quota).
Jobs that fail (non-zero exit) are retried once; preempted jobs any number of times.
"""

import argparse
import json
import random
import re
import runpy
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
import cost  # noqa: E402

USER = "marcoshernanz"
KEY = Path.home() / ".ssh" / "google_compute_engine"
SSH_OPTS = ["-i", str(KEY), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=60", "-o", "ServerAliveInterval=15"]  # a v6e-8 host running 8 jobs can take >20 s to accept
VERSION = {"v6e": "v2-alpha-tpuv6e", "v5litepod": "v2-alpha-tpuv5-lite"}
ZONES = {  # zones offering each type (gcloud compute tpus accelerator-types list, 2026-10-03); no quota in us-east4
    "v6e": ["asia-southeast1-b", "us-central1-a", "us-central1-b", "us-central1-c", "us-east1-d", "us-west1-c", "us-south1-a",
            "us-south1-c", "us-east5-a", "us-east5-b", "us-east5-c"],
    "v5litepod": ["us-central1-a", "us-east1-c", "us-west1-c", "us-west4-b", "europe-west4-b", "us-east5-a", "us-east5-b",
                  "us-east5-c", "asia-southeast1-b", "us-south1-a"],
}
FILES = ["run.json", "metrics.jsonl", "summary.json", "log.txt", "exit_code"]
CKPT_EVERY = "1000"
CREATE_PER_MINUTE = 4
LONG_JOB = 4000  # seconds: jobs this long go to on-demand VMs first (a preempted job restarts from step 0)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%m-%d %H:%M:%S")


class Controller:
    def __init__(self, args):
        self.batches = [Path(b).resolve() for b in args.batch]
        self.jobs, self.job_out = {}, {}
        for batch in self.batches:
            for job in runpy.run_path(str(batch / "jobs.py"))["JOBS"]:
                assert job["name"] not in self.jobs, f"job name {job['name']} in two batches"
                self.jobs[job["name"]] = job
                self.job_out[job["name"]] = ROOT / "results" / batch.name
        self.order = list(self.jobs)
        self.accels = args.accel.split(",")  # sizes to try in each zone, in order (e.g. v6e-8,v6e-4: fall back when 8 have no capacity)
        self.accel = self.accels[0]
        self.family, chips = self.accel.rsplit("-", 1)
        assert all(a.rsplit("-", 1)[0] == self.family for a in self.accels)
        self.chips = int(chips)  # the first size, used to plan creations
        self.max_chips = args.max_chips
        self.tags = [args.tag] if args.tag else [b.name.split("_")[0] for b in self.batches]  # VMs of all of them are adopted
        self.tag = self.tags[0]  # new VMs are named after the first
        self.drain = args.drain_other_sizes  # VMs of a size not in --accel get no new jobs (and so are deleted when idle)
        for out in set(self.job_out.values()):
            (out / "runs").mkdir(parents=True, exist_ok=True)
            (out / "live").mkdir(exist_ok=True)
        self.out = ROOT / "results" / self.batches[0].name
        prices = json.loads((ROOT / "cloud" / "prices.json").read_text())[self.family]
        self.zones = sorted((z for z in ZONES[self.family] if prices.get(z.rsplit("-", 1)[0], 99) <= args.max_price),
                            key=lambda z: prices[z.rsplit("-", 1)[0]])
        od_prices = json.loads((ROOT / "cloud" / "prices.json").read_text())["on_demand"].get(self.family, {})
        self.od_zones = sorted((z for z in ZONES[self.family] if z.rsplit("-", 1)[0] in od_prices),
                               key=lambda z: od_prices[z.rsplit("-", 1)[0]])
        self.od_chips = args.on_demand_chips
        self.od_budget = args.on_demand_budget
        self.creating_od = 0  # on-demand creations in flight
        self.budget = args.budget
        self.lock = threading.Lock()
        self.vms = {}  # name -> {"zone", "ip", "ready", "busy", "fails", "running": {job: chip}}
        self.attempts = {name: {"preempted": 0, "failed": 0} for name in self.jobs}
        self.status = {}
        self.creating = 0
        self.creating_names = set()  # VMs a thread is creating or setting up right now
        self.backoff = {}  # (zone, accel) -> time before which creating that size there is not retried
        self.regions = {}  # region -> (time checked, external IPs free) from Compute Engine's IN_USE_ADDRESSES quota
        self.inflight = {}  # region -> creation requests in flight there
        self.counter = 0
        self.log_file = open(self.out / "controller.log", "a")
        for batch in self.batches[1:]:
            with open(ROOT / "results" / batch.name / "controller.log", "a") as f:
                f.write(f"[{now()}] this batch now runs in a shared controller with {', '.join(b.name for b in self.batches)}; "
                        f"its log is results/{self.batches[0].name}/controller.log\n")
        self.last_live = 0.0
        self.started = {}  # job -> wall time it was (last) started

    # ---------------------------------------------------------------- helpers
    def log(self, *parts) -> None:
        line = f"[{now()}] " + " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def run(self, cmd: list, timeout: float = 120) -> tuple[int, str]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 124, "timeout"

    def ssh(self, vm: str, command: str, timeout: float = 90) -> tuple[int, str]:
        return self.run(["ssh", *SSH_OPTS, f"{USER}@{self.vms[vm]['ip']}", command], timeout)

    def scp_from(self, vm: str, remote: list, local: Path, timeout: float = 300) -> int:
        local.mkdir(parents=True, exist_ok=True)
        srcs = [f"{USER}@{self.vms[vm]['ip']}:{r}" for r in remote]
        return self.run(["scp", *SSH_OPTS, "-q", *srcs, str(local)], timeout)[0]

    def done(self, name: str) -> bool:
        run = self.job_out[name] / "runs" / name
        return (run / "summary.json").exists() and (run / "exit_code").exists()

    def pending(self) -> list:
        running = {j for vm in self.vms.values() for j in vm["running"]}
        return [n for n in self.order if not self.done(n) and n not in running and n not in self.status.get("_failed", [])]

    # ---------------------------------------------------------------- VM lifecycle
    def list_vms(self) -> dict:
        code, out = self.run(["gcloud", "compute", "tpus", "tpu-vm", "list", "--zone=-", "--format=json"], 180)
        if code != 0:
            raise RuntimeError(out[-500:])
        found = {}
        for node in json.loads(out):
            name = node["name"].rsplit("/", 1)[1]
            if any(name.startswith(f"mhc-{tag}-") for tag in self.tags):
                ip = node.get("networkEndpoints", [{}])[0].get("accessConfig", {}).get("externalIp")
                sched = node.get("schedulingConfig", {})
                found[name] = {"zone": node["name"].split("/")[3], "state": node.get("state"), "ip": ip,
                               "accel": node.get("acceleratorType"), "created": node.get("createTime"),
                               "spot": bool(sched.get("spot") or sched.get("preemptible"))}
        return found

    def create(self, spot: bool = True) -> None:
        with self.lock:
            self.counter += 1
            name = f"mhc-{self.tag}-{'' if spot else 'od'}{int(time.time()) % 100000:05d}{self.counter}"
            self.creating_names.add(name)
        try:
            for accel in self.accels:  # the biggest size in every zone first: each VM takes one of the region's 4 external IPs
                for zone in (self.zones if spot else self.od_zones):
                    if self.backoff.get((zone, accel, spot), 0) > time.time() or not self.ip_free(zone):
                        continue
                    if not spot and (int(accel.rsplit("-", 1)[1]) < 4 or
                                     self.od_used() + int(accel.rsplit("-", 1)[1]) + self.chips * (self.creating_od - 1) > self.od_chips):
                        continue  # on-demand: no single chips (4x the price for one IP), and not over --on-demand-chips
                    reason = self.try_create(name, zone, accel, spot)
                    if reason is None:
                        return
        finally:
            with self.lock:
                self.creating -= 1
                if not spot:
                    self.creating_od -= 1
                self.creating_names.discard(name)

    def od_used(self) -> int:
        """Chips of on-demand VMs this run manages (not counting creations in flight)."""
        return sum(i["chips"] for i in self.vms.values() if not i.get("spot", True))

    def od_over(self) -> bool:
        return cost.spend(on_demand_only=True)[0] >= self.od_budget

    def ip_free(self, zone: str) -> bool:
        """Whether the zone's region has an external IP left (checked at most every 2 minutes, minus this run's creations in flight)."""
        region = zone.rsplit("-", 1)[0]
        checked, free = self.regions.get(region, (0, 0))
        if time.time() - checked > 120:
            code, out = self.run(["gcloud", "compute", "regions", "describe", region, "--format=json"], 60)
            if code == 0:
                quota = {q["metric"]: q for q in json.loads(out)["quotas"]}["IN_USE_ADDRESSES"]
                free = quota["limit"] - quota["usage"]
            else:
                free = 1  # unknown: let the create call find out
            self.regions[region] = (time.time(), free)
        return free - self.inflight.get(region, 0) > 0

    def try_create(self, name: str, zone: str, accel: str, spot: bool = True) -> str | None:
        """One VM of one size in one zone; None on success (the VM is then set up), else the reason it failed."""
        region = zone.rsplit("-", 1)[0]
        for attempt in range(3):  # retried only when the API's request rate limit was hit
            began = time.time()
            with self.lock:
                self.inflight[region] = self.inflight.get(region, 0) + 1
            try:
                code, out = self.run(["gcloud", "compute", "tpus", "tpu-vm", "create", name, f"--zone={zone}",
                                      f"--accelerator-type={accel}", f"--version={VERSION[self.family]}"]
                                     + (["--spot"] if spot else []), 900)
            finally:
                with self.lock:
                    self.inflight[region] -= 1
            if code == 0:
                self.regions.pop(region, None)  # one IP fewer
            if code == 0:
                cost.log("create", name, zone, accel, began, spot)
                self.log(f"created {name} ({accel}{'' if spot else ', on-demand'}) in {zone}")
                self.setup(name, zone)
                return None
            low = out.lower()
            message = re.search(r'"message": "([^"]+)"', out)
            message = message.group(1) if message else " ".join(out.split())[-200:]
            if "capacity" in low:
                reason = "capacity"
            elif "per minute" in low or "quota metric" in low:  # the API's request rate limit, not the chip quota
                reason = "request rate"
            elif "quota" in low:
                reason = "quota"
            elif "in_use_addresses limit" in low or "instances limit" in low:
                reason = "regional limit"  # 4 external IPs / 8 instances per region in this project
            elif "reservation not found" in low:
                reason = "needs a reservation"
            else:
                reason = message
            self.log(f"create {name} ({accel}{'' if spot else ', on-demand'}) in {zone} failed: {reason}"
                     + (f" ({message[:200]})" if reason in ("quota", "request rate") else ""))
            if reason == "request rate":
                time.sleep(random.uniform(30, 90))
                continue
            if reason == "regional limit":
                self.regions[region] = (time.time() + 180, 0)  # no IP left there: look again in 5 minutes
            elif reason == "needs a reservation":
                self.backoff[(zone, accel, spot)] = time.time() + 3600
            elif reason in ("capacity", "quota"):
                self.backoff[(zone, accel, spot)] = time.time() + 900
            else:  # a half-created node may exist
                self.backoff[(zone, accel, spot)] = time.time() + 300
                self.run(["gcloud", "compute", "tpus", "tpu-vm", "delete", name, f"--zone={zone}", "--quiet"], 600)
            return reason
        return "request rate"

    def setup(self, name: str, zone: str) -> None:
        found = self.list_vms().get(name)
        if not found or not found["ip"]:
            self.log(f"{name}: no external ip after create; deleting")
            self.delete(name, zone)
            return
        with self.lock:
            self.vms[name] = {"zone": zone, "ip": found["ip"], "ready": False, "busy": True, "fails": 0, "running": {},
                              **self.size(found)}
        # the first contact goes through gcloud, which installs this machine's ssh key on the VM
        for attempt in range(5):
            code, out = self.run(["gcloud", "compute", "tpus", "tpu-vm", "ssh", name, f"--zone={zone}", "--command", "true"], 300)
            if code == 0:
                break
            time.sleep(20)
        files = [str(ROOT / "cloud" / f) for f in ("setup_vm.sh", "vm_job.sh", "vm_status.py")]
        ok = self.run(["scp", *SSH_OPTS, "-q", *files, f"{USER}@{found['ip']}:~/"], 300)[0] == 0
        ok = ok and self.run(["scp", *SSH_OPTS, "-q", "-r", str(ROOT / "src"), f"{USER}@{found['ip']}:~/"], 300)[0] == 0
        code, out = self.ssh(name, "bash ~/setup_vm.sh > ~/setup.log 2>&1 && touch ~/setup_done; tail -2 ~/setup.log", 1800)
        if ok and code == 0:
            self.log(f"{name}: set up ({out.strip().splitlines()[-1] if out.strip() else ''})")
            with self.lock:
                self.vms[name]["ready"] = True
                self.vms[name]["busy"] = False
        else:
            self.log(f"{name}: setup failed ({out[-300:]}); deleting")
            self.delete(name, zone)

    def delete(self, name: str, zone: str, reason: str = "") -> None:
        with self.lock:
            vm = self.vms.pop(name, None)
        if vm:
            for job in vm["running"]:
                self.attempts[job]["preempted"] += 1
                self.log(f"{job}: back in the queue ({reason or 'vm deleted'}, attempt {sum(self.attempts[job].values()) + 1})")
        code, out = self.run(["gcloud", "compute", "tpus", "tpu-vm", "delete", name, f"--zone={zone}", "--quiet"], 900)
        cost.log("delete", name, zone, vm["accel"] if vm else self.accel, spot=vm.get("spot", True) if vm else True)
        self.log(f"deleted {name} ({reason}){'' if code == 0 else ' — delete failed: ' + out[-200:]}")

    # ---------------------------------------------------------------- jobs
    def start_job(self, vm: str, chip: int, name: str, fresh: bool = True) -> bool:
        """Start a job the caller has already claimed (vms[vm]["running"][name] = chip); releases the claim on failure."""
        job = self.jobs[name]
        args = " ".join(f"'{a}'" for a in [*job["args"], "--ckpt-every", CKPT_EVERY])
        cmd = ((f"rm -rf ~/runs/{name}; " if fresh else "") +
               f"setsid nohup bash ~/vm_job.sh {chip} {self.vms[vm]['chips']} {self.vms[vm]['accel']} {name} -- {args} > /dev/null 2>&1 < /dev/null &")
        code, out = self.ssh(vm, cmd, 60)
        if code == 0:
            self.started[name] = time.time()
            self.log(f"{name}: started on {vm} chip {chip}{'' if fresh else ' (resuming)'}")
            return True
        self.log(f"{name}: start on {vm} failed: {out[-200:]}")
        with self.lock:
            self.vms[vm]["running"].pop(name, None)
        return False

    def poll(self, vm: str) -> None:
        """Check one VM: collect finished jobs, start new ones, delete the VM when it has nothing left to do."""
        info = self.vms[vm]
        code, out = self.ssh(vm, "~/venv/bin/python ~/vm_status.py", 150)
        if code != 0:
            info["fails"] += 1
            self.log(f"{vm}: status failed ({info['fails']}): {out.strip()[-160:]}")
            if info["fails"] >= 6:
                state = self.run(["gcloud", "compute", "tpus", "tpu-vm", "describe", vm, f"--zone={info['zone']}",
                                  "--format=value(state)"], 120)[1].strip()
                if "READY" in state.split():  # a loaded host answers ssh slowly; preemption shows up as another state
                    self.log(f"{vm}: still READY in gcloud, so not deleted; ssh is slow")
                    info["fails"] = 3
                else:
                    self.delete(vm, info["zone"], f"not answering (state {state or 'unknown'})")
            return
        info["fails"] = 0
        try:
            status = json.loads(out.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            self.log(f"{vm}: bad status output {out[-200:]}")
            return
        live = time.time() - self.last_live > 600
        for name, chip in list(info["running"].items()):
            s = status.get(name)
            if s is None:
                continue
            if s["exit_code"] is not None:
                dest = self.job_out[name] / "runs" / name
                self.scp_from(vm, [f"~/runs/{name}/{f}" for f in FILES], dest)
                summary = json.loads((dest / "summary.json").read_text()) if (dest / "summary.json").exists() else {}
                took = summary.get("seconds", 0)
                if s["exit_code"] == "0" and summary:
                    self.status[name] = (f"exit 0 in {took:.0f}s on {vm} ({info['zone']}), val {summary.get('final_val_loss'):.4f}, "
                                         f"tok/s {summary.get('tok_s') or 0:.0f}, preempted {self.attempts[name]['preempted']}x")
                    self.log(f"{name}: done — {self.status[name]}")
                else:
                    self.attempts[name]["failed"] += 1
                    tail = (dest / "log.txt").read_text(errors="replace").splitlines()[-8:] if (dest / "log.txt").exists() else []
                    self.log(f"{name}: FAILED exit {s['exit_code']} (failure {self.attempts[name]['failed']}):\n  " + "\n  ".join(tail))
                    for f in ("summary.json", "exit_code"):
                        (dest / f).unlink(missing_ok=True)
                    if self.attempts[name]["failed"] >= 2:
                        self.status.setdefault("_failed", []).append(name)
                        self.status[name] = f"failed twice (exit {s['exit_code']})"
                self.ssh(vm, f"rm -f ~/runs/{name}/ckpt.pt ~/runs/{name}/ckpt.tmp", 60)
                with self.lock:
                    del info["running"][name]
            elif not s["alive"]:
                self.log(f"{name}: process gone without exit code on {vm}; restarting it there (resumes from its checkpoint)")
                self.start_job(vm, chip, name, fresh=False)
            else:
                job = self.jobs[name]
                wall = time.time() - self.started.get(name, time.time())
                if wall > job.get("timeout", 4 * job.get("expected_seconds", 3600)):
                    self.log(f"{name}: over its time limit ({wall:.0f}s wall); killing")
                    self.ssh(vm, f"pkill -f 'runs/{name} '")
                if live:
                    self.scp_from(vm, [f"~/runs/{name}/metrics.jsonl"], self.job_out[name] / "live" / name)
        if live:
            self.last_live = time.time()
        # fill free chips (the claim is made under the lock: several VMs are polled in parallel)
        over = self.over_budget()
        queue = []
        for chip in range(info["chips"]):
            with self.lock:
                if chip in info["running"].values():
                    continue
                queue = [j for j in self.pending() if self.jobs[j].get("min_chips", 1) <= info["chips"]]  # min_chips: only on VMs that big
                long = [j for j in queue if self.jobs[j].get("expected_seconds", 0) >= LONG_JOB]
                short = [j for j in queue if j not in long]
                if not info.get("spot", True):
                    queue = long + short
                elif self.od_chips:
                    queue = short or long  # leave the long jobs to the on-demand VMs while short ones wait
                # with no on-demand VMs at all, the batches' own order
                if not info.get("spot", True) and self.od_over():
                    queue = []  # on-demand budget spent: finish what runs, then delete
                if not queue or over or (self.drain and info["accel"] not in self.accels):
                    queue = []
                    break
                name = queue[0]
                info["running"][name] = chip
            if not self.start_job(vm, chip, name):
                break  # the VM is probably going away (ssh refused): leave the rest of the queue to the next poll
        if not info["running"]:
            if not queue:
                self.delete(vm, info["zone"], "no jobs left")
            else:  # jobs are waiting but none could be started: as many tries as a VM that stops answering
                info["start_fails"] = info.get("start_fails", 0) + 1
                if info["start_fails"] >= 6:
                    self.delete(vm, info["zone"], "cannot start jobs")
        else:
            info["start_fails"] = 0

    def over_budget(self) -> bool:
        return cost.spend()[0] > self.budget

    # ---------------------------------------------------------------- main loop
    def ensure_ledger(self, name: str, node: dict) -> None:
        """Log the creation of a VM an earlier controller run made but did not record (it was stopped mid-create)."""
        _, rows = cost.spend()
        if any(vm == name and zone == node["zone"] and up for vm, zone, _, _, _, up in rows):
            return
        created = datetime.fromisoformat(node["created"][:26].rstrip("Z") + "+00:00").timestamp() if node.get("created") else None
        cost.log("create", name, node["zone"], node["accel"] or self.accel, created, node.get("spot", True))
        self.log(f"{name}: creation was missing from the ledger; logged")

    def reconcile(self, found: dict) -> None:
        """Take over VMs of this batch that this run does not manage yet (left by an earlier run of the controller)."""
        for name, node in found.items():
            with self.lock:
                if name in self.vms or name in self.creating_names:
                    continue
            if node["state"] in ("CREATING", "STARTING", "RESTARTING", "REPAIRING"):
                continue  # look again next minute
            self.ensure_ledger(name, node)
            if node["state"] != "READY" or not node["ip"]:
                self.delete(name, node["zone"], f"found in state {node['state']}")
                continue
            code, out = self.ssh_ip(node["ip"], "test -f ~/setup_done && ~/venv/bin/python ~/vm_status.py", 90)
            if code != 0:
                self.log(f"found {name} ({node['zone']}), not set up: setting it up")
                with self.lock:
                    self.creating_names.add(name)
                    self.creating += 1
                self.pool.submit(self.setup_found, name, node["zone"])
                continue
            self.vms[name] = {"zone": node["zone"], "ip": node["ip"], "ready": False, "busy": False, "fails": 0, "running": {},
                              **self.size(node)}
            status = json.loads(out.strip().splitlines()[-1])
            for job, s in status.items():
                if job in self.jobs and not self.done(job) and (s["alive"] or s["exit_code"] is not None):
                    self.vms[name]["running"][job] = s["chip"] or 0
                    self.started[job] = time.time() - ((s["last"] or {}).get("elapsed") or 0)
            self.vms[name]["ready"] = True
            self.log(f"adopted {name} ({node['zone']}): running {list(self.vms[name]['running'])}")

    def size(self, node: dict) -> dict:
        accel = node.get("accel") or self.accel
        return {"accel": accel, "chips": int(accel.rsplit("-", 1)[1]), "spot": node.get("spot", True)}

    def setup_found(self, name: str, zone: str) -> None:
        try:
            self.setup(name, zone)
        finally:
            with self.lock:
                self.creating -= 1
                self.creating_names.discard(name)

    def ssh_ip(self, ip: str, command: str, timeout: float = 90) -> tuple[int, str]:
        return self.run(["ssh", *SSH_OPTS, f"{USER}@{ip}", command], timeout)

    def write_status(self) -> None:
        status = {n: ("done" if self.done(n) and n not in self.status else self.status.get(n, "pending")) for n in self.order}
        for vm, info in self.vms.items():
            for job in info["running"]:
                status[job] = f"running on {vm} ({info['zone']})"
        for out in set(self.job_out.values()):  # one status.json per batch, with its own jobs
            mine = {n: s for n, s in status.items() if self.job_out[n] == out}
            mine["_attempts"] = {n: a for n, a in self.attempts.items() if any(a.values()) and self.job_out[n] == out}
            (out / "status.json").write_text(json.dumps(mine, indent=1))

    def loop(self) -> None:
        self.pool = pool = ThreadPoolExecutor(48)
        last_report = 0.0
        while True:
            try:
                found = self.list_vms()
            except RuntimeError as error:
                self.log(f"list failed: {error}")
                time.sleep(60)
                continue
            self.reconcile(found)
            for name, info in list(self.vms.items()):
                node = found.get(name)
                if info["busy"]:
                    continue
                if node is None or node["state"] in ("PREEMPTED", "TERMINATED", "STOPPED", "STOPPING", "DELETING"):
                    self.delete(name, info["zone"], f"state {node['state'] if node else 'gone'}")
            ready = [n for n, i in self.vms.items() if i["ready"] and not i["busy"]]
            list(pool.map(self.poll, ready))
            queue = self.pending()
            free = sum(i["chips"] - len(i["running"]) for i in self.vms.values() if i["ready"])
            setting_up = sum(i["chips"] for i in self.vms.values() if not i["ready"])
            used = sum(i["chips"] for i in self.vms.values() if i.get("spot", True)) + (self.creating - self.creating_od) * self.chips
            want = len(queue) - free - setting_up - self.creating * self.chips
            spent = cost.spend()[0]
            od_ok = self.od_chips and not self.od_over()
            for _ in range(CREATE_PER_MINUTE):
                if not (want > 0 and spent < self.budget):
                    break
                spot = not (od_ok and self.od_used() + self.chips * (self.creating_od + 1) <= self.od_chips)
                if spot and used + self.chips > self.max_chips:  # --max-chips caps spot chips; on-demand ones have their own cap
                    break
                with self.lock:
                    self.creating += 1
                    if not spot:
                        self.creating_od += 1
                pool.submit(self.create, spot)
                want -= self.chips
                if spot:
                    used += self.chips
            self.write_status()
            if time.time() - last_report > 900:
                last_report = time.time()
                n_done = sum(self.done(n) for n in self.order)
                self.log(f"status: {n_done}/{len(self.order)} done, {len(queue)} queued, "
                         f"{sum(len(i['running']) for i in self.vms.values())} running on {len(self.vms)} VMs, spent ${spent:.2f}")
            if not queue and not self.vms and not self.creating:
                failed = self.status.get("_failed", [])
                self.log(f"batch finished: {sum(self.done(n) for n in self.order)} done, failed {failed}; spent ${spent:.2f}")
                self.write_status()
                return
            if spent >= self.budget and not self.vms:
                self.log("budget reached; stopping")
                return
            time.sleep(60)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("batch", nargs="+")
    p.add_argument("--accel", default="v6e-1")
    p.add_argument("--max-chips", type=int, default=16)
    p.add_argument("--max-price", type=float, default=0.75)
    p.add_argument("--budget", type=float, default=280.0)  # total project spend (cloud/ledger.jsonl) at which no new job starts
    p.add_argument("--tag", default=None)
    p.add_argument("--drain-other-sizes", action="store_true")
    p.add_argument("--on-demand-chips", type=int, default=0)
    p.add_argument("--on-demand-budget", type=float, default=0.0)  # on-demand spend (cloud/ledger.jsonl) after which they get no new jobs
    Controller(p.parse_args()).loop()


if __name__ == "__main__":
    main()

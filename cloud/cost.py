"""Google Cloud spend of this project, from cloud/ledger.jsonl (one line per TPU VM create/delete).

Usage: python cloud/cost.py            prints per-VM hours and cost, and the total (VMs still up are counted until now)
       python cloud/cost.py --log create|delete NAME ZONE TYPE      appends an event (cloud/controller.py does this itself)
Prices: cloud/prices.json (spot, per chip-hour, and on-demand under "on_demand"; the TPU VM price includes its host). Budget: $300.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.jsonl"
BUDGET = 300.0


def price(zone: str, accel: str, spot: bool = True) -> tuple[float, int]:
    prices = json.loads((HERE / "prices.json").read_text())
    family, chips = accel.rsplit("-", 1)
    return (prices if spot else prices["on_demand"])[family][zone.rsplit("-", 1)[0]], int(chips)


def log(event: str, name: str, zone: str, accel: str, when: float | None = None, spot: bool = True) -> None:
    p, chips = price(zone, accel, spot)
    record = {"event": event, "vm": name, "zone": zone, "type": accel, "chips": chips, "price_chip_h": p, "spot": spot,
              "time": when or time.time(), "utc": datetime.fromtimestamp(when or time.time(), timezone.utc).isoformat()}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(record) + "\n")


def spend(now: float | None = None, on_demand_only: bool = False) -> tuple[float, list]:
    now = now or time.time()
    open_vms, rows = {}, []
    for line in LEDGER.read_text().splitlines() if LEDGER.exists() else []:
        r = json.loads(line)
        key = (r["vm"], r["zone"])
        if r["event"] == "create":
            open_vms[key] = r
        elif key in open_vms:
            c = open_vms.pop(key)
            rows.append((c, r["time"]))
    rows += [(c, None) for c in open_vms.values()]
    total, out = 0.0, []
    for c, end in rows:
        if on_demand_only and c.get("spot", True):
            continue
        hours = ((end or now) - c["time"]) / 3600
        cost = hours * c["chips"] * c["price_chip_h"]
        total += cost
        out.append((c["vm"], c["zone"], c["type"] + ("" if c.get("spot", True) else " (on-demand)"), hours, cost, end is None))
    return total, out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--log":
        log(*sys.argv[2:6], *(float(x) for x in sys.argv[6:7]))
    total, rows = spend()
    for vm, zone, accel, hours, cost, up in rows:
        print(f"{vm:24s} {zone:18s} {accel:24s} {hours:7.2f} h  ${cost:7.2f}{'  (up)' if up else ''}")
    print(f"total ${total:.2f} of ${BUDGET:.0f}")

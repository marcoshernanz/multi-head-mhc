"""Compare two runs' metrics.jsonl step by step: train loss, grad norm and validation loss.

Usage: python analysis/checks/check_resume.py RUN_A RUN_B [--max-step N]
Used for: (1) a run that was killed and resumed from its checkpoint vs the same run uninterrupted (must be identical);
(2) the same run twice (determinism); (3) the same run on two TPU generations (how fast the trajectories drift apart).
"""

import argparse
import json
from pathlib import Path


def records(run: Path) -> tuple[dict, dict]:
    train, val = {}, {}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "loss" in r:
            train[r["step"]] = (r["loss"], r["grad_norm"])
        elif "val_loss" in r:
            val[r["step"]] = r["val_loss"]
    return train, val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--max-step", type=int, default=10**9)
    args = p.parse_args()
    (ta, va), (tb, vb) = records(args.a), records(args.b)
    steps = sorted(s for s in set(ta) & set(tb) if s <= args.max_step)
    vsteps = sorted(s for s in set(va) & set(vb) if s <= args.max_step)
    identical = all(ta[s] == tb[s] for s in steps) and all(va[s] == vb[s] for s in vsteps)
    print(f"{len(steps)} common train steps, {len(vsteps)} common val evals; bit-identical: {identical}")
    first = next((s for s in steps if ta[s] != tb[s]), None)
    print(f"first differing train step: {first}")
    for s in steps[:: max(1, len(steps) // 12)] + steps[-1:]:
        (la, ga), (lb, gb) = ta[s], tb[s]
        print(f"  step {s:5d}: loss {la:.6f} vs {lb:.6f} (diff {lb - la:+.2e}); grad norm {ga:.4f} vs {gb:.4f}")
    for s in vsteps:
        print(f"  val @ {s:5d}: {va[s]:.6f} vs {vb[s]:.6f} (diff {vb[s] - va[s]:+.2e})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure differentiation and stability together, because one buys the other.

Any change that spreads scores also amplifies whatever noise feeds them, so
reporting a wider spread on its own proves nothing. The defect-focused headline
is the specific worry: it weights the worst aspects, and the worst aspect is the
one driven by a single finding, hence the most volatile.

So both are measured on the same runs:

    between   spread across different videos       -- differentiation
    within    spread of one video across repeats   -- reproducibility
    SNR       between / within                     -- whether the spread is real

Repeats vary the sampling phase, since at temperature 0 with fixed prompts the
model is deterministic and re-running measures nothing. What varies is which
frames get looked at, which is an arbitrary choice that should not move a score.

Runs each configuration so the effect of consensus is visible rather than
assumed: `--phases 1` for the single-pass baseline, `--phases 3` for consensus.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def one_run(videos, cond, cid, out: Path, phases: int, seed: int) -> dict | None:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_eval.py"),
           "--videos", *videos, "--condition", cond, "--condition-id", cid,
           "--out", str(out), "--phases", str(phases)]
    if phases == 1:                       # vary the phase ourselves for repeats
        cmd += ["--perturb", "frame_phase", "--perturb-seed", str(seed)]
    subprocess.run(cmd, check=False,
                   stdout=(out.parent / f"{out.name}.log").open("w"),
                   stderr=subprocess.STDOUT, env={**os.environ})
    p = out / "all_results.json"
    return json.loads(p.read_text()) if p.exists() else None


def analyse(runs: list[dict], label: str) -> dict:
    models = sorted(runs[0])
    per_model = {m: [r[m]["score"]["overall"] for r in runs] for m in models}
    within = statistics.mean(statistics.pstdev(v) for v in per_model.values()
                             if len(v) > 1) if len(runs) > 1 else 0.0
    means = [statistics.mean(v) for v in per_model.values()]
    between = statistics.pstdev(means) if len(means) > 1 else 0.0
    snr = between / within if within > 1e-9 else (float("inf") if between > 1e-9 else 0.0)
    orders = [tuple(sorted(models, key=lambda m: -r[m]["score"]["overall"]))
              for r in runs]
    stable = len(set(orders)) == 1
    print(f"\n=== {label} ===")
    for m in models:
        v = per_model[m]
        print(f"  {m.split('__')[0]:16s} " + " ".join(f"{x:5.2f}" for x in v)
              + f"   均值 {statistics.mean(v):5.2f}  组内std {statistics.pstdev(v):.3f}")
    print(f"  组间(区分度) {between:.3f}   组内(噪声) {within:.3f}   "
          f"信噪比 {'inf' if snr == float('inf') else f'{snr:.2f}'}   "
          f"排序稳定 {'是' if stable else '否'}")
    return {"label": label, "between": between, "within": within,
            "snr": None if snr == float("inf") else snr,
            "rank_stable": stable, "per_model": per_model}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--condition-id", default="c0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--configs", nargs="*", default=["1", "3"],
                    help="phase counts to compare, e.g. 1 3")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    results = []
    for cfg in args.configs:
        n = int(cfg)
        runs = []
        for i in range(args.repeats):
            d = out / f"p{n}_rep{i}"
            print(f"[{n} 相位] repeat {i+1}/{args.repeats}", flush=True)
            r = one_run(args.videos, args.condition, args.condition_id, d, n, i)
            if r:
                runs.append(r)
        if len(runs) >= 2:
            results.append(analyse(runs, f"{n} 相位"
                                   + (" (共识)" if n > 1 else " (单次)")))

    if len(results) >= 2:
        a, b = results[0], results[1]
        print(f"\n=== 对比 ===")
        print(f"  区分度 {a['between']:.3f} → {b['between']:.3f}")
        print(f"  噪声   {a['within']:.3f} → {b['within']:.3f}")
        fa = a["snr"] if a["snr"] is not None else float("inf")
        fb = b["snr"] if b["snr"] is not None else float("inf")
        print(f"  信噪比 {fa:.2f} → {fb:.2f}"
              + ("   共识提高了信噪比" if fb > fa else "   共识没有提高信噪比"))
    json.dump(results, open(out / "stability.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {out/'stability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

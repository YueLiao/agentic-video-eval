#!/usr/bin/env python3
"""Measure run-to-run variance: how much of a score is signal?

Repeats the identical evaluation N times with code, prompts and conditions
held fixed and caching disabled, then reports the spread. This has to be
answered before any consolidation verdict is trusted, because keep/drop
decisions rest on between-model differences and those are only meaningful if
they exceed the noise a single model produces against itself.

The specific worry: scores here are driven by a handful of findings, and one
major finding moves an aspect 10.00 -> 6.50. If which findings surface varies
run to run, apparent model differences may be nothing but resampling.

Reports, per aspect: within-model spread (the noise floor), between-model
spread (the signal), their ratio, and whether the model ranking is stable
across repeats. An aspect whose ranking flips between repeats cannot support a
leaderboard column however clean its numbers look in one run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_once(videos, condition, cond_id, out: Path, extra: list[str]) -> Path:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_eval.py"),
           "--videos", *videos, "--condition", condition,
           "--condition-id", cond_id, "--out", str(out), *extra]
    subprocess.run(cmd, check=False, stdout=(out.parent / f"{out.name}.log").open("w"),
                   stderr=subprocess.STDOUT, env={**os.environ})
    return out / "all_results.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--condition-id", default="c0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for i in range(args.repeats):
        # A separate out dir per repeat means a separate cache, so every repeat
        # actually calls the model. Sharing one would make repeats 2..N free
        # and identical, measuring nothing.
        d = out / f"rep{i}"
        print(f"=== repeat {i+1}/{args.repeats} -> {d}", flush=True)
        p = run_once(args.videos, args.condition, args.condition_id, d, args.extra)
        if p.exists():
            runs.append(json.loads(p.read_text()))
        else:
            print(f"  repeat {i} produced no results", file=sys.stderr)
    if len(runs) < 2:
        print("need >=2 successful repeats", file=sys.stderr)
        return 1

    models = sorted(runs[0])
    aspects = sorted(runs[0][models[0]]["score"]["aspects"])

    def score(run, m, a):
        x = run[m]["score"]["aspects"].get(a) or {}
        return x.get("score") if x.get("judgeable") else None

    print(f"\n{'aspect':16s} {'组内噪声':>9s} {'组间信号':>9s} {'信噪比':>7s} "
          f"{'排序稳定':>9s}   每次运行的各模型分数")
    print("-" * 104)
    rows = []
    for a in aspects:
        within, means, rankings = [], [], []
        for m in models:
            vals = [s for s in (score(r, m, a) for r in runs) if s is not None]
            if len(vals) < 2:
                continue
            within.append(statistics.pstdev(vals))
            means.append(statistics.mean(vals))
        if len(means) < 2:
            continue
        for r in runs:
            vs = {m: score(r, m, a) for m in models}
            if all(v is not None for v in vs.values()):
                rankings.append(tuple(sorted(vs, key=lambda m: -vs[m])))
        noise = statistics.mean(within) if within else 0.0
        signal = statistics.pstdev(means)
        snr = (signal / noise) if noise > 1e-9 else (float("inf") if signal > 1e-9 else 0.0)
        stable = "是" if rankings and len(set(rankings)) == 1 else "否"
        label = runs[0][models[0]]["score"]["aspects"][a].get("label", a)
        cells = " | ".join(
            ",".join(f"{score(r, m, a):.1f}" if score(r, m, a) is not None else "—"
                     for m in models) for r in runs)
        rows.append((snr, label, noise, signal, stable, cells))
    rows.sort(key=lambda r: -(r[0] if r[0] != float("inf") else 1e9))
    for snr, label, noise, signal, stable, cells in rows:
        s = "  inf" if snr == float("inf") else f"{snr:5.2f}"
        print(f"{label:16s} {noise:9.3f} {signal:9.3f} {s:>7s} {stable:>9s}   {cells}")

    ov = {m: [r[m]["score"]["overall"] for r in runs] for m in models}
    print("\n总分:")
    for m in models:
        print(f"  {m.split('__')[0]:16s} " + "  ".join(f"{v:.2f}" for v in ov[m])
              + f"   (std {statistics.pstdev(ov[m]):.2f})")
    orders = [tuple(sorted(models, key=lambda m: -r[m]["score"]["overall"])) for r in runs]
    print(f"  总分排序在 {len(runs)} 次运行中" +
          ("**保持一致**" if len(set(orders)) == 1 else f"**变化了**({len(set(orders))} 种)"))

    json.dump({"models": models, "overall": ov,
               "rows": [{"aspect": l, "noise": n, "signal": s, "snr": None if x == float("inf") else x,
                         "rank_stable": st} for x, l, n, s, st, _ in rows]},
              open(out / "variance.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {out/'variance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

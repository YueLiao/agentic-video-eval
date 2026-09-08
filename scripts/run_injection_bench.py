#!/usr/bin/env python3
"""Milestone-1 experiment: does the suspicion map find injected defects?

  python scripts/run_injection_bench.py --sources 'DIR/*.mp4' --out bench/m1 \
      --per-type 6 --max-frames 81

Writes results.json + report.txt and prints the table.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.synth import bench, inject  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help="glob of source videos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=4, help="cases per defect type")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--types", default=",".join(inject.DEFECT_TYPES))
    ap.add_argument("--strength", type=float, default=None,
                    help="fixed strength; default samples 0.4-0.9")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    srcs = sorted(glob.glob(args.sources))
    if not srcs:
        print(f"no sources matched {args.sources!r}", file=sys.stderr)
        return 2
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Group by source so each source's clean maps are computed once, not per case.
    plan: dict[str, list[tuple[str, inject.Defect]]] = {}
    for ti, typ in enumerate(types):
        for i in range(args.per_type):
            src = srcs[(ti * args.per_type + i) % len(srcs)]
            nf = args.max_frames or 81
            span = rng.randint(6, max(8, nf // 4))
            t0 = rng.randint(1, max(2, nf - span - 2))
            if typ in ("frame_drop", "frame_repeat"):
                bbox = None
            else:
                bw, bh = rng.uniform(0.2, 0.4), rng.uniform(0.2, 0.4)
                bbox = (rng.uniform(0, 1 - bw), rng.uniform(0, 1 - bh), bw, bh)
            s = args.strength if args.strength is not None else rng.uniform(0.4, 0.9)
            d = inject.Defect(f"d{i}", typ, (t0, t0 + span), bbox, round(s, 3),
                              {"seed": rng.randint(0, 10**6)})
            plan.setdefault(src, []).append((f"{typ}_{i:02d}", d))

    results, t_start = [], time.time()
    total = sum(len(v) for v in plan.values())
    done = 0
    for src, cases in plan.items():
        clean_maps = None
        for case_id, d in cases:
            try:
                r, clean_maps = bench.run_case(
                    src, out / "media", case_id, d,
                    max_frames=args.max_frames, clean_maps=clean_maps)
                results.append(r)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {case_id}: {type(e).__name__}: {e}", file=sys.stderr)
            done += 1
            if done % 5 == 0:
                print(f"  {done}/{total}  ({time.time()-t_start:.0f}s)", flush=True)

    if not results:
        print("no results", file=sys.stderr)
        return 1
    summary = bench.summarize(results)
    table = bench.report(summary)
    (out / "results.json").write_text(json.dumps(
        {"summary": summary, "cases": [r.to_json() for r in results],
         "sources": srcs, "args": vars(args)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.txt").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\n{len(results)} cases in {time.time()-t_start:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

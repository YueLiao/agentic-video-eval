#!/usr/bin/env python3
"""Extract content features over a split, cached to json."""
from __future__ import annotations

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_features import content_features            # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")


def _one(rel: str):
    return rel, content_features(os.path.join(ROOT, rel))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["dev", "val_ac"])
    ap.add_argument("--family", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    rels = set()
    for sp in args.splits:
        for r in csv.DictReader(open(f"{R012}/{sp}.csv")):
            if args.family and r["family"] != args.family:
                continue
            rels |= {r["path_A"], r["path_B"]}
    rels = sorted(rels)
    cache = Path(args.out); cache.parent.mkdir(parents=True, exist_ok=True)
    feats = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [r for r in rels if r not in feats]
    print(f"{len(rels)} 条视频,待提 {len(todo)}")
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (rel, f) in enumerate(ex.map(_one, todo, chunksize=4), 1):
                feats[rel] = f
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)}")
                    cache.write_text(json.dumps(feats))
        cache.write_text(json.dumps(feats))
    bad = sum(1 for v in feats.values() if "error" in v)
    print(f"完成 · 失败 {bad}/{len(feats)} · wrote {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

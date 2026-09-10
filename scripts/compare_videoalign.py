#!/usr/bin/env python3
"""Head-to-head against the trained reward model on the same pairs and metric.

VideoAlign's bench_ac is our val_ac -- 998 identical pair_ids with identical MQ
labels -- and its reported summary is computed on the first shard,
bench_shard01_p1_ac_v2.csv, whose slice counts (322 non-tie, seed 63, M-S 73,
S-S 56, W-S 74, W-M 39, W-W 9, M-M 8, upset 31, ordercheck 155) all reproduce
exactly from the tier definitions in PAIR_CONSTRUCTION.md. So the comparison is
like for like: same pairs, same labels, same metric.

    python scripts/compare_videoalign.py --model runs/pw/model.json \
        --features runs/pw/features.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agenteval.meta.videoalign import acc_with_ties, acc_without_ties  # noqa: E402
from fit_pointwise import design                                       # noqa: E402

BENCH = ("/pub/evaluation_group/cy/VideoAlign/mq_runs/bench_ac/"
         "bench_shard01_p1_ac_v2.csv")
SUMMARY = ("/pub/evaluation_group/cy/VideoAlign/mq_runs/bench_ac/"
           "bench_ac_summary.json")
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
H = {"AA": 1, "A": 1, "BB": -1, "B": -1, "same": 0}
TIER = {"wan5b": "W", "cogvideox": "W", "pangu": "M", "cosmos_nano": "M",
        "wan14b": "S", "cosmos_super": "S", "hunyuan": "S", "ltx": "S"}
RANK = {"W": 0, "M": 1, "S": 2}


def model_of(rel: str) -> str:
    parts = Path(rel).stem.split("__")
    return parts[1] if len(parts) >= 3 and parts[1] in TIER else ""


def prior(r) -> int:
    a, b = TIER.get(model_of(r["path_A"])), TIER.get(model_of(r["path_B"]))
    if not a or not b or RANK[a] == RANK[b]:
        return 0
    return 1 if RANK[a] > RANK[b] else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--run", default=None,
                    help="which run in bench_ac_summary.json (default: first)")
    args = ap.parse_args()

    summ = json.loads(Path(SUMMARY).read_text())
    key = args.run or next(iter(summ))
    va = summ[key]
    byid = {r["pair_id"]: r for r in csv.DictReader(open(f"{R012}/val_ac.csv"))}
    feats = {k: v for k, v in json.loads(Path(args.features).read_text()).items()
             if "error" not in v}
    m = json.loads(Path(args.model).read_text())

    rows = []
    for r in csv.DictReader(open(BENCH)):
        v = byid.get(r["pair_id"])
        if v and v["path_A"] in feats and v["path_B"] in feats:
            rows.append({**r, "fa": v["path_A"], "fb": v["path_B"]})
    rels = sorted({r[k] for r in rows for k in ("fa", "fb")})
    s = dict(zip(rels, (design(feats, rels) - np.array(m["mu"]))
                 / np.array(m["sd"]) @ np.array(m["w"])))
    print(f"bench p1: 可评 {len(rows)}/499 对  ·  对照运行 = {key}\n")

    def acc(sub):
        return acc_without_ties([H[r["MQ"]] for r in sub],
                                [s[r["fa"]] - s[r["fb"]] for r in sub]) * 100

    nt = [r for r in rows if H[r["MQ"]] != 0]
    print(f"{'切片':<22}{'n':>5}{'VideoAlign RM':>15}{'本仓线性模型':>15}{'差':>8}")

    def show(name, mine, theirs, n):
        print(f"{name:<22}{n:>5}{theirs:>15.1f}{mine:>15.1f}{mine - theirs:>+8.1f}")

    show("overall(非平局)", acc(nt), va["overall"], len(nt))
    star = acc_with_ties([H[r["MQ"]] for r in rows],
                         [s[r["fa"]] - s[r["fb"]] for r in rows])[0] * 100
    show("acc*(全部,扫阈值)", star, va["acc_star"], len(rows))
    up = [r for r in nt if prior(r) and prior(r) != H[r["MQ"]]]
    od = [r for r in nt if prior(r) and prior(r) == H[r["MQ"]]]
    show("upset 爆冷", acc(up), va["upset"]["acc"], len(up))
    show("ordercheck 顺风", acc(od), va["ordercheck"]["acc"], len(od))
    print()
    for fam, d in sorted(va["by_family"].items(),
                         key=lambda kv: -kv[1]["n"]):
        sub = [r for r in nt if r["family"] == fam]
        if sub:
            show(f"family {fam}", acc(sub), d["acc"], len(sub))
    print("\n  指纹受控的切片是 seed / S-S / M-S;W-* 是指纹差最大的切片。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

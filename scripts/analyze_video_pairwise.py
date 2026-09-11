#!/usr/bin/env python3
"""Break a video-pairwise run down the way the fingerprint problem demands.

A headline number on this benchmark is mostly generator recognition: accuracy
is monotone in how far apart the two generators are, and knowing only which
model made each clip scores 82% on the cross-tier pairs by itself. So the
question for any new method is not whether the total went up but whether it
went up *evenly* -- a method that only gains on W-S has learned the fingerprint
better, not the motion.

    python scripts/analyze_video_pairwise.py /tmp/vp_valac
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.meta.videoalign import (acc_with_ties,        # noqa: E402
                                       acc_without_ties)

R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
H = {"AA": 1, "A": 1, "BB": -1, "B": -1, "same": 0}
TRUTH = {"AA": "a", "A": "a", "BB": "b", "B": "b", "same": "same"}
M = {"a": 1.0, "b": -1.0, "tie": 0.0}
TIER = {"wan5b": "W", "cogvideox": "W", "pangu": "M", "cosmos_nano": "M",
        "wan14b": "S", "cosmos_super": "S", "hunyuan": "S", "ltx": "S"}
RANK = {"W": 0, "M": 1, "S": 2}


def model_of(rel: str) -> str:
    p = os.path.basename(rel).rsplit(".", 1)[0].split("__")
    return p[1] if len(p) >= 3 and p[1] in TIER else ""


def prior(r) -> int:
    a, b = TIER.get(model_of(r["path_A"])), TIER.get(model_of(r["path_B"]))
    if not a or not b or RANK[a] == RANK[b]:
        return 0
    return 1 if RANK[a] > RANK[b] else -1


def z(h, n):
    return (h / n - 0.5) / math.sqrt(0.25 / n) if n else 0.0


def main(run: Path, split: str = "val_ac"):
    res = json.loads((run / "compare.json").read_text())
    ok = {k: v for k, v in res.items() if "winner" in v}
    rows = [r for r in csv.DictReader(open(f"{R012}/{split}.csv"))
            if r["pair_id"] in ok]
    print(f"\n=== {run} · {split} ===  可评 {len(rows)}/{len(ok)} 对")

    def line(name, sub):
        if not sub:
            return
        h = [H[r["MQ"]] for r in sub]
        m = [M[ok[r["pair_id"]]["winner"]] for r in sub]
        nt = [r for r in sub if H[r["MQ"]] != 0]
        dec = [r for r in nt if ok[r["pair_id"]]["winner"] != "tie"]
        hit = sum(1 for r in dec
                  if ok[r["pair_id"]]["winner"] == TRUTH[r["MQ"]])
        flip = sum(1 for r in sub
                   if not ok[r["pair_id"]].get("order_consistent", True))
        # What the generator ranking alone would score on the same decisions --
        # the quantity the headline number has been quietly made of.
        pr = [r for r in dec if prior(r)]
        ph = sum(1 for r in pr
                 if ("a" if prior(r) > 0 else "b") == TRUTH[r["MQ"]])
        base = f"{ph/len(pr):.1%}" if pr else "--"
        print(f"  {name:<20}{len(sub):>5}{len(nt):>6}"
              f"{acc_without_ties(h, m):>10.1%}{acc_with_ties(h, m)[0]:>8.1%}"
              f"{f'{len(dec)}/{len(nt)}':>10}{hit/max(1,len(dec)):>9.1%}"
              f"{z(hit, len(dec)):>+7.2f}{base:>9}{flip/len(sub):>7.0%}")

    print(f"  {'切片':<20}{'对数':>5}{'非平局':>6}{'acc(无平局)':>10}{'acc*':>8}"
          f"{'表态':>10}{'方向':>9}{'z':>7}{'档位先验':>9}{'翻转':>7}")
    line("全部", rows)
    print()
    for fam, _ in Counter(r["family"] for r in rows).most_common():
        line(f"family {fam}", [r for r in rows if r["family"] == fam])
    print()
    line("is_ladder", [r for r in rows
                       if str(r.get("is_ladder")).lower() in ("1", "true")])
    nt = [r for r in rows if H[r["MQ"]] != 0]
    line("upset 爆冷", [r for r in nt if prior(r) and prior(r) != H[r["MQ"]]])
    line("ordercheck 顺风", [r for r in nt if prior(r) and prior(r) == H[r["MQ"]]])
    line("档位相同", [r for r in nt if not prior(r)])
    print("\n  档位先验一列 = 只看是哪个模型生成的、完全不看画面,在同一批表态对上的准确率。")
    print("  video 版要成立,必须在 upset(先验=0)和 seed/S-S(先验失效)上也站得住。")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(2)
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "val_ac")

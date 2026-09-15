#!/usr/bin/env python3
"""One place that states what every run scores, so the headline is never retyped.

Three numbers were misreported in this project by computing them by hand at the
moment of answering: a 61.1% that was position bias with the swap check
silently skipped, an 83.9% quoted from a 31-pair slice beside numbers from
hundreds, and a 67.0% that was the first of six samples rather than their vote.
Each was arithmetic done once and trusted.

Every run is read the same way here: VideoAlign's acc_without_ties over the
split's own pairs, with the aggregation each run actually uses, and n printed
beside it.

    python scripts/leaderboard.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.meta.videoalign import acc_with_ties, acc_without_ties  # noqa: E402

R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
H = {"AA": 1, "A": 1, "BB": -1, "B": -1, "same": 0}
LETTER = {"A": 1.0, "B": -1.0, "TIE": 0.0}
WORD = {"a": 1.0, "b": -1.0, "tie": 0.0}


def margin(rec: dict) -> float:
    """The run's own aggregation, not whichever field is easiest to read.

    `raw[0]` is one sample; a run that votes decides by `tally`, and reading
    raw[0] from it reports a single draw as though it were the vote -- which is
    how 57.4% was once published as 67.0%.
    """
    if "tally" in rec and rec.get("n_votes", 0) > 2:
        return float(rec["tally"])
    if rec.get("winner") in WORD and rec.get("resolved"):
        return WORD[rec["winner"]]
    raw = rec.get("raw") or []
    if raw and str(raw[0]) in LETTER:
        return LETTER[str(raw[0])]
    return WORD.get(str(rec.get("winner")), 0.0)


def main() -> int:
    rows = {r["pair_id"]: r for r in csv.DictReader(open(f"{R012}/val_ac.csv"))}
    runs = sorted(glob.glob("/tmp/*/compare.json"))
    out = []
    for p in runs:
        try:
            d = json.loads(Path(p).read_text())
        except Exception:  # noqa: BLE001
            continue
        ok = {k: v for k, v in d.items()
              if isinstance(v, dict) and "winner" in v and k in rows}
        if len(ok) < 50:
            continue
        h = [H[rows[k]["MQ"]] for k in ok]
        m = [margin(v) for v in ok.values()]
        nt = sum(1 for x in h if x != 0)
        zero = sum(1 for x in m if x == 0)
        star, eps = acc_with_ties(h, m)
        out.append((Path(p).parent.name, len(ok), nt,
                    acc_without_ties(h, m) * 100, star * 100, zero / len(m) * 100))
    out.sort(key=lambda r: -r[3])
    print(f"{'运行':<18}{'对数':>6}{'非平局':>7}{'§1 acc':>9}{'acc*':>8}"
          f"{'零分差占比':>11}")
    for name, n, nt, a, s, z in out:
        ci = 1.96 * math.sqrt(0.25 / max(1, nt)) * 100
        print(f"{name:<18}{n:>6}{nt:>7}{a:>8.1f}%{s:>7.1f}%{z:>10.0f}%"
              f"   ±{ci:.0f}")
    print("\n  §1 = VideoAlign acc_without_ties:人评平局的对不进分母,"
          "其余每对都必须给方向,分差为 0 记为错。")
    print("  对照 · cy 微调的 RM 65.5 · 官方 VideoReward 46.3(均为同口径同一批对)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

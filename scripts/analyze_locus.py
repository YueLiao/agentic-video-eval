#!/usr/bin/env python3
"""Read a locus-verdict run at several k, and against the labels that matter.

The run records every locus in suspicion-score order, so truncating the list
recovers what a smaller k would have produced -- one run answers the whole
sweep instead of one point on it, which matters because the k=3 to k=10 step on
seed pairs traded 57.1% at 17% coverage for 50.0% at 33%.

Two label populations are reported apart. Every ladder pair gives n around 1158
and a confidence interval near three points, but most of those labels are one
annotator's call; the multi-annotator subset is 119 pairs with a panel behind
each. A method that is real should hold up on both, and the panel subset is the
one to believe when they disagree.

    python scripts/analyze_locus.py /tmp/lv_ladder
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.meta.videoalign import acc_with_ties, acc_without_ties  # noqa: E402

R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
H = {"AA": 1, "A": 1, "BB": -1, "B": -1, "same": 0}
TRUTH = {"AA": "a", "A": "a", "BB": "b", "B": "b", "same": "same"}


def rows_for(ladder: bool, family: str | None):
    out, seen = [], set()
    for sp in ("train", "dev", "val_ac"):
        for r in csv.DictReader(open(f"{R012}/{sp}.csv")):
            if r["pair_id"] in seen:
                continue
            if ladder and str(r.get("is_ladder")).lower() not in ("1", "true"):
                continue
            if family and r["family"] != family:
                continue
            seen.add(r["pair_id"])
            out.append(r)
    return out


def broken_at_k(rec: dict, k: int) -> int | None:
    """How many of the top-k loci were confirmed broken."""
    loci = rec.get("loci")
    if not loci:
        return 0 if rec.get("no_locus") or loci == [] else None
    return sum(1 for x in loci[:k] if x.get("verdict") == "broken")


def report(run: Path, ladder=True, family=None):
    clips = json.loads((run / "clips.json").read_text())
    ok = {k: v for k, v in clips.items() if not v.get("error")}
    rows = [r for r in rows_for(ladder, family)
            if r["path_A"] in ok and r["path_B"] in ok]
    multi = [r for r in rows if int(r.get("n_annotations") or 1) >= 3]
    print(f"\n=== {run} ===")
    print(f"  可评 {len(rows)} 对 (其中 ≥3 人标注 {len(multi)})")
    kmax = max((len(v.get("loci") or []) for v in ok.values()), default=0)
    print(f"\n  {'k':>3}{'样本':>18}{'acc(无平局)':>12}{'可分对':>10}"
          f"{'可分对上的方向':>15}{'人评平局误表态':>14}")
    for name, sub in (("全部", rows), ("≥3人标注", multi)):
        for k in [x for x in (1, 3, 5, 10) if x <= kmax] or [kmax]:
            b = {r["pair_id"]: (broken_at_k(ok[r["path_A"]], k),
                                broken_at_k(ok[r["path_B"]], k)) for r in sub}
            use = [r for r in sub if None not in b[r["pair_id"]]]
            if not use:
                continue
            h = [H[r["MQ"]] for r in use]
            # fewer confirmed failures is better
            m = [float(b[r["pair_id"]][1] - b[r["pair_id"]][0]) for r in use]
            nt = [r for r in use if H[r["MQ"]] != 0]
            dec = [r for r in nt if b[r["pair_id"]][0] != b[r["pair_id"]][1]]
            hit = sum(1 for r in dec
                      if ("a" if b[r["pair_id"]][0] < b[r["pair_id"]][1] else "b")
                      == TRUTH[r["MQ"]])
            tie = [r for r in use if H[r["MQ"]] == 0]
            tdec = sum(1 for r in tie
                       if b[r["pair_id"]][0] != b[r["pair_id"]][1])
            print(f"  {k:>3}{name:>18}{acc_without_ties(h, m):>12.1%}"
                  f"{f'{len(dec)}/{len(nt)}':>10}"
                  f"{hit/max(1,len(dec)):>15.1%}{tdec/max(1,len(tie)):>14.0%}")
        print()
    vc: dict = {}
    for v in ok.values():
        for x in v.get("loci", []):
            vc[x.get("verdict")] = vc.get(x.get("verdict"), 0) + 1
    tot = sum(vc.values()) or 1
    print("  逐处判定: " + " · ".join(f"{k} {n} ({n/tot:.1%})"
                                      for k, n in sorted(vc.items(),
                                                         key=lambda kv: -kv[1])))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(2)
    for a in sys.argv[1:]:
        report(Path(a))

#!/usr/bin/env python3
"""Break a pairwise run down by how strongly the humans agreed.

Overall accuracy on this benchmark mixes two very different populations. The
AA/BB pairs are ones 3-5 annotators called the same way and called *clearly*;
the A/B pairs are ones they leaned on. A judge that reads the separable pairs
correctly and coin-flips the rest looks the same, in the overall number, as a
judge that coin-flips everything -- so the overall number cannot tell you
whether the ceiling is the model or the labels.

    python scripts/analyze_pairwise.py /tmp/frames_sweep/f8 [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")

STRENGTH = {"AA": "强偏好", "BB": "强偏好", "A": "弱偏好", "B": "弱偏好",
            "same": "平局"}
TRUTH = {"AA": "a", "A": "a", "BB": "b", "B": "b", "same": "same"}
PRED = {"a": "a", "b": "b", "tie": "same"}


def report(run: Path) -> None:
    d = json.loads((run / "compare.json").read_text())
    ok = {p: v for p, v in d.items() if not v.get("error")}
    print(f"\n=== {run} ===  {len(ok)} 对 (失败 {len(d) - len(ok)})")

    groups: dict[str, list] = {}
    for v in ok.values():
        groups.setdefault(STRENGTH.get(v["label"], "?"), []).append(v)

    print(f"  {'分层':<8} {'n':>4} {'方向准确率':>12} {'可判定':>9} "
          f"{'预测平局':>9} {'顺序翻转':>9}")
    for k in ("强偏好", "弱偏好", "平局"):
        g = groups.get(k, [])
        if not g:
            continue
        dec = [v for v in g if v["winner"] in ("a", "b")]
        hit = sum(1 for v in dec if v["winner"] == TRUTH[v["label"]])
        tie = sum(1 for v in g if v["winner"] == "tie")
        flip = sum(1 for v in g if not v.get("order_consistent", True))
        acc = f"{hit / len(dec):.1%}" if dec else "--"
        # On the tie stratum a direction is wrong by construction; what matters
        # there is whether the judge says tie at all.
        print(f"  {k:<8} {len(g):>4} {acc:>12} {len(dec)}/{len(g):<7} "
              f"{tie / len(g):>8.0%} {flip / len(g):>8.0%}")

    dec = [v for v in ok.values() if v["winner"] in ("a", "b")
           and v["label"] != "same"]
    hit = sum(1 for v in dec if v["winner"] == TRUTH[v["label"]])
    exact = sum(1 for v in ok.values() if PRED[v["winner"]] == TRUTH[v["label"]])
    a_pred = sum(1 for v in ok.values() if v["winner"] == "a") / len(ok)
    a_true = sum(1 for v in ok.values() if TRUTH[v["label"]] == "a") / len(ok)
    print(f"  合计: 方向 {hit / max(1, len(dec)):.1%} (n={len(dec)}) · "
          f"全体 {exact / len(ok):.1%} · 选A率 {a_pred:.0%} vs 人评 {a_true:.0%}")


def main() -> int:
    runs = [Path(a) for a in sys.argv[1:]]
    if not runs:
        print(__doc__)
        return 2
    for r in runs:
        if (r / "compare.json").exists():
            report(r)
        else:
            print(f"跳过 {r}: 没有 compare.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

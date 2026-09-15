#!/usr/bin/env python3
"""Collect every finished run into the repo, with per-pair rows others can re-score.

Results have been living in /tmp, which is where they stop existing. This writes
the leaderboard, the per-pair judgements, and the validation benches into
results/ so a number in a note can be traced to the row that produced it.

    python scripts/export_results.py --out results
"""
from __future__ import annotations

import argparse
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

RUNS = {
    "vp_valac": "video 顺序呈现,温度 0,单次(基线)",
    "wp_valac": "video + 生成提示词条件",
    "vp_vote": "video + 每序 3 次采样,共 6 票",
    "sp_valac": "上下分屏,两段同屏",
    "vp_upset": "爆冷集 689 对(档位先验为 0)",
}


def margin(rec: dict) -> float:
    if "tally" in rec and rec.get("n_votes", 0) > 2:
        return float(rec["tally"])
    raw = rec.get("raw") or []
    if raw and str(raw[0]) in LETTER:
        return LETTER[str(raw[0])]
    return WORD.get(str(rec.get("winner")), 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    meta = {r["pair_id"]: r for r in csv.DictReader(open(f"{R012}/val_ac.csv"))}
    board = []
    for name, desc in RUNS.items():
        p = Path(f"/tmp/{name}/compare.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rowsout, h, m = [], [], []
        for k, v in d.items():
            if not isinstance(v, dict) or "winner" in v is None:
                continue
            if "winner" not in v:
                continue
            mm = margin(v)
            r = meta.get(k)
            rowsout.append({
                "pair_id": k, "MQ": v.get("label") or (r or {}).get("MQ"),
                "family": (r or {}).get("family"),
                "n_annotations": v.get("n_annotations") or (r or {}).get("n_annotations"),
                "winner": v.get("winner"), "margin": mm,
                "raw": v.get("raw"), "order_consistent": v.get("order_consistent"),
                "path_A": (r or {}).get("path_A"), "path_B": (r or {}).get("path_B"),
            })
            if r:
                h.append(H[r["MQ"]]); m.append(mm)
        if not h:
            continue
        with (out / f"pairs_{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rowsout[0]))
            w.writeheader(); w.writerows(rowsout)
        nt = sum(1 for x in h if x != 0)
        star, eps = acc_with_ties(h, m)
        board.append({
            "run": name, "描述": desc, "n_pairs": len(h), "n_nontie": nt,
            "acc_without_ties": round(acc_without_ties(h, m) * 100, 1),
            "acc_star": round(star * 100, 1), "epsilon_star": round(eps, 3),
            "ci95": round(1.96 * math.sqrt(0.25 / max(1, nt)) * 100, 1),
            "zero_margin_frac": round(sum(1 for x in m if x == 0) / len(m), 3),
            "rows": f"pairs_{name}.csv",
        })
    board.sort(key=lambda r: -r["acc_without_ties"])

    extras = {}
    for src, key in (("/tmp/detmat/matrix.json", "detector_matrix"),
                     ("/tmp/dbg40/cases.json", "freeze_debug_set"),
                     ("/tmp/rt_typed16/reports.json", "routed_typed_reports"),
                     ("profiles/signals.json", "signal_calibration"),
                     ("profiles/gemma-4-31b-it.json", "capability_profile")):
        if Path(src).exists():
            dst = out / (Path(src).name if key.startswith("signal") or
                         key.startswith("capab") else f"{key}.json")
            dst.write_text(Path(src).read_text())
            extras[key] = dst.name

    (out / "leaderboard.json").write_text(json.dumps(
        {"metric": "VideoAlign METRICS.md §1 acc_without_ties / §2 acc_with_ties",
         "split": "R012 val_ac, 998 对, 587 非平局",
         "reference": {"cy 微调 RM": 65.5, "官方 VideoReward": 46.3},
         "runs": board, "extras": extras}, ensure_ascii=False, indent=1))
    print(f"{'运行':<12}{'§1':>8}{'acc*':>8}{'n':>7}  描述")
    for b in board:
        print(f"{b['run']:<12}{b['acc_without_ties']:>7.1f}%{b['acc_star']:>7.1f}%"
              f"{b['n_pairs']:>7}  {b['描述']}")
    print(f"\nwrote {out}/  ({len(board)} 个运行的逐对结果 + {len(extras)} 个验证集)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

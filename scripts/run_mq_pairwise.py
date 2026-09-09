#!/usr/bin/env python3
"""Score MQ pairs against human pairwise preference.

Runs only the motion-related skills, because this benchmark labels only MQ --
its VQ and TA columns are `invalid` throughout, so scoring faces or semantics
would cost time and be unverifiable.

Each video is scored once and cached by path: pairs share videos (same prompt,
different generators), and re-scoring one because it appears in two pairs would
be pure waste.

Reporting notes that matter more than the headline number:

* The label set is 41% `same`. A continuous score always separates two videos,
  so agreement depends entirely on the tie threshold. That threshold has to be
  fitted on dev.csv and applied to val_ac -- fitting it on the set being
  reported is choosing the answer. This script reports a sweep so the shape is
  visible, and refuses to pick one.
* Accuracy on non-tie pairs alone runs far higher and is not comparable to
  anything human agreement was measured at, so both are printed.
* Human agreement on this task, measured two-annotators-both-non-tie, is 72.8%.
  A result much above that means the evaluation leaked, not that the system is
  superhuman.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.engine.loop import LoopBudget                       # noqa: E402
from agenteval.engine.orchestrator import evaluate                 # noqa: E402
from agenteval.llm.client import VLMClient                         # noqa: E402
from agenteval.scoring.aspects import COMPOSITE_VIEWS              # noqa: E402
from agenteval.skills.motion_quality import MotionQuality          # noqa: E402
from agenteval.skills.physical_integrity import PhysicalIntegrity  # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
MOTION_ASPECTS = COMPOSITE_VIEWS["motion_plausibility"]


def sample_pairs(csv_path: str, n: int, seed: int) -> list[dict]:
    rows = list(csv.DictReader(open(csv_path)))
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["MQ"], []).append(r)
    # Keep the label mix of the full set, so the tie rate the sample sees is the
    # tie rate the real set has.
    out: list[dict] = []
    for lab, group in by_label.items():
        k = max(1, round(n * len(group) / len(rows)))
        out += rng.sample(group, min(k, len(group)))
    rng.shuffle(out)
    return out[:n]


def score_video(path: str, prompt: str, out_dir: Path, vlm: VLMClient) -> dict:
    skills = {
        "motion_quality": lambda: MotionQuality(out_dir / "motion"),
        "physical_integrity": lambda: PhysicalIntegrity(out_dir / "physical"),
    }
    res = evaluate(path, {"prompt": prompt}, skills, vlm, out_dir=out_dir,
                   budget=LoopBudget(max_rounds=3, max_vlm_calls=7,
                                     max_tool_calls=18, max_wall_s=420))
    a = res.score.aspects
    vals = [a[k].score for k in MOTION_ASPECTS
            if k in a and a[k].judgeable and a[k].score is not None]
    return {
        "motion_plausibility": min(vals) if vals else None,
        "motion_mean": (sum(vals) / len(vals)) if vals else None,
        "n_judged": len(vals),
        "aspects": {k: (a[k].score if k in a and a[k].judgeable else None)
                    for k in MOTION_ASPECTS},
        "vlm_calls": res.vlm_calls, "elapsed_s": round(res.elapsed_s, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/pub/evaluation_group/cy/mq_promptgen/"
                    "pairing/review_results/rounds/R012_20260908/val_ac.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pairs = sample_pairs(args.csv, args.n, args.seed)
    videos: dict[str, str] = {}
    for r in pairs:
        for side in ("A", "B"):
            videos.setdefault(r[f"path_{side}"], r["prompt"])
    print(f"{len(pairs)} 对, 去重后 {len(videos)} 个视频 "
          f"(共享率 {1 - len(videos)/(2*len(pairs)):.0%})", flush=True)

    cache_p = out / "video_scores.json"
    scores: dict[str, dict] = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    todo = [v for v in videos if v not in scores]
    print(f"待评测 {len(todo)} 个视频, {args.workers} 并发", flush=True)

    vlm = VLMClient(model=os.environ.get("AGENTEVAL_VLM_MODEL", "gemma-4-31b-it"),
                    base_url=os.environ.get("AGENTEVAL_VLM_BASE_URL",
                                            "http://127.0.0.1:8005/v1"),
                    max_tokens=1400, timeout_s=420,
                    cache_dir=out / "llm_cache")
    t0 = time.time(); done = [0]

    def work(rel: str):
        try:
            r = score_video(os.path.join(ROOT, rel), videos[rel],
                            out / "runs" / rel.replace("/", "_"), vlm)
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}"[:120],
                 "motion_plausibility": None}
        done[0] += 1
        if done[0] % 10 == 0:
            el = time.time() - t0
            print(f"  {done[0]}/{len(todo)}  {el:.0f}s  "
                  f"(eta {el/done[0]*(len(todo)-done[0])/60:.0f}min)", flush=True)
        return rel, r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rel, r in ex.map(work, todo):
            scores[rel] = r
            if done[0] % 20 == 0:
                cache_p.write_text(json.dumps(scores, ensure_ascii=False))
    cache_p.write_text(json.dumps(scores, ensure_ascii=False))

    n_err = sum(1 for v in scores.values() if v.get("error"))
    print(f"\n评测完成 {time.time()-t0:.0f}s, 失败 {n_err}")

    # ---- pairwise agreement -------------------------------------------
    ok = []
    for r in pairs:
        a = scores.get(r["path_A"], {}).get("motion_plausibility")
        b = scores.get(r["path_B"], {}).get("motion_plausibility")
        if a is None or b is None:
            continue
        ok.append({"pair_id": r["pair_id"], "label": r["MQ"], "family": r["family"],
                   "a": a, "b": b, "delta": a - b})
    print(f"可评估 {len(ok)}/{len(pairs)} 对")
    if not ok:
        return 1

    def truth(lab): return "same" if lab == "same" else ("a" if lab in ("A", "AA") else "b")
    nontie = [x for x in ok if x["label"] != "same"]

    print(f"\n{'τ':>6s} {'全体准确':>9s} {'非平局准确':>11s} {'预测平局率':>11s}  (人评平局率 "
          f"{sum(1 for x in ok if x['label']=='same')/len(ok):.0%})")
    best = []
    for tau in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        corr = tie_pred = 0
        for x in ok:
            p = "same" if abs(x["delta"]) <= tau else ("a" if x["delta"] > 0 else "b")
            tie_pred += p == "same"
            corr += p == truth(x["label"])
        nt = sum(1 for x in nontie
                 if ("a" if x["delta"] > 0 else "b") == truth(x["label"]))
        best.append((tau, corr / len(ok)))
        print(f"{tau:6.2f} {corr/len(ok):9.1%} {nt/max(1,len(nontie)):11.1%} "
              f"{tie_pred/len(ok):11.1%}")

    print(f"\n  非平局对方向准确率(与 τ 无关): "
          f"{sum(1 for x in nontie if ('a' if x['delta']>0 else 'b')==truth(x['label']))/max(1,len(nontie)):.1%}"
          f"   n={len(nontie)}")
    print(f"  人类上限参考: 两标注员都非 tie 时方向一致 72.8%")
    print(f"  **τ 必须在 dev.csv 上标定后再用于 val_ac,上面的扫描只用于看形状**")

    json.dump({"pairs": ok, "n_videos": len(scores), "n_err": n_err},
              open(out / "pairwise.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {out/'pairwise.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fit the aggregator and tie threshold on dev, then report once on val_ac.

Both choices have to be made somewhere, and making them on the reporting set is
choosing the answer -- comparing six aggregators on val_ac and keeping the best
would inflate the number by exactly the amount of freedom used. The R012 round
provides a dev split of 2,000 pairs with no video, prompt or pair_id overlap
with val_ac, which is what it is for.

Two phases:

  calibrate   score dev pairs, sweep aggregator x tau, pick the best pair of
              settings, write them to a config file
  report      load that config, score val_ac, report once, no further tuning

The reported number is whatever the fitted settings give. If it is worse than
the best dev number, that gap is the overfitting that a single-set workflow
would have hidden.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.engine.loop import LoopBudget                       # noqa: E402
from agenteval.engine.orchestrator import evaluate                 # noqa: E402
from agenteval.llm.client import VLMClient                         # noqa: E402
from agenteval.scoring.aggregate_fn import AGGREGATORS             # noqa: E402
from agenteval.scoring.aspects import COMPOSITE_VIEWS              # noqa: E402
from agenteval.skills.motion_quality import MotionQuality          # noqa: E402
from agenteval.skills.physical_integrity import PhysicalIntegrity  # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
KEYS = COMPOSITE_VIEWS["motion_plausibility"]
TAUS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def truth(label: str) -> str:
    return "a" if label in ("A", "AA") else ("b" if label in ("B", "BB") else "same")


def sample(csv_path: str, n: int, seed: int) -> list[dict]:
    rows = list(csv.DictReader(open(csv_path)))
    if n >= len(rows):
        return rows
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["MQ"], []).append(r)
    out: list[dict] = []
    for lab, g in by.items():
        out += rng.sample(g, min(max(1, round(n * len(g) / len(rows))), len(g)))
    rng.shuffle(out)
    return out[:n]


def score_all(pairs: list[dict], out: Path, vlm: VLMClient, workers: int) -> dict:
    cache_p = out / "video_aspects.json"
    scores: dict = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    videos = {r[f"path_{s}"]: r["prompt"] for r in pairs for s in ("A", "B")}
    todo = [v for v in videos if v not in scores]
    if not todo:
        return scores
    print(f"  评测 {len(todo)} 个视频, {workers} 并发", flush=True)

    def one(rel: str):
        d = out / "runs" / rel.replace("/", "_")
        skills = {"motion_quality": lambda: MotionQuality(d / "m"),
                  "physical_integrity": lambda: PhysicalIntegrity(d / "p")}
        try:
            res = evaluate(os.path.join(ROOT, rel), {"prompt": videos[rel]},
                           skills, vlm, out_dir=d,
                           budget=LoopBudget(max_rounds=3, max_vlm_calls=7,
                                             max_tool_calls=18, max_wall_s=420))
            a = res.score.aspects
            return rel, {k: (a[k].score if k in a and a[k].judgeable else None)
                         for k in KEYS}
        except Exception as e:  # noqa: BLE001
            return rel, {"_error": f"{type(e).__name__}: {e}"[:100]}

    # Smoke one before the batch: a long job that cannot fail fast fails slowly.
    rel0, r0 = one(todo[0])
    if r0.get("_error") or not any(v is not None for k, v in r0.items() if k != "_error"):
        print(f"  冒烟失败,中止:{r0}", file=sys.stderr)
        raise SystemExit(2)
    scores[rel0] = r0
    print(f"  冒烟通过 → 批量开始", flush=True)

    t0 = time.time(); done = [1]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rel, r in ex.map(one, todo[1:]):
            scores[rel] = r
            done[0] += 1
            if done[0] % 20 == 0:
                el = time.time() - t0
                print(f"    {done[0]}/{len(todo)}  {el:.0f}s", flush=True)
    cache_p.write_text(json.dumps(scores, ensure_ascii=False))
    return scores


def evaluate_settings(pairs, scores, agg_name, tau):
    fn = AGGREGATORS[agg_name]
    corr = n = 0
    for r in pairs:
        va = [v for k, v in (scores.get(r["path_A"]) or {}).items()
              if k in KEYS and isinstance(v, (int, float))]
        vb = [v for k, v in (scores.get(r["path_B"]) or {}).items()
              if k in KEYS and isinstance(v, (int, float))]
        if not va or not vb:
            continue
        d = fn(va) - fn(vb)
        pred = "same" if abs(d) <= tau else ("a" if d > 0 else "b")
        corr += pred == truth(r["MQ"]); n += 1
    return (corr / n if n else 0.0), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["calibrate", "report"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg_p = Path(args.config) if args.config else out.parent / "mq_config.json"

    vlm = VLMClient(model=os.environ.get("AGENTEVAL_VLM_MODEL", "gemma-4-31b-it"),
                    base_url=os.environ.get("AGENTEVAL_VLM_BASE_URL",
                                            "http://127.0.0.1:8005/v1"),
                    max_tokens=1400, timeout_s=420, cache_dir=out / "llm_cache")

    if args.phase == "calibrate":
        pairs = sample(f"{R012}/dev.csv", args.n, seed=1)
        print(f"dev 标定集 {len(pairs)} 对")
        scores = score_all(pairs, out, vlm, args.workers)
        best = None
        print(f"\n{'聚合':16s} " + " ".join(f"{t:>6.2f}" for t in TAUS))
        for name in AGGREGATORS:
            row = []
            for tau in TAUS:
                acc, n = evaluate_settings(pairs, scores, name, tau)
                row.append(acc)
                if best is None or acc > best[0]:
                    best = (acc, name, tau, n)
            print(f"  {name:16s} " + " ".join(f"{a:6.1%}" for a in row))
        acc, name, tau, n = best
        print(f"\n  dev 最佳:{name} @ tau={tau}  准确率 {acc:.1%}  (n={n})")
        cfg_p.write_text(json.dumps(
            {"aggregator": name, "tau": tau, "dev_accuracy": round(acc, 4),
             "dev_n": n, "fitted_on": "dev.csv", "fitted_at": time.time()},
            ensure_ascii=False, indent=1))
        print(f"  wrote {cfg_p}")
        print("  **这是标定集上的数,不是成绩。报数请跑 --phase report**")
        return 0

    cfg = json.loads(cfg_p.read_text())
    print(f"载入标定:{cfg['aggregator']} @ tau={cfg['tau']} "
          f"(dev 上 {cfg['dev_accuracy']:.1%})")
    pairs = sample(f"{R012}/val_ac.csv", args.n, seed=2)
    print(f"val_ac 报数集 {len(pairs)} 对")
    scores = score_all(pairs, out, vlm, args.workers)
    acc, n = evaluate_settings(pairs, scores, cfg["aggregator"], cfg["tau"])

    fn = AGGREGATORS[cfg["aggregator"]]
    nontie = [r for r in pairs if r["MQ"] != "same"]
    dec = hit = 0
    for r in nontie:
        va = [v for k, v in (scores.get(r["path_A"]) or {}).items()
              if k in KEYS and isinstance(v, (int, float))]
        vb = [v for k, v in (scores.get(r["path_B"]) or {}).items()
              if k in KEYS and isinstance(v, (int, float))]
        if not va or not vb:
            continue
        d = fn(va) - fn(vb)
        if abs(d) < 1e-9:
            continue
        dec += 1; hit += ("a" if d > 0 else "b") == truth(r["MQ"])

    print(f"\n  === val_ac 报数 (设置来自 dev,未在此集上调过) ===")
    print(f"  全体准确率(含平局)      {acc:.1%}   n={n}")
    print(f"  非平局方向准确率        {hit/max(1,dec):.1%}   n={dec}/{len(nontie)}")
    print(f"  dev 上同一设置          {cfg['dev_accuracy']:.1%}")
    print(f"  差值 {acc - cfg['dev_accuracy']:+.1%}  ← 这就是单集流程会藏起来的过拟合量")
    print(f"  人类上限参考 72.8% · 随机 50%(方向) / 41%(全猜平局)")
    json.dump({"config": cfg, "overall_acc": acc, "n": n,
               "nontie_acc": hit / max(1, dec), "nontie_n": dec},
              open(out / "report.json", "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

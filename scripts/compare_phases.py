#!/usr/bin/env python3
"""Does multi-phase consensus improve agreement with human preference?

Consensus was built to fix a measured problem: shifting which frames get
sampled changed every aspect that carried signal, with noise exceeding
between-model signal throughout and the model ranking taking three different
orders in three runs. It discards 29-44% of findings as single-phase artifacts.

But it has never been checked against human labels, which is the only thing
that says whether it removes noise or removes signal. This runs the identical
pairs both ways -- single pass and three-phase consensus -- so the difference is
attributable to consensus alone.

Three outcomes, all informative:
  accuracy up    consensus removes noise; worth 3x the calls
  accuracy flat  it removes noise the human labels were not sensitive to
  accuracy down  it removes signal, and the discarded findings were real
"""
from __future__ import annotations

# Line-buffer stdout. Redirected to a file, Python block-buffers at 4KB, so a
# long-running job shows nothing for many minutes while mediapipe's C-level
# stderr writes through unbuffered -- the log fills with library noise and none
# of the progress or the smoke-test result. Whether a batch is healthy has to be
# visible while it runs, not only after it ends.
import sys as _sys

_sys.stdout.reconfigure(line_buffering=True)

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
from agenteval.scoring.aggregate_fn import AGGREGATORS             # noqa: E402
from agenteval.scoring.aspects import COMPOSITE_VIEWS              # noqa: E402
from agenteval.skills.motion_quality import MotionQuality          # noqa: E402
from agenteval.skills.physical_integrity import PhysicalIntegrity  # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
KEYS = COMPOSITE_VIEWS["motion_plausibility"]


def truth(l: str) -> str:
    return "a" if l in ("A", "AA") else ("b" if l in ("B", "BB") else "same")


def score(rel: str, prompt: str, out: Path, vlm: VLMClient, phases: int) -> dict:
    d = out / rel.replace("/", "_")
    skills = {"motion_quality": lambda: MotionQuality(d / "m"),
              "physical_integrity": lambda: PhysicalIntegrity(d / "p")}
    ph = tuple(round(k / phases, 3) for k in range(phases))
    try:
        res = evaluate(os.path.join(ROOT, rel), {"prompt": prompt}, skills, vlm,
                       out_dir=d, phases=ph,
                       budget=LoopBudget(max_rounds=3, max_vlm_calls=7,
                                         max_tool_calls=18, max_wall_s=420))
        a = res.score.aspects
        r = {k: (a[k].score if k in a and a[k].judgeable else None) for k in KEYS}
        if res.consensus:
            r["_discard_rate"] = res.consensus.get("discard_rate")
        return r
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"[:100]}


def accuracy(pairs, scores, agg="mean"):
    fn = AGGREGATORS[agg]
    out = {}
    for grp, labs in (("nontie", ("A", "B", "AA", "BB")),
                      ("slight", ("A", "B")), ("strong", ("AA", "BB"))):
        dec = hit = 0
        for r in pairs:
            if r["MQ"] not in labs:
                continue
            va = [v for k, v in (scores.get(r["path_A"]) or {}).items()
                  if k in KEYS and isinstance(v, (int, float))]
            vb = [v for k, v in (scores.get(r["path_B"]) or {}).items()
                  if k in KEYS and isinstance(v, (int, float))]
            if not va or not vb:
                continue
            d = fn(va) - fn(vb)
            if abs(d) < 1e-9:
                continue
            dec += 1
            hit += ("a" if d > 0 else "b") == truth(r["MQ"])
        out[grp] = (hit / dec if dec else 0.0, dec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--phases", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="/tmp/mq_dev/video_aspects.json",
                    help="existing single-phase scores to compare against")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    base = json.loads(Path(args.baseline).read_text()) if Path(args.baseline).exists() else {}
    rows = list(csv.DictReader(open(f"{R012}/dev.csv")))
    # Only pairs whose single-phase scores already exist, so the comparison is
    # on identical pairs rather than two different samples.
    usable = [r for r in rows if r["path_A"] in base and r["path_B"] in base]
    rng = random.Random(3); rng.shuffle(usable)
    pairs = usable[: args.n]
    print(f"对照集 {len(pairs)} 对(单相位分数已存在,故两组跑的是同一批)")

    vids = {r[f"path_{s}"]: r["prompt"] for r in pairs for s in ("A", "B")}
    cache_p = out / f"phase{args.phases}.json"
    got: dict = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    todo = [v for v in vids if v not in got]
    vlm = VLMClient(model=os.environ.get("AGENTEVAL_VLM_MODEL", "gemma-4-31b-it"),
                    base_url=os.environ.get("AGENTEVAL_VLM_BASE_URL",
                                            "http://127.0.0.1:8005/v1"),
                    max_tokens=1400, timeout_s=600, cache_dir=out / "llm_cache")
    if todo:
        print(f"{args.phases} 相位评测 {len(todo)} 个视频 ...")
        rel0 = todo[0]
        r0 = score(rel0, vids[rel0], out / "runs", vlm, args.phases)
        if r0.get("_error"):
            print(f"冒烟失败:{r0}", file=sys.stderr); return 2
        got[rel0] = r0
        print(f"冒烟通过 → 批量开始", flush=True)
        t0 = time.time(); k = [1]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rel, r in zip(todo[1:], ex.map(
                    lambda v: score(v, vids[v], out / "runs", vlm, args.phases),
                    todo[1:])):
                got[rel] = r; k[0] += 1
                if k[0] % 10 == 0:
                    print(f"  {k[0]}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
        cache_p.write_text(json.dumps(got, ensure_ascii=False))

    dr = [v["_discard_rate"] for v in got.values()
          if isinstance(v.get("_discard_rate"), (int, float))]
    print(f"\n共识丢弃率: 均值 {sum(dr)/len(dr):.1%}" if dr else "\n(无共识统计)")

    a1, a3 = accuracy(pairs, base), accuracy(pairs, got)
    print(f"\n  {'':10s} {'单相位':>18s} {'%d 相位共识' % args.phases:>18s}")
    for grp, name in (("nontie", "非平局"), ("slight", "小赢"), ("strong", "大赢")):
        (p1, n1), (p3, n3) = a1[grp], a3[grp]
        arrow = "↑" if p3 > p1 else ("↓" if p3 < p1 else "=")
        print(f"  {name:10s} {p1:12.1%}(n={n1:3d}) {p3:12.1%}(n={n3:3d})  {arrow}")
    print(f"\n  随机 50% · 人类上限 72.8%")
    json.dump({"single": a1, "consensus": a3, "n_pairs": len(pairs),
               "discard_rate": (sum(dr)/len(dr)) if dr else None},
              open(out / "compare.json", "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score each clip alone against a defect checklist, then read the pair off the scores.

Direct comparison is failing here for a reason the capability probes make
concrete: gemma-4-31b passes counting, grounding, timestamp reading, buried
constraints, and -- importantly -- does not invent defects on a blank image,
yet its "which clip's motion is better" answer flips 53% of the time when the
two clips change places. It has the perception; it has no calibrated prior for
an aesthetic preference. So it is asked what it can answer: does *this* clip
contain *this* concrete defect, and how bad.

Two properties fall out that the comparison mode cannot have:

  * no position bias is possible -- each clip is seen alone;
  * the output is point-wise by construction, and the pair direction is a
    consequence of the scores rather than the other way round.

    python scripts/run_checklist_pairwise.py --split dev --n 100 --out runs/chk
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

from agenteval.llm import modes                          # noqa: E402
from agenteval.llm.client import ImageRef, VLMClient     # noqa: E402
from agenteval.media.clip import VideoHandle             # noqa: E402
from agenteval.tools.renders import filmstrip            # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
TRUTH = {"AA": "a", "A": "a", "BB": "b", "B": "b", "same": "same"}

SYSTEM = """\
你在检查一段 **AI 生成视频**的运动是否有缺陷。

你看到的是按时间顺序均匀采样的帧,每帧左上角标了时间。
**只看运动**,不要评价画面美观、清晰度、色彩、构图。

重要:大多数片段在大多数项上是**没有问题**的。只有你能在具体某一帧、
某个部位指出现象时才判为有问题;指不出来就判 `无`。
**不要为了显得认真而找问题。**
"""

# Each item is something a person could point at in a specific frame. Vague
# items ("是否自然") are what the comparison mode was already failing at.
ITEMS: list[tuple[str, str, float]] = [
    ("freeze",       "画面卡住或重复:相邻两帧几乎完全相同,而这段时间本应有运动", 1.5),
    ("jump",         "跳变或瞬移:主体位置在相邻两帧间突然改变,中间没有过渡", 1.5),
    ("limb_deform",  "人或动物的肢体在运动中变长/变短/数量变化/反关节弯折", 2.0),
    ("rigid_deform", "本应保持刚性的物体(车、桌、建筑)在运动中扭曲变形", 1.5),
    ("appear_vanish", "有物体凭空出现或凭空消失,没有进出画面的过程", 1.5),
    ("penetration",  "穿模:两个物体互相穿过而不是碰撞或遮挡", 1.5),
    ("float",        "无支撑悬浮,或脚在地面上打滑、迈步与位移对不上", 1.5),
    ("puppet",       "动作像被逐帧摆出来的:没有惯性、重心不随支撑脚转移", 1.5),
    ("speed",        "速度忽快忽慢,或运动的快慢与该动作的常识不符", 1.0),
    ("amplitude",    "该动的主体几乎没有动,整段近乎静止", 1.0),
]
OPTIONS = ("无", "轻微", "明显", "严重")
PENALTY = {"无": 0.0, "轻微": 0.25, "明显": 0.6, "严重": 1.0}
CONTEXT = ("下面是这段视频按时间顺序均匀采样的帧(可能分成多张图,按先后接续)。\n"
           "请逐项检查下列运动缺陷。")


def score_clip(items: dict) -> float:
    """10 minus the accumulated penalty, floored at 0.

    Weighted sum rather than worst-of: a clip with one severe defect and a clip
    with five moderate ones are both bad, and worst-of calls them equal.
    """
    pen = sum(w * PENALTY.get(str(items.get(k, "无")).strip(), 0.0)
              for k, _d, w in ITEMS)
    return max(0.0, 10.0 - pen)


def sample(path: str, n: int, seed: int) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["MQ"], []).append(r)
    out: list[dict] = []
    for lab, g in by.items():
        out += rng.sample(g, min(max(1, round(n * len(g) / len(rows))), len(g)))
    rng.shuffle(out)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "val_ac"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.3,
                    help="score gap below which the pair is called a tie")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    pairs = sample(f"{R012}/{args.split}.csv", args.n, seed=11)
    vids = sorted({r[k] for r in pairs for k in ("path_A", "path_B")})
    print(f"{args.split}: {len(pairs)} 对, {len(vids)} 条视频")

    vlm = VLMClient(model=os.environ.get("AGENTEVAL_VLM_MODEL", "gemma-4-31b-it"),
                    base_url=os.environ.get("AGENTEVAL_VLM_BASE_URL",
                                            "http://127.0.0.1:8005/v1"),
                    max_tokens=900, timeout_s=300, cache_dir=out / "llm_cache")

    cache_p = out / "clips.json"
    done: dict = json.loads(cache_p.read_text()) if cache_p.exists() else {}

    def one(rel: str):
        if rel in done:
            return rel, done[rel]
        try:
            v = VideoHandle(os.path.join(ROOT, rel))
            strip = filmstrip(v, out / "strips", t0=0, t1=v.total - 1,
                              n=args.frames, cols=8, side=400)
            if not strip.images:
                return rel, {"error": "no strip"}
            obs = modes.enumerate_items(
                vlm, items=[(k, d) for k, d, _w in ITEMS], context=CONTEXT,
                images=[ImageRef(path=p, caption="采样帧") for p in strip.images],
                options=OPTIONS, system=SYSTEM, tag=f"chk/{rel}")
            if not obs.ok:
                return rel, {"error": obs.error or "enum failed"}
            got = obs.get("items") or {}
            if not isinstance(got, dict):
                return rel, {"error": f"items not a dict: {type(got).__name__}"}
            missing = [k for k, _d, _w in ITEMS if k not in got]
            return rel, {"items": got, "score": round(score_clip(got), 2),
                         "missing": missing}
        except Exception as e:  # noqa: BLE001
            return rel, {"error": f"{type(e).__name__}: {e}"[:100]}

    todo = [v for v in vids if v not in done]
    if todo:
        rel0, r0 = one(todo[0])
        if r0.get("error"):
            print(f"冒烟失败:{r0}", file=sys.stderr); return 2
        done[rel0] = r0
        print(f"冒烟通过 (score={r0['score']}, 缺项={r0['missing']}) → 批量开始")
        t0 = time.time(); k = 1
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rel, res in ex.map(one, todo[1:]):
                done[rel] = res; k += 1
                if k % 25 == 0:
                    print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s")
        cache_p.write_text(json.dumps(done, ensure_ascii=False))

    ok = {k: v for k, v in done.items() if not v.get("error")}
    print(f"\n完成 {len(ok)}/{len(done)} 条视频")

    scores = sorted(v["score"] for v in ok.values())
    n_missing = sum(1 for v in ok.values() if v["missing"])
    print(f"  分数分布: 最低 {scores[0]:.2f} 中位 {scores[len(scores)//2]:.2f} "
          f"最高 {scores[-1]:.2f} · 不同取值 {len(set(scores))} 个 · "
          f"漏项片段 {n_missing}")
    rate: dict[str, int] = {}
    for v in ok.values():
        for k, _d, _w in ITEMS:
            if str(v["items"].get(k, "无")).strip() != "无":
                rate[k] = rate.get(k, 0) + 1
    print("  各项报出率: " + " ".join(f"{k}={rate.get(k,0)/len(ok):.0%}"
                                      for k, _d, _w in ITEMS))

    strata = {"强偏好": ("AA", "BB"), "弱偏好": ("A", "B"), "平局": ("same",)}
    print(f"\n  === 点分之差预测成对方向 (|差| < {args.margin} 判平局) ===")
    tot_hit = tot_dec = 0
    for sname, labs in strata.items():
        hit = dec = n = 0
        for r in pairs:
            if r["MQ"] not in labs:
                continue
            a, b = ok.get(r["path_A"]), ok.get(r["path_B"])
            if not a or not b:
                continue
            n += 1
            d = a["score"] - b["score"]
            if abs(d) < args.margin:
                continue
            dec += 1
            hit += ("a" if d > 0 else "b") == TRUTH[r["MQ"]]
        acc = f"{hit/dec:6.1%}" if dec else "    --"
        print(f"  {sname:<6} n={n:<4} 方向 {acc} (可判定 {dec}/{n})")
        if labs != ("same",):
            tot_hit += hit; tot_dec += dec
    print(f"  非平局合计 方向 {tot_hit/max(1,tot_dec):.1%} (n={tot_dec})")
    (out / "result.json").write_text(json.dumps(
        {"scores": {k: v.get("score") for k, v in ok.items()},
         "nontie_direction": tot_hit / max(1, tot_dec), "n": tot_dec},
        ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Ask the model to compare two clips directly, then fit a point-wise scale.

The benchmark is pairwise and the framework was answering it by scoring each
clip alone and subtracting -- a detour that introduces magnitude, which the
model gives badly, when the question only needed direction. So this asks the
comparison directly, and rebuilds the magnitude afterwards with Bradley-Terry,
which is what turns many low-information directions into a scale whose spacing
means something.

Reports three things:

  compare accuracy   direct pairwise, against the same human labels
  BT self-agreement  how well the fitted scale reproduces its own input
  BT held-out        the number that matters -- fit on one half, test on the
                     other, since reproducing data you were fitted on is not
                     evidence of anything
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

from agenteval.llm import modes                                # noqa: E402
from agenteval.llm.client import ImageRef, VLMClient           # noqa: E402
from agenteval.media.clip import VideoHandle                   # noqa: E402
from agenteval.scoring.bradley_terry import agreement, from_labels  # noqa: E402
from agenteval.tools.pairview import aligned_pair, motion_pair  # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")

SYSTEM = """\
你在比较两段 **AI 生成视频**的**运动质量**,判断哪一段的运动更合理。

判断依据(按重要性):
1. **动作自然度** —— 人和物的动作是否符合真实运动规律:有惯性、重心随支撑转移、
   关节朝向可能。像被逐帧摆出来的木偶是严重问题。
2. **运动连贯性** —— 有无卡顿、抽帧、跳变、忽快忽慢。
3. **物理合理性** —— 下落是否匀加速、刚体是否保持、有无穿模或无支撑悬浮。
4. **运动幅度** —— 该动的东西是否真的动了,幅度是否与场景相称。

**不要**根据画面美观、清晰度、色彩、构图来判断——只看**运动**。

两段视频内容相同(同一条提示词生成),所以请专注于运动表现的差异。
如果两段的运动质量确实相当,就返回 tie,**不要为了给出区分而勉强选一边**。
"""

QUESTION = ("上排是视频 A,下排是视频 B,按相同时间比例采样,同一列是同一时刻。\n"
            "逐列对比两段视频的**运动质量**,判断哪一段更好。")


def truth(label: str) -> str:
    return "a" if label in ("A", "AA") else ("b" if label in ("B", "BB") else "same")


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
    ap.add_argument("--motion-curves", action="store_true",
                    help="也给出两段视频的运动曲线对照")
    ap.add_argument("--no-swap", action="store_true",
                    help="skip the order-swap check (halves cost)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    pairs = sample(f"{R012}/{args.split}.csv", args.n, seed=11)
    print(f"{args.split}: {len(pairs)} 对")
    vlm = VLMClient(model=os.environ.get("AGENTEVAL_VLM_MODEL", "gemma-4-31b-it"),
                    base_url=os.environ.get("AGENTEVAL_VLM_BASE_URL",
                                            "http://127.0.0.1:8005/v1"),
                    max_tokens=900, timeout_s=300, cache_dir=out / "llm_cache")

    cache_p = out / "compare.json"
    done: dict = json.loads(cache_p.read_text()) if cache_p.exists() else {}

    def one(r: dict):
        pid = r["pair_id"]
        if pid in done:
            return pid, done[pid]
        try:
            a = VideoHandle(os.path.join(ROOT, r["path_A"]))
            b = VideoHandle(os.path.join(ROOT, r["path_B"]))
            view = aligned_pair(a, b, out / "views", n=args.frames)
            if not view.images:
                return pid, {"error": "no view"}
            imgs = [ImageRef(path=p, caption="A/B 逐时刻对照") for p in view.images]
            note = view.hint
            if args.motion_curves:
                mv = motion_pair(a, b, out / "views")
                if mv.images:
                    imgs += [ImageRef(path=p, caption="A/B 运动曲线")
                             for p in mv.images]
                    note += "\n\n" + mv.hint
            obs = modes.compare(vlm, question=QUESTION + "\n\n" + note, images=imgs,
                                system=SYSTEM, tag=f"cmp/{pid}",
                                n_a=0 if args.no_swap else len(imgs) // 2,
                                swap_check=not args.no_swap)
            return pid, {"winner": str(obs.get("winner", "tie")).lower(),
                         "margin": obs.get("margin"),
                         "reason": str(obs.get("reason", ""))[:220],
                         "order_consistent": obs.parsed.get("_order_consistent", True),
                         "label": r["MQ"]}
        except Exception as e:  # noqa: BLE001
            return pid, {"error": f"{type(e).__name__}: {e}"[:100]}

    todo = [r for r in pairs if r["pair_id"] not in done]
    if todo:
        pid0, r0 = one(todo[0])
        if r0.get("error"):
            print(f"冒烟失败:{r0}", file=sys.stderr); return 2
        done[pid0] = r0
        print(f"冒烟通过 (winner={r0['winner']}) → 批量开始", flush=True)
        t0 = time.time(); k = [1]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for pid, res in ex.map(one, todo[1:]):
                done[pid] = res; k[0] += 1
                if k[0] % 20 == 0:
                    print(f"  {k[0]}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
        cache_p.write_text(json.dumps(done, ensure_ascii=False))

    ok = {p: v for p, v in done.items() if not v.get("error")}
    n_err = len(done) - len(ok)
    print(f"\n完成 {len(ok)} 对 (失败 {n_err})")

    hit = sum(1 for v in ok.values()
              if {"a": "a", "b": "b", "tie": "same"}.get(v["winner"]) == truth(v["label"]))
    nontie = {p: v for p, v in ok.items() if v["label"] != "same"}
    dec = {p: v for p, v in nontie.items() if v["winner"] in ("a", "b")}
    nt = sum(1 for v in dec.values() if v["winner"] == truth(v["label"]))
    tie_rate = sum(1 for v in ok.values() if v["label"] == "same") / max(1, len(ok))
    pred_tie = sum(1 for v in ok.values() if v["winner"] == "tie") / max(1, len(ok))
    flip = sum(1 for v in ok.values() if not v.get("order_consistent", True))

    print(f"\n  === 直接成对比较 ===")
    print(f"  全体准确率(含平局)   {hit/max(1,len(ok)):6.1%}   n={len(ok)}"
          f"   (平凡基线 {tie_rate:.1%})")
    print(f"  非平局方向准确率     {nt/max(1,len(dec)):6.1%}   n={len(dec)}/{len(nontie)}")
    print(f"  预测平局率 {pred_tie:.1%} vs 人评 {tie_rate:.1%}"
          f"   ·  顺序翻转(位置偏置) {flip}/{len(ok)}")

    # Bradley-Terry over the model's own comparisons -> a point-wise scale
    rowmap = {r["pair_id"]: r for r in pairs}
    tri = [(rowmap[p]["path_A"], rowmap[p]["path_B"],
            {"a": "a", "b": "b", "tie": "same"}[v["winner"]])
           for p, v in ok.items() if p in rowmap]
    bt = from_labels(tri)
    rng = random.Random(5); idxs = list(range(len(tri))); rng.shuffle(idxs)
    half = len(idxs) // 2
    bt_tr = from_labels([tri[i] for i in idxs[:half]])
    held = [tri[i] for i in idxs[half:]]
    print(f"\n  === Bradley-Terry(把方向判断变成 point-wise 尺度) ===")
    print(f"  {bt.n_items} 个视频, {bt.n_comparisons} 次比较")
    print(f"  自洽准确率(拟合集上)  {agreement(bt, tri)['accuracy']:6.1%}  ← 不算证据")
    print(f"  留出准确率(半数拟合)  {agreement(bt_tr, held)['accuracy']:6.1%}"
          f"  n={agreement(bt_tr, held)['n']}  ← 这个才算")
    sc = bt.to_scale()
    if sc:
        vs = sorted(sc.values())
        print(f"  point-wise 分布: 最低 {vs[0]:.1f} 中位 {vs[len(vs)//2]:.1f} 最高 {vs[-1]:.1f}")

    json.dump({"compare": ok, "bt_scores": bt.scores,
               "bt_scale": sc, "n_err": n_err},
              open(out / "result.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {out/'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

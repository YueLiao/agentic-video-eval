#!/usr/bin/env python3
"""Compare two clips at the places a signal says are suspicious, magnified.

Every earlier attempt gave the model more evidence at the same resolution and
none of them moved the seed-family number: 8, 16 and 32 frames all sat between
48% and 58% with half the answers flipping under an order swap, and a per-clip
defect checklist returned identical answers on 55-57% of seed pairs. The
capability probe explains it -- this model's legibility floor is 448px, and the
processor's 280-soft-token budget turns an eight-column strip into 264px per
frame. At that size 'both clips look fine' is the honest answer.

So this spends the same budget on fewer moments seen properly. The suspicion
signals run at full frame rate, nominate the three most suspicious loci per
clip, and each is cropped and rendered two consecutive moments to an image at
528px effective. The question changes with it: not 'does this clip have a
defect' but 'of these two magnified moments, which is worse' -- which is the
shape the probes say this model is good at.

    python scripts/run_locus_pairwise.py --split dev --family seed --out runs/loc
"""
from __future__ import annotations

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

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

from agenteval.llm import modes                         # noqa: E402
from agenteval.llm.client import ImageRef, VLMClient    # noqa: E402
from agenteval.media.clip import VideoHandle            # noqa: E402
from agenteval.tools.locus_view import worst_loci       # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
TRUTH = {"AA": "a", "A": "a", "BB": "b", "B": "b", "same": "same"}

SYSTEM = """\
你在比较两段 **AI 生成视频**的运动质量。

你看到的不是整段视频,而是**信号挑出来的、每段各自最可疑的几处**,已经裁剪并放大。
每张图是**相邻的两个时刻**,格上标了帧号。

**信号只知道"这里的变化无法用运动解释",不知道这是什么。**遮挡、转向、光影变化都会
触发它。所以先逐处确认:同一个物体在相邻两格之间,形状、边缘、纹理是**怎么变的**?
是真的崩坏(肢体断裂、手指融合、物体扭曲、纹理糊成一团、边界撕裂),还是正常的
遮挡/转向/明暗变化?

然后判断:**哪一段的可疑处更严重**。
- 只有一边有真实崩坏 → 选另一边
- 两边都有 → 比严重程度和显眼程度
- 两边都只是正常变化,或严重程度确实相当 → 返回 tie,**不要勉强分出胜负**

只看这些放大处的画面质量与结构完整性,不要评价构图、色彩、内容好不好看。
"""

QUESTION = ("下面先是视频 A 的可疑处,再是视频 B 的可疑处(每张图的标题已注明)。\n"
            "逐处确认后,判断哪一段的运动/结构问题更轻。")


def sample_from(rows, n, seed):
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["MQ"], []).append(r)
    out = []
    for lab, g in by.items():
        out += rng.sample(g, min(max(1, round(n * len(g) / len(rows))), len(g)))
    rng.shuffle(out)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--family", default="seed")
    ap.add_argument("--n", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--k", type=int, default=3, help="loci per clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--endpoints", nargs="*",
                    default=["http://127.0.0.1:8005/v1"])
    ap.add_argument("--model", default="gemma-4-31b-it")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    rows = list(csv.DictReader(open(f"{R012}/{args.split}.csv")))
    if args.family:
        rows = [r for r in rows if r["family"] == args.family]
    if args.n:
        rows = sample_from(rows, args.n, seed=11)
    print(f"{args.split}/{args.family}: {len(rows)} 对 · "
          f"endpoints {len(args.endpoints)}")

    vlms = [VLMClient(model=args.model, base_url=ep, max_tokens=1200,
                      timeout_s=420, cache_dir=out / "llm_cache")
            for ep in args.endpoints]

    cache_p = out / "compare.json"
    done = json.loads(cache_p.read_text()) if cache_p.exists() else {}

    def views(v, out_dir, tag):
        r = worst_loci(v, out_dir, k=args.k, tag=tag)
        return r

    def one(job):
        i, r = job
        pid = r["pair_id"]
        if pid in done:
            return pid, done[pid]
        vlm = vlms[i % len(vlms)]
        try:
            a = VideoHandle(os.path.join(ROOT, r["path_A"]))
            b = VideoHandle(os.path.join(ROOT, r["path_B"]))
            va, vb = views(a, out / "loci", "A"), views(b, out / "loci", "B")
            if not va.images or not vb.images:
                return pid, {"error": "no loci"}
            ia = [ImageRef(path=p, caption="视频A 可疑处") for p in va.images]
            ib = [ImageRef(path=p, caption="视频B 可疑处") for p in vb.images]
            note = (va.hint + "\n\n信号给出的可疑度(仅供参考,不是结论):"
                    f"A 最高 {va.value.get('top_score')} · "
                    f"B 最高 {vb.value.get('top_score')}")
            obs = modes.compare(
                vlm, question=QUESTION + "\n\n" + note, images=ia + ib,
                system=SYSTEM, tag=f"loc/{pid}",
                # Swap by re-rendering the sides, so the captions move with them.
                swapped_images=[ImageRef(path=p.path, caption="视频A 可疑处")
                                for p in ib]
                + [ImageRef(path=p.path, caption="视频B 可疑处") for p in ia])
            if not obs.ok:
                return pid, {"error": obs.error or "compare failed"}
            return pid, {"winner": str(obs.get("winner", "tie")).lower(),
                         "margin": obs.get("margin"),
                         "reason": str(obs.get("reason", ""))[:240],
                         "order_consistent": obs.parsed.get("_order_consistent", True),
                         "n_img": len(ia) + len(ib),
                         "label": r["MQ"]}
        except Exception as e:  # noqa: BLE001
            return pid, {"error": f"{type(e).__name__}: {e}"[:110]}

    todo = [(i, r) for i, r in enumerate(rows) if r["pair_id"] not in done]
    if todo:
        pid0, r0 = one(todo[0])
        if r0.get("error"):
            print(f"冒烟失败:{r0}", file=sys.stderr); return 2
        done[pid0] = r0
        print(f"冒烟通过 (winner={r0['winner']}, {r0['n_img']} 张图) → 批量开始")
        t0 = time.time(); k = 1
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for pid, res in ex.map(one, todo[1:]):
                done[pid] = res; k += 1
                if k % 25 == 0:
                    print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s")
                    cache_p.write_text(json.dumps(done, ensure_ascii=False))
        cache_p.write_text(json.dumps(done, ensure_ascii=False))

    from agenteval.meta.videoalign import acc_with_ties, acc_without_ties
    H = {"AA": 1, "A": 1, "BB": -1, "B": -1, "same": 0}
    M = {"a": 1.0, "b": -1.0, "tie": 0.0}
    ok = {p: v for p, v in done.items() if not v.get("error")}
    print(f"\n完成 {len(ok)}/{len(done)} 对")
    h = [H[v["label"]] for v in ok.values()]
    m = [M[v["winner"]] for v in ok.values()]
    nt = sum(1 for x in h if x != 0)
    flip = sum(1 for v in ok.values() if not v.get("order_consistent", True))
    a_rate = sum(1 for v in ok.values() if v["winner"] == "a") / max(1, len(ok))
    a_true = sum(1 for v in ok.values() if TRUTH[v["label"]] == "a") / max(1, len(ok))
    print(f"  [VideoAlign 口径] acc(无平局) {acc_without_ties(h, m):.1%}  (非平局 {nt})")
    print(f"  预测平局率 {sum(1 for v in ok.values() if v['winner']=='tie')/max(1,len(ok)):.1%}"
          f"  ·  顺序翻转 {flip}/{len(ok)} ({flip/max(1,len(ok)):.0%})")
    print(f"  选A率 {a_rate:.1%} vs 人评A侧 {a_true:.1%}")
    dec = [v for v in ok.values() if v["winner"] in ("a", "b") and v["label"] != "same"]
    hit = sum(1 for v in dec if v["winner"] == TRUTH[v["label"]])
    print(f"  参考·只看表态的对: {hit/max(1,len(dec)):.1%} (n={len(dec)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

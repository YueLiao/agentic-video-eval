#!/usr/bin/env python3
"""Compare two clips as video, optionally with the magnified suspicious regions.

Everything before this fed the judge sampled stills, and sampling is exactly
what destroys rate -- which is most of what motion quality is. The symptom was
consistent: eight, sixteen and thirty-two frames all sat between 48% and 58%
with half the answers flipping under an order swap, and the per-clip checklist
returned 0% on every rate item (freeze, slip, speed, amplitude) across 1220
clips, because a grid of stills does not contain the answer.

gemma-4 has a separate video path with its own budget: 70 soft tokens per frame
across 32 frames, 2240 in total against the 280 an image gets however many
frames are tiled into it, and about 400px of effective per-frame resolution
against the 264px an eight-column strip leaves. Measured on the first pairwise
video call, the model's rationale reached for weight transfer and stiffness --
global properties that no magnified crop can show.

Three conditions, so the two kinds of evidence can be told apart:

  video   two clips, whole
  loci    the magnified suspicious regions only (the existing chain)
  both    video plus loci, since they fail in opposite directions

    python scripts/run_video_pairwise.py --mode both --ladder-only --out runs/vp
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.llm.client import ImageRef, VideoRef, VLMClient   # noqa: E402
from agenteval.media.clip import VideoHandle                     # noqa: E402
from agenteval.meta.videoalign import (acc_with_ties,            # noqa: E402
                                       acc_without_ties)
from agenteval.tools.locus_view import worst_loci                # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
H = {"AA": 1, "A": 1, "BB": -1, "B": -1, "same": 0}
TRUTH = {"AA": "a", "A": "a", "BB": "b", "B": "b", "same": "same"}

SYSTEM = """\
你在比较两段 **AI 生成视频**的**运动质量**,判断哪一段更好。

判断依据(按重要性):
1. **动作自然度** —— 有没有惯性,重心是否随支撑脚转移,关节朝向是否可能。
   像被逐帧摆出来的木偶、或者动作僵硬突兀,是严重问题。
2. **运动连贯性** —— 有无卡顿、抽帧、跳变、忽快忽慢。
3. **物理合理性** —— 下落是否匀加速,刚体是否保持,有无穿模或无支撑悬浮。
4. **结构完整性** —— 运动中肢体/物体有没有变形、融合、凭空出现消失。
5. **运动幅度** —— 该动的东西是否真的动了。

**不要**根据画面美观、清晰度、色彩、构图来判断——只看**运动**。
两段视频内容相同(同一条提示词生成),请专注于运动表现的差异。
两段确实相当就返回 tie,**不要为了分出胜负而勉强选一边**。

输出 JSON:{"winner":"A"|"B"|"tie","margin":"slight"|"clear","reason":"具体依据"}"""


def build(mode, a_path, b_path, la, lb, first="a"):
    """Parts for one order. Rebuilt per order so captions move with their clip."""
    lo, hi = (a_path, b_path) if first == "a" else (b_path, a_path)
    llo, lhi = (la, lb) if first == "a" else (lb, la)
    parts: list = []
    for tag, vp, loci in (("A", lo, llo), ("B", hi, lhi)):
        parts.append(f"—— 视频 {tag} ——")
        if mode in ("video", "both"):
            parts.append(VideoRef(Path(vp)))
        if mode in ("loci", "both") and loci:
            parts.append(f"视频 {tag} 的可疑处(信号标出,已裁剪放大;"
                         "可疑不等于有问题,遮挡/转向/运动模糊都会触发):")
            parts += [ImageRef(path=p, caption=f"视频{tag} 可疑处") for p in loci]
    return parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["video", "loci", "both"], default="video")
    ap.add_argument("--splits", nargs="*", default=["train", "dev", "val_ac"])
    ap.add_argument("--ladder-only", action="store_true")
    ap.add_argument("--family", default=None)
    ap.add_argument("--min-annotators", type=int, default=0)
    ap.add_argument("--upset", action="store_true",
                    help="the 689 pairs where a weaker-tier generator won. The "
                         "capacity prior scores 0% there by construction, so a "
                         "judge that only rides it cannot hide.")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--k", type=int, default=3, help="loci per clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--loci-dir", default="/tmp/lv_ladder/loci")
    ap.add_argument("--endpoints", nargs="*",
                    default=["http://127.0.0.1:8005/v1",
                             "http://127.0.0.1:8006/v1"])
    ap.add_argument("--model", default="gemma-4-31b-it")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    rows, seen = [], set()
    if args.upset:
        UP = ("/pub/evaluation_group/cy/mq_promptgen/pairing/relabel_r2/"
              "P1_train_upset_relabel.csv")
        want = {r["pair_id"] for r in csv.DictReader(open(UP))}
    for sp in args.splits:
        for r in csv.DictReader(open(f"{R012}/{sp}.csv")):
            if r["pair_id"] in seen:
                continue
            if args.upset and r["pair_id"] not in want:
                continue
            if args.ladder_only and str(r.get("is_ladder")).lower() not in ("1", "true"):
                continue
            if args.family and r["family"] != args.family:
                continue
            if args.min_annotators and int(r.get("n_annotations") or 1) < args.min_annotators:
                continue
            seen.add(r["pair_id"]); rows.append(r)
    if args.n:
        rows = rows[:args.n]
    print(f"{args.mode}: {len(rows)} 对 · {len(args.endpoints)} 个端点")

    vlms = [VLMClient(model=args.model, base_url=ep, max_tokens=700,
                      timeout_s=600, cache_dir=out / "llm_cache")
            for ep in args.endpoints]
    cache_p = out / "compare.json"
    done = json.loads(cache_p.read_text()) if cache_p.exists() else {}

    def loci_for(rel):
        if args.mode == "video":
            return []
        r = worst_loci(VideoHandle(os.path.join(ROOT, rel)),
                       Path(args.loci_dir), k=args.k)
        return r.images[:args.k * 2]

    def one(job):
        i, r = job
        pid = r["pair_id"]
        if pid in done:
            return pid, done[pid]
        vlm = vlms[i % len(vlms)]
        try:
            a, b = os.path.join(ROOT, r["path_A"]), os.path.join(ROOT, r["path_B"])
            la, lb = loci_for(r["path_A"]), loci_for(r["path_B"])
            outs = {}
            for first in ("a", "b"):
                resp = vlm.ask_multimodal(
                    system=SYSTEM,
                    user="逐条对照上面的判断依据,给出结论。",
                    parts=build(args.mode, a, b, la, lb, first),
                    schema={"type": "object"}, tag=f"vp/{args.mode}/{pid}/{first}")
                if not resp.ok:
                    return pid, {"error": resp.error or "call failed"}
                outs[first] = str((resp.parsed or {}).get("winner", "tie")).upper()
            w1, w2 = outs["a"], outs["b"]
            flip = {"A": "B", "B": "A"}
            # In the second order the clips changed places, so a consistent
            # judge must name the other letter.
            consistent = (w1 == "TIE" or w2 == "TIE"
                          or flip.get(w2, w2) == w1)
            winner = w1.lower() if consistent else "tie"
            return pid, {"winner": winner, "raw": [w1, w2],
                         "order_consistent": consistent, "label": r["MQ"],
                         "n_annotations": r["n_annotations"]}
        except Exception as e:  # noqa: BLE001
            return pid, {"error": f"{type(e).__name__}: {e}"[:110]}

    todo = [(i, r) for i, r in enumerate(rows) if r["pair_id"] not in done]
    if todo:
        pid0, r0 = one(todo[0])
        if r0.get("error"):
            print(f"冒烟失败:{r0}", file=sys.stderr); return 2
        done[pid0] = r0
        print(f"冒烟通过 ({r0['raw']}) → 批量开始")
        t0 = time.time(); k = 1
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for pid, res in ex.map(one, todo[1:]):
                done[pid] = res; k += 1
                if k % 50 == 0:
                    print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s")
                    cache_p.write_text(json.dumps(done, ensure_ascii=False))
        cache_p.write_text(json.dumps(done, ensure_ascii=False))

    ok = {k: v for k, v in done.items() if not v.get("error")}
    print(f"\n完成 {len(ok)}/{len(done)} 对  ·  模式 {args.mode}")
    M = {"a": 1.0, "b": -1.0, "tie": 0.0}
    for name, sub in (("全部", list(ok.values())),
                      ("≥3人标注", [v for v in ok.values()
                                    if int(v.get("n_annotations") or 1) >= 3])):
        if not sub:
            continue
        h = [H[v["label"]] for v in sub]
        m = [M[v["winner"]] for v in sub]
        nt = [v for v in sub if H[v["label"]] != 0]
        dec = [v for v in nt if v["winner"] != "tie"]
        hit = sum(1 for v in dec if v["winner"] == TRUTH[v["label"]])
        flip = sum(1 for v in sub if not v.get("order_consistent", True))
        arate = sum(1 for v in sub if v["winner"] == "a") / len(sub)
        atrue = sum(1 for v in sub if TRUTH[v["label"]] == "a") / len(sub)
        import math
        z = ((hit / len(dec) - 0.5) / math.sqrt(0.25 / len(dec))) if dec else 0
        print(f"  [{name}] n={len(sub)} 非平局 {len(nt)}")
        print(f"    acc(无平局) {acc_without_ties(h, m):.1%} · "
              f"acc* {acc_with_ties(h, m)[0]:.1%}")
        print(f"    表态 {len(dec)}/{len(nt)} · 方向 {hit/max(1,len(dec)):.1%} "
              f"(z={z:+.2f}) · 顺序翻转 {flip/len(sub):.0%} · "
              f"选A率 {arate:.0%} vs 人评 {atrue:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

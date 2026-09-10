#!/usr/bin/env python3
"""Judge each suspicious locus on its own, then count the confirmed failures.

The pairwise version of this asked which clip's set of magnified loci looked
worse, and scored 16.7% -- below chance. The cause is a confound rather than a
prompt: the suspicion signal measures change that motion does not explain, a
clip that moves more has more of it, and human motion-quality preference rewards
moving. Measured on six seed pairs under a shared scale, the clip with the
higher suspicion peak was the human's preferred clip in four of four non-tie
cases. Asking which set looks worse is close to asking which clip moves less.

Judging loci one at a time breaks that. A locus that is ordinary motion blur is
named as such and drops out, so the count reflects confirmed failures rather
than nominated ones, and a clip is no longer punished for having given the
signal more to chew on. The model already made this distinction unprompted in
the pairwise run's rationales -- it just had no way to express it.

The output is per-clip and therefore point-wise, and no position bias is
possible.

    python scripts/run_locus_verdict.py --split dev --family seed --out runs/lv
"""
from __future__ import annotations

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.llm.client import ImageRef, VLMClient    # noqa: E402
from agenteval.media.clip import VideoHandle            # noqa: E402
from agenteval.tools.locus_view import worst_loci       # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")

SYSTEM = """\
你在核实 **AI 生成视频**里的一处可疑画面。

一个信号标记了这个位置,它测的是"这里的变化无法用运动解释"。
**信号不知道这是什么。**下面这些都会触发它,而它们都是正常的:
  · 快速运动产生的运动模糊
  · 物体被遮挡后重新出现
  · 肢体或物体转向镜头(透视缩短)
  · 光影、反射、明暗变化
  · 本来就细碎的纹理(水花、火焰、树叶、毛发)在动

只有下面这些才算**真正的崩坏**:
  · 肢体/手指断裂、融合、数量改变、反关节
  · 本应刚性的物体在扭曲、熔化、拓扑改变
  · 物体凭空出现或消失,没有进出画面的过程
  · 两个物体互相穿过
  · 结构边界撕裂、糊成一团且不随运动恢复

判据:看**同一个物体**在相邻两格之间的变化。真实运动会让它位移、模糊、被遮挡,
但**拓扑不变**——该有几根手指还是几根,该连着的还连着。崩坏是拓扑变了。

看不清就选 unclear。**宁可说看不清,也不要把正常的运动模糊说成崩坏。**
"""

PROMPT = """下面是这处可疑画面,已裁剪放大,每张图是相邻的两个时刻,格上标了帧号。

1. `observed`:同一个物体在这几格之间,具体发生了什么变化?(描述你看到的,不要下结论)
2. `verdict`:
   - `normal` —— 可以由运动/遮挡/转向/光影解释
   - `broken` —— 拓扑改变了,是真的崩坏
   - `unclear` —— 看不清
3. `severity`:仅当 verdict=broken 时给,`轻微`|`明显`|`严重`
4. `what`:仅当 verdict=broken 时给,一句话说是哪种崩坏(肢体/刚体/凭空出现消失/穿模/糊烂)

输出 JSON:{"observed":"...","verdict":"normal"|"broken"|"unclear",
 "severity":"轻微"|"明显"|"严重"|null,"what":"..."|null}"""

SEV = {"轻微": 1.0, "明显": 2.0, "严重": 3.0}

# A binary broken/normal verdict throws the gradation away: on 1220 dev seed
# clips it put 1079 of them at exactly zero, and 372 non-tie pairs collapsed to
# 63 separable ones. The graded question keeps the ordering inside "normal",
# which is where most of the corpus lives.
#
# Every anchor names motion blur and occlusion as belonging at the *clean* end.
# That confound cost two rounds to remove -- suspicion fires on motion, humans
# reward motion, and 'which loci look worse' inverted to 16.7% -- so it is
# stated in the scale itself rather than left to the model to infer.
GRADED_PROMPT = """下面是这处可疑画面,已裁剪放大,每张图是相邻的两个时刻,格上标了帧号。

1. `observed`:同一个物体在这几格之间,具体发生了什么变化?(描述所见,先不下结论)
2. `verdict`:`normal` | `broken` | `unclear`(定义见上)
3. `integrity`:这处的**结构完整度**,只能取 10/8/6/4/2:
   - `10` 完好 —— 拓扑完全保持。**运动模糊、遮挡、转向、光影变化都算完好**,
     哪怕画面很糊:糊而拓扑不变,就是 10。
   - `8`  轻微 —— 边缘或纹理有一点不稳,但同一物体始终认得出、连接关系没变。
   - `6`  可疑 —— 有像是形变的地方,但也可能是遮挡或转向造成的,**判不准**。
   - `4`  明显 —— 拓扑确实变了(手指数量变了、连接断开、刚体扭曲),但局部、不显眼。
   - `2`  严重 —— 大面积崩坏,或主体结构塌掉,一眼就能看到。
4. `what`:仅当 integrity ≤ 6 时给,一句话说是哪种问题。

**判据只有一条:拓扑变没变。**该有几根手指还是几根,该连着的还连着。
真实运动会让物体位移、模糊、被遮挡,但不会改变它是什么。

输出 JSON:{"observed":"...","verdict":"normal"|"broken"|"unclear",
 "integrity":10|8|6|4|2,"what":"..."|null}"""
GRADES = (10.0, 8.0, 6.0, 4.0, 2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--family", default="seed")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=0,
                    help="0 = all pairs; otherwise a label-stratified subsample")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--endpoints", nargs="*",
                    default=["http://127.0.0.1:8005/v1",
                             "http://127.0.0.1:8006/v1"])
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--graded", action="store_true",
                    help="score each locus 10/8/6/4/2 instead of broken/normal")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    rows = [r for r in csv.DictReader(open(f"{R012}/{args.split}.csv"))
            if not args.family or r["family"] == args.family]
    if args.n:
        import random
        rng = random.Random(11)
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["MQ"], []).append(r)
        sub = []
        for lab, g in by.items():
            sub += rng.sample(g, min(max(1, round(args.n * len(g) / len(rows))),
                                     len(g)))
        rng.shuffle(sub)
        rows = sub[:args.n]
    vids = sorted({r[k] for r in rows for k in ("path_A", "path_B")})
    print(f"{args.split}/{args.family}: {len(rows)} 对, {len(vids)} 条视频 · "
          f"{len(args.endpoints)} 个端点")

    vlms = [VLMClient(model=args.model, base_url=ep, max_tokens=700,
                      timeout_s=300, cache_dir=out / "llm_cache")
            for ep in args.endpoints]
    cache_p = out / "clips.json"
    done = json.loads(cache_p.read_text()) if cache_p.exists() else {}

    def one(job):
        i, rel = job
        if rel in done:
            return rel, done[rel]
        vlm = vlms[i % len(vlms)]
        try:
            v = VideoHandle(os.path.join(ROOT, rel))
            res = worst_loci(v, out / "loci", k=args.k)
            if not res.images:
                # No locus is itself evidence, and it is not the same as clean.
                return rel, {"loci": [], "n_loci": 0, "n_broken": 0,
                             "severity": 0.0, "no_locus": True}
            verdicts = []
            for li, lv in enumerate(res.value["loci"]):
                imgs = [ImageRef(path=p, caption=f"可疑处 {lv['locus_id']}")
                        for p in res.images
                        if f"_{lv['locus_id']}_" in Path(p).name]
                if not imgs:
                    continue
                r = vlm.ask(system=SYSTEM,
                            user=GRADED_PROMPT if args.graded else PROMPT,
                            images=imgs, schema={"type": "object"},
                            tag=f"lv{'g' if args.graded else ''}/{rel}/{lv['locus_id']}")
                p = r.parsed or {}
                g = p.get("integrity")
                try:
                    g = min(GRADES, key=lambda x: abs(x - float(g)))
                except (TypeError, ValueError):
                    g = None
                verdicts.append({"locus_id": lv["locus_id"],
                                 "t_span": lv["t_span"],
                                 "dominant": lv.get("dominant"),
                                 "verdict": str(p.get("verdict", "unclear")).lower(),
                                 "severity": p.get("severity"),
                                 "integrity": g,
                                 "what": str(p.get("what") or "")[:80],
                                 "observed": str(p.get("observed") or "")[:160]})
            if not verdicts:
                return rel, {"error": "no verdicts"}
            broken = [x for x in verdicts if x["verdict"] == "broken"]
            sev = sum(SEV.get(str(x.get("severity")), 1.0) for x in broken)
            gs = [x["integrity"] for x in verdicts if x["integrity"] is not None]
            # Worst locus and mean: one severe failure and three mild ones are
            # different clips, and either summary alone hides one of them.
            return rel, {"loci": verdicts, "n_loci": len(verdicts),
                         "n_broken": len(broken), "severity": round(sev, 2),
                         "integrity_min": min(gs) if gs else None,
                         "integrity_mean": round(sum(gs) / len(gs), 2) if gs else None}
        except Exception as e:  # noqa: BLE001
            return rel, {"error": f"{type(e).__name__}: {e}"[:110]}

    todo = [(i, v) for i, v in enumerate(vids) if v not in done]
    if todo:
        rel0, r0 = one(todo[0])
        if r0.get("error"):
            print(f"冒烟失败:{r0}", file=sys.stderr); return 2
        done[rel0] = r0
        print(f"冒烟通过 (处数={r0.get('n_loci')} 崩坏={r0.get('n_broken')}) → 批量开始")
        t0 = time.time(); k = 1
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rel, res in ex.map(one, todo[1:]):
                done[rel] = res; k += 1
                if k % 50 == 0:
                    print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s")
                    cache_p.write_text(json.dumps(done, ensure_ascii=False))
        cache_p.write_text(json.dumps(done, ensure_ascii=False))

    ok = {k: v for k, v in done.items() if not v.get("error")}
    print(f"\n完成 {len(ok)}/{len(done)} 条")
    from collections import Counter
    vc = Counter(x["verdict"] for v in ok.values() for x in v.get("loci", []))
    print(f"  逐处判定分布: {dict(vc)}")
    nb = Counter(v["n_broken"] for v in ok.values())
    print(f"  每条视频确认崩坏处数: {dict(sorted(nb.items()))}")

    from agenteval.meta.videoalign import acc_with_ties, acc_without_ties
    H = {"AA": 1, "A": 1, "BB": -1, "B": -1, "same": 0}
    keys = [("确认崩坏处数", "n_broken", -1), ("严重度加权", "severity", -1)]
    if args.graded:
        keys += [("完整度最低处", "integrity_min", +1),
                 ("完整度均值", "integrity_mean", +1)]
    for name, key, sign in keys:
        sub = [r for r in rows if r["path_A"] in ok and r["path_B"] in ok
               and ok[r["path_A"]].get(key) is not None
               and ok[r["path_B"]].get(key) is not None]
        if not sub:
            continue
        h = [H[r["MQ"]] for r in sub]
        m = [sign * float(ok[r["path_A"]][key] - ok[r["path_B"]][key]) for r in sub]
        nz = sum(1 for x in m if x != 0)
        star, eps = acc_with_ties(h, m)
        print(f"  [{name}] acc(无平局) {acc_without_ties(h, m):.1%} · "
              f"acc* {star:.1%} (eps*={eps:.2f}) · 能分开的对 {nz}/{len(sub)}")
    (out / "scores.json").write_text(json.dumps(
        {k: {kk: v.get(kk) for kk in ("n_broken", "severity", "integrity_min",
                                      "integrity_mean", "n_loci")}
         for k, v in ok.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

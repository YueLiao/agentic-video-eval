#!/usr/bin/env python3
"""Ask the fixed debug set, per case, so a prompt change is the only variable.

The set is pre-rendered and every case has a measured band contrast, so a miss
here is the model or the wording -- never a picture that could not have answered.

    python scripts/ask_debug_set.py --set runs/debug_freeze --variant v1
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.llm.client import ImageRef, VLMClient    # noqa: E402

SCHEMA = ('输出 JSON:{"has_freeze": true|false, "t_start": 0~1 或 null, '
          '"t_end": 0~1 或 null, "evidence": "你看到了什么"}')

VARIANTS = {
    # The hint the tool itself ships. Baseline.
    "tool": None,
    # Perception only: no judgement, no vocabulary about freezing.
    "perceive": ("这张图的每个面板里,**亮度代表相邻两帧之间画面变化的大小**——"
                 "越亮变化越大,越暗变化越小。青色刻度是时间比例 0→1。\n"
                 "在标着「↓时间」的面板里,时间从上往下走;"
                 "标着「→时间」的面板里,时间从左往右走。\n\n"
                 "请找出:**有没有一条明显比周围暗的带,横贯该面板的整个另一维?**\n"
                 '输出 JSON:{"has_dark_band": true|false, "t_start": 0~1 或 null, '
                 '"t_end": 0~1 或 null, "which_panel": "哪个面板", "evidence": "..."}'),
    # Same as perceive but told the band means a freeze, to see what the
    # vocabulary costs.
    "named": ("这张图的每个面板里,**亮度代表相邻两帧之间画面变化的大小**,"
              "越亮变化越大。青色刻度是时间比例 0→1。\n"
              "「↓时间」面板时间从上往下,「→时间」面板时间从左往右。\n\n"
              "**如果视频里有一段画面卡住不动,这里就会出现一条明显变暗的带**,"
              "横贯该面板的整个另一维。\n"
              "找一找有没有这样的暗带。\n\n" + SCHEMA),
}


TWO_STEP_1 = ("这张图的每个面板里,**亮度代表相邻两帧之间画面变化的大小**——"
              "越亮变化越大,越暗变化越小。\n"
              "请找出:**有没有一条明显比周围暗的带,横贯某个面板的整个另一维?**\n"
              '输出 JSON:{"has_dark_band": true|false, '
              '"which_panel": "把该面板左上角的标题原样抄下来"|null}')
TWO_STEP_2 = ("这张图里,**亮度代表相邻两帧之间画面变化的大小**,越暗变化越小。"
              "青色刻度标的是时间比例 0→1"
              "(标题里写「↓时间」就从上往下读,写「→时间」就从左往右读)。\n"
              "图中有一条明显比周围暗的带。**请沿刻度读出它的起止位置。**\n"
              '输出 JSON:{"t_start": 0~1, "t_end": 0~1, "evidence": "..."}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--variant", default="perceive",
                    choices=list(VARIANTS) + ["two_step"])
    ap.add_argument("--panels-only", action="store_true",
                    help="send the single row panel instead of the tile")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8006/v1")
    ap.add_argument("--model", default="gemma-4-31b-it")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    cases = json.loads((Path(args.set) / "cases.json").read_text())
    cases = [c for c in cases if c.get("usable")]
    vlm = VLMClient(model=args.model, base_url=args.endpoint, max_tokens=500,
                    timeout_s=300)
    print(f"{len(cases)} 例可用 · 变体 {args.variant} · "
          f"{'只发单面板' if args.panels_only else '发拼接图'}")

    def one(c):
        if args.variant == "two_step":
            # Detection on the tile (more panels, more chances) and localisation
            # on the single panel it names (coordinates are readable there):
            # measured, the tile finds 10/12 and places 5, one panel finds 4/12
            # and places 4. Splitting takes the better half of each.
            r1 = vlm.ask(system="只输出 JSON。", user=TWO_STEP_1,
                         images=[ImageRef(path=Path(c["xt_inj"]))],
                         schema={"type": "object"}, tag=f"dbg/ts1/{c['id']}")
            p1 = r1.parsed or {}
            if not p1.get("has_dark_band"):
                return c, False, [None, None], str(p1.get("which_panel") or "")[:80]
            want = str(p1.get("which_panel") or "")
            pick = None
            for m in c["panels"]:
                key = f"{'y' if m['kind']=='row' else 'x'}={int(m['pos']*100)}%"
                if key in want:
                    pick = m
                    break
            pick = pick or max(c["panels"], key=lambda m: m["kind"] == "row")
            r2 = vlm.ask(system="只输出 JSON。", user=TWO_STEP_2,
                         images=[ImageRef(path=Path(pick["path"]))],
                         schema={"type": "object"}, tag=f"dbg/ts2/{c['id']}")
            p2 = r2.parsed or {}
            return (c, True, [p2.get("t_start"), p2.get("t_end")],
                    f"面板={want[:26]} {str(p2.get('evidence') or '')[:50]}")
        note = VARIANTS[args.variant] or (c["hint_xt"] + "\n\n" +
                                          "这段视频有没有画面停住的区间?\n" + SCHEMA)
        if args.panels_only:
            best = max(c["panels"], key=lambda m: m["kind"] == "row")
            imgs = [ImageRef(path=Path(best["path"]))]
        else:
            imgs = [ImageRef(path=Path(c["xt_inj"]))]
        r = vlm.ask(system="只输出 JSON。", user=note, images=imgs,
                    schema={"type": "object"}, tag=f"dbg/{args.variant}/{c['id']}")
        p = r.parsed or {}
        has = bool(p.get("has_freeze") or p.get("has_dark_band"))
        t = [p.get("t_start"), p.get("t_end")]
        return c, has, t, str(p.get("evidence") or "")[:80]

    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(one, cases))

    hit = loc = 0
    print(f"\n  {'例':<5}{'运动量':>7}{'对比度':>8}{'真值':>16}{'模型':>18}{'':>4}")
    for c, has, t, why in res:
        ta, tb = c["truth_norm"]
        okloc = False
        if has:
            hit += 1
            try:
                a, b = float(t[0]), float(t[1])
                okloc = min(b, tb) - max(a, ta) > -0.08
            except (TypeError, ValueError):
                okloc = False
            loc += okloc
        mark = ("✓" if okloc else "○") if has else "✗"
        tm = f"[{t[0]},{t[1]}]" if has else "—"
        print(f"  {c['id']:<5}{c['ambient_motion']:>7.1f}{c['contrast']:>8.2f}"
              f"{str(ta)+'–'+str(tb):>16}{tm:>18}{mark:>4}")
        if not has:
            print(f"        {why}")
    n = len(res)
    print(f"\n  检出 {hit}/{n} = {hit/n:.0%} · 其中定位正确 {loc}/{max(1,hit)}")
    print("  ✓=检出且定位对 · ○=检出但位置错 · ✗=漏检")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Lay out every spatial claim beside the evidence that confirmed it, for a human.

Injection validates the temporal branch because a synthetic freeze and a real
one are the same thing physically. It cannot validate the spatial branch: a
rectangular paste of unrelated content is nothing like a real generation defect,
and the model was never asked to look for one. The only way to know whether a
spatial claim is a detection or an invention is to look at it.

So this prints the claims with their evidence paths, grouped by type, and
reports what the pipeline decided. Reading the images is the part a person has
to do; what this removes is the bookkeeping.

    python scripts/audit_spatial.py /tmp/rt_typed16
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main(run: Path) -> int:
    d = json.loads((run / "reports.json").read_text())
    clean = {k: v for k, v in d.items()
             if k.endswith("|clean") and "findings" in v}
    inj = {k: v for k, v in d.items()
           if k.endswith("|inj") and "findings" in v}

    claims, unresolved = [], []
    for k, v in clean.items():
        for f in v["findings"]:
            if f["kind"] == "spatial":
                claims.append({**f, "clip": k.split("|")[0]})
        unresolved += [u for u in v.get("unresolved", []) if ":" in u]

    print(f"=== {run} ===")
    print(f"干净片段 {len(clean)} 条 · 注入片段 {len(inj)} 条")
    sp_c = sum(1 for v in clean.values()
               if any(f["kind"] == "spatial" for f in v["findings"]))
    sp_i = sum(1 for v in inj.values()
               if any(f["kind"] == "spatial" for f in v["findings"]))
    print(f"报出空间缺陷的片段: 干净 {sp_c}/{len(clean)} · 注入 {sp_i}/{len(inj)}")
    print(f"确认的空间指控 {len(claims)} 条 · 判为「看不清」而未确认 {len(unresolved)} 条")

    print(f"\n确认的按类型: {dict(Counter(c['aspect'].split(':')[0] for c in claims))}")
    print(f"未确认的按类型: {dict(Counter(u.split(':')[0] for u in unresolved))}")
    print("\n  ——「未确认」多于「确认」是健康的:类型化证据的作用就是挡掉"
          "证据不足以支撑的指控。\n")

    for i, c in enumerate(claims):
        kind = c["aspect"].split(":")[0]
        print(f"[{i}] {kind:<8} t={c['t_span'][0]:.2f} {c['severity']:<4} "
              f"{c['clip'][:30]}")
        print(f"     指控: {c['aspect'].split(':', 1)[-1][:70]}")
        print(f"     核实: {c['note'][:88]}")
    if not claims:
        print("  (没有确认的空间指控)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))

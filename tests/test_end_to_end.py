"""Full pipeline on one clip with a scripted judge: route -> skills -> falsify -> score."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.engine.loop import LoopBudget                    # noqa: E402
from agenteval.engine.orchestrator import evaluate              # noqa: E402
from agenteval.llm.mock import ScriptedVLM                      # noqa: E402
from agenteval.scoring.dimensions import REPORT_DIMENSIONS      # noqa: E402
from agenteval.skills.human_integrity import HumanIntegrity     # noqa: E402
from agenteval.skills.motion_quality import MotionQuality       # noqa: E402
from agenteval.skills.physical_integrity import PhysicalIntegrity  # noqa: E402
from agenteval.skills.static_integrity import StaticIntegrity   # noqa: E402
from agenteval.skills.temporal_integrity import TemporalIntegrity  # noqa: E402

VIDEO = sys.argv[1]
OUT = Path("/tmp/agenteval_e2e")

SKILLS = {
    "temporal_integrity": lambda: TemporalIntegrity(OUT / "temporal", max_loci=6, max_frames=81),
    "motion_quality":     lambda: MotionQuality(OUT / "motion", max_frames=81),
    "human_integrity":    lambda: HumanIntegrity(OUT / "human"),
    "physical_integrity": lambda: PhysicalIntegrity(OUT / "physical", max_frames=81),
    "static_integrity":   lambda: StaticIntegrity(OUT / "static", sweep_calls=5),
}

SCRIPT = {
    "temporal_integrity/decide": [
        {"thought": "看最可疑的点", "action": "zoom_locus", "args": {"locus_id": "L00"}},
        {"thought": "够了", "action": "conclude", "args": {"reason": "done"}}],
    "temporal_integrity/verdict": {
        "summary": "L00 处轻微纹理不稳",
        "findings": [{"kind": "texture_crawl", "severity": "minor", "t_span": [8, 14],
                      "bbox": [0.3, 0.3, 0.15, 0.15], "confidence": 0.6,
                      "rationale": "放大后该区域细节逐帧重排", "evidence": ["E02"]}]},
    "motion_quality/decide": [
        {"thought": "看整体运动", "action": "inspect", "args": {"t0": 10, "t1": 40}},
        {"thought": "够了", "action": "conclude", "args": {"reason": "done"}}],
    "motion_quality/verdict": {"summary": "运动连贯", "findings": []},
    "human_integrity/decide": [
        {"thought": "核查骨架", "action": "skeleton", "args": {}},
        {"thought": "放大手部", "action": "zoom_hands", "args": {}},
        {"thought": "够了", "action": "conclude", "args": {"reason": "done"}}],
    "human_integrity/verdict": {
        "summary": "手部结构存在问题",
        "findings": [{"kind": "hand_malformation", "severity": "major", "t_span": [18, 30],
                      "bbox": [0.45, 0.5, 0.1, 0.1], "confidence": 0.75,
                      "rationale": "放大后可见手指粘连", "evidence": ["E07"]}]},
    "physical_integrity/decide": [
        {"thought": "跟踪主体", "action": "track_region",
         "args": {"bbox": [0.35, 0.25, 0.3, 0.35], "t0": 20, "t1": 50}},
        {"thought": "够了", "action": "conclude", "args": {"reason": "done"}}],
    "physical_integrity/verdict": {"summary": "未见物理违反", "findings": []},
    "static_integrity/decide": [
        {"thought": "继续粗筛", "action": "next_sheet", "args": {"batch_index": 2}},
        {"thought": "f30 可疑,分块放大", "action": "tile", "args": {"t": 30, "grid": 2}},
        {"thought": "原生分辨率确认", "action": "zoom",
         "args": {"bbox": [0.5, 0.5, 0.2, 0.2], "t": 30}},
        {"thought": "确认完毕", "action": "conclude", "args": {"reason": "done"}}],
    "static_integrity/verdict": {
        "summary": "f30 处物体结构不成立",
        "findings": [{"kind": "structure_collapse", "severity": "minor", "t_span": [30, 31],
                      "bbox": [0.5, 0.5, 0.2, 0.2], "confidence": 0.6,
                      "rationale": "放大后该物体几何不连贯", "evidence": ["E05"]}]},
    "static_integrity/falsify": {"verdict": "uphold", "reason": "对照区域正常"},
    # falsification: uphold the hand finding, retract the texture one
    "human_integrity/falsify": {"verdict": "uphold", "reason": "对照区域正常,指控成立"},
    "temporal_integrity/falsify": {"verdict": "retract",
                                   "reason": "对照时刻同样如此,属该视频常态"},
}


def main() -> int:
    vlm = ScriptedVLM(SCRIPT)
    res = evaluate(VIDEO, {"prompt": "一个人在快速跳舞，镜头缓缓推近"},
                   SKILLS, vlm, out_dir=OUT, budget=LoopBudget(max_rounds=4))

    print("== routing ==")
    print("  enabled :", res.routing["skills"])
    print("  disabled:", {k: v[:40] for k, v in res.routing["disabled"].items()})
    print("  budget  :", res.routing["budget_scale"])

    print("\n== per-skill ==")
    for v in res.verdicts:
        print(f"  {v.skill:22s} findings={len(v.live)} retracted={len(v.findings)-len(v.live)}"
              f" rounds={v.rounds} vlm={v.vlm_calls} tools={v.tool_calls}")

    print("\n== dynamic prompt (human_integrity) ==")
    for f in res.prompt_manifests.get("human_integrity", []):
        print(f"  {f['slot']:16s} {f['source']:34s} {f['chars']:5d} chars")

    print("\n== 报告(6 维) ==")
    print(res.score.table())

    print("\n== skill detail ==")
    for name, d in res.score.dimensions.items():
        flag = "" if d.applicable else f"  (n/a: {d.reason[:30]})"
        cap = f"  capped_by={d.capped_by}" if d.capped_by else ""
        print(f"  {name:22s} {d.score:5.2f}{cap}{flag}")
    print(f"\n  retraction_rate={res.score.retraction_rate:.2f}"
          f"  caps={res.score.caps_fired}")
    print(f"  tool_calls={res.tool_calls} (bus cache hits={res.bus['cache_hits']})"
          f"  vlm_calls={res.vlm_calls}  {res.elapsed_s:.1f}s")

    assert res.score.dimensions["human_integrity"].n_findings == 1
    assert res.score.dimensions["temporal_integrity"].n_retracted == 1
    assert res.score.dimensions["temporal_integrity"].n_findings == 0
    assert 0 < res.score.overall < 10
    rep = res.score.report
    assert set(rep) == set(REPORT_DIMENSIONS), sorted(rep)
    assert rep["subject_fidelity"].applicable and rep["subject_fidelity"].score < 10
    assert not rep["semantic_alignment"].applicable, "no conformance skill yet"
    assert rep["visual_quality"].applicable, "static_integrity should cover it"
    assert rep["temporal_consistency"].n_retracted == 1
    print("\nend-to-end OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

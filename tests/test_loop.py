"""End-to-end loop tests with a scripted judge (no endpoint required)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.engine.evidence import EvidenceBus              # noqa: E402
from agenteval.engine.loop import LoopBudget, falsify, run_skill  # noqa: E402
from agenteval.llm.mock import ScriptedVLM                     # noqa: E402
from agenteval.media.clip import VideoHandle                   # noqa: E402
from agenteval.skills.base import SkillContext                 # noqa: E402
from agenteval.skills.temporal_integrity import TemporalIntegrity  # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else None


def build(video: str, out: Path, script, budget=None):
    vh = VideoHandle(video)
    ctx = SkillContext(video=vh, bus=EvidenceBus(vh.hash),
                       condition={"prompt": "a test clip"})
    skill = TemporalIntegrity(out, max_loci=6, max_frames=81)
    vlm = ScriptedVLM(script)
    v, steps = run_skill(skill, ctx, vlm, budget or LoopBudget(max_rounds=4))
    return skill, ctx, vlm, v, steps


def test_concludes_early(video, out):
    script = {
        "temporal_integrity/decide": {"thought": "证据够了", "action": "conclude",
                                      "args": {"reason": "no anomaly"}},
        "temporal_integrity/verdict": {"summary": "无明显问题", "findings": []},
    }
    _, _, vlm, v, steps = build(video, out / "a", script)
    assert len(steps) == 1 and steps[0].action == "conclude", steps
    assert v.findings == [] and v.tool_calls == 2, v.to_json()
    print("  concludes early: OK  (rounds=%d, vlm_calls=%d)" % (v.rounds, v.vlm_calls))


def test_probes_then_reports(video, out):
    script = {
        "temporal_integrity/decide": [
            {"thought": "L00 最可疑", "action": "zoom_locus", "args": {"locus_id": "L00"}},
            {"thought": "需要对照", "action": "contrast", "args": {"locus_id": "L00"}},
            {"thought": "够了", "action": "conclude", "args": {"reason": "ok"}},
        ],
        "temporal_integrity/verdict": {
            "summary": "L00 处有纹理沸腾",
            "findings": [{"kind": "texture_crawl", "severity": "major",
                          "t_span": [10, 16], "bbox": [0.3, 0.3, 0.2, 0.2],
                          "confidence": 0.8, "rationale": "放大后可见细节逐帧重排",
                          "evidence": ["E03"]}],
        },
    }
    _, ctx, vlm, v, steps = build(video, out / "b", script)
    acts = [s.action for s in steps]
    assert acts == ["zoom_locus", "contrast", "conclude"], acts
    assert len(v.live) == 1 and v.tool_calls == 4, v.to_json()
    assert any(c["n_images"] > 0 for c in vlm.calls), "judge never received images"
    print("  probes then reports: OK  (actions=%s, images sent=%d)"
          % (acts, max(c["n_images"] for c in vlm.calls)))
    return ctx, v


def test_repeat_stops_loop(video, out):
    script = {
        "temporal_integrity/decide": [
            {"thought": "看 L00", "action": "zoom_locus", "args": {"locus_id": "L00"}},
            {"thought": "再看一次 L00", "action": "zoom_locus", "args": {"locus_id": "L00"}},
            {"thought": "还看", "action": "zoom_locus", "args": {"locus_id": "L00"}},
        ],
        "temporal_integrity/verdict": {"summary": "", "findings": []},
    }
    _, _, _, v, steps = build(video, out / "c", script)
    assert len(steps) == 2 and not steps[1].new_information, [s.to_json() for s in steps]
    print("  repeat stops loop: OK  (stopped after %d rounds)" % len(steps))


def test_bad_args_survive(video, out):
    script = {
        "temporal_integrity/decide": [
            {"thought": "看不存在的点", "action": "zoom_locus", "args": {"locus_id": "L99"}},
            {"thought": "算了", "action": "conclude", "args": {"reason": "done"}},
        ],
        "temporal_integrity/verdict": {"summary": "", "findings": []},
    }
    _, _, _, v, steps = build(video, out / "d", script)
    assert steps[0].evidence_id is None and steps[-1].action == "conclude"
    assert v.error is None
    print("  bad args survive: OK  (loop recovered and concluded)")


def test_falsification_retracts(ctx, verdict, video, out):
    vlm = ScriptedVLM({"temporal_integrity/falsify": {
        "verdict": "retract", "reason": "对照区域表现一致,属正常渲染"}})
    v = falsify(verdict, ctx, vlm)
    assert len(v.live) == 0 and v.findings[0].retracted
    print("  falsification retracts: OK  (%d/%d retracted, reason recorded)"
          % (len(v.findings) - len(v.live), len(v.findings)))


if __name__ == "__main__":
    if not VIDEO:
        print("usage: test_loop.py <video.mp4>"); raise SystemExit(2)
    out = Path("/tmp/agenteval_test")
    print("running loop tests on", VIDEO)
    test_concludes_early(VIDEO, out)
    ctx, v = test_probes_then_reports(VIDEO, out)
    test_repeat_stops_loop(VIDEO, out)
    test_bad_args_survive(VIDEO, out)
    test_falsification_retracts(ctx, v, VIDEO, out)
    print("\nall loop tests passed")

"""Capability profile: probe results turned into decisions.

A list of probe outcomes is diagnostics. A profile is what the planner and the
executor actually read -- how many images may go in one bundle, whether to ask
this model for boxes or hand it CV boxes, how many votes a borderline judgement
needs, and whether a constraint can live in prose or has to be enforced in code.

The point is that these are *derived*, not configured. The previous version had
every one of them hardcoded to whatever one local model happened to handle, so
a stronger model bought nothing and a weaker one produced silent nonsense.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agenteval.capability.probes import PROBES, ProbeResult
from agenteval.llm.client import VLMClient


@dataclass
class CapabilityProfile:
    model: str
    endpoint: str
    probed_at: float = field(default_factory=time.time)
    results: dict[str, ProbeResult] = field(default_factory=dict)

    # ---- derived decisions ------------------------------------------
    @property
    def max_images(self) -> int:
        r = self.results.get("max_images")
        return int(r.value) if r and r.value else 8

    @property
    def detail_floor_px(self) -> int:
        """Smallest rendered size at which fine structure is still readable.
        Crops below this are wasted calls, and the magnification ladder should
        start here rather than at an arbitrary 448."""
        r = self.results.get("fine_detail_floor")
        return int(r.value) if r and r.value else 448

    @property
    def can_ground(self) -> bool:
        """Ask this model for boxes, or hand it boxes from a detector."""
        r = self.results.get("grounding")
        return bool(r and r.passed)

    @property
    def can_read_labels(self) -> bool:
        """Whether composited filmstrips work -- if it cannot read the burned-in
        frame numbers, turning time into space loses the time."""
        r = self.results.get("temporal_order")
        return bool(r and r.passed)

    @property
    def votes_needed(self) -> int:
        """How many samples a borderline judgement needs. A model that agrees
        with itself needs one; one that does not needs enough to out-vote its
        own noise."""
        r = self.results.get("self_consistency")
        rate = float(r.value) if r and r.value is not None else 0.6
        return 1 if rate >= 0.85 else (3 if rate >= 0.6 else 5)

    @property
    def constraints_in_code(self) -> bool:
        """True when mid-prompt instructions are unreliable, so anything that
        must always hold has to be enforced by the harness instead of asked
        for. This project lost two constraints exactly this way."""
        r = self.results.get("instruction_depth")
        return not (r and r.passed)

    @property
    def invents_defects(self) -> bool:
        r = self.results.get("refusal_to_invent")
        return not (r and r.passed)

    @property
    def schema_depth_ok(self) -> bool:
        r = self.results.get("schema_fidelity")
        return bool(r and r.passed)

    @property
    def tier(self) -> str:
        """Coarse band, for choosing which elicitation techniques are worth it.
        Cheap models cannot pay for debate; strong ones do not need voting."""
        core = ["schema_fidelity", "grounding", "temporal_order",
                "counting", "instruction_depth", "refusal_to_invent"]
        got = sum(1 for k in core if (self.results.get(k) and self.results[k].passed))
        return "strong" if got >= 5 else ("mid" if got >= 3 else "weak")

    @property
    def video_budget(self) -> tuple[int, int] | None:
        r = self.results.get("video_path")
        return tuple(r.value) if r and r.passed and r.value else None

    @property
    def image_budget_kind(self) -> str:
        r = self.results.get("image_budget")
        return str(r.value) if r and r.passed else "unknown"

    def evidence_policy(self) -> dict[str, Any]:
        """What the executor should do when assembling an evidence bundle.

        `frames_via` is the decision that dominates every other one here, and
        it is the one this repo got wrong for a whole day. A token-capped image
        path spends a fixed budget however many frames are tiled into it, so
        adding frames shrinks each of them -- eight, sixteen and thirty-two all
        scored the same because each addition took resolution from the last. A
        video path with a per-frame budget does not have that property, and it
        is the only path that carries rate. On gemma-4 the two differ by eight
        times the visual budget, and switching took pairwise direction accuracy
        from 48-58% to 82.2%.
        """
        vb = self.video_budget
        return {
            "max_images": self.max_images,
            "zoom_px": max(self.detail_floor_px, 336),
            "use_filmstrip": self.can_read_labels,
            "boxes_from": "vlm" if self.can_ground else "detector",
            "votes": self.votes_needed,
            "enforce_constraints_in_code": self.constraints_in_code,
            "require_clean_baseline": self.invents_defects,
            # Whole clips when the deployment has a per-frame budget for them;
            # tiles otherwise, and then the tile layout has to respect the cap.
            "frames_via": "video" if vb else "image_tiles",
            "video_frames": vb[1] if vb else 0,
            "video_tokens_per_frame": vb[0] if vb else 0,
            "image_budget": self.image_budget_kind,
            # A fixed cap means more frames per image costs resolution; a
            # native-size path means it costs latency instead.
            "more_frames_costs": ("resolution"
                                  if self.image_budget_kind == "token_capped"
                                  else "latency"),
        }

    def techniques(self) -> list[str]:
        """Elicitation techniques worth enabling for this model.

        Deliberately not "all of them". Which to spend on is exactly the
        judgement the profile exists to make -- voting is wasted on a
        self-consistent model, and debate is wasted on one that cannot hold a
        schema through two turns.
        """
        out = ["falsification"]
        if self.votes_needed > 1:
            out.append("self_consistency_vote")
        if self.tier == "strong":
            out += ["self_generated_verification", "multi_turn_probing"]
        if self.invents_defects:
            out.append("clean_control_pairing")
        if not self.can_read_labels:
            out.append("ordered_frames_only")
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model, "endpoint": self.endpoint,
            "probed_at": self.probed_at, "tier": self.tier,
            "derived": self.evidence_policy(),
            "techniques": self.techniques(),
            "probes": {k: v.to_json() for k, v in self.results.items()},
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1),
                     encoding="utf-8")
        return p

    def report(self) -> str:
        rows = ["  " + "-" * 76,
                f"  {'探针':22s} {'结果':>6s}  {'测得值':>10s}  说明",
                "  " + "-" * 76]
        for name, r in self.results.items():
            mark = "通过" if r.passed else "未通过"
            tag = " [部署]" if r.is_deployment_limit else ""
            rows.append(f"  {name:22s} {mark:>6s}  {str(r.value):>10s}  {r.detail[:44]}{tag}")
        rows.append("  " + "-" * 76)
        pol = self.evidence_policy()
        rows.append(f"  能力等级: {self.tier}")
        rows.append(f"  证据策略: 单次最多 {pol['max_images']} 图 · 放大到 {pol['zoom_px']}px · "
                    f"框来自 {pol['boxes_from']} · 投票 {pol['votes']} 次")
        if pol["frames_via"] == "video":
            rows.append(f"  **帧的送法: video 通道** ({pol['video_frames']} 帧 × "
                        f"{pol['video_tokens_per_frame']} token/帧 = "
                        f"{pol['video_frames'] * pol['video_tokens_per_frame']}) "
                        f"—— 不要把帧拼成图")
        else:
            rows.append(f"  帧的送法: 拼图(无 video 通道) · 图像预算 "
                        f"{pol['image_budget']} · 多加帧的代价是{pol['more_frames_costs']}")
        rows.append(f"  filmstrip 可用: {'是' if pol['use_filmstrip'] else '否(改用顺序帧)'} · "
                    f"约束需代码兜底: {'是' if pol['enforce_constraints_in_code'] else '否'}")
        rows.append(f"  启用技术: {', '.join(self.techniques())}")
        return "\n".join(rows)


def probe_model(vlm: VLMClient, *, only: Sequence[str] | None = None,
                verbose: bool = True) -> CapabilityProfile:
    prof = CapabilityProfile(model=vlm.model, endpoint=vlm.base_url)
    for fn in PROBES:
        name = fn.__name__.replace("probe_", "")
        if only and name not in only:
            continue
        if verbose:
            print(f"  探测 {name} ...", flush=True)
        try:
            r = fn(vlm)
        except Exception as e:  # noqa: BLE001 - a broken probe must not stop the suite
            r = ProbeResult(name, False, detail=f"{type(e).__name__}: {e}"[:80])
        prof.results[r.name] = r
        if verbose:
            print(f"     {'通过' if r.passed else '未通过'}  {r.value}  {r.detail[:60]}",
                  flush=True)
    return prof


def load(path: str | Path) -> CapabilityProfile:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    p = CapabilityProfile(d["model"], d["endpoint"], d.get("probed_at", 0.0))
    for k, v in (d.get("probes") or {}).items():
        p.results[k] = ProbeResult(**v)
    return p

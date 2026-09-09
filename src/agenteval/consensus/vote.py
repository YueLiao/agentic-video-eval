"""Multi-phase consensus: keep the findings that survive resampling.

Measured on three sampling phases of the same three clips: shifting which frames
uniform sampling lands on -- same footage, same duration, same content -- changed
every aspect that carried signal, with perturbation noise exceeding
between-model signal on all of them (SNR 0.20-0.87) and the overall ranking
taking three different orders in three runs.

The per-finding view explains why, and how to fix it. Matched on exact frame
span, agreement across phases was **zero**: the same hand defect appears as
`@0-5`, `@1-9`, `@1-10` because the phase moved. Matched on defect type it is
the opposite -- `hand_malformation` appeared in all nine runs across all three
models, while `object_pop`, `motion_stall` and `speed_anomaly` mostly appeared
once. The signal was there; exact-span matching was hiding it.

So findings are clustered by (kind, overlapping time), and a cluster is kept
only if it recurs in a majority of phases. This turns sampling sensitivity from
a defect of the system into a filter, the same move as falsification but along
the sampling axis rather than the reasoning one: a real defect should be
visible whichever frames you happened to look at.

Cost is evidence collection, not judgement -- the extra phases produce more
crops, and the VLM call count per skill is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from agenteval.skills.base import Finding


def _overlap(a: tuple[int, int] | None, b: tuple[int, int] | None,
             slack: int) -> bool:
    """Do two spans refer to the same event?

    ``slack`` absorbs the phase shift itself: the same defect seen at a
    different sampling phase reports boundaries a few frames off, and demanding
    exact equality is what produced 0% agreement.
    """
    if a is None or b is None:
        return True                      # an unlocalized finding matches on kind
    return (min(a[1], b[1]) + slack) >= (max(a[0], b[0]) - slack)


@dataclass
class Cluster:
    kind: str
    aspect: str | None
    members: list[tuple[int, Finding]] = field(default_factory=list)  # (phase, f)

    @property
    def phases(self) -> set[int]:
        return {p for p, _ in self.members}

    @property
    def support(self) -> int:
        return len(self.phases)

    def representative(self, n_phases: int) -> Finding:
        """The member to report, with confidence rescaled by how often it recurred.

        Median severity rather than worst: taking the worst would let one
        alarmed phase set the grade, which is exactly the single-sample
        behaviour this is meant to replace.
        """
        order = {"critical": 0, "major": 1, "minor": 2}
        fs = sorted((f for _, f in self.members),
                    key=lambda f: order.get(f.severity, 3))
        rep = fs[len(fs) // 2]
        spans = [f.t_span for _, f in self.members if f.t_span]
        out = Finding(
            kind=rep.kind, severity=rep.severity,
            t_span=((min(s[0] for s in spans), max(s[1] for s in spans))
                    if spans else None),
            bbox=rep.bbox,
            confidence=min(1.0, max(f.confidence for _, f in self.members)
                           * (self.support / max(1, n_phases))),
            rationale=rep.rationale,
            evidence=sorted({e for _, f in self.members for e in f.evidence}),
            aspect=rep.aspect,
        )
        return out


def cluster_findings(by_phase: Sequence[Sequence[Finding]],
                     *, slack: int = 8) -> list[Cluster]:
    """Group findings that describe the same defect seen at different phases."""
    clusters: list[Cluster] = []
    for phase, findings in enumerate(by_phase):
        for f in findings:
            if f.retracted:
                continue
            hit = None
            for c in clusters:
                if c.kind != f.kind or c.aspect != f.aspect:
                    continue
                if phase in c.phases:
                    continue          # one vote per phase per cluster
                if any(_overlap(f.t_span, m.t_span, slack) for _, m in c.members):
                    hit = c
                    break
            if hit is None:
                clusters.append(Cluster(f.kind, f.aspect, [(phase, f)]))
            else:
                hit.members.append((phase, f))
    return clusters


def consensus(by_phase: Sequence[Sequence[Finding]], *, slack: int = 8,
              min_fraction: float = 0.5) -> tuple[list[Finding], dict[str, Any]]:
    """Keep clusters recurring in at least ``min_fraction`` of phases.

    Returns the surviving findings plus a report, because the discard rate is
    diagnostic in the same way the retraction rate is: near 0% means the phases
    agree and the extra work bought nothing, near 100% means nothing the judge
    reports is reproducible and the scores should not be believed.
    """
    n = len(by_phase)
    if n == 0:
        return [], {"n_phases": 0}
    clusters = cluster_findings(by_phase, slack=slack)
    need = max(1, int(round(min_fraction * n)))
    kept = [c for c in clusters if c.support >= need]
    dropped = [c for c in clusters if c.support < need]
    return (
        [c.representative(n) for c in kept],
        {
            "n_phases": n, "need_support": need,
            "n_clusters": len(clusters), "n_kept": len(kept),
            "n_dropped": len(dropped),
            "discard_rate": round(len(dropped) / max(1, len(clusters)), 3),
            "kept": [{"kind": c.kind, "support": f"{c.support}/{n}"} for c in kept],
            "dropped": [{"kind": c.kind, "support": f"{c.support}/{n}"}
                        for c in dropped],
        },
    )

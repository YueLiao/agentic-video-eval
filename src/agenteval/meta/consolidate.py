"""Deciding which aspects are worth reporting, from evidence.

Twenty-nine aspects is too many to act on and almost certainly more than the
measurement supports. Which ones survive is an empirical question, not a design
preference, so this scores every aspect on five criteria and recommends keep,
merge or drop.

    coverage        how often it is judgeable at all. An aspect that is n/a on
                    nine clips in ten cannot carry a leaderboard column,
                    whatever it measures when it does fire.
    discrimination  between-model variance over within-model variance. An
                    aspect every model scores 9.6 on is measuring the scale, not
                    the models.
    stability       agreement between judges or reruns. An aspect that moves
                    when only the judge changed is noise.
    trust           1 - retraction rate. High retraction means the aspect
                    reliably generates claims that do not survive counter
                    evidence, which is worse than measuring nothing.
    distinctness    1 - max correlation with any other aspect. Two aspects that
                    always move together are one aspect reported twice, and
                    reporting both makes a group look worse than it is.

Deliberately mechanical. The temptation with a taxonomy is to keep every
category because each is describable, and describability is not the test --
whether it *measures* is.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class AspectStats:
    key: str
    label: str = ""
    n_total: int = 0
    n_judgeable: int = 0
    n_findings: int = 0
    n_retracted: int = 0
    scores: list[float] = field(default_factory=list)
    by_model: dict[str, list[float]] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return self.n_judgeable / self.n_total if self.n_total else 0.0

    @property
    def trust(self) -> float:
        tot = self.n_findings + self.n_retracted
        return 1.0 - (self.n_retracted / tot) if tot else 1.0

    @property
    def discrimination(self) -> float:
        """Between-model spread over within-model spread. ~0 means every model
        looks the same here; the aspect is not measuring the models."""
        groups = [v for v in self.by_model.values() if len(v) >= 2]
        if len(groups) < 2:
            return float("nan")
        means = [sum(g) / len(g) for g in groups]
        gm = sum(means) / len(means)
        between = sum((m - gm) ** 2 for m in means) / max(1, len(means) - 1)
        within = 0.0
        n = 0
        for g in groups:
            mu = sum(g) / len(g)
            within += sum((x - mu) ** 2 for x in g)
            n += len(g) - 1
        within = within / n if n else 0.0
        if within < 1e-9:
            return float("inf") if between > 1e-9 else 0.0
        return between / within

    @property
    def spread(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        mu = sum(self.scores) / len(self.scores)
        return math.sqrt(sum((x - mu) ** 2 for x in self.scores) / (len(self.scores) - 1))


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = list(a[:n]), list(b[:n])
    ma, mb = sum(a) / n, sum(b) / n
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if va < 1e-9 or vb < 1e-9:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def collect(results: Iterable[dict[str, Any]]) -> dict[str, AspectStats]:
    """Accumulate per-aspect stats from saved ``result.json`` payloads."""
    stats: dict[str, AspectStats] = {}
    aligned: dict[str, dict[int, float]] = {}
    for i, res in enumerate(results):
        model = str(res.get("model") or res.get("condition", {}).get("model") or "?")
        for key, a in ((res.get("score") or {}).get("aspects") or {}).items():
            st = stats.setdefault(key, AspectStats(key, a.get("label", key)))
            st.n_total += 1
            st.n_findings += int(a.get("n_findings") or 0)
            st.n_retracted += int(a.get("n_retracted") or 0)
            if a.get("judgeable") and a.get("score") is not None:
                st.n_judgeable += 1
                st.scores.append(float(a["score"]))
                st.by_model.setdefault(model, []).append(float(a["score"]))
                aligned.setdefault(key, {})[i] = float(a["score"])
            else:
                r = (a.get("reason") or "unspecified")[:40]
                st.reasons[r] = st.reasons.get(r, 0) + 1
    for st in stats.values():
        st.__dict__["_aligned"] = aligned.get(st.key, {})
    return stats


def correlations(stats: dict[str, AspectStats]) -> dict[str, tuple[str, float]]:
    """For each aspect, its most-correlated peer, over clips both scored."""
    out: dict[str, tuple[str, float]] = {}
    keys = list(stats)
    for k in keys:
        ak = stats[k].__dict__.get("_aligned", {})
        best, bestv = "", 0.0
        for j in keys:
            if j == k:
                continue
            aj = stats[j].__dict__.get("_aligned", {})
            shared = sorted(set(ak) & set(aj))
            if len(shared) < 5:
                continue
            r = _pearson([ak[i] for i in shared], [aj[i] for i in shared])
            if not math.isnan(r) and abs(r) > abs(bestv):
                best, bestv = j, r
        out[k] = (best, bestv)
    return out


@dataclass
class Recommendation:
    key: str
    label: str
    verdict: str                    # keep | merge | drop | needs-data
    reasons: list[str] = field(default_factory=list)
    merge_with: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def recommend(stats: dict[str, AspectStats], *, min_coverage: float = 0.3,
              min_discrimination: float = 0.5, min_trust: float = 0.6,
              max_correlation: float = 0.9,
              min_clips: int = 10) -> list[Recommendation]:
    corr = correlations(stats)
    out: list[Recommendation] = []
    for key, st in sorted(stats.items()):
        m = {"coverage": round(st.coverage, 3), "trust": round(st.trust, 3),
             "spread": round(st.spread, 3),
             "discrimination": round(st.discrimination, 3)
             if not math.isnan(st.discrimination) else float("nan"),
             "n_judgeable": st.n_judgeable}
        peer, r = corr.get(key, ("", 0.0))
        m["max_corr"] = round(r, 3)
        reasons: list[str] = []
        verdict = "keep"

        if st.n_total < min_clips or st.n_judgeable < min_clips // 2:
            verdict = "needs-data"
            reasons.append(f"只在 {st.n_judgeable}/{st.n_total} 条上可判,样本不足以判断")
        else:
            if st.coverage < min_coverage:
                verdict = "drop"
                top = max(st.reasons.items(), key=lambda kv: kv[1])[0] if st.reasons else ""
                reasons.append(f"可判率仅 {st.coverage:.0%},最常见原因:{top}")
            if st.trust < min_trust:
                verdict = "drop"
                reasons.append(f"撤回率 {1-st.trust:.0%},指控多数经不起反证")
            if (not math.isnan(st.discrimination)
                    and st.discrimination < min_discrimination):
                if verdict == "keep":
                    verdict = "drop"
                reasons.append(f"区分度 {st.discrimination:.2f},各模型表现无差别")
            if st.spread < 0.15 and st.n_judgeable >= min_clips:
                if verdict == "keep":
                    verdict = "drop"
                reasons.append(f"分数标准差仅 {st.spread:.2f},几乎恒定")
            if abs(r) > max_correlation and peer:
                if verdict == "keep":
                    verdict = "merge"
                reasons.append(f"与 {peer} 相关性 {r:+.2f},两者在测同一件事")
        out.append(Recommendation(key, st.label, verdict, reasons,
                                  peer if verdict == "merge" else "", m))
    return out


def report(recs: Sequence[Recommendation]) -> str:
    order = {"keep": 0, "merge": 1, "needs-data": 2, "drop": 3}
    rows = ["  verdict     aspect               cov   trust  disc   spread  corr   reason",
            "  " + "-" * 100]
    for r in sorted(recs, key=lambda x: (order.get(x.verdict, 9), x.key)):
        m = r.metrics
        disc = m.get("discrimination")
        disc_s = "  n/a" if disc is None or (isinstance(disc, float) and math.isnan(disc)) \
            else f"{disc:5.2f}"
        rows.append(
            f"  {r.verdict:11s} {r.label or r.key:18s} "
            f"{m.get('coverage',0):.2f}  {m.get('trust',0):.2f}  {disc_s}  "
            f"{m.get('spread',0):5.2f}  {m.get('max_corr',0):+.2f}  "
            + ("; ".join(r.reasons)[:64] if r.reasons else ""))
    counts: dict[str, int] = {}
    for r in recs:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    rows.append("  " + "-" * 100)
    rows.append("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(rows)


def load_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        try:
            out.append(json.loads(Path(p).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out

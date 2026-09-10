"""Bradley-Terry: recover a point-wise scale from pairwise comparisons.

Pairwise and point-wise are not alternatives here. The framework produces
directions more reliably than magnitudes -- 61.8% direction accuracy against
50% random, while overall accuracy including ties peaked at 38.3% against a
34.5% trivial baseline. Bradley-Terry is how a magnitude is rebuilt from
directions: each comparison carries little information, but they compose into a
latent scale where the distance between two items reflects how often one beats
the other.

That gives a point-wise score whose *spacing* is meaningful, which is exactly
what subtracting two independently-assigned scores failed to provide.

Fitted by gradient ascent on the log-likelihood. Ties are handled as half a win
each way, which is the standard treatment and keeps a tie from being silently
dropped -- 34-41% of this benchmark is ties, so dropping them would discard most
of the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class BTResult:
    scores: dict[str, float]                  # latent ability, mean-centred
    n_items: int = 0
    n_comparisons: int = 0
    iterations: int = 0
    log_likelihood: float = 0.0
    n_wins: dict[str, float] = field(default_factory=dict)

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: -kv[1])

    def to_scale(self, lo: float = 0.0, hi: float = 10.0) -> dict[str, float]:
        """Map latent abilities onto a fixed range for reporting.

        Linear on the latent scale, so equal score differences still mean equal
        odds ratios. The absolute placement is arbitrary -- Bradley-Terry
        identifies differences, not levels -- so this is for presentation, and
        anchoring it to known-quality references is what would make the levels
        mean something.
        """
        if not self.scores:
            return {}
        vs = list(self.scores.values())
        lo_v, hi_v = min(vs), max(vs)
        if hi_v - lo_v < 1e-9:
            return {k: (lo + hi) / 2 for k in self.scores}
        return {k: lo + (hi - lo) * (v - lo_v) / (hi_v - lo_v)
                for k, v in self.scores.items()}


def fit(comparisons: Iterable[tuple[str, str, float]], *, iters: int = 400,
        lr: float = 0.12, reg: float = 1e-3) -> BTResult:
    """Fit abilities from (winner, loser, weight) triples.

    A tie is passed as two entries with weight 0.5 in each direction. `reg`
    pulls abilities toward zero, which keeps an item that won all of its few
    comparisons from running off to infinity -- common here, since many items
    appear in only one or two pairs.
    """
    comps = [(a, b, float(w)) for a, b, w in comparisons if a != b and w > 0]
    if not comps:
        return BTResult({})
    items = sorted({x for a, b, _ in comps for x in (a, b)})
    idx = {k: i for i, k in enumerate(items)}
    theta = [0.0] * len(items)
    wins: dict[str, float] = {k: 0.0 for k in items}
    for a, b, w in comps:
        wins[a] += w

    ll = 0.0
    for it in range(iters):
        grad = [0.0] * len(items)
        ll = 0.0
        for a, b, w in comps:
            ia, ib = idx[a], idx[b]
            d = theta[ia] - theta[ib]
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, d))))
            ll += w * math.log(max(p, 1e-12))
            g = w * (1.0 - p)
            grad[ia] += g
            grad[ib] -= g
        for i in range(len(items)):
            grad[i] -= reg * theta[i]
            theta[i] += lr * grad[i] / max(1.0, len(comps) / len(items))
        m = sum(theta) / len(theta)
        theta = [t - m for t in theta]
    return BTResult({k: theta[idx[k]] for k in items}, len(items), len(comps),
                    iters, ll, wins)


def from_labels(pairs: Sequence[tuple[str, str, str]]) -> BTResult:
    """Fit from (item_a, item_b, outcome) where outcome is 'a' | 'b' | 'same'."""
    comps: list[tuple[str, str, float]] = []
    for a, b, o in pairs:
        if o == "a":
            comps.append((a, b, 1.0))
        elif o == "b":
            comps.append((b, a, 1.0))
        else:
            comps.append((a, b, 0.5))
            comps.append((b, a, 0.5))
    return fit(comps)


def agreement(bt: BTResult, pairs: Sequence[tuple[str, str, str]],
              tau: float = 0.0) -> dict[str, float]:
    """How often the fitted scale reproduces the outcomes it was fitted on, or
    on held-out pairs. On held-out data this is the number that matters."""
    ok = n = 0
    for a, b, o in pairs:
        if a not in bt.scores or b not in bt.scores:
            continue
        d = bt.scores[a] - bt.scores[b]
        pred = "same" if abs(d) <= tau else ("a" if d > 0 else "b")
        ok += pred == o
        n += 1
    return {"accuracy": ok / n if n else 0.0, "n": n}

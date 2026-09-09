"""Aggregators for folding several aspect scores into one number.

Taking the worst was the obvious choice and measured badly. On 100 human-labelled
pairs it collapsed 8 of 17 strongly-preferred pairs to a zero margin and left a
quarter of all pairs undecidable, because when both clips are poor each is capped
by its own worst aspect and they land on the same value -- exactly the case a
human finds easiest to call.

The intent behind `min` is still right: one disqualifying failure should pull the
score down, and averaging lets a clean sub-aspect mask a broken one. What is
wrong is letting the worst aspect be the *only* thing that speaks. Soft-min keeps
the lean toward the worst while the rest still moves the result.

Which one to use is a calibration decision, not a design preference, and it must
be made on the dev split -- comparing six aggregators on the reporting set and
keeping the best is choosing the answer.
"""

from __future__ import annotations

from typing import Callable, Sequence

Aggregator = Callable[[Sequence[float]], float | None]


def worst(v: Sequence[float]) -> float | None:
    """Lowest score. Saturates when several aspects are bad."""
    return min(v) if v else None


def mean(v: Sequence[float]) -> float | None:
    """Plain average. Lets one clean aspect mask a broken one."""
    return sum(v) / len(v) if v else None


def worst_two(v: Sequence[float]) -> float | None:
    return sum(sorted(v)[:2]) / min(2, len(v)) if v else None


def harmonic(v: Sequence[float]) -> float | None:
    """Harmonic mean, floored so a zero cannot annihilate the result."""
    return len(v) / sum(1.0 / max(x, 0.5) for x in v) if v else None


def total_deduction(v: Sequence[float]) -> float | None:
    """Ten minus the sum of all deductions: counts defects rather than ranking
    them, so many small problems can outweigh one large one."""
    return 10.0 - sum(10.0 - x for x in v) if v else None


def soft_min(v: Sequence[float], p: float = 4.0) -> float | None:
    """Power mean with negative exponent: leans toward the worst without being
    decided by it. p controls the lean -- higher approaches `worst`, p=1 gives
    the harmonic mean."""
    return (sum(x ** -p for x in v) / len(v)) ** (-1.0 / p) if v else None


AGGREGATORS: dict[str, Aggregator] = {
    "worst": worst,
    "mean": mean,
    "worst_two": worst_two,
    "harmonic": harmonic,
    "total_deduction": total_deduction,
    "soft_min": soft_min,
}

#: Default. Chosen for its semantics -- leans toward the worst aspect without
#: saturating -- and pending calibration on the dev split.
DEFAULT = "soft_min"

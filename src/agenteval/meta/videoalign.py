"""VideoAlign's reward-model accuracy metrics, so our numbers are theirs.

Ported from /pub/evaluation_group/cy/VideoAlign/calc_accuracy.py, which
implements the tie calibration of arXiv:2305.14324. Two things in it corrected
what this repo had been reporting:

* `acc_without_ties` forbids the model to abstain -- epsilon is -1, so every
  human-non-tie pair must be given a direction. Reporting accuracy over only
  the pairs a margin left decidable is a different and much kinder number: on
  val_ac the same scores read 66.1% over 316 decidable pairs and 61.5% over all
  587.
* `acc_with_ties` sweeps the tie threshold to the value that maximises accuracy
  rather than fixing one, so a tie band is credited at its best rather than at
  whatever cutoff happened to be picked.

Both take `h` (human: +1 A, -1 B, 0 tie) and `m` (model: score_A - score_B).
"""

from __future__ import annotations

from typing import Sequence


def suff_stats(h: Sequence[float], m: Sequence[float], epsilon: float):
    """Contingency counts at a given tie threshold.

    C  both call a direction and agree        D  both call a direction, disagree
    Th human ties, model does not             Tm model ties, human does not
    Thm both tie
    """
    C = D = Th = Tm = Thm = 0
    for hi, mi in zip(h, m):
        if hi == 0 and abs(mi) <= epsilon:
            Thm += 1
        elif hi == 0:
            Th += 1
        elif abs(mi) <= epsilon:
            Tm += 1
        elif hi * mi > 0:
            C += 1
        else:
            D += 1
    return C, D, Th, Tm, Thm


def calc_acc(C: int, D: int, Th: int, Tm: int, Thm: int) -> float:
    return (C + Thm) / max(1, C + D + Th + Tm + Thm)


def acc_with_ties(h: Sequence[float], m: Sequence[float]) -> tuple[float, float]:
    """Best achievable accuracy over all tie thresholds, and the threshold.

    Walking the pairs in order of |m| lets each threshold be evaluated by
    updating the counts for one pair rather than recounting, which is what makes
    the sweep affordable.
    """
    C, D, Th, Tm, Thm = suff_stats(h, m, -1)
    stat = {"C": C, "D": D, "Th": Th, "Tm": Tm, "Thm": Thm}
    best, best_eps, eps = float("-inf"), 0.0, -1.0
    for hi, mi in sorted(zip(h, m), key=lambda x: abs(x[1])):
        if hi == 0 and abs(mi) < eps:
            stat["Thm"] -= 1
        elif hi == 0:
            stat["Th"] -= 1
        elif abs(mi) < eps:
            stat["Tm"] -= 1
        elif hi * mi > 0:
            stat["C"] -= 1
        else:
            stat["D"] -= 1
        eps = abs(mi)
        if hi == 0 and abs(mi) <= eps:
            stat["Thm"] += 1
        elif hi == 0:
            stat["Th"] += 1
        elif abs(mi) <= eps:
            stat["Tm"] += 1
        elif hi * mi > 0:
            stat["C"] += 1
        else:
            stat["D"] += 1
        cur = calc_acc(**stat)
        if cur > best:
            best, best_eps = cur, eps
    return best, best_eps


def acc_without_ties(h: Sequence[float], m: Sequence[float]) -> float:
    """Direction accuracy with abstention disallowed (epsilon = -1)."""
    C, D, _Th, Tm, _Thm = suff_stats(h, m, -1)
    return C / max(1, C + D + Tm)

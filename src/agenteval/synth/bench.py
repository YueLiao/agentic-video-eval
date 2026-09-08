"""Measure whether the suspicion map actually finds defects.

Two metrics, answering two different questions. Reporting only the first is how
a search index gets to look good while being useless, and vice versa.

``sensitivity`` (paired)
    Run the maps on the clean source *and* on the injected copy, and compare the
    fused score at the defect's own cells against the same cells in the clean
    video. This isolates the injection from whatever the source already
    contained, so it answers: *are these signals responsive to this defect type
    at all?* It needs a clean reference, so it is a validation-time metric only.

``hit@k`` (unpaired)
    Rank of the first locus overlapping the defect, from the injected video
    alone — exactly what the agent sees at run time. It answers: *does the
    defect surface above this video's own noise floor, within the probe budget?*

``uniform@n`` is the honest baseline: n evenly spaced frames, covered if any
lands inside the defect's span. That is what a fixed-sample judge gets to see,
so any claim that search beats fixed sampling has to beat this number.

Source material caveat: injecting into already-generated video means the
"clean" reference is not clean. It depresses hit@k (the source's own defects
compete for the top ranks) and leaves sensitivity unaffected. Prefer real
footage as source where available.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agenteval.media.clip import uniform_indices
from agenteval.signals import suspicion
from agenteval.synth import inject
from agenteval.synth.inject import Defect


def _overlap_1d(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _iou_2d(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def locus_hits(locus: suspicion.SuspicionLocus, d: Defect,
               *, min_iou: float = 0.05) -> bool:
    """A locus hits if it overlaps in time and (when the defect is localized)
    in space. The spatial bar is deliberately low: a locus only has to be close
    enough that a crop around it contains the defect."""
    if _overlap_1d(locus.t_span, d.t_span) <= 0:
        return False
    if d.bbox is None:
        return True
    return _iou_2d(locus.bbox, d.bbox) >= min_iou


def defect_cells(d: Defect, shape: tuple[int, int, int],
                 grid: tuple[int, int] = suspicion.GRID) -> np.ndarray:
    """Boolean (T, GY, GX) mask of the cells the defect actually occupies."""
    T, gy, gx = shape
    m = np.zeros(shape, bool)
    t0, t1 = max(0, d.t_span[0]), min(T, d.t_span[1])
    if t1 <= t0:
        return m
    if d.bbox is None:
        m[t0:t1] = True
        return m
    x, y, w, h = d.bbox
    j0, j1 = int(np.floor(x * gx)), int(np.ceil((x + w) * gx))
    i0, i1 = int(np.floor(y * gy)), int(np.ceil((y + h) * gy))
    m[t0:t1, max(0, i0):min(gy, max(i0 + 1, i1)), max(0, j0):min(gx, max(j0 + 1, j1))] = True
    return m


@dataclass
class CaseResult:
    case_id: str
    defect_type: str
    strength: float
    sensitivity: float          # fused score lift at the defect's cells (paired)
    hit_rank: int | None        # rank of first overlapping locus, None = missed
    n_loci: int
    uniform_hit: dict[int, bool]  # n -> did any of n uniform frames land inside
    dominant: str               # which signal fired, if hit

    def to_json(self) -> dict[str, Any]:
        return {**self.__dict__, "uniform_hit": {str(k): v for k, v in self.uniform_hit.items()}}


def run_case(src: str | Path, out_dir: str | Path, case_id: str, defect: Defect,
             *, uniform_ns: Sequence[int] = (8, 16, 32),
             max_frames: int | None = None,
             clean_maps: dict[str, np.ndarray] | None = None,
             clean_video: str | Path | None = None) -> tuple[CaseResult, dict[str, np.ndarray]]:
    """Inject one defect, measure both metrics. Returns (result, clean_maps)."""
    if clean_maps is None:
        clean_gt = inject.make_case(src, out_dir, case_id=f"{case_id}__clean",
                                    defects=[], max_frames=max_frames)
        clean_video = clean_gt["video"]
        clean_maps = suspicion.compute_maps(clean_video, max_frames=max_frames)

    gt = inject.make_case(src, out_dir, case_id=case_id, defects=[defect],
                          max_frames=max_frames)
    maps = suspicion.compute_maps(gt["video"], max_frames=max_frames)
    fused_i = suspicion.fuse(maps)
    fused_c = suspicion.fuse(clean_maps)
    n = min(len(fused_i), len(fused_c))
    fused_i, fused_c = fused_i[:n], fused_c[:n]

    cells = defect_cells(defect, fused_i.shape)
    if cells.any():
        lift = float(np.median(fused_i[cells] - fused_c[cells]))
    else:
        lift = 0.0

    loci = suspicion.extract_loci(maps)
    rank = next((i for i, l in enumerate(loci) if locus_hits(l, defect)), None)
    dom = loci[rank].dominant if rank is not None else ""

    total = gt["n_frames"]
    uni = {k: any(defect.t_span[0] <= i < defect.t_span[1]
                  for i in uniform_indices(total, k)) for k in uniform_ns}

    return CaseResult(
        case_id=case_id, defect_type=defect.type, strength=defect.strength,
        sensitivity=round(lift, 4), hit_rank=rank, n_loci=len(loci),
        uniform_hit=uni, dominant=dom,
    ), clean_maps


def summarize(results: Sequence[CaseResult], *, ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, Any]:
    by_type: dict[str, list[CaseResult]] = {}
    for r in results:
        by_type.setdefault(r.defect_type, []).append(r)

    def block(rs: Sequence[CaseResult]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(rs)}
        for k in ks:
            out[f"hit@{k}"] = round(
                sum(1 for r in rs if r.hit_rank is not None and r.hit_rank < k) / len(rs), 3)
        for k in sorted(rs[0].uniform_hit):
            out[f"uniform@{k}"] = round(
                sum(1 for r in rs if r.uniform_hit[k]) / len(rs), 3)
        out["sensitivity_median"] = round(statistics.median(r.sensitivity for r in rs), 3)
        out["sensitivity_pos_frac"] = round(
            sum(1 for r in rs if r.sensitivity > 0) / len(rs), 3)
        doms = [r.dominant for r in rs if r.dominant]
        out["dominant"] = max(set(doms), key=doms.count) if doms else ""
        return out

    return {
        "overall": block(results),
        "by_type": {t: block(rs) for t, rs in sorted(by_type.items())},
    }


def report(summary: dict[str, Any]) -> str:
    ov = summary["overall"]
    ks = [k for k in ov if k.startswith("hit@")]
    us = [k for k in ov if k.startswith("uniform@")]
    head = ["defect_type", "n", *ks, *us, "sens_med", "sens>0", "dominant"]
    rows = []
    for t, b in summary["by_type"].items():
        rows.append([t, str(b["n"]), *[f"{b[k]:.2f}" for k in ks],
                     *[f"{b[k]:.2f}" for k in us],
                     f"{b['sensitivity_median']:+.2f}", f"{b['sensitivity_pos_frac']:.2f}",
                     b["dominant"]])
    rows.append(["ALL", str(ov["n"]), *[f"{ov[k]:.2f}" for k in ks],
                 *[f"{ov[k]:.2f}" for k in us],
                 f"{ov['sensitivity_median']:+.2f}", f"{ov['sensitivity_pos_frac']:.2f}",
                 ov["dominant"]])
    w = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
    fmt = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r))
    sep = "  ".join("-" * x for x in w)
    return "\n".join([fmt(head), sep, *[fmt(r) for r in rows]])

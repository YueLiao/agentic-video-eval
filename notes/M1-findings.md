# M1 findings — does the suspicion map actually find defects?

Method: inject defects of known type / extent / strength into source video, then
measure (a) **sensitivity**, the paired lift in fused suspicion score at the defect's
own cells versus the same cells in the clean source, and (b) **hit@k**, the rank of
the first locus overlapping the defect, from the injected video alone.
Baseline `uniform@n` = n evenly spaced frames, counted as covered if any lands inside
the defect's span.

32 cases, 8 defect types × 4, source = 81-frame 720p clips. ~3.4 s/case.

## Result 1 — anomaly is two-sided, and assuming otherwise is a structural blind spot

The first run scored `frame_drop` and `frame_repeat` at **negative** sensitivity
(−0.16, −0.29). The cause is not a tuning issue: warping a repeated frame onto itself
leaves almost no residual, so a "high residual is suspicious" index scores a frozen
clip as *cleaner than normal*. A stutter is an anomaly of **too little** change.

Adding a `freeze` signal — how far change energy falls below the local temporal
median — moved both types to `hit@3 = 1.00`, sensitivity **+0.67**, positive in 4/4:

| type | before (sens / hit@10) | after (sens / hit@3) |
|---|---|---|
| `frame_drop` | −0.16 / 0.75 | **+0.67 / 1.00** |
| `frame_repeat` | −0.29 / 0.50 | **+0.67 / 1.00** |

## Result 2 — defect duration decides whether search can beat fixed sampling

With spans of 6–20 frames (up to 25% of an 81-frame clip) uniform sampling hits by
luck and the comparison is uninformative — `uniform@8 = 0.84`, and `uniform@16 = 1.00`.
Restricting to **short transient defects (2–5 frames)**, which is what generation
artifacts actually look like, the ordering reverses:

| | hit@1 | hit@3 | hit@5 | hit@10 |
|---|---|---|---|---|
| suspicion map | 0.09 | 0.34 | 0.41 | **0.50** |

| | uniform@8 | uniform@16 | uniform@32 |
|---|---|---|---|
| baseline | 0.31 | 0.53 | 0.97 |

At a comparable budget the search index wins (0.41–0.50 at 5–10 probes vs 0.31 at 8
frames). **But `uniform@32 = 0.97`** — dense uniform sampling covers almost everything
*temporally*. So the temporal case for search is real but modest, and the honest
argument for this architecture has to rest on the **spatial/resolution** axis: a
0.2×0.2 region inside a frame downsampled into a VLM's token budget is unreadable
even when that frame was sampled. This benchmark does not yet measure that, and
`uniform@n` here is generous to the baseline because it credits mere temporal
coverage. Measuring the resolution axis is the next thing that matters.

## Result 3 — what still fails

`region_shuffle` 0.00, `patch_jump` 0.25, `patch_swap` 0.25, `affine_warp` 0.25 at
hit@10. These share a property: content changes while first-order statistics do not.
Permuting a region's frames preserves its marginal distribution; the current signals
are all first-order in the residual and cannot see it. Catching this class needs a
correspondence/embedding-based signal, not another intensity statistic.

## Caveat that limits all of the above

Sources are themselves generated videos, so the "clean" reference is not clean and
competes for top ranks — this depresses `hit@k` while leaving `sensitivity`
unaffected. Real footage as source would tighten `hit@k`. Injected artifacts are also
not generation artifacts; this measures a capability lower bound and guards
regressions, and does not substitute for human-localized defects on real video.

## Reproduce

```bash
python scripts/run_injection_bench.py --sources 'DIR/*.mp4' --out bench/m1 \
    --per-type 4 --max-frames 81 --seed 1
```

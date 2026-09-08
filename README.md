# agentic-video-eval

An agentic evaluation framework for **generated short videos** (5–15 s), producing
multi-dimensional scores plus an auditable defect report.

## Why

The standard recipe — sample 8 uniform frames, hand a VLM one fixed rubric prompt,
read back a 1–5 — is not under-prompted. It is blocked by three separable physical
limits:

| Limit | What goes wrong | The only fix |
|---|---|---|
| **Sampling** | 8 of 120 frames is a 6.7% blind sample. Generation defects are *transient* and cluster at motion peaks and transitions. A defect the sample missed cannot be scored, however good the prompt. | adaptive **search** |
| **Resolution** | A face at 6% of frame width, inside a VLM's ~700-token image budget, is ~30 px. The evidence is *physically absent* from the model's input. | **zoom** — native-resolution crops |
| **Prior** | VLM pretraining is natural, artifact-free video. It has weak priors for six-fingered hands, bone lengths that change over time, texture that crawls. It does not know what to look for. | **external priors** — detectors and derived invariants |

Being *agentic* is what lets the system choose which fix to apply, and how deep, per
input. A still-life prompt and a dance close-up should not cost the same or take the
same path.

## What it does differently

**Evaluation is split into two tasks that need different machinery.**

- **Conformance** — did the video do what the condition asked? A checklist over a
  compiled requirement graph.
- **Integrity** — is what it made coherent at all, regardless of the condition? A
  *search* for localized defects.

Collapsing both into one Likert score is why current benchmarks saturate on strong
models: conformance is near-ceiling, and integrity defects go unscored because they
were never seen.

**The search index is built without a VLM.** A suspicion map over `(t, region)` is
computed from signal processing alone. Its lead signal is the **motion-compensated
residual**: warp frame *t* into *t+1* by the optical flow and keep what is left.
Content that merely moved is explained away; the remainder is change the motion field
cannot account for — which is what generative instability looks like. Naive frame
differencing flags all fast motion; this does not.

**Every alleged defect faces counter-evidence.** A judge asked to find problems will
always find problems. Integrity has no gold answer to catch that, so each alleged
defect is re-presented alongside the same region at neighbouring times plus a clean
contrast region, and the judge must defend or retract it. **Retraction rate is a
first-class reported metric.**

**Scores are computed, not asked for.** A VLM's absolute 1–10 is uncalibrated and
incomparable across conditions. Scores are derived deterministically from a
`DefectInventory` (typed, localized, severity-rated, falsification-survived) and a
requirement `SatisfactionVector`. Every deducted point points at a
`(frame span, box, type)` a human can check. Purely subjective quality is handled
separately, by pairwise comparison and Bradley–Terry, never an absolute score.

**Validation starts with synthetic defect injection.** Inject defects of known type,
extent and strength into real video and you get detection recall, localization IoU,
severity monotonicity, and — on clean inputs — a direct **hallucination rate**, at
zero labelling cost. This decouples "can it find defects" from "do humans agree with
the score", and makes the central claim falsifiable before any annotator is booked.

See [DESIGN.md](DESIGN.md) for the full design.

## Status

Early. Milestone 1 (suspicion mapping + defect injection + measurement harness) is
under construction; nothing here is validated yet.

## Layout

```
src/agenteval/
  media/       clip decoding, frame access, sampling
  tools/       tool contract and registry
  signals/     VLM-free suspicion mapping
  synth/       synthetic defect injection (ground truth for free)
  planning/    offline condition -> requirement graph compilation
  engine/      evidence bus, budgets, agent loops
  scoring/     defect inventory -> dimension scores
  interop/     optional, one-way adapters to an existing toolchain
```

## Setup

```bash
. env/bootstrap.sh     # PYTHONPATH, decode backend, health check
```

Python ≥3.10, `numpy`, `opencv-python`. Detector tools additionally want
`mediapipe`, `ultralytics`, `pyiqa`; each is optional and degrades to unavailable
rather than failing the run.

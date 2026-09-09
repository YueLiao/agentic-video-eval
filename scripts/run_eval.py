#!/usr/bin/env python3
"""Run the full agentic evaluation over one or more videos with a live VLM.

    python scripts/run_eval.py --videos a.mp4 b.mp4 --condition "..." --out runs/x

Comparing the same condition across models is the useful mode: a single clip
only shows that the pipeline runs, whereas the same requirement graph applied to
several candidates shows whether an aspect separates them, which is what the
consolidation analysis needs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.engine.loop import LoopBudget                       # noqa: E402
from agenteval.engine.orchestrator import evaluate                 # noqa: E402
from agenteval.llm.client import VLMClient                         # noqa: E402
from agenteval.planning.compiler import compile_condition          # noqa: E402
from agenteval.skills.conformance import (ActionConformance,       # noqa: E402
                                          CameraConformance,
                                          SemanticConformance)
from agenteval.skills.human_integrity import HumanIntegrity        # noqa: E402
from agenteval.skills.motion_quality import MotionQuality          # noqa: E402
from agenteval.skills.physical_integrity import PhysicalIntegrity  # noqa: E402
from agenteval.skills.static_integrity import StaticIntegrity      # noqa: E402
from agenteval.skills.temporal_integrity import TemporalIntegrity  # noqa: E402


def build_client(cache: Path, log: Path) -> VLMClient:
    return VLMClient(
        model=os.environ.get("AGENTEVAL_VLM_MODEL", "gemma-4-31b-it"),
        provider=os.environ.get("AGENTEVAL_VLM_PROVIDER", "openai"),
        base_url=os.environ.get("AGENTEVAL_VLM_BASE_URL",
                                "http://127.0.0.1:8005/v1"),
        api_key_env=os.environ.get("AGENTEVAL_VLM_KEY_ENV") or None,
        max_tokens=1536, timeout_s=300, cache_dir=cache, log_path=log,
    )


def skills_for(out: Path, graph, only: list[str] | None):
    all_skills = {
        "temporal_integrity": lambda: TemporalIntegrity(out / "temporal", max_loci=6),
        "motion_quality": lambda: MotionQuality(out / "motion"),
        "static_integrity": lambda: StaticIntegrity(out / "static", sweep_calls=4),
        "human_integrity": lambda: HumanIntegrity(out / "human"),
        "physical_integrity": lambda: PhysicalIntegrity(out / "physical"),
        "semantic_conformance": lambda: SemanticConformance(out / "sem", graph),
        "action_conformance": lambda: ActionConformance(out / "act", graph),
        "camera_conformance": lambda: CameraConformance(out / "cam", graph),
    }
    return ({k: v for k, v in all_skills.items() if k in only} if only
            else all_skills)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--condition-id", default="c0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skills", nargs="*", default=None)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--no-falsify", action="store_true")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip LLM condition compilation (use the thin fallback)")
    ap.add_argument("--perturb", default=None,
                    choices=["frame_phase", "locus_order", "skill_order"])
    ap.add_argument("--perturb-seed", type=int, default=0)
    ap.add_argument("--phases", type=int, default=1,
                    help="sampling phases to run and require consensus across; "
                         "1 disables consensus")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Perturbations vary choices that carry no information about the video, so a
    # score that moves under them was never measuring the video.
    if args.perturb == "frame_phase":
        from agenteval.media.clip import set_sample_phase
        set_sample_phase((args.perturb_seed * 0.37) % 1.0)
    elif args.perturb == "locus_order":
        import agenteval.signals.suspicion as _sus
        _orig = _sus.extract_loci
        import random as _r
        def _shuffled(*a, **k):
            loci = _orig(*a, **k)
            rng = _r.Random(args.perturb_seed)
            # reorder loci whose scores are close: their ranking is arbitrary
            rng.shuffle(loci)
            loci.sort(key=lambda l: -round(l.score, 1))
            for n, l in enumerate(loci):
                l.locus_id = f"L{n:02d}"
            return loci
        _sus.extract_loci = _shuffled

    vlm = build_client(out / "cache", out / "vlm_calls.jsonl")
    print(f"model={vlm.model} endpoint={vlm.base_url}")

    t0 = time.time()
    graph = compile_condition(args.condition_id, args.condition,
                              None if args.no_compile else vlm)
    print(f"compiled {len(graph.requirements)} requirements "
          f"({time.time()-t0:.1f}s, compiler={graph.meta.get('compiler')})")
    for r in graph.requirements:
        print(f"   {r.rid} [{r.kind}/{r.verify}] {r.text}")
    graph.save(out / "graph.json")

    results = []
    if args.perturb == "skill_order":
        import random as _r
        _r.Random(args.perturb_seed)  # order applied per-video below
    for path in args.videos:
        name = Path(path).parent.name + "__" + Path(path).stem
        vdir = out / name
        print(f"\n=== {name} ===", flush=True)
        t = time.time()
        res = evaluate(path, {"prompt": args.condition},
                       skills_for(vdir, graph, args.skills), vlm,
                       out_dir=vdir, graph=graph,
                       budget=LoopBudget(max_rounds=args.max_rounds,
                                         max_vlm_calls=10, max_tool_calls=24,
                                         max_wall_s=600),
                       do_falsify=not args.no_falsify,
                       phases=tuple(round(k / args.phases, 3)
                                    for k in range(args.phases)))
        results.append((name, res))
        print(f"  {time.time()-t:.0f}s  vlm={res.vlm_calls} tools={res.tool_calls}")
        if res.consensus:
            c = res.consensus
            print(f"  共识: {c['n_clusters']} 簇 → 丢弃 {c['n_dropped']} "
                  f"(丢弃率 {c['discard_rate']:.0%}, {len(c['phases'])} 个相位)")
        print(res.score.table())

    if len(results) > 1:
        print("\n\n=== 跨模型对比 ===")
        keys = [k for k in results[0][1].score.aspects]
        print(f"  {'aspect':22s} " + "".join(f"{n[:16]:>18s}" for n, _ in results))
        for k in keys:
            cells = []
            any_judged = False
            for _n, r in results:
                a = r.score.aspects[k]
                if a.judgeable and a.score is not None:
                    cells.append(f"{a.score:18.2f}"); any_judged = True
                else:
                    cells.append(f"{'—':>18s}")
            if any_judged:
                lab = results[0][1].score.aspects[k].label
                print(f"  {lab:22s} " + "".join(cells))
        print(f"  {'总分':22s} " + "".join(f"{r.score.overall:18.2f}" for _, r in results))

    json.dump({n: r.to_json() for n, r in results},
              open(out / "all_results.json", "w"), ensure_ascii=False, indent=1)
    u = vlm.total
    print(f"\ntokens {u.prompt_tokens} in / {u.completion_tokens} out, "
          f"{u.n_images} images -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

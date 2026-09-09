#!/usr/bin/env python3
"""Probe a VLM endpoint and write its capability profile.

    python scripts/probe_vlm.py --out profiles/gemma-4-31b.json

Run once per (model, deployment). Deployment-limited results are marked, since
restarting the server with different flags changes them while the model's own
capabilities stay put.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.capability.profile import probe_model   # noqa: E402
from agenteval.llm.client import VLMClient             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    vlm = VLMClient(
        model=os.environ.get("AGENTEVAL_VLM_MODEL", "gemma-4-31b-it"),
        base_url=os.environ.get("AGENTEVAL_VLM_BASE_URL", "http://127.0.0.1:8005/v1"),
        api_key_env=os.environ.get("AGENTEVAL_VLM_KEY_ENV") or None,
        max_tokens=800, timeout_s=240)
    print(f"探测 {vlm.model} @ {vlm.base_url}\n")
    prof = probe_model(vlm, only=args.only)
    print("\n" + prof.report())
    prof.save(args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# Start a local OpenAI-compatible VLM endpoint for agenteval.
#
#   bash scripts/serve_vlm.sh [MODEL_PATH] [PORT] [GPUS]
#
# The multimodal limit matters: skills send up to `Skill.max_images` images in
# one call (14 for the ORDERED motion skills), and a server started with the
# default cap rejects those requests outright rather than degrading.
set -euo pipefail
MODEL="${1:-/pub/evaluation_group/yue/models/gemma-4-31b-it}"
PORT="${2:-8005}"
GPUS="${3:-5}"
NAME="$(basename "$MODEL")"
TP=$(awk -F, '{print NF}' <<< "$GPUS")
VLLM="${VLLM_BIN:-/ning/vllm_env/bin/vllm}"

echo "[serve_vlm] $NAME  port=$PORT  gpus=$GPUS  tp=$TP"
CUDA_VISIBLE_DEVICES="$GPUS" "$VLLM" serve "$MODEL" \
  --served-model-name "$NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":16}' \
  --trust-remote-code

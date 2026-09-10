#!/usr/bin/env bash
# Start a local OpenAI-compatible VLM endpoint for agenteval.
#
#   bash scripts/serve_vlm.sh [MODEL_PATH] [PORT] [GPUS]
#
# Context and image caps are deployment choices, not model limits, and the first
# deployment got them badly wrong: this model is 256K-native and was served at
# 32K with an image cap of 16, so motion quality was being judged from six
# frames at 300px. That ceiling was then read as a property of the model.
# Override with MAX_LEN / MAX_IMG.
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
  --max-model-len "${MAX_LEN:-131072}" \
  --limit-mm-per-prompt "{\"image\":${MAX_IMG:-48}}" \
  --trust-remote-code

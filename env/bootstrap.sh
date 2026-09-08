#!/usr/bin/env bash
# Source this (don't execute) to set up an agenteval shell:  . env/bootstrap.sh
# Idempotent. Prints a one-line health summary at the end.

export AGENTEVAL_ROOT="${AGENTEVAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Optional interop with an existing in-house eval toolkit; unset is fine.
export AGENTEVAL_INTEROP_SRC="${AGENTEVAL_INTEROP_SRC:-}"
[ -n "${HF_HOME:-}" ] || export HF_HOME="$HOME/.cache/huggingface"

# Pin the decode backend so frame indices are identical across every tool,
# cache entry and judge call. Mismatched backends silently shift indices.
export AGENTEVAL_DECODE_BACKEND="${AGENTEVAL_DECODE_BACKEND:-decord}"

case ":$PYTHONPATH:" in
  *":$AGENTEVAL_ROOT/src:"*) ;;
  *) export PYTHONPATH="$AGENTEVAL_ROOT/src${AGENTEVAL_INTEROP_SRC:+:$AGENTEVAL_INTEROP_SRC}:$PYTHONPATH" ;;
esac

python3 - <<'PY'
import os, socket, importlib.util as u
def has(m):
    try: return u.find_spec(m) is not None
    except Exception: return False
def up(p):
    s = socket.socket(); s.settimeout(0.25)
    try: s.connect(("127.0.0.1", p)); return True
    except Exception: return False
    finally: s.close()
pkgs = ["numpy","cv2","decord","mediapipe","ultralytics","pyiqa","facexlib","timm","open_clip","scipy","krippendorff"]
miss = [m for m in pkgs if not has(m)]
ports = {f"vlm:{p}": up(p) for p in (8003, 8004, 8005)}
print("[agenteval] PYTHONPATH ok; decode=%s" % os.environ["AGENTEVAL_DECODE_BACKEND"])
print("[agenteval] missing packages: %s" % (", ".join(miss) or "none"))
print("[agenteval] VLM endpoints up: %s" % (", ".join(k for k, v in ports.items() if v) or "NONE - start one, see README"))
PY

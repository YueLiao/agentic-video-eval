"""Smoke-test a live VLM endpoint against what the harness actually needs.

Run before trusting any evaluation from a new endpoint. Each check targets a
capability the harness depends on, and each fails loudly rather than degrading,
because a judge that silently cannot follow the response schema produces
plausible-looking scores from parse failures.

    python tests/test_live_vlm.py <video.mp4>

Endpoint comes from AGENTEVAL_VLM_* (see agenteval.llm.client.from_env).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.llm.client import ImageRef, VLMClient          # noqa: E402
from agenteval.media.clip import VideoHandle, bgr_to_jpeg_bytes, uniform_indices  # noqa: E402
from agenteval.tools import renders as R                      # noqa: E402


def client() -> VLMClient:
    return VLMClient(
        model=os.environ.get("AGENTEVAL_VLM_MODEL", "gemma-4-31b-it"),
        provider=os.environ.get("AGENTEVAL_VLM_PROVIDER", "openai"),
        base_url=os.environ.get("AGENTEVAL_VLM_BASE_URL", "http://127.0.0.1:8005/v1"),
        api_key_env=os.environ.get("AGENTEVAL_VLM_KEY_ENV") or None,
        max_tokens=1024, timeout_s=180,
    )


def check(name: str, fn) -> bool:
    t0 = time.time()
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {str(e)[:110]}"
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {name:34s} {time.time()-t0:5.1f}s  {detail}")
    return ok


def main() -> int:
    video = sys.argv[1]
    v = VideoHandle(video)
    vlm = client()
    out = Path("/tmp/agenteval_live")
    print(f"endpoint: {vlm.base_url}  model: {vlm.model}")
    print(f"video: {Path(video).name}  {v.total} frames  {v.duration_s:.1f}s")

    # Warm-up. A freshly started server rejects or stalls its first requests
    # while it finishes compiling, and the retry backoff (2+4+8s) then exhausts
    # inside the first check -- which reads as "this model cannot follow a
    # schema" when nothing of the sort is true.
    for i in range(3):
        w = vlm.ask(system="ping", user="ping", tag="live/warmup")
        if w.ok or w.text:
            print(f"warm-up: ready after {i+1} attempt(s)\n")
            break
    else:
        print("warm-up: endpoint not responding\n")
    results = []

    # 1. text + JSON schema adherence
    def t_json():
        r = vlm.ask(system="只输出 JSON,不要任何多余文字。",
                    user='返回 {"ok": true, "n": 3}',
                    schema={"type": "object"}, tag="live/json")
        detail = f"parsed={r.parsed} attempts={r.attempts}"
        if r.error:
            detail += f" ERROR={r.error[:110]}"
        return (r.ok and isinstance(r.parsed, dict) and r.parsed.get("ok") is True,
                detail)
    results.append(check("text + JSON schema", t_json))

    # 2. single image comprehension
    def t_img():
        fr = v.read([v.total // 2])
        img = ImageRef(data=bgr_to_jpeg_bytes(fr[0]), caption="frame")
        r = vlm.ask(system="只输出 JSON。",
                    user='用一句话描述这张图,返回 {"desc": "..."}',
                    images=[img], schema={"type": "object"}, tag="live/img")
        d = (r.parsed or {}).get("desc", "")
        return bool(r.ok and len(str(d)) > 5), f"desc={str(d)[:60]!r}"
    results.append(check("single image", t_img))

    # 3. many images in one call -- the harness sends up to 14
    def t_many():
        idx = uniform_indices(v.total, 14)
        imgs = [ImageRef(data=bgr_to_jpeg_bytes(f), caption=f"frame {i}")
                for i, f in zip(idx, v.read(idx))]
        r = vlm.ask(system="只输出 JSON。",
                    user=f'一共给了你 {len(imgs)} 张按时间排列的帧。'
                         '返回 {"n_images": <你收到的图片数>, "changed": true/false}',
                    images=imgs, schema={"type": "object"}, tag="live/many")
        n = (r.parsed or {}).get("n_images")
        return bool(r.ok), f"n_images_reported={n} (sent {len(imgs)})"
    results.append(check("14 images in one call", t_many))

    # 4. composite render -- can it read the burned-in frame labels?
    def t_strip():
        res = R.filmstrip(v, out, t0=0, t1=v.total, n=6, cols=3, tag="live")
        img = ImageRef(path=res.images[0], caption="filmstrip")
        r = vlm.ask(system="只输出 JSON。",
                    user='这是一张按时间排列的网格图,每格左上角有帧号(形如 f12)。'
                         '返回 {"frame_labels": [看到的帧号数字...]}',
                    images=[img], schema={"type": "object"}, tag="live/strip")
        labs = (r.parsed or {}).get("frame_labels") or []
        return bool(r.ok and len(labs) >= 3), f"read {len(labs)} labels: {labs[:6]}"
    results.append(check("reads burned-in frame labels", t_strip))

    # 5. the action-selection contract the loop depends on
    def t_action():
        from agenteval.engine.actions import CONCLUDE, Action, decision_instructions
        acts = [Action("zoom_locus", "放大某个可疑点细看",
                       {"locus_id": "str: 可疑点 id, 可用 L00/L01"}), CONCLUDE]
        r = vlm.ask(system="你是视频评测员。",
                    user="可疑点: L00 (t=10-16, 高残差)。\n\n"
                         + decision_instructions(acts),
                    schema={"type": "object"}, tag="live/action")
        a = (r.parsed or {}).get("action")
        return (a in ("zoom_locus", "conclude"),
                f"action={a!r} args={(r.parsed or {}).get('args')}")
    results.append(check("action selection contract", t_action))

    # 6. the verdict schema, including refusing to invent findings
    def t_verdict():
        idx = uniform_indices(v.total, 6)
        imgs = [ImageRef(data=bgr_to_jpeg_bytes(f), caption=f"frame {i}")
                for i, f in zip(idx, v.read(idx))]
        r = vlm.ask(
            system="你是严格的视频评测员。没有发现问题是完全正常的结论,不要编造。",
            user='列出你确认的缺陷。只输出 JSON: '
                 '{"summary":"...","findings":[{"kind":"...","severity":"minor|major|critical",'
                 '"confidence":0.0-1.0,"rationale":"...","evidence":["E01"]}]}',
            images=imgs, schema={"type": "object"}, tag="live/verdict")
        p = r.parsed or {}
        fs = p.get("findings")
        return (r.ok and isinstance(fs, list),
                f"{len(fs) if isinstance(fs, list) else '?'} findings, "
                f"summary={str(p.get('summary',''))[:40]!r}")
    results.append(check("verdict schema", t_verdict))

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} passed")
    u = vlm.total
    print(f"tokens: {u.prompt_tokens} in / {u.completion_tokens} out, "
          f"{u.n_images} images, {u.latency_ms/1000:.1f}s total")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

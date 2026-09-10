"""Capability probing: find out what this endpoint can actually do.

Plug-and-play is not a provider-agnostic HTTP client. It is knowing, per model,
which techniques are worth spending on -- a model that grounds well should be
asked for boxes rather than handed CV boxes, one that reads burned-in frame
labels can use composited filmstrips, one whose self-consistency is poor needs
voting rather than a single sample. Prompts written against whatever one local
model happened to handle do not get better when a stronger model arrives.

Each probe isolates one capability and has a deterministic ground truth, so a
failure means the capability is absent rather than that the question was hard.
They are cheap by construction: single calls on synthetic images or a couple of
real frames.

**Deployment limits are reported separately from model limits.** The first run
of this project served a 256K-context model with `--max-model-len 32768` and an
image cap of 16, then treated the resulting evidence ceiling as a property of
the model. A profile that conflates the two sends you optimising the wrong
thing, so `max_images` and context are probed against the *endpoint* and
labelled as such.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from agenteval.llm.client import ImageRef, VLMClient


@dataclass
class ProbeResult:
    name: str
    passed: bool
    value: Any = None          # the measured capability, not just pass/fail
    detail: str = ""
    elapsed_s: float = 0.0
    is_deployment_limit: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "value": self.value,
                "detail": self.detail, "elapsed_s": round(self.elapsed_s, 1),
                "is_deployment_limit": self.is_deployment_limit}


# ---- synthetic images: ground truth by construction ----------------------

def _swatch(color: tuple[int, int, int], text: str = "", size: int = 320) -> bytes:
    """A solid colour tile with optional burned-in text. Used where the answer
    must be unambiguous -- a real frame invites interpretation, and a probe that
    can be argued with measures nothing."""
    import cv2
    img = np.full((size, size, 3), color, np.uint8)
    if text:
        cv2.putText(img, text, (14, size // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 220, (255, 255, 255), 3, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def _dots(n: int, size: int = 448, seed: int = 0) -> bytes:
    """n small dots on a plain ground: counting with a known answer."""
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 240, np.uint8)
    placed: list[tuple[int, int]] = []
    while len(placed) < n:
        x, y = int(rng.integers(40, size - 40)), int(rng.integers(40, size - 40))
        if all((x - a) ** 2 + (y - b) ** 2 > 80 ** 2 for a, b in placed):
            placed.append((x, y))
            cv2.circle(img, (x, y), 16, (40, 60, 200), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def _box_scene(size: int = 640, box=(0.6, 0.2, 0.18, 0.18)) -> tuple[bytes, tuple]:
    """One distinctly coloured square on a textured ground, for grounding."""
    import cv2
    rng = np.random.default_rng(7)
    img = (rng.integers(150, 200, (size, size, 3))).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 6)
    x, y, w, h = box
    cv2.rectangle(img, (int(x * size), int(y * size)),
                  (int((x + w) * size), int((y + h) * size)), (30, 30, 220), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes(), box


def _iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if (aw * ah + bw * bh - inter) > 0 else 0.0


# ---- the probes ----------------------------------------------------------

def probe_schema(vlm: VLMClient) -> ProbeResult:
    """Can it hold a nested schema? Sets the ceiling on the output contract."""
    t0 = time.time()
    schema = {"type": "object"}
    r = vlm.ask(system="只输出 JSON。",
                user='返回:{"a":{"b":[1,2,3]},"c":{"d":{"e":"x"}},"f":true}',
                schema=schema, tag="probe/schema")
    p = r.parsed or {}
    ok = (isinstance(p.get("a"), dict) and p["a"].get("b") == [1, 2, 3]
          and (p.get("c") or {}).get("d", {}).get("e") == "x" and p.get("f") is True)
    return ProbeResult("schema_fidelity", bool(ok), value=r.attempts,
                       detail=f"attempts={r.attempts} parsed={str(p)[:70]}",
                       elapsed_s=time.time() - t0)


def probe_max_images(vlm: VLMClient, ladder=(4, 8, 16, 24, 32, 48)) -> ProbeResult:
    """Largest image count the endpoint accepts *and* counts correctly.

    Two failure modes with different meanings: an HTTP error is the server's
    per-request cap (deployment), while a wrong count on an accepted request is
    the model losing track (capability). Both cap the evidence bundle, so both
    are reported, but only the first is fixable by restarting the server.
    """
    t0 = time.time()
    best, note = 0, ""
    for n in ladder:
        imgs = [ImageRef(data=_swatch((60, 60, 60), str(i + 1)), caption=f"#{i+1}")
                for i in range(n)]
        r = vlm.ask(system="只输出 JSON。",
                    user=f'这些图上各写了一个数字。返回 {{"count": <图片总数>, '
                         f'"last": <最后一张上的数字>}}',
                    images=imgs, schema={"type": "object"}, tag=f"probe/imgs{n}")
        if not r.ok:
            note = f"n={n} 被拒绝(部署上限): {(r.error or '')[:60]}"
            break
        p = r.parsed or {}
        if int(p.get("count", -1)) != n:
            note = f"n={n} 接受但计数错误(模型上限): 报告 {p.get('count')}"
            break
        best = n
    return ProbeResult("max_images", best > 0, value=best,
                       detail=note or f"至少到 {best} 张仍正确",
                       elapsed_s=time.time() - t0, is_deployment_limit=True)


def probe_counting(vlm: VLMClient, counts=(3, 5, 8)) -> ProbeResult:
    """Counting on a clean synthetic image -- a floor on fine perception."""
    t0 = time.time()
    hits = []
    for n in counts:
        r = vlm.ask(system="只输出 JSON。", user='数一下图中圆点的个数,返回 {"n": <整数>}',
                    images=[ImageRef(data=_dots(n, seed=n))],
                    schema={"type": "object"}, tag=f"probe/count{n}")
        got = (r.parsed or {}).get("n")
        hits.append(int(got) == n if isinstance(got, (int, float, str))
                    and str(got).isdigit() else False)
    acc = sum(hits) / len(hits)
    return ProbeResult("counting", acc >= 0.67, value=round(acc, 2),
                       detail=f"{sum(hits)}/{len(hits)} 正确 (n={list(counts)})",
                       elapsed_s=time.time() - t0)


def probe_grounding(vlm: VLMClient) -> ProbeResult:
    """Can it emit a usable normalized box? Decides whether open-vocabulary
    detection can go through the VLM or needs a CV detector."""
    t0 = time.time()
    img, truth = _box_scene()
    r = vlm.ask(system="只输出 JSON。",
                user='图中有一个红色方块。返回它的归一化边界框:'
                     '{"bbox": [x, y, w, h]},取值 0~1,原点在左上角。',
                images=[ImageRef(data=img)], schema={"type": "object"},
                tag="probe/ground")
    bb = (r.parsed or {}).get("bbox")
    iou = 0.0
    if isinstance(bb, list) and len(bb) >= 4:
        try:
            iou = _iou([float(v) for v in bb[:4]], truth)
        except (TypeError, ValueError):
            iou = 0.0
    return ProbeResult("grounding", iou >= 0.4, value=round(iou, 3),
                       detail=f"IoU={iou:.2f} 报告={bb} 真值={[round(v,2) for v in truth]}",
                       elapsed_s=time.time() - t0)


def probe_temporal_order(vlm: VLMClient) -> ProbeResult:
    """Can it read burned-in frame labels from a composited grid?

    This gates the whole filmstrip approach: compositing time into space only
    helps if the model can tell which cell is which moment.
    """
    import cv2
    t0 = time.time()
    tiles = [np.full((220, 220, 3), (70, 70, 70), np.uint8) for _ in range(6)]
    labels = [3, 17, 42, 58, 91, 120]
    for im, lab in zip(tiles, labels):
        cv2.putText(im, f"f{lab}", (16, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), 3, cv2.LINE_AA)
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    r = vlm.ask(system="只输出 JSON。",
                user='这是一张 3 列 2 行的网格,每格写着帧号(形如 f42),'
                     '按从左上到右下的顺序返回 {"labels": [数字, ...]}',
                images=[ImageRef(data=buf.tobytes())], schema={"type": "object"},
                tag="probe/order")
    got = (r.parsed or {}).get("labels") or []
    try:
        got = [int(x) for x in got]
    except (TypeError, ValueError):
        got = []
    return ProbeResult("temporal_order", got == labels, value=got,
                       detail=f"读出 {got} 真值 {labels}", elapsed_s=time.time() - t0)


def probe_fine_detail(vlm: VLMClient, sizes=(112, 224, 448)) -> ProbeResult:
    """At what rendered size does fine structure become readable?

    Sets where the magnification ladder has to start. Uses a dot count at
    several scales: the smallest size that still counts correctly is the
    model's usable detail floor.
    """
    import cv2
    t0 = time.time()
    floor, detail = None, []
    for s in sizes:
        big = _dots(7, size=448, seed=3)
        arr = cv2.imdecode(np.frombuffer(big, np.uint8), cv2.IMREAD_COLOR)
        small = cv2.resize(arr, (s, s), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        r = vlm.ask(system="只输出 JSON。", user='数一下圆点个数,返回 {"n": <整数>}',
                    images=[ImageRef(data=buf.tobytes())],
                    schema={"type": "object"}, tag=f"probe/detail{s}")
        got = (r.parsed or {}).get("n")
        good = str(got).isdigit() and int(got) == 7
        detail.append(f"{s}px={'ok' if good else got}")
        if good and floor is None:
            floor = s
    return ProbeResult("fine_detail_floor", floor is not None, value=floor,
                       detail=" ".join(detail) + f" → 最小可判读 {floor}px",
                       elapsed_s=time.time() - t0)


def probe_self_consistency(vlm: VLMClient, k: int = 5) -> ProbeResult:
    """How stable is a borderline judgement across samples?

    Run at temperature > 0 on purpose: at 0 the answer is deterministic and this
    would measure nothing. A model that disagrees with itself needs voting, and
    the disagreement rate says how many votes.
    """
    t0 = time.time()
    img, _ = _box_scene()
    answers = []
    saved = vlm.temperature
    vlm.temperature = 0.8
    try:
        for i in range(k):
            r = vlm.ask(system="只输出 JSON。",
                        user='这个红色方块占画面面积的比例最接近哪个?'
                             '返回 {"pick": "1%"|"3%"|"10%"|"30%"}',
                        images=[ImageRef(data=img)], schema={"type": "object"},
                        tag=f"probe/consist{i}")
            answers.append(str((r.parsed or {}).get("pick")))
    finally:
        vlm.temperature = saved
    top = max(set(answers), key=answers.count) if answers else ""
    rate = answers.count(top) / max(1, len(answers))
    return ProbeResult("self_consistency", rate >= 0.6, value=round(rate, 2),
                       detail=f"{k} 次采样最高一致率 {rate:.0%} ({answers})",
                       elapsed_s=time.time() - t0)


def probe_instruction_depth(vlm: VLMClient) -> ProbeResult:
    """Is a constraint buried mid-prompt obeyed?

    Measured because this project lost two constraints to it: a compiler rule
    that dropped camera requirements twice, and a confidence contract that
    returned 0.5 on all fourteen findings until it was moved next to the output
    block. If a model fails this, constraints belong in code, not in prose.
    """
    t0 = time.time()
    filler = "评测视频时应关注运动连贯性与结构完整性。" * 24
    r = vlm.ask(
        system="你是视频评测员。\n" + filler
               + "\n规则 7:回答中必须把 marker 字段设为 'ZK7'。\n" + filler,
        user='返回 {"marker": "...", "ok": true}',
        schema={"type": "object"}, tag="probe/instr")
    got = str((r.parsed or {}).get("marker", ""))
    return ProbeResult("instruction_depth", got == "ZK7", value=got,
                       detail=f"埋在 ~1300 字中段的约束 → 返回 {got!r}",
                       elapsed_s=time.time() - t0)


def probe_refusal_to_invent(vlm: VLMClient) -> ProbeResult:
    """Asked to find defects in a clean image, does it invent them?

    The most consequential failure mode for a judge: a model that always finds
    something makes every video look flawed and the scores meaningless.
    """
    t0 = time.time()
    img = _swatch((120, 140, 160), "")
    r = vlm.ask(system="你是严格的视频评测员。没有发现问题是完全正常的结论,不要编造。",
                user='这是一张纯色测试图。列出你确认的画面缺陷。'
                     '只输出 JSON: {"findings": [...]}',
                images=[ImageRef(data=img)], schema={"type": "object"},
                tag="probe/invent")
    fs = (r.parsed or {}).get("findings")
    n = len(fs) if isinstance(fs, list) else -1
    return ProbeResult("refusal_to_invent", n == 0, value=n,
                       detail=f"在纯色图上报了 {n} 条缺陷 (期望 0)",
                       elapsed_s=time.time() - t0)


def probe_image_budget(vlm: VLMClient) -> ProbeResult:
    """How the deployment spends pixels: fixed token budget, or native size?

    This decides evidence *layout*, and getting it wrong silently wastes every
    round of work built on top. Two deployments measured here differ completely:

      gemma-4-31b   max_soft_tokens 280, pooling 3, patch 16 -- a hard cap. The
                    grid is fitted to the aspect ratio within 280 cells, so a
                    wider image means fewer pixels per region. Adding frames to
                    a strip shrinks every frame.
      qwen3.8-27b   min 65536 / max 16777216 pixels, patch 16, merge 2 -- images
                    are kept at native size up to 16MP. Adding frames costs
                    tokens and latency, not resolution: the same 1542x768
                    evidence ran about 13x slower per call.

    So "give the model more evidence" is a different trade on each, and a
    harness that means to be model-agnostic has to measure this rather than
    assume it. The probe reads it off the deployment's own processor config
    where that is reachable, and otherwise infers a floor empirically by asking
    the same question of one image at growing widths.
    """
    t0 = time.time()
    detail: list[str] = []
    kind, budget = "unknown", None
    try:
        import requests
        r = requests.get(vlm.base_url.rstrip("/") + "/models", timeout=10)
        root = (r.json().get("data") or [{}])[0].get("root")
    except Exception:  # noqa: BLE001
        root = None
    if root:
        for name in ("processor_config.json", "preprocessor_config.json"):
            p = Path(root) / name
            if not p.exists():
                continue
            try:
                cfg = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            ip = cfg.get("image_processor", cfg)
            if ip.get("max_soft_tokens"):
                kind, budget = "token_capped", int(ip["max_soft_tokens"])
                px = int(ip.get("patch_size", 16)) * int(ip.get("pooling_kernel_size", 1))
                detail.append(f"{name}: 硬上限 {budget} token, 每单元 {px}px")
                break
            size = ip.get("size") or {}
            if size.get("longest_edge"):
                kind, budget = "native_capped", int(size["longest_edge"])
                detail.append(f"{name}: 最多 {budget} 像素, 原生保留")
                break
    return ProbeResult("image_budget", kind != "unknown", value=kind,
                       detail=(" · ".join(detail) or "无法从部署读到图像处理配置, "
                               "证据版式应按最保守的固定预算假设") +
                              " [部署]",
                       elapsed_s=time.time() - t0)


PROBES: tuple[Callable[[VLMClient], ProbeResult], ...] = (
    probe_schema, probe_max_images, probe_counting, probe_grounding,
    probe_temporal_order, probe_fine_detail, probe_self_consistency,
    probe_instruction_depth, probe_refusal_to_invent, probe_image_budget,
)

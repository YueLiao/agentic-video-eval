"""View synthesis — making artifacts perceptible to the VLM.

The premise of this module is that the VLM is the strongest artifact recognizer
available, and that the CV layer's job is not to out-judge it but to *put the
evidence where it can see it*. Almost every case of a VLM "missing" a generation
artifact is a presentation failure rather than a capability failure:

* the artifact was never sampled;
* it was sampled but downsampled below legibility;
* it was legible but spread across separate images, and the model had to hold a
  visual memory across them to notice the inconsistency.

The third is the one people underrate. A VLM compares far better *within* one
image than *across* several: temporal inconsistency becomes easy the moment it
is rendered as a spatial pattern. So these renders composite into a single
image wherever the judgement is comparative.

Each render also has an inverse property worth stating: it can manufacture the
appearance of a defect. Amplified high-frequency views make every video look
noisy; residual heatmaps light up on legitimate motion. So each render carries a
hint saying what it exaggerates, and the falsification pass exists to catch
claims that are artifacts of the view rather than of the video.
"""

from __future__ import annotations

import hashlib

from pathlib import Path
from typing import Sequence

import numpy as np

from agenteval.media.clip import VideoHandle, bgr_to_jpeg_bytes, uniform_indices
from agenteval.tools.base import ToolResult

_FONT = None


def _label(img: np.ndarray, text: str, org=(6, 22), scale=0.6) -> np.ndarray:
    import cv2
    out = img.copy()
    cv2.rectangle(out, (org[0] - 4, org[1] - 18),
                  (org[0] + int(len(text) * 11 * scale) + 6, org[1] + 8),
                  (0, 0, 0), -1)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _write(out_dir: Path, stem: str, img: np.ndarray, q: int = 94) -> Path:
    """Content-addressed: the filename carries a hash of the encoded image.

    The stems the callers build describe the *view* -- "strip_0_80",
    "motion_curves" -- and not the clip, so two different videos of the same
    length rendered into the same directory landed on the same path. Measured:
    a 200-clip run wrote exactly two files, and with six workers each request
    was sent whichever clip had written last. Every clip was scored, nothing
    errored, and the numbers were meaningless.

    Hashing the bytes makes a collision mean the images are identical, which is
    a cache hit rather than a bug.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = bgr_to_jpeg_bytes(img, q)
    p = out_dir / f"{stem}_{hashlib.sha1(data).hexdigest()[:10]}.jpg"
    if not p.exists():
        p.write_bytes(data)
    return p


def _crop(frame: np.ndarray, bbox, pad: float, side: int) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    s = max(bw * w, bh * h) * (1 + pad)
    x0 = int(max(0, min(w - 1, cx * w - s / 2)))
    y0 = int(max(0, min(h - 1, cy * h - s / 2)))
    reg = frame[y0:int(min(h, y0 + s)), x0:int(min(w, x0 + s))]
    if reg.size == 0:
        reg = frame
    return cv2.resize(reg, (side, side), interpolation=cv2.INTER_CUBIC)


def _tile(imgs: Sequence[np.ndarray], cols: int, gap: int = 6) -> np.ndarray:
    """Grid of images, padding to a common cell so tiles of different sizes
    (a full frame beside a square crop) compose without distorting either."""
    if not imgs:
        return np.zeros((8, 8, 3), np.uint8)
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[0] == h and im.shape[1] == w:
            padded.append(im)
            continue
        cell = np.full((h, w, 3), 32, np.uint8)
        cell[: im.shape[0], : im.shape[1]] = im
        padded.append(cell)
    imgs = padded
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.full((rows * h + (rows - 1) * gap,
                      cols * w + (cols - 1) * gap, 3), 32, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * (h + gap):r * (h + gap) + h,
               c * (w + gap):c * (w + gap) + w] = im
    return canvas


# ---- the renders --------------------------------------------------------

def filmstrip(video: VideoHandle, out_dir: Path, *, t0: int, t1: int,
              bbox=None, n: int = 8, side: int = 320, cols: int = 4,
              tag: str = "strip") -> ToolResult:
    """Consecutive moments as one labelled grid.

    This is the highest-leverage render. Judging whether something is temporally
    inconsistent means comparing moment to moment, and a model that receives the
    moments as separate images has to carry a visual memory between them. Laid
    out in one image, the comparison becomes ordinary spatial reasoning, which
    is what these models are actually good at.
    """
    t0 = max(0, int(t0)); t1 = min(video.total, max(int(t1), t0 + 2))
    idx = sorted({min(video.total - 1, t0 + i)
                  for i in uniform_indices(t1 - t0, min(n, t1 - t0))})
    tiles = []
    for i, fr in zip(idx, video.read(idx)):
        im = _crop(fr, bbox, 0.35, side) if bbox is not None else _resize_side(fr, side)
        tiles.append(_label(im, f"f{i}"))
    img = _tile(tiles, cols)
    p = _write(out_dir / tag, f"{tag}_{t0}_{t1}", img)
    return ToolResult(
        value={"indices": idx, "t_span": [t0, t1], "layout": f"{cols}col",
               "bbox": [round(v, 4) for v in bbox] if bbox is not None else None},
        images=[p], reliability=1.0, backend="filmstrip",
        hint=(f"第 {t0}-{t1} 帧按时间顺序排成一张图(左上→右下),每格左上角是帧号。"
              "请**逐格对比**:同一个物体的形状、纹理、边缘在相邻格之间应当连续变化。"
              "突然的重排、闪烁、跳变在这种并排布局下比逐张看容易得多。"),
    )


def _resize_side(fr: np.ndarray, side: int) -> np.ndarray:
    import cv2
    h, w = fr.shape[:2]
    s = side / max(h, w)
    return cv2.resize(fr, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)


def ab_contrast(video: VideoHandle, out_dir: Path, *, bbox, t_span,
                side: int = 420, tag: str = "ab") -> ToolResult:
    """Suspect region and a quiet-time reference of the same region, side by
    side in one labelled image.

    Composited rather than sent as two images on purpose: the question is
    "does A differ from B", and asking that within a single frame removes the
    cross-image memory step where the comparison usually fails.
    """
    t0, t1 = int(t_span[0]), int(t_span[1])
    mid = min(video.total - 1, (t0 + t1) // 2)
    span = max(1, t1 - t0)
    ref = mid + 3 * span
    if ref >= video.total:
        ref = max(0, mid - 3 * span)
    frames = video.read([mid, ref])
    if len(frames) < 2:
        return ToolResult(value={"error": "no contrast available"}, reliability=0.0)
    a = _label(_crop(frames[0], bbox, 0.35, side), f"A  f{mid}  (suspect)")
    b = _label(_crop(frames[1], bbox, 0.35, side), f"B  f{ref}  (reference)")
    p = _write(out_dir / tag, f"{tag}_{mid}_{ref}", _tile([a, b], 2))
    return ToolResult(
        value={"suspect_frame": mid, "reference_frame": ref,
               "bbox": [round(v, 4) for v in bbox]},
        images=[p], reliability=1.0, backend="ab_contrast",
        hint=("同一区域的两个时刻:A=可疑时刻,B=参照时刻。"
              "**若 A 的现象在 B 上同样存在,那它是这段视频渲染该区域的常态,不是缺陷。**"
              "只有 A 明显异于 B 时才构成一个'事件'。"),
    )


def residual_heatmap(video: VideoHandle, out_dir: Path, *, t: int,
                     side: int = 512, tag: str = "resid") -> ToolResult:
    """Frame, and the motion-compensated residual over it, side by side.

    Shows the model *what motion cannot explain*: frame t warped into t+1 by the
    optical flow, differenced against the real t+1. Legitimate movement largely
    cancels; what glows is change the scene's own motion does not account for.
    """
    import cv2
    t = max(0, min(video.total - 2, int(t)))
    fr = video.read([t, t + 1])
    if len(fr) < 2:
        return ToolResult(value={"error": "need two frames"}, reliability=0.0)
    a, b = fr
    ga = cv2.cvtColor(_resize_side(a, side), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(_resize_side(b, side), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    yy, xx = np.mgrid[0:ga.shape[0], 0:ga.shape[1]].astype(np.float32)
    warped = cv2.remap(ga, xx + flow[..., 0], yy + flow[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.abs(warped.astype(np.float32) - gb.astype(np.float32))
    p99 = float(np.percentile(err, 99)) or 1.0
    norm = np.clip(err / p99, 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    base = _resize_side(b, side)
    blend = cv2.addWeighted(base, 0.45, heat, 0.55, 0)
    img = _tile([_label(base, f"f{t+1} original"),
                 _label(blend, "motion-compensated residual")], 2)
    p = _write(out_dir / tag, f"{tag}_t{t:04d}", img)
    hot = float((norm > 0.5).mean())
    return ToolResult(
        value={"t": t, "hot_area_frac": round(hot, 4),
               "residual_p99": round(p99, 2)},
        images=[p], reliability=0.8, backend="residual_heatmap",
        hint=("右图亮处 = 用光流补偿后仍解释不掉的变化。"
              "**注意它会在遮挡边界和高速运动的物体轮廓上正常发亮**,那不是缺陷;"
              "真正值得报的是**位于平坦区域内部、与运动无关的**大片亮斑。"
              "请对照左图确认亮处到底对应画面里的什么东西。"),
    )


def highfreq_amplify(video: VideoHandle, out_dir: Path, *, t: int, bbox=None,
                     side: int = 448, gain: float = 6.0,
                     tag: str = "hifreq") -> ToolResult:
    """Original next to a high-pass-amplified view of the same crop.

    Generation leaves its signature in the high frequencies: over-smoothed skin
    that lost its pore texture, checkerboard ringing, texture that is locally
    self-similar in a way real detail never is. Amplifying that band makes it
    legible instead of leaving it a few LSBs below perception.
    """
    import cv2
    t = max(0, min(video.total - 1, int(t)))
    fr = video.read([t])
    if not fr:
        return ToolResult(value={"error": "no frame"}, reliability=0.0)
    base = _crop(fr[0], bbox, 0.3, side) if bbox is not None else _resize_side(fr[0], side)
    blur = cv2.GaussianBlur(base, (0, 0), 2.0)
    hi = cv2.addWeighted(base.astype(np.float32), 1.0, blur.astype(np.float32), -1.0, 128)
    hi = np.clip((hi - 128) * gain + 128, 0, 255).astype(np.uint8)
    img = _tile([_label(base, f"f{t} original"),
                 _label(hi, f"high-pass x{gain:g}")], 2)
    p = _write(out_dir / tag, f"{tag}_t{t:04d}", img)
    return ToolResult(
        value={"t": t, "gain": gain,
               "bbox": [round(v, 4) for v in bbox] if bbox is not None else None},
        images=[p], reliability=0.6, backend="highfreq_amplify",
        hint=("右图是高频放大视图,用来看清原图里几乎不可见的纹理层。"
              "**这个视图会把任何视频都放大得很脏,所以不能只凭它下结论。**"
              "有意义的观察是:该有纹理的地方(皮肤毛孔、织物、树叶)是不是被抹平成了"
              "塑料感的空白;或者出现了规则的网格/重复花纹。"),
    )


def skeleton_overlay(video: VideoHandle, out_dir: Path, *, pose_result,
                     hands_result=None, indices=None, side: int = 360,
                     cols: int = 4, tag: str = "skel") -> ToolResult:
    """Draw the detector's keypoints onto the frames, as one grid.

    Rendering detections instead of reporting numbers lets the model *check the
    detector* rather than trust it. When a bone-length statistic looks alarming,
    an overlay immediately shows whether the limb really deformed or the
    keypoints simply landed in the wrong place — which the number alone can
    never distinguish.
    """
    import cv2
    per = {p["idx"]: p for p in (pose_result.value.get("per_frame") or [])}
    hper = {p["idx"]: p for p in ((hands_result.value.get("per_frame") or [])
                                  if hands_result else [])}
    idx = list(indices) if indices is not None else sorted(per)[:cols * 2]
    if not idx:
        return ToolResult(value={"error": "no pose frames"}, reliability=0.0)
    bones = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
             (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
    tiles = []
    for i, fr in zip(idx, video.read(idx)):
        im = _resize_side(fr, side)
        h, w = im.shape[:2]
        kp = (per.get(i) or {}).get("kpts")
        if kp:
            for a, b in bones:
                ka, kb = kp[a], kp[b]
                if ka[2] > 0.3 and kb[2] > 0.3:
                    cv2.line(im, (int(ka[0] * w), int(ka[1] * h)),
                             (int(kb[0] * w), int(kb[1] * h)), (0, 255, 0), 2)
            for k in kp:
                if k[2] > 0.3:
                    cv2.circle(im, (int(k[0] * w), int(k[1] * h)), 3, (0, 200, 255), -1)
        for hd in (hper.get(i) or {}).get("hands", []):
            for px, py in hd["kpts"]:
                cv2.circle(im, (int(px * w), int(py * h)), 2, (255, 80, 255), -1)
        tiles.append(_label(im, f"f{i}"))
    p = _write(out_dir / tag, f"{tag}_{idx[0]}_{idx[-1]}", _tile(tiles, cols))
    return ToolResult(
        value={"indices": idx, "n": len(idx)}, images=[p],
        reliability=0.7, backend="skeleton_overlay",
        hint=("绿线是检测器估计的骨架,橙点是关节,粉点是手部关键点。"
              "**先判断骨架贴合得对不对**:若关键点本身就落错了位置,"
              "那么任何基于骨长的统计量都不成立,应当以画面为准。"
              "若骨架贴合良好而肢体长度仍在明显变化,那才是真的结构问题。"),
    )


def zoom_grid(video: VideoHandle, out_dir: Path, *, bbox, t_span, n: int = 6,
              side: int = 300, cols: int = 3, tag: str = "zoomgrid") -> ToolResult:
    """A locus, cropped at native resolution and laid out across time."""
    r = filmstrip(video, out_dir, t0=t_span[0], t1=t_span[1], bbox=bbox,
                  n=n, side=side, cols=cols, tag=tag)
    r.hint = ("可疑区域的原生分辨率裁剪,按时间排列(左上→右下),格内左上角为帧号。"
              "这些细节在整帧缩略图中是不可见的。请逐格对比该区域的结构是否连续。")
    r.backend = "zoom_grid"
    return r


# ---- motion-specific renders -------------------------------------------
# Compositing into one image helps *comparative* judgements (is A different from
# B). It actively hurts *rate* judgements: laying frames out in a grid discards
# the timing that motion quality is about. Smoothness, speed, rhythm and
# naturalness need the sequence preserved -- as ordered separate images, as the
# clip itself where the API takes video, or as an explicit plot of the motion
# over time. So motion gets its own encodings rather than reusing the filmstrip.

def _plot(series: dict[str, Sequence[float]], *, w: int = 720, h: int = 260,
          title: str = "", xlabel: str = "frame") -> np.ndarray:
    """Minimal line plot drawn with cv2 (no matplotlib backend to go wrong)."""
    import cv2
    img = np.full((h, w, 3), 250, np.uint8)
    pad_l, pad_b, pad_t, pad_r = 54, 30, 26, 12
    x0, y0, x1, y1 = pad_l, h - pad_b, w - pad_r, pad_t
    cv2.rectangle(img, (x0, y1), (x1, y0), (215, 215, 215), 1)
    allv = [v for s in series.values() for v in s if np.isfinite(v)]
    if not allv:
        return img
    lo, hi = float(min(allv)), float(max(allv))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    colors = [(200, 80, 40), (40, 140, 220), (60, 170, 60), (150, 60, 190)]
    n_max = max(len(s) for s in series.values())
    for ci, (name, s) in enumerate(series.items()):
        col = colors[ci % len(colors)]
        pts = []
        for i, v in enumerate(s):
            if not np.isfinite(v):
                continue
            px = int(x0 + (x1 - x0) * (i / max(1, n_max - 1)))
            py = int(y0 - (y0 - y1) * ((v - lo) / (hi - lo)))
            pts.append((px, py))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, 2, cv2.LINE_AA)
        cv2.putText(img, name, (x0 + 8 + ci * 150, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    for frac in (0.0, 0.5, 1.0):
        v = lo + (hi - lo) * (1 - frac)
        py = int(y1 + (y0 - y1) * frac)
        cv2.putText(img, f"{v:.3g}", (4, py + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (110, 110, 110), 1, cv2.LINE_AA)
    for frac in (0.0, 0.5, 1.0):
        px = int(x0 + (x1 - x0) * frac)
        cv2.putText(img, str(int((n_max - 1) * frac)), (px - 8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (pad_l, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, xlabel, (w // 2 - 20, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (110, 110, 110), 1, cv2.LINE_AA)
    return img


def motion_curves(video: VideoHandle, out_dir: Path, *, stride: int = 1,
                  max_frames: int | None = None, tag: str = "motion") -> ToolResult:
    """Global motion magnitude, acceleration and jerk over time, as a plot.

    Rate is invisible in any grid of frames: a stutter and a slow passage look
    identical laid out spatially. Plotted against the frame axis, a freeze is a
    trough, a jump is a spike, and judder is high-frequency ripple -- shapes a
    VLM reads reliably. The plot is a *summary*, though, so it says where to
    look; the verdict still comes from the frames.
    """
    import cv2
    n = min(video.total, max_frames or video.total)
    idx = list(range(0, n, max(1, stride)))
    gray = video.read_gray(idx, max_side=256)
    if len(gray) < 3:
        return ToolResult(value={"error": "too few frames"}, reliability=0.0)
    mag = []
    for a, b in zip(gray, gray[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag.append(float(np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2).mean()))
    mag_a = np.asarray(mag, np.float64)
    accel = np.diff(mag_a, prepend=mag_a[:1])
    jerk = np.diff(accel, prepend=accel[:1])
    img = _plot({"speed": mag_a.tolist(), "accel": accel.tolist()},
                title="global motion magnitude / acceleration")
    p = _write(out_dir / tag, f"{tag}_curves", img)

    med = float(np.median(mag_a)) or 1e-6
    freeze = [int(idx[i]) for i, v in enumerate(mag_a) if v < 0.15 * med]
    spikes = [int(idx[i]) for i, v in enumerate(np.abs(accel))
              if v > 4 * (float(np.median(np.abs(accel))) or 1e-6)]
    return ToolResult(
        value={"n_pairs": len(mag), "speed_median": round(med, 4),
               "speed_max": round(float(mag_a.max()), 4),
               "jerk_rms": round(float(np.sqrt((jerk ** 2).mean())), 4),
               "near_freeze_frames": freeze[:20],
               "accel_spike_frames": spikes[:20]},
        images=[p], reliability=0.75, backend="motion_curves",
        hint=("横轴帧号,speed=全局光流幅度,accel=其一阶差分。"
              "读法:平台段=匀速;深谷=接近静止(卡顿或本就静止);"
              "尖峰=速度突变(跳变或剪切);高频锯齿=抖动/不平滑。"
              "**注意:曲线只说明运动的时间结构,不能判断运动是否自然合理**——"
              "匀速平滑的曲线也可能对应完全不合物理的运动。"),
    )


def trajectory_plot(track: Sequence[Sequence[float]], out_dir: Path, *,
                    fps: float = 24.0, tag: str = "traj") -> ToolResult:
    """Position, speed and vertical-velocity curves for one tracked point.

    Made for physical plausibility: free fall is a straight line in vertical
    velocity, and a constant-height 'fall' or a mid-air stop shows up as a flat
    segment that no real trajectory produces.
    """
    if len(track) < 4:
        return ToolResult(value={"error": "track too short"}, reliability=0.0)
    arr = np.asarray(track, np.float64)
    x, y = arr[:, 0], arr[:, 1]
    vy = np.diff(y, prepend=y[:1]) * fps
    vx = np.diff(x, prepend=x[:1]) * fps
    speed = np.hypot(vx, vy)
    img = _tile([
        _plot({"x": x.tolist(), "y": y.tolist()}, title="position (normalized)"),
        _plot({"v_y": vy.tolist(), "speed": speed.tolist()},
              title="velocity (per second)"),
    ], 1)
    p = _write(out_dir / tag, f"{tag}_curves", img)
    # straight-line fit on v_y is the free-fall test
    t = np.arange(len(vy), dtype=np.float64)
    try:
        coef = np.polyfit(t, vy, 1)
        resid = float(np.sqrt(((vy - np.polyval(coef, t)) ** 2).mean()))
        denom = float(np.std(vy)) or 1e-6
    except np.linalg.LinAlgError:
        coef, resid, denom = [0.0, 0.0], 0.0, 1.0
    return ToolResult(
        value={"n": len(track), "vy_slope_per_s2": round(float(coef[0]) * fps, 4),
               "vy_linear_residual": round(resid, 4),
               "vy_residual_ratio": round(resid / denom, 4),
               "speed_max": round(float(speed.max()), 4)},
        images=[p], reliability=0.6, backend="trajectory_plot",
        hint=("被跟踪点的位置与速度曲线。自由落体时 v_y 应当是一条直线"
              "(斜率恒定=重力加速度),vy_residual_ratio 接近 0。"
              "中途出现的水平段=物体在空中停住;斜率反号=凭空上升。"
              "**前提是跟踪本身是对的**,请先用画面确认跟的是同一个物体。"),
    )


def ordered_frames(video: VideoHandle, out_dir: Path, *, t0: int, t1: int,
                   n: int = 10, side: int = 384,
                   tag: str = "seq") -> ToolResult:
    """Consecutive frames as *separate, ordered* images.

    The deliberate opposite of :func:`filmstrip`. For motion the sequence itself
    is the evidence, and a grid flattens away the ordering that continuity and
    rhythm are judged from. Captions carry the frame number and timestamp so the
    model can reason about intervals rather than mere order.
    """
    t0 = max(0, int(t0)); t1 = min(video.total, max(int(t1), t0 + 2))
    idx = sorted({min(video.total - 1, t0 + i)
                  for i in uniform_indices(t1 - t0, min(n, t1 - t0))})
    fps = video.fps or 24.0
    paths = []
    for i, fr in zip(idx, video.read(idx)):
        paths.append(_write(out_dir / tag, f"{tag}_t{i:04d}",
                            _label(_resize_side(fr, side), f"f{i}  t={i/fps:.2f}s")))
    return ToolResult(
        value={"indices": idx, "t_span": [t0, t1], "fps": round(fps, 2),
               "dt_s": round((idx[1] - idx[0]) / fps, 4) if len(idx) > 1 else 0.0},
        images=paths, reliability=1.0, backend="ordered_frames",
        hint=(f"第 {t0}-{t1} 帧按时间**顺序**给出的独立帧,每帧标注了帧号与时间戳,"
              f"相邻两帧间隔约 {((idx[1]-idx[0])/fps if len(idx)>1 else 0):.3f} 秒。"
              "请按顺序观察运动是否连贯、速度是否合理、动作是否符合真实物理与生物力学。"),
    )


def motion_trail(video: VideoHandle, out_dir: Path, *, t0: int, t1: int,
                 n: int = 7, side: int = 640, tag: str = "trail") -> ToolResult:
    """All sampled moments blended into one image, so the path is visible.

    A single picture of where things travelled. Smooth motion leaves an evenly
    spaced trail; a teleport leaves a gap; a stall leaves a dense clump.
    """
    import cv2
    t0 = max(0, int(t0)); t1 = min(video.total, max(int(t1), t0 + 2))
    idx = sorted({min(video.total - 1, t0 + i)
                  for i in uniform_indices(t1 - t0, min(n, t1 - t0))})
    frames = [_resize_side(f, side).astype(np.float32) for f in video.read(idx)]
    if len(frames) < 2:
        return ToolResult(value={"error": "too few frames"}, reliability=0.0)
    base = frames[-1].copy()
    acc = np.zeros_like(base)
    for w, f in zip(np.linspace(0.3, 1.0, len(frames)), frames):
        acc = np.maximum(acc, f * w)
    blend = np.clip(0.4 * base + 0.6 * acc, 0, 255).astype(np.uint8)
    img = _tile([_label(frames[-1].astype(np.uint8), f"f{idx[-1]} (last)"),
                 _label(blend, f"trail f{idx[0]}-{idx[-1]}")], 2)
    p = _write(out_dir / tag, f"{tag}_{idx[0]}_{idx[-1]}", img)
    return ToolResult(
        value={"indices": idx, "t_span": [t0, t1]}, images=[p],
        reliability=0.55, backend="motion_trail",
        hint=("右图把多个时刻叠加在一起,用来一眼看出运动轨迹的形状。"
              "均匀铺开=平滑运动;出现断档=瞬移;堆在一处=停滞。"
              "**叠加图会让复杂场景显得很乱**,只在主体明确时可用,"
              "且必须回到顺序帧确认。"),
    )


# ---- paired views: the evidence a graded judgement actually needs ----------
# A grade boundary like "would you notice this at normal viewing speed?" cannot
# be answered from a magnified crop, and a judge given only crops has no way to
# say anything but "yes, obvious" -- which is what put 85% of findings in the
# same grade. Salience has the same problem: whether a defect sits on the
# subject or in the background is a fact about the whole frame, invisible once
# you have cropped to the defect.
#
# So the views that support a graded verdict come in pairs: the full frame at
# viewing scale, and the same moment magnified, in one image so the model can
# see both without holding one in memory.

def paired_view(video: VideoHandle, out_dir: Path, *, bbox, t: int,
                full_side: int = 560, zoom_side: int = 420,
                tag: str = "paired") -> ToolResult:
    """Full frame with the region marked, beside the magnified crop.

    Answers three things one crop cannot: is it visible at normal scale
    (severity), where does it sit in the composition (salience), and what does
    it actually look like (existence).
    """
    import cv2
    t = max(0, min(video.total - 1, int(t)))
    fr = video.read([t])
    if not fr:
        return ToolResult(value={"error": "no frame"}, reliability=0.0)
    full = _resize_side(fr[0], full_side)
    h, w = full.shape[:2]
    x, y, bw, bh = bbox
    p0 = (int(x * w), int(y * h))
    p1 = (int(min(1.0, x + bw) * w), int(min(1.0, y + bh) * h))
    cv2.rectangle(full, p0, p1, (0, 230, 255), 2)
    crop = _crop(fr[0], bbox, 0.3, zoom_side)
    img = _tile([_label(full, f"f{t} 全图(原始观看尺寸)"),
                 _label(crop, f"f{t} 放大 {zoom_side}px")], 2)
    p = _write(out_dir / tag, f"{tag}_t{t:04d}", img)
    return ToolResult(
        value={"t": t, "bbox": [round(v, 4) for v in bbox]},
        images=[p], reliability=1.0, backend="paired_view",
        hint=("左=整帧,按接近正常观看的尺寸显示,黄框标出可疑区域;右=同一帧该区域的放大。\n"
              "**判断严重度时看左边**:这个问题在左图(正常尺寸)里看得出来吗?"
              "看不出来就不是 major。\n"
              "**判断显著位置时也看左边**:黄框落在画面主体上,还是边缘/背景?\n"
              "右图只用来确认问题**是否真的存在**、具体是什么形态。"),
    )


def scale_ladder(video: VideoHandle, out_dir: Path, *, bbox, t: int,
                 tag: str = "ladder") -> ToolResult:
    """The same region at three magnifications, in one image.

    Where a defect first becomes visible as you zoom in *is* its severity, so
    showing the ladder makes the grade boundary something the model can read off
    rather than estimate.
    """
    import cv2
    t = max(0, min(video.total - 1, int(t)))
    fr = video.read([t])
    if not fr:
        return ToolResult(value={"error": "no frame"}, reliability=0.0)
    tiles = [_label(_resize_side(fr[0], 420), "1x 全图")]
    for mult, name in ((2.0, "2x"), (4.0, "4x")):
        x, y, bw, bh = bbox
        cx, cy = x + bw / 2, y + bh / 2
        side = max(bw, bh) * (4.0 / mult)
        b = (max(0.0, cx - side / 2), max(0.0, cy - side / 2),
             min(1.0, side), min(1.0, side))
        tiles.append(_label(_crop(fr[0], b, 0.0, 420), f"{name} 放大"))
    p = _write(out_dir / tag, f"{tag}_t{t:04d}", _tile(tiles, 3))
    return ToolResult(
        value={"t": t, "levels": ["1x", "2x", "4x"]},
        images=[p], reliability=1.0, backend="scale_ladder",
        hint=("同一处在三个放大级别下的样子。**问题在哪一级才变得明显,就对应哪一档**:\n"
              "- 1x(全图)就能看出来 → major 或更重\n"
              "- 2x 才看得出 → minor\n"
              "- 4x 才看得出 → trace\n"
              "这是严重度分档最直接的依据,不要凭印象估。"),
    )


def temporal_extent(video: VideoHandle, out_dir: Path, *, bbox, n: int = 8,
                    side: int = 260, tag: str = "extent") -> ToolResult:
    """The same region sampled across the *whole* clip, for judging extent.

    Extent asks how much of the clip a defect occupies, which cannot be read
    from evidence drawn from one window -- yet a window is what the search
    naturally produces, since it went there because that is where the defect
    was. Without a whole-clip view the judge has no basis to distinguish a flash
    from something running throughout.
    """
    idx = uniform_indices(video.total, n)
    tiles = [_label(_crop(f, bbox, 0.25, side), f"f{i}")
             for i, f in zip(idx, video.read(idx))]
    p = _write(out_dir / tag, f"{tag}_all", _tile(tiles, min(n, 4)))
    return ToolResult(
        value={"indices": idx, "spans_whole_clip": True},
        images=[p], reliability=1.0, backend="temporal_extent",
        hint=(f"同一区域在**整段视频**上的均匀采样({n} 个时刻,覆盖全片)。\n"
              "**判断 extent 用这张图**:数一下有问题的格子占几格——\n"
              "1 格=flash,2-3 格且连续=brief,间隔出现=recurring,几乎每格都有=throughout。"),
    )

def diff_strip(video: VideoHandle, out_dir: Path, *, n: int = 16,
               side: int = 200, cols: int = 8, tag: str = "diff") -> ToolResult:
    """Consecutive-frame absolute difference, as a strip.

    A freeze is the one defect a grid of sampled frames cannot show: the stall
    happens between two samples and the samples themselves look fine. Measured:
    asked to check ten motion defects on a 16-frame grid, the model reported
    appearance defects on 4-21% of clips and freeze, slip, speed and amplitude
    on exactly 0% -- not reluctance, an absence of evidence. Differencing
    *adjacent* frames puts rate back in the picture: a stalled moment is a cell
    that is almost black, and a jump is a cell that is almost white.
    """
    import cv2
    idx = uniform_indices(video.total, min(n + 1, video.total))
    frames = video.read_gray(idx, max_side=side * 2)
    if len(frames) < 2:
        return ToolResult(value={"error": "too few frames"}, reliability=0.0)
    tiles, energy = [], []
    for i, (a, b) in enumerate(zip(frames, frames[1:])):
        d = cv2.absdiff(a, b)
        energy.append(float(d.mean()))
        # Fixed gain, not per-cell normalisation: the cells have to be
        # comparable to each other or "almost black" carries no meaning.
        vis = np.clip(d.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        tiles.append(_label(_resize_side(vis, side),
                            f"f{idx[i]}->{idx[i + 1]}"))
    img = _tile(tiles, cols)
    p = _write(out_dir / tag, f"{tag}_{video.path.stem}"[:110], img)
    e = np.asarray(energy)
    med = float(np.median(e)) or 1e-6
    return ToolResult(
        value={"cell_energy": [round(v, 2) for v in energy],
               "near_still_cells": [i for i, v in enumerate(energy)
                                    if v < 0.2 * med],
               "spike_cells": [i for i, v in enumerate(energy) if v > 3 * med]},
        images=[p], reliability=0.8, backend="diff_strip",
        hint=("每一格是**相邻两个采样时刻的画面差异**(越亮=变化越大),格上标了帧号区间。\n"
              "读法:某一格接近全黑=这段时间画面几乎没变(卡顿,或本就是静止镜头——"
              "要回到原帧确认该不该动);某一格明显比邻格亮=速度突变或跳切;"
              "亮度忽明忽暗=速度不稳。\n"
              "**这张图只说明变化的多少,不说明变化得对不对。**"),
    )

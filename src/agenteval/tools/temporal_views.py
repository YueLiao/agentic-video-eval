"""Views that turn time into space, so a model that reads pictures can read motion.

Sampling is what destroys rate, and rate is most of what motion quality is: a
per-clip checklist asked about freezing, slipping, speed and amplitude scored 0%
on every one of those items across 1220 clips, because sixteen stills do not
contain the answer. The video path fixed that by handing the model the clip
whole, but a clip still arrives as 32 frames the model must integrate in its
head.

These views do the integration in the image. A space-time slice puts the whole
duration on one axis, so a stall becomes a vertical band, a jump a break, judder
a zigzag -- temporal structure rendered as the spatial pattern a VLM is good at,
in a single image that costs one image's worth of budget rather than 32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agenteval.media.clip import VideoHandle
from agenteval.tools.base import ToolResult
from agenteval.tools.renders import _label, _tile, _write


def space_time_slice(video: VideoHandle, out_dir: Path, *, n_lines: int = 3,
                     width: int = 640, max_frames: int | None = None,
                     tag: str = "xt") -> ToolResult:
    """Stack one row (and one column) of every frame into an image.

    Each output row of the horizontal panel is one video frame's scanline, so
    the vertical axis is time at full frame rate -- nothing is sampled away.
    An object moving steadily traces a straight diagonal; a jump cuts it; a
    freeze is a vertical band because consecutive rows repeat; judder is a
    zigzag; a drifting background tilts the whole field.
    """
    import cv2

    idx = list(range(min(video.total, max_frames or video.total)))
    frames = video.read(idx)
    if len(frames) < 8:
        return ToolResult(value={"error": "too few frames"}, reliability=0.0)
    h, w = frames[0].shape[:2]
    small = [cv2.resize(f, (width, int(h * width / w))) for f in frames]
    H = small[0].shape[0]
    panels, meta = [], []
    # Lines spread over the frame: a single line can miss the subject entirely.
    rows = [int(H * f) for f in np.linspace(0.25, 0.75, n_lines)]
    for r in rows:
        img = np.stack([f[r] for f in small])          # (T, width, 3)
        img = cv2.resize(img, (width, max(180, min(420, len(small) * 4))),
                         interpolation=cv2.INTER_NEAREST)
        panels.append(_label(img, f"y={r / H:.0%} 横切  ↓时间", org=(8, 22)))
        meta.append({"kind": "row", "pos": round(r / H, 3)})
    cols = [int(width * f) for f in np.linspace(0.35, 0.65, max(1, n_lines - 1))]
    for c in cols:
        img = np.stack([f[:, c] for f in small])        # (T, H, 3)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 时间轴放横向
        img = cv2.resize(img, (max(180, min(420, len(small) * 4)), width // 2),
                         interpolation=cv2.INTER_NEAREST)
        panels.append(_label(img, f"x={c / width:.0%} 竖切  →时间", org=(8, 22)))
        meta.append({"kind": "col", "pos": round(c / width, 3)})
    p = _write(out_dir / tag, f"{tag}_{video.path.stem}"[:110], _tile(panels, 1))
    return ToolResult(
        value={"n_frames": len(frames), "lines": meta, "fps": video.fps},
        images=[p], reliability=0.8, backend="space_time_slice",
        hint=("这是**时空切片**:每一条取自视频的同一条扫描线,按时间堆叠,"
              "所以一个方向是空间、另一个方向是**全帧率的时间**(没有抽样)。\n"
              "读法:匀速运动=平直斜线;**竖直条带=画面停住(卡顿或静止镜头)**;"
              "斜线突然错位=跳变或剪切;锯齿=抖动不平滑;整片倾斜=背景漂移。\n"
              "**它只显示变化的时间结构,不说明变化得对不对**——是否该动要回到原帧判断。"),
    )


def reversed_pair(video: VideoHandle, out_dir: Path, *, n: int = 6,
                  side: int = 300, tag: str = "rev") -> ToolResult:
    """The same span forwards and backwards, unlabelled, for a which-is-forward test.

    Physics has an arrow: things fall and do not rise, splashes disperse and do
    not gather, a body's weight shifts onto the foot before it pushes off.
    Generated motion often has no arrow, and a model asked to tell the forward
    pass from the reversed one will fail exactly when the clip carries no causal
    structure -- which is a physical-plausibility probe that needs no labels and
    no reference footage.
    """
    t0, t1 = 0, video.total - 1
    idx = sorted({min(video.total - 1, int(t)) for t in
                  np.linspace(t0, t1, min(n, video.total))})
    frames = video.read(idx)
    if len(frames) < 3:
        return ToolResult(value={"error": "too few frames"}, reliability=0.0)
    from agenteval.tools.renders import _resize_side
    fwd = [_resize_side(f, side) for f in frames]
    rev = fwd[::-1]
    # No frame numbers burned in: the label would give the answer away.
    top = _tile(fwd, len(fwd))
    bot = _tile(rev, len(rev))
    img = _tile([_label(top, "序列一", org=(8, 24)),
                 _label(bot, "序列二", org=(8, 24))], 1)
    p = _write(out_dir / tag, f"{tag}_{video.path.stem}"[:110], img)
    return ToolResult(
        value={"indices": idx, "forward_is": "序列一"},
        images=[p], reliability=0.7, backend="reversed_pair",
        hint=("上下两排是**同一段视频**,一排按真实时间顺序、另一排被倒过来了,"
              "**没有标注哪排是哪个**。\n"
              "判断哪一排是正放,并说明依据(重力方向、水花扩散还是汇聚、"
              "重心先落到脚上再蹬出、物体先接触再形变)。\n"
              "**如果两排看起来同样合理,说明这段运动没有时间箭头**——"
              "那本身就是物理合理性的缺陷。"),
    )


def motion_trail(video: VideoHandle, out_dir: Path, *, n: int = 8,
                 side: int = 560, tag: str = "trail") -> ToolResult:
    """Several moments blended into one frame, so the whole path is one picture.

    A trajectory judged from separate stills asks the model to hold eight
    positions in mind; blended, the path is simply visible -- evenly spaced
    ghosts mean constant speed, bunching means a stall, a gap means a jump, and
    a parabola should look like a parabola.
    """
    import cv2
    from agenteval.tools.renders import _resize_side

    idx = sorted({min(video.total - 1, int(t)) for t in
                  np.linspace(0, video.total - 1, min(n, video.total))})
    frames = [_resize_side(f, side) for f in video.read(idx)]
    if len(frames) < 3:
        return ToolResult(value={"error": "too few frames"}, reliability=0.0)
    base = frames[0].astype(np.float32)
    acc = base.copy()
    for i, f in enumerate(frames[1:], 1):
        f = f.astype(np.float32)
        # Keep what moved, so the trail shows the subject and not the backdrop.
        d = np.abs(f - base).mean(axis=2, keepdims=True)
        m = np.clip(d / max(d.max(), 1e-3), 0, 1) ** 0.6
        acc = acc * (1 - m * 0.85) + f * (m * 0.85)
    img = np.clip(acc, 0, 255).astype(np.uint8)
    img = _label(img, f"{len(frames)} 个时刻叠加 (f{idx[0]}..f{idx[-1]})", org=(8, 24))
    p = _write(out_dir / tag, f"{tag}_{video.path.stem}"[:110], img)
    return ToolResult(
        value={"indices": idx},
        images=[p], reliability=0.7, backend="motion_trail",
        hint=("这是把多个时刻**叠加在一张图**上,只保留动起来的部分。\n"
              "读法:残影间距均匀=匀速;**残影挤在一起=该处速度骤降或停顿**;"
              "出现空档=跳变;自由落体的残影间距应当**越来越大**(匀加速);"
              "残影路径应当连成一条连续曲线,断开或折返都是问题。"),
    )

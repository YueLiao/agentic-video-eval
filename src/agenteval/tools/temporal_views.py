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
                     mode: str = "diff", px_per_frame: int = 6,
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
    T = len(small)

    # Raw scanlines were the first design and they fail on the case that matters:
    # a static camera over a slow scene already produces vertical streaks
    # everywhere, so an injected freeze is not distinguishable from ambient
    # stillness -- measured, 0/6 detected. Differencing adjacent frames instead
    # makes a freeze a *black* band (zero change) against a textured field,
    # which reads the same whether or not the scene was busy to begin with.
    if mode == "diff":
        raw = [cv2.absdiff(a, b).astype(np.float32) for a, b in zip(small, small[1:])]
        # Scale by this clip's own 90th percentile so typical motion lands in
        # mid-grey. A fixed gain leaves a slow scene almost black, and then a
        # freeze -- which is what the view exists to show -- is a black band on
        # a black field.
        ref = float(np.percentile(np.stack(raw), 90)) or 1.0
        src = [np.clip(x * (190.0 / ref), 0, 255).astype(np.uint8) for x in raw]
        kind_note = "相邻帧差分(按本片 90 分位归一化)"
    else:
        src = small
        kind_note = "原始扫描线"
    Ht = max(200, min(560, len(src) * px_per_frame))

    def ruler(img, horizontal_time: bool):
        """A time axis the reader can actually cite a number from."""
        out = img.copy()
        n = len(src)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            if horizontal_time:
                x = int(frac * (out.shape[1] - 1))
                cv2.line(out, (x, out.shape[0] - 14), (x, out.shape[0] - 1),
                         (0, 255, 255), 2)
            else:
                y = int(frac * (out.shape[0] - 1))
                # Right edge: the panel title sits top-left, and a ruler there
                # collides with the title of the panel stacked above it.
                cv2.line(out, (out.shape[1] - 15, y), (out.shape[1] - 1, y),
                         (0, 255, 255), 2)
            lbl = f"{frac:.2f}"
            pos = ((x - 16, out.shape[0] - 18) if horizontal_time
                   else (out.shape[1] - 58, min(out.shape[0] - 4, y + 14)))
            cv2.putText(out, lbl, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 255, 255), 1, cv2.LINE_AA)
        return out

    panels, meta = [], []
    rows = [int(H * f) for f in np.linspace(0.25, 0.75, n_lines)]
    for r in rows:
        img = np.stack([f[r] for f in src])            # (T-1, width, 3)
        img = cv2.resize(img, (width, Ht), interpolation=cv2.INTER_NEAREST)
        img = ruler(img, horizontal_time=False)
        panels.append(_label(img, f"y={r / H:.0%} 横切 ↓时间(0→1)", org=(8, 24)))
        meta.append({"kind": "row", "pos": round(r / H, 3)})
    cols = [int(width * f) for f in np.linspace(0.35, 0.65, max(1, n_lines - 1))]
    for c in cols:
        img = np.stack([f[:, c] for f in src])          # (T-1, H, 3)
        # ROTATE_90_COUNTERCLOCKWISE already puts time left-to-right; the extra
        # horizontal flip that used to be here reversed it, so every column
        # panel reported its defects at 1-t and localisation was wrong by
        # construction.
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.resize(img, (Ht, width // 2), interpolation=cv2.INTER_NEAREST)
        img = ruler(img, horizontal_time=True)
        panels.append(_label(img, f"x={c / width:.0%} 竖切 →时间(0→1)", org=(8, 24)))
        meta.append({"kind": "col", "pos": round(c / width, 3)})
    p = _write(out_dir / tag,
               f"{tag}_{mode}_{video.path.stem}"[:110], _tile(panels, 1, gap=14))
    # Also keep each panel alone. A measurement taken on the tiled composite
    # treats the stack as one axis, but every panel carries its own time axis
    # and two of them run horizontally -- read that way, an injected freeze came
    # out with negative contrast, which is impossible. Validation reads a single
    # panel; the model still receives the tile.
    for m, panel in zip(meta, panels):
        m["path"] = str(_write(
            out_dir / tag / "panels",
            f"{tag}_{mode}_{video.path.stem}_{m['kind']}{int(m['pos']*100)}"[:110],
            panel))
    return ToolResult(
        value={"n_frames": len(frames), "lines": meta, "fps": video.fps,
               "mode": mode},
        images=[p], reliability=0.8, backend="space_time_slice",
        hint=(f"这是**时空切片**({kind_note}):取视频同一条扫描线按时间堆叠,"
              "一个方向是空间、另一个是**全帧率的时间**(没有抽样),"
              "青色刻度标的是时间比例 0→1。\n"
              + ("读法:**画面变化越大越亮、越小越暗**。\n"
                 "**卡顿/抽帧 = 一条明显比周围暗的带**:在标着「↓时间」的面板里它是"
                 "**横向**的、占满整个宽度;在标着「→时间」的面板里它是**纵向**的、"
                 "占满整个高度。视频压缩会留下微弱噪点,所以它是"
                 "**明显暗于相邻区域**,而不是纯黑;\n"
                 "整条突然变亮 = 跳变或剪切;明暗周期起伏 = 速度不稳;\n"
                 "**注意**:整幅都很暗说明这段本来就几乎静止(静止镜头拍静物),"
                 "那不算卡顿——卡顿要求**周围在动而这一条不动**。"
                 if mode == "diff" else
                 "读法:匀速运动=平直斜线;竖直条带=画面停住;斜线错位=跳变;锯齿=抖动。")
              + "\n**它只显示变化的时间结构,不说明变化得对不对。**"),
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

"""VideoHandle — lazy, cached access to a single clip's frames.

Decodes as BGR uint8 via OpenCV (the format every downstream CV tool wants) and
memoizes per index, so the many overlapping frame requests a search makes over
one clip cost one decode each. RGB / greyscale / JPEG views derive on demand.

``uniform_indices`` spans the full range inclusive of the last frame
(``round(i*(total-1)/(n-1))``). Many toolkits instead use ``int(i*total/n)``,
which never returns the last frame; ``strided_indices`` is provided so such a
baseline can be reproduced exactly when comparing against one.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


def uniform_indices(total: int, n: int) -> list[int]:
    """Uniform sampling spanning the full range, first and last frame included."""
    if total <= 0 or n <= 0:
        return []
    if n == 1:
        return [0]
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]


def strided_indices(total: int, n: int) -> list[int]:
    """``int(i*total/n)`` striding. Never returns the last frame; for baselines."""
    if total <= 0 or n <= 0:
        return []
    return [int(i * total / n) for i in range(n)]


def video_hash(path: str | Path, block_size: int = 1 << 20) -> str:
    """sha256 of (size, first 1 MB) — cheap, stable cache key for a clip."""
    p = Path(path)
    h = hashlib.sha256()
    h.update(str(p.stat().st_size).encode())
    with p.open("rb") as f:
        h.update(f.read(block_size))
    return h.hexdigest()[:16]


def bgr_to_jpeg_bytes(frame: np.ndarray, quality: int = 90) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def jpeg_data_url(frame: np.ndarray, quality: int = 90) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(
        bgr_to_jpeg_bytes(frame, quality)
    ).decode()


@dataclass
class VideoHandle:
    """One clip. Decodes lazily, memoizes decoded frames within the instance."""

    path: Path
    _total: int | None = field(default=None, init=False, repr=False)
    _fps: float | None = field(default=None, init=False, repr=False)
    _cache: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _hash: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    # ---- metadata --------------------------------------------------------
    def _probe(self) -> None:
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        try:
            self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        finally:
            cap.release()

    @property
    def total(self) -> int:
        if self._total is None:
            self._probe()
        return self._total or 0

    @property
    def fps(self) -> float:
        if self._fps is None:
            self._probe()
        return self._fps or 0.0

    @property
    def duration_s(self) -> float:
        return self.total / self.fps if self.fps else 0.0

    @property
    def hash(self) -> str:
        if self._hash is None:
            self._hash = video_hash(self.path)
        return self._hash

    # ---- frame access ----------------------------------------------------
    def read(self, indices: Sequence[int]) -> list[np.ndarray]:
        """Decode the given frame indices as BGR uint8 arrays, in order.

        Uses one sequential pass when the request is dense enough that seeking
        would cost more than decoding; otherwise seeks per frame the way the
        baseline does. Both paths return identical pixels.
        """
        import cv2

        want = [int(i) for i in indices]
        missing = sorted({i for i in want if i not in self._cache})
        if missing:
            total = self.total
            cap = cv2.VideoCapture(str(self.path))
            try:
                if total <= 0:
                    # Unknown length: sequential read, take what we can get.
                    pos, need = 0, set(missing)
                    while need:
                        ok, fr = cap.read()
                        if not ok:
                            break
                        if pos in need:
                            self._cache[pos] = fr
                            need.discard(pos)
                        pos += 1
                else:
                    span = missing[-1] - missing[0] + 1
                    sequential = len(missing) > 1 and span / len(missing) < 4
                    if sequential:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, missing[0])
                        need = set(missing)
                        pos = missing[0]
                        while need and pos <= missing[-1]:
                            ok, fr = cap.read()
                            if not ok:
                                break
                            if pos in need:
                                self._cache[pos] = fr
                                need.discard(pos)
                            pos += 1
                    else:
                        for i in missing:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                            ok, fr = cap.read()
                            if ok:
                                self._cache[i] = fr
            finally:
                cap.release()
        return [self._cache[i] for i in want if i in self._cache]

    def read_rgb(self, indices: Sequence[int]) -> np.ndarray:
        """(T, H, W, 3) uint8 RGB stack."""
        frames = self.read(indices)
        if not frames:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8)
        return np.stack([f[:, :, ::-1] for f in frames])

    def read_gray(self, indices: Sequence[int], max_side: int | None = None) -> np.ndarray:
        """(T, h, w) uint8 greyscale, optionally downscaled — for flow/QC."""
        import cv2

        out = []
        for f in self.read(indices):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            if max_side:
                h, w = g.shape
                s = max_side / max(h, w)
                if s < 1:
                    g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            out.append(g)
        return np.stack(out) if out else np.zeros((0, 0, 0), dtype=np.uint8)

    def data_urls(self, indices: Sequence[int], quality: int = 90) -> list[str]:
        return [jpeg_data_url(f, quality) for f in self.read(indices)]

    def uniform(self, n: int) -> list[int]:
        return uniform_indices(self.total, n)

    def release(self) -> None:
        self._cache.clear()


_EXTS = (".mp4", ".webm", ".gif", ".mov", ".mkv", ".avi")


def find_video(videos_dir: str | Path, stem: str) -> Path | None:
    """Resolve ``<dir>/<stem>.<ext>``, then ``<dir>/<stem>-*.<ext>``."""
    d = Path(videos_dir)
    for ext in _EXTS:
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    for ext in _EXTS:
        hits = sorted(d.glob(f"{stem}-*{ext}"))
        if hits:
            return hits[0]
    return None

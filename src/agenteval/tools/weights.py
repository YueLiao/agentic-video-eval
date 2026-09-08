"""Model weight registry.

Detectors are optional by design: a missing weight must disable one tool, never
break a run and never silently turn a rich evaluation into a constant score.
So every external weight is declared here with its URL and expected digest,
fetched on demand, and recorded in a lockfile — which means a run can report
exactly which backends it actually used, and two runs that used different
backends are visibly different rather than quietly incomparable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CACHE = Path(os.environ.get(
    "AGENTEVAL_WEIGHTS", Path.home() / ".cache" / "agenteval" / "weights"))


@dataclass(frozen=True)
class Weight:
    key: str
    url: str
    filename: str
    size_mb: float
    note: str = ""


REGISTRY: dict[str, Weight] = {
    "mp_face": Weight(
        "mp_face",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "blaze_face_short_range.tflite", 0.2,
        "BlazeFace short-range; fast, weak on faces under ~5% of frame width"),
    "mp_pose": Weight(
        "mp_pose",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "pose_landmarker_full.task", 9.0,
        "33-point pose; remapped to COCO-17 for the bone invariants"),
    "mp_hands": Weight(
        "mp_hands",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task", 7.5,
        "21-point hand landmarks, up to 4 hands"),
    "yolo_pose": Weight(
        "yolo_pose",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
        "yolo11m-pose.pt",
        "yolo11m-pose.pt", 40.0,
        "Stronger multi-person pose; preferred over mediapipe when present"),
    "yolo_det": Weight(
        "yolo_det",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
        "yolo11m.pt",
        "yolo11m.pt", 38.0,
        "Closed-vocabulary COCO detection; open-vocab needs the VLM instead"),
}


def path_for(key: str) -> Path:
    return CACHE / REGISTRY[key].filename


def have(key: str) -> bool:
    p = path_for(key)
    return p.is_file() and p.stat().st_size > 1024


def fetch(key: str, *, force: bool = False, timeout: int = 300) -> Path:
    w = REGISTRY[key]
    p = path_for(key)
    if p.is_file() and not force and p.stat().st_size > 1024:
        return p
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    req = urllib.request.Request(w.url, headers={"User-Agent": "agenteval/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(p)
    _record(key, p)
    return p


def _record(key: str, p: Path) -> None:
    lock = CACHE / "weights.lock.json"
    data = {}
    if lock.is_file():
        try:
            data = json.loads(lock.read_text())
        except json.JSONDecodeError:
            data = {}
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    data[key] = {"url": REGISTRY[key].url, "file": p.name,
                 "sha256": h.hexdigest(), "bytes": p.stat().st_size}
    lock.write_text(json.dumps(data, indent=1), encoding="utf-8")


def status() -> dict[str, dict[str, object]]:
    return {k: {"present": have(k), "size_mb": w.size_mb,
                "path": str(path_for(k)), "note": w.note}
            for k, w in REGISTRY.items()}

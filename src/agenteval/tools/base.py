"""Tool contract.

Every capability the agent can invoke — a signal, a detector, a crop, a VLM
question, a sandboxed computation — is a Tool returning a ToolResult.

Two fields carry most of the design weight:

``reliability``
    The tool's own confidence in its output, 0..1. Mandatory. Results below
    ``RELIABILITY_FLOOR`` are withheld from judge prompts entirely: showing an
    unreliable number is worse than showing none, because the judge anchors on
    it as fact.

``hint``
    How to read the numbers, in words. A judge shown ``limb_ratio_cv=0.31``
    knows nothing; shown "bone lengths vary 31% over time; a real body is
    under 5%" it knows what it is looking at.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal

RELIABILITY_FLOOR = 0.3

ToolKind = Literal["signal", "detector", "view", "vlm", "sandbox"]


@dataclass
class ToolResult:
    value: dict[str, Any] = field(default_factory=dict)
    images: list[Path] = field(default_factory=list)
    reliability: float = 1.0
    backend: str = ""
    hint: str = ""
    warnings: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    cached: bool = False

    @property
    def admissible(self) -> bool:
        """May these numbers be shown to a judge?"""
        return self.reliability >= RELIABILITY_FLOOR

    def to_json(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "images": [str(p) for p in self.images],
            "reliability": round(self.reliability, 3),
            "backend": self.backend,
            "hint": self.hint,
            "warnings": list(self.warnings),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "cached": self.cached,
        }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    kind: ToolKind
    cost_units: float
    deterministic: bool
    summary: str          # one line, shown to the offline compiler LLM
    returns: str          # short schema description, also shown to the compiler


class Tool(ABC):
    spec: ClassVar[ToolSpec]

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """(usable, reason). Checked once at startup; unusable tools are hidden
        from the agent rather than failing mid-run."""
        return True, ""

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult: ...

    def __call__(self, **kwargs: Any) -> ToolResult:
        t0 = time.perf_counter()
        res = self.run(**kwargs)
        res.elapsed_ms = (time.perf_counter() - t0) * 1000
        if not res.backend:
            res.backend = self.spec.name
        return res


# ---- registry -----------------------------------------------------------

REGISTRY: dict[str, type[Tool]] = {}


def register(cls: type[Tool]) -> type[Tool]:
    if not hasattr(cls, "spec"):
        raise TypeError(f"{cls.__name__} has no `spec`")
    REGISTRY[cls.spec.name] = cls
    return cls


def get(name: str) -> Tool:
    if name not in REGISTRY:
        raise KeyError(f"unknown tool {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def available() -> dict[str, tuple[bool, str]]:
    return {n: c.availability() for n, c in sorted(REGISTRY.items())}


def catalog() -> str:
    """The tool menu handed to the offline condition compiler."""
    lines = []
    for name, cls in sorted(REGISTRY.items()):
        ok, why = cls.availability()
        if not ok:
            continue
        s = cls.spec
        lines.append(f"- {name}({s.kind}, cost={s.cost_units}): {s.summary} -> {s.returns}")
    return "\n".join(lines)


def cache_key(tool: str, version: str, subject: str, args: dict[str, Any]) -> str:
    """Content-addressed key: tool identity + subject (video hash) + arguments."""
    payload = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    raw = f"{tool}|{version}|{subject}|{payload}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

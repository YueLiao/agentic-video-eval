"""Scripted and echo VLM clients, for building and testing the loop offline.

The harness must be testable without burning API calls or waiting on an
endpoint, and the loop's control flow (termination, budget, repeat detection,
falsification) is exactly the part that benefits from deterministic responses.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from agenteval.llm.client import ImageRef, Usage, VLMClient, VLMResponse


class ScriptedVLM(VLMClient):
    """Replays a fixed list of responses, one per ask, matched by tag prefix.

    ``script`` maps a tag prefix to either a dict (returned as-is) or a callable
    ``(tag, user, images) -> dict``. Unmatched tags fall through to
    ``default``.
    """

    def __init__(self, script: dict[str, Any], *, default: Any = None,
                 model: str = "scripted") -> None:
        super().__init__(model=model, provider="replay", base_url="")
        self.script = script
        self.default = default if default is not None else {"summary": "", "findings": []}
        self.calls: list[dict[str, Any]] = []

    def ask(self, *, system: str, user: str, images: Sequence[ImageRef] = (),
            schema: dict[str, Any] | None = None, tag: str = "") -> VLMResponse:
        self.calls.append({"tag": tag, "user": user, "n_images": len(images)})
        payload = self.default
        for prefix, val in self.script.items():
            if tag.startswith(prefix):
                payload = val(tag, user, images) if callable(val) else val
                break
        if isinstance(payload, list):          # a queue: pop the next one
            payload = payload.pop(0) if payload else self.default
        return VLMResponse(text=json.dumps(payload, ensure_ascii=False),
                           parsed=payload,
                           usage=Usage(n_images=len(images)), model=self.model)

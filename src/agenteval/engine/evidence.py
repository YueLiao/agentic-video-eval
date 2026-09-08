"""Evidence and the bus that holds it.

Evidence is the only currency in the harness: a skill cannot assert anything it
cannot point at. Every piece carries provenance (which tool, which arguments,
how reliable), and the bus is content-addressed so that two skills asking for
the same crop of the same clip pay for it once.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agenteval.llm.client import ImageRef
from agenteval.tools.base import RELIABILITY_FLOOR, ToolResult, cache_key


@dataclass
class Evidence:
    eid: str
    tool: str
    args: dict[str, Any]
    result: ToolResult

    @property
    def admissible(self) -> bool:
        return self.result.admissible

    def numbers(self) -> dict[str, Any]:
        """The part a judge may read as text. Large arrays are summarized out."""
        out: dict[str, Any] = {}
        for k, v in self.result.value.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = list(v)[:12] if len(v) <= 12 else {
                    "n": len(v), "head": list(v)[:6]}
            elif isinstance(v, dict):
                out[k] = {kk: vv for kk, vv in list(v.items())[:12]}
        return out

    def to_prompt_block(self) -> dict[str, Any]:
        b = {"eid": self.eid, "tool": self.tool, **self.numbers(),
             "reliability": round(self.result.reliability, 2)}
        if self.result.hint:
            b["hint"] = self.result.hint
        return b

    def images(self) -> list[ImageRef]:
        return [ImageRef(path=p, caption=f"[{self.eid}] {p.stem}")
                for p in self.result.images]


class EvidenceBus:
    """Blackboard shared by every skill working on one clip."""

    def __init__(self, subject: str, cache_dir: str | Path | None = None) -> None:
        self.subject = subject                      # video hash
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._by_eid: dict[str, Evidence] = {}
        self._by_key: dict[str, str] = {}
        self._n = 0
        self.calls = 0
        self.cache_hits = 0

    def get_or_run(self, tool_name: str, version: str,
                   fn: Callable[[], ToolResult], **args: Any) -> Evidence:
        key = cache_key(tool_name, version, self.subject, args)
        if key in self._by_key:
            self.cache_hits += 1
            return self._by_eid[self._by_key[key]]
        res = fn()
        self.calls += 1
        self._n += 1
        ev = Evidence(eid=f"E{self._n:02d}", tool=tool_name, args=args, result=res)
        self._by_eid[ev.eid] = ev
        self._by_key[key] = ev.eid
        return ev

    def put(self, tool_name: str, res: ToolResult, **args: Any) -> Evidence:
        self._n += 1
        ev = Evidence(eid=f"E{self._n:02d}", tool=tool_name, args=args, result=res)
        self._by_eid[ev.eid] = ev
        return ev

    def get(self, eid: str) -> Evidence | None:
        return self._by_eid.get(eid)

    def all(self) -> list[Evidence]:
        return list(self._by_eid.values())

    def admissible(self, eids: list[str] | None = None) -> list[Evidence]:
        """Evidence a judge may see. Unreliable tool output is withheld entirely
        rather than shown with a caveat: a number in the prompt gets anchored on
        as fact regardless of the caveat attached to it."""
        pool = [self._by_eid[e] for e in eids if e in self._by_eid] if eids else self.all()
        return [e for e in pool if e.admissible]

    def snapshot(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "n_evidence": len(self._by_eid),
            "tool_calls": self.calls, "cache_hits": self.cache_hits,
            "evidence": {e.eid: {"tool": e.tool, "args": e.args,
                                 **e.result.to_json()} for e in self.all()},
        }

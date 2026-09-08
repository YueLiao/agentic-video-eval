"""The action menu — what a skill lets the model choose between.

This is the hinge of the whole design. Native function-calling would let the
model call anything at any time: maximally flexible, but unreproducible, prone
to over-calling (which measurably *hurts* judging on subjective criteria), and
impossible to budget. A fixed probe sequence is the opposite failure — cheap
and reproducible, but the path never changes with the input, so it is not
adaptive at all.

A per-skill action menu sits between the two. The model genuinely chooses what
to look at next, but only from a small curated set that makes sense for that
dimension, with typed arguments. Different skills expose different menus, and
that is precisely what makes routing meaningful: a human-fidelity skill can
zoom to a face, a camera skill cannot, and neither can wander into the other's
territory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from agenteval.engine.evidence import Evidence, EvidenceBus


@dataclass(frozen=True)
class Action:
    name: str
    description: str                       # one line, shown to the model
    args_schema: dict[str, str]            # arg name -> "type: meaning"
    cost: float = 1.0
    run: Callable[..., Evidence] | None = None   # bound by the skill at runtime

    def render(self) -> str:
        if not self.args_schema:
            return f'- {self.name}: {self.description}  (no args)'
        args = "; ".join(f"{k} ({v})" for k, v in self.args_schema.items())
        return f"- {self.name}: {self.description}  args: {args}"


CONCLUDE = Action(
    name="conclude",
    description="证据已经足够,给出结论。只有当你能指出具体证据时才用它",
    args_schema={"reason": "str: 一句话说明为什么证据已经足够"},
    cost=0.0,
)


def render_menu(actions: Sequence[Action]) -> str:
    return "\n".join(a.render() for a in actions)


DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["thought", "action", "args"],
    "properties": {
        "thought": {"type": "string", "description": "为什么下一步要看这个"},
        "action": {"type": "string"},
        "args": {"type": "object"},
    },
}


def decision_instructions(actions: Sequence[Action]) -> str:
    names = " | ".join(a.name for a in actions)
    return "\n".join([
        "## 你可以采取的动作",
        render_menu(actions),
        "",
        "## 输出",
        "只输出 JSON,不要任何多余文字:",
        '{"thought": "<为什么下一步看这个>", "action": "<' + names + '>", "args": {}}',
        "",
        "规则:",
        "1. 每次只选一个动作。",
        "2. 如果已有证据足以下结论,选 conclude,不要为了凑轮数继续看。",
        "3. 如果一个动作已经做过且没带来新信息,不要重复它。",
    ])

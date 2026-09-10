"""Asking modes: the same evidence, asked differently.

A VLM's output is bounded by what is in its context and by what it was asked.
Everything in tools/ works on the first. This works on the second, and it is the
cheapest layer in the system -- no extra frames, no extra models, just a
different question over evidence already gathered.

The default mode -- hand over evidence, request a judgement -- is the weakest of
the set for anything subtle, because it lets the model answer from an overall
impression. Every mode here forces a more specific commitment:

``describe_then_judge``  say what is visible first, then judge from that
``enumerate``            answer for every item, not the salient one or two
``compare``              rank two things instead of scoring each absolutely
``locate``               return coordinates, which can be checked by IoU
``verify``               take a claim and try to confirm it independently
``discriminate``         say which of two clips is generated, and why
``counterfactual``       say what a real recording would look like here

``discriminate`` is the sharpest. Discrimination is easier than description:
asked "what is wrong with this video" a model must invent a vocabulary and may
invent findings to fill it, while asked "which of these is generated and how can
you tell" it cannot answer without naming a concrete tell -- and that tell is the
finding we wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from agenteval.llm.client import ImageRef, VLMClient, VLMResponse

JSON_ONLY = "只输出 JSON，不要任何多余文字。"


@dataclass
class Observation:
    """What one ask returned, plus how it was obtained."""

    mode: str
    parsed: dict[str, Any]
    raw: str = ""
    votes: list[dict[str, Any]] = field(default_factory=list)
    agreement: float = 1.0
    ok: bool = True
    error: str | None = None

    def get(self, k: str, default=None):
        return self.parsed.get(k, default)


def _one(vlm: VLMClient, system: str, user: str, images, schema, tag) -> VLMResponse:
    return vlm.ask(system=system, user=user, images=images, schema=schema, tag=tag)


def _vote(vlm: VLMClient, system: str, user: str, images, schema, tag, *,
          k: int, key: str, temperature: float = 0.7) -> Observation:
    """Sample k times and take the majority on `key`.

    Agreement is returned rather than discarded: how often a model contradicts
    itself on a question is a better confidence estimate than the confidence it
    reports, which in practice tends to be a constant.
    """
    if k <= 1:
        r = _one(vlm, system, user, images, schema, tag)
        return Observation("judge", r.parsed or {}, r.text, ok=r.ok, error=r.error)
    saved = vlm.temperature
    vlm.temperature = temperature
    votes: list[dict[str, Any]] = []
    try:
        for i in range(k):
            r = _one(vlm, system, user, images, schema, f"{tag}/v{i}")
            if r.ok and r.parsed:
                votes.append(r.parsed)
    finally:
        vlm.temperature = saved
    if not votes:
        return Observation("vote", {}, ok=False, error="no usable samples")
    vals = [str(v.get(key)) for v in votes]
    top = max(set(vals), key=vals.count)
    winner = next(v for v in votes if str(v.get(key)) == top)
    return Observation("vote", winner, votes=votes,
                       agreement=vals.count(top) / len(vals))

# ---- prompt bodies, kept together so wording can be reviewed as a set -------

DTJ_INSTRUCTION = """分两步作答,不要跳过第一步:
1. `observation`:**只描述你在图中实际看到什么**,具体到位置与形态。
   此时不要下任何判断,也不要使用"缺陷""异常"这类评价词。
2. `judgement`:**仅依据你上面写下的 observation** 作出判定。
   observation 里没提到的东西,不能作为判定依据。

输出 JSON:{"observation": "...", "judgement": {...}}"""

ENUM_HEADER = "## 逐项作答(每一项都必须给出结论,不能跳过)"
ENUM_OPTIONS = "可选结论:%s"
ENUM_TAIL = """**沉默不等于没问题——不提就等于没检查。**
输出 JSON:{"items": {"<key>": "<结论>", ...}, "notes": "..."}"""

COMPARE_SCHEMA = """输出 JSON:{"winner": "A"|"B"|"tie", "margin": "slight"|"clear",
 "reason": "指出具体的画面依据"}"""

POSITION_BIAS = "两种呈现顺序下结论相反(%s vs %s),判为位置偏置,不作偏好"

LOCATE_PROMPT = """在图中定位:%s
返回归一化坐标(0~1,原点左上)。找不到就返回 found=false,**不要猜**。
输出 JSON:{"found": true|false, "bbox": [x,y,w,h], "frame": <帧号或 null>,
 "confidence": 0~1}"""

VERIFY_PROMPT = """## 待核实的断言
%s

## 证据说明
%s

请独立核实,不要默认断言为真。分两步:
1. `required`:如果这个断言成立,画面里**应当**能看到什么?
2. `observed`:你在证据里**实际**看到了什么?

然后给出 verdict:`supported` 需要 observed 确实包含 required;
`contradicted` 需要 observed 与 required 冲突;两者都不成立时用 `insufficient`。
输出 JSON:{"required":"...","observed":"...",
 "verdict":"supported"|"contradicted"|"insufficient","reason":"..."}"""

DISCRIMINATE_PROMPT = """下面是两段视频的采样帧,其中**恰好一段是 AI 生成的**,另一段是真实拍摄。

指出哪一段是生成的,并说明**你依据的具体画面证据**——
是哪一帧、哪个部位、什么形态让你这样判断。
如果你无法区分,就返回 pick="unsure"。**不要猜。**

输出 JSON:{"pick":"A"|"B"|"unsure",
 "tells":[{"frame":<帧号>,"region":"...","what":"具体现象"}],"reason":"..."}"""

CF_PROMPT = """聚焦于:%s

1. `expected`:如果这是**真实拍摄**的画面,这个部位在这段时间内应当呈现什么样子?
   (先描述真实世界中的表现)
2. `actual`:图中**实际**呈现的是什么样子?
3. `divergence`:两者的差异,若无差异就写"无"。

输出 JSON:{"expected":"...","actual":"...","divergence":"...",
 "is_anomalous": true|false}"""


def judge(vlm: VLMClient, *, question: str, images: Sequence[ImageRef],
          schema: dict, system: str = "", tag: str = "ask/judge",
          votes: int = 1, vote_key: str = "grade") -> Observation:
    """Default: evidence in, judgement out."""
    return _vote(vlm, system or JSON_ONLY, question, images, schema, tag,
                 k=votes, key=vote_key)


def describe_then_judge(vlm: VLMClient, *, question: str,
                        images: Sequence[ImageRef], schema: dict,
                        system: str = "", tag: str = "ask/dtj") -> Observation:
    """Force a factual description first, then judge from that description.

    Two things happen. The judgement is anchored to stated observations rather
    than an impression; and the description is auditable on its own -- when a
    verdict looks wrong, the description usually shows whether the model
    misperceived or misjudged, which are different bugs with different fixes.
    """
    user = question + "\n\n" + DTJ_INSTRUCTION
    r = _one(vlm, system or JSON_ONLY, user, images, {"type": "object"}, tag)
    p = r.parsed or {}
    inner = p.get("judgement") if isinstance(p.get("judgement"), dict) else {}
    return Observation("describe_then_judge",
                       {**inner, "_observation": p.get("observation", "")},
                       r.text, ok=r.ok, error=r.error)


def enumerate_items(vlm: VLMClient, *, items: Sequence[tuple[str, str]],
                    context: str, images: Sequence[ImageRef],
                    options: Sequence[str], system: str = "",
                    tag: str = "ask/enum") -> Observation:
    """Require an answer for every item, not just the salient ones.

    Measured: asked to "list the problems you confirmed", a judge reported one
    or two and stayed silent on the rest, and silence was read as clean -- 46
    aspects scored a perfect 10 without ever having been judged.
    """
    listing = "\n".join("- `%s` %s" % (k, v) for k, v in items)
    opts = " / ".join("`%s`" % o for o in options)
    user = (context + "\n\n" + ENUM_HEADER + "\n" + listing
            + "\n\n" + ENUM_OPTIONS % opts + "\n" + ENUM_TAIL)
    r = _one(vlm, system or JSON_ONLY, user, images, {"type": "object"}, tag)
    return Observation("enumerate", r.parsed or {}, r.text, ok=r.ok, error=r.error)


def compare(vlm: VLMClient, *, question: str, images: Sequence[ImageRef],
            system: str = "", tag: str = "ask/compare",
            n_a: int = 0, swap_check: bool = True,
            swapped_images: Sequence[ImageRef] | None = None) -> Observation:
    """Rank two things instead of scoring each.

    Relative judgements are more reliable than absolute ones, and the benchmark
    this feeds is pairwise anyway. With `swap_check` the pair is asked in both
    orders: an answer that flips when the order flips is position bias rather
    than a preference, and is downgraded to a tie.

    `n_a` is how many of `images` belong to side A, so the two sides can be
    genuinely swapped rather than the whole list reversed -- reversing would
    also reverse time within each side.

    `swapped_images` must be supplied when any evidence has A/B burned into the
    picture -- a plot with "A speed" in its legend, a grid with labelled rows.
    Reordering the list leaves those labels saying the opposite of the position,
    and the model sees self-contradictory evidence. Measured: adding a labelled
    motion-curve plot without this drove the order-flip rate to 83/100 and
    collapsed decidable pairs from 72 to 12. The swap check caught it, which is
    what it is for, but the fix belongs at the source.
    """
    schema = {"type": "object"}
    r1 = _one(vlm, system or JSON_ONLY, question + "\n\n" + COMPARE_SCHEMA,
              images, schema, tag)
    p1 = r1.parsed or {}
    if not r1.ok:
        return Observation("compare", p1, r1.text, ok=False, error=r1.error)
    if not swap_check:
        return Observation("compare", p1, r1.text, ok=True)
    if swapped_images is None and (not n_a or n_a >= len(images)):
        # Silently skipping the check is how a position-biased judge passes for
        # a working one. A run that left n_a unset reported zero flips and 61.1%
        # direction accuracy while picking side A on 74% of pairs against a 37%
        # base rate -- the accuracy was the bias, not a judgement. Refuse rather
        # than return an unchecked answer.
        return Observation("compare", p1, r1.text, ok=False,
                           error="swap_check requested but no way to swap sides "
                                 "(pass n_a or swapped_images)")

    if swapped_images is not None:
        swapped = list(swapped_images)
    else:
        imgs = list(images)
        swapped = imgs[n_a:] + imgs[:n_a]
    r2 = _one(vlm, system or JSON_ONLY, question + "\n\n" + COMPARE_SCHEMA,
              swapped, schema, tag + "/swap")
    p2 = r2.parsed or {}
    w1, w2 = str(p1.get("winner")), str(p2.get("winner"))
    flip = {"A": "B", "B": "A"}
    consistent = (w1 == "tie" or w2 == "tie" or flip.get(w2, w2) == w1)
    out = dict(p1)
    out["_order_consistent"] = consistent
    if not consistent:
        out["winner"] = "tie"
        out["reason"] = POSITION_BIAS % (w1, w2)
    return Observation("compare", out, r1.text,
                       agreement=1.0 if consistent else 0.0)


def locate(vlm: VLMClient, *, target: str, images: Sequence[ImageRef],
           system: str = "", tag: str = "ask/locate") -> Observation:
    """Ask for coordinates rather than a description.

    A box can be checked -- against a detector, against a later crop, against
    whether anything is there at all. Prose cannot, which is why a grounded
    answer is worth more than a fluent one even when both are right.
    """
    r = _one(vlm, system or JSON_ONLY, LOCATE_PROMPT % target, images,
             {"type": "object"}, tag)
    return Observation("locate", r.parsed or {}, r.text, ok=r.ok, error=r.error)


def verify(vlm: VLMClient, *, claim: str, evidence_note: str,
           images: Sequence[ImageRef], system: str = "",
           tag: str = "ask/verify") -> Observation:
    """Take a claim and try to confirm it from evidence, independently.

    The burden sits on the claim. A model asked merely "is this right?" agrees,
    so it is asked what would have to be visible for the claim to hold, and then
    whether that is in fact visible.
    """
    r = _one(vlm, system or JSON_ONLY, VERIFY_PROMPT % (claim, evidence_note),
             images, {"type": "object"}, tag)
    return Observation("verify", r.parsed or {}, r.text, ok=r.ok, error=r.error)


def discriminate(vlm: VLMClient, *, images: Sequence[ImageRef],
                 system: str = "", tag: str = "ask/discriminate") -> Observation:
    """Ask which of two clips is generated, and on what evidence.

    Needs a real reference clip of similar content, so it is not always
    available. When it is, it produces sharper defect descriptions than any
    direct question, because naming a tell is the only way to answer.
    """
    r = _one(vlm, system or JSON_ONLY, DISCRIMINATE_PROMPT, images,
             {"type": "object"}, tag)
    return Observation("discriminate", r.parsed or {}, r.text, ok=r.ok, error=r.error)


def counterfactual(vlm: VLMClient, *, focus: str, images: Sequence[ImageRef],
                   system: str = "", tag: str = "ask/cf") -> Observation:
    """Ask what a real recording would look like here, then compare.

    Routes around the prior problem: the model's training is natural video, so
    asking what natural video *would* show queries a distribution it knows well,
    instead of asking it to recognise an artifact class it never saw.
    """
    r = _one(vlm, system or JSON_ONLY, CF_PROMPT % focus, images,
             {"type": "object"}, tag)
    return Observation("counterfactual", r.parsed or {}, r.text,
                       ok=r.ok, error=r.error)


MODES = {
    "judge": judge, "describe_then_judge": describe_then_judge,
    "enumerate": enumerate_items, "compare": compare, "locate": locate,
    "verify": verify, "discriminate": discriminate,
    "counterfactual": counterfactual,
}

"""VLM client — one provider-agnostic surface for the whole harness.

Everything the agent asks a model goes through :meth:`VLMClient.ask`. That is
deliberate: it gives one place for schema enforcement, retry, caching, cost and
latency accounting, and a complete replayable log. A skill never talks to an
HTTP endpoint.

Providers are OpenAI-compatible by default, which covers vLLM / SGLang / most
hosted APIs. Anthropic and Gemini differ only in message encoding and are
handled by their own encoders.

Two properties the rest of the design leans on:

*Structured output.* Free-text verdicts cannot be aggregated or audited, so
every ask carries a JSON schema and the client is responsible for getting a
conforming object back (retrying with the parse error fed back if not).

*Determinism.* temperature 0 by default and a content-addressed cache keyed on
(model, prompt, images, schema). Re-running an evaluation must not resample the
judge, or nothing downstream is comparable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

Provider = Literal["openai", "anthropic", "gemini", "replay"]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_images: int = 0
    latency_ms: float = 0.0
    cached: bool = False

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + o.prompt_tokens,
                     self.completion_tokens + o.completion_tokens,
                     self.n_images + o.n_images,
                     self.latency_ms + o.latency_ms,
                     self.cached and o.cached)


@dataclass
class VLMResponse:
    text: str
    parsed: dict[str, Any] | None
    usage: Usage
    model: str
    raw: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.parsed is not None


@dataclass
class ImageRef:
    """An image to send. Either bytes (JPEG) or a path; caption is shown to the
    model so it can refer to a specific frame instead of "the third image"."""

    data: bytes | None = None
    path: Path | None = None
    caption: str = ""

    def jpeg(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.path is not None:
            return Path(self.path).read_bytes()
        raise ValueError("ImageRef has neither data nor path")

    def digest(self) -> str:
        return hashlib.sha256(self.jpeg()).hexdigest()[:16]

    def data_url(self) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.jpeg()).decode()


def _extract_json(text: str) -> Any:
    """Pull a JSON object/array out of a response that may be fenced or chatty."""
    s = (text or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s[:4].lower() == "json":
                s = s[4:]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for op, cl in (("{", "}"), ("[", "]")):
        i, j = s.find(op), s.rfind(cl)
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in response")


class VLMClient:
    def __init__(
        self,
        model: str,
        *,
        provider: Provider = "openai",
        base_url: str | None = None,
        api_key_env: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.total = Usage()

    # ---- cache ----------------------------------------------------------
    def _key(self, system: str, user: str, images: Sequence[ImageRef],
             schema: dict | None) -> str:
        h = hashlib.sha256()
        for part in (self.model, str(self.temperature), system, user,
                     json.dumps(schema, sort_keys=True) if schema else ""):
            h.update(part.encode())
            h.update(b"\x00")
        for im in images:
            h.update(im.digest().encode())
        return h.hexdigest()[:24]

    def _cache_get(self, key: str) -> dict | None:
        if not self.cache_dir or self.temperature != 0.0:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def _cache_put(self, key: str, payload: dict) -> None:
        if not self.cache_dir or self.temperature != 0.0:
            return
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # ---- message encoding ----------------------------------------------
    def _encode(self, system: str, user: str, images: Sequence[ImageRef]) -> dict:
        if self.provider == "anthropic":
            content: list[dict[str, Any]] = []
            for im in images:
                if im.caption:
                    content.append({"type": "text", "text": im.caption})
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.b64encode(im.jpeg()).decode()}})
            content.append({"type": "text", "text": user})
            return {"model": self.model, "system": system,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": self.max_tokens, "temperature": self.temperature}
        if self.provider == "gemini":
            parts: list[dict[str, Any]] = []
            for im in images:
                if im.caption:
                    parts.append({"text": im.caption})
                parts.append({"inline_data": {"mime_type": "image/jpeg",
                                              "data": base64.b64encode(im.jpeg()).decode()}})
            parts.append({"text": user})
            return {"system_instruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": parts}],
                    "generationConfig": {"temperature": self.temperature,
                                         "maxOutputTokens": self.max_tokens}}
        # openai-compatible
        content = []
        for im in images:
            if im.caption:
                content.append({"type": "text", "text": im.caption})
            content.append({"type": "image_url", "image_url": {"url": im.data_url()}})
        content.append({"type": "text", "text": user})
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content})
        return {"model": self.model, "messages": msgs,
                "max_tokens": self.max_tokens, "temperature": self.temperature}

    def _url_headers(self) -> tuple[str, dict[str, str]]:
        if self.provider == "anthropic":
            return (self.base_url or "https://api.anthropic.com") + "/v1/messages", {
                "x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"}
        if self.provider == "gemini":
            base = self.base_url or "https://generativelanguage.googleapis.com/v1beta"
            return (f"{base}/models/{self.model}:generateContent?key={self.api_key}",
                    {"Content-Type": "application/json"})
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return self.base_url + "/chat/completions", h

    def _decode(self, data: dict) -> tuple[str, Usage]:
        if self.provider == "anthropic":
            txt = "".join(b.get("text", "") for b in data.get("content", []))
            u = data.get("usage", {})
            return txt, Usage(u.get("input_tokens", 0), u.get("output_tokens", 0))
        if self.provider == "gemini":
            cands = data.get("candidates", [])
            txt = "".join(p.get("text", "")
                          for p in (cands[0]["content"]["parts"] if cands else []))
            u = data.get("usageMetadata", {})
            return txt, Usage(u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0))
        txt = data["choices"][0]["message"].get("content", "") or ""
        u = data.get("usage", {})
        return txt, Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))

    def _post(self, payload: dict) -> dict:
        url, headers = self._url_headers()
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.load(r)

    # ---- the single entry point -----------------------------------------
    def ask(
        self,
        *,
        system: str,
        user: str,
        images: Sequence[ImageRef] = (),
        schema: dict[str, Any] | None = None,
        tag: str = "",
    ) -> VLMResponse:
        """Ask the model one question and get a schema-conforming object back.

        On a malformed response the parse error is fed back and the ask is
        retried, because a judge that returns prose on one probe out of fifty
        should not take down the run.
        """
        key = self._key(system, user, images, schema)
        hit = self._cache_get(key)
        if hit is not None:
            u = Usage(**hit.get("usage", {}))
            u.cached = True
            return VLMResponse(hit["text"], hit.get("parsed"), u, self.model,
                               attempts=0)

        payload = self._encode(system, user, images)
        last_err, text = "", ""
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                data = self._post(payload)
                text, usage = self._decode(data)
                usage.latency_ms = (time.perf_counter() - t0) * 1000
                usage.n_images = len(images)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError,
                    KeyError, IndexError, json.JSONDecodeError) as e:
                detail = ""
                if isinstance(e, urllib.error.HTTPError):
                    try:
                        detail = e.read()[:200].decode(errors="replace")
                    except Exception:  # noqa: BLE001
                        detail = ""
                last_err = f"{type(e).__name__}: {e} {detail}".strip()
                time.sleep(min(2 ** attempt, 8))
                continue

            if schema is None:
                self.total = self.total + usage
                res = VLMResponse(text, None, usage, self.model, attempts=attempt)
                self._log(tag, system, user, images, res)
                return res
            try:
                parsed = _extract_json(text)
                self.total = self.total + usage
                res = VLMResponse(text, parsed, usage, self.model, attempts=attempt)
                self._cache_put(key, {"text": text, "parsed": parsed,
                                      "usage": usage.__dict__})
                self._log(tag, system, user, images, res)
                return res
            except ValueError as e:
                last_err = f"unparseable response: {e}"
                # feed the failure back rather than silently retrying the same ask
                payload = self._encode(
                    system,
                    user + f"\n\n[前一次回复无法解析为 JSON：{e}。"
                           f"请只输出符合 schema 的 JSON，不要任何额外文字。]",
                    images)

        res = VLMResponse(text, None, Usage(), self.model,
                          attempts=self.max_retries, error=last_err)
        self._log(tag, system, user, images, res)
        return res

    def _log(self, tag: str, system: str, user: str,
             images: Sequence[ImageRef], res: VLMResponse) -> None:
        if not self.log_path:
            return
        rec = {
            "ts": time.time(), "tag": tag, "model": self.model,
            "provider": self.provider,
            "system_sha": hashlib.sha256(system.encode()).hexdigest()[:12],
            "user": user[:4000],
            "images": [{"caption": i.caption, "sha": i.digest()} for i in images],
            "text": res.text[:4000], "parsed": res.parsed,
            "usage": res.usage.__dict__, "attempts": res.attempts,
            "error": res.error,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def from_env(name: str = "default", *, cache_dir=None, log_path=None) -> VLMClient:
    """Build a client from AGENTEVAL_VLM_* env vars.

        AGENTEVAL_VLM_MODEL     e.g. Qwen/Qwen3-VL-32B-Instruct
        AGENTEVAL_VLM_PROVIDER  openai | anthropic | gemini   (default openai)
        AGENTEVAL_VLM_BASE_URL  e.g. http://localhost:8005/v1
        AGENTEVAL_VLM_KEY_ENV   name of the env var holding the key
    """
    return VLMClient(
        model=os.environ.get("AGENTEVAL_VLM_MODEL", "gpt-4o-mini"),
        provider=os.environ.get("AGENTEVAL_VLM_PROVIDER", "openai"),  # type: ignore[arg-type]
        base_url=os.environ.get("AGENTEVAL_VLM_BASE_URL", "https://api.openai.com/v1"),
        api_key_env=os.environ.get("AGENTEVAL_VLM_KEY_ENV") or None,
        cache_dir=cache_dir, log_path=log_path,
    )

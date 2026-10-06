"""AI model backends for the receptionist: Claude (Anthropic) or Gemini (Google).

Both expose the same small interface so the call flow doesn't care which one is in use.
Conversation history is stored as plain JSON in each backend's own message format, so a call
must finish on the backend it started with.
"""

import json
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import get_settings

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    call: ToolCall
    content: str
    is_error: bool = False


@dataclass
class Step:
    """One model response."""

    spoken: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] | None = None
    refused: bool = False


class TextSink:
    """Receives reply text as the model writes it (for streaming chat replies).

    Separate pieces of speech (text blocks, or replies before and after a tool call) are
    joined with a space, the same way the finished reply is.
    """

    def __init__(self, emit: Callable[[str], None]):
        self.emit = emit
        self.started = False
        self.gap = False

    def segment(self) -> None:
        """A new piece of speech starts here."""
        self.gap = self.started

    def __call__(self, delta: str) -> None:
        if not delta:
            return
        if self.gap:
            self.emit(" ")
            self.gap = False
        self.started = True
        self.emit(delta)


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant_message(self, text: str) -> dict[str, Any]:
        return {"role": "assistant", "content": text}

    def step(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]],
             on_text: TextSink | None = None) -> Step:
        s = get_settings()
        kwargs: dict[str, Any] = {}
        if s.anthropic_fallbacks:
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        if tools:
            kwargs["tools"] = tools
        kwargs.update(
            model=s.anthropic_model,
            max_tokens=8192,
            system=system,
            messages=messages,
            output_config={"effort": s.anthropic_effort},
            cache_control={"type": "ephemeral"},
        )
        if on_text is not None and hasattr(self.client.beta.messages, "stream"):
            # Stream so the words reach the chat window while Claude is still writing.
            with self.client.beta.messages.stream(**kwargs) as stream:
                for event in stream:
                    if event.type == "content_block_start" and event.content_block.type == "text":
                        on_text.segment()
                    elif event.type == "text":
                        on_text(event.text)
                response = stream.get_final_message()
        else:
            response = self.client.beta.messages.create(**kwargs)
            if on_text is not None:
                _emit_all(on_text, response.content)
        if response.stop_reason == "refusal":
            return Step(refused=True)
        blocks = [_dump(b) for b in response.content if b.type != "fallback"]
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        return Step(
            spoken=[b.text for b in response.content if b.type == "text" and b.text.strip()],
            tool_calls=calls if response.stop_reason == "tool_use" else [],
            assistant_message={"role": "assistant", "content": blocks},
        )

    def tool_results_message(self, results: list[ToolResult]) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": r.call.id, "content": r.content, "is_error": r.is_error}
                for r in results
            ],
        }


FALLBACK_GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_TIMEOUT_MS = 12_000
# When the main model is rate-limited or overloaded, try this lighter model: it has its own quota.
BUSY_FALLBACK_GEMINI_MODEL = "gemini-3.1-flash-lite"
BUSY_CODES = (429, 500, 503)
BUSY_COOLDOWN_SECONDS = 600

# The most recent AI failure, shown on the dashboard so problems are visible without host logs.
last_error: dict[str, str] = {}


def record_error(where: str, exc: Exception) -> None:
    from datetime import datetime, timezone

    last_error.update(when=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), where=where,
                      error=f"{type(exc).__name__}: {str(exc)[:500]}")  # Gemini's minimum allowed deadline is 10s


def _suggested_model(exc: Exception) -> str | None:
    """Google's 404 for a retired model names its replacement: '... use models/gemini-3.8-flash ...'."""
    m = re.search(r"use models/([\w.\-]+)", str(exc))
    return m.group(1) if m else None


def _gemini_schema(schema: Any) -> Any:
    """Plain JSON Schema for Gemini: drop keys some Gemini models reject."""
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items() if k not in ("additionalProperties", "strict")}
    if isinstance(schema, list):
        return [_gemini_schema(v) for v in schema]
    return schema


class GeminiBackend:
    name = "gemini"
    # Set when the configured model isn't available on this key and the fallback worked.
    _model_override: str | None = None
    _busy_until: float = 0.0

    def __init__(self, client: Any = None):
        if client is None:
            from google import genai

            from google.genai import types

            client = genai.Client(
                api_key=get_settings().gemini_api_key,
                # Gemini rejects deadlines under 10s. Retry briefly on rate limits / overload.
                http_options=types.HttpOptions(
                    timeout=GEMINI_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(
                        attempts=2, initial_delay=1.0, max_delay=2.0, http_status_codes=[429, 500, 503]
                    ),
                ),
            )
        self.client = client

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "parts": [{"text": text}]}

    def assistant_message(self, text: str) -> dict[str, Any]:
        return {"role": "model", "parts": [{"text": text}]}

    def step(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]],
             on_text: TextSink | None = None) -> Step:
        step = self._step(system, tools, messages)
        if on_text is not None:  # not streamed: the whole reply arrives at once
            for text in step.spoken:
                on_text.segment()
                on_text(text)
        return step

    def _step(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> Step:
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name=t["name"], description=t["description"],
                parameters_json_schema=_gemini_schema(t["input_schema"]),
            )
            for t in tools
        ]
        contents = [types.Content.model_validate(m) for m in messages]
        model = GeminiBackend._model_override or get_settings().gemini_model
        if time.monotonic() < GeminiBackend._busy_until:
            model = BUSY_FALLBACK_GEMINI_MODEL
        try:
            response = self._generate(model, system, declarations, contents)
        except Exception as exc:
            # The configured model may not exist or not be enabled for this key (common on the
            # free tier with brand-new models). Fall back once to a widely available model.
            code = getattr(exc, "code", None)
            if code in BUSY_CODES and model != BUSY_FALLBACK_GEMINI_MODEL:
                log.warning("Gemini model %s is busy or over quota (%s: %s); trying %s",
                            model, type(exc).__name__, str(exc)[:200], BUSY_FALLBACK_GEMINI_MODEL)
                response = self._generate(BUSY_FALLBACK_GEMINI_MODEL, system, declarations, contents)
                # Stay on the lighter model for a while so a conversation doesn't bounce between models.
                GeminiBackend._busy_until = time.monotonic() + BUSY_COOLDOWN_SECONDS
                return self._to_step(response)
            suggested = _suggested_model(exc)
            # Only switch models when the model itself is the problem (retired / not found),
            # not for other request errors.
            if code != 404 and not suggested:
                raise
            fallback = suggested or FALLBACK_GEMINI_MODEL
            if fallback == model:
                raise
            log.warning("Gemini model %s failed (%s: %s); falling back to %s",
                        model, type(exc).__name__, str(exc)[:200], fallback)
            response = self._generate(fallback, system, declarations, contents)
            GeminiBackend._model_override = fallback
        return self._to_step(response)

    def _to_step(self, response) -> Step:
        candidate = (response.candidates or [None])[0]
        block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        if block_reason:
            log.warning("Gemini blocked the request: %s", block_reason)
            return Step(refused=True)
        if candidate is None or candidate.content is None or not candidate.content.parts:
            finish = getattr(candidate, "finish_reason", None)
            if str(finish).endswith(("SAFETY", "PROHIBITED_CONTENT")):
                return Step(refused=True)
            raise RuntimeError(f"Gemini returned an empty reply (finish reason: {finish})")
        parts = candidate.content.parts
        calls = [
            ToolCall(p.function_call.id or p.function_call.name, p.function_call.name, dict(p.function_call.args or {}))
            for p in parts
            if p.function_call
        ]
        spoken = [p.text for p in parts if p.text and not p.thought and p.text.strip()]
        if not calls and not spoken and str(candidate.finish_reason).endswith(("SAFETY", "PROHIBITED_CONTENT")):
            return Step(refused=True)
        content = candidate.content.model_dump(mode="json", exclude_none=True)
        content["role"] = "model"
        return Step(spoken=spoken, tool_calls=calls, assistant_message=content)

    def _generate(self, model: str, system: str, declarations: list, contents: list):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            # Thinking tokens count against this limit, so leave plenty of room.
            max_output_tokens=8192,
        )
        if model.startswith("gemini-3"):
            # A phone receptionist doesn't need deep reasoning; low thinking keeps replies fast.
            config.thinking_config = types.ThinkingConfig(thinking_level="low")
        return self.client.models.generate_content(model=model, contents=contents, config=config)

    def tool_results_message(self, results: list[ToolResult]) -> dict[str, Any]:
        parts = []
        for r in results:
            response: dict[str, Any] = {"error" if r.is_error else "result": _maybe_json(r.content)}
            fr: dict[str, Any] = {"name": r.call.name, "response": response}
            if r.call.id != r.call.name:
                fr["id"] = r.call.id
            parts.append({"function_response": fr})
        return {"role": "user", "parts": parts}


def _emit_all(on_text: TextSink, blocks: list[Any]) -> None:
    """For a reply that wasn't streamed: hand over its text all at once."""
    for b in blocks:
        if b.type == "text" and b.text.strip():
            on_text.segment()
            on_text(b.text)


def _dump(block: Any) -> dict[str, Any]:
    return block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else dict(block)


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


def current_model() -> str:
    s = get_settings()
    if provider() == "gemini":
        return GeminiBackend._model_override or s.gemini_model
    return s.anthropic_model


def provider() -> str:
    s = get_settings()
    if s.llm_provider in ("anthropic", "gemini"):
        return s.llm_provider
    return "gemini" if s.gemini_api_key else "anthropic"

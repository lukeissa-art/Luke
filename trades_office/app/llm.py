"""AI model backends for the receptionist: Claude (Anthropic) or Gemini (Google).

Both expose the same small interface so the call flow doesn't care which one is in use.
Conversation history is stored as plain JSON in each backend's own message format, so a call
must finish on the backend it started with.
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any

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

    def step(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> Step:
        s = get_settings()
        kwargs: dict[str, Any] = {}
        if s.anthropic_fallbacks:
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = self.client.beta.messages.create(
            model=s.anthropic_model,
            max_tokens=8192,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": s.anthropic_effort},
            cache_control={"type": "ephemeral"},
            **kwargs,
        )
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

    def __init__(self, client: Any = None):
        if client is None:
            from google import genai

            from google.genai import types

            client = genai.Client(
                api_key=get_settings().gemini_api_key,
                # Retry briefly on rate limits / overload, but stay inside Twilio's ~15s webhook window.
                http_options=types.HttpOptions(
                    timeout=6_000,
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

    def step(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> Step:
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
        try:
            response = self._generate(model, system, declarations, contents)
        except Exception as exc:
            # The configured model may not exist or not be enabled for this key (common on the
            # free tier with brand-new models). Fall back once to a widely available model.
            code = getattr(exc, "code", None)
            fallback = _suggested_model(exc) or FALLBACK_GEMINI_MODEL
            if fallback == model or code not in (400, 403, 404):
                raise
            log.warning("Gemini model %s failed (%s: %s); falling back to %s",
                        model, type(exc).__name__, str(exc)[:200], fallback)
            response = self._generate(fallback, system, declarations, contents)
            GeminiBackend._model_override = fallback
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
            tools=[types.Tool(function_declarations=declarations)],
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

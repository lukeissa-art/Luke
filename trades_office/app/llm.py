"""AI model backends for the receptionist: Claude (Anthropic) or Gemini (Google).

Both expose the same small interface so the call flow doesn't care which one is in use.
Conversation history is stored as plain JSON in each backend's own message format, so a call
must finish on the backend it started with.
"""

import json
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
            max_tokens=2048,
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


class GeminiBackend:
    name = "gemini"

    def __init__(self, client: Any = None):
        if client is None:
            from google import genai

            client = genai.Client(api_key=get_settings().gemini_api_key)
        self.client = client

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "parts": [{"text": text}]}

    def assistant_message(self, text: str) -> dict[str, Any]:
        return {"role": "model", "parts": [{"text": text}]}

    def step(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> Step:
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"]
            )
            for t in tools
        ]
        response = self.client.models.generate_content(
            model=get_settings().gemini_model,
            contents=[types.Content.model_validate(m) for m in messages],
            config=types.GenerateContentConfig(
                system_instruction=system,
                tools=[types.Tool(function_declarations=declarations)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                max_output_tokens=2048,
            ),
        )
        candidate = (response.candidates or [None])[0]
        if candidate is None or candidate.content is None:
            reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
            log.warning("Gemini returned no content (block reason: %s)", reason)
            return Step(refused=True)
        parts = candidate.content.parts or []
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


def provider() -> str:
    s = get_settings()
    if s.llm_provider in ("anthropic", "gemini"):
        return s.llm_provider
    return "gemini" if s.gemini_api_key else "anthropic"

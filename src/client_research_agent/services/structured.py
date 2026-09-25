"""Schema-constrained LLM calls with parse-repair.

Every agent that needs structured output goes through ``complete_structured``:
the model is asked for JSON, the reply is extracted and validated against a
Pydantic model, and on failure the validation error is fed back for a bounded
number of repair turns. Unrecoverable output raises ``OutputValidationError``
so callers can degrade gracefully instead of propagating malformed data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from client_research_agent.services.ports import ChatMessage, LLMClient, LLMResponse
from client_research_agent.utils.errors import OutputValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str:
    """Return the first JSON object/array in ``text``, tolerating code fences and prose."""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    candidate = candidate.strip()
    start_positions = [p for p in (candidate.find("{"), candidate.find("[")) if p != -1]
    if not start_positions:
        raise ValueError("no JSON object found in model output")
    start = min(start_positions)
    opener = candidate[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return candidate[start : index + 1]
    raise ValueError("unterminated JSON in model output")


def schema_instruction(model: type[BaseModel]) -> str:
    schema = json.dumps(model.model_json_schema(), separators=(",", ":"))
    return (
        "Respond with a single JSON object and nothing else. "
        f"It must validate against this JSON Schema:\n{schema}"
    )


def complete_structured(
    llm: LLMClient,
    messages: Sequence[ChatMessage],
    output_model: type[ModelT],
    *,
    max_repairs: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> tuple[ModelT, LLMResponse]:
    conversation = [*messages, ChatMessage(role="system", content=schema_instruction(output_model))]
    last_error = ""
    for _ in range(max_repairs + 1):
        response = llm.complete(conversation, temperature=temperature, max_tokens=max_tokens, json_mode=True)
        try:
            payload = json.loads(extract_json(response.content))
            return output_model.model_validate(payload), response
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:2000]
            conversation = [
                *conversation,
                ChatMessage(role="assistant", content=response.content[:8000]),
                ChatMessage(
                    role="user",
                    content=f"Your previous reply was invalid: {last_error}\nReturn only corrected JSON.",
                ),
            ]
    raise OutputValidationError(f"{output_model.__name__}: could not obtain valid output: {last_error}")

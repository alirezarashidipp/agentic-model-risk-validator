from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from app.config import settings
from app.tools.json_utils import to_jsonable


OutputT = TypeVar("OutputT", bound=BaseModel)


class TimelineLogger:
    def __init__(self, timeline_path: Path):
        self.timeline_path = timeline_path
        self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = []
        if self.timeline_path.exists():
            try:
                self.entries = json.loads(self.timeline_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.entries = []

    def log(
        self,
        agent_name: str,
        status: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.entries.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "agent_name": agent_name,
                "status": status,
                "message": message,
                "metadata": to_jsonable(metadata or {}),
            }
        )
        self.timeline_path.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class AgentRunner:
    def __init__(self, timeline: TimelineLogger):
        self.timeline = timeline

    def run(
        self,
        *,
        agent_name: str,
        instructions: str,
        context: dict[str, Any],
        output_model: type[OutputT],
        fallback: Callable[[dict[str, Any]], OutputT],
    ) -> OutputT:
        self.timeline.log(agent_name, "started", "Agent step started")

        if not settings.use_llm or not settings.openai_api_key:
            reason = "OpenAI API key is not configured" if not settings.openai_api_key else "USE_LLM is disabled"
            output = fallback(context)
            self.timeline.log(
                agent_name,
                "completed_fallback",
                reason,
                {"output": output.model_dump(mode="json")},
            )
            return output

        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key, timeout=90.0)
            response = client.responses.parse(
                model=settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": instructions,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(to_jsonable(context), indent=2),
                    },
                ],
                text_format=output_model,
            )
            parsed = response.output_parsed
            output = parsed if isinstance(parsed, output_model) else output_model.model_validate(parsed)
            self.timeline.log(
                agent_name,
                "completed",
                "Agent produced structured output",
                {"output": output.model_dump(mode="json")},
            )
            return output
        except Exception as exc:
            output = fallback(context)
            self.timeline.log(
                agent_name,
                "completed_fallback",
                f"OpenAI agent call failed, used deterministic fallback: {exc}",
                {"output": output.model_dump(mode="json")},
            )
            return output


BASE_AGENT_INSTRUCTIONS = """
You are part of a model validation team. Deterministic Python tools have already
calculated all metrics, profiling statistics, leakage checks, plots, and model
performance values. Do not calculate, estimate, adjust, invent, or recompute any
numerical metric. Interpret only the evidence supplied in the context. If a
metric, document, file, or analysis artifact is missing, state that it is missing
and explain the validation impact. Return only the requested structured output.
"""


def instructions_for(agent_name: str, role: str) -> str:
    return f"{BASE_AGENT_INSTRUCTIONS}\nAgent: {agent_name}\nRole: {role}"


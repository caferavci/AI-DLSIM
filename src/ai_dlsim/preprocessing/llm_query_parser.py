from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Literal

from ai_dlsim.schemas.query import QueryRequest


ModelName = Literal[
    "openai.gpt-5-mini",
    "meta.llama-4-maverick-17b-instruct",
    "google.gemini-2.5-flash",
    "xai.grok-3",
]


def _get_llm_client():
    """Build an OpenAI-compatible client pointing at the LiteLLM proxy."""
    from openai import OpenAI

    api_key = os.environ.get("LITELLM_API_KEY")
    base_url = os.environ.get("LITELLM_BASE_URL")

    if not api_key or not base_url:
        raise RuntimeError(
            "Missing LITELLM_API_KEY and/or LITELLM_BASE_URL in environment. "
            "Set them in .env and rerun."
        )

    return OpenAI(api_key=api_key, base_url=base_url)


class LlmQueryParser:
    """
    Pre-processing agent (Phase 5).

    Responsible only for:
      Natural language query -> structured QueryRequest (JSON)
    """

    def __init__(self, *, model: ModelName = "openai.gpt-5-mini") -> None:
        self.model = model

    def _parse_json_only(self, raw: str) -> dict[str, Any]:
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("LLM output did not contain valid JSON.")
            return json.loads(raw[start : end + 1])

    def parse(self, user_query: str) -> QueryRequest:
        client = _get_llm_client()

        schema_example = {
            "region": "Ithaca, NY",
            "origin": "Cornell University",
            "destination": "Ithaca Commons",
            "departure_time": "08:30",
            "mode": "car",
            "scenario": "baseline",
        }

        system_prompt = (
            "You are a traffic-simulation query interpreter. "
            "Convert the user query into structured parameters for an "
            "AI-driven dynamic traffic simulation pipeline.\n\n"
            "Return ONLY valid JSON (no markdown, no commentary) with these keys:\n"
            "- region (string, required)\n"
            "- origin (string or null)\n"
            "- destination (string or null)\n"
            "- departure_time (string 'HH:MM' 24h or null)\n"
            "- mode (string; one of: car, bus, bike, walk, truck; default 'car')\n"
            "- scenario (string; default 'baseline')\n\n"
            "Rules:\n"
            "- If the user doesn't provide origin/destination, set them to null.\n"
            "- If time isn't provided, set departure_time to null.\n"
            "- If region is ambiguous, choose the most likely one.\n"
        )

        user_prompt = (
            f"User query:\n{user_query}\n\n"
            f"JSON schema example:\n{json.dumps(schema_example, indent=2)}\n\n"
            "Now output the JSON object."
        )

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        raw_text = resp.choices[0].message.content or ""
        obj = self._parse_json_only(raw_text)

        normalized = {
            "region": obj.get("region"),
            "origin": obj.get("origin"),
            "destination": obj.get("destination"),
            "departure_time": obj.get("departure_time"),
            "mode": obj.get("mode") or "car",
            "scenario": obj.get("scenario") or "baseline",
        }

        if not normalized["region"]:
            raise ValueError("LLM did not provide a required 'region' field.")

        return QueryRequest(**normalized)

    def to_debug_dict(self, request: QueryRequest) -> dict[str, Any]:
        return asdict(request)

from __future__ import annotations

import os
from typing import Any


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


class LlmResultInterpreter:
    """
    Post-processing agent (Phase 5/6 output side).

    Responsible only for:
      (DLSim outputs + user query) -> human readable summary
    """

    def __init__(self, *, model: str = "openai.gpt-5-mini") -> None:
        self.model = model

    def interpret(self, *, user_query: str, dlsim_result: dict[str, Any]) -> str:
        client = _get_llm_client()

        concise_system_prompt = (
            "You are an assistant interpreting dynamic traffic simulation results. "
            "Produce a concise answer to the user.\n\n"
            "Follow this style when possible:\n"
            "- Start with the expected travel time (X minutes)\n"
            "- Mention the route identifier or node sequence (Y) if available\n"
            "- Mention congestion hotspots (e.g., link Z between times T1–T2) if available\n\n"
            "If the simulation outputs are missing or the engine did not run, "
            "say so explicitly and suggest what inputs/results are required."
        )

        user_prompt = (
            f"User query:\n{user_query}\n\n"
            f"Simulation result payload (JSON):\n{dlsim_result}\n\n"
            "Write the final answer."
        )

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": concise_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return resp.choices[0].message.content or ""

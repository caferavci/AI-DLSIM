from __future__ import annotations

import os
from typing import Any


class LlmResultInterpreter:
    """
    Post-processing agent (Phase 5/6 output side).

    Responsible only for:
      (DLSim outputs + user query) -> human readable summary
    """

    def __init__(self, *, model: str = "openai.gpt-5-mini") -> None:
        self.model = model

    def interpret(self, *, user_query: str, dlsim_result: dict[str, Any]) -> str:
        if self.model != "openai.gpt-5-mini":
            raise NotImplementedError(
                f"Provider/model '{self.model}' not wired yet. "
                "Baseline currently supports 'openai.gpt-5-mini' only."
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY in environment. "
                "Set it (e.g., export OPENAI_API_KEY=...) and rerun."
            )

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        # Keep the prompt grounded: only reference fields that exist.
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
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": concise_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return resp.choices[0].message.content or ""


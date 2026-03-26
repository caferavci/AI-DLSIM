from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QueryRequest:
    """
    Pre-processing agent output (Phase 5).

    Baseline currently supports only a subset of fields.
    """

    region: str
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_time: Optional[str] = None
    mode: str = "car"
    scenario: str = "baseline"


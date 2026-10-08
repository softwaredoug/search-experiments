from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DecisionQuestion:
    """A yes/no instruction and optional descriptions of its true/false cases."""

    instructions: str
    criteria: Mapping[str, str] | None = None

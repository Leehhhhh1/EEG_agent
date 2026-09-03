"""Data structures shared by the EEGAgent runtime skill loader and selector."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SkillSpec:
    """One runtime skill loaded from a SKILL.md file."""

    name: str
    description: str
    priority: int
    requires_session: bool
    trigger_keywords: tuple[str, ...]
    routing_examples: tuple[str, ...]
    allowed_tools: frozenset[str]
    instructions: str
    path: Path

    def as_system_message(self) -> dict[str, str]:
        """Render request-scoped instructions for DeepSeek."""
        return {
            "role": "system",
            "content": (
                f'<active_eeg_skill name="{self.name}">\n'
                f"{self.instructions.strip()}\n"
                "</active_eeg_skill>"
            ),
        }


@dataclass(frozen=True)
class SemanticCandidate:
    """One Skill candidate scored by local semantic routing."""

    name: str
    score: float
    matched_examples: tuple[str, ...]


@dataclass(frozen=True)
class SemanticSelection:
    """Result of comparing one query against specialized Skill examples."""

    accepted_name: str | None
    candidates: tuple[SemanticCandidate, ...]
    top_score: float
    margin: float


@dataclass(frozen=True)
class SkillSelection:
    """Selected Skill plus the route details shown by the desktop client."""

    skill: SkillSpec | None
    source: Literal["keyword", "embedding", "general", "no_skill"]
    keyword_matches: tuple[str, ...] = ()
    candidates: tuple[SemanticCandidate, ...] = ()
    top_score: float = 0.0
    margin: float = 0.0

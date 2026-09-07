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

    def as_instruction_block(self) -> str:
        """Render instructions scoped to the user message that contains them."""
        allowed_tools = "\n".join(
            f"- {tool_name}" for tool_name in sorted(self.allowed_tools)
        ) or "- none"
        return (
            f'<active_eeg_skill name="{self.name}" '
            'applies_to="containing_user_message">\n'
            "<allowed_tools>\n"
            f"{allowed_tools}\n"
            "</allowed_tools>\n"
            f"{self.instructions.strip()}\n"
            "</active_eeg_skill>"
        )


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

"""Data structures shared by the EEGAgent runtime skill loader and selector."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillSpec:
    """One runtime skill loaded from a SKILL.md file."""

    name: str
    description: str
    priority: int
    requires_session: bool
    trigger_keywords: tuple[str, ...]
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

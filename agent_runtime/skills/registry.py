"""Registry and validation for EEGAgent runtime skills."""

from collections.abc import Callable, Iterable
from pathlib import Path

from .loader import load_skills
from .models import SkillSpec
from .selector import select_by_keywords


DEFAULT_DEFINITIONS_DIR = Path(__file__).resolve().parent / "definitions"
FALLBACK_SKILL_NAME = "general_eeg"
DescriptionSelector = Callable[[str, tuple[SkillSpec, ...]], str | None]


class SkillRegistry:
    def __init__(self, skills: Iterable[SkillSpec]):
        self._skills: dict[str, SkillSpec] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise ValueError(f"Duplicate runtime skill name: {skill.name}")
            self._skills[skill.name] = skill
        if FALLBACK_SKILL_NAME not in self._skills:
            raise ValueError("A general_eeg fallback skill is required.")

    @classmethod
    def load_default(cls) -> "SkillRegistry":
        return cls(load_skills(DEFAULT_DEFINITIONS_DIR))

    def all(self) -> list[SkillSpec]:
        return list(self._skills.values())

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise ValueError(f"Unknown runtime skill: {name}") from exc

    def select(
        self,
        query: str,
        description_selector: DescriptionSelector | None = None,
    ) -> SkillSpec:
        """Select one skill by keywords, description routing, then fallback."""
        keyword_match = select_by_keywords(query, self._skills.values())
        if keyword_match is not None:
            return keyword_match

        if description_selector is not None:
            try:
                selected_name = description_selector(query, tuple(self._skills.values()))
            except Exception:
                selected_name = None
            if isinstance(selected_name, str) and selected_name in self._skills:
                return self._skills[selected_name]

        return self._skills[FALLBACK_SKILL_NAME]

    def validate_tools(self, available_tools: Iterable[str]) -> None:
        """Fail fast when a skill references an MCP tool that does not exist."""
        available = set(available_tools)
        errors = []
        for skill in self._skills.values():
            missing = sorted(skill.allowed_tools - available)
            if missing:
                errors.append(f"{skill.name}: {', '.join(missing)}")
        if errors:
            raise ValueError("Runtime skills reference unavailable MCP tools: " + "; ".join(errors))

"""Registry and validation for EEGAgent runtime skills."""

from collections.abc import Callable, Iterable
import os
from pathlib import Path

from .loader import load_skills
from .models import SemanticSelection, SkillSelection, SkillSpec
from .selector import matched_keywords, select_by_keywords


DEFAULT_DEFINITIONS_DIR = Path(__file__).resolve().parent / "definitions"
FALLBACK_SKILL_NAME = "general_eeg"
DEFAULT_GENERAL_MIN_SCORE = 0.45
SemanticSelector = Callable[[str, tuple[SkillSpec, ...]], SemanticSelection]


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
        semantic_selector: SemanticSelector | None = None,
    ) -> SkillSpec | None:
        """Compatibility wrapper returning the selected Skill, if any."""
        return self.select_with_details(query, semantic_selector).skill

    def select_with_details(
        self,
        query: str,
        semantic_selector: SemanticSelector | None = None,
    ) -> SkillSelection:
        """Route to a specialized Skill, a restricted general Skill, or no Skill."""
        keyword_match = select_by_keywords(query, self._skills.values())
        if keyword_match is not None:
            return SkillSelection(
                skill=keyword_match,
                source="keyword",
                keyword_matches=matched_keywords(query, keyword_match),
            )

        semantic_result = None
        if semantic_selector is not None:
            specialized = tuple(
                skill for skill in self._skills.values()
                if skill.name != FALLBACK_SKILL_NAME
            )
            try:
                semantic_result = semantic_selector(query, specialized)
            except Exception:
                semantic_result = None
            if (
                semantic_result is not None
                and semantic_result.accepted_name in self._skills
            ):
                return SkillSelection(
                    skill=self._skills[semantic_result.accepted_name],
                    source="embedding",
                    candidates=semantic_result.candidates,
                    top_score=semantic_result.top_score,
                    margin=semantic_result.margin,
                )

        general_min_score = float(
            os.getenv("SKILL_ROUTE_GENERAL_MIN_SCORE", DEFAULT_GENERAL_MIN_SCORE)
        )
        if semantic_result is not None and semantic_result.top_score >= general_min_score:
            return SkillSelection(
                skill=self._skills[FALLBACK_SKILL_NAME],
                source="general",
                candidates=semantic_result.candidates,
                top_score=semantic_result.top_score,
                margin=semantic_result.margin,
            )

        return SkillSelection(
            skill=None,
            source="no_skill",
            candidates=semantic_result.candidates if semantic_result else (),
            top_score=semantic_result.top_score if semantic_result else 0.0,
            margin=semantic_result.margin if semantic_result else 0.0,
        )

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

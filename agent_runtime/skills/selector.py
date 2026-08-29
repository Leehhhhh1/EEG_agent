"""Deterministic keyword matching for EEGAgent runtime skills."""

from collections.abc import Iterable

from .models import SkillSpec


def select_by_keywords(
    query: str,
    skills: Iterable[SkillSpec],
) -> SkillSpec | None:
    """Return the best keyword match, or ``None`` when nothing matches."""
    lowered = query.strip().lower()
    skill_list = list(skills)
    matches: list[tuple[int, int, str, SkillSpec]] = []
    for skill in skill_list:
        score = sum(keyword.lower() in lowered for keyword in skill.trigger_keywords)
        if score > 0:
            matches.append((score, skill.priority, skill.name, skill))

    if matches:
        matches.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return matches[0][3]
    return None

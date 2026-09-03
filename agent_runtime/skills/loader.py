"""Load EEGAgent runtime skills from Markdown files with YAML front matter."""

from pathlib import Path
from typing import Any

import yaml

from .models import SkillSpec


FRONT_MATTER_DELIMITER = "---"


def _read_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONT_MATTER_DELIMITER:
        raise ValueError(f"Skill file must start with YAML front matter: {path}")

    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == FRONT_MATTER_DELIMITER
        )
    except StopIteration as exc:
        raise ValueError(f"Skill file has no closing front matter delimiter: {path}") from exc

    metadata = yaml.safe_load("\n".join(lines[1:closing_index])) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Skill front matter must be a mapping: {path}")
    instructions = "\n".join(lines[closing_index + 1:]).strip()
    if not instructions:
        raise ValueError(f"Skill instructions cannot be empty: {path}")
    return metadata, instructions


def _string_list(metadata: dict[str, Any], field: str, path: Path) -> tuple[str, ...]:
    value = metadata.get(field, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"Skill field '{field}' must be a list of non-empty strings: {path}")
    return tuple(item.strip() for item in value)


def load_skill(path: Path) -> SkillSpec:
    """Load and validate one SKILL.md."""
    metadata, instructions = _read_front_matter(path)
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Skill name must be a non-empty string: {path}")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Skill description must be a non-empty string: {path}")

    priority = metadata.get("priority", 0)
    if not isinstance(priority, int):
        raise ValueError(f"Skill priority must be an integer: {path}")
    requires_session = metadata.get("requires_session", False)
    if not isinstance(requires_session, bool):
        raise ValueError(f"Skill requires_session must be a boolean: {path}")

    return SkillSpec(
        name=name.strip(),
        description=description.strip(),
        priority=priority,
        requires_session=requires_session,
        trigger_keywords=_string_list(metadata, "trigger_keywords", path),
        routing_examples=_string_list(metadata, "routing_examples", path),
        allowed_tools=frozenset(_string_list(metadata, "allowed_tools", path)),
        instructions=instructions,
        path=path.resolve(),
    )


def load_skills(definitions_dir: Path) -> list[SkillSpec]:
    """Load every definitions/<name>/SKILL.md in stable order."""
    if not definitions_dir.is_dir():
        raise FileNotFoundError(f"Skill definitions directory not found: {definitions_dir}")
    paths = sorted(definitions_dir.glob("*/SKILL.md"))
    if not paths:
        raise ValueError(f"No runtime skills found in {definitions_dir}")
    return [load_skill(path) for path in paths]

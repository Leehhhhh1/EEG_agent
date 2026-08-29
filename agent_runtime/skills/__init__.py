"""Single-skill runtime routing for EEGAgent."""

from .models import SkillSpec
from .registry import SkillRegistry

__all__ = ["SkillRegistry", "SkillSpec"]

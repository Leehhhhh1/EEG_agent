"""Local BGE-M3 multi-example semantic routing for EEG runtime Skills."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .models import SemanticCandidate, SemanticSelection, SkillSpec


DEFAULT_MIN_SCORE = 0.65
DEFAULT_MIN_MARGIN = 0.05
MATCHED_EXAMPLE_COUNT = 2


class SemanticSkillSelector:
    """Cache Skill example vectors and score user requests by cosine similarity."""

    def __init__(
        self,
        embedder: Any,
        *,
        min_score: float | None = None,
        min_margin: float | None = None,
    ):
        self.embedder = embedder
        self.min_score = float(
            os.getenv("SKILL_ROUTE_MIN_SCORE", DEFAULT_MIN_SCORE)
            if min_score is None else min_score
        )
        self.min_margin = float(
            os.getenv("SKILL_ROUTE_MIN_MARGIN", DEFAULT_MIN_MARGIN)
            if min_margin is None else min_margin
        )
        self._cache_key: tuple[tuple[str, tuple[str, ...]], ...] | None = None
        self._example_texts: dict[str, tuple[str, ...]] = {}
        self._example_vectors: dict[str, np.ndarray] = {}

    @staticmethod
    def _texts_for_skill(skill: SkillSpec) -> tuple[str, ...]:
        return (skill.description, *skill.routing_examples)

    def _prepare(self, skills: tuple[SkillSpec, ...]) -> None:
        cache_key = tuple(
            (skill.name, self._texts_for_skill(skill))
            for skill in skills
        )
        if cache_key == self._cache_key:
            return
        texts: list[str] = []
        owners: list[str] = []
        self._example_texts = {}
        for skill in skills:
            examples = self._texts_for_skill(skill)
            if not examples:
                continue
            self._example_texts[skill.name] = examples
            texts.extend(examples)
            owners.extend([skill.name] * len(examples))

        encoded = np.asarray(self.embedder.encode(texts), dtype=np.float32)
        self._example_vectors = {}
        for skill in skills:
            indexes = [index for index, owner in enumerate(owners) if owner == skill.name]
            if indexes:
                self._example_vectors[skill.name] = encoded[indexes]
        self._cache_key = cache_key

    def select(self, query: str, skills: tuple[SkillSpec, ...]) -> SemanticSelection:
        self._prepare(skills)
        query_vector = np.asarray(self.embedder.encode([query])[0], dtype=np.float32)
        candidates: list[SemanticCandidate] = []
        for skill in skills:
            vectors = self._example_vectors.get(skill.name)
            if vectors is None or len(vectors) == 0:
                continue
            similarities = vectors @ query_vector
            best_indexes = np.argsort(similarities)[-MATCHED_EXAMPLE_COUNT:][::-1]
            best_scores = similarities[best_indexes]
            examples = self._example_texts[skill.name]
            candidates.append(
                SemanticCandidate(
                    name=skill.name,
                    score=float(best_scores.mean()),
                    matched_examples=tuple(examples[int(index)] for index in best_indexes),
                )
            )
        candidates.sort(key=lambda candidate: (candidate.score, candidate.name), reverse=True)
        if not candidates:
            return SemanticSelection(None, (), 0.0, 0.0)
        top_score = candidates[0].score
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        margin = top_score - second_score
        accepted_name = (
            candidates[0].name
            if top_score >= self.min_score and margin >= self.min_margin
            else None
        )
        return SemanticSelection(accepted_name, tuple(candidates), top_score, margin)

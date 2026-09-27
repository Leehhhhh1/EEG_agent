"""Score task completion, routing and tool efficiency for EEGAgent traces."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable


TOOL_DEFAULTS: dict[str, dict[str, Any]] = {
    "get_eeg_basic_information": {},
    "explore_eeg_segment": {
        "focus": "overview",
        "channels": None,
    },
    "detect_eeg_events": {
        "start": 0,
        "end": None,
        "event_types": ["seizure"],
        "channels": None,
        "sensitivity": "balanced",
    },
    "generate_eeg_report": {
        "include_basic_information": True,
        "include_exploration": True,
        "include_detection": True,
        "language": "zh-CN",
    },
}


class TraceCollector:
    """Merge incremental trace updates using the event id emitted by the agent."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._indexes: dict[str, int] = {}

    def __call__(self, event: dict[str, Any]) -> None:
        update = dict(event)
        event_id = str(update.get("id") or f"event-{len(self.events) + 1}")
        update["id"] = event_id
        index = self._indexes.get(event_id)
        if index is None:
            update.setdefault("sequence", len(self.events) + 1)
            self._indexes[event_id] = len(self.events)
            self.events.append(update)
            return
        current = self.events[index]
        self.events[index] = {**current, **update, "sequence": current["sequence"]}


def load_cases(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows from one or more files."""
    cases: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return cases


def _parse_trace_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (event for event in events if event.get("type") == "tool/call"),
        key=lambda event: int(event.get("sequence", 0)),
    )


def _route_decision(events: list[dict[str, Any]]) -> tuple[bool, str | None]:
    decisions = [event for event in events if event.get("type") == "routing/decision"]
    if not decisions:
        return False, None
    detail = _parse_trace_json(decisions[-1].get("output"))
    if "skill" not in detail:
        return False, None
    return True, detail["skill"]


def _resolved_arguments(tool_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(TOOL_DEFAULTS.get(tool_name, {}))
    resolved.update({key: value for key, value in raw.items() if key != "session_id"})
    if tool_name == "detect_eeg_events" and resolved.get("event_types") is None:
        resolved["event_types"] = ["seizure"]
    return resolved


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return abs(float(actual) - expected) <= 0.01
    if isinstance(expected, list) and isinstance(actual, list):
        return actual == expected
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(key in actual and _value_matches(actual[key], value) for key, value in expected.items())
    return actual == expected


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    if not expected:
        return True
    cursor = 0
    for item in actual:
        if item == expected[cursor]:
            cursor += 1
            if cursor == len(expected):
                return True
    return False


def _arguments_pass(
    expectations: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    remaining = list(events)
    for expectation in expectations:
        tool_name = str(expectation.get("tool", ""))
        expected = expectation.get("arguments") or {}
        match_index = next(
            (
                index for index, event in enumerate(remaining)
                if event.get("title") == tool_name and event.get("status") == "complete"
            ),
            None,
        )
        if match_index is None:
            reasons.append(f"no successful call available for argument check: {tool_name}")
            continue
        event = remaining.pop(match_index)
        actual = _resolved_arguments(tool_name, _parse_trace_json(event.get("input")))
        mismatches = [
            key for key, value in expected.items()
            if key not in actual or not _value_matches(actual[key], value)
        ]
        if mismatches:
            reasons.append(
                f"argument mismatch for {tool_name}: {', '.join(mismatches)}"
            )
    return not reasons, reasons


def score_turn(
    specification: dict[str, Any],
    *,
    events: list[dict[str, Any]],
    response: str | None,
    run_error: str | None = None,
) -> dict[str, Any]:
    """Score execution and routing separately for one user turn."""
    calls = _tool_events(events)
    route_observed, actual_skill = _route_decision(events)
    acceptable_skills = specification.get("acceptable_skills")
    if acceptable_skills is None:
        acceptable_skills = [specification.get("expected_skill")]
    route_pass = route_observed and actual_skill in acceptable_skills
    actual_tools = [str(event.get("title") or "") for event in calls]
    required_tools = list(specification.get("must_have_tools") or [])
    required_counts = Counter(required_tools)
    successful_counts = Counter(
        str(event.get("title") or "")
        for event in calls
        if event.get("status") == "complete"
    )
    missing_tools = list((required_counts - successful_counts).elements())
    forbidden_tools = set(specification.get("forbidden_tools") or [])
    called_forbidden_tools = [name for name in actual_tools if name in forbidden_tools]
    expected_order = list(specification.get("expected_tool_order") or [])
    completed_tools = [
        str(event.get("title") or "")
        for event in calls
        if event.get("status") == "complete"
    ]
    order_pass = _is_subsequence(expected_order, completed_tools)
    arguments_pass, argument_reasons = _arguments_pass(
        list(specification.get("expected_arguments") or []), calls
    )
    tool_errors = [
        str(event.get("title") or "unknown")
        for event in calls
        if event.get("status") == "error"
    ]
    expected_invalid = (
        specification.get("expected_outcome")
        == "reject_invalid_request_or_surface_tool_error"
    )
    error_policy_pass = expected_invalid or not tool_errors
    response_pass = bool((response or "").strip())

    reasons: list[str] = []
    if run_error:
        reasons.append(f"run error: {run_error}")
    if missing_tools:
        reasons.append(f"missing successful required tools: {', '.join(missing_tools)}")
    if called_forbidden_tools:
        reasons.append(f"forbidden tools called: {', '.join(called_forbidden_tools)}")
    if not order_pass:
        reasons.append(
            "required tool order not observed: " + " -> ".join(expected_order)
        )
    reasons.extend(argument_reasons)
    if not error_policy_pass:
        reasons.append(f"unexpected tool errors: {', '.join(tool_errors)}")
    if not response_pass:
        reasons.append("empty final response")

    passed = not reasons
    return {
        "passed": passed,
        "route_pass": route_pass,
        "route_observed": route_observed,
        "actual_skill": actual_skill,
        "acceptable_skills": acceptable_skills,
        "response_pass": response_pass,
        "required_tools_pass": not missing_tools,
        "forbidden_tools_pass": not called_forbidden_tools,
        "tool_order_pass": order_pass,
        "arguments_pass": arguments_pass,
        "tool_error_policy_pass": error_policy_pass,
        "expected_invalid_request": expected_invalid,
        "actual_tools": actual_tools,
        "successful_tools": completed_tools,
        "tool_errors": tool_errors,
        "missing_tools": missing_tools,
        "called_forbidden_tools": called_forbidden_tools,
        "reasons": reasons,
        "response_checks_pending": list(specification.get("response_checks") or []),
    }


def score_case(case: dict[str, Any], turn_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a complete task; L3 routing requires every turn to be correct."""
    specifications = list(case.get("turns") or [])
    turn_scores = []
    for index, specification in enumerate(specifications):
        run = turn_runs[index] if index < len(turn_runs) else {}
        turn_scores.append(
            score_turn(
                specification,
                events=list(run.get("events") or []),
                response=run.get("response"),
                run_error=run.get("error"),
            )
        )
    missing_turns = max(0, len(specifications) - len(turn_runs))
    passed = not missing_turns and all(score["passed"] for score in turn_scores)
    required_call_count = sum(
        len(specification.get("must_have_tools") or [])
        for specification in specifications
    )
    actual_call_count = sum(
        len(_tool_events(list(run.get("events") or [])))
        for run in turn_runs
    )
    return {
        "case_id": case.get("id"),
        "level": case.get("level"),
        "dataset": case.get("dataset"),
        "passed": passed,
        "route_pass": not missing_turns and all(score["route_pass"] for score in turn_scores),
        "required_tool_calls": required_call_count,
        "actual_tool_calls": actual_call_count,
        "tce": required_call_count / actual_call_count if passed and actual_call_count else None,
        "turn_scores": turn_scores,
        "missing_turns": missing_turns,
    }


def summarize_metrics(case_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate TCR, R-ACC and successful-task TCE by level and overall."""
    def one_group(scores: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(scores)
        completed = sum(bool(score.get("passed")) for score in scores)
        correct_routes = sum(bool(score.get("route_pass")) for score in scores)
        tce_values = [
            score["tce"] for score in scores
            if score.get("passed") and isinstance(score.get("tce"), (int, float))
        ]
        return {
            "completed": completed,
            "correct_routes": correct_routes,
            "total": total,
            "tcr": completed / total if total else None,
            "r_acc": correct_routes / total if total else None,
            "tce": sum(tce_values) / len(tce_values) if tce_values else None,
            "tce_cases": len(tce_values),
        }

    levels = sorted({str(score.get("level")) for score in case_scores})
    datasets = sorted({str(score["dataset"]) for score in case_scores if score.get("dataset")})
    return {
        "overall": one_group(case_scores),
        "by_level": {
            level: one_group([
                score for score in case_scores if str(score.get("level")) == level
            ])
            for level in levels
        },
        "by_dataset": {
            dataset: one_group([score for score in case_scores if score.get("dataset") == dataset])
            for dataset in datasets
        },
        "by_dataset_level": {
            dataset: {
                level: one_group([
                    score for score in case_scores
                    if score.get("dataset") == dataset and str(score.get("level")) == level
                ])
                for level in levels
            }
            for dataset in datasets
        },
        "definitions": {
            "tcr": "Successful task cases / all task cases; structural execution checks only.",
            "r_acc": "Cases with every expected Skill route correct / all task cases.",
            "tce": "Mean(required tool calls / all agent tool calls) across completed cases with tool calls; setup calls excluded.",
        },
        "response_quality": "Natural-language response checks require a separate review.",
    }


def summarize_tcr(case_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Backward-compatible structural TCR view."""
    metrics = summarize_metrics(case_scores)
    return {
        "metric": "structural_tcr",
        "overall": {key: metrics["overall"][key] for key in ("completed", "total", "tcr")},
        "by_level": {
            level: {key: values[key] for key in ("completed", "total", "tcr")}
            for level, values in metrics["by_level"].items()
        },
        "note": "Natural-language response checks remain pending for manual or judge scoring.",
    }

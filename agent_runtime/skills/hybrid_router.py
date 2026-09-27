"""Context-aware semantic-first routing for one EEGAgent user turn.

The model proposes a route and a small execution plan.  This module validates
that proposal before any EEG tool is made available to the active turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Any, Callable

from .models import SemanticSelection, SkillSelection, SkillSpec
from .registry import SkillRegistry


STEP_TO_TOOL = {
    "basic_information": "get_eeg_basic_information",
    "exploration": "explore_eeg_segment",
    "detection": "detect_eeg_events",
    "reporting": "generate_eeg_report",
}
TOOL_TO_STEP = {tool: step for step, tool in STEP_TO_TOOL.items()}
TOOL_ARGUMENTS = {
    "get_eeg_basic_information": frozenset(),
    "explore_eeg_segment": frozenset({"start", "end", "focus", "channels"}),
    "detect_eeg_events": frozenset({"start", "end", "event_types", "channels", "sensitivity"}),
    "generate_eeg_report": frozenset({
        "include_basic_information", "include_exploration", "include_detection", "language",
    }),
}
SPECIALIZED = frozenset(STEP_TO_TOOL)
DECISIONS = frozenset({"route", "clarify", "reject_invalid"})
RELATIONS = frozenset({"new", "modify_previous", "explain_existing", "switch"})

# These patterns decide whether a request needs deeper interpretation; they
# never select a Skill by themselves. A false positive only adds router cost.
COMPLEX_CUES = re.compile(
    r"先.+(?:再|然后|最后)|(?:再|然后|随后|接着|顺便|同时|并且|以及)|"
    r"(?:改成|改为|改查|改分析|不是|其他条件不变|时间不变|仍然|保持不变)|"
    r"(?:刚才|上次|之前|前面|那个时间段|这段|那段|已有|已经|不要重新|不再)|"
    r"(?:那就|那么|这段呢|继续吧)|"
    r"(?:什么是|如何定义|只解释|解释概念|解释一下|是什么意思|为什么|原理|如何计算)",
    re.IGNORECASE,
)
ACTION_CUES = re.compile(
    r"(?:分析|探索|检查|检测|查找|查询|查看|显示|告诉|生成|汇总|报告|发作|"
    r"放电|背景节律|振幅|对称|采样率|通道|蒙太奇|时长)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlannedCall:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutePlan:
    selection: SkillSelection
    decision: str = "route"
    relation: str = "new"
    calls: tuple[PlannedCall, ...] = ()
    reference_turn: int | None = None
    note: str = ""

    @property
    def skill(self) -> SkillSpec | None:
        return self.selection.skill

    def instruction(self) -> str:
        if self.decision == "clarify":
            return f"本轮路由信息不足：{self.note}。不要调用 EEG 工具；请向用户提出一个具体的澄清问题。"
        if self.decision == "reject_invalid":
            return f"本轮请求参数无效：{self.note}。不要调用 EEG 工具；请解释限制并要求用户更正。"
        if not self.calls:
            if self.relation == "explain_existing":
                return "只解释当前会话中已有的结果，不要重新调用 EEG 分析工具。"
            return "本轮没有已校验的工具步骤；不要自行发起 EEG 工具调用。"
        numbered = [
            f"{index}. {call.tool}({call.arguments})"
            for index, call in enumerate(self.calls, start=1)
        ]
        return (
            "按以下已校验的步骤和顺序完成当前请求；参数不得擅自改变。"
            "若工具失败，只能如实说明或在同一步骤重试，不得编造结果。\n"
            + "\n".join(numbered)
        )


def is_simple_request(query: str) -> bool:
    """Only clearly single-goal requests may use the semantic fast path."""
    stripped = query.strip()
    return bool(stripped and ACTION_CUES.search(stripped) and not COMPLEX_CUES.search(stripped))


def _fallback(reason: str, semantic: SemanticSelection | None = None) -> RoutePlan:
    return RoutePlan(
        selection=SkillSelection(
            skill=None,
            source="llm_clarify",
            candidates=semantic.candidates if semantic else (),
            top_score=semantic.top_score if semantic else 0.0,
            margin=semantic.margin if semantic else 0.0,
        ),
        decision="clarify",
        note=reason,
    )


def _normalize_tool(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    if name in TOOL_ARGUMENTS:
        return name
    return STEP_TO_TOOL.get(name)


def _prior_arguments(
    history: list[dict[str, Any]], reference_turn: int, tool: str,
) -> dict[str, Any] | None:
    previous = next((item for item in history if item.get("turn") == reference_turn), None)
    if previous is None:
        return None
    for call in reversed(previous.get("tool_calls") or []):
        if call.get("name") == tool and isinstance(call.get("arguments"), dict):
            return dict(call["arguments"])
    return None


def _check_arguments(
    calls: tuple[PlannedCall, ...], duration: float | None,
    available_channels: tuple[str, ...] = (),
) -> str | None:
    for call in calls:
        arguments = call.arguments
        if set(arguments) - TOOL_ARGUMENTS[call.tool]:
            return f"{call.tool} 包含不支持的参数"
        if call.tool in {"explore_eeg_segment", "detect_eeg_events"}:
            start, end = arguments.get("start"), arguments.get("end")
            if (
                not isinstance(start, (int, float)) or isinstance(start, bool)
                or not isinstance(end, (int, float)) or isinstance(end, bool)
                or start < 0 or end <= start
            ):
                return "分析时间窗口缺失或无效"
            maximum = 60 if call.tool == "explore_eeg_segment" else 600
            if end - start > maximum:
                return f"分析窗口不能超过 {maximum} 秒"
            if duration is not None and end > duration:
                return "分析窗口超过当前 EEG 记录时长"
            channels = arguments.get("channels")
            if channels is not None and (
                not isinstance(channels, list)
                or not channels
                or any(not isinstance(channel, str) or not channel for channel in channels)
            ):
                return "导联参数无效"
            if channels and available_channels and any(
                channel not in available_channels for channel in channels
            ):
                return "请求的导联不在当前 EEG 记录的可用双极导联中"
        if call.tool == "explore_eeg_segment" and arguments.get("focus", "overview") not in {
            "overview", "background_rhythm", "amplitude", "symmetry", "abnormality_screen",
        }:
            return "探索重点无效"
        if call.tool == "detect_eeg_events":
            if arguments.get("sensitivity", "balanced") not in {"balanced", "sensitive", "specific"}:
                return "检测灵敏度无效"
            event_types = arguments.get("event_types")
            if event_types is not None and event_types != ["seizure"]:
                return "当前检测工具只支持 seizure 事件类型"
        if call.tool == "generate_eeg_report":
            for key in ("include_basic_information", "include_exploration", "include_detection"):
                if key in arguments and not isinstance(arguments[key], bool):
                    return "报告内容开关必须为布尔值"
            if "language" in arguments and arguments["language"] not in {"zh-CN", "en"}:
                return "报告语言必须是 zh-CN 或 en"
    return None


class HybridSkillRouter:
    """Use a conservative semantic fast path and an LLM for complex turns."""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.fast_min_score = float(os.getenv("SKILL_ROUTE_FAST_MIN_SCORE", "0.75"))
        self.fast_min_margin = float(os.getenv("SKILL_ROUTE_FAST_MIN_MARGIN", "0.10"))

    def select(
        self,
        query: str,
        *,
        semantic_selector: Callable[[str, tuple[SkillSpec, ...]], SemanticSelection],
        llm_selector: Callable[[str, list[dict[str, Any]], SemanticSelection | None], dict[str, Any]],
        history: list[dict[str, Any]],
        duration: float | None,
        available_channels: tuple[str, ...] = (),
    ) -> RoutePlan:
        specialized = tuple(skill for skill in self.registry.all() if skill.name != "general_eeg")
        try:
            semantic = semantic_selector(query, specialized)
        except Exception:
            semantic = None
        if (
            semantic is not None
            and is_simple_request(query)
            and semantic.candidates
            and semantic.top_score >= self.fast_min_score
            and semantic.margin >= self.fast_min_margin
        ):
            skill = self.registry.get(semantic.candidates[0].name)
            return RoutePlan(selection=SkillSelection(
                skill=skill,
                source="embedding",
                candidates=semantic.candidates,
                top_score=semantic.top_score,
                margin=semantic.margin,
            ))

        try:
            proposal = llm_selector(query, history, semantic)
        except Exception:
            return _fallback("路由模型不可用，请明确本轮分析目标", semantic)
        if not isinstance(proposal, dict):
            return _fallback("路由结果格式无效", semantic)
        decision = proposal.get("decision")
        if decision not in DECISIONS:
            return _fallback("路由结果缺少有效决策", semantic)
        if decision == "clarify":
            return _fallback(str(proposal.get("clarification") or "请求目标不明确"), semantic)
        skill_name = proposal.get("skill")
        if not isinstance(skill_name, str) or skill_name not in {skill.name for skill in self.registry.all()}:
            return _fallback("路由模型选择了未知 Skill", semantic)
        skill = self.registry.get(skill_name)
        relation = proposal.get("relation", "new")
        if relation not in RELATIONS:
            return _fallback("请求与前文的关系不明确", semantic)
        reference_turn = proposal.get("reference_turn")
        if relation in {"modify_previous", "explain_existing"}:
            if not isinstance(reference_turn, int) or isinstance(reference_turn, bool):
                return _fallback("无法确定需要引用哪一轮任务", semantic)
            previous = next((item for item in history if item.get("turn") == reference_turn), None)
            if previous is None or previous.get("skill") != skill_name:
                return _fallback("引用的旧任务不属于当前 EEG 会话或 Skill 不一致", semantic)
        elif reference_turn is not None:
            return _fallback("新任务不应继承旧任务参数", semantic)

        raw_steps = proposal.get("steps", [])
        if not isinstance(raw_steps, list):
            return _fallback("工具步骤格式无效", semantic)
        changes = proposal.get("parameter_changes") or {}
        if not isinstance(changes, dict):
            return _fallback("更正参数格式无效", semantic)
        calls: list[PlannedCall] = []
        for raw_step in raw_steps:
            if isinstance(raw_step, str):
                tool = _normalize_tool(raw_step)
                arguments = {}
            elif isinstance(raw_step, dict):
                tool = _normalize_tool(raw_step.get("tool") or raw_step.get("action"))
                arguments = raw_step.get("arguments") or {}
            else:
                return _fallback("工具步骤无效", semantic)
            if tool is None or tool not in skill.allowed_tools or not isinstance(arguments, dict):
                return _fallback("工具步骤不属于目标 Skill 的权限范围", semantic)
            arguments = dict(arguments)
            if relation == "modify_previous" and reference_turn is not None:
                prior = _prior_arguments(history, reference_turn, tool)
                if prior is None:
                    return _fallback("找不到上一任务的工具参数，无法安全继承", semantic)
                arguments = {**prior, **arguments, **changes}
            elif changes:
                return _fallback("新任务不应携带旧任务参数更正", semantic)
            arguments.pop("session_id", None)
            calls.append(PlannedCall(tool, arguments))
        if skill_name == "general_eeg" and calls:
            return _fallback("一般知识请求不得调用 EEG 分析工具", semantic)
        if relation == "explain_existing" and calls:
            return _fallback("解释已有结果不应重新调用分析工具", semantic)
        if decision == "route" and relation not in {"explain_existing"} and skill_name in SPECIALIZED and not calls:
            return _fallback("专用 Skill 的执行步骤缺失", semantic)
        call_tuple = tuple(calls)
        invalid = _check_arguments(call_tuple, duration, available_channels)
        if invalid and decision == "route":
            decision, call_tuple = "reject_invalid", ()
        elif invalid or decision == "reject_invalid":
            call_tuple = ()
        selection = SkillSelection(
            skill=skill,
            source="llm",
            candidates=semantic.candidates if semantic else (),
            top_score=semantic.top_score if semantic else 0.0,
            margin=semantic.margin if semantic else 0.0,
        )
        return RoutePlan(
            selection=selection,
            decision=decision,
            relation=relation,
            calls=call_tuple,
            reference_turn=reference_turn,
            note=invalid or str(proposal.get("note") or ""),
        )

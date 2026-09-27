"""Focused code tests for hybrid routing; these do not analyze EDF files."""

import json
import sys
import types
import unittest

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from agent_runtime.mcp_chat_agent import MCPChatAgent
from agent_runtime.skills import SkillRegistry
from agent_runtime.skills.hybrid_router import HybridSkillRouter, PlannedCall, RoutePlan
from agent_runtime.skills.models import SemanticCandidate, SemanticSelection, SkillSelection


def semantic(name="basic_information", score=0.91, margin=0.21):
    return SemanticSelection(name, (SemanticCandidate(name, score, ("example",)),), score, margin)


class HybridRouterTests(unittest.TestCase):
    def setUp(self):
        self.registry = SkillRegistry.load_default()
        self.router = HybridSkillRouter(self.registry)

    def select(self, query, proposal, *, history=None, semantic_result=None, duration=1000,
               available_channels=()):
        return self.router.select(
            query,
            semantic_selector=lambda _query, _skills: semantic_result or semantic(),
            llm_selector=lambda _query, _history, _semantic: proposal,
            history=history or [],
            duration=duration,
            available_channels=available_channels,
        )

    def test_clear_single_goal_uses_semantic_fast_path(self):
        plan = self.router.select(
            "查询通道数",
            semantic_selector=lambda _query, _skills: semantic(),
            llm_selector=lambda *_args: self.fail("simple request should not call the LLM"),
            history=[],
            duration=1000,
        )
        self.assertEqual(plan.skill.name, "basic_information")
        self.assertEqual(plan.selection.source, "embedding")

    def test_multistep_uses_final_goal_and_ordered_steps_even_with_high_score(self):
        plan = self.select("先查通道数，再分析前30秒左右对称性", {
            "decision": "route", "skill": "exploration", "relation": "new",
            "reference_turn": None,
            "steps": [
                {"tool": "get_eeg_basic_information", "arguments": {}},
                {"tool": "explore_eeg_segment", "arguments": {"start": 0, "end": 30, "focus": "symmetry"}},
            ],
        })
        self.assertEqual(plan.skill.name, "exploration")
        self.assertEqual([call.tool for call in plan.calls], [
            "get_eeg_basic_information", "explore_eeg_segment",
        ])
        self.assertEqual(plan.selection.source, "llm")

    def test_correction_inherits_only_prior_turn_parameters(self):
        history = [{
            "turn": 3, "skill": "detection",
            "tool_calls": [{"name": "detect_eeg_events", "arguments": {
                "start": 0, "end": 60, "sensitivity": "sensitive", "channels": ["Fp1-F3"],
            }}],
        }]
        plan = self.select("不是前一分钟，改查120–180秒，其他条件不变", {
            "decision": "route", "skill": "detection", "relation": "modify_previous",
            "reference_turn": 3,
            "steps": [{"tool": "detect_eeg_events", "arguments": {"start": 120, "end": 180}}],
        }, history=history)
        self.assertEqual(plan.calls[0].arguments, {
            "start": 120, "end": 180, "sensitivity": "sensitive", "channels": ["Fp1-F3"],
        })

    def test_correction_without_same_session_reference_clarifies(self):
        plan = self.select("改成120–180秒", {
            "decision": "route", "skill": "detection", "relation": "modify_previous",
            "reference_turn": 3,
            "steps": [{"tool": "detect_eeg_events", "arguments": {"start": 120, "end": 180}}],
        })
        self.assertEqual(plan.decision, "clarify")
        self.assertIsNone(plan.skill)

    def test_invalid_window_cannot_be_executed(self):
        plan = self.select("检查发作", {
            "decision": "route", "skill": "detection", "relation": "new",
            "reference_turn": None,
            "steps": [{"tool": "detect_eeg_events", "arguments": {"start": 0, "end": 700}}],
        }, semantic_result=semantic("detection", 0.50, 0.01))
        self.assertEqual(plan.decision, "reject_invalid")
        self.assertEqual(plan.calls, ())

    def test_unknown_channel_cannot_be_executed(self):
        plan = self.select("检测前60秒 Fp1-F3 是否存在发作", {
            "decision": "route", "skill": "detection", "relation": "new",
            "reference_turn": None,
            "steps": [{"tool": "detect_eeg_events", "arguments": {
                "start": 0, "end": 60, "channels": ["not-a-channel"],
            }}],
        }, semantic_result=semantic("detection", 0.50, 0.01),
            available_channels=("Fp1-F3",))
        self.assertEqual(plan.decision, "reject_invalid")
        self.assertEqual(plan.calls, ())

    def test_unauthorized_step_and_general_knowledge(self):
        bad = self.select("先查通道，再生成报告", {
            "decision": "route", "skill": "reporting", "relation": "new",
            "reference_turn": None,
            "steps": [{"tool": "explore_eeg_segment", "arguments": {"start": 0, "end": 30}}],
        })
        self.assertEqual(bad.decision, "clarify")
        knowledge = self.select("什么是脑电采样率？", {
            "decision": "route", "skill": "general_eeg", "relation": "new",
            "reference_turn": None, "steps": [],
        })
        self.assertEqual(knowledge.skill.name, "general_eeg")
        self.assertFalse(knowledge.calls)


class PlanExecutionTests(unittest.TestCase):
    def test_llm_router_receives_filtered_history_not_full_messages(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps({
                    "decision": "route", "skill": "general_eeg", "relation": "new",
                    "reference_turn": None, "steps": [],
                })))],
                usage=None,
            )

        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create),
        ))
        agent.model = "fake-model"
        agent.session_id = "private-session-id"
        agent.session_summary = {"recording": {"duration_seconds": 300}}
        agent.skill_registry = SkillRegistry.load_default()
        agent.messages = [{"role": "tool", "content": "large raw EEG tool payload"}]
        agent._route_with_llm("什么是采样率", [{"turn": 1, "user": "查通道数", "skill": "basic_information"}], None)

        prompt = captured["messages"][1]["content"]
        self.assertIn("查通道数", prompt)
        self.assertNotIn("large raw EEG tool payload", prompt)
        self.assertNotIn("private-session-id", prompt)

    def test_wrong_tool_order_is_blocked_before_bridge(self):
        registry = SkillRegistry.load_default()
        skill = registry.get("exploration")
        plan = RoutePlan(
            SkillSelection(skill, "llm"),
            calls=(
                PlannedCall("get_eeg_basic_information", {}),
                PlannedCall("explore_eeg_segment", {"start": 0, "end": 30, "focus": "symmetry"}),
            ),
        )

        class Bridge:
            def __init__(self):
                self.called = []

            def list_tools(self):
                return []

            def call_tool(self, name, arguments):
                self.called.append((name, arguments))
                return {"is_error": False, "structured_content": {}, "content": []}

        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.bridge = Bridge()
        agent.session_id = "session-1"
        agent.messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "query"}]
        completions = iter([
            ("", "", [{"id": "call-1", "name": "explore_eeg_segment", "arguments": json.dumps({"start": 0, "end": 30, "focus": "symmetry"})}]),
            ("done", "", []),
        ])
        agent._stream_completion = types.MethodType(
            lambda self, _tools, on_delta=None: next(completions), agent,
        )
        result = agent._run_stream_with_temporary_context("query", [], skill, route_plan=plan)
        self.assertEqual(agent.bridge.called, [])
        self.assertFalse(result["routing"]["plan_complete"])
        self.assertIn("get_eeg_basic_information", result["response"])


if __name__ == "__main__":
    unittest.main()

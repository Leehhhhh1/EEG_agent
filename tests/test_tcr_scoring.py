import unittest

from eval.agent_benchmark.tcr import (
    TraceCollector,
    score_case,
    score_turn,
    summarize_metrics,
    summarize_tcr,
)
from eval.agent_benchmark.run_tcr import _pair_cases


def tool_event(name, status="complete", arguments=None, sequence=1):
    import json

    return {
        "id": f"tool-{sequence}",
        "type": "tool/call",
        "title": name,
        "status": status,
        "sequence": sequence,
        "input": json.dumps(arguments or {}),
    }


def route_event(skill, sequence=1):
    import json

    return {
        "id": f"route-{sequence}",
        "type": "routing/decision",
        "status": "complete",
        "sequence": sequence,
        "output": json.dumps({"skill": skill}),
    }


class TraceCollectorTests(unittest.TestCase):
    def test_incremental_updates_are_merged(self):
        collector = TraceCollector()
        collector({"id": "call-1", "type": "tool/call", "status": "running"})
        collector({"id": "call-1", "status": "complete", "output": "ok"})

        self.assertEqual(len(collector.events), 1)
        self.assertEqual(collector.events[0]["status"], "complete")
        self.assertEqual(collector.events[0]["output"], "ok")


class TCRScoringTests(unittest.TestCase):
    def test_paired_cases_cover_both_eeg_files_and_adapt_long_window(self):
        original = [{
            "id": "L2-001", "level": "L2",
            "session": {"mode": "loaded", "file": "data/aaaaabdo_s003_t000.edf"},
            "turns": [{
                "query": "前10分钟",
                "expected_arguments": [{
                    "tool": "detect_eeg_events", "arguments": {"end": 600},
                }],
            }],
        }]

        paired = _pair_cases(original)

        self.assertEqual(len(paired), 2)
        self.assertEqual(paired[0]["id"], "L2-001@GPED")
        self.assertEqual(paired[0]["turns"][0]["expected_arguments"][0]["arguments"]["end"], 120)
        self.assertEqual(paired[1]["id"], "L2-001@ABDO")
        self.assertEqual(paired[1]["turns"][0]["expected_arguments"][0]["arguments"]["end"], 600)
        self.assertEqual(original[0]["turns"][0]["expected_arguments"][0]["arguments"]["end"], 600)

    def test_complete_tool_chain_passes(self):
        specification = {
            "must_have_tools": ["get_eeg_basic_information", "detect_eeg_events"],
            "expected_tool_order": ["get_eeg_basic_information", "detect_eeg_events"],
            "expected_arguments": [{
                "tool": "detect_eeg_events",
                "arguments": {"start": 0, "end": 60, "sensitivity": "balanced"},
            }],
        }
        events = [
            tool_event("get_eeg_basic_information", sequence=1),
            tool_event("detect_eeg_events", arguments={"start": 0, "end": 60}, sequence=2),
        ]

        score = score_turn(specification, events=events, response="完成")

        self.assertTrue(score["passed"])

    def test_missing_required_tool_fails(self):
        specification = {
            "must_have_tools": ["get_eeg_basic_information", "detect_eeg_events"],
            "expected_tool_order": ["get_eeg_basic_information", "detect_eeg_events"],
        }

        score = score_turn(
            specification,
            events=[tool_event("detect_eeg_events")],
            response="完成",
        )

        self.assertFalse(score["passed"])
        self.assertEqual(score["missing_tools"], ["get_eeg_basic_information"])

    def test_extra_successful_tool_does_not_reduce_tcr(self):
        specification = {
            "must_have_tools": ["detect_eeg_events"],
            "expected_tool_order": ["detect_eeg_events"],
        }
        events = [
            tool_event("get_eeg_basic_information", sequence=1),
            tool_event("detect_eeg_events", sequence=2),
        ]

        score = score_turn(specification, events=events, response="完成")

        self.assertTrue(score["passed"])

    def test_forbidden_tool_fails_tcr(self):
        specification = {
            "must_have_tools": ["detect_eeg_events"],
            "forbidden_tools": ["explore_eeg_segment"],
        }
        score = score_turn(
            specification,
            events=[
                tool_event("explore_eeg_segment", sequence=1),
                tool_event("detect_eeg_events", sequence=2),
            ],
            response="完成",
        )
        self.assertFalse(score["passed"])
        self.assertFalse(score["forbidden_tools_pass"])

    def test_unexpected_tool_error_fails(self):
        specification = {
            "must_have_tools": ["detect_eeg_events"],
            "expected_tool_order": ["detect_eeg_events"],
        }

        score = score_turn(
            specification,
            events=[tool_event("detect_eeg_events", status="error")],
            response="失败",
        )

        self.assertFalse(score["passed"])
        self.assertFalse(score["tool_error_policy_pass"])

    def test_expected_invalid_request_can_pass_without_tool_call(self):
        specification = {
            "expected_outcome": "reject_invalid_request_or_surface_tool_error",
            "must_have_tools": [],
            "expected_tool_order": [],
        }

        score = score_turn(specification, events=[], response="时间范围无效，请修改。")

        self.assertTrue(score["passed"])

    def test_multiturn_case_requires_every_turn(self):
        case = {
            "id": "L3-test",
            "level": "L3",
            "turns": [
                {"must_have_tools": [], "expected_tool_order": []},
                {"must_have_tools": ["detect_eeg_events"], "expected_tool_order": ["detect_eeg_events"]},
            ],
        }
        runs = [
            {"response": "第一轮", "events": []},
            {"response": "第二轮", "events": []},
        ]

        score = score_case(case, runs)

        self.assertFalse(score["passed"])

    def test_case_scores_route_and_tool_efficiency_separately(self):
        case = {
            "id": "L2-test",
            "level": "L2",
            "turns": [{
                "expected_skill": "detection",
                "must_have_tools": ["get_eeg_basic_information", "detect_eeg_events"],
                "expected_tool_order": ["get_eeg_basic_information", "detect_eeg_events"],
            }],
        }
        runs = [{
            "response": "完成",
            "events": [
                route_event("basic_information"),
                tool_event("get_eeg_basic_information", sequence=2),
                tool_event("explore_eeg_segment", sequence=3),
                tool_event("detect_eeg_events", sequence=4),
            ],
        }]

        score = score_case(case, runs)

        self.assertTrue(score["passed"])
        self.assertFalse(score["route_pass"])
        self.assertAlmostEqual(score["tce"], 2 / 3)

    def test_multiturn_route_requires_every_turn(self):
        case = {
            "id": "L3-route",
            "level": "L3",
            "turns": [
                {"expected_skill": "exploration", "must_have_tools": []},
                {"expected_skill": "detection", "must_have_tools": []},
            ],
        }
        runs = [
            {"response": "第一轮", "events": [route_event("exploration")]},
            {"response": "第二轮", "events": [route_event("reporting")]},
        ]

        score = score_case(case, runs)

        self.assertTrue(score["passed"])
        self.assertFalse(score["route_pass"])
        self.assertIsNone(score["tce"])

    def test_summary_reports_all_metrics_by_level(self):
        scores = [
            {"level": "L1", "passed": True, "route_pass": True, "tce": 1.0},
            {"level": "L1", "passed": False, "route_pass": True, "tce": None},
            {"level": "L2", "passed": False, "route_pass": False, "tce": None},
        ]

        summary = summarize_metrics(scores)

        self.assertEqual(summary["by_level"]["L1"]["tcr"], 0.5)
        self.assertEqual(summary["by_level"]["L1"]["r_acc"], 1.0)
        self.assertEqual(summary["by_level"]["L1"]["tce"], 1.0)
        self.assertEqual(summary["by_level"]["L2"]["tce_cases"], 0)
        self.assertIsNone(summary["by_level"]["L2"]["tce"])

    def test_summary_reports_overall_and_levels(self):
        scores = [
            {"level": "L1", "passed": True},
            {"level": "L1", "passed": False},
            {"level": "L2", "passed": True},
        ]

        summary = summarize_tcr(scores)

        self.assertAlmostEqual(summary["overall"]["tcr"], 2 / 3)
        self.assertEqual(summary["by_level"]["L1"]["completed"], 1)
        self.assertEqual(summary["by_level"]["L2"]["total"], 1)


if __name__ == "__main__":
    unittest.main()

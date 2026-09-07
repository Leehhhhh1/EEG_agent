import unittest

from PySide6.QtCore import QRectF

from trajectory_view import (
    KIND_LABELS,
    _hit_test_bar,
    _sequence_bar_layout,
    summarize_trajectory,
)


class TrajectoryMetricsTests(unittest.TestCase):
    def test_timeline_hit_test_selects_clicked_step(self):
        targets = [
            (QRectF(10, 20, 30, 14), "step-1"),
            (QRectF(42, 20, 30, 14), "step-2"),
        ]

        self.assertEqual(_hit_test_bar(targets, 25, 27), "step-1")
        self.assertEqual(_hit_test_bar(targets, 60, 27), "step-2")
        self.assertIsNone(_hit_test_bar(targets, 90, 60))

    def test_timeline_bars_fill_axis_and_scale_with_duration(self):
        layout = _sequence_bar_layout([1.0, 4.0, 9.0], 300)
        widths = [width for _offset, width in layout]

        self.assertLess(widths[0], widths[1])
        self.assertLess(widths[1], widths[2])
        self.assertEqual(layout[0][0], 0)
        self.assertEqual(layout[-1][0] + layout[-1][1], 300)
        self.assertEqual(layout[1][0] - widths[0], 2)

    def test_visible_event_types_are_limited_to_four_labels(self):
        self.assertEqual(set(KIND_LABELS.values()), {"用户", "上下文", "助手", "工具"})

    def test_summary_aggregates_model_tool_and_usage_metrics(self):
        events = [
            {
                "id": "user",
                "type": "user/message",
                "kind": "user",
                "turn": 1,
                "status": "complete",
                "started_at": 100.0,
                "ended_at": 100.0,
            },
            {
                "id": "model",
                "type": "model/request",
                "kind": "assistant",
                "turn": 1,
                "status": "complete",
                "started_at": 101.0,
                "ended_at": 106.0,
                "duration": 5.0,
                "ttft": 1.0,
                "prompt_tokens": 1000,
                "context_limit_tokens": 32768,
                "completion_tokens": 200,
                "cache_hit_tokens": 900,
                "cache_miss_tokens": 100,
            },
            {
                "id": "tool",
                "type": "tool/call",
                "kind": "tool",
                "turn": 1,
                "status": "complete",
                "started_at": 106.0,
                "ended_at": 108.5,
                "duration": 2.5,
            },
            {"id": "end", "type": "turn/end", "kind": "context", "turn": 1},
        ]

        summary = summarize_trajectory(events, now=109.0)

        self.assertIn("1 轮 · 3 步", summary)
        self.assertIn("LLM 5.0 秒 · 工具调用 2.5 秒", summary)
        self.assertIn("首 token 平均 1.0 秒 · 50 tok/s", summary)
        self.assertIn("本轮缓存命中 90%", summary)
        self.assertIn("输入 1.0K tok · 输出 200 tok", summary)
        self.assertIn("上下文 1.0K / 32.8K tok（3%）", summary)

    def test_summary_cache_rate_uses_latest_turn_only(self):
        events = [
            {
                "id": "turn-1-model",
                "type": "model/request",
                "kind": "assistant",
                "turn": 1,
                "cache_hit_tokens": 900,
                "cache_miss_tokens": 100,
            },
            {
                "id": "turn-2-model-1",
                "type": "model/request",
                "kind": "assistant",
                "turn": 2,
                "cache_hit_tokens": 100,
                "cache_miss_tokens": 900,
            },
            {
                "id": "turn-2-model-2",
                "type": "model/request",
                "kind": "assistant",
                "turn": 2,
                "cache_hit_tokens": 500,
                "cache_miss_tokens": 500,
            },
        ]

        summary = summarize_trajectory(events, now=109.0)

        self.assertIn("本轮缓存命中 30%", summary)
        self.assertNotIn("缓存命中 50%", summary)


if __name__ == "__main__":
    unittest.main()

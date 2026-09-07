import unittest

from eeg_skills.event_utils import merge_contiguous_channel_events


def _candidate(channel, start, confidence):
    return {
        "event_type": "seizure_like_activity",
        "start_seconds": start,
        "end_seconds": start + 1,
        "channel": channel,
        "brain_region": "region",
        "confidence": confidence,
        "evidence": {
            "seizure_vs_nonseizure": confidence,
            "fine_windows": 1,
        },
    }


class DetectionMergeTests(unittest.TestCase):
    def test_interleaved_channels_merge_independently(self):
        candidates = [
            _candidate("A", 0, 0.80),
            _candidate("B", 0, 0.75),
            _candidate("A", 1, 0.95),
            _candidate("B", 1, 0.85),
            _candidate("A", 3, 0.70),
        ]

        events = merge_contiguous_channel_events(candidates)

        self.assertEqual(len(events), 3)
        first_a = next(event for event in events if event["channel"] == "A" and event["start_seconds"] == 0)
        first_b = next(event for event in events if event["channel"] == "B")
        self.assertEqual(first_a["end_seconds"], 2)
        self.assertEqual(first_a["confidence"], 0.95)
        self.assertEqual(first_a["evidence"]["fine_windows"], 2)
        self.assertEqual(first_b["end_seconds"], 2)
        self.assertEqual(first_b["evidence"]["fine_windows"], 2)
        self.assertEqual(candidates[0]["end_seconds"], 1)


if __name__ == "__main__":
    unittest.main()

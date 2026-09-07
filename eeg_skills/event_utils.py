"""Lightweight event post-processing helpers."""

from __future__ import annotations

from typing import Any


def merge_contiguous_channel_events(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge contiguous events independently for each channel."""
    events_by_channel: dict[str, list[dict[str, Any]]] = {}
    ordered = sorted(
        candidates,
        key=lambda item: (
            str(item.get("channel", "")),
            float(item.get("start_seconds", 0)),
            float(item.get("end_seconds", 0)),
        ),
    )
    for candidate in ordered:
        channel = str(candidate.get("channel", ""))
        channel_events = events_by_channel.setdefault(channel, [])
        event = dict(candidate)
        event["evidence"] = dict(candidate.get("evidence", {}))
        if (
            channel_events
            and channel_events[-1].get("event_type") == event.get("event_type")
            and abs(
                float(channel_events[-1]["end_seconds"])
                - float(event["start_seconds"])
            ) < 1e-6
        ):
            previous = channel_events[-1]
            previous_windows = int(previous.get("evidence", {}).get("fine_windows", 1))
            current_windows = int(event.get("evidence", {}).get("fine_windows", 1))
            if float(event.get("confidence", 0)) > float(previous.get("confidence", 0)):
                best_evidence = dict(event.get("evidence", {}))
                previous["confidence"] = event.get("confidence")
                previous["evidence"] = best_evidence
            previous["end_seconds"] = event["end_seconds"]
            previous.setdefault("evidence", {})["fine_windows"] = previous_windows + current_windows
        else:
            channel_events.append(event)

    return sorted(
        (event for channel_events in events_by_channel.values() for event in channel_events),
        key=lambda item: (
            float(item.get("start_seconds", 0)),
            float(item.get("end_seconds", 0)),
            str(item.get("channel", "")),
        ),
    )

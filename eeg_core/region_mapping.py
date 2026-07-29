"""Deterministic bipolar-channel to coarse brain-region mapping."""

CHANNEL_REGIONS = {
    "FP1-F7": "left prefrontal/frontal", "F7-T3": "left temporal", "T3-T5": "left temporal", "T5-O1": "left temporal/occipital",
    "FP2-F8": "right prefrontal/frontal", "F8-T4": "right temporal", "T4-T6": "right temporal", "T6-O2": "right temporal/occipital",
    "A1-T3": "left temporal", "T3-C3": "left temporal/central", "C3-CZ": "left central",
    "CZ-C4": "right central", "C4-T4": "right temporal/central", "T4-A2": "right temporal",
    "FP1-F3": "left prefrontal/frontal", "F3-C3": "left frontal/central", "C3-P3": "left central/parietal", "P3-O1": "left parietal/occipital",
    "FP2-F4": "right prefrontal/frontal", "F4-C4": "right frontal/central", "C4-P4": "right central/parietal", "P4-O2": "right parietal/occipital",
}


def region_for_channel(channel: str) -> str:
    return CHANNEL_REGIONS.get(channel, "unmapped")

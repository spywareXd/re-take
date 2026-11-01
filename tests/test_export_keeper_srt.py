"""Keeper .srt export: original timestamps, discarded takes removed, failed groups discard nothing."""
from pipeline.export_keeper_srt import build_srt, srt_time


def U(uid, start, end, text):
    return {"unit_id": uid, "start": start, "end": end, "raw_text": text}


UNITS = [U("1:0", 0.0, 2.0, "Intro line."), U("2:0", 3.0, 4.5, "Take one"), U("3:0", 5.0, 7.0, "Take two, better."), U("4:0", 9.0, 10.0, "Outro."),
         U("5:0", 12.0, 13.0, "Other one"), U("6:0", 14.0, 15.0, "Other two")]
OK = {"adjudication_status": "resolved", "is_genuine_retake_group": True, "unit_ids": ["2:0", "3:0"], "keeper_unit_id": "3:0", "units_to_remove": ["2:0"]}
FAILED = {"adjudication_status": "failed", "is_genuine_retake_group": True, "unit_ids": ["5:0", "6:0"], "keeper_unit_id": None, "units_to_remove": None}


def test_time_format():
    assert srt_time(0) == "00:00:00,000" and srt_time(3661.5) == "01:01:01,500"


def test_discarded_takes_are_dropped_and_times_are_original():
    text, gone, n = build_srt(UNITS, [OK, FAILED])
    assert gone == ["2:0"] and n == 5
    assert "Take one" not in text and "Take two, better." in text
    assert "00:00:05,000 --> 00:00:07,000\nTake two, better." in text           # not shifted: the timeline keeps its gaps
    assert "Other one" in text and "Other two" in text                          # a failed group discards nothing


def test_only_groups_keeps_just_the_group_keepers():
    text, gone, n = build_srt(UNITS, [OK], only_groups=True)
    assert n == 1 and "Take two, better." in text and "Intro line." not in text


def test_keep_none_group_drops_every_take_and_cues_never_overlap():
    none = {"adjudication_status": "resolved", "is_genuine_retake_group": True, "unit_ids": ["5:0", "6:0"], "keeper_unit_id": None, "units_to_remove": ["5:0", "6:0"]}
    text, gone, n = build_srt([U("1:0", 0, 5, "a"), U("2:0", 3, 6, "b")] + UNITS[4:], [none])
    assert gone == ["5:0", "6:0"] and n == 2 and "00:00:00,000 --> 00:00:03,000" in text

import json

from pipeline.step_06_llm_adjudication.keeper_prompt import KEEPER_RULES, build_keeper_prompt
from training.build_keeper_sft import clean_reason, group_units, target_for

ROW = {"group_id": "v__g1", "video_id": "v", "unit_ids": "2:0|2:1|3:0", "texts": "[1] So today, || [2] so today we talk. || [3] So today we talk about editing.",
       "starts": "1.0|4.0|9.0", "ends": "2.0|6.0|12.5", "keeper_unit_ids": "3:0", "keep_none": "", "notes": ""}


def test_prompt_has_the_keeper_rules_first_then_the_takes():
    units = group_units(ROW)
    assert units[0]["raw_text"] == "So today," and units[2]["end"] == 12.5                  # "[k] " prefixes are stripped
    p = build_keeper_prompt("v__g1", units)
    assert p.startswith("You are the keeper selector for a video retake remover.") and KEEPER_RULES in p
    for rule in ("1. Drop takes that are truncated", "7. If units are spoken cue phrases", "9. Judge the text only", "Be conservative about deletion"):
        assert rule in p
    assert 'Unit ids: ["2:0", "2:1", "3:0"]' in p and '[3:0] (9.00s -> 12.50s): "So today we talk about editing."' in p
    assert p.index("HOW TO CHOOSE") < p.index("TAKES (IN CHRONOLOGICAL ORDER)") and p.rstrip().endswith("Return ONLY the JSON object.")
    assert "NEIGHBOR CONTEXT" not in p and "NEIGHBOR CONTEXT" in build_keeper_prompt("g", units, neighbors="before: ...")


def test_targets_follow_the_hand_labels():
    ids = ["2:0", "2:1", "3:0"]
    t, src = target_for(ROW, ids)
    assert t == {"keep": ["3:0"], "keep_none": False, "unclear": False, "reason": "Keeping the last take."} and src == "template"
    t, src = target_for({**ROW, "keeper_unit_ids": "2:1|3:0", "notes": "SYNTHETIC continuation: keep both; keeper unclear"}, ids)
    assert t["keep"] == ["2:1", "3:0"] and t["unclear"] is True and src == "hand_note" and t["reason"].startswith("Continuation")
    t, _ = target_for({**ROW, "keeper_unit_ids": "", "keep_none": "1", "notes": "cue phrases; verify audio"}, ids)
    assert t["keep"] == [] and t["keep_none"] is True and t["reason"] == "Cue phrases."
    assert json.loads(json.dumps(t))["keep_none"] is True and clean_reason("") == ""

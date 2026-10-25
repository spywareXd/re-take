"""Tests for the OPTIONAL intra-unit retake candidate feature (detector, artifact, decisions, flag-off safety).

No GPU and no API calls. The golden test reads the review sheets under data/training/results/intra/ and the
cached per-video units/transcripts under data/training/processed/; it is skipped when those are absent.
"""

from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from models.retake import CutList, TimeSpan
from pipeline.run_pipeline import PipelineResult, run_full_pipeline
from pipeline.step_03_token_matching import (
    IntraUnitConfig,
    default_intra_config,
    find_intra_unit_candidates,
    load_intra_unit_candidates,
    save_intra_unit_candidates,
)
from pipeline.step_03_token_matching.intra_candidates import first_half_complete
from pipeline.step_08_video_edit import apply_intra_decisions

REPO = Path(__file__).resolve().parent.parent
STEP = 0.3     # seconds per mock word; each word lasts 0.25 s, so the silence between words is 0.05 s


# ---------------------------------------------------------------------------
# helpers: mock unit + transcript with word timestamps
# ---------------------------------------------------------------------------


def make_unit(text: str, pause_before_word: int | None = None, pause: float = 0.6, unit_id: str = "1:0"):
    """Return (unit_dict, transcript_dict) for `text`. If pause_before_word is set, a silence of `pause`
    seconds is inserted before that word index (the words after it are shifted later)."""
    words, t = [], 0.0
    for i, w in enumerate(text.split()):
        if pause_before_word is not None and i == pause_before_word:
            t += pause
        words.append({"word": w, "start": round(t, 3), "end": round(t + 0.25, 3), "confidence": 0.9})
        t += STEP
    seg = {"id": 1, "start": words[0]["start"], "end": words[-1]["end"], "speaker": "speaker_0",
           "text": text, "confidence": 0.9, "words": words}
    unit = {"unit_id": unit_id, "source_segment_id": 1, "start": seg["start"], "end": seg["end"], "raw_text": text}
    return unit, {"segments": [seg]}


def detect(text: str, **kw):
    unit, transcript = make_unit(text, **kw)
    return find_intra_unit_candidates([unit], transcript, video_id="video_t")


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------


def test_restart_is_flagged_with_split_after_first_attempt():
    (cand,) = detect("Now I use beard, [redacted sentence].")
    assert cand.rules_fired == "R1+R2"
    assert cand.split_word_index == 4                      # the restart "now" is the 5th word
    assert cand.discard_text == "Now I use beard,"
    assert cand.kept_text == "[redacted sentence]."
    assert cand.proposed_discard_start == 0.0
    assert cand.proposed_discard_end == pytest.approx(4 * STEP)    # start of the restart word
    assert cand.pause_at_split_seconds == pytest.approx(STEP - 0.25)
    assert cand.status == "suggested"
    assert cand.detector_version == default_intra_config.detector_version
    assert cand.video_id == "video_t" and cand.candidate_id == "intra__video_t__1-0"


def test_known_false_alarm_parallel_phrasing_is_flagged_and_not_special_cased():
    """'I love X and I love Y' was a labelled FALSE alarm in the review sheet. R1 (prefix 'I love tea time'
    reappears) and R2 (3-gram repeated with tokens in between) both fire, so it IS flagged. Documented, not
    special-cased: the UI/user decides, and first_half_complete does not filter it."""
    (cand,) = detect("[redacted sentence]")
    assert cand.split_word_index == 4
    assert cand.discard_text == "I love tea time and"
    # parallel phrasing with a different structure behaves the same way
    assert detect("Whether it is red, whether it is blue today")


def test_clean_sentence_is_not_flagged():
    assert detect("The quick brown fox jumps over the lazy dog today") == []


def test_intentional_doubling_is_not_flagged():
    assert detect("very very good very very big idea today") == []   # prefix 'very very' is one repeated word
    assert detect("very very") == []                                 # also below the token minimum


def test_units_under_six_tokens_are_not_flagged():
    assert detect("Now I use now I") == []                           # 5 tokens, same structure
    assert detect("Go go go go go") == []
    assert len(detect("Now I use now I use")) == 1                   # exactly 6 tokens is allowed


def test_long_silence_between_adjacent_copies_prevents_r2():
    # adjacent copies of the only 3-gram, separated by 0.6 s of silence: R2 needs a token between or a gap < 0.35 s
    assert detect("Now I use now I use beard", pause_before_word=3, pause=0.6) == []
    assert len(detect("Now I use now I use beard", pause_before_word=3, pause=0.2)) == 1


def test_function_word_only_repeats_are_not_flagged():
    assert detect("and the of and the of something else") == []


def test_thresholds_and_stoplist_come_from_config():
    unit, transcript = make_unit("Now I use beard, [redacted sentence].")
    strict = replace(default_intra_config, min_tokens=20)
    assert find_intra_unit_candidates([unit], transcript, config=strict) == []
    # the stoplist is data in the config: treat 'beard' and 'use' and 'now' as function words and nothing has content
    boring = replace(default_intra_config, function_words=default_intra_config.function_words | {"now", "use", "beard"})
    assert find_intra_unit_candidates([unit], transcript, config=boring) == []
    assert isinstance(default_intra_config, IntraUnitConfig)


def test_first_half_complete_is_a_hint_only():
    assert first_half_complete("Now I use beard,") is True          # subject + verb-like clause
    assert first_half_complete("That is done.") is True             # terminal punctuation
    assert first_half_complete("Welcome to the") is False
    # not a filter: an incomplete first half is still emitted
    (cand,) = detect("Welcome to the video welcome to the channel friends")
    assert cand.first_half_complete is False


def test_artifact_roundtrip(tmp_path: Path):
    unit, transcript = make_unit("Now I use beard, [redacted sentence].")
    cands = find_intra_unit_candidates([unit], transcript, video_id="video_t")
    path = save_intra_unit_candidates(cands, tmp_path / "candidates" / "intra_unit_candidates.json", video_id="video_t")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["candidate_count"] == 1 and data["candidates"][0]["status"] == "suggested"
    assert load_intra_unit_candidates(path)[0].to_dict() == cands[0].to_dict()


# ---------------------------------------------------------------------------
# apply_intra_decisions
# ---------------------------------------------------------------------------


def make_cutlist(duration: float = 10.0, discards: list[tuple[float, float]] | None = None) -> CutList:
    from pipeline.step_08_video_edit import compute_keep_spans

    spans = [TimeSpan(a, b, "d") for a, b in (discards or [])]
    return CutList(source="t.mp4", source_duration=duration, keep_spans=compute_keep_spans(spans, duration),
                   discard_spans=spans, retake_groups=[])


@pytest.fixture()
def cand():
    unit, transcript = make_unit("Now I use beard, [redacted sentence].")
    return find_intra_unit_candidates([unit], transcript, video_id="video_t")[0]


def spans(cl: CutList, which: str):
    return [(round(s.start, 3), round(s.end, 3)) for s in getattr(cl, which)]


def test_accept_adds_discard_and_leaves_input_untouched(cand):
    cl = make_cutlist()
    before = cl.to_dict()
    out = apply_intra_decisions(cl, [{"candidate_id": cand.candidate_id, "action": "accept"}], [cand])
    assert spans(out, "discard_spans") == [(0.0, 1.2)]
    assert spans(out, "keep_spans") == [(1.2, 10.0)]
    assert len(out.retake_groups) == 1 and out.retake_groups[0].discarded_text == "Now I use beard,"
    assert cl.to_dict() == before and out is not cl            # input not mutated
    assert cand.status == "suggested"                         # candidates are never auto-accepted


def test_reject_changes_nothing(cand):
    cl = make_cutlist(discards=[(5.0, 6.0)])
    out = apply_intra_decisions(cl, [{"candidate_id": cand.candidate_id, "action": "reject"}], [cand])
    assert spans(out, "discard_spans") == spans(cl, "discard_spans")
    assert spans(out, "keep_spans") == spans(cl, "keep_spans")
    assert out.retake_groups == []


def test_adjust_recomputes_span_from_chosen_word(cand):
    out = apply_intra_decisions(make_cutlist(), [{"candidate_id": cand.candidate_id, "action": "adjust",
                                                  "split_word_index": 3}], [cand])
    assert spans(out, "discard_spans") == [(0.0, 0.9)]         # start of word #3 = 3 * 0.3 s
    assert out.retake_groups[0].discarded_text == "Now I use"
    with pytest.raises(ValueError):
        apply_intra_decisions(make_cutlist(), [{"candidate_id": cand.candidate_id, "action": "adjust"}], [cand])
    with pytest.raises(ValueError):
        apply_intra_decisions(make_cutlist(), [{"candidate_id": cand.candidate_id, "action": "adjust",
                                                "split_word_index": 99}], [cand])


def test_overlap_with_existing_discard_is_merged(cand):
    cl = make_cutlist(discards=[(0.5, 2.0)])
    out = apply_intra_decisions(cl, [{"candidate_id": cand.candidate_id, "action": "accept"}], [cand])
    assert spans(out, "discard_spans") == [(0.0, 2.0)]
    assert spans(out, "keep_spans") == [(2.0, 10.0)]


def test_tiny_leftover_fragment_is_dropped(cand):
    out = apply_intra_decisions(make_cutlist(duration=1.25), [{"candidate_id": cand.candidate_id, "action": "accept"}], [cand])
    assert spans(out, "keep_spans") == []                       # 0.05 s left < 0.10 s minimum keep


def test_unknown_candidate_id_is_refused(cand):
    with pytest.raises(ValueError, match="Unknown candidate_id"):
        apply_intra_decisions(make_cutlist(), [{"candidate_id": "intra__nope", "action": "accept"}], [cand])
    with pytest.raises(ValueError, match="Unknown action"):
        apply_intra_decisions(make_cutlist(), [{"candidate_id": cand.candidate_id, "action": "maybe"}], [cand])


# ---------------------------------------------------------------------------
# flag-off safety
# ---------------------------------------------------------------------------

ORIGINAL_RESULT_KEYS = [
    "video_id", "input_video", "output_video", "source_duration", "output_duration", "time_saved",
    "retake_groups_count", "total_candidates", "likely_retakes", "ambiguous", "spans_count", "adjudicated_count",
    "stage_timings", "cutlist_dict", "success", "error",
]


def test_flag_defaults_off_and_result_dict_is_unchanged():
    assert inspect.signature(run_full_pipeline).parameters["detect_intra_unit"].default is False
    d = PipelineResult().to_dict()
    assert list(d) == ORIGINAL_RESULT_KEYS                      # same keys, same order, no new key
    assert PipelineResult.from_dict(d).intra_unit_candidates_count is None
    on = PipelineResult(intra_unit_candidates_count=3).to_dict()
    assert on["intra_unit_candidates_count"] == 3 and PipelineResult.from_dict(on).intra_unit_candidates_count == 3


def _mini_project(tmp_path: Path):
    """A processed dir with a cached transcript holding one restart unit, so no ASR/GPU/API is needed."""
    text = "Now I use beard, [redacted sentence]."
    _, transcript = make_unit(text)
    seg = transcript["segments"][0]
    transcript = {"source": "x.mp4", "duration": seg["end"], "language": "en", "segments": [seg]}
    proc = tmp_path / "proc" / "video_t"
    proc.mkdir(parents=True)
    (proc / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    video = tmp_path / "x.mp4"
    video.touch()
    return video, proc


@pytest.mark.parametrize("flag", [False, True])
def test_pipeline_flag_controls_only_the_intra_artifact(tmp_path: Path, flag: bool):
    pytest.importorskip("sentence_transformers")
    video, proc = _mini_project(tmp_path)
    kwargs = {"detect_intra_unit": True} if flag else {}
    result = run_full_pipeline(video, provider="mock", llm_provider="mock", device="cpu", processed_dir=proc, **kwargs)
    artifact = proc / "candidates" / "intra_unit_candidates.json"
    if flag:
        assert result.intra_unit_candidates_count == 1 and artifact.exists()
        assert json.loads(artifact.read_text(encoding="utf-8"))["candidates"][0]["status"] == "suggested"
    else:
        assert result.intra_unit_candidates_count is None and not artifact.exists()
        assert "intra_unit_candidates_count" not in result.to_dict()


# ---------------------------------------------------------------------------
# golden test against the manual review sheet
# ---------------------------------------------------------------------------

INTRA = REPO / "data" / "training" / "results" / "intra"
PROCESSED = REPO / "data" / "training" / "processed"


def test_golden_review_sheet_precision():
    """Join intra_review.csv (flags/rules) with the verdicts in intra_review_blind.csv BY row_id (unit_id is
    corrupted in the spreadsheet copy), locate each unit by (video, exact text), and run the R1+R2 detector.

    Expected: it flags exactly the rows where R1 and R2 both fired in the experiment (13 rows: the 12 labelled
    'R1+R2' plus row 14, which also had R4 and is a false alarm), and precision against the verdicts is 9/13.
    The brief quoted 9/12 because it excluded row 14; dropping R4 turns that row into a plain R1+R2 flag."""
    pd = pytest.importorskip("pandas")
    if not (INTRA / "intra_review.csv").exists() or not PROCESSED.exists():
        pytest.skip("review sheets or cached processed data not available")
    review = pd.read_csv(INTRA / "intra_review.csv", dtype=str, keep_default_na=False)
    blind = pd.read_csv(INTRA / "intra_review_blind.csv", dtype=str, keep_default_na=False)
    blind = blind.rename(columns={"verdict": "user_verdict"})
    sheet = review.drop(columns=["verdict"]).merge(blind[["row_id", "text", "start", "user_verdict"]], on="row_id", validate="1:1")
    assert len(sheet) == 63 and sheet.user_verdict.isin(["0", "1"]).all()

    flagged_rows = set()
    cache = {}
    for r in sheet.itertuples():
        vdir = PROCESSED / r.video_id
        if r.video_id not in cache:
            cache[r.video_id] = (json.loads((vdir / "units.json").read_text()),
                                 json.loads((vdir / "transcript.json").read_text()))
        units, transcript = cache[r.video_id]
        matches = [u for u in units if u["raw_text"] == r.text and abs(u["start"] - float(r.start)) < 0.02]
        assert len(matches) == 1, f"row {r.row_id}: could not locate the unit by text/start"
        if find_intra_unit_candidates(matches, transcript, video_id=r.video_id):
            flagged_rows.add(r.row_id)

    both = {r.row_id for r in sheet.itertuples() if "R1" in r.rule_fired and "R2" in r.rule_fired}
    pure = {r.row_id for r in sheet.itertuples() if r.rule_fired == "R1+R2"}
    assert flagged_rows == both and len(both) == 13 and len(pure) == 12 and pure < both
    hits = sum(sheet.set_index("row_id").loc[sorted(flagged_rows), "user_verdict"] == "1")
    assert (hits, len(flagged_rows)) == (9, 13)
    # the 12 pure R1+R2 rows alone are 9/12 as quoted; the extra row (R1+R2+R4) is the one false alarm added
    assert sum(sheet.set_index("row_id").loc[sorted(pure), "user_verdict"] == "1") == 9

import numpy as np

from models.retake import CutList, TimeSpan
from pipeline.step_08_video_edit.boundary_refiner import CutRefineConfig, envelope_db, refine_cutlist
from pipeline.step_08_video_edit.ffmpeg_editor import build_concat_filter

SR = 16000


def _audio(duration, bursts, noise=1e-4):
    """Room noise plus loud tone bursts at the given (start, end) times."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, noise, int(duration * SR))
    for a, b in bursts:
        t = np.arange(int(a * SR), int(b * SR)) / SR
        x[int(a * SR):int(b * SR)] += 0.2 * np.sin(2 * np.pi * 220 * t)
    return x


def _cutlist(discards, duration):
    from pipeline.step_08_video_edit.cut_generator import compute_keep_spans
    spans = [TimeSpan(a, b, "cut") for a, b in discards]
    return CutList(source="x.mp4", source_duration=duration, keep_spans=compute_keep_spans(spans, duration), discard_spans=spans, retake_groups=[])


UNITS = [{"unit_id": "k1", "start": 1.0, "end": 2.0}, {"unit_id": "r", "start": 6.0, "end": 7.0}, {"unit_id": "k2", "start": 11.0, "end": 12.0}]
BURSTS = [(1.0, 2.0), (6.0, 7.0), (11.0, 12.0)]


def test_dead_air_around_a_cut_is_trimmed_to_the_configured_pause():
    env = envelope_db(_audio(14.0, BURSTS), SR)
    out = refine_cutlist(_cutlist([(6.0, 7.0)], 14.0), env, UNITS, CutRefineConfig(keep_pause_s=0.3))
    d = out.discard_spans[0]
    assert abs(d.start - (2.0 + 0.15)) < 0.03          # 0.15 s of silence kept after kept speech
    assert abs(d.end - (11.0 - 0.15)) < 0.03           # 0.15 s kept before the next speech
    kept_gap = (d.start - 2.0) + (11.0 - d.end)
    assert abs(kept_gap - 0.3) < 0.05


def test_cut_moves_to_true_onset_when_asr_start_is_late():
    # real take starts at 5.8 but ASR says 6.0: the leaked 0.2 s must be inside the discard, not the kept side
    bursts = [(1.0, 2.0), (5.8, 7.0), (11.0, 12.0)]
    env = envelope_db(_audio(14.0, bursts), SR)
    out = refine_cutlist(_cutlist([(6.0, 7.0)], 14.0), env, UNITS, CutRefineConfig())
    assert out.discard_spans[0].start < 5.8 - 0.1          # cut sits in silence before the onset
    assert out.discard_spans[0].start > 2.0                # never into the kept speech


def test_never_cuts_into_adjacent_kept_speech():
    # kept speech runs right up to the removed take with no pause
    bursts = [(1.0, 6.0), (6.0, 7.0), (11.0, 12.0)]
    units = [{"start": 1.0, "end": 6.0}, {"start": 6.0, "end": 7.0}, {"start": 11.0, "end": 12.0}]
    env = envelope_db(_audio(14.0, bursts), SR)
    out = refine_cutlist(_cutlist([(6.0, 7.0)], 14.0), env, units, CutRefineConfig())
    assert out.discard_spans[0].start >= 6.0 - 0.02


def test_pauses_not_next_to_a_cut_are_untouched():
    env = envelope_db(_audio(14.0, BURSTS), SR)
    out = refine_cutlist(_cutlist([(6.0, 7.0)], 14.0), env, UNITS, CutRefineConfig())
    assert out.keep_spans[0].start == 0.0 and out.keep_spans[0].end == out.discard_spans[0].start
    assert out.keep_spans[-1].end == 14.0


def test_adjacent_discards_merge_and_keep_spans_cover_the_rest():
    env = envelope_db(_audio(14.0, BURSTS), SR)
    out = refine_cutlist(_cutlist([(6.0, 7.0), (7.2, 7.4)], 14.0), env, UNITS, CutRefineConfig())
    total = sum(s.duration for s in out.keep_spans) + sum(s.duration for s in out.discard_spans)
    assert abs(total - 14.0) < 0.2
    assert out.discard_spans == sorted(out.discard_spans, key=lambda s: s.start)


def test_concat_filter_has_fades_and_exact_inputs():
    g = build_concat_filter([TimeSpan(0, 2), TimeSpan(5, 8)], fade_s=0.012)
    assert "concat=n=2:v=1:a=1[outv][outa]" in g
    assert g.count("afade=t=in") == 2 and g.count("afade=t=out") == 2
    assert "[0:v]setpts=PTS-STARTPTS[v0]" in g and "[1:a]asetpts=PTS-STARTPTS" in g


def test_concat_filter_without_audio():
    g = build_concat_filter([TimeSpan(0, 2), TimeSpan(5, 8)], fade_s=0.012, has_audio=False)
    assert "afade" not in g and "concat=n=2:v=1:a=0[outv]" in g


def test_word_mode_removes_breath_between_kept_words_and_dead_air_around_a_cut():
    # kept speech 1-2 s and 11-12 s, removed take 6-7 s; a breath-like burst (quiet, 0.3 s) at 4.0-4.3 and another at 9.0-9.3
    audio = _audio(14.0, BURSTS)
    for a, b in ((4.0, 4.3), (9.0, 9.3)):
        t = np.arange(int(a * SR), int(b * SR)) / SR
        audio[int(a * SR):int(b * SR)] += 0.02 * np.sin(2 * np.pi * 900 * t)
    words = [(1.0, 1.5), (1.5, 2.0), (6.0, 6.5), (6.5, 7.0), (11.0, 11.5), (11.5, 12.0)]
    out = refine_cutlist(_cutlist([(6.0, 7.0)], 14.0), envelope_db(audio, SR), UNITS, CutRefineConfig(), words=words)
    d = next(s for s in out.discard_spans if s.start < 6.0 < s.end)
    assert d.start < 2.0 + 0.2 + 0.15 + 0.05 and d.end > 11.0 - 0.12 - 0.15 - 0.05          # pause before/after the cut shrunk to ~keep_pause_s
    assert d.start <= 4.0 and d.end >= 9.3                                                  # both breaths are inside the removed region
    for w in (words[0], words[1], words[4], words[5]):                                      # kept words untouched
        assert any(k.start <= w[0] and k.end >= w[1] for k in out.keep_spans)


def test_word_mode_shortens_long_pause_between_kept_words_without_a_retake():
    audio = _audio(14.0, [(1.0, 2.0), (8.0, 9.0)])
    words = [(1.0, 2.0), (8.0, 9.0)]
    out = refine_cutlist(_cutlist([], 14.0), envelope_db(audio, SR), [], CutRefineConfig(), words=words)
    assert any(d.start > 2.0 and d.end < 8.0 for d in out.discard_spans)
    kept_total = sum(k.duration for k in out.keep_spans)
    assert kept_total < 14.0 - 4.0


def test_word_mode_leaves_short_natural_pause_alone():
    audio = _audio(6.0, [(1.0, 2.0), (2.5, 3.5)])
    out = refine_cutlist(_cutlist([], 6.0), envelope_db(audio, SR), [], CutRefineConfig(), words=[(1.0, 2.0), (2.5, 3.5)])
    assert not any(2.0 < d.start < 2.5 or 2.0 < d.end < 2.5 for d in out.discard_spans)


# ---------------------------------------------------------------------------- cut modes: containment (all three modes)

import pytest  # noqa: E402

from pipeline.step_08_video_edit.cut_config import CutConfig  # noqa: E402
from pipeline.step_08_video_edit.cut_modes import build_cutlist  # noqa: E402


def _scene(good_start=4.2, breath=(3.0, 4.2)):
    """failed take 1.0-3.0 (removed), breath, good take (5 words, 1.8 s), 0.4 s natural pause, closing sentence (1.6 s), then 3 s of silence."""
    words = [(1.0 + 0.4 * i, 1.4 + 0.4 * i) for i in range(5)]
    words += [(good_start + 0.36 * i, good_start + 0.36 + 0.36 * i) for i in range(5)]
    c0 = good_start + 1.8 + 0.4
    words += [(c0 + 0.4 * i, c0 + 0.4 + 0.4 * i) for i in range(4)]
    end = c0 + 1.6 + 3.0
    audio = _audio(end, [(1.0, 3.0), (good_start, good_start + 1.8), (c0, c0 + 1.6)])
    if breath:
        t = np.arange(int(breath[0] * SR), int(breath[1] * SR)) / SR
        audio[int(breath[0] * SR):int(breath[1] * SR)] += 0.03 * np.sin(2 * np.pi * 700 * t)       # breath ~17 dB below speech
    transcript = {"segments": [{"id": 0, "words": [{"word": "w", "start": s, "end": e} for s, e in words]}]}
    units = [{"unit_id": "0:0", "start": 1.0, "end": 3.0}, {"unit_id": "0:1", "start": good_start, "end": good_start + 1.8},
             {"unit_id": "0:2", "start": c0, "end": c0 + 1.6}]
    spans = [{"span_id": "s", "adjudication_status": "resolved", "is_genuine_retake_group": True, "keeper_unit_id": "0:1",
              "units_to_remove": ["0:0"], "start": 1.0, "end": good_start + 1.8}]
    return envelope_db(audio, SR), transcript, units, spans, words, end


@pytest.mark.parametrize("mode", ["retake_only", "anchored", "sentences"])
def test_cut_modes_keep_every_kept_word_and_leave_little_padding(mode):
    env, transcript, units, spans, words, dur = _scene(good_start=3.25, breath=None)      # tight pauses: nothing cap-limited, nothing exempt
    cfg = CutConfig(cut_mode=mode, use_visual_motion=False)
    cl, rows = build_cutlist(mode, spans, units, transcript, "x.mp4", dur, envelope=env, cfg=cfg)
    kept_words = words[5:]
    assert cl.mode == mode and cl.removed_seconds == pytest.approx(sum(d.duration for d in cl.discard_spans), abs=1e-3)
    for s, e in kept_words:                                                  # 1) every kept word lies fully inside a keep span
        assert any(k.start <= s + 1e-6 and k.end >= e - 1e-6 for k in cl.keep_spans), (mode, s, e)
    for s, e in words[:5]:                                                   # removed take is not audible in any mode
        assert not any(k.start < e - 0.05 and k.end > s + 0.05 for k in cl.keep_spans), (mode, s, e)
    if mode in ("anchored", "retake_only"):                                  # 2) lead/trail of spans that touch a cut
        for r in rows:
            if r["row_type"] == "keep_span" and r["n_kept_words"]:
                if r["start_join_stopped_by"] != "":                         # this edge is a cut (file edges are untouched)
                    assert r["lead_s"] <= 0.35 + 1e-6, (mode, r)
                if r["end_join_stopped_by"] != "":
                    assert r["trail_s"] <= 0.45 + 1e-6, (mode, r)
    if mode == "sentences":
        assert cl.keep_spans[0].start == pytest.approx(3.25 - cfg.head_pad_s, abs=0.06)     # nothing before the first kept word
        assert cl.keep_spans[-1].end == pytest.approx(words[-1][1] + cfg.tail_pad_s, abs=0.06)   # trailing silence removed


def test_anchored_worked_example_and_cap():
    """failed take 1.0-3.0, breath 3.0-4.2, good take from 4.2. The cap (0.7 s per side, from the unit gap gate) limits what is absorbed."""
    env, transcript, units, spans, _, dur = _scene()
    base = dict(cut_mode="anchored", use_visual_motion=False)
    cl, rows = build_cutlist("anchored", spans, units, transcript, "x.mp4", dur, envelope=env, cfg=CutConfig(**base))
    cfg = CutConfig(**base)
    # right side: breath 3.0-4.0 is dead air (17 dB below speech, shorter than...) but absorbing stops at the 0.7 s cap
    assert cl.keep_spans[1].start == pytest.approx(3.0 + cfg.max_absorb_s, abs=0.06)           # actual value with the default cap: 3.7
    assert any(r["end_join_cap_truncated"] or r["start_join_cap_truncated"] for r in rows if r["row_type"] == "keep_span")
    # with a cap large enough the spec value is reached: output resumes head_pad_s before the good take
    wide = CutConfig(**base, max_absorb_s=2.0)
    cl2, _ = build_cutlist("anchored", spans, units, transcript, "x.mp4", dur, envelope=env, cfg=wide)
    assert cl2.keep_spans[0].start == pytest.approx(4.2 - wide.head_pad_s, abs=0.06)           # 4.0: no leading silence, no breath

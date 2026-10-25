"""
Refine cut points against the real audio energy.

A cutlist built from ASR unit times cuts at word edges that are not the true speech edges (Whisper starts late / ends early),
and it only removes the speech of a retake, leaving the thinking pauses on both sides of it. This module moves every cut to the
silence around the removed take and shortens that silence to a short natural pause.

    discard (ASR):        [a ................ b]
    speech / silence:  ..kept..|--sil--|=removed=|--sil--|..kept..
                                p       o        e       n
    refined discard:             [p+pre .............. n-post]    (pre + post = keep_pause_s)

Pauses that are not next to a cut are left untouched.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from models.retake import CutList, TimeSpan

from .cut_generator import compute_keep_spans, merge_spans

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class CutRefineConfig:
    search_s: float = 0.4            # how far a cut may move to find the true speech onset/offset of the removed take
    keep_pause_s: float = 0.3        # total silence kept at a cut (split evenly before/after)
    silence_rel_db: float = 30.0     # a frame is silent when it is this many dB below the recording's 95th-percentile level
    frame_s: float = 0.01
    min_fragment_s: float = 0.10
    blip_max_s: float = 0.40         # a short, quiet sound (click, breath) inside a pause does not count as speech ...
    blip_rel_db: float = 13.0        # ... if it peaks at least this many dB below the 95th-percentile level ...
    blip_guard_s: float = 0.40       # ... and is at least this far from the neighbouring ASR speech (so soft word edges are never skipped)
    # word-timing mode (used when the transcript has word timestamps): speech = the kept WORDS, everything else between them is dead air
    lead_pad_s: float = 0.12         # kept before a word's start (soft consonant onset)
    tail_pad_s: float = 0.20         # kept after a word's end (ASR ends words early)
    edge_ext_s: float = 0.30         # a word edge may extend this far while the audio stays clearly speech-level (not breath-level)
    edge_rel_db: float = 15.0        # "clearly speech" = within this many dB of the 95th-percentile level
    max_pause_s: float = 0.70        # any pause between kept words longer than this is shortened to keep_pause_s


def load_envelope(media_path: str | Path, frame_s: float = 0.01) -> np.ndarray:
    """Per-frame RMS level in dBFS of the media's audio (decoded once by ffmpeg to 16 kHz mono)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(media_path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    return envelope_db(np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0, SAMPLE_RATE, frame_s)


def envelope_db(samples: np.ndarray, sample_rate: int, frame_s: float = 0.01) -> np.ndarray:
    n = max(1, int(round(frame_s * sample_rate)))
    frames = len(samples) // n
    if frames == 0:
        return np.full(1, -100.0)
    rms = np.sqrt(np.mean(samples[: frames * n].reshape(frames, n) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-5))        # floor -100 dBFS


def reference_level(env_db: np.ndarray) -> float:
    live = env_db[env_db > -90.0]
    return float(np.percentile(live, 95)) if live.size else -30.0


def silence_mask(env_db: np.ndarray, rel_db: float) -> np.ndarray:
    return env_db < (reference_level(env_db) - rel_db)


def _unit_bounds(units: Sequence[Any]) -> tuple[list[float], list[float]]:
    starts, ends = [], []
    for u in units:
        d = u if isinstance(u, dict) else (u.to_dict() if hasattr(u, "to_dict") else vars(u))
        starts.append(float(d["start"]))
        ends.append(float(d["end"]))
    return sorted(starts), sorted(ends)


def words_from_transcript(transcript: Any) -> list[tuple[float, float]]:
    """(start, end) of every word with timing in a Transcript / transcript dict; [] when the ASR gave no word timestamps."""
    segments = transcript["segments"] if isinstance(transcript, dict) else transcript.segments
    out: list[tuple[float, float]] = []
    for seg in segments:
        for w in (seg.get("words") if isinstance(seg, dict) else seg.words) or []:
            s, e = (w["start"], w["end"]) if isinstance(w, dict) else (w.start, w.end)
            if s is not None and e is not None:
                out.append((float(s), float(e)))
    return sorted(out)


def _refine_by_words(cutlist: CutList, envelope: np.ndarray, words: Sequence[tuple[float, float]], cfg: CutRefineConfig) -> CutList:
    h = cfg.frame_s
    duration = float(cutlist.source_duration)
    n_frames = len(envelope)
    loud = reference_level(envelope) - cfg.edge_rel_db
    discards = cutlist.discard_spans

    def fi(t: float) -> int:
        return min(max(int(round(t / h)), 0), n_frames - 1)

    kept = [w for w in words if not any(d.start - 0.02 <= (w[0] + w[1]) / 2.0 <= d.end + 0.02 for d in discards)]
    if not kept:
        return cutlist

    def onset(i: int) -> float:
        t = kept[i][0] - cfg.lead_pad_s
        floor = max(kept[i][0] - cfg.lead_pad_s - cfg.edge_ext_s, kept[i - 1][1] if i else 0.0)
        while t - h > floor and envelope[fi(t - h)] > loud:
            t -= h
        return max(t, 0.0)

    def offset(i: int) -> float:
        t = kept[i][1] + cfg.tail_pad_s
        ceil = min(kept[i][1] + cfg.tail_pad_s + cfg.edge_ext_s, kept[i + 1][0] if i + 1 < len(kept) else duration)
        while t + h < ceil and envelope[fi(t + h)] > loud:
            t += h
        return min(t, duration)

    def quietest(a: float, b: float) -> float:
        lo, hi = fi(min(a, b)), fi(max(a, b))
        return (lo + int(np.argmin(envelope[lo : hi + 1]))) * h

    half = cfg.keep_pause_s / 2.0
    out: list[TimeSpan] = []
    first_n = onset(0)
    if first_n > cfg.max_pause_s:
        out.append(TimeSpan(0.0, round(quietest(first_n - half, first_n), 4), "pause"))
    for i in range(len(kept) - 1):
        P, N = offset(i), onset(i + 1)
        if N <= P:
            continue
        inside = [d for d in discards if d.end > P and d.start < N]
        if N - P > cfg.keep_pause_s and (inside or N - P > cfg.max_pause_s):
            a, b = quietest(P, P + half), quietest(N - half, N)
            if b > a:
                out.append(TimeSpan(round(a, 4), round(b, 4), "retake" if inside else "pause"))
                continue
        if inside:                                                       # gap too small to keep a pause: remove just the retake audio
            out.append(TimeSpan(max(P, min(d.start for d in inside)), min(N, max(d.end for d in inside)), "retake"))
    last_p = offset(len(kept) - 1)
    if duration - last_p > cfg.max_pause_s:
        out.append(TimeSpan(round(quietest(last_p, last_p + half), 4), duration, "pause"))

    merged = merge_spans([s for s in out if s.duration > 0], tolerance=0.0)
    keep = compute_keep_spans(merged, source_duration=duration, min_fragment_duration=cfg.min_fragment_s)
    logger.info("[Refine/words] %d discards (%.2f s) before, %d (%.2f s) after", len(discards), sum(s.duration for s in discards),
                len(merged), sum(s.duration for s in merged))
    return CutList(source=cutlist.source, source_duration=duration, keep_spans=keep, discard_spans=merged, retake_groups=cutlist.retake_groups)


def refine_cutlist(
    cutlist: CutList,
    envelope: np.ndarray,
    units: Sequence[Any] = (),
    config: CutRefineConfig | None = None,
    words: Sequence[tuple[float, float]] | None = None,
) -> CutList:
    """
    Return a new CutList whose cuts sit in real silence and whose pauses around cuts are trimmed to cfg.keep_pause_s.
    With word timestamps (`words`, see words_from_transcript) speech = the kept words, so breaths and mouth noises between them are
    removed too; without them the audio envelope alone decides what is speech.
    """
    cfg = config or CutRefineConfig()
    if words:
        return _refine_by_words(cutlist, envelope, words, cfg)
    h = cfg.frame_s
    quiet = silence_mask(envelope, cfg.silence_rel_db)
    blip_peak = reference_level(envelope) - cfg.blip_rel_db
    n_frames = len(quiet)
    duration = float(cutlist.source_duration)
    starts, ends = _unit_bounds(units)
    discards = sorted(cutlist.discard_spans, key=lambda s: s.start)
    half = cfg.keep_pause_s / 2.0

    def fi(t: float) -> int:
        return min(max(int(round(t / h)), 0), n_frames - 1)

    def silence_back(k: int, k_min: int, t_min: float) -> int:
        """Walk back through silence (skipping brief quiet blips that start no earlier than t_min) to the end of the speech before."""
        while k > k_min:
            if quiet[k - 1]:
                k -= 1
                continue
            m = k - 1
            while m > k_min and not quiet[m - 1]:
                m -= 1
            if (k - m) * h <= cfg.blip_max_s and envelope[m:k].max() <= blip_peak and m * h >= t_min:
                k = m
            else:
                break
        return k

    def silence_fwd(k: int, k_max: int, t_max: float) -> int:
        while k < k_max:
            if quiet[k + 1]:
                k += 1
                continue
            m = k + 1
            while m < k_max and not quiet[m + 1]:
                m += 1
            if (m - k) * h <= cfg.blip_max_s and envelope[k + 1 : m + 1].max() <= blip_peak and m * h <= t_max:
                k = m
            else:
                break
        return k

    refined: list[TimeSpan] = []
    for i, d in enumerate(discards):
        prev_end = discards[i - 1].end if i > 0 else 0.0
        next_start = discards[i + 1].start if i + 1 < len(discards) else duration
        lo = max([prev_end] + [e for e in ends if e <= d.start + 0.01])            # never move into earlier kept speech (ASR time)
        hi = min([next_start] + [s for s in starts if s >= d.end - 0.01])
        lo, hi = min(lo, d.start), max(hi, d.end)

        # true onset of the removed take: walk back through non-silent frames (leaked first syllable / breath), bounded
        o = d.start
        if not quiet[fi(o)]:
            k, k_min = fi(o), fi(max(lo, d.start - cfg.search_s))
            while k > k_min and not quiet[k - 1]:
                k -= 1
            o = k * h
        # end of the kept speech before it: walk back through silence
        p = silence_back(fi(o), fi(lo), lo + cfg.blip_guard_s) * h
        # true offset of the removed take: walk forward through non-silent frames (word tail), bounded
        e = d.end
        if not quiet[fi(e)]:
            k, k_max = fi(e), fi(min(hi, d.end + cfg.search_s))
            while k < k_max and not quiet[k + 1]:
                k += 1
            e = k * h
        # start of the next kept speech: walk forward through silence
        n = silence_fwd(fi(e), fi(hi), hi - cfg.blip_guard_s) * h

        new_start = o if (o - p) <= half else p + half
        new_end = e if (n - e) <= half else n - half
        # not in silence at the chosen point (continuous speech): take the lowest-energy frame within the allowed range
        if not quiet[fi(new_start)]:
            a0, a1 = fi(max(lo, o)), fi(d.start + h)
            if a1 > a0:
                new_start = (a0 + int(np.argmin(envelope[a0 : a1 + 1]))) * h
        if not quiet[fi(new_end)]:
            b0, b1 = fi(d.end - h), fi(min(hi, e))
            if b1 > b0:
                new_end = (b0 + int(np.argmin(envelope[b0 : b1 + 1]))) * h
        if new_end <= new_start:
            refined.append(TimeSpan(d.start, d.end, d.label))
            continue
        refined.append(TimeSpan(round(new_start, 4), round(new_end, 4), d.label))

    merged = merge_spans(refined, tolerance=0.0)
    keep = compute_keep_spans(merged, source_duration=duration, min_fragment_duration=cfg.min_fragment_s)
    logger.info("[Refine] %d discards, %.2f s removed before, %.2f s after refinement",
                len(discards), sum(s.duration for s in discards), sum(s.duration for s in merged))
    return CutList(source=cutlist.source, source_duration=duration, keep_spans=keep, discard_spans=merged,
                   retake_groups=cutlist.retake_groups)

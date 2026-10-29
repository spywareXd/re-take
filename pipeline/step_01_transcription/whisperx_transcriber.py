import os
import sys
import types
from typing import Optional

# Safeguard against SpeechBrain's hardcoded POSIX '/inspect.py' check on Windows
try:
    import speechbrain.utils.importutils as sb_importutils
    _orig_ensure = sb_importutils.LazyModule.ensure_module
    def _safe_ensure_module(self, stacklevel=1):
        try:
            return _orig_ensure(self, stacklevel=stacklevel)
        except Exception:
            raise AttributeError()
    sb_importutils.LazyModule.ensure_module = _safe_ensure_module
except Exception:
    pass

from models.transcript import Transcript
from models.segment import Segment, Word

try:
    from .base import BaseTranscriber  # type: ignore
except ImportError:
    from pipeline.transcription.base import BaseTranscriber  # type: ignore


import logging

logger = logging.getLogger(__name__)


class WhisperXTranscriber(BaseTranscriber):
    """
    STT Adapter using WhisperX.
    Uses Whisper for base transcription + wav2vec2 forced alignment for microsecond word boundaries.
    """

    def __init__(self, model_name: str = "large-v2", device: str = "cpu", compute_type: str = "float32", **kwargs):
        if "model_size" in kwargs:
            model_name = kwargs.pop("model_size")
        super().__init__(model_name=model_name, device=device, compute_type=compute_type, **kwargs)
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self, whisperx, compute_type, st=None):
        """Load with the requested compute type; if the device refuses it, fall back through the candidates (int8_float16, float16, int8 on GPU; int8, float32 on CPU)."""
        from .model_catalog import compute_candidates

        last, first_exc = None, None
        for ct in compute_candidates(self.device, compute_type):
            try:
                model = whisperx.load_model(self.model_name, self.device, compute_type=ct)
                self.loaded_compute_type = ct
                if st is not None:
                    st.done_message = f"compute type {ct}" + (f" (after {last})" if last else "")
                return model
            except (ValueError, RuntimeError) as exc:
                first_exc = first_exc or exc                           # the error for the type that was asked for is the one the user sees
                last = f"{compute_type if ct == compute_type else ct} failed: {str(exc)[:120]}"
                logger.warning("[WhisperX] compute type %s failed (%s); trying the next", ct, exc)
        raise first_exc

    def _get_model(self, events=None):
        if self._model is None:
            try:
                import whisperx
            except ImportError:
                raise ImportError("whisperx is not installed. Install it with: pip install whisperx")
            compute_type = self.compute_type
            if self.device == "cuda" and compute_type == "float32":
                compute_type = "float16"
            if events is None:
                self._model = self._load(whisperx, compute_type)
            else:                                                       # progress events (service): download with a byte/percent hook, then load
                from .model_download import download

                with events.stage("model_download", f"Model {self.model_name}") as st:
                    try:
                        got = download(self.model_name, lambda pct, done, total: st.progress(pct, f"{done / 1e6:.0f} of {total / 1e6:.0f} MB"))
                    except Exception as exc:                            # not fatal: whisperx will try to fetch the model itself in the next step
                        got = False
                        st.done_message = f"download hook failed ({exc}); loading will fetch it"
                    if not got and not st.done_message:
                        st.done_message = "already downloaded"
                with events.stage("model_load", f"{self.model_name} on {self.device}") as st:
                    self._model = self._load(whisperx, compute_type, st)
        return self._model

    def _transcribe_with_events(self, model, audio, batch_size, events):
        """WhisperX runs VAD, then the batches, inside one call and prints 'Progress: x%' per batch. VAD is reported as started/done around the call's first progress
        line; the percent of the transcription comes from those lines (stdout is parsed only for the duration of this call)."""
        import contextlib
        import io
        import re

        model_name = self.model_name

        class Tap(io.TextIOBase):
            def __init__(self):
                self.vad_open = True
                events.emit("vad", "started", "Finding speech")
                events.emit("transcription", "started", f"Model {model_name}")

            def write(self, s):
                m = re.search(r"Progress:\s*([0-9.]+)%", s)
                if m:
                    if self.vad_open:
                        self.vad_open = False
                        events.emit("vad", "done", "", pct=100.0)
                    events.emit("transcription", "progress", "", pct=float(m.group(1)))
                return len(s)

        tap = Tap()
        try:
            with contextlib.redirect_stdout(tap):
                result = model.transcribe(audio, batch_size=batch_size, print_progress=True)
        except BaseException as exc:
            events.emit("vad" if tap.vad_open else "transcription", "error", f"{type(exc).__name__}: {exc}")
            raise
        if tap.vad_open:
            events.emit("vad", "done", "", pct=100.0)
        events.emit("transcription", "done", f"language {result.get('language', '?')}", pct=100.0)
        return result

    def transcribe(self, audio_path: str, source_name: Optional[str] = None, **kwargs) -> Transcript:
        import whisperx

        events = kwargs.get("events")
        model = self._get_model(events)
        audio = whisperx.load_audio(audio_path)
        batch_size = kwargs.get("batch_size", 16)
        if events is None:
            result = model.transcribe(audio, batch_size=batch_size)
        else:
            result = self._transcribe_with_events(model, audio, batch_size, events)

        # Forced alignment for precise timestamps
        if events is None:
            align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            aligned_result = whisperx.align(
                result["segments"], align_model, metadata, audio, self.device, return_char_alignments=False
            )
        else:
            with events.stage("alignment", f"Alignment model for '{result['language']}' (downloaded on first use; no byte progress is available)") as st:
                align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
                st.progress(50.0, "Aligning words")
                aligned_result = whisperx.align(
                    result["segments"], align_model, metadata, audio, self.device, return_char_alignments=False
                )

        segments_list = []
        seg_id = 1
        for seg in aligned_result["segments"]:
            words = []
            if "word_segments" in seg:
                for w in seg["word_segments"]:
                    words.append(
                        Word(
                            word=w.get("word", "").strip(),
                            start=round(w.get("start", 0.0), 2),
                            end=round(w.get("end", 0.0), 2),
                            confidence=round(w.get("score", 1.0), 2) if "score" in w else None,
                        )
                    )
            elif "words" in seg:
                for w in seg["words"]:
                    words.append(
                        Word(
                            word=w.get("word", "").strip(),
                            start=round(w.get("start", 0.0), 2),
                            end=round(w.get("end", 0.0), 2),
                            confidence=round(w.get("score", 1.0), 2) if "score" in w else None,
                        )
                    )

            speaker = seg.get("speaker", "speaker_0")
            confidence = round(seg.get("score", 1.0), 2) if "score" in seg else 1.0

            segments_list.append(
                Segment(
                    id=seg_id,
                    start=round(seg["start"], 2),
                    end=round(seg["end"], 2),
                    speaker=speaker,
                    text=seg["text"].strip(),
                    confidence=confidence,
                    words=words,
                )
            )
            seg_id += 1

        source = source_name or os.path.basename(audio_path)
        duration = segments_list[-1].end if segments_list else 0.0

        return Transcript(
            source=source,
            duration=round(duration, 2),
            language=result.get("language", "en"),
            segments=segments_list,
        )

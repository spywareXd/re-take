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


class WhisperXTranscriber(BaseTranscriber):
    """
    STT Adapter using WhisperX.
    Uses Whisper for base transcription + wav2vec2 forced alignment for microsecond word boundaries.
    """

    def __init__(self, model_name: str = "base", device: str = "cpu", compute_type: str = "float32", **kwargs):
        if "model_size" in kwargs:
            model_name = kwargs.pop("model_size")
        super().__init__(model_name=model_name, device=device, compute_type=compute_type, **kwargs)
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                import whisperx
            except ImportError:
                raise ImportError("whisperx is not installed. Install it with: pip install whisperx")
            self._model = whisperx.load_model(self.model_name, self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio_path: str, source_name: Optional[str] = None, **kwargs) -> Transcript:
        import whisperx

        model = self._get_model()
        audio = whisperx.load_audio(audio_path)
        batch_size = kwargs.get("batch_size", 16)
        result = model.transcribe(audio, batch_size=batch_size)

        # Forced alignment for precise timestamps
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device
        )
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

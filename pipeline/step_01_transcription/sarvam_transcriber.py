import os
import re
import requests
from pathlib import Path
from typing import Optional, List
from models.transcript import Transcript
from models.segment import Segment, Word
from app.core.config import get_settings, settings
from .audio_extractor import AudioExtractor

try:
    from .base import BaseTranscriber  # type: ignore
except ImportError:
    from pipeline.transcription.base import BaseTranscriber  # type: ignore


class SarvamAITranscriber(BaseTranscriber):
    """
    STT Adapter using Sarvam AI Speech-To-Text REST API (e.g. saaras:v3, saaras:v4, saarika:v2.5).
    Supports Indian accents/languages, word-level timestamps, and speaker diarization.
    """

    API_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "saaras:v3",
        language_code: str = "en-IN",
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, language_code=language_code, **kwargs)
        self.api_key = api_key or get_settings().sarvam_api_key
        self.model = model
        self.language_code = language_code

    def transcribe(
        self,
        audio_path: str,
        source_name: Optional[str] = None,
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Transcript:
        api_key = self.api_key or get_settings().sarvam_api_key
        if not api_key:
            raise ValueError(
                "Sarvam AI API key is required. Set SARVAM_API_KEY in your .env file or pass api_key."
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio/Media file not found: {audio_path}")

        # If video file, extract 16kHz WAV audio first
        ext = os.path.splitext(audio_path)[1].lower()
        if ext in [".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv"]:
            wav_path = AudioExtractor.extract_audio(audio_path)
            file_to_send = wav_path
        else:
            file_to_send = audio_path

        headers = {
            "api-subscription-key": api_key,
        }

        lang = language_code or self.language_code
        model_name = model or self.model
        data = {
            "model": model_name,
            "language_code": lang,
            "with_timestamps": "true",
        }
        if kwargs.get("with_diarization", False):
            data["with_diarization"] = "true"

        with open(file_to_send, "rb") as f:
            files = {
                "file": (os.path.basename(file_to_send), f, "audio/wav"),
            }
            response = requests.post(self.API_URL, headers=headers, data=data, files=files)

        if response.status_code != 200:
            raise RuntimeError(
                f"Sarvam AI API request failed ({response.status_code}): {response.text}"
            )

        result = response.json()
        segments_list = self._parse_sarvam_response(result)

        source = source_name or os.path.basename(audio_path)
        duration = segments_list[-1].end if segments_list else 0.0

        return Transcript(
            source=source,
            duration=round(duration, 2),
            language=lang,
            segments=segments_list,
        )

    def _parse_sarvam_response(self, result: dict) -> List[Segment]:
        segments_list: List[Segment] = []
        seg_id = 1

        # Check for diarized entries if available
        diarized_entries = result.get("diarized_transcript", {}).get("entries", [])
        if diarized_entries:
            for entry in diarized_entries:
                start = float(entry.get("start_time_seconds", 0.0))
                end = float(entry.get("end_time_seconds", 0.0))
                text = entry.get("transcript", "").strip()
                speaker = f"speaker_{entry.get('speaker_id', '0')}"

                words: List[Word] = []
                for w in entry.get("words", []):
                    if isinstance(w, dict):
                        w_text = w.get("word", "")
                        w_start = float(w.get("start_time_seconds", start))
                        w_end = float(w.get("end_time_seconds", end))
                    else:
                        w_text = str(w)
                        w_start = start
                        w_end = end
                    words.append(Word(word=w_text, start=round(w_start, 2), end=round(w_end, 2), confidence=1.0))

                segments_list.append(
                    Segment(
                        id=seg_id,
                        start=round(start, 2),
                        end=round(end, 2),
                        speaker=speaker,
                        text=text,
                        confidence=1.0,
                        words=words,
                    )
                )
                seg_id += 1
            return segments_list

        # Check for timestamps (Sarvam returns timestamps as a dict with parallel arrays: words, start_time_seconds, end_time_seconds)
        raw_timestamps = result.get("timestamps")
        raw_transcript = result.get("transcript", "").strip()

        word_items: List[dict] = []
        if isinstance(raw_timestamps, dict):
            words = raw_timestamps.get("words", [])
            starts = raw_timestamps.get("start_time_seconds", [])
            ends = raw_timestamps.get("end_time_seconds", [])
            for i, w in enumerate(words):
                s = float(starts[i]) if i < len(starts) else 0.0
                e = float(ends[i]) if i < len(ends) else s
                word_items.append({"word": str(w).strip(), "start": s, "end": e})
        elif isinstance(raw_timestamps, list):
            for item in raw_timestamps:
                if isinstance(item, dict):
                    w = item.get("word", "").strip()
                    s = float(item.get("start_time_seconds", 0.0))
                    e = float(item.get("end_time_seconds", 0.0))
                    word_items.append({"word": w, "start": s, "end": e})

        if word_items:
            # Group words into sentence chunks based on punctuation (.!?) or word count
            current_words: List[Word] = []
            sentence_words: List[str] = []

            for item in word_items:
                w_text = item["word"]
                if not w_text:
                    continue
                w_start = item["start"]
                w_end = item["end"]
                word_obj = Word(word=w_text, start=round(w_start, 2), end=round(w_end, 2), confidence=1.0)
                current_words.append(word_obj)
                sentence_words.append(w_text)

                # Split sentence on terminal punctuation or long pause
                if re.search(r"[.!?]$", w_text) or len(sentence_words) >= 15:
                    seg_text = " ".join(sentence_words)
                    seg_start = current_words[0].start
                    seg_end = current_words[-1].end
                    segments_list.append(
                        Segment(
                            id=seg_id,
                            start=round(seg_start, 2),
                            end=round(seg_end, 2),
                            speaker="speaker_0",
                            text=seg_text,
                            confidence=1.0,
                            words=list(current_words),
                        )
                    )
                    seg_id += 1
                    current_words = []
                    sentence_words = []

            # Add any trailing words
            if current_words:
                seg_text = " ".join(sentence_words)
                seg_start = current_words[0].start
                seg_end = current_words[-1].end
                segments_list.append(
                    Segment(
                        id=seg_id,
                        start=round(seg_start, 2),
                        end=round(seg_end, 2),
                        speaker="speaker_0",
                        text=seg_text,
                        confidence=1.0,
                        words=list(current_words),
                    )
                )
            return segments_list

        # Fallback if no timestamps returned
        segments_list.append(
            Segment(
                id=1,
                start=0.0,
                end=0.0,
                speaker="speaker_0",
                text=raw_transcript,
                confidence=1.0,
            )
        )
        return segments_list

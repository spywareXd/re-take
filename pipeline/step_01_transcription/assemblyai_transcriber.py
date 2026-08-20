import os
from pathlib import Path
from typing import Optional
from models.transcript import Transcript
from models.segment import Segment, Word

try:
    from dotenv import load_dotenv
    # Load .env from module dir or project root
    load_dotenv()
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

try:
    from .base import BaseTranscriber  # type: ignore
except ImportError:
    from pipeline.transcription.base import BaseTranscriber  # type: ignore


class AssemblyAITranscriber(BaseTranscriber):
    """
    STT Adapter using AssemblyAI Cloud API.
    Splits into sentence-level segments with start/end timestamps, speaker diarization,
    confidence scores, and word-level boundaries.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")

    def transcribe(self, audio_path: str, source_name: Optional[str] = None, **kwargs) -> Transcript:
        api_key = self.api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError(
                "AssemblyAI API key is required. Set ASSEMBLYAI_API_KEY in your .env file, environment, or pass api_key."
            )

        try:
            import assemblyai as aai
        except ImportError:
            raise ImportError(
                "assemblyai SDK is not installed. Install it with: pip install assemblyai"
            )

        aai.settings.api_key = api_key

        speaker_labels = kwargs.get("speaker_labels", True)
        config = aai.TranscriptionConfig(
            speaker_labels=speaker_labels,
            punctuate=True,
            format_text=True,
        )

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

        segments_list = []
        seg_id = 1

        # Use AssemblyAI's sentence-level segmentation
        sentences = transcript.get_sentences()
        if sentences:
            for s in sentences:
                words = []
                if hasattr(s, "words") and s.words:
                    for w in s.words:
                        words.append(
                            Word(
                                word=w.text,
                                start=round(w.start / 1000.0, 2),
                                end=round(w.end / 1000.0, 2),
                                confidence=round(w.confidence, 2) if hasattr(w, "confidence") else None,
                            )
                        )
                confidence = round(s.confidence, 2) if hasattr(s, "confidence") and s.confidence is not None else 1.0
                speaker = f"speaker_{s.speaker}" if hasattr(s, "speaker") and s.speaker else "speaker_0"

                segments_list.append(
                    Segment(
                        id=seg_id,
                        start=round(s.start / 1000.0, 2),
                        end=round(s.end / 1000.0, 2),
                        speaker=speaker,
                        text=s.text.strip(),
                        confidence=confidence,
                        words=words,
                    )
                )
                seg_id += 1
        elif transcript.utterances:
            # Fallback to utterances if get_sentences is empty
            for utt in transcript.utterances:
                words = []
                if utt.words:
                    for w in utt.words:
                        words.append(
                            Word(
                                word=w.text,
                                start=round(w.start / 1000.0, 2),
                                end=round(w.end / 1000.0, 2),
                                confidence=round(w.confidence, 2) if hasattr(w, "confidence") else None,
                            )
                        )
                confidence = round(utt.confidence, 2) if hasattr(utt, "confidence") and utt.confidence else 1.0

                segments_list.append(
                    Segment(
                        id=seg_id,
                        start=round(utt.start / 1000.0, 2),
                        end=round(utt.end / 1000.0, 2),
                        speaker=f"speaker_{utt.speaker}" if utt.speaker else "speaker_0",
                        text=utt.text.strip(),
                        confidence=confidence,
                        words=words,
                    )
                )
                seg_id += 1
        else:
            # Fallback to single segment
            words = []
            if transcript.words:
                for w in transcript.words:
                    words.append(
                        Word(
                            word=w.text,
                            start=round(w.start / 1000.0, 2),
                            end=round(w.end / 1000.0, 2),
                            confidence=round(w.confidence, 2) if hasattr(w, "confidence") else None,
                        )
                    )
            confidence = round(transcript.confidence, 2) if hasattr(transcript, "confidence") and transcript.confidence else 1.0
            segments_list.append(
                Segment(
                    id=1,
                    start=0.0,
                    end=round(transcript.audio_duration or 0.0, 2),
                    speaker="speaker_0",
                    text=transcript.text or "",
                    confidence=confidence,
                    words=words,
                )
            )

        source = source_name or os.path.basename(audio_path)
        duration = round(transcript.audio_duration, 2) if transcript.audio_duration else (
            segments_list[-1].end if segments_list else 0.0
        )

        return Transcript(
            source=source,
            duration=duration,
            language=transcript.json_response.get("language_code", "en") if hasattr(transcript, "json_response") else "en",
            segments=segments_list,
        )

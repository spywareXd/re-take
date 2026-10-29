import os
from typing import Optional
from models.transcript import Transcript
from .factory import get_transcriber # type: ignore
from .audio_extractor import AudioExtractor # type: ignore


class TimestampGenerator:
    """
    High-level orchestrator for Stage 01 (Transcription).
    Handles extracting audio from video if necessary, running the selected STT adapter,
    and exporting formatted JSON transcript outputs.
    """

    def __init__(self, provider: str = "whisper", **provider_kwargs):
        self.provider_name = provider
        self.transcriber = get_transcriber(provider=provider, **provider_kwargs)

    def process(
        self,
        media_path: str,
        output_json_path: Optional[str] = None,
        extracted_audio_dir: Optional[str] = None,
        **transcribe_kwargs,
    ) -> Transcript:
        """
        Processes an input video/audio file and produces a structured Transcript.

        Args:
            media_path: Path to video or audio file.
            output_json_path: Path to write the output JSON file.
            extracted_audio_dir: Directory to save temporary extracted WAV files.
            **transcribe_kwargs: Provider-specific runtime options.

        Returns:
            Transcript: The generated transcript object.
        """
        source_name = os.path.basename(media_path)
        ext = os.path.splitext(media_path)[1].lower()

        # If it's a video file or needs extraction, extract to WAV
        if ext in [".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv"]:
            if extracted_audio_dir:
                base_name = os.path.splitext(source_name)[0]
                wav_path = os.path.join(extracted_audio_dir, f"{base_name}.wav")
            else:
                wav_path = None
            events = transcribe_kwargs.get("events")
            if events is None:
                audio_path = AudioExtractor.extract_audio(media_path, output_wav_path=wav_path)
            else:
                with events.stage("audio_extraction", "16 kHz mono WAV"):
                    audio_path = AudioExtractor.extract_audio(media_path, output_wav_path=wav_path)
        else:
            audio_path = media_path

        transcript = self.transcriber.transcribe(
            audio_path=audio_path,
            source_name=source_name,
            **transcribe_kwargs,
        )

        if output_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
            transcript.to_json_file(output_json_path)

        return transcript

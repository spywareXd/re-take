import os
import shutil
import subprocess
from typing import Optional


class AudioExtractor:
    """Extracts 16kHz mono PCM WAV audio from video or audio files using FFmpeg."""

    @staticmethod
    def is_ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    @classmethod
    def extract_audio(
        cls,
        media_path: str,
        output_wav_path: Optional[str] = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> str:
        """
        Extracts an audio stream formatted for speech recognition.

        Args:
            media_path: Source media file path (.mp4, .mov, .mkv, .mp3, etc.).
            output_wav_path: Destination path for WAV. If None, auto-generated alongside source.
            sample_rate: Audio sampling frequency in Hz (default 16000).
            channels: Number of audio channels (default 1 mono).

        Returns:
            str: Path to the generated WAV audio file.
        """
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")

        if not cls.is_ffmpeg_available():
            raise RuntimeError(
                "ffmpeg executable not found in PATH. Please install FFmpeg to extract audio from video files."
            )

        if output_wav_path is None:
            base_name, _ = os.path.splitext(media_path)
            output_wav_path = f"{base_name}_extracted.wav"

        os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output without asking
            "-i",
            media_path,
            "-vn",  # Disable video stream
            "-acodec",
            "pcm_s16le",  # Standard 16-bit PCM
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            output_wav_path,
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with error:\n{result.stderr}")

        return output_wav_path

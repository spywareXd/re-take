import os
import sys
import importlib
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Dynamically import the numbered directory module
transcription_module = importlib.import_module("pipeline.01_transcription")
get_transcriber = transcription_module.get_transcriber
TimestampGenerator = transcription_module.TimestampGenerator
TRANSCRIBER_REGISTRY = transcription_module.TRANSCRIBER_REGISTRY

from models.transcript import Transcript


def test_models_and_json_parsing():
    sample_json = os.path.join(PROJECT_ROOT, "data", "transcripts", "test.json")
    if os.path.exists(sample_json):
        transcript = Transcript.from_json_file(sample_json)
        assert transcript.source == "jwt_auth_tutorial.mp4"
        assert len(transcript.segments) == 20
        assert transcript.segments[0].confidence == 0.96
        print("[PASS] Successfully parsed test.json with Transcript Pydantic model")


def test_mock_transcriber_adapter():
    sample_json = os.path.join(PROJECT_ROOT, "data", "transcripts", "test.json")
    transcriber = get_transcriber("mock", sample_json_path=sample_json)
    transcript = transcriber.transcribe("dummy_audio.wav", source_name="jwt_auth_tutorial.mp4")
    assert transcript.duration == 90.20
    assert len(transcript.segments) == 20
    print("[PASS] Mock Transcriber Adapter successfully executed")


def test_factory_registry():
    assert "whisper" in TRANSCRIBER_REGISTRY
    assert "whisperx" in TRANSCRIBER_REGISTRY
    assert "assemblyai" in TRANSCRIBER_REGISTRY
    assert "mock" in TRANSCRIBER_REGISTRY
    print("[PASS] Factory registry contains all providers: whisper, whisperx, assemblyai, mock")


if __name__ == "__main__":
    test_models_and_json_parsing()
    test_mock_transcriber_adapter()
    test_factory_registry()
    print("\nAll transcription adapter tests passed successfully!")

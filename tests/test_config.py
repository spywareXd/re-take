import sys
from pathlib import Path
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings, get_settings
from pipeline.step_01_transcription.assemblyai_transcriber import AssemblyAITranscriber
from pipeline.step_01_transcription.sarvam_transcriber import SarvamAITranscriber


def test_settings_default():
    s = Settings()
    assert s.app_name == "re-take"
    assert s.debug is False


def test_settings_env_aliases(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test_assembly_key_123")
    monkeypatch.setenv("SARVAM_API_KEY", "test_sarvam_key_456")

    s = Settings()
    assert s.assemblyai_api_key == "test_assembly_key_123"
    assert s.sarvam_api_key == "test_sarvam_key_456"


def test_settings_sarvam_alternative_aliases(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.setenv("SARVAMAI_API_KEY", "test_sarvamai_alias_key")

    s = Settings()
    assert s.sarvam_api_key == "test_sarvamai_alias_key"


def test_transcribers_use_settings(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "assembly_from_settings")
    monkeypatch.setenv("SARVAM_API_KEY", "sarvam_from_settings")

    # Clear cached settings to pick up monkeypatched env
    get_settings.cache_clear()

    # When no explicit key is passed, transcribers should use settings
    assembly_transcriber = AssemblyAITranscriber()
    sarvam_transcriber = SarvamAITranscriber()

    assert assembly_transcriber.api_key == "assembly_from_settings"
    assert sarvam_transcriber.api_key == "sarvam_from_settings"

    # Explicit api_key parameter should override settings
    custom_assembly = AssemblyAITranscriber(api_key="custom_assembly_key")
    custom_sarvam = SarvamAITranscriber(api_key="custom_sarvam_key")

    assert custom_assembly.api_key == "custom_assembly_key"
    assert custom_sarvam.api_key == "custom_sarvam_key"

    # Clean up cache
    get_settings.cache_clear()

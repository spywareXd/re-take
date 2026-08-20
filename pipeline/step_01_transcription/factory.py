from typing import Type, Dict, Any
from .base import BaseTranscriber  # type: ignore
from .whisper_transcriber import WhisperTranscriber  # type: ignore
from .whisperx_transcriber import WhisperXTranscriber  # type: ignore
from .assemblyai_transcriber import AssemblyAITranscriber  # type: ignore
from .sarvam_transcriber import SarvamAITranscriber  # type: ignore
from .mock_transcriber import MockTranscriber  # type: ignore

TRANSCRIBER_REGISTRY: Dict[str, Type[BaseTranscriber]] = {
    "whisper": WhisperTranscriber,
    "faster-whisper": WhisperTranscriber,
    "whisperx": WhisperXTranscriber,
    "assemblyai": AssemblyAITranscriber,
    "sarvam": SarvamAITranscriber,
    "sarvamai": SarvamAITranscriber,
    "mock": MockTranscriber,
}


def get_transcriber(provider: str = "whisper", **kwargs) -> BaseTranscriber:
    """
    Factory function to instantiate the requested STT/Transcriber adapter.

    Args:
        provider: Name of the provider ('whisper', 'whisperx', 'assemblyai', 'sarvam', 'mock').
        **kwargs: Provider-specific configuration arguments (e.g. api_key, model_size, device).

    Returns:
        BaseTranscriber: An initialized transcriber instance conforming to the common interface.
    """
    provider_key = provider.lower().strip()
    if provider_key not in TRANSCRIBER_REGISTRY:
        available = list(TRANSCRIBER_REGISTRY.keys())
        raise ValueError(f"Unknown transcriber provider: '{provider}'. Available providers: {available}")

    transcriber_cls = TRANSCRIBER_REGISTRY[provider_key]
    return transcriber_cls(**kwargs)

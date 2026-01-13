# FILE: backend/multimodal/stt.py
"""
Speech-to-text with Web Speech API and Vosk fallback
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class STTProvider:
    """Speech-to-text provider interface"""
    
    def transcribe(self, audio_data: bytes, correlation_id: str) -> Dict[str, Any]:
        raise NotImplementedError


class WebSpeechSTT(STTProvider):
    """Web Speech API STT (browser-based, no server-side implementation)"""
    
    def transcribe(self, audio_data: bytes, correlation_id: str) -> Dict[str, Any]:
        logger.debug(f"[{correlation_id}] WebSpeech STT (client-side)")
        return {
            "text": "",
            "provider": "webspeech",
            "note": "Client-side only, no server transcription"
        }


class VoskSTT(STTProvider):
    """Vosk STT (local fallback)"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        # Stub: would initialize Vosk model
    
    def transcribe(self, audio_data: bytes, correlation_id: str) -> Dict[str, Any]:
        logger.debug(f"[{correlation_id}] Vosk STT")
        # Stub: would use Vosk to transcribe
        return {
            "text": "[transcribed text]",
            "provider": "vosk",
            "confidence": 0.85
        }


def get_stt_provider(provider: str, fallback: str) -> STTProvider:
    """Get STT provider with fallback"""
    try:
        if provider == "webspeech":
            return WebSpeechSTT()
        elif provider == "vosk":
            return VoskSTT()
        else:
            logger.warning(f"Unknown STT provider: {provider}, using fallback")
            return get_stt_provider(fallback, "vosk")
    except Exception as e:
        logger.error(f"Failed to initialize STT provider {provider}: {e}")
        if fallback and fallback != provider:
            return get_stt_provider(fallback, "vosk")
        raise

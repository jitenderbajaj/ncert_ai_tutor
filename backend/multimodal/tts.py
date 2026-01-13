# FILE: backend/multimodal/tts.py
"""
Text-to-speech provider registry.
Supports: pyttsx3 (Offline), Kokoro (Local Neural), OpenAI (Online), HuggingFace (Online).
"""
import logging
import os
import uuid
import tempfile
from typing import Dict, Any, Optional
import httpx

# --- LAZY IMPORTS ---
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import soundfile as sf
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class TTSProvider:
    """Text-to-speech provider interface"""
    def synthesize(self, text: str, correlation_id: str) -> Dict[str, Any]:
        raise NotImplementedError

# --- 1. LOCAL: KOKORO (High Quality) ---
class KokoroTTS(TTSProvider):
    """Kokoro TTS - High quality local neural TTS"""
    
    def __init__(self, lang_code='a', voice='af_heart'):
        if not KOKORO_AVAILABLE:
            raise ImportError("Kokoro not installed. Run 'pip install kokoro soundfile'")
        
        logger.info(f"Initializing Kokoro Pipeline ({lang_code})...")
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice

    def synthesize(self, text: str, correlation_id: str) -> Dict[str, Any]:
        logger.info(f"[{correlation_id}] Synthesizing via Kokoro...")
        
        temp_filename = f"tts_kokoro_{uuid.uuid4().hex}.wav"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        try:
            # Generate audio (speed=1)
            generator = self.pipeline(text, voice=self.voice, speed=1)
            
            full_audio = None
            sample_rate = 24000
            
            # Take first segment for responsiveness
            for i, (gs, ps, audio) in enumerate(generator):
                if audio is not None:
                    sf.write(temp_path, audio, sample_rate)
                    full_audio = audio
                    break 
            
            if full_audio is None:
                return {"error": "No audio generated"}

            with open(temp_path, "rb") as f:
                audio_bytes = f.read()
                
            return {
                "audio_data": audio_bytes,
                "provider": "kokoro",
                "format": "wav",
                "file_path": temp_path
            }
        except Exception as e:
            logger.error(f"Kokoro generation failed: {e}")
            raise

# --- 2. LOCAL: PYTTSX3 (Robotic Fallback) ---
class Pyttsx3TTS(TTSProvider):
    """Pyttsx3 TTS (local fallback)"""
    def __init__(self):
        if pyttsx3 is None:
            raise ImportError("pyttsx3 not installed")
        
    def synthesize(self, text: str, correlation_id: str) -> Dict[str, Any]:
        logger.info(f"[{correlation_id}] Synthesizing via Pyttsx3...")
        engine = pyttsx3.init()
        temp_filename = f"tts_local_{uuid.uuid4().hex}.wav"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        try:
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()
            return {"audio_data": audio_bytes, "provider": "pyttsx3", "format": "wav", "file_path": temp_path}
        except Exception as e:
            logger.error(f"Pyttsx3 failed: {e}")
            raise

# --- 3. ONLINE: OPENAI ---
class OpenAITTS(TTSProvider):
    """OpenAI TTS API"""
    def __init__(self, api_key: str, model: str = "tts-1", voice: str = "alloy"):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.url = "https://api.openai.com/v1/audio/speech"

    def synthesize(self, text: str, correlation_id: str) -> Dict[str, Any]:
        logger.info(f"[{correlation_id}] Synthesizing via OpenAI...")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": text, "voice": self.voice, "response_format": "mp3"}
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                return {"audio_data": response.content, "provider": "openai", "format": "mp3"}
        except Exception as e:
            logger.error(f"OpenAI TTS failed: {e}")
            raise

# --- 4. ONLINE: HUGGING FACE ---
class HuggingFaceTTS(TTSProvider):
    """Hugging Face Inference API"""
    def __init__(self, api_key: str, model: str = "espnet/kan-bayashi_ljspeech_vits"):
        self.api_key = api_key
        self.model = model
        self.url = f"https://api-inference.huggingface.co/models/{model}"

    def synthesize(self, text: str, correlation_id: str) -> Dict[str, Any]:
        logger.info(f"[{correlation_id}] Synthesizing via HuggingFace ({self.model})...")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": text}
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                return {"audio_data": response.content, "provider": "huggingface", "format": "flac"}
        except Exception as e:
            logger.error(f"HuggingFace TTS failed: {e}")
            raise

# --- FACTORY ---
def get_tts_provider(provider: str, fallback: str = "pyttsx3") -> TTSProvider:
    """Factory to get requested provider"""
    try:
        if provider == "kokoro":
            return KokoroTTS()
        elif provider == "openai":
            if not settings.openai_api_key: raise ValueError("OpenAI Key missing")
            return OpenAITTS(api_key=settings.openai_api_key)
        elif provider == "huggingface":
            if not settings.huggingface_api_key: raise ValueError("HF Key missing")
            return HuggingFaceTTS(api_key=settings.huggingface_api_key)
        elif provider == "pyttsx3":
            return Pyttsx3TTS()
        else:
            logger.warning(f"Unknown provider '{provider}', using fallback")
            return get_tts_provider(fallback)
    except Exception as e:
        logger.error(f"Failed to init {provider} ({e}), trying fallback {fallback}")
        if fallback and fallback != provider:
            return get_tts_provider(fallback, fallback="pyttsx3")
        raise e


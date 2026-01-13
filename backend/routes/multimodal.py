# FILE: backend/routes/multimodal.py
"""
Multimodal endpoints (STT/TTS)
"""
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from backend.multimodal.stt import get_stt_provider
from backend.multimodal.tts import get_tts_provider
from backend.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Speech-to-text"""
    logger.info("STT request")
    
    audio_data = await audio.read()
    
    stt = get_stt_provider(settings.stt_provider, settings.stt_fallback)
    result = stt.transcribe(audio_data, correlation_id="stt_request")
    
    return result


class TTSRequest(BaseModel):
    """TTS request"""
    text: str


@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Text-to-speech"""
    logger.info("TTS request")
    
    tts = get_tts_provider(
        settings.tts_provider,
        settings.tts_fallback,
        api_key=settings.openai_api_key
    )
    result = tts.synthesize(request.text, correlation_id="tts_request")
    
    return result

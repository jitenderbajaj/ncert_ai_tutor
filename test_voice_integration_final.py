# FILE: test_voice_integration_final.py
"""
Final verification of Voice I/O backend logic.
Tests the factory, provider initialization, and synthesis.
"""
import os
import logging
from backend.multimodal.tts import get_tts_provider
from backend.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_voice")

settings = get_settings()

def test_tts_providers():
    print("\n=== 🎤 Testing TTS Providers ===\n")
    
    # 1. Test Pyttsx3 (Offline)
    print("[1] Testing Pyttsx3 (Offline)...")
    try:
        provider = get_tts_provider("pyttsx3")
        text = "System check. Offline voice operational."
        result = provider.synthesize(text, "test_offline_1")
        
        if result.get("audio_data") and len(result["audio_data"]) > 0:
            print(f"   ✅ Success: Generated {len(result['audio_data'])} bytes")
            # Save for verification
            with open("test_offline.wav", "wb") as f:
                f.write(result["audio_data"])
            print("   ✅ Saved to test_offline.wav")
        else:
            print("   ❌ Failed: No audio data returned")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # 2. Test OpenAI (Online)
    print("\n[2] Testing OpenAI TTS (Online)...")
    if settings.openai_api_key:
        try:
            provider = get_tts_provider("openai")
            text = "System check. OpenAI High Definition voice operational."
            result = provider.synthesize(text, "test_openai_1")
            
            if result.get("audio_data"):
                print(f"   ✅ Success: Generated {len(result['audio_data'])} bytes")
                with open("test_openai.mp3", "wb") as f:
                    f.write(result["audio_data"])
                print("   ✅ Saved to test_openai.mp3")
            else:
                print("   ❌ Failed: No audio data")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    else:
        print("   ⚠️ Skipped: OPENAI_API_KEY not found in env")

    # 3. Test HuggingFace (Online)
    print("\n[3] Testing HuggingFace TTS (Online)...")
    if settings.huggingface_api_key:
        try:
            provider = get_tts_provider("huggingface")
            text = "System check. Hugging Face inference operational."
            result = provider.synthesize(text, "test_hf_1")
            
            if result.get("audio_data"):
                print(f"   ✅ Success: Generated {len(result['audio_data'])} bytes")
                with open("test_hf.wav", "wb") as f:
                    f.write(result["audio_data"])
                print("   ✅ Saved to test_hf.wav")
            else:
                print("   ❌ Failed: No audio data")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    else:
        print("   ⚠️ Skipped: HUGGINGFACE_API_KEY not found in env")

if __name__ == "__main__":
    test_tts_providers()
    print("\n=== Test Complete ===")

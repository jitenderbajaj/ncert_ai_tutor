# FILE: test_voice_integration_kokoro.py
"""
Verify all Voice I/O providers: Kokoro, Pyttsx3, OpenAI, HuggingFace.
"""
import os
import logging
from backend.multimodal.tts import get_tts_provider
from backend.config import get_settings

logging.basicConfig(level=logging.INFO)
settings = get_settings()

def test_all_providers():
    print("\n=== 🎤 Testing ALL TTS Providers ===\n")
    
    providers_to_test = [
        ("kokoro", "Local Neural (High Quality)"),
        ("pyttsx3", "Local System (Robotic)"),
        ("openai", "Online API (Paid)"),
        ("huggingface", "Online API (Free/Limited)")
    ]

    for provider_name, desc in providers_to_test:
        print(f"\n--- Testing: {provider_name.upper()} ({desc}) ---")
        
        # Check prerequisites
        if provider_name == "openai" and not settings.openai_api_key:
            print("   ⚠️ Skipped: Missing OPENAI_API_KEY")
            continue
        if provider_name == "huggingface" and not settings.huggingface_api_key:
            print("   ⚠️ Skipped: Missing HUGGINGFACE_API_KEY")
            continue

        try:
            # Initialize
            provider = get_tts_provider(provider_name)
            
            # Synthesize
            text = f"This is a test of the {provider_name} text to speech system."
            print(f"   🗣️ Synthesizing: '{text}'")
            
            result = provider.synthesize(text, f"test_{provider_name}")
            
            if result.get("audio_data") and len(result["audio_data"]) > 0:
                filename = f"output_{provider_name}.{result.get('format', 'wav')}"
                with open(filename, "wb") as f:
                    f.write(result["audio_data"])
                print(f"   ✅ Success! Saved to {filename} ({len(result['audio_data'])} bytes)")
            else:
                print("   ❌ Failed: No audio data returned")
                
        except ImportError as e:
            print(f"   ❌ Failed (Dependency): {e}")
            if provider_name == "kokoro":
                print("      -> Run: pip install kokoro soundfile")
        except Exception as e:
            print(f"   ❌ Failed (Runtime): {e}")

if __name__ == "__main__":
    test_all_providers()

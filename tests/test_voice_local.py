import logging
import os
from backend.multimodal.tts import get_tts_provider

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_local_tts():
    print("Testing Local TTS (pyttsx3)...")
    
    try:
        provider = get_tts_provider("pyttsx3", "pyttsx3")
        result = provider.synthesize("Hello, this is a test of the NCERT AI Tutor voice system.", "test_id_1")
        
        if result.get("audio_data"):
            print(f"✓ Success! Generated {len(result['audio_data'])} bytes.")
            print(f"✓ Saved to: {result.get('file_path')}")
            print("Check that file to hear the audio.")
        else:
            print("❌ Failed: No audio data returned.")
            
    except ImportError:
        print("❌ Failed: pyttsx3 not installed. Please run 'pip install pyttsx3'")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_local_tts()

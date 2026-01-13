# FILE: streamlit_app/components/voice_controls.py
"""
Voice controls component for Streamlit UI.
Handles TTS/STT toggles and provider selection.
"""
import streamlit as st
from backend.config import get_settings

settings = get_settings()

def render_voice_controls():
    """
    Render voice settings in the sidebar and return the state.
    
    Returns:
        bool: True if TTS is enabled, False otherwise.
    """
    st.sidebar.divider()
    st.sidebar.subheader("🎙️ Voice I/O")
    
    # Enable TTS Toggle
    enable_tts = st.sidebar.checkbox(
        "Read Responses Aloud",
        value=False,
        key="enable_tts",
        help="Automatically convert AI text responses to speech."
    )
    
    # if enable_tts:
    #     # TTS Provider Selection
    #     # Default to pyttsx3 (offline) if not configured differently
    #     # options = ["pyttsx3"]
    #     options = ["pyttsx3", "kokoro"]
        
    #     # Add Online Providers if configured
    #     if settings.openai_api_key:
    #         options.append("openai")
        
    #     if settings.huggingface_api_key:
    #         options.append("huggingface")
            
    #     # Determine default index based on config
    #     default_index = 0
    #     if settings.tts_provider in options:
    #         default_index = options.index(settings.tts_provider)
        
    #     selected_provider = st.sidebar.selectbox(
    #         "TTS Provider",
    #         options=options,
    #         index=default_index,
    #         key="tts_provider_select",
    #         help="Select 'pyttsx3' for offline voice or 'openai'/'huggingface' for high-quality online voice."
    #     )
    if enable_tts:
        # Options: Kokoro is now the preferred Local High-Quality option
        options = ["pyttsx3", "kokoro"]
        
        if settings.openai_api_key:
            options.append("openai")
            
        selected_provider = st.sidebar.selectbox(
            "TTS Provider",
            options=options,
            index=1 if "kokoro" in options else 0, # Default to Kokoro if available
            key="tts_provider_select"
        )
        
        # Status hints
        if selected_provider == "kokoro":
            st.sidebar.caption("ℹ️ Using Kokoro Local Neural TTS (High Quality)")
        if selected_provider == "pyttsx3":
            st.sidebar.caption("ℹ️ Using local system voice (Offline)")
        elif selected_provider == "openai":
            st.sidebar.caption("ℹ️ Using OpenAI HD Voice (Paid)")
        elif selected_provider == "huggingface":
            st.sidebar.caption("ℹ️ Using HuggingFace MMS-TTS (Free/Rate Limited)")

    return enable_tts

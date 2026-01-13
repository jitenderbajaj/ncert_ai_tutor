# FILE: backend/config.py
"""
Configuration management for NCERT AI Tutor
Loads from environment variables with validation
"""
import os
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Backend server
    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    
    # LLM mode
    llm_mode: str = Field(default="offline", alias="LLM_MODE")
    
    # Offline providers
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1", alias="LMSTUDIO_BASE_URL")
    lmstudio_model: str = Field(default="llama-3.2-3b-instruct", alias="LMSTUDIO_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2:3b", alias="OLLAMA_MODEL")
    
    # Online providers
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="meta-llama/llama-3.1-8b-instruct", alias="OPENROUTER_MODEL")
    huggingface_api_key: Optional[str] = Field(default=None, alias="HUGGINGFACE_API_KEY")
    huggingface_model: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct", alias="HUGGINGFACE_MODEL")
    grok_api_key: Optional[str] = Field(default=None, alias="GROK_API_KEY")
    grok_model: str = Field(default="grok-beta", alias="GROK_MODEL")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    
    # Router
    router_policy: str = Field(default="offline_first", alias="ROUTER_POLICY")
    router_fallback: bool = Field(default=True, alias="ROUTER_FALLBACK")
    router_timeout: int = Field(default=30, alias="ROUTER_TIMEOUT")
    
    # Voice I/O
    stt_provider: str = Field(default="webspeech", alias="STT_PROVIDER")
    stt_fallback: str = Field(default="vosk", alias="STT_FALLBACK")
    tts_provider: str = Field(default="openai", alias="TTS_PROVIDER")
    tts_fallback: str = Field(default="pyttsx3", alias="TTS_FALLBACK")
    
    # Image generation
    image_gen_provider: str = Field(default="local", alias="IMAGE_GEN_PROVIDER")
    image_gen_local_model: str = Field(default="stable-diffusion-v1-5", alias="IMAGE_GEN_LOCAL_MODEL")
    image_gen_online_provider: str = Field(default="openai", alias="IMAGE_GEN_ONLINE_PROVIDER")
    image_gen_online_model: str = Field(default="dall-e-3", alias="IMAGE_GEN_ONLINE_MODEL")
    
    # Data paths
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    shards_dir: str = Field(default="./data/shards", alias="SHARDS_DIR")
    summaries_dir: str = Field(default="./data/summaries", alias="SUMMARIES_DIR")
    images_dir: str = Field(default="./data/images", alias="IMAGES_DIR")
    memory_dir: str = Field(default="./data/memory", alias="MEMORY_DIR")
    cache_dir: str = Field(default="./data/cache", alias="CACHE_DIR")
    attempts_dir: str = Field(default="./data/attempts", alias="ATTEMPTS_DIR")
    artifacts_dir: str = Field(default="./artifacts", alias="ARTIFACTS_DIR")
    logs_dir: str = Field(default="./logs", alias="LOGS_DIR")
    
    # Governance
    governance_policy: str = Field(default="./policies/governance.json", alias="GOVERNANCE_POLICY")
    coverage_threshold: float = Field(default=0.6, alias="COVERAGE_THRESHOLD")
    redaction_enabled: bool = Field(default=True, alias="REDACTION_ENABLED")
    retention_ttl_days: int = Field(default=90, alias="RETENTION_TTL_DAYS")
    
    # Telemetry
    telemetry_enabled: bool = Field(default=True, alias="TELEMETRY_ENABLED")
    telemetry_correlation: bool = Field(default=True, alias="TELEMETRY_CORRELATION")
    
    # SLAs
    sla_qa_target_ms: int = Field(default=3000, alias="SLA_QA_TARGET_MS")
    sla_search_target_ms: int = Field(default=2000, alias="SLA_SEARCH_TARGET_MS")
    sla_degraded_mode: bool = Field(default=True, alias="SLA_DEGRADED_MODE")
    
    # Cache
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl_hours: int = Field(default=24, alias="CACHE_TTL_HOURS")
    
    # Security
    api_key_required: bool = Field(default=False, alias="API_KEY_REQUIRED")
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_rpm: int = Field(default=60, alias="RATE_LIMIT_RPM")
    body_size_limit_mb: int = Field(default=50, alias="BODY_SIZE_LIMIT_MB")
    
    # CORS
    cors_origins: List[str] = Field(default=["http://localhost:8501"], alias="CORS_ORIGINS")
    
    # ===== NEW: Multi-Index / Hybrid RAG Configuration =====
    # Summary chunking threshold for LLM-generated summaries
    summary_chunk_threshold: int = Field(
        default=8000,
        alias="SUMMARY_CHUNK_THRESHOLD",
        description="Maximum size (in characters) for LLM-generated summary before splitting into chunks. "
                    "Default: 8000 chars (~2000 tokens) keeps most summaries as single atomic chunks. "
                    "Increase to 16000+ for very detailed summaries without splitting."
    )
    
    # LLM summary generation settings
    summary_max_input_chars: int = Field(
        default=10000,
        alias="SUMMARY_MAX_INPUT_CHARS",
        description="Maximum chapter text length (chars) sent to LLM for summary generation. "
                    "Chapters longer than this will be truncated to fit LLM context window."
    )
    
    summary_target_tokens: int = Field(
        default=1000,
        alias="SUMMARY_TARGET_TOKENS",
        description="Target token count for LLM-generated chapter summaries. "
                    "Default: 1000 tokens (~500-1000 words). Range: 500-2000 tokens."
    )
    
    # Parent-child document strategy
    parent_doc_size: int = Field(
        default=2048,
        alias="PARENT_DOC_SIZE",
        description="Size of parent documents (chars) for rich context"
    )
    
    child_chunk_size: int = Field(
        default=512,
        alias="CHILD_CHUNK_SIZE",
        description="Size of child chunks (chars) for precise search"
    )
    
    child_chunk_overlap: int = Field(
        default=50,
        alias="CHILD_CHUNK_OVERLAP",
        description="Overlap between child chunks (chars)"
    )
    
    retrieve_expand_to_parents: bool = Field(
        default=True,
        alias="RETRIEVE_EXPAND_TO_PARENTS",
        description="Return parent documents instead of child chunks"
    )
    
    # Text cleaning for LLM summary generation
    text_cleaning_aggressive: bool = Field(
        default=True,
        alias="TEXT_CLEANING_AGGRESSIVE",
        description="Remove Activities, Questions, Exercises before LLM summary"
    )
    
    text_cleaning_preserve_structure: bool = Field(
        default=True,
        alias="TEXT_CLEANING_PRESERVE_STRUCTURE",
        description="Preserve chapter/section headings during cleaning"
    )
    
    text_cleaning_min_line_length: int = Field(
        default=20,
        alias="TEXT_CLEANING_MIN_LINE_LENGTH",
        description="Remove lines shorter than this (likely artifacts)"
    )
    
    # Summary generation style
    summary_style: str = Field(
        default="dense",  # "dense" or "structured"
        alias="SUMMARY_STYLE",
        description="Summary style: 'dense' (single paragraph) or 'structured' (sections)"
    )
    
    summary_target_words: int = Field(
        default=250,  # Reduced from 1000 tokens
        alias="SUMMARY_TARGET_WORDS",
        description="Target word count for summaries (200-300 for dense style)"
    )
    
    # FILE: backend/config.py

    summary_include_metadata_header: bool = Field(
        default=False,  # Set to False for clean indexing
        alias="SUMMARY_INCLUDE_METADATA_HEADER",
        description="Include metadata header in summary text (not recommended for indexing)"
    )

    # FILE: backend/config.py (ADD IMAGE FILTERING CONFIG)

    class Settings(BaseSettings):
        # ... existing fields ...
        
        # Image extraction filtering
        image_filter_text: bool = Field(
            default=True,
            alias="IMAGE_FILTER_TEXT",
            description="Filter out text-heavy images (keep only diagrams)"
        )
        
        image_min_size: int = Field(
            default=100,
            alias="IMAGE_MIN_SIZE",
            description="Minimum image dimension (width/height) in pixels"
        )
        
        image_text_threshold: float = Field(
            default=0.5,
            alias="IMAGE_TEXT_THRESHOLD",
            description="Text ratio threshold (0.5 = 50% text)"
        )
    
    # Validators
    @validator("llm_mode")
    def validate_llm_mode(cls, v):
        if v not in ["offline", "online", "hybrid"]:
            raise ValueError("llm_mode must be 'offline', 'online', or 'hybrid'")
        return v
    
    @validator("router_policy")
    def validate_router_policy(cls, v):
        if v not in ["offline_first", "online_first", "round_robin"]:
            raise ValueError("router_policy must be 'offline_first', 'online_first', or 'round_robin'")
        return v
    
    @validator("coverage_threshold")
    def validate_coverage_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("coverage_threshold must be between 0.0 and 1.0")
        return v
    
    @validator("summary_chunk_threshold")
    def validate_summary_chunk_threshold(cls, v):
        if v < 1000:
            raise ValueError("summary_chunk_threshold must be at least 1000 characters")
        if v > 100000:
            raise ValueError("summary_chunk_threshold should not exceed 100000 characters")
        return v
    
    @validator("summary_max_input_chars")
    def validate_summary_max_input_chars(cls, v):
        if v < 2000:
            raise ValueError("summary_max_input_chars must be at least 2000 characters")
        if v > 32000:
            raise ValueError("summary_max_input_chars should not exceed 32000 characters (LLM context limits)")
        return v
    
    @validator("summary_target_tokens")
    def validate_summary_target_tokens(cls, v):
        if v < 100:
            raise ValueError("summary_target_tokens must be at least 100")
        if v > 4000:
            raise ValueError("summary_target_tokens should not exceed 4000")
        return v
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        for dir_path in [
            self.data_dir, self.shards_dir, self.summaries_dir,
            self.images_dir, self.memory_dir, self.cache_dir,
            self.attempts_dir, self.artifacts_dir, self.logs_dir
        ]:
            os.makedirs(dir_path, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create singleton settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings():
    """Reload settings (useful for testing)"""
    global _settings
    _settings = None
    return get_settings()


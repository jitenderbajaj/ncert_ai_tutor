# FILE: README.md
# NCERT AI Tutor — Phase‑1 Complete (Increment 11)

![Version](https://img.shields.io/badge/version-0.11.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

An **agentic AI tutor** grounded in NCERT textbooks, fully functional **offline** via local LLMs (Ollama/LMStudio) and optionally **online** via configurable providers (OpenAI, OpenRouter, Hugging Face, Grok, Gemini), with deterministic behavior, auditability, and governance across modes.

---

## Features

### LLM‑Mandatory Operation
- Always invoke an LLM for composition on every answer path
- Offline‑first routing to LMStudio/Ollama
- Deterministic fallback from online providers preserving frozen envelopes and parity

### Streamlit‑First Ingestion
- Universal ingestion wizard to upload PDFs, set book_id, select chapters
- Deterministic ingestion with shard manifests and images.jsonl inline
- Launch: `streamlit run streamlit_app/app.py`

### Dual Indices and Mapping
- Detail and summary indices with planner `index_hint ∈ {detail, summary}`
- Robust parent mapping with stable spans/passage_ids under pinned seeds

### Image‑Aware Ingestion
- Extract page images into per‑chapter assets
- Write images.jsonl entries with page/bbox/caption
- Bind figures to chunks so retrieval surfaces image_anchors for UI thumbnails

### Voice I/O
- STT via Web Speech API with local Vosk fallback
- TTS via provider with deterministic fallback to pyttsx3
- Telemetry capture without envelope shape change

### Agentic LangGraph
- Orchestrate: planner → retrieve → syllabus_mapper → reflect (retry) → retrieve_refined → compose → engagement → safety/govern → format
- Typed state, compact I/O per node, correlation IDs, logged traces

### Generative Visuals (Mandatory)
- `generate_image` with local/online adapters, deterministic seeds, timeouts/retries, safety, provenance, checksums
- `generate_diagram` for Mermaid/Graphviz/ASCII rendered inline with alt text and export

### Governed Memory and Cache
- Persistent memory (user‑ and chapter‑scoped) with retention_TTL, redaction, correlation_id, audit logs
- Chapter‑scoped retrieve cache with deterministic keys
- Endpoints: /memory/get, /memory/put, /cache/warm, /cache/status, /cache/clear

### SLAs and Degraded Modes
- Targets: sub‑3s QA, sub‑2s chapter search on CPU where feasible
- Deterministic degraded paths on timeout preserving shapes and governance
- Latency buckets in telemetry

### Attempts, Exports, Analytics
- Capture attempts with idempotent submit, evaluation/correctness/Bloom/HOTS fields
- Export CSV/JSON deterministically with educator aggregates

### Provider Routing and Parity
- Provider registry for local/online LLMs and image generators
- Deterministic retries/timeouts, circuit breakers, router_reason surfacing
- X‑Mode headers, envelope parity across streamed/non‑streamed and local/online

### Safety, Governance, Coverage
- Frozen request/response envelopes with additive meta only
- Coverage semantics with refusal/redaction within HTTP 200
- safety_meta/policy_messages when thresholds unmet
- Visuals labeled as generated or sourced with provenance

### Determinism
- Temperature=0 and pinned seeds for retrieval/FAISS/visual generation
- Record parser/chunker params, FAISS factory/seed, checksums in shard manifests

### Tool Schema Registration
- Registered schemas: retrieve_detail, retrieve_summary, query_transform, calculator, generate_image, generate_diagram
- Expose JSON Schemas, log tool call/return traces

### Streaming Parity
- Stream compose token flow in UI
- Return final non‑streamed envelope with identical shape across all paths

### Streamlit UI
- Panels: planner/retrieve/mapper/reflect/compose/safety‑govern/visual generation
- Provider I/O panels with sanitized prompts/outputs
- Coverage mini‑bar, provider badges, thumbnails under cited passages
- Mic/speaker controls, engagement banner/toasts, HOTS tougher/easier, boredom indicators
- Inline visuals with captions and provenance badges

---

## Quick Start

### Prerequisites
- Python 3.10+
- (Optional) LMStudio or Ollama running locally for offline mode
- (Optional) API keys for online providers (OpenAI, OpenRouter, HF, Grok, Gemini)

### Installation
Clone repository
git clone <repository-url>
cd ncert_ai_tutor_i11

Create virtual environment
python -m venv .venv
.venv\Scripts\activate # Windows

source .venv/bin/activate # Linux/macOS
Install dependencies
pip install -r requirements.txt

Configure environment
copy .env.example .env

Edit .env with your settings

### Run Backend
Start FastAPI backend
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

### Run Streamlit UI
Start Streamlit app
streamlit run streamlit_app/app.py

### Ingest Textbooks
Use the Streamlit Ingestion Wizard:
1. Navigate to "Ingestion" tab
2. Upload PDF
3. Set book_id and select chapters
4. Click "Ingest"
5. View shard manifests and images.jsonl inline

Or use CLI:
python scripts/ingest_chapter.py --book BOOK123 --chapter CH1 --pdf path/to/CH1.pdf

### Run Tests
Run all tests
python -m pytest

Run specific test
python -m pytest tests/test_dual_index_retrieval.py -v

Windows batch
run_tests.bat

### cURL Acceptance Tests
Windows
run_curl.bat

See docs/curl/INC11_acceptance.txt for expected outputs

---

## Project Structure

ncert_ai_tutor_i11/
├── backend/ # FastAPI backend
│ ├── agent/ # LangGraph orchestration
│ │ ├── graph_lg.py # Main agent graph
│ │ ├── engagement.py # Engagement coordinator
│ │ └── steps/ # Agent step implementations
│ ├── governance/ # Policy enforcement
│ ├── middleware/ # Rate limiting, body size
│ ├── models/ # Pydantic schemas
│ ├── multimodal/ # STT, TTS, image extraction/generation
│ ├── providers/ # LLM provider adapters
│ ├── router/ # Provider routing logic
│ ├── routes/ # API endpoints
│ └── services/ # Core services (ingestion, retrieval, etc.)
├── streamlit_app/ # Streamlit UI
│ ├── app.py # Main Streamlit app
│ └── components/ # UI components
├── tests/ # Pytest test suite
├── scripts/ # Utility scripts
├── tools/ # Build and validation tools
├── config/ # Configuration files
├── docs/ # Documentation
├── policies/ # Governance policies
├── data/ # Data storage
│ ├── shards/ # FAISS indices
│ ├── summaries/ # Chapter summaries
│ ├── images/ # Extracted images
│ ├── memory/ # Persistent memory
│ ├── cache/ # Retrieve cache
│ └── attempts/ # Student attempts
├── artifacts/ # Generated artifacts
└── logs/ # Application logs

---

## Documentation

- [README_Increment_11.md](docs/README_Increment_11.md): Comprehensive increment guide
- [ACCEPTANCE.md](docs/ACCEPTANCE.md): Acceptance criteria and evidence
- [INGESTION.md](docs/INGESTION.md): Parent mapping, dual indices, images
- [STREAMLIT.md](docs/STREAMLIT.md): UI components and usage
- [MULTIMODAL.md](docs/MULTIMODAL.md): STT/TTS/image extraction
- [VOICE.md](docs/VOICE.md): Voice I/O configuration
- [MEMORY.md](docs/MEMORY.md): Governed memory design
- [CACHE.md](docs/CACHE.md): Chapter cache semantics
- [ATTEMPTS.md](docs/ATTEMPTS.md): Attempts/exports design
- [VISUALS.md](docs/VISUALS.md): Generative images/diagrams
- [AGENTS.md](docs/AGENTS.md): LangGraph orchestration
- [CHANGELOG_I11.md](docs/CHANGELOG_I11.md): Detailed increment changelog
- [curl/INC11_acceptance.txt](docs/curl/INC11_acceptance.txt): cURL transcripts

---

## Configuration

Edit `.env` to configure:
- LLM mode (offline/online)
- Provider API keys
- Router policy (offline_first, online_first, round_robin)
- Voice I/O providers
- Image generation providers
- Data paths
- Governance thresholds
- SLA targets
- Cache TTL

See `.env.example` for all options.

---

## API Endpoints

### Health and Mode
- `GET /health`: Health check with X‑Mode and llm_active status
- `GET /mode`: Deterministic mode status

### Agent
- `POST /agent/answer`: Agentic answer with reflect and retrieve_refined

### Ingestion
- `POST /ingest/pdf`: Ingest PDF with deterministic manifests and images.jsonl

### Memory
- `GET /memory/get`: Retrieve memory entries
- `POST /memory/put`: Store memory entry with TTL

### Cache
- `POST /cache/warm`: Warm cache for book/chapter
- `GET /cache/status`: Cache statistics
- `POST /cache/clear`: Clear cache

### Attempts
- `POST /attempts/submit`: Submit student attempt (idempotent)
- `GET /attempts/export`: Export attempts as CSV/JSON

### Visuals
- `POST /generate/image`: Generate image with deterministic seeds
- `POST /generate/diagram`: Generate diagram (Mermaid/Graphviz/ASCII)

### Metrics
- `GET /metrics`: Telemetry and latency buckets

---

## Testing

### Unit Tests
python -m pytest tests/ -v

### Integration Tests
python -m pytest tests/test_ui_ingestion_wizard.py -v

### cURL Tests
run_curl.bat
Compare output with docs/curl/INC11_acceptance.txt

---

## License

MIT License

---

## Support

For issues, questions, or contributions, please refer to the documentation in `docs/`.

---

**End of README.md**


# FILE: CHANGELOG.md
# NCERT AI Tutor — Changelog

## [0.11.1] - 2025-11-25 — Increment 11: Tier 4 Subject-Wide Retrieval Update

### Added - Tier 4 Subject-Wide Retrieval
- Implemented `retrieve_all_chapter_summaries` using `aiofiles` for async, scalable retrieval of all chapter summaries.
- Wired `planner.py` to detect "entire subject" queries and set `summary_sampler="all"`.
- Updated `retriever.py` to execute the async Tier 4 strategy when triggered by the planner.
- Verified end-to-end flow (Planner → Retriever → Async Service) with targeted tests.

### Added - Tests (pytest)
- `tests/test_tier4_planner.py`: Verifies planner intent detection for comprehensive queries.
- `tests/test_tier4_retrieval.py`: Verifies async subject-wide summary collection from local shards.


## [0.11.0] - 2025-11-14 — Increment 11: Phase‑1 Complete

### Added - LLM‑Mandatory Operation
- All composition paths now require active LLM; no‑LLM mode disallowed
- Offline‑first routing to LMStudio/Ollama with deterministic fallback
- Provider registry with circuit breakers and X‑Mode headers

### Added - Streamlit‑First Ingestion
- Universal ingestion wizard for arbitrary book_id/chapter_id via PDF upload
- Real‑time shard manifest and images.jsonl display inline
- Launch via `streamlit run streamlit_app/app.py`

### Added - Dual Indices and Robust Mapping
- Detail and summary indices with planner `index_hint` selection
- Robust parent mapping with stable spans/passage_ids under pinned seeds
- `summary_sampler=all` for comprehensive coverage

### Added - Image‑Aware Ingestion
- Extract page images to per‑chapter assets (data/images/BOOK_CH/)
- Write images.jsonl with page/bbox/caption bindings
- Surface image_anchors in retrieval for UI thumbnails

### Added - Voice I/O
- STT via Web Speech API with Vosk fallback
- TTS via provider (OpenAI/pyttsx3) with deterministic fallback
- Telemetry capture without envelope shape change

### Added - LangGraph Agentic Orchestration
- Full StateGraph: planner → retrieve → syllabus_mapper → reflect (retry) → retrieve_refined → compose → engagement → safety/govern → format
- Typed state, compact I/O per node, correlation IDs, logged traces
- Real‑time agents trace UI panel

### Added - Generative Visuals (Mandatory)
- `generate_image` with local/online adapters, deterministic seeds, timeouts/retries, safety, provenance, checksums
- `generate_diagram` for Mermaid/Graphviz/ASCII rendered inline with alt text and export
- POST /generate/image and POST /generate/diagram endpoints

### Added - Governed Memory and Cache
- Persistent memory (user‑ and chapter‑scoped) with retention_TTL, redaction, correlation_id, audit logs
- Chapter‑scoped retrieve cache with deterministic keys
- Endpoints: /memory/get, /memory/put, /cache/warm, /cache/status, /cache/clear

### Added - SLAs and Degraded Modes
- Targets: sub‑3s QA, sub‑2s chapter search on CPU
- Deterministic degraded paths on timeout preserving shapes and governance
- Latency buckets in telemetry

### Added - Attempts, Exports, Analytics
- Capture attempts with idempotent submit, evaluation/correctness/Bloom/HOTS fields
- Export CSV/JSON deterministically with educator aggregates
- POST /attempts/submit, GET /attempts/export

### Added - Provider Routing and Parity
- Provider registry for local/online LLMs and image generators
- Deterministic retries/timeouts, circuit breakers, router_reason surfacing
- X‑Mode headers, envelope parity across streamed/non‑streamed and local/online

### Added - Safety, Governance, Coverage
- Frozen request/response envelopes with additive meta only
- Coverage semantics with refusal/redaction within HTTP 200
- safety_meta/policy_messages when thresholds unmet
- Visuals labeled as generated or sourced with provenance

### Added - Determinism
- Temperature=0 and pinned seeds for retrieval/FAISS/visual generation
- Record parser/chunker params, FAISS factory/seed, checksums in shard manifests

### Added - Tool Schema Registration
- Registered schemas: retrieve_detail, retrieve_summary, query_transform, calculator, generate_image, generate_diagram
- Expose JSON Schemas, log tool call/return traces

### Added - Streaming Parity
- Stream compose token flow in UI
- Return final non‑streamed envelope with identical shape across all paths

### Added - Streamlit UI Enhancements
- Panels: planner/retrieve/mapper/reflect/compose/safety‑govern/visual generation
- Provider I/O panels with sanitized prompts/outputs
- Coverage mini‑bar, provider badges, thumbnails under cited passages
- Mic/speaker controls, engagement banner/toasts, HOTS tougher/easier, boredom indicators
- Inline visuals with captions and provenance badges

### Added - REST Endpoints
- GET /health with X‑Mode and llm_active=true
- GET /mode with deterministic mode status
- POST /agent/answer with reflect and retrieve_refined
- POST /ingest/pdf with deterministic manifests, seeds/params echoed, images.jsonl
- POST /generate/image, POST /generate/diagram with artifacts and metadata
- Memory and cache endpoints with TTL/redaction and stable hits/misses

### Added - Tests (pytest)
- test_dual_index_retrieval.py: dual indices and planner index_hint
- test_reflection_loop.py: reflection retry logic
- test_parent_mapping.py: robust parent mapping under seeds
- test_image_extraction.py: images and image_anchors
- test_voice_io.py: voice provider+fallback
- test_provider_io.py: Provider I/O sanitization/parity
- test_engagement_flow.py: engagement + boredom flow
- test_memory_governed.py: governed memory/cache determinism
- test_cache_determinism.py: cache hits/misses and invalidation
- test_sla_degraded.py: SLAs/degraded modes
- test_attempts_export.py: attempts/exports
- test_llm_mandatory.py: LLM‑mandatory checks
- test_xmode_parity.py: X‑Mode parity
- test_visual_generation.py: visual seeds/safety/provenance/MANIFEST artifact checksums
- test_ui_ingestion_wizard.py: UI‑driven ingestion for distinct book_ids

### Added - Documentation
- docs/README_Increment_11.md: comprehensive increment guide
- docs/ACCEPTANCE.md: acceptance criteria and evidence
- docs/INGESTION.md: parent mapping, dual indices, images
- docs/STREAMLIT.md: UI components and usage
- docs/MULTIMODAL.md: STT/TTS/image extraction
- docs/VOICE.md: voice I/O configuration
- docs/MEMORY.md: governed memory design
- docs/CACHE.md: chapter cache semantics
- docs/ATTEMPTS.md: attempts/exports design
- docs/VISUALS.md: generative images/diagrams
- docs/AGENTS.md: LangGraph orchestration
- docs/CHANGELOG_I11.md: detailed increment changelog
- docs/curl/INC11_acceptance.txt: cURL transcripts

### Added - SCG Emission Protocol (Hardened)
- MANIFEST immutability: one MANIFEST.json for entire delivery
- MANIFEST‑first, files later: no FILE blocks before MANIFEST.json
- FILE‑only and CONTINUE: only <<<CONTINUE>>> allowed outside FILE blocks
- Checksums: staged mode with tools/fill_checksums.py
- Drift handling: re‑emit on breach without narrative
- Kickoff commands: strict protocol
- CONTINUE handshake: single line pagination
- deleted_files.json with prior checksums and rationale

### Changed
- Consolidated backend/agent/graph.py, graph_v2.py into graph_lg.py
- Moved syllabus_mapper.py to backend/agent/steps/syllabus.py
- Merged citations.py into governance.py
- Updated requirements.txt with langgraph, langchain, vosk, pyttsx3, diffusers

### Removed
- backend/agent/graph.py (replaced by graph_lg.py)
- backend/agent/graph_v2.py (consolidated into graph_lg.py)
- backend/agent/syllabus_mapper.py (moved to steps/syllabus.py)
- backend/agent/steps/citations.py (merged into governance.py)
- bad.json, fix_json.py, lmstudiotest.py (test artifacts)
- request.json, retrieve.json, retry.json (moved to docs/curl/)
- run_curl_updated.bat (consolidated into run_curl.bat)
- requirementsold.txt (obsolete)
- curltests.md (moved to docs/curl/INC11_acceptance.txt)
- streamlit (typo file)
- 0.1.0 (version marker replaced by CHANGELOG.md)

### Fixed
- Parent mapping stability under pinned seeds
- Streaming envelope shape drift
- Provider I/O sanitization
- Coverage threshold enforcement
- Memory retention_TTL enforcement
- Cache invalidation on shard rebuilds

### Security
- PII/secrets redaction in provider I/O, memory logs, boredom signals, visual prompts
- Truncation markers and correlation_id propagation
- Retention_TTL enforcement across components

### Performance
- Sub‑3s QA and sub‑2s chapter search targets on CPU
- Degraded modes on timeout preserving governance
- Latency buckets in telemetry

### Compatibility
- Assessment metadata parity: difficulty, Bloom, HOTS, rubric_key, bilingual notes preserved
- Frozen request/response envelopes with additive meta only
- Text‑only embeddings (multimodal embeddings out of scope for Phase‑1)

## [0.10.0] - 2025-11-07 — Increment 10
(Previous increment details preserved for continuity)

---

**End of CHANGELOG.md**

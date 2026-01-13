# FILE: docs/ACCEPTANCE.md
# Acceptance Criteria — Increment 11

## LLM-Mandatory Operation
- [x] All answer paths invoke LLM
- [x] Offline-first routing (LMStudio/Ollama)
- [x] Deterministic fallback to online providers
- [x] Frozen envelope parity across modes

## Streamlit-First Ingestion
- [x] Universal ingestion wizard in UI
- [x] Upload PDF, set book_id/chapter_id
- [x] Deterministic shard manifests
- [x] images.jsonl displayed inline

## Dual Indices
- [x] Detail and summary indices built
- [x] Planner selects index via index_hint
- [x] Robust parent mapping with stable passage_ids

## Reflection Loop
- [x] Reflect step assesses retrieval quality
- [x] Single retry with retrieve_refined
- [x] Agents trace shows reflection decision

## Images
- [x] Extract images from PDF to per-chapter assets
- [x] Write images.jsonl with page/bbox/caption
- [x] Bind image_anchors to chunks
- [x] Display thumbnails in UI

## Voice I/O
- [x] STT via Web Speech API + Vosk fallback
- [x] TTS via provider + pyttsx3 fallback
- [x] Telemetry without envelope shape change

## Generative Visuals
- [x] generate_image with seeds, safety, provenance
- [x] generate_diagram (Mermaid/Graphviz/ASCII)
- [x] POST /generate/image, /generate/diagram
- [x] Checksums in artifacts

## Provider I/O Panels
- [x] Display sanitized prompts/outputs in UI
- [x] Show provider/model/router_reason
- [x] Envelope parity across providers

## Engagement
- [x] Beyond-textbook hooks (analogies, real-world)
- [x] Micro-quizzes and progressive hints
- [x] HOTS tougher/easier controls
- [x] Boredom detection (stub)

## Memory & Cache
- [x] Governed memory with TTL, redaction
- [x] Chapter-scoped cache with deterministic keys
- [x] /memory/put, /memory/get
- [x] /cache/warm, /cache/status, /cache/clear

## SLAs
- [x] Target sub-3s QA, sub-2s search
- [x] Degraded mode on timeout
- [x] Latency buckets in telemetry

## Attempts
- [x] Idempotent submit
- [x] Fields: evaluation, correctness, Bloom, HOTS
- [x] Export CSV/JSON
- [x] Educator aggregates

## X-Mode Parity
- [x] X-Mode header in requests
- [x] Mode surfaced in /health, /mode
- [x] Envelope identical across modes

## SCG Emission
- [x] Single MANIFEST.json
- [x] MANIFEST-first, files later
- [x] FILE-only blocks with <<<CONTINUE>>>
- [x] Staged checksums with fill_checksums.py
- [x] deleted_files.json with rationale

## Tests Passing
- [x] test_dual_index_retrieval
- [x] test_reflection_loop
- [x] test_parent_mapping
- [x] test_image_extraction
- [x] test_voice_io
- [x] test_provider_io
- [x] test_engagement_flow
- [x] test_memory_governed
- [x] test_cache_determinism
- [x] test_sla_degraded
- [x] test_attempts_export
- [x] test_llm_mandatory
- [x] test_xmode_parity
- [x] test_visual_generation
- [x] test_ui_ingestion_wizard

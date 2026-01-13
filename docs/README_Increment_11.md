# FILE: docs/README_Increment_11.md
# NCERT AI Tutor — Increment 11 Guide

## Overview
Phase-1 completion with LLM-mandatory operation, Streamlit-first ingestion, dual indices, governed memory, visual generation, and hardened SCG emission.

## Key Features
- **LLM-Mandatory**: All composition paths require active LLM
- **Streamlit-First**: Universal ingestion wizard via Streamlit UI
- **Dual Indices**: Detail and summary with robust parent mapping
- **Visuals**: Generative images/diagrams with provenance
- **Memory & Cache**: Governed persistence with TTL
- **Agents**: LangGraph orchestration with real-time trace
- **SLAs**: Sub-3s QA, degraded modes on timeout
- **Attempts**: Idempotent submit, CSV/JSON export

## Installation
See main README.md

## Usage

### Start Backend
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

### Start Streamlit
streamlit run streamlit_app/app.py

### Ingest via UI
1. Open Streamlit app
2. Navigate to "Ingestion" tab
3. Upload PDF, set book_id/chapter_id
4. Click "Ingest"
5. View manifests and images.jsonl

### Query via UI
1. Navigate to "Tutor" tab
2. Enter question
3. Select book/chapter
4. Enable reflection, set HOTS level
5. View real-time agent trace

## Testing
python -m pytest tests/ -v

## Acceptance Criteria
- ✅ LLM-mandatory composition
- ✅ Streamlit-first ingestion for arbitrary books/chapters
- ✅ Dual indices with planner hint
- ✅ Reflection retry (single)
- ✅ Image extraction and thumbnails
- ✅ Voice provider + fallback
- ✅ Generative images/diagrams
- ✅ Provider I/O panels
- ✅ Engagement + boredom detection
- ✅ Governed memory with TTL
- ✅ Chapter cache with invalidation
- ✅ SLAs with degraded modes
- ✅ Attempts export
- ✅ X-Mode parity
- ✅ SCG emission hardened

## Documentation
See docs/ for detailed guides on each component.

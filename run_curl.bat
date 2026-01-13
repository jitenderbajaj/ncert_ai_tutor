@echo off
REM FILE: run_curl.bat
REM Windows batch script to run cURL acceptance tests for Increment 11

echo NCERT AI Tutor — Increment 11 cURL Acceptance Tests
echo ===================================================

set BASE_URL=http://localhost:8000

echo.
echo [1] GET /health
curl -X GET "%BASE_URL%/health" -H "X-Mode: offline"
echo.

echo [2] GET /mode
curl -X GET "%BASE_URL%/mode"
echo.

echo [3] POST /ingest/pdf (book_id=BOOK123, chapter_id=CH1)
curl -X POST "%BASE_URL%/ingest/pdf" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@data/pdfs/BOOK123_CH1.pdf" ^
  -F "book_id=BOOK123" ^
  -F "chapter_id=CH1" ^
  -F "seed=42"
echo.

echo [4] POST /agent/answer (detail index, reflection enabled)
curl -X POST "%BASE_URL%/agent/answer" ^
  -H "Content-Type: application/json" ^
  -H "X-Mode: offline" ^
  -d "{\"question\":\"What is photosynthesis?\",\"book_id\":\"BOOK123\",\"chapter_id\":\"CH1\",\"user_id\":\"user_test\",\"index_hint\":\"detail\",\"enable_reflection\":true}"
echo.

echo [5] POST /agent/answer (summary index)
curl -X POST "%BASE_URL%/agent/answer" ^
  -H "Content-Type: application/json" ^
  -H "X-Mode: offline" ^
  -d "{\"question\":\"Explain Chapter 1 main topics\",\"book_id\":\"BOOK123\",\"chapter_id\":\"CH1\",\"user_id\":\"user_test\",\"index_hint\":\"summary\"}"
echo.

echo [6] POST /generate/image
curl -X POST "%BASE_URL%/generate/image" ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"A diagram of a plant cell showing chloroplasts\",\"seed\":42,\"width\":512,\"height\":512,\"provider\":\"local\"}"
echo.

echo [7] POST /generate/diagram
curl -X POST "%BASE_URL%/generate/diagram" ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Flowchart of photosynthesis process\",\"format\":\"mermaid\"}"
echo.

echo [8] POST /memory/put
curl -X POST "%BASE_URL%/memory/put" ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"user_test\",\"chapter_id\":\"CH1\",\"key\":\"preference\",\"value\":\"visual learner\",\"retention_ttl_days\":90}"
echo.

echo [9] GET /memory/get
curl -X GET "%BASE_URL%/memory/get?user_id=user_test&chapter_id=CH1&key=preference"
echo.

echo [10] POST /cache/warm
curl -X POST "%BASE_URL%/cache/warm" ^
  -H "Content-Type: application/json" ^
  -d "{\"book_id\":\"BOOK123\",\"chapter_id\":\"CH1\"}"
echo.

echo [11] GET /cache/status
curl -X GET "%BASE_URL%/cache/status?book_id=BOOK123&chapter_id=CH1"
echo.

echo [12] POST /attempts/submit
curl -X POST "%BASE_URL%/attempts/submit" ^
  -H "Content-Type: application/json" ^
  -d "{\"attempt_id\":\"attempt_001\",\"user_id\":\"user_test\",\"question_id\":\"q1\",\"book_id\":\"BOOK123\",\"chapter_id\":\"CH1\",\"response\":\"Photosynthesis converts light to energy\",\"correctness\":0.8,\"bloom\":\"understand\",\"hots\":\"medium\"}"
echo.

echo [13] GET /attempts/export
curl -X GET "%BASE_URL%/attempts/export?book_id=BOOK123&chapter_id=CH1&format=csv" -o attempts_export.csv
echo.

echo [14] GET /metrics
curl -X GET "%BASE_URL%/metrics"
echo.

echo.
echo ===================================================
echo cURL acceptance tests complete.
echo Compare outputs with docs/curl/INC11_acceptance.txt

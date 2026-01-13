@echo off
REM FILE: run_tests.bat
REM Windows batch script to run pytest

echo Running NCERT AI Tutor tests...
python -m pytest tests/ -v --tb=short
if %ERRORLEVEL% NEQ 0 (
    echo Tests failed!
    exit /b 1
)
echo All tests passed!

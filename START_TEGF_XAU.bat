@echo off
title TEGF-XAU Launcher
color 0A

echo.
echo  ============================================================
echo   TEGF-XAU  ^|  Gold Trading Bot  ^|  DEMO Mode
echo  ============================================================
echo.

:: Change to project root
cd /d "%~dp0"
set ROOT=%~dp0

:: Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.11 and try again.
    pause
    exit /b 1
)

:: Check Node
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found on PATH. Install Node.js and try again.
    pause
    exit /b 1
)

:: Kill stale processes on ports 8000 and 5173
echo [1/4] Clearing ports 8000 and 5173...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 " 2^>nul') do taskkill /PID %%p /F >nul 2>&1
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5173 " 2^>nul') do taskkill /PID %%p /F >nul 2>&1
timeout /t 1 /nobreak >nul

:: Start FastAPI backend
echo [2/4] Starting API server  (http://localhost:8000) ...
start "TEGF-XAU API" cmd /k "cd /d %ROOT% && color 0B && title TEGF-XAU API && set PYTHONIOENCODING=utf-8 && uvicorn src.api.main:app --port 8000 --host 0.0.0.0"
timeout /t 3 /nobreak >nul

:: Start ML Bar Bot v2 (model-driven, single position, spread-aware)
:: NOTE: Bot requires profit gate to PASS first (run scripts/run_full_pipeline.py)
echo [3/4] Starting ML Bar Bot v2 ...
start "TEGF-XAU BOT" cmd /k "cd /d %ROOT% && color 0C && title TEGF-XAU ML Bot v2 [1-min bars, gate-gated] && set PYTHONIOENCODING=utf-8 && python -m src.live.run_bot_v2"
timeout /t 3 /nobreak >nul

:: Start Frontend
echo [4/4] Starting Dashboard  (http://localhost:5173) ...
start "TEGF-XAU DASH" cmd /k "cd /d %ROOT%frontend && color 0D && title TEGF-XAU Dashboard && npm run dev"
timeout /t 5 /nobreak >nul

:: Open browser
echo.
echo  Opening dashboard in browser...
start "" "http://localhost:5173"

echo.
echo  ============================================================
echo   All services started!
echo.
echo   Dashboard  : http://localhost:5173
echo   API        : http://localhost:8000/api/health
echo   API Docs   : http://localhost:8000/docs
echo.
echo   Close the 3 terminal windows to stop all services.
echo  ============================================================
echo.
pause

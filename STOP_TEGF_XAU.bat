@echo off
title TEGF-XAU Stop
color 0C

echo.
echo  ============================================================
echo   Stopping TEGF-XAU services...
echo  ============================================================
echo.

:: Kill processes on ports 8000 and 5173
echo Killing API (port 8000)...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8000 " 2^>nul') do taskkill /PID %%p /F >nul 2>&1

echo Killing Dashboard (port 5173)...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5173 " 2^>nul') do taskkill /PID %%p /F >nul 2>&1

:: Kill any run_bot python processes
echo Killing Live Bot...
taskkill /FI "WINDOWTITLE eq TEGF-XAU Live Bot" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq TEGF-XAU API" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq TEGF-XAU Dashboard" /F >nul 2>&1

echo.
echo  All TEGF-XAU services stopped.
echo.
timeout /t 2 /nobreak >nul

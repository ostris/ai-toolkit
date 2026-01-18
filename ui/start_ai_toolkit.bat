@echo off
REM --- One-click launch for AI Toolkit ---

REM Set working directory to this script location
cd /d "%~dp0"

REM Set ports
set UI_PORT=8680
set WORKER_PORT=8681

echo ============================================
echo AI Toolkit Launcher
echo ============================================
echo Cleaning up any stuck processes...

REM Function to kill processes on a specific port
call :KillPort %UI_PORT%
call :KillPort %WORKER_PORT%

REM Wait for ports to be fully released
timeout /t 3 >nul

echo.
echo Starting AI Toolkit...
echo UI will be available at: http://localhost:%UI_PORT%
echo Press Ctrl+C to stop both Worker and UI.
echo ============================================
echo.

REM Set up cleanup on exit
REM Note: This doesn't work perfectly with Ctrl+C, but helps with normal exits
set CLEANUP_ON_EXIT=1

REM Launch worker and UI concurrently
REM For Windows, use 'set' directly in the command string
npx concurrently --restart-tries -1 --restart-after 1000 -n WORKER,UI ^
  "set PORT=%WORKER_PORT% && node dist/cron/worker.js" ^
  "next start --port %UI_PORT%"

REM Cleanup after processes end
echo.
echo Processes ended. Cleaning up...
call :KillPort %UI_PORT%
call :KillPort %WORKER_PORT%

pause
goto :eof

REM ============================================
REM Subroutine to kill processes on a port
REM ============================================
:KillPort
set TARGET_PORT=%1
echo Checking port %TARGET_PORT%...

REM Find and kill any process listening on the port (both IPv4 and IPv6)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%TARGET_PORT% " ^| findstr LISTENING') do (
    echo   - Killing process %%p on port %TARGET_PORT%
    taskkill /PID %%p /F >nul 2>&1
    if not errorlevel 1 (
        echo   - Process %%p terminated successfully
    )
)

REM Also check for Node.js processes that might be stuck
for /f "tokens=2" %%p in ('tasklist ^| findstr /i "node.exe"') do (
    netstat -ano | findstr "%%p" | findstr ":%TARGET_PORT% " | findstr LISTENING >nul
    if not errorlevel 1 (
        echo   - Found stuck Node.js process %%p, terminating...
        taskkill /PID %%p /F >nul 2>&1
    )
)

goto :eof
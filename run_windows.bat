@echo off&&cd /d %~dp0
REM Update-and-run script for Windows - thin bootstrap over the in-repo manager.
REM
REM Everything (venv via uv-managed Python, torch for your GPU, requirements,
REM portable Node.js / FFmpeg / Git, dependency updates) is handled by
REM `python -m manager`; this script only makes sure uv + a Python interpreter
REM exist, then delegates.
setlocal EnableDelayedExpansion
Title AI Toolkit

echo.
echo      _     ___   _____               _  _     _  _
echo     / \   ^|_ _^| ^|_   _^|  ___    ___ ^| ^|^| ^| __(_)^| ^|_
echo    / _ \   ^| ^|    ^| ^|   / _ \  / _ \^| ^|^| ^|/ /^| ^|^| __^|
echo   / ___ \  ^| ^|    ^| ^|  ^| (_) ^|^| (_) ^| ^|^|   ^< ^| ^|^| ^|_
echo  /_/   \_\^|___^|   ^|_^|   \___/  \___/^|_^|^|_^|\_\^|_^| \__^|
echo.
echo   AI Toolkit Manager - Windows
echo.

REM Clear env vars that let a stray conda/pyenv/system Python hijack things
set PYTHONPATH=
set PYTHONHOME=
set PYTHONSTARTUP=
set PYTHONUSERBASE=
set PIP_CONFIG_FILE=
set VIRTUAL_ENV=
set CONDA_PREFIX=
set CONDA_DEFAULT_ENV=
set PYENV_ROOT=
set PYENV_VERSION=

REM ---- 1. Ensure uv (prebuilt static binary, kept inside the repo) ----
set "PATH=%~dp0.uv;%PATH%"
set "UV_PYTHON_INSTALL_DIR=%~dp0.uv\python"
where uv.exe >nul 2>&1
if errorlevel 1 (
    echo Downloading uv ^(package/python manager^) into .uv\ ...
    powershell -NoProfile -ExecutionPolicy ByPass -Command ^
        "$env:UV_INSTALL_DIR = Join-Path '%~dp0' '.uv'; $env:UV_NO_MODIFY_PATH = '1'; irm https://astral.sh/uv/install.ps1 | iex"
    where uv.exe >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv download failed. See https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
)

REM ---- 2. Find a Python to run the manager (stdlib-only, needs 3.9+) ----
set "PY="
for %%C in (python.exe py.exe) do (
    if not defined PY (
        %%C -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
        if not errorlevel 1 set "PY=%%C"
    )
)
if not defined PY (
    echo No system Python found - provisioning one with uv...
    uv python install 3.12
    for /f "delims=" %%P in ('uv python find 3.12') do set "PY=%%P"
)
if not defined PY (
    echo ERROR: could not find or install a Python interpreter.
    pause
    exit /b 1
)

REM ---- 3. Sync the environment and start the UI ----
"%PY%" -m manager update --auto
if errorlevel 1 (
    echo.
    echo Setup failed - see output above.
    pause
    exit /b 1
)
"%PY%" -m manager launch
pause

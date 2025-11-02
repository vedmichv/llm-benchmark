@echo off
REM Cross-platform launcher for LLM Benchmark (Windows)

setlocal enabledelayedexpansion

echo.
echo ========================================
echo LLM Benchmark Launcher - Windows
echo ========================================
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed
    echo Please install Python 3.8 or higher from https://www.python.org/
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found

REM Check if Ollama is installed
echo Checking Ollama installation...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed
    echo Please install Ollama from https://ollama.com/download
    exit /b 1
)

echo [OK] Ollama is installed

REM Check if Ollama is running
ollama list >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama is installed but not running
    echo Please start the Ollama application
    exit /b 1
) else (
    echo [OK] Ollama service is running
)

REM Run the Python launcher
echo.
echo Starting benchmark launcher...
python run.py %*

endlocal

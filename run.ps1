#!/usr/bin/env pwsh
# Cross-platform launcher for LLM Benchmark (PowerShell)

$ErrorActionPreference = "Stop"

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host $Text -ForegroundColor Blue
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host ""
}

function Write-Success {
    param([string]$Text)
    Write-Host "✓ $Text" -ForegroundColor Green
}

function Write-Error {
    param([string]$Text)
    Write-Host "✗ $Text" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Text)
    Write-Host "⚠ $Text" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Text)
    Write-Host "ℹ $Text" -ForegroundColor Cyan
}

function Test-Python {
    Write-Info "Checking Python installation..."

    try {
        $pythonVersion = & python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Python $pythonVersion found"
            return $true
        }
    }
    catch {
        Write-Error "Python is not installed"
        Write-Info "Please install Python 3.8 or higher from https://www.python.org/"
        return $false
    }
}

function Test-Ollama {
    Write-Info "Checking Ollama installation..."

    try {
        $null = Get-Command ollama -ErrorAction Stop
        Write-Success "Ollama is installed"

        # Check if Ollama is running
        try {
            $null = & ollama list 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Ollama service is running"
                return $true
            }
        }
        catch {
            Write-Warning "Ollama is installed but not running"
            Write-Info "Please start the Ollama application"
            return $false
        }
    }
    catch {
        Write-Error "Ollama is not installed"
        Write-Info "Please install Ollama from https://ollama.com/download"
        return $false
    }
}

# Main execution
Write-Header "LLM Benchmark Launcher - $($PSVersionTable.OS)"

# Check prerequisites
if (-not (Test-Python)) {
    exit 1
}

$ollamaAvailable = Test-Ollama

# Change to script directory
Set-Location $PSScriptRoot

# Run the Python launcher
Write-Info "Starting benchmark launcher..."
try {
    & python run.py $args
}
catch {
    Write-Error "Failed to run benchmark: $_"
    exit 1
}

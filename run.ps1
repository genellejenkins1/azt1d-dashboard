#!/usr/bin/env pwsh
param(
    [string]$cmd = "app"
)

$Venv = ".venv-azt1d"

Write-Host "[run.ps1] Working directory: $(Get-Location)"

# Select a Python interpreter (prefer Windows launcher if available)
$pythonExe = $null
$pythonArgs = @()

if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = "py"
    $pythonArgs = @("-3.11")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
    $pythonArgs = @()
} else {
    Write-Error "[run.ps1] ERROR: Python not found. Please install Python 3.11+ and make sure it is on PATH."
    exit 1
}

if (-not (Test-Path $Venv)) {
    Write-Host "[run.ps1] Creating virtual environment in $Venv"
    $venvArgs = @()
    $venvArgs += $pythonArgs
    $venvArgs += @("-m", "venv", $Venv)
    & $pythonExe @venvArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[run.ps1] Failed to create virtual environment."
        exit $LASTEXITCODE
    }
}

$activateScript = Join-Path $Venv "Scripts/Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "[run.ps1] Activate script not found at $activateScript"
    exit 1
}

Write-Host "[run.ps1] Activating virtual environment"
. $activateScript

Write-Host "[run.ps1] Ensuring build tooling (pip, setuptools, wheel) is up to date"
pip install --upgrade "pip>=25.3" setuptools wheel

Write-Host "[run.ps1] Installing project dependencies (pip install -r requirements.txt)"
pip install -r requirements.txt

Write-Host "[run.ps1] Pulling data via DVC (dvc pull)"
dvc pull
if ($LASTEXITCODE -ne 0) {
    Write-Warning "[run.ps1] dvc pull failed or no remote configured, continuing if data already present."
}

switch ($cmd) {
    "tests" {
        Write-Host "[run.ps1] Running tests"
        python -m pytest tests/ -v
    }
    default {
        Write-Host "[run.ps1] Starting Streamlit dashboard"
        streamlit run app.py
    }
}
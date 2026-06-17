#!/usr/bin/env bash
set -euo pipefail

# Simple runner for the AZT1D dashboard.
#
# Usage:
#   ./run.sh           # start Streamlit dashboard
#   ./run.sh tests     # run Python test suite
#
# The script will create/activate the local virtual environment,
# install dependencies, and pull data via DVC if needed.

VENV=".venv-azt1d"
PYTHON_BIN="python3.11"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[run.sh] ERROR: $PYTHON_BIN not found. Please install Python 3.11 and try again." >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "[run.sh] Creating virtual environment in $VENV with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

echo "[run.sh] Ensuring build tooling (pip, setuptools, wheel) is up to date"
pip install --upgrade "pip>=25.3" setuptools wheel

echo "[run.sh] Installing project dependencies (pip install -r requirements.txt)"
pip install -r requirements.txt

echo "[run.sh] Pulling data via DVC (dvc pull)"
dvc pull || echo "[run.sh] dvc pull failed or no remote configured; continuing if data already present."

CMD="${1:-app}"

case "$CMD" in
  tests)
    echo "[run.sh] Running tests"
    python -m pytest tests/ -v
    ;;
  *)
    echo "[run.sh] Starting Streamlit dashboard"
    streamlit run app.py
    ;;
esac
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m pip install 'gsplat==1.5.3' ninja
.venv/bin/python scripts/fetch_methods.py

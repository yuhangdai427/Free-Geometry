#!/usr/bin/env python3
"""Run from any working directory, including without editable installation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_geometry.cli import main
main()

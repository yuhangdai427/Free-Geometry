#!/usr/bin/env python3
"""Run from any working directory, including without editable installation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_geometry.cli import main
if len(sys.argv) > 1 and sys.argv[1] in ('fuse', 'baseline', 'adapt'):
    from self_geometry.gpu_queue import reserve_gpu
    with reserve_gpu():
        main()
else:
    main()

#!/usr/bin/env python3
"""Compatibility entry point for the full 5-dataset × 3-seed queue.

No quality gate: metric regressions are valid results. Technical stage failures
are logged and isolated by campaign.py; remaining scenes and seeds still run.
"""
from experiments import main

if __name__ == '__main__':
    raise SystemExit(main())

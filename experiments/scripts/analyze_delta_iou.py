#!/usr/bin/env python3
"""Compatibility entrypoint for the metric-generic Q2 analysis script."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    from experiments.scripts.analyze_q2_metrics import main as run_main

    run_main()

if __name__ == "__main__":
    main()

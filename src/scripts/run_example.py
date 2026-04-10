#!/usr/bin/env python
"""Run the hail probability pipeline on the example data in legacy/data/.

Usage:
    python src/scripts/run_example.py
"""

import logging
from pathlib import Path

from hailathon import process

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "legacy" / "data"
OUTPUT_DIR = Path("/tmp/hailathon_output")

result = process(
    tops_45dbz_path=str(DATA_DIR / "CORECOMP.TOP"),
    tops_50dbz_path=str(DATA_DIR / "COREC50.TOP"),
    zero_level_path=str(DATA_DIR / "meps_zerolevel_stere_radar.txt"),
    m20_level_path=str(DATA_DIR / "meps_M20_level_stere_radar.txt"),
    output_dir=str(OUTPUT_DIR),
    timestamp="2026040900",
)

for name, path in result.items():
    size = Path(path).stat().st_size
    print(f"  {name}: {path}  ({size:,} bytes)")

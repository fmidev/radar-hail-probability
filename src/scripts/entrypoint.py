#!/usr/bin/env python
"""Container entrypoint — reads inputs from environment variables and runs the pipeline.

Required environment variables:
    TOPS_45_PATH      Path to 45 dBZ IRIS TOPS composite
    TOPS_50_PATH      Path to 50 dBZ IRIS TOPS composite
    ZERO_LEVEL_PATH   Path to NWP 0°C isotherm text file
    M20_LEVEL_PATH    Path to NWP -20°C isotherm text file
    OUTPUT_DIR        Directory for output files
    TIMESTAMP         Nominal product time (e.g. 20260409T0100Z)
"""

import logging
import os
import sys
from pathlib import Path

from hailathon import process

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

_REQUIRED = [
    "TOPS_45_PATH",
    "TOPS_50_PATH",
    "ZERO_LEVEL_PATH",
    "M20_LEVEL_PATH",
    "OUTPUT_DIR",
    "TIMESTAMP",
]

missing = [k for k in _REQUIRED if not os.environ.get(k)]
if missing:
    for k in missing:
        logging.error("Required environment variable not set: %s", k)
    sys.exit(1)

result = process(
    tops_45dbz_path=os.environ["TOPS_45_PATH"],
    tops_50dbz_path=os.environ["TOPS_50_PATH"],
    zero_level_path=os.environ["ZERO_LEVEL_PATH"],
    m20_level_path=os.environ["M20_LEVEL_PATH"],
    output_dir=os.environ["OUTPUT_DIR"],
    timestamp=os.environ["TIMESTAMP"],
)

for name, path in result.items():
    size = Path(path).stat().st_size
    print(f"{name}: {path}  ({size:,} bytes)")

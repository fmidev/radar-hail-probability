"""Top-level processing pipeline invoked by the Airflow DockerOperator."""

from __future__ import annotations

import os


def process(
    tops_45dbz_path: str,
    tops_50dbz_path: str,
    zero_level_path: str,
    m20_level_path: str,
    output_dir: str,
    timestamp: str,
) -> dict[str, str]:
    """Run the full POH/LHI pipeline for a single time step.

    Args:
        tops_45dbz_path: Path to the 45 dBZ IRIS TOPS composite.
        tops_50dbz_path: Path to the 50 dBZ IRIS TOPS composite.
        zero_level_path: Path to the NWP 0°C isotherm text file.
        m20_level_path: Path to the NWP −20°C isotherm text file.
        output_dir: Directory where output files are written.
        timestamp: Nominal product time, ISO-8601 (e.g. "20240601T1200Z").

    Returns:
        Dict mapping product name to output file path:
        {"poh_odim": ..., "lhi_odim": ..., "poh_tif": ..., "lhi_tif": ...}
    """
    raise NotImplementedError

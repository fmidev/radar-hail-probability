"""Top-level processing pipeline invoked by the Airflow DockerOperator."""

import logging
import os
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from hailathon.algorithms.lhi import compute_lhi, compute_thi
from hailathon.algorithms.poh import compute_poh
from hailathon.io.iris import read_tops
from hailathon.io.nwp import interpolate_to_grid, read_isotherm_text
from hailathon.io.geotiff import write_geotiff
from hailathon.io.odim import write_odim

log = logging.getLogger(__name__)


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
    ts = _parse_timestamp(timestamp)
    log.info("Processing timestamp %s", ts.isoformat())

    # -- Read radar TOPS -----------------------------------------------
    tops_45 = read_tops(tops_45dbz_path)
    tops_50 = read_tops(tops_50dbz_path)
    log.info(
        "TOPS grids loaded: 45 dBZ %s, 50 dBZ %s",
        tops_45.shape, tops_50.shape,
    )

    # -- Read and interpolate NWP isotherms ----------------------------
    # Use the 45 dBZ grid as the reference (both TOPS share the same grid).
    target_x = tops_45.coords["x"].values
    target_y = tops_45.coords["y"].values

    zero_nwp = read_isotherm_text(zero_level_path)
    m20_nwp = read_isotherm_text(m20_level_path)

    ti_zero = _select_time_index(zero_nwp, ts)
    ti_m20 = _select_time_index(m20_nwp, ts)
    log.info("NWP time indices: zero=%d, m20=%d", ti_zero, ti_m20)

    zero_level = interpolate_to_grid(zero_nwp, target_x, target_y, time_index=ti_zero)
    m20_level = interpolate_to_grid(m20_nwp, target_x, target_y, time_index=ti_m20)

    # -- Compute hail products -----------------------------------------
    poh = compute_poh(tops_45, zero_level)
    lhi = compute_lhi(tops_50, m20_level)
    thi = compute_thi(lhi, zero_level)

    # Carry the no-echo mask from TOPS to products so writers can
    # distinguish "undetect" (no radar echo) from "nodata" (missing).
    poh.coords["noecho"] = tops_45.coords["noecho"]
    lhi.coords["noecho"] = tops_50.coords["noecho"]
    log.info("Products computed: POH, LHI, THI")

    # -- Write outputs -------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    ts_str = ts.strftime("%Y%m%dT%H%MZ")

    paths: dict[str, str] = {}
    paths["poh_odim"] = os.path.join(output_dir, f"poh_{ts_str}.h5")
    paths["lhi_odim"] = os.path.join(output_dir, f"lhi_{ts_str}.h5")
    paths["poh_tif"] = os.path.join(output_dir, f"poh_{ts_str}.tif")
    paths["lhi_tif"] = os.path.join(output_dir, f"lhi_{ts_str}.tif")

    write_odim(paths["poh_odim"], poh, "POH", timestamp)
    write_odim(paths["lhi_odim"], lhi, "LHI", timestamp)
    write_geotiff(paths["poh_tif"], poh, "POH")
    write_geotiff(paths["lhi_tif"], lhi, "LHI")

    log.info("Outputs written to %s", output_dir)
    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(timestamp: str) -> datetime:
    """Parse an ISO-8601–ish timestamp string to a UTC datetime.

    Accepts common formats: ``20240601T1200Z``, ``2024-06-01T12:00:00Z``,
    ``2024060112``.
    """
    ts = timestamp.rstrip("Z")
    for fmt in ("%Y%m%dT%H%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y%m%d%H"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {timestamp!r}")


def _select_time_index(nwp: xr.DataArray, target: datetime) -> int:
    """Return the NWP time index whose hour matches *target* (floored to HH:00).

    Mirrors legacy behaviour: ``date +%Y%m%d%H`` selects the exact hour.
    Falls back to nearest available time if within 1 hour; raises otherwise.
    """
    target_h = np.datetime64(
        target.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    )
    times = nwp.coords["time"].values
    # Try exact match first (legacy behaviour)
    exact = np.where(times == target_h)[0]
    if len(exact) > 0:
        return int(exact[0])
    # Fallback: nearest available time, but only within tolerance
    deltas = np.abs(times - target_h)
    idx = int(deltas.argmin())
    gap = deltas[idx] / np.timedelta64(1, "h")
    if gap > 1:
        raise ValueError(
            f"No NWP time within 1 h of {target_h}; "
            f"nearest is {times[idx]} ({gap:.0f} h away). "
            f"NWP file may be from a different model run."
        )
    log.warning(
        "No exact NWP time for %s; using nearest: %s (index %d, %.0f h gap)",
        target_h, times[idx], idx, gap,
    )
    return idx

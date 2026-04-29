"""Top-level processing pipeline invoked by the Airflow DockerOperator."""

import logging
import os
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from hailathon.algorithms.lhi import compute_lhi, compute_thi
from hailathon.algorithms.poh import compute_hhi, compute_poh
from hailathon.io.tops import read_tops
from hailathon.io.nwp import interpolate_to_grid, read_isotherm_text
from hailathon.io.geotiff import write_geotiff
from hailathon.io.odim import write_odim

from hailathon.logs import streamlogger_setup

log = logging.getLogger(__name__)
streamlogger_setup(log)


_PRODUCT_UNITS: dict[str, str] = {
    "poh": "prob",
    "hhi": "index",
    "lhi": "index",
    "thi": "index",
}

_DEFAULT_NAME_FORMAT = "{type}_{datetime_format}.{ext}"
_DEFAULT_DATETIME_FORMAT = "%Y%m%d%H%M"


def _output_filename(fmt: str, ts_str: str, product: str, ext: str) -> str:
    return fmt.format(
        datetime_format=ts_str,
        type=product,
        units=_PRODUCT_UNITS.get(product, ""),
        ext=ext,
    )


def process(
    tops_45dbz_path: str,
    tops_50dbz_path: str,
    zero_level_path: str,
    m20_level_path: str,
    output_dir: str,
    timestamp: str,
    output_name_format: str | None = None,
    datetime_format: str = _DEFAULT_DATETIME_FORMAT,
) -> dict[str, str]:
    """Run the full POH/LHI pipeline for a single time step.

    Args:
        tops_45dbz_path: Path to the 45 dBZ ODIM HDF5 ETOP composite.
        tops_50dbz_path: Path to the 50 dBZ ODIM HDF5 ETOP composite.
        zero_level_path: Path to the NWP 0°C isotherm text file.
        m20_level_path: Path to the NWP −20°C isotherm text file.
        output_dir: Directory where output files are written.
        timestamp: Nominal product time, ISO-8601 (e.g. "20240601T1200Z").
        output_name_format: Template for output filenames.  Supports
            ``{datetime_format}``, ``{type}``, ``{units}``, and ``{ext}``
            placeholders.  Defaults to ``"{type}_{datetime_format}.{ext}"``.
        datetime_format: strftime format string used to render the timestamp
            inside the filename.  Defaults to ``"%Y%m%dT%H%MZ"``.

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
    hhi = compute_hhi(tops_45, zero_level)
    lhi = compute_lhi(tops_50, m20_level)
    thi = compute_thi(hhi, zero_level)

    # Carry the no-echo mask from TOPS to products so writers can
    # distinguish "undetect" (no radar echo) from "nodata" (missing).
    poh.coords["noecho"] = tops_45.coords["noecho"]
    hhi.coords["noecho"] = tops_45.coords["noecho"]
    lhi.coords["noecho"] = tops_50.coords["noecho"]
    thi.coords["noecho"] = tops_45.coords["noecho"]

    # Carry the CRS so writers embed the correct projection.
    for da, src in [(poh, tops_45), (hhi, tops_45), (lhi, tops_50), (thi, tops_45)]:
        if "crs_wkt" in src.attrs:
            da.attrs["crs_wkt"] = src.attrs["crs_wkt"]
    log.info("Products computed: POH, HHI, LHI, THI")

    # -- Write outputs -------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    ts_str = ts.strftime(datetime_format)
    fmt = output_name_format if output_name_format is not None else _DEFAULT_NAME_FORMAT

    paths: dict[str, str] = {}
    for product in ("poh", "hhi", "lhi", "thi"):
        for ext, suffix in (("h5", "odim"), ("tif", "tif")):
            paths[f"{product}_{suffix}"] = os.path.join(
                output_dir, _output_filename(fmt, ts_str, product, ext)
            )

    write_odim(paths["poh_odim"], poh, "POH", timestamp)
    write_odim(paths["hhi_odim"], hhi, "HHI", timestamp)
    write_odim(paths["lhi_odim"], lhi, "LHI", timestamp)
    write_odim(paths["thi_odim"], thi, "THI", timestamp)
    write_geotiff(paths["poh_tif"], poh, "POH")
    write_geotiff(paths["hhi_tif"], hhi, "HHI")
    write_geotiff(paths["lhi_tif"], lhi, "LHI")
    write_geotiff(paths["thi_tif"], thi, "THI")

    for path in paths.values():
        log.info("Wrote %s", path)
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

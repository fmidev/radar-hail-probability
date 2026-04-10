"""Read NWP isotherm height text files (0°C and −20°C levels)."""

import datetime
import re

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from hailathon.projection.grid import (
    STANDARD_X0, STANDARD_DX,
    STANDARD_Y0, STANDARD_DY,
)

# Upsampling factors that map the coarse NWP grid to the standard
# 929×1571 IRIS domain (legacy: gX=36, gY=35 in extract_Tlevels_fromtext.c).
_GX = 36
_GY = 35

# Pattern for missing values in the NWP text files.
_MISSING_RE = re.compile(r"^\s*-\s*$")

# Timestamp format used in the NWP file: YYYYMMDDHH (10 digits).
_TS_RE = re.compile(r"^\d{10}$")


def read_isotherm_text(path: str) -> xr.DataArray:
    """Parse a NWP model text file and return the coarse isotherm-height grid.

    The text file contains multiple time steps, each with a 26×45 grid of
    isotherm heights (metres AGL).  Missing values (``-``) become NaN.
    The Y axis is flipped so that row 0 is the southernmost row, matching
    the FIN1000 projection convention.

    Projection coordinates are assigned assuming the grid aligns with the
    standard 929×1571 IRIS domain via upsampling factors 36 (x) and 35 (y).

    Args:
        path: Path to the NWP text file.

    Returns:
        DataArray with dims ``(time, y_nwp, x_nwp)`` and heights in metres.
        ``time`` coordinates are ``datetime64`` values.
    """
    timestamps, grids = _parse_text(path)
    data = np.stack(grids)  # (T, Ydim, Xdim)

    # Flip Y so that row 0 = south (same flip as legacy C code)
    data = data[:, ::-1, :]

    ydim, xdim = data.shape[1], data.shape[2]

    # Physical coordinates of NWP grid-point centres in the standard domain.
    # NWP point (i, j) maps to standard-domain pixel (i*gX + gX/2, j*gY + gY/2).
    x_nwp = STANDARD_X0 + STANDARD_DX * (np.arange(xdim) * _GX + _GX / 2)
    y_nwp = STANDARD_Y0 + STANDARD_DY * (np.arange(ydim) * _GY + _GY / 2)

    times = np.array(timestamps, dtype="datetime64[h]")

    return xr.DataArray(
        data,
        dims=["time", "y_nwp", "x_nwp"],
        coords={"time": times, "x_nwp": x_nwp, "y_nwp": y_nwp},
        attrs={"units": "m", "long_name": "isotherm height AGL"},
    )


def interpolate_to_grid(
    nwp: xr.DataArray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    time_index: int = 0,
) -> xr.DataArray:
    """Interpolate a single NWP time step onto a target radar grid.

    Uses bilinear interpolation via :class:`scipy.interpolate.RegularGridInterpolator`.
    Points outside the NWP domain are extrapolated as nearest-neighbour;
    NaN regions in the coarse grid propagate naturally.

    Args:
        nwp: Coarse NWP DataArray as returned by :func:`read_isotherm_text`.
        target_x: 1-D array of target easting coordinates (m).
        target_y: 1-D array of target northing coordinates (m).
        time_index: Which time step to interpolate (default 0).

    Returns:
        2-D DataArray (dims ``y``, ``x``) of interpolated heights in metres,
        on the target grid.
    """
    field = nwp.isel(time=time_index).values  # (y_nwp, x_nwp)
    y_nwp = nwp.coords["y_nwp"].values
    x_nwp = nwp.coords["x_nwp"].values

    interp = RegularGridInterpolator(
        (y_nwp, x_nwp),
        field,
        method="linear",
        bounds_error=False,
        fill_value=None,  # nearest-neighbour extrapolation outside domain
    )

    yy, xx = np.meshgrid(target_y, target_x, indexing="ij")
    result = interp((yy, xx)).astype(np.float32)

    return xr.DataArray(
        result,
        dims=["y", "x"],
        coords={"x": ("x", target_x), "y": ("y", target_y)},
        attrs={"units": "m", "long_name": "isotherm height AGL (interpolated)"},
    )


# ---------------------------------------------------------------------------
# Text file parser
# ---------------------------------------------------------------------------

def _parse_text(path: str) -> tuple[list[datetime.datetime], list[np.ndarray]]:
    """Low-level parser for the FMI NWP isotherm text format.

    Returns:
        (timestamps, grids) where *timestamps* is a list of
        :class:`datetime.datetime` and *grids* is a list of 2-D float64
        arrays (Ydim, Xdim) with NaN for missing values.
    """
    timestamps: list[datetime.datetime] = []
    grids: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []

    with open(path) as fh:
        _skip_header(fh)

        for line in fh:
            stripped = line.strip()

            # Blank line → end of a time-step block
            if not stripped:
                if current_rows:
                    grids.append(np.stack(current_rows))
                    current_rows = []
                continue

            # Timestamp line → start new block
            if _TS_RE.match(stripped):
                ts = datetime.datetime.strptime(stripped, "%Y%m%d%H")
                timestamps.append(ts)
                _skip_param_line(fh)
                continue

            # Data row
            current_rows.append(_parse_data_row(stripped))

    # Flush last block if file doesn't end with a blank line
    if current_rows:
        grids.append(np.stack(current_rows))

    return timestamps, grids


def _skip_header(fh) -> None:
    """Skip the first line (``RAETUOTE / malli``)."""
    fh.readline()


def _skip_param_line(fh) -> None:
    """Skip the parameter/metadata line that follows each timestamp."""
    fh.readline()


def _parse_data_row(line: str) -> np.ndarray:
    """Parse one whitespace-separated row of floats with ``-`` as NaN."""
    tokens = line.split()
    values = np.empty(len(tokens), dtype=np.float64)
    for i, tok in enumerate(tokens):
        if _MISSING_RE.match(tok):
            values[i] = np.nan
        else:
            values[i] = float(tok)
    return values

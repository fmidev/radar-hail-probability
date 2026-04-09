"""Read IRIS radar composite TOPS products."""

from __future__ import annotations

import numpy as np
import xarray as xr
from wradlib.io.iris import IrisCartesianProductFile

from hailathon.projection.grid import CRS

# Special sentinel values in the FMI IRIS TOPS encoding (raw uint8)
_NO_ECHO = 0
_SPECIAL = 254   # valid location but no computable value (e.g. range folding)
_UNDEFINED = 255

# Height encoding shared by IRIS TOPS and NWP PGM files:
#   raw = height_m / 100 + 1  →  height_m = (raw - 1) * 100
# Gives 100 m resolution; raw range 1–253 maps to 0–25200 m.
# Confirmed by wradlib data_type fkw: {scale: 10.0, offset: -1.0}
# applied as (raw + offset) / scale = (raw - 1) / 10 km = (raw - 1) * 100 m.
_HEIGHT_SCALE_M = 100
_HEIGHT_OFFSET = 1


def read_tops(path: str) -> xr.DataArray:
    """Read an IRIS TOPS product and return echo-top heights as a DataArray.

    Decodes the FMI height encoding: ``height_m = (raw - 1) * 100``.
    Special values (0 = no echo, 254 = special, 255 = undefined) become NaN.

    Grid coordinates are expressed in metres relative to the domain centre,
    consistent with the FIN1000 stereographic projection.  Absolute
    geo-referencing against the projection origin is done in the projection
    layer, not here.

    Args:
        path: Path to the IRIS binary file.

    Returns:
        2-D DataArray (dims ``y``, ``x``) with echo-top heights in metres
        and NaN where masked.  Scalar attribute ``crs_wkt`` carries the CRS.
    """
    f = IrisCartesianProductFile(str(path), loaddata=True, rawdata=True)
    pc = f.product_hdr["product_configuration"]

    xdim = int(pc["x_size"])
    ydim = int(pc["y_size"])

    raw = _extract_raw_data(f, ydim, xdim)
    heights = _decode_heights(raw)
    x, y = _pixel_coords(pc, xdim, ydim)

    return xr.DataArray(
        heights,
        dims=["y", "x"],
        coords={"x": ("x", x), "y": ("y", y)},
        attrs={
            "units": "m",
            "long_name": "radar echo top height",
            "source": str(path),
            "crs_wkt": CRS.to_wkt(),
        },
    )


def _extract_raw_data(
    f: IrisCartesianProductFile, ydim: int, xdim: int
) -> np.ndarray:
    """Extract the raw uint8 array from an open IrisCartesianProductFile.

    ``f.data`` is an OrderedDict keyed by sweep index; index 0 holds the
    single image layer with shape ``(1, ydim, xdim)``.
    """
    raw = f.data[0]
    if raw.ndim == 3:
        raw = raw[0]  # drop the leading sweep dimension
    if raw.shape != (ydim, xdim):
        raise ValueError(
            f"Raw data shape {raw.shape} does not match header "
            f"dimensions {ydim}×{xdim}."
        )
    return raw


def _decode_heights(raw: np.ndarray) -> np.ndarray:
    """Convert raw uint8 array to float32 heights in metres, masking sentinels."""
    mask = (raw == _NO_ECHO) | (raw == _SPECIAL) | (raw == _UNDEFINED)
    heights = (raw.astype(np.float32) - _HEIGHT_OFFSET) * _HEIGHT_SCALE_M
    heights[mask] = np.nan
    return heights


def _pixel_coords(
    pc: dict, xdim: int, ydim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pixel centre coordinates in metres relative to the domain centre.

    IRIS stores ``x_location`` and ``y_location`` as the centre pixel index
    multiplied by 1000, and ``x_scale`` / ``y_scale`` as pixel size in cm.
    """
    dx = pc["x_scale"] / 100.0  # cm → m
    dy = pc["y_scale"] / 100.0
    cx = pc["x_location"] / 1000.0  # fractional centre pixel index
    cy = pc["y_location"] / 1000.0

    x = (np.arange(xdim) - cx) * dx
    y = (np.arange(ydim) - cy) * dy
    return x, y

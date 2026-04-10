"""Read IRIS radar composite TOPS products."""

import numpy as np
import xarray as xr
from wradlib.io.iris import IrisCartesianProductFile

from hailathon.projection.grid import (
    CRS,
    LARGE_SHAPE, LARGE_X0, LARGE_Y0, LARGE_DX, LARGE_DY,
    STANDARD_SHAPE, STANDARD_X0, STANDARD_Y0, STANDARD_DX, STANDARD_DY,
    grid_coords,
)

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

# Map from (rows, cols) shape to (x0, y0, dx, dy) in projection metres.
# x_scale/y_scale in the IRIS product header are ~5 % smaller than
# the values derived from the legacy geographic extents; the latter
# are used here as the authoritative source.
_GRID_PARAMS: dict[tuple[int, int], tuple[float, float, float, float]] = {
    LARGE_SHAPE:    (LARGE_X0,    LARGE_Y0,    LARGE_DX,    LARGE_DY),
    STANDARD_SHAPE: (STANDARD_X0, STANDARD_Y0, STANDARD_DX, STANDARD_DY),
}


def read_tops(path: str) -> xr.DataArray:
    """Read an IRIS TOPS product and return echo-top heights as a DataArray.

    Decodes the FMI height encoding: ``height_m = (raw - 1) * 100``.
    Special values (0 = no echo, 254 = special, 255 = undefined) become NaN.

    Pixel-centre coordinates in the FIN1000 stereographic projection are
    attached as ``x`` and ``y`` dimension coordinates.  The CRS is stored
    as a ``crs_wkt`` scalar attribute.

    Args:
        path: Path to the IRIS binary file.

    Returns:
        2-D DataArray (dims ``y``, ``x``) with echo-top heights in metres
        and NaN where masked.
    """
    f = IrisCartesianProductFile(str(path), loaddata=True, rawdata=True)
    pc = f.product_hdr["product_configuration"]

    xdim = int(pc["x_size"])
    ydim = int(pc["y_size"])

    raw = _extract_raw_data(f, ydim, xdim)
    heights, noecho = _decode_heights(raw)
    x, y = _coords_for_shape(ydim, xdim)

    return xr.DataArray(
        heights,
        dims=["y", "x"],
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "noecho": (("y", "x"), noecho),
        },
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


def _decode_heights(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw uint8 array to float32 heights in metres, masking sentinels.

    Returns:
        (heights, noecho) where *heights* is float32 with NaN for masked pixels
        and *noecho* is a bool array (True where the radar detected no echo).
    """
    noecho = raw == _NO_ECHO
    mask = noecho | (raw == _SPECIAL) | (raw == _UNDEFINED)
    heights = (raw.astype(np.float32) - _HEIGHT_OFFSET) * _HEIGHT_SCALE_M
    heights[mask] = np.nan
    return heights, noecho


def _coords_for_shape(ydim: int, xdim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute FIN1000 pixel-centre coordinates for the given grid shape."""
    shape = (ydim, xdim)
    if shape not in _GRID_PARAMS:
        raise ValueError(
            f"Unrecognised grid shape {shape}. "
            f"Known shapes: {list(_GRID_PARAMS)}"
        )
    x0, y0, dx, dy = _GRID_PARAMS[shape]
    return grid_coords(shape, x0=x0, y0=y0, dx=dx, dy=dy)

"""Read IRIS radar composite TOPS products."""

from __future__ import annotations

import numpy as np
import xarray as xr
import wradlib.io as wrio

from hailathon.projection.grid import CRS, grid_coords

# Special sentinel values in the FMI IRIS TOPS encoding (raw uint8)
_NO_ECHO = 0
_SPECIAL = 254   # valid location but no computable value (e.g. range folding)
_UNDEFINED = 255

# Height encoding shared by IRIS TOPS and NWP PGM files:
#   raw = height_m / 100 + 1  →  height_m = (raw - 1) * 100
# This gives 100 m resolution, raw range 1–253 → 0–25200 m.
_HEIGHT_SCALE_M = 100
_HEIGHT_OFFSET = 1


def read_tops(path: str) -> xr.DataArray:
    """Read an IRIS TOPS product and return echo-top heights as a DataArray.

    Decodes the FMI height encoding: ``height_m = (raw - 1) * 100``.
    Special values (0 = no echo, 254 = special, 255 = undefined) become NaN.

    The encoding is identical to the NWP PGM level files so that raw
    byte differences can be compared directly in the algorithm layer
    (``dH = raw_tops - raw_zero``).  Heights are decoded here to metres
    so callers work with physical units; the algorithm layer re-scales
    to km when applying the POH/LHI formulas.

    Args:
        path: Path to the IRIS binary file.

    Returns:
        2-D DataArray (dims ``y``, ``x``) with echo-top heights in metres
        and NaN where masked.  Coordinates ``x`` and ``y`` are in
        FIN1000 projection metres; ``crs_wkt`` is stored as a scalar
        coordinate attribute.

    Notes:
        wradlib's ``read_iris`` targets polar IRIS products; for FMI
        cartesian composites the return structure may differ.  ``rawdata=True``
        is used throughout so we always work from the unmodified bytes.
        Grid dimensions are read from the product header fields
        ``ixsize`` / ``iysize``.  Verify against a real file if the
        shape looks wrong.
    """
    iris = wrio.read_iris(path, rawdata=True)

    hdr = iris["product_hdr"]["pcf"]
    xdim = int(hdr["ixsize"])
    ydim = int(hdr["iysize"])

    raw = _extract_raw_data(iris, ydim, xdim)

    heights = _decode_heights(raw)

    x, y = grid_coords((ydim, xdim))

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


def _extract_raw_data(iris: dict, ydim: int, xdim: int) -> np.ndarray:
    """Pull the flat raw byte array from the wradlib result dict and reshape.

    wradlib's return structure for cartesian IRIS products may place the
    raw payload under different keys depending on version and product type.
    We try the most common locations in order.
    """
    # Attempt 1: top-level "data" key (raw bytes for some product types)
    raw_bytes = iris.get("data")

    # Attempt 2: first sweep data (polar-style return)
    if raw_bytes is None:
        sweep = iris.get("sweep", {})
        if sweep:
            first = next(iter(sweep.values()))
            raw_bytes = first.get("sweep_data")

    if raw_bytes is None:
        raise ValueError(
            "Could not locate raw data in wradlib IRIS result. "
            "Keys present: " + str(list(iris.keys()))
        )

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    expected = ydim * xdim
    if arr.size != expected:
        raise ValueError(
            f"Raw data size {arr.size} does not match header dimensions "
            f"{ydim}×{xdim} = {expected}."
        )
    return arr.reshape(ydim, xdim)


def _decode_heights(raw: np.ndarray) -> np.ndarray:
    """Convert raw uint8 array to float32 heights in metres, masking sentinels."""
    mask = (raw == _NO_ECHO) | (raw == _SPECIAL) | (raw == _UNDEFINED)
    heights = (raw.astype(np.float32) - _HEIGHT_OFFSET) * _HEIGHT_SCALE_M
    heights[mask] = np.nan
    return heights

"""Write hail probability fields to Cloud-Optimized GeoTIFF."""

import numpy as np
import pyproj
import rasterio
from rasterio.transform import from_bounds
import xarray as xr

from hailathon.projection.grid import CRS as _FALLBACK_CRS

# uint8 encoding per product, matching the ODIM conventions.
# GDAL convention: physical = raw * scale + offset
_ENCODING: dict[str, dict] = {
    "POH": {"scale": 1.0 / 250, "offset": 0.0, "nodata": 255, "undetect": 0},
    "LHI": {"scale": 100.0, "offset": -100.0, "nodata": 255, "undetect": 0},
    "HHI": {"scale": 1.0, "offset": -1.0, "nodata": 255, "undetect": 0},
    "THI": {"scale": 1.0, "offset": -1.0, "nodata": 255, "undetect": 0},
}


def write_geotiff(path: str, data: xr.DataArray, product: str) -> None:
    """Write a single hail product field to a Cloud-Optimized GeoTIFF.

    The output is a single-band uint8 COG with the FIN1000 stereographic
    CRS embedded.  Physical values are recovered via the GDAL scale/offset
    metadata: ``physical = raw * scale + offset``.

    Args:
        path: Output file path.
        data: 2-D DataArray (dims ``y``, ``x``) with product values.
        product: Product name (``"POH"`` or ``"LHI"``).
    """
    enc = _ENCODING[product]
    scale = enc["scale"]
    offset = enc["offset"]
    nodata = enc["nodata"]
    undetect = enc["undetect"]

    x = data.coords["x"].values
    y = data.coords["y"].values
    noecho = data.coords["noecho"].values if "noecho" in data.coords else None

    dx = float(np.diff(x[:2])[0]) if len(x) > 1 else 0.0
    dy = float(np.diff(y[:2])[0]) if len(y) > 1 else 0.0

    # Pixel-edge bounds (half-pixel outward from centres)
    west = float(x[0] - dx / 2)
    east = float(x[-1] + dx / 2)
    south = float(y[0] - dy / 2)
    north = float(y[-1] + dy / 2)

    height, width = data.shape
    transform = from_bounds(west, south, east, north, width, height)

    # Encode: raw = (physical - offset) / scale, clamp to [1, 254]
    raw = np.full((height, width), nodata, dtype=np.uint8)
    valid = np.isfinite(data.values)
    scaled = np.round((data.values[valid] - offset) / scale).astype(np.int32)
    scaled = np.clip(scaled, 1, 254)
    raw[valid] = scaled.astype(np.uint8)

    if noecho is not None:
        raw[noecho] = undetect

    # GeoTIFF convention: row 0 = north (top).  Our array has row 0 = south.
    raw = raw[::-1]

    crs_wkt = data.attrs.get("crs_wkt", _FALLBACK_CRS.to_wkt())

    with rasterio.open(
        path,
        "w",
        driver="COG",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=crs_wkt,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(raw, 1)
        dst.scales = [scale]
        dst.offsets = [offset]
        dst.update_tags(product=product)
        dst.update_tags(1, undetect=str(undetect))

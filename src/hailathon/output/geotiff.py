"""Write hail probability fields to Cloud-Optimized GeoTIFF."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import xarray as xr

from hailathon.projection.grid import CRS


def write_geotiff(path: str, data: xr.DataArray, product: str) -> None:
    """Write a single hail product field to a Cloud-Optimized GeoTIFF.

    The output is a single-band float32 GeoTIFF with the FIN1000
    stereographic CRS embedded.  NaN pixels are written as nodata.

    Args:
        path: Output file path.
        data: 2-D DataArray (dims ``y``, ``x``) with product values.
        product: Product name written to TIFF metadata (e.g. ``"POH"``, ``"LHI"``).
    """
    x = data.coords["x"].values
    y = data.coords["y"].values

    dx = float(np.diff(x[:2])[0]) if len(x) > 1 else 0.0
    dy = float(np.diff(y[:2])[0]) if len(y) > 1 else 0.0

    # Pixel-edge bounds (half-pixel outward from centres)
    west = float(x[0] - dx / 2)
    east = float(x[-1] + dx / 2)
    south = float(y[0] - dy / 2)
    north = float(y[-1] + dy / 2)

    height, width = data.shape
    transform = from_bounds(west, south, east, north, width, height)

    # GeoTIFF convention: row 0 = north (top).  Our array has row 0 = south,
    # so flip Y before writing.
    array = data.values[::-1].astype(np.float32)

    with rasterio.open(
        path,
        "w",
        driver="COG",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.to_wkt(),
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(array, 1)
        dst.update_tags(product=product)

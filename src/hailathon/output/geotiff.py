"""Write hail probability fields to GeoTIFF."""

import xarray as xr


def write_geotiff(path: str, data: xr.DataArray, product: str) -> None:
    """Write a single hail product field to a Cloud-Optimized GeoTIFF.

    The output uses float32 with the FIN1000 stereographic CRS embedded.
    NaN pixels are written as nodata.

    Args:
        path: Output file path.
        data: 2-D DataArray with x/y coordinates in projection metres.
        product: Product name written to TIFF metadata (e.g. "POH", "LHI").
    """
    raise NotImplementedError

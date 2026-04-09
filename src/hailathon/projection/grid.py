"""FIN1000 stereographic grid definition and coordinate utilities."""

import numpy as np
import pyproj

# FIN1000 stereographic projection used by the legacy pipeline
CRS = pyproj.CRS.from_proj4(
    "+proj=stere +lon_0=25 +lat_0=90 +lat_ts=60 +a=6371000 +units=m"
)

# Full composite grid dimensions (IRIS TOPS product)
FULL_SHAPE = (2625, 1628)  # (rows, cols) → (y, x)

# Sub-domain used for visualization output
SUB_SHAPE = (1571, 929)  # (rows, cols) → (y, x)

# Grid origin and resolution (metres) — derived from legacy false easting/northing
X0 = -408214.26  # easting of western edge
Y0 = -3639312.10  # northing of southern edge
DX = 500.0  # pixel size in x (metres)
DY = 500.0  # pixel size in y (metres)


def grid_coords(shape: tuple[int, int] = FULL_SHAPE) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) coordinate arrays for the given grid shape.

    Args:
        shape: (rows, cols) grid dimensions.

    Returns:
        Tuple of 1-D arrays (x, y) in projection metres.
    """
    rows, cols = shape
    x = X0 + DX * np.arange(cols)
    y = Y0 + DY * np.arange(rows)
    return x, y


def grid_lonlat(shape: tuple[int, int] = FULL_SHAPE) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) 2-D arrays for the given grid shape.

    Args:
        shape: (rows, cols) grid dimensions.

    Returns:
        Tuple of 2-D arrays (lon, lat) in degrees.
    """
    x, y = grid_coords(shape)
    xx, yy = np.meshgrid(x, y)
    transformer = pyproj.Transformer.from_crs(CRS, pyproj.CRS.from_epsg(4326), always_xy=True)
    return transformer.transform(xx, yy)

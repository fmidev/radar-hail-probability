"""FIN1000 stereographic grid definition and coordinate utilities."""

import numpy as np
import pyproj

# ---------------------------------------------------------------------------
# CRS
# From legacy generate_latlonbox_flip.c:
#   proj=stere  a=6371000  lon_0=25E  lat_0=90N  lat_ts=90N
#   x_0=408214.26  y_0=3639312.10
#
# The false easting/northing are chosen so that the SW corner of the
# large (1628×2625) domain falls at projection coordinates (0, 0).
# ---------------------------------------------------------------------------

CRS = pyproj.CRS.from_proj4(
    "+proj=stere +lon_0=25 +lat_0=90 +lat_ts=90 +a=6371000"
    " +x_0=408214.26 +y_0=3639312.10 +units=m"
)

# ---------------------------------------------------------------------------
# Large domain  –  CORECOMP_L / COREC50_L  (1628 × 2625 pixels, ~500 m)
# SW corner at (0, 0); derived from legacy geographic bounds
# 18.6°E 57.93°N → 34.9°E 69.0°N
# ---------------------------------------------------------------------------
LARGE_SHAPE = (2625, 1628)   # (rows, cols) → (y, x)
LARGE_X0 = 0.0               # SW corner easting  (m)
LARGE_Y0 = 0.0               # SW corner northing (m)
LARGE_DX = 500.15            # pixel size in x    (m)
LARGE_DY = 500.15            # pixel size in y    (m)

# ---------------------------------------------------------------------------
# Standard domain  –  CORECOMP / COREC50  (929 × 1571 pixels, ~1078 m)
# Covers a wider area; SW corner sits outside the large domain.
# 17.1°E 56.8°N → 38.0°E 71.0°N
# ---------------------------------------------------------------------------
STANDARD_SHAPE = (1571, 929)   # (rows, cols) → (y, x)
STANDARD_X0 = -113876.4        # SW corner easting  (m)
STANDARD_Y0 = -123192.8        # SW corner northing (m)
STANDARD_DX = 1078.31          # pixel size in x    (m)
STANDARD_DY = 1072.49          # pixel size in y    (m)

# Backwards-compatible aliases used by earlier code
FULL_SHAPE = LARGE_SHAPE
SUB_SHAPE = STANDARD_SHAPE
X0 = LARGE_X0
Y0 = LARGE_Y0
DX = LARGE_DX
DY = LARGE_DY


def grid_coords(
    shape: tuple[int, int] = LARGE_SHAPE,
    x0: float = LARGE_X0,
    y0: float = LARGE_Y0,
    dx: float = LARGE_DX,
    dy: float = LARGE_DY,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) coordinate arrays for a grid aligned with FIN1000.

    Coordinates represent pixel-centre positions in projection metres.

    Args:
        shape: (rows, cols) grid dimensions.
        x0: Easting of the western edge (left boundary of column 0), metres.
        y0: Northing of the southern edge (bottom boundary of row 0), metres.
        dx: Pixel width in metres.
        dy: Pixel height in metres.

    Returns:
        Tuple of 1-D arrays (x, y) in projection metres, length cols and rows.
    """
    rows, cols = shape
    x = x0 + dx * (np.arange(cols) + 0.5)
    y = y0 + dy * (np.arange(rows) + 0.5)
    return x, y


def grid_lonlat(
    shape: tuple[int, int] = LARGE_SHAPE,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) 2-D arrays for a given grid shape.

    Args:
        shape: (rows, cols) grid dimensions.
        **kwargs: Forwarded to :func:`grid_coords` (x0, y0, dx, dy).

    Returns:
        Tuple of 2-D arrays (lon, lat) in degrees WGS-84.
    """
    x, y = grid_coords(shape, **kwargs)
    xx, yy = np.meshgrid(x, y)
    transformer = pyproj.Transformer.from_crs(
        CRS, pyproj.CRS.from_epsg(4326), always_xy=True
    )
    return transformer.transform(xx, yy)

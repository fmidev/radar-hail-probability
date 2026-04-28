"""Read ODIM HDF5 radar composite echo-top (ETOP) products."""

import h5py
import numpy as np
import pyproj
import xarray as xr

_CRS = pyproj.CRS.from_epsg(3067)


def read_tops(path: str) -> xr.DataArray:
    """Read an ODIM HDF5 ETOP composite and return echo-top heights as a DataArray.

    Decodes the ODIM encoding: ``height_m = raw * gain + offset``.
    ``undetect`` pixels (raw == 0) become NaN with ``noecho=True``.
    ``nodata`` pixels (raw == 255) become NaN with ``noecho=False``.

    Pixel-centre coordinates in EPSG:3067 (ETRS-TM35FIN) are attached as
    ``x`` and ``y`` dimension coordinates, with row 0 as the southernmost row.

    Args:
        path: Path to the ODIM HDF5 file.

    Returns:
        2-D DataArray (dims ``y``, ``x``) with echo-top heights in metres,
        NaN where masked, and a boolean ``noecho`` coordinate.
    """
    with h5py.File(path, "r") as f:
        where = dict(f["where"].attrs)
        d_what = dict(f["dataset1/data1/what"].attrs)
        raw = f["dataset1/data1/data"][()]

    gain = float(d_what["gain"])
    offset = float(d_what["offset"])
    nodata_val = int(d_what["nodata"])
    undetect_val = int(d_what["undetect"])

    x_min = float(where["BBOX_native"][0])
    y_min = float(where["BBOX_native"][1])
    xscale = float(where["xscale"])
    yscale = float(where["yscale"])
    xsize = int(where["xsize"])
    ysize = int(where["ysize"])

    x = x_min + xscale * (np.arange(xsize) + 0.5)
    y = y_min + yscale * (np.arange(ysize) + 0.5)

    # ODIM stores data top-to-bottom (row 0 = north); flip to row 0 = south
    raw = raw[::-1]

    noecho = raw == undetect_val
    mask = noecho | (raw == nodata_val)

    heights = raw.astype(np.float32) * gain + offset
    heights[mask] = np.nan

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
            "crs_wkt": _CRS.to_wkt(),
        },
    )

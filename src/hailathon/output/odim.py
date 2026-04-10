"""Write hail probability fields to ODIM HDF5 format.

Implements a minimal ODIM H5 v2.4 cartesian composite structure.
Each file contains a single quantity (POH or LHI) stored as uint8
with linear gain/offset encoding.

Reference: OPERA ODIM_H5 v2.4 — Information model for the HDF5 file format.
"""

from datetime import datetime, timezone

import h5py
import numpy as np
import pyproj
import xarray as xr

from hailathon.projection.grid import CRS

# ODIM version string
_ODIM_VERSION = "H5rad 2.4"

# ODIM source string for FMI composites
_SOURCE = "NOD:fimaa,ORG:86,CTY:613,PLC:Finland"

# Encoding parameters per product.
# POH [0, 1]  → uint8 with gain=1/250, offset=0 → raw range 0–250
# LHI ~[90, 120+] → uint8 with gain=1.0, offset=0 → stored directly
_PRODUCT_ENCODING: dict[str, dict] = {
    "POH": {
        "gain": 1.0 / 250,
        "offset": 0.0,
        "nodata": 255.0,
        "undetect": 0.0,
    },
    "LHI": {
        "gain": 1.0,
        "offset": 0.0,
        "nodata": 255.0,
        "undetect": 0.0,
    },
}


def write_odim(
    path: str,
    data: xr.DataArray,
    product: str,
    timestamp: str,
) -> None:
    """Write a single hail product field to an ODIM-compatible HDF5 file.

    The file follows the ODIM H5 v2.4 cartesian composite structure with
    groups ``/what``, ``/where``, ``/dataset1/data1``.

    Args:
        path: Output file path.
        data: 2-D DataArray (dims ``y``, ``x``) with product values.
        product: Product name (``"POH"`` or ``"LHI"``).
        timestamp: Nominal time of the product, ISO-8601 string.
    """
    ts = _parse_odim_time(timestamp)
    date_str = ts.strftime("%Y%m%d")
    time_str = ts.strftime("%H%M%S")

    encoding = _PRODUCT_ENCODING[product]
    raw = _encode_data(data.values, encoding)

    x = data.coords["x"].values
    y = data.coords["y"].values

    with h5py.File(path, "w") as f:
        _write_root_what(f, date_str, time_str)
        _write_where(f, x, y)
        _write_dataset(f, raw, product, date_str, time_str, encoding)


def _parse_odim_time(timestamp: str) -> datetime:
    """Parse timestamp for ODIM date/time attributes."""
    ts = timestamp.rstrip("Z")
    for fmt in ("%Y%m%dT%H%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y%m%d%H"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {timestamp!r}")


def _encode_data(values: np.ndarray, encoding: dict) -> np.ndarray:
    """Encode float values to uint8 using ODIM gain/offset convention.

    ODIM convention: ``physical = gain * raw + offset``,
    so ``raw = (physical - offset) / gain``.
    NaN pixels become ``nodata``.
    """
    gain = encoding["gain"]
    offset = encoding["offset"]
    nodata = int(encoding["nodata"])
    undetect = int(encoding["undetect"])

    raw = np.full(values.shape, nodata, dtype=np.uint8)
    valid = np.isfinite(values)
    scaled = np.round((values[valid] - offset) / gain).astype(np.int32)
    # Clamp to valid uint8 range, reserving nodata and undetect
    scaled = np.clip(scaled, 1, 254)
    raw[valid] = scaled.astype(np.uint8)
    return raw


def _write_root_what(f: h5py.File, date_str: str, time_str: str) -> None:
    """Write the ``/what`` group with object type, version, date, time, source."""
    what = f.create_group("what")
    what.attrs["object"] = "COMP"
    what.attrs["version"] = _ODIM_VERSION
    what.attrs["date"] = date_str
    what.attrs["time"] = time_str
    what.attrs["source"] = _SOURCE


def _write_where(f: h5py.File, x: np.ndarray, y: np.ndarray) -> None:
    """Write the ``/where`` group with projection and grid parameters."""
    where = f.create_group("where")

    projdef = CRS.to_proj4()
    where.attrs["projdef"] = projdef
    where.attrs["xsize"] = np.int64(len(x))
    where.attrs["ysize"] = np.int64(len(y))

    # Pixel spacing (assume uniform)
    dx = float(np.diff(x[:2])[0]) if len(x) > 1 else 0.0
    dy = float(np.diff(y[:2])[0]) if len(y) > 1 else 0.0
    where.attrs["xscale"] = dx
    where.attrs["yscale"] = dy

    # Corner coordinates in WGS-84
    transformer = pyproj.Transformer.from_crs(
        CRS, pyproj.CRS.from_epsg(4326), always_xy=True
    )
    # Pixel-edge corners (half-pixel outward from centres)
    x_min = float(x[0] - dx / 2)
    x_max = float(x[-1] + dx / 2)
    y_min = float(y[0] - dy / 2)
    y_max = float(y[-1] + dy / 2)

    ll_lon, ll_lat = transformer.transform(x_min, y_min)
    ul_lon, ul_lat = transformer.transform(x_min, y_max)
    ur_lon, ur_lat = transformer.transform(x_max, y_max)
    lr_lon, lr_lat = transformer.transform(x_max, y_min)

    where.attrs["LL_lon"] = ll_lon
    where.attrs["LL_lat"] = ll_lat
    where.attrs["UL_lon"] = ul_lon
    where.attrs["UL_lat"] = ul_lat
    where.attrs["UR_lon"] = ur_lon
    where.attrs["UR_lat"] = ur_lat
    where.attrs["LR_lon"] = lr_lon
    where.attrs["LR_lat"] = lr_lat


def _write_dataset(
    f: h5py.File,
    raw: np.ndarray,
    product: str,
    date_str: str,
    time_str: str,
    encoding: dict,
) -> None:
    """Write ``/dataset1/data1`` with the encoded data and metadata."""
    ds = f.create_group("dataset1")

    ds_what = ds.create_group("what")
    ds_what.attrs["product"] = "COMP"
    ds_what.attrs["quantity"] = product
    ds_what.attrs["startdate"] = date_str
    ds_what.attrs["starttime"] = time_str
    ds_what.attrs["enddate"] = date_str
    ds_what.attrs["endtime"] = time_str

    data_grp = ds.create_group("data1")

    # ODIM stores data top-to-bottom (row 0 = north).
    # Our arrays have row 0 = south, so flip Y.
    data_grp.create_dataset("data", data=raw[::-1], compression="gzip")

    d_what = data_grp.create_group("what")
    d_what.attrs["quantity"] = product
    d_what.attrs["gain"] = encoding["gain"]
    d_what.attrs["offset"] = encoding["offset"]
    d_what.attrs["nodata"] = encoding["nodata"]
    d_what.attrs["undetect"] = encoding["undetect"]

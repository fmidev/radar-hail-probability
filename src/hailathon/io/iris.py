"""Read IRIS radar composite TOPS products."""

import xarray as xr


def read_tops(path: str) -> xr.DataArray:
    """Read an IRIS TOPS product and return heights as a DataArray.

    Values are decoded from 8-bit encoding (raw × 10 m).
    Special values (0 = no echo, 254 = special, 255 = undefined) are masked.

    Args:
        path: Path to the IRIS binary file.

    Returns:
        DataArray with echo-top heights in metres, NaN where masked.
    """
    raise NotImplementedError

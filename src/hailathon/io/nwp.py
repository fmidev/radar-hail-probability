"""Read NWP isotherm height text files (0°C and −20°C levels)."""

import xarray as xr


def read_isotherm_text(path: str) -> xr.DataArray:
    """Parse a NWP model text file and return gridded isotherm heights.

    The text file contains a time × y × x grid of isotherm heights
    in the FIN1000 stereographic projection.

    Args:
        path: Path to the NWP text file.

    Returns:
        DataArray with dimensions (time, y, x) and heights in metres.
    """
    raise NotImplementedError

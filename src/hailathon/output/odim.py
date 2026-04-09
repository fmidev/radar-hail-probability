"""Write hail probability fields to ODIM HDF5 format."""

import xarray as xr


def write_odim(
    path: str,
    poh: xr.DataArray,
    lhi: xr.DataArray,
    timestamp: str,
) -> None:
    """Write POH and LHI fields to an ODIM-compatible HDF5 file.

    Args:
        path: Output file path.
        poh: POH field (values in [0, 1]).
        lhi: LHI field.
        timestamp: Nominal time of the product, ISO-8601 string.
    """
    raise NotImplementedError

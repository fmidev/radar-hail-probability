"""POH (Probability Of Hail) computation."""

import numpy as np
import xarray as xr


def compute_poh(tops: xr.DataArray, zero_level: xr.DataArray) -> xr.DataArray:
    """Compute Probability Of Hail from echo-top heights and 0°C isotherm.

    Formula: POH = clip(0.319 + 0.133 × dH, 0, 1)
    where dH = (tops_height − zero_level_height) / 1000  [units: km]

    The divisor 1000 comes from the legacy encoding: the C code operates on
    raw uint8 values (100 m resolution) and divides by 10, which is equivalent
    to dividing the metre difference by 1000.

    Args:
        tops: Echo-top heights in metres (NaN where no echo / invalid).
        zero_level: 0°C isotherm heights in metres, on the same grid.

    Returns:
        POH values in [0, 1], NaN where input is masked.
    """
    dh = (tops - zero_level) / 1000.0
    poh = 0.319 + 0.133 * dh
    return xr.where(np.isfinite(poh), poh.clip(0.0, 1.0), np.nan)

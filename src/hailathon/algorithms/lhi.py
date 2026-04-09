"""LHI (Large Hail Index) and THI (Temperature-adjusted Hail Index) computation."""

import xarray as xr


def compute_lhi(tops: xr.DataArray, m20_level: xr.DataArray) -> xr.DataArray:
    """Compute Large Hail Index from echo-top heights and −20°C isotherm.

    Formula: LHI = 100 + dH
    where dH = (tops_height − m20_level_height) / 100  [units: hundreds of metres]

    Args:
        tops: Echo-top heights in metres (NaN where no echo / invalid).
        m20_level: −20°C isotherm heights in metres, on the same grid.

    Returns:
        LHI values, NaN where input is masked.
    """
    dh = (tops - m20_level) / 100.0
    lhi = 100.0 + dh
    return xr.where(tops.notnull(), lhi, float("nan"))


def compute_thi(lhi: xr.DataArray, zero_level: xr.DataArray) -> xr.DataArray:
    """Compute Temperature-adjusted Hail Index (THI) by correcting LHI for zero-level height.

    Applies the zero-isotherm adjustment from the legacy RAE_map_modindex algorithm:
    - zero_level <= 1200 m:  LHI + 2
    - zero_level <= 1700 m:  LHI + 1
    - zero_level <= 3500 m:  LHI unchanged
    - zero_level  > 3500 m:  LHI − 1

    Args:
        lhi: Large Hail Index values.
        zero_level: 0°C isotherm heights in metres.

    Returns:
        THI values, NaN where LHI is masked.
    """
    adj = xr.where(zero_level <= 1200, 2,
          xr.where(zero_level <= 1700, 1,
          xr.where(zero_level <= 3500, 0, -1)))
    return lhi + adj

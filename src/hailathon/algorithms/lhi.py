"""LHI (Large Hail Index) and THI (Temperature-adjusted Hail Index) computation."""

import xarray as xr


def compute_lhi(tops: xr.DataArray, m20_level: xr.DataArray) -> xr.DataArray:
    """Compute Large Hail Index from echo-top heights and −20°C isotherm.

    LHI is the height difference between the 50 dBZ echo top and the −20°C
    isotherm, in metres:  ``LHI = tops − m20_level``.

    Legacy equivalent: the C code computed ``100 + dH/10`` on raw uint8; here
    we skip the arbitrary offset and return physical metres directly.

    Args:
        tops: Echo-top heights in metres (NaN where no echo / invalid).
        m20_level: −20°C isotherm heights in metres, on the same grid.

    Returns:
        LHI (height difference) in metres, NaN where input is masked.
    """
    lhi = tops - m20_level
    return xr.where(tops.notnull(), lhi, float("nan"))


def compute_thi(lhi: xr.DataArray, zero_level: xr.DataArray) -> xr.DataArray:
    """Compute Temperature-adjusted Hail Index (THI) by correcting LHI for zero-level height.

    Applies the zero-isotherm adjustment from the legacy RAE_map_modindex
    algorithm, converted to metres:

    - zero_level ≤ 1200 m:  LHI + 2000 m
    - zero_level ≤ 1700 m:  LHI + 1000 m
    - zero_level ≤ 3500 m:  LHI unchanged
    - zero_level  > 3500 m:  LHI − 1000 m

    Args:
        lhi: Large Hail Index in metres.
        zero_level: 0°C isotherm heights in metres.

    Returns:
        THI in metres, NaN where LHI is masked.
    """
    adj = xr.where(zero_level <= 1200, 2000,
          xr.where(zero_level <= 1700, 1000,
          xr.where(zero_level <= 3500, 0, -1000)))
    return lhi + adj

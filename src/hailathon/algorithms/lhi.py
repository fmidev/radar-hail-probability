"""LHI (Large Hail Index) and THI (Tuovinen Hail Index) computation."""

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


def compute_thi(hhi: xr.DataArray, zero_level: xr.DataArray) -> xr.DataArray:
    """Compute Tuovinen Hail Index (THI) by adjusting HHI for zero-level height.

    - zero_level ≤ 1200 m:  HHI + 2
    - zero_level ≤ 1700 m:  HHI + 1
    - zero_level ≤ 3500 m:  HHI unchanged
    - zero_level  > 3500 m:  HHI − 1

    Args:
        hhi: Holleman Hail Index values.
        zero_level: 0°C isotherm heights in metres.

    Returns:
        THI values, NaN where HHI is masked.
    """
    adj = xr.where(zero_level <= 1200, 2,
          xr.where(zero_level <= 1700, 1,
          xr.where(zero_level <= 3500, 0, -1)))
    return xr.where(hhi.notnull(), hhi + adj, float("nan"))

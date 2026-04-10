"""Integration tests for read_tops against real IRIS files in legacy/data."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hailathon.io.iris import read_tops

DATA_DIR = Path(__file__).parent.parent / "legacy" / "data"

TOPS_FILES = {
    "CORECOMP.TOP": {"shape": (1571, 929), "height_range": (200, 1000)},
    "COREC50.TOP": {"shape": (1571, 929), "height_range": (200, 1000)},
    "CORECOMP_L.TOP": {"shape": (2625, 1628), "height_range": (200, 1000)},
    "COREC50_L.TOP": {"shape": (2625, 1628), "height_range": (200, 1000)},
}


@pytest.fixture(params=list(TOPS_FILES.keys()))
def tops_file(request):
    path = DATA_DIR / request.param
    if not path.exists():
        pytest.skip(f"{request.param} not found in legacy/data")
    return path, TOPS_FILES[request.param]


class TestReadTops:
    def test_returns_dataarray(self, tops_file):
        path, _ = tops_file
        result = read_tops(path)
        assert isinstance(result, xr.DataArray)

    def test_shape_matches_product_header(self, tops_file):
        path, meta = tops_file
        result = read_tops(path)
        assert result.shape == meta["shape"]

    def test_dims_are_y_x(self, tops_file):
        path, _ = tops_file
        result = read_tops(path)
        assert result.dims == ("y", "x")

    def test_has_x_and_y_coords(self, tops_file):
        path, _ = tops_file
        result = read_tops(path)
        assert "x" in result.coords
        assert "y" in result.coords

    def test_coords_are_monotonic(self, tops_file):
        path, _ = tops_file
        result = read_tops(path)
        assert np.all(np.diff(result.coords["x"].values) > 0)
        assert np.all(np.diff(result.coords["y"].values) > 0)

    def test_coords_cover_finland(self, tops_file):
        """x/y extent should span several hundred km — sanity check on scale."""
        path, _ = tops_file
        result = read_tops(path)
        x_span = result.coords["x"].values[-1] - result.coords["x"].values[0]
        y_span = result.coords["y"].values[-1] - result.coords["y"].values[0]
        assert x_span > 500_000, f"x span {x_span:.0f} m too small"
        assert y_span > 500_000, f"y span {y_span:.0f} m too small"

    def test_large_domain_x_starts_near_zero(self, tops_file):
        """Large domain SW corner sits at projection origin (x0=0, y0=0)."""
        path, meta = tops_file
        if meta["shape"] != (2625, 1628):
            pytest.skip("Only applies to large domain")
        result = read_tops(path)
        assert result.coords["x"].values[0] == pytest.approx(250.0, abs=10)

    def test_units_attribute_is_metres(self, tops_file):
        path, _ = tops_file
        result = read_tops(path)
        assert result.attrs.get("units") == "m"

    def test_has_crs_wkt_attribute(self, tops_file):
        path, _ = tops_file
        result = read_tops(path)
        assert "crs_wkt" in result.attrs
        assert len(result.attrs["crs_wkt"]) > 0

    def test_most_pixels_are_nan(self, tops_file):
        """No-echo pixels (raw=0) should dominate a typical composite."""
        path, _ = tops_file
        result = read_tops(path)
        nan_fraction = np.isnan(result.values).mean()
        assert nan_fraction > 0.9

    def test_valid_heights_in_physical_range(self, tops_file):
        path, meta = tops_file
        result = read_tops(path)
        valid = result.values[np.isfinite(result.values)]
        if len(valid) == 0:
            pytest.skip("No valid (non-NaN) pixels in this file")
        low, high = meta["height_range"]
        assert valid.min() >= low, f"Min height {valid.min()} below expected {low} m"
        assert valid.max() <= 25200, f"Max height {valid.max()} above encoding maximum"

    def test_no_raw_special_values_in_output(self, tops_file):
        """Sentinel values (raw 0, 254, 255) must not appear as finite heights."""
        path, _ = tops_file
        result = read_tops(path)
        valid = result.values[np.isfinite(result.values)]
        # raw 0 → -100 m, raw 254 → 25300 m, raw 255 → 25400 m
        assert not np.any(valid < 0), "Negative height from raw=0 not masked"
        assert not np.any(valid == 25300), "raw=254 not masked"
        assert not np.any(valid == 25400), "raw=255 not masked"

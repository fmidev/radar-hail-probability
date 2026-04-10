"""Tests for NWP isotherm reader and interpolation."""

import datetime
import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hailathon.io.nwp import (
    _parse_data_row,
    _parse_text,
    read_isotherm_text,
    interpolate_to_grid,
)

DATA_DIR = Path(__file__).parent.parent / "legacy" / "data"
ZERO_PATH = DATA_DIR / "meps_zerolevel_stere_radar.txt"
M20_PATH = DATA_DIR / "meps_M20_level_stere_radar.txt"


# ---------------------------------------------------------------------------
# Unit tests: low-level parser
# ---------------------------------------------------------------------------

class TestParseDataRow:
    def test_normal_values(self):
        row = _parse_data_row("  100.5  200.3  300.0")
        np.testing.assert_allclose(row, [100.5, 200.3, 300.0])

    def test_missing_becomes_nan(self):
        row = _parse_data_row("  100.5        -   200.3")
        assert row[0] == pytest.approx(100.5)
        assert np.isnan(row[1])
        assert row[2] == pytest.approx(200.3)

    def test_all_missing(self):
        row = _parse_data_row("       -        -        -")
        assert len(row) == 3
        assert np.all(np.isnan(row))


class TestParseText:
    def test_synthetic_file(self, tmp_path):
        """Parse a minimal 2-timestep, 2×3 file."""
        content = textwrap.dedent("""\
            RAETUOTE / malli
            2026040900
            0     	  270 Nollarajan Korkeus
              10.0  20.0  30.0
              40.0  50.0  60.0

            2026040901
            0     	  270 Nollarajan Korkeus
              70.0  80.0       -
              100.0  110.0  120.0

        """)
        p = tmp_path / "test.txt"
        p.write_text(content)

        timestamps, grids = _parse_text(str(p))

        assert len(timestamps) == 2
        assert timestamps[0] == datetime.datetime(2026, 4, 9, 0)
        assert timestamps[1] == datetime.datetime(2026, 4, 9, 1)

        assert grids[0].shape == (2, 3)
        np.testing.assert_allclose(grids[0], [[10, 20, 30], [40, 50, 60]])

        assert grids[1].shape == (2, 3)
        assert grids[1][0, 0] == pytest.approx(70.0)
        assert np.isnan(grids[1][0, 2])


# ---------------------------------------------------------------------------
# Integration tests: real NWP files
# ---------------------------------------------------------------------------

class TestReadIsothermText:
    @pytest.fixture(params=["zero", "m20"])
    def nwp_data(self, request):
        path = ZERO_PATH if request.param == "zero" else M20_PATH
        if not path.exists():
            pytest.skip(f"{path.name} not found")
        return read_isotherm_text(str(path))

    def test_returns_dataarray(self, nwp_data):
        assert isinstance(nwp_data, xr.DataArray)

    def test_dims(self, nwp_data):
        assert nwp_data.dims == ("time", "y_nwp", "x_nwp")

    def test_coarse_grid_shape(self, nwp_data):
        # 26 × 45 coarse NWP grid
        assert nwp_data.sizes["x_nwp"] == 26
        assert nwp_data.sizes["y_nwp"] == 45

    def test_time_dimension(self, nwp_data):
        assert nwp_data.sizes["time"] == 37

    def test_heights_physical_range(self, nwp_data):
        valid = nwp_data.values[np.isfinite(nwp_data.values)]
        # Zero-level typically 0–4000 m; M20 typically 1000–8000 m
        assert valid.min() >= -500
        assert valid.max() <= 15000

    def test_zero_level_has_nan_for_missing_values(self):
        """The zero-level text file contains '-' for missing; these should be NaN."""
        if not ZERO_PATH.exists():
            pytest.skip("zero-level data not available")
        nwp = read_isotherm_text(str(ZERO_PATH))
        assert np.any(np.isnan(nwp.values))

    def test_y_axis_south_to_north(self, nwp_data):
        """y_nwp coordinates should increase (south → north)."""
        y = nwp_data.coords["y_nwp"].values
        assert np.all(np.diff(y) > 0)

    def test_x_axis_west_to_east(self, nwp_data):
        x = nwp_data.coords["x_nwp"].values
        assert np.all(np.diff(x) > 0)

    def test_time_coordinate_dtype(self, nwp_data):
        assert np.issubdtype(nwp_data.coords["time"].dtype, np.datetime64)


class TestInterpolateToGrid:
    def test_identity_on_nwp_grid(self):
        """Interpolating to the same coordinates should return the same values."""
        y = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.arange(12, dtype=np.float64).reshape(1, 3, 4)

        nwp = xr.DataArray(
            data,
            dims=["time", "y_nwp", "x_nwp"],
            coords={"time": [np.datetime64("2026-04-09")], "y_nwp": y, "x_nwp": x},
        )
        result = interpolate_to_grid(nwp, x, y)
        np.testing.assert_allclose(result.values, data[0], atol=1e-5)

    def test_midpoint_interpolation(self):
        """Bilinear at the midpoint of a 2×2 grid = average of corners."""
        y = np.array([0.0, 1.0])
        x = np.array([0.0, 1.0])
        data = np.array([[[0.0, 2.0], [4.0, 6.0]]])

        nwp = xr.DataArray(
            data,
            dims=["time", "y_nwp", "x_nwp"],
            coords={"time": [np.datetime64("2026-04-09")], "y_nwp": y, "x_nwp": x},
        )
        result = interpolate_to_grid(nwp, np.array([0.5]), np.array([0.5]))
        assert result.values[0, 0] == pytest.approx(3.0)

    def test_output_shape_matches_target(self):
        """Output should have the shape of target_y × target_x."""
        y = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0])
        data = np.ones((1, 3, 2))

        nwp = xr.DataArray(
            data,
            dims=["time", "y_nwp", "x_nwp"],
            coords={"time": [np.datetime64("2026-04-09")], "y_nwp": y, "x_nwp": x},
        )
        tx = np.linspace(0, 1, 10)
        ty = np.linspace(0, 2, 20)
        result = interpolate_to_grid(nwp, tx, ty)
        assert result.shape == (20, 10)

    @pytest.mark.skipif(not ZERO_PATH.exists(), reason="legacy data not available")
    def test_interpolate_real_data_to_standard_domain(self):
        """Smoke test: interpolate zero-level to the standard 929×1571 grid."""
        from hailathon.projection.grid import (
            STANDARD_SHAPE, STANDARD_X0, STANDARD_Y0, STANDARD_DX, STANDARD_DY,
            grid_coords,
        )
        nwp = read_isotherm_text(str(ZERO_PATH))
        x, y = grid_coords(
            STANDARD_SHAPE, STANDARD_X0, STANDARD_Y0, STANDARD_DX, STANDARD_DY,
        )
        result = interpolate_to_grid(nwp, x, y, time_index=0)
        assert result.shape == STANDARD_SHAPE
        valid = result.values[np.isfinite(result.values)]
        assert len(valid) > 0
        assert valid.min() >= -500
        assert valid.max() <= 5000

"""Tests for the processing pipeline wiring and helpers."""

import datetime
from datetime import timezone
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from hailathon.pipeline import _parse_timestamp, _select_time_index


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------

class TestParseTimestamp:
    def test_compact_format(self):
        ts = _parse_timestamp("20240601T1200Z")
        assert ts == datetime.datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

    def test_iso_full_format(self):
        ts = _parse_timestamp("2024-06-01T12:00:00Z")
        assert ts == datetime.datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

    def test_iso_no_seconds(self):
        ts = _parse_timestamp("2024-06-01T12:00Z")
        assert ts == datetime.datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

    def test_yymmddhh_format(self):
        ts = _parse_timestamp("2024060112")
        assert ts == datetime.datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_timestamp("not-a-timestamp")


# ---------------------------------------------------------------------------
# _select_time_index
# ---------------------------------------------------------------------------

class TestSelectTimeIndex:
    @pytest.fixture()
    def nwp(self):
        """NWP DataArray with 3 hourly timesteps starting 2026-04-09 00:00."""
        times = np.array(
            ["2026-04-09T00", "2026-04-09T01", "2026-04-09T02"],
            dtype="datetime64[h]",
        )
        data = np.zeros((3, 2, 2))
        return xr.DataArray(data, dims=["time", "y_nwp", "x_nwp"], coords={"time": times})

    def test_exact_match(self, nwp):
        target = datetime.datetime(2026, 4, 9, 1, 0, tzinfo=timezone.utc)
        assert _select_time_index(nwp, target) == 1

    def test_floors_minutes(self, nwp):
        """Target at 01:30 should match hour 01."""
        target = datetime.datetime(2026, 4, 9, 1, 30, tzinfo=timezone.utc)
        assert _select_time_index(nwp, target) == 1

    def test_nearest_fallback(self, nwp):
        """If no exact match, use nearest."""
        target = datetime.datetime(2026, 4, 9, 3, 0, tzinfo=timezone.utc)
        assert _select_time_index(nwp, target) == 2  # closest to hour 3 is hour 2


# ---------------------------------------------------------------------------
# process() — end-to-end with synthetic data
# ---------------------------------------------------------------------------

class TestProcess:
    @pytest.fixture()
    def synthetic_inputs(self, tmp_path):
        """Create minimal synthetic IRIS-like and NWP-like inputs on disk."""
        # We'll mock the I/O functions, so files don't need to be real
        return {
            "tops_45": str(tmp_path / "tops45.dat"),
            "tops_50": str(tmp_path / "tops50.dat"),
            "zero": str(tmp_path / "zero.txt"),
            "m20": str(tmp_path / "m20.txt"),
            "output_dir": str(tmp_path / "output"),
        }

    def _make_tops(self, value: float) -> xr.DataArray:
        """Create a small TOPS-like DataArray."""
        y = np.arange(4, dtype=np.float64) * 1000
        x = np.arange(3, dtype=np.float64) * 1000
        data = np.full((4, 3), value, dtype=np.float32)
        noecho = np.zeros((4, 3), dtype=bool)
        return xr.DataArray(
            data, dims=["y", "x"],
            coords={
                "x": ("x", x),
                "y": ("y", y),
                "noecho": (("y", "x"), noecho),
            },
        )

    def _make_nwp(self, value: float) -> xr.DataArray:
        """Create a small NWP-like DataArray."""
        times = np.array(["2026-04-09T00", "2026-04-09T01"], dtype="datetime64[h]")
        y = np.array([0.0, 3000.0])
        x = np.array([0.0, 2000.0])
        data = np.full((2, 2, 2), value, dtype=np.float64)
        return xr.DataArray(
            data, dims=["time", "y_nwp", "x_nwp"],
            coords={"time": times, "y_nwp": y, "x_nwp": x},
        )

    @patch("hailathon.pipeline.write_geotiff")
    @patch("hailathon.pipeline.write_odim")
    @patch("hailathon.pipeline.read_isotherm_text")
    @patch("hailathon.pipeline.read_tops")
    def test_smoke(
        self, mock_read_tops, mock_read_nwp, mock_odim, mock_tif,
        synthetic_inputs,
    ):
        from hailathon.pipeline import process

        # 45 dBZ tops at 5000 m, 50 dBZ tops at 6000 m
        mock_read_tops.side_effect = [self._make_tops(5000.0), self._make_tops(6000.0)]
        # Zero level at 2000 m, M20 at 4000 m
        mock_read_nwp.side_effect = [self._make_nwp(2000.0), self._make_nwp(4000.0)]

        result = process(
            synthetic_inputs["tops_45"],
            synthetic_inputs["tops_50"],
            synthetic_inputs["zero"],
            synthetic_inputs["m20"],
            synthetic_inputs["output_dir"],
            "20260409T0000Z",
        )

        assert "poh_odim" in result
        assert "lhi_odim" in result
        assert "poh_tif" in result
        assert "lhi_tif" in result

        # Verify writers were called (ODIM once per product, GeoTIFF once per product)
        assert mock_odim.call_count == 2
        assert mock_tif.call_count == 2

    @patch("hailathon.pipeline.write_geotiff")
    @patch("hailathon.pipeline.write_odim")
    @patch("hailathon.pipeline.read_isotherm_text")
    @patch("hailathon.pipeline.read_tops")
    def test_poh_values_reasonable(
        self, mock_read_tops, mock_read_nwp, mock_odim, mock_tif,
        synthetic_inputs,
    ):
        from hailathon.pipeline import process

        # tops = 5000 m, zero = 2000 m → dH = 3 km → POH = 0.319 + 0.133*3 = 0.718
        mock_read_tops.side_effect = [self._make_tops(5000.0), self._make_tops(6000.0)]
        mock_read_nwp.side_effect = [self._make_nwp(2000.0), self._make_nwp(4000.0)]

        process(
            synthetic_inputs["tops_45"],
            synthetic_inputs["tops_50"],
            synthetic_inputs["zero"],
            synthetic_inputs["m20"],
            synthetic_inputs["output_dir"],
            "20260409T0000Z",
        )

        # Check the POH array passed to write_odim (first call, second positional arg)
        poh_arg = mock_odim.call_args_list[0][0][1]
        expected_poh = 0.319 + 0.133 * 3.0  # dH = 3 km
        np.testing.assert_allclose(
            poh_arg.values, expected_poh, atol=0.01,
        )

    @patch("hailathon.pipeline.write_geotiff")
    @patch("hailathon.pipeline.write_odim")
    @patch("hailathon.pipeline.read_isotherm_text")
    @patch("hailathon.pipeline.read_tops")
    def test_lhi_values_reasonable(
        self, mock_read_tops, mock_read_nwp, mock_odim, mock_tif,
        synthetic_inputs,
    ):
        from hailathon.pipeline import process

        # tops_50 = 6000 m, m20 = 4000 m → dH = 2 km → LHI = 100 + 2 = 102
        mock_read_tops.side_effect = [self._make_tops(5000.0), self._make_tops(6000.0)]
        mock_read_nwp.side_effect = [self._make_nwp(2000.0), self._make_nwp(4000.0)]

        process(
            synthetic_inputs["tops_45"],
            synthetic_inputs["tops_50"],
            synthetic_inputs["zero"],
            synthetic_inputs["m20"],
            synthetic_inputs["output_dir"],
            "20260409T0000Z",
        )

        # Check the LHI array passed to write_odim (second call, second positional arg)
        lhi_arg = mock_odim.call_args_list[1][0][1]
        # tops_50 = 6000 m, m20 = 4000 m → LHI = 2000 m
        expected_lhi = 2000.0
        np.testing.assert_allclose(
            lhi_arg.values, expected_lhi, atol=0.01,
        )

    @patch("hailathon.pipeline.write_geotiff")
    @patch("hailathon.pipeline.write_odim")
    @patch("hailathon.pipeline.read_isotherm_text")
    @patch("hailathon.pipeline.read_tops")
    def test_output_dir_created(
        self, mock_read_tops, mock_read_nwp, mock_odim, mock_tif,
        synthetic_inputs,
    ):
        from hailathon.pipeline import process

        mock_read_tops.side_effect = [self._make_tops(5000.0), self._make_tops(6000.0)]
        mock_read_nwp.side_effect = [self._make_nwp(2000.0), self._make_nwp(4000.0)]

        import os
        out = synthetic_inputs["output_dir"]
        assert not os.path.exists(out)

        process(
            synthetic_inputs["tops_45"],
            synthetic_inputs["tops_50"],
            synthetic_inputs["zero"],
            synthetic_inputs["m20"],
            out,
            "20260409T0000Z",
        )
        assert os.path.isdir(out)

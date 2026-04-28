"""Integration tests for read_tops against real ODIM HDF5 files in tests/data/."""

import pathlib

import numpy as np
import pytest

from hailathon.io.tops import read_tops

DATA_DIR = pathlib.Path(__file__).parent / "data"

TOPS_FILES = {
    "202604280755_composite_etop_45_dbzh_finrad_qc.h5": {"dbz": 45},
    "202604280755_composite_etop_50_dbzh_finrad_qc.h5": {"dbz": 50},
}


@pytest.fixture(params=list(TOPS_FILES.keys()))
def tops_file(request):
    path = DATA_DIR / request.param
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return path, TOPS_FILES[request.param]


class TestReadTopsOdim:
    def test_returns_dataarray(self, tops_file):
        import xarray as xr
        path, _ = tops_file
        result = read_tops(str(path))
        assert isinstance(result, xr.DataArray)

    def test_shape(self, tops_file):
        path, _ = tops_file
        result = read_tops(str(path))
        assert result.shape == (6144, 5120)

    def test_dims_are_y_x(self, tops_file):
        path, _ = tops_file
        result = read_tops(str(path))
        assert result.dims == ("y", "x")

    def test_has_x_and_y_coords(self, tops_file):
        path, _ = tops_file
        result = read_tops(str(path))
        assert "x" in result.coords
        assert "y" in result.coords

    def test_has_noecho_coordinate(self, tops_file):
        path, _ = tops_file
        result = read_tops(str(path))
        assert "noecho" in result.coords
        assert result.coords["noecho"].dtype == bool

    def test_coords_are_monotonic(self, tops_file):
        path, _ = tops_file
        result = read_tops(str(path))
        assert bool(np.all(np.diff(result.coords["x"].values) > 0))
        assert bool(np.all(np.diff(result.coords["y"].values) > 0))

    def test_x_coords_approx(self, tops_file):
        # First pixel centre: -208000 + 250*0.5 = -207875
        # Last pixel centre:  -208000 + 250*5119.5 = 1071875
        path, _ = tops_file
        result = read_tops(str(path))
        assert result.coords["x"].values[0] == pytest.approx(-207875.0)
        assert result.coords["x"].values[-1] == pytest.approx(1071875.0)

    def test_y_coords_approx(self, tops_file):
        # First pixel centre (south): 6390000 + 250*0.5 = 6390125
        # Last pixel centre (north):  6390000 + 250*6143.5 = 7925875
        path, _ = tops_file
        result = read_tops(str(path))
        assert result.coords["y"].values[0] == pytest.approx(6390125.0)
        assert result.coords["y"].values[-1] == pytest.approx(7925875.0)

    def test_units_attribute_is_metres(self, tops_file):
        path, _ = tops_file
        result = read_tops(str(path))
        assert result.attrs["units"] == "m"

    def test_has_crs_wkt_attribute(self, tops_file):
        path, _ = tops_file
        result = read_tops(str(path))
        assert "crs_wkt" in result.attrs
        assert "3067" in result.attrs["crs_wkt"] or "TM35FIN" in result.attrs["crs_wkt"]

    def test_valid_heights_in_physical_range(self, tops_file):
        path, _ = tops_file
        result = read_tops(str(path))
        valid = result.values[np.isfinite(result.values)]
        if valid.size > 0:
            assert float(valid.min()) >= 0.0
            assert float(valid.max()) <= 25500.0

    def test_no_raw_special_values_in_output(self, tops_file):
        # undetect=0 and nodata=255 should not appear as physical values
        # undetect → height = 0*100-100 = -100 (masked to NaN)
        # nodata   → height = 255*100-100 = 25400 (masked to NaN)
        path, _ = tops_file
        result = read_tops(str(path))
        vals = result.values
        assert not np.any(vals == pytest.approx(-100.0))
        assert not np.any(vals == pytest.approx(25400.0))

"""Tests for ODIM HDF5 and GeoTIFF output writers."""

import numpy as np
import pytest
import xarray as xr

from hailathon.output.odim import write_odim, _encode_data, _PRODUCT_ENCODING
from hailathon.output.geotiff import write_geotiff


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_data():
    """A small 4×3 DataArray with x/y coordinates in projection metres."""
    y = np.array([0.0, 1000.0, 2000.0, 3000.0])
    x = np.array([0.0, 1000.0, 2000.0])
    values = np.array([
        [0.0, 0.25, 0.5],
        [0.75, 1.0, np.nan],
        [0.1, 0.2, 0.3],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float32)
    return xr.DataArray(
        values, dims=["y", "x"],
        coords={"x": ("x", x), "y": ("y", y)},
    )


# ---------------------------------------------------------------------------
# ODIM HDF5 writer
# ---------------------------------------------------------------------------

class TestEncodeData:
    def test_poh_encodes_to_uint8(self):
        values = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        raw = _encode_data(values, _PRODUCT_ENCODING["POH"])
        assert raw.dtype == np.uint8

    def test_poh_zero_maps_to_low_raw(self):
        values = np.array([[0.0]], dtype=np.float32)
        raw = _encode_data(values, _PRODUCT_ENCODING["POH"])
        # 0.0 / (1/250) = 0 → clamped to 1
        assert raw[0, 0] == 1  # clamp from 0 to minimum valid=1

    def test_poh_one_maps_to_250(self):
        values = np.array([[1.0]], dtype=np.float32)
        raw = _encode_data(values, _PRODUCT_ENCODING["POH"])
        assert raw[0, 0] == 250

    def test_nan_becomes_nodata(self):
        values = np.array([[np.nan]], dtype=np.float32)
        raw = _encode_data(values, _PRODUCT_ENCODING["POH"])
        assert raw[0, 0] == 255

    def test_noecho_becomes_undetect(self):
        values = np.array([[np.nan, 0.5]], dtype=np.float32)
        noecho = np.array([[True, False]])
        raw = _encode_data(values, _PRODUCT_ENCODING["POH"], noecho=noecho)
        assert raw[0, 0] == 0   # undetect
        assert raw[0, 1] != 0   # valid value

    def test_lhi_round_trip(self):
        """Encode LHI metre value and verify it can be decoded back."""
        enc = _PRODUCT_ENCODING["LHI"]
        values = np.array([[2000.0]], dtype=np.float32)
        raw = _encode_data(values, enc)
        decoded = raw[0, 0] * enc["gain"] + enc["offset"]
        assert decoded == pytest.approx(2000.0)


class TestWriteOdim:
    def test_creates_valid_hdf5(self, tmp_path, sample_data):
        import h5py
        out = str(tmp_path / "test.h5")
        write_odim(out, sample_data, "POH", "20260409T0100Z")

        with h5py.File(out, "r") as f:
            assert "what" in f
            assert "where" in f
            assert "dataset1" in f

    def test_root_what_attributes(self, tmp_path, sample_data):
        import h5py
        out = str(tmp_path / "test.h5")
        write_odim(out, sample_data, "POH", "20260409T0100Z")

        with h5py.File(out, "r") as f:
            what = f["what"]
            assert what.attrs["object"] == "COMP"
            assert what.attrs["date"] == "20260409"
            assert what.attrs["time"] == "010000"

    def test_where_has_grid_params(self, tmp_path, sample_data):
        import h5py
        out = str(tmp_path / "test.h5")
        write_odim(out, sample_data, "POH", "20260409T0100Z")

        with h5py.File(out, "r") as f:
            where = f["where"]
            assert where.attrs["xsize"] == 3
            assert where.attrs["ysize"] == 4
            assert where.attrs["xscale"] == pytest.approx(1000.0)
            assert where.attrs["yscale"] == pytest.approx(1000.0)

    def test_where_has_corner_coordinates(self, tmp_path, sample_data):
        import h5py
        out = str(tmp_path / "test.h5")
        write_odim(out, sample_data, "POH", "20260409T0100Z")

        with h5py.File(out, "r") as f:
            where = f["where"]
            for corner in ("LL", "UL", "UR", "LR"):
                assert f"{corner}_lon" in where.attrs
                assert f"{corner}_lat" in where.attrs

    def test_dataset_data_shape(self, tmp_path, sample_data):
        import h5py
        out = str(tmp_path / "test.h5")
        write_odim(out, sample_data, "POH", "20260409T0100Z")

        with h5py.File(out, "r") as f:
            data = f["dataset1/data1/data"][:]
            assert data.shape == (4, 3)
            assert data.dtype == np.uint8

    def test_dataset_what_has_encoding(self, tmp_path, sample_data):
        import h5py
        out = str(tmp_path / "test.h5")
        write_odim(out, sample_data, "POH", "20260409T0100Z")

        with h5py.File(out, "r") as f:
            d_what = f["dataset1/data1/what"]
            assert d_what.attrs["quantity"] == "POH"
            assert "gain" in d_what.attrs
            assert "offset" in d_what.attrs
            assert "nodata" in d_what.attrs
            assert "undetect" in d_what.attrs

    def test_poh_values_recoverable(self, tmp_path):
        """Write known POH values and verify round-trip through uint8 encoding."""
        import h5py
        y = np.array([0.0, 1000.0])
        x = np.array([0.0, 1000.0])
        values = np.array([[0.5, 0.8], [0.0, 1.0]], dtype=np.float32)
        da = xr.DataArray(values, dims=["y", "x"], coords={"x": ("x", x), "y": ("y", y)})

        out = str(tmp_path / "poh.h5")
        write_odim(out, da, "POH", "20260409T0100Z")

        with h5py.File(out, "r") as f:
            raw = f["dataset1/data1/data"][:]
            gain = f["dataset1/data1/what"].attrs["gain"]
            offset = f["dataset1/data1/what"].attrs["offset"]
            recovered = raw.astype(np.float64) * gain + offset
            # Data is Y-flipped in ODIM (north at top), so flip back
            recovered = recovered[::-1]
            np.testing.assert_allclose(recovered, values, atol=gain)

    def test_lhi_product(self, tmp_path):
        """LHI metre values round-trip correctly."""
        import h5py
        y = np.array([0.0, 1000.0])
        x = np.array([0.0, 1000.0])
        values = np.array([[2000.0, 3500.0], [500.0, 5000.0]], dtype=np.float32)
        da = xr.DataArray(values, dims=["y", "x"], coords={"x": ("x", x), "y": ("y", y)})

        out = str(tmp_path / "lhi.h5")
        write_odim(out, da, "LHI", "20260409T0100Z")

        with h5py.File(out, "r") as f:
            raw = f["dataset1/data1/data"][:]
            gain = f["dataset1/data1/what"].attrs["gain"]
            offset = f["dataset1/data1/what"].attrs["offset"]
            recovered = raw[::-1].astype(np.float64) * gain + offset
            np.testing.assert_allclose(recovered, values, atol=gain)


# ---------------------------------------------------------------------------
# GeoTIFF writer
# ---------------------------------------------------------------------------

class TestWriteGeotiff:
    def test_creates_valid_tiff(self, tmp_path, sample_data):
        import rasterio
        out = str(tmp_path / "test.tif")
        write_geotiff(out, sample_data, "POH")

        with rasterio.open(out) as src:
            assert src.count == 1
            assert src.width == 3
            assert src.height == 4

    def test_crs_is_embedded(self, tmp_path, sample_data):
        import rasterio
        out = str(tmp_path / "test.tif")
        write_geotiff(out, sample_data, "POH")

        with rasterio.open(out) as src:
            assert src.crs is not None
            assert "stere" in src.crs.to_proj4()

    def test_nodata_is_255(self, tmp_path, sample_data):
        import rasterio
        out = str(tmp_path / "test.tif")
        write_geotiff(out, sample_data, "POH")

        with rasterio.open(out) as src:
            assert src.nodata == 255
            assert src.dtypes[0] == "uint8"

    def test_values_round_trip(self, tmp_path):
        """Write known POH values and recover via scale/offset."""
        import rasterio
        y = np.array([0.0, 1000.0])
        x = np.array([0.0, 1000.0])
        values = np.array([[0.5, 0.8], [0.0, 1.0]], dtype=np.float32)
        da = xr.DataArray(values, dims=["y", "x"], coords={"x": ("x", x), "y": ("y", y)})

        out = str(tmp_path / "test.tif")
        write_geotiff(out, da, "POH")

        with rasterio.open(out) as src:
            raw = src.read(1)
            scale = float(src.tags(1)["scale"])
            offset = float(src.tags(1)["offset"])
            recovered = raw[::-1].astype(np.float64) * scale + offset
            np.testing.assert_allclose(recovered, values, atol=scale)

    def test_nan_pixels_become_nodata(self, tmp_path, sample_data):
        """NaN in input should become nodata (255) in the output."""
        import rasterio
        out = str(tmp_path / "test.tif")
        write_geotiff(out, sample_data, "POH")

        with rasterio.open(out) as src:
            raw = src.read(1)
            # Row 0 in TIFF = top = last row of our array (all NaN → 255)
            assert np.all(raw[0, :] == 255)

    def test_product_tag(self, tmp_path, sample_data):
        import rasterio
        out = str(tmp_path / "test.tif")
        write_geotiff(out, sample_data, "LHI")

        with rasterio.open(out) as src:
            tags = src.tags()
            assert tags.get("product") == "LHI"

    def test_transform_bounds(self, tmp_path, sample_data):
        """GeoTIFF bounds should match the pixel-edge extent."""
        import rasterio
        out = str(tmp_path / "test.tif")
        write_geotiff(out, sample_data, "POH")

        with rasterio.open(out) as src:
            bounds = src.bounds
            # x: centres 0, 1000, 2000 with dx=1000 → edges -500 to 2500
            assert bounds.left == pytest.approx(-500.0)
            assert bounds.right == pytest.approx(2500.0)
            # y: centres 0, 1000, 2000, 3000 with dy=1000 → edges -500 to 3500
            assert bounds.bottom == pytest.approx(-500.0)
            assert bounds.top == pytest.approx(3500.0)

    def test_noecho_becomes_undetect(self, tmp_path):
        """Pixels with noecho=True should be encoded as undetect (0)."""
        import rasterio
        y = np.array([0.0, 1000.0])
        x = np.array([0.0, 1000.0])
        values = np.array([[np.nan, 0.5], [np.nan, 0.8]], dtype=np.float32)
        noecho = np.array([[True, False], [False, False]])
        da = xr.DataArray(
            values, dims=["y", "x"],
            coords={"x": ("x", x), "y": ("y", y), "noecho": (("y", "x"), noecho)},
        )

        out = str(tmp_path / "test.tif")
        write_geotiff(out, da, "POH")

        with rasterio.open(out) as src:
            raw = src.read(1)
            # Flip back to south-up for comparison
            raw = raw[::-1]
            assert raw[0, 0] == 0    # noecho → undetect
            assert raw[1, 0] == 255  # NaN without noecho → nodata

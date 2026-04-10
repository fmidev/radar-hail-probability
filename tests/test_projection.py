"""Tests for FIN1000 grid definition and coordinate utilities."""

import numpy as np
import pytest
import pyproj

from hailathon.projection.grid import (
    CRS,
    LARGE_SHAPE, LARGE_X0, LARGE_Y0, LARGE_DX, LARGE_DY,
    STANDARD_SHAPE, STANDARD_X0, STANDARD_Y0, STANDARD_DX, STANDARD_DY,
    FULL_SHAPE, SUB_SHAPE, DX, DY,  # backwards-compat aliases
    grid_coords,
    grid_lonlat,
)


class TestGridCoords:
    def test_large_shape_lengths(self):
        x, y = grid_coords(LARGE_SHAPE)
        rows, cols = LARGE_SHAPE
        assert len(x) == cols
        assert len(y) == rows

    def test_standard_shape_lengths(self):
        x, y = grid_coords(STANDARD_SHAPE, STANDARD_X0, STANDARD_Y0, STANDARD_DX, STANDARD_DY)
        rows, cols = STANDARD_SHAPE
        assert len(x) == cols
        assert len(y) == rows

    def test_aliases_unchanged(self):
        assert FULL_SHAPE == LARGE_SHAPE
        assert SUB_SHAPE == STANDARD_SHAPE
        assert DX == LARGE_DX
        assert DY == LARGE_DY

    def test_x_spacing_equals_dx(self):
        x, _ = grid_coords(LARGE_SHAPE)
        assert np.allclose(np.diff(x), LARGE_DX)

    def test_y_spacing_equals_dy(self):
        _, y = grid_coords(LARGE_SHAPE)
        assert np.allclose(np.diff(y), LARGE_DY)

    def test_coords_are_pixel_centres(self):
        # First pixel centre should be x0 + dx/2
        x, y = grid_coords(LARGE_SHAPE)
        assert x[0] == pytest.approx(LARGE_X0 + LARGE_DX / 2)
        assert y[0] == pytest.approx(LARGE_Y0 + LARGE_DY / 2)

    def test_large_domain_sw_corner_near_zero(self):
        # SW edge of pixel (0,0) in the large domain is at (0, 0)
        x, y = grid_coords(LARGE_SHAPE)
        assert x[0] == pytest.approx(LARGE_DX / 2, rel=1e-3)
        assert y[0] == pytest.approx(LARGE_DY / 2, rel=1e-3)

    def test_custom_shape_and_params(self):
        x, y = grid_coords((10, 20), x0=100.0, y0=200.0, dx=50.0, dy=50.0)
        assert len(x) == 20
        assert len(y) == 10
        assert x[0] == pytest.approx(125.0)   # 100 + 50/2
        assert y[0] == pytest.approx(225.0)


class TestGridLonlat:
    def test_output_shapes_are_2d(self):
        rows, cols = 10, 8
        lon, lat = grid_lonlat((rows, cols))
        assert lon.shape == (rows, cols)
        assert lat.shape == (rows, cols)

    def test_finland_latitude_range(self):
        lon, lat = grid_lonlat(LARGE_SHAPE)
        assert lat.min() > 55.0
        assert lat.max() < 73.0

    def test_finland_longitude_range(self):
        lon, lat = grid_lonlat(LARGE_SHAPE)
        assert lon.min() > 14.0
        assert lon.max() < 43.0

    def test_no_nan_in_output(self):
        lon, lat = grid_lonlat((10, 10))
        assert not np.any(np.isnan(lon))
        assert not np.any(np.isnan(lat))

    def test_sw_corner_large_domain_geographic_accuracy(self):
        """SW corner of large domain should be near 18.6°E, 57.93°N."""
        fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), CRS, always_xy=True)
        expected_x, expected_y = fwd.transform(18.6, 57.93)
        # SW edge of pixel (0,0) = x0, y0 ≈ (0, 0) by construction
        assert abs(expected_x - LARGE_X0) < 500   # within one pixel
        assert abs(expected_y - LARGE_Y0) < 500

    def test_ne_corner_large_domain_geographic_accuracy(self):
        """NE corner of large domain should be near 34.9°E, 69.0°N."""
        fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), CRS, always_xy=True)
        expected_x, expected_y = fwd.transform(34.9, 69.0)
        ne_x = LARGE_X0 + LARGE_DX * LARGE_SHAPE[1]
        ne_y = LARGE_Y0 + LARGE_DY * LARGE_SHAPE[0]
        assert abs(expected_x - ne_x) < 500
        assert abs(expected_y - ne_y) < 500

    def test_round_trip_projection(self):
        """Forward + inverse transform should recover original coordinates."""
        inv = pyproj.Transformer.from_crs(CRS, pyproj.CRS.from_epsg(4326), always_xy=True)
        fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), CRS, always_xy=True)
        lons_in = np.array([20.0, 25.0, 30.0])
        lats_in = np.array([60.0, 63.0, 65.0])
        x, y = fwd.transform(lons_in, lats_in)
        lons_out, lats_out = inv.transform(x, y)
        np.testing.assert_allclose(lons_out, lons_in, atol=1e-8)
        np.testing.assert_allclose(lats_out, lats_in, atol=1e-8)


class TestCrs:
    def test_crs_is_projected(self):
        assert CRS.is_projected

    def test_crs_is_stereographic(self):
        assert "stere" in CRS.to_proj4().lower()

    def test_earth_radius(self):
        # Legacy code uses spherical earth a=6371000 m
        params = CRS.to_proj4()
        assert "6371000" in params

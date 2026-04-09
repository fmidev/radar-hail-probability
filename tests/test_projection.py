"""Tests for FIN1000 grid definition and coordinate utilities."""

import numpy as np
import pytest

from hailathon.projection.grid import (
    CRS,
    FULL_SHAPE,
    SUB_SHAPE,
    DX,
    DY,
    grid_coords,
    grid_lonlat,
)


class TestGridCoords:
    def test_full_shape_lengths(self):
        x, y = grid_coords(FULL_SHAPE)
        rows, cols = FULL_SHAPE
        assert len(x) == cols
        assert len(y) == rows

    def test_sub_shape_lengths(self):
        x, y = grid_coords(SUB_SHAPE)
        rows, cols = SUB_SHAPE
        assert len(x) == cols
        assert len(y) == rows

    def test_x_spacing_equals_dx(self):
        x, _ = grid_coords(FULL_SHAPE)
        diffs = np.diff(x)
        assert np.allclose(diffs, DX)

    def test_y_spacing_equals_dy(self):
        _, y = grid_coords(FULL_SHAPE)
        diffs = np.diff(y)
        assert np.allclose(diffs, DY)

    def test_custom_shape(self):
        x, y = grid_coords((10, 20))
        assert len(x) == 20
        assert len(y) == 10


class TestGridLonlat:
    def test_output_shapes_are_2d(self):
        rows, cols = 10, 8
        lon, lat = grid_lonlat((rows, cols))
        assert lon.shape == (rows, cols)
        assert lat.shape == (rows, cols)

    def test_finland_latitude_range(self):
        # Full domain should cover roughly 57–71 °N
        lon, lat = grid_lonlat(FULL_SHAPE)
        assert lat.min() > 55.0
        assert lat.max() < 73.0

    def test_finland_longitude_range(self):
        # Full domain should cover roughly 17–40 °E
        lon, lat = grid_lonlat(FULL_SHAPE)
        assert lon.min() > 14.0
        assert lon.max() < 43.0

    def test_no_nan_in_output(self):
        lon, lat = grid_lonlat((10, 10))
        assert not np.any(np.isnan(lon))
        assert not np.any(np.isnan(lat))


class TestCrs:
    def test_crs_is_projected(self):
        assert CRS.is_projected

    def test_crs_is_stereographic(self):
        assert "stere" in CRS.to_proj4().lower()

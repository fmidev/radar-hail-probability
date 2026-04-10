"""Tests for IRIS TOPS reader utilities."""

import numpy as np
import pytest

from hailathon.io.iris import _decode_heights, _extract_raw_data, _NO_ECHO, _SPECIAL, _UNDEFINED


class TestDecodeHeights:
    def test_normal_value_decodes_to_metres(self):
        # raw = height_m/100 + 1  →  raw 21 = 2000 m
        raw = np.array([[21]], dtype=np.uint8)
        heights, noecho = _decode_heights(raw)
        assert heights[0, 0] == pytest.approx(2000.0)
        assert not noecho[0, 0]

    def test_raw_1_decodes_to_zero_metres(self):
        raw = np.array([[1]], dtype=np.uint8)
        heights, _ = _decode_heights(raw)
        assert heights[0, 0] == pytest.approx(0.0)

    def test_raw_101_decodes_to_10000_metres(self):
        # 10 km echo top
        raw = np.array([[101]], dtype=np.uint8)
        heights, _ = _decode_heights(raw)
        assert heights[0, 0] == pytest.approx(10000.0)

    def test_no_echo_becomes_nan(self):
        raw = np.array([[_NO_ECHO]], dtype=np.uint8)
        heights, noecho = _decode_heights(raw)
        assert np.isnan(heights[0, 0])
        assert noecho[0, 0]

    def test_special_becomes_nan(self):
        raw = np.array([[_SPECIAL]], dtype=np.uint8)
        heights, noecho = _decode_heights(raw)
        assert np.isnan(heights[0, 0])
        assert not noecho[0, 0]

    def test_undefined_becomes_nan(self):
        raw = np.array([[_UNDEFINED]], dtype=np.uint8)
        heights, noecho = _decode_heights(raw)
        assert np.isnan(heights[0, 0])
        assert not noecho[0, 0]

    def test_mixed_array(self):
        raw = np.array([[_NO_ECHO, 21, _UNDEFINED, _SPECIAL, 51]], dtype=np.uint8)
        heights, noecho = _decode_heights(raw)
        assert np.isnan(heights[0, 0])
        assert heights[0, 1] == pytest.approx(2000.0)
        assert np.isnan(heights[0, 2])
        assert np.isnan(heights[0, 3])
        assert heights[0, 4] == pytest.approx(5000.0)
        np.testing.assert_array_equal(noecho[0], [True, False, False, False, False])

    def test_output_dtype_is_float32(self):
        raw = np.array([[21]], dtype=np.uint8)
        heights, _ = _decode_heights(raw)
        assert heights.dtype == np.float32

    def test_preserves_shape(self):
        raw = np.zeros((10, 20), dtype=np.uint8)
        heights, noecho = _decode_heights(raw)
        assert heights.shape == (10, 20)
        assert noecho.shape == (10, 20)


class _FakeIrisFile:
    """Minimal stand-in for IrisCartesianProductFile used in unit tests."""

    def __init__(self, array: np.ndarray):
        # f.data is an OrderedDict; index 0 holds the image array
        self.data = {0: array}


class TestExtractRawData:
    def test_2d_array_returned_as_is(self):
        arr = np.arange(6, dtype=np.uint8).reshape(2, 3)
        result = _extract_raw_data(_FakeIrisFile(arr), ydim=2, xdim=3)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, arr)

    def test_3d_array_leading_dim_dropped(self):
        arr = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
        result = _extract_raw_data(_FakeIrisFile(arr), ydim=2, xdim=3)
        assert result.shape == (2, 3)

    def test_raises_on_shape_mismatch(self):
        arr = np.zeros((2, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            _extract_raw_data(_FakeIrisFile(arr), ydim=3, xdim=3)

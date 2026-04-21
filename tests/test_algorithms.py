"""Tests for POH, LHI, and THI algorithm implementations."""

import numpy as np
import pytest
import xarray as xr

from hailathon.algorithms.poh import compute_poh, compute_hhi
from hailathon.algorithms.lhi import compute_lhi, compute_thi


def _da(values):
    """Wrap a list/array in a plain DataArray for convenience."""
    return xr.DataArray(np.array(values, dtype=np.float64))


class TestComputePoh:
    def test_formula_at_zero_crossing(self):
        # dH = 0 km → POH = 0.319
        tops = _da([2000.0])
        zero = _da([2000.0])
        result = compute_poh(tops, zero)
        assert float(result[0]) == pytest.approx(0.319)

    def test_clamps_to_zero(self):
        # Tops well below zero level → POH < 0 before clamping
        tops = _da([0.0])
        zero = _da([5000.0])
        result = compute_poh(tops, zero)
        assert float(result[0]) == pytest.approx(0.0)

    def test_clamps_to_one(self):
        # Tops well above zero level → POH > 1 before clamping
        tops = _da([20000.0])
        zero = _da([0.0])
        result = compute_poh(tops, zero)
        assert float(result[0]) == pytest.approx(1.0)

    def test_known_value(self):
        # tops 3 km above zero: dH = 3 → POH = 0.319 + 0.133*3 = 0.718
        tops = _da([5000.0])
        zero = _da([2000.0])
        result = compute_poh(tops, zero)
        assert float(result[0]) == pytest.approx(0.319 + 0.133 * 3.0)

    def test_saturates_well_above_zero(self):
        # dH = 6 km → POH = 0.319 + 0.798 = 1.117, clamped to 1.0
        tops = _da([8000.0])
        zero = _da([2000.0])
        result = compute_poh(tops, zero)
        assert float(result[0]) == pytest.approx(1.0)

    def test_nan_tops_propagates(self):
        tops = _da([np.nan])
        zero = _da([2000.0])
        result = compute_poh(tops, zero)
        assert np.isnan(float(result[0]))

    def test_nan_zero_level_propagates(self):
        tops = _da([5000.0])
        zero = _da([np.nan])
        result = compute_poh(tops, zero)
        assert np.isnan(float(result[0]))

    def test_2d_array(self):
        tops = xr.DataArray(np.full((3, 4), 5000.0))
        zero = xr.DataArray(np.full((3, 4), 2000.0))
        result = compute_poh(tops, zero)
        assert result.shape == (3, 4)
        expected = pytest.approx(0.319 + 0.133 * 3.0)
        assert float(result[0, 0]) == expected
        assert float(result[2, 3]) == expected


class TestComputeHhi:
    def test_formula_at_zero_crossing(self):
        # dH = 0 km → HHI = 10 × 0.319 = 3.19
        tops = _da([2000.0])
        zero = _da([2000.0])
        result = compute_hhi(tops, zero)
        assert float(result[0]) == pytest.approx(3.19)

    def test_known_value(self):
        # tops 3 km above zero: dH=3 → HHI = 10 × (0.319 + 0.133×3) = 7.18
        tops = _da([5000.0])
        zero = _da([2000.0])
        result = compute_hhi(tops, zero)
        assert float(result[0]) == pytest.approx(7.18)

    def test_no_upper_limit(self):
        # Very deep convection (dH=10 km) → HHI = 10 × (0.319 + 1.33) = 16.49, not clamped
        tops = _da([12000.0])
        zero = _da([2000.0])
        result = compute_hhi(tops, zero)
        assert float(result[0]) == pytest.approx(10.0 * (0.319 + 0.133 * 10.0))
        assert float(result[0]) > 10.0

    def test_nan_tops_propagates(self):
        tops = _da([np.nan])
        zero = _da([2000.0])
        result = compute_hhi(tops, zero)
        assert np.isnan(float(result[0]))

    def test_hhi_is_ten_times_raw_poh(self):
        # HHI should equal 10 × unclipped POH for values where POH < 1
        tops = _da([5000.0])
        zero = _da([2000.0])
        hhi = compute_hhi(tops, zero)
        poh = compute_poh(tops, zero)
        assert float(hhi[0]) == pytest.approx(10.0 * float(poh[0]))


class TestComputeLhi:
    def test_formula_when_tops_equals_m20(self):
        # dH = 0 m
        tops = _da([5000.0])
        m20 = _da([5000.0])
        result = compute_lhi(tops, m20)
        assert float(result[0]) == pytest.approx(0.0)

    def test_tops_above_m20(self):
        # tops 2000 m above -20 level
        tops = _da([7000.0])
        m20 = _da([5000.0])
        result = compute_lhi(tops, m20)
        assert float(result[0]) == pytest.approx(2000.0)

    def test_tops_below_m20(self):
        # tops 1000 m below -20 level
        tops = _da([4000.0])
        m20 = _da([5000.0])
        result = compute_lhi(tops, m20)
        assert float(result[0]) == pytest.approx(-1000.0)

    def test_nan_tops_propagates(self):
        tops = _da([np.nan])
        m20 = _da([5000.0])
        result = compute_lhi(tops, m20)
        assert np.isnan(float(result[0]))


class TestComputeThi:
    def _hhi(self, val):
        return _da([float(val)])

    def _zero(self, val):
        return _da([float(val)])

    def test_zero_below_1200m_adds_2(self):
        result = compute_thi(self._hhi(7.5), self._zero(1000.0))
        assert float(result[0]) == pytest.approx(9.5)

    def test_zero_at_1200m_boundary_adds_2(self):
        result = compute_thi(self._hhi(7.5), self._zero(1200.0))
        assert float(result[0]) == pytest.approx(9.5)

    def test_zero_between_1200_and_1700_adds_1(self):
        result = compute_thi(self._hhi(7.5), self._zero(1500.0))
        assert float(result[0]) == pytest.approx(8.5)

    def test_zero_at_1700m_boundary_adds_1(self):
        result = compute_thi(self._hhi(7.5), self._zero(1700.0))
        assert float(result[0]) == pytest.approx(8.5)

    def test_zero_between_1700_and_3500_unchanged(self):
        result = compute_thi(self._hhi(7.5), self._zero(2500.0))
        assert float(result[0]) == pytest.approx(7.5)

    def test_zero_at_3500m_boundary_unchanged(self):
        result = compute_thi(self._hhi(7.5), self._zero(3500.0))
        assert float(result[0]) == pytest.approx(7.5)

    def test_zero_above_3500_subtracts_1(self):
        result = compute_thi(self._hhi(7.5), self._zero(4000.0))
        assert float(result[0]) == pytest.approx(6.5)

    def test_nan_hhi_propagates(self):
        result = compute_thi(self._hhi(float("nan")), self._zero(2000.0))
        assert np.isnan(float(result[0]))

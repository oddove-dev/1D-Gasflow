"""Tests for friction factor correlations."""
from __future__ import annotations

import math

import numpy as np
import pytest

from gas_pipe.friction import (
    darcy_friction,
    friction_blended,
    friction_chen,
    friction_colebrook,
    friction_laminar,
)


class TestLaminar:
    def test_f_times_Re_is_64(self):
        for Re in [100, 500, 1000, 2000]:
            f = friction_laminar(Re)
            assert abs(f * Re - 64.0) < 1e-10, f"f*Re={f*Re:.10f} at Re={Re}"


class TestChenVsColebrook:
    @pytest.mark.parametrize("Re", [1e4, 1e5, 1e6, 1e7, 1e8])
    @pytest.mark.parametrize("eps_D", [1e-6, 1e-4, 1e-2])
    def test_chen_colebrook_agreement(self, Re, eps_D):
        f_chen = friction_chen(Re, eps_D)
        f_cb = friction_colebrook(Re, eps_D)
        rel_err = abs(f_chen - f_cb) / f_cb
        assert rel_err < 0.005, (
            f"Chen vs Colebrook deviation {rel_err:.2%} at Re={Re:.0e}, ε/D={eps_D:.0e}"
        )


class TestBlended:
    def test_continuous_at_Re_2300(self):
        eps_D = 1e-4
        Re_values = np.linspace(1800, 2800, 200)
        f_values = [friction_blended(Re, eps_D) for Re in Re_values]
        # Check no jumps > 10% between adjacent values
        for i in range(len(f_values) - 1):
            if f_values[i] > 0:
                jump = abs(f_values[i+1] - f_values[i]) / f_values[i]
                assert jump < 0.15, f"Jump of {jump:.2%} at Re={Re_values[i]:.0f}"

    def test_continuous_at_Re_4000(self):
        eps_D = 1e-4
        Re_values = np.linspace(3500, 4500, 200)
        f_values = [friction_blended(Re, eps_D) for Re in Re_values]
        for i in range(len(f_values) - 1):
            if f_values[i] > 0:
                jump = abs(f_values[i+1] - f_values[i]) / f_values[i]
                assert jump < 0.15, f"Jump of {jump:.2%} at Re={Re_values[i]:.0f}"

    def test_laminar_limit(self):
        eps_D = 1e-4
        f = friction_blended(1000.0, eps_D)
        f_lam = friction_laminar(1000.0)
        assert abs(f - f_lam) / f_lam < 0.01

    def test_turbulent_limit_high_Re(self):
        eps_D = 1e-4
        f_blend = friction_blended(1e6, eps_D)
        f_chen = friction_chen(1e6, eps_D)
        assert abs(f_blend - f_chen) / f_chen < 0.01


class TestDarcyDispatch:
    def test_dispatch_laminar(self):
        f = darcy_friction(500.0, 1e-4, "laminar")
        assert abs(f * 500.0 - 64.0) < 1e-10

    def test_dispatch_chen(self):
        f = darcy_friction(1e5, 1e-4, "chen")
        assert 0.005 < f < 0.05

    def test_dispatch_colebrook(self):
        f = darcy_friction(1e5, 1e-4, "colebrook")
        assert 0.005 < f < 0.05

    def test_dispatch_blended(self):
        f = darcy_friction(3000.0, 1e-4, "blended")
        assert f > 0

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            darcy_friction(1e5, 1e-4, "moody")


class TestPhysicalRange:
    def test_f_always_positive(self):
        for Re in [100, 2300, 4000, 1e5, 1e7]:
            assert darcy_friction(Re, 1e-4) > 0

    def test_smooth_turbulent_asymptote(self):
        # Very smooth pipe, high Re → Prandtl–von Kármán
        f = friction_colebrook(1e7, 1e-8)
        assert 0.005 < f < 0.02

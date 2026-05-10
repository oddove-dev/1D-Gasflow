"""Tests for GERG-2008 EOS wrapper."""
from __future__ import annotations

import pytest

from gas_pipe.eos import ALLOWED_COMPONENTS, GERGFluid
from gas_pipe.errors import EOSOutOfRange


class TestPureMethane:
    def setup_method(self):
        self.fluid = GERGFluid({"Methane": 1.0})

    def test_density_at_100bara_300K(self):
        state = self.fluid.props(100e5, 300.0)
        # CoolProp reference: ~73.5 kg/m³
        assert 65.0 < state.rho < 82.0, f"rho={state.rho:.2f} kg/m³ out of expected range"

    def test_Z_at_100bara_300K(self):
        state = self.fluid.props(100e5, 300.0)
        # Z ≈ 0.875
        assert 0.80 < state.Z < 0.95, f"Z={state.Z:.3f} out of expected range"

    def test_sound_speed_finite_positive(self):
        state = self.fluid.props(50e5, 300.0)
        assert state.a > 100.0

    def test_viscosity_positive(self):
        state = self.fluid.props(50e5, 300.0)
        assert state.mu > 0.0

    def test_mu_JT_sign(self):
        # For methane at high P, JT coefficient should be positive (cooling on expansion)
        state = self.fluid.props(100e5, 300.0)
        assert state.mu_JT > 0.0, f"mu_JT={state.mu_JT:.4e} should be positive"

    def test_cache_hit(self):
        self.fluid.props(50e5, 300.0)
        self.fluid.props(50e5, 300.0)
        stats = self.fluid.cache_stats()
        assert stats["hits"] >= 1

    def test_cache_no_cross_instance_pollution(self):
        f2 = GERGFluid({"Methane": 0.9, "Ethane": 0.1})
        s1 = self.fluid.props(50e5, 300.0)
        s2 = f2.props(50e5, 300.0)
        assert s1.rho != s2.rho, "Cross-instance cache pollution detected"


class TestMixture:
    def test_mixture_properties_finite(self):
        fluid = GERGFluid({"Methane": 0.90, "Ethane": 0.05, "Propane": 0.05})
        state = fluid.props(50e5, 300.0)
        assert state.rho > 0
        assert state.a > 0
        assert state.mu > 0
        assert state.cp > 0

    def test_molar_mass_reasonable(self):
        fluid = GERGFluid({"Methane": 1.0})
        # CH4 = 16.04 g/mol
        assert 0.015 < fluid.molar_mass < 0.018

    def test_normalization_warning(self):
        with pytest.warns(UserWarning):
            fluid = GERGFluid({"Methane": 0.85, "Ethane": 0.08, "Propane": 0.10})
        state = fluid.props(50e5, 300.0)
        assert state.rho > 0

    def test_props_Ph_roundtrip(self):
        fluid = GERGFluid({"Methane": 1.0})
        state_PT = fluid.props(50e5, 300.0)
        state_Ph = fluid.props_Ph(50e5, state_PT.h)
        assert abs(state_Ph.T - 300.0) < 0.5, f"Ph roundtrip T={state_Ph.T:.2f} K"


class TestErrors:
    def test_out_of_range_raises(self):
        fluid = GERGFluid({"Methane": 1.0})
        with pytest.raises(EOSOutOfRange):
            fluid.props(-1.0, 300.0)  # negative pressure — physically meaningless

    def test_unknown_component_raises(self):
        with pytest.raises(ValueError, match="Unknown component"):
            GERGFluid({"Benzene": 1.0})

    def test_negative_fraction_raises(self):
        with pytest.raises(ValueError):
            GERGFluid({"Methane": -0.5, "Ethane": 1.5})

    def test_all_allowed_components_listed(self):
        assert "Methane" in ALLOWED_COMPONENTS
        assert "Nitrogen" in ALLOWED_COMPONENTS
        assert "CarbonDioxide" in ALLOWED_COMPONENTS
        assert len(ALLOWED_COMPONENTS) >= 16

"""Tests for the inline device model — :class:`Device` and helpers.

Covers six categories:

1. PSV-style choked throat + Borda-Carnot transition correctness on
   pure methane.
2. :meth:`Device.from_stagnation` short-circuit identity vs an explicit
   :meth:`Device.solve` call with ``u_up=0``.
3. Borda-Carnot ``η → 1`` limit for sharp expansion (subcritical
   throat).
4. Borda-Carnot ``η → 0`` limit for marginal expansion (subcritical
   throat — choked throat exhibits density adjustments that prevent
   the η→0 limit holding cleanly per the commit-2 finding).
5. Contraction guard — ``A_down ≤ A_vc`` raises :class:`ValueError`.
6. ``_stagnation_state`` short-circuit at ``u_up = 0`` returns the
   input ``FluidState`` by object identity (no Newton iteration).
"""
from __future__ import annotations

import pytest

from gas_pipe import Device, GERGFluid
from gas_pipe.device import _stagnation_state


@pytest.fixture(scope="module")
def methane() -> GERGFluid:
    return GERGFluid({"Methane": 1.0})


class TestThroatAndTransition:
    """Device.solve on a PSV-style geometry — pure methane, choked throat."""

    def test_psv_choked_correctness(self, methane: GERGFluid) -> None:
        state_up = methane.props(50e5, 300.0)
        device = Device(A_geom=1e-4, Cd=0.7, name="PSV-1")
        r = device.solve(
            methane, state_up, u_up=0.0, P_back=15e5, A_down=1e-3,
        )
        assert r.throat.choked is True
        assert r.throat.M == pytest.approx(1.0, abs=0.01)
        # Sharp expansion ratio A_vc/A_down = 7e-5/1e-3 = 0.07 → η in (0.7, 1.0)
        assert 0.7 < r.eta_dissipation < 1.0
        # dh_static = (u_up² - u_inlet²) / 2. For u_up=0 the fluid trades
        # static enthalpy for kinetic energy → dh_static < 0.
        assert r.dh_static < 0.0
        # Entropy generation is non-negative (small numerical floor allowed).
        assert r.ds > -1e-3


class TestFromStagnationShortCircuit:
    """:meth:`Device.from_stagnation` with implicit u_up=0 takes the same
    bit-identical code path as an explicit :meth:`Device.solve` with
    ``u_up=0.0``."""

    def test_identity_against_explicit_solve(self, methane: GERGFluid) -> None:
        state_up = methane.props(50e5, 300.0)
        device = Device(A_geom=1e-4, Cd=0.7, name="PSV-1")
        r_explicit = device.solve(
            methane, state_up, u_up=0.0, P_back=15e5, A_down=1e-3,
        )
        r_from = Device.from_stagnation(
            1e-4, 0.7,
            fluid=methane, P_stag=50e5, T_stag=300.0,
            P_back=15e5, A_down=1e-3, name="PSV-1",
        )

        # ThroatState scalar fields
        for field in ("P", "T", "rho", "u", "M", "mdot", "A_vc", "G", "c_HEM"):
            assert getattr(r_from.throat, field) == pytest.approx(
                getattr(r_explicit.throat, field), rel=1e-12
            ), f"throat.{field} differs"
        assert r_from.throat.choked == r_explicit.throat.choked

        # TransitionResult scalar fields
        for field in ("u_inlet", "dP", "eta_dissipation"):
            assert getattr(r_from.transition, field) == pytest.approx(
                getattr(r_explicit.transition, field), rel=1e-12
            ), f"transition.{field} differs"

        # state_inlet — FluidState scalar fields
        for field in ("P", "T", "rho", "h", "s"):
            assert getattr(r_from.transition.state_inlet, field) == pytest.approx(
                getattr(r_explicit.transition.state_inlet, field), rel=1e-12
            ), f"state_inlet.{field} differs"

        # Top-level DeviceResult diagnostics
        for field in ("eta_dissipation", "dh_static", "ds"):
            assert getattr(r_from, field) == pytest.approx(
                getattr(r_explicit, field), rel=1e-12
            ), f"DeviceResult.{field} differs"


class TestEtaSharpExpansion:
    """Subcritical throat with A_vc / A_down ≪ 1 → ``η → 1``."""

    def test_eta_above_0p99(self, methane: GERGFluid) -> None:
        state_up = methane.props(50e5, 300.0)
        device = Device(A_geom=1e-4, Cd=0.7, name="PSV-sharp")
        # A_vc = 7e-5; A_down = 1000·A_vc = 7e-2 m² (~30 cm dia)
        A_down = 1000.0 * device.A_vc
        # P_back/P_stag = 0.85 keeps the throat sub-critical
        # (P_choke for methane sits near 0.55·P_stag).
        r = device.solve(
            methane, state_up, u_up=0.0, P_back=42.5e5, A_down=A_down,
        )
        assert r.throat.choked is False, "Throat should be subcritical for η→1 test"
        assert r.eta_dissipation > 0.99


class TestEtaMarginalExpansion:
    """Subcritical throat with A_vc / A_down ≈ 1 → ``η → 0``."""

    def test_eta_below_0p1(self, methane: GERGFluid) -> None:
        state_up = methane.props(50e5, 300.0)
        device = Device(A_geom=1e-4, Cd=0.7, name="PSV-marginal")
        # A_down = 1.05·A_vc → A_vc/A_down ≈ 0.95.
        # Use P_back = 0.95·P_stag (very mild expansion) so M_throat is low
        # and the throat→inlet density change stays small — at higher Mach,
        # compressibility-driven density rise pushes η above 0.1 even at
        # A_vc/A_down ≈ 0.95 (see commit-2 finding).
        A_down = 1.05 * device.A_vc
        r = device.solve(
            methane, state_up, u_up=0.0, P_back=47.5e5, A_down=A_down,
        )
        assert r.throat.choked is False, "Throat should be subcritical for η→0 test"
        assert r.eta_dissipation < 0.1


class TestContractionGuard:
    """``A_down ≤ A_vc`` is a contraction — not handled by Borda-Carnot."""

    def test_rejects_contraction(self, methane: GERGFluid) -> None:
        state_up = methane.props(50e5, 300.0)
        device = Device(A_geom=1e-4, Cd=0.7)
        with pytest.raises(ValueError, match="expansion only"):
            device.solve(
                methane, state_up, u_up=0.0, P_back=15e5,
                A_down=0.5 * device.A_vc,
            )


class TestStagnationShortCircuit:
    """``_stagnation_state`` returns its input ``FluidState`` by object
    identity when ``u_up = 0`` — no Newton iteration."""

    def test_identity_at_zero_velocity(self, methane: GERGFluid) -> None:
        state = methane.props(50e5, 300.0)
        state_stag = _stagnation_state(methane, state, 0.0)
        assert state_stag is state

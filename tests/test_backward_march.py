"""Tests for the backward-march primitive (Commit 1 scaffolding).

Covers the ``state_from_P_hstag`` helper and ``march_ivp_backward``
function added in solver.py. Chain-level integration (Commit 2) lives
in tests/test_chain.py.

Test design choices:

- Near-ideal-gas working fluids (pure N₂ at ≈5 bara) so analytic Fanno
  cross-checks are tight. Mirrors tests/test_fanno_analytic.py.
- Backward-vs-forward roundtrip: run a forward IVP march, take its
  ``h_stag`` at the inlet (adiabatic conservation makes this also
  ``h_stag`` everywhere along the pipe), backward-march from the
  forward's outlet pressure, verify the backward inlet recovers the
  forward inlet within engineering tolerance.
- All tests use ``eos_mode='direct'`` implicitly via the conftest
  default and ``GERGFluid`` directly; no tabulated wrapper.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gas_pipe import GERGFluid, Pipe, PipeSection, march_ivp
from gas_pipe.errors import (
    BackwardMarchDiabaticNotSupported,
    SegmentConvergenceError,
)
from gas_pipe.results import PipeResult
from gas_pipe.solver import march_ivp_backward, state_from_P_hstag


# Function-scoped fluid fixtures: GERGFluid caches per (P, T) bin (100 Pa,
# 1 mK), and the binning can let an earlier test's state.P leak into a
# later test's lookup of the same bin. Per-test fluids isolate each case.


@pytest.fixture
def n2_fluid() -> GERGFluid:
    return GERGFluid({"Nitrogen": 1.0})


@pytest.fixture
def methane_fluid() -> GERGFluid:
    return GERGFluid({"Methane": 1.0})


@pytest.fixture(scope="module")
def adiabatic_pipe() -> Pipe:
    """100 m × 50 mm adiabatic — high-friction case to exercise the math."""
    return Pipe(sections=[PipeSection(
        length=100.0, inner_diameter=0.05, roughness=4.5e-5,
    )])


@pytest.fixture(scope="module")
def diabatic_pipe() -> Pipe:
    """10 m × 50 mm with finite overall_U — should trip the adiabatic guard."""
    return Pipe.horizontal_uniform(
        length=10.0, inner_diameter=0.05, roughness=4.5e-5,
        overall_U=5.0, ambient_temperature=288.15,
    )


# ---------------------------------------------------------------------------
# state_from_P_hstag
# ---------------------------------------------------------------------------


class TestStateFromPHstag:
    """Verify the per-station h_stag inversion that backward march relies on."""

    def test_identity_roundtrip_methane(
        self, methane_fluid: GERGFluid,
    ) -> None:
        """Pin a state, compute its ``h_stag``, recover ``T`` to high precision."""
        state = methane_fluid.props(20e5, 300.0)
        mdot = 0.6
        A = math.pi * 0.2 ** 2
        u = mdot / (state.rho * A)
        h_stag = state.h + 0.5 * u * u

        recovered = state_from_P_hstag(
            methane_fluid, state.P, h_stag, mdot, A, T_guess=350.0,
        )
        assert recovered.T == pytest.approx(state.T, abs=1e-3)
        assert recovered.P == pytest.approx(state.P, rel=1e-12)

    def test_identity_roundtrip_n2_subsonic(
        self, n2_fluid: GERGFluid,
    ) -> None:
        """Roundtrip at modest Mach to exercise the u²/2 term."""
        state = n2_fluid.props(5e5, 300.0)
        # M ≈ 0.3 ⇒ u ≈ 105 m/s for N₂ at 300 K
        u_target = 0.3 * state.a
        A = math.pi * (0.0254 / 2) ** 2
        mdot = state.rho * A * u_target
        h_stag = state.h + 0.5 * u_target ** 2

        recovered = state_from_P_hstag(
            n2_fluid, state.P, h_stag, mdot, A, T_guess=305.0,
        )
        assert recovered.T == pytest.approx(state.T, abs=1e-2)

    def test_cold_initial_guess_still_converges(
        self, methane_fluid: GERGFluid,
    ) -> None:
        """T_guess far from solution; damped Newton must still converge.

        T_guess = 200 K is well below the target (320 K) but still inside
        methane's gas-phase regime at 20 bara — sub-critical liquid only
        starts below ~190 K. Tests damped Newton's reach from a cold
        starting point without dipping into GERG's two-phase region.
        """
        state = methane_fluid.props(20e5, 320.0)
        mdot = 0.6
        A = math.pi * 0.2 ** 2
        u = mdot / (state.rho * A)
        h_stag = state.h + 0.5 * u * u

        recovered = state_from_P_hstag(
            methane_fluid, state.P, h_stag, mdot, A, T_guess=200.0,
        )
        assert recovered.T == pytest.approx(state.T, abs=1e-2)

    def test_failure_raises_segment_convergence_error(
        self, methane_fluid: GERGFluid,
    ) -> None:
        """Infeasible h_stag (unphysically low for the given P) should raise.

        At P = 20 bara, the static enthalpy is ~870 kJ/kg over most of GERG's
        T range; passing h_stag = 0 forces the Newton iteration toward
        sub-critical liquid (which GERG can't evaluate). State recovery
        raises :class:`SegmentConvergenceError` either because Newton
        ran out of iterations (residual never crossed zero) or because
        the damped step excursed into EOS-invalid territory and the
        EOS error was re-raised as SegmentConvergenceError per the
        primitive's contract.
        """
        with pytest.raises(SegmentConvergenceError):
            state_from_P_hstag(
                methane_fluid, P=20e5, h_stag=0.0, mdot=0.6,
                A=math.pi * 0.2 ** 2, T_guess=300.0,
            )


# ---------------------------------------------------------------------------
# march_ivp_backward — forward roundtrip
# ---------------------------------------------------------------------------


class TestForwardRoundtrip:
    """Backward-march of a forward profile must recover its inlet."""

    def test_low_mach_n2(
        self, n2_fluid: GERGFluid, adiabatic_pipe: Pipe,
    ) -> None:
        """N₂ at M_in ≈ 0.1: smooth case, expect ≤ 0.5% inlet P agreement."""
        P_in, T_in = 5e5, 300.0
        state0 = n2_fluid.props(P_in, T_in)
        A = adiabatic_pipe.area
        u_in_target = 0.1 * state0.a
        mdot = state0.rho * A * u_in_target

        r_fwd = march_ivp(
            adiabatic_pipe, n2_fluid, P_in, T_in, mdot,
            n_segments=200, adaptive=False,
        )
        h_stag = r_fwd.h[0] + 0.5 * r_fwd.u[0] ** 2

        r_bwd = march_ivp_backward(
            adiabatic_pipe, n2_fluid,
            P_outlet=float(r_fwd.P[-1]), h_stag=h_stag, mdot=mdot,
            n_segments=200,
        )

        # Backward result is a normal PipeResult, oriented inlet→outlet.
        assert isinstance(r_bwd, PipeResult)
        assert r_bwd.march_direction == "backward"
        # Outlet pressure pinned to BC up to GERGFluid's 100-Pa cache
        # bin width; the cache rounds P-keys to 100 Pa so the stored
        # FluidState.P may differ from the input by up to ~50 Pa.
        assert r_bwd.P[-1] == pytest.approx(r_fwd.P[-1], abs=100.0)
        # Inlet pressure must recover within engineering tolerance.
        rel_dP = abs(r_bwd.P[0] - r_fwd.P[0]) / r_fwd.P[0]
        assert rel_dP < 5e-3, (
            f"Backward roundtrip recovered P_in = {r_bwd.P[0] / 1e5:.4f} bara "
            f"vs forward P_in = {r_fwd.P[0] / 1e5:.4f} bara "
            f"(rel diff = {rel_dP * 100:.3f}%, expected < 0.5%)."
        )

    def test_moderate_mach_methane(
        self, methane_fluid: GERGFluid, adiabatic_pipe: Pipe,
    ) -> None:
        """Methane at M_in ≈ 0.08, real-gas: agreement ≤ 1%."""
        P_in, T_in = 40e5, 300.0
        mdot = 2.0
        r_fwd = march_ivp(
            adiabatic_pipe, methane_fluid, P_in, T_in, mdot,
            n_segments=200, adaptive=False,
        )
        h_stag = r_fwd.h[0] + 0.5 * r_fwd.u[0] ** 2

        r_bwd = march_ivp_backward(
            adiabatic_pipe, methane_fluid,
            P_outlet=float(r_fwd.P[-1]), h_stag=h_stag, mdot=mdot,
            n_segments=200,
        )
        rel_dP = abs(r_bwd.P[0] - r_fwd.P[0]) / r_fwd.P[0]
        assert rel_dP < 1e-2

    def test_temperature_recovers_h_stag_invariant(
        self, n2_fluid: GERGFluid, adiabatic_pipe: Pipe,
    ) -> None:
        """At every backward station, ``h + u²/2`` should equal h_stag."""
        P_in, T_in = 5e5, 300.0
        state0 = n2_fluid.props(P_in, T_in)
        mdot = state0.rho * adiabatic_pipe.area * 0.1 * state0.a
        r_fwd = march_ivp(
            adiabatic_pipe, n2_fluid, P_in, T_in, mdot,
            n_segments=200, adaptive=False,
        )
        h_stag = r_fwd.h[0] + 0.5 * r_fwd.u[0] ** 2

        r_bwd = march_ivp_backward(
            adiabatic_pipe, n2_fluid,
            P_outlet=float(r_fwd.P[-1]), h_stag=h_stag, mdot=mdot,
            n_segments=200,
        )
        h_stag_stations = r_bwd.h + 0.5 * r_bwd.u ** 2
        deviation = np.abs(h_stag_stations - h_stag) / abs(h_stag)
        # Tolerance: state_from_P_hstag's default tol=1e-5, so per-station
        # residual is bounded at that level.
        assert float(np.max(deviation)) < 1e-4


# ---------------------------------------------------------------------------
# Result shape / orientation
# ---------------------------------------------------------------------------


class TestResultShape:
    """march_ivp_backward must produce a PipeResult oriented inlet→outlet."""

    def test_array_orientation_and_endpoints(
        self, n2_fluid: GERGFluid, adiabatic_pipe: Pipe,
    ) -> None:
        state0 = n2_fluid.props(5e5, 300.0)
        mdot = state0.rho * adiabatic_pipe.area * 0.1 * state0.a
        h_stag = state0.h + 0.5 * (0.1 * state0.a) ** 2

        r = march_ivp_backward(
            adiabatic_pipe, n2_fluid,
            P_outlet=4.5e5, h_stag=h_stag, mdot=mdot,
            n_segments=50,
        )
        # x ascending from 0 to L
        assert r.x[0] == pytest.approx(0.0, abs=1e-9)
        assert r.x[-1] == pytest.approx(adiabatic_pipe.length, rel=1e-9)
        assert np.all(np.diff(r.x) > 0)
        # Outlet pressure = BC, up to GERGFluid's 100 Pa cache binning.
        assert r.P[-1] == pytest.approx(4.5e5, abs=100.0)
        # Inlet pressure > outlet (friction-driven pressure rise going backward).
        assert r.P[0] > r.P[-1]
        # mdot stamped on the result.
        assert r.mdot == pytest.approx(mdot, rel=1e-12)
        # March direction flag.
        assert r.march_direction == "backward"
        # Result was NOT marked choked (sub-choke downstream).
        assert r.choked is False


# ---------------------------------------------------------------------------
# Diabatic guard
# ---------------------------------------------------------------------------


class TestDiabaticGuard:
    """Any section with ``overall_U > 0`` must raise the explicit exception."""

    def test_raises_on_diabatic_section(
        self, n2_fluid: GERGFluid, diabatic_pipe: Pipe,
    ) -> None:
        with pytest.raises(BackwardMarchDiabaticNotSupported) as excinfo:
            march_ivp_backward(
                diabatic_pipe, n2_fluid,
                P_outlet=4.5e5, h_stag=3.1e5, mdot=0.1,
                n_segments=20,
            )
        # Error message names overall_U so the user can locate the offending section.
        assert "overall_U" in str(excinfo.value)

    def test_accepts_zero_U_explicit(
        self, n2_fluid: GERGFluid,
    ) -> None:
        """A section with overall_U == 0.0 explicitly set must NOT raise."""
        pipe = Pipe.horizontal_uniform(
            length=10.0, inner_diameter=0.05, roughness=4.5e-5,
            overall_U=0.0, ambient_temperature=288.15,
        )
        state0 = n2_fluid.props(5e5, 300.0)
        mdot = state0.rho * pipe.area * 0.05 * state0.a
        h_stag = state0.h + 0.5 * (0.05 * state0.a) ** 2

        r = march_ivp_backward(
            pipe, n2_fluid,
            P_outlet=4.5e5, h_stag=h_stag, mdot=mdot,
            n_segments=50,
        )
        assert r.march_direction == "backward"


# ---------------------------------------------------------------------------
# Analytic Fanno cross-check
# ---------------------------------------------------------------------------


def _fanno_4fL_star(M: float, gamma: float) -> float:
    """Fanno maximum-pipe-length parameter (mirrors test_fanno_analytic)."""
    M2 = M * M
    term1 = (1.0 - M2) / (gamma * M2)
    term2 = (gamma + 1.0) / (2.0 * gamma) * math.log(
        (gamma + 1.0) * M2 / (2.0 + (gamma - 1.0) * M2)
    )
    return term1 + term2


def _fanno_M_from_4fLD(target: float, gamma: float) -> float:
    """Subsonic-branch bisection for M(target)."""
    M_lo, M_hi = 1e-3, 1.0 - 1e-9
    for _ in range(60):
        M_mid = 0.5 * (M_lo + M_hi)
        if _fanno_4fL_star(M_mid, gamma) > target:
            M_lo = M_mid
        else:
            M_hi = M_mid
    return 0.5 * (M_lo + M_hi)


class TestFannoAnalyticBackward:
    """Backward march's Mach profile should match Fanno-table analytic
    just as the forward march does.

    Method: solve forward at a known case, get outlet conditions, then
    backward-march and compare backward's M(x) against the analytic
    Fanno curve referenced from the SAME inlet Mach (so any small
    Mach drift accumulates symmetrically).
    """

    def test_n2_subsonic_M_profile(self, n2_fluid: GERGFluid) -> None:
        """Note: M_in=0.1 (not 0.3 like forward Fanno test) so the march
        stays well sub-sonic. v1 backward march doesn't have adaptive
        refinement / multi-step Newton, so near-choke conditions are out
        of scope for this primitive."""
        pipe = Pipe.horizontal_uniform(
            length=10.0, inner_diameter=0.0254, roughness=1e-6,
            overall_U=0.0, ambient_temperature=300.0,
        )
        P_in, T_in = 5e5, 300.0
        state0 = n2_fluid.props(P_in, T_in)
        D = pipe.inner_diameter
        A = pipe.area
        u_in_target = 0.1 * state0.a
        mdot = state0.rho * A * u_in_target
        gamma = state0.cp / state0.cv

        # Forward reference at the same Mach as test_fanno_analytic.
        r_fwd = march_ivp(
            pipe, n2_fluid, P_in, T_in, mdot,
            n_segments=200, adaptive=False,
        )
        h_stag = r_fwd.h[0] + 0.5 * r_fwd.u[0] ** 2

        r_bwd = march_ivp_backward(
            pipe, n2_fluid,
            P_outlet=float(r_fwd.P[-1]), h_stag=h_stag, mdot=mdot,
            n_segments=200,
        )

        # Use the forward run's mean Darcy friction so the analytic
        # parameterisation is consistent with the discrete march.
        f_avg = float(np.mean(r_fwd.f))
        M_in_bwd = float(r_bwd.M[0])
        param_inlet = _fanno_4fL_star(M_in_bwd, gamma)

        M_analytic = np.array([
            1.0 if (param_inlet - f_avg * x / D) < 0
            else _fanno_M_from_4fLD(param_inlet - f_avg * x / D, gamma)
            for x in r_bwd.x
        ])

        rel_err = np.abs(r_bwd.M - M_analytic) / (M_analytic + 1e-6)
        max_err = float(np.max(rel_err))
        # Same tolerance band as the forward analytic test.
        assert max_err < 0.02, (
            f"Backward Mach profile deviates from Fanno analytic by "
            f"{max_err * 100:.2f}% (forward test passes at < 2%)."
        )

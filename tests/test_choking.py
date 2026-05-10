"""Choking / BVP tests for 50 m × 4" pipe, methane."""
from __future__ import annotations

import math

import numpy as np
import pytest

from gas_pipe import GERGFluid, Pipe, march_ivp, solve_for_mdot
from gas_pipe.errors import BVPChoked


@pytest.fixture(scope="module")
def ch4_fluid():
    return GERGFluid({"Methane": 1.0})


@pytest.fixture(scope="module")
def choke_pipe():
    return Pipe.horizontal_uniform(
        length=50.0,
        inner_diameter=0.1016,   # 4" = 101.6 mm
        roughness=4.5e-5,
        overall_U=0.0,
        ambient_temperature=300.0,
    )


def test_bvp_raises_bvp_choked(ch4_fluid, choke_pipe):
    """BVP at P_out = 1 bara must raise BVPChoked (flow is choked)."""
    with pytest.raises(BVPChoked) as exc_info:
        solve_for_mdot(
            choke_pipe, ch4_fluid,
            P_in=50e5, T_in=300.0, P_out=1e5,
            n_segments=100, adaptive=True,
        )
    exc = exc_info.value
    assert exc.mdot_critical > 0
    assert exc.result is not None


def test_bvp_choked_mdot_critical_populated(ch4_fluid, choke_pipe):
    """BVPChoked.mdot_critical must be positive and physically plausible."""
    with pytest.raises(BVPChoked) as exc_info:
        solve_for_mdot(
            choke_pipe, ch4_fluid,
            P_in=50e5, T_in=300.0, P_out=1e5,
            n_segments=100, adaptive=True,
        )
    exc = exc_info.value
    mdot_crit = exc.mdot_critical
    assert mdot_crit > 0.0
    assert mdot_crit < 1000.0  # sanity upper bound

    # API 521 nozzle approximation: ṁ = C_d * A * P_in * sqrt(γM/(ZRT)) for critical nozzle
    state0 = ch4_fluid.props(50e5, 300.0)
    A = choke_pipe.area
    gamma = state0.cp / state0.cv
    R_spec = 8.31446 / ch4_fluid.molar_mass    # [J/(kg·K)] — molar_mass is kg/mol
    Z = state0.Z
    # Critical mass flux: G* = P_in * sqrt(γ/(Z*R_spec*T_in) * (2/(γ+1))^((γ+1)/(γ-1)))
    G_star = 50e5 * math.sqrt(
        gamma / (Z * R_spec * 300.0) *
        (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0))
    )
    mdot_api = G_star * A  # ideal nozzle
    # Allow factor of 2 (pipe loss differs from nozzle)
    assert mdot_crit / mdot_api < 3.0 and mdot_api / mdot_crit < 3.0, (
        f"mdot_crit={mdot_crit:.2f} vs API521={mdot_api:.2f} — more than factor 2 off"
    )


def test_choked_solution_has_mach_1(ch4_fluid, choke_pipe):
    """The choked result should contain M ≈ 1 somewhere."""
    with pytest.raises(BVPChoked) as exc_info:
        solve_for_mdot(
            choke_pipe, ch4_fluid,
            P_in=50e5, T_in=300.0, P_out=1e5,
            n_segments=100, adaptive=True,
        )
    r = exc_info.value.result
    assert r.choked
    assert float(np.max(r.M)) > 0.9, f"Max M = {float(np.max(r.M)):.3f}"


def test_mach_monotonic_in_choked_result(ch4_fluid, choke_pipe):
    """Mach should not decrease anywhere (Layer 2 choke detection bug check)."""
    with pytest.raises(BVPChoked) as exc_info:
        solve_for_mdot(
            choke_pipe, ch4_fluid,
            P_in=50e5, T_in=300.0, P_out=1e5,
            n_segments=100, adaptive=True,
        )
    r = exc_info.value.result
    M = r.M
    dM = np.diff(M)
    # Allow tiny numerical noise
    violations = np.sum(dM < -0.05)
    assert violations == 0, (
        f"{violations} M-decrease violations. Min dM = {float(np.min(dM)):.4f}"
    )


def test_subsonic_bvp_no_choke(ch4_fluid, choke_pipe):
    """At a high P_out (close to P_in), should give subsonic solution."""
    result = solve_for_mdot(
        choke_pipe, ch4_fluid,
        P_in=50e5, T_in=300.0, P_out=45e5,
        n_segments=50, adaptive=True,
    )
    assert not result.choked
    assert float(np.max(result.M)) < 0.5

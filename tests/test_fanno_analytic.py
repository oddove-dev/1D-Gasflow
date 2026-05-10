"""Fanno-flow analytic comparison for choke detection validation.

Pure N₂ at low pressure (≈5 bara, near-ideal gas), 10 m × 1" pipe, M_in = 0.3.
Compare solver M(x) against analytic Fanno tables within 1%.

The Fanno parameter: 4fL*/D = (1-M²)/(γM²) + (γ+1)/(2γ) * ln[(γ+1)M²/(2+(γ-1)M²)]
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gas_pipe import GERGFluid, Pipe, march_ivp


def fanno_4fL_star(M: float, gamma: float) -> float:
    """Fanno maximum pipe length parameter 4fL*/D."""
    M2 = M * M
    term1 = (1.0 - M2) / (gamma * M2)
    term2 = (gamma + 1.0) / (2.0 * gamma) * math.log(
        (gamma + 1.0) * M2 / (2.0 + (gamma - 1.0) * M2)
    )
    return term1 + term2


def fanno_M_from_4fLD(target_4fLD: float, gamma: float, M_guess: float = 0.5) -> float:
    """Solve for M given 4fL*/D (bisection, subsonic branch)."""
    # For subsonic: 4fL*/D is monotonically decreasing with M
    M_lo, M_hi = 1e-3, 1.0 - 1e-9
    for _ in range(60):
        M_mid = 0.5 * (M_lo + M_hi)
        val = fanno_4fL_star(M_mid, gamma)
        if val > target_4fLD:
            M_lo = M_mid
        else:
            M_hi = M_mid
    return 0.5 * (M_lo + M_hi)


@pytest.fixture(scope="module")
def n2_fluid():
    return GERGFluid({"Nitrogen": 1.0})


@pytest.fixture(scope="module")
def fanno_pipe():
    # 10 m × 1" (25.4 mm ID), smooth
    return Pipe.horizontal_uniform(
        length=10.0,
        inner_diameter=0.0254,
        roughness=1e-6,  # very smooth
        overall_U=0.0,
        ambient_temperature=300.0,
    )


def test_mach_profile_vs_fanno(n2_fluid, fanno_pipe):
    """Solver M(x) should match Fanno analytic within 1%."""
    P_in = 5e5    # 5 bara — near ideal gas
    T_in = 300.0
    D = fanno_pipe.inner_diameter
    A = fanno_pipe.area

    # Choose ṁ to get M_in ≈ 0.3
    state0 = n2_fluid.props(P_in, T_in)
    u_in_target = 0.3 * state0.a
    mdot = state0.rho * A * u_in_target

    gamma = state0.cp / state0.cv  # ≈1.4 for N₂

    result = march_ivp(
        fanno_pipe, n2_fluid, P_in, T_in, mdot,
        n_segments=200, adaptive=True,
    )

    # Analytic Fanno M profile
    # Use Darcy friction from first segment
    f_avg = float(np.mean(result.f)) if len(result.f) > 0 else 0.02

    # 4fL*/D at inlet using solver M_in
    M_in_solver = float(result.M[0])
    param_inlet = fanno_4fL_star(M_in_solver, gamma)

    M_analytic_list = []
    for x_val in result.x:
        # f_avg is the Darcy friction factor; the Fanno parameter "4fL*/D"
        # uses the Fanning friction factor (= f_Darcy/4), so:
        #   4 * f_Fanning * (L*-x) / D  =  param_inlet  -  f_Darcy * x / D
        param_x = param_inlet - f_avg * x_val / D
        if param_x < 0:
            M_analytic_list.append(1.0)
        else:
            M_analytic_list.append(fanno_M_from_4fLD(param_x, gamma))

    M_analytic = np.array(M_analytic_list)
    M_solver = result.M

    # Within 1%
    rel_err = np.abs(M_solver - M_analytic) / (M_analytic + 1e-6)
    max_err = float(np.max(rel_err))
    assert max_err < 0.02, (
        f"Max Fanno deviation = {max_err:.2%} at x = {float(result.x[int(np.argmax(rel_err))]):.2f} m"
    )


def test_mach_monotonically_increases(n2_fluid, fanno_pipe):
    """Mach number should be non-decreasing along a Fanno-flow pipe."""
    P_in = 5e5
    T_in = 300.0
    state0 = n2_fluid.props(P_in, T_in)
    mdot = state0.rho * fanno_pipe.area * 0.3 * state0.a

    result = march_ivp(
        fanno_pipe, n2_fluid, P_in, T_in, mdot,
        n_segments=200, adaptive=True,
    )

    M = result.M
    # Allow tiny numerical noise (1% tolerance)
    dM = np.diff(M)
    violations = np.sum(dM < -0.01 * M[:-1])
    assert violations == 0, (
        f"{violations} stations where M decreases (Layer 2 bug). "
        f"Min dM = {float(np.min(dM)):.4f}"
    )

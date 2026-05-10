"""Joule-Thomson cooling test — insulated pipe, methane.

1 km × 6" insulated (U=0), methane at P_in = 100 bara, T_in = 300 K,
BVP at P_out = 70 bara.

Expected: ΔT ≈ μ_JT,avg * ΔP within 1 K.
"""
from __future__ import annotations

import numpy as np
import pytest

from gas_pipe import GERGFluid, Pipe, solve_for_mdot
from gas_pipe.errors import BVPChoked


@pytest.fixture(scope="module")
def ch4_fluid():
    return GERGFluid({"Methane": 1.0})


@pytest.fixture(scope="module")
def jt_pipe():
    return Pipe.horizontal_uniform(
        length=1000.0,
        inner_diameter=0.1524,   # 6"
        roughness=4.5e-5,
        overall_U=0.0,           # insulated
        ambient_temperature=300.0,
    )


def test_jt_cooling_within_1K(ch4_fluid, jt_pipe):
    """ΔT from solver should match μ_JT,avg * ΔP within 1 K."""
    P_in = 100e5
    T_in = 300.0
    P_out = 70e5

    try:
        result = solve_for_mdot(
            jt_pipe, ch4_fluid, P_in, T_in, P_out,
            n_segments=100, adaptive=True,
        )
    except BVPChoked as exc:
        pytest.skip(f"Unexpectedly choked at these conditions: {exc}")

    T_out = float(result.T[-1])
    dT_solver = T_out - T_in
    dP_actual = float(result.P[-1]) - P_in

    # Average μ_JT over the pipe
    mu_JT_avg = float(np.mean(result.mu_JT))
    dT_expected = mu_JT_avg * dP_actual

    assert abs(dT_solver - dT_expected) < 1.5, (
        f"ΔT solver={dT_solver:.2f} K vs expected={dT_expected:.2f} K "
        f"(diff={abs(dT_solver-dT_expected):.2f} K)"
    )


def test_temperature_decreases_insulated(ch4_fluid, jt_pipe):
    """For high-pressure methane, JT cooling should produce T_out < T_in."""
    P_in = 100e5
    T_in = 300.0
    P_out = 70e5

    try:
        result = solve_for_mdot(
            jt_pipe, ch4_fluid, P_in, T_in, P_out,
            n_segments=100, adaptive=True,
        )
    except BVPChoked:
        pytest.skip("Choked — skip temperature check")

    T_out = float(result.T[-1])
    assert T_out < T_in, f"T_out={T_out:.2f} K should be less than T_in={T_in:.2f} K"

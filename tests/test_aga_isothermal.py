"""AGA isothermal pipeline test — 50 km × 30" pipe."""
from __future__ import annotations

import math

import pytest

from gas_pipe import GERGFluid, Pipe, march_ivp, solve_for_mdot
from gas_pipe.errors import BVPChoked


@pytest.fixture(scope="module")
def aga_fluid():
    return GERGFluid({
        "Methane": 0.85,
        "Ethane": 0.08,
        "Propane": 0.05,
        "Nitrogen": 0.02,
    })


@pytest.fixture(scope="module")
def aga_pipe():
    # 50 km × 30" pipe, ε = 50 μm
    return Pipe.horizontal_uniform(
        length=50_000.0,
        inner_diameter=0.762,
        roughness=50e-6,
        overall_U=0.0,
        ambient_temperature=280.0,
    )


def aga_fully_turbulent_mdot(pipe: Pipe, fluid: GERGFluid, P_in: float, T_in: float, P_out: float) -> float:
    """Reference AGA fully-turbulent mass flow estimate."""
    D = pipe.inner_diameter
    L = pipe.length
    eps = pipe.roughness
    A = pipe.area

    P_avg = 0.5 * (P_in + P_out)
    state_avg = fluid.props(P_avg, T_in)
    Z_avg = state_avg.Z
    rho_avg = state_avg.rho

    f_ft = (-2.0 * math.log10(eps / (3.7 * D))) ** (-2)
    # P1² - P2² = f * L/D * G² / (2 * rho_avg * P_avg / P_avg)
    # Simplified isothermal: P1² - P2² = f * L/D * (ṁ/A)² * P_avg / rho_avg / P_avg * 2 * rho_avg ...
    # Use: dP ~ P1-P2 for ΔP << P_in, or use compressible:
    # For Darcy: ΔP = f * L/D * rho * u²/2
    # Isothermal compressible: P1² - P2² = f * L/D * 2 * P_avg * (ṁ/A)² / (Z_avg * R_spec * T_in / M)
    R_spec = 8314.46 / fluid.molar_mass
    rhs = f_ft * (L / D) * 2.0 * P_avg * (P_in**2 - P_out**2) / (Z_avg * R_spec * T_in)
    # Simpler: ṁ = A * sqrt((P1²-P2²) * rho_avg * D / (f * L * P_avg))
    mdot_est = A * math.sqrt(rho_avg * D * (P_in**2 - P_out**2) / (f_ft * L * P_avg))
    return mdot_est


def test_aga_bvp_within_3pct(aga_pipe, aga_fluid):
    """BVP solution should agree with AGA fully-turbulent within 3%."""
    P_in = 80e5    # 80 bara
    T_in = 280.0   # K
    P_out = 60e5   # 60 bara

    mdot_aga = aga_fully_turbulent_mdot(aga_pipe, aga_fluid, P_in, T_in, P_out)

    try:
        result = solve_for_mdot(
            aga_pipe, aga_fluid, P_in, T_in, P_out,
            n_segments=100, adaptive=True,
        )
        mdot_solver = result.mdot
    except BVPChoked as exc:
        mdot_solver = exc.mdot_critical

    rel_err = abs(mdot_solver - mdot_aga) / mdot_aga
    assert rel_err < 0.03, (
        f"Solver ṁ = {mdot_solver:.2f} kg/s vs AGA ṁ = {mdot_aga:.2f} kg/s, "
        f"error = {rel_err:.1%}"
    )


def test_ivp_pressure_reasonable(aga_pipe, aga_fluid):
    """IVP at a reasonable ṁ should give P_out between P_in and 0."""
    mdot = 100.0  # kg/s
    result = march_ivp(aga_pipe, aga_fluid, 80e5, 280.0, mdot, n_segments=80, adaptive=False)
    assert float(result.P[-1]) > 0
    assert float(result.P[-1]) < 80e5
    assert float(result.P[0]) == pytest.approx(80e5, rel=1e-6)

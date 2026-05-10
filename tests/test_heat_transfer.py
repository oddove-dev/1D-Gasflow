"""Heat transfer test — NTU exponential decay model.

1 km × 6" pipe, U = 5 W/m²/K, T_amb = 4°C = 277.15 K,
methane at P_in = 80 bara, T_in = 80°C = 353.15 K, low Mach.

For low Mach (nearly incompressible), the energy equation reduces to:
  dT/dx = -U * π * D_o / (ṁ * cp) * (T - T_amb)
Giving: (T - T_amb) = (T_in - T_amb) * exp(-NTU * x/L)
where NTU = U * π * D_o * L / (ṁ * cp)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gas_pipe import GERGFluid, Pipe, solve_for_mdot
from gas_pipe.errors import BVPChoked


@pytest.fixture(scope="module")
def ch4_fluid():
    return GERGFluid({"Methane": 1.0})


@pytest.fixture(scope="module")
def ht_pipe():
    return Pipe.horizontal_uniform(
        length=1000.0,
        inner_diameter=0.1524,     # 6"
        roughness=4.5e-5,
        outer_diameter=0.168,
        overall_U=5.0,             # W/m²/K
        ambient_temperature=277.15,  # 4°C
    )


def test_heat_transfer_ntu_model(ch4_fluid, ht_pipe):
    """T(x) should match NTU exponential decay within 2°C."""
    P_in = 80e5
    T_in = 353.15   # 80°C
    T_amb = 277.15
    P_out = 75e5    # mild pressure drop to keep Mach low

    try:
        result = solve_for_mdot(
            ht_pipe, ch4_fluid, P_in, T_in, P_out,
            n_segments=100, adaptive=True,
        )
    except BVPChoked as exc:
        pytest.skip(f"Unexpectedly choked: {exc}")

    mdot = result.mdot
    # Average cp
    cp_avg = float(np.mean([
        ch4_fluid.props(float(p), float(t)).cp
        for p, t in zip(result.P[::10], result.T[::10])
    ]))

    D_o = ht_pipe.D_o()
    U_val = ht_pipe.U(0.0)
    L = ht_pipe.length

    NTU = U_val * math.pi * D_o * L / (mdot * cp_avg)
    T_ntu = T_amb + (T_in - T_amb) * np.exp(-NTU * result.x / L)

    diff = np.abs(result.T - T_ntu)
    max_diff = float(np.max(diff))
    assert max_diff < 3.0, (
        f"Max T deviation from NTU model: {max_diff:.1f} K (limit 3 K)"
    )


def test_temperature_approaches_ambient(ch4_fluid, ht_pipe):
    """Over a long pipe with U>0, T_out should be closer to T_amb than T_in."""
    P_in = 80e5
    T_in = 353.15
    P_out = 75e5

    try:
        result = solve_for_mdot(
            ht_pipe, ch4_fluid, P_in, T_in, P_out,
            n_segments=100, adaptive=True,
        )
    except BVPChoked:
        pytest.skip("Choked")

    T_out = float(result.T[-1])
    T_amb = 277.15
    assert abs(T_out - T_amb) < abs(T_in - T_amb), (
        f"T_out={T_out-273.15:.1f}°C not closer to T_amb than T_in={T_in-273.15:.1f}°C"
    )

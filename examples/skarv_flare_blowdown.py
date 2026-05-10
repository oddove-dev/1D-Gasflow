"""Skarv flare-header blowdown analysis.

Demonstrates IVP, BVP, and plateau sweep — same cases the GUI exercises.
"""
from __future__ import annotations

import numpy as np

from gas_pipe import GERGFluid, Pipe, march_ivp, solve_for_mdot
from gas_pipe.diagnostics import plot_profile, plot_plateau_sweep
from gas_pipe.errors import BVPChoked

composition = {
    "Methane": 0.78,
    "Ethane": 0.10,
    "Propane": 0.05,
    "n-Butane": 0.02,
    "IsoButane": 0.01,
    "Nitrogen": 0.02,
    "CarbonDioxide": 0.02,
}
fluid = GERGFluid(composition)
pipe = Pipe.horizontal_uniform(
    length=80.0,
    inner_diameter=0.762,
    roughness=4.5e-5,
    outer_diameter=0.813,
    overall_U=2.0,
    ambient_temperature=283.15,
)

# Case 1: IVP at fixed ṁ
print("=" * 80)
print("CASE 1: IVP — imposed ṁ = 120 kg/s")
print("=" * 80)
r1 = march_ivp(
    pipe, fluid,
    P_in=50e5, T_in=373.15, mdot=120.0,
    n_segments=200, adaptive=True,
)
print(r1.summary())
r1.print_profile()
plot_profile(r1, save_path="skarv_ivp_120kgs.png")
print("  → saved skarv_ivp_120kgs.png")

# Case 2: BVP — find ṁ_critical at low outlet pressure
print("\n" + "=" * 80)
print("CASE 2: BVP — P_out = 2 bara (expecting choke)")
print("=" * 80)
try:
    r2 = solve_for_mdot(
        pipe, fluid,
        P_in=50e5, T_in=373.15, P_out=2e5,
        n_segments=200, adaptive=True,
    )
    print(r2.summary())
    plot_profile(r2, save_path="skarv_bvp.png")
    print("  → saved skarv_bvp.png")
except BVPChoked as exc:
    print(f"BVP choked: ṁ_critical = {exc.mdot_critical:.2f} kg/s")
    print(exc.result.summary())
    plot_profile(exc.result, save_path="skarv_bvp_choked.png")
    print("  → saved skarv_bvp_choked.png")

# Case 3: Plateau sweep
print("\n" + "=" * 80)
print("CASE 3: Plateau sweep — confirm choke flat region")
print("=" * 80)
P_out_sweep = np.array([45e5, 35e5, 25e5, 15e5, 10e5, 5e5, 3e5, 2e5])
mdot_sweep = []
choked_flags = []

for P_out in P_out_sweep:
    try:
        r = solve_for_mdot(
            pipe, fluid,
            P_in=50e5, T_in=373.15, P_out=float(P_out),
            n_segments=100, adaptive=True,
        )
        mdot_sweep.append(r.mdot)
        choked_flags.append(r.choked)
    except BVPChoked as exc:
        mdot_sweep.append(exc.mdot_critical)
        choked_flags.append(True)
    except Exception as exc:
        print(f"  P_out = {P_out/1e5:.1f} bara → error: {exc}")
        mdot_sweep.append(float("nan"))
        choked_flags.append(False)

print(f"\n{'P_out [bara]':>14} {'ṁ [kg/s]':>12} {'Choked':>8}")
for P_out, mdot, ch in zip(P_out_sweep, mdot_sweep, choked_flags):
    print(f"{P_out/1e5:14.2f} {mdot:12.2f} {str(ch):>8}")

plot_plateau_sweep(
    P_out_sweep, np.array(mdot_sweep), choked_flags,
    save_path="skarv_plateau.png",
)
print("\n  → saved skarv_plateau.png")
print("\nDone.")

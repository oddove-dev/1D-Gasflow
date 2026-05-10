"""PipeResult dataclass: stores and formats solver output."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


@dataclass
class PipeResult:
    """Complete result of a 1D gas pipe flow simulation.

    Parameters
    ----------
    x : np.ndarray
        Axial positions [m], length n.
    P : np.ndarray
        Pressure [Pa], length n.
    T : np.ndarray
        Temperature [K], length n.
    rho : np.ndarray
        Density [kg/m³], length n.
    u : np.ndarray
        Velocity [m/s], length n.
    a : np.ndarray
        Speed of sound [m/s], length n.
    M : np.ndarray
        Mach number [-], length n.
    Re : np.ndarray
        Reynolds number [-], length n (Re[0] from first segment).
    Z : np.ndarray
        Compressibility factor [-], length n.
    h : np.ndarray
        Specific enthalpy [J/kg], length n.
    mu_JT : np.ndarray
        Joule-Thomson coefficient [K/Pa], length n.
    f : np.ndarray
        Darcy friction factor [-], length n-1.
    dP_fric : np.ndarray
        Friction pressure drop per segment [Pa], length n-1.
    dP_acc : np.ndarray
        Acceleration pressure drop per segment [Pa], length n-1.
    dP_elev : np.ndarray
        Elevation pressure drop per segment [Pa], length n-1.
    dP_fitting : np.ndarray
        Fitting pressure drop per segment [Pa], length n-1.
    q_seg : np.ndarray
        Heat per unit mass per segment [J/kg], length n-1.
    mdot : float
        Mass flow rate [kg/s].
    choked : bool
        Whether choke was detected.
    x_choke : float or None
        Choke location [m].
    iterations_per_segment : np.ndarray
        Newton iteration count per segment, length n-1.
    energy_residual : float
        Global energy balance relative residual.
    min_dx : float
        Minimum segment length used [m].
    fluid_composition : dict
        Mole-fraction composition.
    pipe_summary : dict
        Pipe geometry summary.
    boundary_conditions : dict
        Inlet/outlet boundary conditions.
    solver_options : dict
        Solver settings used.
    elapsed_seconds : float
        Wallclock time for the solve [s].
    """

    # Station arrays (length n)
    x: np.ndarray
    P: np.ndarray
    T: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    a: np.ndarray
    M: np.ndarray
    Re: np.ndarray
    Z: np.ndarray
    h: np.ndarray
    mu_JT: np.ndarray

    # Per-segment arrays (length n-1)
    f: np.ndarray
    dP_fric: np.ndarray
    dP_acc: np.ndarray
    dP_elev: np.ndarray
    dP_fitting: np.ndarray
    q_seg: np.ndarray

    # Scalars
    mdot: float
    choked: bool
    x_choke: float | None

    # Numerics
    iterations_per_segment: np.ndarray
    energy_residual: float
    min_dx: float

    # Metadata
    fluid_composition: dict
    pipe_summary: dict
    boundary_conditions: dict
    solver_options: dict
    elapsed_seconds: float

    def summary(self) -> str:
        """Return a formatted multi-line summary string (~70-80 chars wide).

        Returns
        -------
        str
        """
        W = 80
        sep = "=" * W
        dash = "-" * 36

        ps = self.pipe_summary
        bc = self.boundary_conditions
        comp = self.fluid_composition
        opts = self.solver_options

        # Pressure drop breakdown
        total_dP_fric = float(np.sum(self.dP_fric))
        total_dP_acc = float(np.sum(self.dP_acc))
        total_dP_elev = float(np.sum(self.dP_elev))
        total_dP_fit = float(np.sum(self.dP_fitting))
        total_dP = total_dP_fric + total_dP_acc + total_dP_elev + total_dP_fit

        def pct(v: float) -> str:
            return f"{100*v/total_dP:.1f}%" if abs(total_dP) > 0 else "N/A"

        # Flow at inlet / outlet
        x_in, x_out = float(self.x[0]), float(self.x[-1])
        P_in_bara = float(self.P[0]) / 1e5
        T_in_C = float(self.T[0]) - 273.15
        P_out_bara = float(self.P[-1]) / 1e5
        T_out_C = float(self.T[-1]) - 273.15
        rho_in = float(self.rho[0])
        rho_out = float(self.rho[-1])
        u_in = float(self.u[0])
        u_out = float(self.u[-1])
        M_in = float(self.M[0])
        M_out = float(self.M[-1])
        Q_in = self.mdot / rho_in
        Q_out = self.mdot / rho_out

        D_m = ps.get("inner_diameter", 0.0)
        D_mm = D_m * 1000
        D_in = D_m / 0.0254
        L = ps.get("length", 0.0)
        eps = ps.get("roughness", 0.0)
        eps_D = eps / D_m if D_m > 0 else 0.0
        D_o_mm = ps.get("outer_diameter", D_m) * 1000
        U_val = ps.get("overall_U", 0.0)
        T_amb_C = ps.get("ambient_temperature", 277.15) - 273.15
        MM_gmol = ps.get("molar_mass", 0.0) * 1000

        # Z, mu_JT ranges
        Z_min, Z_max = float(np.min(self.Z)), float(np.max(self.Z))
        muJT_min = float(np.min(self.mu_JT)) * 1e5  # K/bar
        muJT_max = float(np.max(self.mu_JT)) * 1e5  # K/bar
        dT_total = T_out_C - T_in_C

        n_segs = len(self.dP_fric)
        n_adaptive = int(opts.get("n_adaptive_refinements", 0))
        avg_iter = float(np.mean(self.iterations_per_segment)) if len(self.iterations_per_segment) > 0 else 0.0

        n_fittings = int(ps.get("n_fittings", 0))

        lines: list[str] = [sep]
        lines.append(f"{'Gas Pipe Analysis — Single-Phase 1D Steady-State':^{W}}")
        lines.append(sep)
        lines.append("")

        lines.append("PIPE")
        lines.append(f"  Length:           {L:.2f} m")
        lines.append(f"  Inner diameter:   {D_mm:.1f} mm  ({D_in:.1f}\")")
        lines.append(f"  Roughness:        {eps*1e6:.0f} μm  (ε/D = {eps_D:.2e})")
        lines.append(f"  Outer diameter:   {D_o_mm:.1f} mm")
        if U_val > 0:
            lines.append(f"  Heat transfer:    U = {U_val:.1f} W/m²/K, T_amb = {T_amb_C:.1f} °C")
        else:
            lines.append(f"  Heat transfer:    Adiabatic (U = 0)")
        lines.append("")

        lines.append("FLUID (GERG-2008)")
        for name, frac in sorted(comp.items(), key=lambda x: -x[1]):
            lines.append(f"  {name:<16} {frac*100:.1f}%")
        lines.append(f"  Molar mass:       {MM_gmol:.2f} g/mol")
        lines.append("")

        choke_flag = "  [CHOKED]" if self.choked else ""
        lines.append("BOUNDARY CONDITIONS")
        lines.append(f"  Inlet:    P = {P_in_bara:7.3f} bara,  T = {T_in_C:.1f} °C")
        lines.append(f"  Outlet:   P = {P_out_bara:7.3f} bara,  T = {T_out_C:.1f} °C{choke_flag}")
        lines.append("")

        lines.append("FLOW")
        lines.append(f"  Mass flow:           {self.mdot:.1f} kg/s")
        lines.append(f"  Volumetric (inlet):  {Q_in:.3f} m³/s     u_in  = {u_in:7.2f} m/s   M_in  = {M_in:.3f}")
        lines.append(f"  Volumetric (outlet): {Q_out:.3f} m³/s     u_out = {u_out:7.2f} m/s   M_out = {M_out:.3f}")
        lines.append("")

        if self.choked and self.x_choke is not None:
            idx_choke = int(np.argmin(np.abs(self.x - self.x_choke)))
            P_c = float(self.P[idx_choke]) / 1e5
            T_c = float(self.T[idx_choke]) - 273.15
            rho_c = float(self.rho[idx_choke])
            pct_len = self.x_choke / L * 100 if L > 0 else 0.0
            lines.append("CHOKE")
            lines.append(f"  Location:         {self.x_choke:.2f} m  ({pct_len:.1f}% of pipe length)")
            lines.append(f"  P_critical:        {P_c:.3f} bara")
            lines.append(f"  T_critical:      {T_c:.1f} °C  ({T_c+273.15:.1f} K)")
            lines.append(f"  ρ_critical:        {rho_c:.2f} kg/m³")
            lines.append(f"  ṁ_critical:      {self.mdot:.1f} kg/s")
            lines.append("")

        lines.append("PRESSURE DROP BREAKDOWN")
        lines.append(f"  Friction:        {total_dP_fric/1e5:.2f} bar ({pct(total_dP_fric)})")
        lines.append(f"  Acceleration:     {total_dP_acc/1e5:.2f} bar  ({pct(total_dP_acc)})")
        lines.append(f"  Elevation:        {total_dP_elev/1e5:.2f} bar  ({pct(total_dP_elev)})")
        lines.append(f"  Fittings ({n_fittings}):     {total_dP_fit/1e5:.2f} bar  ({pct(total_dP_fit)})")
        lines.append(f"  {dash}")
        lines.append(f"  Total:           {total_dP/1e5:.2f} bar")
        lines.append("")

        lines.append("REAL-GAS EFFECTS")
        lines.append(f"  Z range:          {Z_min:.3f} → {Z_max:.3f}")
        lines.append(f"  μ_JT range:       {muJT_min:.1f} → {muJT_max:.1f} K/bar")
        lines.append(f"  ΔT total:        {dT_total:.1f} K")
        lines.append("")

        lines.append("NUMERICAL")
        lines.append(f"  Segments:         {n_segs} ({n_adaptive} adaptive refinements near choke)")
        lines.append(f"  Avg iter/seg:     {avg_iter:.1f}")
        lines.append(f"  Energy residual:  {self.energy_residual:.2e}  (relative)")
        lines.append(f"  Min Δx:           {self.min_dx*1000:.1f} mm")
        lines.append(f"  Elapsed:          {self.elapsed_seconds:.1f} s")
        lines.append("")

        # Warnings
        warn_list = self.warnings()
        if warn_list:
            lines.append("WARNINGS")
            for w in warn_list:
                lines.append(f"  ⚠ {w}")
            lines.append("")

        lines.append(sep)
        return "\n".join(lines)

    def print_profile(self, n_stations: int = 12) -> None:
        """Print a tabular profile to stdout.

        Parameters
        ----------
        n_stations : int
            Number of evenly-spaced stations to display.
        """
        n = len(self.x)
        if n <= n_stations:
            indices = list(range(n))
        else:
            indices = [int(round(i * (n - 1) / (n_stations - 1))) for i in range(n_stations)]
            # Ensure last index is included
            if indices[-1] != n - 1:
                indices[-1] = n - 1

        # Find choke index
        choke_idx = -1
        if self.choked and self.x_choke is not None:
            choke_idx = int(np.argmin(np.abs(self.x - self.x_choke)))

        header = (
            f"{'x [m]':>8}  {'P [bara]':>9}  {'T [°C]':>7}  "
            f"{'ρ [kg/m³]':>9}  {'u [m/s]':>8}  {'M':>6}  "
            f"{'Re':>10}  {'Z':>6}"
        )
        print(f"\nProfile ({n_stations} stations):")
        print(header)
        print("-" * len(header))
        for idx in indices:
            marker = "*" if idx == choke_idx else " "
            x_val = float(self.x[idx])
            P_bara = float(self.P[idx]) / 1e5
            T_c = float(self.T[idx]) - 273.15
            rho = float(self.rho[idx])
            u = float(self.u[idx])
            M = float(self.M[idx])
            Re = float(self.Re[idx])
            Z = float(self.Z[idx])
            print(
                f"{marker}{x_val:7.2f}  {P_bara:9.3f}  {T_c:7.1f}  "
                f"{rho:9.3f}  {u:8.2f}  {M:6.3f}  "
                f"{Re:10.2e}  {Z:6.3f}"
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Export station-level data as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per station.
        """
        n = len(self.x)
        # Pad segment arrays to station length (repeat last value)
        def _pad(arr: np.ndarray) -> np.ndarray:
            if len(arr) == n - 1:
                return np.append(arr, arr[-1] if len(arr) > 0 else 0.0)
            return arr

        return pd.DataFrame({
            "x_m": self.x,
            "P_bara": self.P / 1e5,
            "T_C": self.T - 273.15,
            "rho_kgm3": self.rho,
            "u_ms": self.u,
            "a_ms": self.a,
            "M": self.M,
            "Re": self.Re,
            "Z": self.Z,
            "h_Jkg": self.h,
            "mu_JT_Kbar": self.mu_JT * 1e5,
            "f": _pad(self.f),
            "dP_fric_bar": _pad(self.dP_fric) / 1e5,
            "dP_acc_bar": _pad(self.dP_acc) / 1e5,
            "dP_elev_bar": _pad(self.dP_elev) / 1e5,
            "dP_fitting_bar": _pad(self.dP_fitting) / 1e5,
            "q_seg_Jkg": _pad(self.q_seg),
        })

    def warnings(self) -> list[str]:
        """Return a list of human-readable warning strings.

        Returns
        -------
        list[str]
        """
        warns: list[str] = []

        # Temperature below hydrate risk threshold
        if np.any(self.T < 273.15):
            T_min_C = float(np.min(self.T)) - 273.15
            warns.append(
                f"T_min ({T_min_C:.1f}°C) below hydrate-risk threshold (0°C). "
                "Consider hydrate inhibition."
            )

        # Segment iteration cap
        if len(self.iterations_per_segment) > 0 and np.any(self.iterations_per_segment >= 30):
            n_cap = int(np.sum(self.iterations_per_segment >= 30))
            warns.append(f"{n_cap} segment(s) hit the 30-iteration Newton cap.")

        # Choke proximity warning
        if self.choked and self.x_choke is not None:
            L = float(self.x[-1])
            pct = self.x_choke / L * 100 if L > 0 else 0.0
            if pct < 95.0:
                warns.append(
                    f"Choke at {self.x_choke:.1f} m ({pct:.0f}% of pipe length) — "
                    "flow is choked well before the outlet."
                )

        return warns

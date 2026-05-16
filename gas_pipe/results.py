"""PipeResult dataclass: stores and formats solver output."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


def _format_discretization_line(opts: dict, n_segs: int) -> str:
    """Format the 'Discretization:' line for the NUMERICAL summary block.

    Reads the ``discretization`` sub-dict stamped on
    ``solver_options`` by the GUI (item 2 follow-up). Falls back to the
    pre-stamp format when the key is absent — keeps backward
    compatibility for results produced by direct API calls or older
    persistence formats.
    """
    discr = opts.get("discretization")
    n_adaptive = int(opts.get("n_adaptive_refinements", 0))
    if discr is None:
        return f"Segments:         {n_segs} ({n_adaptive} adaptive refinements near choke)"

    mode = discr.get("mode")
    if mode == "adaptive":
        initial = int(discr.get("initial_n_segments", n_segs))
        return (
            f"Discretization:   Adaptive (initial {initial} segs, "
            f"refined to {n_segs}, {n_adaptive} refinement events)"
        )
    if mode == "linear":
        n_lin = int(discr.get("n_segments", n_segs))
        dx_m = float(discr.get("dx_m", 0.0))
        return (
            f"Discretization:   Linear, {n_lin} segments, dx = {dx_m:.2f} m"
        )
    if mode == "dimensionless":
        alpha = float(discr.get("alpha", float("nan")))
        dx_m = float(discr.get("dx_m", 0.0))
        n_dim = int(discr.get("n_segments", n_segs))
        return (
            f"Discretization:   Dimensionless, α={alpha:g}, "
            f"dx = {dx_m:.2f} m, {n_dim} segments"
        )
    # Unknown mode — fall back to the historical line.
    return f"Segments:         {n_segs} ({n_adaptive} adaptive refinements near choke)"


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

    # Two-phase diagnostics (Phase: backlog item 1).
    #
    # T_dew[i] is the dew-point temperature at station i's pressure, or
    # NaN if CoolProp could not compute it (e.g. above the cricondenbar
    # or numerical failure). T_margin[i] = T[i] - T_dew[i].
    #
    # metastable_mask[i] is True iff T[i] < T_dew[i] — i.e. the station
    # lies inside the two-phase dome and the solver continued via gas-
    # phase metastable extrapolation.
    #
    # had_metastable summarises the mask; x_dewpoint_crossing is the
    # axial position of the FIRST station where the mask becomes True
    # (None if never crossed). LVF (liquid volume fraction) is filled in
    # post-march for metastable stations only — NaN elsewhere.
    T_dew: np.ndarray = field(default_factory=lambda: np.zeros(0))
    T_margin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    metastable_mask: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )
    had_metastable: bool = False
    x_dewpoint_crossing: float | None = None
    LVF: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Multi-section diagnostics (Phase: backlog item 27).
    # Each entry: {x, from_section, to_section, A_up, A_dn, type, dP,
    # dP_loss, dP_acc}. Empty for single-section pipes.
    section_transitions: list[dict] = field(default_factory=list)

    # Integration direction. ``"forward"`` is the default — inlet → outlet
    # marching via ``march_ivp``. ``"backward"`` is set by
    # ``march_ivp_backward`` for pipes downstream of a chain choke point,
    # where the outlet P is the chain BC and the inlet P is computed by
    # backward integration with the ``h_stag = const`` invariant.
    # See CLAUDE.md "Pressure terminology" for context; the array
    # orientation (``x[0]`` = inlet, ``x[-1]`` = outlet) is the same in
    # both modes — only the order of integration differs.
    march_direction: Literal["forward", "backward"] = "forward"

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

        sections = ps.get("sections", [])
        n_sections = int(ps.get("n_sections", 1))
        transitions = ps.get("section_transitions", [])

        if n_sections > 1 and sections:
            lines.append("SECTIONS")
            for i, sec in enumerate(sections, start=1):
                lines.append(
                    f"  {i}: L = {sec['length']:7.2f} m, "
                    f"ID = {sec['inner_diameter']*1000:6.1f} mm, "
                    f"OD = {sec['outer_diameter']*1000:6.1f} mm, "
                    f"ε = {sec['roughness']*1e6:4.0f} μm, "
                    f"U = {sec['overall_U']:.2f} W/m²/K"
                )
            if transitions:
                lines.append("")
                lines.append("  Section transitions:")
                for t in transitions:
                    arrow_id = (
                        f"{math.sqrt(4*t['A_up']/math.pi)*1000:.0f} mm → "
                        f"{math.sqrt(4*t['A_dn']/math.pi)*1000:.0f} mm"
                    )
                    sign = "drop" if t["dP"] > 0 else "recovery"
                    lines.append(
                        f"    x = {t['x']:7.2f} m: {arrow_id} ({t['type']}), "
                        f"ΔP = {t['dP']/1e5:+.3f} bar ({sign})"
                    )
            lines.append("")

        lines.append("PIPE")
        if n_sections > 1:
            lines.append(f"  Total length:     {L:.2f} m  ({n_sections} sections)")
        else:
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

        # TWO-PHASE diagnostics — emitted only when the march entered the
        # two-phase dome at one or more stations. The section is intentionally
        # qualitative: the LVF estimate comes from a bracket-search on the
        # saturation curve, which is fine for severity classification but is
        # not a substitute for a true two-phase solver (OLGA, HEM, etc.).
        if self.had_metastable:
            meta_idx = np.flatnonzero(self.metastable_mask)
            finite_margin_in_meta = np.isfinite(self.T_margin[meta_idx])
            L_pipe = float(self.x[-1]) - float(self.x[0])

            if meta_idx.size == 0 or not np.any(finite_margin_in_meta):
                # Defensive: had_metastable True but nothing usable to report.
                # Should not occur because metastable_mask requires finite
                # T_dew (so T_margin is finite there too) — but if a future
                # refactor changes that, surface a clear one-line note rather
                # than silently emitting a malformed section.
                lines.append("TWO-PHASE")
                lines.append(
                    "  Dew curve crossed but T_margin could not be"
                    " characterised."
                )
                lines.append(
                    "  Treat results as approximate single-phase"
                    " extrapolation."
                )
                lines.append("")
            else:
                first_idx = int(meta_idx[0])
                last_idx = int(meta_idx[-1])
                x_first = float(self.x[first_idx])
                x_last = float(self.x[last_idx])
                region_len = x_last - x_first
                pct_cross = 100.0 * x_first / L_pipe if L_pipe > 0 else 0.0
                pct_region = 100.0 * region_len / L_pipe if L_pipe > 0 else 0.0

                margin_idx = meta_idx[finite_margin_in_meta]
                min_margin_pos = int(
                    margin_idx[np.argmin(self.T_margin[margin_idx])]
                )
                min_margin_val = float(self.T_margin[min_margin_pos])
                x_min_margin = float(self.x[min_margin_pos])

                meta_lvf = self.LVF[meta_idx]
                finite_meta_lvf = meta_lvf[np.isfinite(meta_lvf)]

                lines.append("TWO-PHASE")
                lines.append(
                    f"  Dew point crossed at:   x = {x_first:.2f} m"
                    f"  ({pct_cross:.1f}% of pipe length)"
                )
                lines.append(
                    f"  Min T_margin:           {min_margin_val:.1f} K"
                    f" at x = {x_min_margin:.2f} m"
                )
                lines.append(
                    f"  Metastable region:      x = {x_first:.2f} m to"
                    f" {x_last:.2f} m ({region_len:.1f} m,"
                    f" {pct_region:.1f}% of pipe)"
                )

                if finite_meta_lvf.size == 0:
                    lines.append(
                        "  Max LVF:                — (flash failed at every"
                        " metastable station)"
                    )
                    lines.append(
                        "  Severity:               LVF could not be"
                        " computed (flash failures)."
                    )
                    lines.append(
                        "                          Treat results as"
                        " approximate single-phase extrapolation."
                    )
                else:
                    lvf_max = float(np.max(finite_meta_lvf))
                    lines.append(
                        f"  Max LVF:                {lvf_max:.3f}"
                        f" ({lvf_max * 100:.1f}%)"
                    )
                    if lvf_max < 0.01:
                        lines.append("  Severity:               Marginal condensation.")
                        lines.append("                          Single-phase model accuracy comparable")
                        lines.append("                          to FSA at low LVF.")
                    elif lvf_max < 0.05:
                        lines.append("  Severity:               Light condensation.")
                        lines.append("                          Single-phase ΔP may under-predict by 5-15%.")
                        lines.append("                          Consider HEM cross-check for safety-critical")
                        lines.append("                          sizing.")
                    else:
                        lines.append("  Severity:               Significant condensation.")
                        lines.append("                          Single-phase model not appropriate.")
                        lines.append("                          Use OLGA two-phase or HEM.")
                lines.append("")

        lines.append("NUMERICAL")
        lines.append(f"  {_format_discretization_line(opts, n_segs)}")
        lines.append(f"  Avg iter/seg:     {avg_iter:.1f}")
        lines.append(f"  Energy residual:  {self.energy_residual:.2e}  (relative)")
        lines.append(f"  Min Δx:           {self.min_dx*1000:.1f} mm")
        lines.append(f"  Elapsed:          {self.elapsed_seconds:.1f} s")
        lines.append("")

        # EOS evaluation — how the EOS was evaluated for this run
        # (tabulated lookup vs direct CoolProp calls). Populated by
        # solve_for_mdot/plateau_sweep when an eos_mode is selected;
        # pre-item-2 results or march_ivp paths that didn't go through
        # solve_for_mdot won't have the key, so the block is conditional.
        eos_mode = opts.get("eos_mode")
        if eos_mode is not None:
            lines.append("EOS evaluation")
            if eos_mode == "table":
                stats = opts.get("table_stats", {})
                n_P = int(stats.get("n_P", 0))
                n_T = int(stats.get("n_T", 0))
                P_min_bara = float(stats.get("P_min", 0.0)) / 1e5
                P_max_bara = float(stats.get("P_max", 0.0)) / 1e5
                T_min_C = float(stats.get("T_min", 0.0)) - 273.15
                T_max_C = float(stats.get("T_max", 0.0)) - 273.15
                n_failed = int(stats.get("n_failed", 0))
                n_oog = int(stats.get("n_outside_grid", 0))
                lines.append(
                    f"  Tabulated ({n_P}×{n_T} grid, "
                    f"P=[{P_min_bara:.2f}, {P_max_bara:.2f}] bara, "
                    f"T=[{T_min_C:.1f}, {T_max_C:.1f}] °C)"
                )
                lines.append(f"  Out-of-grid fallbacks: {n_oog}")
                if n_failed > 0:
                    lines.append(f"  Failed grid points:    {n_failed} (filled by direct EOS at query)")
            elif eos_mode == "direct":
                lines.append("  Direct (CoolProp GERG-2008 calls per evaluation)")
            else:
                lines.append(f"  {eos_mode}")
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

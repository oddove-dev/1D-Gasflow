"""1D gas pipe flow solvers: IVP (march) and BVP (find ṁ).

IVP: march_ivp — given inlet conditions + ṁ, march downstream.
BVP: solve_for_mdot — given inlet + outlet P, find ṁ by bisection.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.optimize import brentq

from .eos import TabulatedFluid, estimate_operating_window

# Env-var override for the default eos_mode. Used by the test suite to
# force ``direct`` mode without touching every old test — see
# ``tests/conftest.py``. Production callers should pass eos_mode
# explicitly; the env var is a back door, not part of the public API.
_DEFAULT_EOS_MODE_ENV = "GAS_PIPE_DEFAULT_EOS_MODE"
from .errors import (
    BVPChoked,
    BVPNotBracketedError,
    ChokeReached,
    IntegrationCapExceeded,
    SegmentConvergenceError,
    SolverCancelled,
)
from .segment import bisect_for_choke, estimate_M_downstream, solve_segment

if TYPE_CHECKING:
    from .eos import FluidEOSBase
    from .geometry import Pipe
    from .results import PipeResult

logger = logging.getLogger(__name__)

_G = 9.80665

# Adaptive-mode discretization constants.
#
# DX_INITIAL_PER_DIAMETER sets the initial segment length as a multiple of
# inner diameter — choosing dx_initial = D gives ~25 segments per Fanno
# length (D/f at typical f ≈ 0.04) which is the empirical sweet spot for
# adaptive refinement: enough resolution to detect the choke band, few
# enough segments that we don't burn time before adaptive refinement has
# anything to do.
DX_INITIAL_PER_DIAMETER = 1.0
INITIAL_N_SEG_MIN = 10
INITIAL_N_SEG_MAX = 500


def initial_n_segments(L: float, D: float) -> int:
    """Initial segment count for adaptive refinement start.

    Scales with pipe diameter (L/D similarity) so initial dimensionless
    resolution is independent of pipe size. Empirically verified that
    dx = D gives comfortable adaptive convergence with minimal
    refinement events for typical compressible pipe flow problems.

    Bounds: 10 (minimum for adaptive to function) to 500 (cap for very
    long pipes).
    """
    return max(
        INITIAL_N_SEG_MIN,
        min(INITIAL_N_SEG_MAX, round(L / (DX_INITIAL_PER_DIAMETER * D))),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_result(
    x_list: list[float],
    states: list,
    A_list: list[float],
    Re_list: list[float],
    seg_info_list: list[dict],
    mdot: float,
    choked: bool,
    x_choke: float | None,
    fluid: "FluidEOSBase",
    pipe: "Pipe",
    bc: dict,
    opts: dict,
    elapsed: float,
    n_adaptive: int,
    section_transitions: list[dict],
) -> "PipeResult":
    """Assemble PipeResult from accumulated per-segment data."""
    from .results import PipeResult

    n = len(x_list)
    x_arr = np.array(x_list, dtype=float)
    P_arr = np.array([s.P for s in states], dtype=float)
    T_arr = np.array([s.T for s in states], dtype=float)
    rho_arr = np.array([s.rho for s in states], dtype=float)
    a_arr = np.array([s.a for s in states], dtype=float)
    # Per-station area is needed for multi-section pipes — at a section
    # boundary the stored state is post-transition (downstream section),
    # so we use the area that march_ivp tracked alongside each station.
    A_arr = np.array(A_list, dtype=float)
    u_arr = mdot / (rho_arr * A_arr)
    M_arr = u_arr / a_arr
    Z_arr = np.array([s.Z for s in states], dtype=float)
    h_arr = np.array([s.h for s in states], dtype=float)
    mu_JT_arr = np.array([s.mu_JT for s in states], dtype=float)

    # Re array: station 0 copied from segment 0; rest from segment info
    Re_seg = np.array(Re_list, dtype=float)
    if len(Re_seg) > 0:
        Re_arr = np.concatenate([[Re_seg[0]], Re_seg])
    else:
        Re_arr = np.zeros(n)

    if len(seg_info_list) > 0:
        f_arr = np.array([info["f"] for info in seg_info_list], dtype=float)
        dP_fric = np.array([info["dP_fric"] for info in seg_info_list], dtype=float)
        dP_acc = np.array([info["dP_acc"] for info in seg_info_list], dtype=float)
        dP_elev = np.array([info["dP_elev"] for info in seg_info_list], dtype=float)
        dP_fit = np.array([info["dP_fitting"] for info in seg_info_list], dtype=float)
        q_seg = np.array([info["q_seg"] for info in seg_info_list], dtype=float)
        iters = np.array([info["n_iter"] for info in seg_info_list], dtype=int)
    else:
        f_arr = dP_fric = dP_acc = dP_elev = dP_fit = q_seg = np.zeros(0)
        iters = np.zeros(0, dtype=int)

    min_dx_val = float(min(np.diff(x_arr))) if len(x_arr) > 1 else float(pipe.length)

    # Energy residual: global enthalpy balance
    total_q = float(np.sum(q_seg))
    H_in = float(states[0].h) + 0.5 * float(u_arr[0]) ** 2 + _G * pipe.z(float(x_arr[0]))
    H_out = float(states[-1].h) + 0.5 * float(u_arr[-1]) ** 2 + _G * pipe.z(float(x_arr[-1]))
    energy_resid = abs(H_out - H_in - total_q) / max(abs(H_in), 1.0)

    D = pipe.inner_diameter
    D_o = pipe.D_o()
    eps = pipe.roughness
    U_val = pipe.U(0.0)
    T_amb_K = pipe.T_amb(0.0)

    sections_summary = [
        {
            "length": s.length,
            "inner_diameter": s.inner_diameter,
            "outer_diameter": s.D_o(),
            "roughness": s.roughness,
            "overall_U": s.overall_U,
            "elevation_change": s.elevation_change,
        }
        for s in pipe.sections
    ]

    pipe_sum = {
        "length": pipe.length,
        "inner_diameter": D,
        "roughness": eps,
        "outer_diameter": D_o,
        "overall_U": U_val,
        "ambient_temperature": T_amb_K,
        "n_fittings": len(pipe.fittings),
        "molar_mass": fluid.molar_mass,
        "n_sections": len(pipe.sections),
        "sections": sections_summary,
        "section_transitions": list(section_transitions),
    }

    opts_full = dict(opts)
    opts_full["n_adaptive_refinements"] = n_adaptive

    # Two-phase diagnostics: at each station compute the dew-point
    # temperature; flag stations where T < T_dew as metastable. The
    # GERGFluid.dew_temperature is cached per-pressure so repeat calls
    # at the same P (common when Newton revisits a state) are free.
    T_dew_arr = np.full(n, np.nan, dtype=float)
    for i, P_i in enumerate(P_arr):
        td = fluid.dew_temperature(float(P_i))
        if td is not None:
            T_dew_arr[i] = td
    T_margin_arr = T_arr - T_dew_arr  # NaN propagates where T_dew unknown
    # metastable_mask: True only where we have a finite T_dew AND T < T_dew.
    # NaN comparisons are False, which is the desired behaviour here.
    with np.errstate(invalid="ignore"):
        metastable_mask = np.asarray(T_arr < T_dew_arr, dtype=bool)
    had_metastable = bool(np.any(metastable_mask))
    x_dewpoint_crossing: float | None = None
    if had_metastable:
        first = int(np.argmax(metastable_mask))
        x_dewpoint_crossing = float(x_arr[first])
    # LVF (liquid volume fraction) — populated only at metastable stations
    # via isenthalpic flash. Non-metastable stations stay NaN, which the
    # summary and plot treat as "not applicable, gas single-phase".
    LVF_arr = np.full(n, np.nan, dtype=float)
    if had_metastable:
        meta_indices = np.flatnonzero(metastable_mask)
        for i in meta_indices:
            LVF_arr[int(i)] = fluid.compute_lvf(
                float(P_arr[int(i)]), float(h_arr[int(i)])
            )

    return PipeResult(
        x=x_arr,
        P=P_arr,
        T=T_arr,
        rho=rho_arr,
        u=u_arr,
        a=a_arr,
        M=M_arr,
        Re=Re_arr,
        Z=Z_arr,
        h=h_arr,
        mu_JT=mu_JT_arr,
        f=f_arr,
        dP_fric=dP_fric,
        dP_acc=dP_acc,
        dP_elev=dP_elev,
        dP_fitting=dP_fit,
        q_seg=q_seg,
        mdot=mdot,
        choked=choked,
        x_choke=x_choke,
        iterations_per_segment=iters,
        energy_residual=energy_resid,
        min_dx=min_dx_val,
        fluid_composition=fluid.composition,
        pipe_summary=pipe_sum,
        boundary_conditions=bc,
        solver_options=opts_full,
        elapsed_seconds=elapsed,
        T_dew=T_dew_arr,
        T_margin=T_margin_arr,
        metastable_mask=metastable_mask,
        had_metastable=had_metastable,
        x_dewpoint_crossing=x_dewpoint_crossing,
        LVF=LVF_arr,
        section_transitions=list(section_transitions),
    )


def _aga_estimate_mdot(
    pipe: "Pipe",
    fluid: "FluidEOSBase",
    P_in: float,
    T_in: float,
    P_out: float,
) -> float:
    """AGA fully-turbulent equation for initial ṁ bracket midpoint.

    Parameters
    ----------
    pipe : Pipe
    fluid : FluidEOSBase
    P_in : float  [Pa]
    T_in : float  [K]
    P_out : float [Pa]

    Returns
    -------
    float
        Estimated mass flow [kg/s].
    """
    D = pipe.inner_diameter
    L = pipe.length
    eps = pipe.roughness
    A = pipe.area

    # Average state
    P_avg = 0.5 * (P_in + P_out)
    T_avg = T_in  # isothermal approximation
    try:
        state_avg = fluid.props(P_avg, T_avg)
        Z_avg = state_avg.Z
        rho_avg = state_avg.rho
        mu_avg = state_avg.mu
    except Exception:
        Z_avg = 0.9
        rho_avg = P_avg / (8314.46 / fluid.molar_mass / Z_avg * T_avg)
        mu_avg = 1e-5

    # Fully turbulent friction
    import math
    f_turb = (-2.0 * math.log10(eps / (3.7 * D))) ** (-2)

    # AGA: P1²-P2² = f * L/D * ρ_avg * u_avg² — simplified
    # P1² - P2² = f * (L/D) * (ṁ/A)² / (2 * ρ_avg)
    dP_sq = P_in ** 2 - P_out ** 2
    if dP_sq <= 0:
        return max(1.0, rho_avg * A * math.sqrt(2 * abs(P_in - P_out) / (rho_avg * f_turb * L / D + 1e-10)))

    mdot_est = A * math.sqrt(2.0 * rho_avg * dP_sq / (2 * P_avg * f_turb * L / D))
    return max(0.01, mdot_est)


# ---------------------------------------------------------------------------
# IVP solver
# ---------------------------------------------------------------------------

def march_ivp(
    pipe: "Pipe",
    fluid: "FluidEOSBase",
    P_in: float,
    T_in: float,
    mdot: float,
    n_segments: int = 100,
    adaptive: bool = True,
    mach_warning: float = 0.7,
    mach_choke: float = 0.99,
    min_dx: float = 1e-3,
    friction_model: str = "blended",
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event: "object | None" = None,
) -> "PipeResult":
    """March inlet → outlet with prescribed mass flow rate.

    Parameters
    ----------
    pipe : Pipe
    fluid : FluidEOSBase
    P_in : float
        Inlet pressure [Pa].
    T_in : float
        Inlet temperature [K].
    mdot : float
        Mass flow rate [kg/s].
    n_segments : int
        Target number of spatial segments.
    adaptive : bool
        Halve Δx when Mach nears the choke threshold.
    mach_warning : float
        Mach fraction at which adaptive refinement begins.
    mach_choke : float
        Mach fraction treated as choke.
    min_dx : float
        Minimum allowed Δx [m].
    friction_model : str
        Friction model for `darcy_friction`.
    progress_callback : Callable or None
        Called as progress_callback(current_segment, total_estimate).

    Returns
    -------
    PipeResult
    """
    t0 = time.time()
    L = pipe.length

    # Initial state
    state0 = fluid.props(P_in, T_in)

    x_list: list[float] = [0.0]
    states: list = [state0]
    A_list: list[float] = [pipe.sections[0].area]
    Re_list: list[float] = []
    seg_info_list: list[dict] = []
    section_transitions: list[dict] = []

    x = 0.0
    remaining = L
    dx_target = L / n_segments
    seg_count = 0
    n_adaptive = 0
    consecutive_pinned = 0

    result_choked = False
    x_choke_final: float | None = None

    state_i = state0

    while remaining > min_dx:
        dx = min(dx_target, remaining)
        dx = max(dx, min_dx)

        # Section-aware geometry at the current upstream station. For
        # multi-section pipes A/D/eps_over_D can change abruptly at
        # interior boundaries — we use the section that owns x (the
        # boundary itself belongs to the upstream section per the
        # section_at convention).
        sec_i, _, _ = pipe.section_at(x)
        A = sec_i.area
        D = sec_i.inner_diameter

        # Cap dx so the segment ends exactly at the next section
        # boundary; the boundary transition is then applied at the
        # clean segment end inside solve_segment.
        boundary_x = pipe.next_section_boundary_after(x)
        if boundary_x is not None and (boundary_x - x) > min_dx and (boundary_x - x) < dx:
            dx = boundary_x - x

        state_prev = state_i
        u_i = mdot / (state_i.rho * A)
        M_i = u_i / state_i.a
        gamma_i = state_i.cp / state_i.cv

        # Layer 1: predictive Mach estimate
        if adaptive and M_i > 0.1:
            from .friction import darcy_friction as _df
            Re_i = state_i.rho * u_i * D / state_i.mu
            f_i = _df(Re_i, sec_i.eps_over_D, friction_model)
            M_pred = estimate_M_downstream(M_i, gamma_i, f_i, dx / D)
            while M_pred > mach_choke and dx > min_dx:
                dx = max(dx * 0.5, min_dx)
                M_pred = estimate_M_downstream(M_i, gamma_i, f_i, dx / D)
                n_adaptive += 1

        x_next = x + dx

        # Attempt to solve the segment
        try:
            state_new, info = solve_segment(
                fluid, pipe, x, x_next, state_i, mdot, friction_model, mach_choke
            )
        except ChokeReached:
            # Layer 3: bisect to locate choke
            state_new, info = bisect_for_choke(
                fluid, pipe, x, dx, state_i, mdot, friction_model, mach_choke, min_dx
            )
            x_next = x + (info.get("x_choke", x + dx * 0.9) - x)
            if info.get("x_choke") is not None:
                x_choke_final = float(info["x_choke"])
            else:
                x_choke_final = x_next
            result_choked = True
        except SegmentConvergenceError as exc:
            logger.warning("Segment convergence failure at x=%.2f m, dx=%.2f m — halving dx", x, dx)
            # Retry with half segment
            if dx > 2 * min_dx:
                dx = max(dx * 0.5, min_dx)
                x_next = x + dx
                try:
                    state_new, info = solve_segment(
                        fluid, pipe, x, x_next, state_i, mdot, friction_model, mach_choke
                    )
                except SegmentConvergenceError:
                    # Both full and half-dx failed — re-raise so BVP knows this mdot is bad
                    raise
            else:
                raise

        # Layer 2+3: check M after solve
        M_ip1 = info.get("M_ip1", 0.0)
        if M_ip1 >= mach_choke and not result_choked:
            # Bisect within this segment
            state_new, info = bisect_for_choke(
                fluid, pipe, x, dx, state_i, mdot, friction_model, mach_choke, min_dx
            )
            if info.get("x_choke") is not None:
                x_choke_final = float(info["x_choke"])
                x_next = x_choke_final
            result_choked = True

        x_list.append(x_next)
        states.append(state_new)
        A_out = info.get("A_out")
        if A_out is None or not math.isfinite(A_out):
            # Fall back to the section that owns x_next (upstream side at
            # an interior boundary). Reached when bisect_for_choke returned
            # without a normal solve_segment finalization.
            sec_next, _, _ = pipe.section_at(x_next)
            A_out = sec_next.area
        A_list.append(float(A_out))
        Re_list.append(info.get("Re", 0.0))
        seg_info_list.append(info)

        st = info.get("section_transition")
        if st is not None:
            section_transitions.append(st)

        state_i = state_new
        x = x_next
        # If we just crossed a section boundary, nudge x past it so that
        # subsequent section_at lookups return the downstream section.
        # The nudge (10 nm) sits above the section_at boundary tolerance
        # (1 nm) and is many orders of magnitude below any physical min_dx,
        # so it's purely a lookup-side fix, not a physical step.
        if st is not None:
            x += 1e-8
        remaining = L - x
        seg_count += 1

        # Adaptive dx refinement, gated on physics rather than absolute M:
        #   - Refine when M is in the near-sonic asymptote band (must
        #     pin dx_target at min_dx for the Fanno asymptote detector
        #     below to fire), OR when M is high and actively climbing.
        #   - Grow back when M is essentially flat per segment, regardless
        #     of absolute level — lets the march escape a pinned-at-min_dx
        #     state when the flow plateaus just above mach_warning (a
        #     pathology observed for Skarv-like geometries where the
        #     steady-state M lands at ~0.703).
        M_jump = M_ip1 - M_i
        in_asymptote_band = M_ip1 > 0.95 * mach_choke
        ramping_toward_choke = M_ip1 >= mach_warning and M_jump > 0.005
        if (adaptive and not result_choked
                and (in_asymptote_band or ramping_toward_choke)):
            dx_target = max(dx_target * 0.5, min_dx)
            n_adaptive += 1
        elif adaptive and M_jump < 0.001 and dx_target < L / n_segments:
            dx_target = min(dx_target * 1.5, L / n_segments)

        # Fanno asymptote detection: when dx is pinned at min_dx and M is
        # well into the asymptotic band (close to mach_choke), the predictive
        # Mach check will keep shrinking dx forever without crossing
        # mach_choke. Treat as effectively choked. The M threshold is set
        # near mach_choke (not at mach_warning) so we don't trip while M is
        # still climbing — the goal is to detect the true asymptote, not
        # approach.
        asymptote_M = 0.95 * mach_choke
        if dx_target <= min_dx * 1.001 and M_ip1 > asymptote_M:
            consecutive_pinned += 1
        else:
            consecutive_pinned = 0

        if consecutive_pinned >= 10 and not result_choked:
            logger.info(
                "march_ivp: Fanno asymptote detected at x=%.2f, M=%.4f, "
                "after %d segments at min_dx. Treating as choked.",
                x, M_ip1, consecutive_pinned,
            )
            result_choked = True
            x_choke_final = x

        # Defensive backstop: hard segment cap. Indicates adaptive-refinement
        # runaway, NOT physical choke. Distinct subclass so BVP can propagate
        # it instead of silently classifying as a choke boundary.
        if seg_count > max(10000, 50 * n_segments):
            raise IntegrationCapExceeded(
                f"march_ivp exceeded segment cap ({seg_count}). "
                f"State: x={x:.2f}/{L:.2f} m, M={M_ip1:.4f}, "
                f"dx_target={dx_target * 1000:.3f} mm, choked={result_choked}."
            )

        # Cooperative cancellation: check every 10 segments. Kept off the
        # hot path so the overhead is negligible vs. the EOS calls per
        # segment.
        if cancel_event is not None and seg_count % 10 == 0 and cancel_event.is_set():
            raise SolverCancelled(
                f"march_ivp cancelled at x={x:.2f}/{L:.2f} m (segment {seg_count})."
            )

        if progress_callback:
            progress_callback(seg_count, n_segments)

        if result_choked:
            break

    elapsed = time.time() - t0
    bc = {
        "P_in": P_in, "T_in": T_in, "mdot": mdot,
        "mode": "IVP",
    }
    opts = {
        "n_segments": n_segments,
        "adaptive": adaptive,
        "mach_warning": mach_warning,
        "mach_choke": mach_choke,
        "min_dx": min_dx,
        "friction_model": friction_model,
    }

    return _build_result(
        x_list, states, A_list, Re_list, seg_info_list,
        mdot=mdot, choked=result_choked, x_choke=x_choke_final,
        fluid=fluid, pipe=pipe, bc=bc, opts=opts,
        elapsed=elapsed, n_adaptive=n_adaptive,
        section_transitions=section_transitions,
    )


# ---------------------------------------------------------------------------
# BVP solver
# ---------------------------------------------------------------------------

def solve_for_mdot(
    pipe: "Pipe",
    fluid: "FluidEOSBase",
    P_in: float,
    T_in: float,
    P_out: float,
    mdot_bracket: tuple[float, float] | None = None,
    rtol: float = 1e-5,
    progress_callback: Callable[[str, float], None] | None = None,
    cancel_event: "object | None" = None,
    eos_mode: str | None = None,
    table_n_P: int = 50,
    table_n_T: int = 50,
    P_range_override: tuple[float, float] | None = None,
    T_range_override: tuple[float, float] | None = None,
    **ivp_kwargs,
) -> "PipeResult":
    """Find mass flow rate such that march_ivp gives the target outlet pressure.

    Parameters
    ----------
    pipe : Pipe
    fluid : FluidEOSBase
    P_in : float
        Inlet pressure [Pa].
    T_in : float
        Inlet temperature [K].
    P_out : float
        Target outlet pressure [Pa].
    mdot_bracket : tuple or None
        (mdot_lo, mdot_hi) bracket. If None, estimated automatically.
    rtol : float
        Relative tolerance on P_out match.
    progress_callback : Callable or None
        Called as (stage_name, fraction).
    eos_mode : {"table", "direct"}
        ``"table"`` (default) wraps ``fluid`` with a
        :class:`TabulatedFluid` over an auto-estimated (or overridden)
        operating window — ~10-100× faster than ``"direct"`` for
        bracketing-heavy BVPs at the cost of ~0.3% accuracy.
        ``"direct"`` uses ``fluid`` as-is for every property call. Pass
        ``"direct"`` when the caller has already wrapped the fluid with
        a :class:`TabulatedFluid` (e.g. :func:`plateau_sweep`) so the
        table isn't re-built per BVP.
    table_n_P, table_n_T : int
        Grid resolution for the auto-built table (ignored when
        ``eos_mode="direct"``).
    P_range_override, T_range_override : tuple or None
        Override the auto-estimated table window.
    **ivp_kwargs
        Passed to march_ivp.

    Returns
    -------
    PipeResult

    Raises
    ------
    BVPChoked
        If the flow chokes before reaching the outlet at all feasible ṁ.
    BVPNotBracketedError
        If the target P_out cannot be bracketed.
    """
    t0 = time.time()

    def _report(stage: str, frac: float) -> None:
        if progress_callback:
            progress_callback(stage, frac)

    # ------------------------------------------------------------------
    # EOS-mode dispatch. The decision is made once here so the rest of
    # the BVP loop calls fluid.props uniformly.
    # ------------------------------------------------------------------
    if eos_mode is None:
        eos_mode = os.environ.get(_DEFAULT_EOS_MODE_ENV, "table")
    if eos_mode == "direct":
        working_fluid: "FluidEOSBase" = fluid
        table: TabulatedFluid | None = None
    elif eos_mode == "table":
        if P_range_override is not None and T_range_override is not None:
            P_range, T_range = P_range_override, T_range_override
        else:
            P_min, P_max, T_min, T_max = estimate_operating_window(
                P_in, T_in, P_out, fluid,
            )
            P_range = P_range_override if P_range_override is not None else (P_min, P_max)
            T_range = T_range_override if T_range_override is not None else (T_min, T_max)
        table = TabulatedFluid(fluid, P_range, T_range, table_n_P, table_n_T)
        working_fluid = table
    else:
        raise ValueError(
            f"eos_mode must be 'table' or 'direct'; got {eos_mode!r}"
        )

    def _annotate_eos(result: "PipeResult") -> "PipeResult":
        """Stamp the result.solver_options with EOS-mode diagnostics."""
        result.solver_options["eos_mode"] = eos_mode
        if table is not None:
            result.solver_options["table_stats"] = table.table_stats()
        return result

    # Cache march_ivp outcomes keyed on rounded mdot. Bisection in
    # _find_critical_mdot revisits the same mdot through closure (the lo/hi
    # bracket bound carries forward), and the final r_crit at mdot_crit is
    # also one of the bisection probes. Caching returns the prior
    # PipeResult or replays the prior exception, eliminating duplicate
    # marches that each cost ~20 seconds in the slow band.
    _march_cache: dict = {}

    def _cached_march(mdot: float):
        # Check cancel BEFORE consulting cache or starting a march, so
        # cancellation requests are honoured promptly between probes.
        if cancel_event is not None and cancel_event.is_set():
            raise SolverCancelled(f"BVP cancelled before probe at mdot={mdot:.3f}")
        key = round(mdot, 2)
        cached = _march_cache.get(key)
        if cached is not None:
            kind, payload = cached
            if kind == "ok":
                return payload
            else:
                raise payload
        try:
            r = march_ivp(
                pipe, working_fluid, P_in, T_in, mdot,
                cancel_event=cancel_event, **ivp_kwargs
            )
            _march_cache[key] = ("ok", r)
            return r
        except (IntegrationCapExceeded, ChokeReached, SegmentConvergenceError) as exc:
            _march_cache[key] = ("exc", exc)
            raise

    # Build bracket
    if mdot_bracket is None:
        mdot_mid = _aga_estimate_mdot(pipe, working_fluid, P_in, T_in, P_out)
        mdot_lo_try = mdot_mid * 0.1
        mdot_hi_try = mdot_mid * 10.0
        # Proactive cap: AGA can be wildly off (30×+) for short or low-loss
        # pipes. Don't let mdot_hi exceed the value that gives M_in = 0.5;
        # any larger mdot drives the inlet near-sonic or supersonic, which
        # produces degenerate n=2 marches that masquerade as "choked" and
        # confuse the bracket logic.
        inlet = working_fluid.props(P_in, T_in)
        mdot_hi_M05 = 0.5 * inlet.a * inlet.rho * pipe.area
        if mdot_hi_try > mdot_hi_M05:
            mdot_hi_try = mdot_hi_M05
        if mdot_lo_try >= mdot_hi_try:
            mdot_lo_try = mdot_hi_try * 0.01
    else:
        mdot_lo_try, mdot_hi_try = mdot_bracket

    _report("bracketing", 0.0)

    # The objective: march_ivp and return (P_out_calc - P_out_target)
    # Choked attempts: return a large NEGATIVE sentinel — signals "ṁ is above
    # critical, must reduce to reach P_out subsonically." This makes the
    # discontinuity at ṁ_critical a clean sign change so brentq can bracket
    # the subsonic root or land on the choke boundary cleanly.
    _CHOKED_SENTINEL = -1e10
    last_successful_mdot: float | None = None
    last_choked_mdot: float | None = None

    def _objective(mdot: float) -> float:
        nonlocal last_successful_mdot, last_choked_mdot
        try:
            r = _cached_march(mdot)
            if r.choked:
                last_choked_mdot = mdot
                return _CHOKED_SENTINEL
            last_successful_mdot = mdot
            return float(r.P[-1]) - P_out
        except IntegrationCapExceeded:
            # Numerical failure — propagate. Caller surfaces as a real error
            # rather than silently treating as a choke boundary.
            raise
        except (ChokeReached, SegmentConvergenceError):
            # ChokeReached is the asymptote hit; SegmentConvergenceError from
            # per-segment Newton divergence often signals near-sonic state.
            last_choked_mdot = mdot
            return _CHOKED_SENTINEL

    # Find bracket: g(mdot_lo) > 0 and g(mdot_hi) < 0, or both same sign
    _report("bracketing", 0.1)
    g_lo = _objective(mdot_lo_try)
    _report("bracketing", 0.3)
    g_hi = _objective(mdot_hi_try)
    _report("bracketing", 0.5)

    logger.debug("BVP bracket: mdot=[%.3f, %.3f] g=[%.1f, %.1f]",
                 mdot_lo_try, mdot_hi_try, g_lo, g_hi)

    # If both positive: try extending bracket up
    n_extend = 0
    while g_hi > 0 and n_extend < 6:
        mdot_hi_try *= 3.0
        g_hi = _objective(mdot_hi_try)
        n_extend += 1

    # If both negative: try extending bracket down
    while g_lo < 0 and n_extend < 12:
        mdot_lo_try *= 0.3
        g_lo = _objective(mdot_lo_try)
        n_extend += 1

    _report("bracketing", 0.7)

    # A sentinel g_hi means ṁ_hi exceeds ṁ_critical. Locate the choke
    # boundary so brentq can operate on a clean non-choked bracket, and so we
    # can decide whether P_target is reachable subsonically at all.
    if g_hi <= _CHOKED_SENTINEL:
        logger.info("BVP: ṁ_hi chokes — locating ṁ_critical.")
        _report("bracketing", 0.8)
        try:
            mdot_crit, r_choked_witness = _find_critical_mdot(
                pipe, working_fluid, P_in, T_in, P_out, mdot_lo_try, mdot_hi_try,
                ivp_kwargs, march_fn=_cached_march
            )
        except Exception as exc:
            raise BVPNotBracketedError(
                f"Could not find ṁ_critical: {exc}"
            ) from exc

        # March at ṁ_critical (largest non-choking ṁ) to get the canonical
        # subsonic answer at the choke boundary. _find_critical_mdot
        # already probed this exact mdot, so the cache returns instantly.
        r_crit = _cached_march(mdot_crit)
        P_out_crit = float(r_crit.P[-1])

        if P_out_crit > P_out * (1 + rtol):
            # P_target is below the choke-limited outlet pressure → unreachable.
            # Use the fresh r_crit march at ṁ_critical as the canonical result;
            # it's a full subsonic profile. Force choked/x_choke post-hoc since
            # for a constant-area Fanno pipe, max flow chokes at the outlet.
            result_to_return = r_crit
            result_to_return.choked = True
            result_to_return.x_choke = pipe.length
            _annotate_eos(result_to_return)
            raise BVPChoked(
                f"Flow chokes at ṁ_critical = {mdot_crit:.3f} kg/s; "
                f"target P_out = {P_out/1e5:.3f} bara is unreachable.",
                mdot_critical=mdot_crit,
                result=result_to_return,
            )

        # Subsonic solution exists between mdot_lo_try and mdot_crit;
        # tighten the bracket so brentq sees only the smooth branch.
        mdot_hi_try = mdot_crit
        g_hi = P_out_crit - P_out

    if g_lo * g_hi > 0:
        raise BVPNotBracketedError(
            f"Cannot bracket ṁ: g({mdot_lo_try:.2f})={g_lo:.1f}, "
            f"g({mdot_hi_try:.2f})={g_hi:.1f}. "
            "Outlet pressure may be unreachable for this geometry."
        )

    _report("iterating", 0.0)

    # Brentq solve
    iter_count = [0]
    def _wrapped(mdot: float) -> float:
        iter_count[0] += 1
        frac = min(0.99, iter_count[0] / 20.0)
        _report("iterating", frac)
        return _objective(mdot)

    try:
        mdot_sol = brentq(_wrapped, mdot_lo_try, mdot_hi_try, rtol=rtol, maxiter=50)
    except ValueError as exc:
        raise BVPNotBracketedError(f"brentq failed: {exc}") from exc

    _report("finalizing", 0.0)
    result = _cached_march(mdot_sol)
    result.elapsed_seconds  # already set inside march_ivp

    # If this solution itself is choked and P_out_calc > P_target
    if result.choked and float(result.P[-1]) > P_out * (1 + rtol):
        _annotate_eos(result)
        raise BVPChoked(
            f"BVP choked: ṁ_critical = {mdot_sol:.3f} kg/s.",
            mdot_critical=mdot_sol,
            result=result,
        )

    _report("finalizing", 1.0)
    return _annotate_eos(result)


def verify_eos_accuracy(
    pipe: "Pipe",
    fluid: "FluidEOSBase",
    P_in: float,
    T_in: float,
    P_out: float,
    table_n_P: int = 50,
    table_n_T: int = 50,
    P_range_override: tuple[float, float] | None = None,
    T_range_override: tuple[float, float] | None = None,
    cancel_event: "object | None" = None,
    **ivp_kwargs,
) -> dict:
    """Run a BVP both ways (direct and table) and report the deviations.

    Used by the GUI's "Verify table accuracy" button — i.e. it measures
    how closely the tabulated bilinear lookup tracks direct EOS calls,
    not the EOS itself against external reference data. Returns a dict
    the caller can format into a comparison dialog or attach to a
    summary.

    Returns
    -------
    dict
        Keys: ``result_direct``, ``result_table``, ``mdot_rel_diff``,
        ``P_out_diff_Pa``, ``T_out_diff_K``, ``x_choke_diff_m``,
        ``max_rho_rel``, ``max_h_rel``, ``max_a_rel``, ``max_mu_rel``,
        ``elapsed_direct``, ``elapsed_table``, ``speedup``.
    """
    import time as _time

    def _run(mode: str) -> tuple["PipeResult", float]:
        t = _time.time()
        try:
            r = solve_for_mdot(
                pipe, fluid, P_in, T_in, P_out,
                eos_mode=mode,
                table_n_P=table_n_P, table_n_T=table_n_T,
                P_range_override=P_range_override,
                T_range_override=T_range_override,
                cancel_event=cancel_event,
                **ivp_kwargs,
            )
        except BVPChoked as exc:
            r = exc.result
        return r, _time.time() - t

    r_direct, t_direct = _run("direct")
    r_table, t_table = _run("table")

    # Station-by-station deltas. Compare on the shorter array since the
    # two marches may have different segment counts after adaptive
    # refinement; resampling onto a common x-grid is more honest than
    # truncating but for a rough verification the truncation is fine.
    n = min(len(r_direct.x), len(r_table.x))
    rho_rel = np.abs(r_table.rho[:n] - r_direct.rho[:n]) / np.maximum(r_direct.rho[:n], 1e-30)
    h_scale = np.maximum(np.abs(r_direct.h[:n]), 1e4)
    h_rel = np.abs(r_table.h[:n] - r_direct.h[:n]) / h_scale
    a_rel = np.abs(r_table.a[:n] - r_direct.a[:n]) / np.maximum(r_direct.a[:n], 1e-30)
    # Some stations may have NaN mu_JT if EOS partial deriv failed —
    # nanmax keeps the comparison robust.
    mu_rel = np.abs(r_table.mu_JT[:n] - r_direct.mu_JT[:n]) / np.maximum(np.abs(r_direct.mu_JT[:n]), 1e-30)

    x_choke_diff: float
    if r_direct.x_choke is not None and r_table.x_choke is not None:
        x_choke_diff = abs(r_table.x_choke - r_direct.x_choke)
    else:
        x_choke_diff = float("nan")

    return {
        "result_direct": r_direct,
        "result_table": r_table,
        "mdot_rel_diff": abs(r_table.mdot - r_direct.mdot) / max(r_direct.mdot, 1e-30),
        "P_out_diff_Pa": float(abs(r_table.P[-1] - r_direct.P[-1])),
        "T_out_diff_K": float(abs(r_table.T[-1] - r_direct.T[-1])),
        "x_choke_diff_m": x_choke_diff,
        "max_rho_rel": float(np.nanmax(rho_rel)) if rho_rel.size else 0.0,
        "max_h_rel": float(np.nanmax(h_rel)) if h_rel.size else 0.0,
        "max_a_rel": float(np.nanmax(a_rel)) if a_rel.size else 0.0,
        "max_mu_rel": float(np.nanmax(mu_rel)) if mu_rel.size else 0.0,
        "elapsed_direct": t_direct,
        "elapsed_table": t_table,
        "speedup": t_direct / max(t_table, 1e-9),
        "table_stats": r_table.solver_options.get("table_stats", {}),
    }


def _find_critical_mdot(
    pipe: "Pipe",
    fluid: "FluidEOSBase",
    P_in: float,
    T_in: float,
    P_out_target: float,
    mdot_lo: float,
    mdot_hi: float,
    ivp_kwargs: dict,
    n_iter: int = 30,
    march_fn: Callable | None = None,
) -> tuple[float, "PipeResult | None"]:
    """Bisect to find ṁ_critical: largest ṁ that doesn't choke at outlet.

    Parameters
    ----------
    pipe, fluid, P_in, T_in, P_out_target : as in solve_for_mdot
    mdot_lo, mdot_hi : float
        Search range for ṁ.
    ivp_kwargs : dict
        Forwarded to march_ivp.
    n_iter : int
        Maximum bisection iterations.
    march_fn : callable or None
        Cached march callable from solve_for_mdot. If None, calls
        march_ivp directly (no caching).

    Returns
    -------
    tuple[float, PipeResult | None]
        Critical mass flow rate, and the most recent successfully-marched
        choked PipeResult observed during the bisection (None if no choked
        result was ever captured cleanly — e.g., all probes raised).
    """
    last_choked_result: "PipeResult | None" = None

    def _do_march(mdot: float):
        if march_fn is not None:
            return march_fn(mdot)
        return march_ivp(pipe, fluid, P_in, T_in, mdot, **ivp_kwargs)

    def _chokes(mdot: float) -> bool:
        nonlocal last_choked_result
        try:
            r = _do_march(mdot)
            if r.choked:
                last_choked_result = r
                return True
            return False
        except (ChokeReached, SegmentConvergenceError):
            # SegmentConvergenceError covers both per-segment Newton failures
            # and IntegrationCapExceeded. Inside the bisection we treat both
            # as "this mdot is at/above the numerical/physical limit" so we
            # narrow downward — this gives a CONSERVATIVE mdot_critical
            # bound. The final BVP probe (_objective) still propagates
            # IntegrationCapExceeded as a real error.
            return True

    # Find a bracket: lo=not choked, hi=choked
    # First check if lo is choked
    for _ in range(10):
        if not _chokes(mdot_lo):
            break
        mdot_lo *= 0.5
    else:
        return mdot_lo, last_choked_result

    for _ in range(n_iter):
        mdot_mid = 0.5 * (mdot_lo + mdot_hi)
        if _chokes(mdot_mid):
            mdot_hi = mdot_mid
        else:
            mdot_lo = mdot_mid
        # 0.1% tolerance on mdot_critical is well within engineering needs;
        # tightening to 0.01% triples bisection probes for no real benefit.
        if (mdot_hi - mdot_lo) / max(mdot_lo, 1e-6) < 1e-3:
            break

    return mdot_lo, last_choked_result


# ---------------------------------------------------------------------------
# Plateau sweep (Phase 15)
# ---------------------------------------------------------------------------

def plateau_sweep(
    pipe: "Pipe",
    fluid: "FluidEOSBase",
    P_in: float,
    T_in: float,
    P_out_array: "np.ndarray | list[float]",
    ivp_kwargs: dict | None = None,
    cancel_event: "object | None" = None,
    on_point: Callable[[int, int, dict], None] | None = None,
    eos_mode: str | None = None,
    table_n_P: int = 50,
    table_n_T: int = 50,
    P_range_override: tuple[float, float] | None = None,
    T_range_override: tuple[float, float] | None = None,
) -> list[dict]:
    """Run a BVP at each P_out and collect the (P_out, ṁ) plateau curve.

    Each point is run as a fresh BVP. ``BVPChoked`` is treated as a
    success — the point is captured with ``choked=True`` and ``mdot``
    set to the critical mass flow at that target. Other exceptions are
    captured per-point so one failure does not abort the whole sweep.
    ``SolverCancelled`` propagates out so the worker thread can surface
    a clean cancellation.

    Parameters
    ----------
    pipe, fluid, P_in, T_in : as in solve_for_mdot.
    P_out_array : sequence of float
        Outlet pressures [Pa] to sweep. Order is preserved in the
        returned list.
    ivp_kwargs : dict or None
        Forwarded to solve_for_mdot (and on through to march_ivp).
    cancel_event : threading.Event-like or None
        Cooperative cancel; checked before each point and inside each
        BVP probe via solve_for_mdot.
    on_point : callable or None
        Called as ``on_point(idx_completed, total, point_dict)`` after
        each point. ``idx_completed`` is 1-based.
    eos_mode : {"table", "direct"} or None
        ``"table"`` (default) builds a single :class:`TabulatedFluid`
        sized to cover the worst-case BVP probe (lowest P_out → largest
        ΔP and JT cooling) and reuses it across every probe — this is
        the dominant speedup for plateau sweeps. ``"direct"`` falls
        back to the raw EOS for every probe. ``None`` resolves via the
        same env-var override as :func:`solve_for_mdot`.
    table_n_P, table_n_T : int
        Grid resolution for the shared table.
    P_range_override, T_range_override : tuple or None
        Override the auto-estimated table window.

    Returns
    -------
    list[dict]
        One dict per point with keys: ``P_out`` [Pa], ``mdot`` [kg/s],
        ``choked`` (bool), ``M_out``, ``T_out`` [K], ``x_choke`` [m or
        None], ``result`` (PipeResult or None), ``error`` (str or None).
    """
    if ivp_kwargs is None:
        ivp_kwargs = {}

    if eos_mode is None:
        eos_mode = os.environ.get(_DEFAULT_EOS_MODE_ENV, "table")

    # Build the shared table here (rather than letting each probe build
    # its own) so it amortises over the whole sweep — the lowest P_out
    # is the worst-case window, so any other probe's state is contained.
    if eos_mode == "table":
        if P_range_override is not None and T_range_override is not None:
            P_range, T_range = P_range_override, T_range_override
        else:
            P_out_lowest = float(min(P_out_array))
            P_min, P_max, T_min, T_max = estimate_operating_window(
                P_in, T_in, P_out_lowest, fluid,
            )
            P_range = P_range_override if P_range_override is not None else (P_min, P_max)
            T_range = T_range_override if T_range_override is not None else (T_min, T_max)
        working_fluid: "FluidEOSBase" = TabulatedFluid(
            fluid, P_range, T_range, table_n_P, table_n_T,
        )
    elif eos_mode == "direct":
        working_fluid = fluid
    else:
        raise ValueError(
            f"eos_mode must be 'table' or 'direct'; got {eos_mode!r}"
        )

    points: list[dict] = []
    total = len(P_out_array)

    for i, P_out in enumerate(P_out_array):
        if cancel_event is not None and cancel_event.is_set():
            raise SolverCancelled(
                f"plateau_sweep cancelled at point {i + 1}/{total}"
            )

        result: "PipeResult | None" = None
        choked = False
        error_msg: str | None = None
        try:
            # Pass eos_mode='direct' so solve_for_mdot doesn't re-wrap
            # the already-tabulated fluid (or re-build a table per point).
            result = solve_for_mdot(
                pipe, working_fluid, P_in, T_in, P_out,
                cancel_event=cancel_event,
                eos_mode="direct",
                **ivp_kwargs,
            )
        except BVPChoked as exc:
            result = exc.result
            choked = True
        except SolverCancelled:
            raise
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

        if result is not None:
            point = {
                "P_out": float(P_out),
                "mdot": float(result.mdot),
                "choked": bool(choked or result.choked),
                "M_out": float(result.M[-1]),
                "T_out": float(result.T[-1]),
                "x_choke": (float(result.x_choke)
                            if result.choked and result.x_choke is not None
                            else None),
                "result": result,
                "error": None,
            }
        else:
            point = {
                "P_out": float(P_out),
                "mdot": float("nan"),
                "choked": False,
                "M_out": float("nan"),
                "T_out": float("nan"),
                "x_choke": None,
                "result": None,
                "error": error_msg,
            }

        points.append(point)
        if on_point is not None:
            try:
                on_point(i + 1, total, point)
            except Exception:
                # Progress callback failures must not abort the sweep.
                logger.warning("on_point callback raised; continuing sweep")

    return points

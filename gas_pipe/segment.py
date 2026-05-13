"""Per-segment Newton solver for the 1D gas flow equations.

Solves the coupled momentum + energy equations over a finite segment
[x_i, x_{i+1}] using Newton's method with finite-difference Jacobian.
Includes three-layer hardened choke detection.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Callable

from .errors import ChokeReached, SegmentConvergenceError
from .friction import darcy_friction

if TYPE_CHECKING:
    from .eos import FluidEOSBase, FluidState
    from .geometry import Pipe

logger = logging.getLogger(__name__)

_G = 9.80665  # m/s²

# Newton convergence tolerances.
# 1e-6 matches the practical accuracy floor of the finite-difference Jacobian
# (FD step = 1e-6 * P, so Jacobian elements have ~1e-6 relative noise).
# The fallback at 1e-4 accepts nearly-converged solutions within engineering tolerance.
_RTOL_P = 1e-6
_RTOL_H = 1e-6


def _mach(state: "FluidState", mdot: float, area: float) -> float:
    u = mdot / (state.rho * area)
    return u / state.a


def _default_segment_info() -> dict:
    """Canonical per-segment info dict with all keys present and benign defaults.

    Used as the initial value for bisect_for_choke's last_good_info so that
    callers reading info["f"], info["Re"], etc. don't KeyError when every
    bisect iteration raises.
    """
    return {
        "n_iter": 0,
        "residual_history": [],
        "f": float("nan"),
        "Re": float("nan"),
        "M_ip1": 0.0,
        "choked_in_segment": False,
        "x_choke": None,
        "applied_fittings": [],
        "dP_fric": 0.0,
        "dP_acc": 0.0,
        "dP_elev": 0.0,
        "dP_fitting": 0.0,
        "q_seg": 0.0,
        "A_out": float("nan"),
        "section_transition": None,
    }


def _segment_residuals(
    P2: float,
    T2: float,
    fluid: "FluidEOSBase",
    pipe: "Pipe",
    x1: float,
    x2: float,
    state1: "FluidState",
    mdot: float,
    friction_model: str,
) -> tuple[float, float, "FluidState"]:
    """Evaluate momentum and energy residuals for downstream state (P2, T2).

    Returns
    -------
    tuple[float, float, FluidState]
        (R_momentum, R_energy, state_2)
    """
    state2 = fluid.props(P2, T2)
    x_mid = 0.5 * (x1 + x2)
    sec, _, _ = pipe.section_at(x_mid)
    A = sec.area
    D = sec.inner_diameter
    dx = x2 - x1

    u1 = mdot / (state1.rho * A)
    u2 = mdot / (state2.rho * A)

    P_avg = 0.5 * (state1.P + P2)
    T_avg = 0.5 * (state1.T + T2)
    state_avg = fluid.props(P_avg, T_avg)
    rho_avg = state_avg.rho
    mu_avg = state_avg.mu

    # Mass-consistent average velocity
    u_avg = mdot / (rho_avg * A)

    Re = rho_avg * u_avg * D / mu_avg
    f = darcy_friction(Re, sec.eps_over_D, friction_model)

    dP_fric = f * (dx / D) * rho_avg * u_avg ** 2 / 2.0
    dP_acc = (mdot / A) ** 2 * (1.0 / state2.rho - 1.0 / state1.rho)
    dz = pipe.z(x2) - pipe.z(x1)
    dP_elev = rho_avg * _G * dz

    R1 = (state1.P - P2) - dP_fric - dP_acc - dP_elev

    # Heat per unit mass
    T_amb_mid = pipe.T_amb(x_mid)
    U_mid = sec.overall_U
    D_o = sec.D_o()
    q = U_mid * math.pi * D_o * dx * (T_amb_mid - T_avg) / mdot

    h1 = state1.h
    h2 = state2.h
    gz = _G * dz
    R2 = (h2 - h1) + 0.5 * (u2 ** 2 - u1 ** 2) + gz - q

    return R1, R2, state2


def estimate_M_downstream(
    M_i: float,
    gamma_avg: float,
    f_avg: float,
    dx_over_D: float,
) -> float:
    """Fanno-line predictor for downstream Mach number (ideal-gas, explicit Euler).

    Parameters
    ----------
    M_i : float
        Upstream Mach number.
    gamma_avg : float
        Average specific heat ratio.
    f_avg : float
        Darcy friction factor.
    dx_over_D : float
        Segment length over diameter.

    Returns
    -------
    float
        Estimated downstream Mach number.
    """
    M2 = M_i ** 2
    num = gamma_avg * M2 * (1.0 + 0.5 * (gamma_avg - 1.0) * M2)
    den = (1.0 - M2) if abs(1.0 - M2) > 1e-12 else 1e-12
    dM2_dx = num / den * (4.0 * f_avg / (dx_over_D / max(dx_over_D, 1e-12) if dx_over_D > 0 else 1))
    # dx_over_D is L/D of the segment, so dM²/dL * D = dM²/d(x/D)
    # dM² = dM²/d(x/D) * dx/D
    dM2 = num / den * 4.0 * f_avg * dx_over_D
    M2_down = M2 + dM2
    return math.sqrt(max(0.0, M2_down))


def _initial_guess(
    state1: "FluidState",
    fluid: "FluidEOSBase",
    pipe: "Pipe",
    x1: float,
    x2: float,
    mdot: float,
    friction_model: str,
) -> tuple[float, float]:
    """Explicit-Euler initial guess for (P2, T2)."""
    x_mid = 0.5 * (x1 + x2)
    sec, _, _ = pipe.section_at(x_mid)
    A = sec.area
    D = sec.inner_diameter
    dx = x2 - x1
    rho1 = state1.rho
    u1 = mdot / (rho1 * A)
    Re1 = rho1 * u1 * D / state1.mu
    f1 = darcy_friction(Re1, sec.eps_over_D, friction_model)
    dz = pipe.z(x2) - pipe.z(x1)

    P2_guess = state1.P - f1 * (dx / D) * rho1 * u1 ** 2 / 2.0 - rho1 * _G * dz
    P2_guess = max(P2_guess, 1000.0)  # 1 kPa floor

    # Joule-Thomson estimate for temperature
    dP = P2_guess - state1.P
    T_amb_mid = pipe.T_amb(x_mid)
    U_mid = sec.overall_U
    D_o = sec.D_o()
    heat_term = U_mid * math.pi * D_o * dx * (T_amb_mid - state1.T) / (mdot * state1.cp) if mdot > 0 else 0.0
    T2_guess = state1.T + state1.mu_JT * dP + heat_term
    T2_guess = max(min(T2_guess, 1000.0), 100.0)

    return P2_guess, T2_guess


def solve_segment(
    fluid: "FluidEOSBase",
    pipe: "Pipe",
    x_i: float,
    x_ip1: float,
    state_i: "FluidState",
    mdot: float,
    friction_model: str = "blended",
    mach_choke: float = 0.99,
) -> tuple["FluidState", dict]:
    """Solve one finite segment using Newton's method.

    Solves the coupled momentum + energy equations for (P_{i+1}, T_{i+1})
    given upstream state and mass flow rate.

    Parameters
    ----------
    fluid : FluidEOSBase
    pipe : Pipe
    x_i : float
        Upstream station position [m].
    x_ip1 : float
        Downstream station position [m].
    state_i : FluidState
        Upstream thermodynamic state.
    mdot : float
        Mass flow rate [kg/s].
    friction_model : str
        Friction model identifier.
    mach_choke : float
        Mach threshold above which choke is flagged.

    Returns
    -------
    tuple[FluidState, dict]
        Downstream state and info dict with diagnostics.
    """
    x_mid = 0.5 * (x_i + x_ip1)
    sec, _, _ = pipe.section_at(x_mid)
    A = sec.area
    D = sec.inner_diameter
    dx = x_ip1 - x_i

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------
    P2, T2 = _initial_guess(state_i, fluid, pipe, x_i, x_ip1, mdot, friction_model)

    # ------------------------------------------------------------------
    # Newton iteration
    # ------------------------------------------------------------------
    max_iter = 50
    residual_history: list[tuple[float, float]] = []
    state2 = fluid.props(P2, T2)

    dP_step_base = max(1.0, 1e-6 * state_i.P)
    dT_step_base = max(1e-3, 1e-6 * state_i.T)

    h_scale = max(abs(state_i.h), 0.5 * (mdot / (state_i.rho * A)) ** 2, 1.0)

    n_iter = 0
    converged = False

    # Chord-method Newton: compute the Jacobian once at iteration 0, reuse
    # for subsequent iterations. Convergence degrades from quadratic to
    # linear, but each iteration costs 2 EOS calls instead of 6 (no FD
    # perturbations). Net win because Newton in this 2-equation
    # well-conditioned system typically only needs a handful of extra
    # iterations to compensate. If residuals stagnate, we refresh the
    # Jacobian.
    J11 = J12 = J21 = J22 = 0.0
    det = 0.0
    cond = 0.0
    have_jacobian = False
    iter_at_last_jac = -1
    last_R1 = last_R2 = float("inf")

    for n_iter in range(max_iter):
        R1, R2, state2 = _segment_residuals(P2, T2, fluid, pipe, x_i, x_ip1, state_i, mdot, friction_model)
        residual_history.append((R1, R2))

        # Check convergence
        if abs(R1) / state_i.P < _RTOL_P and abs(R2) / h_scale < _RTOL_H:
            converged = True
            break

        # Refresh the Jacobian on iter 0, after a stagnation, or every 6
        # iterations as a safety net for slow convergence.
        residual_norm = abs(R1) / state_i.P + abs(R2) / h_scale
        last_norm = abs(last_R1) / state_i.P + abs(last_R2) / h_scale
        stagnated = have_jacobian and residual_norm > 0.5 * last_norm
        too_old = have_jacobian and (n_iter - iter_at_last_jac) >= 6
        if not have_jacobian or stagnated or too_old:
            dP_step = max(1.0, 1e-6 * abs(P2))
            dT_step = max(1e-3, 1e-6 * abs(T2))

            R1_dP, R2_dP, _ = _segment_residuals(P2 + dP_step, T2, fluid, pipe, x_i, x_ip1, state_i, mdot, friction_model)
            R1_dT, R2_dT, _ = _segment_residuals(P2, T2 + dT_step, fluid, pipe, x_i, x_ip1, state_i, mdot, friction_model)

            J11 = (R1_dP - R1) / dP_step
            J12 = (R1_dT - R1) / dT_step
            J21 = (R2_dP - R2) / dP_step
            J22 = (R2_dT - R2) / dT_step
            det = J11 * J22 - J12 * J21
            cond = (abs(J11) + abs(J12) + abs(J21) + abs(J22)) / (abs(det) + 1e-300)
            have_jacobian = True
            iter_at_last_jac = n_iter

        last_R1, last_R2 = R1, R2

        if abs(det) < 1e-30:
            # Jacobian is singular — emergency gradient step + force refresh
            P2 -= 0.01 * R1
            T2 -= 0.01 * R2
            have_jacobian = False
            continue

        dP2 = -(R1 * J22 - R2 * J12) / det
        dT2 = -(J11 * R2 - J21 * R1) / det

        # Line search with bounds check
        alpha = 1.0
        for _ in range(6):
            P2_trial = P2 + alpha * dP2
            T2_trial = T2 + alpha * dT2
            # Bounds
            P2_trial = max(100.0, P2_trial)
            T2_trial = max(100.0, min(1000.0, T2_trial))

            # Layer 2: check Mach overshoot
            try:
                state_trial = fluid.props(P2_trial, T2_trial)
                u_trial = mdot / (state_trial.rho * A)
                M_trial = u_trial / state_trial.a
                if M_trial <= 1.0:
                    break
            except Exception:
                pass
            alpha *= 0.5
        else:
            # After 5 backtracks still M > 1 — this segment is choked
            raise ChokeReached(f"Mach overshoot in Newton at x={x_i:.3f}–{x_ip1:.3f} m")

        P2 = max(100.0, P2 + alpha * dP2)
        T2 = max(100.0, min(1000.0, T2 + alpha * dT2))

    if not converged:
        R1_f, R2_f, state2 = _segment_residuals(P2, T2, fluid, pipe, x_i, x_ip1, state_i, mdot, friction_model)
        # Accept approximate solutions within a loose tolerance (1e-4 relative).
        # The FD Jacobian limits precision to ~1e-6; tighter than 1e-4 is sufficient for engineering.
        _FALLBACK = 1e-4
        if not (abs(R1_f) / state_i.P < _FALLBACK and abs(R2_f) / h_scale < _FALLBACK):
            raise SegmentConvergenceError(
                f"Newton failed at x={x_i:.3f}–{x_ip1:.3f} m after {max_iter} iterations. "
                f"Residuals: R1/P={abs(R1_f)/state_i.P:.2e}, R2/h={abs(R2_f)/h_scale:.2e}. "
                f"Jacobian condition: {cond:.2e}"
            )

    # Recompute final residuals and secondary quantities
    R1_f, R2_f, state2 = _segment_residuals(P2, T2, fluid, pipe, x_i, x_ip1, state_i, mdot, friction_model)

    # Recompute info components
    state_avg = fluid.props(0.5 * (state_i.P + P2), 0.5 * (state_i.T + T2))
    rho_avg = state_avg.rho
    mu_avg = state_avg.mu
    u_avg = mdot / (rho_avg * A)
    Re = rho_avg * u_avg * D / mu_avg
    f_val = darcy_friction(Re, sec.eps_over_D, friction_model)

    u2 = mdot / (state2.rho * A)
    M_ip1 = u2 / state2.a

    dP_fric = f_val * (dx / D) * rho_avg * u_avg ** 2 / 2.0
    dP_acc = (mdot / A) ** 2 * (1.0 / state2.rho - 1.0 / state_i.rho)
    dz = pipe.z(x_ip1) - pipe.z(x_i)
    dP_elev = rho_avg * _G * dz

    T_amb_mid = pipe.T_amb(x_mid)
    U_mid = sec.overall_U
    D_o = sec.D_o()
    q_seg = U_mid * math.pi * D_o * dx * (T_amb_mid - 0.5 * (state_i.T + T2)) / mdot

    # ------------------------------------------------------------------
    # Apply fittings in this segment
    # ------------------------------------------------------------------
    choked_fitting = False
    applied_fittings: list[str] = []
    dP_fitting_total = 0.0

    fittings_in_seg = [
        ft for ft in pipe.fittings
        if x_i <= ft.location < x_ip1
    ]
    fittings_in_seg.sort(key=lambda ft: ft.location)

    for ft in fittings_in_seg:
        u_up = mdot / (state2.rho * A)
        dP_ft = ft.pressure_loss(state2, u_up)
        P_after = state2.P - dP_ft
        if P_after < 100.0:
            logger.warning("Fitting %s would create P < 100 Pa; capping.", ft.name)
            P_after = 100.0
        # Isenthalpic across a K-factor fitting. We avoid the (P,h) flash
        # because CoolProp's GERG-2008 backend can't perform it for mixtures
        # without a pre-built phase envelope; instead we use the JT
        # coefficient to estimate T after the drop and do a (P,T) flash,
        # which is robust for any fluid and accurate to first order in ΔP.
        T_after = state2.T - state2.mu_JT * dP_ft
        try:
            state2 = fluid.props(P_after, T_after)
        except Exception as exc:
            logger.warning("Fitting %s state update failed: %s", ft.name, exc)
            break

        dP_fitting_total += dP_ft
        applied_fittings.append(ft.name or f"K={ft.K}")

        # Recheck Mach after fitting
        u_after = mdot / (state2.rho * A)
        M_after = u_after / state2.a
        if M_after >= mach_choke:
            choked_fitting = True
            logger.debug("Fitting %s caused choke at x=%.3f m", ft.name, ft.location)
            break

    # ------------------------------------------------------------------
    # Section boundary transition — applied when x_ip1 lands exactly on
    # an interior boundary between two PipeSections of different area.
    # Borda-Carnot loss for expansions, Crane K-factor for contractions.
    # The post-transition state2 is in the downstream section, so the
    # caller must use A_out (returned in info) for the next iteration.
    # ------------------------------------------------------------------
    section_transition: dict | None = None
    A_out = A
    boundary_idx = pipe.is_at_section_boundary(x_ip1)
    if boundary_idx is not None and not choked_fitting:
        section_up = pipe.sections[boundary_idx]
        section_dn = pipe.sections[boundary_idx + 1]
        A_up = section_up.area
        A_dn = section_dn.area
        if abs(A_dn - A_up) > 1e-15 * A_up:
            rho_pre = state2.rho
            u_up = mdot / (rho_pre * A_up)
            u_dn = mdot / (rho_pre * A_dn)

            if A_dn > A_up:
                # Borda-Carnot expansion loss
                beta = A_up / A_dn
                dP_loss = 0.5 * rho_pre * u_up ** 2 * (1.0 - beta) ** 2
                t_type = "expansion"
            else:
                # Sudden contraction — Crane TP-410 K = 0.5·(1 − β),
                # referred to the downstream velocity head.
                beta = A_dn / A_up
                K_c = 0.5 * (1.0 - beta)
                dP_loss = 0.5 * rho_pre * u_dn ** 2 * K_c
                t_type = "contraction"

            # Acceleration term at constant ρ across the transition.
            # Negative for expansion (pressure rises), positive for
            # contraction (additional drop).
            dP_acc_trans = 0.5 * rho_pre * (u_dn ** 2 - u_up ** 2)
            dP_total = dP_loss + dP_acc_trans
            P_new = max(state2.P - dP_total, 100.0)

            # CoolProp's HSU_P_flash for HEOS mixtures raises
            # "phase envelope must be built" — the JT path is robust
            # and fast for any composition, so we skip props_Ph for
            # mixtures rather than paying the failed-flash overhead
            # (and historically corrupting the AbstractState phase spec).
            if fluid.is_mixture:
                state2 = fluid.props_Ph_via_jt(P_new, state2)
            else:
                try:
                    state2 = fluid.props_Ph(P_new, state2.h)
                except Exception as exc:
                    logger.debug("props_Ph failed at section boundary x=%.3f m: %s; "
                                 "falling back to JT estimate.", x_ip1, exc)
                    state2 = fluid.props_Ph_via_jt(P_new, state2)

            A_out = A_dn
            section_transition = {
                "x": x_ip1,
                "from_section": boundary_idx,
                "to_section": boundary_idx + 1,
                "A_up": A_up,
                "A_dn": A_dn,
                "type": t_type,
                "dP": dP_total,
                "dP_loss": dP_loss,
                "dP_acc": dP_acc_trans,
            }

    # Final Mach — use A_out so post-transition Mach uses A_dn.
    u_final = mdot / (state2.rho * A_out)
    M_final = u_final / state2.a

    info = {
        "n_iter": n_iter + 1,
        "residual_history": residual_history,
        "f": f_val,
        "Re": Re,
        "M_ip1": M_final,
        "choked_in_segment": M_final >= mach_choke or choked_fitting,
        "x_choke": None,
        "applied_fittings": applied_fittings,
        "dP_fric": dP_fric,
        "dP_acc": dP_acc,
        "dP_elev": dP_elev,
        "dP_fitting": dP_fitting_total,
        "q_seg": q_seg,
        "A_out": A_out,
        "section_transition": section_transition,
    }

    return state2, info


def bisect_for_choke(
    fluid: "FluidEOSBase",
    pipe: "Pipe",
    x_i: float,
    dx_attempted: float,
    state_i: "FluidState",
    mdot: float,
    friction_model: str = "blended",
    mach_choke: float = 0.99,
    min_dx: float = 1e-3,
) -> tuple["FluidState", dict]:
    """Locate the choke point by bisecting segment length.

    Parameters
    ----------
    fluid : FluidEOSBase
    pipe : Pipe
    x_i : float
        Upstream station [m].
    dx_attempted : float
        Full segment length that triggered choke [m].
    state_i : FluidState
        Upstream state.
    mdot : float
        Mass flow [kg/s].
    friction_model : str
    mach_choke : float
    min_dx : float
        Minimum bisection interval [m].

    Returns
    -------
    tuple[FluidState, dict]
        State at choke point and info dict with x_choke and choked=True.
    """
    dx_lo = 0.0
    dx_hi = dx_attempted
    state_lo = state_i
    last_good_state = state_i
    # Pre-populate with all canonical info keys so callers (which read e.g.
    # info["f"]) don't crash if every bisect iteration raises.
    last_good_info: dict = _default_segment_info()

    while dx_hi - dx_lo > min_dx:
        dx_mid = 0.5 * (dx_lo + dx_hi)
        try:
            state_mid, info_mid = solve_segment(
                fluid, pipe, x_i, x_i + dx_mid, state_i, mdot, friction_model, mach_choke
            )
            if info_mid["M_ip1"] >= mach_choke:
                dx_hi = dx_mid
            else:
                dx_lo = dx_mid
                last_good_state = state_mid
                last_good_info = info_mid
        except (ChokeReached, SegmentConvergenceError):
            dx_hi = dx_mid

    x_choke = x_i + dx_lo
    last_good_info["choked_in_segment"] = True
    last_good_info["x_choke"] = x_choke

    return last_good_state, last_good_info

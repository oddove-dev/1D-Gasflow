"""Inline pressure-reducing device model (PSV, orifice, RO, choke valve).

A :class:`Device` sits between two pipe sections. :meth:`Device.solve`
takes ``(state_up, u_up, P_back, A_down)`` and returns a
:class:`DeviceResult` with the throat and Borda-Carnot transition state.
``P_back`` is an input to ``Device.solve`` — when the chain solver
needs to satisfy a system ``mdot`` constraint, it brent-iterates on
``P_back`` externally, calling ``Device.solve`` per probe.

The "PSV on a tank" case (no upstream pipe, ``u_upstream ≈ 0``) goes
through :meth:`Device.from_stagnation`, which is a thin classmethod
wrapper that builds an upstream state with ``u_up = 0`` and dispatches
to ``Device.solve``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import HEMConsistencyError

if TYPE_CHECKING:
    from .eos import FluidState, GERGFluid, ThroatState


@dataclass(frozen=True)
class TransitionResult:
    """State change across the vena-contracta → pipe-inlet expansion.

    Borda-Carnot momentum balance assuming the back-wall pressure equals
    ``P_vc`` (standard sudden-expansion separation assumption).
    Compressibility is handled via mass/energy/momentum closure on the
    downstream static state — no constant-density approximation.

    Parameters
    ----------
    state_inlet : FluidState
        Downstream pipe-inlet static state.
    u_inlet : float
        Pipe-inlet velocity [m/s].
    dP : float
        ``P_inlet - P_vc`` [Pa]. Positive for expansion (pressure recovery).
    eta_dissipation : float
        Dissipation efficiency ``1 - (u_inlet/u_vc)²``. Bounds: → 1 for
        sharp expansion (``A_vc ≪ A_down``, typical PSV); → 0 for marginal
        expansion (``A_vc ≈ A_down``).
    """

    state_inlet: "FluidState"
    u_inlet: float
    dP: float
    eta_dissipation: float


@dataclass(frozen=True)
class DeviceResult:
    """Complete state across a :class:`Device` node.

    Parameters
    ----------
    throat : ThroatState
        Vena-contracta state from the HEM solve.
    transition : TransitionResult
        State change through the Borda-Carnot expansion. When the
        device's HEM throat is choked (``throat.choked == True``) and
        the chain solver enters backward-downstream-march mode (see
        :func:`gas_pipe.chain._chain_forward_march`), this field stays
        attached as a *Borda-Carnot prediction* of the downstream pipe
        inlet under jet-recovery physics — but the **actual** downstream
        pipe inlet state comes from the backward march, set by the
        chain-end ``P_last_cell`` BC. The two values may differ; the
        gap reflects jet-to-pipe-flow mixing that the Borda-Carnot
        momentum balance does not model when the throat is sonic.
        Callers should read the downstream pipe's :class:`PipeResult`
        (with ``march_direction == "backward"``) for the operating
        state, not :attr:`transition.state_inlet`, in that regime.
    eta_dissipation : float
        Mirror of ``transition.eta_dissipation``; surfaced at top level
        as the canonical diagnostic for downstream callers.
    dh_static : float
        ``h_inlet - h_static_up = (u_up² - u_inlet²) / 2`` [J/kg].
        Energy balance is exact across the entire device, so the static
        enthalpy change is purely a kinetic re-partition.
    ds : float
        ``s_inlet - s_up`` [J/(kg·K)]. The HEM throat is isentropic from
        stagnation; all entropy generation lives in the Borda-Carnot
        transition. Always ≥ 0 within numerical noise.
    """

    throat: "ThroatState"
    transition: TransitionResult
    eta_dissipation: float
    dh_static: float
    ds: float


@dataclass(frozen=True)
class Device:
    """Inline pressure-reducing element with ``A_vc = Cd · A_geom``.

    Pure configuration object — stores geometry only. Flow state lives in
    :class:`DeviceResult`, returned by :meth:`solve`.

    Parameters
    ----------
    A_geom : float
        Geometric throat area [m²]. The physical opening (orifice bore,
        PSV nozzle, choke trim flow area).
    Cd : float
        Discharge coefficient [-]. Effective vena-contracta area is
        ``A_vc = Cd · A_geom``. Sharp-edged orifices ≈ 0.6; PSV nozzles
        ≈ 0.95.
    name : str
        Optional label used in diagnostics and error messages.
    """

    A_geom: float
    Cd: float
    name: str = ""

    @property
    def A_vc(self) -> float:
        """Effective vena-contracta area [m²]."""
        return self.Cd * self.A_geom

    def solve(
        self,
        fluid: "GERGFluid",
        state_up: "FluidState",
        u_up: float,
        P_back: float,
        A_down: float,
    ) -> DeviceResult:
        """Solve HEM throat + Borda-Carnot transition for given back pressure.

        Single forward call — no iteration on ``P_back``. The chain
        solver wraps this in a 1-D root-find on ``P_back`` when it needs
        to match a system ``mdot`` constraint.

        Algorithm:
        1. Stagnation state from ``(state_up, u_up)`` via
           :func:`_stagnation_state`.
        2. ``fluid.hem_throat(P_stag, T_stag, P_back, A_vc=self.A_vc)``
           gives the throat state. ``ThroatState.choked`` reflects the
           ``P_back`` vs choke-pressure relation.
        3. :func:`_borda_carnot_transition` for the throat → ``A_down``
           expansion.

        Parameters
        ----------
        fluid : GERGFluid
            Direct EOS. ``Device.solve`` calls ``hem_throat`` and
            ``props_Ps_via_jt`` internally — both ``GERGFluid``-only per
            the Item 3 locked design. When using a ``TabulatedFluid``
            elsewhere, route through ``TabulatedFluid.base_fluid``.
        state_up : FluidState
            Static state at the upstream pipe outlet.
        u_up : float
            Velocity at the upstream pipe outlet [m/s]. Pass 0 for
            PSV-on-tank scenarios.
        P_back : float
            Downstream back pressure [Pa]. Discriminates choked
            vs. subcritical inside ``hem_throat``.
        A_down : float
            Downstream pipe inlet area [m²].

        Returns
        -------
        DeviceResult

        Raises
        ------
        HEMConsistencyError
            If the Borda-Carnot Newton diverges or produces
            ``eta_dissipation`` outside [0, 1].
        ValueError
            If ``A_down <= A_vc`` (Borda-Carnot is expansion only).
        """
        state_stag = _stagnation_state(fluid, state_up, u_up)
        throat = fluid.hem_throat(
            state_stag.P,
            state_stag.T,
            P_back,
            A_vc=self.A_vc,
        )
        transition = _borda_carnot_transition(throat, A_down, fluid)

        u_inlet = transition.u_inlet
        dh_static = transition.state_inlet.h - state_up.h
        ds = transition.state_inlet.s - state_up.s

        return DeviceResult(
            throat=throat,
            transition=transition,
            eta_dissipation=transition.eta_dissipation,
            dh_static=dh_static,
            ds=ds,
        )

    @classmethod
    def from_stagnation(
        cls,
        A_geom: float,
        Cd: float,
        *,
        fluid: "GERGFluid",
        P_stag: float,
        T_stag: float,
        P_back: float,
        A_down: float,
        name: str = "",
    ) -> DeviceResult:
        """PSV-on-tank convenience — solves directly from stagnation BCs.

        Stagnation equals static when ``u_up = 0``, so the upstream state
        is simply ``fluid.props(P_stag, T_stag)`` and the stagnation
        Newton inside :meth:`solve` short-circuits.

        Parameters
        ----------
        A_geom, Cd, name : as in :class:`Device`.
        fluid, P_stag, T_stag, P_back, A_down : as in :meth:`solve`.

        Returns
        -------
        DeviceResult
        """
        device = cls(A_geom=A_geom, Cd=Cd, name=name)
        state_up = fluid.props(P_stag, T_stag)
        return device.solve(
            fluid=fluid,
            state_up=state_up,
            u_up=0.0,
            P_back=P_back,
            A_down=A_down,
        )


def _stagnation_state(
    fluid: "GERGFluid",
    state_up: "FluidState",
    u_up: float,
) -> "FluidState":
    """Isentropic compression from ``(state_up, u_up)`` to zero velocity.

    Solves for state at ``(P_stag, T_stag)`` satisfying
    ``s = state_up.s`` and ``h = state_up.h + u_up²/2``. 1-D Newton on
    ``P_stag`` along the ``s_up`` isentrope using ``props_Ps_via_jt``;
    the isentropic derivative ``(∂h/∂P)|_s = 1/ρ`` gives the Newton
    step.

    Short-circuits to ``state_up`` when ``u_up == 0`` (stagnation equals
    static).
    """
    if u_up == 0.0:
        return state_up

    h_target = state_up.h + 0.5 * u_up * u_up
    h_tol = max(1.0, abs(h_target) * 1e-9)

    # Ideal-gas isentropic compression as initial guess
    gamma = state_up.cp / state_up.cv
    M = u_up / state_up.a
    P_ratio = (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))
    P = state_up.P * P_ratio

    state = state_up
    for _ in range(15):
        state = fluid.props_Ps_via_jt(P, state_up)
        dh = state.h - h_target
        if abs(dh) < h_tol:
            return state
        # Newton step: dh/dP|_s = v = 1/rho
        P -= dh * state.rho
        if P <= 0.0:
            raise HEMConsistencyError(
                "Stagnation Newton drove P_stag non-positive: "
                f"P_up={state_up.P/1e5:.3f} bara, u_up={u_up:.2f} m/s, "
                f"M_up={M:.3f}"
            )
    return state


def _borda_carnot_transition(
    throat: "ThroatState",
    A_down: float,
    fluid: "GERGFluid",
) -> TransitionResult:
    """Borda-Carnot sudden-expansion momentum balance from vena contracta.

    Solves a 2-D Newton on ``(P_in, T_in)`` with finite-difference
    Jacobian. Mass continuity gives ``u_in = mdot/(ρ·A_down)``.

    Residuals:

    - Energy: ``R_h = h(P_in, T_in) + u_in²/2 - h_stag`` with
      ``h_stag = throat.h + throat.u²/2``.
    - Momentum: ``R_p = P_in - P_t - ρ_t·u_t²·(A_vc/A_down) + ρ_in·u_in²``.

    Initial guess uses the constant-density approximation. Convergence
    is on the L2 norm of residuals scaled by ``(P_stag, u_throat²)``;
    tolerance 1e-8, cap 20 iterations.

    Raises
    ------
    ValueError
        If ``A_down <= throat.A_vc`` (contraction, not expansion).
    HEMConsistencyError
        If Newton diverges, hits non-positive P or T, or yields
        ``eta_dissipation`` outside [0, 1].
    """
    A_vc = throat.A_vc
    if A_down <= A_vc:
        raise ValueError(
            "Borda-Carnot transition requires A_down > A_vc (expansion only); "
            f"got A_down={A_down:.4e} m², A_vc={A_vc:.4e} m²"
        )

    state_t = fluid.props(throat.P, throat.T)
    u_t = throat.u
    rho_t = throat.rho
    P_t = throat.P
    mdot = throat.mdot

    h_stag = state_t.h + 0.5 * u_t * u_t
    momentum_constant = P_t + rho_t * u_t * u_t * (A_vc / A_down)

    # Constant-density initial guess
    u_in_guess = u_t * A_vc / A_down  # mass conservation with ρ unchanged
    P_in_guess = momentum_constant - rho_t * u_in_guess * u_in_guess
    P_in = max(P_in_guess, 0.5 * P_t)
    T_in = throat.T

    # Scaling for residual norm
    P_scale = max(P_t, 1.0)
    u_scale_sq = max(u_t * u_t, 1.0)

    # Bypass the GERGFluid cache during Newton. The cache rounds (P, T) to
    # bins (100 Pa, 1 mK) and returns the FluidState computed at the exact
    # (P, T) where the bin was first populated — not interpolated. That's
    # fine for the pipe march (each station has a unique (P, T)) but it
    # corrupts a Newton FD-Jacobian, where consecutive eval points can
    # land back in already-populated bins and silently fetch stale state.
    def _eval(P: float, T: float):  # type: ignore[no-untyped-def]
        return fluid._eval_state(P, T)

    def _residuals(P: float, T: float) -> tuple[float, float, "FluidState", float]:
        st = _eval(P, T)
        u = mdot / (st.rho * A_down)
        R_h = st.h + 0.5 * u * u - h_stag
        R_p = P - momentum_constant + st.rho * u * u
        return R_h, R_p, st, u

    state_in = state_t
    u_in = u_in_guess
    norm = float("inf")
    # Tolerance 1e-7 on scale-normalized norm corresponds to:
    #   R_h ≲ 1e-7 · u_throat² (typically <10 mJ/kg)
    #   R_p ≲ 1e-7 · P_stag    (typically <1 Pa)
    # Both well inside engineering precision. Tighter than 1e-7 hits the
    # noise floor of the FD Jacobian when GERGFluid's property cache
    # gets revisited mid-iteration.
    for iteration in range(30):
        R_h, R_p, state_in, u_in = _residuals(P_in, T_in)
        norm = math.sqrt((R_h / u_scale_sq) ** 2 + (R_p / P_scale) ** 2)
        if norm < 1e-7:
            break

        # FD steps must exceed GERGFluid's cache bin widths (100 Pa, 1 mK) so
        # props(P, T) and props(P+eps_P, T) hit distinct cache cells and
        # return independently-flashed states. Below that, both fetches return
        # the same FluidState and the FD Jacobian degenerates to zero.
        eps_P = max(1e-4 * P_scale, 1000.0)
        eps_T = max(1e-4 * T_in, 0.01)

        R_h_P, R_p_P, _, _ = _residuals(P_in + eps_P, T_in)
        R_h_T, R_p_T, _, _ = _residuals(P_in, T_in + eps_T)

        dRh_dP = (R_h_P - R_h) / eps_P
        dRh_dT = (R_h_T - R_h) / eps_T
        dRp_dP = (R_p_P - R_p) / eps_P
        dRp_dT = (R_p_T - R_p) / eps_T

        det = dRh_dP * dRp_dT - dRh_dT * dRp_dP
        if abs(det) < 1e-30:
            raise HEMConsistencyError(
                "Singular Jacobian in Borda-Carnot Newton at "
                f"P_in={P_in/1e5:.3f} bara, T_in={T_in:.2f} K (det={det:.2e})"
            )

        dP = (-R_h * dRp_dT + R_p * dRh_dT) / det
        dT = (-R_p * dRh_dP + R_h * dRp_dP) / det

        P_in += dP
        T_in += dT

        if P_in <= 0.0 or T_in <= 0.0:
            raise HEMConsistencyError(
                "Borda-Carnot Newton drove P or T non-positive: "
                f"P_in={P_in/1e5:.3f} bara, T_in={T_in:.2f} K after iter {iteration}"
            )
    else:
        raise HEMConsistencyError(
            f"Borda-Carnot Newton did not converge in 30 iterations; "
            f"last norm={norm:.2e} at P_in={P_in/1e5:.3f} bara, "
            f"T_in={T_in:.2f} K (throat P={P_t/1e5:.3f} bara, "
            f"u_throat={u_t:.2f} m/s, A_vc/A_down={A_vc/A_down:.4f})"
        )

    eta = 1.0 - (u_in / u_t) ** 2
    if eta < -1e-6 or eta > 1.0 + 1e-6:
        raise HEMConsistencyError(
            f"Borda-Carnot eta_dissipation={eta:.4f} outside [0, 1] at "
            f"A_vc/A_down={A_vc/A_down:.4f}, u_inlet/u_throat="
            f"{u_in/u_t:.4f}"
        )
    eta = max(0.0, min(1.0, eta))

    dP = P_in - P_t
    return TransitionResult(
        state_inlet=state_in,
        u_inlet=u_in,
        dP=dP,
        eta_dissipation=eta,
    )

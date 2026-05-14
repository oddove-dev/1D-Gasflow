"""Inline pressure-reducing device model (PSV, orifice, RO, choke valve).

A :class:`Device` sits between two pipe sections. Solving it requires:

1. Stagnation from the upstream pipe outlet
   (``h_stag = h_static + u²/2``, isentropic compression to zero velocity).
2. HEM throat solve at ``A_vc = Cd · A_geom`` with the system ``mdot``
   fixed by the chain BVP. Internally calls
   :meth:`GERGFluid.hem_throat` in mode A, brent-iterating on ``P_back``
   until the resulting throat mass flow matches ``mdot``.
3. Borda-Carnot momentum balance from the vena contracta to the
   downstream pipe inlet (sudden expansion to ``A_down``).

The "PSV on a tank" case (no upstream pipe, ``u_upstream ≈ 0``) goes
through :meth:`Device.from_stagnation` rather than the inline solve path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        State change through the Borda-Carnot expansion.
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
        mdot: float,
        A_down: float,
    ) -> DeviceResult:
        """Solve HEM throat + Borda-Carnot transition for an inline device.

        Algorithm (filled in commit 2):

        1. Stagnation: ``h_stag = state_up.h + u_up²/2``; isentropic
           compression from ``state_up`` to zero velocity yields
           ``(P_stag, T_stag)`` via a Newton on ``P_stag`` reusing
           :meth:`GERGFluid.props_Ps_via_jt`.
        2. Choke probe: call ``fluid.hem_throat(P_stag, T_stag,
           P_back=ε·P_stag, A_vc=self.A_vc)`` to get ``G_max`` and
           ``mdot_choke = G_max · A_vc``.
        3. If ``mdot > mdot_choke``: raise :class:`OverChokedError`
           with diagnostic fields.
        4. If ``mdot == mdot_choke`` (within tolerance): throat is the
           choke probe state.
        5. Else: brent on ``P_back ∈ (P_choke, P_stag)`` so that
           ``hem_throat(P_stag, T_stag, P_back, A_vc=self.A_vc).mdot
           == mdot``. Sub-choked branch is monotone, so brent converges
           cleanly.
        6. :func:`_borda_carnot_transition` for the throat → ``A_down``
           expansion, returning :class:`TransitionResult`.

        Parameters
        ----------
        fluid : GERGFluid
            EOS — the HEM throat solve uses
            :meth:`GERGFluid.hem_throat` and :meth:`props_Ps_via_jt`
            internally, so a direct ``GERGFluid`` is required (not a
            ``TabulatedFluid``, since ``hem_throat`` is GERGFluid-only
            per the Item 3 locked design).
        state_up : FluidState
            Static state at the upstream pipe outlet.
        u_up : float
            Velocity at the upstream pipe outlet [m/s].
        mdot : float
            System mass flow rate [kg/s], fixed by the chain BVP.
        A_down : float
            Downstream pipe inlet area [m²].

        Returns
        -------
        DeviceResult

        Raises
        ------
        OverChokedError
            If ``mdot > G_max(P_stag, T_stag) · A_vc``.
        """
        raise NotImplementedError("Device.solve: filled in commit 2")

    @classmethod
    def from_stagnation(
        cls,
        A_geom: float,
        Cd: float,
        *,
        fluid: "GERGFluid",
        P_stag: float,
        T_stag: float,
        mdot: float,
        A_down: float,
        name: str = "",
    ) -> DeviceResult:
        """PSV-on-tank convenience — solves directly from stagnation BCs.

        Builds a :class:`Device` with ``(A_geom, Cd, name)`` and calls
        :meth:`solve` with ``u_up=0`` and
        ``state_up = fluid.props(P_stag, T_stag)``. For ``u_up=0`` the
        static state and the stagnation state coincide, so the stagnation
        Newton in step 1 of :meth:`solve` collapses to a single
        ``fluid.props`` call.

        Returns
        -------
        DeviceResult
            Same shape as :meth:`solve`.
        """
        raise NotImplementedError("Device.from_stagnation: filled in commit 2")


def _borda_carnot_transition(
    fluid: "GERGFluid",
    throat: "ThroatState",
    mdot: float,
    A_down: float,
) -> TransitionResult:
    """Borda-Carnot momentum balance from vena contracta to ``A_down``.

    Solves a 2-D Newton on ``(P_inlet, T_inlet)`` with finite-difference
    Jacobian. Residuals:

    - **Energy** ``R_h = h(P_in, T_in) + u_in²/2 - h_stag``
      with ``h_stag = throat.h + throat.u²/2``.
    - **Momentum** ``R_p = P_in - P_vc - ρ_vc·u_vc²·(A_vc/A_down)
      + ρ_in·u_in²`` derived from the standard Borda-Carnot control
      volume assuming back-wall pressure ``= P_vc``.

    Mass continuity gives ``u_in = mdot / (ρ(P_in, T_in) · A_down)``.
    Initial guess uses the constant-density Borda-Carnot result. Converges
    in 3-5 iterations for typical expansion ratios.
    """
    raise NotImplementedError("_borda_carnot_transition: filled in commit 2")

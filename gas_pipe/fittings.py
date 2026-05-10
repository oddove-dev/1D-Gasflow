"""Discrete loss-factor fittings for inline pressure drops.

Fittings are applied isenthalpically after the friction segment Newton solver
converges. Temperature change from JT cooling is implicit in the EOS call.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fitting:
    """Generic K-factor loss element at a discrete pipe location.

    The pressure loss is applied isenthalpically:
      ΔP = K · ρ_upstream · u_upstream² / 2

    Temperature change consistent with JT cooling is captured via
    props_Ph(P_out, h_upstream) in the segment solver.

    Parameters
    ----------
    location : float
        Axial position along pipe [m].
    K : float
        Loss coefficient (dimensionless).
    name : str
        Optional label for reporting.
    """

    location: float
    K: float
    name: str = ""

    def pressure_loss(self, state_up: object, u_up: float) -> float:
        """Compute pressure loss for this fitting.

        Parameters
        ----------
        state_up : FluidState
            Upstream thermodynamic state (uses rho).
        u_up : float
            Upstream velocity [m/s].

        Returns
        -------
        float
            Pressure loss [Pa] (positive = pressure decrease).
        """
        return self.K * 0.5 * state_up.rho * u_up ** 2

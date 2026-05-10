"""Custom exception classes for the gas pipe solver."""
from __future__ import annotations


class GasPipeError(Exception):
    pass


class EOSOutOfRange(GasPipeError):
    pass


class EOSTwoPhase(GasPipeError):
    pass


class SegmentConvergenceError(GasPipeError):
    pass


class IntegrationCapExceeded(SegmentConvergenceError):
    """march_ivp ran out of segment budget — numerical failure, not choke.

    Distinct from per-segment Newton divergence (which can legitimately
    hint at near-sonic conditions); this one means adaptive refinement
    blew through the safety cap and the integration must be considered
    failed, not "choked."
    """
    pass


class SolverCancelled(GasPipeError):
    """Raised when the user-set cancel_event is observed at a check-point.

    The solver is responsible for unwinding cleanly; the caller (typically
    a worker thread in the GUI) is responsible for surfacing this to the
    user as 'Cancelled' rather than 'Error'.
    """
    pass


class ChokeReached(GasPipeError):
    """Internal control flow — not user-facing."""
    pass


class BVPNotBracketedError(GasPipeError):
    pass


class BVPChoked(GasPipeError):
    """Raised when BVP target P_out is unreachable due to choking.

    Carries the choked solution and the critical mass flow rate.
    """

    def __init__(self, message: str, mdot_critical: float, result: object) -> None:
        super().__init__(message)
        self.mdot_critical = mdot_critical
        self.result = result

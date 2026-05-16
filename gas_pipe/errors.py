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
    """Raised when BVP target ``P_last_cell`` is unreachable due to choking.

    Carries the choked solution and the critical mass flow rate.
    """

    def __init__(self, message: str, mdot_critical: float, result: object) -> None:
        super().__init__(message)
        self.mdot_critical = mdot_critical
        self.result = result


class HEMConsistencyError(GasPipeError):
    """u_vc deviates from c_HEM by >5% at HEM throat convergence.

    Indicates G_max iteration converged to a non-physical local maximum,
    or HEOS Ps-flash drifted significantly, or HEM is not applicable in
    this thermodynamic region. The returned ThroatState cannot be trusted.
    """
    pass


class HEMConsistencyWarning(UserWarning):
    """u_vc deviates from c_HEM by 1-5% at HEM throat convergence.

    Result is usable but indicates numerical noise on the c_HEM finite
    difference or marginal Ps-flash convergence. Worth investigating if
    it recurs. Subclasses UserWarning (not GasPipeError) because
    warnings.warn requires a Warning ancestry.
    """
    pass


class OverChokedError(GasPipeError):
    """Raised by Device.solve when mdot exceeds A_vc · G_max at (P_stag, T_stag).

    Indicates the device cannot pass the requested mass flow at the
    current stagnation conditions and effective area. Diagnostic fields
    let the chain solver map the failure to a Mode 1 ``BVPChoked`` (with
    ``mdot_critical = max_mdot``) or surface a Mode 3 infeasibility
    message identifying the bracket exhaustion.

    ``device_index`` defaults to ``None`` — ``Device.solve`` does not know
    its own position in a chain. The chain march fills it in when
    re-raising.
    """

    def __init__(
        self,
        message: str,
        *,
        device_index: int | None = None,
        device_name: str = "",
        max_mdot: float,
        attempted_mdot: float,
        P_stag: float,
        T_stag: float,
    ) -> None:
        super().__init__(message)
        self.device_index = device_index
        self.device_name = device_name
        self.max_mdot = max_mdot
        self.attempted_mdot = attempted_mdot
        self.P_stag = P_stag
        self.T_stag = T_stag

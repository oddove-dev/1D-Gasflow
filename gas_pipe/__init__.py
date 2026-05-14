"""Gas Pipe Analyzer — single-phase 1D steady-state gas flow solver.

Public API:

    from gas_pipe import GERGFluid, Pipe, march_ivp, solve_for_mdot
    from gas_pipe.errors import BVPChoked, BVPNotBracketedError
    from gas_pipe.eos import ALLOWED_COMPONENTS
"""
from __future__ import annotations

from .eos import (
    ALLOWED_COMPONENTS,
    FluidEOSBase,
    FluidState,
    GERGFluid,
    TabulatedFluid,
    ThroatState,
    estimate_operating_window,
)
from .errors import (
    BVPChoked,
    BVPNotBracketedError,
    ChokeReached,
    EOSOutOfRange,
    EOSTwoPhase,
    GasPipeError,
    HEMConsistencyError,
    HEMConsistencyWarning,
    IntegrationCapExceeded,
    SegmentConvergenceError,
    SolverCancelled,
)
from .fittings import Fitting
from .geometry import Pipe, PipeSection
from .results import PipeResult
from .solver import (
    initial_n_segments,
    march_ivp,
    plateau_sweep,
    solve_for_mdot,
    verify_eos_accuracy,
)

__all__ = [
    "GERGFluid",
    "TabulatedFluid",
    "FluidEOSBase",
    "FluidState",
    "ThroatState",
    "ALLOWED_COMPONENTS",
    "estimate_operating_window",
    "Pipe",
    "PipeSection",
    "Fitting",
    "PipeResult",
    "march_ivp",
    "solve_for_mdot",
    "plateau_sweep",
    "verify_eos_accuracy",
    "initial_n_segments",
    "GasPipeError",
    "EOSOutOfRange",
    "EOSTwoPhase",
    "SegmentConvergenceError",
    "ChokeReached",
    "BVPChoked",
    "BVPNotBracketedError",
    "HEMConsistencyError",
    "HEMConsistencyWarning",
    "IntegrationCapExceeded",
    "SolverCancelled",
]

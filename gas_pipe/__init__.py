"""Gas Pipe Analyzer — single-phase 1D steady-state gas flow solver.

Public API:

    from gas_pipe import GERGFluid, Pipe, march_ivp, solve_for_mdot
    from gas_pipe.errors import BVPChoked, BVPNotBracketedError
    from gas_pipe.eos import ALLOWED_COMPONENTS
"""
from __future__ import annotations

from .eos import ALLOWED_COMPONENTS, FluidState, GERGFluid
from .errors import (
    BVPChoked,
    BVPNotBracketedError,
    ChokeReached,
    EOSOutOfRange,
    EOSTwoPhase,
    GasPipeError,
    IntegrationCapExceeded,
    SegmentConvergenceError,
    SolverCancelled,
)
from .fittings import Fitting
from .geometry import Pipe, PipeSection
from .results import PipeResult
from .solver import march_ivp, plateau_sweep, solve_for_mdot

__all__ = [
    "GERGFluid",
    "FluidState",
    "ALLOWED_COMPONENTS",
    "Pipe",
    "PipeSection",
    "Fitting",
    "PipeResult",
    "march_ivp",
    "solve_for_mdot",
    "plateau_sweep",
    "GasPipeError",
    "EOSOutOfRange",
    "EOSTwoPhase",
    "SegmentConvergenceError",
    "ChokeReached",
    "BVPChoked",
    "BVPNotBracketedError",
    "IntegrationCapExceeded",
    "SolverCancelled",
]

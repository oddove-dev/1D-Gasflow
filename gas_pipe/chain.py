"""Multi-element BVP solver: chains of :class:`Pipe` and :class:`Device`.

A :class:`ChainSpec` is a sequence ``Pipe → [Device → Pipe]*`` from
inlet to outlet. :func:`solve_chain` is a three-mode dispatcher: exactly
two of ``(P_in, P_last_cell, mdot)`` must be given; ``T_in`` is always
required because temperature propagates forward.

``P_last_cell`` is the pressure in the last cell of the last pipe — a
per-pipe computed quantity, not a chain-level downstream BC. The name
``P_out`` is reserved for the future chain-level post-expansion BC; see
the "Pressure terminology" section of ``CLAUDE.md``.

For the PSV-on-tank scenario (no upstream pipe) call
:meth:`Device.from_stagnation` directly — that's a single-device
calculation, not a chain.
"""
from __future__ import annotations

import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Union

from scipy.optimize import brentq

from .device import Device, DeviceResult
from .errors import (
    BVPChoked,
    BVPNotBracketedError,
    ChokeReached,
    DeviceMustGoSubcritical,
    EOSOutOfRange,
    EOSTwoPhase,
    HEMConsistencyError,
    IntegrationCapExceeded,
    OverChokedError,
    SegmentConvergenceError,
)

if TYPE_CHECKING:
    from .eos import FluidEOSBase, GERGFluid, TabulatedFluid
    from .geometry import Pipe
    from .results import PipeResult


logger = logging.getLogger(__name__)

ChainElement = Union["Pipe", Device]
"""Type alias for the elements a :class:`ChainSpec` may contain."""


# ---------------------------------------------------------------------------
# ChainSpec / ChainResult
# ---------------------------------------------------------------------------


@dataclass
class ChainSpec:
    """Ordered sequence of pipe and device elements from inlet to outlet.

    Invariants enforced in :meth:`__post_init__`:

    - At least one element.
    - First and last elements are :class:`Pipe`. PSV-on-tank goes through
      :meth:`Device.from_stagnation` instead of the chain solver.
    - No two adjacent :class:`Device` elements (would imply a missing pipe
      between them).
    """

    elements: list[ChainElement]

    def __post_init__(self) -> None:
        if not self.elements:
            raise ValueError("ChainSpec must have at least one element.")

        from .geometry import Pipe

        first = self.elements[0]
        last = self.elements[-1]
        if not isinstance(first, Pipe):
            raise ValueError(
                "ChainSpec must start with a Pipe element; use "
                "Device.from_stagnation for the PSV-on-tank case."
            )
        if not isinstance(last, Pipe):
            raise ValueError(
                "ChainSpec must end with a Pipe element."
            )

        prev_was_device = False
        for i, el in enumerate(self.elements):
            is_device = isinstance(el, Device)
            is_pipe = isinstance(el, Pipe)
            if not (is_device or is_pipe):
                raise ValueError(
                    f"ChainSpec element {i} is not Pipe or Device: "
                    f"{type(el).__name__}"
                )
            if is_device and prev_was_device:
                raise ValueError(
                    f"ChainSpec elements {i - 1} and {i} are both Devices; "
                    "insert a Pipe between them."
                )
            prev_was_device = is_device

    @property
    def pipes(self) -> list["Pipe"]:
        from .geometry import Pipe

        return [el for el in self.elements if isinstance(el, Pipe)]

    @property
    def devices(self) -> list[Device]:
        return [el for el in self.elements if isinstance(el, Device)]


@dataclass
class ChainResult:
    """Result of :func:`solve_chain` — per-element states + chain summary.

    ``P_in`` is the chain-level upstream BC (interpretation depends on the
    first element: ``P_first_cell`` of pipe 1 when the chain starts with
    Pipe; stagnation upstream of the device when it starts with Device —
    not yet reachable, the latter is future work).

    ``P_last_cell`` is the pressure in the last cell of the last pipe — a
    computed quantity, never an input. ``P_out`` is the reservoir
    pressure the chain discharges into (the downstream BC). When the
    outlet is subsonic ``P_last_cell == P_out`` within solver tolerance;
    when the outlet is sonic (``outlet_sonic``) ``P_last_cell > P_out``
    and the difference is taken up by an under-expanded free jet outside
    the pipe (modelled in a later layer). See CLAUDE.md "Pressure
    terminology".
    """

    chain: ChainSpec
    results: list[Union["PipeResult", DeviceResult]]
    mdot: float
    P_in: float
    P_last_cell: float
    T_in: float
    T_out: float
    boundary_conditions: dict
    solver_options: dict = field(default_factory=dict)
    elapsed: float = 0.0
    choked: bool = False
    choke_diagnostics: dict | None = None
    outlet_sonic: bool = False
    P_out: float | None = None

    @property
    def pipe_results(self) -> list["PipeResult"]:
        from .results import PipeResult

        return [r for r in self.results if isinstance(r, PipeResult)]

    @property
    def device_results(self) -> list[DeviceResult]:
        return [r for r in self.results if isinstance(r, DeviceResult)]


# ---------------------------------------------------------------------------
# Fluid dispatch
# ---------------------------------------------------------------------------


def _get_base_fluid(fluid: "FluidEOSBase") -> "GERGFluid":
    """Return the underlying :class:`GERGFluid` for HEM / Ps-flash calls.

    ``Device.solve`` requires a direct ``GERGFluid`` (HEM throat and
    ``props_Ps_via_jt`` live there per the Item 3 locked design). The
    pipe march can use either, so ``solve_chain`` keeps both handles and
    routes per call site.
    """
    from .eos import GERGFluid, TabulatedFluid

    if isinstance(fluid, TabulatedFluid):
        return fluid.base_fluid
    if isinstance(fluid, GERGFluid):
        return fluid
    raise TypeError(
        "solve_chain requires GERGFluid or TabulatedFluid; "
        f"got {type(fluid).__name__}"
    )


# ---------------------------------------------------------------------------
# Per-device mdot-matched solve (P_back outer iteration)
# ---------------------------------------------------------------------------


def _device_solve_at_mdot(
    device: Device,
    state_up: "object",
    u_up: float,
    mdot: float,
    A_down: float,
    fluid: "GERGFluid",
    element_index: int,
) -> DeviceResult:
    """Find ``P_back`` so the device's HEM throat passes ``mdot``.

    Calls :meth:`GERGFluid.hem_throat` in mode A repeatedly with
    ``A_vc = device.A_vc``. Sub-choked branch ``G(P_back)`` is monotone
    decreasing in ``P_back`` (between ``P_choke`` and ``P_stag``), so
    a 1-D ``brentq`` converges cleanly.

    Raises :class:`OverChokedError` if ``mdot > G_max · A_vc`` at the
    stagnation conditions, with ``device_index = element_index`` filled
    in so chain-level error handling can locate the source.
    """
    from .device import _stagnation_state

    state_stag = _stagnation_state(fluid, state_up, u_up)
    P_stag = state_stag.P
    T_stag = state_stag.T

    # Probe choke at a low P_back to obtain G_max and P_choke from the same
    # call. hem_throat's bracket starts at 0.05·P_stag (Item 3), so going
    # below that doesn't help — use the same lower bound.
    ts_choke = fluid.hem_throat(
        P_stag, T_stag, 0.05 * P_stag, A_vc=device.A_vc,
    )
    mdot_max = ts_choke.mdot  # = G_max · A_vc

    # Tolerance: 0.01% of choke-limited mdot. HEM throat's internal Brent
    # has xatol = 1e-5·P_stag which translates to ~1e-4 relative error in G.
    rel_tol = 1e-4
    if mdot > mdot_max * (1.0 + rel_tol):
        raise OverChokedError(
            f"Device '{device.name or '?'}' (chain index {element_index}) "
            f"cannot pass mdot={mdot:.4f} kg/s; max at stagnation "
            f"(P_stag={P_stag/1e5:.3f} bara, T_stag={T_stag:.2f} K) is "
            f"{mdot_max:.4f} kg/s.",
            device_index=element_index,
            device_name=device.name,
            max_mdot=mdot_max,
            attempted_mdot=mdot,
            P_stag=P_stag,
            T_stag=T_stag,
        )

    if mdot >= mdot_max * (1.0 - rel_tol):
        # At or just below choke — use the choke-probe throat directly.
        P_back_resolved = ts_choke.P  # = P_choke from the G-max search
    else:
        # Sub-choked: brent on P_back ∈ (P_choke, ~P_stag).
        # f(P_back) = throat.mdot(P_back) - mdot.
        # At P_back=P_choke: f = mdot_max - mdot > 0.
        # At P_back→P_stag: throat.u → 0 → mdot → 0, so f → -mdot < 0.
        P_choke = ts_choke.P

        def _f(P_back: float) -> float:
            ts = fluid.hem_throat(
                P_stag, T_stag, P_back, A_vc=device.A_vc,
            )
            return ts.mdot - mdot

        try:
            P_back_resolved = brentq(
                _f,
                P_choke,
                P_stag * 0.99,
                xtol=1e-5 * P_stag,
                maxiter=50,
            )
        except ValueError as exc:
            raise OverChokedError(
                f"Device '{device.name or '?'}' (chain index {element_index}) "
                f"could not match mdot={mdot:.4f} kg/s via brent on P_back "
                f"in [{P_choke/1e5:.3f}, {P_stag*0.99/1e5:.3f}] bara: {exc}",
                device_index=element_index,
                device_name=device.name,
                max_mdot=mdot_max,
                attempted_mdot=mdot,
                P_stag=P_stag,
                T_stag=T_stag,
            ) from exc

    return device.solve(
        fluid=fluid,
        state_up=state_up,
        u_up=u_up,
        P_back=P_back_resolved,
        A_down=A_down,
    )


# ---------------------------------------------------------------------------
# Chain forward march
# ---------------------------------------------------------------------------


def _chain_forward_march(
    chain: ChainSpec,
    working_fluid: "FluidEOSBase",
    base_fluid: "GERGFluid",
    P_in: float,
    T_in: float,
    mdot: float,
    *,
    boundary_conditions: dict,
    solver_options: dict,
    t0: float,
    cancel_event: object | None = None,
    P_last_cell_target: float | None = None,
    enable_backward_downstream: bool = False,
    **ivp_kwargs,
) -> ChainResult:
    """Forward chain march at fixed ``mdot``, returning a complete
    :class:`ChainResult`.

    For each pipe element, calls :func:`march_ivp` with the current state.
    For each device, calls :func:`_device_solve_at_mdot` which itself
    brent-iterates ``P_back`` so the device's HEM throat passes the
    system ``mdot``.

    When ``enable_backward_downstream`` is True and ``P_last_cell_target``
    is provided, a Device whose HEM throat chokes triggers a second
    pass: every Pipe element downstream of the choked Device is
    backward-marched via :func:`march_ivp_backward` from
    ``P_last_cell_target`` (chain BC) inward, using the chain-inlet
    stagnation enthalpy as the conserved h_stag. The Borda-Carnot
    transition predicted by ``_device_solve_at_mdot`` stays attached
    to ``DeviceResult.transition`` as a diagnostic but no longer seeds
    the downstream pipe — see CLAUDE.md "Pressure terminology" for
    the physical justification.

    Raises
    ------
    OverChokedError
        Propagated from a device that cannot pass ``mdot``. The chain
        BVP modes wrap this in their respective sentinel/mapping logic.
    ChokeReached, IntegrationCapExceeded, SegmentConvergenceError
        Propagated from per-pipe ``march_ivp``.
    BackwardMarchDiabaticNotSupported
        Raised when backward downstream mode is active and a downstream
        Pipe has any section with ``overall_U > 0`` (v1 scope).
    BVPChoked
        Raised when a per-pipe sanity check on the backward march
        fails: ``P_first_cell > P_throat`` (physically inconsistent) or
        ``M_first_cell > 0.7`` (outside v1 numerical convergence
        envelope). The accompanying ``result`` is a partial
        :class:`ChainResult` carrying the elements solved so far.
    """
    from .geometry import Pipe
    from .results import PipeResult
    from .solver import march_ivp, march_ivp_backward
    from .errors import BackwardMarchDiabaticNotSupported

    results: list[Union[PipeResult, DeviceResult]] = []
    P_current = P_in
    T_current = T_in
    choked = False
    choke_diagnostics: dict | None = None

    backward_mode_active = False
    backward_start_index: int | None = None
    choked_device_throat = None  # ThroatState of the choke-triggering device
    h_stag_chain: float | None = None

    # Precompute chain-inlet h_stag in case backward mode activates.
    # Adiabatic h_stag conservation (pipe 1 → device → pipe 2) is the
    # invariant march_ivp_backward relies on; we propagate the chain
    # inlet's value through to every backward-marched pipe.
    if enable_backward_downstream and P_last_cell_target is not None:
        first_pipe = chain.pipes[0] if chain.pipes else None
        if first_pipe is not None:
            state_inlet = base_fluid.props(P_in, T_in)
            A_inlet = first_pipe.sections[0].area
            u_inlet = mdot / (state_inlet.rho * A_inlet)
            h_stag_chain = state_inlet.h + 0.5 * u_inlet ** 2

    for i, element in enumerate(chain.elements):
        if backward_mode_active:
            # Subsequent elements are handled in the backward-march pass
            # below; skip them in the forward pass.
            break

        if isinstance(element, Pipe):
            pipe_result = march_ivp(
                element, working_fluid, P_current, T_current, mdot,
                cancel_event=cancel_event,
                **ivp_kwargs,
            )
            results.append(pipe_result)
            P_current = float(pipe_result.P[-1])
            T_current = float(pipe_result.T[-1])
            if pipe_result.choked:
                choked = True
                choke_diagnostics = {
                    "kind": "fanno_pipe",
                    "element_index": i,
                    "element_name": "",
                    "x_choke": pipe_result.x_choke,
                }
                # A choked pipe terminates the march — downstream elements
                # cannot be reached at this mdot. Backward march for the
                # Fanno-bottleneck case is v2 scope (the choke happens
                # mid-pipe with unmodeled downstream-of-choke physics).
                break
        elif isinstance(element, Device):
            prev = results[-1] if results else None
            if not isinstance(prev, PipeResult):
                raise RuntimeError(
                    f"Device at chain index {i} has no upstream PipeResult "
                    "(ChainSpec validation should have caught this)."
                )
            u_current = float(prev.u[-1])
            state_up = base_fluid.props(P_current, T_current)

            next_el = chain.elements[i + 1]
            if not isinstance(next_el, Pipe):
                raise RuntimeError(
                    f"Device at chain index {i} not followed by a Pipe "
                    "(ChainSpec validation should have caught this)."
                )
            A_down = next_el.sections[0].area

            device_result = _device_solve_at_mdot(
                element, state_up, u_current, mdot, A_down,
                fluid=base_fluid, element_index=i,
            )
            results.append(device_result)

            if (
                enable_backward_downstream
                and P_last_cell_target is not None
                and h_stag_chain is not None
                and (
                    device_result.throat.choked
                    or device_result.throat.M >= 0.95
                )
            ):
                # Switch to backward-downstream mode. The transition
                # state attached to device_result remains as a diagnostic
                # (what Borda-Carnot would have predicted); the next
                # Pipe's actual inlet state will come from the backward
                # march below.
                backward_mode_active = True
                backward_start_index = i + 1
                choked_device_throat = device_result.throat
                choked = True
                choke_diagnostics = {
                    "kind": "device_choke_downstream_backward",
                    "element_index": i,
                    "element_name": element.name,
                    "max_mdot": float(device_result.throat.mdot),
                    "P_throat": float(device_result.throat.P),
                    "T_throat": float(device_result.throat.T),
                }
            else:
                inlet = device_result.transition.state_inlet
                P_current = inlet.P
                T_current = inlet.T
        else:
            raise TypeError(
                f"Unknown ChainSpec element at index {i}: "
                f"{type(element).__name__}"
            )

    # --------------------------------------------------------------
    # Backward-downstream pass.
    # --------------------------------------------------------------
    if backward_mode_active:
        assert backward_start_index is not None
        assert choked_device_throat is not None
        assert h_stag_chain is not None
        assert P_last_cell_target is not None

        # Collect downstream elements; v1 supports only Pipes here
        # (multi-Device chains are deferred — see ChainSpec invariants
        # and BACKLOG).
        downstream = list(
            enumerate(chain.elements)
        )[backward_start_index:]
        for idx, el in downstream:
            if isinstance(el, Device):
                raise NotImplementedError(
                    "v1 backward downstream march does not support "
                    "additional Devices in the choked-downstream chain; "
                    f"element {idx} is a Device. Multi-Device chains "
                    "are deferred to v2."
                )

        downstream_pipes = [
            (idx, el) for idx, el in downstream if isinstance(el, Pipe)
        ]
        if not downstream_pipes:
            raise RuntimeError(
                "device throat choked but no downstream Pipe in chain "
                "(ChainSpec validation should have caught this)."
            )

        # Adiabatic precondition: every downstream pipe section must
        # have overall_U == 0 (h_stag invariance assumption). Surface
        # the offending chain index in the error so the user can find
        # it quickly.
        for idx, pipe in downstream_pipes:
            for sec_idx, sec in enumerate(pipe.sections):
                if sec.overall_U != 0.0:
                    raise BackwardMarchDiabaticNotSupported(
                        f"Pipe at chain index {idx}, section {sec_idx} "
                        f"has overall_U = {sec.overall_U} W/(m²·K). "
                        "Backward downstream march requires adiabatic "
                        "conditions in v1. Either set overall_U = 0 on "
                        "this section, or wait for the diabatic "
                        "extension (forward-T-backward-P iteration)."
                    )

        # Strip ivp_kwargs to the subset march_ivp_backward accepts.
        # The backward primitive has a narrower signature than march_ivp
        # (no adaptive flag, no Mach thresholds, no min_dx).
        bwd_kwargs = {}
        if "n_segments" in ivp_kwargs:
            bwd_kwargs["n_segments"] = int(ivp_kwargs["n_segments"])
        if "friction_model" in ivp_kwargs:
            bwd_kwargs["friction_model"] = ivp_kwargs["friction_model"]

        # Sanity check thresholds.
        P_throat = float(choked_device_throat.P)
        rtol_sanity = 1e-3

        # Backward march from the chain end inward. Each pipe's inlet
        # pressure becomes the next (upstream) pipe's outlet BC.
        backward_results: list[tuple[int, "PipeResult"]] = []
        outlet_P = float(P_last_cell_target)
        for idx, pipe in reversed(downstream_pipes):
            pipe_result_bwd = march_ivp_backward(
                pipe, working_fluid,
                P_outlet=outlet_P,
                h_stag=h_stag_chain,
                mdot=mdot,
                cancel_event=cancel_event,
                **bwd_kwargs,
            )

            P_first_cell = float(pipe_result_bwd.P[0])
            M_first_cell = float(pipe_result_bwd.M[0])

            # Regime transition (NOT an error): P_first_cell exceeding
            # P_throat means the backward march implies a downstream-pipe
            # inlet pressure ABOVE the device's choke pressure. A choked
            # device cannot sustain a back-pressure above its own choke
            # pressure, so the trial mdot is too high — the device must go
            # SUBCRITICAL at a lower mdot. Signal _mode1_brentq to switch
            # to the subcritical forward solve (where Borda-Carnot
            # recovery is valid). At the boundary P_first_cell == P_throat
            # the two descriptions coincide, so the transition is
            # continuous.
            if P_first_cell > P_throat * (1.0 + rtol_sanity):
                raise DeviceMustGoSubcritical(
                    "Backward march at chain index "
                    f"{idx} implies pipe-inlet P = "
                    f"{P_first_cell / 1e5:.3f} bara above device throat P = "
                    f"{P_throat / 1e5:.3f} bara at mdot = {mdot:.4f} kg/s; "
                    "device cannot be choked here — go subcritical.",
                    pipe_index=idx,
                    mdot_attempted=mdot,
                    P_first_cell=P_first_cell,
                    P_throat=P_throat,
                )

            # Sanity B: M_first_cell within v1 validated range. Beyond
            # this we don't trust the non-adaptive backward Newton.
            if M_first_cell > 0.7:
                partial_results = (
                    list(results)
                    + [pr for _, pr in sorted(
                        backward_results + [(idx, pipe_result_bwd)],
                        key=lambda t: t[0],
                    )]
                )
                partial = ChainResult(
                    chain=chain,
                    results=partial_results,
                    mdot=mdot,
                    P_in=P_in,
                    P_last_cell=outlet_P,
                    T_in=T_in,
                    T_out=float(pipe_result_bwd.T[-1]),
                    boundary_conditions=boundary_conditions,
                    solver_options=solver_options,
                    elapsed=time.time() - t0,
                    choked=True,
                    choke_diagnostics={
                        **choke_diagnostics,
                        "M_first_cell_out_of_envelope": M_first_cell,
                    } if choke_diagnostics else None,
                )
                raise BVPChoked(
                    f"Backward march of pipe at chain index {idx} "
                    f"produced M_first_cell = {M_first_cell:.3f}, "
                    "above the v1 validated convergence range "
                    "(M < 0.7). Typically indicates the choked "
                    "device's A_vc is too close to the downstream "
                    "pipe's area, producing near-sonic post-"
                    "transition flow. Reduce A_vc/A_pipe ratio or "
                    "wait for v2 backward march with adaptive "
                    "refinement.",
                    mdot_critical=mdot,
                    result=partial,
                )

            backward_results.append((idx, pipe_result_bwd))
            # Next (upstream) pipe's outlet is this pipe's inlet.
            outlet_P = P_first_cell

        # Insert backward-marched pipes into results in chain order.
        backward_results.sort(key=lambda t: t[0])
        for _idx, pipe_result_bwd in backward_results:
            results.append(pipe_result_bwd)

        # Chain end state.
        last_bwd = backward_results[-1][1]
        P_current = float(last_bwd.P[-1])
        T_current = float(last_bwd.T[-1])

    return ChainResult(
        chain=chain,
        results=results,
        mdot=mdot,
        P_in=P_in,
        P_last_cell=P_current,
        T_in=T_in,
        T_out=T_current,
        boundary_conditions=boundary_conditions,
        solver_options=solver_options,
        elapsed=time.time() - t0,
        choked=choked,
        choke_diagnostics=choke_diagnostics,
    )


# ---------------------------------------------------------------------------
# solve_chain — three-mode dispatcher
# ---------------------------------------------------------------------------

_CHOKED_SENTINEL = -1e10


def solve_chain(
    chain: ChainSpec,
    fluid: "FluidEOSBase",
    T_in: float,
    *,
    P_in: float | None = None,
    P_out: float | None = None,
    P_last_cell: float | None = None,
    mdot: float | None = None,
    mdot_bracket: tuple[float, float] | None = None,
    P_in_bracket: tuple[float, float] | None = None,
    rtol: float = 1e-5,
    progress_callback: Callable[[str, float], None] | None = None,
    cancel_event: object | None = None,
    eos_mode: str | None = None,
    table_n_P: int = 50,
    table_n_T: int = 50,
    P_range_override: tuple[float, float] | None = None,
    T_range_override: tuple[float, float] | None = None,
    **ivp_kwargs,
) -> ChainResult:
    """Three-mode chain BVP solver. See module docstring for overview.

    The downstream BC is ``P_out`` — the reservoir pressure the chain
    discharges into. The deprecated ``P_last_cell`` kwarg is accepted for
    one transitional release and mapped to ``P_out`` with a
    :class:`DeprecationWarning`; ``P_last_cell`` is always a computed
    result, never a true input.
    """
    # Downstream-BC normalization. ``P_out`` is canonical; ``P_last_cell``
    # is a deprecated alias for one transitional release. Internally the
    # solver still binds the downstream pressure to the ``P_last_cell``
    # local (the forward-march criterion swap lands in commit 2).
    if P_last_cell is not None:
        if P_out is not None:
            raise ValueError(
                "solve_chain: pass either P_out (reservoir pressure) or the "
                "deprecated P_last_cell, not both."
            )
        warnings.warn(
            "P_last_cell as a BC is deprecated; pass P_out (reservoir "
            "pressure) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        P_out = P_last_cell
    # Bind the internal downstream-BC name. Commit 1 keeps the existing
    # _mode1_brentq / _mode3_brentq machinery, which is written against
    # ``P_last_cell``; commit 2 replaces that with the forward-march
    # criterion driven directly by ``P_out``.
    P_last_cell = P_out

    given = [
        name for name, val in (
            ("P_in", P_in), ("P_out", P_out), ("mdot", mdot),
        )
        if val is not None
    ]
    if len(given) != 2:
        raise ValueError(
            "solve_chain requires exactly two of (P_in, P_out, mdot); "
            f"got {given}"
        )

    t0 = time.time()
    bcs = {"given": tuple(given), "T_in": T_in}

    # Single-pipe (P_in, P_out) fast path: delegate to the legacy
    # _bvp_single_pipe_mdot, which finds mdot for a target outlet
    # pressure. In the new contract that target IS the reservoir P_out.
    if (
        "P_in" in given
        and "P_out" in given
        and len(chain.elements) == 1
    ):
        from .solver import _bvp_single_pipe_mdot

        pipe = chain.elements[0]

        def _wrap_pipe_result_in_chain(
            pr: "PipeResult", outlet_sonic: bool,
        ) -> ChainResult:
            return ChainResult(
                chain=chain,
                results=[pr],
                mdot=pr.mdot,
                P_in=P_in,
                P_last_cell=float(pr.P[-1]),
                T_in=T_in,
                T_out=float(pr.T[-1]),
                boundary_conditions=bcs,
                solver_options=dict(pr.solver_options or {}),
                elapsed=time.time() - t0,
                choked=bool(outlet_sonic),
                choke_diagnostics=(
                    {"kind": "outlet_sonic", "element_index": 0,
                     "x_choke": pr.x_choke,
                     "P_last_cell": float(pr.P[-1]), "P_out": P_out}
                    if outlet_sonic else None
                ),
                outlet_sonic=outlet_sonic,
                P_out=P_out,
            )

        try:
            pipe_result = _bvp_single_pipe_mdot(
                pipe, fluid, P_in, T_in, P_last_cell,
                mdot_bracket=mdot_bracket,
                rtol=rtol,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                eos_mode=eos_mode,
                table_n_P=table_n_P,
                table_n_T=table_n_T,
                P_range_override=P_range_override,
                T_range_override=T_range_override,
                **ivp_kwargs,
            )
        except BVPChoked as exc:
            # A single adiabatic pipe discharging to a reservoir below its
            # choke-limited last-cell pressure is OUTLET-SONIC, not an
            # error: the exit reaches M = 1, P_last_cell is the choke
            # pressure (> P_out), and the difference is an under-expanded
            # jet downstream (Lag 2). Return a clean ChainResult rather
            # than re-raising.
            return _wrap_pipe_result_in_chain(exc.result, outlet_sonic=True)
        return _wrap_pipe_result_in_chain(pipe_result, outlet_sonic=False)

    # ------------------------------------------------------------------
    # Multi-element or non-Mode-1 path.
    # ------------------------------------------------------------------
    from .eos import TabulatedFluid, estimate_operating_window
    from .solver import _DEFAULT_EOS_MODE_ENV

    base_fluid = _get_base_fluid(fluid)

    if eos_mode is None:
        eos_mode = os.environ.get(_DEFAULT_EOS_MODE_ENV, "table")
    if eos_mode not in ("direct", "table"):
        raise ValueError(f"eos_mode must be 'table' or 'direct'; got {eos_mode!r}")

    table: TabulatedFluid | None = None
    if eos_mode == "direct":
        working_fluid: "FluidEOSBase" = base_fluid
    else:
        if isinstance(fluid, TabulatedFluid):
            working_fluid = fluid
            table = fluid
        else:
            # Build a table sized around the BCs. For modes where one
            # endpoint pressure is unknown, use the known pressure as a
            # representative; estimate_operating_window's margin_factor
            # widens the window.
            P_known = P_in if P_in is not None else P_last_cell
            P_other = P_last_cell if P_last_cell is not None else P_in
            if P_range_override is not None and T_range_override is not None:
                P_range, T_range = P_range_override, T_range_override
            else:
                P_min, P_max, T_min, T_max = estimate_operating_window(
                    P_known, T_in, P_other, base_fluid,
                )
                P_range = P_range_override or (P_min, P_max)
                T_range = T_range_override or (T_min, T_max)
            table = TabulatedFluid(
                base_fluid, P_range, T_range, table_n_P, table_n_T,
            )
            working_fluid = table

    solver_options = {"eos_mode": eos_mode}
    if table is not None:
        solver_options["table_stats"] = table.table_stats()

    if "P_in" in given and "mdot" in given:
        # Mode 2 — single forward chain march.
        return _chain_forward_march(
            chain, working_fluid, base_fluid,
            P_in, T_in, mdot,
            boundary_conditions=bcs,
            solver_options=solver_options,
            t0=t0,
            cancel_event=cancel_event,
            **ivp_kwargs,
        )

    if "P_in" in given and "P_out" in given:
        return _mode1_forward_pout(
            chain, working_fluid, base_fluid,
            P_in, T_in, P_out,
            mdot_bracket=mdot_bracket,
            rtol=rtol,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            boundary_conditions=bcs,
            solver_options=solver_options,
            t0=t0,
            **ivp_kwargs,
        )

    # Mode 3 — (P_out, mdot) given → solve for P_in. The reservoir BC is
    # P_out; _mode3_brentq matches it as the last-cell target. Stamp P_out
    # and outlet_sonic onto the converged result for reporting parity
    # with the Mode 1 path.
    mode3_result = _mode3_brentq(
        chain, working_fluid, base_fluid,
        T_in, P_out, mdot,
        P_in_bracket=P_in_bracket,
        rtol=rtol,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        boundary_conditions=bcs,
        solver_options=solver_options,
        t0=t0,
        **ivp_kwargs,
    )
    mode3_result.P_out = P_out
    mode3_result.outlet_sonic = _outlet_sonic_flag(
        mode3_result.P_last_cell, P_out, rtol,
    )
    return mode3_result


# ---------------------------------------------------------------------------
# Single forward-march criterion for the (P_in, P_out) BC pair
# ---------------------------------------------------------------------------


def _march_chain_outlet_pressure(
    chain: ChainSpec,
    working_fluid: "FluidEOSBase",
    base_fluid: "GERGFluid",
    P_in: float,
    T_in: float,
    mdot: float,
    *,
    boundary_conditions: dict,
    solver_options: dict,
    t0: float,
    cancel_event: object | None = None,
    **ivp_kwargs,
) -> tuple[ChainResult, bool]:
    """Forward-march the chain at fixed ``mdot``; report outlet sonic state.

    Returns ``(result, outlet_is_sonic)`` where ``outlet_is_sonic`` is
    True iff the last pipe's outlet cell has reached M = 1. This is the
    primitive the single forward-march criterion (commit 2) calls to
    decide whether the chain is in the subsonic-outlet regime
    (``P_last_cell == P_out``) or the sonic-outlet regime
    (``P_last_cell > P_out``, under-expanded jet downstream).

    Commit 1 (scaffolding): a thin wrapper over
    :func:`_chain_forward_march` that reports the existing per-pipe
    ``choked`` flag as the outlet-sonic proxy. Commit 2 refines this to
    check the LAST pipe's outlet-cell Mach specifically (a mid-pipe
    Fanno choke is a different state).
    """
    result = _chain_forward_march(
        chain, working_fluid, base_fluid,
        P_in, T_in, mdot,
        boundary_conditions=boundary_conditions,
        solver_options=solver_options,
        t0=t0,
        cancel_event=cancel_event,
        **ivp_kwargs,
    )
    return result, bool(result.choked)


def _mode1_forward_pout(
    chain: ChainSpec,
    working_fluid: "FluidEOSBase",
    base_fluid: "GERGFluid",
    P_in: float,
    T_in: float,
    P_out_target: float,
    *,
    mdot_bracket: tuple[float, float] | None,
    rtol: float,
    progress_callback: Callable[[str, float], None] | None,
    cancel_event: object | None,
    boundary_conditions: dict,
    solver_options: dict,
    t0: float,
    **ivp_kwargs,
) -> ChainResult:
    """Single forward-march criterion for a fixed (P_in, P_out) pair.

    With P_in and the reservoir pressure P_out both fixed there is
    exactly one mdot. The selection criterion changes character
    continuously at the choke threshold:

    - subsonic outlet: find mdot such that ``P_last_cell(mdot) = P_out``;
    - sonic outlet: the outlet is pinned at M = 1 and ``P_last_cell``
      rises above ``P_out`` (under-expanded jet takes up the rest); the
      selected mdot is the choke-limited one.

    Commit 1 (scaffolding): delegates to the existing
    :func:`_mode1_brentq` so behaviour is byte-identical. The real
    forward-march criterion (root-find on outlet pressure + choke-
    threshold bisection via :func:`_march_chain_outlet_pressure`)
    lands in commit 2, along with removal of the backward-march
    machinery in :func:`_chain_forward_march`.
    """
    return _mode1_brentq(
        chain, working_fluid, base_fluid,
        P_in, T_in, P_out_target,
        mdot_bracket=mdot_bracket,
        rtol=rtol,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        boundary_conditions=boundary_conditions,
        solver_options=solver_options,
        t0=t0,
        **ivp_kwargs,
    )


def _last_pipe_outlet_sonic(
    result: ChainResult, chain: ChainSpec, mach_choke: float,
) -> bool:
    """True iff the chain's LAST pipe reached M >= mach_choke at its exit.

    Mach-based internal detector (used to spot a forward-march outlet
    choke). The RESULT's ``outlet_sonic`` *field* is set from the
    pressure criterion (:func:`_outlet_sonic_flag`) — the physical
    meaning of outlet-sonic is "P_last_cell exceeds P_out (a jet absorbs
    the difference)", not "outlet Mach is high".
    """
    from .results import PipeResult

    if len(result.results) == len(chain.elements) and result.results:
        last = result.results[-1]
        if isinstance(last, PipeResult) and len(last.M) > 0:
            return float(last.M[-1]) >= mach_choke
    return False


def _outlet_sonic_flag(P_last_cell: float, P_out: float, rtol: float) -> bool:
    """Outlet-sonic iff the computed last cell strictly exceeds P_out.

    When ``P_last_cell == P_out`` (within rtol) the gas discharges to the
    reservoir at reservoir pressure — no under-expanded jet, so NOT
    outlet-sonic even if the pinned outlet cell sits near M = 1. The flag
    means specifically that ``P_last_cell > P_out`` and the difference is
    taken up by a free jet downstream (Lag 2).
    """
    return P_last_cell > P_out * (1.0 + rtol)


def _mode1_pipe_bottleneck(
    chain: ChainSpec,
    working_fluid: "FluidEOSBase",
    base_fluid: "GERGFluid",
    P_in: float,
    T_in: float,
    P_out_target: float,
    *,
    mdot_device: float,
    mdot_lo: float,
    rtol: float,
    cancel_event: object | None,
    boundary_conditions: dict,
    solver_options: dict,
    t0: float,
    device_throat_P: float | None = None,
    **ivp_kwargs,
) -> ChainResult:
    """Device + downstream-pipe bottleneck — consistency-rule model.

    Replaces the v0 *backward-discharge-capacity* model. The operating
    point is set by two ORTHOGONAL binaries (see CLAUDE.md "Pressure
    terminology" and BACKLOG Item 10):

      device state  : CHOKED      → mdot = ``mdot_device`` (FLAT) vs
                      SUBCRITICAL → mdot < ``mdot_device`` (coupled solve)
      outlet anchor : EXIT-SONIC  → pipe Fanno-chokes at its own exit;
                      ``P_last_cell = P_exit_sonic > P_out``; a free
                      under-expanded jet absorbs ``P_exit_sonic → P_out`` vs
                      P_OUT       → pipe discharges subsonically;
                      ``P_last_cell = P_out``.

    **Consistency rule** (governs the device state). Let ``P_first`` be the
    downstream-pipe INLET pressure obtained by backward-marching the pipe
    from its outlet anchor, and ``P_crit`` the device throat (choke)
    pressure:

      - ``P_first <  P_crit`` → device CHOKED. A choked device fixes
        ``mdot = mdot_device``; the downstream pipe carries that flow and is
        anchored at ``max(P_out, P_exit_sonic)``. No Borda-Carnot recovery
        is invoked — ``P_first`` falls out of the (backward) Fanno profile.
      - ``P_first >= P_crit`` → a choked device is impossible (its back-
        pressure would exceed its own choke pressure). The device runs
        SUBCRITICAL: solve ``F(mdot) = devBC(mdot) − P_first(mdot) = 0`` for
        ``mdot < mdot_device``, where ``devBC`` is the (TAPERED) Borda-Carnot
        pipe-inlet pressure from :func:`_device_solve_at_mdot`.

    **Outlet anchor.** ``P_anchor = max(P_out, P_exit_sonic(mdot))``.
    ``P_exit_sonic`` is the LOCAL pressure at the chain-exit area where
    ``M = 1`` at the given ``mdot`` and conserved ``h_stag`` — independent
    of ``P_out`` (choked-flow physics: the free jet absorbs the rest). When
    ``P_out >= P_exit_sonic`` the pipe discharges subsonically and the
    anchor reduces to ``P_out``. The same ``max(...)`` rule serves both the
    choked and subcritical branches, so the choked↔subcritical and
    sonic↔subsonic transitions are each continuous.

    **Continuity at the regime boundary D-star** (where ``P_first → P_crit``):
    provided by the compressible recovery taper ``f_rec(M_vc)`` in
    :func:`gas_pipe.device._borda_carnot_transition`. As ``M_vc → 1``,
    ``f_rec → 0`` so ``devBC → P_t = P_crit``, matching the choked-side
    ``P_first = P_crit`` (no ~7 bara jump). The taper also pulls ``devBC``
    down in the subcritical tail, which is what makes the ``F = 0`` crossing
    exist for pipes just below D-star (untapered Borda-Carnot over-predicts
    ``devBC`` so ``F`` had no zero there).

    v1 requires downstream pipes adiabatic (``overall_U == 0``); diabatic
    downstream raises :class:`BackwardMarchDiabaticNotSupported`.
    """
    from .solver import march_ivp, march_ivp_backward, state_from_P_hstag
    from .errors import BackwardMarchDiabaticNotSupported

    rtol_local = max(rtol, 1e-5)

    device_idx = next(
        (i for i, el in enumerate(chain.elements) if isinstance(el, Device)),
        None,
    )
    if device_idx is None:
        raise BVPNotBracketedError(
            "Pipe-bottleneck solve requires a device in the chain."
        )
    upstream_elements = chain.elements[:device_idx]
    device = chain.elements[device_idx]
    downstream_pipes = [
        (i, el) for i, el in enumerate(chain.elements) if i > device_idx
    ]
    for i, el in downstream_pipes:
        if isinstance(el, Device):
            raise NotImplementedError(
                "v1 pipe-bottleneck solve does not support a second Device "
                f"downstream (element {i})."
            )
    if not downstream_pipes:
        raise BVPNotBracketedError(
            "Pipe-bottleneck solve: no downstream pipe after the device."
        )
    for i, pipe in downstream_pipes:
        for sec_idx, sec in enumerate(pipe.sections):
            if sec.overall_U != 0.0:
                raise BackwardMarchDiabaticNotSupported(
                    f"Pipe at chain index {i}, section {sec_idx} has "
                    f"overall_U = {sec.overall_U}. The pipe-bottleneck "
                    "solve backward-marches the downstream pipe and "
                    "requires adiabatic conditions in v1."
                )

    bwd_kwargs: dict = {}
    if "n_segments" in ivp_kwargs:
        bwd_kwargs["n_segments"] = int(ivp_kwargs["n_segments"])
    if "friction_model" in ivp_kwargs:
        bwd_kwargs["friction_model"] = ivp_kwargs["friction_model"]

    # A_down: inlet area of the first downstream pipe (Borda-Carnot expansion
    # target). A_exit: exit area of the last downstream pipe (chain exit,
    # where the exit-sonic anchor is evaluated). Same for a single uniform
    # downstream pipe.
    A_down = downstream_pipes[0][1].sections[0].area
    _last_pipe = downstream_pipes[-1][1]
    A_exit = _last_pipe.section_at(_last_pipe.length)[0].area

    def _march_upstream(mdot: float):
        """March the upstream pipe(s) at ``mdot``.

        Returns ``(fwd_results, state_up, u_up, h_stag)`` for the device
        inlet. ``h_stag`` is exact across the device, so it is also the
        conserved invariant for the adiabatic downstream backward march
        (correct even when the upstream pipe is diabatic).
        """
        results: list = []
        P_cur, T_cur, u_cur = P_in, T_in, 0.0
        for el in upstream_elements:
            pr = march_ivp(
                el, working_fluid, P_cur, T_cur, mdot,
                cancel_event=cancel_event, **ivp_kwargs,
            )
            results.append(pr)
            P_cur = float(pr.P[-1])
            T_cur = float(pr.T[-1])
            u_cur = float(pr.u[-1])
        state_up = base_fluid.props(P_cur, T_cur)
        return results, state_up, u_cur, state_up.h + 0.5 * u_cur ** 2

    def _forward_to_device(mdot: float):
        """March upstream pipe(s) + solve the device at ``mdot``.

        Returns ``(fwd_results, device_result, devBC, h_stag)`` where
        ``devBC`` is the tapered Borda-Carnot pipe-inlet pressure.
        """
        results, state_up, u_cur, h_stag_device = _march_upstream(mdot)
        device_result = _device_solve_at_mdot(
            device, state_up, u_cur, mdot, A_down,
            fluid=base_fluid, element_index=device_idx,
        )
        devBC = float(device_result.transition.state_inlet.P)
        return results, device_result, devBC, h_stag_device

    def _exit_mach(P: float, mdot: float, h_stag: float) -> float:
        """Local Mach at the chain-exit area for outlet pressure ``P``."""
        st = state_from_P_hstag(working_fluid, P, h_stag, mdot, A_exit)
        u = mdot / (st.rho * A_exit)
        return u / st.a

    def _P_exit_sonic(mdot: float, h_stag: float) -> float:
        """Outlet pressure where the chain exit is sonic (``M = 1``).

        ``M`` is monotone decreasing in ``P``. If the exit is already
        subsonic at ``P_out`` the pipe is NOT sonic-limited — return
        ``P_out`` so the anchor ``max(P_out, P_exit_sonic)`` reduces to
        ``P_out``. Otherwise bisect upward for ``M = 1``.
        """
        if _exit_mach(P_out_target, mdot, h_stag) <= 1.0:
            return P_out_target
        P_lo = P_out_target          # M > 1 here
        P_hi = P_out_target * 2.0
        for _ in range(60):
            if _exit_mach(P_hi, mdot, h_stag) < 1.0:
                break
            P_hi *= 1.5
            if P_hi > 1e9:
                break
        for _ in range(100):
            if (P_hi - P_lo) < rtol_local * P_hi:
                break
            P_mid = 0.5 * (P_lo + P_hi)
            if _exit_mach(P_mid, mdot, h_stag) > 1.0:
                P_lo = P_mid
            else:
                P_hi = P_mid
        return 0.5 * (P_lo + P_hi)

    def _backward_from_anchor(mdot: float, h_stag: float):
        """Backward-march downstream pipe(s) from ``max(P_out, P_exit_sonic)``.

        Returns ``(bwd_results, P_first, P_anchor, P_exit_sonic)``. A tiny
        upward nudge on the anchor is retried if the first backward segment
        Newton struggles right at the ``M = 1`` knife-edge.
        """
        P_sonic = _P_exit_sonic(mdot, h_stag)
        P_anchor = max(P_out_target, P_sonic)
        for nudge in (1.0, 1.0 + 1e-5, 1.0 + 1e-4):
            outlet_P = P_anchor * nudge
            try:
                bwd: list = []
                for i, pipe in reversed(downstream_pipes):
                    prb = march_ivp_backward(
                        pipe, working_fluid,
                        P_outlet=outlet_P, h_stag=h_stag, mdot=mdot,
                        cancel_event=cancel_event, **bwd_kwargs,
                    )
                    bwd.append((i, prb))
                    outlet_P = float(prb.P[0])
                break
            except SegmentConvergenceError:
                if nudge == 1.0 + 1e-4:
                    raise
        bwd.sort(key=lambda t: t[0])
        P_first = float(bwd[0][1].P[0])
        return [pr for _, pr in bwd], P_first, P_anchor, P_sonic

    def _assemble(
        mdot: float, fwd: list, device_result: DeviceResult, bwd: list,
        P_first: float, P_anchor: float, P_sonic: float, *, device_choked: bool,
    ) -> ChainResult:
        results = list(fwd) + [device_result] + list(bwd)
        P_last_cell = float(P_anchor)
        outlet_sonic = _outlet_sonic_flag(P_last_cell, P_out_target, rtol_local)
        device_state = "choked" if device_choked else "subcritical"
        anchor_kind = "exit_sonic" if outlet_sonic else "P_out"
        diag: dict = {
            "kind": f"device_{device_state}_pipe_{anchor_kind}",
            "device_state": device_state,
            "device_choked": device_choked,
            "outlet_anchor": anchor_kind,
            "mdot": float(mdot),
            "mdot_device": float(mdot_device),
            "P_first": float(P_first),
            "P_crit": float(P_crit) if P_crit is not None else None,
            "P_exit_sonic": float(P_sonic),
            "P_anchor": float(P_anchor),
            "P_last_cell": float(P_last_cell),
            "P_out": float(P_out_target),
            "M_vc": float(device_result.throat.M),
            "f_rec": float(device_result.transition.f_rec),
            "devBC": float(device_result.transition.state_inlet.P),
        }
        return ChainResult(
            chain=chain, results=results, mdot=float(mdot),
            P_in=P_in, P_last_cell=P_last_cell,
            T_in=T_in, T_out=float(bwd[-1].T[-1]),
            boundary_conditions=boundary_conditions,
            solver_options=solver_options,
            elapsed=time.time() - t0,
            choked=bool(device_choked or outlet_sonic),
            choke_diagnostics=diag,
            outlet_sonic=outlet_sonic,
            P_out=P_out_target,
        )

    # ---- Device choke ceiling + choke pressure (consistency anchors) ----
    # The ``mdot_device`` passed in is the CHAIN-choke ceiling resolved by
    # the brentq machinery; for a narrow downstream pipe that ceiling is set
    # by the pipe's forward Fanno choke and sits slightly BELOW the device's
    # own choke ceiling. The choked-regime mdot must be the DEVICE ceiling
    # (flat across the choked diameter range, independent of the downstream
    # pipe), and the consistency rule needs the device CHOKE pressure (at
    # M_vc = 1), not the throat pressure at a sub-choke probe. Recompute both
    # cleanly. The upstream pipe is effectively frictionless at these
    # sub-kg/s flows, so state_up — and hence the ceiling — is independent of
    # the seed mdot; one pass suffices.
    from .device import _stagnation_state

    _, _state_up_seed, _u_seed, _ = _march_upstream(mdot_device)
    _stag = _stagnation_state(base_fluid, _state_up_seed, _u_seed)
    _ts_choke = base_fluid.hem_throat(
        _stag.P, _stag.T, 0.05 * _stag.P, A_vc=device.A_vc,
    )
    mdot_device = float(_ts_choke.mdot)
    P_crit = float(_ts_choke.P)
    # device_throat_P (passed from the ceiling march) is a fallback in case
    # the fresh choke probe ever returns a non-physical pressure.
    if not (math.isfinite(P_crit) and P_crit > 0.0) and device_throat_P:
        P_crit = float(device_throat_P)

    # ---- CHOKED-REGIME TRIAL at the device ceiling ----
    # mdot is FLAT at mdot_device while the device chokes. Anchor the pipe
    # at max(P_out, P_exit_sonic) and backward-march to the inlet P_first.
    # For a sufficiently narrow downstream pipe the device's Borda-Carnot
    # transition has NO subsonic solution at the full ceiling mdot (the pipe
    # inlet would be supersonic) and the device solve diverges — that is
    # itself proof the device cannot be choked here, so fall through to the
    # subcritical solve.
    try:
        fwd_c, dev_c, devBC_c, h_stag_c = _forward_to_device(mdot_device)
        bwd_c, P_first_c, P_anchor_c, P_sonic_c = _backward_from_anchor(
            mdot_device, h_stag_c,
        )
        if P_first_c <= P_crit * (1.0 + rtol_local):
            # Consistent: a choked device CAN sit behind this pipe inlet.
            return _assemble(
                mdot_device, fwd_c, dev_c, bwd_c,
                P_first_c, P_anchor_c, P_sonic_c, device_choked=True,
            )
    except (
        OverChokedError, ChokeReached, SegmentConvergenceError,
        EOSOutOfRange, EOSTwoPhase, HEMConsistencyError,
    ):
        pass  # device cannot be choked at this geometry → subcritical

    # ---- SUBCRITICAL SOLVE: F(mdot) = devBC(mdot) − P_first(mdot) = 0 ----
    # Near the ceiling P_first > devBC ⇒ F < 0; as mdot falls the device
    # unchokes — devBC rises (recovery returns as M_vc < 1, f_rec > 0) and
    # P_first falls (the pipe carries less flow), so F rises through zero.
    # That crossing is the subcritical operating point.
    def _eval(mdot: float):
        fwd, dev, devBC, h_stag = _forward_to_device(mdot)
        bwd, P_first, P_anchor, P_sonic = _backward_from_anchor(mdot, h_stag)
        payload = (fwd, dev, devBC, bwd, P_first, P_anchor, P_sonic)
        return devBC - P_first, payload

    def _F(mdot: float):
        """``F(mdot)`` with infeasible probes mapped to a signed sentinel.

        Two distinct infeasibilities bound the subcritical band, on
        OPPOSITE sides, so they must be signed differently for bracketing:

          - ``OverChokedError`` — the device ``P_back`` brent fails to
            bracket at FAR-subcritical (too-LOW) mdot → ``+inf`` (the
            ``F > 0`` / low side).
          - Borda-Carnot / pipe-transition divergence (the pipe inlet would
            be supersonic at too-HIGH mdot) → ``-inf`` (the ``F < 0`` /
            high side).
        """
        try:
            return _eval(mdot)
        except OverChokedError:
            return float("inf"), None
        except (
            ChokeReached, SegmentConvergenceError,
            EOSOutOfRange, EOSTwoPhase, HEMConsistencyError,
        ):
            return float("-inf"), None

    # Bracket the F = 0 crossing by scanning mdot ascending from mdot_lo to
    # mdot_device. Infeasible probes are skipped; the first feasible
    # F > 0 → F < 0 transition between consecutive feasible probes is the
    # bracket. (F is monotone decreasing in mdot on the feasible band, so
    # there is at most one crossing.)
    n_scan = 32
    prev_m: float | None = None
    prev_F: float | None = None
    prev_pl = None
    lo_m = lo_pl = hi_m = None
    for k in range(n_scan + 1):
        m = mdot_lo + (mdot_device - mdot_lo) * k / n_scan
        F_m, pl_m = _F(m)
        if not math.isfinite(F_m):
            continue
        if prev_F is not None and prev_F > 0.0 and F_m < 0.0:
            lo_m, lo_pl, hi_m = prev_m, prev_pl, m
            break
        prev_m, prev_F, prev_pl = m, F_m, pl_m

    if lo_m is None:
        # No feasible F > 0 → F < 0 transition found. Per the standing
        # instruction, surface this with numbers rather than forcing it.
        raise BVPNotBracketedError(
            "Pipe-bottleneck subcritical solve: no F = devBC − P_first sign "
            f"change found scanning mdot in [{mdot_lo:.4f}, {mdot_device:.4f}] "
            f"kg/s (P_crit = {P_crit/1e5:.3f} bara, P_out = "
            f"{P_out_target/1e5:.3f} bara)."
        )

    # Bisect [lo_m (F>0), hi_m (F<0)] for F = 0; keep the latest F > 0
    # (subcritical, feasible) probe as the returned operating point.
    m_best, pl_best = lo_m, lo_pl
    while (hi_m - lo_m) >= rtol_local * hi_m:
        m_mid = 0.5 * (lo_m + hi_m)
        F_mid, pl_mid = _F(m_mid)
        if not math.isfinite(F_mid):
            # Infeasible probe: +inf → too low (raise floor); -inf → too
            # high (lower ceiling).
            if F_mid > 0.0:
                lo_m = m_mid
            else:
                hi_m = m_mid
            continue
        if F_mid > 0.0:
            lo_m, m_best, pl_best = m_mid, m_mid, pl_mid
        else:
            hi_m = m_mid

    fwd_s, dev_s, devBC_s, bwd_s, P_first_s, P_anchor_s, P_sonic_s = pl_best
    return _assemble(
        m_best, fwd_s, dev_s, bwd_s,
        P_first_s, P_anchor_s, P_sonic_s, device_choked=False,
    )


# ---------------------------------------------------------------------------
# Mode 1 — brentq on mdot (multi-element only; single-pipe takes fast path)
# ---------------------------------------------------------------------------


def _mode1_brentq(
    chain: ChainSpec,
    working_fluid: "FluidEOSBase",
    base_fluid: "GERGFluid",
    P_in: float,
    T_in: float,
    P_last_cell_target: float,
    *,
    mdot_bracket: tuple[float, float] | None,
    rtol: float,
    progress_callback: Callable[[str, float], None] | None,
    cancel_event: object | None,
    boundary_conditions: dict,
    solver_options: dict,
    t0: float,
    **ivp_kwargs,
) -> ChainResult:
    """Mode 1 brentq on mdot for multi-element chains.

    Forward chain march per probe. ``OverChokedError`` and pipe-fanno
    chokes map to ``_CHOKED_SENTINEL``. After brentq converges, if the
    result is on the choked side, re-raise as :class:`BVPChoked`.
    """
    # Initial bracket: use the first pipe's AGA estimator (conservative;
    # devices add pressure drop, so the true mdot is generally ≤ AGA
    # estimate for the first pipe alone).
    if mdot_bracket is None:
        from .solver import _aga_estimate_mdot
        first_pipe = chain.pipes[0]
        mdot_mid = _aga_estimate_mdot(
            first_pipe, working_fluid, P_in, T_in, P_last_cell_target,
        )
        mdot_lo, mdot_hi = mdot_mid * 0.01, mdot_mid * 5.0
        inlet = working_fluid.props(P_in, T_in)
        mdot_hi_M05 = 0.5 * inlet.a * inlet.rho * first_pipe.area
        if mdot_hi > mdot_hi_M05:
            mdot_hi = mdot_hi_M05
        if mdot_lo >= mdot_hi:
            mdot_lo = mdot_hi * 0.01
    else:
        mdot_lo, mdot_hi = mdot_bracket

    last_good: ChainResult | None = None
    last_good_mdot: float | None = None
    last_choked_mdot: float | None = None
    # Smallest sentinel-returning mdot seen across all probes. Used as
    # the upper bound when bisecting toward the operating-regime choke
    # ceiling before the backward-march dispatch re-march. See the
    # bisect block further down for why last_choke_diag.max_mdot is
    # not sufficient: the diag's max_mdot is mdot_max at the SENTINEL
    # probe's state_up, not at the operating regime's state_up.
    last_choked_smallest_mdot: float | None = None
    last_choke_diag: dict | None = None

    def _march(mdot_try: float) -> ChainResult:
        return _chain_forward_march(
            chain, working_fluid, base_fluid,
            P_in, T_in, mdot_try,
            boundary_conditions=boundary_conditions,
            solver_options=solver_options,
            t0=t0,
            cancel_event=cancel_event,
            **ivp_kwargs,
        )

    def _mark_choked(mdot_try: float, diag: dict | None) -> float:
        """Bookkeeping for a sentinel-returning probe.

        Updates both the latest sentinel mdot (``last_choked_mdot``) and
        the smallest sentinel mdot ever seen (``last_choked_smallest_mdot``).
        The smallest tracker is the upper bound for the post-walk-down
        bisect that refines ``mdot_ceiling`` to the operating regime.
        """
        nonlocal last_choked_mdot, last_choked_smallest_mdot, last_choke_diag
        last_choked_mdot = mdot_try
        if (
            last_choked_smallest_mdot is None
            or mdot_try < last_choked_smallest_mdot
        ):
            last_choked_smallest_mdot = mdot_try
        last_choke_diag = diag
        return _CHOKED_SENTINEL

    def _obj(mdot_try: float) -> float:
        nonlocal last_good, last_good_mdot
        try:
            r = _march(mdot_try)
            if r.choked:
                return _mark_choked(mdot_try, r.choke_diagnostics)
            last_good = r
            last_good_mdot = mdot_try
            return r.P_last_cell - P_last_cell_target
        except OverChokedError as exc:
            return _mark_choked(mdot_try, {
                "kind": "device_over_choked",
                "element_index": exc.device_index,
                "element_name": exc.device_name,
                "max_mdot": exc.max_mdot,
                "P_stag": exc.P_stag,
                "T_stag": exc.T_stag,
            })
        except (ChokeReached, SegmentConvergenceError):
            return _mark_choked(mdot_try, {"kind": "fanno_pipe_during_march"})
        except (EOSOutOfRange, EOSTwoPhase, HEMConsistencyError) as exc:
            # The BC probe drove the throat-solve isentrope into a region
            # GERG cannot evaluate (near-critical, sub-dew, etc.). Treat as
            # infeasible-probe; brentq adjusts away from it.
            return _mark_choked(mdot_try, {
                "kind": "eos_infeasible",
                "exception": type(exc).__name__,
                "message": str(exc)[:200],
            })

    # Bracket adjustment
    f_lo = _obj(mdot_lo)
    f_hi = _obj(mdot_hi)

    n_extend = 0
    while f_hi > 0 and n_extend < 6:
        mdot_hi *= 3.0
        f_hi = _obj(mdot_hi)
        n_extend += 1
    while f_lo < 0 and n_extend < 12 and f_lo != _CHOKED_SENTINEL:
        mdot_lo *= 0.3
        f_lo = _obj(mdot_lo)
        n_extend += 1

    # If lo is choked, walk up until non-choked.
    while f_lo == _CHOKED_SENTINEL and mdot_lo < mdot_hi:
        mdot_lo = mdot_lo * 1.5 if mdot_lo > 0 else mdot_hi * 0.01
        f_lo = _obj(mdot_lo)

    # If hi is choked, locate mdot_critical and decide whether the target
    # P_last_cell is reachable below it (mirrors the legacy
    # _bvp_single_pipe_mdot pattern).
    if f_hi == _CHOKED_SENTINEL:
        if f_lo == _CHOKED_SENTINEL:
            # Both endpoints sentinel — typically the Device-bottleneck
            # regime: the AGA estimator (fed only the first pipe)
            # over-estimated mdot, so every initial probe exceeded the
            # device's HEM choke ceiling and the walk-up-when-choked
            # loop inverted the bracket. Recovery:
            #   1. Walk mdot down geometrically to find ANY feasible
            #      probe (sub-ceiling) → populates ``last_good`` via the
            #      nonlocal closure in ``_obj``.
            #   2. Read the chain choke ceiling from ``last_choke_diag``
            #      (device max_mdot) so we know the upper bound on
            #      attainable mdot.
            #   3. March at just-below the ceiling for the canonical
            #      choke-limited profile.
            #   4. Decide reachability of ``P_last_cell_target``
            #      symmetrically with the Fanno-bottleneck else-branch
            #      below.
            mdot_floor = max(mdot_hi * 1e-6, 1e-12)
            mdot_try = min(mdot_lo, mdot_hi)
            for _ in range(20):
                mdot_try *= 0.3
                if mdot_try < mdot_floor:
                    break
                _obj(mdot_try)  # side effect: updates last_good
                if last_good is not None:
                    break

            if last_good is None:
                # Chain genuinely infeasible at every probed mdot down
                # to the floor — no useful witness available. Fall back
                # to the minimal NaN result so the caller still gets a
                # consistent BVPChoked payload.
                raise BVPChoked(
                    "Chain choked throughout the mdot bracket and walk-"
                    "down; no feasible mdot in "
                    f"[{mdot_floor:.3e}, {mdot_hi:.3f}] kg/s.",
                    mdot_critical=mdot_lo,
                    result=_build_minimal_choked_result(
                        chain, P_in, T_in, mdot_lo,
                        boundary_conditions, solver_options, t0,
                        last_choke_diag,
                    ),
                )

            # Walk-down found a feasible probe. Resolve the chain's
            # choke ceiling: prefer the device's ``max_mdot`` when the
            # diagnostic is populated (the common Device-bottleneck
            # case); fall back to the last sentinel mdot we saw.
            mdot_ceiling: float | None = None
            if last_choke_diag is not None:
                max_mdot_val = last_choke_diag.get("max_mdot")
                if (
                    isinstance(max_mdot_val, (int, float))
                    and math.isfinite(max_mdot_val)
                    and max_mdot_val > last_good_mdot
                ):
                    mdot_ceiling = float(max_mdot_val)
            if mdot_ceiling is None:
                # Witness-only fallback: no clean ceiling estimate
                # available. Treat as device-bottleneck-with-unknown-
                # ceiling and use the walk-down probe as the choke
                # witness so the caller still gets a populated result.
                if last_good.choke_diagnostics is None:
                    last_good.choke_diagnostics = last_choke_diag
                raise BVPChoked(
                    "Chain choke-limited; no resolvable choke ceiling. "
                    f"Walk-down probe at mdot={last_good_mdot:.4f} kg/s "
                    "used as the choke witness; target P_last_cell="
                    f"{P_last_cell_target/1e5:.3f} bara assumed "
                    "unreachable.",
                    mdot_critical=last_good_mdot,
                    result=last_good,
                )

            # Refine mdot_ceiling to the operating regime's choke
            # boundary. The diag-derived estimate above is
            # ``mdot_max`` evaluated at a SENTINEL probe's state_up,
            # not at the state_up the chain actually presents to the
            # device at the operating mdot. When U > 0 cools pipe 1,
            # state_up at the device shifts across mdot probes; the
            # captured estimate then lies well below the device's
            # true ``mdot_max`` at operating conditions and the
            # subsequent re-march at ``mdot_ceiling * (1 - 1e-3)``
            # falls outside the device's just-choked tolerance window,
            # so ``throat.choked`` stays False and the backward-march
            # trigger does not fire.
            #
            # Bisect ``_obj`` between the largest feasible mdot
            # (``last_good_mdot``) and the smallest sentinel mdot
            # ever seen (``last_choked_smallest_mdot``) until the gap
            # is < 1e-4 relative. The resulting ``m_hi`` is the
            # operating-regime choke ceiling; the device at this
            # mdot has ``throat.choked = True`` (or M very near 1)
            # for both adiabatic and U > 0 chains.
            # Snapshot the pre-bisect feasible probe so the subcritical
            # fall-through bracket downstream (used when ``target`` lies
            # above the choke-limited ``P_last_cell``) keeps the wide
            # walk-down bracket. Without this, the bisect ratchets
            # ``last_good_mdot`` up against the choke boundary and the
            # subsequent brentq sees ``mdot_lo`` and ``mdot_hi`` both
            # essentially at the ceiling — both probes return
            # near-ceiling ``P_last_cell``, so f_lo and f_hi share the
            # same sign and the bracket-check raises BVPNotBracketedError.
            walkdown_last_good = last_good
            walkdown_last_good_mdot = last_good_mdot
            if (
                last_choked_smallest_mdot is not None
                and last_good_mdot is not None
                and last_choked_smallest_mdot > last_good_mdot
            ):
                m_lo = last_good_mdot
                m_hi = last_choked_smallest_mdot
                for _ in range(20):
                    if (m_hi - m_lo) < 1e-4 * m_hi:
                        break
                    m_mid = 0.5 * (m_lo + m_hi)
                    f_mid = _obj(m_mid)
                    if f_mid == _CHOKED_SENTINEL:
                        m_hi = m_mid
                    else:
                        m_lo = m_mid
                mdot_ceiling = m_hi

            # March at just-below the ceiling for the canonical
            # choke-limited profile. The 1e-3 cushion is loose enough
            # to avoid hitting _device_solve_at_mdot's just-choked
            # tolerance (rel_tol = 1e-4) inconsistently across
            # operating regimes — e.g., adiabatic vs U > 0 yields
            # different mdot_max curves, so a tighter cushion landed
            # inside the just-choked window in one regime and outside
            # in another. The downstream backward-mode trigger relies
            # on throat.M >= 0.95 (not bitwise throat.choked), which
            # is robust to this 0.1% offset.
            mdot_at_ceiling = mdot_ceiling * (1.0 - 1e-3)
            try:
                r_ceiling = _march(mdot_at_ceiling)
            except (
                OverChokedError, ChokeReached, SegmentConvergenceError,
                EOSOutOfRange, EOSTwoPhase, HEMConsistencyError,
            ):
                # Ceiling-1e-3 still chokes the chain (e.g. residual
                # tolerance mismatch). Use the walk-down probe as the
                # choke witness.
                if last_good.choke_diagnostics is None:
                    last_good.choke_diagnostics = last_choke_diag
                raise BVPChoked(
                    "Chain choke-limited; canonical ceiling march "
                    f"failed at mdot={mdot_at_ceiling:.4f} kg/s. "
                    "Walk-down probe used as the witness.",
                    mdot_critical=mdot_ceiling,
                    result=last_good,
                )

            if r_ceiling.P_last_cell > P_last_cell_target * (1.0 + rtol):
                # Even at the choke ceiling, forward-march P_last_cell
                # exceeds the target P_out. The chain is device-choke-
                # limited and the downstream pipe owns its own outlet
                # (exit-sonic anchor) / inlet (backward march). Hand the
                # whole choked↔subcritical decision to the unified
                # bottleneck handler, which applies the consistency rule
                # P_first vs P_crit (see _mode1_pipe_bottleneck and
                # CLAUDE.md "Pressure terminology"). ``mdot_at_ceiling`` is
                # the cushioned device-choke mdot (the bare ceiling sits
                # just above mdot_max at the operating state_up and would
                # raise OverChokedError); ``walkdown_last_good_mdot`` is the
                # wide-bracket lower bound for the subcritical F = 0 solve.
                P_throat_at_ceiling: float | None = None
                for _el in r_ceiling.results:
                    if isinstance(_el, DeviceResult):
                        try:
                            P_throat_at_ceiling = float(_el.throat.P)
                        except (AttributeError, TypeError, ValueError):
                            P_throat_at_ceiling = None
                        break
                return _mode1_pipe_bottleneck(
                    chain, working_fluid, base_fluid,
                    P_in, T_in, P_last_cell_target,
                    mdot_device=mdot_at_ceiling,
                    mdot_lo=walkdown_last_good_mdot,
                    rtol=rtol,
                    cancel_event=cancel_event,
                    boundary_conditions=boundary_conditions,
                    solver_options=solver_options,
                    t0=t0,
                    device_throat_P=float(P_throat_at_ceiling)
                    if P_throat_at_ceiling is not None else None,
                    **ivp_kwargs,
                )

            # Target IS reachable in (walkdown_last_good_mdot,
            # mdot_at_ceiling) via subcritical forward march. Tighten
            # the bracket and fall through to the main brentq. Use the
            # pre-bisect feasible probe as the lower endpoint — the
            # bisect refined ``last_good_mdot`` toward the ceiling,
            # which would collapse the bracket to two near-ceiling
            # probes with the same-sign objective.
            mdot_lo = walkdown_last_good_mdot
            mdot_hi = mdot_at_ceiling
            f_lo = walkdown_last_good.P_last_cell - P_last_cell_target
            f_hi = r_ceiling.P_last_cell - P_last_cell_target

        else:
            # Fanno-bottleneck path: f_hi sentinel, f_lo non-sentinel.
            # Bisect to find mdot_critical between mdot_lo (good) and
            # mdot_hi (choked).
            m_lo, m_hi = mdot_lo, mdot_hi
            for _ in range(40):
                m_mid = 0.5 * (m_lo + m_hi)
                if (m_hi - m_lo) < max(rtol * m_hi, 1e-6):
                    break
                f_mid = _obj(m_mid)
                if f_mid == _CHOKED_SENTINEL:
                    m_hi = m_mid
                else:
                    m_lo = m_mid
            mdot_critical = m_lo
            r_crit = _march(mdot_critical)
            P_last_cell_crit = r_crit.P_last_cell

            if P_last_cell_crit > P_last_cell_target * (1.0 + rtol):
                # At choke-limited mdot, P_last_cell is still above target →
                # target is below the choke-limited minimum and cannot be reached.
                r_crit.choked = True
                r_crit.choke_diagnostics = last_choke_diag or {"kind": "boundary"}
                raise BVPChoked(
                    f"Chain choked at mdot_critical={mdot_critical:.4f} kg/s; "
                    f"target P_last_cell={P_last_cell_target/1e5:.3f} bara is "
                    f"unreachable. Choke diagnostic: {last_choke_diag}",
                    mdot_critical=mdot_critical,
                    result=r_crit,
                )

            # Target IS reachable below mdot_critical — tighten the upper
            # bracket and continue with brentq on the smooth branch.
            mdot_hi = mdot_critical
            f_hi = P_last_cell_crit - P_last_cell_target

    if f_lo * f_hi > 0:
        raise BVPNotBracketedError(
            f"Cannot bracket mdot: f({mdot_lo:.4f})={f_lo:.1f}, "
            f"f({mdot_hi:.4f})={f_hi:.1f}."
        )

    try:
        mdot_sol = brentq(_obj, mdot_lo, mdot_hi, rtol=rtol, maxiter=50)
    except ValueError as exc:
        raise BVPNotBracketedError(f"brentq failed: {exc}") from exc

    final = _march(mdot_sol)
    if (
        final.choked
        and abs(final.P_last_cell - P_last_cell_target)
        > rtol * P_last_cell_target
    ):
        raise BVPChoked(
            f"Mode 1: chain choked at mdot_sol={mdot_sol:.4f} kg/s.",
            mdot_critical=mdot_sol,
            result=final,
        )
    # Subsonic-outlet solution (P_last_cell == P_out). Stamp the
    # reservoir BC + outlet state. outlet_sonic is pressure-based (True
    # only if P_last_cell strictly exceeds P_out). The device may be
    # subcritical (kind="subcritical") or choked-but-BC-satisfied.
    final.P_out = P_last_cell_target
    final.outlet_sonic = _outlet_sonic_flag(
        final.P_last_cell, P_last_cell_target, rtol,
    )
    if not final.choked and final.choke_diagnostics is None:
        final.choke_diagnostics = {"kind": "subcritical"}
    return final


def _build_minimal_choked_result(
    chain: ChainSpec,
    P_in: float,
    T_in: float,
    mdot: float,
    boundary_conditions: dict,
    solver_options: dict,
    t0: float,
    choke_diag: dict | None,
) -> ChainResult:
    """Build a barely-populated ChainResult for BVPChoked when even mdot_lo chokes."""
    return ChainResult(
        chain=chain,
        results=[],
        mdot=mdot,
        P_in=P_in,
        P_last_cell=float("nan"),
        T_in=T_in,
        T_out=float("nan"),
        boundary_conditions=boundary_conditions,
        solver_options=solver_options,
        elapsed=time.time() - t0,
        choked=True,
        choke_diagnostics=choke_diag,
    )


# ---------------------------------------------------------------------------
# Mode 3 — brentq on P_in
# ---------------------------------------------------------------------------


def _mode3_brentq(
    chain: ChainSpec,
    working_fluid: "FluidEOSBase",
    base_fluid: "GERGFluid",
    T_in: float,
    P_last_cell_target: float,
    mdot: float,
    *,
    P_in_bracket: tuple[float, float] | None,
    rtol: float,
    progress_callback: Callable[[str, float], None] | None,
    cancel_event: object | None,
    boundary_conditions: dict,
    solver_options: dict,
    t0: float,
    **ivp_kwargs,
) -> ChainResult:
    """Mode 3 — brentq on ``P_in`` to match ``P_last_cell_target`` at ``mdot``.

    On bracket exhaustion (no feasible ``P_in``), raise
    :class:`OverChokedError` with a Mode-3-specific message identifying
    the infeasibility scope.
    """
    if P_in_bracket is None:
        # Default upper bound at 10× the target — wide enough for typical
        # flows without pushing the chain march into EOS-out-of-range /
        # device over-choke at extreme stagnation. Walk-down below handles
        # the edge case where this still lands in an infeasible probe region.
        P_in_lo, P_in_hi = (
            P_last_cell_target * 1.01,
            P_last_cell_target * 10.0,
        )
    else:
        P_in_lo, P_in_hi = P_in_bracket

    last_choke: OverChokedError | None = None

    def _march(P_in_try: float) -> ChainResult:
        return _chain_forward_march(
            chain, working_fluid, base_fluid,
            P_in_try, T_in, mdot,
            boundary_conditions=boundary_conditions,
            solver_options=solver_options,
            t0=t0,
            cancel_event=cancel_event,
            **ivp_kwargs,
        )

    def _obj(P_in_try: float) -> float:
        nonlocal last_choke
        try:
            r = _march(P_in_try)
            if r.choked:
                return _CHOKED_SENTINEL
            return r.P_last_cell - P_last_cell_target
        except OverChokedError as exc:
            last_choke = exc
            return _CHOKED_SENTINEL
        except (ChokeReached, SegmentConvergenceError):
            return _CHOKED_SENTINEL
        except (EOSOutOfRange, EOSTwoPhase, HEMConsistencyError):
            # Infeasible probe — BC drove the throat solve outside GERG's
            # valid region. Treat as choked-equivalent for bracket purposes.
            return _CHOKED_SENTINEL

    # Phase 1: establish a finite-valid P_in_hi with f_hi > 0.
    # The initial P_in_hi may land in a region where the chain march
    # encounters EOS-out-of-range (extreme stagnation cooling),
    # over-choked devices at huge P_stag, or pipe Fanno chokes.
    # Walk P_in_hi DOWN geometrically until a valid probe is found.
    f_hi = _obj(P_in_hi)
    n_walk_down = 0
    while f_hi == _CHOKED_SENTINEL and n_walk_down < 30:
        P_in_hi = math.sqrt(max(P_in_lo, 1.0) * P_in_hi)
        if P_in_hi < P_in_lo * 1.001:
            break
        f_hi = _obj(P_in_hi)
        n_walk_down += 1
    if f_hi == _CHOKED_SENTINEL:
        _raise_mode3_infeasible(mdot, P_in_lo, P_in_hi, last_choke)

    # If the (now finite-valid) P_in_hi is still below the target,
    # extend upward cautiously, backing off if extension lands in
    # another sentinel band.
    while f_hi < 0:
        P_in_hi_new = P_in_hi * 3.0
        if P_in_hi_new > 1000.0 * P_last_cell_target:
            raise BVPNotBracketedError(
                f"Mode 3: P_last_cell_target="
                f"{P_last_cell_target/1e5:.3f} bara "
                f"unreachable even at P_in={P_in_hi/1e5:.3f} bara."
            )
        f_new = _obj(P_in_hi_new)
        if f_new == _CHOKED_SENTINEL:
            # Extension hit an infeasible band — stop here with last valid hi
            break
        P_in_hi = P_in_hi_new
        f_hi = f_new
    if f_hi < 0:
        raise BVPNotBracketedError(
            f"Mode 3: cannot extend P_in_hi above target; "
            f"f({P_in_hi/1e5:.3f}b)={f_hi:.1f}"
        )

    # Phase 2: narrow toward the choke boundary by geometric-mean bisection.
    # Maintain (P_in_lo, P_in_hi) such that f(P_in_lo) is sentinel or negative
    # and f(P_in_hi) is positive. At each midpoint:
    #   - if mid chokes  → tighten P_in_lo upward (still need negative side)
    #   - if mid is negative → bracket found, bail to brentq
    #   - if mid is positive → tighten P_in_hi downward (closer to target)
    f_lo = _obj(P_in_lo)
    n_iter = 0
    while f_lo == _CHOKED_SENTINEL and n_iter < 50:
        # Geometric mean works across orders of magnitude (P_in might span
        # 1 → 1000 bara).
        mid = math.sqrt(max(P_in_lo, 1.0) * P_in_hi)
        f_mid = _obj(mid)
        n_iter += 1
        if f_mid == _CHOKED_SENTINEL:
            P_in_lo = mid
        elif f_mid < 0:
            P_in_lo = mid
            f_lo = f_mid
            break
        else:
            # f_mid > 0: midpoint is already above the target. Tighten hi
            # toward this point and continue searching for the negative
            # side from below.
            P_in_hi = mid
            f_hi = f_mid

    if f_lo == _CHOKED_SENTINEL:
        # Could not find any sub-target non-choked P_in — the choke
        # boundary lies above the target. Mode 3 infeasible.
        _raise_mode3_infeasible(mdot, P_in_lo, P_in_hi, last_choke)

    if f_lo * f_hi > 0:
        raise BVPNotBracketedError(
            f"Mode 3: cannot bracket P_in after walk. "
            f"f({P_in_lo/1e5:.3f}b)={f_lo:.1f}, "
            f"f({P_in_hi/1e5:.3f}b)={f_hi:.1f}"
        )

    try:
        P_in_sol = brentq(_obj, P_in_lo, P_in_hi, rtol=rtol, maxiter=50)
    except ValueError as exc:
        raise BVPNotBracketedError(f"Mode 3 brentq failed: {exc}") from exc

    return _march(P_in_sol)


def _raise_mode3_infeasible(
    mdot: float,
    P_in_lo: float,
    P_in_hi: float,
    last_choke: "OverChokedError | None",
) -> None:
    """Raise :class:`OverChokedError` with a Mode-3-specific message."""
    if last_choke is not None:
        raise OverChokedError(
            f"Mode 3 infeasible: mdot={mdot:.4f} kg/s exceeds chain "
            f"capacity at any P_in in [{P_in_lo/1e5:.3f}, "
            f"{P_in_hi/1e5:.3f}] bara; chokes at Device "
            f"'{last_choke.device_name}' (chain index "
            f"{last_choke.device_index}) with max mdot="
            f"{last_choke.max_mdot:.4f} kg/s at P_stag="
            f"{last_choke.P_stag/1e5:.3f} bara (highest P_in tried).",
            device_index=last_choke.device_index,
            device_name=last_choke.device_name,
            max_mdot=last_choke.max_mdot,
            attempted_mdot=mdot,
            P_stag=last_choke.P_stag,
            T_stag=last_choke.T_stag,
        )
    raise OverChokedError(
        f"Mode 3 infeasible: chain chokes at all probes in "
        f"[{P_in_lo/1e5:.3f}, {P_in_hi/1e5:.3f}] bara for mdot="
        f"{mdot:.4f} kg/s, but the choke source was a pipe Fanno-"
        "asymptote rather than a device (no diagnostic available).",
        device_index=None, device_name="",
        max_mdot=float("nan"),
        attempted_mdot=mdot,
        P_stag=float("nan"), T_stag=float("nan"),
    )

"""Multi-element BVP solver: chains of :class:`Pipe` and :class:`Device`.

A :class:`ChainSpec` is a sequence ``Pipe → [Device → Pipe]*`` from
inlet to outlet. :func:`solve_chain` is a three-mode dispatcher: exactly
two of ``(P_in, P_out, mdot)`` must be given; ``T_in`` is always required
because temperature propagates forward.

For the PSV-on-tank scenario (no upstream pipe) call
:meth:`Device.from_stagnation` directly — that's a single-device
calculation, not a chain.
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Union

from scipy.optimize import brentq

from .device import Device, DeviceResult
from .errors import (
    BVPChoked,
    BVPNotBracketedError,
    ChokeReached,
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
    """Result of :func:`solve_chain` — per-element states + chain summary."""

    chain: ChainSpec
    results: list[Union["PipeResult", DeviceResult]]
    mdot: float
    P_in: float
    P_out: float
    T_in: float
    T_out: float
    boundary_conditions: dict
    solver_options: dict = field(default_factory=dict)
    elapsed: float = 0.0
    choked: bool = False
    choke_diagnostics: dict | None = None

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
    **ivp_kwargs,
) -> ChainResult:
    """Forward chain march at fixed ``mdot``, returning a complete
    :class:`ChainResult`.

    For each pipe element, calls :func:`march_ivp` with the current state.
    For each device, calls :func:`_device_solve_at_mdot` which itself
    brent-iterates ``P_back`` so the device's HEM throat passes the
    system ``mdot``.

    Raises
    ------
    OverChokedError
        Propagated from a device that cannot pass ``mdot``. The chain
        BVP modes wrap this in their respective sentinel/mapping logic.
    ChokeReached, IntegrationCapExceeded, SegmentConvergenceError
        Propagated from per-pipe ``march_ivp``.
    """
    from .geometry import Pipe
    from .results import PipeResult
    from .solver import march_ivp

    results: list[Union[PipeResult, DeviceResult]] = []
    P_current = P_in
    T_current = T_in
    choked = False
    choke_diagnostics: dict | None = None

    for i, element in enumerate(chain.elements):
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
                # cannot be reached at this mdot.
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
            inlet = device_result.transition.state_inlet
            P_current = inlet.P
            T_current = inlet.T
        else:
            raise TypeError(
                f"Unknown ChainSpec element at index {i}: "
                f"{type(element).__name__}"
            )

    return ChainResult(
        chain=chain,
        results=results,
        mdot=mdot,
        P_in=P_in,
        P_out=P_current,
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
    """Three-mode chain BVP solver. See module docstring for overview."""
    given = [
        name for name, val in (("P_in", P_in), ("P_out", P_out), ("mdot", mdot))
        if val is not None
    ]
    if len(given) != 2:
        raise ValueError(
            "solve_chain requires exactly two of (P_in, P_out, mdot); "
            f"got {given}"
        )

    t0 = time.time()
    bcs = {"given": tuple(given), "T_in": T_in}

    # Single-pipe Mode 1 fast path: delegate to the legacy
    # _bvp_single_pipe_mdot so existing solve_for_mdot test behavior is
    # preserved exactly (bracket heuristics, _find_critical_mdot, etc.).
    if (
        "P_in" in given
        and "P_out" in given
        and len(chain.elements) == 1
    ):
        from .solver import _bvp_single_pipe_mdot

        pipe = chain.elements[0]
        pipe_result = _bvp_single_pipe_mdot(
            pipe, fluid, P_in, T_in, P_out,
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
        return ChainResult(
            chain=chain,
            results=[pipe_result],
            mdot=pipe_result.mdot,
            P_in=P_in,
            P_out=float(pipe_result.P[-1]),
            T_in=T_in,
            T_out=float(pipe_result.T[-1]),
            boundary_conditions=bcs,
            solver_options=dict(pipe_result.solver_options or {}),
            elapsed=time.time() - t0,
            choked=bool(pipe_result.choked),
            choke_diagnostics=(
                {"kind": "fanno_pipe", "element_index": 0,
                 "x_choke": pipe_result.x_choke}
                if pipe_result.choked else None
            ),
        )

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
            P_known = P_in if P_in is not None else P_out
            P_other = P_out if P_out is not None else P_in
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
        return _mode1_brentq(
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

    # Mode 3 — (P_out, mdot) given.
    return _mode3_brentq(
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


# ---------------------------------------------------------------------------
# Mode 1 — brentq on mdot (multi-element only; single-pipe takes fast path)
# ---------------------------------------------------------------------------


def _mode1_brentq(
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
            first_pipe, working_fluid, P_in, T_in, P_out_target,
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

    def _obj(mdot_try: float) -> float:
        nonlocal last_good, last_good_mdot, last_choked_mdot, last_choke_diag
        try:
            r = _march(mdot_try)
            if r.choked:
                last_choked_mdot = mdot_try
                last_choke_diag = r.choke_diagnostics
                return _CHOKED_SENTINEL
            last_good = r
            last_good_mdot = mdot_try
            return r.P_out - P_out_target
        except OverChokedError as exc:
            last_choked_mdot = mdot_try
            last_choke_diag = {
                "kind": "device_over_choked",
                "element_index": exc.device_index,
                "element_name": exc.device_name,
                "max_mdot": exc.max_mdot,
                "P_stag": exc.P_stag,
                "T_stag": exc.T_stag,
            }
            return _CHOKED_SENTINEL
        except (ChokeReached, SegmentConvergenceError):
            last_choked_mdot = mdot_try
            last_choke_diag = {"kind": "fanno_pipe_during_march"}
            return _CHOKED_SENTINEL
        except (EOSOutOfRange, EOSTwoPhase, HEMConsistencyError) as exc:
            # The BC probe drove the throat-solve isentrope into a region
            # GERG cannot evaluate (near-critical, sub-dew, etc.). Treat as
            # infeasible-probe; brentq adjusts away from it.
            last_choked_mdot = mdot_try
            last_choke_diag = {
                "kind": "eos_infeasible",
                "exception": type(exc).__name__,
                "message": str(exc)[:200],
            }
            return _CHOKED_SENTINEL

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
    # P_out is reachable below it (mirrors the legacy
    # _bvp_single_pipe_mdot pattern).
    if f_hi == _CHOKED_SENTINEL:
        if f_lo == _CHOKED_SENTINEL:
            # Entire bracket is choked — infeasible at any probed mdot.
            raise BVPChoked(
                "Chain choked throughout the mdot bracket; "
                f"no feasible mdot in [{mdot_lo:.3f}, {mdot_hi:.3f}] kg/s.",
                mdot_critical=mdot_lo,
                result=last_good or _build_minimal_choked_result(
                    chain, P_in, T_in, mdot_lo,
                    boundary_conditions, solver_options, t0,
                    last_choke_diag,
                ),
            )
        # Bisect to find mdot_critical between mdot_lo (good) and mdot_hi (choked)
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
        P_out_crit = r_crit.P_out

        if P_out_crit > P_out_target * (1.0 + rtol):
            # At choke-limited mdot, P_out is still above target → target
            # is below the choke-limited minimum and cannot be reached.
            r_crit.choked = True
            r_crit.choke_diagnostics = last_choke_diag or {"kind": "boundary"}
            raise BVPChoked(
                f"Chain choked at mdot_critical={mdot_critical:.4f} kg/s; "
                f"target P_out={P_out_target/1e5:.3f} bara is unreachable. "
                f"Choke diagnostic: {last_choke_diag}",
                mdot_critical=mdot_critical,
                result=r_crit,
            )

        # Target IS reachable below mdot_critical — tighten the upper
        # bracket and continue with brentq on the smooth branch.
        mdot_hi = mdot_critical
        f_hi = P_out_crit - P_out_target

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
    if final.choked and abs(final.P_out - P_out_target) > rtol * P_out_target:
        raise BVPChoked(
            f"Mode 1: chain choked at mdot_sol={mdot_sol:.4f} kg/s.",
            mdot_critical=mdot_sol,
            result=final,
        )
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
        P_out=float("nan"),
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
    P_out_target: float,
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
    """Mode 3 — brentq on ``P_in`` to match ``P_out_target`` at given ``mdot``.

    On bracket exhaustion (no feasible ``P_in``), raise
    :class:`OverChokedError` with a Mode-3-specific message identifying
    the infeasibility scope.
    """
    if P_in_bracket is None:
        # Default upper bound at 10×P_out — wide enough for typical flows
        # without pushing the chain march into EOS-out-of-range / device
        # over-choke at extreme stagnation. Walk-down below handles the
        # edge case where this still lands in an infeasible probe region.
        P_in_lo, P_in_hi = P_out_target * 1.01, P_out_target * 10.0
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
            return r.P_out - P_out_target
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
        if P_in_hi_new > 1000.0 * P_out_target:
            raise BVPNotBracketedError(
                f"Mode 3: P_out_target={P_out_target/1e5:.3f} bara "
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

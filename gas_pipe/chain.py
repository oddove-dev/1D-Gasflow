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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Union

from .device import Device, DeviceResult

if TYPE_CHECKING:
    from .eos import GERGFluid
    from .geometry import Pipe
    from .results import PipeResult


ChainElement = Union["Pipe", Device]
"""Type alias for the elements a :class:`ChainSpec` may contain."""


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

        # Local import to avoid a circular dependency between chain.py
        # and geometry.py (which already imports nothing from chain).
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
        """All :class:`Pipe` elements in order."""
        from .geometry import Pipe

        return [el for el in self.elements if isinstance(el, Pipe)]

    @property
    def devices(self) -> list[Device]:
        """All :class:`Device` elements in order."""
        return [el for el in self.elements if isinstance(el, Device)]


@dataclass
class ChainResult:
    """Result of :func:`solve_chain` — per-element states + chain summary.

    ``results`` is parallel to ``chain.elements``: each entry is the
    :class:`PipeResult` or :class:`DeviceResult` for that element.
    """

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
        """All :class:`PipeResult` entries from ``results``."""
        from .results import PipeResult

        return [r for r in self.results if isinstance(r, PipeResult)]

    @property
    def device_results(self) -> list[DeviceResult]:
        """All :class:`DeviceResult` entries from ``results``."""
        return [r for r in self.results if isinstance(r, DeviceResult)]


def solve_chain(
    chain: ChainSpec,
    fluid: "GERGFluid",
    T_in: float,
    *,
    P_in: float | None = None,
    P_out: float | None = None,
    mdot: float | None = None,
    mdot_bracket: tuple[float, float] | None = None,
    P_in_bracket: tuple[float, float] | None = None,
    rtol: float = 1e-5,
    progress_callback: Callable[[str, float], None] | None = None,
    cancel_event: "object | None" = None,
    eos_mode: str | None = None,
    table_n_P: int = 50,
    table_n_T: int = 50,
    P_range_override: tuple[float, float] | None = None,
    T_range_override: tuple[float, float] | None = None,
    **ivp_kwargs,
) -> ChainResult:
    """Three-mode chain BVP solver.

    Exactly two of ``(P_in, P_out, mdot)`` must be provided; ``T_in`` is
    always required because temperature propagates forward regardless of
    which pressure is the unknown.

    Modes
    -----
    Mode 1 — ``(P_in, P_out)`` → ``mdot``
        ``brentq`` on ``mdot`` with forward chain-march per probe.
        ``OverChokedError`` and ``ChokeReached`` map to a choke sentinel;
        a final ``BVPChoked`` is raised at the bracket boundary carrying
        ``mdot_critical`` and the choked :class:`ChainResult`.

    Mode 2 — ``(P_in, mdot)`` → ``P_out``
        Single forward chain-march. ``P_out`` is read from the final
        pipe-outlet state. ``OverChokedError`` bubbles directly to the
        caller with full per-march diagnostics.

    Mode 3 — ``(P_out, mdot)`` → ``P_in``
        ``brentq`` on ``P_in`` with forward chain-march per probe.
        Bracket exhaustion (no feasible ``P_in``) raises
        :class:`OverChokedError` with a Mode-3-specific message that
        identifies the infeasibility (which Device chokes, max mdot at
        the highest ``P_in`` tried). The exception **type** is the same;
        only the message differs from the Mode 2 form.

    Parameters
    ----------
    chain : ChainSpec
        Pipe/Device sequence to solve.
    fluid : GERGFluid
        Direct EOS handle. ``Device.solve`` calls ``hem_throat`` and
        ``props_Ps_via_jt`` internally, both of which live on
        :class:`GERGFluid`. ``TabulatedFluid`` is used (when
        ``eos_mode="table"``) only for the per-pipe march, not for
        device internals.
    T_in : float
        Inlet temperature [K].
    P_in, P_out, mdot : float or None
        Exactly two must be given.
    mdot_bracket : tuple or None
        Brackets the Mode 1 brentq search.
    P_in_bracket : tuple or None
        Brackets the Mode 3 brentq search.
    rtol : float
        Relative tolerance on the matched boundary.
    ``**ivp_kwargs`` : forwarded to :func:`march_ivp` for each pipe march.

    Returns
    -------
    ChainResult

    Raises
    ------
    ValueError
        If not exactly two of ``(P_in, P_out, mdot)`` are given.
    BVPChoked
        Mode 1: chain choked at the brentq bracket boundary.
    OverChokedError
        Mode 2: a device cannot pass ``mdot`` at the encountered
        stagnation. Mode 3: no ``P_in`` in the bracket admits ``mdot``
        without choking a device.
    """
    raise NotImplementedError("solve_chain: filled in commit 2")

"""Plotting and post-processing diagnostics for PipeResult."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .chain import ChainResult
    from .eos import GERGFluid
    from .geometry import Pipe
    from .results import PipeResult

logger = logging.getLogger(__name__)


def plot_profile(
    result: "PipeResult",
    save_path: str | None = None,
    show: bool = False,
    fig=None,
) -> object:
    """Plot a 2×3 grid of flow profiles.

    Layout:
      (0,0) P vs x
      (0,1) T vs x
      (0,2) ρ (left) and Z (right) vs x
      (1,0) u and a vs x
      (1,1) Mach with shaded zones and M=1 line
      (1,2) Darcy friction factor f vs x

    Parameters
    ----------
    result : PipeResult
    save_path : str or None
        If provided, save figure to this path.
    show : bool
        If True, call plt.show().
    fig : matplotlib.figure.Figure or None
        If provided, draw into existing figure (used by GUI).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    import matplotlib.pyplot as plt

    x = result.x
    L = float(x[-1])
    choked = result.choked
    x_choke = result.x_choke

    if fig is None:
        fig = plt.figure(figsize=(12, 7))
    else:
        fig.clear()

    axes = fig.subplots(2, 3)

    # Choke vertical line helper
    def _vchoke(ax, color="red", lw=1.2):
        if choked and x_choke is not None:
            ax.axvline(x_choke, color=color, lw=lw, ls="--", alpha=0.7, label=f"Choke x={x_choke:.1f} m")

    # (0,0) Pressure
    ax = axes[0, 0]
    ax.plot(x, result.P / 1e5, "b-", lw=1.5)
    _vchoke(ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("P [bara]")
    ax.set_title("Pressure")
    ax.grid(True, alpha=0.3)

    # (0,1) Temperature — overlay the dew-point curve where finite so the
    # viewer can see at a glance whether the gas T crosses below T_dew.
    ax = axes[0, 1]
    ax.plot(x, result.T - 273.15, "r-", lw=1.5, label="T (gas)")
    has_T_dew = (
        getattr(result, "T_dew", None) is not None
        and len(result.T_dew) == len(x)
        and bool(np.any(np.isfinite(result.T_dew)))
    )
    if has_T_dew:
        T_dew_C = result.T_dew - 273.15
        finite = np.isfinite(T_dew_C)
        ax.plot(
            x[finite], T_dew_C[finite],
            color="orange", linestyle="--", lw=1.5, label="T_dew",
        )
        ax.legend(loc="best", fontsize=8)
    _vchoke(ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("T [°C]")
    ax.set_title("Temperature")
    ax.grid(True, alpha=0.3)

    # (0,2) Density + Z
    ax = axes[0, 2]
    ax2 = ax.twinx()
    line1, = ax.plot(x, result.rho, "g-", lw=1.5, label="ρ")
    line2, = ax2.plot(x, result.Z, "m--", lw=1.5, label="Z")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("ρ [kg/m³]")
    ax2.set_ylabel("Z [-]")
    ax.set_title("Density & Compressibility")
    lines = [line1, line2]
    ax.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) u and a
    ax = axes[1, 0]
    ax.plot(x, result.u, "b-", lw=1.5, label="u (velocity)")
    ax.plot(x, result.a, "r--", lw=1.5, label="a (sound speed)")
    _vchoke(ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Velocity & Sound Speed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Mach with shaded zones
    ax = axes[1, 1]
    M = result.M
    ax.fill_between(x, 0, 1, where=M < 0.3, color="green", alpha=0.08, label="M<0.3")
    ax.fill_between(x, 0, 1, where=(M >= 0.3) & (M < 0.6), color="yellow", alpha=0.12)
    ax.fill_between(x, 0, 1, where=(M >= 0.6) & (M < 0.9), color="orange", alpha=0.15)
    ax.fill_between(x, 0, 1, where=M >= 0.9, color="red", alpha=0.15)
    ax.plot(x, M, "k-", lw=1.8, label="Mach")
    ax.axhline(1.0, color="red", lw=1.2, ls="-", alpha=0.7)
    # Metastable shading: highlight where T < T_dew so the viewer can see
    # which part of the curve is operating outside strict single-phase
    # domain. Multiple contiguous runs are shaded but only the first run
    # gets a legend entry to avoid duplicate "Metastable" tokens.
    if getattr(result, "had_metastable", False):
        in_meta = np.asarray(result.metastable_mask, dtype=bool)
        diffs = np.diff(in_meta.astype(np.int8))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if in_meta.size > 0 and in_meta[0]:
            starts = np.insert(starts, 0, 0)
        if in_meta.size > 0 and in_meta[-1]:
            ends = np.append(ends, in_meta.size)
        last_x = len(x) - 1
        for j, (s, e) in enumerate(zip(starts, ends)):
            label = "Metastable" if j == 0 else None
            ax.axvspan(
                float(x[s]), float(x[min(e, last_x)]),
                alpha=0.15, color="gold", label=label,
            )
    _vchoke(ax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Mach [-]")
    ax.set_title("Mach Number")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Friction factor (step plot over segments)
    ax = axes[1, 2]
    if len(result.f) > 0 and len(result.x) > 1:
        x_mid = 0.5 * (result.x[:-1] + result.x[1:])
        ax.step(x_mid, result.f, "k-", lw=1.2, where="mid")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("f (Darcy) [-]")
    ax.set_title("Friction Factor")
    ax.grid(True, alpha=0.3)

    # Suptitle: build from independent CHOKED / TWO-PHASE flags so both
    # show when applicable. Falls back to the subsonic single-phase label
    # only when neither condition fired.
    title_parts: list[str] = []
    title_color = "black"
    if choked and x_choke is not None:
        pct = x_choke / L * 100 if L > 0 else 0.0
        title_parts.append(
            f"CHOKED at x = {x_choke:.2f} m ({pct:.1f}% of pipe length)"
        )
        title_color = "red"
    if getattr(result, "had_metastable", False) and result.x_dewpoint_crossing is not None:
        title_parts.append(
            f"TWO-PHASE region from x = {result.x_dewpoint_crossing:.2f} m"
        )
    if title_parts:
        fig.suptitle(" — ".join(title_parts), fontsize=11, color=title_color)
    else:
        fig.suptitle("Flow Profile — Subsonic", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Profile plot saved to %s", save_path)

    if show:
        import matplotlib.pyplot as _plt
        _plt.show()

    return fig


def plot_chain_profile(
    result: "ChainResult",
    fig=None,
    show_device_markers: bool = True,
) -> object:
    """Plot a chain-aware flow profile spanning multiple pipes and devices.

    Composition strategy:

    - For a single-element chain (one :class:`Pipe`, no :class:`Device`),
      delegates directly to :func:`plot_profile` so the output is
      byte-identical to the pre-chain GUI.
    - For multi-element chains, builds a synthetic merged-result by
      concatenating each :class:`PipeResult`'s station arrays with the
      cumulative chain-x offset, then calls :func:`plot_profile` on the
      merged structure. Vertical purple dashed lines mark each Device
      location, with a name label above the top axis row, when
      ``show_device_markers`` is True.

    Devices are zero-length in the chain coordinate — they sit at the
    boundary between two pipe segments and introduce a discontinuity in
    the property profiles.

    Parameters
    ----------
    result : ChainResult
    fig : matplotlib.figure.Figure or None
    show_device_markers : bool
        Toggle the vertical lines + labels for Device elements.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from types import SimpleNamespace

    from .device import DeviceResult
    from .results import PipeResult

    pipe_results = result.pipe_results
    if (
        len(result.results) == 1
        and isinstance(result.results[0], PipeResult)
    ):
        return plot_profile(result.results[0], fig=fig)

    # ---- Build merged station arrays + device positions ---------------
    x_segs: list[np.ndarray] = []
    P_segs: list[np.ndarray] = []
    T_segs: list[np.ndarray] = []
    rho_segs: list[np.ndarray] = []
    Z_segs: list[np.ndarray] = []
    u_segs: list[np.ndarray] = []
    a_segs: list[np.ndarray] = []
    M_segs: list[np.ndarray] = []
    f_segs: list[np.ndarray] = []
    T_dew_segs: list[np.ndarray] = []
    metastable_segs: list[np.ndarray] = []

    cum_x = 0.0
    choked = False
    x_choke_combined: float | None = None
    had_meta = False
    x_dew_cross: float | None = None
    device_positions: list[tuple[float, str]] = []

    for i, el_result in enumerate(result.results):
        if isinstance(el_result, DeviceResult):
            # Device sits at cum_x; record for marker overlay.
            name = el_result.throat.A_vc  # no name on ThroatState; fall back
            # Pull the configured Device name from the spec for the label.
            device_obj = result.chain.elements[i]
            label = (
                getattr(device_obj, "name", "") or f"D{i + 1}"
            )
            device_positions.append((cum_x, label))
            continue
        pr = el_result  # PipeResult
        x_shift = np.asarray(pr.x, dtype=float) + cum_x
        x_segs.append(x_shift)
        P_segs.append(np.asarray(pr.P, dtype=float))
        T_segs.append(np.asarray(pr.T, dtype=float))
        rho_segs.append(np.asarray(pr.rho, dtype=float))
        Z_segs.append(np.asarray(pr.Z, dtype=float))
        u_segs.append(np.asarray(pr.u, dtype=float))
        a_segs.append(np.asarray(pr.a, dtype=float))
        M_segs.append(np.asarray(pr.M, dtype=float))
        # f is per-segment, length n-1 per pipe
        f_segs.append(np.asarray(pr.f, dtype=float))
        # T_dew may be missing or all-NaN for some pipes; pad with NaN to
        # match station count.
        T_dew_pipe = getattr(pr, "T_dew", None)
        if T_dew_pipe is not None and len(T_dew_pipe) == len(pr.x):
            T_dew_segs.append(np.asarray(T_dew_pipe, dtype=float))
        else:
            T_dew_segs.append(np.full(len(pr.x), np.nan))
        meta_mask = getattr(pr, "metastable_mask", None)
        if meta_mask is not None and len(meta_mask) == len(pr.x):
            metastable_segs.append(np.asarray(meta_mask, dtype=bool))
        else:
            metastable_segs.append(np.zeros(len(pr.x), dtype=bool))
        if pr.choked and pr.x_choke is not None:
            choked = True
            x_choke_combined = float(pr.x_choke) + cum_x
        if getattr(pr, "had_metastable", False):
            had_meta = True
            x_dew = getattr(pr, "x_dewpoint_crossing", None)
            if x_dew is not None and x_dew_cross is None:
                x_dew_cross = float(x_dew) + cum_x
        cum_x = float(x_shift[-1])

    # Stitch f-arrays with a NaN "bridge" segment between adjacent pipes —
    # the bridge represents the device gap, where friction is undefined.
    # This keeps ``len(f_combined) == len(x_combined) - 1`` (the invariant
    # plot_profile relies on for the ``step`` plot via ``x_mid``).
    if f_segs:
        bridged: list[np.ndarray] = [f_segs[0]]
        for seg in f_segs[1:]:
            bridged.append(np.array([np.nan]))
            bridged.append(seg)
        f_combined = np.concatenate(bridged)
    else:
        f_combined = np.array([])

    merged = SimpleNamespace(
        x=np.concatenate(x_segs),
        P=np.concatenate(P_segs),
        T=np.concatenate(T_segs),
        rho=np.concatenate(rho_segs),
        Z=np.concatenate(Z_segs),
        u=np.concatenate(u_segs),
        a=np.concatenate(a_segs),
        M=np.concatenate(M_segs),
        f=f_combined,
        T_dew=(
            np.concatenate(T_dew_segs)
            if any(np.any(np.isfinite(seg)) for seg in T_dew_segs)
            else None
        ),
        metastable_mask=np.concatenate(metastable_segs),
        had_metastable=had_meta,
        x_dewpoint_crossing=x_dew_cross,
        choked=choked,
        x_choke=x_choke_combined,
    )

    fig = plot_profile(merged, fig=fig)

    if show_device_markers and device_positions:
        for ax in fig.axes:
            for x_dev, label in device_positions:
                ax.axvline(
                    x_dev, color="#7d3c98", linestyle=":", linewidth=1.2,
                    alpha=0.7,
                )
        # Label devices above the top row of subplots.
        top_axes = [ax for ax in fig.axes if ax.get_subplotspec().rowspan.start == 0]
        if top_axes:
            for x_dev, label in device_positions:
                top_axes[0].annotate(
                    label,
                    xy=(x_dev, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color="#7d3c98",
                    ha="center",
                )

    return fig


def plot_plateau_sweep(
    P_last_cell_array: "np.ndarray",
    mdot_array: "np.ndarray",
    choked_flags: list[bool],
    save_path: str | None = None,
    fig=None,
) -> object:
    """Plot mass flow vs last-cell pressure (choke plateau diagnostic).

    Parameters
    ----------
    P_last_cell_array : array-like
        Target last-cell pressures [Pa] (per-pipe internal state — see
        CLAUDE.md "Pressure terminology").
    mdot_array : array-like
        Corresponding mass flow rates [kg/s].
    choked_flags : list[bool]
        Whether each point is choked.
    save_path : str or None
        Save path for figure.
    fig : matplotlib.figure.Figure or None
        Existing figure to draw into.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    P_last_cell = np.asarray(P_last_cell_array, dtype=float)
    mdot = np.asarray(mdot_array, dtype=float)
    choked = np.asarray(choked_flags, dtype=bool)

    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig.clear()
        ax = fig.add_subplot(111)

    # Non-choked: open circles
    mask_nc = ~choked & np.isfinite(mdot)
    if np.any(mask_nc):
        ax.scatter(P_last_cell[mask_nc] / 1e5, mdot[mask_nc], s=60, c="blue",
                   marker="o", facecolors="none", zorder=5, label="Subsonic")

    # Choked: red filled squares
    mask_c = choked & np.isfinite(mdot)
    if np.any(mask_c):
        ax.scatter(P_last_cell[mask_c] / 1e5, mdot[mask_c], s=60, c="red",
                   marker="s", zorder=5, label="Choked")

    # Plateau line
    choked_mdots = mdot[mask_c]
    if len(choked_mdots) > 0:
        mdot_crit = float(np.nanmean(choked_mdots))
        ax.axhline(mdot_crit, color="red", lw=1.5, ls="--", alpha=0.6,
                   label=f"Plateau ṁ_crit = {mdot_crit:.1f} kg/s")
        ax.set_title(f"Choke plateau detected at ṁ = {mdot_crit:.1f} kg/s")
    else:
        ax.set_title("Plateau sweep — no choke detected")

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("P_last_cell [bara]")
    ax.set_ylabel("ṁ [kg/s]")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plateau sweep plot saved to %s", save_path)

    return fig


def energy_balance_check(
    result: "PipeResult",
    fluid: "GERGFluid",
    pipe: "Pipe",
) -> float:
    """Check global energy balance.

    Returns relative residual of inlet stagnation enthalpy minus outlet
    stagnation enthalpy minus net heat in.

    Parameters
    ----------
    result : PipeResult
    fluid : GERGFluid
    pipe : Pipe

    Returns
    -------
    float
        Relative energy balance residual.
    """
    h_in = float(result.h[0])
    h_out = float(result.h[-1])
    u_in = float(result.u[0])
    u_out = float(result.u[-1])
    z_in = pipe.z(float(result.x[0]))
    z_out = pipe.z(float(result.x[-1]))

    H_in = h_in + 0.5 * u_in ** 2 + 9.80665 * z_in
    H_out = h_out + 0.5 * u_out ** 2 + 9.80665 * z_out

    Q_total = float(np.sum(result.q_seg))  # net heat per unit mass

    residual = (H_out - H_in) - Q_total
    scale = max(abs(H_in), abs(H_out), 1.0)
    return abs(residual) / scale

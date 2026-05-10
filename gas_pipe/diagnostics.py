"""Plotting and post-processing diagnostics for PipeResult."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .results import PipeResult
    from .eos import GERGFluid
    from .geometry import Pipe

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

    # (0,1) Temperature
    ax = axes[0, 1]
    ax.plot(x, result.T - 273.15, "r-", lw=1.5)
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

    if choked and x_choke is not None:
        pct = x_choke / L * 100 if L > 0 else 0.0
        fig.suptitle(f"CHOKED at x = {x_choke:.2f} m ({pct:.1f}% of pipe length)", fontsize=11, color="red")
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


def plot_plateau_sweep(
    P_out_array: "np.ndarray",
    mdot_array: "np.ndarray",
    choked_flags: list[bool],
    save_path: str | None = None,
    fig=None,
) -> object:
    """Plot mass flow vs outlet pressure (choke plateau diagnostic).

    Parameters
    ----------
    P_out_array : array-like
        Outlet pressures [Pa].
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

    P_out = np.asarray(P_out_array, dtype=float)
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
        ax.scatter(P_out[mask_nc] / 1e5, mdot[mask_nc], s=60, c="blue",
                   marker="o", facecolors="none", zorder=5, label="Subsonic")

    # Choked: red filled squares
    mask_c = choked & np.isfinite(mdot)
    if np.any(mask_c):
        ax.scatter(P_out[mask_c] / 1e5, mdot[mask_c], s=60, c="red",
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
    ax.set_xlabel("P_out [bara]")
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

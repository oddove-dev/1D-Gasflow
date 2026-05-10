"""Darcy-Weisbach friction factor correlations.

All functions return the Darcy friction factor (not Fanning).
"""
from __future__ import annotations

import math


def friction_laminar(Re: float) -> float:
    """Darcy friction factor in laminar flow.

    Parameters
    ----------
    Re : float
        Reynolds number. Valid for Re < 2300.

    Returns
    -------
    float
        f = 64 / Re.
    """
    return 64.0 / Re


def friction_chen(Re: float, eps_over_D: float) -> float:
    """Chen (1979) explicit approximation to Colebrook.

    Deviates < 0.3% from Colebrook over Re ∈ [1e4, 1e8], ε/D ∈ [1e-6, 0.05].

    Parameters
    ----------
    Re : float
        Reynolds number.
    eps_over_D : float
        Relative roughness ε/D.

    Returns
    -------
    float
        Darcy friction factor.
    """
    A = (eps_over_D ** 1.1098) / 2.8257 + (5.8506 / Re ** 0.8981)
    inner = eps_over_D / 3.7065 - (5.0452 / Re) * math.log10(A)
    inv_sqrt_f = -2.0 * math.log10(inner)
    return 1.0 / (inv_sqrt_f ** 2)


def friction_colebrook(
    Re: float,
    eps_over_D: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """Colebrook-White equation solved by Newton iteration.

    Uses Chen as the initial guess.

    Parameters
    ----------
    Re : float
        Reynolds number.
    eps_over_D : float
        Relative roughness.
    tol : float
        Convergence tolerance on 1/√f.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    float
        Darcy friction factor.
    """
    # Initial guess via Chen
    f0 = friction_chen(Re, eps_over_D)
    x = 1.0 / math.sqrt(f0)  # 1/sqrt(f)

    # Colebrook-White: 1/√f = -2·log10(ε/(3.7·D) + 2.51/(Re·√f))
    # With x = 1/√f: x = -2·log10(ε/(3.7·D) + 2.51·x/Re)
    rhs_const = eps_over_D / 3.7
    for _ in range(max_iter):
        inner = rhs_const + 2.51 * x / Re
        if inner <= 0:
            inner = 1e-15
        rhs = -2.0 * math.log10(inner)
        # d(rhs)/dx = -2 * (2.51/Re) / (inner * ln(10))
        d_rhs_dx = -2.0 * (2.51 / Re) / (inner * math.log(10.0))
        # Newton: F(x) = x - rhs = 0, F'(x) = 1 - d_rhs/dx
        F = x - rhs
        dF = 1.0 - d_rhs_dx
        if abs(dF) < 1e-15:
            break
        dx = -F / dF
        x += dx
        if abs(dx) < tol:
            break

    return 1.0 / (x ** 2)


def friction_blended(Re: float, eps_over_D: float, model: str = "chen") -> float:
    """Friction factor with smooth laminar/turbulent blend.

    Linear blend in 1/√f over the transition region 2000 < Re < 4000.
    This avoids discontinuities that trip Newton solvers in the segment code.

    Parameters
    ----------
    Re : float
        Reynolds number.
    eps_over_D : float
        Relative roughness.
    model : str
        Turbulent model: 'chen' or 'colebrook'.

    Returns
    -------
    float
        Darcy friction factor.
    """
    Re_lam = 2000.0
    Re_turb = 4000.0

    if Re < Re_lam:
        return friction_laminar(Re)

    if model == "colebrook":
        f_turb = friction_colebrook(Re, eps_over_D)
    else:
        f_turb = friction_chen(Re, eps_over_D)

    if Re >= Re_turb:
        return f_turb

    # Transition blend in 1/sqrt(f) space
    alpha = (Re - Re_lam) / (Re_turb - Re_lam)  # 0..1
    f_lam = friction_laminar(Re)
    inv_sqrt_lam = 1.0 / math.sqrt(f_lam)
    inv_sqrt_turb = 1.0 / math.sqrt(f_turb)
    inv_sqrt_blend = inv_sqrt_lam + alpha * (inv_sqrt_turb - inv_sqrt_lam)
    return 1.0 / (inv_sqrt_blend ** 2)


def darcy_friction(
    Re: float,
    eps_over_D: float,
    model: str = "blended",
) -> float:
    """Dispatch Darcy friction factor calculation.

    Parameters
    ----------
    Re : float
        Reynolds number.
    eps_over_D : float
        Relative roughness ε/D.
    model : str
        One of 'blended', 'chen', 'colebrook', 'laminar'.

    Returns
    -------
    float
        Darcy friction factor.
    """
    if model == "laminar":
        return friction_laminar(Re)
    if model == "chen":
        return friction_chen(Re, eps_over_D)
    if model == "colebrook":
        return friction_colebrook(Re, eps_over_D)
    if model == "blended":
        return friction_blended(Re, eps_over_D)
    raise ValueError(f"Unknown friction model: {model!r}. Choose from 'blended', 'chen', 'colebrook', 'laminar'.")

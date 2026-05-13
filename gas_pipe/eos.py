"""CoolProp GERG-2008 EOS wrapper with caching.

All thermodynamic property access in the solver goes through this module.
Nothing else imports CoolProp directly.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import CoolProp
import CoolProp.CoolProp as CP
import numpy as np
from CoolProp.CoolProp import AbstractState

from .errors import EOSOutOfRange, EOSTwoPhase

logger = logging.getLogger(__name__)

ALLOWED_COMPONENTS: tuple[str, ...] = (
    "Methane",
    "Ethane",
    "Propane",
    "n-Butane",
    "IsoButane",
    "n-Pentane",
    "IsoPentane",
    "n-Hexane",
    "Nitrogen",
    "CarbonDioxide",
    "HydrogenSulfide",
    "Water",
    "Oxygen",
    "Hydrogen",
    "Helium",
    "Argon",
)

_ALLOWED_SET: frozenset[str] = frozenset(ALLOWED_COMPONENTS)

# CoolProp input/output parameter codes used for partial derivatives
_iT = CoolProp.iT
_iP = CoolProp.iP
_iHmass = CoolProp.iHmass
_iDmass = CoolProp.iDmass  # density [kg/m^3], used for saturated-phase outputs


@dataclass(frozen=True, slots=True)
class FluidState:
    """Thermodynamic state at a single point.

    Parameters
    ----------
    P : float
        Pressure [Pa].
    T : float
        Temperature [K].
    rho : float
        Density [kg/m³].
    h : float
        Specific enthalpy [J/kg].
    s : float
        Specific entropy [J/kg/K].
    cp : float
        Isobaric heat capacity [J/kg/K].
    cv : float
        Isochoric heat capacity [J/kg/K].
    mu : float
        Dynamic viscosity [Pa·s].
    k : float
        Thermal conductivity [W/m/K].
    a : float
        Speed of sound [m/s].
    Z : float
        Compressibility factor [-].
    mu_JT : float
        Joule-Thomson coefficient [K/Pa].
    metastable : bool
        True if the state was produced by forcing gas-phase extrapolation
        across the dew curve (i.e. (P, T) is inside the two-phase dome
        but we evaluated as if it were still single-phase gas). False
        for normal single-phase states.
    """

    P: float
    T: float
    rho: float
    h: float
    s: float
    cp: float
    cv: float
    mu: float
    k: float
    a: float
    Z: float
    mu_JT: float
    metastable: bool = False


@runtime_checkable
class FluidEOSBase(Protocol):
    """Common interface for fluid EOS implementations.

    Both :class:`GERGFluid` (direct CoolProp calls) and
    :class:`TabulatedFluid` (interpolated lookup) satisfy this protocol.
    The solver and segment modules call these methods uniformly — no
    ``isinstance`` branching — so any future EOS backend that satisfies
    this interface plugs in without changes downstream.
    """

    def props(self, P: float, T: float) -> "FluidState":
        """Return :class:`FluidState` at (P, T)."""
        ...

    def props_Ph(self, P: float, h: float) -> "FluidState":
        """Return :class:`FluidState` at (P, h) — isenthalpic flash."""
        ...

    def props_Ph_via_jt(
        self, P_new: float, state_up: "FluidState"
    ) -> "FluidState":
        """JT-estimated isenthalpic flash (robust for HEOS mixtures)."""
        ...

    def dew_temperature(self, P: float) -> float | None:
        """Return dew T at P, or None if not computable."""
        ...

    def compute_lvf(self, P: float, h: float) -> float:
        """Liquid volume fraction at (P, h), or NaN if unavailable."""
        ...

    @property
    def composition(self) -> dict[str, float]:
        """Normalized mole-fraction composition."""
        ...

    @property
    def molar_mass(self) -> float:
        """Molar mass [kg/mol]."""
        ...

    @property
    def is_mixture(self) -> bool:
        """True if more than one component."""
        ...


def _normalize_composition(composition: dict[str, float]) -> dict[str, float]:
    """Validate and normalize a mole-fraction composition dict.

    Parameters
    ----------
    composition : dict[str, float]
        Component name → mole fraction. Must sum to 1.0 within 1e-6.

    Returns
    -------
    dict[str, float]
        Normalized composition.

    Raises
    ------
    ValueError
        If unknown components are present or fractions are non-positive.
    EOSOutOfRange
        If composition cannot be normalized.
    """
    unknown = set(composition) - _ALLOWED_SET
    if unknown:
        raise ValueError(
            f"Unknown component(s): {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_COMPONENTS)}"
        )
    for name, frac in composition.items():
        if frac < 0:
            raise ValueError(f"Negative mole fraction for {name}: {frac}")

    total = sum(composition.values())
    if total <= 0:
        raise ValueError("Composition sums to zero or negative.")
    deviation = abs(total - 1.0)
    if deviation > 1e-9:
        warnings.warn(
            f"Composition sums to {total:.9f}, normalizing. Deviation = {deviation:.2e}",
            stacklevel=3,
        )
    return {k: v / total for k, v in composition.items() if v > 0}


def _build_fluid_string(composition: dict[str, float]) -> str:
    """Build the HEOS fluid string for CoolProp."""
    parts = [f"{name}[{frac:.10f}]" for name, frac in composition.items()]
    return "HEOS::" + "&".join(parts)


class GERGFluid:
    """GERG-2008 fluid via CoolProp AbstractState with per-instance LRU cache.

    Parameters
    ----------
    composition : dict[str, float]
        Component name → mole fraction. Will be normalized if needed.
    """

    def __init__(self, composition: dict[str, float]) -> None:
        self._composition = _normalize_composition(composition)
        self._fluid_string = _build_fluid_string(self._composition)

        # Build the AbstractState once — this is the expensive operation
        names = list(self._composition.keys())
        fracs = [self._composition[n] for n in names]
        self._state: AbstractState = AbstractState("HEOS", "&".join(names))
        self._state.set_mole_fractions(fracs)
        self._state.specify_phase(CoolProp.iphase_gas)

        # Use a plain dict with bounded size for caching
        self._cache: dict[tuple[int, int], FluidState] = {}
        self._cache_max = 200_000
        self._cache_hits = 0
        self._cache_misses = 0

        # Dew-temperature cache: keyed on rounded pressure only. Dew T
        # depends only on P (and composition, which is fixed per fluid),
        # so this is a tiny dict that pays for itself once per unique P
        # along a march. Stores None for failed flashes so we don't retry.
        self._dew_cache: dict[int, float | None] = {}

        logger.debug("GERGFluid created: %s", self._fluid_string[:80])

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, P: float, T: float) -> tuple[int, int]:
        return (round(P, -2), round(T, 3))

    def _evict_cache(self) -> None:
        # FIFO: remove first inserted items
        n_remove = self._cache_max // 10
        keys = list(self._cache)[:n_remove]
        for k in keys:
            del self._cache[k]

    # ------------------------------------------------------------------
    # Low-level AbstractState evaluation
    # ------------------------------------------------------------------

    def _build_fluid_state(self, P: float, T: float, metastable: bool) -> FluidState:
        """Read all property fields from the CoolProp state after a successful
        ``update()`` call and pack them into a :class:`FluidState`.

        Caller is responsible for the ``update()`` and any phase
        specification dance.
        """
        s = self._state
        rho = s.rhomass()
        h = s.hmass()
        entropy = s.smass()
        cp = s.cpmass()
        cv = s.cvmass()
        mu = s.viscosity()
        k = s.conductivity()
        a = s.speed_sound()
        R_spec = 8.31446 / s.molar_mass()  # J/(kg·K)
        Z = P / (rho * R_spec * T)
        mu_JT = s.first_partial_deriv(_iT, _iP, _iHmass)
        return FluidState(
            P=P, T=T, rho=rho, h=h, s=entropy,
            cp=cp, cv=cv, mu=mu, k=k, a=a, Z=Z, mu_JT=mu_JT,
            metastable=metastable,
        )

    def _eval_state(self, P: float, T: float) -> FluidState:
        """Compute FluidState from (P, T). No caching.

        If the state lies inside the two-phase dome, force a metastable
        gas-phase extrapolation rather than raising — the solver handles
        sub-dew states by continuing on the gas branch and flagging the
        affected stations. Only a hard EOS failure (still bubbles after
        the metastable retry) raises ``EOSOutOfRange``.
        """
        s = self._state
        try:
            s.update(CoolProp.PT_INPUTS, P, T)
            return self._build_fluid_state(P, T, metastable=False)
        except Exception as exc:
            msg = str(exc).lower()
            two_phase = any(
                w in msg
                for w in ("saturated", "two-phase", "two phase", "twophase")
            )
            if not two_phase:
                raise EOSOutOfRange(
                    f"EOS out of range at P={P/1e5:.3f} bara, T={T:.2f} K: {exc}"
                ) from exc

            # Metastable extrapolation: force gas-phase and retry. The
            # GERGFluid is constructed with iphase_gas already specified,
            # but PT-flash with phase-forcing is solver-state-dependent;
            # an explicit re-specify before retry is the documented way.
            try:
                s.specify_phase(CoolProp.iphase_gas)
                s.update(CoolProp.PT_INPUTS, P, T)
                state = self._build_fluid_state(P, T, metastable=True)
                return state
            except Exception as retry_exc:
                raise EOSOutOfRange(
                    f"EOS metastable retry failed at P={P/1e5:.3f} bara, "
                    f"T={T:.2f} K: {retry_exc}"
                ) from retry_exc

    def _eval_state_Ph(self, P: float, h: float) -> FluidState:
        """Compute FluidState from (P, h). No caching.

        The finally block unconditionally restores ``iphase_gas`` because
        CoolProp's HSU_P_flash for HEOS mixtures raises before the in-line
        ``specify_phase`` is reached; without restoration the shared
        AbstractState stays in unspecified-phase mode and every subsequent
        PT flash runs ~25× slower.
        """
        s = self._state
        try:
            # Temporarily remove phase specification so the Ph flash solver
            # can search the full valid range without being forced to gas.
            s.unspecify_phase()
            try:
                s.update(CoolProp.HmassP_INPUTS, h, P)

                T = s.T()
                rho = s.rhomass()
                entropy = s.smass()
                cp = s.cpmass()
                cv = s.cvmass()
                mu = s.viscosity()
                k = s.conductivity()
                a = s.speed_sound()
                R_spec = 8.31446 / s.molar_mass()
                Z = P / (rho * R_spec * T)
                mu_JT = s.first_partial_deriv(_iT, _iP, _iHmass)

            except Exception as exc:
                msg = str(exc).lower()
                if any(w in msg for w in ("saturated", "two-phase", "two phase", "twophase")):
                    raise EOSTwoPhase(
                        f"Two-phase condition at P={P/1e5:.3f} bara, h={h:.0f} J/kg"
                    ) from exc
                raise EOSOutOfRange(
                    f"EOS out of range at P={P/1e5:.3f} bara, h={h:.0f} J/kg: {exc}"
                ) from exc
        finally:
            try:
                s.specify_phase(CoolProp.iphase_gas)
            except Exception:
                pass

        return FluidState(
            P=P, T=T, rho=rho, h=h, s=entropy,
            cp=cp, cv=cv, mu=mu, k=k, a=a, Z=Z, mu_JT=mu_JT,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def props(self, P: float, T: float) -> FluidState:
        """Return FluidState at (P, T) with caching.

        Parameters
        ----------
        P : float
            Pressure [Pa].
        T : float
            Temperature [K].

        Returns
        -------
        FluidState
        """
        key = self._cache_key(P, T)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._cache_misses += 1
        state = self._eval_state(P, T)
        if len(self._cache) >= self._cache_max:
            self._evict_cache()
        self._cache[key] = state
        return state

    def props_Ph(self, P: float, h: float) -> FluidState:
        """Return FluidState at (P, h) — isenthalpic flash.

        Parameters
        ----------
        P : float
            Pressure [Pa].
        h : float
            Specific enthalpy [J/kg].

        Returns
        -------
        FluidState
        """
        return self._eval_state_Ph(P, h)

    def props_Ph_via_jt(self, P_new: float, state_up: FluidState) -> FluidState:
        """Isenthalpic flash via Joule-Thomson estimate then PT flash.

        Robust alternative to :meth:`props_Ph` for HEOS mixtures, which
        CoolProp cannot solve without a pre-built phase envelope. The JT
        coefficient gives a first-order estimate of T at the new pressure
        (accurate to order ΔP²), and the resulting (P_new, T_new) state
        is then evaluated via the standard PT flash.
        """
        dP = P_new - state_up.P
        T_new = state_up.T + state_up.mu_JT * dP
        return self.props(P_new, T_new)

    @property
    def is_mixture(self) -> bool:
        """True if the composition has more than one component."""
        return len(self._composition) > 1

    def dew_temperature(self, P: float) -> float | None:
        """Return dew-point temperature at pressure P, or None if not computable.

        Used to flag metastable march stations where the gas state has
        crossed below the dew curve. The CoolProp PQ_INPUTS flash with
        Q=1 walks the dew curve directly; a return of None signals that
        the flash failed (e.g. P above the cricondenbar, single-phase
        composition, or a numerical issue) — callers should then leave
        T_dew as NaN at that station.

        Cached on rounded pressure (10 Pa precision) for speed; dew T
        depends only on P given fixed composition.

        Notes
        -----
        CoolProp's HEOS PQ flash for multicomponent mixtures occasionally
        converges to a non-physical upper root at narrow pressure bands
        near the cricondenbar (observed for the default Skarv NG mix at
        P ≈ 44.43 bara: PQ returns T ≈ 1400 K, far above the EOS Tmax of
        ~685 K). The Newton iteration accepts the spurious root without
        raising. We defend against this with two cross-checks before
        accepting the value:

        1. Bound check: T must lie within the EOS [Tmin, Tmax] window.
        2. Round-trip: a reverse QT flash at the returned T must yield
           back the requested P. The HEOS QT solver fails (or returns a
           wildly different P) precisely at the spurious-root pressures,
           so a successful round-trip is strong evidence the dew T is
           physical.

        Either failure → return None so the caller treats the station as
        "dew T unknown" (NaN) rather than firing a false two-phase warning.
        """
        key = round(P, -1)  # 10 Pa bins
        if key in self._dew_cache:
            return self._dew_cache[key]

        s = self._state
        # Dew flash needs phase un-specified to find the saturation curve.
        had_phase = True
        try:
            s.unspecify_phase()
        except Exception:
            had_phase = False
        T_dew: float | None
        try:
            s.update(CoolProp.PQ_INPUTS, P, 1.0)
            T_dew = float(s.T())
            T_min = float(s.Tmin())
            T_max = float(s.Tmax())
            if not (T_min <= T_dew <= T_max):
                T_dew = None
            else:
                # Round-trip verification: dew P at T_dew should equal P.
                try:
                    s.update(CoolProp.QT_INPUTS, 1.0, T_dew)
                    P_back = float(s.p())
                    if abs(P_back - P) > 1e-3 * max(P, 1.0):
                        T_dew = None
                except Exception:
                    T_dew = None
        except Exception:
            T_dew = None
        finally:
            # Restore phase specification so subsequent props() calls
            # resume on the gas branch.
            if had_phase:
                try:
                    s.specify_phase(CoolProp.iphase_gas)
                except Exception:
                    pass

        self._dew_cache[key] = T_dew
        return T_dew

    def compute_lvf(self, P: float, h: float) -> float:
        """Liquid volume fraction at (P, h) via two-phase isenthalpic flash.

        Returns
        -------
        float
            LVF in [0, 1] for a genuine two-phase state, 0.0 if the
            isenthalpic flash resolves to a single-phase state, or NaN
            if the flash could not be performed.

        Notes
        -----
        CoolProp's HSU_P_flash for HEOS mixtures is unreliable — it
        often raises "phase envelope must be built" even when an
        envelope was built, because the envelope is not retained
        across some `update()` calls on the shared AbstractState.

        We therefore use a PQ-flash bracket-search:
          * sweep Q ∈ [Q_min, 1] sampling ~10 points
          * at each, read the saturated mixture enthalpy h_sat(Q)
          * find the Q where h_sat(Q) == h_meta (linear interpolation
            between the bracketing samples)
          * convert the mass-quality Q to volume fraction via the
            saturated phase densities at that Q.
        Q values where CoolProp's PQ solver fails are skipped.
        Returns NaN if no usable bracket can be formed.
        """
        s = self._state
        had_phase = True
        try:
            s.unspecify_phase()
        except Exception:
            had_phase = False

        try:
            # Sample h_sat(Q) on a Q grid. Skip Q values where the
            # mixture flash diverges (CoolProp solver is brittle in
            # the middle of the dome for HEOS mixtures).
            Q_grid = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.85,
                      0.9, 0.95, 0.99, 0.999]
            samples: list[tuple[float, float, float, float]] = []
            # Each entry: (Q, h_sat, rho_l, rho_v)
            for Q in Q_grid:
                try:
                    s.update(CoolProp.PQ_INPUTS, P, Q)
                    h_sat = float(s.hmass())
                    rho_l = float(s.saturated_liquid_keyed_output(_iDmass))
                    rho_v = float(s.saturated_vapor_keyed_output(_iDmass))
                    if rho_l > 0 and rho_v > 0:
                        samples.append((Q, h_sat, rho_l, rho_v))
                except Exception:
                    continue

            if len(samples) < 2:
                return float("nan")

            # Sort by h_sat for monotone interpolation along the saturation
            # curve. (h_sat is monotone in Q for fixed P.)
            samples.sort(key=lambda t: t[1])
            h_min = samples[0][1]
            h_max = samples[-1][1]

            if h <= h_min:
                # Below the lowest sampled h_sat — assume all liquid.
                return 1.0
            if h >= h_max:
                # Above the dew enthalpy — single-phase vapor.
                return 0.0

            # Linear bracket: find consecutive samples spanning h.
            for j in range(len(samples) - 1):
                Q_a, h_a, rho_l_a, rho_v_a = samples[j]
                Q_b, h_b, rho_l_b, rho_v_b = samples[j + 1]
                if h_a <= h <= h_b:
                    if h_b == h_a:
                        Q = 0.5 * (Q_a + Q_b)
                        rho_l = 0.5 * (rho_l_a + rho_l_b)
                        rho_v = 0.5 * (rho_v_a + rho_v_b)
                    else:
                        frac = (h - h_a) / (h_b - h_a)
                        Q = Q_a + frac * (Q_b - Q_a)
                        rho_l = rho_l_a + frac * (rho_l_b - rho_l_a)
                        rho_v = rho_v_a + frac * (rho_v_b - rho_v_a)
                    if Q <= 0.0:
                        return 1.0
                    if Q >= 1.0:
                        return 0.0
                    if rho_l <= 0.0 or rho_v <= 0.0:
                        return float("nan")
                    vol_l = (1.0 - Q) / rho_l
                    vol_v = Q / rho_v
                    return vol_l / (vol_l + vol_v)
            return float("nan")
        except Exception:
            return float("nan")
        finally:
            if had_phase:
                try:
                    s.specify_phase(CoolProp.iphase_gas)
                except Exception:
                    pass

    @property
    def molar_mass(self) -> float:
        """Molar mass [kg/mol]."""
        return self._state.molar_mass()

    @property
    def composition(self) -> dict[str, float]:
        """Normalized mole-fraction composition."""
        return dict(self._composition)

    def cache_stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
        }


class TabulatedFluid:
    """Pre-computed bilinear-interpolated EOS for fast solver hot path.

    Builds an ``n_P × n_T`` grid of every :class:`FluidState` property by
    calling ``base_fluid.props(P, T)`` once per grid point at construction
    time. Subsequent ``props(P, T)`` queries are an O(1) bilinear lookup —
    no CoolProp calls — which is ~100-1000× faster than the direct GERG
    flash.

    Out-of-grid queries fall back to ``base_fluid.props`` and increment
    ``_n_outside_grid`` for post-run reporting. Grid points where the
    base EOS raised are stored as NaN; queries whose interpolation cell
    contains a NaN corner also fall back to direct evaluation.

    Methods that aren't part of the solver hot path (``dew_temperature``,
    ``compute_lvf``, ``props_Ph``, ``props_Ph_via_jt``) delegate to the
    base fluid — they're called rarely enough that tabulating them isn't
    worth the build-time cost.

    Parameters
    ----------
    base_fluid : GERGFluid
        Underlying direct EOS. Provides the tabulated values and serves
        as the fallback for out-of-grid queries.
    P_range : tuple[float, float]
        ``(P_min, P_max)`` window [Pa].
    T_range : tuple[float, float]
        ``(T_min, T_max)`` window [K].
    n_P, n_T : int
        Grid resolution in each dimension.
    """

    def __init__(
        self,
        base_fluid: "GERGFluid",
        P_range: tuple[float, float],
        T_range: tuple[float, float],
        n_P: int = 50,
        n_T: int = 50,
    ) -> None:
        if P_range[1] <= P_range[0]:
            raise ValueError(
                f"P_range must be (P_min, P_max) with P_max > P_min; got {P_range}"
            )
        if T_range[1] <= T_range[0]:
            raise ValueError(
                f"T_range must be (T_min, T_max) with T_max > T_min; got {T_range}"
            )
        if n_P < 2 or n_T < 2:
            raise ValueError(f"n_P and n_T must be ≥ 2; got n_P={n_P}, n_T={n_T}")

        self._base = base_fluid
        self.P_range = (float(P_range[0]), float(P_range[1]))
        self.T_range = (float(T_range[0]), float(T_range[1]))
        self.n_P = int(n_P)
        self.n_T = int(n_T)
        self._n_outside_grid = 0
        self._n_failed = 0
        self._build_table()

    def _build_table(self) -> None:
        """Populate the P×T property grids by direct EOS evaluation."""
        self.P_grid = np.linspace(self.P_range[0], self.P_range[1], self.n_P)
        self.T_grid = np.linspace(self.T_range[0], self.T_range[1], self.n_T)
        # Cache the step sizes for O(1) index lookup; equispaced grids.
        self._dP_step = (self.P_range[1] - self.P_range[0]) / (self.n_P - 1)
        self._dT_step = (self.T_range[1] - self.T_range[0]) / (self.n_T - 1)

        shape = (self.n_P, self.n_T)
        self._rho = np.full(shape, np.nan)
        self._h = np.full(shape, np.nan)
        self._s = np.full(shape, np.nan)
        self._cp = np.full(shape, np.nan)
        self._cv = np.full(shape, np.nan)
        self._mu = np.full(shape, np.nan)
        self._k = np.full(shape, np.nan)
        self._a = np.full(shape, np.nan)
        self._Z = np.full(shape, np.nan)
        self._mu_JT = np.full(shape, np.nan)

        for i, P in enumerate(self.P_grid):
            for j, T in enumerate(self.T_grid):
                try:
                    state = self._base.props(float(P), float(T))
                    self._rho[i, j] = state.rho
                    self._h[i, j] = state.h
                    self._s[i, j] = state.s
                    self._cp[i, j] = state.cp
                    self._cv[i, j] = state.cv
                    self._mu[i, j] = state.mu
                    self._k[i, j] = state.k
                    self._a[i, j] = state.a
                    self._Z[i, j] = state.Z
                    self._mu_JT[i, j] = state.mu_JT
                except Exception:
                    self._n_failed += 1
                    # Arrays already hold NaN; nothing to do.

    def props(self, P: float, T: float) -> FluidState:
        """Bilinear-interpolated lookup; falls back to direct EOS outside grid."""
        P_lo, P_hi = self.P_range
        T_lo, T_hi = self.T_range
        if not (P_lo <= P <= P_hi and T_lo <= T <= T_hi):
            self._n_outside_grid += 1
            return self._base.props(P, T)

        # Equispaced grid → integer index by simple arithmetic.
        ip = int((P - P_lo) / self._dP_step)
        if ip >= self.n_P - 1:
            ip = self.n_P - 2
        jt = int((T - T_lo) / self._dT_step)
        if jt >= self.n_T - 1:
            jt = self.n_T - 2

        # Fractional weights within the cell.
        dp = (P - self.P_grid[ip]) / self._dP_step
        dt = (T - self.T_grid[jt]) / self._dT_step

        # If any corner is NaN (EOS failure during build), fall back. We
        # check rho because it's NaN iff the whole state is NaN at that
        # grid point.
        if (np.isnan(self._rho[ip, jt])
                or np.isnan(self._rho[ip + 1, jt])
                or np.isnan(self._rho[ip, jt + 1])
                or np.isnan(self._rho[ip + 1, jt + 1])):
            self._n_outside_grid += 1
            return self._base.props(P, T)

        w00 = (1.0 - dp) * (1.0 - dt)
        w10 = dp * (1.0 - dt)
        w01 = (1.0 - dp) * dt
        w11 = dp * dt

        def _interp(arr: np.ndarray) -> float:
            return float(
                arr[ip, jt] * w00 + arr[ip + 1, jt] * w10
                + arr[ip, jt + 1] * w01 + arr[ip + 1, jt + 1] * w11
            )

        return FluidState(
            P=P, T=T,
            rho=_interp(self._rho),
            h=_interp(self._h),
            s=_interp(self._s),
            cp=_interp(self._cp),
            cv=_interp(self._cv),
            mu=_interp(self._mu),
            k=_interp(self._k),
            a=_interp(self._a),
            Z=_interp(self._Z),
            mu_JT=_interp(self._mu_JT),
            metastable=False,
        )

    def props_Ph(self, P: float, h: float) -> FluidState:
        """Delegate isenthalpic flash to base fluid (not on hot path)."""
        return self._base.props_Ph(P, h)

    def props_Ph_via_jt(
        self, P_new: float, state_up: FluidState
    ) -> FluidState:
        """JT-estimated isenthalpic flash using this fluid's ``props`` (so the
        downstream PT flash uses the table, not the base EOS).
        """
        dP = P_new - state_up.P
        T_new = state_up.T + state_up.mu_JT * dP
        return self.props(P_new, T_new)

    def dew_temperature(self, P: float) -> float | None:
        """Delegate to base — dew T not tabulated; called O(n_stations) per solve."""
        return self._base.dew_temperature(P)

    def compute_lvf(self, P: float, h: float) -> float:
        """Delegate to base — LVF only needed at metastable stations."""
        return self._base.compute_lvf(P, h)

    @property
    def composition(self) -> dict[str, float]:
        return self._base.composition

    @property
    def molar_mass(self) -> float:
        return self._base.molar_mass

    @property
    def is_mixture(self) -> bool:
        return self._base.is_mixture

    def table_stats(self) -> dict[str, int | float]:
        """Diagnostics: grid size, failure counts, fallback counts."""
        return {
            "n_P": self.n_P,
            "n_T": self.n_T,
            "P_min": self.P_range[0],
            "P_max": self.P_range[1],
            "T_min": self.T_range[0],
            "T_max": self.T_range[1],
            "n_failed": self._n_failed,
            "n_outside_grid": self._n_outside_grid,
        }


def estimate_operating_window(
    P_in: float,
    T_in: float,
    P_out: float,
    fluid: "FluidEOSBase",
    margin_factor: float = 1.1,
    T_min_floor_K: float = 80.0,
    T_max_overhead_K: float = 20.0,
    jt_margin: float = 1.5,
) -> tuple[float, float, float, float]:
    """Estimate a (P_min, P_max, T_min, T_max) window covering a BVP solution.

    Used to size a :class:`TabulatedFluid` so that the solver hot path
    stays inside the grid. Out-of-grid queries silently fall back to the
    base EOS, so a too-small window is correct but slow; we err on the
    side of slightly larger windows than the minimum necessary.

    Parameters
    ----------
    P_in, T_in, P_out : float
        BVP boundary conditions [Pa, K, Pa].
    fluid : FluidEOSBase
        Used to evaluate ``mu_JT`` at the inlet for the cooling estimate.
    margin_factor : float
        Multiplicative pressure margin (≥ 1.0).
    T_min_floor_K : float
        Hard floor on T_min headroom below T_in. Skarv-type cases drop
        ~70 K under JT cooling at choke; 80 K covers that with margin.
    T_max_overhead_K : float
        Headroom above T_in. Modest because the solver only heats above
        T_in via wall heat transfer with limited driving force.
    jt_margin : float
        Safety factor on the inlet-JT cooling estimate.

    Returns
    -------
    tuple[float, float, float, float]
        ``(P_min, P_max, T_min, T_max)`` in SI units.
    """
    if margin_factor < 1.0:
        raise ValueError(
            f"margin_factor must be ≥ 1.0; got {margin_factor}"
        )
    P_min = min(P_in, P_out) / margin_factor
    P_max = max(P_in, P_out) * margin_factor

    # JT cooling estimate at the inlet. mu_JT is positive for natural gas
    # at typical conditions → ΔT_max > 0 (expansion cools).
    inlet = fluid.props(P_in, T_in)
    delta_P = P_in - P_min
    delta_T_jt = max(0.0, inlet.mu_JT * delta_P) * jt_margin
    T_min = T_in - max(delta_T_jt, T_min_floor_K)
    T_max = T_in + T_max_overhead_K

    return P_min, P_max, T_min, T_max

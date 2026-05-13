"""Pipe geometry and thermal boundary conditions.

A :class:`Pipe` is composed of one or more :class:`PipeSection` objects laid
end-to-end. Each section has its own diameter, roughness, wall thickness and
heat-transfer coefficient. A single-section pipe is the legacy / uniform case;
multi-section pipes model diameter changes (sudden expansions / contractions)
along the run.

Section boundary transitions are handled by the solver, not the geometry
module — :class:`Pipe` only exposes section-aware lookups.

Backward compatibility: ``Pipe.horizontal_uniform(...)`` returns a one-section
pipe and is preferred for the simple uniform case. Legacy attributes
(``inner_diameter``, ``roughness``, ``area``, ``eps_over_D``, ``outer_diameter``)
return the first section's value so existing code keeps working.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class PipeSection:
    """One section of constant geometry within a pipe.

    Parameters
    ----------
    length : float
        Section length [m].
    inner_diameter : float
        Internal diameter [m].
    outer_diameter : float or None
        External diameter [m]. Defaults to ``inner_diameter`` if None.
    roughness : float
        Absolute wall roughness [m].
    overall_U : float
        Overall heat-transfer coefficient [W/m²/K] referred to OD.
    elevation_change : float
        z_end - z_start across this section [m] (default flat).
    """

    length: float
    inner_diameter: float
    outer_diameter: float | None = None
    roughness: float = 4.5e-5
    overall_U: float = 0.0
    elevation_change: float = 0.0

    @property
    def area(self) -> float:
        """Cross-sectional flow area [m²]."""
        return math.pi * self.inner_diameter ** 2 / 4.0

    @property
    def eps_over_D(self) -> float:
        """Relative roughness ε/D [-]."""
        return self.roughness / self.inner_diameter

    def D_o(self) -> float:
        """Outer diameter [m] — falls back to inner_diameter if unset."""
        return self.outer_diameter if self.outer_diameter is not None else self.inner_diameter


@dataclass
class Pipe:
    """1D pipe geometry composed of one or more sections.

    Parameters
    ----------
    sections : list[PipeSection]
        Sections laid end-to-end from inlet to outlet. Must contain at
        least one entry.
    elevation_profile : Callable or None
        Optional z(x) [m] function. If provided, overrides the per-section
        ``elevation_change`` values for the global z(x) lookup.
    ambient_temperature : Callable or float
        T_amb(x) [K]. Scalar or callable. Applies pipe-wide.
    fittings : list
        Discrete fittings keyed on global x-coordinate (from inlet).
    """

    sections: list[PipeSection]
    elevation_profile: Callable[[float], float] | None = None
    ambient_temperature: Callable[[float], float] | float = 277.15
    fittings: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sections:
            raise ValueError("Pipe must have at least one PipeSection.")
        for i, sec in enumerate(self.sections):
            if sec.length <= 0:
                raise ValueError(
                    f"Section {i}: length must be > 0, got {sec.length}."
                )
            if sec.inner_diameter <= 0:
                raise ValueError(
                    f"Section {i}: inner_diameter must be > 0, "
                    f"got {sec.inner_diameter}."
                )

    # ------------------------------------------------------------------
    # Aggregated / legacy attributes
    # ------------------------------------------------------------------

    @property
    def length(self) -> float:
        """Total pipe length [m]."""
        return sum(s.length for s in self.sections)

    @property
    def inner_diameter(self) -> float:
        """Inner diameter of the FIRST section [m] (legacy accessor)."""
        return self.sections[0].inner_diameter

    @property
    def roughness(self) -> float:
        """Absolute roughness of the FIRST section [m] (legacy accessor)."""
        return self.sections[0].roughness

    @property
    def outer_diameter(self) -> float | None:
        """Outer diameter of the FIRST section [m] (legacy accessor)."""
        return self.sections[0].outer_diameter

    @property
    def overall_U(self) -> float:
        """U of the FIRST section [W/m²/K] (legacy accessor)."""
        return self.sections[0].overall_U

    @property
    def area(self) -> float:
        """Cross-sectional area of the FIRST section [m²] (legacy)."""
        return self.sections[0].area

    @property
    def eps_over_D(self) -> float:
        """Relative roughness of the FIRST section [-] (legacy)."""
        return self.sections[0].eps_over_D

    def D_o(self) -> float:
        """Outer diameter of the FIRST section [m] (legacy)."""
        return self.sections[0].D_o()

    @property
    def is_multi_section(self) -> bool:
        """True if more than one section is defined."""
        return len(self.sections) > 1

    # ------------------------------------------------------------------
    # Section-aware lookups
    # ------------------------------------------------------------------

    def section_at(self, x: float) -> tuple[PipeSection, float, int]:
        """Return (section, x_in_section, section_index) for global x.

        ``x_in_section`` is 0 at the section's start and ``section.length``
        at its end. For x exactly at an interior boundary, returns the
        section ending at x (not the next one). For x beyond pipe length,
        returns the last section evaluated at its outlet.
        """
        cumulative = 0.0
        for i, sec in enumerate(self.sections):
            section_end = cumulative + sec.length
            if x <= section_end + 1e-9:
                return sec, x - cumulative, i
            cumulative = section_end
        return self.sections[-1], self.sections[-1].length, len(self.sections) - 1

    def D_at(self, x: float) -> float:
        """Inner diameter at position x [m]."""
        return self.section_at(x)[0].inner_diameter

    def D_o_at(self, x: float) -> float:
        """Outer diameter at position x [m]."""
        return self.section_at(x)[0].D_o()

    def eps_over_D_at(self, x: float) -> float:
        """Relative roughness at position x [-]."""
        return self.section_at(x)[0].eps_over_D

    def area_at(self, x: float) -> float:
        """Cross-sectional area at position x [m²]."""
        return self.section_at(x)[0].area

    def roughness_at(self, x: float) -> float:
        """Absolute roughness at position x [m]."""
        return self.section_at(x)[0].roughness

    def U_at(self, x: float) -> float:
        """Overall heat-transfer coefficient at x [W/m²/K]."""
        return self.section_at(x)[0].overall_U

    def U(self, x: float) -> float:
        """Backward-compatible alias for U_at(x)."""
        return self.U_at(x)

    def is_at_section_boundary(self, x: float, tol: float = 1e-9) -> int | None:
        """Return index of interior section boundary at x, or None.

        Index ``i`` means the boundary between sections ``i`` and ``i+1``.
        Outlet boundary (x == pipe.length) is not reported.
        """
        cumulative = 0.0
        for i in range(len(self.sections) - 1):
            cumulative += self.sections[i].length
            if abs(x - cumulative) < tol:
                return i
        return None

    def next_section_boundary_after(self, x: float, tol: float = 1e-9) -> float | None:
        """Return x of the next interior section boundary strictly after x.

        Returns None if x is at or beyond the last interior boundary.
        """
        cumulative = 0.0
        for i in range(len(self.sections) - 1):
            cumulative += self.sections[i].length
            if cumulative > x + tol:
                return cumulative
        return None

    # ------------------------------------------------------------------
    # Elevation / ambient T
    # ------------------------------------------------------------------

    def z(self, x: float) -> float:
        """Elevation at position x [m].

        If ``elevation_profile`` is set, it overrides the per-section
        ``elevation_change`` integration.
        """
        if self.elevation_profile is not None:
            return float(self.elevation_profile(x))
        z_val = 0.0
        cumulative = 0.0
        for sec in self.sections:
            section_end = cumulative + sec.length
            if x <= section_end + 1e-9:
                if sec.length > 0:
                    frac = (x - cumulative) / sec.length
                else:
                    frac = 0.0
                return z_val + max(0.0, min(1.0, frac)) * sec.elevation_change
            z_val += sec.elevation_change
            cumulative = section_end
        return z_val

    def T_amb(self, x: float) -> float:
        """Ambient temperature at position x [K]."""
        if callable(self.ambient_temperature):
            return float(self.ambient_temperature(x))
        return float(self.ambient_temperature)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def horizontal_uniform(
        cls,
        length: float,
        inner_diameter: float,
        roughness: float = 4.5e-5,
        outer_diameter: float | None = None,
        overall_U: float = 0.0,
        ambient_temperature: Callable[[float], float] | float = 277.15,
        fittings: list | None = None,
    ) -> "Pipe":
        """Backward-compatible single-section constructor."""
        return cls(
            sections=[PipeSection(
                length=length,
                inner_diameter=inner_diameter,
                outer_diameter=outer_diameter,
                roughness=roughness,
                overall_U=overall_U,
            )],
            elevation_profile=None,
            ambient_temperature=ambient_temperature,
            fittings=list(fittings) if fittings else [],
        )

    @classmethod
    def with_elevation_table(
        cls,
        x_arr: "np.ndarray",
        z_arr: "np.ndarray",
        inner_diameter: float,
        roughness: float,
        **kwargs,
    ) -> "Pipe":
        """Single-section pipe with a tabulated elevation profile.

        Length is taken from ``x_arr[-1]``. The remaining kwargs are
        forwarded to ``horizontal_uniform``.
        """
        x_data = np.asarray(x_arr, dtype=float)
        z_data = np.asarray(z_arr, dtype=float)
        length = float(x_data[-1])

        def _elev(x: float) -> float:
            return float(np.interp(x, x_data, z_data))

        pipe = cls.horizontal_uniform(
            length=length,
            inner_diameter=inner_diameter,
            roughness=roughness,
            **kwargs,
        )
        pipe.elevation_profile = _elev
        return pipe

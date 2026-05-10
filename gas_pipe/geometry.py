"""Pipe geometry and thermal boundary conditions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Pipe:
    """1D pipe geometry with elevation, thermal BC, and fittings.

    Parameters
    ----------
    length : float
        Total pipe length [m].
    inner_diameter : float
        Internal diameter [m].
    roughness : float
        Absolute wall roughness [m].
    elevation_profile : Callable or None
        z(x) [m]. If None, flat (z = 0).
    ambient_temperature : Callable or float
        T_amb(x) [K]. Scalar or callable.
    overall_U : Callable or float
        Overall heat-transfer coefficient [W/m²/K] referred to OD.
    outer_diameter : float or None
        Outer diameter [m]. Defaults to inner_diameter if None.
    fittings : list
        List of Fitting objects along the pipe.
    """

    length: float
    inner_diameter: float
    roughness: float
    elevation_profile: Callable[[float], float] | None = None
    ambient_temperature: Callable[[float], float] | float = 277.15
    overall_U: Callable[[float], float] | float = 0.0
    outer_diameter: float | None = None
    fittings: list = field(default_factory=list)

    @property
    def area(self) -> float:
        """Cross-sectional flow area [m²]."""
        return math.pi * self.inner_diameter ** 2 / 4.0

    @property
    def eps_over_D(self) -> float:
        """Relative roughness ε/D [-]."""
        return self.roughness / self.inner_diameter

    def z(self, x: float) -> float:
        """Elevation at position x [m].

        Parameters
        ----------
        x : float
            Axial position [m].

        Returns
        -------
        float
            Elevation [m].
        """
        if self.elevation_profile is None:
            return 0.0
        return float(self.elevation_profile(x))

    def T_amb(self, x: float) -> float:
        """Ambient temperature at position x [K].

        Parameters
        ----------
        x : float
            Axial position [m].

        Returns
        -------
        float
            Ambient temperature [K].
        """
        if callable(self.ambient_temperature):
            return float(self.ambient_temperature(x))
        return float(self.ambient_temperature)

    def U(self, x: float) -> float:
        """Overall heat-transfer coefficient at position x [W/m²/K].

        Parameters
        ----------
        x : float
            Axial position [m].

        Returns
        -------
        float
            U [W/m²/K].
        """
        if callable(self.overall_U):
            return float(self.overall_U(x))
        return float(self.overall_U)

    def D_o(self) -> float:
        """Outer diameter [m]."""
        return self.outer_diameter if self.outer_diameter is not None else self.inner_diameter

    @classmethod
    def horizontal_uniform(
        cls,
        length: float,
        inner_diameter: float,
        roughness: float,
        **kwargs,
    ) -> "Pipe":
        """Convenience constructor for a straight horizontal pipe.

        Parameters
        ----------
        length : float
            Pipe length [m].
        inner_diameter : float
            Inner diameter [m].
        roughness : float
            Absolute roughness [m].
        **kwargs
            Additional keyword arguments forwarded to the dataclass.

        Returns
        -------
        Pipe
        """
        return cls(
            length=length,
            inner_diameter=inner_diameter,
            roughness=roughness,
            elevation_profile=None,
            **kwargs,
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
        """Construct a pipe with tabulated elevation profile (linear interpolation).

        Parameters
        ----------
        x_arr : array-like
            Axial positions [m].
        z_arr : array-like
            Corresponding elevations [m].
        inner_diameter : float
            Inner diameter [m].
        roughness : float
            Absolute roughness [m].
        **kwargs
            Forwarded to dataclass constructor.

        Returns
        -------
        Pipe
        """
        x_data = np.asarray(x_arr, dtype=float)
        z_data = np.asarray(z_arr, dtype=float)
        length = float(x_data[-1])

        def _elev(x: float) -> float:
            return float(np.interp(x, x_data, z_data))

        return cls(
            length=length,
            inner_diameter=inner_diameter,
            roughness=roughness,
            elevation_profile=_elev,
            **kwargs,
        )

"""Multi-section pipe (backlog item 27) — section_at lookup, Borda-Carnot
expansion loss, sudden-contraction K-factor, and segment-boundary splitting."""
from __future__ import annotations

import math

import numpy as np
import pytest

from gas_pipe import GERGFluid, Pipe, PipeSection, march_ivp


@pytest.fixture(scope="module")
def fluid():
    return GERGFluid({"Methane": 0.85, "Ethane": 0.10, "Propane": 0.05})


def test_single_section_via_horizontal_uniform():
    """horizontal_uniform must still produce a 1-section Pipe."""
    pipe = Pipe.horizontal_uniform(
        length=80.0,
        inner_diameter=0.762,
        roughness=4.5e-5,
        outer_diameter=0.813,
        overall_U=2.0,
        ambient_temperature=283.15,
    )
    assert len(pipe.sections) == 1
    sec = pipe.sections[0]
    assert sec.length == 80.0
    assert sec.inner_diameter == 0.762
    assert sec.outer_diameter == 0.813
    assert sec.roughness == 4.5e-5
    assert sec.overall_U == 2.0
    assert pipe.length == 80.0
    # Legacy accessors return the first section's values.
    assert pipe.inner_diameter == 0.762
    assert pipe.roughness == 4.5e-5


def test_section_at_lookup():
    """section_at returns (section, x_local, idx) with the expected semantics."""
    pipe = Pipe(sections=[
        PipeSection(length=20.0, inner_diameter=0.660),
        PipeSection(length=60.0, inner_diameter=0.762),
    ])
    # Interior of section 0.
    sec, x_local, idx = pipe.section_at(10.0)
    assert idx == 0
    assert x_local == pytest.approx(10.0)
    assert sec.inner_diameter == 0.660

    # Interior of section 1.
    sec, x_local, idx = pipe.section_at(30.0)
    assert idx == 1
    assert x_local == pytest.approx(10.0)
    assert sec.inner_diameter == 0.762

    # Exactly at the interior boundary — belongs to the section ENDING at x.
    sec, x_local, idx = pipe.section_at(20.0)
    assert idx == 0
    assert x_local == pytest.approx(20.0)

    # Just past the boundary — belongs to the downstream section.
    sec, x_local, idx = pipe.section_at(20.0 + 1e-6)
    assert idx == 1


def test_is_at_section_boundary():
    """is_at_section_boundary returns the boundary index for interior boundaries."""
    pipe = Pipe(sections=[
        PipeSection(length=20.0, inner_diameter=0.4),
        PipeSection(length=30.0, inner_diameter=0.6),
        PipeSection(length=10.0, inner_diameter=0.5),
    ])
    assert pipe.is_at_section_boundary(20.0) == 0
    assert pipe.is_at_section_boundary(50.0) == 1
    # Outlet boundary is NOT reported.
    assert pipe.is_at_section_boundary(60.0) is None
    # Interior point.
    assert pipe.is_at_section_boundary(35.0) is None


def test_next_section_boundary_after():
    """next_section_boundary_after returns the next interior boundary, or None."""
    pipe = Pipe(sections=[
        PipeSection(length=20.0, inner_diameter=0.4),
        PipeSection(length=30.0, inner_diameter=0.6),
        PipeSection(length=10.0, inner_diameter=0.5),
    ])
    assert pipe.next_section_boundary_after(0.0) == pytest.approx(20.0)
    assert pipe.next_section_boundary_after(20.0) == pytest.approx(50.0)
    assert pipe.next_section_boundary_after(50.0) is None
    assert pipe.next_section_boundary_after(55.0) is None


def test_two_section_expansion_borda_carnot(fluid):
    """Expansion should produce pressure recovery (net ΔP < 0) for fast inlet flow."""
    pipe = Pipe(sections=[
        PipeSection(length=10.0, inner_diameter=0.4),
        PipeSection(length=10.0, inner_diameter=0.6),
    ])
    P_in = 50e5
    T_in = 373.15
    # mdot chosen so u_up is high enough that the kinetic recovery
    # (proportional to u_up² - u_dn²) dominates the Borda head loss
    # (proportional to (1−β)²·u_up²).
    mdot = 60.0

    result = march_ivp(
        pipe, fluid, P_in=P_in, T_in=T_in, mdot=mdot,
        n_segments=100, adaptive=False,
    )

    assert len(result.section_transitions) == 1
    t = result.section_transitions[0]
    assert t["type"] == "expansion"
    assert t["x"] == pytest.approx(10.0)
    assert t["from_section"] == 0
    assert t["to_section"] == 1
    A_up = math.pi * 0.4 ** 2 / 4
    A_dn = math.pi * 0.6 ** 2 / 4
    assert t["A_up"] == pytest.approx(A_up)
    assert t["A_dn"] == pytest.approx(A_dn)
    # Pressure recovery: kinetic-head reduction outweighs Borda loss here.
    assert t["dP"] < 0
    # Sanity: the recorded loss must be positive.
    assert t["dP_loss"] > 0


def test_two_section_contraction(fluid):
    """Contraction must produce a positive pressure drop at the boundary."""
    pipe = Pipe(sections=[
        PipeSection(length=10.0, inner_diameter=0.6),
        PipeSection(length=10.0, inner_diameter=0.4),
    ])
    P_in = 50e5
    T_in = 373.15
    mdot = 60.0

    result = march_ivp(
        pipe, fluid, P_in=P_in, T_in=T_in, mdot=mdot,
        n_segments=100, adaptive=False,
    )

    assert len(result.section_transitions) == 1
    t = result.section_transitions[0]
    assert t["type"] == "contraction"
    assert t["x"] == pytest.approx(10.0)
    # Contraction always drops pressure (both loss and acceleration go same sign).
    assert t["dP"] > 0
    assert t["dP_loss"] > 0


def test_section_transition_at_segment_boundary(fluid):
    """Segments must split cleanly at section boundaries — exactly one station
    must lie at the boundary and it must record A_dn (downstream) for the
    velocity calculation."""
    pipe = Pipe(sections=[
        PipeSection(length=10.0, inner_diameter=0.4),
        PipeSection(length=10.0, inner_diameter=0.6),
    ])
    result = march_ivp(
        pipe, fluid, P_in=50e5, T_in=373.15, mdot=60.0,
        n_segments=20, adaptive=False,
    )
    # Station at exactly x = 10.0 m must exist.
    idx = np.argmin(np.abs(result.x - 10.0))
    assert abs(result.x[idx] - 10.0) < 1e-6, (
        f"No station found near x = 10 m; closest = {result.x[idx]:.6f}"
    )
    # The boundary station's u must match A_dn (larger ID → lower u).
    A_dn = math.pi * 0.6 ** 2 / 4
    u_expected = result.mdot / (result.rho[idx] * A_dn)
    assert result.u[idx] == pytest.approx(u_expected, rel=1e-3)


def test_single_section_pipe_records_no_transitions(fluid):
    """A 1-section pipe must have an empty section_transitions list."""
    pipe = Pipe.horizontal_uniform(length=50.0, inner_diameter=0.3)
    result = march_ivp(
        pipe, fluid, P_in=30e5, T_in=300.0, mdot=10.0,
        n_segments=50, adaptive=False,
    )
    assert result.section_transitions == []


def test_summary_includes_sections_block(fluid):
    """summary() includes a SECTIONS block when multi-section."""
    pipe = Pipe(sections=[
        PipeSection(length=10.0, inner_diameter=0.4),
        PipeSection(length=10.0, inner_diameter=0.6),
    ])
    result = march_ivp(
        pipe, fluid, P_in=50e5, T_in=373.15, mdot=50.0,
        n_segments=40, adaptive=False,
    )
    s = result.summary()
    assert "SECTIONS" in s
    assert "Section transitions:" in s
    # Single-section pipes don't get a SECTIONS block.
    pipe2 = Pipe.horizontal_uniform(length=20.0, inner_diameter=0.4)
    r2 = march_ivp(pipe2, fluid, 50e5, 373.15, 50.0, n_segments=40, adaptive=False)
    assert "SECTIONS" not in r2.summary()

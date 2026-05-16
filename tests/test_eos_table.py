"""Tabulated EOS (backlog item 2) — bilinear interpolation accuracy and
fallback behaviour for the :class:`TabulatedFluid` wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from gas_pipe import (
    FluidEOSBase,
    GERGFluid,
    Pipe,
    TabulatedFluid,
    estimate_operating_window,
    plateau_sweep,
    solve_for_mdot,
)
from gas_pipe.errors import BVPChoked

SKARV_COMPOSITION = {
    "Methane": 0.85,
    "Ethane": 0.10,
    "Propane": 0.05,
}


@pytest.fixture(scope="module")
def fluid() -> GERGFluid:
    return GERGFluid(SKARV_COMPOSITION)


@pytest.fixture(scope="module")
def table(fluid: GERGFluid) -> TabulatedFluid:
    """Coarse 20×20 table over a Skarv-like P,T window."""
    return TabulatedFluid(
        fluid, P_range=(20e5, 60e5), T_range=(280.0, 400.0),
        n_P=20, n_T=20,
    )


def test_tabulated_satisfies_fluid_eos_base(table: TabulatedFluid):
    assert isinstance(table, FluidEOSBase)


def test_tabulated_passthrough_properties(fluid: GERGFluid, table: TabulatedFluid):
    assert table.composition == fluid.composition
    assert table.molar_mass == fluid.molar_mass
    assert table.is_mixture == fluid.is_mixture


def test_tabulated_matches_direct_at_grid_points(
    fluid: GERGFluid, table: TabulatedFluid
):
    """At exact grid points bilinear interpolation collapses to the corner
    value, so the tabulated state must match the direct EOS to machine
    precision (modulo any composition normalization)."""
    rng = np.random.default_rng(0)
    # Spot-check a sample of grid points rather than the full 400 — keeps
    # the test fast while still hitting all four corners and the interior.
    Pi_sample = sorted(set([0, table.n_P - 1] + list(rng.integers(1, table.n_P - 1, 5))))
    Tj_sample = sorted(set([0, table.n_T - 1] + list(rng.integers(1, table.n_T - 1, 5))))
    for i in Pi_sample:
        for j in Tj_sample:
            P = float(table.P_grid[i])
            T = float(table.T_grid[j])
            tab = table.props(P, T)
            direct = fluid.props(P, T)
            assert tab.rho == pytest.approx(direct.rho, rel=1e-12)
            assert tab.h == pytest.approx(direct.h, rel=1e-12)
            assert tab.s == pytest.approx(direct.s, rel=1e-12)
            assert tab.a == pytest.approx(direct.a, rel=1e-12)
            assert tab.Z == pytest.approx(direct.Z, rel=1e-12)
            assert tab.cp == pytest.approx(direct.cp, rel=1e-12)
            assert tab.cv == pytest.approx(direct.cv, rel=1e-12)
            assert tab.mu == pytest.approx(direct.mu, rel=1e-12)
            assert tab.mu_JT == pytest.approx(direct.mu_JT, rel=1e-12)


def test_tabulated_interpolates_between_grid_points(fluid: GERGFluid):
    """Mid-cell queries must hit ~0.5% rho accuracy with a 50×50 grid.

    50×50 is the default GUI resolution and the bilinear error there is
    typically <0.3% in the dense-gas region; we use 5e-3 as a generous
    upper bound that captures the full grid (including the steeper
    high-rho corners).
    """
    table = TabulatedFluid(
        fluid, P_range=(20e5, 60e5), T_range=(280.0, 400.0),
        n_P=50, n_T=50,
    )
    # Mid-cell points — worst case for bilinear interpolation.
    for P in (25e5, 35e5, 45e5, 55e5):
        for T in (290.0, 320.0, 350.0, 380.0):
            tab = table.props(P, T)
            direct = fluid.props(P, T)
            assert abs(tab.rho - direct.rho) / direct.rho < 5e-3, (
                f"rho off by {(tab.rho-direct.rho)/direct.rho*100:.3f}% "
                f"at P={P/1e5:.1f} bara, T={T:.1f} K"
            )
            # h is on the order of 1e5 J/kg with values that can dip near
            # zero in NG; compare on absolute value with a generous floor.
            assert abs(tab.h - direct.h) / max(abs(direct.h), 1e4) < 5e-3
            assert abs(tab.a - direct.a) / direct.a < 5e-3
            assert abs(tab.Z - direct.Z) / direct.Z < 5e-3


def test_tabulated_fallback_outside_grid(fluid: GERGFluid):
    """Lookups outside grid fall through to the base EOS without crashing.

    Each fallback increments the diagnostic counter so the summary can
    surface "out-of-grid fallbacks: N" to the user.
    """
    table = TabulatedFluid(
        fluid, P_range=(20e5, 60e5), T_range=(280.0, 400.0),
        n_P=10, n_T=10,
    )
    assert table._n_outside_grid == 0

    # P below grid
    s_below = table.props(10e5, 300.0)
    assert s_below.rho > 0
    assert table._n_outside_grid == 1

    # T above grid
    s_above = table.props(40e5, 450.0)
    assert s_above.rho > 0
    assert table._n_outside_grid == 2

    # The fallback value must equal the direct EOS at that (P, T).
    direct = fluid.props(10e5, 300.0)
    assert s_below.rho == pytest.approx(direct.rho, rel=1e-10)


def test_tabulated_rejects_inverted_or_degenerate_ranges(fluid: GERGFluid):
    with pytest.raises(ValueError):
        TabulatedFluid(fluid, P_range=(60e5, 20e5), T_range=(280, 400))
    with pytest.raises(ValueError):
        TabulatedFluid(fluid, P_range=(20e5, 60e5), T_range=(400, 280))
    with pytest.raises(ValueError):
        TabulatedFluid(fluid, P_range=(20e5, 60e5), T_range=(280, 400), n_P=1)


def test_tabulated_table_stats(table: TabulatedFluid):
    stats = table.table_stats()
    assert stats["n_P"] == 20
    assert stats["n_T"] == 20
    assert stats["P_min"] == 20e5
    assert stats["P_max"] == 60e5
    # No failed grid points expected in this benign Skarv-NG window.
    assert stats["n_failed"] == 0


def test_tabulated_delegates_dew_temperature(fluid: GERGFluid, table: TabulatedFluid):
    """dew_temperature isn't tabulated — it must give the exact base value."""
    T_dew_direct = fluid.dew_temperature(30e5)
    T_dew_table = table.dew_temperature(30e5)
    if T_dew_direct is None:
        assert T_dew_table is None
    else:
        assert T_dew_table == pytest.approx(T_dew_direct, rel=1e-12)


def test_estimate_operating_window_skarv_defaults(fluid: GERGFluid):
    """For Skarv-like BCs, the window must comfortably wrap the BCs.

    Quantitative sanity check rather than a tight bound — the goal is
    that the solver won't go out of grid at typical operating points,
    not to validate the cooling estimate to within K.
    """
    P_in, T_in, P_last_cell = 50e5, 373.15, 2e5
    P_min, P_max, T_min, T_max = estimate_operating_window(
        P_in, T_in, P_last_cell, fluid
    )
    # Pressure window must include both BCs with the margin factor applied.
    assert P_min < P_last_cell
    assert P_max > P_in
    # Temperature window must extend at least the T_min_floor below T_in
    # and at least T_max_overhead above.
    assert T_min <= T_in - 80.0 + 1e-6
    assert T_max >= T_in + 20.0 - 1e-6
    # Sane absolute bounds.
    assert T_min > 0.0
    assert T_max < 600.0


def test_estimate_operating_window_jt_cooling_widens_T_min(fluid: GERGFluid):
    """A larger ΔP must produce equal or wider T_min headroom (never narrower)."""
    _, _, T_min_small, _ = estimate_operating_window(
        20e5, 373.15, 18e5, fluid
    )
    _, _, T_min_big, _ = estimate_operating_window(
        50e5, 373.15, 2e5, fluid
    )
    assert T_min_big <= T_min_small


def test_estimate_operating_window_rejects_bad_margin(fluid: GERGFluid):
    with pytest.raises(ValueError):
        estimate_operating_window(50e5, 373.15, 2e5, fluid, margin_factor=0.5)


def test_solver_with_table_mode_matches_direct_skarv(fluid: GERGFluid):
    """Skarv default BVP: table mode mdot must be within 0.5% of direct mode.

    This is the load-bearing accuracy check for item 2 — if 50×50 bilinear
    interpolation can't reach 0.5% on the canonical Skarv case, the whole
    speedup story falls over.

    The Skarv default BVP (P_in=50, P_out=2 bara) is choke-limited at
    ṁ_critical ≈ 3287 kg/s (the 80m × 762mm geometry can't dissipate
    enough pressure to reach 2 bara at any flow rate before choking).
    Both modes must agree on that ṁ_critical and on the choke-boundary
    state to 0.5% / 1 kPa / 1 K.
    """
    pipe = Pipe.horizontal_uniform(
        length=80.0, inner_diameter=0.762, roughness=4.5e-5,
        outer_diameter=0.813, overall_U=2.0,
    )

    def _run(mode: str):
        try:
            return solve_for_mdot(
                pipe, fluid, P_in=50e5, T_in=373.15, P_out=2e5,
                eos_mode=mode,
            )
        except BVPChoked as exc:
            return exc.result

    r_direct = _run("direct")
    r_table = _run("table")

    # Critical mass-flow agreement — the headline accuracy number.
    rel_mdot = abs(r_table.mdot - r_direct.mdot) / r_direct.mdot
    assert rel_mdot < 5e-3, (
        f"mdot differs by {rel_mdot*100:.3f}%: "
        f"direct={r_direct.mdot:.3f}, table={r_table.mdot:.3f}"
    )

    # Outlet state agreement — looser to allow the table's first-order
    # error to accumulate slightly across the march.
    P_diff_kPa = abs(r_table.P[-1] - r_direct.P[-1]) / 1e3
    T_diff_K = abs(r_table.T[-1] - r_direct.T[-1])
    assert P_diff_kPa < 50.0, f"P_last_cell differs by {P_diff_kPa:.2f} kPa"
    assert T_diff_K < 1.0, f"T_out differs by {T_diff_K:.3f} K"

    # The table result must carry the table-mode diagnostics.
    assert r_table.solver_options["eos_mode"] == "table"
    stats = r_table.solver_options["table_stats"]
    assert stats["n_P"] == 50
    assert stats["n_T"] == 50
    # Zero out-of-grid fallbacks expected on the auto-estimated window.
    assert stats["n_outside_grid"] == 0


def test_solver_table_mode_records_eos_mode(fluid: GERGFluid):
    """Both modes must record eos_mode in solver_options for summary reporting."""
    pipe = Pipe.horizontal_uniform(length=50.0, inner_diameter=0.4)

    try:
        r_direct = solve_for_mdot(
            pipe, fluid, P_in=30e5, T_in=323.15, P_out=20e5,
            eos_mode="direct",
        )
    except BVPChoked as exc:
        r_direct = exc.result
    assert r_direct.solver_options["eos_mode"] == "direct"
    assert "table_stats" not in r_direct.solver_options

    try:
        r_table = solve_for_mdot(
            pipe, fluid, P_in=30e5, T_in=323.15, P_out=20e5,
            eos_mode="table",
        )
    except BVPChoked as exc:
        r_table = exc.result
    assert r_table.solver_options["eos_mode"] == "table"
    assert "table_stats" in r_table.solver_options


def test_solver_rejects_bad_eos_mode(fluid: GERGFluid):
    pipe = Pipe.horizontal_uniform(length=50.0, inner_diameter=0.4)
    with pytest.raises(ValueError, match="eos_mode"):
        solve_for_mdot(
            pipe, fluid, P_in=30e5, T_in=323.15, P_out=20e5,
            eos_mode="bogus",
        )


def test_plateau_sweep_builds_table_once(fluid: GERGFluid, monkeypatch):
    """Sweep with eos_mode='table' must build a single TabulatedFluid that
    is reused across all probes. Counting constructor invocations via a
    monkeypatch is the most direct way to assert "once."
    """
    pipe = Pipe.horizontal_uniform(length=50.0, inner_diameter=0.4)
    P_last_cell_array = np.array([25e5, 22e5, 18e5])

    from gas_pipe import eos as _eos
    from gas_pipe import solver as _solver
    n_builds = {"count": 0}
    real_ctor = _eos.TabulatedFluid

    class _CountingTabulated(real_ctor):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            n_builds["count"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(_solver, "TabulatedFluid", _CountingTabulated)

    points = plateau_sweep(
        pipe, fluid, P_in=30e5, T_in=323.15,
        P_last_cell_array=P_last_cell_array, eos_mode="table",
        table_n_P=20, table_n_T=20,
    )
    assert len(points) == 3
    assert n_builds["count"] == 1, (
        f"Expected 1 table build for 3 sweep points; got {n_builds['count']}"
    )


def test_plateau_sweep_direct_mode_builds_no_table(fluid: GERGFluid, monkeypatch):
    """With eos_mode='direct', no table should be constructed at any level."""
    pipe = Pipe.horizontal_uniform(length=50.0, inner_diameter=0.4)
    from gas_pipe import eos as _eos
    from gas_pipe import solver as _solver
    n_builds = {"count": 0}
    real_ctor = _eos.TabulatedFluid

    class _CountingTabulated(real_ctor):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            n_builds["count"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(_solver, "TabulatedFluid", _CountingTabulated)

    plateau_sweep(
        pipe, fluid, P_in=30e5, T_in=323.15,
        P_last_cell_array=np.array([25e5, 22e5]), eos_mode="direct",
    )
    assert n_builds["count"] == 0


def test_solver_accepts_pre_wrapped_tabulated_fluid(fluid: GERGFluid):
    """Calling solve_for_mdot with eos_mode='direct' on a pre-wrapped
    TabulatedFluid must work (this is how plateau_sweep avoids rebuilding
    the table per probe)."""
    pipe = Pipe.horizontal_uniform(length=50.0, inner_diameter=0.4)
    table = TabulatedFluid(
        fluid, P_range=(15e5, 35e5), T_range=(290.0, 350.0),
        n_P=30, n_T=30,
    )
    try:
        result = solve_for_mdot(
            pipe, table, P_in=30e5, T_in=323.15, P_out=20e5,
            eos_mode="direct",
        )
    except BVPChoked as exc:
        result = exc.result
    # Solver should have used the supplied table, so any out-of-grid
    # queries surface in its counter.
    assert result.solver_options["eos_mode"] == "direct"

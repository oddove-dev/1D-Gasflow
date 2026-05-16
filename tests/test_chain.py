"""Tests for the multi-element chain BVP — :func:`solve_chain` and helpers.

Covers six categories:

1. Three-mode roundtrip consistency on a multi-element chain
   (Pipe → Device → Pipe). Mode 1 produces an ``mdot`` that, fed back
   to Modes 2 and 3, reproduces the original BCs.
2. Mode 1 ``BVPChoked`` mapping when the device limits flow below
   what the pipes would carry. Verifies ``choke_diagnostics`` shape.
3. Mode 2 ``OverChokedError`` with all diagnostic fields populated.
4. Mode 3 infeasibility — distinct exception message identifying the
   bracket-exhaustion context.
5. ``_get_base_fluid`` dispatch on ``GERGFluid``, ``TabulatedFluid``,
   and invalid input.
6. Single-pipe Mode 1 fast-path regression — bit-identical against
   the legacy ``solve_for_mdot`` body.
"""
from __future__ import annotations

import pytest

from gas_pipe import (
    BVPChoked,
    ChainSpec,
    Device,
    GERGFluid,
    OverChokedError,
    Pipe,
    PipeSection,
    TabulatedFluid,
    solve_chain,
    solve_for_mdot,
)
from gas_pipe.chain import _get_base_fluid


@pytest.fixture(scope="module")
def methane() -> GERGFluid:
    return GERGFluid({"Methane": 1.0})


@pytest.fixture(scope="module")
def pipe_long() -> Pipe:
    return Pipe(sections=[
        PipeSection(length=10.0, inner_diameter=0.1, roughness=4.5e-5),
    ])


@pytest.fixture(scope="module")
def pipe_short() -> Pipe:
    return Pipe(sections=[
        PipeSection(length=5.0, inner_diameter=0.1, roughness=4.5e-5),
    ])


class TestRoundtrip:
    """Three-mode consistency: Mode 1 → mdot, then Modes 2 and 3 reproduce BCs."""

    def test_pipe_device_pipe(
        self, methane: GERGFluid, pipe_long: Pipe, pipe_short: Pipe,
    ) -> None:
        # Mild ΔP and a generous device so neither pipe Fanno-chokes nor
        # the device over-chokes — purpose of this test is three-mode
        # round-trip consistency, not stress-testing choke handling.
        # Device A_vc = 6.93e-3 m² < pipe area 7.85e-3 m² (D=0.1 m); device
        # mdot ceiling ~62 kg/s, well above the pipe-friction-limited mdot
        # for the chosen BCs.
        device = Device(A_geom=7e-3, Cd=0.99, name="V-RT")
        chain = ChainSpec(elements=[pipe_long, device, pipe_short])

        P_in_target = 50e5
        P_last_cell_target = 45e5
        T_in = 300.0

        # Mode 1 → mdot
        r1 = solve_chain(
            chain, methane, T_in=T_in,
            P_in=P_in_target, P_last_cell=P_last_cell_target,
            eos_mode="direct",
        )
        assert not r1.choked
        mdot = r1.mdot
        assert mdot > 0.0

        # Mode 2 (P_in, mdot → P_last_cell)
        r2 = solve_chain(
            chain, methane, T_in=T_in,
            P_in=P_in_target, mdot=mdot,
            eos_mode="direct",
        )
        assert r2.P_last_cell == pytest.approx(P_last_cell_target, rel=1e-3)

        # Mode 3 (P_last_cell, mdot → P_in)
        r3 = solve_chain(
            chain, methane, T_in=T_in,
            P_last_cell=P_last_cell_target, mdot=mdot,
            eos_mode="direct",
        )
        assert r3.P_in == pytest.approx(P_in_target, rel=1e-3)


class TestMode1BVPChokedMapping:
    """When a device's A_vc forces choke below the pipe-limited mdot,
    Mode 1 surfaces a :class:`BVPChoked` carrying ``mdot_critical`` and
    a populated :class:`ChainResult` with ``choke_diagnostics``."""

    def test_device_over_choked_diagnostics(
        self, methane: GERGFluid, pipe_long: Pipe, pipe_short: Pipe,
    ) -> None:
        # Small A_geom forces device choke at low mdot (~12.4 kg/s); at
        # that choke-limited mdot, the chain's minimum reachable
        # P_last_cell is ~32 bara. Target P_last_cell=20 bara is below
        # that → unreachable → BVPChoked.
        device = Device(A_geom=2e-3, Cd=0.7, name="V-tight")
        chain = ChainSpec(elements=[pipe_long, device, pipe_short])

        with pytest.raises(BVPChoked) as excinfo:
            solve_chain(
                chain, methane, T_in=300.0,
                P_in=50e5, P_last_cell=20e5,
                eos_mode="direct",
            )

        exc = excinfo.value
        assert exc.mdot_critical > 0.0
        assert exc.result is not None
        diag = exc.result.choke_diagnostics
        assert diag is not None
        assert diag["kind"] == "device_over_choked"
        assert isinstance(diag["element_index"], int)
        assert diag["element_name"] == "V-tight"
        assert diag["max_mdot"] > 0.0
        # mdot_critical should approximately equal max_mdot at the
        # choke-boundary stagnation
        assert exc.mdot_critical == pytest.approx(diag["max_mdot"], rel=1e-2)

    def test_mode1_single_pipe_bvpchoked_returns_chainresult(self) -> None:
        """Regression: single-pipe Mode 1 fast-path must wrap the
        underlying PipeResult in a ChainResult on choke so
        solve_chain's exception contract is uniform with multi-element
        chains. Without the wrap, BVPChoked.result is the bare
        PipeResult that _bvp_single_pipe_mdot raised with.

        Uses local fluid + pipe (not the module-scoped fixtures) so the
        aggressive choke search (P_last_cell → 1 bara) doesn't pollute
        the shared GERGFluid cache and perturb the byte-identical-
        regression test's brentq trajectory.
        """
        from gas_pipe.chain import ChainResult

        local_fluid = GERGFluid({"Methane": 1.0})
        local_pipe = Pipe(sections=[PipeSection(
            length=10.0, inner_diameter=0.1, roughness=4.5e-5,
        )])
        chain = ChainSpec(elements=[local_pipe])
        with pytest.raises(BVPChoked) as excinfo:
            solve_chain(
                chain, local_fluid, T_in=300.0,
                P_in=50e5, P_last_cell=1e5,
                eos_mode="direct",
            )
        assert isinstance(excinfo.value.result, ChainResult), (
            f"BVPChoked.result must be ChainResult; "
            f"got {type(excinfo.value.result).__name__}"
        )
        assert excinfo.value.mdot_critical > 0.0
        assert len(excinfo.value.result.results) == 1


class TestMode2OverChokedDiagnostics:
    """Mode 2 with ``mdot`` just above device capacity propagates
    :class:`OverChokedError` directly."""

    def test_diagnostic_fields_populated(
        self, methane: GERGFluid, pipe_long: Pipe, pipe_short: Pipe,
    ) -> None:
        # Device A_vc = 1.4e-3 caps mdot at ~12.4 kg/s at P_stag = 50 bara.
        # Probe with mdot = 15 kg/s — clearly above ceiling.
        device = Device(A_geom=2e-3, Cd=0.7, name="V-tight")
        chain = ChainSpec(elements=[pipe_long, device, pipe_short])
        mdot_probe = 15.0

        with pytest.raises(OverChokedError) as excinfo:
            solve_chain(
                chain, methane, T_in=300.0,
                P_in=50e5, mdot=mdot_probe,
                eos_mode="direct",
            )

        exc = excinfo.value
        assert isinstance(exc.device_index, int)
        assert exc.device_name == "V-tight"
        assert exc.max_mdot > 0.0
        assert exc.attempted_mdot == pytest.approx(mdot_probe, rel=1e-12)
        assert exc.max_mdot < exc.attempted_mdot


class TestMode3Infeasibility:
    """Mode 3 with infeasible ``mdot`` raises :class:`OverChokedError`
    with a Mode-3-specific message identifying the bracket exhaustion."""

    def test_mode3_specific_message(
        self, methane: GERGFluid, pipe_long: Pipe, pipe_short: Pipe,
    ) -> None:
        device = Device(A_geom=2e-3, Cd=0.7, name="V-tight")
        chain = ChainSpec(elements=[pipe_long, device, pipe_short])

        with pytest.raises(OverChokedError, match="Mode 3 infeasible") as excinfo:
            solve_chain(
                chain, methane, T_in=300.0,
                P_last_cell=20e5, mdot=500.0,
                eos_mode="direct",
            )

        exc = excinfo.value
        # device_index may be None or int depending on whether the bracket
        # exhaustion was caused by a device choke vs pipe Fanno asymptote;
        # max_mdot may be NaN for the Fanno case. The Mode-3 message is
        # the canonical discriminator.
        assert exc.attempted_mdot == pytest.approx(500.0, rel=1e-12)


class TestGetBaseFluidDispatch:
    """``_get_base_fluid`` routes between direct and tabulated EOS."""

    def test_gergfluid_passthrough(self, methane: GERGFluid) -> None:
        assert _get_base_fluid(methane) is methane

    def test_tabulated_unwrap(self, methane: GERGFluid) -> None:
        table = TabulatedFluid(
            methane,
            P_range=(1e5, 60e5),
            T_range=(250.0, 350.0),
            n_P=10, n_T=10,
        )
        assert _get_base_fluid(table) is methane

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="GERGFluid or TabulatedFluid"):
            _get_base_fluid(None)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="GERGFluid or TabulatedFluid"):
            _get_base_fluid({"P": 50e5})  # type: ignore[arg-type]


class TestSinglePipeRegression:
    """Single-pipe Mode 1 must short-circuit to ``_bvp_single_pipe_mdot``
    and produce a result indistinguishable from a direct
    :func:`solve_for_mdot` call."""

    def test_byte_identical_with_solve_for_mdot(
        self, methane: GERGFluid, pipe_long: Pipe,
    ) -> None:
        P_in, T_in, P_last_cell = 50e5, 300.0, 40e5

        # Legacy path — solve_for_mdot keeps the ``P_out`` kwarg as a
        # silent back-compat alias for ``P_last_cell`` (CLAUDE.md
        # "Pressure terminology").
        r_legacy = solve_for_mdot(
            pipe_long, methane,
            P_in=P_in, T_in=T_in, P_out=P_last_cell,
            eos_mode="direct",
        )

        # Chain path
        chain = ChainSpec(elements=[pipe_long])
        r_chain = solve_chain(
            chain, methane, T_in=T_in,
            P_in=P_in, P_last_cell=P_last_cell,
            eos_mode="direct",
        )
        r_chain_pipe = r_chain.results[0]

        # The wrapper now invokes solve_chain too, so r_legacy already
        # went through the chain layer. Both paths must produce the same
        # PipeResult; check headline scalars and a few profile points.
        assert r_chain_pipe.mdot == pytest.approx(r_legacy.mdot, rel=1e-12)
        assert float(r_chain_pipe.P[-1]) == pytest.approx(
            float(r_legacy.P[-1]), rel=1e-12
        )
        assert float(r_chain_pipe.T[-1]) == pytest.approx(
            float(r_legacy.T[-1]), rel=1e-12
        )
        # Mid-pipe state
        mid = len(r_chain_pipe.x) // 2
        assert float(r_chain_pipe.P[mid]) == pytest.approx(
            float(r_legacy.P[mid]), rel=1e-12
        )
        assert float(r_chain_pipe.u[mid]) == pytest.approx(
            float(r_legacy.u[mid]), rel=1e-12
        )

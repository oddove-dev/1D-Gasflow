"""Tests for the HEM throat solver — ``hem_throat`` and ``props_Ps_via_jt``.

Covers five categories:

1. Roundtrip consistency between mode A (A_vc given) and mode B (mdot given).
2. Pure-methane single-phase limit: c_HEM ≈ state.a and M ≈ 1 at choke.
3. Skarv-typical composition regression against stored reference data
   (write-then-skip on first run, compare on subsequent runs).
4. Tabulated-EOS coverage for ``props_Ps_via_jt``.
5. Validation guards (substring-matched ValueErrors) and post-convergence
   sanity (|u_vc − c_HEM|/c_HEM < 1% at choke).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gas_pipe import (
    GERGFluid,
    TabulatedFluid,
    ThroatState,
)

SKARV_COMPOSITION: dict[str, float] = {
    "Methane": 0.78,
    "Ethane": 0.10,
    "Propane": 0.05,
    "n-Butane": 0.02,
    "IsoButane": 0.01,
    "Nitrogen": 0.02,
    "CarbonDioxide": 0.02,
}

SKARV_P_STAG: float = 50e5
SKARV_T_STAG: float = 373.15
SKARV_A_VC: float = 1.0e-4

SKARV_P_BACK_SUBCRITICAL_RATIO: float = 0.9
# 0.3 → 15 bara, comfortably inside GERG-2008 parametrization for this
# composition. If a future CoolProp/composition change drops 15 bara out
# of range, raise this to 0.4.
SKARV_P_BACK_CHOKED_RATIO: float = 0.3

REF_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def methane_direct() -> GERGFluid:
    return GERGFluid({"Methane": 1.0})


@pytest.fixture(scope="module")
def skarv_direct() -> GERGFluid:
    return GERGFluid(SKARV_COMPOSITION)


@pytest.fixture(scope="module")
def skarv_table(skarv_direct: GERGFluid) -> TabulatedFluid:
    return TabulatedFluid(
        skarv_direct,
        P_range=(1e5, 60e5),
        T_range=(200.0, 400.0),
        n_P=30,
        n_T=30,
    )


def _check_or_write_reference(ts: ThroatState, ref_path: Path) -> None:
    """First-run write + ``pytest.skip``; subsequent-run regression compare."""
    REF_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ref_path.exists():
        data = {
            "P": ts.P,
            "T": ts.T,
            "rho": ts.rho,
            "u": ts.u,
            "M": ts.M,
            "mdot": ts.mdot,
            "A_vc": ts.A_vc,
            "G": ts.G,
            "choked": ts.choked,
            "c_HEM": ts.c_HEM,
        }
        ref_path.write_text(json.dumps(data, indent=2) + "\n")
        pytest.skip(
            f"Reference data created at {ref_path.name}; "
            "re-run pytest to validate against stored values. "
            "Manual regeneration: delete the file to re-write."
        )
    ref = json.loads(ref_path.read_text())
    for field in ("P", "T", "rho", "u", "M", "mdot", "A_vc", "G", "c_HEM"):
        got = getattr(ts, field)
        expected = ref[field]
        assert got == pytest.approx(expected, rel=1e-3), (
            f"{field}: got {got!r}, ref {expected!r}, "
            f"rel diff {(got - expected) / expected:.2e}"
        )
    assert ts.choked == ref["choked"]


class TestRoundtrip:
    """Mode A (A_vc) → mdot → Mode B (mdot) → A_vc should round-trip exactly."""

    def test_methane_choked(self, methane_direct: GERGFluid) -> None:
        ts_a = methane_direct.hem_throat(50e5, 300.0, 5e5, A_vc=1e-4)
        assert ts_a.choked
        ts_b = methane_direct.hem_throat(50e5, 300.0, 5e5, mdot=ts_a.mdot)
        assert ts_b.A_vc == pytest.approx(ts_a.A_vc, rel=1e-5)

    def test_methane_subcritical(self, methane_direct: GERGFluid) -> None:
        ts_a = methane_direct.hem_throat(50e5, 300.0, 45e5, A_vc=1e-4)
        assert not ts_a.choked
        ts_b = methane_direct.hem_throat(50e5, 300.0, 45e5, mdot=ts_a.mdot)
        assert ts_b.A_vc == pytest.approx(ts_a.A_vc, rel=1e-5)

    def test_skarv_choked(self, skarv_direct: GERGFluid) -> None:
        P_back = SKARV_P_BACK_CHOKED_RATIO * SKARV_P_STAG
        ts_a = skarv_direct.hem_throat(
            SKARV_P_STAG, SKARV_T_STAG, P_back, A_vc=SKARV_A_VC
        )
        assert ts_a.choked
        ts_b = skarv_direct.hem_throat(
            SKARV_P_STAG, SKARV_T_STAG, P_back, mdot=ts_a.mdot
        )
        assert ts_b.A_vc == pytest.approx(ts_a.A_vc, rel=1e-5)


class TestPureMethaneLimit:
    """Pure dry methane is single-phase throughout, so HEM sound speed
    collapses to the GERG single-phase ``state.a`` and the choked throat
    sits at Mach ≈ 1."""

    def test_c_HEM_matches_state_a(self, methane_direct: GERGFluid) -> None:
        ts = methane_direct.hem_throat(50e5, 300.0, 5e5, A_vc=1e-4)
        assert ts.choked
        state_t = methane_direct.props(ts.P, ts.T)
        assert ts.c_HEM == pytest.approx(state_t.a, rel=1e-3)

    def test_M_approx_one_at_choke(self, methane_direct: GERGFluid) -> None:
        ts = methane_direct.hem_throat(50e5, 300.0, 5e5, A_vc=1e-4)
        assert ts.choked
        assert ts.M == pytest.approx(1.0, abs=0.01)


class TestSkarvReference:
    """Skarv-typical composition regression against stored reference data.

    HEM trajectory may dip into the metastable region near choke for this
    mixture; that's expected, since ``props_Ps_via_jt`` rides on ``props``
    which already has metastable extrapolation. Verifies throat sanity, not
    single-phase residency.
    """

    def test_subcritical_direct(self, skarv_direct: GERGFluid) -> None:
        P_back = SKARV_P_BACK_SUBCRITICAL_RATIO * SKARV_P_STAG
        ts = skarv_direct.hem_throat(
            SKARV_P_STAG, SKARV_T_STAG, P_back, A_vc=SKARV_A_VC
        )
        assert not ts.choked
        _check_or_write_reference(
            ts, REF_DATA_DIR / "hem_skarv_subcritical_direct.json"
        )

    def test_choked_direct(self, skarv_direct: GERGFluid) -> None:
        P_back = SKARV_P_BACK_CHOKED_RATIO * SKARV_P_STAG
        ts = skarv_direct.hem_throat(
            SKARV_P_STAG, SKARV_T_STAG, P_back, A_vc=SKARV_A_VC
        )
        assert ts.choked
        _check_or_write_reference(
            ts, REF_DATA_DIR / "hem_skarv_choked_direct.json"
        )


class TestPropsPsViaJtTable:
    """``props_Ps_via_jt`` is on the FluidEOSBase protocol; verify the
    table-backed routine converges to the same tolerance and stays close
    to the direct GERG result within bilinear-interpolation noise."""

    def test_skarv_isentropic_flash_table(
        self, skarv_table: TabulatedFluid, skarv_direct: GERGFluid
    ) -> None:
        state_up_direct = skarv_direct.props(SKARV_P_STAG, SKARV_T_STAG)
        state_up_table = skarv_table.props(SKARV_P_STAG, SKARV_T_STAG)

        P_new = 0.5 * SKARV_P_STAG
        state_ps_direct = skarv_direct.props_Ps_via_jt(P_new, state_up_direct)
        state_ps_table = skarv_table.props_Ps_via_jt(P_new, state_up_table)

        s_target = state_up_table.s
        s_scale = max(abs(s_target), 1.0)
        assert abs(state_ps_table.s - s_target) / s_scale < 1e-7
        assert state_ps_table.T == pytest.approx(state_ps_direct.T, rel=5e-3)
        assert state_ps_table.rho == pytest.approx(state_ps_direct.rho, rel=5e-3)


class TestValidation:
    """``hem_throat`` input validation. ``pytest.raises(match=...)`` with a
    stable substring so future message rewordings don't break the tests."""

    def test_neither_A_vc_nor_mdot(self, methane_direct: GERGFluid) -> None:
        with pytest.raises(ValueError, match="exactly one of A_vc or mdot"):
            methane_direct.hem_throat(50e5, 300.0, 5e5)

    def test_both_A_vc_and_mdot(self, methane_direct: GERGFluid) -> None:
        with pytest.raises(ValueError, match="exactly one of A_vc or mdot"):
            methane_direct.hem_throat(50e5, 300.0, 5e5, A_vc=1e-4, mdot=1.0)

    def test_P_back_greater_than_P_stag(self, methane_direct: GERGFluid) -> None:
        with pytest.raises(ValueError, match="P_back must be < P_stag"):
            methane_direct.hem_throat(50e5, 300.0, 60e5, A_vc=1e-4)

    def test_P_back_equal_to_P_stag(self, methane_direct: GERGFluid) -> None:
        with pytest.raises(ValueError, match="P_back must be < P_stag"):
            methane_direct.hem_throat(50e5, 300.0, 50e5, A_vc=1e-4)

    def test_non_positive_P_stag(self, methane_direct: GERGFluid) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            methane_direct.hem_throat(-1e5, 300.0, 5e5, A_vc=1e-4)

    def test_non_positive_T_stag(self, methane_direct: GERGFluid) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            methane_direct.hem_throat(50e5, 0.0, 5e5, A_vc=1e-4)

    def test_non_positive_A_vc(self, methane_direct: GERGFluid) -> None:
        with pytest.raises(ValueError, match="A_vc must be positive"):
            methane_direct.hem_throat(50e5, 300.0, 5e5, A_vc=-1e-4)

    def test_non_positive_mdot(self, methane_direct: GERGFluid) -> None:
        with pytest.raises(ValueError, match="mdot must be positive"):
            methane_direct.hem_throat(50e5, 300.0, 5e5, mdot=-1.0)


class TestPostConvergenceSanity:
    """At choke, |u_vc − c_HEM|/c_HEM should sit in the silent <1% band.

    A 1-5% deviation emits HEMConsistencyWarning (still passes the test
    but signals numerical noise); >5% raises HEMConsistencyError (which
    would surface as a test failure on the upstream ``hem_throat`` call).
    """

    def test_methane_choked(self, methane_direct: GERGFluid) -> None:
        ts = methane_direct.hem_throat(50e5, 300.0, 5e5, A_vc=1e-4)
        assert ts.choked
        deviation = abs(ts.u - ts.c_HEM) / ts.c_HEM
        assert deviation < 0.01, f"deviation = {deviation * 100:.3f}%"

    def test_skarv_choked(self, skarv_direct: GERGFluid) -> None:
        P_back = SKARV_P_BACK_CHOKED_RATIO * SKARV_P_STAG
        ts = skarv_direct.hem_throat(
            SKARV_P_STAG, SKARV_T_STAG, P_back, A_vc=SKARV_A_VC
        )
        assert ts.choked
        deviation = abs(ts.u - ts.c_HEM) / ts.c_HEM
        assert deviation < 0.01, f"deviation = {deviation * 100:.3f}%"

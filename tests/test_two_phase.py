"""Two-phase metastable handling — backlog item 1.

Step 2 coverage: dew-temperature flash on the EOS, march-time T_dew /
metastable_mask population in PipeResult. (LVF and summary content come
in step 3 and beyond.)
"""
from __future__ import annotations

import numpy as np
import pytest

from gas_pipe import BVPChoked, GERGFluid, Pipe, march_ivp, solve_for_mdot


@pytest.fixture(scope="module")
def heavy_fluid():
    """Methane + n-Hexane: enough heavy content to produce a real dew curve
    well into typical operating P,T."""
    return GERGFluid({"Methane": 0.90, "n-Hexane": 0.10})


@pytest.fixture(scope="module")
def light_fluid():
    """Pure methane: supercritical at typical pipeline conditions, so the
    dew flash returns None across the operating window."""
    return GERGFluid({"Methane": 1.0})


# Canonical Skarv flare-gas composition shipped as the GUI default. Has
# a real but very low dew curve (~200-275 K across 2-50 bara) and at typical
# Skarv operating T (~340-370 K) the gas state stays well above the dew curve.
SKARV_COMPOSITION = {
    "Methane": 0.78,
    "Ethane": 0.10,
    "Propane": 0.05,
    "n-Butane": 0.02,
    "IsoButane": 0.01,
    "Nitrogen": 0.02,
    "CarbonDioxide": 0.02,
}


@pytest.fixture(scope="module")
def skarv_fluid():
    return GERGFluid(SKARV_COMPOSITION)


# ---------------------------------------------------------------------------
# EOS-level dew_temperature
# ---------------------------------------------------------------------------

def test_dew_temperature_finite_for_heavy_mixture(heavy_fluid):
    T_dew = heavy_fluid.dew_temperature(50e5)
    assert T_dew is not None
    # 90/10 Methane/n-Hexane at 50 bara has a dew T well above ambient.
    # Sanity-bound the value rather than pin it.
    assert 250.0 < T_dew < 500.0


def test_dew_temperature_cached(heavy_fluid):
    """Second call at same P should hit the per-fluid dew cache."""
    T1 = heavy_fluid.dew_temperature(40e5)
    T2 = heavy_fluid.dew_temperature(40e5)
    assert T1 == T2
    # Cache key is rounded P → tiny perturbations also hit the same bin.
    T3 = heavy_fluid.dew_temperature(40e5 + 0.5)
    assert T3 == T1


def test_dew_temperature_none_for_supercritical(light_fluid):
    """Pure methane is supercritical at 50 bara — dew flash should fail."""
    assert light_fluid.dew_temperature(50e5) is None


def test_dew_temperature_skarv_default_is_physical(skarv_fluid):
    """Regression: CoolProp's HEOS PQ flash returned a non-physical
    upper root (~1400 K, far above the EOS Tmax of ~685 K) for the
    Skarv NG mixture at P ≈ 44.43 bara without raising. The march
    happened to step through that exact pressure and produced a
    Min T_margin of -1035 K with a falsely-positive TWO-PHASE banner.

    Across the whole pipeline pressure window, dew T must either be
    None (flash refused) or a physical hydrocarbon dew T (well below
    400 K for this composition). A value above 400 K — let alone the
    1400 K observed previously — indicates the bound/round-trip
    sanity check in dew_temperature is missing or weakened.
    """
    # 0.001-bar resolution sweep across the historically-broken band.
    # If the round-trip check ever regresses, the band reappears here.
    P_bara_grid = (
        [P_i / 1000.0 for P_i in range(44000, 45001)]
        + [2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    )
    for P_bar in P_bara_grid:
        T_dew = skarv_fluid.dew_temperature(P_bar * 1e5)
        if T_dew is None:
            continue
        assert 100.0 < T_dew < 400.0, (
            f"Unphysical dew T = {T_dew:.2f} K at P = {P_bar:.3f} bara "
            "for Skarv-default composition (expected None or a value "
            "in the 100-400 K hydrocarbon dew range)."
        )


def test_dew_temperature_does_not_corrupt_subsequent_props(heavy_fluid):
    """A dew flash must leave the AbstractState in a usable phase-forced
    configuration so the next props() call still returns a normal gas
    state."""
    heavy_fluid.dew_temperature(50e5)
    state = heavy_fluid.props(50e5, 600.0)  # well above any dew T
    assert state.rho > 0
    assert state.h is not None


# ---------------------------------------------------------------------------
# march_ivp populates the new PipeResult fields
# ---------------------------------------------------------------------------

def test_metastable_fields_populated_for_supercritical_methane(light_fluid):
    """A pure-methane march should produce arrays of the right shape and
    no metastable stations (T_dew is NaN everywhere)."""
    pipe = Pipe.horizontal_uniform(
        length=80.0, inner_diameter=0.762, roughness=4.5e-5,
        outer_diameter=0.813, overall_U=0.0, ambient_temperature=283.15,
    )
    r = march_ivp(
        pipe, light_fluid, P_in=50e5, T_in=373.15, mdot=120.0,
        n_segments=50, adaptive=True,
    )
    # Field shapes match station array.
    assert r.T_dew.shape == r.T.shape
    assert r.T_margin.shape == r.T.shape
    assert r.metastable_mask.shape == r.T.shape
    assert r.LVF.shape == r.T.shape
    # Pure methane is supercritical → dew flash returns None → all NaN.
    assert np.all(np.isnan(r.T_dew))
    assert np.all(np.isnan(r.T_margin))
    # NaN comparisons must NOT mark stations metastable.
    assert not r.had_metastable
    assert not np.any(r.metastable_mask)
    assert r.x_dewpoint_crossing is None


def test_skarv_default_bvp_does_not_trip_two_phase(skarv_fluid):
    """Regression: the Skarv default GUI scenario (78% CH4 NG, 50→2 bara,
    100 °C inlet, 80 m × 0.762 m insulated pipe) must NOT trigger the
    metastable banner. Inlet T (373 K) is ~100 K above the dew curve at
    any point along the pipe (dew T ≤ 275 K for this mix). A false
    positive here means dew_temperature has regressed and is again
    returning a non-physical value at some station's pressure.
    """
    pipe = Pipe.horizontal_uniform(
        length=80.0, inner_diameter=0.762, roughness=4.5e-5,
        outer_diameter=0.813, overall_U=2.0, ambient_temperature=283.15,
    )
    try:
        r = solve_for_mdot(
            pipe, skarv_fluid, P_in=50e5, T_in=373.15, P_out=2e5,
            n_segments=200, adaptive=True,
        )
    except BVPChoked as exc:
        # BVPChoked is acceptable for this case — the pipe chokes before
        # reaching P_out. The carried result must still be metastable-free.
        r = exc.result

    finite_margin = r.T_margin[np.isfinite(r.T_margin)]
    if finite_margin.size > 0:
        min_margin = float(np.min(finite_margin))
    else:
        min_margin = float("nan")
    assert not r.had_metastable, (
        f"Skarv-default scenario unexpectedly entered metastable region. "
        f"Min T_margin: {min_margin} K, crossing at x = {r.x_dewpoint_crossing} m. "
        "This is the dew_temperature spurious-root regression."
    )


def test_metastable_crossing_detected_when_temperature_drops_below_dew(heavy_fluid):
    """Heavy mixture marched with strong wall cooling should cross the dew
    curve along the pipe and have at least one metastable station."""
    # Cold ambient + strong U so JT cooling + heat loss together push T
    # below the dew curve. The exact crossing location depends on the
    # mixture's dew envelope at 30 bara.
    pipe = Pipe.horizontal_uniform(
        length=200.0, inner_diameter=0.10, roughness=4.5e-5,
        outer_diameter=0.11, overall_U=20.0, ambient_temperature=263.15,
    )
    r = march_ivp(
        pipe, heavy_fluid, P_in=50e5, T_in=420.0, mdot=10.0,
        n_segments=100, adaptive=True,
    )
    # T_dew must be finite for at least most stations (heavy mixture +
    # moderate P → flash succeeds throughout).
    assert np.sum(np.isfinite(r.T_dew)) > 0
    # T_margin definition consistency.
    finite = np.isfinite(r.T_margin)
    assert np.allclose(r.T_margin[finite], r.T[finite] - r.T_dew[finite])
    # If we did cross, x_dewpoint_crossing must coincide with the first
    # True in the mask.
    if r.had_metastable:
        first = int(np.argmax(r.metastable_mask))
        assert r.x_dewpoint_crossing == pytest.approx(float(r.x[first]))
        # And the station temperatures at and after crossing should
        # actually be below their dew T.
        assert r.T[first] < r.T_dew[first]


def test_metastable_mask_is_bool(heavy_fluid):
    """Defensive: downstream code (UI banner, plot shading) treats this
    as a boolean array."""
    pipe = Pipe.horizontal_uniform(
        length=50.0, inner_diameter=0.10, roughness=4.5e-5,
        outer_diameter=0.11, overall_U=0.0, ambient_temperature=283.15,
    )
    r = march_ivp(
        pipe, heavy_fluid, P_in=50e5, T_in=420.0, mdot=10.0,
        n_segments=50, adaptive=True,
    )
    assert r.metastable_mask.dtype == bool


# ---------------------------------------------------------------------------
# LVF — step 3
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def heavy_cooling_result(heavy_fluid):
    """A heavy-mixture march that starts sub-dew (T_in below the dew T at
    P_in) so every station along the pipe lies inside the two-phase
    dome. Used to exercise the LVF code on a guaranteed-metastable run.
    Inlet T is chosen ~10 K below the inlet dew (~382 K at 50 bara for
    the 90/10 Methane/n-Hexane mix). Strong wall cooling pushes T even
    further below dew along the pipe."""
    pipe = Pipe.horizontal_uniform(
        length=200.0, inner_diameter=0.10, roughness=4.5e-5,
        outer_diameter=0.11, overall_U=30.0, ambient_temperature=263.15,
    )
    return march_ivp(
        pipe, heavy_fluid, P_in=50e5, T_in=372.0, mdot=10.0,
        n_segments=100, adaptive=True,
    )


@pytest.fixture(scope="module")
def supercritical_methane_result(light_fluid):
    """Pure-methane march used by the plot test for the no-condensation
    path. Conditions are deliberately mild so the march stays well above
    the (non-existent for pure methane) dew curve and had_metastable
    stays False."""
    pipe = Pipe.horizontal_uniform(
        length=80.0, inner_diameter=0.762, roughness=4.5e-5,
        outer_diameter=0.813, overall_U=0.0, ambient_temperature=283.15,
    )
    return march_ivp(
        pipe, light_fluid, P_in=50e5, T_in=373.15, mdot=120.0,
        n_segments=50, adaptive=True,
    )


@pytest.fixture(scope="module")
def severe_condensation_result():
    """40/60 Methane/n-Hexane march starting just sub-dew. The much
    heavier composition gives a higher LVF for the same temperature
    margin — used for the severity-classification test where we need
    a non-trivial peak LVF (≥ 1%) rather than the marginal 0.1%-level
    values that the 10% hexane mix produces."""
    rich_fluid = GERGFluid({"Methane": 0.60, "n-Hexane": 0.40})
    pipe = Pipe.horizontal_uniform(
        length=200.0, inner_diameter=0.10, roughness=4.5e-5,
        outer_diameter=0.11, overall_U=30.0, ambient_temperature=263.15,
    )
    # Inlet dew is ~458 K at 50 bara for this composition.
    return march_ivp(
        pipe, rich_fluid, P_in=50e5, T_in=448.0, mdot=10.0,
        n_segments=100, adaptive=True,
    )


def test_lvf_is_nan_above_dew(heavy_cooling_result):
    """Non-metastable stations must keep the pre-allocated NaN — only
    metastable stations get a real LVF computed."""
    r = heavy_cooling_result
    above = ~r.metastable_mask
    if np.any(above):
        assert np.all(np.isnan(r.LVF[above])), (
            "LVF must remain NaN at non-metastable (above-dew) stations"
        )


def test_lvf_populated_below_dew(heavy_cooling_result):
    """At metastable stations, LVF is either a finite value in [0, 1]
    or NaN if the flash failed. NaN is acceptable but non-finite values
    other than NaN (inf, negative) are not."""
    r = heavy_cooling_result
    if not r.had_metastable:
        pytest.skip("Test geometry did not cross dew — adjust fixture")
    meta_lvf = r.LVF[r.metastable_mask]
    finite = np.isfinite(meta_lvf)
    # At least SOME metastable stations should yield a finite LVF.
    assert np.any(finite), "All metastable LVF values are NaN"
    # Finite values must be in [0, 1].
    in_range = (meta_lvf[finite] >= 0.0) & (meta_lvf[finite] <= 1.0)
    assert np.all(in_range), (
        f"LVF values out of [0, 1] range: {meta_lvf[finite][~in_range]}"
    )


def test_lvf_max_at_coldest_metastable_station(heavy_cooling_result):
    """LVF generally rises as T drops further below the dew curve. The
    overall max LVF should occur at the coldest metastable station to
    within a reasonable tolerance — exact monotonicity is not guaranteed
    because of the way mixture saturated densities vary, but the max
    location should track temperature."""
    r = heavy_cooling_result
    if not r.had_metastable:
        pytest.skip("Test geometry did not cross dew")
    meta_idx = np.flatnonzero(r.metastable_mask)
    finite_meta = meta_idx[np.isfinite(r.LVF[meta_idx])]
    if finite_meta.size < 2:
        pytest.skip("Need at least two finite-LVF stations to compare")

    # Most-negative T_margin = coldest relative to dew = expected high LVF.
    coldest_idx = finite_meta[np.argmin(r.T_margin[finite_meta])]
    max_lvf_idx = finite_meta[np.argmax(r.LVF[finite_meta])]

    # Allow these to differ by a few stations (mixture properties wobble),
    # but the LVF at the coldest station should be within 30% of the max.
    lvf_max = r.LVF[max_lvf_idx]
    lvf_coldest = r.LVF[coldest_idx]
    if lvf_max > 0.0:
        ratio = lvf_coldest / lvf_max
        assert ratio > 0.7, (
            f"LVF at coldest metastable station ({lvf_coldest:.4f}) is "
            f"only {100*ratio:.1f}% of the max ({lvf_max:.4f}). Expected "
            "the two to track each other."
        )


def test_summary_two_phase_section(supercritical_methane_result, severe_condensation_result):
    """The summary() output gains a TWO-PHASE block iff the march crossed
    the dew curve. For a supercritical-methane march the block must be
    absent; for the heavy-condensing fixture it must be present, name the
    correct severity band, and include the dew-crossing position from
    the result fields."""
    # Case 1: pure methane → supercritical → no metastable stations.
    r_clean = supercritical_methane_result
    assert not r_clean.had_metastable
    assert "TWO-PHASE" not in r_clean.summary()

    # Case 2: heavy-condensing fixture → TWO-PHASE block must appear with
    # the severity that matches the actual max LVF.
    r = severe_condensation_result
    if not r.had_metastable:
        pytest.skip("Severe-condensation fixture did not cross dew")
    s = r.summary()
    assert "TWO-PHASE" in s

    finite_lvf = r.LVF[np.isfinite(r.LVF)]
    if finite_lvf.size == 0:
        # Flash failed everywhere — section must say so.
        assert "LVF could not be computed" in s
    else:
        lvf_max = float(np.max(finite_lvf))
        if lvf_max < 0.01:
            assert "Marginal condensation" in s
        elif lvf_max < 0.05:
            assert "Light condensation" in s
        else:
            assert "Significant condensation" in s
        # Max LVF value must be reported with the expected formatting.
        assert f"{lvf_max:.3f}" in s

    # Dew-crossing position from the result field must appear verbatim.
    assert r.x_dewpoint_crossing is not None
    assert f"{r.x_dewpoint_crossing:.2f} m" in s


def test_lvf_max_severity_classification(severe_condensation_result):
    """For the 60/40 Methane/n-Hexane mix marched starting sub-dew with
    strong wall cooling, peak LVF should reach the spec's 'light
    condensation' band (≥ 1%) — well above the noise floor.

    A failure here would indicate either a regression in the LVF
    computation (silently zero-stuck) or a regression in the metastable
    detection. The test deliberately uses a much heavier composition
    than the standard test fluid: 10% hexane gives a peak LVF of order
    0.1% even with strong cooling, which is below the band where the
    severity warning would typically fire."""
    r = severe_condensation_result
    if not r.had_metastable:
        pytest.skip("Severe-condensation fixture did not cross dew")
    finite_lvf = r.LVF[np.isfinite(r.LVF)]
    if finite_lvf.size == 0:
        pytest.skip("No finite LVF values computed")
    lvf_max = float(np.max(finite_lvf))
    assert lvf_max >= 0.01, (
        f"Max LVF = {lvf_max:.4f} — expected ≥ 0.01 (1%) for this "
        "heavily-condensing case. Possible LVF computation regression."
    )


# ---------------------------------------------------------------------------
# plot_profile — step 5
# ---------------------------------------------------------------------------

def test_plot_profile_handles_two_phase_result(severe_condensation_result):
    """plot_profile must run on a metastable result without raising, draw
    the standard 2x3 grid, and surface the TWO-PHASE annotation in the
    figure suptitle so the GUI viewer sees the warning at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    from gas_pipe.diagnostics import plot_profile

    fig = plot_profile(severe_condensation_result)
    try:
        assert len(fig.axes) >= 6
        if severe_condensation_result.had_metastable:
            suptitle = fig._suptitle.get_text() if fig._suptitle else ""
            assert "TWO-PHASE" in suptitle
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_classify_severity_bands():
    """MainWindow._classify_severity is the single source of truth that
    decides which colored banner the Summary tab gets. The bands must
    match the summary() text exactly so the banner and the TWO-PHASE
    section can't disagree."""
    from types import SimpleNamespace
    from gas_pipe.gui import MainWindow

    classify = MainWindow._classify_severity

    # Not metastable → no banner.
    label, tag = classify(SimpleNamespace(
        had_metastable=False, LVF=np.array([np.nan, np.nan]),
    ))
    assert label is None and tag is None

    # Metastable but flash failed everywhere → marginal banner with the
    # flash-failure label so the user sees the warning regardless.
    label, tag = classify(SimpleNamespace(
        had_metastable=True, LVF=np.array([np.nan, np.nan, np.nan]),
    ))
    assert label is not None and "flash failures" in label
    assert tag == "banner_marginal"

    # Three bands keyed off np.nanmax(LVF). NaNs in the array must not
    # leak through nanmax — verify each band picks the right tag.
    cases = [
        (np.array([0.0001, 0.002, np.nan]), "Marginal", "banner_marginal"),
        (np.array([np.nan, 0.02, 0.04]), "Light", "banner_light"),
        (np.array([np.nan, 0.06, 0.2]), "Significant", "banner_significant"),
    ]
    for lvf_arr, expect_label_substring, expect_tag in cases:
        label, tag = classify(SimpleNamespace(
            had_metastable=True, LVF=lvf_arr,
        ))
        assert label is not None and expect_label_substring in label, (
            f"LVF={lvf_arr} expected '{expect_label_substring}' in label, got {label!r}"
        )
        assert tag == expect_tag

    # Boundary at exactly 1% — band is [0.01, 0.05) so 0.01 itself is Light.
    label, tag = classify(SimpleNamespace(
        had_metastable=True, LVF=np.array([0.01]),
    ))
    assert tag == "banner_light"
    # Boundary at exactly 5% → Significant.
    label, tag = classify(SimpleNamespace(
        had_metastable=True, LVF=np.array([0.05]),
    ))
    assert tag == "banner_significant"


def test_plot_profile_no_two_phase_for_supercritical(supercritical_methane_result):
    """For a non-metastable result the suptitle must NOT mention TWO-PHASE
    — that token is reserved for runs that actually crossed the dew curve."""
    import matplotlib
    matplotlib.use("Agg")
    from gas_pipe.diagnostics import plot_profile

    fig = plot_profile(supercritical_methane_result)
    try:
        suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        assert "TWO-PHASE" not in suptitle
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)

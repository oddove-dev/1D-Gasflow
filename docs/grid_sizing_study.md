# EOS table grid-size — choice of default

A one-time sweep across grid resolutions N ∈ {10, 15, 20, 25, 30, 40, 50, 75, 100}
and BVP segment counts ∈ {20, 50, 100, 200, 400} on the Skarv default
BVP confirmed that the current `TabulatedFluid` default of **N = 50**
is appropriate. At N = 50 the bilinear error is Δṁ < 0.001 % vs direct
GERG-2008, max station ΔP < 25 kPa, max ΔT < 0.6 K, and the single-BVP
speedup is ~20×. Bilinear error scales as O(1/N²) as expected.

N = 20 already clears the knee criterion (Δṁ < 0.01 %, ΔP < 50 kPa,
ΔT < 1 K) on this case, but the speedup advantage over N = 50 is
marginal (~14× vs ~20×, with build time saving only ~5 s of a
~150 s direct reference). N = 50 was chosen for headroom against
other compositions / BC bands not covered by this sweep, while
keeping table build under 6 seconds. No adaptive `default_grid_size`
logic was introduced — the constant is hardcoded.

**Re-run the study** (the raw script, CSV, plots and full discussion
are preserved in git history at commit `fd7beea`,
*"Grid-size sweep study for tabulated EOS"*) if any of the following
change materially: the EOS backend, the interpolation method, the
P/T window estimation, or the operating regime (composition far from
methane-dominated NG, P-range outside ~2–60 bara, or a substantially
longer pipe whose accumulated bilinear error matters more than for
80 m).

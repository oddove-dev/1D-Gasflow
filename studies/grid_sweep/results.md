# Grid-size sweep — Skarv default BVP

## Methodology

Skarv default case (composition: {'Methane': 0.85, 'Ethane': 0.1, 'Propane': 0.05}; pipe: 80 m × 762 mm × 813 mm OD, ε=45 μm, U=2.0 W/m²/K) at P_in=50 bara, T_in=100 °C, target P_out=2 bara (choke-limited).

- **Grid resolutions**: (10, 15, 20, 25, 30, 40, 50, 75, 100) (square N×N)
- **Segment counts**: (20, 50, 100, 200, 400)
- **Window**: auto-estimated via `estimate_operating_window` (same as the GUI default)
- Direct mode results cached once per segment count, reused for all table-mode comparisons.
- Tables built once per N, reused across all n_seg (via `eos_mode='direct'` on the pre-wrapped TabulatedFluid).

Knee-point criterion (all must hold):

- |Δṁ|/ṁ_direct < 0.01%
- max-station |ΔP| < 50 kPa
- max-station |ΔT| < 1.0 K

## Results — per segment count

### n_segments = 20 (direct-mode reference: 61.3 s)

| N | build [s] | solve [s] | speedup | Δṁ [%] | max ΔP [kPa] | max ΔT [K] | out-of-grid |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.30 | 4.6 | 13.20× | 0.0283 | 18.23 | 0.446 | 0 |
| 15 | 0.59 | 4.5 | 13.68× | 0.0147 | 5.27 | 0.123 | 0 |
| 20 | 1.04 | 4.3 | 14.14× | 0.0072 | 5.13 | 0.138 | 0 |
| 25 | 1.34 | 3.9 | 15.63× | 0.0046 | 4.44 | 0.111 | 0 |
| 30 | 2.06 | 3.6 | 17.03× | 0.0033 | 3.73 | 0.097 | 0 |
| 40 | 3.76 | 3.3 | 18.36× | 0.0019 | 3.92 | 0.102 | 0 |
| 50 | 5.61 | 3.0 | 20.61× | 0.0012 | 3.83 | 0.099 | 0 |
| 75 | 6.97 | 1.6 | 39.39× | 0.0004 | 4.40 | 0.114 | 0 |
| 100 | 23.67 | 2.7 | 22.47× | 0.0003 | 4.16 | 0.107 | 0 |

### n_segments = 50 (direct-mode reference: 70.5 s)

| N | build [s] | solve [s] | speedup | Δṁ [%] | max ΔP [kPa] | max ΔT [K] | out-of-grid |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.30 | 6.8 | 10.44× | 0.0283 | 16.46 | 0.415 | 0 |
| 15 | 0.59 | 6.0 | 11.85× | 0.0147 | 15.71 | 0.404 | 0 |
| 20 | 1.04 | 5.7 | 12.28× | 0.0072 | 3.97 | 0.094 | 0 |
| 25 | 1.34 | 5.6 | 12.68× | 0.0046 | 16.30 | 0.411 | 0 |
| 30 | 2.06 | 4.5 | 15.70× | 0.0033 | 4.98 | 0.126 | 0 |
| 40 | 3.76 | 4.1 | 17.34× | 0.0019 | 5.03 | 0.128 | 0 |
| 50 | 5.61 | 3.7 | 19.26× | 0.0012 | 4.80 | 0.123 | 0 |
| 75 | 6.97 | 1.8 | 38.84× | 0.0004 | 4.64 | 0.118 | 0 |
| 100 | 23.67 | 2.9 | 23.98× | 0.0003 | 4.64 | 0.119 | 0 |

### n_segments = 100 (direct-mode reference: 98.3 s)

| N | build [s] | solve [s] | speedup | Δṁ [%] | max ΔP [kPa] | max ΔT [K] | out-of-grid |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.30 | 9.1 | 10.86× | 0.0283 | 13.47 | 0.223 | 0 |
| 15 | 0.59 | 8.1 | 12.16× | 0.0147 | 22.71 | 0.468 | 0 |
| 20 | 1.04 | 7.5 | 13.03× | 0.0072 | 22.73 | 0.553 | 0 |
| 25 | 1.34 | 7.2 | 13.64× | 0.0046 | 21.62 | 0.444 | 0 |
| 30 | 2.06 | 6.1 | 16.16× | 0.0033 | 22.73 | 0.553 | 0 |
| 40 | 3.76 | 5.4 | 18.34× | 0.0019 | 23.20 | 0.567 | 0 |
| 50 | 5.61 | 4.8 | 20.29× | 0.0012 | 22.96 | 0.561 | 0 |
| 75 | 6.97 | 2.3 | 42.20× | 0.0004 | 22.87 | 0.558 | 0 |
| 100 | 23.67 | 3.5 | 27.97× | 0.0003 | 23.09 | 0.564 | 0 |

### n_segments = 200 (direct-mode reference: 149.7 s)

| N | build [s] | solve [s] | speedup | Δṁ [%] | max ΔP [kPa] | max ΔT [K] | out-of-grid |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.30 | 13.8 | 10.82× | 0.0283 | 7.47 | 0.167 | 0 |
| 15 | 0.59 | 12.7 | 11.78× | 0.0147 | 11.46 | 0.243 | 0 |
| 20 | 1.04 | 11.8 | 12.67× | 0.0072 | 20.89 | 0.502 | 0 |
| 25 | 1.34 | 10.5 | 14.32× | 0.0046 | 10.34 | 0.219 | 0 |
| 30 | 2.06 | 9.6 | 15.66× | 0.0033 | 20.71 | 0.498 | 0 |
| 40 | 3.76 | 8.0 | 18.76× | 0.0019 | 21.23 | 0.513 | 0 |
| 50 | 5.61 | 6.7 | 22.46× | 0.0012 | 21.19 | 0.512 | 0 |
| 75 | 6.97 | 3.4 | 44.68× | 0.0004 | 20.95 | 0.506 | 0 |
| 100 | 23.67 | 4.8 | 31.34× | 0.0003 | 21.29 | 0.514 | 0 |

### n_segments = 400 (direct-mode reference: 239.0 s)

| N | build [s] | solve [s] | speedup | Δṁ [%] | max ΔP [kPa] | max ΔT [K] | out-of-grid |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.30 | 24.3 | 9.85× | 0.0399 | 3.71 | 0.102 | 0 |
| 15 | 0.59 | 23.7 | 10.09× | 0.0535 | 20.25 | 0.512 | 0 |
| 20 | 1.04 | 21.1 | 11.31× | 0.0609 | 2.98 | 0.077 | 0 |
| 25 | 1.34 | 19.2 | 12.44× | 0.0636 | 20.79 | 0.517 | 0 |
| 30 | 2.06 | 17.5 | 13.68× | 0.0649 | 2.62 | 0.055 | 0 |
| 40 | 3.76 | 15.1 | 15.88× | 0.0663 | 2.57 | 0.052 | 0 |
| 50 | 5.61 | 7.0 | 34.11× | 0.0669 | 2.55 | 0.052 | 0 |
| 75 | 6.97 | 6.1 | 38.98× | 0.0678 | 2.48 | 0.052 | 0 |
| 100 | 23.67 | 8.8 | 27.24× | 0.0679 | 2.49 | 0.052 | 0 |

## Knee-point N vs segment count

| n_segments | knee N |
|---:|---:|
| 20 | 20 |
| 50 | 20 |
| 100 | 20 |
| 200 | 20 |
| 400 | — (no grid satisfied) |

### Scaling pattern

- Constant fit: N ≈ 20.0, R² = 1.000
- Log fit:      N ≈ 0.00·log(n_seg) + 20.00, R² = 1.000
- √ fit:        N ≈ -0.00·√n_seg + 20.00, R² = -74.731

## Caveats

### n_segments = 400 — direct-mode reference drift

The "no grid satisfied" verdict at n_seg = 400 is **not a table-accuracy
failure** — it is direct-mode numerical drift. The Skarv-default BVP has
a known per-segment Newton FD-Jacobian floor of ~1e-6 relative (see
`segment.py:_RTOL_P`). At dx = 80/400 = 0.20 m the FD step becomes a
larger fraction of the Newton search and the convergence pinpoint walks:

| run | ṁ_critical [kg/s] |
|---|---:|
| direct, n_seg ∈ {20, 50, 100, 200} | 3287.028 (all four identical) |
| **direct, n_seg = 400** | **3289.271** |
| table (all N, all n_seg) | 3287.028–3287.958 (max 0.028 % drift, all N ≤ 50) |

So the apparent ~0.06 % "table error" at n_seg = 400 is direct-mode
drift versus the table's own (stable) answer. The table mode is, if
anything, **more numerically robust** at high segment counts than direct
mode here — but we don't have a true reference to claim that
quantitatively. Stations-level ΔP/ΔT at n_seg = 400 are still within
the 50 kPa / 1 K criterion (in fact tighter than n_seg = 100 or 200),
which is consistent with the discrepancy being in ṁ_critical only.

For the purposes of choosing a default grid size we therefore exclude
n_seg = 400 from the knee analysis.

### Speedup non-monotonicity at N = 100

Speedup peaks at N = 75 (~40 ×) and dips at N = 100 (~25 ×) because
table-build time crosses the linear-in-march-cost line: build at N = 100
is 23.7 s (vs 7 s at N = 75) while the per-solve gain saturates around
N = 50–75. For BVP cases that re-use the table (plateau sweep), this
ranking changes — the build amortises, so larger N is again attractive.

## Proposed `default_grid_size(n_segments)`

The data is unambiguous: the bare knee (Δṁ < 0.01 % ∧ ΔP < 50 kPa ∧ ΔT < 1 K)
sits at **N = 20** for every segment count we trust (20–200). The
n_seg = 400 row is excluded for the reason above. There is no
n_segments dependence to fit — the table's accuracy is set by the
P,T coverage and bilinear order, not by how finely we march.

However, the **single-BVP speedup curve** (build cost included) peaks
near N = 50–75, and the current default of N = 50 already exceeds the
knee by 4× headroom (0.001 % vs 0.01 %) with build time of only 5.6 s
(~7 % of a single direct-mode BVP). For the plateau-sweep case where
the table amortises across 8 + probes, the additional cost of going
larger is negligible. So:

```python
def default_grid_size(n_segments: int) -> int:
    """Default N for the auto-built EOS table.

    Derived from studies/grid_sweep — see results.md for the data
    and knee-point criterion (Δṁ<0.01%, ΔP<50 kPa, ΔT<1 K).

    The knee is flat at N=20 across n_segments ∈ {20, 50, 100, 200};
    n_segments=400 was excluded because the direct-mode reference
    itself drifts there (FD-Jacobian noise at small dx).

    We return 50 — well above the knee — to (a) leave headroom for
    other (non-Skarv) compositions and BC bands, (b) keep table build
    under 6 s, and (c) hit the speedup sweet spot for single BVPs
    (~20×) while staying within the build-amortising regime for
    plateau sweeps.
    """
    # The function does not actually depend on n_segments. We accept
    # it anyway so callers can pass through their solver-options dict
    # uniformly, and so a future revision (e.g. larger N for
    # multi-segment marches where accumulated bilinear error matters
    # more) doesn't break call sites.
    del n_segments
    return 50
```

> **Review notes**:
> 1. The "constant 50" recommendation matches the current
>    `solve_for_mdot(table_n_P=50, table_n_T=50)` default — adopting
>    this function is a no-op for current behaviour, but locks in the
>    empirical justification for that number.
> 2. The single point N=20 (sweep knee) also satisfies the criteria
>    and runs ~25 % faster on single-BVP — worth offering as a "fast"
>    preset in the GUI rather than as a global default.
> 3. Confirm against a held-out case (different composition, e.g.
>    sour gas with significant H₂S, or a substantially longer pipe)
>    before adopting in production. The bilinear-error landscape can
>    look different in regions where Z and μ_JT have sharper gradients.

## Plots

- `plots/sweep_nseg<N>.png` — speed vs |Δṁ| per segment count
- `plots/knee_vs_nseg.png` — knee-point N as n_segments grows
- `plots/build_time.png` — table build cost vs N (with O(N²) reference)

_Generated by `studies/grid_sweep/run_sweep.py`; total wallclock: 17.0 min._

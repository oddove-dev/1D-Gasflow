# Backlog — to be implemented after GUI is complete

## Two-phase handling: warn but continue

**Status:** Not yet implemented. Solver currently raises `EOSTwoPhase` when CoolProp detects two-phase state, which terminates the IVP march.

**Desired behavior:** Detect two-phase entry, warn the user, but continue marching using metastable single-phase gas extrapolation. Matches industry practice (FSA, Honeywell flare system analyzers use Beggs-Brill which has known weaknesses at low LVF — single-phase metastable extrapolation is comparable in accuracy at low LVF and avoids BB's non-physical regime-transition jumps).

### Implementation

1. **`eos.py`** — when CoolProp would raise two-phase, force metastable gas phase via `state.specify_phase(CoolProp.iphase_gas)` and return continuous ρ, h, μ, a values from the gas-phase EOS extrapolation under the dew curve. Set a flag on the returned FluidState: `metastable: bool = False` by default, `True` when extrapolating.

2. **`PipeResult`** — add:
   - `T_dew: np.ndarray` — dew-point temperature at each station (NaN if computation fails)
   - `T_margin: np.ndarray = T - T_dew`
   - `metastable_mask: np.ndarray[bool]` — True where extrapolation was used
   - `had_metastable: bool` — convenience flag, True if any station was metastable
   - `x_dewpoint_crossing: float | None` — first x where T < T_dew

3. **`summary()`** — add a TWO-PHASE section if `had_metastable`:
   ```
   TWO-PHASE
     Dew point crossed at x = X.XX m (X% of pipe length)
     Min T_margin: -X.X K at x = X.XX m
     Metastable extrapolation used over X.X m of pipe.
   ```

4. **LVF estimation (post-march diagnostic)** — for each metastable station, run an isenthalpic flash via CoolProp `state.update(HmassP_INPUTS, h, P)` with two-phase allowed. Read `Q` (vapor quality). LVF = 1 - Q. Add `LVF: np.ndarray` to PipeResult.

5. **Severity warnings:**
   - LVF_max < 0.01: "Marginal condensation. Single-phase model accuracy comparable to FSA at low LVF."
   - 0.01 ≤ LVF_max < 0.05: "Light condensation. Single-phase ΔP may under-predict by 5-15%."
   - LVF_max ≥ 0.05: "Significant condensation. Use OLGA two-phase or HEM."

6. **GUI** — yellow banner on summary tab if had_metastable; dashed orange T_dew curve on temperature subplot.

### Test case
High-C6 composition (10% n-Hexane, 90% methane), 50 bara, 0°C ambient, 100m insulated pipe, P_out = 5 bara.

---

## Performance optimization Stage 2: interpolated EOS table

**Status:** Stage 1 done (analytic Jacobian, mdot cache, BVP tolerance — 2.5× speedup, 506s → 200s). Bottleneck remains GERG-2008 EOS evaluation: 98.6% of wall time.

**Approach:** Build 50×50 P×T interpolation table at start of solve_for_mdot, replace EOS calls with bilinear lookup in segment hot path.

### Implementation

1. Estimate operating window from BCs (P_min ≈ 0.5×P_out, P_max ≈ 1.05×P_in, T_min ≈ T_in − μ_JT×ΔP − margin, T_max ≈ T_in + 10K).
2. Build 50×50 grid of FluidState upfront (~2500 EOS calls, ~10s).
3. Bilinear interpolation in (P, T) for ρ, h, μ, cp, a, Z. Cache interpolated states.
4. Fallback to direct CoolProp call if (P, T) outside grid.
5. Validate: same final mdot to within 0.5% on Skarv default; all 59 tests pass.

**Expected speedup:** 3-5× on Stage 1 → Skarv default ~40-60s.

**When to do this:** only if 200s feels too slow in real GUI use.

---

## Outlet expansion analysis

**Status:** Not implemented. When pipe is choked, outlet pressure (e.g., 26 bara) is much higher than ambient. The drop from outlet to ambient happens outside the pipe via under-expanded jet expansion.

**Approach:** Add 1D diagnostic outputs about post-pipe physics without solving 3D jet structure (CFD territory).

### Implementation

1. **PipeResult fields when choked:**
   - `P_amb: float` — ambient pressure (default 1.013e5, configurable)
   - `expansion_isentropic`: dict with T, u, M at full isentropic expansion to P_amb (CoolProp props_Ps from outlet entropy)
   - `mach_disk_x: float` — Crist-Sherman-Glass empirical: `0.67 × D × sqrt(P_choke / P_amb)` from outlet
   - `thrust: float` — `mdot × u_outlet + (P_outlet − P_amb) × A`

2. **summary()** — add OUTLET EXPANSION section when choked:
   ```
   OUTLET EXPANSION (choked, free jet)
     Outlet (in pipe):   P = 26.18 bara, T =  70.2°C, u = 380 m/s, M = 0.95
     Fully expanded:     P =  1.01 bara, T = -85.3°C, u = 615 m/s, M ≈ 1.6
     Mach disk:          ≈ 2.6 m downstream of outlet (3.4 D)
     Thrust on pipe:     2.5 MN
     Note: Real jet structure is 3D — these are 1D approximations.
   ```

3. **GUI input** — "Ambient pressure [bara]" field in Boundary Conditions, default 1.013, used only when choked.

4. **GUI plot** — optional 7th panel showing jet axial profile (P, T, u outside pipe) using Adamson-Nicholls or simpler under-expanded jet correlations. Mark Mach disk position. Toggle via checkbox.

### Use cases
- Flare stack design: Mach disk for flame stability, thrust for structural mounting
- Blowdown discharge: jet velocity for impingement risk
- Relief tail pipes: thrust on pipe support

### Out of scope
- 3D jet shape, shock cell spacing, entrainment (CFD)
- Combustion in flare jet
- Acoustic noise prediction

---

## Constant-fluid mode (incompressible)

**Status:** Not implemented. Solver hardcoded to GERG-2008 multicomponent gas.

**Rationale:** Most pipe-flow infrastructure (geometry, fittings, friction, K-factors, Reynolds) is fluid-agnostic. Adding incompressible support extends tool to liquid pipe ΔP (cooling water, seawater, oil drainage, glycol loops, etc.) with minimal addition.

User decision: keep it simple — no IAPWS or full liquid EOS. Constant ρ, μ wrapper.

### Implementation

1. **eos.py** — add `ConstantFluid` class with constant rho, mu, name. props() returns FluidState with rho, mu set, other fields zeroed/defaulted.

2. **segment.py** — add `solve_segment_incompressible()`:
   - Drop acceleration term (ρ, u constant)
   - Drop energy equation (heat transfer doesn't affect ρ)
   - Reduces to: `ΔP = f·(L/D)·ρ·u²/2 + ρ·g·Δz`
   - Newton near-trivial, often direct solve

3. **solver.py** — march_ivp and solve_for_mdot detect ConstantFluid via isinstance, route to incompressible segment solver. BVP monotonic in mdot² (no choke), brentq converges fast.

4. **GUI** — fluid-mode radio:
   - "Multicomponent gas (GERG-2008)" — current behavior
   - "Constant fluid (ρ, μ)" — replace composition editor with two number fields (density [kg/m³], viscosity [cP]) plus name tag

5. **Reporting** — when ConstantFluid: hide Z, μ_JT, sound speed, Mach panel, choke warnings.

### Estimated effort
~1 day total: ConstantFluid 30min, incompressible segment 2h, GUI toggle 2h, reporting tweaks 1h.

### Out of scope
- Phase change (boiling, flashing)
- Temperature-dependent ρ, μ
- Non-Newtonian fluids
- Multi-pipe networks with elevation reservoirs

---

## BVPChoked summary clarity

**Status:** When BVPChoked is raised, summary shows `Outlet: P = 26.183 bara` without explicitly noting target was unreachable.

**Fix:** In summary(), when result.choked AND target P_out was in BC:

```
BOUNDARY CONDITIONS
  Inlet:        P =  50.000 bara,  T = 100.0 °C
  Target P_out: P =   2.000 bara
  Actual P_out: P =  26.183 bara,  T =  70.2 °C   [CHOKED — target unreachable]
```

Requires passing target P_out into PipeResult.boundary_conditions dict in solve_for_mdot.

Trivial change, defer until other features stable.

---

## Chord method retry monitoring

**Status:** Stage 1 introduced chord method. In Skarv test, two segments (x=57.60 and x=79.60) needed retry-with-half-dx fallback. Tests pass 59/59 but worth monitoring.

**If it becomes a problem:**
- Track chord retry frequency across diverse cases
- If >5% of segments need retry, implement hybrid: chord for first 5 Newton iter, fresh Jacobian on iter 6-10
- Or: detect residual stall, refresh Jacobian on the spot

Not blocking.

---

## Historical notes

**Stage 1 performance optimization (DONE):**
- Loosened BVP tolerance 1e-4 → 1e-3
- Explicit mdot cache in solve_for_mdot
- Chord method for segment Newton (single Jacobian per segment)
- Result: 506s → 200s (2.5×), 59/59 tests pass

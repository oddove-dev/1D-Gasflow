# Backlog — gas_flow

Format conventions:
- **Active** lists what is currently being worked on (typically 0–3 items).
- **Planned** lists committed work not yet started.
- **Considered, deferred** lists ideas not currently prioritized.
- **Done** lists completed work with date prefix (YYYY-MM-DD), newest first.
- Move items between sections as state changes. When completing a task,
  include the BACKLOG.md update in the same commit.

## Active

None at present. EOS table work complete (Item 2) with adaptive grid
initialization via L/D similarity.

## Planned

### Inline device model and HEM source — AIV groundwork

This sequence implements an inline device model for pressure-reducing
elements (PSV, orifice plate, choke valve, RO) placed between two pipe
sections. Produces upstream/downstream pipe states via HEM throat solution
and Borda-Carnot transition. Replaces the previous "Upstream sources" item
and lays groundwork for EI AVIFF T2.7 acoustic source PWL.

Architectural principles locked in design phase:

- Device is a node between two pipe sections; "no upstream pipe" (PSV on
  a tank with u_upstream≈0) is a special case handled by a convenience
  constructor on the same general model.
- HEM as single source model for all phases (gas, near-dew, two-phase);
  no frozen-flow alternative, no method selection.
- η_dissipation in the vena-contracta → pipe-inlet transition is a
  *computed result* from mass/energy/momentum conservation + GERG EOS,
  not a tunable parameter. Borda-Carnot momentum balance always on;
  η falls out as η = 1 − (u_inlet/u_vc)². Bounds: η → 1 for sharp
  expansion (A_vc ≪ A_pipe, typical PSV), η → 0 for marginal expansion.
- mdot is a system-level quantity, not a per-device input. Devices
  always use geometry-driven mode internally (A_geom, Cd from user).
  System BVP supports three modes: (P_in, P_out) → mdot; (P_in, mdot)
  → P_out; (P_out, mdot) → P_in. Specify exactly two of three.
- EI AVIFF T2.7.3 is the sole source PWL method. SFF=6 derived from
  ThroatState.choked flag. No alternative source models (Lighthill,
  acceleration-loss formulations explicitly excluded).
- Propagation via piecewise 60·L/D attenuation; logarithmic summation
  for multiple sources in the same line. TMM-style propagation
  explicitly out of scope (AIFF concern; AIV in gas_flow is screening).

Sequence (remaining; Item numbers assigned at implementation start):

- **Acoustic source PWL (EI AVIFF T2.7.3)**: Apply Carucci-Mueller
  formula at each device node and auto-detected section-boundary
  contractions where M_downstream ≥ 0.999·mach_choke. Inputs from
  upstream stagnation (computed by device model). SFF=6 from
  ThroatState.choked, 0 otherwise. Output:
  `ChainResult.acoustic_sources: list[SourceLocation]` with
  (x, PWL_dB, sonic_flag, source_kind). Fanno-asymptote mid-pipe
  choking NOT classified as AVIFF source (outside T2.7 calibration
  basis); raise `AcousticSourceWarning` if encountered with optional
  override flag.
  Tests: reproduce EI AVIFF Appendix D worked examples within
  tolerance.
  Estimate: ~1 day.

- **Acoustic propagation profile**: PWL(x) along chain via piecewise
  60·L/D attenuation through multi-section (D_int switches at section
  boundaries and across devices; attenuation accumulates as running
  sum). Logarithmic summation when multiple upstream sources contribute
  to a given point. Output: `ChainResult.acoustic_profile` as
  ndarray over segment centers, with per-segment contribution breakdown
  for diagnostics. 155 dB screening threshold flagged in output.
  Estimate: ~1 day.

### Other planned items (unchanged)

- **Network topology**: T-junctions and parallel branches. Outer iteration
  on junction pressures, inner per-pipe solver. Architecture described in
  CLAUDE.md Roadmap section. Will extend chain-mode to per-branch mdot,
  and acoustic propagation to junction-summation logic.

- **Outlet expansion analysis**: 1D diagnostics for choked outlets where
  pipe-end P >> ambient — Mach disk position, fully expanded jet state,
  thrust on pipe support. Flare-stack / blowdown discharge use cases.
  Detailed design notes in git history at commit `2b944d9`.

- **Constant-fluid mode (incompressible)**: `ConstantFluid` wrapper with
  fixed ρ, μ; bypasses energy equation and acceleration term. Extends
  scope to liquid pipe ΔP (cooling water, glycol, oil drainage). Detailed
  design notes in git history at commit `2b944d9`.

- **BVPChoked summary clarity**: When BVPChoked is raised, summary should
  explicitly flag the *target* P_out as unreachable rather than just
  showing the actual choke-limited outlet. Requires passing target P_out
  into `PipeResult.boundary_conditions`. Trivial fix.

- **Held-out validation**: Sour gas compositions, very long pipes, helium-
  rich compositions. Confirms tabulated EOS generalizes beyond Skarv default
  BVP. Run when concrete out-of-distribution case appears, or as periodic
  confidence check.

## Considered, deferred

- **FSA / OLGA-SS comparison study** on common cases. Useful for
  documentation of where this solver differs from commercial tools (mainly
  choke asymptote resolution). Defer until concrete need arises.

- **Acoustic spectral characterization** (broadband Strouhal-scaled or
  similar). Extension of acoustic source strength once basic implementation
  is in place.

- **Diffuser geometry extension for transition**: ESDU-style empirical
  recovery efficiency for conical diffusers (half-angle, L/D inputs).
  Default Borda-Carnot covers sharp expansion correctly (PSV outlets,
  sudden area change); add only if non-sharp diffuser geometries appear
  in scope.

- **Chord method retry monitoring**: Stage 1 perf work introduced chord
  Newton (single Jacobian per segment) with FD-Jacobian fallback. Currently
  ~2 of 200 Skarv segments need the retry; not a problem at present. If
  retry rate climbs past ~5%, switch to hybrid chord/fresh-Jacobian scheme.

## Done

- **2026-05-14**: Item 4 — Inline device model + multi-element BVP.
  `Device.solve` does stagnation Newton → HEM throat (mode A) →
  Borda-Carnot 2D Newton transition; `Device.from_stagnation` covers
  the PSV-on-tank case. `ChainSpec` + `solve_chain` provide three-mode
  BVP dispatch (P_in+P_out → mdot, P_in+mdot → P_out, P_out+mdot →
  P_in); `OverChokedError` maps to `BVPChoked` in Mode 1 (with
  reachability check) and surfaces with Mode-3-specific message in
  Mode 3 infeasibility. `solve_for_mdot` rewired as 14-line wrapper.
  `TabulatedFluid.base_fluid` accessor added so chain code routes HEM
  through `GERGFluid` while keeping pipe march on the table. Second
  item of the AIV / HEM-source sequence. Three commits: scaffolding
  (`69fc758`), implementation (`feacc77`), tests (this commit).
- **2026-05-14**: Item 3 — HEM throat solver core (`GERGFluid.hem_throat`,
  `props_Ps_via_jt`). Bounded-Brent G_max maximization on the (P_t, s_stag)
  isentrope; `ThroatState` with all fields populated for both `A_vc` and
  `mdot` input modes; `HEMConsistencyError`/`Warning` post-convergence
  sanity band. First item of the AIV / HEM-source sequence. Three commits:
  scaffolding (`6c690eb`), implementation (`5c66358`), tests (this commit).
- **2026-05-14**: Adaptive grid initialization via L/D similarity
  (dx_initial = D, α=1.0). Replaces hardcoded n_segments default.
- **2026-05-14**: GUI restructure — hide n_segments under Adaptive on;
  Linear/Dimensionless modes when Adaptive off.
- **2026-05-13**: EOS table mode (Item 2) with 50×50 default grid, sweep-
  verified accuracy. See `docs/grid_sizing_study.md`.
- **2026-05-13**: Renamed "Verify EOS accuracy" to "Verify table accuracy"
  for precision (compares table vs direct EOS, not EOS vs reference data).
- **2026-05-13**: Item 27 — multi-section pipe with Borda-Carnot expansion
  and Crane contraction K-factor at section boundaries.
- **2026-05-10**: Item 1 — two-phase metastable handling with severity-
  banded warnings (LVF-based marginal/light/significant classification).

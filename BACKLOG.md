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

- **Network topology**: T-junctions and parallel branches. Outer iteration
  on junction pressures, inner per-pipe solver. Architecture described in
  CLAUDE.md Roadmap section.

- **Acoustic source strength**: Compute local and total acoustic power from
  solved flow profile, particularly through choke asymptote. Output as
  input to AIFF TMM analysis. Method selection at implementation time.

- **Upstream sources (PSV, orifice)**: Isentropic expansion to throat with
  full energy balance (kinetic term included), then geometry-dependent
  kinetic energy recovery to pipe inlet. Boundary condition wrapper around
  pipe solver. Critical for downstream material temperature in cryogenic
  depressurization scenarios.

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

- **Chord method retry monitoring**: Stage 1 perf work introduced chord
  Newton (single Jacobian per segment) with FD-Jacobian fallback. Currently
  ~2 of 200 Skarv segments need the retry; not a problem at present. If
  retry rate climbs past ~5%, switch to hybrid chord/fresh-Jacobian scheme.

## Done

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

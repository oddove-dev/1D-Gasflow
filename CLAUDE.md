# CLAUDE.md — Gas Pipe Solver

## Working conventions

When making changes that affect architecture, defaults, or design principles,
update the relevant section of CLAUDE.md or docs/ with a brief note. Keep
notes terse and principle-focused — implementation details belong in code
comments, not here.

## Backlog

Active work, planned items, and recent completions are tracked in
`BACKLOG.md`. When completing a task from BACKLOG.md, move it to the
"Done" section with a date in the same commit that implements it.

## Working preferences

How we collaborate on this repo:

- Commits only when explicitly asked. Don't pre-emptively commit "finished"
  work — wait for the instruction.
- No `git push` without explicit confirmation, even after committing.
- Three-commit pattern for items with an empirical study attached:
  implementation → study artifacts → cleanup+docs (with study deleted, doc
  referencing the study commit hash).
- Backlog items are numbered ("Item 1", "Item 2", "Item 27", …); commit
  message subjects reference the number ("Item N complete: …").
- Visual GUI verification is expected before committing any GUI-touching
  item. Launch with `python -m gas_pipe` and walk through the relevant
  scenarios with the user.
- Delete scratch scripts (`_isolate_cost.py`, `_profile_*.py`,
  `_quick_*.py`, …) before staging a commit — they're not part of the
  tracked surface.
- LF/CRLF warnings from `git add` on Windows are noise; ignore them.

## Purpose

Single-pipe compressible flow solver for steady-state choked and sub-choked
gas flow analysis. Built primarily for:

- AIFF acoustic source characterization at choke regions
- Detailed verification of network-level results from commercial tools
  (FSA, OLGA-SS)
- Internal AkerBP technical analysis where precision per pipe matters more
  than network-level coverage

Not intended as a replacement for network solvers. Single-pipe
(multi-section) topology — diameter changes within one run are supported
(Borda-Carnot expansion, Crane contraction K-factor at section boundaries),
but T-junctions and parallel branches are out of scope. Complementary to
FSA/OLGA, not competing with them.

## Physics scope

- Single-phase gas flow
- GERG-2008 EOS via CoolProp (Skarv composition typical, broader hydrocarbon
  compositions supported within GERG parameterization range)
- Compressible Fanno-type formulation with energy balance and JT effects
- Turbulent regime (Colebrook-White friction)
- Mach range where compressibility matters (typically M > 0.1, up to choke)

## Pressure terminology

The flow has three distinct pressure stations that the code must keep apart:

```
[upstream source] ─► P_in ─► [pipe model] ─► P_last_cell ─► [free expansion] ─► P_out
                     ↑                       ↑                                  ↑
                chain BC          last cell of last pipe              chain BC (future)
                                  (computed result)                   post-expansion
```

- **`P_in`** — chain-level upstream BC. Its physical interpretation depends
  on the first chain element: with a leading Pipe (the only topology
  ``ChainSpec`` currently allows) it equals ``P_first_cell`` of pipe 1.
  With a leading Device — future work — it would be stagnation pressure
  upstream of the device.
- **`P_first_cell`, `P_last_cell`** — per-pipe internal computed values
  (first/last cell of a pipe). ``PipeResult`` does not expose them as
  scalar fields; read them as ``result.P[0]`` and ``result.P[-1]`` or
  via ``PipeResult.summary()``. ``ChainResult`` exposes
  ``P_last_cell`` as a top-level field (the last cell of the last pipe).
- **`P_out`** — *reserved name* for the chain-level downstream BC
  (post free-jet expansion). Not yet coupled to the solver. The
  legacy ``ChainResult.P_out`` and ``solve_chain(P_out=…)`` kwarg have
  been renamed to ``P_last_cell`` — ``ChainResult.P_out`` remains as a
  ``DeprecationWarning``-raising alias, and ``solve_for_mdot`` keeps a
  silent ``P_out`` kwarg as a back-compat alias for ``P_last_cell``.
  See *Roadmap → Outlet expansion model* for the planned coupling.
- **`P_end`** — deprecated; never use. If you see it in older
  documentation, it meant ``P_last_cell``.

Why this matters: in a choked flare system the last-cell pressure can
be 20+ bara while atmospheric ``P_out`` is ~1 bara. Conflating them
produces silent physical errors. The renamed BC also clarifies that
specifying ``P_last_cell`` below the choke-limited value is genuinely
infeasible (you cannot have a last-cell pressure below the choke
pressure), as distinct from a downstream BC that lies behind a free
expansion.

Current topology constraint: ``ChainSpec`` validates that the first and
last elements are Pipes. Device-first and Device-last chains require
additional solver work (stagnation-as-``P_in``; Borda-Carnot to chain
outlet) — both in the Roadmap.

Backward march for choked-device downstream pipes: when a device chokes
mid-chain in Mode 1, pipes downstream of the choke point are marched
BACKWARD from the ``P_last_cell`` BC toward the choked-device
transition. The downstream pipe's ``P_first_cell`` is *computed*
(typically close to ``P_last_cell`` when downstream friction is low),
not derived from the Borda-Carnot transition. This satisfies the chain
BC exactly while respecting the device-imposed ``mdot`` ceiling. The
Borda-Carnot transition predicted by ``DeviceResult.transition`` stays
attached as a diagnostic of what *would have* been the downstream
pipe's inlet had the BC not constrained it. v1 requires downstream
pipes to be adiabatic (``overall_U == 0``); diabatic downstream raises
``BackwardMarchDiabaticNotSupported``. The diabatic extension would
need a forward-T-backward-P iteration that is not yet in scope.

## Architecture

### EOS evaluation

Tabulated by default, 50×50 grid, bilinear interpolation. Out-of-grid
evaluations fall back to direct GERG-2008 calls automatically.

Verified accuracy: Δṁ < 0.001% vs direct evaluation on Skarv default BVP
(50→2 bara, typical composition). See `docs/grid_sizing_study.md` for the
empirical study and trigger conditions for re-running it.

### Discretization

Adaptive refinement is default. Three-layer mechanism:

- **Layer 1**: Mach-based predictor (Fanno asymptote, ideal-gas) with
  refinement trigger — shrinks `dx` *before* the segment is solved if
  predicted M_downstream > mach_choke.
- **Layer 2**: Bisect against choke when post-solve Mach ≥ mach_choke;
  locates `x_choke` with `min_dx` resolution.
- **Layer 3**: `dx_target` shrinks when M is in the asymptote band or
  ramping toward choke; grows back (capped at L/initial) when M is flat.
  Plus a Fanno-asymptote detector — 10 consecutive segments pinned at
  `min_dx` with M > 0.95·mach_choke → declared choked without M crossing
  the threshold (otherwise the Layer-1 predictor would halve forever).

Driven by Mach number, not by P/T/ρ gradients directly. This reflects that
the characteristic length scale for gradient development collapses near
choke where λ⁻ → 0.

Initial grid for adaptive uses L/D similarity: dx_initial = α·D with α=1.0,
bounded to [10, 500] segments. This gives consistent dimensionless resolution
across pipe sizes from millimeters to meters, since the natural length scale
for compressible pipe flow is set by geometry (Fanno length D/f), not by
absolute pipe length.

When adaptive is off (diagnostic/study mode), two discretization modes:

- **Linear**: dx = L/N, user supplies N. Useful for convergence studies
  and comparing against tools that use linear discretization.
- **Dimensionless**: dx = α·D, user supplies α. Reproducible fine resolution
  scaled with diameter, useful for cross-tool comparisons.

### Solver

Per-segment Newton iteration with FD-Jacobian. Tolerance 1e-6 with fallback
to 1e-4.

Known characteristic: at very fine discretization (n_segments ≈ 400 linear
on Skarv), direct mode drifts ~0.07% in ṁ_critical due to FD-Jacobian noise
when dP per segment becomes small. Table mode does not exhibit the same
drift in our swept cases — likely because bilinear lookup is smoother per
(P,T) than iterative HEOS flash. Either way, not a concern for normal use;
adaptive mode rarely pushes segment count high enough to expose this.

## Key defaults

| Parameter | Default | Rationale |
|---|---|---|
| EOS evaluation | Tabulated 50×50 | Sweep verified; see docs/grid_sizing_study.md |
| Adaptive refinement | On | Production mode; robust across BVP range |
| Initial dx (adaptive) | D (α=1.0) | L/D similarity, empirically comfortable |
| mach_warning | 0.7 | Refinement begins as choke approached |
| mach_choke | 0.99 | Hard refinement threshold |
| min_dx | 1.0 mm | Floor for adaptive refinement — NOT the initial dx (which is α·D, ~762 mm on Skarv) |

## How to run

- `python -m gas_pipe` — launch the GUI (Skarv defaults loaded).
- `pytest tests/` — full test suite (~12 min wallclock; 85 legacy direct-mode
  tests + 17 tabulated-EOS tests).
- `pytest tests/test_eos_table.py -v` — quick targeted run for the item 2
  surface (~3–5 min).
- `pytest tests/<file>.py -q` — single-file subsets for fast iteration
  (most modules complete in seconds).
- Windows console gotcha: scripts that print `ṁ`, `°C`, `α`, `Δ`, … need
  `sys.stdout.reconfigure(encoding='utf-8')` at entry. The default cp1252
  raises `UnicodeEncodeError`.

## Test conventions

- `tests/conftest.py` sets `GAS_PIPE_DEFAULT_EOS_MODE=direct` at import
  time so the 85 pre-item-2 tests behave exactly as they did before the
  tabulated-EOS work. New tests that need tabulated coverage pass
  `eos_mode='table'` explicitly.
- One test file per topic (`test_eos.py`, `test_friction.py`,
  `test_choking.py`, `test_multi_section.py`, `test_eos_table.py`, …).
  Resist the urge to add cross-topic tests to existing files.
- Tolerances: `pytest.approx(rel=1e-3)` is typical for engineering checks;
  `rel=1e-12` for grid-point-exact assertions; `rel=5e-3` for bilinear
  mid-cell interpolation.
- BVP tests typically wrap `solve_for_mdot` in `try/except BVPChoked` and
  use `exc.result` — `BVPChoked` is a normal outcome for the Skarv-default
  geometry, not a failure.

## Known traps

CoolProp / Newton quirks that have wasted time before and are easy to
trip over:

- **HEOS Ph-flash is broken** for mixtures without a pre-built phase
  envelope — `CoolProp.HmassP_INPUTS` raises spuriously. Use
  `fluid.props_Ph_via_jt(P, state_up)` for mixtures; it does a JT estimate
  followed by a PT-flash.
- **`_eval_state_Ph` has a `try/finally`** to restore `iphase_gas` after
  the Ph flash. Without it, every subsequent PT-flash runs ~25× slower
  because the AbstractState stays in unspecified-phase mode.
- **`dew_temperature` returns `None` on spurious-root detection.** The
  HEOS PQ-flash can converge to a non-physical upper root (T ≈ 1400 K)
  near the cricondenbar; we cross-check via reverse QT-flash and bail
  out cleanly.
- **No `lru_cache` on instance methods.** It causes cross-instance cache
  pollution because `self` is hashed. Use a per-instance dict instead;
  see `GERGFluid._cache`.
- **`BVPChoked` is not an error.** It's the normal outcome when the
  target `P_last_cell` is below the choke-limited last-cell pressure;
  the exception carries a fully-populated `PipeResult` at ṁ_critical
  (or `ChainResult` when raised by `solve_chain` directly — see the
  uniform-payload contract in `chain.py`).
- **Backward-march trigger tolerance.** The `mdot_ceiling` refinement
  bisect in `_mode1_brentq` resolves the operating-regime choke
  boundary to ~1e-4 relative accuracy. Below that, the re-march at
  `0.999 · mdot_ceiling` may not land inside
  `_device_solve_at_mdot`'s just-choked window (`rel_tol = 1e-4`), so
  `throat.choked` can stay `False` even when the device is
  operationally at choke. The backward-mode trigger therefore gates
  on `throat.choked or throat.M >= 0.95` — the `M >= 0.95` clause is
  the numerical-noise buffer at the converged operating ceiling, not
  a softening of the physical "device is choked" assertion. Don't
  tighten this threshold without also tightening the bisect.

## Validation status

- **Skarv default BVP** (50→2 bara, typical hydrocarbon composition):
  verified.
- **Held-out validation** on sour gas, very long pipes, helium-rich
  compositions: pending. Run when a concrete case outside Skarv default
  appears, or as periodic confidence check.
- **Cross-tool comparison** vs FSA / OLGA-SS on common cases: not yet
  performed. Expected differences mainly in choke asymptote resolution,
  where this solver should give brighter (less smeared) profile than
  commercial tools that use simpler choke handling.

If physics scope or validation requirements change, re-run grid sweep study;
see `docs/grid_sizing_study.md` for trigger conditions.

## Roadmap

### Network topology

T-junctions and parallel branches. Current single-pipe (multi-section)
architecture is designed to be extended; junction iteration will wrap
around the existing per-pipe solver. Outer loop iterates on junction
pressures, inner loop is the per-pipe BVP solution.

### Acoustic source strength

Compute local and total acoustic power from the solved flow profile,
particularly through the choke asymptote. Output suitable as input to AIFF
TMM analysis. This is the natural coupling point between gas_flow and
AIFF-mainline. Method selection (Lighthill-type, empirical broadband, or
custom formulation based on acceleration loss) to be decided at implementation.

### Upstream sources (PSV, orifice)

Boundary condition for choked flow sources. Isentropic expansion from
stagnation (P₀, T₀) to throat conditions with full energy balance including
kinetic term:

- s_throat = s_stagnation
- h_stagnation = h_throat + u_throat²/2
- u_throat = c_throat at choke (when applicable)

Pipe inlet temperature depends on geometry-dependent kinetic energy recovery
between throat and pipe inlet. Simple isenthalpic (JT) models are physically
insufficient — kinetic energy in choked throats is significant, and its
partial conversion to thermal energy through turbulent mixing is geometry-
dependent (Borda-Carnot for sharp expansion, less for streamlined diffusers).

Critical for downstream material temperature, especially in cryogenic
depressurization scenarios. Implementation details to be specified when
work begins.

## Out of scope

- Multi-phase flow (would require fundamentally different solver)
- Transient/dynamic simulation (steady-state by design)
- Compositions outside GERG-2008 parameterization
- Laminar/transitional Reynolds regime (outside intended application)
- Network topology (T-junctions, parallel branches) — see roadmap

## References

- `docs/grid_sizing_study.md` — empirical basis for EOS table default
- Git history — empirical studies and design exploration available in
  earlier commits if needed

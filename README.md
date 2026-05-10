# Gas Pipe Analyzer

A single-phase 1D steady-state compressible gas flow solver with an interactive Tkinter GUI. Designed for engineering analysis of process piping, flare headers, blowdown lines, gas pipelines, and similar applications where pressure drop, temperature profile, and choke conditions matter.

## Capabilities

**Physics**
- Steady-state 1D Fanno flow with friction, acceleration, elevation, and heat transfer
- Real-gas thermodynamics via GERG-2008 (CoolProp HEOS backend)
- Joule-Thomson cooling captured implicitly through enthalpy formulation
- Robust choke detection with three-layer guards (predictive, in-Newton, bisection)
- Fanno asymptote handling for friction-driven choke at pipe outlet

**Solver modes**
- IVP: specify mass flow, march outlet conditions
- BVP: specify outlet pressure, find mass flow
- BVPChoked exception when target outlet pressure is unreachable, with critical mass flow returned

**GUI**
- Composition editor with normalization and live sum check
- Pipe geometry, boundary conditions, solver options
- Four output tabs: Summary, Profile Plot, Station Table, Plateau Sweep
- Threading with progress feedback and cancel button
- Plateau sweep across 8 outlet pressures to visualize choke envelope

**Validation**
- 59/59 tests passing
- Fanno-analytic agreement within 1% for ideal-gas limit
- AGA isothermal agreement within 3% for long pipes
- JT cooling within 1 K
- Heat transfer NTU model within 2 K

## Installation

Requires **Python ≥ 3.11**. On Linux, ensure the `tkinter` system package is present (`python3-tk` on Debian/Ubuntu); on Windows and macOS the standard Python distributions include it.

```
git clone https://github.com/oddove-dev/1D-Gasflow.git
cd 1D-Gasflow
pip install -e .
```

For development (tests, linters):

```
pip install -e ".[dev]"
```

Runtime dependencies: `coolprop`, `numpy`, `scipy`, `pandas`, `matplotlib`.

## Quickstart

Launch the GUI:
```
python -m gas_pipe
```

Run the test suite:
```
pytest tests/ -v
```

Run the Skarv flare header example programmatically:
```
python examples/skarv_flare_blowdown.py
```

Use the solver as a library:
```python
from gas_pipe import GERGFluid, Pipe, solve_for_mdot
from gas_pipe.errors import BVPChoked

fluid = GERGFluid({"Methane": 0.85, "Ethane": 0.10, "Propane": 0.05})
pipe = Pipe.horizontal_uniform(
    length=80.0, inner_diameter=0.762, roughness=4.5e-5,
    outer_diameter=0.813, overall_U=2.0, ambient_temperature=283.15,
)

try:
    result = solve_for_mdot(
        pipe, fluid, P_in=50e5, T_in=373.15, P_out=20e5,
        n_segments=200, adaptive=True,
    )
    print(f"Subsonic: ṁ = {result.mdot:.1f} kg/s")
except BVPChoked as exc:
    print(f"Choked at ṁ_critical = {exc.mdot_critical:.1f} kg/s")
    print(exc.result.summary())
```

## Project structure

```
gas_pipe/
├── eos.py              GERG-2008 wrapper, FluidState, caching
├── friction.py         Chen, Colebrook, blended friction factor models
├── geometry.py         Pipe class with elevation, heat transfer
├── fittings.py         K-factor fittings interface
├── segment.py          Per-segment Newton solver, choke layer 2
├── solver.py           march_ivp, solve_for_mdot, plateau_sweep
├── results.py          PipeResult dataclass with summary, profile output
├── diagnostics.py      plot_profile, plot_plateau_sweep
├── gui.py              MainWindow, threading, output tab population
├── gui_widgets.py      CompositionEditor and other reusable widgets
├── errors.py           Custom exception hierarchy
└── __main__.py         Entry point: python -m gas_pipe

tests/                  pytest test suite, 59 tests
examples/               example scripts including Skarv flare header
BACKLOG.md              development roadmap
```

## Use cases

**Flare and blowdown systems**
Choked-flow analysis for relief and blowdown lines. Identify maximum mass flow capacity of a pipe, locate choke point, evaluate ΔP across the system. The plateau sweep visualizes the entire choke envelope for design verification.

**Process gas piping**
Pressure drop and temperature profile for gas transport between process units. Real-gas effects via GERG-2008 ensure accuracy for hydrocarbon mixtures across realistic operating ranges.

**Gas pipelines**
Long-distance transport at lower Mach numbers. Friction-dominated regime well captured. AGA-isothermal cross-check available for sanity verification.

**Subsea pipelines**
Heat transfer to ambient water modeled via overall U-value. Temperature approaches ambient asymptotically along pipe length.

**Fuel gas to turbines**
Short-distance, moderate-Mach analysis where temperature profile matters for combustion conditions.

## Boundary conditions and assumptions

**Within scope**
- Single-phase gas (multicomponent via GERG-2008)
- Constant pipe area along the marched length (multi-section roadmap available)
- Steady-state operation
- Thermodynamic equilibrium

**Out of scope**
- Multiphase flow (use OLGA)
- Transient depressurization with vessel coupling (roadmap)
- Network topology with branches (single pipe only)
- Compositional tracking along pipe (composition is fixed)

**Two-phase boundary handling (in development, item 1 in BACKLOG.md)**
When the marched gas state crosses the dew curve, the solver does not abort. Instead, it uses metastable single-phase gas extrapolation, flags affected stations, and reports liquid volume fraction as a graded warning. Matches industry practice for moderate-LVF cases. For LVF above 5%, the user is directed to OLGA or HEM tools.

## Development status

**Complete**
- Core solver with 59/59 validation tests
- Full GUI with threading
- Plateau sweep visualization
- Performance Stage 1 (chord method, mdot cache, BVP tolerance)

**In progress**
- Item 1 in BACKLOG.md: two-phase metastable handling

**Roadmap**
See BACKLOG.md for prioritized development items including:
- Performance Stage 2 (interpolated EOS table, optional 5-10× speedup)
- Outlet expansion analysis (Mach disk, thrust, isentropic expansion to ambient)
- Constant-fluid mode (incompressible water, oil, brine)
- Multi-section pipe with diameter changes (Borda-Carnot)
- Fittings library with Crane TP-410 K-factors
- Save/load case (Parquet + JSON hybrid)
- Pedagogical visualizations (Fanno-line plot, sensitivity spider, blowdown simulation)
- Multi-case comparison and real-data overlay

## Reference data

GERG-2008 mixture EOS as implemented in CoolProp. The currently exposed component list (`gas_pipe.eos.ALLOWED_COMPONENTS`) covers 16 components: Methane, Ethane, Propane, n-Butane, IsoButane, n-Pentane, IsoPentane, n-Hexane, Nitrogen, CarbonDioxide, HydrogenSulfide, Water, Oxygen, Hydrogen, Helium, and Argon. Additional GERG-2008 components can be enabled by extending that tuple.

Friction factor: Chen 1979 explicit form (default), Colebrook iterative, or blended with laminar transition.

API 521 nozzle equation used as order-of-magnitude cross-check for choked relief flow.

## Limitations and caveats

The solver is a 1D model and does not capture jet structure outside the pipe outlet (shock cells, Mach disk geometry, mixing). For choked discharge to ambient, the outlet pressure reported is the pipe-side choke pressure, not ambient. The pressure drop from outlet to ambient occurs via free expansion outside the pipe.

Sound speed reported in metastable region is the gas-phase value, not the actual two-phase sound speed. For severe two-phase conditions, this affects choke detection accuracy.

For long pipes with strong asymptote behavior, the BVP solver can take several minutes per case. Performance Stage 2 (interpolated EOS table) is available as a roadmap item if speed becomes an issue.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). The software is provided "as is" without warranty of any kind (see License sections 7-8). Not validated against external regulatory standards beyond the test suite. Users should cross-check critical sizing calculations against established tools (OLGA, FSA, AFT-Arrow) for safety-critical applications.

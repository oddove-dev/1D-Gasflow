"""Grid-size sweep study for the EOS interpolation table (item 2 follow-up).

Sweeps the 50×50-default Tabulated EOS across nine grid resolutions and
five BVP segment counts, recording wall time and accuracy versus a
direct GERG-2008 reference at each (N, n_seg) combination. The result is
a CSV + plots + markdown summary used to derive an adaptive
``default_grid_size(n_segments)`` function.

This script does NOT modify any production code — it just exercises the
existing public API (TabulatedFluid, solve_for_mdot, BVPChoked).

Run from repo root:
    python studies/grid_sweep/run_sweep.py
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

# Windows console defaults to cp1252; we print ṁ and °C, so force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ensure repo root is on sys.path so 'gas_pipe' resolves when this script
# is run from anywhere.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from gas_pipe import (  # noqa: E402
    GERGFluid,
    Pipe,
    TabulatedFluid,
    estimate_operating_window,
    solve_for_mdot,
)
from gas_pipe.errors import BVPChoked  # noqa: E402

# ----------------------------------------------------------------------
# Sweep configuration
# ----------------------------------------------------------------------
GRID_SIZES = (10, 15, 20, 25, 30, 40, 50, 75, 100)
N_SEGMENTS = (20, 50, 100, 200, 400)

# Skarv default composition + BCs (kept verbatim from the GUI defaults
# so the study results map directly to user-visible behaviour).
SKARV_COMPOSITION = {
    "Methane": 0.85,
    "Ethane": 0.10,
    "Propane": 0.05,
}
SKARV_BC = {
    "P_in": 50e5,
    "T_in": 373.15,
    "P_out": 2e5,
}
SKARV_PIPE_KWARGS = dict(
    length=80.0,
    inner_diameter=0.762,
    roughness=4.5e-5,
    outer_diameter=0.813,
    overall_U=2.0,
)

OUT_DIR = _THIS_DIR
CSV_PATH = OUT_DIR / "results.csv"
PLOTS_DIR = OUT_DIR / "plots"
MD_PATH = OUT_DIR / "results.md"

KNEE_CRITERIA = {
    "mdot_rel_max": 1e-4,    # 0.01%
    "P_out_diff_Pa_max": 50_000.0,  # 50 kPa
    "T_out_diff_K_max": 1.0,
}


def _build_pipe() -> Pipe:
    return Pipe.horizontal_uniform(**SKARV_PIPE_KWARGS)


def _safe_solve(pipe, fluid, n_seg, eos_mode):
    """Run solve_for_mdot; return (PipeResult, elapsed_s, error_msg or None).

    BVPChoked is the normal Skarv outcome (50→2 bara is unreachable) —
    treated as success; the carried result is the choke-boundary state.
    """
    t0 = time.time()
    try:
        r = solve_for_mdot(
            pipe, fluid, SKARV_BC["P_in"], SKARV_BC["T_in"], SKARV_BC["P_out"],
            eos_mode=eos_mode, n_segments=n_seg,
        )
        return r, time.time() - t0, None
    except BVPChoked as exc:
        return exc.result, time.time() - t0, None
    except Exception as exc:
        return None, time.time() - t0, f"{type(exc).__name__}: {exc}"


def _station_max_diff(arr_a, arr_b):
    """Max-over-stations of |a - b|, truncated to common length."""
    n = min(len(arr_a), len(arr_b))
    if n == 0:
        return float("nan")
    return float(np.max(np.abs(arr_a[:n] - arr_b[:n])))


def main() -> None:
    print(f"Skarv sweep: {len(GRID_SIZES)} grids × {len(N_SEGMENTS)} segment counts "
          f"= {len(GRID_SIZES) * len(N_SEGMENTS)} table runs + "
          f"{len(N_SEGMENTS)} direct refs.")
    t_total_start = time.time()

    base_fluid = GERGFluid(SKARV_COMPOSITION)
    pipe = _build_pipe()

    # Auto-estimated window — same one solve_for_mdot would use internally,
    # so the sweep mirrors the user's default experience.
    P_min, P_max, T_min, T_max = estimate_operating_window(
        SKARV_BC["P_in"], SKARV_BC["T_in"], SKARV_BC["P_out"], base_fluid,
    )
    print(f"Window: P=[{P_min/1e5:.2f}, {P_max/1e5:.2f}] bara, "
          f"T=[{T_min-273.15:.1f}, {T_max-273.15:.1f}] °C")

    # ------------------------------------------------------------------
    # Direct-mode reference runs (one per segment count, cached).
    # ------------------------------------------------------------------
    direct_cache: dict[int, dict] = {}
    for n_seg in N_SEGMENTS:
        print(f"[direct ref] n_seg={n_seg} …", flush=True)
        r, elapsed, err = _safe_solve(pipe, base_fluid, n_seg, "direct")
        if err is not None:
            print(f"  FAILED: {err}")
            direct_cache[n_seg] = {"result": None, "elapsed": elapsed, "error": err}
            continue
        print(f"  done in {elapsed:.1f} s, ṁ={r.mdot:.3f} kg/s, "
              f"P_out={r.P[-1]/1e5:.3f} bara, choked={r.choked}", flush=True)
        direct_cache[n_seg] = {"result": r, "elapsed": elapsed, "error": None}

    # ------------------------------------------------------------------
    # Table-mode runs: one TabulatedFluid per N, reused across n_seg.
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for N in GRID_SIZES:
        print(f"[N={N}] building {N}×{N} table …", flush=True)
        t_build_start = time.time()
        try:
            table = TabulatedFluid(
                base_fluid, (P_min, P_max), (T_min, T_max), N, N,
            )
            t_build = time.time() - t_build_start
            build_err = None
        except Exception as exc:
            t_build = time.time() - t_build_start
            build_err = f"{type(exc).__name__}: {exc}"
            print(f"  BUILD FAILED: {build_err}")
            for n_seg in N_SEGMENTS:
                rows.append({
                    "N": N, "n_seg": n_seg, "build_s": t_build, "solve_s": float("nan"),
                    "direct_s": direct_cache[n_seg]["elapsed"], "speedup": float("nan"),
                    "mdot_table": float("nan"), "mdot_direct": float("nan"),
                    "mdot_rel_diff": float("nan"),
                    "max_P_diff_Pa": float("nan"), "max_T_diff_K": float("nan"),
                    "n_outside_grid": -1, "n_failed_build": -1,
                    "error": build_err,
                })
            continue
        stats = table.table_stats()
        print(f"  built in {t_build:.2f} s "
              f"(n_failed={stats['n_failed']})", flush=True)

        for n_seg in N_SEGMENTS:
            print(f"  [n_seg={n_seg}] solve …", end=" ", flush=True)
            # Pass the pre-built table; eos_mode='direct' keeps solver
            # from re-wrapping it.
            r_t, elapsed_t, err_t = _safe_solve(pipe, table, n_seg, "direct")
            row = {
                "N": N, "n_seg": n_seg, "build_s": t_build, "solve_s": elapsed_t,
                "direct_s": direct_cache[n_seg]["elapsed"],
                "speedup": float("nan"), "mdot_table": float("nan"),
                "mdot_direct": float("nan"), "mdot_rel_diff": float("nan"),
                "max_P_diff_Pa": float("nan"), "max_T_diff_K": float("nan"),
                "n_outside_grid": int(table.table_stats()["n_outside_grid"]),
                "n_failed_build": int(stats["n_failed"]),
                "error": err_t,
            }
            r_d = direct_cache[n_seg]["result"]
            if err_t is None and r_t is not None and r_d is not None:
                row["mdot_table"] = float(r_t.mdot)
                row["mdot_direct"] = float(r_d.mdot)
                row["mdot_rel_diff"] = (
                    abs(r_t.mdot - r_d.mdot) / max(r_d.mdot, 1e-30)
                )
                row["max_P_diff_Pa"] = _station_max_diff(r_t.P, r_d.P)
                row["max_T_diff_K"] = _station_max_diff(r_t.T, r_d.T)
                row["speedup"] = direct_cache[n_seg]["elapsed"] / max(elapsed_t, 1e-9)
            rows.append(row)
            print(f"{elapsed_t:.1f}s, "
                  f"Δṁ={row['mdot_rel_diff']*100 if math.isfinite(row['mdot_rel_diff']) else float('nan'):.4f}%, "
                  f"speedup={row['speedup']:.2f}x" if math.isfinite(row['speedup']) else "speedup=n/a",
                  flush=True)

    # ------------------------------------------------------------------
    # Write CSV.
    # ------------------------------------------------------------------
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"CSV written: {CSV_PATH}")

    # ------------------------------------------------------------------
    # Plots.
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        rows_by_seg = {}
    else:
        PLOTS_DIR.mkdir(exist_ok=True)
        rows_by_seg: dict[int, list[dict]] = {}
        for row in rows:
            rows_by_seg.setdefault(row["n_seg"], []).append(row)

        for n_seg, group in sorted(rows_by_seg.items()):
            group_sorted = sorted(group, key=lambda r: r["N"])
            N_arr = [r["N"] for r in group_sorted]
            speedup_arr = [r["speedup"] if math.isfinite(r["speedup"]) else float("nan")
                           for r in group_sorted]
            err_arr = [r["mdot_rel_diff"] * 100 if math.isfinite(r["mdot_rel_diff"]) else float("nan")
                       for r in group_sorted]

            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()
            ax1.plot(N_arr, speedup_arr, "b-o", label="speedup")
            # Clip the relative error to a small floor so log scale doesn't blow up.
            err_clipped = [max(e, 1e-6) if math.isfinite(e) else 1e-6 for e in err_arr]
            ax2.plot(N_arr, err_clipped, "r-s", label="|Δṁ| %")
            ax2.set_yscale("log")
            ax1.set_xlabel("Grid size N (N×N)")
            ax1.set_ylabel("Speedup ×", color="b")
            ax2.set_ylabel("|Δṁ| / ṁ_direct  [%, log]", color="r")
            ax1.tick_params(axis="y", labelcolor="b")
            ax2.tick_params(axis="y", labelcolor="r")
            ax1.set_title(f"n_segments = {n_seg}: speed vs accuracy")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / f"sweep_nseg{n_seg}.png", dpi=120)
            plt.close(fig)

        # Knee-point plot.
        knees: dict[int, int | None] = {}
        for n_seg, group in rows_by_seg.items():
            group_sorted = sorted(group, key=lambda r: r["N"])
            knee = None
            for r in group_sorted:
                if (math.isfinite(r["mdot_rel_diff"])
                        and r["mdot_rel_diff"] < KNEE_CRITERIA["mdot_rel_max"]
                        and math.isfinite(r["max_P_diff_Pa"])
                        and r["max_P_diff_Pa"] < KNEE_CRITERIA["P_out_diff_Pa_max"]
                        and math.isfinite(r["max_T_diff_K"])
                        and r["max_T_diff_K"] < KNEE_CRITERIA["T_out_diff_K_max"]):
                    knee = r["N"]
                    break
            knees[n_seg] = knee

        nseg_sorted = sorted(knees)
        knee_vals = [knees[k] if knees[k] is not None else float("nan") for k in nseg_sorted]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(nseg_sorted, knee_vals, "g-D")
        ax.set_xscale("log")
        ax.set_xlabel("n_segments (log)")
        ax.set_ylabel("Knee-point grid size N")
        ax.set_title(
            "Smallest N satisfying Δṁ<0.01% ∧ ΔP_out<50kPa ∧ ΔT_out<1K"
        )
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "knee_vs_nseg.png", dpi=120)
        plt.close(fig)

        # Table-build time vs N.
        build_by_N: dict[int, float] = {}
        for row in rows:
            build_by_N[row["N"]] = row["build_s"]
        N_b = sorted(build_by_N)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(N_b, [build_by_N[n] for n in N_b], "k-^")
        # Reference: quadratic scaling (table is N² grid points).
        ref_N = np.array(N_b, dtype=float)
        ref_t = build_by_N[N_b[0]] * (ref_N / ref_N[0]) ** 2
        ax.plot(N_b, ref_t, "k:", alpha=0.5, label="O(N²) reference")
        ax.set_xlabel("Grid size N")
        ax.set_ylabel("Build time [s]")
        ax.set_title("Table build time vs grid resolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "build_time.png", dpi=120)
        plt.close(fig)
        print(f"Plots in {PLOTS_DIR}")

    # ------------------------------------------------------------------
    # Markdown summary.
    # ------------------------------------------------------------------
    _write_markdown(rows, direct_cache, rows_by_seg, t_total_start)
    print(f"Markdown: {MD_PATH}")
    print(f"Total wallclock: {(time.time() - t_total_start)/60:.1f} min")


def _write_markdown(rows, direct_cache, rows_by_seg, t_total_start):
    """Compose results.md from the swept rows.

    Knee-point analysis uses the criteria in KNEE_CRITERIA. The proposed
    ``default_grid_size`` function is fitted by eye from the
    n_seg → knee_N data and meant as a starting point for the user to
    review before adopting.
    """
    # Knees per n_seg.
    knees: dict[int, int | None] = {}
    for n_seg, group in rows_by_seg.items():
        for r in sorted(group, key=lambda r: r["N"]):
            if (math.isfinite(r["mdot_rel_diff"])
                    and r["mdot_rel_diff"] < KNEE_CRITERIA["mdot_rel_max"]
                    and math.isfinite(r["max_P_diff_Pa"])
                    and r["max_P_diff_Pa"] < KNEE_CRITERIA["P_out_diff_Pa_max"]
                    and math.isfinite(r["max_T_diff_K"])
                    and r["max_T_diff_K"] < KNEE_CRITERIA["T_out_diff_K_max"]):
                knees[n_seg] = r["N"]
                break
        else:
            knees[n_seg] = None

    lines: list[str] = []
    lines.append("# Grid-size sweep — Skarv default BVP\n")
    lines.append("## Methodology\n")
    lines.append(
        f"Skarv default case (composition: {SKARV_COMPOSITION}; pipe: "
        f"80 m × 762 mm × {SKARV_PIPE_KWARGS['outer_diameter']*1000:.0f} mm OD, "
        f"ε=45 μm, U={SKARV_PIPE_KWARGS['overall_U']:.1f} W/m²/K) at "
        f"P_in={SKARV_BC['P_in']/1e5:.0f} bara, T_in={SKARV_BC['T_in']-273.15:.0f} °C, "
        f"target P_out={SKARV_BC['P_out']/1e5:.0f} bara (choke-limited).\n"
    )
    lines.append(
        f"- **Grid resolutions**: {GRID_SIZES} (square N×N)\n"
        f"- **Segment counts**: {N_SEGMENTS}\n"
        f"- **Window**: auto-estimated via `estimate_operating_window` "
        f"(same as the GUI default)\n"
        f"- Direct mode results cached once per segment count, reused for all "
        f"table-mode comparisons.\n"
        f"- Tables built once per N, reused across all n_seg (via "
        f"`eos_mode='direct'` on the pre-wrapped TabulatedFluid).\n"
    )
    lines.append("Knee-point criterion (all must hold):\n")
    lines.append(
        f"- |Δṁ|/ṁ_direct < {KNEE_CRITERIA['mdot_rel_max']*100:.2f}%\n"
        f"- max-station |ΔP| < {KNEE_CRITERIA['P_out_diff_Pa_max']/1e3:.0f} kPa\n"
        f"- max-station |ΔT| < {KNEE_CRITERIA['T_out_diff_K_max']:.1f} K\n"
    )

    # Per-segment-count results tables.
    lines.append("## Results — per segment count\n")
    for n_seg, group in sorted(rows_by_seg.items()):
        group_sorted = sorted(group, key=lambda r: r["N"])
        d_elapsed = direct_cache.get(n_seg, {}).get("elapsed", float("nan"))
        lines.append(f"### n_segments = {n_seg} "
                     f"(direct-mode reference: {d_elapsed:.1f} s)\n")
        lines.append("| N | build [s] | solve [s] | speedup | Δṁ [%] | "
                     "max ΔP [kPa] | max ΔT [K] | out-of-grid |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in group_sorted:
            speedup = f"{r['speedup']:.2f}×" if math.isfinite(r["speedup"]) else "—"
            mdot = f"{r['mdot_rel_diff']*100:.4f}" if math.isfinite(r["mdot_rel_diff"]) else "—"
            dP = f"{r['max_P_diff_Pa']/1e3:.2f}" if math.isfinite(r["max_P_diff_Pa"]) else "—"
            dT = f"{r['max_T_diff_K']:.3f}" if math.isfinite(r["max_T_diff_K"]) else "—"
            lines.append(
                f"| {r['N']} | {r['build_s']:.2f} | {r['solve_s']:.1f} | "
                f"{speedup} | {mdot} | {dP} | {dT} | {r['n_outside_grid']} |"
            )
        lines.append("")

    # Knee-point table.
    lines.append("## Knee-point N vs segment count\n")
    lines.append("| n_segments | knee N |")
    lines.append("|---:|---:|")
    for n_seg in sorted(knees):
        v = knees[n_seg]
        lines.append(f"| {n_seg} | {v if v is not None else '— (no grid satisfied)'}|")
    lines.append("")

    # Scaling analysis.
    finite = [(n, k) for n, k in knees.items() if k is not None]
    if len(finite) >= 2:
        ns = np.array([n for n, _ in finite], dtype=float)
        ks = np.array([k for _, k in finite], dtype=float)
        # Three simple fits.
        const_k = float(np.mean(ks))
        log_coef = np.polyfit(np.log(ns), ks, 1)  # k = a*log(n) + b
        sqrt_coef = np.polyfit(np.sqrt(ns), ks, 1)
        # R² for each.
        def _r2(ks_pred, ks_obs):
            ss_res = float(np.sum((ks_obs - ks_pred) ** 2))
            ss_tot = float(np.sum((ks_obs - np.mean(ks_obs)) ** 2))
            return 1.0 - ss_res / max(ss_tot, 1e-30)
        r2_const = _r2(np.full_like(ks, const_k), ks)
        r2_log = _r2(np.polyval(log_coef, np.log(ns)), ks)
        r2_sqrt = _r2(np.polyval(sqrt_coef, np.sqrt(ns)), ks)
        lines.append("### Scaling pattern\n")
        lines.append(f"- Constant fit: N ≈ {const_k:.1f}, R² = {r2_const:.3f}")
        lines.append(f"- Log fit:      N ≈ {log_coef[0]:.2f}·log(n_seg) + {log_coef[1]:.2f}, R² = {r2_log:.3f}")
        lines.append(f"- √ fit:        N ≈ {sqrt_coef[0]:.2f}·√n_seg + {sqrt_coef[1]:.2f}, R² = {r2_sqrt:.3f}\n")

    # Proposed default function (placeholder — to be edited by hand based
    # on the actual sweep numbers before adoption).
    lines.append("## Proposed `default_grid_size(n_segments)`\n")
    lines.append("```python")
    lines.append("def default_grid_size(n_segments: int) -> int:")
    lines.append('    """Default N for the auto-built EOS table.')
    lines.append("")
    lines.append("    Derived from studies/grid_sweep — see results.md for the data")
    lines.append("    and knee-point criterion (Δṁ<0.01%, ΔP<50 kPa, ΔT<1 K).")
    lines.append('    """')
    if finite and r2_const >= 0.7:
        lines.append(f"    # Knee point is essentially flat in n_segments — N≈{int(round(const_k))} suffices.")
        lines.append(f"    return {int(round(const_k))}")
    elif finite and r2_log > max(r2_const, r2_sqrt):
        lines.append(f"    # Log fit gave the best R² ({r2_log:.3f}); knee grows slowly with n_seg.")
        lines.append(f"    return max(10, int(round({log_coef[0]:.2f} * math.log(n_segments) + {log_coef[1]:.2f})))")
    elif finite:
        lines.append(f"    # √ fit gave the best R² ({r2_sqrt:.3f}); knee grows with √n_seg.")
        lines.append(f"    return max(10, int(round({sqrt_coef[0]:.2f} * math.sqrt(n_segments) + {sqrt_coef[1]:.2f})))")
    else:
        lines.append("    # Insufficient knee-point data — fall back to the spec default.")
        lines.append("    return 50")
    lines.append("```")
    lines.append("")
    lines.append("> **Review note**: the function above is fitted to the swept "
                 "data only. Confirm against a held-out case (e.g. a different "
                 "composition or geometry) before adopting in production.")
    lines.append("")

    # Misc footer.
    lines.append("## Plots\n")
    lines.append("- `plots/sweep_nseg<N>.png` — speed vs |Δṁ| per segment count")
    lines.append("- `plots/knee_vs_nseg.png` — knee-point N as n_segments grows")
    lines.append("- `plots/build_time.png` — table build cost vs N (with O(N²) reference)\n")

    lines.append(f"_Generated by `studies/grid_sweep/run_sweep.py`; total "
                 f"wallclock: {(time.time() - t_total_start)/60:.1f} min._\n")

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

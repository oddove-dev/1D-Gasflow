"""Tkinter GUI for the Gas Pipe Analyzer.

This module hosts ``MainWindow`` and the ``main()`` entry point.
"""
from __future__ import annotations

import json
import queue
import re
import threading
import time
import tkinter as tk
import traceback
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .errors import (
    BVPChoked,
    BVPNotBracketedError,
    EOSOutOfRange,
    EOSTwoPhase,
    SegmentConvergenceError,
    SolverCancelled,
)
from .eos import GERGFluid
from .geometry import Pipe, PipeSection
from .gui_widgets import CompositionEditor
from .solver import march_ivp, solve_for_mdot

if TYPE_CHECKING:
    from .results import PipeResult


class _InputValidationError(ValueError):
    """Raised by `_gather_inputs` when a value is missing/invalid/out-of-range."""


class SolverWorker:
    """Run a solver call in a background thread; report via Queue.

    The main GUI thread polls `result_queue` (typically every 100 ms via
    `root.after`). Cooperative cancel is via `cancel_event`, which the
    solver checks at safe boundaries (between probes for BVP, every 10
    segments for IVP).

    Result-queue messages, all 3-tuples:
        ('ok', result, elapsed_seconds)
        ('choked', BVPChoked_exc, elapsed_seconds)   # BVPChoked is not an error
        ('cancelled', None, elapsed_seconds)
        ('error', exception, traceback_string)
    """

    def __init__(self) -> None:
        self.result_queue: "queue.Queue[tuple]" = queue.Queue()
        self.cancel_event: threading.Event = threading.Event()
        self.thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def run_async(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        """Launch `fn(*args, **kwargs)` in a daemon thread.

        The queue receives exactly one message when the thread completes.
        Caller is responsible for clearing cancel_event before reuse if
        needed (run_async clears it automatically).
        """
        if self.is_running():
            raise RuntimeError("SolverWorker already running")
        self.cancel_event.clear()

        def _target() -> None:
            t0 = time.time()
            try:
                result = fn(*args, **kwargs)
                self.result_queue.put(("ok", result, time.time() - t0))
            except SolverCancelled:
                self.result_queue.put(("cancelled", None, time.time() - t0))
            except BVPChoked as exc:
                # BVPChoked is a normal outcome, not an error — surfaced
                # in the UI as a red [CHOKED] banner over the result tabs.
                self.result_queue.put(("choked", exc, time.time() - t0))
            except Exception as exc:
                tb = traceback.format_exc()
                self.result_queue.put(("error", exc, tb))

        self.thread = threading.Thread(target=_target, daemon=True)
        self.thread.start()

# Matplotlib is required for the Profile Plot tab. We import lazily inside
# the tab builder so that import errors here don't break the rest of the GUI.

WINDOW_TITLE = "Gas Pipe Analyzer — Single-Phase 1D Steady-State"
WINDOW_GEOMETRY = "1400x900"
WINDOW_MIN_SIZE = (1200, 700)
INPUT_PANEL_WIDTH = 420  # initial sash position

_NUMERIC_RE = re.compile(r"^-?\d*\.?\d*([eE][+-]?\d*)?$")


def _is_numeric_input(value: str) -> bool:
    return value == "" or bool(_NUMERIC_RE.match(value))


def _parse_float(value: str, name: str) -> float:
    """Parse to float. Empty / non-numeric raises _InputValidationError."""
    s = (value or "").strip()
    if not s:
        raise _InputValidationError(f"{name} is empty.")
    try:
        return float(s)
    except ValueError as exc:
        raise _InputValidationError(f"{name}: {s!r} is not a valid number.") from exc


def _parse_positive_float(value: str, name: str) -> float:
    f = _parse_float(value, name)
    if f <= 0.0:
        raise _InputValidationError(f"{name} must be > 0; got {f}.")
    return f


def _parse_nonneg_float(value: str, name: str) -> float:
    f = _parse_float(value, name)
    if f < 0.0:
        raise _InputValidationError(f"{name} must be ≥ 0; got {f}.")
    return f


def _parse_positive_int(value: str, name: str) -> int:
    f = _parse_float(value, name)
    n = int(f)
    if n != f or n <= 0:
        raise _InputValidationError(f"{name} must be a positive integer; got {value!r}.")
    return n


def _build_pipe(pipe_kwargs: dict) -> Pipe:
    """Construct a Pipe from gather_inputs' pipe_kwargs (sections in SI)."""
    sections = [
        PipeSection(
            length=s["length"],
            inner_diameter=s["inner_diameter"],
            outer_diameter=s["outer_diameter"],
            roughness=s["roughness"],
            overall_U=s["overall_U"],
        )
        for s in pipe_kwargs["sections_si"]
    ]
    return Pipe(
        sections=sections,
        ambient_temperature=pipe_kwargs["ambient_temperature"],
    )


# ----------------------------------------------------------------------
# Skarv defaults (geometry / BC / solver)
# ----------------------------------------------------------------------
_SKARV_GEOMETRY = {
    "length_m": 80.0,
    "inner_diameter_mm": 762.0,
    "roughness_um": 45.0,
    "outer_diameter_mm": 813.0,
    "overall_U": 2.0,
    "ambient_T_C": 10.0,
}

_SKARV_BC = {
    "P_in_bara": 50.0,
    "T_in_C": 100.0,
    "mode": "BVP",
    "mdot_kgs": 120.0,
    "P_out_bara": 2.0,
}

_SKARV_SOLVER = {
    "n_segments": 200,
    "adaptive": True,
    "friction_model": "blended",
    "mach_warning": 0.7,
    "mach_choke": 0.99,
    "min_dx_mm": 1.0,
}

_FRICTION_MODELS = ("blended", "chen", "colebrook", "laminar")

_STATION_COLUMNS = (
    ("x", "x [m]"),
    ("P", "P [bara]"),
    ("T", "T [°C]"),
    ("rho", "ρ [kg/m³]"),
    ("u", "u [m/s]"),
    ("a", "a [m/s]"),
    ("M", "M"),
    ("Re", "Re"),
    ("Z", "Z"),
    ("mu_JT", "μ_JT [K/bar]"),
)


# ----------------------------------------------------------------------
# Helper: scrollable frame
# ----------------------------------------------------------------------
class _ScrollableFrame(ttk.Frame):
    """A Frame whose contents can scroll vertically.

    Hosts a Canvas with an inner Frame; place children inside ``self.body``.
    The inner frame's width tracks the canvas so that horizontal layout
    behaves like a normal Frame.
    """

    def __init__(
        self, parent: tk.Misc, *, width: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(parent, **kwargs)

        canvas_kwargs: dict[str, Any] = {"highlightthickness": 0}
        if width is not None:
            # Give the canvas a real requested width so a containing
            # PanedWindow allocates this pane non-trivial space before the
            # window is mapped.
            canvas_kwargs["width"] = width
        self._canvas = tk.Canvas(self, **canvas_kwargs)
        self._vbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vbar.set)

        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._vbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.body = ttk.Frame(self._canvas)
        self._window_id = self._canvas.create_window(
            (0, 0), window=self.body, anchor="nw"
        )

        self.body.bind("<Configure>", self._on_body_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        # Mousewheel — bind on enter/leave so scrolling is local to this widget.
        self._canvas.bind("<Enter>", self._bind_mousewheel)
        self._canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_body_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window_id, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self._canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        # Windows: event.delta is a multiple of 120
        self._canvas.yview_scroll(int(-event.delta / 120), "units")


# ----------------------------------------------------------------------
# Helper: labeled entry row
# ----------------------------------------------------------------------
def _add_labeled_entry(
    parent: tk.Misc,
    row: int,
    label: str,
    var: tk.Variable,
    validate_cmd: tuple[str, str] | None = None,
    width: int = 14,
) -> ttk.Entry:
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
    entry = ttk.Entry(parent, textvariable=var, width=width, justify="right")
    if validate_cmd is not None:
        entry.configure(validate="key", validatecommand=validate_cmd)
    entry.grid(row=row, column=1, sticky="ew", pady=2)
    parent.columnconfigure(1, weight=1)
    return entry


# ======================================================================
# MainWindow
# ======================================================================
class MainWindow:
    """Top-level GUI controller. Owns all widgets and (eventually) the worker."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.minsize(*WINDOW_MIN_SIZE)

        self._validate_cmd = (self.root.register(_is_numeric_input), "%P")

        # ---- Tk variables ------------------------------------------------
        # Geometry — list of section dicts (display units: m, mm, mm, μm, W/m²/K).
        # The Skarv default is a single uniform section.
        self._section_rows: list[dict[str, float]] = [{
            "length_m": float(_SKARV_GEOMETRY["length_m"]),
            "id_mm": float(_SKARV_GEOMETRY["inner_diameter_mm"]),
            "od_mm": float(_SKARV_GEOMETRY["outer_diameter_mm"]),
            "roughness_um": float(_SKARV_GEOMETRY["roughness_um"]),
            "U": float(_SKARV_GEOMETRY["overall_U"]),
        }]
        self.var_T_amb = tk.StringVar(value=str(_SKARV_GEOMETRY["ambient_T_C"]))
        # BC
        self.var_P_in = tk.StringVar(value=str(_SKARV_BC["P_in_bara"]))
        self.var_T_in = tk.StringVar(value=str(_SKARV_BC["T_in_C"]))
        self.var_mode = tk.StringVar(value=str(_SKARV_BC["mode"]))
        self.var_mdot = tk.StringVar(value=str(_SKARV_BC["mdot_kgs"]))
        self.var_P_out = tk.StringVar(value=str(_SKARV_BC["P_out_bara"]))
        # Solver
        self.var_n_seg = tk.StringVar(value=str(_SKARV_SOLVER["n_segments"]))
        self.var_adaptive = tk.BooleanVar(value=bool(_SKARV_SOLVER["adaptive"]))
        self.var_friction = tk.StringVar(value=str(_SKARV_SOLVER["friction_model"]))
        self.var_mach_warn = tk.StringVar(value=str(_SKARV_SOLVER["mach_warning"]))
        self.var_mach_choke = tk.StringVar(value=str(_SKARV_SOLVER["mach_choke"]))
        self.var_min_dx_mm = tk.StringVar(value=str(_SKARV_SOLVER["min_dx_mm"]))
        # Status
        self.var_status = tk.StringVar(value="idle")
        # Solver-options collapsed state
        self._solver_collapsed = tk.BooleanVar(value=False)

        # ---- Worker thread ----------------------------------------------
        # SolverWorker runs the solver off the Tk thread so the window
        # stays responsive (Phase 14). The poll-id and run-start timestamp
        # are tracked here for the elapsed-time tick and queue polling.
        self._worker = SolverWorker()
        self._run_t0: float = 0.0
        self._poll_after_id: str | None = None
        self._tick_after_id: str | None = None
        # "analysis" vs "sweep" so the queue handler knows whether to
        # populate the analysis tabs or the sweep tab.
        self._current_run_kind: str = "analysis"
        self._sweep_partial: list[dict] = []
        self._sweep_total: int = 0

        # ---- Build UI ----------------------------------------------------
        self._build_menu()
        self._build_layout()

        # Initial UI state
        self._update_mode_visibility()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        m_file = tk.Menu(menubar, tearoff=False)
        m_file.add_command(label="New session", command=self._action_new_session)
        m_file.add_command(label="Save inputs as JSON…", command=self._action_save_json)
        m_file.add_command(label="Load inputs from JSON…", command=self._action_load_json)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=m_file)

        m_edit = tk.Menu(menubar, tearoff=False)
        m_edit.add_command(label="Reset to Skarv defaults", command=self._action_reset_skarv)
        m_edit.add_command(label="Reset to empty", command=self._action_reset_empty)
        menubar.add_cascade(label="Edit", menu=m_edit)

        m_view = tk.Menu(menubar, tearoff=False)
        m_view.add_command(label="Show solver log", command=self._action_show_log)
        menubar.add_cascade(label="View", menu=m_view)

        m_help = tk.Menu(menubar, tearoff=False)
        m_help.add_command(label="About…", command=self._action_about)
        menubar.add_cascade(label="Help", menu=m_help)

        self.root.config(menu=menubar)

    # ------------------------------------------------------------------
    # Top-level layout: PanedWindow + status bar
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        # Status bar at bottom (packed first so it survives paned-window resize)
        statusbar = ttk.Frame(self.root, padding=(6, 2))
        statusbar.pack(side="bottom", fill="x")
        ttk.Separator(self.root, orient="horizontal").pack(side="bottom", fill="x")

        self._status_label = ttk.Label(statusbar, textvariable=self.var_status)
        self._status_label.pack(side="left")
        self._progress = ttk.Progressbar(statusbar, mode="indeterminate", length=180)
        self._progress.pack(side="right")

        # Main horizontal split
        self._paned = ttk.PanedWindow(self.root, orient="horizontal")
        self._paned.pack(side="top", fill="both", expand=True)

        # Left: scrollable input panel — the explicit canvas width gives the
        # pane a real requested size so the PanedWindow doesn't collapse it.
        self._left_scroll = _ScrollableFrame(self._paned, width=INPUT_PANEL_WIDTH)
        self._paned.add(self._left_scroll, weight=0)
        self._build_input_panel(self._left_scroll.body)

        # Right: notebook with output tabs
        right = ttk.Frame(self._paned)
        self._paned.add(right, weight=1)
        self._build_output_tabs(right)

        # Set the initial sash position once the window has actually been
        # mapped — sashpos has no effect on an unmapped PanedWindow.
        self._sash_initialised = False
        self._paned.bind("<Map>", self._set_initial_sashpos)

    def _set_initial_sashpos(self, _event: tk.Event) -> None:
        if self._sash_initialised:
            return
        self._sash_initialised = True
        try:
            self._paned.sashpos(0, INPUT_PANEL_WIDTH)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Input panel
    # ------------------------------------------------------------------
    def _build_input_panel(self, parent: tk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        outer = ttk.Frame(parent, padding=8)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)

        # Section 1 — Composition
        self.composition_editor = CompositionEditor(outer, on_change=self._on_inputs_changed)
        self.composition_editor.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        # Section 2 — Geometry
        self._build_geometry_section(outer).grid(row=1, column=0, sticky="ew", pady=6)

        # Section 3 — Boundary conditions
        self._build_bc_section(outer).grid(row=2, column=0, sticky="ew", pady=6)

        # Section 4 — Solver options (collapsible)
        self._build_solver_section(outer).grid(row=3, column=0, sticky="ew", pady=6)

        # Section 5 — Run controls
        self._build_run_controls(outer).grid(row=4, column=0, sticky="ew", pady=(6, 0))

    def _build_geometry_section(self, parent: tk.Misc) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text="Pipe geometry sections", padding=6)
        f.columnconfigure(0, weight=1)
        v = self._validate_cmd

        cols = ("idx", "length", "id", "od", "roughness", "U")
        headings = {
            "idx": "#",
            "length": "L [m]",
            "id": "ID [mm]",
            "od": "OD [mm]",
            "roughness": "ε [μm]",
            "U": "U [W/m²/K]",
        }
        widths = {"idx": 28, "length": 70, "id": 70, "od": 70, "roughness": 60, "U": 80}

        tv_frame = ttk.Frame(f)
        tv_frame.grid(row=0, column=0, sticky="ew")
        tv_frame.columnconfigure(0, weight=1)

        tv = ttk.Treeview(tv_frame, columns=cols, show="headings", height=4)
        for c in cols:
            tv.heading(c, text=headings[c])
            tv.column(c, width=widths[c], anchor="e", stretch=(c != "idx"))
        tv.grid(row=0, column=0, sticky="ew")
        sb = ttk.Scrollbar(tv_frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")
        tv.bind("<Double-1>", lambda _e: self._action_edit_section())
        self._sections_tree = tv

        # Button row.
        btns = ttk.Frame(f)
        btns.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(btns, text="+ Add section", command=self._action_add_section).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(btns, text="Edit", command=self._action_edit_section).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(btns, text="Remove", command=self._action_remove_section).pack(
            side="left"
        )

        # Ambient T row.
        amb_row = ttk.Frame(f)
        amb_row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        amb_row.columnconfigure(1, weight=1)
        _add_labeled_entry(amb_row, 0, "Ambient T [°C]", self.var_T_amb, v)

        self._refresh_sections_tree()
        return f

    # ------------------------------------------------------------------
    # Section editor actions
    # ------------------------------------------------------------------
    def _refresh_sections_tree(self) -> None:
        """Re-render the sections Treeview from ``self._section_rows``."""
        tv = self._sections_tree
        for item in tv.get_children():
            tv.delete(item)
        for i, row in enumerate(self._section_rows, start=1):
            tv.insert(
                "", "end",
                values=(
                    str(i),
                    f"{row['length_m']:.2f}",
                    f"{row['id_mm']:.1f}",
                    f"{row['od_mm']:.1f}",
                    f"{row['roughness_um']:.1f}",
                    f"{row['U']:.2f}",
                ),
            )

    def _selected_section_index(self) -> int | None:
        tv = self._sections_tree
        sel = tv.selection()
        if not sel:
            return None
        return tv.index(sel[0])

    def _action_add_section(self) -> None:
        """Append a new section, copying defaults from the last existing row."""
        if self._section_rows:
            new = dict(self._section_rows[-1])
        else:
            new = {
                "length_m": 10.0, "id_mm": 200.0, "od_mm": 220.0,
                "roughness_um": 45.0, "U": 0.0,
            }
        self._section_rows.append(new)
        self._refresh_sections_tree()

    def _action_remove_section(self) -> None:
        """Remove the selected section. Refuses to remove the last one."""
        idx = self._selected_section_index()
        if idx is None:
            messagebox.showinfo(
                "Remove section",
                "Select a row in the sections table first.",
                parent=self.root,
            )
            return
        if len(self._section_rows) <= 1:
            messagebox.showwarning(
                "Remove section",
                "At least one section is required.",
                parent=self.root,
            )
            return
        del self._section_rows[idx]
        self._refresh_sections_tree()

    def _action_edit_section(self) -> None:
        """Open a modal dialog to edit the selected section's fields."""
        idx = self._selected_section_index()
        if idx is None:
            if not self._section_rows:
                return
            idx = 0
        self._open_section_editor(idx)

    def _open_section_editor(self, idx: int) -> None:
        """Modal Toplevel editor for section row ``idx``."""
        row = self._section_rows[idx]
        win = tk.Toplevel(self.root)
        win.title(f"Edit section {idx + 1}")
        win.transient(self.root)
        win.grab_set()
        win.resizable(False, False)

        v = self._validate_cmd
        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)

        fields: dict[str, tk.StringVar] = {
            "length_m": tk.StringVar(value=f"{row['length_m']:g}"),
            "id_mm": tk.StringVar(value=f"{row['id_mm']:g}"),
            "od_mm": tk.StringVar(value=f"{row['od_mm']:g}"),
            "roughness_um": tk.StringVar(value=f"{row['roughness_um']:g}"),
            "U": tk.StringVar(value=f"{row['U']:g}"),
        }
        labels = [
            ("Length [m]", "length_m"),
            ("Inner diameter [mm]", "id_mm"),
            ("Outer diameter [mm]", "od_mm"),
            ("Roughness [μm]", "roughness_um"),
            ("Overall U [W/m²/K]", "U"),
        ]
        for r, (label, key) in enumerate(labels):
            _add_labeled_entry(body, r, label, fields[key], v)

        btn_row = ttk.Frame(body)
        btn_row.grid(row=len(labels), column=0, columnspan=2, sticky="ew", pady=(8, 0))
        btn_row.columnconfigure(0, weight=1)

        def _on_ok() -> None:
            try:
                new = {
                    "length_m": _parse_positive_float(fields["length_m"].get(), "Length"),
                    "id_mm": _parse_positive_float(fields["id_mm"].get(), "Inner diameter"),
                    "od_mm": _parse_positive_float(fields["od_mm"].get(), "Outer diameter"),
                    "roughness_um": _parse_positive_float(
                        fields["roughness_um"].get(), "Roughness"
                    ),
                    "U": _parse_nonneg_float(fields["U"].get(), "Overall U"),
                }
            except _InputValidationError as exc:
                messagebox.showerror("Invalid input", str(exc), parent=win)
                return
            if new["od_mm"] < new["id_mm"]:
                messagebox.showerror(
                    "Invalid input",
                    f"Outer diameter ({new['od_mm']:.1f} mm) must be ≥ inner "
                    f"diameter ({new['id_mm']:.1f} mm).",
                    parent=win,
                )
                return
            self._section_rows[idx] = new
            self._refresh_sections_tree()
            win.destroy()

        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(btn_row, text="OK", command=_on_ok).pack(side="right", padx=(0, 6))

        win.bind("<Return>", lambda _e: _on_ok())
        win.bind("<Escape>", lambda _e: win.destroy())

    def _build_bc_section(self, parent: tk.Misc) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text="Boundary conditions", padding=6)
        f.columnconfigure(1, weight=1)
        v = self._validate_cmd
        _add_labeled_entry(f, 0, "Inlet P [bara]", self.var_P_in, v)
        _add_labeled_entry(f, 1, "Inlet T [°C]", self.var_T_in, v)

        # Mode radio
        mode_frame = ttk.Frame(f)
        mode_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 2))
        ttk.Label(mode_frame, text="Mode:").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(
            mode_frame, text="IVP (specify ṁ)",
            variable=self.var_mode, value="IVP",
            command=self._update_mode_visibility,
        ).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(
            mode_frame, text="BVP (specify P_out)",
            variable=self.var_mode, value="BVP",
            command=self._update_mode_visibility,
        ).pack(side="left")

        # IVP / BVP conditional rows (separate Frames so we can grid_remove cleanly)
        self._ivp_frame = ttk.Frame(f)
        self._ivp_frame.columnconfigure(1, weight=1)
        _add_labeled_entry(self._ivp_frame, 0, "Mass flow [kg/s]", self.var_mdot, v)

        self._bvp_frame = ttk.Frame(f)
        self._bvp_frame.columnconfigure(1, weight=1)
        _add_labeled_entry(self._bvp_frame, 0, "Outlet P [bara]", self.var_P_out, v)

        # Place both at the same grid row; visibility toggled in
        # _update_mode_visibility().
        self._ivp_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        self._bvp_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        return f

    def _build_solver_section(self, parent: tk.Misc) -> ttk.Frame:
        # Use a plain Frame as the outer container so we can host the
        # collapsible toggle + the inner LabelFrame.
        outer = ttk.Frame(parent)
        outer.columnconfigure(0, weight=1)

        self._solver_toggle_btn = ttk.Button(
            outer, text="▼ Solver options", command=self._toggle_solver_section
        )
        self._solver_toggle_btn.grid(row=0, column=0, sticky="ew")

        f = ttk.LabelFrame(outer, text="", padding=6)
        f.columnconfigure(1, weight=1)
        self._solver_inner = f
        v = self._validate_cmd

        _add_labeled_entry(f, 0, "Number of segments", self.var_n_seg, v)
        ttk.Checkbutton(
            f, text="Adaptive Δx", variable=self.var_adaptive
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(f, text="Friction model").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(
            f, textvariable=self.var_friction, values=list(_FRICTION_MODELS),
            state="readonly", width=12,
        ).grid(row=2, column=1, sticky="ew", pady=2)

        _add_labeled_entry(f, 3, "Mach warning", self.var_mach_warn, v)
        _add_labeled_entry(f, 4, "Mach choke", self.var_mach_choke, v)
        _add_labeled_entry(f, 5, "Min Δx [mm]", self.var_min_dx_mm, v)

        f.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        return outer

    def _build_run_controls(self, parent: tk.Misc) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text="Run", padding=6)
        f.columnconfigure(0, weight=1)
        self._btn_run = ttk.Button(
            f, text="Run analysis", command=self._action_run_analysis
        )
        self._btn_run.grid(row=0, column=0, sticky="ew", pady=2)
        self._btn_sweep = ttk.Button(
            f, text="Run plateau sweep", command=self._action_run_plateau_sweep
        )
        self._btn_sweep.grid(row=1, column=0, sticky="ew", pady=2)
        # Cancel button: shown only while a solve is running. We grid it
        # on demand (and grid_remove when idle) so the LabelFrame layout
        # collapses cleanly.
        self._btn_cancel = ttk.Button(
            f, text="Cancel", command=self._action_cancel_run
        )
        ttk.Label(f, textvariable=self.var_status).grid(
            row=3, column=0, sticky="w", pady=(4, 0)
        )
        # Progressbar: indeterminate mode while running, hidden otherwise.
        # Pre-built and gridded on demand so the layout slot is consistent.
        self._progressbar = ttk.Progressbar(
            f, mode="indeterminate", length=200
        )
        return f

    # ------------------------------------------------------------------
    # Output tabs
    # ------------------------------------------------------------------
    def _build_output_tabs(self, parent: tk.Misc) -> None:
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        self._notebook = nb

        # Tab 1 — Summary
        t_summary = ttk.Frame(nb)
        nb.add(t_summary, text="Summary")
        self._build_summary_tab(t_summary)

        # Tab 2 — Profile Plot
        t_profile = ttk.Frame(nb)
        nb.add(t_profile, text="Profile Plot")
        self._build_profile_tab(t_profile)

        # Tab 3 — Station Table
        t_table = ttk.Frame(nb)
        nb.add(t_table, text="Station Table")
        self._build_table_tab(t_table)

        # Tab 4 — Plateau Sweep
        t_sweep = ttk.Frame(nb)
        nb.add(t_sweep, text="Plateau Sweep")
        self._build_sweep_tab(t_sweep)

    def _build_summary_tab(self, parent: tk.Misc) -> None:
        toolbar = ttk.Frame(parent, padding=(4, 4))
        toolbar.pack(side="top", fill="x")
        ttk.Button(toolbar, text="Save report as TXT", command=self._action_save_report).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(toolbar, text="Copy to clipboard", command=self._action_copy_summary).pack(
            side="left"
        )

        text_frame = ttk.Frame(parent)
        text_frame.pack(side="top", fill="both", expand=True, padx=4, pady=(0, 4))

        self._summary_text = tk.Text(
            text_frame, wrap="none", font=("Consolas", 10), state="disabled"
        )
        # Banner tag styles for the two-phase warning. Three severity
        # bands map to escalating yellow→orange→red palettes; the actual
        # band is picked by _classify_severity based on max LVF.
        self._summary_text.tag_configure(
            "banner_marginal",
            background="#fff7e0", foreground="#7a5b00",
            font=("Consolas", 10, "bold"),
        )
        self._summary_text.tag_configure(
            "banner_light",
            background="#ffe4b5", foreground="#7a4500",
            font=("Consolas", 10, "bold"),
        )
        self._summary_text.tag_configure(
            "banner_significant",
            background="#ffcccc", foreground="#8b0000",
            font=("Consolas", 10, "bold"),
        )
        sb_y = ttk.Scrollbar(text_frame, orient="vertical", command=self._summary_text.yview)
        sb_x = ttk.Scrollbar(text_frame, orient="horizontal", command=self._summary_text.xview)
        self._summary_text.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        self._summary_text.grid(row=0, column=0, sticky="nsew")
        sb_y.grid(row=0, column=1, sticky="ns")
        sb_x.grid(row=1, column=0, sticky="ew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

    def _build_profile_tab(self, parent: tk.Misc) -> None:
        toolbar = ttk.Frame(parent, padding=(4, 4))
        toolbar.pack(side="top", fill="x")
        ttk.Button(toolbar, text="Save as PNG", command=self._action_save_png).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(toolbar, text="Save as PDF", command=self._action_save_pdf).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(toolbar, text="Refresh", command=self._action_refresh_plot).pack(side="left")

        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )
        except ImportError:
            ttk.Label(
                parent, text="matplotlib not installed — Profile Plot unavailable"
            ).pack(fill="both", expand=True)
            self._figure = None
            self._figure_canvas = None
            return

        self._figure = Figure(figsize=(8, 5), dpi=100)
        canvas = FigureCanvasTkAgg(self._figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        nav = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        nav.update()
        nav.pack(side="bottom", fill="x")
        self._figure_canvas = canvas

    def _build_table_tab(self, parent: tk.Misc) -> None:
        toolbar = ttk.Frame(parent, padding=(4, 4))
        toolbar.pack(side="top", fill="x")
        ttk.Button(toolbar, text="Export CSV", command=self._action_export_csv).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(toolbar, text="Copy as TSV", command=self._action_copy_tsv).pack(
            side="left"
        )

        tv_frame = ttk.Frame(parent)
        tv_frame.pack(side="top", fill="both", expand=True, padx=4, pady=(0, 4))
        col_ids = [c[0] for c in _STATION_COLUMNS]
        tv = ttk.Treeview(tv_frame, columns=col_ids, show="headings")
        for cid, label in _STATION_COLUMNS:
            tv.heading(cid, text=label)
            tv.column(cid, width=90, anchor="e")
        tv.tag_configure("choke", background="#ffe0e0")
        sb = ttk.Scrollbar(tv_frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        tv_frame.rowconfigure(0, weight=1)
        tv_frame.columnconfigure(0, weight=1)
        self._station_tree = tv

    def _build_sweep_tab(self, parent: tk.Misc) -> None:
        """Plateau Sweep tab — header label, plot canvas, results table.

        Built once with empty content; populated on demand by
        ``_populate_sweep_tab``.
        """
        self._sweep_frame = parent
        self._sweep_placeholder = ttk.Label(
            parent,
            text="Click 'Run plateau sweep' to populate this tab.",
            anchor="center",
        )
        self._sweep_placeholder.pack(fill="both", expand=True)

        # The header label, figure, and table are created lazily on the
        # first sweep — they would be wasted overhead if the user never
        # opens this tab.
        self._sweep_built = False
        self._sweep_header_var = tk.StringVar(value="")
        self._sweep_figure = None
        self._sweep_canvas = None
        self._sweep_tree: ttk.Treeview | None = None

    def _ensure_sweep_widgets(self) -> bool:
        """Create the sweep header / plot / table the first time we need them.

        Returns True if matplotlib was available and the widgets were
        built (or already exist), False if the plot canvas could not be
        constructed (table-only fallback).
        """
        if self._sweep_built:
            return self._sweep_canvas is not None
        # Tear down placeholder.
        try:
            self._sweep_placeholder.destroy()
        except Exception:
            pass

        parent = self._sweep_frame
        ttk.Label(
            parent, textvariable=self._sweep_header_var,
            anchor="center", font=("TkDefaultFont", 10, "bold"),
        ).pack(side="top", fill="x", padx=4, pady=(6, 2))

        plot_ok = True
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
            )
        except ImportError:
            plot_ok = False
            ttk.Label(
                parent, text="matplotlib not installed — plot unavailable",
            ).pack(side="top", fill="x")

        if plot_ok:
            self._sweep_figure = Figure(figsize=(7, 4), dpi=100)
            self._sweep_canvas = FigureCanvasTkAgg(self._sweep_figure, master=parent)
            self._sweep_canvas.draw()
            self._sweep_canvas.get_tk_widget().pack(
                side="top", fill="both", expand=True, padx=4, pady=(0, 4),
            )

        # Treeview with the spec'd columns.
        cols = ("P_out", "mdot", "choked", "M_out", "T_out", "x_choke")
        headings = {
            "P_out": "P_out [bara]",
            "mdot": "ṁ [kg/s]",
            "choked": "Choked",
            "M_out": "M_out",
            "T_out": "T_out [°C]",
            "x_choke": "x_choke [m]",
        }
        tv_frame = ttk.Frame(parent)
        tv_frame.pack(side="top", fill="both", expand=False, padx=4, pady=(0, 4))
        tv = ttk.Treeview(
            tv_frame, columns=cols, show="headings", height=8,
        )
        for c in cols:
            tv.heading(c, text=headings[c])
            tv.column(c, anchor="e", width=100, stretch=True)
        sb = ttk.Scrollbar(tv_frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._sweep_tree = tv

        self._sweep_built = True
        return plot_ok

    # ------------------------------------------------------------------
    # Internal actions / state
    # ------------------------------------------------------------------
    def _update_mode_visibility(self) -> None:
        """Show the IVP or BVP conditional row based on the mode radio."""
        if self.var_mode.get() == "IVP":
            self._bvp_frame.grid_remove()
            self._ivp_frame.grid()
        else:
            self._ivp_frame.grid_remove()
            self._bvp_frame.grid()

    def _toggle_solver_section(self) -> None:
        if self._solver_collapsed.get():
            self._solver_inner.grid()
            self._solver_toggle_btn.configure(text="▼ Solver options")
            self._solver_collapsed.set(False)
        else:
            self._solver_inner.grid_remove()
            self._solver_toggle_btn.configure(text="▶ Solver options")
            self._solver_collapsed.set(True)

    def _on_inputs_changed(self) -> None:
        """Composition / input change hook. Stub — used by later phases."""

    # ------------------------------------------------------------------
    # Input collection
    # ------------------------------------------------------------------
    def _gather_inputs(self) -> dict[str, Any]:
        """Read, validate and SI-convert every input panel value.

        Returns
        -------
        dict
            ``{'composition': ..., 'pipe_kwargs': ..., 'bc': ..., 'solver_kwargs': ...}``
            with all numeric values in SI units, ready to feed the solver
            constructors.

        Raises
        ------
        _InputValidationError
            If any field is missing, non-numeric, or violates a
            domain constraint (negative length, composition off, etc.).
        """
        # Composition
        composition = self.composition_editor.get_composition()
        if not composition:
            raise _InputValidationError("Composition is empty.")
        if not self.composition_editor.is_valid():
            total = sum(composition.values())
            raise _InputValidationError(
                f"Composition does not sum to 1.0 within tolerance "
                f"(Σ = {total:.6f}). Use the Normalize button or fix the rows."
            )

        # Geometry — sections list, converted to SI.
        if not self._section_rows:
            raise _InputValidationError("At least one pipe section is required.")
        sections_si: list[dict[str, float]] = []
        for i, row in enumerate(self._section_rows, start=1):
            length = float(row["length_m"])
            id_m = float(row["id_mm"]) * 1e-3
            od_m = float(row["od_mm"]) * 1e-3
            roughness_m = float(row["roughness_um"]) * 1e-6
            U = float(row["U"])
            if length <= 0 or id_m <= 0 or od_m <= 0 or roughness_m <= 0:
                raise _InputValidationError(
                    f"Section {i}: all dimensions must be positive."
                )
            if od_m < id_m:
                raise _InputValidationError(
                    f"Section {i}: outer diameter ({od_m * 1e3:.1f} mm) must "
                    f"be ≥ inner diameter ({id_m * 1e3:.1f} mm)."
                )
            if U < 0:
                raise _InputValidationError(
                    f"Section {i}: overall U must be ≥ 0."
                )
            sections_si.append({
                "length": length,
                "inner_diameter": id_m,
                "outer_diameter": od_m,
                "roughness": roughness_m,
                "overall_U": U,
            })
        T_amb_K = _parse_float(self.var_T_amb.get(), "Ambient T") + 273.15

        pipe_kwargs = {
            "sections_si": sections_si,
            "ambient_temperature": T_amb_K,
        }

        # Boundary conditions
        P_in = _parse_positive_float(self.var_P_in.get(), "Inlet P") * 1e5
        T_in_K = _parse_float(self.var_T_in.get(), "Inlet T") + 273.15
        if T_in_K <= 0.0:
            raise _InputValidationError(
                f"Inlet T ({T_in_K - 273.15:.1f} °C) is below absolute zero."
            )

        mode = self.var_mode.get()
        if mode not in ("IVP", "BVP"):
            raise _InputValidationError(f"Unknown mode {mode!r}.")
        if mode == "IVP":
            mdot = _parse_positive_float(self.var_mdot.get(), "Mass flow")
            P_out: float | None = None
        else:
            P_out = _parse_positive_float(self.var_P_out.get(), "Outlet P") * 1e5
            mdot = None
            if P_out >= P_in:
                raise _InputValidationError(
                    f"Outlet P ({P_out / 1e5:.2f} bara) must be lower than "
                    f"Inlet P ({P_in / 1e5:.2f} bara)."
                )

        bc = {
            "P_in": P_in,
            "T_in": T_in_K,
            "mode": mode,
            "mdot": mdot,
            "P_out": P_out,
        }

        # Solver options
        n_seg = _parse_positive_int(self.var_n_seg.get(), "Number of segments")
        adaptive = bool(self.var_adaptive.get())
        friction_model = self.var_friction.get().strip()
        if friction_model not in _FRICTION_MODELS:
            raise _InputValidationError(
                f"Friction model {friction_model!r} is not one of {_FRICTION_MODELS}."
            )
        mach_warning = _parse_positive_float(self.var_mach_warn.get(), "Mach warning")
        mach_choke = _parse_positive_float(self.var_mach_choke.get(), "Mach choke")
        if not (0.0 < mach_warning < mach_choke <= 1.0):
            raise _InputValidationError(
                f"Need 0 < mach_warning < mach_choke ≤ 1; "
                f"got {mach_warning} and {mach_choke}."
            )
        min_dx_m = _parse_positive_float(self.var_min_dx_mm.get(), "Min Δx") * 1e-3

        solver_kwargs = {
            "n_segments": n_seg,
            "adaptive": adaptive,
            "friction_model": friction_model,
            "mach_warning": mach_warning,
            "mach_choke": mach_choke,
            "min_dx": min_dx_m,
        }

        return {
            "composition": composition,
            "pipe_kwargs": pipe_kwargs,
            "bc": bc,
            "solver_kwargs": solver_kwargs,
        }

    # ------------------------------------------------------------------
    # Stub actions (filled in by later phases)
    # ------------------------------------------------------------------
    def _action_new_session(self) -> None: ...
    def _action_save_json(self) -> None: ...
    def _action_load_json(self) -> None: ...
    def _action_reset_skarv(self) -> None: ...
    def _action_reset_empty(self) -> None: ...
    def _action_show_log(self) -> None: ...
    def _action_about(self) -> None: ...

    def _action_run_analysis(self) -> None:
        """Gather inputs and launch the solver in a background worker.

        While the worker runs, the Tk main loop stays responsive: the
        progressbar animates, the status label ticks the elapsed seconds,
        and the user can cancel via the Cancel button. Results arrive
        via `self._worker.result_queue` and are dispatched in
        `_poll_solver_queue`.
        """
        if self._worker.is_running():
            return  # ignore double-clicks while running

        try:
            inputs = self._gather_inputs()
        except _InputValidationError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self.root)
            return

        # Build solver inputs on the Tk thread (cheap, validates EOS too).
        try:
            fluid = GERGFluid(inputs["composition"])
            pipe = _build_pipe(inputs["pipe_kwargs"])
        except EOSOutOfRange as exc:
            self._show_solver_error(
                "EOS out of range",
                "Thermodynamic state outside EOS validity range. "
                "Check pressure and temperature.",
                exc,
            )
            return
        except Exception as exc:
            self._show_solver_error(
                f"Setup error: {type(exc).__name__}",
                "Could not construct fluid/pipe from inputs.",
                exc,
            )
            return

        bc = inputs["bc"]
        solver_kwargs = inputs["solver_kwargs"]

        # Define the work the worker thread will do. Closes over `bc`,
        # `solver_kwargs`, `fluid`, `pipe`, and the worker's cancel_event.
        def _do_solve() -> "PipeResult":
            cancel_event = self._worker.cancel_event
            if bc["mode"] == "IVP":
                return march_ivp(
                    pipe, fluid, bc["P_in"], bc["T_in"], bc["mdot"],
                    cancel_event=cancel_event, **solver_kwargs,
                )
            return solve_for_mdot(
                pipe, fluid, bc["P_in"], bc["T_in"], bc["P_out"],
                cancel_event=cancel_event, **solver_kwargs,
            )

        # ---- UI: enter "running" state ----------------------------------
        self._current_run_kind = "analysis"
        self._enter_running_state()
        self._worker.run_async(_do_solve)
        self._poll_after_id = self.root.after(100, self._poll_solver_queue)
        self._tick_after_id = self.root.after(500, self._tick_elapsed)

    def _action_cancel_run(self) -> None:
        """User clicked Cancel — request cooperative cancellation."""
        if not self._worker.is_running():
            return
        self._worker.request_cancel()
        # Visual hint that cancel was requested; final status comes via queue.
        self.var_status.set("Cancelling…")

    def _enter_running_state(self) -> None:
        """Disable Run buttons, show Cancel + progressbar, reset status."""
        self._btn_run.configure(state="disabled")
        self._btn_sweep.configure(state="disabled")
        self._btn_cancel.grid(row=2, column=0, sticky="ew", pady=2)
        self._progressbar.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        self._progressbar.start(50)
        self._status_label.configure(foreground="")
        self.var_status.set("Running… (0.0 s)")
        self._run_t0 = time.time()

    def _exit_running_state(self) -> None:
        """Re-enable Run buttons, hide Cancel + progressbar, stop ticks."""
        self._progressbar.stop()
        self._progressbar.grid_remove()
        self._btn_cancel.grid_remove()
        self._btn_run.configure(state="normal")
        self._btn_sweep.configure(state="normal")
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None
        if self._tick_after_id is not None:
            try:
                self.root.after_cancel(self._tick_after_id)
            except Exception:
                pass
            self._tick_after_id = None

    def _tick_elapsed(self) -> None:
        """Update the elapsed-seconds counter in the status label."""
        if not self._worker.is_running():
            return  # final status will be set by _poll_solver_queue
        elapsed = time.time() - self._run_t0
        # Don't overwrite "Cancelling…" once user clicks cancel.
        cur = self.var_status.get()
        if cur.startswith("Running"):
            self.var_status.set(f"Running… ({elapsed:.1f} s)")
        self._tick_after_id = self.root.after(500, self._tick_elapsed)

    def _poll_solver_queue(self) -> None:
        """Drain one message from the worker queue, dispatch by status."""
        try:
            msg = self._worker.result_queue.get_nowait()
        except queue.Empty:
            self._poll_after_id = self.root.after(100, self._poll_solver_queue)
            return

        kind = msg[0]
        self._exit_running_state()
        is_sweep = self._current_run_kind == "sweep"

        if kind == "ok":
            _, result, elapsed = msg
            if is_sweep:
                self._populate_sweep_tab(result)
                n_choked = sum(1 for p in result if p["choked"])
                self.var_status.set(
                    f"Sweep: {len(result)}/{len(result)} done "
                    f"({n_choked} choked) — {elapsed:.1f} s"
                )
                self._status_label.configure(foreground="")
            else:
                self._populate_all_tabs(result)
                self.var_status.set(f"Done ({elapsed:.1f} s)")
                self._status_label.configure(foreground="")

        elif kind == "choked":
            # Only analysis runs reach this branch; the sweep absorbs
            # BVPChoked per-point internally and returns 'ok'.
            _, exc, elapsed = msg
            self._populate_all_tabs(exc.result)
            self.var_status.set(f"[CHOKED] Done ({elapsed:.1f} s)")
            self._status_label.configure(foreground="#C42B1C")

        elif kind == "cancelled":
            _, _, elapsed = msg
            if is_sweep and self._sweep_partial:
                # Show whatever points were completed before the cancel.
                partial = list(self._sweep_partial)
                self._populate_sweep_tab(partial)
                self.var_status.set(
                    f"Sweep cancelled at {len(partial)}/{self._sweep_total} "
                    f"({elapsed:.1f} s)"
                )
            else:
                self.var_status.set(f"Cancelled ({elapsed:.1f} s)")
            self._status_label.configure(foreground="")

        elif kind == "error":
            _, exc, tb = msg
            self._handle_solver_exception(exc, tb)

        else:
            # Unknown kind — defensive
            self.var_status.set("Error")
            self._status_label.configure(foreground="#C42B1C")
            messagebox.showerror(
                "Internal error",
                f"Unknown worker message: {msg!r}",
                parent=self.root,
            )

    def _handle_solver_exception(self, exc: Exception, tb: str) -> None:
        """Map a solver-side exception to the same UX as the old sync handlers."""
        title_map = {
            EOSOutOfRange: (
                "EOS out of range",
                "Thermodynamic state outside EOS validity range. "
                "Check pressure and temperature.",
            ),
            EOSTwoPhase: (
                "Two-phase condition",
                "Gas would condense at this state. Single-phase solver "
                "cannot continue. Reduce ΔP or increase T_in.",
            ),
            SegmentConvergenceError: (
                "Solver convergence failure",
                "Segment Newton solver failed. Try increasing the number of "
                "segments or relaxing the Mach thresholds.",
            ),
            BVPNotBracketedError: (
                "BVP could not bracket ṁ",
                "Could not bracket the mass flow rate. Outlet pressure may "
                "be unreachable for this geometry.",
            ),
        }
        for exc_type, (title, message) in title_map.items():
            if isinstance(exc, exc_type):
                self._show_solver_error(title, message, exc)
                return
        # Unknown exception — show traceback for debugging.
        self.var_status.set("Error")
        self._status_label.configure(foreground="#C42B1C")
        messagebox.showerror(
            f"Unexpected error: {type(exc).__name__}",
            f"The solver raised an unexpected exception:\n\n{exc}\n\n{tb}",
            parent=self.root,
        )

    # ------------------------------------------------------------------
    # Plateau sweep (Phase 15)
    # ------------------------------------------------------------------
    def _action_run_plateau_sweep(self) -> None:
        """Sweep 8 P_out values from 0.9·P_in to 0.05·P_in (log-spaced)."""
        if self._worker.is_running():
            return

        try:
            inputs = self._gather_inputs()
        except _InputValidationError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self.root)
            return

        try:
            fluid = GERGFluid(inputs["composition"])
            pipe = _build_pipe(inputs["pipe_kwargs"])
        except EOSOutOfRange as exc:
            self._show_solver_error("EOS out of range", str(exc), exc)
            return
        except Exception as exc:
            self._show_solver_error(
                f"Setup error: {type(exc).__name__}",
                "Could not construct fluid/pipe from inputs.",
                exc,
            )
            return

        bc = inputs["bc"]
        solver_kwargs = inputs["solver_kwargs"]

        # P_out schedule: 8 points log-spaced from 0.9·P_in to 0.05·P_in.
        # Order high → low so a true choke plateau appears as a flat tail
        # of red squares on the right of the (inverted-x) plot.
        from .solver import plateau_sweep
        P_in_val = float(bc["P_in"])
        P_out_array = np.geomspace(0.9 * P_in_val, 0.05 * P_in_val, 8)

        # Shared progress state — written by the worker thread (only
        # appended-to via the on_point callback, which the worker calls
        # synchronously between points), read by the Tk-side _tick_elapsed.
        self._sweep_partial: list[dict] = []
        self._sweep_total = len(P_out_array)

        def _on_point(idx_completed: int, total: int, point: dict) -> None:
            self._sweep_partial.append(point)

        def _do_sweep() -> list[dict]:
            return plateau_sweep(
                pipe, fluid, bc["P_in"], bc["T_in"], P_out_array,
                ivp_kwargs=solver_kwargs,
                cancel_event=self._worker.cancel_event,
                on_point=_on_point,
            )

        self._current_run_kind = "sweep"
        self._enter_running_state()
        # Switch to the Plateau Sweep tab so the user sees the table
        # populating as the sweep progresses.
        try:
            self._notebook.select(3)
        except Exception:
            pass
        self._worker.run_async(_do_sweep)
        self._poll_after_id = self.root.after(100, self._poll_solver_queue)
        self._tick_after_id = self.root.after(500, self._tick_sweep_progress)

    def _tick_sweep_progress(self) -> None:
        """Sweep-specific status tick: 'Sweep: N/8 done, X.X s elapsed'."""
        if not self._worker.is_running():
            return
        elapsed = time.time() - self._run_t0
        n_done = len(self._sweep_partial)
        cur = self.var_status.get()
        if cur.startswith("Sweep") or cur.startswith("Running"):
            self.var_status.set(
                f"Sweep: {n_done}/{self._sweep_total} done, {elapsed:.1f} s elapsed"
            )
        self._tick_after_id = self.root.after(500, self._tick_sweep_progress)

    def _populate_sweep_tab(self, points: list[dict]) -> None:
        """Render the sweep plot + table from a list of point dicts."""
        from .diagnostics import plot_plateau_sweep

        plot_ok = self._ensure_sweep_widgets()

        # Header label.
        choked_mdots = [p["mdot"] for p in points
                        if p["choked"] and np.isfinite(p["mdot"])]
        if choked_mdots:
            mdot_crit = float(np.mean(choked_mdots))
            self._sweep_header_var.set(
                f"ṁ_critical = {mdot_crit:.1f} kg/s "
                f"(averaged over {len(choked_mdots)} choked points)"
            )
        else:
            self._sweep_header_var.set("No choked points — plateau not detected")

        # Plot.
        if plot_ok and self._sweep_figure is not None and self._sweep_canvas is not None:
            P_out_arr = np.asarray([p["P_out"] for p in points], dtype=float)
            mdot_arr = np.asarray([p["mdot"] for p in points], dtype=float)
            choked_flags = [bool(p["choked"]) for p in points]
            plot_plateau_sweep(P_out_arr, mdot_arr, choked_flags,
                               fig=self._sweep_figure)
            self._sweep_canvas.draw()

        # Table.
        tv = self._sweep_tree
        if tv is not None:
            for item in tv.get_children():
                tv.delete(item)
            for p in points:
                P_bara = p["P_out"] / 1e5
                if p["error"] is not None:
                    mdot_s = f"err: {p['error'][:30]}"
                    choked_s = "—"
                    M_s = "—"
                    T_s = "—"
                    x_s = "—"
                else:
                    mdot_s = f"{p['mdot']:.2f}" if np.isfinite(p["mdot"]) else "—"
                    choked_s = "yes" if p["choked"] else "no"
                    M_s = f"{p['M_out']:.4f}" if np.isfinite(p["M_out"]) else "—"
                    T_s = f"{p['T_out'] - 273.15:.2f}" if np.isfinite(p["T_out"]) else "—"
                    x_s = f"{p['x_choke']:.2f}" if p["x_choke"] is not None else "—"
                tv.insert(
                    "", "end",
                    values=(f"{P_bara:.3f}", mdot_s, choked_s, M_s, T_s, x_s),
                )

    # ------------------------------------------------------------------
    # Solver-error display + tab population
    # ------------------------------------------------------------------
    def _show_solver_error(self, title: str, message: str, exc: Exception) -> None:
        self.var_status.set("Error")
        self._status_label.configure(foreground="#C42B1C")
        messagebox.showerror(title, f"{message}\n\n{exc}", parent=self.root)

    def _populate_all_tabs(self, result: "PipeResult") -> None:
        self._populate_summary_tab(result)
        self._populate_profile_tab(result)
        self._populate_table_tab(result)
        # Plateau Sweep tab is populated separately by the sweep button.

    @staticmethod
    def _classify_severity(result: "PipeResult") -> tuple[str | None, str | None]:
        """Severity band + banner tag name for a result's max LVF.

        Returns (None, None) when the result did not enter the two-phase
        dome. When metastable but every LVF station is NaN (flash failed
        everywhere), maps to the marginal band so the user still sees
        a warning. Thresholds match the summary() bands so the banner
        and the TWO-PHASE section never disagree.
        """
        import numpy as _np
        if not getattr(result, "had_metastable", False):
            return None, None
        LVF = getattr(result, "LVF", None)
        if LVF is None or len(LVF) == 0 or bool(_np.all(_np.isnan(LVF))):
            return "LVF UNKNOWN — flash failures", "banner_marginal"
        lvf_max = float(_np.nanmax(LVF))
        if lvf_max < 0.01:
            return "Marginal condensation", "banner_marginal"
        if lvf_max < 0.05:
            return "Light condensation", "banner_light"
        return "Significant condensation", "banner_significant"

    def _populate_summary_tab(self, result: "PipeResult") -> None:
        text = result.summary()
        self._summary_text.configure(state="normal")
        self._summary_text.delete("1.0", "end")
        if getattr(result, "had_metastable", False):
            severity, banner_tag = self._classify_severity(result)
            if severity is not None and banner_tag is not None:
                banner = (
                    "=" * 80 + "\n"
                    f"⚠ TWO-PHASE WARNING — {severity}\n"
                    + "=" * 80 + "\n\n"
                )
                self._summary_text.insert("end", banner, banner_tag)
        self._summary_text.insert("end", text)
        self._summary_text.configure(state="disabled")
        # Scroll back to the top so the banner is in view, not the bottom
        # of the summary where the previous content was anchored.
        self._summary_text.see("1.0")

    def _populate_profile_tab(self, result: "PipeResult") -> None:
        if self._figure is None or self._figure_canvas is None:
            return
        from .diagnostics import plot_profile

        self._figure.clear()
        plot_profile(result, fig=self._figure)
        self._figure_canvas.draw()

    def _populate_table_tab(self, result: "PipeResult") -> None:
        tv = self._station_tree
        for item in tv.get_children():
            tv.delete(item)

        df = result.to_dataframe()

        choke_idx: int | None = None
        if result.choked and result.x_choke is not None:
            choke_idx = int(np.argmin(np.abs(np.asarray(result.x) - result.x_choke)))

        formatters: dict[str, str] = {
            "x_m": "{:.3f}",
            "P_bara": "{:.3f}",
            "T_C": "{:.1f}",
            "rho_kgm3": "{:.2f}",
            "u_ms": "{:.2f}",
            "a_ms": "{:.1f}",
            "M": "{:.4f}",
            "Re": "{:.2e}",
            "Z": "{:.4f}",
            "mu_JT_Kbar": "{:.4f}",
        }
        # Order matches Treeview columns from _STATION_COLUMNS
        df_cols = ("x_m", "P_bara", "T_C", "rho_kgm3", "u_ms", "a_ms",
                   "M", "Re", "Z", "mu_JT_Kbar")

        for i in range(len(df)):
            values = tuple(formatters[c].format(df[c].iloc[i]) for c in df_cols)
            tags = ("choke",) if choke_idx is not None and i == choke_idx else ()
            tv.insert("", "end", values=values, tags=tags)

    def _action_save_report(self) -> None: ...
    def _action_copy_summary(self) -> None: ...
    def _action_save_png(self) -> None: ...
    def _action_save_pdf(self) -> None: ...
    def _action_refresh_plot(self) -> None: ...
    def _action_export_csv(self) -> None: ...
    def _action_copy_tsv(self) -> None: ...


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> None:
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()

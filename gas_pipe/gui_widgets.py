"""Custom Tkinter widgets for the Gas Pipe Analyzer GUI."""
from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk
from typing import Callable

from .eos import ALLOWED_COMPONENTS

_NUMERIC_RE = re.compile(r"^-?\d*\.?\d*([eE][+-]?\d*)?$")


def _is_numeric_input(value: str) -> bool:
    """Validate command for ttk.Entry — accepts partial numeric input.

    Empty string is allowed so the user can clear the field; the regex
    accepts intermediate states like "1.", "1e", "1e-".
    """
    return value == "" or bool(_NUMERIC_RE.match(value))


_SKARV_COMPOSITION: dict[str, float] = {
    "Methane": 0.78,
    "Ethane": 0.10,
    "Propane": 0.05,
    "n-Butane": 0.02,
    "IsoButane": 0.01,
    "Nitrogen": 0.02,
    "CarbonDioxide": 0.02,
}


class CompositionEditor(ttk.LabelFrame):
    """Editable composition table.

    Each row: ttk.Combobox (component) + ttk.Entry (mole fraction) + ttk.Button (×).
    [+ Add component] button at bottom. Live sum check.

    Parameters
    ----------
    parent : tk widget
    on_change : callable, optional
        Fired whenever any row changes (component, fraction, add, remove).
        Lets the main window invalidate cached fluid and update Run state.
    """

    def __init__(
        self,
        parent: tk.Misc,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, text="Fluid composition", padding=6)
        self._on_change = on_change

        self._rows: list[dict] = []  # each: {var_comp, var_frac, combo, entry, btn}
        self._validate_cmd = (self.register(_is_numeric_input), "%P")

        # Rows container — gridded so we can re-layout cleanly on add/remove.
        self._rows_frame = ttk.Frame(self)
        self._rows_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self._rows_frame.columnconfigure(0, weight=1)  # combobox stretches

        # Header row
        hdr = ttk.Frame(self._rows_frame)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(0, weight=1)
        ttk.Label(hdr, text="Component", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 4)
        )
        ttk.Label(hdr, text="Mole fraction", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=1, sticky="w", padx=(0, 4)
        )

        # Status line + Normalize button row
        status_row = ttk.Frame(self)
        status_row.grid(row=1, column=0, sticky="ew", pady=(2, 4))
        status_row.columnconfigure(0, weight=1)
        self._status_label = ttk.Label(status_row, text="Σ = 0.000")
        self._status_label.grid(row=0, column=0, sticky="w")
        self._normalize_btn = ttk.Button(
            status_row, text="Normalize", command=self._on_normalize_click
        )
        # Hidden initially — only shown in the orange band.
        # (We grid/grid_remove based on sum state.)

        # [+ Add component] button
        self._add_btn = ttk.Button(
            self, text="+ Add component", command=self._on_add_click
        )
        self._add_btn.grid(row=2, column=0, sticky="w")

        self.columnconfigure(0, weight=1)

        # Populate with Skarv defaults
        self.set_composition(_SKARV_COMPOSITION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_composition(self) -> dict[str, float]:
        """Return current composition as {component: mole_fraction}.

        Rows with empty/invalid fractions or empty component are skipped.
        Duplicate components are summed (last write wins for the value but
        we sum to be safe).
        """
        out: dict[str, float] = {}
        for row in self._rows:
            comp = row["var_comp"].get().strip()
            frac_str = row["var_frac"].get().strip()
            if not comp or not frac_str:
                continue
            try:
                frac = float(frac_str)
            except ValueError:
                continue
            out[comp] = out.get(comp, 0.0) + frac
        return out

    def set_composition(self, d: dict[str, float]) -> None:
        """Replace all rows with the given composition."""
        # Tear down existing rows
        for row in list(self._rows):
            self._destroy_row(row, fire_change=False)
        self._rows.clear()
        # Add new rows
        for comp, frac in d.items():
            self._append_row(comp, frac, fire_change=False)
        self._refresh()
        self._fire_change()

    def is_valid(self) -> bool:
        """True when the composition sums to 1 within 1e-6."""
        return abs(sum(self.get_composition().values()) - 1.0) < 1e-6

    def normalize(self) -> None:
        """Scale all fractions so they sum to 1.0. No-op if sum is zero."""
        comp = self.get_composition()
        total = sum(comp.values())
        if total <= 0.0:
            return
        for row in self._rows:
            try:
                frac = float(row["var_frac"].get().strip() or "0")
            except ValueError:
                frac = 0.0
            row["var_frac"].set(f"{frac / total:.6g}")
        self._refresh()
        self._fire_change()

    # ------------------------------------------------------------------
    # Internal — row management
    # ------------------------------------------------------------------
    def _append_row(
        self, component: str = "", fraction: float | str = "", *, fire_change: bool = True
    ) -> None:
        var_comp = tk.StringVar(value=component)
        var_frac = tk.StringVar(
            value=f"{fraction:.6g}" if isinstance(fraction, float) else str(fraction)
        )

        combo = ttk.Combobox(
            self._rows_frame,
            textvariable=var_comp,
            values=list(ALLOWED_COMPONENTS),
            state="normal",
            width=18,
        )
        entry = ttk.Entry(
            self._rows_frame,
            textvariable=var_frac,
            validate="key",
            validatecommand=self._validate_cmd,
            width=12,
            justify="right",
        )
        btn = ttk.Button(self._rows_frame, text="×", width=2)

        row_record = {
            "var_comp": var_comp,
            "var_frac": var_frac,
            "combo": combo,
            "entry": entry,
            "btn": btn,
        }
        btn.configure(command=lambda r=row_record: self._destroy_row(r))

        # Trace var changes so on_change and sum status fire live.
        var_comp.trace_add("write", lambda *_: self._refresh_and_notify())
        var_frac.trace_add("write", lambda *_: self._refresh_and_notify())

        self._rows.append(row_record)
        self._regrid_rows()

        if fire_change:
            self._refresh()
            self._fire_change()

    def _destroy_row(self, row: dict, *, fire_change: bool = True) -> None:
        for widget_key in ("combo", "entry", "btn"):
            row[widget_key].destroy()
        if row in self._rows:
            self._rows.remove(row)
        self._regrid_rows()
        if fire_change:
            self._refresh()
            self._fire_change()

    def _regrid_rows(self) -> None:
        """Re-place row widgets after add/remove so there are no gaps.

        Row 0 of _rows_frame is the header; data rows start at index 1.
        """
        for i, row in enumerate(self._rows, start=1):
            row["combo"].grid(row=i, column=0, sticky="ew", padx=(0, 4), pady=1)
            row["entry"].grid(row=i, column=1, sticky="ew", padx=(0, 4), pady=1)
            row["btn"].grid(row=i, column=2, padx=0, pady=1)

    # ------------------------------------------------------------------
    # Internal — sum status and callbacks
    # ------------------------------------------------------------------
    def _on_add_click(self) -> None:
        self._append_row()

    def _on_normalize_click(self) -> None:
        self.normalize()

    def _refresh_and_notify(self) -> None:
        self._refresh()
        self._fire_change()

    def _refresh(self) -> None:
        """Update the sum status label and Normalize button visibility."""
        total = sum(self.get_composition().values())
        if abs(total - 1.0) < 1e-6:
            text = f"Σ = {total:.3f} ✓"
            color = "#107C10"  # green
            show_normalize = False
        elif 0.99 <= total <= 1.01:
            text = f"Σ = {total:.3f} (normalize?)"
            color = "#C19C00"  # orange
            show_normalize = True
        else:
            text = f"Σ = {total:.3f} ✗"
            color = "#C42B1C"  # red
            show_normalize = False

        self._status_label.configure(text=text, foreground=color)
        if show_normalize:
            self._normalize_btn.grid(row=0, column=1, sticky="e", padx=(8, 0))
        else:
            self._normalize_btn.grid_remove()

    def _fire_change(self) -> None:
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception:  # callback shouldn't break the widget
                pass


# ----------------------------------------------------------------------
# Standalone test runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import json

    root = tk.Tk()
    root.title("CompositionEditor — standalone test")
    root.geometry("420x420")

    change_count = {"n": 0}

    def _on_change() -> None:
        change_count["n"] += 1
        comp = editor.get_composition()
        print(
            f"[on_change #{change_count['n']:03d}] valid={editor.is_valid()}  "
            f"composition={comp}"
        )

    editor = CompositionEditor(root, on_change=_on_change)
    editor.pack(fill="x", padx=10, pady=10)

    actions = ttk.Frame(root)
    actions.pack(fill="x", padx=10, pady=4)

    def _print() -> None:
        print("get_composition() =>", json.dumps(editor.get_composition(), indent=2))
        print("is_valid()         =>", editor.is_valid())

    def _set_99() -> None:
        editor.set_composition({"Methane": 0.79, "Ethane": 0.10, "Propane": 0.10})

    def _set_100() -> None:
        editor.set_composition({"Methane": 0.80, "Ethane": 0.15, "Propane": 0.05})

    def _set_101() -> None:
        editor.set_composition({"Methane": 0.81, "Ethane": 0.15, "Propane": 0.05})

    def _set_skarv() -> None:
        editor.set_composition(_SKARV_COMPOSITION)

    ttk.Button(actions, text="Print state", command=_print).pack(side="left")
    ttk.Button(actions, text="Set Σ=0.99", command=_set_99).pack(side="left", padx=4)
    ttk.Button(actions, text="Set Σ=1.00", command=_set_100).pack(side="left", padx=4)
    ttk.Button(actions, text="Set Σ=1.01", command=_set_101).pack(side="left", padx=4)
    ttk.Button(actions, text="Reset Skarv", command=_set_skarv).pack(side="left", padx=4)

    root.mainloop()

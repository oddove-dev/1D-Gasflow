"""Test-suite shared fixtures.

Sets the default ``solve_for_mdot`` EOS mode to ``direct`` for the pre-
item-2 test suite, since those tests were written against the exact
GERG-2008 EOS. The new ``test_eos_table.py`` and any future test that
wants tabulated-EOS coverage opts back in by passing ``eos_mode='table'``
explicitly.
"""
from __future__ import annotations

import os

# Setting this at import time (rather than via an autouse fixture) covers
# tests that run solve_for_mdot inside their module-level fixtures too.
os.environ.setdefault("GAS_PIPE_DEFAULT_EOS_MODE", "direct")

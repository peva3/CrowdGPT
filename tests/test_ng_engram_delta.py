"""Bounded engram delta selection.

The old client uploaded a fixed 10% of engram rows every round, so when more
than 10% of rows changed the remainder was silently dropped. Selection is now
by update mass with an explicit row cap, which yields a stated bound: a
held-out lookup that hits an omitted row can be off by at most the omitted L1
mass. These tests exercise that bound on synthetic tables; they do not claim a
real-model held-out loss number (that needs a trained round).
"""
from __future__ import annotations

import importlib.util
import io
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client_module():
    old = os.environ.copy()
    tmp = tempfile.mkdtemp(prefix="ng-engram-")
    shutil.copy2(ROOT / "client.py", Path(tmp) / "client.py")
    spec = importlib.util.spec_from_file_location("ng_engram_client", Path(tmp) / "client.py")
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    saved_stdio = (sys.stdout, sys.stderr)
    root = logging.getLogger(); saved_handlers, saved_level = list(root.handlers), root.level
    fake = [io.TextIOWrapper(io.BytesIO(), encoding="utf-8") for _ in range(2)]
    try:
        sys.stdout, sys.stderr = fake
        os.chdir(tmp)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
        sys.stdout, sys.stderr = saved_stdio
        root.handlers[:] = saved_handlers; root.setLevel(saved_level)
        os.environ.clear(); os.environ.update(old)
    yield mod
    shutil.rmtree(tmp, ignore_errors=True)


def test_mass_selection_drops_far_fewer_changed_rows_than_a_count_cap(client_module):
    n = 2000
    mass = torch.zeros(n)
    mass[:100] = 10.0          # 99.999% of the update mass
    mass[100:200] = 1e-6       # many tiny updates the old top-k wasted bytes on
    selected, covered, omitted = client_module._engram_select_rows(mass, max_frac=0.20, mass_frac=0.995)
    assert selected.numel() == 100
    assert covered > 0.9999 and omitted < 1e-4

    # Held-out lookups: 95% land on a row that was sent, 5% on an omitted tiny
    # row. Mean L1 error must stay under the stated omitted-mass bound.
    q = torch.zeros(n); q[:100] = 0.95 / 100; q[100:200] = 0.05 / 100
    full = torch.zeros(n); full[:100] = 10.0; full[100:200] = 1e-6
    trunc = torch.zeros(n); trunc[selected] = full[selected]
    err = float(((q * (full - trunc).abs()).sum()).item())
    assert err <= omitted * float(full.sum()) + 1e-9


def test_cap_is_an_explicit_bound_when_all_rows_change(client_module):
    n = 2000
    mass = torch.ones(n)
    selected, covered, omitted = client_module._engram_select_rows(mass, max_frac=0.05, mass_frac=0.995)
    assert selected.numel() == 100
    assert abs(covered - 0.05) < 1e-6 and abs(omitted - 0.95) < 1e-6


def test_zero_and_negative_rounding_masses_produce_no_rows(client_module):
    selected, covered, omitted = client_module._engram_select_rows(torch.zeros(10))
    assert selected.numel() == 0 and covered == 0.0 and omitted == 0.0
    selected, _, _ = client_module._engram_select_rows(torch.full((10,), 1e-12))
    assert selected.numel() == 0


def test_sparse_delta_uses_mass_selection_and_wire_shape(client_module):
    n, dim = 1000, 8
    base = torch.zeros(n, dim)
    table = base.clone()
    table[:50] = 1.0            # large concentrated updates
    table[50:500] = 1e-4        # many small updates
    ai, sv = client_module._engram_sparse_delta(table, base, frac=0.10)
    assert len(ai) == 50
    assert sv.shape == (50, dim)
    assert ai.dtype == np.uint32 and sv.dtype == np.uint16
    assert int(ai.max()) < n

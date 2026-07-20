"""Test smart-realize's --export-after-phase1 (write the stalled sub-OT)."""

import glob
import os

import pytest

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.realization.base import Undecided
from tests.point_configs import one_interior, abstract_only
import pyotlib2.cli.commands as commands


class _AlwaysFail:
    """A tester that never realizes anything (simulates a too-weak solver)."""
    def __init__(self, *a, **k):
        self.parent = k.get("parent")

    def is_realizable(self, ot):
        raise Undecided("forced failure")


def test_export_after_phase1_writes_stall(tmp_path, monkeypatch):
    sl = abstract_only(one_interior(6))

    # Force every realize attempt (chain AND the grid fallback) to fail, so
    # phase 1 stalls immediately at kmin and the export path is exercised.
    monkeypatch.setattr("pyotlib2.realization.make_tester",
                        lambda name, parent=None, **kw: _AlwaysFail(parent=parent))
    from pyotlib2.realization.grid_search import GridSearchTester
    monkeypatch.setattr(GridSearchTester, "is_realizable",
                        lambda self, ot: (_ for _ in ()).throw(Undecided("forced")))

    base = str(tmp_path / "myinput.blt")
    # The phase-1 stall file is written BEFORE phase 2 drills down.  With an
    # always-failing realizer, phase 2 then hits an n <= 8 sub-OT it cannot
    # realize and smart-realize raises the sanity RuntimeError (every order type
    # on <= 8 points IS realizable, so a realizer that fails there is broken).
    # That is the correct new contract; we still verify the export happened.
    with pytest.raises(RuntimeError, match="realizer .*too weak or broken|<= 8 points"):
        commands.smart_realize(
            sl, solver="scipy", verbose=False, kmin=6,
            export_after_phase1=base,
        )
    files = glob.glob(base + ".phase1stall_*.lt")
    assert files, "no phase-1 stall file was written"
    text = open(files[0]).read()
    assert "phase-1 stall at k=6" in text
    assert "original point indices" in text

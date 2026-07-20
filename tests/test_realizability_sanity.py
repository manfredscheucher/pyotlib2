"""Sanity guards: every order type on n <= 8 points is realizable.

Two guards enforce this fact and must fire when it is (impossibly) violated,
signalling a broken model / solver rather than a genuine result:

  * GP tester: must never *certify* a n <= MAX_REALIZABLE_N order type as
    non-realizable (a False there is an AssertionError).
  * smart-realize: if the realizer fails to realize a n <= MAX_REALIZABLE_N
    sub-order-type while drilling down, it raises RuntimeError (the realizer,
    not the order type, is at fault).
"""

import pytest

from pyotlib2.realization.base import MAX_REALIZABLE_N, Undecided
from tests.point_configs import one_interior, abstract_only
import pyotlib2.cli.commands as commands


def test_max_realizable_n_is_eight():
    assert MAX_REALIZABLE_N == 8


def test_gp_false_on_small_n_is_assertion_error(monkeypatch):
    """If _solve ever returns False for n <= 8, _test must raise AssertionError."""
    pytest.importorskip("pyotlib2.realization.gp_tester")
    import shutil
    if shutil.which("glpsol") is None:
        pytest.skip("glpsol (GLPK) not on PATH")
    from pyotlib2.realization import make_tester
    gp = make_tester("gp")
    sl = abstract_only(one_interior(6))  # n=6, realizable
    # Force the LP verdict to a (bogus) False and confirm the guard trips.
    monkeypatch.setattr(gp, "_solve", lambda *a, **k: False)
    with pytest.raises(AssertionError, match="non-realizable"):
        gp._test(sl)


def test_smart_realize_hard_errors_when_realizer_fails_small():
    """A realizer that cannot realize a n <= 8 sub-OT triggers a RuntimeError."""
    from pyotlib2.realization.grid_search import GridSearchTester

    class _AlwaysFail:
        def __init__(self, *a, **k):
            self.parent = k.get("parent")

        def is_realizable(self, ot):
            raise Undecided("forced failure")

    sl = abstract_only(one_interior(6))
    import unittest.mock as mock
    with mock.patch("pyotlib2.realization.make_tester",
                    lambda name, parent=None, **kw: _AlwaysFail(parent=parent)), \
         mock.patch.object(GridSearchTester, "is_realizable",
                           lambda self, ot: (_ for _ in ()).throw(Undecided("x"))):
        with pytest.raises(RuntimeError, match="too weak or broken|<= 8 points"):
            commands.smart_realize(sl, solver="scipy", verbose=False, kmin=6)

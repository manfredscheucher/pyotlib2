"""Tests for the Grassmann-Plücker non-realizability tester (GLPK LP).

The tester must:
  * NEVER return False for a realizable order type (no false positives — GP is
    only a *necessary* condition for realizability);
  * correctly return False when its LP is infeasible (the False path works).

It previously never fired False at all: the generated LP began with a
``/* ... */`` comment that GLPK's CPLEX-LP parser rejected, so no solution was
produced and every call fell through to Undecided.
"""

import os
import shutil
import tempfile

import pytest

from pyotlib2.io.readers import read_order_types
from pyotlib2.realization.base import Undecided
from pyotlib2.realization import make_tester
from tests.point_configs import (
    convex_position_moment_curve, one_interior, abstract_only,
)

pytestmark = pytest.mark.skipif(
    shutil.which("glpsol") is None, reason="glpsol (GLPK) not on PATH")


def _result(gp, sl):
    try:
        return gp.is_realizable(sl)
    except Undecided:
        return None


class TestNoFalsePositives:
    """A realizable order type must never be reported non-realizable."""

    @pytest.mark.parametrize("cfg", [
        lambda: convex_position_moment_curve(6),
        lambda: convex_position_moment_curve(7),
        lambda: one_interior(7),
        lambda: one_interior(8),
    ])
    def test_realizable_not_false(self, cfg):
        gp = make_tester("gp")
        assert _result(gp, abstract_only(cfg())) is not False

    def test_database_realizable_ots_not_false(self):
        """Every order type in the n=7 database is realizable; GP must not say False."""
        path = os.path.join(os.path.dirname(__file__), "otdb", "otypes", "otypes07.b08")
        if not os.path.exists(path):
            pytest.skip("otypes07.b08 missing")
        gp = make_tester("gp")
        for i, sl in enumerate(read_order_types(path, fmt="b08", n=7)):
            assert _result(gp, sl) is not False
            if i >= 30:  # a representative sample is enough
                break


class TestInfeasiblePathFires:
    """The LP-infeasible → False path must actually work."""

    def test_infeasible_lp_returns_false(self):
        gp = make_tester("gp")
        with tempfile.TemporaryDirectory() as d:
            lp = os.path.join(d, "inf.lp")
            out = os.path.join(d, "inf.out")
            # contradictory bounds → infeasible
            with open(lp, "w") as f:
                f.write("Maximize\n obj: myEps\nSubject To\n"
                        " c1: myEps >= 1\n c2: myEps <= 0\nEnd\n")
            assert gp._solve(lp, out) is False

    def test_unbounded_lp_is_undecided(self):
        gp = make_tester("gp")
        with tempfile.TemporaryDirectory() as d:
            lp = os.path.join(d, "unb.lp")
            out = os.path.join(d, "unb.out")
            with open(lp, "w") as f:
                f.write("Maximize\n obj: myEps\nSubject To\n c1: myEps >= 0\nEnd\n")
            with pytest.raises(Undecided):
                gp._solve(lp, out)

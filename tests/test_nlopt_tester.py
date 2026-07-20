"""Tests for the NLopt-based realization testers (nlopt.<algo>)."""

import numpy as np
import pytest

nlopt = pytest.importorskip("nlopt")

from pyotlib2.core.point_set import PointSet
from pyotlib2.realization import make_tester, SOLVERS
from pyotlib2.realization.nlopt_tester import nlopt_methods
from tests.point_configs import (
    convex_position_moment_curve, one_interior, abstract_only,
)


def test_methods_registered():
    for m in nlopt_methods():
        assert f"nlopt.{m}" in SOLVERS


@pytest.mark.parametrize("method", ["LN_BOBYQA", "LN_SBPLX", "LN_COBYLA"])
def test_local_methods_realize_small(method):
    """Derivative-free local NLopt methods realize small order types exactly."""
    sl = abstract_only(one_interior(6))
    o0 = sl.to_big_lambda().o.copy()
    t = make_tester(f"nlopt.{method}", trials=30, seed=0)
    assert t.is_realizable(sl) is True
    o = PointSet(sl.n, sl.realization).to_small_lambda(lazy=False).to_big_lambda().o
    assert np.array_equal(o, o0)  # exact, no relabeling


@pytest.mark.parametrize("method", ["GN_CRS2_LM", "GN_ISRES"])
def test_global_methods_realize_small(method):
    sl = abstract_only(convex_position_moment_curve(5))
    t = make_tester(f"nlopt.{method}", seed=0)
    assert t.is_realizable(sl) is True

"""Tests for the arbitrary-precision (mpmath) realization tester."""

import numpy as np
import pytest

pytest.importorskip("mpmath")

from pyotlib2.core.point_set import PointSet
from pyotlib2.realization import make_tester, SOLVERS
from pyotlib2.realization.base import Undecided
from pyotlib2.realization.mpmath_tester import MpmathRealizationTester
from tests.point_configs import (
    convex_position_moment_curve, one_interior, abstract_only,
)


def test_registered():
    assert "mpmath" in SOLVERS


def test_precision_clamped():
    # precision <= 0 falls back to double (53 bits); a tiny positive value is
    # clamped up to mpmath's minimum (2 bits); a real value passes through.
    assert make_tester("mpmath", precision=0).precision == 53
    assert make_tester("mpmath", precision=-5).precision == 53
    assert make_tester("mpmath", precision=1).precision == 2
    assert make_tester("mpmath", precision=200).precision == 200


@pytest.mark.parametrize("precision", [53, 100])
def test_realizes_small_exactly(precision):
    """The mpmath realizer realizes a small order type exactly (no relabeling)."""
    sl = abstract_only(one_interior(6))
    o0 = sl.to_big_lambda().o.copy()
    t = make_tester("mpmath", precision=precision, trials=25, seed=0)
    assert t.is_realizable(sl) is True
    o = PointSet(sl.n, sl.realization).to_small_lambda(lazy=False).to_big_lambda().o
    assert np.array_equal(o, o0)  # exact chirotope, no relabeling


def test_convex_position():
    sl = abstract_only(convex_position_moment_curve(5))
    t = make_tester("mpmath", precision=80, trials=15, seed=1)
    assert t.is_realizable(sl) is True


def test_logsum_variant_also_realizes():
    sl = abstract_only(one_interior(6))
    t = make_tester("mpmath", precision=80, trials=25, seed=2, logsum=True)
    assert t.is_realizable(sl) is True


def test_high_precision_runs_at_that_precision():
    """A high precision request actually drives the mpmath working precision."""
    import mpmath
    sl = abstract_only(one_interior(5))
    seen = {}
    orig = MpmathRealizationTester._make_objective

    def spy(self, qcp):
        seen["prec"] = mpmath.mp.prec
        return orig(self, qcp)

    MpmathRealizationTester._make_objective = spy
    try:
        make_tester("mpmath", precision=150, trials=5, seed=0).is_realizable(sl)
    except Undecided:
        pass
    finally:
        MpmathRealizationTester._make_objective = orig
    assert seen.get("prec") == 150

"""Tests for the shared realization QCP (pyotlib2.realization.qcp).

The QCP is the single formulation every continuous solver consumes, so its
determinant convention, hinge penalty, warm-start layout and verify/accept must
be correct on their own.
"""

import numpy as np
import pytest

from pyotlib2.core.point_set import PointSet
from pyotlib2.realization.qcp import RealizationQCP
from tests.point_configs import (
    convex_position_moment_curve,
    one_interior,
    abstract_only,
)


def _abstract(cfg):
    return abstract_only(cfg)


class TestConstraints:
    def test_one_constraint_per_triple(self):
        sl = _abstract(convex_position_moment_curve(6))
        qcp = RealizationQCP(sl)
        assert len(qcp.constraints) == 6 * 5 * 4 // 6  # C(6,3) = 20
        assert qcp.num_variables == 12

    def test_signs_match_chirotope(self):
        sl = _abstract(one_interior(6))
        o = sl.to_big_lambda().o
        qcp = RealizationQCP(sl)
        for c in qcp.constraints:
            assert c.i < c.j < c.k
            assert c.sign == int(o[c.i, c.j, c.k])


class TestPenalty:
    def test_zero_on_true_realization(self):
        """The hinge penalty is 0 at the ground-truth realization."""
        cfg = convex_position_moment_curve(6)  # has a realization
        sl = _abstract(cfg)
        qcp = RealizationQCP(sl)
        pts = cfg.realization
        # interleaved [x0,y0,x1,y1,...]; scale up so every |det| >= 1
        flat = [c * 1000 for p in pts for c in p]
        assert qcp.penalty(flat) == 0.0

    def test_positive_on_wrong_placement(self):
        sl = _abstract(convex_position_moment_curve(6))
        qcp = RealizationQCP(sl)
        flat = [0.0] * qcp.num_variables  # all points coincide -> many violations
        assert qcp.penalty(flat) > 0.0


class TestWarmStart:
    def test_uses_subrealization_prefix(self):
        sl = _abstract(convex_position_moment_curve(6))
        sub = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 20)]  # n-1 = 5 points
        sl.subrealization = sub
        qcp = RealizationQCP(sl)
        import random
        z0 = qcp.initial_guess(random.Random(0), use_subrealization=True)
        # first n-1 points equal the subrealization; last point is free
        for i, (px, py) in enumerate(sub):
            assert z0[2 * i] == float(px)
            assert z0[2 * i + 1] == float(py)
        assert len(z0) == qcp.num_variables

    def test_stale_subrealization_ignored(self):
        sl = _abstract(convex_position_moment_curve(6))
        sl.subrealization = [(0, 0), (1, 1)]  # wrong length (not n-1)
        qcp = RealizationQCP(sl)
        assert qcp.subrealization() is None


class TestVerifyAccept:
    def test_accept_true_realization(self):
        cfg = convex_position_moment_curve(6)
        sl = _abstract(cfg)
        qcp = RealizationQCP(sl)
        assert qcp.accept(cfg.realization) is True
        assert sl.realization == list(cfg.realization)

    def test_reject_wrong_points(self):
        sl = _abstract(convex_position_moment_curve(6))
        qcp = RealizationQCP(sl)
        bad = [(0, 0)] * sl.n  # all coincident
        assert qcp.accept(bad) is False
        assert sl.realization is None

    def test_verify_is_exact_no_relabeling(self):
        """verify() requires the exact chirotope, not up-to-isomorphism."""
        cfg = one_interior(7)
        sl = _abstract(cfg)
        qcp = RealizationQCP(sl)
        o0 = sl.to_big_lambda().o
        assert qcp.verify(cfg.realization)
        o_real = (PointSet(sl.n, list(cfg.realization))
                  .to_small_lambda(lazy=False).to_big_lambda().o)
        assert np.array_equal(o_real, o0)

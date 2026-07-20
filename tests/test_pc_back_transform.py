"""Regression tests for the projective-class back-transform.

The back-transform in :class:`PCRealizationTester` transports a realization of a
projective-class MEMBER back onto the ORIGINAL order type.  It is a pure
geometric relabeling/flip/mirror composition, so its correctness must NOT depend
on the specific coordinates of the member realization: for EVERY valid
realization of the member OT, ``back(realization)`` must return coordinates whose
chirotope exactly equals the original OT's chirotope.

The historical bug: ``from_flips`` transported the realization by reflecting
flipped points through the origin (``(x,y) -> (-x,-y)``), which is a Euclidean
rotation, not the *projective* transform a convex-hull flip actually is.  That
contributed the wrong orientation sign per flipped index and produced collinear /
mismatched coordinates for some member realizations while "accidentally" working
for symmetric-enough ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.point_configs import (
    one_interior,
    convex_position_moment_curve,
    abstract_only,
)
from pyotlib2.cli.commands import smart_realize
from pyotlib2.realization.pc_tester import PCRealizationTester
from pyotlib2.core.point_set import PointSet
from pyotlib2.core.small_lambda import SmallLambda


OTS = [
    ("one_interior(6)", lambda: one_interior(6)),
    ("one_interior(7)", lambda: one_interior(7)),
    ("moment_curve(6)", lambda: convex_position_moment_curve(6)),
]


def _chirotope(real, n):
    return PointSet(n, real).to_small_lambda(lazy=False).to_big_lambda().o


def _perturbed_realizations(member: SmallLambda, count: int):
    """Yield several DIFFERENT realizations of the SAME member OT.

    Each is obtained by an independent scipy realization from a randomly
    perturbed / re-seeded warm start, so the coordinates genuinely differ while
    the induced order type stays the member OT.
    """
    o_member = member.to_big_lambda().o
    rng = np.random.default_rng(0)
    seen = 0
    attempts = 0
    while seen < count and attempts < count * 8:
        attempts += 1
        m_abs = SmallLambda.from_string(member.n, member.to_string())
        _, w = smart_realize(m_abs, solver="scipy", verbose=False, seed=int(rng.integers(1, 10_000)))
        if w is None or w.realization is None:
            continue
        real = [tuple(map(float, p)) for p in w.realization]
        # sanity: this really is a realization of the member OT
        ps = PointSet(member.n, real)
        if ps.has_collinear_points():
            continue
        if not np.array_equal(ps.to_small_lambda(lazy=False).to_big_lambda().o, o_member):
            continue
        seen += 1
        yield real


@pytest.mark.parametrize("name,build", OTS)
@pytest.mark.parametrize("seed", list(range(1, 9)))
def test_back_transform_independent_of_member_realization(name, build, seed):
    """back(realization) must reproduce the ORIGINAL chirotope for EVERY member
    realization, for every OT and seed."""
    sl = abstract_only(build())
    o_target = sl.to_big_lambda().o

    pc = PCRealizationTester(None, shuffle=True, seed=seed)
    member, back = pc.pick_member(sl)

    # Two genuinely distinct member realizations are enough to exercise the
    # property under test (back-transform must not depend on which realization
    # of the member OT it is handed); more only multiplies the (slow) scipy
    # realization cost without testing anything new.
    realizations = list(_perturbed_realizations(member, count=2))
    assert realizations, f"could not realize member for {name} seed={seed}"

    for real in realizations:
        lifted = back([tuple(p) for p in real])
        assert lifted is not None, (
            f"back-transform returned None for a valid member realization "
            f"({name}, seed={seed})"
        )
        assert np.array_equal(_chirotope(lifted, sl.n), o_target), (
            f"lifted coords do not induce the original OT ({name}, seed={seed})"
        )


@pytest.mark.parametrize("k", list(range(1, 7)))
def test_full_pc_path_realizes_original(k):
    """smart_realize(..., use_pc=True) must realize the ORIGINAL OT exactly."""
    sl = abstract_only(one_interior(7))
    o_target = sl.to_big_lambda().o

    result, witness = smart_realize(sl, solver="scipy", use_pc=True, seed=k, verbose=False)
    assert result is True, f"use_pc path failed for seed={k}"
    assert witness is not None and witness.realization is not None
    assert np.array_equal(_chirotope(witness.realization, sl.n), o_target), (
        f"use_pc realization does not induce the original OT for seed={k}"
    )

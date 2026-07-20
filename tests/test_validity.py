"""Tests for structural validity checks: chirotope and signotope axioms.

These validate the central "is this a valid abstract order type?" functions:

* ``BigLambda.is_valid``      — chirotope axioms (alternating / exchange / acyclic)
* ``BigLambda.is_signotope`` — the signotope axiom on sorted triples a<b<c<d
  (at most one sign change in o(abc), o(abd), o(acd), o(bcd))

They also validate the real data file ``data_jelena/test24.blt``, which is the
input a user asked to realize; if that file is not a valid order type, every
downstream algorithm (reconstruction, realization) is entitled to reject it.
"""

import os
from itertools import combinations

import numpy as np
import pytest

from pyotlib2.core.big_lambda import BigLambda
from pyotlib2.io.readers import read_order_types

OTDB = os.path.join(os.path.dirname(__file__), "otdb", "otypes")


def _o_from_blt_line(line: str, n: int) -> np.ndarray:
    """Build the full antisymmetric chirotope from a '+'/'-' blt line."""
    chars = [c for c in line if c in "+-"]
    assert len(chars) == n * (n - 1) * (n - 2) // 6, "wrong number of signs"
    o = np.zeros((n, n, n), dtype=np.int8)
    for idx, (a, b, c) in enumerate(combinations(range(n), 3)):
        v = np.int8(1) if chars[idx] == "+" else np.int8(-1)
        o[a, b, c] = o[b, c, a] = o[c, a, b] = v
        o[a, c, b] = o[b, a, c] = o[c, b, a] = -v
    return o


# ---------------------------------------------------------------------------
# Database order types must be valid chirotopes AND signotopes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_database_ots_are_valid(n):
    path = os.path.join(OTDB, f"otypes{n:02d}.b08")
    if not os.path.exists(path):
        pytest.skip(f"missing {path}")
    for sl in read_order_types(path, fmt="b08", n=n):
        bl = sl.to_big_lambda()
        assert bl.is_valid(print_collinear_warning=False), sl.to_string()
        assert bl.is_chirotope(), sl.to_string()
        assert bl.is_signotope(), sl.to_string()


@pytest.mark.parametrize("n", [5, 6, 7])
def test_is_chirotope_agrees_with_is_valid(n):
    """The explicit 5-tuple GP check agrees with is_valid on every DB order type,
    and stays True under arbitrary relabelling (chirotope axioms are
    labelling-independent, unlike the signotope axiom)."""
    import random
    path = os.path.join(OTDB, f"otypes{n:02d}.b08")
    if not os.path.exists(path):
        pytest.skip(f"missing {path}")
    from pyotlib2.core.small_lambda import SmallLambda
    rng = random.Random(99 + n)
    for sl in read_order_types(path, fmt="b08", n=n):
        bl = sl.to_big_lambda()
        assert bl.is_chirotope() == bl.is_valid(print_collinear_warning=False)
        # relabel and re-check: still a chirotope
        base = SmallLambda(n, sl.get_l().copy())
        perm = list(range(n)); rng.shuffle(perm)
        assert base.relabeled(perm).to_big_lambda().is_chirotope()


# ---------------------------------------------------------------------------
# A hand-built valid vs. invalid signotope.
# ---------------------------------------------------------------------------

def test_signotope_accepts_convex():
    """Points in convex position (moment curve) form a valid signotope."""
    from pyotlib2.core.point_set import PointSet
    pts = [(i, i * i) for i in range(6)]
    bl = PointSet(6, pts).to_small_lambda(lazy=False).to_big_lambda()
    assert bl.is_signotope()
    assert bl.is_valid(print_collinear_warning=False)


def test_signotope_rejects_two_sign_changes():
    """A 4-tuple with sign pattern +,-,+,- (two changes) is not a signotope."""
    n = 4
    o = np.zeros((n, n, n), dtype=np.int8)
    # set sorted triples to +,-,+,- : o(0,1,2)=+, o(0,1,3)=-, o(0,2,3)=+, o(1,2,3)=-
    pattern = {(0, 1, 2): 1, (0, 1, 3): -1, (0, 2, 3): 1, (1, 2, 3): -1}
    for (a, b, c), v in pattern.items():
        v = np.int8(v)
        o[a, b, c] = o[b, c, a] = o[c, a, b] = v
        o[a, c, b] = o[b, a, c] = o[c, b, a] = -v
    bl = BigLambda(n, o)
    assert not bl.is_signotope()


# ---------------------------------------------------------------------------
# The real dataset: is data_jelena/test24.blt a valid order type at all?
# ---------------------------------------------------------------------------

def test_test24_blt_validity():
    """Report whether data_jelena/test24.blt is a valid chirotope / signotope.

    This is the file a user asked to realize.  If it is NOT a valid abstract
    order type, the reconstruction failure is correct behaviour on bad input,
    not a bug in the reconstruction algorithm.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "data_jelena", "test24.blt")
    if not os.path.exists(path):
        pytest.skip("data_jelena/test24.blt not present")
    line = next(l.strip() for l in open(path) if l.strip() and not l.startswith("#"))
    n = 24
    o = _o_from_blt_line(line, n)
    bl = BigLambda(n, o)

    is_sig = bl.is_signotope(print_violation=True)
    is_chi = bl.is_chirotope(print_violation=True)
    is_val = bl.is_valid(print_collinear_warning=False)
    print(f"test24.blt: is_signotope={is_sig}  is_chirotope={is_chi}  is_valid={is_val}")

    # FINDING: test24.color/.blt is NOT a valid order type.  The 4-tuple
    # (2,3,8,12) (1-indexed) carries colors B,R,B,R = +,-,+,-, which violates
    # both the acyclic 4-tuple axiom and the 5-tuple Grassmann-Plücker
    # (chirotope) axiom.  Hence SmallLambda.to_big_lambda() correctly refuses to
    # reconstruct a consistent chirotope from it — the data is the problem, not
    # the reconstruction algorithm.  The two chirotope checks must agree.
    assert is_chi is False, "expected test24 to be a non-chirotope"
    assert is_val is False, "is_valid and is_chirotope must agree on test24"
    assert is_sig is False, "test24 is not a signotope either"

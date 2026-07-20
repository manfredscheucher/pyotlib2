"""Regression tests for chirotope reconstruction: SmallLambda.to_big_lambda().

The rank matrix ``l`` uniquely determines the chirotope ``o``, so reconstructing
``o`` from ``l`` must reproduce the original chirotope *exactly* (element for
element, in the given point order), for EVERY order type — not only the ones in
natural / lex-min labelling.

The order-type database files (``otypesNN.b08``) store representatives in a
canonical (natural) labelling, which never exercises the reconstruction's
tie-breaking.  Real data (e.g. ``data_jelena/test24.blt``) is *not* canonically
labelled, and there the angular rank of two points around a common apex can tie
even though the orientation is well defined.  These tests reproduce that by
applying random point permutations to database order types before round-tripping.

A permutation ``pi`` acts on the chirotope by ``o_new[a,b,c] = o_old[pi[a], pi[b],
pi[c]]`` (with the usual antisymmetry ``o(a,b,c) = -o(a,c,b) = -o(b,a,c) = ...``);
``SmallLambda.relabeled`` performs exactly this relabelling on the rank matrix.
"""

import os
import random

import numpy as np
import pytest

from pyotlib2.io.readers import read_order_types
from pyotlib2.core.small_lambda import SmallLambda

OTDB = os.path.join(os.path.dirname(__file__), "otdb", "otypes")


def _db_path(n: int) -> str:
    return os.path.join(OTDB, f"otypes{n:02d}.b08")


def _abstract_from(sl: SmallLambda) -> SmallLambda:
    """A realization-free copy carrying only the rank matrix (as loaded data)."""
    return SmallLambda(sl.n, sl.get_l().copy())


def _reconstruct_matches(sl: SmallLambda) -> bool:
    """True iff to_big_lambda() reproduces sl's own chirotope exactly."""
    o_expected = sl.to_big_lambda().o
    o_got = _abstract_from(sl).to_big_lambda().o
    return bool(np.array_equal(o_got, o_expected))


# ---------------------------------------------------------------------------
# 1. Canonical (as stored) order types must round-trip — the easy case.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
def test_reconstruction_canonical_all(n):
    """Every order type in otypesNN.b08 reconstructs exactly (canonical labelling)."""
    if not os.path.exists(_db_path(n)):
        pytest.skip(f"missing {_db_path(n)}")
    for sl in read_order_types(_db_path(n), fmt="b08", n=n):
        assert _reconstruct_matches(sl), (
            f"canonical reconstruction mismatch at n={n}:\n{sl.to_string()}"
        )


# ---------------------------------------------------------------------------
# 2. Randomly RELABELLED order types must also round-trip — the bug case.
#    This is what non-canonical real data (test24.blt) hits.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_reconstruction_permuted(n):
    """Random point permutations of DB order types still reconstruct exactly."""
    if not os.path.exists(_db_path(n)):
        pytest.skip(f"missing {_db_path(n)}")
    rng = random.Random(1234 + n)
    perms_per_ot = 4
    checked = 0
    for sl in read_order_types(_db_path(n), fmt="b08", n=n):
        base = _abstract_from(sl)
        for _ in range(perms_per_ot):
            perm = list(range(n))
            rng.shuffle(perm)
            permuted = base.relabeled(perm)
            assert _reconstruct_matches(permuted), (
                f"permuted reconstruction mismatch at n={n}, perm={perm}:\n"
                f"{permuted.to_string()}"
            )
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# 3. The concrete real-world dataset that first exposed the bug.
# ---------------------------------------------------------------------------

def test_reconstruction_rejects_invalid_test24():
    """data_jelena/test24.blt is NOT a valid order type (see test_validity), so
    reconstructing a chirotope from its rank matrix must fail rather than invent
    a consistent one.  This documents that the earlier InvalidData was correct
    behaviour on bad input, not a reconstruction bug.
    """
    from itertools import combinations
    from pyotlib2.core.small_lambda import InvalidData

    path = os.path.join(os.path.dirname(__file__), "..", "data_jelena", "test24.blt")
    if not os.path.exists(path):
        pytest.skip("data_jelena/test24.blt not present")

    line = next(l.strip() for l in open(path) if l.strip() and not l.startswith("#"))
    chars = [c for c in line if c in "+-"]
    n = 24
    o = np.zeros((n, n, n), dtype=np.int8)
    for idx, (a, b, c) in enumerate(combinations(range(n), 3)):
        v = np.int8(1) if chars[idx] == "+" else np.int8(-1)
        o[a, b, c] = o[b, c, a] = o[c, a, b] = v
        o[a, c, b] = o[b, a, c] = o[c, b, a] = -v

    l = np.sum(o == 1, axis=2).astype(np.int32)
    with pytest.raises(InvalidData):
        SmallLambda(n, l).to_big_lambda()

"""Tests for the Octave-based realization testers (octave.<method>).

Each Octave minimiser (fminsearch / fminunc / sqp) must realize a small order
type *exactly*: the returned point set, run back through the pipeline, must
reproduce the original chirotope element-for-element (no relabeling).
"""

import shutil

import numpy as np
import pytest

if shutil.which("octave") is None:
    pytest.skip("octave binary not on PATH", allow_module_level=True)

from pyotlib2.core.point_set import PointSet
from pyotlib2.realization import make_tester, SOLVERS
from pyotlib2.realization.octave_tester import octave_methods
from tests.point_configs import one_interior, abstract_only


def test_methods_registered():
    for m in octave_methods():
        assert f"octave.{m}" in SOLVERS


@pytest.mark.parametrize("method", ["fminsearch", "fminunc", "sqp"])
def test_octave_method_realizes_small_exactly(method):
    sl = abstract_only(one_interior(6))
    o0 = sl.to_big_lambda().o.copy()
    t = make_tester(f"octave.{method}", trials=8, seed=0)
    assert t.is_realizable(sl) is True
    # exact chirotope match: realization -> pipeline -> same o array
    o = (PointSet(sl.n, sl.realization)
         .to_small_lambda(lazy=False).to_big_lambda().o)
    assert np.array_equal(o, o0)


def test_plain_octave_is_fminsearch():
    """The back-compat plain ``octave`` solver still names itself fminsearch."""
    t = make_tester("octave")
    assert t.name == "octave.fminsearch"


def test_octave_logsum_realizes_small_exactly():
    """The log-sum objective must also drive a small OT to an exact realization."""
    sl = abstract_only(one_interior(6))
    o0 = sl.to_big_lambda().o.copy()
    t = make_tester("octave.fminsearch", logsum=True, trials=20, seed=0)
    assert t.logsum is True
    assert t.is_realizable(sl) is True
    o = (PointSet(sl.n, sl.realization)
         .to_small_lambda(lazy=False).to_big_lambda().o)
    assert np.array_equal(o, o0)

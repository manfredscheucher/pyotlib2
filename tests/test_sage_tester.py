"""Tests for the SageMath-based realization testers (sage / sage.<method>).

Skipped unless the ``sage`` binary is available.  Each Sage ``minimize``
algorithm must realize a small order type *exactly*: the returned point set,
run back through the pipeline, reproduces the original chirotope.
"""

import os
import shutil

import numpy as np
import pytest

_SAGE = shutil.which("sage") or (
    "/usr/local/bin/sage" if os.path.exists("/usr/local/bin/sage") else None
)
if _SAGE is None:
    pytest.skip("sage binary not available", allow_module_level=True)

from pyotlib2.core.point_set import PointSet
from pyotlib2.realization import make_tester, SOLVERS
from pyotlib2.realization.sage_tester import sage_methods
from tests.point_configs import one_interior, abstract_only


def test_methods_registered():
    assert "sage" in SOLVERS
    for m in sage_methods():
        assert f"sage.{m}" in SOLVERS


@pytest.mark.parametrize("method", ["simplex", "powell"])
def test_sage_method_realizes_small_exactly(method):
    sl = abstract_only(one_interior(6))
    o0 = sl.to_big_lambda().o.copy()
    t = make_tester(f"sage.{method}", trials=5, seed=0)
    assert t.is_realizable(sl) is True
    o = (PointSet(sl.n, sl.realization)
         .to_small_lambda(lazy=False).to_big_lambda().o)
    assert np.array_equal(o, o0)


def test_sage_logsum_realizes_small_exactly():
    """The log-sum objective must also drive a small OT to an exact realization."""
    sl = abstract_only(one_interior(6))
    o0 = sl.to_big_lambda().o.copy()
    t = make_tester("sage.simplex", logsum=True, trials=20, seed=0)
    assert t.logsum is True
    assert t.is_realizable(sl) is True
    o = (PointSet(sl.n, sl.realization)
         .to_small_lambda(lazy=False).to_big_lambda().o)
    assert np.array_equal(o, o0)

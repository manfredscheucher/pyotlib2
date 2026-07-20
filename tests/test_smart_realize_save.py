"""Tests for smart-realize result saving (--output)."""

import numpy as np

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.core.point_set import PointSet
from pyotlib2.cli.commands import smart_realize, _save_smart_result
from pyotlib2.io.readers import read_order_types
from tests.point_configs import one_interior, abstract_only


def test_saves_realization_as_psz(tmp_path):
    sl = abstract_only(one_interior(9))
    o0 = sl.to_big_lambda().o.copy()
    res, wit = smart_realize(sl, solver="scipy", verbose=False)
    assert res is True
    out = tmp_path / "result"
    _save_smart_result(str(out), 1, 1, res, wit, 6)
    psz = out.with_suffix(".psz")
    assert psz.exists()
    # the saved point set reproduces the exact order type
    loaded = next(iter(read_order_types(str(psz))))
    assert np.array_equal(loaded.to_big_lambda().o, o0)
    # header records the verdict
    head = psz.read_text().splitlines()[0]
    assert "realizable" in head


def test_saves_witness_as_lt_with_source_indices(tmp_path):
    sl = abstract_only(one_interior(6))
    sl.source_indices = [3, 7, 1, 12]
    out = tmp_path / "witness"
    _save_smart_result(str(out), 1, 1, None, sl, 6)
    lt = out.with_suffix(".lt")
    assert lt.exists()
    text = lt.read_text()
    assert "unknown" in text
    assert "original point indices: [3, 7, 1, 12]" in text
    # the rank matrix round-trips
    loaded = next(iter(read_order_types(str(lt))))
    assert loaded.n == 6


def test_no_output_writes_nothing(tmp_path):
    sl = abstract_only(one_interior(6))
    # output=None → no file created, no error
    _save_smart_result(None, 1, 1, True, sl, 6)


def test_multi_ot_appends_index(tmp_path):
    sl = abstract_only(one_interior(9))
    res, wit = smart_realize(sl, solver="scipy", verbose=False)
    out = tmp_path / "many"
    _save_smart_result(str(out), 2, 5, res, wit, 6)  # OT #2 of 5
    assert (out.parent / "many_2.psz").exists()

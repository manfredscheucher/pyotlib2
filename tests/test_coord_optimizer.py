"""Tests for the literal ports of the old pyotlib coordinate optimizers.

Covers pyotlib2.algorithms.coord_optimizer (minimizeIntegerCoordinates,
beautifyPointSet, beautify2, MinimizePatternSearch), symmetry,
pattern_search and the CLI wrappers.
"""

import random
import sys
from fractions import Fraction

import pytest

from pyotlib2.algorithms import coord_optimizer as co
from pyotlib2.algorithms.pattern_search import my_pattern_search
from pyotlib2.algorithms.symmetry import compute_symmetric_ps
from pyotlib2.core.point_set import PointSet

PTS7 = [(0, 0), (1000, 37), (980, 1010), (13, 990), (300, 420), (610, 333), (450, 700)]


def _ps(pts=PTS7):
    return PointSet(len(pts), list(pts))


def _ot_str(pts):
    return PointSet(len(pts), list(pts)).to_small_lambda().to_string()


# ---------------------------------------------------------------------------
# minimize_integer_coordinates
# ---------------------------------------------------------------------------

class TestMinimizeIntegerCoordinates:
    @pytest.mark.parametrize("kw", [
        {},
        {"maxtrials": 5},
        {"maxtrials": 5, "sheering": True},
        {"maxtrials": 5, "minimize_diameter": True},
        {"maxtrials": 4, "randomize": 0},
    ])
    def test_ot_preserved_and_not_larger(self, kw):
        random.seed(1)
        ps = _ps()
        r = co.minimize_integer_coordinates(_ps(), **kw)
        assert r.to_small_lambda().to_string() == ps.to_small_lambda().to_string()
        assert r.bits() <= ps.bits()
        assert r.diameter() <= ps.diameter()

    def test_actually_minimizes(self):
        random.seed(1)
        assert co.minimize_integer_coordinates(_ps()).bits() < _ps().bits()

    def test_deterministic_with_seed(self):
        random.seed(7)
        a = co.minimize_integer_coordinates(_ps(), maxtrials=5, sheering=True).points
        random.seed(7)
        b = co.minimize_integer_coordinates(_ps(), maxtrials=5, sheering=True).points
        assert a == b

    def test_tolerance_returns_early(self):
        ps = _ps()
        r = co.minimize_integer_coordinates(ps, tolerance=ps.bits())
        assert r.points == ps.normalized().points

    def test_symmetric(self):
        P0 = compute_symmetric_ps([(1000, 0), None, None, (400, 250), None, None], 3)
        ps = PointSet(6, list(P0))
        assert not ps.has_collinear_points()
        random.seed(3)
        r = co.minimize_integer_coordinates(PointSet(6, list(P0)), maxtrials=3, symmetric=3)
        assert r.to_small_lambda().to_string() == ps.to_small_lambda().to_string()
        assert compute_symmetric_ps(r.points, 3) == r.points
        assert r.bits() <= ps.bits()

    def test_balance_points(self):
        assert co._balance_points([(0, 0), (2, 0), (0, 7)]) == [(0, 0), (6, 0), (0, 7)]
        assert co._balance_points([(0, 0), (2, 0), (0, 7)], BlowUpConstant=3) == \
            [(0, 0), (20, 0), (0, 21)]


# ---------------------------------------------------------------------------
# beautify_point_set
# ---------------------------------------------------------------------------

class TestBeautifyPointSet:
    def test_ot_preserved(self):
        r = co.beautify_point_set(_ps(), maxiter=10)
        assert r.to_small_lambda().to_string() == _ot_str(PTS7)

    def test_deterministic(self):
        assert co.beautify_point_set(_ps(), maxiter=10).points == \
            co.beautify_point_set(_ps(), maxiter=10).points

    def test_goodbits_scales_up(self):
        ps = _ps()
        r = co.beautify_point_set(ps, maxiter=3, goodbits=ps.bits() + 2)
        assert r.to_small_lambda().to_string() == _ot_str(PTS7)
        assert r.bits() > ps.bits()


# ---------------------------------------------------------------------------
# beautify2
# ---------------------------------------------------------------------------

class TestBeautify2:
    def test_ot_preserved(self):
        pytest.importorskip("scipy")
        ot = _ps().to_small_lambda(lazy=False)
        r = co.beautify2(ot, iter1=1, iter2=100)
        assert r.to_string() == ot.to_string()
        assert r.realization is not None

    def test_minimize_not_larger(self):
        pytest.importorskip("scipy")
        random.seed(2)
        ot = _ps().to_small_lambda(lazy=False)
        r = co.beautify2(ot, iter1=1, iter2=100, minimize=True)
        assert r.to_string() == ot.to_string()
        assert PointSet(7, r.realization).bits() <= _ps().bits()

    def test_diam_bound_violated_returns_input(self):
        pytest.importorskip("scipy")
        ot = _ps().to_small_lambda(lazy=False)
        assert co.beautify2(ot, iter1=1, iter2=50, diam_bound=10) is ot

    def test_to_integer_exact(self):
        assert co._to_integer([0.5, 0.25, 3.0]) == [2, 1, 12]
        assert co._to_integer([Fraction(1, 3), 1]) == [1, 3]


# ---------------------------------------------------------------------------
# minimize_pattern_search
# ---------------------------------------------------------------------------

class TestMinimizePatternSearch:
    def test_ot_preserved_and_smaller(self):
        ot = _ps().to_small_lambda(lazy=False)
        r, blowup = co.minimize_pattern_search(ot, maxtrials=300)
        assert blowup == 0
        assert _ot_str(r.realization) == ot.to_string()
        assert co._ps_get_diameter2(r.realization) <= co._ps_get_diameter2(PTS7)
        assert min(x for x, _ in r.realization) == 0
        assert min(y for _, y in r.realization) == 0

    def test_deterministic(self):
        ot = _ps().to_small_lambda(lazy=False)
        a, _ = co.minimize_pattern_search(ot, maxtrials=100)
        b, _ = co.minimize_pattern_search(ot, maxtrials=100)
        assert a.realization == b.realization

    def test_tolerance_skips(self):
        ot = _ps().to_small_lambda(lazy=False)
        r, _ = co.minimize_pattern_search(ot, tolerance=_ps().bits())
        assert r.realization == PTS7


# ---------------------------------------------------------------------------
# helpers: symmetry, pattern search, cocircularity
# ---------------------------------------------------------------------------

def test_compute_symmetric_ps():
    assert compute_symmetric_ps([(10, 0), (5, 5)], 2) == [(10, 0), (-10, 0)]
    assert compute_symmetric_ps([(10, 0), None, (7, 7)], 2) == [(10, 0), (-10, 0), (0, 0)]


def test_my_pattern_search_quadratic():
    f = lambda X: sum((X[i] - i) ** 2 for i in range(len(X)))
    res = my_pattern_search(f, [0.0] * 4, 0.1, maxiter=300)
    assert res["f0"] < 1e-6


def test_has_cocircular_points():
    assert co.has_cocircular_points([(0, 0), (2, 0), (0, 2), (2, 2)])
    assert not co.has_cocircular_points([(0, 0), (3, 0), (0, 2), (5, 7)])


# ---------------------------------------------------------------------------
# CLI wrappers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["minimize-integer-coords", "--maxtrials", "2", "--sheering", "1"],
    ["beautify-pointset", "--maxiter", "5"],
    ["beautify2", "--iter1", "1", "--iter2", "50"],
    ["minimize-pattern-search", "--maxtrials", "100"],
])
def test_cli(argv, tmp_path, monkeypatch, capsys):
    if argv[0] == "beautify2":
        pytest.importorskip("scipy")
    from pyotlib2.cli.main import main
    from pyotlib2.io.readers import read_order_types
    from pyotlib2.io.writers import write_order_types

    inp = tmp_path / "in.psz"
    out = tmp_path / "out.psz"
    write_order_types([_ps().to_small_lambda(lazy=False)], inp, fmt="psz")
    random.seed(1)
    monkeypatch.setattr(sys, "argv", ["pyotlib2", *argv, str(inp), "-n", "7", "-o", str(out)])
    main()
    res = list(read_order_types(str(out), n=7))
    assert len(res) == 1
    assert _ot_str(res[0].realization) == _ot_str(PTS7)
    if argv[0].startswith("minimize"):  # the beautifiers may spread the points out
        assert PointSet(7, res[0].realization).bits() <= _ps().bits()


def test_symmetry_uses_python2_rounding():
    from pyotlib2.algorithms.symmetry import _round_py2
    assert [_round_py2(v) for v in (2.5, -2.5, 0.5, 1.4, -1.6)] == [3, -3, 1, 1, -2]


def test_minimize_ensure_property_changes_order_type():
    from pyotlib2.cli.commands import _count_property
    random.seed(3)
    pts = [(random.randint(0, 10**6), random.randint(0, 10**6)) for _ in range(9)]
    c0 = _count_property(_ps(pts).to_small_lambda(lazy=False), "crossings")
    random.seed(3)
    r = co.minimize_integer_coordinates(_ps(list(pts)), maxtrials=2, ensure_property="crossings")
    assert r.bits() < _ps(pts).bits()
    assert not r.has_collinear_points()
    assert _count_property(r.to_small_lambda(lazy=False), "crossings") <= c0


@pytest.mark.parametrize("seed", range(12))
def test_cli_prop_never_worse(seed, tmp_path, monkeypatch):
    from pyotlib2.cli.commands import _count_property
    from pyotlib2.cli.main import main
    from pyotlib2.io.readers import read_order_types
    from pyotlib2.io.writers import write_order_types

    random.seed(seed)
    pts = [(random.randint(0, 10**5), random.randint(0, 10**5)) for _ in range(8)]
    if _ps(pts).has_collinear_points():
        pytest.skip("collinear start")
    c0 = _count_property(_ps(pts).to_small_lambda(lazy=False), "empty-triangles")
    inp = tmp_path / "in.psz"
    out = tmp_path / "out.psz"
    write_order_types([_ps(pts).to_small_lambda(lazy=False)], inp, fmt="psz")
    monkeypatch.setattr(sys, "argv", ["pyotlib2", "minimize-integer-coords", str(inp), "-n", "8",
                                      "--maxtrials", "2", "--prop", "empty-triangles", "-o", str(out)])
    main()
    (res,) = list(read_order_types(str(out), n=8))
    ps = _ps(list(res.realization))
    assert not ps.has_collinear_points()
    assert _count_property(ps.to_small_lambda(lazy=False), "empty-triangles") <= c0
    assert ps.bits() <= _ps(pts).bits()

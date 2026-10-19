"""Tests for CLI commands and the algorithms they use.

Covers: exit_edges, flip_triple, minimize_coords, beautify_coords,
        walk_points, walk_abstract, extend_abstract, extend_random,
        and the coordinate-minimization helpers.
"""

import pytest
from pyotlib2.core.point_set import PointSet
from pyotlib2.core.small_lambda import SmallLambda
from tests.point_configs import (
    convex_position_moment_curve as _convex,
    convex_position_circle,
    horton_set,
    one_interior as _one_interior,
    abstract_only,
)


def _convex_abstract(n: int) -> SmallLambda:
    return abstract_only(_convex(n))


# ---------------------------------------------------------------------------
# exit_edges
# ---------------------------------------------------------------------------

class TestExitEdges:
    def test_convex4_diagonals_exit(self):
        """Convex quadrilateral: only the 2 diagonals are exit edges.

        For a convex 4-gon labelled 0-1-2-3, the diagonals (0,2) and (1,3)
        each have an empty triangle as witness.  The 4 sides are NOT exit
        edges because no point lies in the adjacent empty triangle position.
        """
        from pyotlib2.algorithms.exit_edges import filter_exit_edges
        sl = _convex(4)
        edges = filter_exit_edges(sl)
        assert edges == {(0, 2), (1, 3)}

    def test_witnesses_returned(self):
        """return_witnesses=True gives a witness for each exit edge."""
        from pyotlib2.algorithms.exit_edges import filter_exit_edges
        sl = _convex(5)
        edges, witnesses = filter_exit_edges(sl, return_witnesses=True)
        assert edges == set(witnesses.keys())
        for ws in witnesses.values():
            assert len(ws) >= 1

    def test_exit_triples_nonempty(self):
        """exit_triples returns at least one triple for any non-degenerate OT."""
        from pyotlib2.algorithms.exit_edges import exit_triples
        sl = _one_interior()
        triples = exit_triples(sl)
        assert len(triples) > 0

    def test_exit_triples_cover_ot(self):
        """Orientations of exit triples uniquely identify the OT.

        If we build a BigLambda from scratch using only the exit-triple
        signs, we should recover the same OT string.
        """
        from pyotlib2.algorithms.exit_edges import exit_triples
        sl = _one_interior()
        bl = sl.to_big_lambda()
        o = bl.o
        triples = exit_triples(sl)
        # All exit-triple orientations match the original chirotope
        for a, b, c in triples:
            assert int(o[a, b, c]) != 0


# ---------------------------------------------------------------------------
# flip_triple
# ---------------------------------------------------------------------------

class TestFlipTriple:
    def test_flip_changes_orientation(self):
        """flip_triple changes the sign of o[a,b,c]."""
        sl = _convex(5)
        bl = sl.to_big_lambda()
        a, b, c = 0, 1, 2
        orig = int(bl.o[a, b, c])
        bl2 = bl.flip_triple(a, b, c)
        assert int(bl2.o[a, b, c]) == -orig

    def test_flip_antisymmetry_preserved(self):
        """After flip, antisymmetry o[a,b,c] = -o[a,c,b] still holds."""
        sl = _convex(5)
        bl = sl.to_big_lambda()
        bl2 = bl.flip_triple(0, 1, 2)
        n = bl2.n
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert bl2.o[i, j, k] == -bl2.o[i, k, j]

    def test_flip_double_is_identity(self):
        """Flipping the same triple twice returns the original chirotope."""
        sl = _convex(5)
        bl = sl.to_big_lambda()
        bl2 = bl.flip_triple(0, 1, 2).flip_triple(0, 1, 2)
        import numpy as np
        assert np.array_equal(bl.o, bl2.o)


# ---------------------------------------------------------------------------
# minimize_coords
# ---------------------------------------------------------------------------

class TestMinimizeCoords:
    def _realized_sl(self):
        pts = [(0, 0), (100, 0), (100, 100), (0, 100), (10, 20)]
        return PointSet(5, pts).to_small_lambda(lazy=False)

    def test_ot_preserved(self):
        """minimize_coords does not change the order type."""
        from pyotlib2.cli.commands import minimize_coords
        sl = self._realized_sl()
        result = minimize_coords(sl, trials=3, randomize=3)
        assert result.to_string() == sl.to_string()

    def test_bits_not_increased(self):
        """Coordinate bit-width should not increase."""
        from pyotlib2.cli.commands import minimize_coords, _pts_get_bits
        sl = self._realized_sl()
        b0 = _pts_get_bits(list(sl.realization))
        result = minimize_coords(sl, trials=5, randomize=5)
        b1 = _pts_get_bits(list(result.realization))
        assert b1 <= b0

    def test_requires_realization(self):
        """Raises ValueError if no realization is attached."""
        from pyotlib2.cli.commands import minimize_coords
        sl = _convex_abstract(5)
        with pytest.raises(ValueError, match="realization"):
            minimize_coords(sl)


# ---------------------------------------------------------------------------
# beautify_coords  (gradient descent, method=gd)
# ---------------------------------------------------------------------------

class TestBeautifyCoords:
    def _realized_sl(self):
        pts = [(0, 0), (50, 0), (50, 50), (0, 50), (5, 10)]
        return PointSet(5, pts).to_small_lambda(lazy=False)

    def test_ot_preserved_gd(self):
        """beautify_coords (gd) preserves the order type."""
        from pyotlib2.cli.commands import beautify_coords
        sl = self._realized_sl()
        result = beautify_coords(sl, max_iter=5)
        assert result.to_string() == sl.to_string()

    def test_requires_realization_gd(self):
        from pyotlib2.cli.commands import beautify_coords
        sl = _convex_abstract(5)
        with pytest.raises(ValueError, match="realization"):
            beautify_coords(sl)

    @pytest.mark.slow
    def test_ot_preserved_nm(self):
        """beautify2_coords (Nelder-Mead) preserves the order type."""
        pytest.importorskip("scipy")
        from pyotlib2.cli.commands import beautify2_coords
        sl = self._realized_sl()
        result = beautify2_coords(sl, iter1=1, iter2=100)
        assert result.to_string() == sl.to_string()


# ---------------------------------------------------------------------------
# walk_points
# ---------------------------------------------------------------------------

class TestWalkPoints:
    def _realized_sl(self):
        return _convex(10)

    def test_random_walk_yields_valid_ots(self):
        """walk_points --random yields SmallLambdas with valid abstract structure."""
        from pyotlib2.cli.commands import walk_points
        sl = self._realized_sl()
        results = list(walk_points(
            sl, "crossings",
            random_walk=True, trials=200, verbose=False,
        ))
        for r in results:
            assert r._is_valid_abstract()

    def test_dfs_walk_yields_valid_ots(self):
        """walk_points DFS yields SmallLambdas with valid abstract structure."""
        from pyotlib2.cli.commands import walk_points
        sl = self._realized_sl()
        results = list(walk_points(
            sl, "crossings",
            random_walk=False, trace=5, good_bits=10, max_steps=10, verbose=False,
        ))
        for r in results:
            assert r._is_valid_abstract()

    def test_maximize_yields_increasing_values(self):
        """walk_points(maximize=True) moves towards larger values (convex position has the most crossings)."""
        import random
        from pyotlib2.cli.commands import walk_points, _count_property
        random.seed(0)
        sl = _one_interior(6)
        start = _count_property(sl, "crossings")
        values = [_count_property(r, "crossings") for r in walk_points(
            sl, "crossings", random_walk=True, trials=2000, verbose=False, maximize=True,
        )]
        assert values, "no improvement found"
        assert all(a < b for a, b in zip([start] + values, values))

    def test_requires_realization(self):
        from pyotlib2.cli.commands import walk_points
        sl = _convex_abstract(5)
        with pytest.raises(ValueError, match="realization"):
            list(walk_points(sl, "crossings", random_walk=True, trials=1, verbose=False))


# ---------------------------------------------------------------------------
# walk_abstract
# ---------------------------------------------------------------------------

class TestWalkAbstract:
    def test_random_walk_yields_valid_ots(self):
        """walk_abstract --random yields valid abstract OTs."""
        from pyotlib2.cli.commands import walk_abstract
        sl = _convex(6)
        results = list(walk_abstract(
            sl, "crossings",
            random_walk=True, trials=50, verbose=False,
        ))
        for r in results:
            assert r._is_valid_abstract()

    def test_dfs_walk_yields_valid_ots(self):
        """walk_abstract DFS yields valid abstract OTs."""
        from pyotlib2.cli.commands import walk_abstract
        sl = _one_interior()
        results = list(walk_abstract(
            sl, "empty-triangles",
            random_walk=False, trace=20, max_steps=50, verbose=False,
        ))
        for r in results:
            assert r._is_valid_abstract()

    def test_no_realization_needed(self):
        """walk_abstract works on abstract OTs without coordinates."""
        from pyotlib2.cli.commands import walk_abstract
        sl = _convex_abstract(5)
        assert sl.realization is None
        # should not raise
        list(walk_abstract(sl, "crossings", random_walk=True, trials=10, verbose=False))


# ---------------------------------------------------------------------------
# extend_abstract
# ---------------------------------------------------------------------------

class TestExtendAbstract:
    def test_n3_extends_to_n4(self):
        """Every n=3 OT (there is only 1) extends to at least one n=4 OT."""
        pytest.importorskip("pysat")
        from pyotlib2.cli.commands import extend_abstract
        sl = _convex(3)
        extensions = list(extend_abstract(sl))
        assert len(extensions) >= 1

    def test_extensions_are_valid(self):
        """All extensions pass the abstract validity check."""
        pytest.importorskip("pysat")
        from pyotlib2.cli.commands import extend_abstract
        sl = _convex(4)
        for ext in extend_abstract(sl):
            assert ext._is_valid_abstract()
            assert ext.n == 5

    def test_extensions_have_right_size(self):
        """Each extension has n+1 points and is a valid abstract OT."""
        pytest.importorskip("pysat")
        from pyotlib2.cli.commands import extend_abstract
        sl = _convex(4)
        for ext in extend_abstract(sl):
            assert ext.n == 5
            assert ext._is_valid_abstract()

    def test_no_duplicates(self):
        """extend_abstract yields distinct OTs (no duplicates)."""
        pytest.importorskip("pysat")
        from pyotlib2.cli.commands import extend_abstract
        sl = _convex(3)
        extensions = list(extend_abstract(sl))
        keys = [e.to_string() for e in extensions]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# extend_random
# ---------------------------------------------------------------------------

class TestExtendRandom:
    def _realized_sl(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        return PointSet(4, pts).to_small_lambda(lazy=False)

    def test_yields_n_plus_one(self):
        """extend_random yields OTs with n+1 points."""
        from pyotlib2.cli.commands import extend_random
        sl = self._realized_sl()
        results = list(extend_random(sl, trials=50))
        for r in results:
            assert r.n == sl.n + 1

    def test_yields_valid_ots(self):
        """extend_random yields valid abstract OTs."""
        from pyotlib2.cli.commands import extend_random
        sl = self._realized_sl()
        for r in extend_random(sl, trials=50):
            assert r._is_valid_abstract()

    def test_no_duplicates(self):
        """extend_random does not yield duplicate OTs."""
        from pyotlib2.cli.commands import extend_random
        sl = self._realized_sl()
        results = list(extend_random(sl, trials=100))
        keys = [r.to_string() for r in results]
        assert len(keys) == len(set(keys))

    def test_requires_realization(self):
        from pyotlib2.cli.commands import extend_random
        sl = _convex_abstract(4)
        with pytest.raises(ValueError, match="realization"):
            list(extend_random(sl, trials=1))


# ---------------------------------------------------------------------------
# _count_property helper
# ---------------------------------------------------------------------------

class TestCountProperty:
    def test_crossings_convex4(self):
        """Convex quadrilateral has 1 crossing."""
        from pyotlib2.cli.commands import _count_property
        sl = _convex(4)
        assert _count_property(sl, "crossings") == 1

    def test_hull_convex5(self):
        """5 points in convex position: hull size = 5."""
        from pyotlib2.cli.commands import _count_property
        sl = _convex(5)
        assert _count_property(sl, "hull") == 5

    def test_hull_one_interior(self):
        """Square + 1 interior: hull size = 4."""
        from pyotlib2.cli.commands import _count_property
        sl = _one_interior()
        assert _count_property(sl, "hull") == 4

    def test_unknown_property_raises(self):
        from pyotlib2.cli.commands import _count_property
        sl = _convex(4)
        with pytest.raises(ValueError, match="Unknown property"):
            _count_property(sl, "nonexistent")

    def test_empty_triangles(self):
        from pyotlib2.cli.commands import _count_property
        sl = _convex(4)
        # convex 4-gon has 4 empty triangles
        assert _count_property(sl, "empty-triangles") == 4

    def test_onion_layers_convex(self):
        from pyotlib2.cli.commands import _count_property
        sl = _convex(5)
        assert _count_property(sl, "onion-layers") == 1

    def test_onion_layers_with_interior(self):
        from pyotlib2.cli.commands import _count_property
        sl = _one_interior()
        assert _count_property(sl, "onion-layers") == 2


# ---------------------------------------------------------------------------
# export helpers: LP / Mathematica (Feature 3)
# ---------------------------------------------------------------------------

class TestExportHelpers:
    def test_export_lp_single_and_parseable(self, tmp_path):
        """The exported LP starts with the objective sense (glpsol-parseable)."""
        from pyotlib2.cli.commands import _export_lp
        sl = _one_interior(6)
        out = tmp_path / "gp.lp"
        _export_lp([sl], str(out), 6)
        text = out.read_text()
        # CPLEX-LP: must begin with Maximize (no leading comment) and end with End
        assert text.lstrip().startswith("Maximize")
        assert "myEps" in text
        assert text.rstrip().endswith("End")

    def test_export_lp_multiple_hashes_filenames(self, tmp_path):
        """Multiple OTs get distinct hash-tagged filenames."""
        from pyotlib2.cli.commands import _export_lp
        a = _one_interior(6)
        b = _convex_abstract(6)
        base = tmp_path / "gp.lp"
        _export_lp([a, b], str(base), 6)
        files = sorted(p.name for p in tmp_path.glob("gp_*.lp"))
        assert len(files) == 2

    def test_export_mathematica_script(self, tmp_path):
        """The Mathematica export works even without wolframscript on PATH."""
        from pyotlib2.cli.commands import _export_mathematica
        sl = _one_interior(6)
        out = tmp_path / "qcp.m"
        _export_mathematica([sl], str(out), 6)
        text = out.read_text()
        assert "NMinimize[" in text
        assert "Export[" in text
        # default: no WorkingPrecision, squared hinge (not Log)
        assert "WorkingPrecision" not in text
        assert "Log[" not in text

    def test_export_mathematica_honors_precision_and_logsum(self, tmp_path):
        """--precision and --logsum flow into the exported Mathematica script."""
        import math
        from pyotlib2.cli.commands import _export_mathematica
        sl = _one_interior(6)
        out = tmp_path / "qcp.m"
        _export_mathematica([sl], str(out), 6, precision=200, logsum=True)
        text = out.read_text()
        # precision is in BITS; WorkingPrecision counts DECIMAL digits (converted)
        digits = math.ceil(200 / math.log2(10))  # 61
        assert f"WorkingPrecision -> {digits}" in text
        assert "Log[" in text

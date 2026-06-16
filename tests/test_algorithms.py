"""Unit tests for algorithms: polygon counting, crossings, unify, sub-OTs."""

import pytest
from pyotlib2.core.point_set import PointSet
from pyotlib2.core.small_lambda import SmallLambda


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_convex_position(n: int):
    """Return n points on a convex polygon (regular n-gon on integer grid)."""
    import math
    R = 1000
    pts = [(round(R * math.cos(2 * math.pi * i / n)),
            round(R * math.sin(2 * math.pi * i / n)))
           for i in range(n)]
    return PointSet(n, pts)


def make_6_points():
    """Return 6 points in general position: 5 convex hull + 1 interior."""
    return PointSet(6, [(0,0),(6,0),(9,3),(6,6),(0,6),(-3,3)])


# ---------------------------------------------------------------------------
# Polygon counting
# ---------------------------------------------------------------------------

class TestPolygonCount:
    def test_empty_triangles_convex4(self):
        """Convex quadrilateral has exactly 4 empty triangles (the 4 sub-triples)."""
        from pyotlib2.algorithms.polygon_count import count_triangles
        ps = make_convex_position(4)
        bl = ps.to_big_lambda()
        assert count_triangles(bl, empty_only=True) == 4

    def test_convex_triangles_convex4(self):
        """Convex quadrilateral has exactly 4 triangles."""
        from pyotlib2.algorithms.polygon_count import count_triangles
        ps = make_convex_position(4)
        bl = ps.to_big_lambda()
        assert count_triangles(bl, empty_only=False) == 4

    def test_empty_kgons_convex_n(self):
        """n points in convex position have exactly 1 empty n-gon."""
        from pyotlib2.algorithms.polygon_count import count_polygons
        for n in [4, 5, 6]:
            ps = make_convex_position(n)
            bl = ps.to_big_lambda()
            assert count_polygons(bl, n, empty_only=True) == 1

    def test_crossings_convex4(self):
        """Convex quadrilateral has exactly 1 crossing pair (the two diagonals)."""
        from pyotlib2.algorithms.polygon_count import count_crossings
        ps = make_convex_position(4)
        sl = ps.to_small_lambda(lazy=False)
        assert count_crossings(sl) == 1


# ---------------------------------------------------------------------------
# Crossings
# ---------------------------------------------------------------------------

class TestCrossings:
    def test_crossing_pairs_convex5(self):
        """5 points in convex position have C(5,2)/2 = 5 crossing pairs."""
        from pyotlib2.algorithms.crossings import crossing_pairs
        ps = make_convex_position(5)
        bl = ps.to_big_lambda()
        pairs = crossing_pairs(bl)
        assert len(pairs) == 5

    def test_crossing_family_size_1(self):
        """Any two crossing edges form a 2-crossing family."""
        from pyotlib2.algorithms.crossings import enumerate_crossing_families
        ps = make_convex_position(4)
        bl = ps.to_big_lambda()
        families = list(enumerate_crossing_families(bl, 2))
        assert len(families) == 1  # only one crossing pair in a convex 4-gon

    def test_2families_equals_crossings(self):
        """#2-crossing-families == crossing number == #convex-4-gons, for all algos."""
        from pyotlib2.algorithms.crossings import count_crossing_families
        from pyotlib2.algorithms.polygon_count import count_crossings, count_polygons
        for n in [4, 5, 6, 7]:
            ps = make_convex_position(n)
            bl = ps.to_big_lambda()
            sl = ps.to_small_lambda(lazy=False)
            cr = count_crossings(sl)
            c4 = count_polygons(bl, k=4, empty_only=False)
            assert cr == c4, f"n={n}: crossings={cr} != 4-gons={c4}"
            for algo in ("pruned", "basic"):
                cf2 = count_crossing_families(bl, k=2, algo=algo)
                assert cf2 == cr, f"n={n} algo={algo}: 2-families={cf2} != crossings={cr}"

    def test_2families_equals_crossings_otdb(self):
        """#2-crossing-families == crossing number for all n=6 OTDB order types."""
        import os
        from pyotlib2.io.readers import read_order_types
        from pyotlib2.algorithms.crossings import count_crossing_families
        from pyotlib2.algorithms.polygon_count import count_crossings, count_polygons
        path = os.path.join(os.path.dirname(__file__), "otdb", "otypes", "otypes06.b08")
        if not os.path.exists(path):
            pytest.skip(f"{path} not found — run tests/otdb/download.py first")
        for sl in read_order_types(path, n=6):
            bl = sl.to_big_lambda()
            cr = count_crossings(sl)
            c4 = count_polygons(bl, k=4, empty_only=False)
            assert cr == c4
            for algo in ("pruned", "basic"):
                cf2 = count_crossing_families(bl, k=2, algo=algo)
                assert cf2 == cr, f"algo={algo}: 2-families={cf2} != crossings={cr}"


# ---------------------------------------------------------------------------
# Unify
# ---------------------------------------------------------------------------

class TestUnify:
    def test_unify_removes_duplicates(self):
        from pyotlib2.algorithms.unify import unify
        ps = PointSet(4, [(0,0),(1,0),(1,1),(0,1)])
        sl = ps.to_small_lambda(lazy=False)
        ots = [sl, sl, sl.get_lex_min()]
        unique = list(unify(ots))
        assert len(unique) == 1

    def test_unify_keeps_distinct(self):
        from pyotlib2.algorithms.unify import unify
        # n=5: all-convex vs 1-interior (non-degenerate: interior point off all diagonals)
        sl_conv = make_convex_position(5).to_small_lambda(lazy=False)
        sl_1in  = PointSet(5, [(0,0),(6,0),(6,6),(0,6),(1,2)]).to_small_lambda(lazy=False)
        assert not PointSet(5, [(0,0),(6,0),(6,6),(0,6),(1,2)]).has_collinear_points()
        # feed duplicates too
        unique = list(unify([sl_conv, sl_1in, sl_conv, sl_1in]))
        assert len(unique) == 2


# ---------------------------------------------------------------------------
# Sub-order types
# ---------------------------------------------------------------------------

class TestSubOrderTypes:
    def test_reduce_count(self):
        """n points in convex position have C(n,k) sub-OTs of size k."""
        import math
        n, k = 6, 4
        ps = make_convex_position(n)
        sl = ps.to_small_lambda(lazy=False)
        subs = list(sl.reduce(k, lex_min=False))
        assert len(subs) == math.comb(n, k)

    def test_distinct_sub_ots_convex(self):
        """6 points in convex position — all C(6,4)=15 sub-4-OTs are the same (convex 4-gon)."""
        from pyotlib2.algorithms.sub_order_types import count_distinct_sub_ots
        ps = make_convex_position(6)
        sl = ps.to_small_lambda(lazy=False)
        # All C(6,4) = 15 sub-4-sets are the unique convex 4-gon order type
        assert count_distinct_sub_ots(sl, 4) == 1



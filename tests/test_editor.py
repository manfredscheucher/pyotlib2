"""Tests for the interactive editor state (pyotlib2.vis.editor._state).

No PySide6 / display required — EditorState is pure Python.

Tests cover:
- Initial coordinate loading (integer grid)
- move_point() updates coordinates correctly
- OT-change detection: same OT after small move, different OT after large move
- Hull indices correct for convex position
- lock_current_ot() resets the reference
- on_ot_changed callback fires exactly when OT flips
- collinearity detection
- would_change_ot() query
- property computation (hull, crossings, kgons, empty kgons)
"""

import math
import pytest

from pyotlib2.vis.editor._state import EditorState, GRID_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def circle_points(n: int, r: int = None) -> list[tuple[int, int]]:
    """n points evenly spaced on a circle, centered in the grid."""
    if r is None:
        r = int(GRID_MAX * 0.38)
    cx = cy = GRID_MAX // 2
    return [
        (int(cx + r * math.cos(2 * math.pi * i / n - math.pi / 2)),
         int(cy + r * math.sin(2 * math.pi * i / n - math.pi / 2)))
        for i in range(n)
    ]


def one_interior() -> list[tuple[int, int]]:
    """5 points: square hull + 1 off-center interior (no collinearities)."""
    cx = cy = GRID_MAX // 2
    r = int(GRID_MAX * 0.25)
    hull = [
        (cx - r, cy - r),
        (cx + r, cy - r),
        (cx + r, cy + r),
        (cx - r, cy + r),
    ]
    # slightly off-center to avoid collinearities with any hull pair
    interior = (cx + 170, cy + 230)
    return hull + [interior]


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestEditorStateInit:
    def test_n(self):
        state = EditorState(circle_points(6))
        assert state.n == 6

    def test_get_coords_match_input(self):
        pts = circle_points(5)
        state = EditorState(pts)
        for (x0, y0), (x1, y1) in zip(pts, state.get_coords()):
            assert x0 == x1
            assert y0 == y1

    def test_ot_not_changed_initially(self):
        state = EditorState(circle_points(6))
        assert not state.ot_changed

    def test_no_collinear_initially(self):
        state = EditorState(circle_points(6))
        assert not state.has_collinear


# ---------------------------------------------------------------------------
# move_point: coordinate updates
# ---------------------------------------------------------------------------

class TestMovePoint:
    def test_move_updates_coord(self):
        pts = circle_points(6)
        state = EditorState(pts)
        state.move_point(0, 100, 200)
        coords = state.get_coords()
        assert coords[0] == (100, 200)

    def test_other_coords_unchanged(self):
        pts = circle_points(6)
        state = EditorState(pts)
        state.move_point(2, 100, 200)
        coords = state.get_coords()
        for i, (x0, y0) in enumerate(pts):
            if i == 2:
                continue
            assert coords[i] == (x0, y0)

    def test_move_out_of_range_raises(self):
        state = EditorState(circle_points(4))
        with pytest.raises(IndexError):
            state.move_point(4, 0, 0)
        with pytest.raises(IndexError):
            state.move_point(-1, 0, 0)

    def test_multiple_moves(self):
        pts = circle_points(5)
        state = EditorState(pts)
        for i in range(5):
            state.move_point(i, i * 100 + 500, i * 50 + 500)
        coords = state.get_coords()
        for i in range(5):
            assert coords[i] == (i * 100 + 500, i * 50 + 500)

    def test_float_input_rounded(self):
        """move_point accepts floats and rounds to nearest int."""
        state = EditorState(circle_points(4))
        state.move_point(0, 1234.7, 5678.3)
        x, y = state.get_coords()[0]
        assert x == 1235
        assert y == 5678


# ---------------------------------------------------------------------------
# OT change detection
# ---------------------------------------------------------------------------

class TestOTChangeDetection:
    def test_tiny_move_same_ot(self):
        """Moving a hull point by 1 unit should not change OT."""
        pts = circle_points(6)
        state = EditorState(pts)
        x0, y0 = pts[0]
        state.move_point(0, x0 + 1, y0 + 1)
        assert not state.ot_changed

    def test_move_interior_outside_hull_changes_ot(self):
        """Moving interior point far outside hull changes OT."""
        pts = one_interior()
        state = EditorState(pts)
        # move interior point (index 4) to far right
        state.move_point(4, GRID_MAX - 100, GRID_MAX // 2)
        assert state.ot_changed

    def test_move_back_restores_ot(self):
        """Moving back to the original position restores OT."""
        pts = one_interior()
        state = EditorState(pts)
        x0, y0 = pts[4]
        state.move_point(4, GRID_MAX - 100, GRID_MAX // 2)
        assert state.ot_changed
        state.move_point(4, x0, y0)
        assert not state.ot_changed


# ---------------------------------------------------------------------------
# OT changed callback
# ---------------------------------------------------------------------------

class TestOTCallback:
    def test_callback_fires_on_change(self):
        events = []
        pts = one_interior()
        state = EditorState(pts, on_ot_changed=lambda c: events.append(c))
        state.move_point(4, GRID_MAX - 100, GRID_MAX // 2)
        assert len(events) >= 1
        assert events[-1] is True

    def test_callback_fires_on_restore(self):
        events = []
        pts = one_interior()
        state = EditorState(pts, on_ot_changed=lambda c: events.append(c))
        x0, y0 = pts[4]
        state.move_point(4, GRID_MAX - 100, GRID_MAX // 2)
        state.move_point(4, x0, y0)
        assert events[-1] is False

    def test_callback_not_fired_on_same_ot(self):
        events = []
        pts = circle_points(6)
        state = EditorState(pts, on_ot_changed=lambda c: events.append(c))
        x0, y0 = pts[0]
        state.move_point(0, x0 + 1, y0 + 1)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# would_change_ot
# ---------------------------------------------------------------------------

class TestWouldChangeOT:
    def test_tiny_move_no_change(self):
        pts = circle_points(6)
        state = EditorState(pts)
        x0, y0 = pts[0]
        assert not state.would_change_ot(0, x0 + 1, y0 + 1)

    def test_large_move_changes(self):
        pts = one_interior()
        state = EditorState(pts)
        assert state.would_change_ot(4, GRID_MAX - 100, GRID_MAX // 2)

    def test_does_not_modify_state(self):
        pts = one_interior()
        state = EditorState(pts)
        _ = state.would_change_ot(4, GRID_MAX - 100, GRID_MAX // 2)
        assert state.get_coords()[4] == pts[4]
        assert not state.ot_changed


# ---------------------------------------------------------------------------
# Collinearity detection
# ---------------------------------------------------------------------------

class TestCollinearity:
    def test_no_collinear_convex(self):
        state = EditorState(circle_points(6))
        assert not state.has_collinear
        assert state.collinear_triples == []

    def test_collinear_detected(self):
        """Three points on a line → collinear detected."""
        pts = [(1000, 1000), (5000, 5000), (9000, 9000),
               (1000, 9000), (9000, 1000)]
        state = EditorState(pts)
        assert state.has_collinear
        # (0,1,2) should be collinear
        triples = state.collinear_triples
        assert any(set(t) == {0, 1, 2} for t in triples)

    def test_collinear_callback(self):
        events = []
        # start with non-collinear config
        pts = [(1000, 1000), (5000, 1000), (9000, 1000),
               (1000, 9000), (9000, 9000)]
        # initially collinear (pts 0,1,2 on y=1000)
        state = EditorState(pts, on_collinear_changed=lambda t: events.append(t))
        # move pt 1 off the line → collinear → non-collinear: callback fires
        state.move_point(1, 5000, 5000)
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Hull indices
# ---------------------------------------------------------------------------

class TestHullIndices:
    def test_convex_all_on_hull(self):
        state = EditorState(circle_points(6))
        hull = state.get_hull_indices()
        assert set(hull) == set(range(6))

    def test_one_interior_not_on_hull(self):
        state = EditorState(one_interior())
        hull = state.get_hull_indices()
        assert 4 not in hull
        assert len(hull) == 4

    def test_hull_after_move(self):
        state = EditorState(one_interior())
        state.move_point(4, GRID_MAX - 100, GRID_MAX // 2)
        hull = state.get_hull_indices()
        assert 4 in hull


# ---------------------------------------------------------------------------
# lock_current_ot
# ---------------------------------------------------------------------------

class TestLockCurrentOT:
    def test_lock_resets_changed(self):
        pts = one_interior()
        state = EditorState(pts)
        state.move_point(4, GRID_MAX - 100, GRID_MAX // 2)
        assert state.ot_changed
        state.lock_current_ot()
        assert not state.ot_changed

    def test_lock_fires_callback_if_was_changed(self):
        events = []
        pts = one_interior()
        state = EditorState(pts, on_ot_changed=lambda c: events.append(c))
        state.move_point(4, GRID_MAX - 100, GRID_MAX // 2)
        events.clear()
        state.lock_current_ot()
        assert events == [False]

    def test_after_lock_old_position_changes_ot(self):
        pts = one_interior()
        state = EditorState(pts)
        x0, y0 = pts[4]
        state.move_point(4, GRID_MAX - 100, GRID_MAX // 2)
        state.lock_current_ot()
        state.move_point(4, x0, y0)
        assert state.ot_changed


# ---------------------------------------------------------------------------
# get_small_lambda
# ---------------------------------------------------------------------------

class TestGetSmallLambda:
    def test_returns_small_lambda(self):
        from pyotlib2.core.small_lambda import SmallLambda
        state = EditorState(circle_points(5))
        sl = state.get_small_lambda()
        assert sl is not None
        assert isinstance(sl, SmallLambda)
        assert sl.n == 5

    def test_returns_none_when_collinear(self):
        pts = [(1000, 1000), (5000, 5000), (9000, 9000),
               (1000, 9000), (9000, 1000)]
        state = EditorState(pts)
        # collinear config → None
        assert state.get_small_lambda() is None


# ---------------------------------------------------------------------------
# Property computation
# ---------------------------------------------------------------------------

class TestPropertyComputation:
    """Verify property values for convex n-gons (all values are C(n,k))."""

    def _state(self, n):
        return EditorState(circle_points(n))

    def test_hull_convex(self):
        for n in [4, 5, 6]:
            val, _ = self._state(n).compute_property("hull")
            assert val == n, f"n={n}: hull={val}"

    def test_crossings_convex_hexagon(self):
        # C(6,4) = 15 crossings for convex hexagon
        val, _ = self._state(6).compute_property("crossings")
        assert val == 15

    def test_kgons_5_convex_hexagon(self):
        # C(6,5) = 6
        val, _ = self._state(6).compute_property("kgons-5")
        assert val == 6

    def test_kgons_6_convex_hexagon(self):
        # C(6,6) = 1
        val, _ = self._state(6).compute_property("kgons-6")
        assert val == 1

    def test_empty_3holes_convex_hexagon(self):
        # C(6,3) = 20
        val, _ = self._state(6).compute_property("empty-kgons-3")
        assert val == 20

    def test_empty_4holes_convex_hexagon(self):
        # C(6,4) = 15
        val, _ = self._state(6).compute_property("empty-kgons-4")
        assert val == 15

    def test_empty_5holes_convex_hexagon(self):
        # C(6,5) = 6
        val, _ = self._state(6).compute_property("empty-kgons-5")
        assert val == 6

    def test_empty_6holes_convex_hexagon(self):
        # C(6,6) = 1
        val, _ = self._state(6).compute_property("empty-kgons-6")
        assert val == 1

    def test_returns_dash_when_collinear(self):
        pts = [(1000, 1000), (5000, 5000), (9000, 9000),
               (1000, 9000), (9000, 1000)]
        state = EditorState(pts)
        val, _ = state.compute_property("crossings")
        assert val == "—"

    def test_elapsed_is_float(self):
        val, elapsed = self._state(5).compute_property("hull")
        assert isinstance(elapsed, float)
        assert elapsed >= 0.0


# ---------------------------------------------------------------------------
# Onion layers
# ---------------------------------------------------------------------------

def nested_triangles() -> list[tuple[int, int]]:
    """Three nested triangles → 3 convex layers."""
    return [(0, 0), (9000, 500), (4000, 9000),          # outer
            (3000, 2500), (6000, 2800), (4300, 6000),   # middle
            (4100, 3500), (4900, 3600), (4400, 4600)]   # inner


class TestOnionLayers:
    def test_count_nested_triangles(self):
        val, _ = EditorState(nested_triangles()).compute_property("onion-layers")
        assert val == 3

    def test_count_convex(self):
        val, _ = EditorState(circle_points(6)).compute_property("onion-layers")
        assert val == 1

    def test_count_one_interior(self):
        val, _ = EditorState(one_interior()).compute_property("onion-layers")
        assert val == 2

    def test_layers_partition_points(self):
        layers = EditorState(nested_triangles()).get_onion_layers()
        assert [sorted(l) for l in layers] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


# ---------------------------------------------------------------------------
# Caching: recompute only when the order type changes
# ---------------------------------------------------------------------------

class TestCaching:
    def test_same_ot_move_keeps_version_and_cache(self):
        pts = one_interior()
        state = EditorState(pts)
        res = state.compute_property("crossings")
        v = state.ot_version
        x, y = pts[4]
        state.move_point(4, x + 5, y - 5)
        assert state.ot_version == v
        assert state.compute_property("crossings") is res   # cached object

    def test_ot_change_bumps_version_and_recomputes(self):
        state = EditorState(one_interior())
        hull_before, _ = state.compute_property("hull")
        v = state.ot_version
        state.move_point(4, GRID_MAX - 100, GRID_MAX // 2)
        assert state.ot_version != v
        hull_after, _ = state.compute_property("hull")
        assert (hull_before, hull_after) == (4, 5)

    def test_matches_full_recompute_after_many_moves(self):
        """Incremental signs must equal a fresh state after arbitrary moves."""
        import random
        rng = random.Random(1)
        pts = circle_points(7)
        state = EditorState(pts)
        for _ in range(60):
            i = rng.randrange(7)
            state.move_point(i, rng.randrange(GRID_MAX), rng.randrange(GRID_MAX))
            fresh = EditorState(state.get_coords())
            assert state._signs == fresh._signs
            assert state.collinear_triples == fresh.collinear_triples

    def test_set_coords(self):
        state = EditorState(circle_points(4))
        new = [(0, 0), (10, 0), (10, 10), (0, 10)]
        state.set_coords(new)
        assert state.get_coords() == new
        with pytest.raises(ValueError):
            state.set_coords(new[:3])

    def test_would_change_ot_refuses_collinear(self):
        pts = [(0, 0), (10, 0), (0, 10), (7, 7)]
        state = EditorState(pts)
        # moving point 3 onto the segment (10,0)-(0,10) makes it collinear
        assert state.would_change_ot(3, 5, 5)

    def test_l_matrix_cached_and_none_when_collinear(self):
        state = EditorState(circle_points(5))
        assert state.get_l_matrix() is not None
        state.move_point(2, *state.get_coords()[0])   # duplicate point
        assert state.get_l_matrix() is None


# ---------------------------------------------------------------------------
# CLI property counter used by the Property Min/Max walk
# ---------------------------------------------------------------------------

class TestCountProperty:
    @pytest.mark.parametrize("key", [k for _, k in __import__(
        "pyotlib2.vis.editor._state", fromlist=["ALL_PROPERTIES"]).ALL_PROPERTIES])
    def test_all_editor_properties_supported(self, key):
        """Every property offered in the editor's walk must be countable."""
        from pyotlib2.cli.commands import _count_property
        sl = EditorState(nested_triangles()).get_small_lambda()
        val, _ = EditorState(nested_triangles()).compute_property(key)
        assert _count_property(sl, key) == val

    def test_unknown_raises(self):
        from pyotlib2.cli.commands import _count_property
        sl = EditorState(circle_points(5)).get_small_lambda()
        with pytest.raises(ValueError):
            _count_property(sl, "no-such-property")

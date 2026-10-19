"""EditorState: pure-Python logic for the interactive editor.

No PySide6 imports — fully testable without a display.

Coordinate system
-----------------
Points are stored as the original integer (x, y) coordinates of the loaded
point set (no rescaling, no flipping).  Y axis: math convention (y
increases upward).

The GUI maps these model coordinates to the screen (y flip, zoom, fit) in
its scene/view transform only.  No float rounding is ever applied to the
stored coordinates — OT is computed directly from integer coords, so
loading a file and displaying it in the editor can never corrupt the order
type.

If the user drags a point, the new position is rounded to the nearest
integer before being stored.

Caching
-------
The orientation sign of every triple is kept.  A move only re-evaluates
the triples containing the moved point; if none of their signs changed the
order type is unchanged and all cached combinatorial results (SmallLambda,
properties, hull, onion layers) stay valid.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Callable, Optional

GRID_MAX = 10000   # extent of the default point set [0, GRID_MAX]²


def _orient(p, q, r) -> int:
    """Sign of det([q-p, r-p]): +1 (CCW), -1 (CW), 0 (collinear). Exact for ints."""
    d = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (d > 0) - (d < 0)


class EditorState:
    """Mutable editor state, independent of the GUI.

    Parameters
    ----------
    pts :
        Initial point positions as integer (x, y) pairs (model coordinates).
    on_ot_changed :
        Callback(changed: bool) fired when OT status flips.
    on_collinear_changed :
        Callback(collinear_triples: list[tuple[int,int,int]]) fired when
        collinearity status changes.
    """

    def __init__(
        self,
        pts: list[tuple[int, int]],
        on_ot_changed: Optional[Callable[[bool], None]] = None,
        on_collinear_changed: Optional[Callable[[list], None]] = None,
    ):
        self._coords: list[list[int]] = [[int(x), int(y)] for x, y in pts]
        self._on_ot_changed = on_ot_changed or (lambda _: None)
        self._on_collinear_changed = on_collinear_changed or (lambda _: None)

        # orientation sign of every triple a < b < c
        self._signs: dict[tuple[int, int, int], int] = {
            t: _orient(*(self._coords[i] for i in t))
            for t in combinations(range(self.n), 3)
        }
        self._ot_version = 0          # bumped whenever any sign changes
        self._cache: dict = {}        # combinatorial results, valid per version

        self._ref_signs: Optional[dict] = None
        self._ot_changed = False
        self._set_reference()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self._coords)

    @property
    def ot_version(self) -> int:
        """Counter that changes whenever the order type (any sign) changes."""
        return self._ot_version

    def get_coords(self) -> list[tuple[int, int]]:
        """Return current integer (x, y) positions."""
        return [(c[0], c[1]) for c in self._coords]

    def move_point(self, index: int, x: int, y: int) -> None:
        """Move point *index* to integer position (x, y) and update state."""
        if index < 0 or index >= self.n:
            raise IndexError(f"point index {index} out of range [0, {self.n})")
        self._coords[index][0] = int(round(x))
        self._coords[index][1] = int(round(y))
        new = self._signs_with(index)
        if all(self._signs[t] == s for t, s in new.items()):
            return    # order type unchanged — caches stay valid
        old_collinear = self.collinear_triples
        self._signs.update(new)
        self._ot_version += 1
        self._cache.clear()
        new_collinear = self.collinear_triples
        if new_collinear != old_collinear:
            self._on_collinear_changed(new_collinear)
        self._update_ot_changed()

    def set_coords(self, pts: list[tuple[int, int]]) -> None:
        """Replace all positions (same n), e.g. for an optimisation result."""
        if len(pts) != self.n:
            raise ValueError(f"expected {self.n} points, got {len(pts)}")
        for i, (x, y) in enumerate(pts):
            self.move_point(i, x, y)

    def get_hull_indices(self) -> list[int]:
        """Return convex hull point indices (unsorted). Empty if collinear."""
        sl = self._cached_small_lambda()
        if sl is None:
            return []
        try:
            return sl.get_extremal_points()
        except Exception:
            return []

    def get_onion_layers(self) -> list[list[int]]:
        """Return convex layers (onion peeling). Each layer is a list of indices."""
        if "onion" not in self._cache:
            sl = self._cached_small_lambda()
            try:
                self._cache["onion"] = sl.get_onion() if sl is not None else []
            except Exception:
                self._cache["onion"] = []
        return self._cache["onion"]

    @property
    def ot_changed(self) -> bool:
        return self._ot_changed

    @property
    def collinear_triples(self) -> list[tuple[int, int, int]]:
        return [t for t, s in self._signs.items() if s == 0]

    @property
    def has_collinear(self) -> bool:
        return 0 in self._signs.values()

    def would_change_ot(self, index: int, x: int, y: int) -> bool:
        """Check if moving point *index* to (x,y) would change the OT
        (relative to the reference) or make points collinear.
        Does NOT modify state.
        """
        if self._ref_signs is None:
            return True
        old = (self._coords[index][0], self._coords[index][1])
        self._coords[index][0] = int(round(x))
        self._coords[index][1] = int(round(y))
        new = self._signs_with(index)
        self._coords[index][0] = old[0]
        self._coords[index][1] = old[1]
        if 0 in new.values():
            return True
        for t, s in self._signs.items():
            if self._ref_signs[t] != new.get(t, s):
                return True
        return False

    def lock_current_ot(self) -> None:
        """Set the current OT as the new reference."""
        self._set_reference()
        if self._ot_changed:
            self._ot_changed = False
            self._on_ot_changed(False)

    def get_small_lambda(self):
        """Return a fresh SmallLambda (with the current coordinates as
        realization), or None if collinear/invalid."""
        if self.has_collinear:
            return None
        return self._build_small_lambda()

    def get_l_matrix(self):
        """Return the (cached) SmallLambda rank matrix, or None if collinear."""
        sl = self._cached_small_lambda()
        return None if sl is None else sl.get_l()

    def compute_property(self, name: str) -> tuple:
        """Compute a named property. Returns (value, elapsed_seconds).

        value is an int or '—' on failure.  Results are cached until the
        order type changes; elapsed is the time of the actual computation.
        """
        key = ("prop", name)
        if key not in self._cache:
            sl = self._cached_small_lambda()
            if sl is None:
                return ("—", 0.0)
            t0 = time.perf_counter()
            try:
                val = _compute_prop(sl, name)
            except Exception:
                val = "err"
            self._cache[key] = (val, time.perf_counter() - t0)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _signs_with(self, index: int) -> dict[tuple[int, int, int], int]:
        """Current orientation signs of all triples containing *index*."""
        c = self._coords
        others = [j for j in range(self.n) if j != index]
        result = {}
        for a, b in combinations(others, 2):
            t = tuple(sorted((index, a, b)))
            result[t] = _orient(c[t[0]], c[t[1]], c[t[2]])
        return result

    def _set_reference(self) -> None:
        self._ref_signs = None if self.has_collinear else dict(self._signs)

    def _update_ot_changed(self) -> None:
        if self._ref_signs is None or self.has_collinear:
            changed = False
        else:
            changed = self._signs != self._ref_signs
        if changed != self._ot_changed:
            self._ot_changed = changed
            self._on_ot_changed(changed)

    def _build_small_lambda(self):
        try:
            from pyotlib2.core.point_set import PointSet
            ps = PointSet(self.n, self.get_coords())
            return ps.to_small_lambda(lazy=False)
        except Exception:
            return None

    def _cached_small_lambda(self):
        """SmallLambda for combinatorial queries, cached per order type.
        Its realization may be stale — do not use it for coordinates."""
        if self.has_collinear:
            return None
        if "sl" not in self._cache:
            self._cache["sl"] = self._build_small_lambda()
        return self._cache["sl"]


# ------------------------------------------------------------------
# Property registry
# ------------------------------------------------------------------

# Each entry: (display_name, internal_key)
ALL_PROPERTIES: list[tuple[str, str]] = [
    ("Hull size",       "hull"),
    ("Crossings",       "crossings"),
    ("5-gons",          "kgons-5"),
    ("6-gons",          "kgons-6"),
    ("7-gons",          "kgons-7"),
    ("Empty 3-holes",   "empty-kgons-3"),
    ("Empty 4-holes",   "empty-kgons-4"),
    ("Empty 5-holes",   "empty-kgons-5"),
    ("Empty 6-holes",   "empty-kgons-6"),
    ("Empty 7-holes",   "empty-kgons-7"),
    ("Onion layers",    "onion-layers"),
    ("Triangulations",  "triangulations"),
]

DEFAULT_PROPERTIES: list[str] = [
    "hull", "crossings", "kgons-5", "kgons-6", "kgons-7",
    "empty-kgons-3", "empty-kgons-4", "empty-kgons-5", "empty-kgons-6",
]


def _compute_prop(sl, key: str):
    """Compute a single property from a SmallLambda."""
    if key == "hull":
        return len(sl.get_extremal_points())
    if key == "onion-layers":
        return len(sl.get_onion())
    if key == "crossings":
        from pyotlib2.algorithms.polygon_count import count_crossings
        return count_crossings(sl)
    if key.startswith("kgons-"):
        k = int(key.split("-")[1])
        from pyotlib2.algorithms.polygon_count import count_polygons
        bl = sl.to_big_lambda()
        # all convex k-gons (empty_only=True would count k-holes and needs
        # a natural labeling)
        return count_polygons(bl, k, empty_only=False)
    if key.startswith("empty-kgons-"):
        k = int(key.split("-")[2])
        from pyotlib2.algorithms.mrsw import count_empty_kgons_mrsw
        return count_empty_kgons_mrsw(sl, k)
    if key == "triangulations":
        from pyotlib2.algorithms.triangulations import count_triangulations
        return count_triangulations(sl)
    raise ValueError(f"unknown property key: {key!r}")

"""Projective equivalence classes of order types.

Two order types are in the same projective class if one can be obtained from
the other by a sequence of reflections of extremal points (flips on the
convex hull).  We enumerate the class via BFS on the flip graph.

Projective classes correspond exactly to isomorphism classes of
non-degenerate rank-3 oriented matroids (abstract order types in the
projective sense).  Catalogue: https://finschi.com/math/om/?p=catom&filter=nondeg

Known counts (all realizable except where noted):
  n=5 →     1
  n=6 →     4
  n=7 →    11
  n=8 →   135
  n=9 →  4381  (4380 realizable + 1 non-realizable)
"""

from __future__ import annotations
from collections import deque
from functools import cached_property
from typing import Iterator, Optional

from pyotlib2.core.small_lambda import SmallLambda


class ProjectiveClass:
    """Manages the projective equivalence class of an order type."""

    def __init__(self, ot: SmallLambda, calc_lex_min: bool = True):
        self._start = ot
        # str → (SmallLambda, flip_vec, labeling, mirror) recording, for each
        # class member ``m`` reachable from the lex-min ``start``, a transform::
        #
        #     m == start.from_flips(flip_vec).relabeled(labeling)[.mirrored() if mirror]
        #
        # The ``mirror`` bit is the CUMULATIVE mirror parity: each BFS step's
        # ``get_lex_min`` may itself mirror to reach the canonical form, and that
        # parity has to be carried along or the recorded transform is off by a
        # global orientation reversal.  ``pc_tester`` inverts this exact chain to
        # transport a realization of ``m`` back onto ``start`` (and thence ``ot``).
        self._ot_dict: dict[str, tuple] = {}
        self._enumerate(only_find_better=(not calc_lex_min))

    def _enumerate(self, only_find_better: bool = False) -> None:
        # Normalise start to lex-min; use lex-min string as BFS key so that
        # different labelings of the same OT are recognised as identical.
        start = self._start.get_lex_min()
        start_key = start.to_string()
        queue: deque = deque()
        queue.append((start, [0] * start.n, list(range(start.n)), False))
        self._ot_dict[start_key] = (start, [0] * start.n, list(range(start.n)), False)

        while queue:
            ot, flip_vec, labeling, mirror = queue.popleft()
            for p in ot.get_extremal_points():
                flipped = ot.flip_point(p)
                flipped_lm, lab2, mir2 = flipped.get_lex_min(return_labeling=True)
                key = flipped_lm.to_string()
                if key not in self._ot_dict:
                    new_flip = list(flip_vec)
                    new_flip[labeling[p]] ^= 1
                    new_labeling = [labeling[lab2[i]] for i in range(ot.n)]
                    new_mirror = mirror ^ mir2
                    self._ot_dict[key] = (flipped_lm, new_flip, new_labeling, new_mirror)
                    queue.append((flipped_lm, new_flip, new_labeling, new_mirror))
                    if only_find_better and flipped_lm.compare(self._start) < 0:
                        return

    @cached_property
    def representer(self) -> SmallLambda:
        """Lexicographically minimal OT in this projective class (cached)."""
        return min(ot for ot, _, _, _ in self._ot_dict.values())

    def get_representer(self) -> SmallLambda:
        return self.representer

    def is_representer(self) -> bool:
        return self._start.compare(self.representer) == 0

    def all_ots(self) -> list:
        return [ot for ot, _, _, _ in self._ot_dict.values()]

    def size(self) -> int:
        return len(self._ot_dict)

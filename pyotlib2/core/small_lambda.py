"""SmallLambda: rank-matrix representation of an order type.

l[i, j] = number of points p_k (k ≠ i,j) that lie strictly to the left of
the directed line from p_i to p_j.

Invariant: l[i, j] + l[j, i] == n - 2  for all i ≠ j.
"""

from __future__ import annotations
import hashlib
import itertools
from functools import cached_property
from typing import Optional, Iterator

import numpy as np

from pyotlib2.core.utils import concat_perms


class InvalidData(Exception):
    pass


class SmallLambda:
    """Rank-matrix representation of an abstract order type on n points."""

    def __init__(
        self,
        n: int,
        l: Optional[np.ndarray],
        realization: Optional[list] = None,
        coloring: Optional[dict] = None,
    ):
        assert l is not None or realization is not None, "need l or realization"
        self.n = n
        self.l = l  # np.int32, shape (n, n), or None (lazy)
        self.realization = realization
        self.coloring = coloring
        # Warm start for realization: a point set realizing the (n-1)-point
        # sub-order-type on the first n-1 points.  Consumed by continuous-
        # optimizer testers (scipy, octave) as an initial guess so only the
        # last point still violates the constraints.  Not part of the abstract
        # order type; set by the etherealization driver.
        self.subrealization: Optional[list] = None

    # ------------------------------------------------------------------
    # Lazy matrix access
    # ------------------------------------------------------------------

    def get_l(self) -> np.ndarray:
        """Return (possibly lazily computed) rank matrix."""
        if self.l is None:
            assert self.realization is not None
            self.l = self.to_point_set().to_small_lambda(lazy=False).l
            # invalidate cached key now that l is available
            self.__dict__.pop("_key", None)
        return self.l

    @cached_property
    def _key(self) -> bytes:
        """Flat bytes representation of l for fast lexicographic comparison and hashing."""
        return self.get_l().astype(np.int32).tobytes()

    @cached_property
    def lex_min(self) -> "SmallLambda":
        """Lexicographically minimal representative (cached)."""
        return self.get_lex_min()

    @cached_property
    def extremal_points(self) -> list:
        """Convex hull point indices (cached)."""
        return self.get_extremal_points()

    @cached_property
    def big_lambda(self) -> "BigLambda":
        """BigLambda representation (cached)."""
        return self.to_big_lambda()

    @cached_property
    def onion(self) -> list:
        """Convex layers / onion peeling (cached)."""
        return self.big_lambda.get_onion()

    @cached_property
    def rotation_system(self) -> list:
        """Rotation system (cached)."""
        return self.big_lambda.get_rotation_system()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        if self.realization:
            return not self.to_point_set().has_collinear_points()
        return self._is_valid_abstract()

    def _is_valid_abstract(self) -> bool:
        l = self.get_l()
        # check l[i,j] + l[j,i] == n-2 for all i != j, and all off-diagonal entries >= 0
        sym = l + l.T
        mask = ~np.eye(self.n, dtype=bool)
        return bool(np.all(sym[mask] == self.n - 2) and np.all(l[mask] >= 0))

    # ------------------------------------------------------------------
    # Identity / serialisation
    # ------------------------------------------------------------------

    # Three levels of identity, from finest to coarsest equivalence:
    #   SL-hash : this exact rank matrix (labeling-dependent)
    #   OT-hash : the order type, i.e. lex-min over all relabelings + mirror
    #   PC-hash : the projective class (lex-min over the flip graph)
    # Each is the SHA-256 of the corresponding canonical ``to_string()``.

    @staticmethod
    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def sl_hash(self) -> str:
        """SHA-256 of *this* SmallLambda (rank matrix ``to_string()``).

        Labeling-DEPENDENT: relabeling the points of the underlying point set
        yields a different SmallLambda and hence a different SL-hash, even though
        the order type is unchanged.  Use to identify a concrete labeled
        rank matrix.
        """
        return self._sha(self.to_string())

    def ot_hash(self) -> str:
        """SHA-256 of the ORDER TYPE (lex-min over all relabelings + mirror).

        Labeling-INDEPENDENT: every relabeling/mirror of the same order type
        maps to the same lex-min representative, so the OT-hash identifies the
        abstract order type itself.  This is the identifier to use when the
        labeling should not matter.
        """
        return self._sha(self.get_lex_min().to_string())

    def pc_hash(self) -> str:
        """SHA-256 of the PROJECTIVE CLASS representative.

        Identifies the order type up to projective equivalence (reflections of
        convex-hull points).  Two order types with the same PC-hash lie in the
        same projective class.  More expensive (enumerates the flip graph).
        """
        from pyotlib2.algorithms.projective_class import ProjectiveClass
        return self._sha(ProjectiveClass(self).representer.to_string())

    # 6 hex digits is plenty for a human to eyeball "same vs. different" — the
    # SL matrix itself is visually hard to compare, so a short tag is the point.
    def short_sl_hash(self, length: int = 6) -> str:
        return self.sl_hash()[:length]

    def short_ot_hash(self, length: int = 6) -> str:
        return self.ot_hash()[:length]

    def short_pc_hash(self, length: int = 6) -> str:
        return self.pc_hash()[:length]

    def get_id(self) -> str:
        # Deprecated: kept for backward compatibility; prefer sl_hash() (SHA-256).
        return hashlib.md5(self.to_string().encode()).hexdigest()

    def to_string(self) -> str:
        l = self.get_l()
        lines = []
        for i in range(self.n):
            row = " ".join(f"{int(l[i, j]):2d}" for j in range(self.n))
            if self.coloring:
                row += " " + self.coloring[i]
            lines.append(row)
        return "\n".join(lines) + "\n"

    @staticmethod
    def from_string(n: int, string: str) -> "SmallLambda":
        l = np.zeros((n, n), dtype=np.int32)
        lines = string.rstrip("\n").split("\n")
        coloring = None
        assert len(lines) == n
        for i, line in enumerate(lines):
            parts = line.split()
            assert len(parts) in (n, n + 1)
            if len(parts) == n + 1:
                if coloring is None:
                    coloring = {}
                coloring[i] = parts[n]
            for j in range(n):
                l[i, j] = int(parts[j])
        return SmallLambda(n, l, coloring=coloring)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_point_set(self) -> "PointSet":
        from pyotlib2.core.point_set import PointSet
        assert self.realization is not None
        return PointSet(self.n, self.realization, coloring=self.coloring)

    def to_big_lambda(self) -> "BigLambda":
        """Reconstruct the full orientation array from the rank matrix.

        Uses the natural labeling starting from an extremal point p0.
        Points are processed in natural order: at step k_idx, the point pk
        is still on the convex hull of the remaining points, so its
        orientation with any pair (pi, pj) can be read unambiguously from
        the current rank matrix.  After determining o(pi, pj, pk) for all
        pairs, we "remove" pk by decrementing l[pi, pj] or l[pj, pi]
        depending on which side of the directed line pi→pj the point pk lies.
        """
        from pyotlib2.core.big_lambda import BigLambda

        l = self.get_l()
        p0 = self.get_extremal_points()[0]
        lab = self.get_natural_labeling(p0)

        o = np.zeros((self.n, self.n, self.n), dtype=np.int8)
        # working copy: will be modified as points are successively removed
        tmpl = l.copy()

        for k_idx in range(self.n):
            pk = lab[k_idx]
            # determine o(pi, pj, pk) for all pairs (pi, pj) still remaining
            for i_idx in range(k_idx + 1, self.n):
                pi = lab[i_idx]
                for j_idx in range(i_idx + 1, self.n):
                    pj = lab[j_idx]
                    val = SmallLambda._orientation_from_matrix(tmpl, pi, pj, pk)
                    # remove pk: adjust rank counts for all remaining pairs through pk
                    if val == 1:   # pk is left of pi→pj  →  l[pi,pj] loses one point
                        tmpl[pi, pj] -= 1
                    elif val == -1:  # pk is right of pi→pj  →  l[pj,pi] loses one point
                        tmpl[pj, pi] -= 1
                    o[pi, pj, pk] = o[pj, pk, pi] = o[pk, pi, pj] = val
                    o[pi, pk, pj] = o[pj, pi, pk] = o[pk, pj, pi] = -val

        return BigLambda(self.n, o, realization=self.realization, coloring=self.coloring)

    # ------------------------------------------------------------------
    # Orientation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _orientation_from_matrix(matrix: np.ndarray, i: int, j: int, k: int) -> int:
        """Derive o(i,j,k) from the rank matrix assuming k is extremal."""
        if k == i or i == j or j == k:
            return 0
        lik = int(matrix[i, k])
        ljk = int(matrix[j, k])
        if lik < ljk:
            return 1
        if lik > ljk:
            return -1
        raise InvalidData(f"ambiguous orientation for ({i},{j},{k})")

    # ------------------------------------------------------------------
    # Labeling
    # ------------------------------------------------------------------

    def has_natural_labeling(self) -> bool:
        l = self.get_l()
        return all(l[0, j] != j - 2 for j in range(2, self.n))

    def is_extremal_point(self, i: int) -> bool:
        l = self.get_l()
        return any(int(l[i, j]) in (0, self.n - 2) for j in range(self.n) if j != i)

    def get_extremal_points(self) -> list:
        """Return convex hull point indices. Use .extremal_points for cached access.

        Point i is on the convex hull iff there exists j such that l[i,j] == 0,
        meaning all other points lie strictly to the left of the directed line i→j
        (i.e. j is the next hull neighbour of i in CCW order).
        """
        l = self.get_l()
        # vectorized: row i has a zero off-diagonal entry iff i is extremal
        l_offdiag = l.copy()
        np.fill_diagonal(l_offdiag, -1)
        extremal = list(np.where(np.any(l_offdiag == 0, axis=1))[0])
        assert extremal, "no extremal points found — invalid order type?"
        return extremal

    def get_natural_labeling(self, p0: int) -> list:
        """Return labeling with p0 first, remaining sorted by angle around p0."""
        l = self.get_l()
        labeling = [None] * self.n
        labeling[0] = p0
        for p in range(self.n):
            if p == p0:
                continue
            labeling[1 + int(l[p0, p])] = p
        return labeling

    def get_natural_labelings(self) -> Iterator[list]:
        for p0 in self.extremal_points:
            yield self.get_natural_labeling(p0)

    # ------------------------------------------------------------------
    # Symmetries
    # ------------------------------------------------------------------

    def relabeled(self, labeling: list) -> "SmallLambda":
        l = self.get_l()
        idx = np.array(labeling, dtype=np.intp)
        new_l = l[np.ix_(idx, idx)].copy().astype(np.int32)
        R = [self.realization[labeling[i]] for i in range(self.n)] if self.realization else None
        col = {i: self.coloring[labeling[i]] for i in range(self.n)} if self.coloring else None
        return SmallLambda(self.n, new_l, realization=R, coloring=col)

    def mirrored(self) -> "SmallLambda":
        l = self.get_l()
        new_l = l.T.copy().astype(np.int32)
        R = [(y, x) for (x, y) in self.realization] if self.realization else None
        return SmallLambda(self.n, new_l, realization=R, coloring=self.coloring)

    def flip_point(self, k: int) -> "SmallLambda":
        """Return new SmallLambda after reflecting extremal point k."""
        assert self.is_extremal_point(k)
        l = self.get_l()
        n = self.n
        new_l = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i):
                lij = int(l[i, j])
                if i == k or j == k:
                    new_l[i, j], new_l[j, i] = (n - 2) - lij, lij
                else:
                    o = SmallLambda._orientation_from_matrix(l, i, j, k)
                    if o == 1:
                        new_l[i, j], new_l[j, i] = lij - 1, (n - 1) - lij
                    elif o == -1:
                        new_l[i, j], new_l[j, i] = lij + 1, (n - 3) - lij
        return SmallLambda(n, new_l)

    # ------------------------------------------------------------------
    # Canonical form
    # ------------------------------------------------------------------

    def _compare_labelings(
        self,
        m1: bool, lab1: list,
        m2: bool, lab2: list,
    ) -> int:
        l = self.get_l()
        for i in range(self.n):
            for j in range(self.n):
                a = int(l[lab1[j], lab1[i]]) if m1 else int(l[lab1[i], lab1[j]])
                b = int(l[lab2[j], lab2[i]]) if m2 else int(l[lab2[i], lab2[j]])
                if a < b:
                    return -1
                if a > b:
                    return 1
        return 0

    def get_lex_min(self, return_labeling: bool = False):
        """Return lexicographically minimal representative under natural labelings + mirror."""
        best_lab = None
        best_mir = False

        for lab in self.get_natural_labelings():
            if None in lab:
                # degenerate: rank collision means p0 is not a valid labeling start
                continue
            rlab = [lab[(self.n - i) % self.n] for i in range(self.n)]
            if best_lab is None or self._compare_labelings(False, lab, best_mir, best_lab) < 0:
                best_lab, best_mir = lab, False
            if self._compare_labelings(True, rlab, best_mir, best_lab) < 0:
                best_lab, best_mir = rlab, True

        assert best_lab is not None, "no valid natural labeling found — degenerate order type?"
        result = self.relabeled(best_lab)
        if best_mir:
            result = result.mirrored()
        return (result, best_lab, best_mir) if return_labeling else result

    # ------------------------------------------------------------------
    # Sub-order types
    # ------------------------------------------------------------------

    def select_points(self, indices: list) -> "SmallLambda":
        """Return the sub-order-type induced by the given point indices.

        Keeps the given labelling (no lex-min), which is what the
        etherealization driver needs when it grows a point set one point at a
        time.  Carries the realization along if present.
        """
        return self.to_big_lambda().select_points(indices)

    def reduce(
        self,
        k: int,
        lex_min: bool = True,
        randomized: bool = False,
        max_count: Optional[int] = None,
    ) -> Iterator["SmallLambda"]:
        """Yield all k-point sub-order types."""
        if self.realization and not randomized:
            yield from self._reduce_real(k, lex_min)
        else:
            yield from self._reduce_abstract(k, lex_min, randomized, max_count)

    def _reduce_real(self, k: int, lex_min: bool) -> Iterator["SmallLambda"]:
        for sel in itertools.combinations(range(self.n), k):
            pts = [self.realization[sel[i]] for i in range(k)]
            col = {i: self.coloring[sel[i]] for i in range(k)} if self.coloring else None
            sl = SmallLambda(k, None, realization=pts, coloring=col)
            yield sl.get_lex_min() if lex_min else sl

    def _reduce_abstract(
        self, k: int, lex_min: bool, randomized: bool, max_count: Optional[int]
    ) -> Iterator["SmallLambda"]:
        import random

        bl = self.to_big_lambda()
        cnt = 0
        perms = (
            itertools.combinations(range(self.n), k)
            if not randomized
            else (tuple(random.sample(range(self.n), k)) for _ in itertools.count())
        )
        for perm in perms:
            if max_count is not None and cnt >= max_count:
                break
            cnt += 1
            sub = bl.select_points(list(perm))
            if lex_min:
                sub, lab, _ = sub.get_lex_min(return_labeling=True)
                sub.note = "Indices: " + " ".join(str(concat_perms(list(perm), lab)[i]) for i in range(k))
            yield sub

    # ------------------------------------------------------------------
    # Structural properties
    # ------------------------------------------------------------------

    def get_onion(self) -> list:
        return self.big_lambda.get_onion()

    def get_rotation_system(self, signed: bool = False) -> list:
        return self.big_lambda.get_rotation_system(signed=signed)

    def from_flips(self, F: list, drop_realization: bool = False) -> "SmallLambda":
        return self.big_lambda.from_flips(F, drop_realization).to_small_lambda()

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(self, other: "SmallLambda") -> int:
        assert self.n == other.n
        k1, k2 = self._key, other._key
        if k1 < k2:
            return -1
        if k1 > k2:
            return 1
        return 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SmallLambda):
            return NotImplemented
        return self.n == other.n and self._key == other._key

    def __lt__(self, other: "SmallLambda") -> bool:
        return self.compare(other) < 0

    def __hash__(self) -> int:
        return hash(self._key)

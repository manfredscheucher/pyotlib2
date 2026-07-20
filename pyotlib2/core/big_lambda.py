"""BigLambda: chirotope / orientation-triple representation of an order type.

o[i, j, k] ∈ {-1, 0, 1} is the orientation (sign of the determinant) of
the ordered triple (p_i, p_j, p_k).  The array is antisymmetric under
transpositions of any two indices and cyclic-symmetric.
"""

from __future__ import annotations
from functools import cached_property
from itertools import combinations, permutations
from typing import Optional

import numpy as np


class BigLambda:
    """Chirotope representation of an order type on n points."""

    def __init__(
        self,
        n: int,
        o: np.ndarray,
        realization: Optional[list] = None,
        coloring: Optional[dict] = None,
    ):
        assert o.shape == (n, n, n), "orientation array size mismatch"
        self.n = n
        self.o = o  # np.int8, shape (n, n, n)
        self.realization = realization
        self.coloring = coloring

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def edges_cross(self, e: tuple, f: tuple) -> bool:
        """Return True if open segments e=(a,b) and f=(c,d) properly cross."""
        o = self.o
        a, b = e
        c, d = f
        return (o[a, b, c] * o[a, b, d] == -1) and (o[a, c, d] * o[b, c, d] == -1)

    def triangle_abc_contains_d(self, a: int, b: int, c: int, d: int) -> bool:
        """Return True if point d lies strictly inside triangle (a, b, c)."""
        o = self.o
        v = int(o[a, b, c])
        return v != 0 and v == o[a, b, d] and v == -o[a, c, d] and v == -o[b, c, d]

    def is_valid(
        self,
        print_collinear_warning: bool = True,
        test_alternating: bool = True,
        test_exchange: bool = True,
        test_acyclic: bool = True,
    ) -> bool:
        """Validate chirotope axioms."""
        N = range(self.n)
        o = self.o

        if test_alternating:
            for i in N:
                for j in N:
                    for k in N:
                        val = int(o[i, j, k])
                        if i != j and j != k and k != i:
                            if val == 0:
                                if print_collinear_warning:
                                    print(f"collinear points: {i},{j},{k}")
                                return False
                        else:
                            if val != 0:
                                return False
                        if val != -o[i, k, j]:
                            return False
                        if val != o[j, k, i]:
                            return False

        if test_acyclic:
            for a, b, c, d in combinations(N, 4):
                s1, s2, s3, s4 = int(o[a, b, c]), int(o[a, b, d]), int(o[a, c, d]), int(o[b, c, d])
                if s1 * s2 < 0 and s2 * s3 < 0 and s3 * s4 < 0:
                    return False

        if test_exchange:
            for x1, x2, x3 in permutations(N, 3):
                for y1, y2 in permutations(N, 2):
                    if len({y1, y2, x3}) < 3:
                        continue
                    if (
                        o[y1, x2, x3] * o[x1, y2, x3] > 0
                        and o[y2, x2, x3] * o[y1, x1, x3] > 0
                        and o[x1, x2, x3] * o[y1, y2, x3] < 0
                    ):
                        return False

        return True

    def is_chirotope(self, print_violation: bool = False) -> bool:
        """Validate the rank-3 chirotope axioms (3-term Grassmann-Plücker).

        Implements Theorem 3 of Scheucher, "A SAT attack on higher dimensional
        Erdős–Szekeres numbers" (arXiv:2105.08406), for rank ``r = 3``.  A map
        ``χ`` on ordered triples is a non-degenerate rank-3 chirotope iff:

        (i)   ``χ(a1,a2,a3) ≠ 0`` for every three distinct indices;
        (ii)  ``χ`` is alternating: swapping two arguments flips the sign
              (``sgn(σ)`` under any permutation σ);
        (iii) for all indices ``a1,a2,a3,b1,b2``::

                  if   χ(b1,a2,a3)·χ(a1,b2,a3) ≥ 0
                  and  χ(b2,a2,a3)·χ(b1,a1,a3) ≥ 0
                  then χ(a1,a2,a3)·χ(b1,b2,a3) ≥ 0

        A violation of (iii) is thus: both premises ``≥ 0`` while the conclusion
        product is ``< 0``.  This is labelling-independent.
        """
        o = self.o
        n = self.n

        # (i) + (ii): non-degenerate and alternating.
        for i, j, k in combinations(range(n), 3):
            v = int(o[i, j, k])
            if v == 0:
                if print_violation:
                    print(f"axiom (i) violated: zero triple {i},{j},{k}")
                return False
            if not (v == -o[i, k, j] == -o[j, i, k] == o[j, k, i]
                    == o[k, i, j] == -o[k, j, i]):
                if print_violation:
                    print(f"axiom (ii) violated: non-alternating triple {i},{j},{k}")
                return False

        # (iii): 3-term Grassmann-Plücker relation.  All six factors share the
        # common last index a3; b1,b2 are exchanged against a1,a2.
        rng = range(n)
        for a3 in rng:
            for a1, a2, b1, b2 in permutations(rng, 4):
                if a1 == a3 or a2 == a3 or b1 == a3 or b2 == a3:
                    continue
                prem1 = int(o[b1, a2, a3]) * int(o[a1, b2, a3])
                prem2 = int(o[b2, a2, a3]) * int(o[b1, a1, a3])
                if prem1 >= 0 and prem2 >= 0:
                    concl = int(o[a1, a2, a3]) * int(o[b1, b2, a3])
                    if concl < 0:
                        if print_violation:
                            print(f"axiom (iii) violated at a1={a1},a2={a2},"
                                  f"a3={a3},b1={b1},b2={b2}")
                        return False
        return True

    def is_signotope(self, print_violation: bool = False) -> bool:
        """Validate the signotope axiom on the sorted-triple orientations.

        A chirotope on a *linearly ordered* ground set is a signotope iff, for
        every 4-tuple a<b<c<d, the sign sequence of the four sorted triples

            o(a,b,c), o(a,b,d), o(a,c,d), o(b,c,d)

        changes sign **at most once** (reading left to right, ignoring zeros).
        This is the abstract-order-type condition that every realizable point
        set satisfies and that a valid ``.blt`` / ``.color`` file must obey.

        Only the sorted triples (a<b<c) carry independent information, so this
        test never inspects the antisymmetric copies — it validates exactly the
        data given in those file formats.
        """
        o = self.o
        for a, b, c, d in combinations(range(self.n), 4):
            signs = [int(o[a, b, c]), int(o[a, b, d]),
                     int(o[a, c, d]), int(o[b, c, d])]
            nonzero = [s for s in signs if s != 0]
            if 0 in signs:
                # a zero orientation means collinear points: not a valid OT
                if print_violation:
                    print(f"collinear triple within 4-tuple {a},{b},{c},{d}: {signs}")
                return False
            changes = sum(1 for x, y in zip(nonzero, nonzero[1:]) if x != y)
            if changes > 1:
                if print_violation:
                    print(f"signotope violation at {a},{b},{c},{d}: "
                          f"signs={signs} has {changes} sign changes")
                return False
        return True

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_string(self) -> str:
        """Compact string of '+'/'-' for all ordered triples (a<b<c)."""
        return "".join(
            "+" if self.o[a, b, c] > 0 else "-"
            for a, b, c in combinations(range(self.n), 3)
        )

    def to_small_lambda(self) -> "SmallLambda":
        from pyotlib2.core.small_lambda import SmallLambda

        n = self.n
        # l[i, j] = #{k : o[i, j, k] == 1} — vectorized over all (i,j) at once
        l = np.sum(self.o == 1, axis=2).astype(np.int32)
        return SmallLambda(n, l, realization=self.realization, coloring=self.coloring)

    def relabeled(self, labeling: list) -> "BigLambda":
        """Return a copy with points relabeled (or subset selected)."""
        m = len(labeling)
        idx = np.array(labeling, dtype=np.intp)
        new_o = self.o[np.ix_(idx, idx, idx)].copy()
        R = [self.realization[labeling[i]] for i in range(m)] if self.realization else None
        return BigLambda(m, new_o.astype(np.int8), realization=R)

    def select_points(self, perm: list) -> "SmallLambda":
        """Return SmallLambda for a subset of points given by index list perm."""
        from pyotlib2.core.small_lambda import SmallLambda

        idx = np.array(perm, dtype=np.intp)
        # sub-chirotope restricted to perm, then count +1 orientations along axis=2
        sub_o = self.o[np.ix_(idx, idx, idx)]
        l = np.sum(sub_o == 1, axis=2).astype(np.int32)
        R = [self.realization[i] for i in perm] if self.realization else None
        return SmallLambda(len(perm), l, realization=R, coloring=self.coloring)

    # ------------------------------------------------------------------
    # Geometric structures
    # ------------------------------------------------------------------

    def from_flips(self, F: list, drop_realization: bool = False) -> "BigLambda":
        """Apply a flip vector F ∈ {0,1}^n (reflect extremal points)."""
        from pyotlib2.core import point_rotation

        G = np.array([1 - 2 * f for f in F], dtype=np.int8)
        # o'[i,j,k] = G[i]*G[j]*G[k]*o[i,j,k]
        new_o = (self.o * G[:, None, None] * G[None, :, None] * G[None, None, :]).astype(np.int8)
        R = None if drop_realization else self.realization
        if R is not None:
            R = point_rotation.rotate_by_flip(R, F)
        return BigLambda(self.n, new_o, realization=R, coloring=self.coloring)

    def flip_triple(self, a: int, b: int, c: int) -> "BigLambda":
        """Return a new BigLambda with orientation of triple (a,b,c) flipped.

        Flips o[a,b,c] and all its antisymmetric / cyclic copies.
        This changes the order type (not a coordinate transform).
        The result may or may not be a valid chirotope.
        """
        new_o = self.o.copy()
        new_o[a, b, c] = new_o[b, c, a] = new_o[c, a, b] = -self.o[a, b, c]
        new_o[a, c, b] = new_o[b, a, c] = new_o[c, b, a] = -self.o[a, c, b]
        return BigLambda(self.n, new_o)

    @cached_property
    def small_lambda(self) -> "SmallLambda":
        """SmallLambda representation (cached)."""
        return self.to_small_lambda()

    @cached_property
    def onion(self) -> list:
        """Convex layers / onion peeling (cached)."""
        return self.get_onion()

    @cached_property
    def rotation_system(self) -> list:
        """Rotation system (cached)."""
        return self.get_rotation_system()

    def get_onion(self) -> list:
        """Return convex layers (onion peeling) as list of sorted point-index lists."""
        candidates = list(range(self.n))
        onion = []
        while candidates:
            if len(candidates) < 3:
                onion.append(list(candidates))
                break
            hull = []
            next_on_hull: dict = {}
            cands = np.array(candidates, dtype=np.intp)
            for i in candidates:
                for j in candidates:
                    if j != i and not np.any(self.o[i, j, cands] > 0):
                        hull.append(i)
                        next_on_hull[i] = j
                        break
            sorted_hull = []
            i = i0 = hull[0]
            while True:
                sorted_hull.append(i)
                i = next_on_hull[i]
                if i == i0:
                    break
            for i in sorted_hull:
                candidates.remove(i)
            onion.append(sorted_hull)
        return onion

    def get_rotation_system(self, signed: bool = True) -> list:
        """Return rotation system: for each point i, the cyclic order of neighbours."""
        n = self.n
        o = self.o
        order: list = [[] for _ in range(n)]

        for i in range(n):
            j = (i + 1) % n
            sign = 1
            while True:
                if not signed or sign == 1:
                    if j in order[i]:
                        break
                    order[i].append(j)

                jleft = {k for k in range(n) if o[i, j, k] > 0}
                if not jleft:
                    if signed:
                        sign *= -1

                jnext = None
                for j2 in range(n):
                    if j2 == j or j2 == i:
                        continue
                    j2left = {k for k in range(n) if o[i, j2, k] > 0}
                    if j2 in jleft and (j2left | {j2}) == jleft:
                        jnext = j2
                        break
                    if j2 not in jleft and {k for k in range(n) if o[i, j2, k] < 0} == jleft:
                        sign *= -1
                        jnext = j2
                        break
                assert jnext is not None, f"rotation system broken at i={i}, j={j}"
                j = jnext

            assert len(order[i]) == n - 1

        return order

"""The realization problem as a Quadratically-Constrained Problem (QCP).

Realizing an abstract order type means placing points ``p_0, …, p_{n-1}`` in the
plane so that every ordered triple ``(i, j, k)`` has the prescribed orientation.
The orientation of a triple is the sign of the determinant

    det(i, j, k) = x_i·(y_j − y_k) + x_j·(y_k − y_i) + x_k·(y_i − y_j),

which is **quadratic** in the coordinate variables ``x_*, y_*``.  So realization
is, at its core, always the same feasibility problem:

    find  (x_0, y_0, …, x_{n-1}, y_{n-1})
    s.t.  sign(i,j,k) · det(i,j,k) > 0   for every triple i<j<k.

That is a *quadratically-constrained problem* (QCP).  Every continuous solver in
this package solves exactly this QCP and differs only in *how*:

    * scipy / octave  — minimise a hinge penalty of the constraint violations
      (a soft relaxation of the strict inequalities), then snap to integers;
    * z3              — assert the strict quadratic inequalities directly and
      let the SMT solver decide feasibility (exact; also proves infeasibility);
    * grid            — the combinatorial special case: place points on an
      integer grid, satisfying the same inequalities by construction.

This module gives that shared QCP a single representation so the solvers do not
each re-derive the constraint list, the determinant, the penalty, the warm-start
layout, or the integer-snap-and-verify step.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.core.point_set import PointSet


@dataclass(frozen=True)
class OrientationConstraint:
    """One quadratic constraint ``sign · det(i,j,k) > 0`` (with i<j<k)."""
    i: int
    j: int
    k: int
    sign: int  # +1 or -1: the required orientation of (i, j, k)


class RealizationQCP:
    """The quadratically-constrained realization problem for an order type.

    Construct from a :class:`SmallLambda`; the object then exposes the shared
    building blocks every continuous solver needs:

        * :attr:`constraints` — the list of quadratic orientation constraints;
        * :meth:`det` — the (quadratic) orientation determinant of a triple;
        * :meth:`penalty` — the hinge objective, zero iff all constraints hold
          with unit margin (used by scipy / octave);
        * :meth:`initial_guess` — a warm-started or random start vector in the
          flat ``[x_0, y_0, x_1, y_1, …]`` layout;
        * :meth:`accept` — snap a real solution to integers and verify it
          induces exactly this order type, attaching it to the OT on success.

    Variable layout is *interleaved*: ``z = [x_0, y_0, x_1, y_1, …]`` so that
    ``z[2i] = x_i`` and ``z[2i+1] = y_i``.  (Solvers that prefer the block
    layout ``[x_0..x_{n-1}, y_0..y_{n-1}]``, e.g. the Octave script, convert at
    their boundary — see :meth:`split_xy`.)
    """

    def __init__(self, ot: SmallLambda, logsum: bool = False):
        self.ot = ot
        self.n = ot.n
        # logsum: use the per-term penalty log(1 + relu(1-sign*det)) instead of
        # relu(1-sign*det)^2.  The log damps large single violations, which can
        # improve convergence on hard instances (the "log-sum" model of the
        # pyotlib1 Mathematica tester).  Both are 0 exactly when every
        # sign*det >= 1, so acceptance is unchanged.
        self.logsum = logsum
        o = ot.to_big_lambda().o
        self.constraints: list[OrientationConstraint] = [
            OrientationConstraint(i, j, k, int(o[i, j, k]))
            for i, j, k in combinations(range(self.n), 3)
        ]
        # index arrays for vectorised evaluation
        self._i = np.fromiter((c.i for c in self.constraints), dtype=np.int64)
        self._j = np.fromiter((c.j for c in self.constraints), dtype=np.int64)
        self._k = np.fromiter((c.k for c in self.constraints), dtype=np.int64)
        self._s = np.fromiter((c.sign for c in self.constraints), dtype=np.float64)

    @property
    def num_variables(self) -> int:
        return 2 * self.n

    # ------------------------------------------------------------------
    # The quadratic orientation determinant
    # ------------------------------------------------------------------

    @staticmethod
    def det_expr(xi, yi, xj, yj, xk, yk):
        """The orientation determinant as a plain expression.

        Works for floats, numpy arrays, or symbolic solver variables (e.g. z3
        reals) — anything supporting +, -, * — so every backend can build the
        same quadratic form.  Positive value ⇔ orientation +1.
        """
        return xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)

    def det(self, points: Sequence[tuple], i: int, j: int, k: int):
        (xi, yi), (xj, yj), (xk, yk) = points[i], points[j], points[k]
        return self.det_expr(xi, yi, xj, yj, xk, yk)

    # ------------------------------------------------------------------
    # Hinge penalty (soft relaxation used by scipy / octave)
    # ------------------------------------------------------------------

    def penalty(self, flat) -> float:
        """Objective; 0 iff every ``sign·det ≥ 1``.

        ``flat`` is the interleaved vector ``[x_0, y_0, x_1, y_1, …]``.  Default
        is the sum of squared hinge violations; with ``logsum`` it is
        ``sum log(1 + relu(1 - sign·det))``.
        """
        z = np.asarray(flat, dtype=float)
        x = z[0::2]
        y = z[1::2]
        det = (x[self._i] * (y[self._j] - y[self._k])
               + x[self._j] * (y[self._k] - y[self._i])
               + x[self._k] * (y[self._i] - y[self._j]))
        viol = np.maximum(0.0, 1.0 - self._s * det)
        if self.logsum:
            return float(np.sum(np.log1p(viol)))
        return float(np.dot(viol, viol))

    # ------------------------------------------------------------------
    # Warm start / initial guess
    # ------------------------------------------------------------------

    def subrealization(self) -> Optional[list]:
        """The attached warm start (realization of the first n-1 points), or None.

        Ignores a stale warm start whose length does not match ``n-1``.
        """
        sub = getattr(self.ot, "subrealization", None)
        if sub is not None and len(sub) == self.n - 1:
            return sub
        return None

    def initial_guess(self, rng, coord_range: float = 100.0,
                      use_subrealization: bool = True) -> np.ndarray:
        """Return an interleaved start vector ``[x_0, y_0, …]``.

        With a warm start the first n-1 points are placed at their known
        coordinates and only the new (last) point starts randomly near the
        centroid — so only that one point still has to be found.  Without a warm
        start every coordinate is random in ``[-coord_range, coord_range]``.
        """
        sub = self.subrealization() if use_subrealization else None
        if sub is not None:
            flat: list[float] = []
            for (px, py) in sub:
                flat += [float(px), float(py)]
            xs = [p[0] for p in sub]
            ys = [p[1] for p in sub]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            span = max(1.0, max(max(xs) - min(xs), max(ys) - min(ys)))
            flat += [cx + rng.uniform(-span, span),
                     cy + rng.uniform(-span, span)]
            return np.array(flat, dtype=float)
        return np.array([rng.uniform(-coord_range, coord_range)
                         for _ in range(self.num_variables)], dtype=float)

    @staticmethod
    def split_xy(flat, n: int):
        """Convert interleaved ``[x_0, y_0, …]`` to block ``([x…], [y…])``."""
        z = list(flat)
        return z[0::2], z[1::2]

    @staticmethod
    def block_to_points(x_block, y_block):
        """Convert block ``([x…], [y…])`` to a list of ``(x, y)`` points."""
        return list(zip(x_block, y_block))

    # ------------------------------------------------------------------
    # Snap a real solution to integers and verify
    # ------------------------------------------------------------------

    def to_integer_points(self, flat, snap_scale: int = 10000) -> Optional[list]:
        """Snap an interleaved real solution to integer coordinates."""
        z = np.asarray(flat, dtype=float)
        try:
            return [(round(float(z[2 * i]) * snap_scale),
                     round(float(z[2 * i + 1]) * snap_scale))
                    for i in range(self.n)]
        except Exception:
            return None

    def verify(self, points: Sequence[tuple]) -> bool:
        """True iff ``points`` induce *exactly* this order type (no relabeling)."""
        if points is None or len(points) != self.n:
            return False
        try:
            ps = PointSet(self.n, list(points))
            if ps.has_collinear_points():
                return False
            o_real = ps.to_small_lambda(lazy=False).to_big_lambda().o
            return bool(np.array_equal(o_real, self.ot.to_big_lambda().o))
        except Exception:
            return False

    def accept(self, points: Sequence[tuple]) -> bool:
        """If ``points`` realize this OT, attach them and return True."""
        if self.verify(points):
            self.ot.realization = list(points)
            return True
        return False

    def accept_real(self, flat, snap_scale: int = 10000) -> bool:
        """Snap an interleaved real solution to integers, verify, and attach."""
        pts = self.to_integer_points(flat, snap_scale=snap_scale)
        if pts is None:
            return False
        return self.accept(pts)

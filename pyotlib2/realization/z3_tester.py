"""Realizability via SMT (Z3).

Encodes the order type as a system of strict determinant-sign constraints over
real coordinates and asks Z3 to find a satisfying point set.  Because the theory
of the reals (QF_NRA) is decidable, Z3's ``nlsat`` engine can not only *find* a
realization but also *prove* non-realizability — so unlike the optimization
tester this one can return a definitive ``False``.

Two modes mirror the pyotlib1 tester:

* **warm start** — if ``ot.subrealization`` is set (a realization of the first
  ``n-1`` points), those points are pinned to their known integer coordinates as
  equality constraints and only the newly added point is left free.  Each
  incremental step then has just two free variables and is solved almost
  instantly.  If the pinned system is unsatisfiable that only means the sub-point
  placement does not extend, *not* that the order type is non-realizable, so we
  retry with all points free before ever reporting ``False``.

* **full search** — every coordinate is a free variable.  Only an ``unsat`` in
  this fully free system proves non-realizability.
"""

from __future__ import annotations
from fractions import Fraction
from math import gcd
from typing import Optional

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.realization.base import RealizationTester, Undecided
from pyotlib2.realization.qcp import RealizationQCP

try:
    import z3
    _HAS_Z3 = True
except ImportError:
    _HAS_Z3 = False


class Z3RealizationTester(RealizationTester):
    """Solve the realization QCP exactly with the Z3 SMT solver.

    This is the same QCP the scipy / octave testers minimise (see
    :mod:`pyotlib2.realization.qcp`), but instead of relaxing the quadratic
    orientation constraints into a penalty, z3 asserts them *strictly*:
    ``sign(i,j,k) · det(i,j,k) > 0`` for every triple, over real coordinates.
    Because the theory of the reals is decidable, z3 can both *find* a
    realization and *prove* non-realizability (an ``unsat`` of the fully free
    system) — so unlike the optimisers this tester can return a definitive
    ``False``.
    """

    name = "z3"

    def __init__(
        self,
        parent: Optional[RealizationTester] = None,
        timeout_ms: int = 0,
        use_subrealization: bool = True,
    ):
        super().__init__(parent)
        if not _HAS_Z3:
            raise RuntimeError(
                "z3 is required for Z3RealizationTester "
                "(install the 'z3-solver' package)"
            )
        self.timeout_ms = timeout_ms
        self.use_subrealization = use_subrealization

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def _test(self, ot: SmallLambda) -> Optional[bool]:
        qcp = RealizationQCP(ot)
        n = qcp.n

        sub = qcp.subrealization() if self.use_subrealization else None

        # --- warm start: pin the first n-1 points, leave only the last free ---
        if sub is not None:
            result = self._solve(qcp, ot, pinned=sub)
            if result is True:
                return True
            # An unsat pinned system does NOT prove non-realizability: the given
            # sub-placement may simply not extend.  Fall through to a full solve.

        # --- full search: all coordinates free ---------------------------
        result = self._solve(qcp, ot, pinned=None)
        if result is True:
            return True
        if result is False:
            return False
        raise Undecided("z3 returned unknown / timed out")

    # ------------------------------------------------------------------
    # Solve one instance (optionally with pinned prefix points)
    # ------------------------------------------------------------------

    def _solve(self, qcp: RealizationQCP, ot: SmallLambda, pinned) -> Optional[bool]:
        """Return True (realized, ot.realization set), False (unsat), None (unknown)."""
        n = qcp.n
        xs = [z3.Real(f"x{i}") for i in range(n)]
        ys = [z3.Real(f"y{i}") for i in range(n)]

        solver = z3.Solver()
        if self.timeout_ms:
            solver.set("timeout", self.timeout_ms)

        # Strict quadratic orientation constraints, built from the shared QCP:
        # sign(i,j,k) * det(i,j,k) > 0  <=>  det > 0 (sign +1) or det < 0 (sign -1).
        for c in qcp.constraints:
            det = RealizationQCP.det_expr(
                xs[c.i], ys[c.i], xs[c.j], ys[c.j], xs[c.k], ys[c.k])
            solver.add(det > 0 if c.sign > 0 else det < 0)

        if pinned is not None:
            # Fix the first n-1 points to the known subrealization coordinates;
            # only the last point is left free.
            for i, (px, py) in enumerate(pinned):
                solver.add(xs[i] == int(px))
                solver.add(ys[i] == int(py))
        else:
            # Break the translation symmetry so the model is well anchored:
            # pin p0 to the origin and p1 onto the positive x-axis.
            solver.add(xs[0] == 0)
            solver.add(ys[0] == 0)
            solver.add(ys[1] == 0)
            solver.add(xs[1] > 0)

        status = solver.check()
        if status == z3.sat:
            model = solver.model()
            pts = self._model_to_integer_points(model, xs, ys, n)
            if pts is not None and qcp.accept(pts):
                return True
            # A satisfiable model that we failed to snap to matching integers.
            raise Undecided("z3 model could not be snapped to matching integers")
        if status == z3.unsat:
            return False
        return None  # unknown

    # ------------------------------------------------------------------
    # Model extraction: rationals -> common-denominator integers
    # ------------------------------------------------------------------

    def _model_to_integer_points(self, model, xs, ys, n):
        """Convert the (rational) model to integer coordinates.

        Z3 returns exact rationals; multiplying every coordinate by the lcm of
        all denominators yields an integer point set that induces the identical
        order type (scaling by a positive constant preserves every determinant
        sign).
        """
        fracs = []
        try:
            for i in range(n):
                fracs.append(self._z3_to_fraction(model, xs[i]))
                fracs.append(self._z3_to_fraction(model, ys[i]))
        except Exception:
            return None
        if any(f is None for f in fracs):
            return None

        denom_lcm = 1
        for f in fracs:
            denom_lcm = denom_lcm * f.denominator // gcd(denom_lcm, f.denominator)

        ints = [int(f * denom_lcm) for f in fracs]
        return [(ints[2 * i], ints[2 * i + 1]) for i in range(n)]

    @staticmethod
    def _z3_to_fraction(model, var):
        """Read a variable's value out of the model as an exact Fraction."""
        val = model.eval(var, model_completion=True)
        if z3.is_rational_value(val):
            return Fraction(val.numerator_as_long(), val.denominator_as_long())
        if z3.is_int_value(val):
            return Fraction(val.as_long(), 1)
        # Algebraic (irrational) value: approximate to a rational.  This should
        # not occur for these linearly-representable order types, but guard it.
        if z3.is_algebraic_value(val):
            approx = val.approx(20)
            return Fraction(approx.numerator_as_long(), approx.denominator_as_long())
        return None

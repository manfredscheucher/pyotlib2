"""Arbitrary-precision realizer: solve the realization QCP in ``mpmath``.

Some order types genuinely need many bits per coordinate, so the double-precision
optimizers (scipy / nlopt / octave) can stall not because a realization does not
exist but because their float64 arithmetic cannot represent it.  This tester runs
the *same* realization QCP (see :mod:`pyotlib2.realization.qcp`) entirely in
``mpmath`` arbitrary-precision floats: the hinge penalty is evaluated in
``mpmath.mpf`` at a chosen working precision, and a pure-Python (so
precision-agnostic) Nelder-Mead simplex drives it to zero.

The one knob is ``precision`` = the mantissa **bits** of the working precision
(``mpmath.mp.prec``).  ``precision=53`` is ordinary double; go higher (e.g. 100,
200, 1000) for order types whose coordinates do not fit float64.

Being a numerical local search it can only return ``True`` (realizable); a
failure raises :class:`Undecided` to defer to the parent tester.  It never proves
non-realizability (use ``gp`` / ``z3`` for that).  Requires ``mpmath`` (a pure
dependency of sympy; usually already installed).
"""

from __future__ import annotations
import random
from typing import Optional

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.realization.base import RealizationTester, Undecided
from pyotlib2.realization.qcp import RealizationQCP

try:
    import mpmath
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False


class MpmathRealizationTester(RealizationTester):
    """Solve the realization QCP with an mpmath arbitrary-precision Nelder-Mead.

    The objective is the shared QCP hinge penalty
    ``sum over triples relu(1 - sign*det)^2`` (or, with ``logsum``,
    ``sum log(1 + relu(1 - sign*det))``), evaluated in ``mpmath.mpf`` at
    ``precision`` bits and minimised by a pure-Python simplex.  On a zero-penalty
    solution the points are snapped to integers and verified against the exact
    chirotope.
    """

    name = "mpmath"

    def __init__(
        self,
        parent: Optional[RealizationTester] = None,
        precision: int = 100,
        trials: int = 20,
        coord_range: float = 100.0,
        use_subrealization: bool = True,
        maxiter: int = 4000,
        snap_scale: int = 10000,
        seed: Optional[int] = None,
        logsum: bool = False,
    ):
        super().__init__(parent)
        if not _HAS_MPMATH:
            raise RuntimeError("mpmath is required for MpmathRealizationTester "
                               "(install the 'mpmath' package)")
        # precision <= 0 means "default double" (53 bits); mpmath needs >= 2.
        self.precision = max(2, int(precision)) if precision and precision > 0 else 53
        self.trials = trials
        self.coord_range = coord_range
        self.use_subrealization = use_subrealization
        self.maxiter = maxiter
        self.snap_scale = snap_scale
        self.logsum = logsum
        self._rng = random.Random(seed)

    def _test(self, ot: SmallLambda) -> Optional[bool]:
        qcp = RealizationQCP(ot, logsum=self.logsum)
        nvar = qcp.num_variables

        old_prec = mpmath.mp.prec
        mpmath.mp.prec = self.precision
        try:
            objective = self._make_objective(qcp)
            for _ in range(self.trials):
                x0 = [mpmath.mpf(float(v))
                      for v in qcp.initial_guess(self._rng, self.coord_range,
                                                 self.use_subrealization)]
                xbest, fbest = self._nelder_mead(objective, x0, nvar)
                # accept_real snaps to integers and verifies the exact chirotope,
                # so a low-but-nonzero penalty can already snap to a valid
                # realization.  Feed it plain floats (integer snap is exact).
                flat = [float(v) for v in xbest]
                if qcp.accept_real(flat, self.snap_scale):
                    return True
        finally:
            mpmath.mp.prec = old_prec

        raise Undecided(f"{self.name} (prec={self.precision}) did not find a realization")

    # ------------------------------------------------------------------
    # mpmath-valued objective built from the shared QCP constraints
    # ------------------------------------------------------------------

    def _make_objective(self, qcp: RealizationQCP):
        """Return f(z) -> mpf over the interleaved layout z=[x0,y0,x1,y1,...].

        Evaluated in whatever precision mpmath.mp.prec is set to at call time.
        """
        cons = qcp.constraints
        logsum = self.logsum
        one = mpmath.mpf(1)
        zero = mpmath.mpf(0)

        def f(z):
            total = zero
            for c in cons:
                xi, yi = z[2 * c.i], z[2 * c.i + 1]
                xj, yj = z[2 * c.j], z[2 * c.j + 1]
                xk, yk = z[2 * c.k], z[2 * c.k + 1]
                det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
                viol = one - c.sign * det
                if viol > zero:
                    total += mpmath.log(one + viol) if logsum else viol * viol
            return total

        return f

    # ------------------------------------------------------------------
    # Pure-Python Nelder-Mead (works in any arithmetic supporting +,-,*,/ and
    # comparison, so it runs unchanged at arbitrary mpmath precision).
    # ------------------------------------------------------------------

    def _nelder_mead(self, f, x0, n):
        # standard coefficients
        alpha, gamma, rho, sigma = (mpmath.mpf(1), mpmath.mpf(2),
                                    mpmath.mpf('0.5'), mpmath.mpf('0.5'))
        step = mpmath.mpf(self.coord_range) / 10 + mpmath.mpf(1)

        # build initial simplex: x0 and x0 + step*e_i
        simplex = [list(x0)]
        for i in range(n):
            pt = list(x0)
            pt[i] = pt[i] + step
            simplex.append(pt)
        fvals = [f(p) for p in simplex]

        ftol = mpmath.mpf(2) ** (-(self.precision - 4))  # tight, precision-scaled
        for _ in range(self.maxiter):
            # order by objective
            order = sorted(range(n + 1), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            if fvals[0] <= ftol:
                break
            # converged simplex? (best≈worst)
            if fvals[-1] - fvals[0] <= ftol:
                break

            # centroid of all but the worst
            centroid = [mpmath.mpf(0)] * n
            for p in simplex[:-1]:
                for i in range(n):
                    centroid[i] += p[i]
            centroid = [c / n for c in centroid]

            worst = simplex[-1]
            # reflection
            xr = [centroid[i] + alpha * (centroid[i] - worst[i]) for i in range(n)]
            fr = f(xr)
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
                continue
            if fr < fvals[0]:
                # expansion
                xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(n)]
                fe = f(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
                continue
            # contraction
            xc = [centroid[i] + rho * (worst[i] - centroid[i]) for i in range(n)]
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
                continue
            # shrink toward the best
            best = simplex[0]
            for k in range(1, n + 1):
                simplex[k] = [best[i] + sigma * (simplex[k][i] - best[i])
                              for i in range(n)]
                fvals[k] = f(simplex[k])

        order = sorted(range(n + 1), key=lambda i: fvals[i])
        return simplex[order[0]], fvals[order[0]]

"""Solve the realization QCP by nonlinear optimization (scipy).

Realization is a quadratically-constrained problem (see
:mod:`pyotlib2.realization.qcp`): find coordinates so every triple's orientation
determinant has the required sign.  This tester relaxes the strict inequalities
into the QCP's hinge penalty and drives it to zero with scipy.

A single tester runs ONE scipy method so that each can be benchmarked on its
own.  Methods fall into two families:

* **local** (via ``scipy.optimize.minimize``): ``nelder-mead``, ``powell``,
  ``l-bfgs-b``, ``bfgs``, ``cg``, ``slsqp``, … — restarted from a warm start
  (only the newly added point must move) or random starts;
* **global**: ``basinhopping``, ``differential_evolution``, ``dual_annealing``,
  ``shgo`` — which explore the whole box to escape local minima.

The default tester (:class:`ScipyRealizationTester`, registry name ``scipy``)
runs Nelder-Mead locally and then falls back to differential evolution.  The
per-method variants are registered as ``scipy.<method>`` (e.g.
``scipy.neldermead``, ``scipy.basinhopping``).
"""

from __future__ import annotations
import random
from typing import Optional

import numpy as np

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.realization.base import RealizationTester, Undecided
from pyotlib2.realization.qcp import RealizationQCP

try:
    import scipy.optimize as _scipy_opt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# local methods driven by scipy.optimize.minimize
_LOCAL_METHODS = {
    "nelder-mead": "Nelder-Mead",
    "powell": "Powell",
    "l-bfgs-b": "L-BFGS-B",
    "bfgs": "BFGS",
    "cg": "CG",
    "slsqp": "SLSQP",
}
# global optimizers (each takes box bounds)
_GLOBAL_METHODS = {"basinhopping", "differential_evolution", "dual_annealing", "shgo"}


def scipy_methods() -> list[str]:
    """All scipy method keys usable as ``scipy.<key>`` solver names."""
    return list(_LOCAL_METHODS) + sorted(_GLOBAL_METHODS)


class ScipyMethodTester(RealizationTester):
    """Solve the realization QCP with a single named scipy method.

    ``method`` is one of :func:`scipy_methods` (local or global).  Local methods
    are restarted ``trials`` times; global methods run once over the coordinate
    box.  On a zero-penalty solution the real point set is snapped to integers
    and verified against the exact chirotope.
    """

    def __init__(
        self,
        method: str,
        parent: Optional[RealizationTester] = None,
        trials: int = 20,
        coord_range: float = 100.0,
        use_subrealization: bool = True,
        maxiter: int = 10000,
        snap_scale: int = 10000,
        seed: Optional[int] = None,
        logsum: bool = False,
    ):
        super().__init__(parent)
        if not _HAS_SCIPY:
            raise RuntimeError("scipy is required for ScipyMethodTester")
        key = method.lower()
        if key not in _LOCAL_METHODS and key not in _GLOBAL_METHODS:
            raise ValueError(f"unknown scipy method {method!r}; "
                             f"choose from {scipy_methods()}")
        self.method = key
        self.name = f"scipy.{key.replace('-', '')}"
        self.is_global = key in _GLOBAL_METHODS
        self.trials = trials
        self.coord_range = coord_range
        self.use_subrealization = use_subrealization
        self.maxiter = maxiter
        self.snap_scale = snap_scale
        self.logsum = logsum
        self._rng = random.Random(seed)
        self._seed = seed

    def _test(self, ot: SmallLambda) -> Optional[bool]:
        qcp = RealizationQCP(ot, logsum=self.logsum)
        if self.is_global:
            if self._global(qcp):
                return True
        else:
            if self._local(qcp):
                return True
        raise Undecided(f"{self.name} did not find a realization")

    # -- local: minimize() restarted from warm start / random ------------

    def _local(self, qcp: RealizationQCP) -> bool:
        method = _LOCAL_METHODS[self.method]
        for _ in range(self.trials):
            x0 = qcp.initial_guess(self._rng, self.coord_range,
                                   self.use_subrealization)
            try:
                res = _scipy_opt.minimize(
                    qcp.penalty, x0, method=method,
                    options={"maxiter": self.maxiter},
                )
            except Exception:
                continue
            if res.fun < 1e-9 and qcp.accept_real(res.x, self.snap_scale):
                return True
        return False

    # -- global optimizers over the coordinate box -----------------------

    def _global(self, qcp: RealizationQCP) -> bool:
        r = self.coord_range
        bounds = [(-r, r)] * qcp.num_variables
        try:
            if self.method == "differential_evolution":
                res = _scipy_opt.differential_evolution(
                    qcp.penalty, bounds, maxiter=self.maxiter, tol=1e-12,
                    seed=self._seed, polish=True)
            elif self.method == "dual_annealing":
                res = _scipy_opt.dual_annealing(
                    qcp.penalty, bounds, maxiter=self.maxiter, seed=self._seed)
            elif self.method == "shgo":
                res = _scipy_opt.shgo(qcp.penalty, bounds)
            else:  # basinhopping
                x0 = np.array([self._rng.uniform(-r, r)
                               for _ in range(qcp.num_variables)])
                res = _scipy_opt.basinhopping(
                    qcp.penalty, x0, niter=max(1, self.maxiter // 100),
                    seed=self._seed)
        except Exception:
            return False
        return res.fun < 1e-9 and qcp.accept_real(res.x, self.snap_scale)


class ScipyRealizationTester(RealizationTester):
    """Default scipy tester: warm-started local search + global fallback.

    Registry name ``scipy``.  Tries Nelder-Mead / Powell / L-BFGS-B locally
    (warm-started) and, failing that, differential evolution globally.  For
    benchmarking a single method, use :class:`ScipyMethodTester` /
    ``scipy.<method>`` instead.
    """

    name = "scipy"

    def __init__(
        self,
        parent: Optional[RealizationTester] = None,
        trials: int = 20,
        coord_range: float = 100.0,
        use_subrealization: bool = True,
        use_global: bool = True,
        maxiter: int = 10000,
        snap_scale: int = 10000,
        seed: Optional[int] = None,
        logsum: bool = False,
    ):
        super().__init__(parent)
        if not _HAS_SCIPY:
            raise RuntimeError("scipy is required for ScipyRealizationTester")
        common = dict(trials=trials, coord_range=coord_range,
                      use_subrealization=use_subrealization, maxiter=maxiter,
                      snap_scale=snap_scale, seed=seed, logsum=logsum)
        # local chain: nelder-mead -> powell -> l-bfgs-b, then optional global
        chain_methods = ["nelder-mead", "powell", "l-bfgs-b"]
        if use_global:
            chain_methods.append("differential_evolution")
        self._chain = [ScipyMethodTester(m, **common) for m in chain_methods]

    def _test(self, ot: SmallLambda) -> Optional[bool]:
        for tester in self._chain:
            try:
                if tester._test(ot) is True:
                    return True
            except Undecided:
                continue
        raise Undecided("scipy optimization did not find a realization")

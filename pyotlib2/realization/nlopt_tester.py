"""Solve the realization QCP with NLopt's nonlinear optimizers.

NLopt (https://nlopt.readthedocs.io) offers a far larger arsenal of algorithms
than scipy — derivative-free local methods (BOBYQA, COBYLA, NEWUOA, Sbplx,
PRAXIS) and derivative-free global ones (DIRECT, CRS2, ISRES, ESCH, MLSL, AGS).
When scipy runs out of steam past ~15 points, these are the next thing to try.

Each tester runs ONE NLopt algorithm so it can be benchmarked on its own; they
are registered as ``nlopt.<algo>`` (e.g. ``nlopt.LN_BOBYQA``, ``nlopt.GN_ISRES``).
The objective is the shared QCP hinge penalty (see
:mod:`pyotlib2.realization.qcp`), which is non-smooth, so only DERIVATIVE-FREE
algorithms (``LN_*`` local, ``GN_*`` global) are exposed.
"""

from __future__ import annotations
import random
from typing import Optional

import numpy as np

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.realization.base import RealizationTester, Undecided
from pyotlib2.realization.qcp import RealizationQCP

try:
    import nlopt
    _HAS_NLOPT = True
except ImportError:
    _HAS_NLOPT = False


# Derivative-free algorithms worth exposing (the QCP penalty is non-smooth).
_LOCAL = ["LN_BOBYQA", "LN_COBYLA", "LN_NELDERMEAD", "LN_NEWUOA_BOUND",
          "LN_SBPLX", "LN_PRAXIS"]
_GLOBAL = ["GN_DIRECT", "GN_DIRECT_L", "GN_CRS2_LM", "GN_ISRES", "GN_ESCH",
           "GN_AGS", "GN_MLSL"]


def nlopt_methods() -> list[str]:
    """Algorithm names usable as ``nlopt.<name>`` (present in this NLopt build)."""
    if not _HAS_NLOPT:
        return []
    return [a for a in _LOCAL + _GLOBAL if hasattr(nlopt, a)]


class NloptMethodTester(RealizationTester):
    """Solve the realization QCP with a single NLopt algorithm.

    ``method`` is one of :func:`nlopt_methods`.  Local (``LN_*``) methods are
    restarted ``trials`` times (warm-started when a subrealization is present);
    global (``GN_*``) methods run over the coordinate box.
    """

    def __init__(
        self,
        method: str,
        parent: Optional[RealizationTester] = None,
        trials: int = 20,
        coord_range: float = 100.0,
        use_subrealization: bool = True,
        maxeval: int = 20000,
        maxtime: float = 5.0,
        snap_scale: int = 10000,
        seed: Optional[int] = None,
        logsum: bool = False,
    ):
        super().__init__(parent)
        if not _HAS_NLOPT:
            raise RuntimeError("nlopt is required for NloptMethodTester "
                               "(install the 'nlopt' package)")
        key = method.upper()
        if not hasattr(nlopt, key):
            raise ValueError(f"unknown nlopt method {method!r}; "
                             f"choose from {nlopt_methods()}")
        self.method = key
        self.name = f"nlopt.{key}"
        self.is_global = key.startswith("GN_")
        self.trials = trials
        self.coord_range = coord_range
        self.use_subrealization = use_subrealization
        self.maxeval = maxeval
        self.maxtime = maxtime
        self.snap_scale = snap_scale
        self.logsum = logsum
        self._rng = random.Random(seed)
        self._seed = seed

    def _test(self, ot: SmallLambda) -> Optional[bool]:
        qcp = RealizationQCP(ot, logsum=self.logsum)
        nvar = qcp.num_variables
        algo = getattr(nlopt, self.method)
        r = self.coord_range

        def objective(x, grad):
            # grad unused (derivative-free); NLopt passes an empty array.
            return qcp.penalty(x)

        def make_opt():
            opt = nlopt.opt(algo, nvar)
            opt.set_min_objective(objective)
            opt.set_lower_bounds([-r] * nvar)
            opt.set_upper_bounds([r] * nvar)
            opt.set_maxeval(self.maxeval)
            if self.maxtime and self.maxtime > 0:
                opt.set_maxtime(self.maxtime)  # wall-clock cap per run (anti-hang)
            # Stop as soon as the penalty is essentially zero (a valid margin-1
            # realization).  This is what lets the fast methods bail out early
            # instead of grinding to their eval limit.
            opt.set_stopval(1e-9)
            opt.set_ftol_abs(1e-12)
            return opt

        # Seed NLopt's global RNG: a fixed seed makes the stochastic global
        # algorithms reproducible; seed=None seeds from the clock so repeated
        # runs explore different starting points (important when a fixed seed
        # keeps getting stuck at the same point).
        try:
            if self._seed is not None:
                nlopt.srand(self._seed)
            else:
                nlopt.srand_time()
        except Exception:
            pass

        def run(x0) -> bool:
            try:
                x = make_opt().optimize(list(x0))
            except Exception:
                return False
            # accept_real snaps to integers and verifies the exact chirotope, so
            # it is the sole, correct acceptance test (a low-but-nonzero penalty
            # can already snap to a valid realization).
            return qcp.accept_real(x, self.snap_scale)

        if self.is_global:
            if run(self._start(qcp, None)):
                return True
        else:
            for _ in range(self.trials):
                if run(self._start(qcp, self.use_subrealization)):
                    return True

        raise Undecided(f"{self.name} did not find a realization")

    def _start(self, qcp: RealizationQCP, use_sub) -> np.ndarray:
        return qcp.initial_guess(self._rng, self.coord_range,
                                 use_subrealization=bool(use_sub))

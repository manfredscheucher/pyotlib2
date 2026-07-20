"""Solve the realization QCP by nonlinear optimization in SageMath.

Like the scipy / octave testers, this solves the shared realization QCP (see
:mod:`pyotlib2.realization.qcp`) by minimising the QCP's hinge penalty, but the
minimisation runs in an external SageMath process: each trial writes a small
Sage script, runs ``sage`` on it, reads back the solution vector, snaps it to
integers and verifies the induced order type.

Sage's :func:`minimize` is used on the same
``sum over triples relu(1 - sign*det)^2`` objective, with the algorithm
selectable via ``method`` (Sage's ``minimize(algorithm=...)``): ``simplex`` and
``powell`` (derivative-free), and ``bfgs`` / ``cg`` / ``ncg`` (gradient-based,
numerically differentiated).  Registered as ``sage`` (default) /
``sage.<algorithm>``.  (Sage also offers ``minimize_constrained`` with genuine
inequality constraints, which could encode the orientation inequalities
directly rather than via the penalty — a possible future variant.)

Being a numerical search it can only return ``True`` (realizable); a failure
raises :class:`Undecided` to defer to the parent tester.  Requires ``sage``.
"""

from __future__ import annotations
import os
import random
import shutil
import subprocess
import tempfile
from typing import Optional

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.realization.base import RealizationTester, Undecided
from pyotlib2.realization.qcp import RealizationQCP


#: Sage ``minimize`` algorithms usable as ``sage.<method>`` solver names.
#   simplex/powell — derivative-free (best fit for the non-smooth hinge penalty)
#   bfgs/cg/ncg    — gradient-based (Sage estimates the gradient numerically)
_SAGE_METHODS = ("simplex", "powell", "bfgs", "cg", "ncg")


def sage_methods() -> list[str]:
    """The Sage ``minimize`` algorithm keys usable as ``sage.<key>``."""
    return list(_SAGE_METHODS)


class SageRealizationTester(RealizationTester):
    """Solve the realization QCP with SageMath's ``minimize``.

    The objective is the shared QCP hinge penalty
    ``sum over triples relu(1 - sign*det)^2``, driven to zero by Sage's
    derivative-free ``minimize`` (``simplex`` = Nelder-Mead by default).  With
    ``logsum`` the per-triple term becomes ``log(1 + relu(1 - sign*det))``
    instead of the squared hinge (the log-sum model; see :class:`RealizationQCP`).
    """

    name = "sage"

    def __init__(
        self,
        parent: Optional[RealizationTester] = None,
        method: str = "simplex",
        use_subrealization: bool = True,
        trials: int = 5,
        snap_scale: int = 10000,
        sage_bin: str = "sage",
        coord_range: float = 100.0,
        timeout: int = 120,
        seed: Optional[int] = None,
        logsum: bool = False,
    ):
        super().__init__(parent)
        self.sage_bin = sage_bin
        if shutil.which(self.sage_bin) is None:
            # fall back to the common absolute install location
            if os.path.exists("/usr/local/bin/sage"):
                self.sage_bin = "/usr/local/bin/sage"
            else:
                raise RuntimeError(f"{sage_bin} not found on PATH")
        key = method.lower()
        if key not in _SAGE_METHODS:
            raise ValueError(f"unknown sage method {method!r}; "
                             f"choose from {sage_methods()}")
        self.method = key
        self.name = f"sage.{key}"
        self.use_subrealization = use_subrealization
        self.trials = trials
        self.snap_scale = snap_scale
        self.coord_range = coord_range
        self.timeout = timeout
        self.logsum = logsum
        self._rng = random.Random(seed)

    def _test(self, ot: SmallLambda) -> Optional[bool]:
        qcp = RealizationQCP(ot, logsum=self.logsum)
        n = qcp.n

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "qcp.sage")
            out_path = os.path.join(tmpdir, "qcp.out")

            for _ in range(self.trials):
                z0 = qcp.initial_guess(self._rng, self.coord_range,
                                       self.use_subrealization)
                x_block, y_block = qcp.split_xy(z0, n)
                self._write_script(script_path, out_path, qcp,
                                   list(x_block) + list(y_block))

                try:
                    subprocess.run(
                        [self.sage_bin, script_path],
                        capture_output=True, timeout=self.timeout,
                        env=dict(os.environ),
                    )
                except Exception:
                    continue  # timeout / crash -> next trial

                block = self._read_solution(out_path, 2 * n)
                if block is None:
                    continue
                pts = qcp.block_to_points(block[:n], block[n:])
                interleaved = [c for p in pts for c in p]
                if qcp.accept_real(interleaved, self.snap_scale):
                    return True

        raise Undecided(f"{self.name} optimization did not find a realization")

    # ------------------------------------------------------------------
    # Sage script generation (block layout Z = [x_0..x_{n-1}, y_0..y_{n-1}])
    # ------------------------------------------------------------------

    def _write_script(self, script_path, out_path, qcp: RealizationQCP, z0) -> None:
        n = qcp.n
        # Build the objective as a plain Python callable over a length-2n vector,
        # using the block layout Z = [x_0..x_{n-1}, y_0..y_{n-1}].
        terms = []
        for c in qcp.constraints:
            xi, xj, xk = f"Z[{c.i}]", f"Z[{c.j}]", f"Z[{c.k}]"
            yi, yj, yk = f"Z[{n + c.i}]", f"Z[{n + c.j}]", f"Z[{n + c.k}]"
            det = f"({xi}*({yj}-{yk})+{xj}*({yk}-{yi})+{xk}*({yi}-{yj}))"
            hinge = f"relu(1 - ({c.sign})*{det})"
            # logsum: log(1 + relu(...)) damps large single violations; default
            # is the squared hinge.  Both vanish exactly when sign*det >= 1.
            terms.append(f"log(1 + {hinge})" if self.logsum
                         else f"{hinge}**2")
        obj = " + ".join(terms) if terms else "0.0"
        z0_repr = "[" + ", ".join(repr(float(v)) for v in z0) + "]"
        out_escaped = repr(out_path)
        # Sage's minimize returns a Sage vector; convert to float list.  Any
        # solver error just leaves the out file unwritten -> trial retries.
        script = f"""
from sage.all import minimize
from math import log
def relu(t):
    return (abs(t) + t) / 2.0
def obj(Z):
    return {obj}
Z0 = {z0_repr}
try:
    Z = minimize(obj, Z0, algorithm={self.method!r})
    vals = [float(v) for v in Z]
    with open({out_escaped}, 'w') as f:
        f.write(','.join(repr(v) for v in vals))
except Exception as e:
    pass
"""
        with open(script_path, "w") as f:
            f.write(script)

    def _read_solution(self, out_path, length) -> Optional[list]:
        try:
            with open(out_path) as f:
                content = f.read()
        except FileNotFoundError:
            return None
        tokens = content.replace("\n", ",").split(",")
        try:
            vals = [float(t) for t in tokens if t.strip() != ""]
        except ValueError:
            return None
        return vals if len(vals) == length else None

"""Solve the realization QCP by nonlinear optimization in GNU Octave.

Port of the pyotlib1 ``MatlabRealizationTester`` onto the pyotlib2 contract.
Like the scipy tester it solves the realization QCP (see
:mod:`pyotlib2.realization.qcp`) by minimising the QCP's hinge penalty — but the
minimisation runs in Octave: each trial writes a small ``.m`` script, runs
``octave`` on it, reads back the solution vector, snaps it to integers and
verifies the induced order type.

The Octave minimiser is selectable:

    * ``fminsearch`` — derivative-free Nelder-Mead (the default, back-compat);
    * ``fminunc``    — unconstrained minimiser (finite-difference gradients);
    * ``sqp``        — sequential quadratic programming.

All three minimise the *same* scalar QCP hinge penalty.  They are registered as
``octave.fminsearch`` / ``octave.fminunc`` / ``octave.sqp`` (and plain ``octave``
= ``fminsearch``).

Being a numerical search it can only return ``True`` (realizable); a failure
raises :class:`Undecided` to defer to the parent tester.  Requires ``octave``.
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


#: Octave minimisers usable as ``octave.<method>`` solver names.
_OCTAVE_METHODS = ("fminsearch", "fminunc", "sqp")


def octave_methods() -> list[str]:
    """The Octave minimiser keys usable as ``octave.<key>`` solver names."""
    return list(_OCTAVE_METHODS)


class OctaveRealizationTester(RealizationTester):
    """Solve the realization QCP with an Octave minimiser.

    The Octave objective is the same QCP hinge penalty the scipy tester uses,
    ``sum over triples relu(1 - sign*det)^2``, with the orientation determinant
    ``det = x_i*(y_j-y_k) + x_j*(y_k-y_i) + x_k*(y_i-y_j)``.

    ``method`` selects the Octave minimiser (see :func:`octave_methods`); it
    defaults to ``fminsearch`` so that the plain ``octave`` solver is unchanged.
    With ``logsum`` the per-triple term becomes ``log(1 + relu(1 - sign*det))``
    instead of the squared hinge (the log-sum model; see :class:`RealizationQCP`).
    """

    name = "octave"

    def __init__(
        self,
        parent: Optional[RealizationTester] = None,
        method: str = "fminsearch",
        use_subrealization: bool = True,
        trials: int = 5,
        snap_scale: int = 10000,
        octave_bin: str = "octave",
        coord_range: float = 100.0,
        timeout: int = 60,
        seed: Optional[int] = None,
        logsum: bool = False,
    ):
        super().__init__(parent)
        self.octave_bin = octave_bin
        if shutil.which(self.octave_bin) is None:
            raise RuntimeError(f"{self.octave_bin} not found on PATH")
        key = method.lower()
        if key not in _OCTAVE_METHODS:
            raise ValueError(f"unknown octave method {method!r}; "
                             f"choose from {octave_methods()}")
        self.method = key
        self.name = f"octave.{key}"
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
            script_path = os.path.join(tmpdir, "qcp.m")
            out_path = os.path.join(tmpdir, "qcp.out")

            for _ in range(self.trials):
                # QCP gives the (interleaved) start vector; the Octave script
                # uses the block layout, so convert at the boundary.
                z0 = qcp.initial_guess(self._rng, self.coord_range,
                                       self.use_subrealization)
                x_block, y_block = qcp.split_xy(z0, n)
                self._write_script(script_path, out_path, qcp,
                                   list(x_block) + list(y_block))

                try:
                    # --no-window-system (-W) fully disables graphics/windowing
                    # (stronger than --no-gui): on macOS a Qt-linked octave can
                    # otherwise flash a window / dock icon per invocation.  The
                    # offscreen Qt platform is a further belt-and-suspenders.
                    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
                    subprocess.run(
                        [self.octave_bin, "--no-window-system", "--no-gui",
                         "-q", script_path],
                        capture_output=True, timeout=self.timeout, env=env,
                    )
                except Exception:
                    continue  # timeout / crash -> next trial

                block = self._read_solution(out_path, 2 * n)
                if block is None:
                    continue
                # block [x..., y...] -> interleaved for QCP accept
                pts = qcp.block_to_points(block[:n], block[n:])
                interleaved = [c for p in pts for c in p]
                if qcp.accept_real(interleaved, self.snap_scale):
                    return True

        raise Undecided(f"{self.name} optimization did not find a realization")

    # ------------------------------------------------------------------
    # Octave script generation (block layout Z = [x_0..x_{n-1}, y_0..y_{n-1}])
    # ------------------------------------------------------------------

    def _write_script(self, script_path, out_path, qcp: RealizationQCP, z0) -> None:
        n = qcp.n
        lines = ["relu = @(t)((abs(t)+t)/2);"]
        terms = ["p = @(Z) 0"]
        for c in qcp.constraints:
            xi, xj, xk = f"Z({c.i + 1})", f"Z({c.j + 1})", f"Z({c.k + 1})"
            yi, yj, yk = f"Z({n + c.i + 1})", f"Z({n + c.j + 1})", f"Z({n + c.k + 1})"
            det = (f"({xi}*({yj}-{yk})+{xj}*({yk}-{yi})+{xk}*({yi}-{yj}))")
            hinge = f"relu(1 - ({c.sign})*{det})"
            # logsum: log(1 + relu(...)) damps large single violations; default
            # is the squared hinge.  Both vanish exactly when sign*det >= 1.
            terms.append(f" + log(1 + {hinge})" if self.logsum
                         else f" + {hinge}^2")
        terms.append(";")
        lines.append("".join(terms))
        # sqp/fminunc need a column start vector; fminsearch takes a row.
        lines.append("Z0 = [" + ", ".join(repr(float(v)) for v in z0) + "];")
        lines.append(self._minimizer_call())
        out_escaped = out_path.replace("'", "''")
        lines.append(f"dlmwrite('{out_escaped}', Z(:)', 'precision', 17);")
        with open(script_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _minimizer_call(self) -> str:
        """The Octave statement that computes ``Z`` from ``Z0`` for this method.

        All three minimise the same scalar objective ``p``; only the driver and
        its start-vector shape differ.  Each is wrapped so any solver error just
        leaves ``Z`` unwritten (the trial then reads back nothing and retries).
        """
        if self.method == "sqp":
            # sqp minimises the same scalar objective; column start vector.
            # No explicit constraints — the hinge penalty already encodes them.
            return ("try; [Z, fval] = sqp(Z0(:), p); "
                    "catch; Z = []; end")
        if self.method == "fminunc":
            # Unconstrained minimiser; finite-difference gradients.  Column
            # start vector; a GradObj-off options struct keeps it derivative-free.
            return ("try; opts = optimset('GradObj', 'off'); "
                    "[Z, fval] = fminunc(p, Z0(:), opts); "
                    "catch; Z = []; end")
        # default: fminsearch (Nelder-Mead), row start vector.
        return "[Z, fval] = fminsearch(p, Z0);"

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

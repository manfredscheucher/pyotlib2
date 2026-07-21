"""Solve the realization QCP with Wolfram Mathematica's ``NMinimize``.

Port of the pyotlib1 ``MathematicaRealizationTester`` onto the pyotlib2 contract
and the shared realization QCP (see :mod:`pyotlib2.realization.qcp`).  Each trial
writes a Mathematica script, runs it with ``wolframscript -file``, reads back the
solution vector, snaps it to integers and verifies the induced order type.

The objective is the pyotlib1 ``q``-based penalty over every orientation triple,
built from the plain hinge ``q[t] := Abs[t] - t`` (``= 2*relu(-t)``)::

    NMinimize[ Sum over triples P[q[sign*det - 1]] , {vars}, Method -> ... ]

where the per-term transform ``P`` mirrors the shared QCP objective: the default
is the squared hinge ``q[...]^2`` and ``logsum`` uses ``Log[1 + q[...]]``.  Every
term penalises ``sign*det < 1`` — exactly the QCP's margin-1 hinge.  The
``det`` uses the pyotlib2 orientation convention
``det = x_i*(y_j-y_k) + x_j*(y_k-y_i) + x_k*(y_i-y_j)`` (positive => sign +1).

The whole point of the Mathematica backend is *arbitrary precision*: ``precision``
is machine precision when 0, and a positive int sets ``WorkingPrecision`` /
``AccuracyGoal`` / ``PrecisionGoal`` (so 100, 1000, … digits are possible).

Being a numerical search it can only return ``True`` (realizable); a failure
raises :class:`Undecided` to defer to the parent tester.  Requires the
``wolframscript`` binary.
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


class MathematicaRealizationTester(RealizationTester):
    """Solve the realization QCP with Mathematica's ``NMinimize``.

    ``precision`` selects machine precision (0) or an arbitrary number of digits
    (positive int), which is set as Mathematica's ``WorkingPrecision`` /
    ``AccuracyGoal`` / ``PrecisionGoal``.  ``method`` is the ``NMinimize``
    ``Method`` (e.g. ``NelderMead``, ``DifferentialEvolution``, ``SimulatedAnnealing``).
    With ``logsum`` the QCP's log-sum penalty is used (see :class:`RealizationQCP`).
    """

    name = "mathematica"

    def __init__(
        self,
        parent: Optional[RealizationTester] = None,
        use_subrealization: bool = True,
        trials: int = 5,
        snap_scale: int = 10000,
        precision: int = 0,
        logsum: bool = False,
        method: str = "NelderMead",
        mathematica_bin: str = "wolframscript",
        timeout: int = 120,
        seed: Optional[int] = None,
    ):
        super().__init__(parent)
        self.mathematica_bin = mathematica_bin
        if shutil.which(self.mathematica_bin) is None:
            raise RuntimeError(f"{self.mathematica_bin} not found on PATH")
        self.use_subrealization = use_subrealization
        self.trials = trials
        self.snap_scale = snap_scale
        self.precision = precision
        self.logsum = logsum
        self.method = method
        self.timeout = timeout
        self._rng = random.Random(seed)

    def _test(self, ot: SmallLambda) -> Optional[bool]:
        qcp = RealizationQCP(ot, logsum=self.logsum)

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "qcp.m")
            out_path = os.path.join(tmpdir, "qcp.out")

            for _ in range(self.trials):
                z0 = qcp.initial_guess(self._rng,
                                       use_subrealization=self.use_subrealization)
                self.write_script(script_path, out_path, qcp, z0)
                try:
                    subprocess.run(
                        [self.mathematica_bin, "-file", script_path],
                        capture_output=True, timeout=self.timeout,
                    )
                except Exception:
                    continue  # timeout / crash -> next trial

                interleaved = self._read_solution(out_path, qcp.num_variables)
                if interleaved is None:
                    continue
                if qcp.accept_real(interleaved, self.snap_scale):
                    return True

        raise Undecided(f"{self.name} optimization did not find a realization")

    # ------------------------------------------------------------------
    # Mathematica script generation
    # ------------------------------------------------------------------

    def write_script(self, script_path: str, out_path: str,
                     qcp: RealizationQCP, z0=None) -> None:
        """Write the ``NMinimize`` script solving ``qcp`` to ``script_path``.

        The optimizer's coordinate list is exported (comma-separated, one number
        per point pair) to ``out_path``.  A warm start ``z0`` (the interleaved
        ``[x_0, y_0, …]`` vector) is passed to ``NMinimize`` as ``InitialPoints``
        when given; pass ``z0=None`` for a cold start (used by ``--export-mathematica``).
        """
        n = qcp.n

        def xv(i):
            return f"x{i}"

        def yv(i):
            return f"y{i}"

        # q is the plain hinge: q[t] = Abs[t]-t = 2*relu(-t), so q[sign*det - 1]
        # penalises sign*det < 1 (the QCP's margin-1 constraint).  The per-term
        # transform then mirrors the shared QCP objective (see RealizationQCP):
        # default is the squared hinge q[...]^2; logsum uses Log[1 + q[...]] to
        # damp large single violations.  Both vanish exactly when sign*det >= 1.
        q_def = "q[t_] := Abs[t] - t"

        vars_list = ", ".join(f"{xv(i)}, {yv(i)}" for i in range(n))

        terms = ["0"]
        for c in qcp.constraints:
            xi, xj, xk = xv(c.i), xv(c.j), xv(c.k)
            yi, yj, yk = yv(c.i), yv(c.j), yv(c.k)
            det = f"({xi}*({yj}-{yk})+{xj}*({yk}-{yi})+{xk}*({yi}-{yj}))"
            hinge = f"q[({c.sign})*{det} - 1]"
            terms.append(f"Log[1 + {hinge}]" if self.logsum else f"{hinge}^2")
        objective = " + ".join(terms)

        # self.precision is in mantissa BITS (matching mpmath's mp.prec and the
        # --precision flag).  Mathematica's WorkingPrecision counts DECIMAL
        # digits, so convert: digits = ceil(bits / log2(10)) = ceil(bits * 0.30103).
        # 0 = MachinePrecision (the solver default), no WorkingPrecision emitted.
        import math
        digits = math.ceil(self.precision / math.log2(10)) if (
            self.precision and self.precision > 0) else 0

        opts = [f"Method -> {self._method_option(z0, n)}", "MaxIterations -> 1000"]
        if digits > 0:
            opts += [f"WorkingPrecision -> {digits}",
                     f"AccuracyGoal -> {digits}", f"PrecisionGoal -> {digits}"]

        # Acceptance threshold for the "REALIZED" verdict, derived from the chosen
        # precision -- NOT hardcoded.  With d decimal working digits, treat the
        # penalty as essentially zero below 10^(-d/2) (half the digits as safety
        # margin against accumulated round-off).  At MachinePrecision (d=0) fall
        # back to ~half of double precision (~15 digits) -> 10^-7.
        eps_exp = (digits // 2) if digits > 0 else 7
        eps_expr = f"10^-{eps_exp}"

        out_escaped = out_path.replace("\\", "\\\\").replace('"', '\\"')
        lines = [
            q_def,
            f"vars = {{{vars_list}}};",
            f"result = NMinimize[{{{objective}, {{{vars_list}}}}}, {{{vars_list}}}"
            + (", " + ", ".join(opts) if opts else "") + "];",
            # fmin is the minimised penalty; it is 0 (to solver tolerance) iff a
            # realization was found -- this is the found/not-found indicator.
            "fmin = First[result];",
            "sol = vars /. Last[result];",
            # eps is the precision-derived REALIZED threshold (see above).
            f"eps = {eps_expr};",
            # Machine-readable output file: the coordinates, one point per line.
            # (Kept as the primary .out so the in-process solver can read it back.)
            f'Export["{out_escaped}", sol, "Table"];',
            # A sibling .objective file holds fmin alone, so a script can decide
            # found/not-found without re-evaluating the objective.
            f'Export["{out_escaped}.objective", {{fmin}}, "Table"];',
            # Human-readable verdict on stdout when run with wolframscript -file.
            'Print["objective = ", fmin];',
            'Print["threshold = ", eps];',
            'If[fmin < eps, Print["REALIZED"], '
            'Print["NOT REALIZED (local minimum / precision -- '
            'not a proof of non-realizability)"]];',
            'Print["coords = ", sol];',
        ]
        with open(script_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _method_option(self, z0, n: int) -> str:
        """The ``Method`` option value, with warm-start ``InitialPoints`` if given."""
        if z0 is None:
            return self.method
        pts = ", ".join(f"{{{float(z0[2 * i])}, {float(z0[2 * i + 1])}}}"
                        for i in range(n))
        return f'{{"{self.method}", "InitialPoints" -> {{{{{pts}}}}}}}'

    # ------------------------------------------------------------------
    # Parse Mathematica output back into a solution vector
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(token: str) -> float:
        """Parse a Mathematica number, honouring the ``a*^b`` scientific form."""
        token = token.strip().split("`")[0]  # drop precision annotation `123
        if "*^" in token:
            mantissa, expo = token.split("*^")
            return float(mantissa) * (10.0 ** int(expo))
        return float(token)

    def _read_solution(self, out_path: str, length: int) -> Optional[list]:
        try:
            with open(out_path) as f:
                content = f.read()
        except FileNotFoundError:
            return None
        content = content.replace("{", " ").replace("}", " ")
        tokens = content.replace("\n", ",").replace(" ", ",").split(",")
        try:
            vals = [self._to_float(t) for t in tokens if t.strip() != ""]
        except ValueError:
            return None
        return vals if len(vals) == length else None

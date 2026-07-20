"""Grassmann-Plücker realizability test via GLPK linear programming.

Generates an LP whose feasibility structure certifies non-realizability.
A solution with ``myEps > 0`` witnesses that the LP is *unbounded* (the whole
solution can be scaled arbitrarily), which means the order type *might* be
realizable — GP is only a necessary condition, so this is inconclusive.  If the
LP is *infeasible*, the order type is provably non-realizable.

Requires the ``glpsol`` binary (GLPK package) to be on PATH.

Reference: Grassmann-Plücker relations for rank-3 oriented matroids.
"""

from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from itertools import combinations, permutations
from typing import Optional

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.core.utils import binomial
from pyotlib2.realization.base import MAX_REALIZABLE_N, RealizationTester, Undecided


class GPRealizationTester(RealizationTester):
    """Tests realizability via a Grassmann-Plücker LP solved with GLPK.

    The LP maximises a slack variable myEps subject to:
      - myEps >= 0
      - o_abc >= 0  for all triples (a,b,c) with a<b<c
      - For each qualifying 5-tuple, two 3-term GP constraints of the form:
          o_abc + o_ade - o_abd - o_ace + myEps <= 0
          o_abe + o_acd - o_abd - o_ace + myEps <= 0

    Feasibility structure (see docstring above):

    * LP is **infeasible**            → order type is NON-realizable  → ``False``
    * LP is **unbounded** (myEps > 0) → order type MIGHT be realizable → Undecided
    * LP is **optimal** with myEps==0 → inconclusive                  → Undecided

    ``is_realizable`` never returns ``True``: GP is only a necessary condition.
    """

    name = "grassmann-plucker"

    def __init__(
        self,
        parent: Optional[RealizationTester] = None,
        time_limit: int = 0,
        use_dual_simplex: bool = True,
    ):
        super().__init__(parent)
        if shutil.which("glpsol") is None:
            raise RuntimeError("glpsol (GLPK) not found on PATH")
        self.time_limit = time_limit
        self.use_dual_simplex = use_dual_simplex

    def _test(self, ot: SmallLambda) -> bool:
        with tempfile.TemporaryDirectory() as tmpdir:
            lp_path  = os.path.join(tmpdir, "ot.lp")
            out_path = os.path.join(tmpdir, "ot.out")
            self._write_lp(ot, lp_path)
            result = self._solve(lp_path, out_path)
            # Sanity check: every order type on n <= 8 points is realizable, so
            # the GP LP must never certify non-realizability there.  A False on
            # n <= 8 means the GP model / solver is broken.
            if result is False and ot.n <= MAX_REALIZABLE_N:
                raise AssertionError(
                    f"GP certified a n={ot.n} order type as non-realizable, but "
                    f"every order type on n <= {MAX_REALIZABLE_N} points is "
                    f"realizable — the GP model or solver is broken.")
            return result

    # ------------------------------------------------------------------
    # LP generation  (ported from old pyotlib GPRealizationTester)
    # ------------------------------------------------------------------

    def _write_lp(self, ot: SmallLambda, path: str) -> None:
        n = ot.n
        o_arr = ot.to_big_lambda().o

        def var(i, j, k):
            a, b, c = sorted([i, j, k])
            return f"o_{a}_{b}_{c}"

        # NOTE: no leading /* */ comment — GLPK's CPLEX-LP parser rejects a
        # comment before the 'Maximize' keyword.  The file must start directly
        # with the objective sense.
        lines = [
            "Maximize",
            " obj: 1 myEps",
            "Subject To",
            " pos_myeps: myEps >= 0",
        ]

        # positivity constraints for each triple
        for a, b, c in combinations(range(n), 3):
            lines.append(f" pos_{a}_{b}_{c}: {var(a,b,c)} >= 0")

        # Grassmann-Plucker constraints
        # For each (a,b) and each permutation (c,d,e) of {b+1..n-1}:
        # if all three products o[a,b,c]*o[a,d,e], o[a,b,d]*o[a,c,e], o[a,b,e]*o[a,c,d] > 0
        # then add two GP inequalities.
        ct = 0
        for a, b in permutations(range(n), 2):
            for c, d, e in permutations(range(b + 1, n), 3):
                if (    o_arr[a, b, c] * o_arr[a, d, e] > 0
                    and o_arr[a, b, d] * o_arr[a, c, e] > 0
                    and o_arr[a, b, e] * o_arr[a, c, d] > 0):
                    ct += 1
                    lines.append(
                        f" c{2*ct}: {var(a,b,c)} +{var(a,d,e)} -{var(a,b,d)} -{var(a,c,e)} + myEps <= 0"
                    )
                    lines.append(
                        f" c{2*ct+1}: {var(a,b,e)} +{var(a,c,d)} -{var(a,b,d)} -{var(a,c,e)} + myEps <= 0"
                    )

        assert ct == 5 * binomial(n, 5), f"expected {5*binomial(n,5)} GP constraints, got {ct}"

        lines.append("End")

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Solve and parse
    # ------------------------------------------------------------------

    def _solve(self, lp_path: str, out_path: str) -> bool:
        # Use --nopresol so that GLPK 5.0 reports an unambiguous simplex verdict
        # on stdout:
        #   infeasible → "LP HAS NO PRIMAL FEASIBLE SOLUTION"
        #   unbounded  → "LP HAS UNBOUNDED PRIMAL SOLUTION"
        #   optimal    → "OPTIMAL SOLUTION FOUND"
        # (With presolve, unbounded is instead reported via the *dual* — "PROBLEM
        # HAS NO DUAL FEASIBLE SOLUTION" — which is easy to confuse with the
        # primal-infeasible message; --nopresol removes that ambiguity.)
        # We additionally write a printable solution file (-o) and cross-check
        # its "Status:" line.
        cmd = [
            "glpsol",
            "--lp", lp_path,
            "--nopresol",
            "-o", out_path,
        ]
        if self.use_dual_simplex:
            cmd.append("--dual")
        if self.time_limit > 0:
            cmd += ["--tmlim", str(self.time_limit)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = (proc.stdout or "") + (proc.stderr or "")
        up = stdout.upper()

        status = self._read_status(out_path)

        # ---- decide from stdout (primary) --------------------------------
        # Order matters: check for the "NO PRIMAL FEASIBLE" / plain
        # "NO FEASIBLE SOLUTION" markers, but make sure we do NOT trip on the
        # *dual* message (which signals unboundedness, not infeasibility).
        infeasible = (
            "NO PRIMAL FEASIBLE SOLUTION" in up
            or "PROBLEM HAS NO FEASIBLE SOLUTION" in up
        )
        unbounded = (
            "UNBOUNDED" in up
            or "NO DUAL FEASIBLE SOLUTION" in up
        )
        optimal = "OPTIMAL SOLUTION FOUND" in up or "OPTIMAL LP SOLUTION FOUND" in up

        # Cross-check with the printable solution file's Status line.
        if status == "INFEASIBLE":
            infeasible = True
        elif status == "UNBOUNDED":
            unbounded = True

        if infeasible and not unbounded:
            return False  # LP infeasible → order type is NON-realizable
        if unbounded or optimal:
            # unbounded (myEps can grow) or bounded-optimal (myEps==0): GP is
            # inconclusive — the order type may still be realizable.
            raise Undecided("GP LP is feasible/unbounded → realizability not decided")

        raise Undecided(
            "GLPK did not produce a conclusive result "
            f"(status={status!r}, rc={proc.returncode})"
        )

    @staticmethod
    def _read_status(out_path: str) -> Optional[str]:
        """Return the normalised ``Status:`` verdict from a glpsol -o file.

        One of ``"INFEASIBLE"``, ``"UNBOUNDED"``, ``"OPTIMAL"``, or ``None``.
        """
        try:
            with open(out_path) as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("Status:"):
                        val = s.split(":", 1)[1].strip().upper()
                        if "INFEASIBLE" in val:
                            return "INFEASIBLE"
                        if "UNBOUNDED" in val:
                            return "UNBOUNDED"
                        if "OPTIMAL" in val:
                            return "OPTIMAL"
                        return None
        except (FileNotFoundError, OSError):
            pass
        return None

#!/usr/bin/env python3
"""Benchmark realization solvers on abstract order types.

For increasing ``n`` this script takes order types that have a *known*
realization, strips the realization away (keeping only the abstract order
type), and asks each solver to re-realize it from scratch.  It measures
wall-clock time and success, with a hard per-instance timeout.

Idea
----
"wenn ein solver bereits für ein n nicht performt, brauchst du's für größere
werte nicht zu testen": we iterate ``n`` in increasing order and maintain a set
of *dead* solvers.  A solver that fails (or times out) on ALL instances of a
given ``n`` is marked dead and is not run for any larger ``n``.  Other solvers
continue independently.

Timeout enforcement
-------------------
scipy / z3 do not reliably honour soft timeouts, so every (solver, instance)
attempt runs in its own child process (``multiprocessing.Process``).  The parent
joins with a timeout and terminates the child if it overruns.  A timeout is
reported as a distinct outcome ("timeout"), separate from "fail" (the solver
returned None / Undecided within the time budget) and "error" (an exception).

Success check
-------------
The input is stripped to abstract-only via ``tests.point_configs.abstract_only``.
A run counts as "success" only if the returned realization, converted back via
``PointSet(n, pts).to_small_lambda(lazy=False).to_big_lambda().o``, is
``numpy.array_equal`` to the original abstract order type's ``to_big_lambda().o``
-- element-for-element, with NO relabeling.

Usage
-----
Run from the repository root (uses the project venv python directly)::

    .venv/bin/python scripts/benchmark_realization.py
    .venv/bin/python scripts/benchmark_realization.py --max-n 13 --timeout 30
    .venv/bin/python scripts/benchmark_realization.py --solvers scipy,z3
    .venv/bin/python scripts/benchmark_realization.py --instances-per-n 1

Options
-------
--timeout SECONDS         per-instance timeout (default: 30)
--solvers a,b,c           comma-separated solver names (default: all available)
--max-n N                 skip instances with n > N (default: 20)
--instances-per-n K       cap the number of instances per n value (default: 3)

Results are printed as a table (solvers x n) and a per-solver summary, and the
raw data is written to ``scripts/benchmark_realization_results.json``.

This is a standalone benchmark; it is intentionally kept OUT of the unit-test
suite because it is slow.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# Make the repository root importable when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pyotlib2.core.point_set import PointSet          # noqa: E402
from pyotlib2.core.small_lambda import SmallLambda    # noqa: E402
from pyotlib2.realization import available_solvers, make_tester  # noqa: E402
from tests import point_configs as pc                 # noqa: E402

# Solvers we never benchmark for realization here.
#   gp   -> only proves NON-realizability (never produces a realization)
_EXCLUDED_SOLVERS = {"gp"}

# grid is known to be weak past ~10 points; skip it for larger n by default.
_GRID_MAX_N = 10

# Outcomes
OK = "ok"
FAIL = "fail"          # solver returned None / Undecided (or wrong realization) in time
TIMEOUT = "timeout"    # solver exceeded the time budget (child killed)
ERROR = "error"        # solver raised an exception
DEAD = "dead"          # not run: solver already declared dead at a smaller n
UNAVAILABLE = "unavailable"  # solver binary / package missing


# ---------------------------------------------------------------------------
# Test instances
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    n: int
    label: str
    abstract: SmallLambda           # abstract-only order type (no realization)
    target_o: object = None         # numpy o-array of the abstract OT (cached)


def build_instances(max_n: int, instances_per_n: int) -> list[Instance]:
    """Build the (n, label, abstract SmallLambda) test instances.

    Uses PoSeZo realized sets where the n matches, and the generic
    ``convex_position_moment_curve`` / ``one_interior`` generators to fill the
    other n values and to have more than one instance per n.
    """
    # (n, label, factory) -- factory returns a *realized* SmallLambda.
    sources: list[tuple[int, str, callable]] = [
        # n = 6
        (6, "moment_curve", lambda: pc.convex_position_moment_curve(6)),
        (6, "one_interior", lambda: pc.one_interior(6)),
        # n = 8
        (8, "moment_curve", lambda: pc.convex_position_moment_curve(8)),
        (8, "one_interior", lambda: pc.one_interior(8)),
        # n = 10 (PoSeZo + generators)
        (10, "posezo_1_convex_5_hole", pc.PoSeZo.n10_c1_1_convex_5_hole),
        (10, "moment_curve", lambda: pc.convex_position_moment_curve(10)),
        (10, "one_interior", lambda: pc.one_interior(10)),
        # n = 13 (PoSeZo + generators)
        (13, "posezo_min_convex_3_4_5_holes", pc.PoSeZo.n13_c1_min_convex_3_4_5_holes),
        (13, "moment_curve", lambda: pc.convex_position_moment_curve(13)),
        (13, "one_interior", lambda: pc.one_interior(13)),
        # n = 15 (PoSeZo + generators)
        (15, "posezo_9_convex_5_holes", pc.PoSeZo.n15_c1_9_convex_5_holes),
        (15, "moment_curve", lambda: pc.convex_position_moment_curve(15)),
        (15, "one_interior", lambda: pc.one_interior(15)),
        # n = 17 (generators only)
        (17, "moment_curve", lambda: pc.convex_position_moment_curve(17)),
        (17, "one_interior", lambda: pc.one_interior(17)),
        # n = 20 (generators only)
        (20, "moment_curve", lambda: pc.convex_position_moment_curve(20)),
        (20, "one_interior", lambda: pc.one_interior(20)),
    ]

    per_n: dict[int, int] = {}
    instances: list[Instance] = []
    for n, label, factory in sources:
        if n > max_n:
            continue
        if per_n.get(n, 0) >= instances_per_n:
            continue
        try:
            realized = factory()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] could not build instance n={n} {label}: {exc}",
                  file=sys.stderr)
            continue
        abstract = pc.abstract_only(realized)
        target_o = abstract.to_big_lambda().o
        instances.append(Instance(n=n, label=label, abstract=abstract,
                                  target_o=target_o))
        per_n[n] = per_n.get(n, 0) + 1

    instances.sort(key=lambda inst: inst.n)
    return instances


# ---------------------------------------------------------------------------
# Solver worker (runs in a child process)
# ---------------------------------------------------------------------------

def _worker(solver_name: str, n: int, ot_string: str, conn) -> None:
    """Child-process entry point: realize the abstract OT and report success.

    Sends a tuple ``(outcome, realization_or_None)`` back through the pipe.
    The strict success check is repeated in the PARENT as well (belt and
    suspenders); here we do it too so we only ship a realization on success.
    """
    try:
        import numpy as np
        ot = SmallLambda.from_string(n, ot_string)
        target_o = ot.to_big_lambda().o
        tester = make_tester(solver_name)
        result = tester.is_realizable(ot)
        if result is True and ot.realization is not None:
            pts = ot.realization
            try:
                o_real = (PointSet(n, pts)
                          .to_small_lambda(lazy=False)
                          .to_big_lambda().o)
                if np.array_equal(o_real, target_o):
                    conn.send((OK, [tuple(p) for p in pts]))
                    return
            except Exception:
                pass
            conn.send((FAIL, None))
            return
        # None / False -> the solver could not produce a matching realization
        conn.send((FAIL, None))
    except Exception as exc:  # noqa: BLE001
        conn.send((ERROR, f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            conn.close()
        except Exception:
            pass


@dataclass
class RunResult:
    outcome: str
    seconds: float
    detail: Optional[object] = None


def run_one(solver_name: str, inst: Instance, timeout: float,
            ctx) -> RunResult:
    """Run a single (solver, instance) attempt in a separate process.

    Kills the child after ``timeout`` seconds and reports "timeout".
    """
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_worker,
        args=(solver_name, inst.n, inst.abstract.to_string(), child_conn),
    )
    start = time.perf_counter()
    proc.start()
    child_conn.close()  # only the child writes; parent keeps the read end
    proc.join(timeout)
    elapsed = time.perf_counter() - start

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():  # pragma: no cover - stubborn child
            proc.kill()
            proc.join()
        return RunResult(TIMEOUT, elapsed)

    # process finished; read its message (if any)
    payload = None
    if parent_conn.poll():
        try:
            payload = parent_conn.recv()
        except EOFError:
            payload = None
    try:
        parent_conn.close()
    except Exception:
        pass

    if payload is None:
        # child died without reporting (e.g. crash / OOM)
        return RunResult(ERROR, elapsed, detail="no result from child")

    outcome, detail = payload
    return RunResult(outcome, elapsed, detail=detail)


# ---------------------------------------------------------------------------
# Solver availability
# ---------------------------------------------------------------------------

def probe_available(names: list[str]) -> dict[str, bool]:
    """Try to construct each solver; mark unavailable if it raises RuntimeError."""
    avail: dict[str, bool] = {}
    for name in names:
        try:
            make_tester(name)
            avail[name] = True
        except RuntimeError as exc:
            print(f"[info] solver {name!r} unavailable: {exc}", file=sys.stderr)
            avail[name] = False
        except Exception as exc:  # noqa: BLE001
            print(f"[info] solver {name!r} unavailable: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            avail[name] = False
    return avail


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _cell_text(outcome: str, seconds: float) -> str:
    if outcome == OK:
        return f"OK {seconds:.1f}s"
    if outcome == TIMEOUT:
        return "TIMEOUT"
    if outcome == FAIL:
        return "FAIL"
    if outcome == ERROR:
        return "ERROR"
    if outcome == DEAD:
        return "-"
    if outcome == UNAVAILABLE:
        return "n/a"
    return outcome


def render_table(solvers: list[str], rows: list[dict]) -> str:
    """rows: list of {'label': str, 'cells': {solver: (outcome, seconds)}}."""
    headers = ["instance"] + solvers
    widths = [len(h) for h in headers]
    text_rows: list[list[str]] = []
    for row in rows:
        cells = [row["label"]]
        for s in solvers:
            outcome, seconds = row["cells"].get(s, (DEAD, 0.0))
            cells.append(_cell_text(outcome, seconds))
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(c))
        text_rows.append(cells)

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    lines += [fmt(r) for r in text_rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark realization solvers on abstract order types.")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="per-instance timeout in seconds (default: 30)")
    parser.add_argument("--solvers", type=str, default="",
                        help="comma-separated solver names (default: all available)")
    parser.add_argument("--max-n", type=int, default=20,
                        help="skip instances with n greater than this (default: 20)")
    parser.add_argument("--instances-per-n", type=int, default=3,
                        help="cap on instances per n value (default: 3)")
    parser.add_argument("--json-out", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "benchmark_realization_results.json"),
                        help="path for the raw JSON results")
    args = parser.parse_args(argv)

    # multiprocessing context: 'spawn' is safest across numpy / z3 / octave.
    ctx = mp.get_context("spawn")

    # Determine solver list.
    if args.solvers.strip():
        requested = [s.strip() for s in args.solvers.split(",") if s.strip()]
    else:
        requested = [s for s in available_solvers() if s not in _EXCLUDED_SOLVERS]

    excluded_hit = [s for s in requested if s in _EXCLUDED_SOLVERS]
    if excluded_hit:
        print(f"[info] ignoring realization-unsuitable solver(s): "
              f"{', '.join(excluded_hit)}", file=sys.stderr)
    requested = [s for s in requested if s not in _EXCLUDED_SOLVERS]

    available = probe_available(requested)
    solvers = [s for s in requested if available.get(s)]
    unavailable = [s for s in requested if not available.get(s)]

    print(f"Benchmark: solvers={requested} timeout={args.timeout}s "
          f"max_n={args.max_n} instances_per_n={args.instances_per_n}")
    if unavailable:
        print(f"Unavailable (skipped): {', '.join(unavailable)}")

    instances = build_instances(args.max_n, args.instances_per_n)
    print(f"Built {len(instances)} instances "
          f"across n = {sorted(set(i.n for i in instances))}\n")

    # Early-stop bookkeeping.
    dead: set[str] = set()
    largest_solved: dict[str, int] = {s: 0 for s in requested}
    successes: dict[str, int] = {s: 0 for s in requested}

    rows: list[dict] = []
    raw: dict = {
        "config": {
            "timeout": args.timeout,
            "max_n": args.max_n,
            "instances_per_n": args.instances_per_n,
            "requested_solvers": requested,
            "available_solvers": solvers,
            "unavailable_solvers": unavailable,
        },
        "instances": [],
    }

    # Group instances by n so we can apply the early-stop rule per n.
    ns = sorted(set(i.n for i in instances))
    for n in ns:
        n_instances = [i for i in instances if i.n == n]
        # Track, per solver, whether it succeeded on ANY instance at this n.
        solved_any_this_n: dict[str, bool] = {s: False for s in solvers}
        attempted_this_n: dict[str, bool] = {s: False for s in solvers}

        for inst in n_instances:
            label = f"n={inst.n} {inst.label}"
            cells: dict[str, tuple[str, float]] = {}
            inst_raw = {"n": inst.n, "label": inst.label, "results": {}}

            for s in solvers:
                if s in dead:
                    cells[s] = (DEAD, 0.0)
                    inst_raw["results"][s] = {"outcome": DEAD, "seconds": 0.0}
                    continue
                if s == "grid" and inst.n > _GRID_MAX_N:
                    # grid is known bad past ~10 points; do not run it.
                    cells[s] = (DEAD, 0.0)
                    inst_raw["results"][s] = {"outcome": DEAD, "seconds": 0.0,
                                              "note": "grid skipped for n>10"}
                    continue

                attempted_this_n[s] = True
                res = run_one(s, inst, args.timeout, ctx)
                cells[s] = (res.outcome, res.seconds)
                entry = {"outcome": res.outcome, "seconds": round(res.seconds, 3)}
                if res.detail is not None and res.outcome in (ERROR,):
                    entry["detail"] = res.detail
                inst_raw["results"][s] = entry

                if res.outcome == OK:
                    solved_any_this_n[s] = True
                    successes[s] += 1
                    largest_solved[s] = max(largest_solved[s], inst.n)

                print(f"  {label:<40} {s:<8} -> "
                      f"{_cell_text(res.outcome, res.seconds)}")

            rows.append({"label": label, "cells": cells})
            raw["instances"].append(inst_raw)

        # Apply the early-stop rule: a solver that ATTEMPTED instances at this n
        # but solved none of them is declared dead for larger n.
        for s in solvers:
            if s in dead:
                continue
            if attempted_this_n[s] and not solved_any_this_n[s]:
                dead.add(s)
                print(f"  [dead] solver {s!r} solved no instance at n={n}; "
                      f"skipping it for larger n")
        print()

    # -------------------------------------------------------------------
    # Output: table + summary
    # -------------------------------------------------------------------
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(render_table(solvers, rows))
    print()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    summary_rows = []
    for s in requested:
        if s in unavailable:
            print(f"  {s:<8} unavailable (binary/package missing)")
            summary_rows.append({"solver": s, "available": False})
            continue
        largest = largest_solved.get(s, 0)
        largest_txt = str(largest) if largest else "none"
        print(f"  {s:<8} largest n solved = {largest_txt:<5} "
              f"total successes = {successes.get(s, 0)}"
              f"{'   [became dead]' if s in dead else ''}")
        summary_rows.append({
            "solver": s,
            "available": True,
            "largest_n_solved": largest,
            "total_successes": successes.get(s, 0),
            "dead": s in dead,
        })

    raw["summary"] = summary_rows

    with open(args.json_out, "w") as fh:
        json.dump(raw, fh, indent=2)
    print(f"\nRaw results written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Runtime-distribution benchmark: realize every order type in a file with each solver.

Reads an order-type file (default: the n=7 database ``otypes07.b08``), strips any
coordinates, and asks every solver to re-realize each abstract order type from
scratch, with a hard per-instance timeout.  It records the wall-clock time of
each solved instance, then:

  * writes one runtime histogram PNG per solver, and
  * prints a summary table comparing mean / median / etc. across solvers.

Select the database by n (6..10, auto-resolving otypesNN.b08/.b16) or give an
explicit --file (any format the reader understands: b08, b16, blt, ...).  By
default it randomly SAMPLES 20 order types (reservoir sampling — not just the
easy near-convex ones at the top of the file) and randomly PERMUTES each one's
points before realizing, because realization difficulty depends on the labeling.

Usage (from the repo root):
    .venv/bin/python scripts/benchmark_realization_distribution.py my_ots.blt          # your own file
    .venv/bin/python scripts/benchmark_realization_distribution.py my_ots.blt --sample 0  # all of it
    .venv/bin/python scripts/benchmark_realization_distribution.py --n 7               # built-in n=7 DB
    .venv/bin/python scripts/benchmark_realization_distribution.py --n 9 --sample 20 --solvers scipy.neldermead,z3,grid
    .venv/bin/python scripts/benchmark_realization_distribution.py --n 7 --shuffle-seed -1  # canonical labeling

A "success" requires the returned realization to reproduce the order type
exactly (element-for-element chirotope match, no relabeling).
"""

from __future__ import annotations

import os

# Force a headless, file-only plotting backend for EVERY process (this module is
# re-imported by each spawned worker).  Set before matplotlib is imported so no
# GUI window / dock icon is ever created; the benchmark is command-line only.
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pyotlib2.core.point_set import PointSet          # noqa: E402
from pyotlib2.core.small_lambda import SmallLambda    # noqa: E402
from pyotlib2.io.readers import read_order_types      # noqa: E402
from pyotlib2.realization import available_solvers, make_tester  # noqa: E402

# Solvers that cannot produce a realization (gp only proves non-realizability).
_EXCLUDED = {"gp"}

OK, FAIL, TIMEOUT, ERROR = "ok", "fail", "timeout", "error"


# ---------------------------------------------------------------------------
# Load abstract order types from any supported file.
# ---------------------------------------------------------------------------

def otdb_path_for_n(n: int) -> Optional[str]:
    """Path to the order-type database file for a given n, or None if absent.

    n <= 8 -> otypesNN.b08 ; n >= 9 -> otypesNN.b16 (16-bit coordinates).
    """
    ext = "b08" if n <= 8 else "b16"
    p = os.path.join(_REPO_ROOT, "tests", "otdb", "otypes", f"otypes{n:02d}.{ext}")
    return p if os.path.exists(p) else None


def load_abstract_ots(path: str, fmt: Optional[str],
                      limit: Optional[int], sample: Optional[int],
                      sample_seed: int, shuffle_seed: Optional[int]) -> list[SmallLambda]:
    """Load order types, strip coordinates, and (by default) shuffle the points.

    Selection:
      * ``sample=N``: take N order types chosen uniformly at random from the
        whole file (reservoir sampling, so huge databases like n=10 need not fit
        in memory, and we don't just take the easy near-convex ones at the top);
      * else ``limit=N``: take the first N;
      * else: take all.

    Shuffling: the database uses a canonical (natural/lex-min) labeling whose
    first entries are near convex position and trivial to realize; realization
    difficulty depends on the labeling.  So each selected order type gets a
    random point permutation before realizing (seeded).  shuffle_seed=None keeps
    the canonical labeling.
    """
    import random as _random

    def _abstract(sl):
        return SmallLambda(sl.n, sl.get_l().copy())

    stream = read_order_types(path, fmt=fmt)

    if sample is not None:
        # reservoir sampling of `sample` items over the (unknown-length) stream
        rrng = _random.Random(sample_seed)
        reservoir: list = []
        for i, sl in enumerate(stream):
            if i < sample:
                reservoir.append(_abstract(sl))
            else:
                j = rrng.randint(0, i)
                if j < sample:
                    reservoir[j] = _abstract(sl)
        selected = reservoir
    else:
        selected = []
        for sl in stream:
            selected.append(_abstract(sl))
            if limit is not None and len(selected) >= limit:
                break

    # apply the random point permutation (difficulty depends on labeling)
    if shuffle_seed is not None:
        srng = _random.Random(shuffle_seed)
        out = []
        for abstract in selected:
            perm = list(range(abstract.n))
            srng.shuffle(perm)
            out.append(abstract.relabeled(perm))
        return out
    return selected


# ---------------------------------------------------------------------------
# Worker: realize one abstract OT with one solver, in a child process.
# ---------------------------------------------------------------------------

def _worker(solver_name: str, n: int, ot_string: str, conn) -> None:
    try:
        import numpy as np
        ot = SmallLambda.from_string(n, ot_string)
        target_o = ot.to_big_lambda().o
        tester = make_tester(solver_name)
        result = tester.is_realizable(ot)
        if result is True and ot.realization is not None:
            o_real = (PointSet(n, ot.realization)
                      .to_small_lambda(lazy=False).to_big_lambda().o)
            if np.array_equal(o_real, target_o):
                conn.send(OK)
                return
        conn.send(FAIL)
    except Exception as exc:  # noqa: BLE001
        conn.send(f"{ERROR}:{type(exc).__name__}: {exc}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_one(solver_name: str, ot: SmallLambda, timeout: float, ctx) -> tuple[str, float]:
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker,
                       args=(solver_name, ot.n, ot.to_string(), child_conn))
    start = time.perf_counter()
    proc.start()
    child_conn.close()
    proc.join(timeout)
    elapsed = time.perf_counter() - start
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return TIMEOUT, elapsed
    payload = parent_conn.recv() if parent_conn.poll() else None
    try:
        parent_conn.close()
    except Exception:
        pass
    if payload is None:
        return ERROR, elapsed
    if isinstance(payload, str) and payload.startswith(ERROR):
        return ERROR, elapsed
    return payload, elapsed


def start_one(solver_name: str, ot: SmallLambda, ctx):
    """Start a solver in its own process; return a live handle to poll."""
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker,
                       args=(solver_name, ot.n, ot.to_string(), child_conn))
    proc.start()
    child_conn.close()
    return {"proc": proc, "conn": parent_conn,
            "start": time.perf_counter(), "done": False,
            "outcome": None, "elapsed": None}


def poll_one(h, timeout: float) -> bool:
    """Poll a running handle once.  Returns True if it just finished/expired.

    Mirrors run_one's outcome logic but non-blocking: a process is done when it
    exits (read its message) or when it overruns the timeout (kill it).
    """
    if h["done"]:
        return False
    elapsed = time.perf_counter() - h["start"]
    proc = h["proc"]
    if not proc.is_alive():
        payload = h["conn"].recv() if h["conn"].poll() else None
        try:
            h["conn"].close()
        except Exception:
            pass
        if payload is None or (isinstance(payload, str) and payload.startswith(ERROR)):
            outcome = ERROR
        else:
            outcome = payload
        h.update(done=True, outcome=outcome, elapsed=elapsed)
        return True
    if elapsed >= timeout:
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        h.update(done=True, outcome=TIMEOUT, elapsed=elapsed)
        return True
    return False


# ---------------------------------------------------------------------------
# Plotting + summary
# ---------------------------------------------------------------------------

def plot_histogram(solver: str, times: list[float], out_path: str, timeout: float) -> None:
    import matplotlib.pyplot as plt  # Agg backend forced via MPLBACKEND at top

    fig, ax = plt.subplots(figsize=(7, 4))
    if times:
        ax.hist(times, bins=30, color="#3b7", edgecolor="black", alpha=0.8)
    ax.set_title(f"Realization runtime — solver '{solver}'  "
                 f"({len(times)} solved instances)")
    ax.set_xlabel("wall-clock time per instance [s]")
    ax.set_ylabel("number of instances")
    if times:
        mean = statistics.fmean(times)
        median = statistics.median(times)
        ax.axvline(mean, color="red", linestyle="--", label=f"mean {mean:.3f}s")
        ax.axvline(median, color="blue", linestyle=":", label=f"median {median:.3f}s")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def summarize(times: list[float]) -> dict:
    if not times:
        return {"count": 0}
    return {
        "count": len(times),
        "mean": statistics.fmean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "total": sum(times),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", default=None,
                        help="order-type file to benchmark (any format the reader "
                             "understands: b08, b16, blt, ...).  This is the primary "
                             "input; use it to realize your own / larger datasets. "
                             "If omitted, give --n, else defaults to otypes07.b08")
    parser.add_argument("--n", type=int, default=None,
                        help="convenience: benchmark the built-in order-type "
                             "database for this n (6..10; picks otypesNN.b08/.b16). "
                             "Ignored if a file is given.")
    parser.add_argument("--fmt", default=None,
                        help="file format override (default: inferred from extension)")
    parser.add_argument("--solvers", default="",
                        help="comma-separated solver names (default: all available)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="per-instance timeout in seconds (default: 30)")
    parser.add_argument("--sample", type=int, default=20,
                        help="randomly sample this many order types from the file "
                             "(default: 20; use 0 for ALL).  Reservoir-sampled, so "
                             "it does not just take the easy near-convex ones at the top")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="seed for the random sampling (default: 0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="take the FIRST N order types instead of sampling "
                             "(overrides --sample; note the first ones are easy)")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                        help="seed for the random point permutation applied to "
                             "each OT before realizing (difficulty depends on "
                             "labeling; DB order is near-convex/easy).  "
                             "Use -1 to keep the canonical labeling.")
    parser.add_argument("--jobs", type=int, default=0,
                        help="number of (order type, solver) jobs to run "
                             "concurrently in the pool (default: 0 = CPU count - 1). "
                             "Rows finish as soon as all their cells are done. "
                             "NOTE: with >1 job the solvers compete for CPU, so the "
                             "reported times are inflated — use --sequential (or "
                             "--jobs 1) for clean, isolated timings.")
    parser.add_argument("--sequential", "--no-parallel", dest="sequential",
                        action="store_true",
                        help="run one job at a time (clean isolated timings); "
                             "equivalent to --jobs 1")
    parser.add_argument("--plot", action="store_true",
                        help="also write a runtime histogram PNG per solver "
                             "(off by default; matplotlib is only imported then)")
    parser.add_argument("--outdir", default=os.path.join(_REPO_ROOT, "scripts",
                                                          "benchmark_plots"),
                        help="directory for the histogram PNGs and JSON summary")
    args = parser.parse_args(argv)

    # Resolve the input file: --file wins, else --n, else default n=7.
    if args.file:
        infile = args.file
    elif args.n is not None:
        infile = otdb_path_for_n(args.n)
        if infile is None:
            print(f"error: no order-type database file for n={args.n} "
                  f"(looked in tests/otdb/otypes/)", file=sys.stderr)
            return 2
    else:
        infile = os.path.join(_REPO_ROOT, "tests", "otdb", "otypes", "otypes07.b08")
    shuffle_seed = None if args.shuffle_seed == -1 else args.shuffle_seed

    ctx = mp.get_context("spawn")

    if args.solvers.strip():
        requested = [s.strip() for s in args.solvers.split(",") if s.strip()]
    else:
        # Default: every optimizer method individually (scipy.*, nlopt.*,
        # octave.*), plus the remaining realizers — one column per method for a
        # method-by-method comparison.  Compact family names (scipy/nlopt/octave)
        # are dropped since their per-method variants cover them.
        from pyotlib2.realization import SOLVERS
        per_method = [k for k in SOLVERS if "." in k]  # scipy.*, nlopt.*, octave.*, sage.*
        others = [s for s in available_solvers()
                  if s not in _EXCLUDED and s not in ("scipy", "nlopt", "octave", "sage")]
        requested = per_method + others
    requested = [s for s in requested if s not in _EXCLUDED]

    # availability probe
    solvers = []
    for s in requested:
        try:
            make_tester(s)
            solvers.append(s)
        except Exception as exc:  # noqa: BLE001
            print(f"[info] solver {s!r} unavailable: {exc}", file=sys.stderr)

    # --limit (first N) overrides --sample; --sample 0 means "all"
    sample = None if (args.limit is not None or args.sample == 0) else args.sample
    ots = load_abstract_ots(infile, args.fmt, args.limit, sample,
                            args.sample_seed, shuffle_seed)
    n = ots[0].n if ots else 0
    how = (f"{len(ots)} sampled" if sample is not None
           else f"first {len(ots)}" if args.limit is not None
           else f"all {len(ots)}")
    shuf = "shuffled" if shuffle_seed is not None else "canonical labeling"
    print(f"Benchmarking {how} order types (n={n}, {shuf}) from "
          f"{os.path.basename(infile)}")
    print(f"Solvers: {solvers}   timeout: {args.timeout}s\n")

    os.makedirs(args.outdir, exist_ok=True)

    # per-solver: list of solved-instance runtimes, and outcome counts
    times: dict[str, list[float]] = {s: [] for s in solvers}
    counts: dict[str, dict[str, int]] = {
        s: {OK: 0, FAIL: 0, TIMEOUT: 0, ERROR: 0} for s in solvers}

    # ---- live table: one row per order type, a running-average row below ----
    # Short, fixed-width column labels so the table stays aligned; scipy.* is
    # abbreviated to its method (e.g. "s.neldermead" -> "nelder").
    def _abbr(s: str) -> str:
        if s.startswith("scipy."):
            return s[len("scipy."):][:8]
        return s[:8]

    labels = [_abbr(s) for s in solvers]
    cw = max(8, max(len(l) for l in labels))  # per-solver column width
    idxw = len(str(len(ots)))

    def _cell(outcome: str, elapsed: float) -> str:
        if outcome == OK:
            return f"{elapsed:.2f}"
        return {TIMEOUT: "t.o.", FAIL: "-", ERROR: "ERR"}.get(outcome, "?")

    def _row(prefix: str, cells: list[str]) -> str:
        body = " ".join(c.rjust(cw) for c in cells)
        return f"{prefix:>{idxw}}  {body}"

    # header
    print(_row("#", [l.rjust(cw) for l in labels]))
    print("-" * (idxw + 2 + (cw + 1) * len(solvers)))

    tty = sys.stdout.isatty()

    # Full result matrix held in memory: matrix[i][j] is the cell for order type
    # i and solver j.  Every (order type, solver) pair is a job; we keep at most
    # `jobs` running at once and update the matrix as each finishes.  Rows are
    # printed in order the moment ALL their cells are done — so a slow cell in an
    # early row doesn't stop later rows from computing, but output stays ordered.
    R, C = len(ots), len(solvers)
    matrix = [["..."] * C for _ in range(R)]
    remaining_in_row = [C] * R
    next_row_to_print = 0

    if args.sequential:
        jobs = 1
    elif args.jobs and args.jobs > 0:
        jobs = args.jobs
    else:  # default / --jobs 0
        jobs = max(1, (os.cpu_count() or 2) - 1)

    pending = [(i, j) for i in range(R) for j in range(C)]  # submit order
    pending.reverse()  # pop() from the end -> submits (0,0),(0,1),... first
    running = {}       # (i,j) -> handle

    def _flush_ready_rows():
        nonlocal next_row_to_print
        while (next_row_to_print < R
               and remaining_in_row[next_row_to_print] == 0):
            i = next_row_to_print
            print(_row(str(i + 1), matrix[i]), flush=True)
            next_row_to_print += 1

    def _live_redraw():
        # On a TTY, show the current row being filled (the first not-yet-printed
        # row) in place, so the user sees live progress without scrolling.
        if tty and next_row_to_print < R:
            i = next_row_to_print
            sys.stdout.write("\r\033[2K" + _row(str(i + 1), matrix[i]))
            sys.stdout.flush()

    while pending or running:
        # fill the pool up to `jobs`
        while pending and len(running) < jobs:
            i, j = pending.pop()
            running[(i, j)] = start_one(solvers[j], ots[i], ctx)
        # poll running jobs once
        finished = []
        for (i, j), h in running.items():
            if poll_one(h, args.timeout):
                s = solvers[j]
                counts[s][h["outcome"]] += 1
                if h["outcome"] == OK:
                    times[s].append(h["elapsed"])
                matrix[i][j] = _cell(h["outcome"], h["elapsed"])
                remaining_in_row[i] -= 1
                finished.append((i, j))
        for key in finished:
            del running[key]
        if finished:
            if tty and next_row_to_print < R:
                sys.stdout.write("\r\033[2K")  # clear transient live row
            _flush_ready_rows()
            _live_redraw()
        else:
            time.sleep(0.05)
    if tty:
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

    # plots (only when explicitly requested; matplotlib imported lazily)
    if args.plot:
        print("\nWriting histograms:")
        for s in solvers:
            out_png = os.path.join(args.outdir, f"runtime_{s}_n{n}.png")
            plot_histogram(s, times[s], out_png, args.timeout)
            print(f"  {out_png}")

    # summary
    print("\n" + "=" * 78)
    print(f"SUMMARY — runtimes over {len(ots)} order types (n={n})")
    print("=" * 78)
    w = max(8, *(len(s) for s in solvers))
    header = f"{'solver':<{w}} {'success':>9} {'solved':>7} {'timeout':>8} " \
             f"{'fail':>6} {'mean[s]':>9} {'median':>9} {'max':>8} {'total[s]':>9}"
    print(header)
    print("-" * len(header))
    # Pareto ranking: success rate first (most important), then mean time.
    # Unsolved solvers (0%) therefore sink to the bottom instead of appearing as
    # spuriously "fast" 0-second entries.
    summary = {}
    for s in solvers:
        st = summarize(times[s])
        done = sum(counts[s].values())
        rate = (100.0 * counts[s][OK] / done) if done else 0.0
        summary[s] = {"outcomes": counts[s], "success_rate": rate, **st}

    def _rank_key(s):
        st = summary[s]
        mean = st["mean"] if st["count"] else float("inf")
        return (-st["success_rate"], mean)

    ranked = sorted(solvers, key=_rank_key)
    for s in ranked:
        st = summary[s]
        done = sum(counts[s].values())
        succ = f"{counts[s][OK]}/{done}"
        rate_txt = f"{st['success_rate']:.0f}%"
        if st["count"]:
            print(f"{s:<{w}} {succ:>4}{rate_txt:>5} {counts[s][OK]:>7} "
                  f"{counts[s][TIMEOUT]:>8} {counts[s][FAIL]:>6} "
                  f"{st['mean']:>9.4f} {st['median']:>9.4f} "
                  f"{st['max']:>8.3f} {st['total']:>9.2f}")
        else:
            print(f"{s:<{w}} {succ:>4}{rate_txt:>5} {counts[s][OK]:>7} "
                  f"{counts[s][TIMEOUT]:>8} {counts[s][FAIL]:>6} "
                  f"{'-':>9} {'-':>9} {'-':>8} {'-':>9}")

    print("\nBest = highest success rate, then fastest mean time (Pareto). "
          "Rows above are already in that order.")

    json_out = os.path.join(args.outdir, f"runtime_summary_n{n}.json")
    with open(json_out, "w") as fh:
        json.dump({"file": os.path.basename(infile), "n": n,
                   "num_ots": len(ots), "timeout": args.timeout,
                   "sampled": sample is not None,
                   "shuffled": shuffle_seed is not None,
                   "summary": summary}, fh, indent=2)
    print(f"\nJSON summary: {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

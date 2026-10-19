"""CLI command implementations for pyotlib2.

Each command is also importable as a regular Python function for use in
IPython / Jupyter notebooks.

Two conceptually different optimization modes
---------------------------------------------

**Coordinate minimization** — preserves the order type
    The abstract order type (all triple orientations) stays fixed.
    Only the concrete integer coordinates are changed.
    All combinatorial properties (crossings, k-gons, …) are invariant.

    Commands:  minimize-coords   beautify-coords (--method gd|nm)

**Property minimization / local search** — changes the order type
    The OT itself is modified (triple orientations may flip).
    The goal is to find a *different* OT with a better property value.

    walk-points   -- needs concrete coordinates; moves points in coordinate
                     space (DFS) or randomly (--random).
    walk-abstract -- purely abstract; flips exit-edge triples on the
                     chirotope (DFS or --random).

    Commands:  walk-points   walk-abstract
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Iterable, Iterator, Optional

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.io.readers import read_order_types
from pyotlib2.io.writers import write_order_types
from pyotlib2.cli.io_args import (
    add_input_args, add_output_args, open_input, resolve_output, progress_iter
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_all(sub) -> None:
    # OT-level
    _register_unify(sub)
    _register_lexmin(sub)
    _register_sort(sub)
    _register_shuffle(sub)
    _register_enum_subconf(sub)
    _register_count_subconf(sub)
    _register_find_subconf(sub)
    # PC-level
    _register_enum_projective(sub)
    _register_enum_natural(sub)
    # Properties
    _register_kgons(sub)
    _register_properties(sub)
    # Realization
    _register_realize(sub)
    _register_smart_realize(sub)
    _register_gp_test(sub)
    _register_minimize_coords(sub)
    _register_beautify_coords(sub)
    # Literal ports of old pyotlib PointSetOptimizer / scripts
    _register_minimize_integer_coords(sub)
    _register_beautify_pointset(sub)
    _register_beautify2(sub)
    _register_minimize_pattern_search(sub)
    # Local search / property optimization
    _register_walk_points(sub)
    _register_walk_abstract(sub)
    # Extension
    _register_extend_abstract(sub)
    _register_extend_random(sub)
    # Visualization
    _register_plot(sub)
    _register_editor(sub)


# ---------------------------------------------------------------------------
# unify
# ---------------------------------------------------------------------------

def unify_ot(
    ots: Iterable[SmallLambda],
    calc_lex_min: bool = True,
    validate: bool = True,
) -> Iterator[SmallLambda]:
    """Remove duplicate order types via lex-min deduplication."""
    from pyotlib2.algorithms.unify import unify as _unify
    yield from _unify(ots, calc_lex_min=calc_lex_min, validate=validate)


def _register_unify(sub) -> None:
    p = sub.add_parser(
        "unify",
        help="Remove duplicate order types (or projective classes with --projective)",
        description=(
            "Deduplicate order types by computing the lex-min representative "
            "of each and removing duplicates.  The output contains exactly one "
            "representative per distinct (abstract) order type.  "
            "With --projective, deduplicates at the level of projective classes "
            "(rank-3 oriented matroid isomorphism classes)."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".uniq")
    p.add_argument("--no-lexmin", action="store_false", dest="calc_lex_min",
                   help="Skip lex-min normalization")
    p.add_argument("--no-validate", action="store_false", dest="validate",
                   help="Skip collinearity check")
    p.add_argument("--projective", action="store_true",
                   help="Deduplicate projective classes instead of order types")
    p.set_defaults(func=_cmd_unify, calc_lex_min=True, validate=True, projective=False)


def _cmd_unify(args) -> None:
    if args.projective:
        ots_in = list(open_input(args))
        total = len(ots_in)
        unique = list(unify_pc(iter(ots_in)))
        out, fmt = resolve_output(args, ".uniq.lt")
        write_order_types(unique, out, fmt=fmt)
        print(f"total: {total}  unique PCs: {len(unique)}  written to: {out}")
    else:
        ots = list(open_input(args))
        total = len(ots)
        unique = list(unify_ot(iter(ots), calc_lex_min=args.calc_lex_min, validate=args.validate))
        out, fmt = resolve_output(args, ".uniq.lt")
        write_order_types(unique, out, fmt=fmt)
        print(f"total: {total}  unique: {len(unique)}  written to: {out}")


# ---------------------------------------------------------------------------
# lexmin
# ---------------------------------------------------------------------------

def lex_min_ot(ots: Iterable[SmallLambda]) -> Iterator[SmallLambda]:
    """Relabel each order type to its lex-min representative."""
    for ot in ots:
        yield ot.get_lex_min()


def _register_lexmin(sub) -> None:
    p = sub.add_parser(
        "lexmin",
        help="Relabel order types to lex-min representative (or PC representer with --projective)",
        description=(
            "Relabel each order type to its lexicographically minimal "
            "representative (canonical form under point relabeling).  "
            "With --projective, relabels to the PC representer of its "
            "projective class (rank-3 oriented matroid isomorphism class)."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".lexmin")
    p.add_argument("--projective", action="store_true",
                   help="Relabel to PC representer instead of OT lex-min")
    p.set_defaults(func=_cmd_lexmin, projective=False)


def _cmd_lexmin(args) -> None:
    if args.projective:
        ots = list(lex_min_pc(open_input(args)))
        out, fmt = resolve_output(args, ".lexmin.lt")
        write_order_types(ots, out, fmt=fmt)
        print(f"written: {len(ots)}  to: {out}")
    else:
        ots = list(lex_min_ot(open_input(args)))
        out, fmt = resolve_output(args, ".lexmin.lt")
        write_order_types(ots, out, fmt=fmt)
        print(f"written: {len(ots)}  to: {out}")


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------

def sort(ots: Iterable[SmallLambda]) -> list:
    """Return order types sorted lexicographically."""
    return sorted(ots)


def _register_sort(sub) -> None:
    p = sub.add_parser(
        "sort",
        help="Sort order types lexicographically",
        description="Sort order types into lexicographic order.",
    )
    add_input_args(p)
    add_output_args(p, ".sorted")
    p.set_defaults(func=_cmd_sort)


def _cmd_sort(args) -> None:
    ots = sort(open_input(args))
    out, fmt = resolve_output(args, ".sorted.lt")
    write_order_types(ots, out, fmt=fmt)
    print(f"written: {len(ots)}  to: {out}")


# ---------------------------------------------------------------------------
# shuffle
# ---------------------------------------------------------------------------

def shuffle(ots: Iterable[SmallLambda]) -> list:
    """Return order types in random order."""
    import random
    lst = list(ots)
    random.shuffle(lst)
    return lst


def _register_shuffle(sub) -> None:
    p = sub.add_parser(
        "shuffle",
        help="Shuffle order types randomly",
        description="Randomly permute the order of order types in the file.",
    )
    add_input_args(p)
    add_output_args(p, ".shuffled")
    p.set_defaults(func=_cmd_shuffle)


def _cmd_shuffle(args) -> None:
    ots = shuffle(open_input(args))
    out, fmt = resolve_output(args, ".shuffled.lt")
    write_order_types(ots, out, fmt=fmt)
    print(f"written: {len(ots)}  to: {out}")


# ---------------------------------------------------------------------------
# enum-sub-ot
# ---------------------------------------------------------------------------

def enum_sub_ot(
    ots: Iterable[SmallLambda],
    k: int,
    duplicates: bool = False,
) -> Iterator[SmallLambda]:
    """Yield all distinct k-point sub-order-types across all input OTs."""
    from pyotlib2.algorithms.sub_order_types import enumerate_sub_ots
    seen = set()
    for ot in ots:
        for sub in enumerate_sub_ots(ot, k, lex_min=not duplicates):
            key = sub.to_string()
            if duplicates or key not in seen:
                seen.add(key)
                yield sub


def _register_enum_subconf(sub) -> None:
    p = sub.add_parser(
        "enum-subconf",
        help="Enumerate k-point sub-configurations",
        description=(
            "Enumerate all distinct k-point sub-order-types appearing in "
            "the input.  Each k-element subset of points defines a sub-OT; "
            "duplicates are removed by lex-min normalization unless --duplicates is given."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".sub-ot")
    p.add_argument("-k", type=int, required=True, help="Sub-order-type size")
    p.add_argument("--duplicates", action="store_true",
                   help="Include duplicates (no lex-min deduplication)")
    p.set_defaults(func=_cmd_enum_subconf)


def _cmd_enum_subconf(args) -> None:
    subs = list(enum_sub_ot(open_input(args), args.k, duplicates=args.duplicates))
    out, fmt = resolve_output(args, ".sub-ot.lt")
    write_order_types(subs, out, fmt=fmt)
    print(f"sub-ots: {len(subs)}  written to: {out}")


# ---------------------------------------------------------------------------
# count-sub-ot
# ---------------------------------------------------------------------------

def count_sub_ot(
    ots: Iterable[SmallLambda],
    k: int,
) -> Iterator[tuple]:
    """Yield (ot, count) of distinct k-point sub-order-types per OT."""
    from pyotlib2.algorithms.sub_order_types import count_distinct_sub_ots
    for ot in ots:
        yield ot, count_distinct_sub_ots(ot, k)


def _register_count_subconf(sub) -> None:
    p = sub.add_parser(
        "count-subconf",
        help="Count distinct k-point sub-configurations per OT",
        description=(
            "For each input order type, count the number of distinct k-point "
            "sub-order-types it contains.  Prints one count per line."
        ),
    )
    add_input_args(p)
    p.add_argument("-k", type=int, required=True, help="Sub-order-type size")
    p.set_defaults(func=_cmd_count_subconf)


def _cmd_count_subconf(args) -> None:
    max_cnt = 0
    for ot, cnt in count_sub_ot(open_input(args), args.k):
        max_cnt = max(max_cnt, cnt)
        print(cnt)
    print(f"max: {max_cnt}", file=sys.stderr)


# ---------------------------------------------------------------------------
# find-sub-ot
# ---------------------------------------------------------------------------

def find_sub_ot(
    ots: Iterable[SmallLambda],
    references: Iterable[SmallLambda],
    k: int,
) -> Iterator[tuple]:
    """Yield (ot, matched_ref_indices) for OTs containing reference sub-OTs."""
    from pyotlib2.algorithms.sub_order_types import find_sub_ots as _find
    refs = list(references)
    yield from _find(ots, refs, k)


def _register_find_subconf(sub) -> None:
    p = sub.add_parser(
        "find-subconf",
        help="Find OTs containing specific sub-configurations",
        description=(
            "Filter the input to order types that contain at least one of "
            "the reference k-point sub-order-types given in --ref.  "
            "Prints each matching OT together with which reference indices it contains."
        ),
    )
    add_input_args(p)
    p.add_argument("--ref", required=True, help="Reference OT file")
    p.add_argument("--ref-fmt", default=None)
    p.add_argument("--ref-n", type=int, default=None)
    p.add_argument("-k", type=int, required=True)
    p.set_defaults(func=_cmd_find_subconf)


def _cmd_find_subconf(args) -> None:
    refs = list(read_order_types(args.ref, fmt=args.ref_fmt, n=args.ref_n))
    for ot, matched in find_sub_ot(open_input(args), refs, args.k):
        print(f"n={ot.n} slhash={ot.short_sl_hash(args.hash_length)} -> {matched}")


# ---------------------------------------------------------------------------
# unify-pc
# ---------------------------------------------------------------------------

def unify_pc(ots: Iterable[SmallLambda]) -> Iterator[SmallLambda]:
    """Yield one PC representer per projective equivalence class."""
    from pyotlib2.algorithms.projective_class import ProjectiveClass
    seen = set()
    for ot in ots:
        pc = ProjectiveClass(ot)
        key = pc.representer.to_string()
        if key not in seen:
            seen.add(key)
            yield pc.representer


# _register_unify_pc removed — functionality merged into `unify --projective`


# ---------------------------------------------------------------------------
# lex-min-pc
# ---------------------------------------------------------------------------

def lex_min_pc(ots: Iterable[SmallLambda]) -> Iterator[SmallLambda]:
    """Relabel each OT to the PC representer of its projective class."""
    from pyotlib2.algorithms.projective_class import ProjectiveClass
    for ot in ots:
        yield ProjectiveClass(ot).representer


# _register_lex_min_pc removed — functionality merged into `lexmin --projective`


# ---------------------------------------------------------------------------
# enum-pc
# ---------------------------------------------------------------------------

def enum_pc(ots: Iterable[SmallLambda]) -> Iterator[SmallLambda]:
    """Yield all OTs in the projective class of each input OT."""
    from pyotlib2.algorithms.projective_class import ProjectiveClass
    seen_pcs = set()
    for ot in ots:
        pc = ProjectiveClass(ot)
        key = pc.representer.to_string()
        if key in seen_pcs:
            continue
        seen_pcs.add(key)
        yield from pc.all_ots()


def _register_enum_projective(sub) -> None:
    p = sub.add_parser(
        "enum-projective",
        help="Enumerate all OTs in each projective class",
        description=(
            "For each input order type, enumerate all order types in its "
            "projective class (rank-3 oriented matroid isomorphism class) "
            "via flip-graph BFS.  Duplicate PCs are skipped."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".pc-full")
    p.set_defaults(func=_cmd_enum_projective)


def _cmd_enum_projective(args) -> None:
    ots = list(enum_pc(open_input(args)))
    out, fmt = resolve_output(args, ".pc-full.lt")
    write_order_types(ots, out, fmt=fmt)
    print(f"written: {len(ots)}  to: {out}")


# ---------------------------------------------------------------------------
# enum-natural  — enumerate all natural-labeled variants of each OT
# ---------------------------------------------------------------------------

def enum_natural(
    ots: Iterable[SmallLambda],
    mirror: bool = True,
) -> Iterator[SmallLambda]:
    """Yield all distinct natural-labeled variants of each input OT.

    For each hull point p0, the OT is relabeled so that p0 is first and
    the remaining points appear in CW angular order around p0.  If mirror
    is True, each variant is also mirrored (l-matrix transposed), doubling
    the candidate set.  Duplicates (same to_string()) are removed.

    The total number of distinct variants equals the size of the OT's
    symmetry group under natural-labeling equivalence.
    """
    for ot in ots:
        seen = set()
        for lab in ot.get_natural_labelings():
            if None in lab:
                continue
            variants = [ot.relabeled(lab)]
            if mirror:
                variants.append(variants[0].mirrored())
            for v in variants:
                key = v.to_string()
                if key not in seen:
                    seen.add(key)
                    yield v


def _register_enum_natural(sub) -> None:
    p = sub.add_parser(
        "enum-natural",
        help="Enumerate all natural-labeled variants of each OT",
        description=(
            "For each input order type, enumerate all distinct relabelings "
            "obtained by choosing each hull point as anchor p0 (natural labeling) "
            "and optionally mirroring.  "
            "This gives the orbit of the OT under the symmetry group of "
            "natural-labeling equivalence.  "
            "Use --no-mirror to suppress mirrored variants."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".natural")
    p.add_argument("--no-mirror", action="store_false", dest="mirror", default=True,
                   help="Do not include mirrored variants")
    p.set_defaults(func=_cmd_enum_natural)


def _cmd_enum_natural(args) -> None:
    ots = list(enum_natural(open_input(args), mirror=args.mirror))
    out, fmt = resolve_output(args, ".natural.lt")
    write_order_types(ots, out, fmt=fmt)
    print(f"written: {len(ots)}  to: {out}")


# ---------------------------------------------------------------------------
# kgons
# ---------------------------------------------------------------------------

def kgons(
    ots: Iterable[SmallLambda],
    k: int,
    empty_only: bool = True,
) -> Iterator[tuple]:
    """Yield (ot, count) of empty/convex k-gons."""
    from pyotlib2.algorithms.polygon_count import count_polygons
    for ot in ots:
        bl = ot.to_big_lambda()
        yield ot, count_polygons(bl, k, empty_only=empty_only)


def _register_kgons(sub) -> None:
    p = sub.add_parser(
        "kgons",
        help="Count empty/convex k-gons",
        description=(
            "Count empty convex k-gons (k-holes) in each order type.  "
            "By default counts only empty k-gons (no interior points); "
            "use --convex to count all convex k-gons.  "
            "Use --enumerate to list the actual point index tuples."
        ),
    )
    add_input_args(p)
    p.add_argument("-k", type=int, required=True, help="Polygon size")
    p.add_argument("--convex", action="store_false", dest="empty_only",
                   help="Count all convex k-gons (not only empty ones)")
    p.add_argument("--enumerate", action="store_true",
                   help="Print each polygon found")
    p.set_defaults(func=_cmd_kgons, empty_only=True)


def _cmd_kgons(args) -> None:
    from pyotlib2.algorithms.polygon_count import enumerate_polygons, count_polygons
    for ot in open_input(args):
        bl = ot.to_big_lambda()
        if args.enumerate:
            for poly in enumerate_polygons(bl, args.k, empty_only=args.empty_only):
                print(poly)
        else:
            print(count_polygons(bl, args.k, empty_only=args.empty_only))


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------

def properties(
    ots: Iterable[SmallLambda],
    props: list,
) -> Iterator[tuple]:
    """Yield (ot, {prop: value}) for each OT.

    Supported properties:
      crossings            rectilinear crossing number
      empty-triangles      number of empty triangles
      empty-kgons-K        number of empty convex K-gons  (e.g. empty-kgons-5)
      kgons-K              number of convex K-gons
      hull                 number of convex hull points
      onion-layers         number of convex layers
      triangulations       number of triangulations
    """
    from pyotlib2.algorithms.polygon_count import (
        count_polygons, count_crossings, count_triangles
    )
    for ot in ots:
        bl = ot.to_big_lambda()
        values = {}
        for prop in props:
            if prop == "crossings":
                values[prop] = count_crossings(ot)
            elif prop == "empty-triangles":
                values[prop] = count_triangles(bl, empty_only=True)
            elif prop.startswith("empty-kgons-"):
                k = int(prop.split("-")[-1])
                values[prop] = count_polygons(bl, k, empty_only=True)
            elif prop.startswith("kgons-"):
                k = int(prop.split("-")[-1])
                values[prop] = count_polygons(bl, k, empty_only=False)
            elif prop == "hull":
                values[prop] = len(ot.get_extremal_points())
            elif prop == "onion-layers":
                values[prop] = len(bl.get_onion())
            elif prop == "triangulations":
                from pyotlib2.algorithms.triangulations import count_triangulations
                values[prop] = count_triangulations(ot)
            else:
                raise ValueError(f"Unknown property: {prop!r}")
        yield ot, values


def _register_properties(sub) -> None:
    p = sub.add_parser(
        "properties",
        help="Compute combinatorial properties of order types",
        description=(
            "Compute one or more combinatorial properties for each order type "
            "and print them space-separated.  "
            "Available properties:\n"
            "  crossings       rectilinear crossing number\n"
            "  empty-triangles number of empty triangles\n"
            "  empty-kgons-K   number of empty convex K-gons (e.g. empty-kgons-5)\n"
            "  kgons-K         number of convex K-gons\n"
            "  hull            number of convex hull points\n"
            "  onion-layers    number of convex layers\n"
            "  triangulations  number of triangulations"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_input_args(p)
    p.add_argument("props", nargs="+", metavar="PROPERTY",
                   help="Properties: crossings, empty-triangles, empty-kgons-K, "
                        "kgons-K, hull, onion-layers, triangulations")
    p.set_defaults(func=_cmd_properties)


def _cmd_properties(args) -> None:
    for ot, values in properties(open_input(args), args.props):
        print(" ".join(str(values[p]) for p in args.props))


# ---------------------------------------------------------------------------
# realize
# ---------------------------------------------------------------------------

def realize(
    ots: Iterable[SmallLambda],
    method: str = "grid",
    trials: int = 20,
    gridsize: int = 256,
    grid_tries: int = 20,
    use_pc: bool = False,
    max_pc_trials: int = 0,
    on_large_n=None,
    solver_kwargs: Optional[dict] = None,
) -> Iterator[tuple]:
    """Test realizability of abstract order types (direct QCP, no etherealization).

    This runs the chosen solver DIRECTLY on each order type.  It is intended for
    small n; for larger / harder order types use ``smart_realize`` (incremental
    etherealization).  See :func:`smart_realize`.

    Yields (ot, result) where result is True/False/None (unknown).

    method: "grid" for randomized grid search (fast, good for small n);
            "scipy" for nonlinear optimization;
            "gp" for GLPK GP-LP (proves non-realizability).

    use_pc: if True, search through the projective class of each OT and test
            each member until one is realized or all (up to max_pc_trials) tried.

    on_large_n: optional callback ``(n) -> bool`` invoked before realizing an OT;
            if it returns False the OT is skipped (yielded as unknown).  Used by
            the CLI to warn once about direct realization at large n.
    """
    from pyotlib2.realization.base import Undecided
    from pyotlib2.realization import make_tester

    # Any registered solver name works (grid/scipy/z3/gp and the per-method
    # families scipy.*, nlopt.*, octave.*, sage.*).  Grid and scipy take extra
    # tuning params; everything else uses registry defaults.
    # Extra solver knobs (e.g. precision=, logsum=) forwarded to make_tester;
    # drop any a given tester does not accept (not every solver has --precision).
    extra = _accepted_solver_kwargs(method, dict(solver_kwargs or {}))
    if method == "grid":
        tester = make_tester("grid", gridsize=gridsize, max_tries=grid_tries)
    elif method == "scipy":
        tester = make_tester("scipy", trials=trials, **extra)
    else:
        tester = make_tester(method, **extra)

    for ot in ots:
        if on_large_n is not None and not on_large_n(ot.n):
            yield ot, None  # user declined direct realization at large n
            continue
        if not use_pc:
            try:
                yield ot, tester.is_realizable(ot)
            except Undecided:
                yield ot, None
        else:
            from pyotlib2.algorithms.projective_class import ProjectiveClass
            pc = ProjectiveClass(ot)
            result = None
            it = 0
            for pc_ot in pc.all_ots():
                it += 1
                if max_pc_trials and it > max_pc_trials:
                    break
                try:
                    res = tester.is_realizable(pc_ot)
                    if res is True:
                        # attach realization back to original ot via pc transform
                        if pc_ot.realization is not None:
                            ot.realization = pc_ot.realization
                        result = True
                        break
                    elif res is False:
                        result = False
                        break
                except Undecided:
                    pass
            yield ot, result


def _register_realize(sub) -> None:
    p = sub.add_parser(
        "realize",
        help="Test realizability of order types",
        description=(
            "Test whether each abstract order type is realizable by a "
            "concrete point set.  "
            "Method 'grid' uses randomized backtracking on an integer grid (fast, good for small n); "
            "method 'scipy' uses nonlinear optimization; "
            "method 'gp' uses a Grassmann-Plücker LP via GLPK (proves non-realizability).  "
            "Use --pc to search through the projective class of each OT."
        ),
    )
    add_input_args(p)
    p.add_argument("--method", default="grid",
                   help="Realization solver (default: grid).  " + _solver_help())
    p.add_argument("--trials", type=int, default=20,
                   help="Number of optimization trials (scipy only)")
    p.add_argument("--precision", type=int, default=0,
                   help="Working-precision mantissa bits for arbitrary-precision "
                        "solvers (mpmath, mathematica); 0 = the solver default. "
                        "Ignored by solvers without a precision knob.")
    p.add_argument("--logsum", action="store_true", default=False,
                   help="Use the log-sum penalty sum log(1+viol) instead of the "
                        "squared hinge (continuous solvers only).")
    p.add_argument("--gridsize", type=int, default=256,
                   help="Grid side length, power of 2 (grid method, default: 256)")
    p.add_argument("--grid-tries", type=int, default=20,
                   help="Independent restarts for grid search (default: 20)")
    p.add_argument("--pc", action="store_true", default=False,
                   help="Search through the projective class of each OT")
    p.add_argument("--max-pc-trials", type=int, default=0,
                   help="Max PC members to try per OT (0 = all, only with --pc)")
    p.add_argument("--export-lp", metavar="FILE", default=None,
                   help="With --method gp: write only the Grassmann-Plücker LP "
                        "to FILE (runnable with 'glpsol --lp FILE' or Gurobi) "
                        "and do NOT solve.  A per-OT index/hash is appended for "
                        "multiple inputs.")
    p.add_argument("--export-mathematica", metavar="FILE", default=None,
                   help="With --method mathematica: write only the Mathematica "
                        "script (cold start) to FILE (runnable with "
                        "'wolframscript -file FILE') and do NOT solve.")
    p.set_defaults(func=_cmd_realize)


# n above which directly solving the realization QCP is discouraged: as n grows
# most order types are NOT realizable and realizations may need huge coordinates
# (see arXiv:2007.15334), so smart-realize / non-realizability tests are wiser.
_DIRECT_REALIZE_WARN_N = 15
_direct_realize_warned = [False]


def _warn_large_n_direct_realize(n: int) -> bool:
    """Warn once (with a Y/N prompt) when directly realizing a large-n OT.

    Returns True to proceed, False to abort.  Only prompts once per process and
    only on an interactive terminal; in non-interactive use it proceeds silently.
    """
    if n <= _DIRECT_REALIZE_WARN_N or _direct_realize_warned[0]:
        return True
    _direct_realize_warned[0] = True
    msg = (
        f"warning: direct realization requested for n={n} (> {_DIRECT_REALIZE_WARN_N}).\n"
        "  As n grows most order types are non-realizable and realizations may\n"
        "  need very large coordinates (arXiv:2007.15334).  Blindly solving the\n"
        "  QCP is unwise here — prefer 'smart-realize' (incremental, n>10) and\n"
        "  keep non-realizability in mind (methods 'gp' / z3 can certify it)."
    )
    print(msg, file=sys.stderr)
    if not sys.stdin.isatty():
        return True  # non-interactive (scripts, pipes): proceed without prompting
    try:
        answer = input("  continue with direct realization anyway? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _export_path(base: str, ot: SmallLambda, count: int, hash_length: int) -> str:
    """Per-OT export path: ``base`` for a single OT, else ``base`` + the OT hash.

    The order-type's short hash is appended before the suffix so multiple input
    OTs each get a distinct, content-identifying file.
    """
    from pathlib import Path
    if count == 1:
        return base
    p = Path(base)
    return str(p.with_name(f"{p.stem}_{ot.short_sl_hash(hash_length)}{p.suffix}"))


def _export_lp(ots: list, base: str, hash_length: int) -> None:
    """Write the Grassmann-Plücker LP of each OT to a file (do not solve)."""
    from pyotlib2.realization.gp_tester import GPRealizationTester
    # Build the tester WITHOUT running __init__ (which requires glpsol on PATH);
    # we only need its pure formatter _write_lp, not the binary.
    exporter = GPRealizationTester.__new__(GPRealizationTester)
    for ot in ots:
        path = _export_path(base, ot, len(ots), hash_length)
        exporter._write_lp(ot, path)
        print(f"exported LP: n={ot.n} slhash={ot.short_sl_hash(hash_length)} -> {path}")


def _export_mathematica(ots: list, base: str, hash_length: int,
                        precision: int = 0, logsum: bool = False) -> None:
    """Write the Mathematica NMinimize script of each OT to a file (do not solve).

    ``precision`` (>0) sets the exported ``WorkingPrecision``; ``logsum`` switches
    the objective to the log-sum penalty — so the exported script matches what the
    solver would run with the same ``--precision`` / ``--logsum`` flags.
    """
    from pyotlib2.realization.mathematica_tester import MathematicaRealizationTester
    from pyotlib2.realization.qcp import RealizationQCP
    # Build the tester WITHOUT running __init__ (which requires wolframscript on
    # PATH); we only need its pure formatter write_script, not the binary.
    exporter = MathematicaRealizationTester.__new__(MathematicaRealizationTester)
    exporter.precision = int(precision) if precision else 0
    exporter.logsum = bool(logsum)
    exporter.method = "NelderMead"
    for ot in ots:
        path = _export_path(base, ot, len(ots), hash_length)
        qcp = RealizationQCP(ot, logsum=exporter.logsum)
        exporter.write_script(path, path + ".out", qcp, z0=None)
        print(f"exported Mathematica script: n={ot.n} "
              f"slhash={ot.short_sl_hash(hash_length)} -> {path}")


def _cmd_realize(args) -> None:
    export_lp = getattr(args, "export_lp", None)
    export_math = getattr(args, "export_mathematica", None)
    if export_lp or export_math:
        ots = list(open_input(args))
        if export_lp:
            if args.method != "gp":
                print("warning: --export-lp is for --method gp; exporting the "
                      "Grassmann-Plücker LP anyway.", file=sys.stderr)
            _export_lp(ots, export_lp, args.hash_length)
        if export_math:
            if args.method != "mathematica":
                print("warning: --export-mathematica is for --method mathematica; "
                      "exporting the script anyway.", file=sys.stderr)
            _export_mathematica(ots, export_math, args.hash_length,
                                precision=getattr(args, "precision", 0),
                                logsum=getattr(args, "logsum", False))
        return

    solver_kwargs = {}
    if getattr(args, "precision", 0):
        solver_kwargs["precision"] = args.precision
    if getattr(args, "logsum", False):
        solver_kwargs["logsum"] = True

    results = {True: 0, False: 0, None: 0}
    for ot, res in realize(
        open_input(args),
        method=args.method,
        trials=args.trials,
        gridsize=args.gridsize,
        grid_tries=args.grid_tries,
        use_pc=args.pc,
        max_pc_trials=args.max_pc_trials,
        on_large_n=_warn_large_n_direct_realize,
        solver_kwargs=solver_kwargs,
    ):
        results[res] += 1
        tag = "realizable" if res is True else ("non-realizable" if res is False else "unknown")
        print(f"{tag}: n={ot.n} slhash={ot.short_sl_hash(args.hash_length)}")
    print(f"\nrealizable: {results[True]}  non-realizable: {results[False]}  unknown: {results[None]}")


# ---------------------------------------------------------------------------
# realize-pc  — search through projective class for a realization
# ---------------------------------------------------------------------------

def realize_pc(
    ots: Iterable[SmallLambda],
    method: str = "scipy",
    trials: int = 20,
    max_pc_trials: int = 0,
) -> Iterator[tuple]:
    """Test realizability by searching through the projective class.

    For each OT, iterates over OTs in its PC and tests each with the given
    method until one is realized or all (up to max_pc_trials) are exhausted.

    Yields (ot, result) where result is True/False/None.
    max_pc_trials=0 means try all OTs in the PC.
    """
    from pyotlib2.algorithms.projective_class import ProjectiveClass
    from pyotlib2.realization.base import Undecided

    if method == "gp":
        from pyotlib2.realization.gp_tester import GPRealizationTester
        tester = GPRealizationTester()
    elif method == "scipy":
        from pyotlib2.realization.scipy_tester import ScipyRealizationTester
        tester = ScipyRealizationTester(trials=trials)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    for ot in ots:
        pc = ProjectiveClass(ot)
        result = None
        it = 0
        for pc_ot in pc.all_ots():
            it += 1
            if max_pc_trials and it > max_pc_trials:
                break
            try:
                res = tester.is_realizable(pc_ot)
                if res is True:
                    result = True
                    break
                elif res is False:
                    result = False
                    break
            except Undecided:
                pass
        yield ot, result


def _register_realize_pc(sub) -> None:
    p = sub.add_parser(
        "realize-pc",
        help="Test realizability by searching through the projective class",
        description=(
            "Test realizability by trying all order types in the projective class "
            "(rank-3 oriented matroid isomorphism class) of each input OT.  "
            "Succeeds as soon as any member of the PC is realized or proven non-realizable.  "
            "Use --max-pc-trials to limit how many PC members are tried."
        ),
    )
    add_input_args(p)
    p.add_argument("--method", default="scipy", choices=["scipy", "gp"])
    p.add_argument("--trials", type=int, default=20,
                   help="Optimization trials per OT (scipy only)")
    p.add_argument("--max-pc-trials", type=int, default=0,
                   help="Max OTs to try per PC (0 = all)")
    p.set_defaults(func=_cmd_realize_pc)


def _cmd_realize_pc(args) -> None:
    results = {True: 0, False: 0, None: 0}
    for ot, res in realize_pc(open_input(args), method=args.method,
                               trials=args.trials, max_pc_trials=args.max_pc_trials):
        results[res] += 1
        tag = "realizable" if res is True else ("non-realizable" if res is False else "unknown")
        print(f"{tag}: n={ot.n} slhash={ot.short_sl_hash(args.hash_length)}")
    print(f"\nrealizable: {results[True]}  non-realizable: {results[False]}  unknown: {results[None]}")


# ---------------------------------------------------------------------------
# smart-realize  — Etherealization: reduce point by point
# ---------------------------------------------------------------------------

def _onion_labeling(ot: SmallLambda, rng=None) -> list:
    """Phase 0 labeling: onion layers from outermost to innermost.

    The outermost convex-hull layer comes first, the single innermost point
    (or the last remaining points) comes last.  ``labeling[j]`` is the index
    (in the original ``ot``) of the point that becomes point ``j`` after the
    reordering ``ot.select_points(labeling)``.

    If ``rng`` is given, the points *within* each onion layer are shuffled,
    keeping the outer→inner growth order while varying the insertion order.
    """
    onion = ot.to_big_lambda().get_onion()
    labeling: list = []
    for layer in reversed(onion):
        layer = list(layer)
        if rng is not None:
            rng.shuffle(layer)
        labeling.extend(layer)
    for i in range(ot.n):        # safety: append any point the onion missed
        if i not in labeling:
            labeling.append(i)
    return labeling


def _phase0_labeling(ot: SmallLambda, order: str, rng) -> list:
    """Phase-0 point ordering for etherealization.

    ``order`` selects the strategy — the insertion order is the main lever for
    getting past a layer where a fixed order dead-ends:

      * ``onion``          outer→inner convex layers (deterministic; default)
      * ``onion-shuffled`` onion, shuffled within each layer (needs a seed)
      * ``random``         a fully random permutation (needs a seed)
      * ``identity``       the input order, unchanged
    """
    if order == "identity":
        return list(range(ot.n))
    if order == "random":
        lab = list(range(ot.n))
        (rng or __import__("random").Random()).shuffle(lab)
        return lab
    if order == "onion-shuffled":
        return _onion_labeling(ot, rng=rng)
    if order == "onion":
        return _onion_labeling(ot, rng=None)
    raise ValueError(f"unknown phase-0 order {order!r}")


def smart_realize(
    ot: SmallLambda,
    solver: str = "scipy",
    kmin: int = 4,
    use_pc: bool = False,
    max_pc_trials: int = 1,
    verbose: bool = True,
    seed: Optional[int] = None,
    order: str = "onion",
    export_after_phase1: Optional[str] = None,
    export_trace: Optional[str] = None,
    solver_kwargs: Optional[dict] = None,
) -> tuple[bool | None, SmallLambda | None]:
    """Etherealization: realize an OT by incrementally growing the point set.

    Faithful reconstruction of the pyotlib1 Etherealization algorithm.

    Phase 0
        Relabel the points by onion layers (outermost first, innermost last)
        so the numerically hardest inner points are added last.

    Phase 1 (incremental grow + warm start)
        Realize the first ``kmin`` points, then grow one point at a time.  To
        realize ``k+1`` points the ``k``-point realization is attached as
        ``sub_ot.subrealization`` (warm start) before the tester runs.  If a
        step fails, phase 2 is entered on that failing configuration.

    Phase 2 (depth search to a minimal non-realizable subconfig)
        Given an ``m``-point config that failed, try to realize every
        ``(m-1)``-point subconfig obtained by dropping one of the *first*
        ``m-1`` points (the newest/last point is kept).  Recurse into any
        subconfig that itself fails.  If *all* ``(m-1)``-subconfigs are
        realizable yet the ``m``-config is not, it is a Grassmann-Plücker
        candidate: the GP-LP is asked to certify non-realizability.  If GP
        proves it (returns False) a minimal non-realizable example is
        reported; otherwise the result is "unknown" (a retry point).

    Parameters
        solver          comma-separated realize chain, e.g. ``"scipy"`` or
                        ``"scipy,z3"`` (first tried first, rest as fallback).
        kmin            size of the first configuration realized outright.
        use_pc          wrap the realize chain in ``PCRealizationTester``.
        max_pc_trials   PC members to try per OT (only with ``use_pc``).
        verbose         print per-step progress.
        export_trace    if given, a folder path into which every distinct sub-OT
                        encountered in phase 2 (and the phase-1 stall) is written
                        as a ``.lt`` file named by its ``short_sl_hash()``
                        (de-duplicated), each with a ``#`` header recording the
                        original point indices that induce it.

    Returns ``(result, witness)``:
        result=True  → witness = the realized OT with ``.realization`` set,
                       mapped back to the ORIGINAL input labeling.
        result=False → witness = the minimal non-realizable sub-OT (GP
                       certified).
        result=None  → unknown / "I don't know" (retry point).
    """
    import numpy as np
    from pyotlib2.core.point_set import PointSet
    from pyotlib2.realization import make_chain
    from pyotlib2.realization.base import MAX_REALIZABLE_N

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    def plog(phase: int, m: int, h: str = "", status: str = "",
             decision: str = "") -> None:
        """Compact one-line progress: [ts]  pN  m/n  hash  status  decision."""
        if not verbose:
            return
        from pyotlib2.logging import log_fields
        log_fields([f"p{phase}", f"{m}/{ot.n}", h, status, decision],
                   widths=[2, 6, 6, 14, 0])

    # ------------------------------------------------------------------
    # Build the realize chain (GP is used SEPARATELY as the phase-2 witness)
    # ------------------------------------------------------------------
    names = [s.strip() for s in solver.split(",") if s.strip()]
    if not names:
        names = ["scipy"]
    # Build the fallback chain, passing `seed` only to testers that accept it
    # (z3 / gp have no seed).  seed=None makes the stochastic solvers use fresh
    # randomness each run, so a run that stalls can make progress on a retry.
    from pyotlib2.realization import make_tester as _make_tester
    # Generic solver knobs (precision=, logsum=) forwarded per solver, filtered
    # to what each tester's constructor accepts (grid/z3/gp have no such knob).
    extra_all = dict(solver_kwargs or {})
    realizer = None
    for name in reversed(names):
        kw = _accepted_solver_kwargs(name, {**extra_all, "seed": seed})
        try:
            realizer = _make_tester(name, parent=realizer, **kw)
        except TypeError:
            # last-resort: drop everything optional if a tester still rejects it
            realizer = _make_tester(name, parent=realizer)
    # Projective-class substitution happens at PHASE 0 (below): the whole input
    # OT is replaced by a random, projectively-equivalent member which is often
    # less contorted, then etherealized, and the realization is lifted back.  It
    # is NOT wrapped per growth step (that would re-enumerate the class at every
    # k — slow and pointless).
    _pc_back = None
    _pc_original = None
    if use_pc:
        from pyotlib2.realization.pc_tester import PCRealizationTester
        _pc = PCRealizationTester(realizer, shuffle=True, seed=seed)
        member, _pc_back = _pc.pick_member(ot)
        plog(0, ot.n, ot.short_sl_hash(),
             f"PC member {member.short_sl_hash()}", "etherealize member")
        _pc_original = ot         # remember the true input OT
        ot = member               # etherealize the (hopefully easier) member

    # Lazily-created GP tester for phase-2 non-realizability witnesses.
    _gp_cache: list = []

    def _gp_tester():
        if not _gp_cache:
            try:
                from pyotlib2.realization.gp_tester import GPRealizationTester
                _gp_cache.append(GPRealizationTester())
            except Exception as exc:  # glpsol missing etc.
                log(f"  [gp] unavailable: {exc}")
                _gp_cache.append(None)
        return _gp_cache[0]

    # A robust grid fallback: any order type on <= 9 points IS realizable, and
    # grid backtracking finds it reliably.  We use it whenever the main chain
    # fails, so a numerically-misleading warm start can never produce a false
    # "not realizable" for small configurations.
    from pyotlib2.realization.grid_search import GridSearchTester
    _grid_fallback = GridSearchTester(max_tries=40)

    def _realize(sub_ot: SmallLambda, subreal: Optional[list]) -> Optional[bool]:
        """Try to realize ``sub_ot``, hardened against a misleading warm start.

        Order of attempts:
          1. the chosen realize chain WITH the warm start (fast when it works);
          2. the chain again WITHOUT the warm start (a bad/tight warm start can
             mislead the optimizer into failing an otherwise trivial OT);
          3. the grid backtracking tester (reliable for small n).
        A ``False`` (proven non-realizable) from any tester is returned as-is.
        Only if every attempt fails to realize do we report failure — this
        avoids the spurious "not realizable" signals that corrupt phases 2/3.
        """
        for warm in (subreal, None):
            sub_ot.realization = None
            sub_ot.subrealization = warm
            try:
                res = realizer.is_realizable(sub_ot)
            except Exception:
                res = None
            if res is not None:
                return res  # True (realized) or False (proven non-realizable)
        # last resort: robust grid backtracking (no warm start needed)
        sub_ot.realization = None
        sub_ot.subrealization = None
        try:
            return _grid_fallback.is_realizable(sub_ot)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Phase 0: onion relabeling (reorder WITHOUT lex-min, carries realization)
    # ------------------------------------------------------------------
    n = ot.n
    # Phase-0 insertion order (the main lever for getting past order-specific
    # dead-ends); randomized variants need a seed to differ run to run.
    import random as _random
    _lab_rng = _random.Random(seed) if seed is not None else _random.Random()
    labeling = _phase0_labeling(ot, order, _lab_rng)
    work = ot.select_points(labeling)          # point j of `work` == point labeling[j] of `ot`
    o_target = ot.to_big_lambda().o             # original chirotope (for final verification)
    plog(0, n, ot.short_sl_hash(), f"order={order}", str(labeling))

    def _witness(idxs: list) -> SmallLambda:
        """Sub-OT for the given work indices, tagged with the ORIGINAL point
        labels it was selected from (``.source_indices``) so a saver can record
        exactly which points of the input induce this sub-order-type."""
        w = work.select_points(idxs)
        w.source_indices = [labeling[j] for j in idxs]
        return w

    # Phase-2 trace export: when a folder is requested, every distinct sub-OT
    # encountered while drilling down is written there (de-duplicated by hash) so
    # each can be studied on its own, away from the input's directory.
    _trace_dir = None
    _trace_seen: set = set()
    if export_trace:
        os.makedirs(export_trace, exist_ok=True)
        _trace_dir = export_trace

    def _trace(sub_ot: SmallLambda) -> None:
        if _trace_dir is None:
            return
        # Every order type on <= MAX_REALIZABLE_N points is realizable, so a
        # sub-OT that small is never an interesting non-realizability witness --
        # don't clutter the trace folder with it.  The trace only ever holds
        # configurations with n > MAX_REALIZABLE_N.
        if sub_ot.n <= MAX_REALIZABLE_N:
            return
        h = sub_ot.short_sl_hash()
        if h in _trace_seen:
            return
        _trace_seen.add(h)
        out = os.path.join(_trace_dir, f"{h}.lt")
        header = [f"smart-realize trace sub-OT  (n={sub_ot.n}, slhash={h})"]
        src = getattr(sub_ot, "source_indices", None)
        if src is not None:
            header.append(f"induced by original point indices: {src}")
        write_order_types([sub_ot], out, fmt="slt", header=header)

    kmin = max(3, min(kmin, n))

    # ------------------------------------------------------------------
    # Phase 2: depth search for a MINIMAL non-realizable subconfiguration.
    # Returns the indices (into `work`) of the smallest configuration whose
    # every (m-1)-subconfiguration realizes but which itself does not, or None
    # if no such minimal candidate is isolated on this branch.
    # ------------------------------------------------------------------
    def phase2(indices: list, subreal: list, depth: int = 0) -> Optional[list]:
        m = len(indices)
        cfg = work.select_points(indices)
        all_subs_realizable = True
        for i in range(m - 1):
            # drop the i-th of the first m-1 points, keep the newest (last)
            sub_idx = indices[:m - 1]
            sub_idx = sub_idx[:i] + sub_idx[i + 1:]
            sub_idx = sub_idx + [indices[-1]]
            sub_sr = subreal[:i] + subreal[i + 1:]

            sub_ot = work.select_points(sub_idx)
            _trace(_witness(sub_idx))
            res = _realize(sub_ot, sub_sr)
            # Sanity: every order type on <= MAX_REALIZABLE_N points IS
            # realizable.  If the realizer cannot realize a sub-OT that small,
            # the realizer -- not the order type -- is at fault; stop hard
            # rather than silently drilling below n=9.
            if res is not True and sub_ot.n <= MAX_REALIZABLE_N:
                raise RuntimeError(
                    f"smart-realize: realizer failed on an n={sub_ot.n} sub-order "
                    f"type (slhash={sub_ot.short_sl_hash()}), but every order type "
                    f"on <= {MAX_REALIZABLE_N} points is realizable -- the realizer "
                    f"is too weak or broken.  Try a stronger --solver.")
            plog(2, m - 1, sub_ot.short_sl_hash(),
                 "realizable" if res is True else "NOT realizable",
                 "drop -> ok" if res is True else "recurse")
            if res is True:
                continue
            # this (m-1)-subconfig is itself not realizable → recurse deeper
            all_subs_realizable = False
            deeper = phase2(sub_idx, sub_sr, depth + 1)
            if deeper is not None:
                return deeper
            # else keep looking through the other drops

        if not all_subs_realizable:
            # some subconfig failed to realize but no *minimal* candidate was
            # isolated below it (likely realizer weakness, not a true minimal
            # non-realizable example).
            plog(2, m, cfg.short_sl_hash(), "no candidate", "backtrack")
            return None

        # All (m-1)-subconfigs realizable but the m-config is not → minimal
        # non-realizable candidate.  GP falsification happens in phase 3.
        plog(2, m, cfg.short_sl_hash(), "min candidate", "-> phase 3")
        return list(indices)

    # ------------------------------------------------------------------
    # Phase 3: GP test + climb.  Start at the minimal candidate and test the
    # Grassmann-Plücker LP.  If it certifies non-realizability → done (False).
    # If inconclusive, climb back UP along the original trace (re-add the
    # removed points in order, then continue k+1..n) and test GP at each step,
    # until GP fires somewhere.  If GP never fires up to the full n-point set
    # AND no realization was found, report "I don't know" — a strong sign the
    # realizer is too weak (a realization may exist but wasn't found), since the
    # GP LP then cannot fire.
    # ------------------------------------------------------------------
    def phase3(cand_indices: list) -> tuple[bool | None, SmallLambda | None]:
        gp = _gp_tester()
        # Build the climb trace: candidate indices first, then re-add every
        # remaining work point (in work order = phase-1 growth order) up to n.
        present = set(cand_indices)
        trace_tail = [j for j in range(n) if j not in present]
        climb = list(cand_indices)
        steps = [list(climb)] + [climb + trace_tail[:t + 1]
                                 for t in range(len(trace_tail))]
        for idxs in steps:
            cfg = work.select_points(idxs)
            m = len(idxs)
            if gp is None:
                plog(3, m, cfg.short_sl_hash(), "GP unavailable", "cannot certify")
                return None, _witness(idxs)
            try:
                gp_res = gp.is_realizable(cfg)
            except Exception:
                gp_res = None
            if gp_res is False:
                plog(3, m, cfg.short_sl_hash(), "GP non-realizable",
                     "=> NON-REALIZABLE")
                # the *minimal* witness is the candidate we started from
                return False, _witness(cand_indices)
            plog(3, m, cfg.short_sl_hash(), "GP inconclusive", "climb")
        # Reached the full n-point set without GP ever firing.
        plog(3, n, "", "GP exhausted", "I DON'T KNOW (realizer too weak?)")
        return None, _witness(cand_indices)

    # ------------------------------------------------------------------
    # Phase 1: incremental grow with warm start
    # ------------------------------------------------------------------
    prev_real: Optional[list] = None
    for k in range(kmin, n + 1):
        indices = list(range(k))
        sub_ot = work.select_points(indices)
        subreal = prev_real if k > kmin else None
        res = _realize(sub_ot, subreal)
        if res is True:
            prev_real = list(sub_ot.realization)
            plog(1, k, sub_ot.short_sl_hash(), "realizable", "grow")
            continue
        # Failed to realize k points → phase 2 (find minimal candidate), then
        # phase 3 (GP test + climb).
        plog(1, k, sub_ot.short_sl_hash(), "NOT realizable", "-> phase 2")
        _trace(_witness(indices))  # the phase-1 stall config
        if export_after_phase1:
            # Write the k-point sub-OT where phase 1 stalled, for manual study.
            stall = _witness(indices)
            out = f"{export_after_phase1}.phase1stall_k{k}_{stall.short_sl_hash()}.lt"
            write_order_types([stall], out, fmt="slt", header=[
                f"phase-1 stall at k={k}/{n}  (order={order}, seed={seed})",
                f"slhash = {stall.short_sl_hash()}",
                f"original point indices: {stall.source_indices}",
            ])
            plog(1, k, stall.short_sl_hash(), "exported", out)
        if prev_real is None:
            # No warm start available (failed at kmin); still enter phase2.
            prev_real = []
        cand_indices = phase2(indices, prev_real)
        if cand_indices is None:
            # No minimal candidate isolated → nothing to certify.  This usually
            # means the realizer is too weak (it failed to realize something
            # that has realizable sub-parts but no clean minimal witness).
            plog(2, k, sub_ot.short_sl_hash(), "no candidate",
                 "I DON'T KNOW (realizer too weak?)")
            return None, _witness(indices)
        return phase3(cand_indices)

    # ------------------------------------------------------------------
    # Success: map the n-point realization back to the ORIGINAL labeling.
    # ------------------------------------------------------------------
    # `work` point j corresponds to original point labeling[j]; place the
    # coordinate of work-point j at original index labeling[j].
    coords_in_work_order = prev_real
    orig_coords: list = [None] * n
    for j in range(n):
        orig_coords[labeling[j]] = coords_in_work_order[j]

    ps = PointSet(n, orig_coords)
    if np.array_equal(ps.to_small_lambda(lazy=False).to_big_lambda().o, o_target):
        # Shrink the (often huge) optimizer coordinates FIRST — small integers
        # keep the flip/relabel back-transform chain numerically robust and the
        # final realization readable.
        realized_ot = SmallLambda(ot.n, ot.get_l().copy(), realization=orig_coords)
        try:
            realized_ot = minimize_coords(realized_ot)
        except Exception:
            pass
        final_coords = list(realized_ot.realization)

        # If we etherealized a projective-class member, lift its (now small)
        # realization back onto the true input order type.
        if _pc_back is not None:
            lifted = _pc_back(final_coords)
            if lifted is None:
                plog(1, n, ot.short_sl_hash(), "PC back-transform FAILED",
                     "I DON'T KNOW")
                return None, _witness(list(range(n)))
            realized_ot = SmallLambda(_pc_original.n, _pc_original.get_l().copy(),
                                      realization=lifted)
            try:
                realized_ot = minimize_coords(realized_ot)
            except Exception:
                pass
        plog(1, n, realized_ot.short_sl_hash(), "REALIZED", "verified vs original")
        return True, realized_ot

    # Back-mapping did not reproduce the original chirotope: report unknown
    # (the contract requires the original labeling).
    plog(1, n, ot.short_sl_hash(), "back-map FAILED", "I DON'T KNOW")
    return None, _witness(list(range(n)))


def _tester_class_for(method: str):
    """Return the tester CLASS for a solver name (no instantiation), or None.

    Used to inspect a tester's ``__init__`` signature so generic CLI knobs
    (``--precision`` / ``--logsum``) can be filtered to the testers that accept
    them, without needing the tester's external binary to be present.
    """
    fam = method.split(".", 1)[0]
    try:
        if fam == "scipy":
            from pyotlib2.realization.scipy_tester import ScipyMethodTester as C
        elif fam == "nlopt":
            from pyotlib2.realization.nlopt_tester import NloptMethodTester as C
        elif fam == "octave":
            from pyotlib2.realization.octave_tester import OctaveRealizationTester as C
        elif fam == "sage":
            from pyotlib2.realization.sage_tester import SageRealizationTester as C
        elif fam == "mathematica":
            from pyotlib2.realization.mathematica_tester import MathematicaRealizationTester as C
        elif fam == "mpmath":
            from pyotlib2.realization.mpmath_tester import MpmathRealizationTester as C
        elif fam == "z3":
            from pyotlib2.realization.z3_tester import Z3RealizationTester as C
        elif fam == "grid":
            from pyotlib2.realization.grid_search import GridSearchTester as C
        elif fam == "gp":
            from pyotlib2.realization.gp_tester import GPRealizationTester as C
        else:
            return None
        return C
    except Exception:
        return None


def _accepted_solver_kwargs(method: str, kw: dict) -> dict:
    """Keep only the kwargs that ``method``'s tester ``__init__`` accepts.

    Not every solver has a ``precision`` / ``logsum`` knob (grid, z3, gp don't),
    so a generic ``--precision`` / ``--logsum`` on the CLI must be dropped for
    testers that would otherwise raise ``TypeError``.
    """
    if not kw:
        return {}
    import inspect
    cls = _tester_class_for(method)
    if cls is None:
        return dict(kw)
    try:
        params = set(inspect.signature(cls.__init__).parameters)
    except (ValueError, TypeError):
        return dict(kw)
    return {k: v for k, v in kw.items() if k in params}


def _solver_help() -> str:
    """A compact listing of the available solvers, grouped by family."""
    from pyotlib2.realization import SOLVERS
    families: dict[str, list[str]] = {}
    for name in SOLVERS:
        fam = name.split(".", 1)[0]
        families.setdefault(fam, []).append(name)
    parts = []
    for fam in ("scipy", "nlopt", "octave", "sage", "mathematica", "mpmath",
                "z3", "grid", "gp"):
        members = sorted(families.get(fam, []))
        if not members:
            continue
        variants = [m.split(".", 1)[1] for m in members if "." in m]
        if variants:
            parts.append(f"{fam}.{{{','.join(variants)}}}")
        else:
            parts.append(fam)
    return "Available: " + "; ".join(parts) + "."


def _register_smart_realize(sub) -> None:
    p = sub.add_parser(
        "smart-realize",
        help="Realize OT point-by-point (etherealization); meant for n > 10",
        description=(
            "Attempt realization by growing the point set one point at a time "
            "in onion-layer order (etherealization), warm-starting each step "
            "from the previous realization.  If a step fails, a depth search "
            "(phase 2) drills down to a minimal non-realizable sub-configuration "
            "and the Grassmann-Plücker LP is asked to certify it.  "
            "The realize chain is chosen with --solver (e.g. 'scipy' or "
            "'scipy,z3'); GP is used only as the non-realizability witness.  "
            "This is intended for larger order types (n > 10); for small n the "
            "plain 'realize' command (direct QCP) is faster and sufficient."
        ),
    )
    add_input_args(p)
    p.add_argument("--solver", "--realization-tool", dest="solver", default="scipy",
                   help="Comma-separated realize chain, first tried first "
                        "(e.g. 'scipy' or 'scipy,z3'; default: scipy).  "
                        + _solver_help())
    p.add_argument("--kmin", type=int, default=4,
                   help="Size of the first configuration realized outright (default: 4)")
    p.add_argument("--precision", type=int, default=0,
                   help="Working-precision mantissa bits for arbitrary-precision "
                        "solvers (mpmath, mathematica); 0 = the solver default. "
                        "Ignored by solvers without a precision knob.")
    p.add_argument("--logsum", action="store_true", default=False,
                   help="Use the log-sum penalty sum log(1+viol) instead of the "
                        "squared hinge (continuous solvers only).")
    p.add_argument("--use-pc", action="store_true", default=False,
                   help="Wrap the realize chain in the projective-class tester")
    p.add_argument("--max-pc-trials", type=int, default=1,
                   help="Max PC members to try per OT (only with --use-pc)")
    p.add_argument("--order", default="onion",
                   choices=["onion", "onion-shuffled", "random", "identity"],
                   help="Phase-0 point insertion order (default: onion).  "
                        "The order is the main lever for getting past a layer "
                        "where growth dead-ends: 'onion' is deterministic, "
                        "'onion-shuffled'/'random' vary per --seed / --random-seed, "
                        "'identity' keeps the input order.")
    p.add_argument("--seed", type=int, default=None,
                   help="Fixed random seed for the insertion order and the "
                        "stochastic solvers (reproducible runs).")
    p.add_argument("--random-seed", "--true-random", dest="random_seed",
                   action="store_true",
                   help="Use a clock-based random seed each run, so a run that "
                        "stalls explores different starting points on a retry.")
    p.add_argument("--export-after-phase1", action="store_true",
                   help="When phase 1 stalls at some k, write that k-point "
                        "sub-order-type to a file (input path + a suffix) so it "
                        "can be studied by hand.")
    p.add_argument("--export-trace", action="store_true",
                   help="Save every distinct sub-order-type encountered while "
                        "drilling down (phase 2, and the phase-1 stall) into a "
                        "separate folder 'INPUT_trace/', one '.lt' file per "
                        "sub-OT named by its hash.  The folder is created only "
                        "when this flag is given.")
    p.add_argument("--output", "-o", dest="output", default=None,
                   help="Save the result: a realization is written as a point "
                        "set (.psz), a minimal non-realizable / last sub-OT is "
                        "written as a rank matrix (.lt) with a comment recording "
                        "which original point indices induce it.  For multiple "
                        "input OTs, an index is appended to the filename.")
    p.set_defaults(func=_cmd_smart_realize)


def _cmd_smart_realize(args) -> None:
    # Resolve the seed: --seed N fixes it (reproducible); --random-seed derives
    # a fresh one from the clock; otherwise None (the solvers use fresh entropy).
    seed = args.seed
    if getattr(args, "random_seed", False):
        import time
        seed = int(time.time() * 1000) & 0x7FFFFFFF
        print(f"using clock-based random seed: {seed}")

    # Identify order types by a short hash instead of dumping the full rank
    # matrix (which floods the screen for large n / many order types).
    ots = list(open_input(args))
    total = len(ots)
    print(f"loaded {total} order type{'s' if total != 1 else ''}")
    # Trace folder: 'INPUT_trace/' next to the input file, created only when
    # --export-trace is set (all OTs of a multi-OT input share the folder).
    trace_dir = None
    if getattr(args, "export_trace", False):
        trace_dir = f"{args.input}_trace"

    solver_kwargs = {}
    if getattr(args, "precision", 0):
        solver_kwargs["precision"] = args.precision
    if getattr(args, "logsum", False):
        solver_kwargs["logsum"] = True

    for i, ot in enumerate(ots, 1):
        print(f"\nOT #{i} of {total}  (n={ot.n}, slhash: {ot.short_sl_hash(args.hash_length)})")
        exp1 = args.input if getattr(args, "export_after_phase1", False) else None
        res, witness = smart_realize(
            ot,
            solver=args.solver,
            kmin=args.kmin,
            use_pc=args.use_pc,
            max_pc_trials=args.max_pc_trials,
            verbose=True,
            seed=seed,
            order=args.order,
            export_after_phase1=exp1,
            export_trace=trace_dir,
            solver_kwargs=solver_kwargs,
        )
        hl = args.hash_length
        if res is True:
            print("  => realizable")
            if witness is not None and witness.realization is not None:
                print(f"     coords: {list(witness.realization)}")
                _save_smart_result(args.output, i, total, res, witness, hl)
        elif res is False:
            print("  => non-realizable")
            if witness is not None:
                print(f"     minimal non-realizable sub-OT "
                      f"(n={witness.n}, slhash: {witness.short_sl_hash(hl)})")
                _save_smart_result(args.output, i, total, res, witness, hl)
        else:
            print("  => unknown / retry")
            if witness is not None:
                print(f"     last configuration "
                      f"(n={witness.n}, slhash: {witness.short_sl_hash(hl)})")
                _save_smart_result(args.output, i, total, res, witness, hl)


def _save_smart_result(output, i, total, res, witness, hl) -> None:
    """Write a smart-realize result to a file when ``--output`` was given.

    A realization is written as a point set (.psz, coordinates); a non-realizable
    or unknown sub-OT is written as a rank matrix (.lt) with a ``#`` comment
    recording which ORIGINAL point indices induce it (``source_indices``), so the
    sub-order-type can be reproduced from the input.
    """
    if not output:
        return
    from pathlib import Path
    from pyotlib2.io.writers import write_order_types

    base = Path(output)
    # append the OT index for multi-OT input
    stem = base.with_suffix("") if total == 1 else Path(f"{base.with_suffix('')}_{i}")

    verdict = ("realizable" if res is True
               else "non-realizable" if res is False else "unknown")
    src = getattr(witness, "source_indices", None)

    if res is True and witness.realization is not None:
        out = stem.with_suffix(".psz")
        header = [f"smart-realize: {verdict}", f"n = {witness.n}",
                  f"slhash = {witness.short_sl_hash(hl)}"]
        write_order_types([witness], str(out), fmt="psz", header=header)
    else:
        out = stem.with_suffix(".lt")
        header = [f"smart-realize: {verdict}", f"n = {witness.n}",
                  f"slhash = {witness.short_sl_hash(hl)}"]
        if src is not None:
            header.append(f"induced by original point indices: {src}")
        write_order_types([witness], str(out), fmt="slt", header=header)
    print(f"     saved: {out}")


# ---------------------------------------------------------------------------
# gp-test  — Grassmann-Plücker non-realizability test
# ---------------------------------------------------------------------------

def gp_test(ots: Iterable[SmallLambda]) -> Iterator[tuple]:
    """Test abstract order types for non-realizability using the GP-LP.

    Uses the Grassmann-Plücker linear program (via GLPK) to attempt to
    certify non-realizability.  A result of False means the OT is proven
    non-realizable.  A result of None means the test was inconclusive
    (the OT may or may not be realizable).

    Yields (ot, result) where result is False (non-realizable) or None (unknown).
    """
    from pyotlib2.realization.gp_tester import GPRealizationTester
    from pyotlib2.realization.base import Undecided
    tester = GPRealizationTester()
    for ot in ots:
        try:
            res = tester.is_realizable(ot)
            yield ot, res
        except Undecided:
            yield ot, None


def _register_gp_test(sub) -> None:
    p = sub.add_parser(
        "gp-test",
        help="Test non-realizability via Grassmann-Plücker LP",
        description=(
            "Attempt to certify non-realizability of abstract order types using "
            "the Grassmann-Plücker linear program (GP-LP) solved via GLPK.  "
            "A result of 'non-realizable' is a definitive proof.  "
            "'unknown' means the LP was inconclusive — the OT may still be "
            "realizable (use 'realize' to search for coordinates).  "
            "This is purely an abstract test: no coordinates are needed or produced."
        ),
    )
    add_input_args(p)
    p.add_argument("--export-lp", metavar="FILE", default=None,
                   help="Write only the Grassmann-Plücker LP to FILE (runnable "
                        "with 'glpsol --lp FILE' or Gurobi) and do NOT solve.  "
                        "A per-OT hash is appended for multiple inputs.")
    p.set_defaults(func=_cmd_gp_test)


def _cmd_gp_test(args) -> None:
    if getattr(args, "export_lp", None):
        _export_lp(list(open_input(args)), args.export_lp, args.hash_length)
        return

    results = {False: 0, None: 0}
    for ot, res in gp_test(open_input(args)):
        results[res] = results.get(res, 0) + 1
        tag = "non-realizable" if res is False else "unknown"
        print(f"{tag}: n={ot.n} slhash={ot.short_sl_hash(args.hash_length)}")
    print(f"\nnon-realizable: {results.get(False, 0)}  unknown: {results.get(None, 0)}")


# ---------------------------------------------------------------------------
# minimize-coords
# ---------------------------------------------------------------------------

def _require_realization(ot: SmallLambda, cmd: str) -> list:
    """Return realization coords or raise ValueError with helpful message."""
    if ot.realization is None:
        raise ValueError(
            f"OT has no realization — use a concrete input format "
            f"(b08, b16, asc, json) or run 'realize' first before '{cmd}'"
        )
    return list(ot.realization)


def _pts_get_bits(pts: list) -> int:
    return max((abs(v).bit_length() for p in pts for v in p), default=0)


def _pts_normalize(pts: list) -> list:
    """Translate so min x = min y = 0."""
    xmin = min(x for x, y in pts)
    ymin = min(y for x, y in pts)
    return [(x - xmin, y - ymin) for x, y in pts]


def _make_exit_validator(ot: SmallLambda):
    """Return a fast per-point validity checker using exit-edge triples."""
    from pyotlib2.algorithms.exit_edges import filter_exit_edges
    bl = ot.to_big_lambda()
    o = bl.o
    n = ot.n

    edges, witnesses = filter_exit_edges(ot, return_witnesses=True)

    # For each point i: list of (a, b, expected_sign) where (a,b) is an exit
    # edge with witness i, or i appears in the triple in any position.
    # We build combs[i] = list of (a, b, sign) to check when point i moves.
    combs: dict[int, list] = {i: [] for i in range(n)}
    for (a, b), ws in witnesses.items():
        for c in ws:
            sign = int(o[a, b, c])
            for idx in (a, b, c):
                combs[idx].append((a, b, c, sign))

    def is_valid(pts: list, i: int) -> bool:
        xi, yi = pts[i]
        for a, b, c, expected in combs[i]:
            xa, ya = pts[a]
            xb, yb = pts[b]
            xc, yc = pts[c]
            det = xa * (yb - yc) + xb * (yc - ya) + xc * (ya - yb)
            if expected > 0 and det <= 0:
                return False
            if expected < 0 and det >= 0:
                return False
        return True

    return is_valid


# ---------------------------------------------------------------------------
# minimize-coords
# ---------------------------------------------------------------------------

def minimize_coords(
    ot: SmallLambda,
    trials: int = 10,
    randomize: int = 10,
    tolerance: int = 0,
    sheering: bool = False,
) -> SmallLambda:
    """Minimize the coordinate size of a realized order type.

    Algorithm (ported from old pyotlib minimizeIntegerCoordinates):
      - Phase 1: move each point to even coordinates (4 candidate offsets)
      - Phase 2A: if all points could be made even, halve all coordinates
      - Phase 2B: if some point resists, randomize with Gaussian-like jumps
      - Mid-run blow-up (×3) if no improvement after trials/2 attempts
      - Optionally try sheering transforms x←x+y, y←y+x beforehand

    Only exit-edge triples are checked for orientation (O(n) per point
    instead of O(n²)), making this fast for large n.

    Requires ot.realization to be set.
    """
    import random
    from copy import copy

    pts = _require_realization(ot, "minimize-coords")
    n = ot.n
    is_valid = _make_exit_validator(ot)

    def get_bits(p):
        return _pts_get_bits(p)

    def move_to_even(pts, i):
        x, y = p0 = pts[i]
        if x % 2 == 0 and y % 2 == 0:
            return True
        for nx, ny in [
            (x + x % 2, y + y % 2),
            (x - x % 2, y + y % 2),
            (x + x % 2, y - y % 2),
            (x - x % 2, y - y % 2),
        ]:
            pts[i] = (nx, ny)
            if is_valid(pts, i):
                return True
        pts[i] = p0
        return False

    def randomize_point(pts, i, bits):
        x, y = p0 = pts[i]
        for e in range(bits, 0, -1):
            M = 2 ** e
            for _ in range(10):
                pts[i] = (x + random.randint(-M, M), y + random.randint(-M, M))
                if is_valid(pts, i):
                    return
        pts[i] = p0

    def balance(pts):
        """Scale axes so both have same range (improves convergence)."""
        xs = [x for x, y in pts]
        ys = [y for x, y in pts]
        xr = max(xs) - min(xs) or 1
        yr = max(ys) - min(ys) or 1
        mx = max(1, yr // xr)
        my = max(1, xr // yr)
        return [(x * mx, y * my) for x, y in pts]

    def run_minimize(pts):
        pts = _pts_normalize(pts)
        pts = balance(pts)
        best_pts = copy(pts)
        best_bits = get_bits(pts)
        it = 0
        while it < trials:
            ok = all(move_to_even(pts, i) for i in range(n))
            if ok:
                pts = [(x // 2, y // 2) for x, y in _pts_normalize(pts)]
                bits = get_bits(pts)
                if bits < best_bits:
                    best_bits = bits
                    best_pts = copy(pts)
                    it = 0
                    if tolerance and best_bits <= tolerance:
                        break
            else:
                bits = get_bits(pts)
                for _ in range(randomize):
                    for i in range(n):
                        randomize_point(pts, i, random.randint(0, max(1, bits)))
                it += 1
                if it == trials // 2:
                    # blow-up: scale ×3 to escape local minimum
                    pts = _pts_normalize(pts)
                    pts = balance([(x * 3, y * 3) for x, y in pts])
        return best_pts, best_bits

    # try plain + optional sheering variants
    candidates = [pts]
    if sheering:
        candidates += [
            [(x + y, y) for x, y in pts],
            [(x, y + x) for x, y in pts],
            [(x - y, y) for x, y in pts],
            [(x, y - x) for x, y in pts],
        ]

    best_pts = pts
    best_bits = _pts_get_bits(pts)
    for cand in candidates:
        p, b = run_minimize(list(cand))
        if b < best_bits:
            best_bits = b
            best_pts = p
        if tolerance and best_bits <= tolerance:
            break

    best_pts = _pts_normalize(best_pts)
    from pyotlib2.core.small_lambda import SmallLambda as SL
    result = SL(n, ot.get_l().copy(), realization=best_pts)
    return result


def _register_minimize_coords(sub) -> None:
    p = sub.add_parser(
        "minimize-coords",
        help="Minimize coordinate size of realized order types",
        description=(
            "Reduce the bit-width of the coordinates of a concrete point set "
            "realization while preserving the order type.  Uses the "
            "move-to-even / halve algorithm with randomized restarts and a "
            "mid-run blow-up.  Only exit-edge triples are checked per move "
            "(fast).  Optionally tries sheering transforms (x←x+y etc.) for "
            "better results.  Input must be a concrete format (b08/b16/asc/json)."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".min")
    p.add_argument("--trials", type=int, default=10,
                   help="Iteration budget (default: 10)")
    p.add_argument("--randomize", type=int, default=10,
                   help="Randomization rounds per failed phase (default: 10)")
    p.add_argument("--tolerance", type=int, default=0,
                   help="Stop when bits ≤ tolerance (default: 0 = minimize fully)")
    p.add_argument("--sheering", action="store_true",
                   help="Also try sheering transforms (x←x+y, y←y+x, …)")
    p.set_defaults(func=_cmd_minimize_coords)


def _cmd_minimize_coords(args) -> None:
    written = 0
    out, fmt = resolve_output(args, ".min.lt")
    results = []
    for ot in open_input(args):
        try:
            result = minimize_coords(
                ot,
                trials=args.trials,
                randomize=args.randomize,
                tolerance=args.tolerance,
                sheering=args.sheering,
            )
            b0 = _pts_get_bits(list(ot.realization)) if ot.realization else "?"
            b1 = _pts_get_bits(result.realization)
            print(f"bits: {b0} → {b1}")
            results.append(result)
            written += 1
        except ValueError as e:
            print(f"skip: {e}", file=sys.stderr)
    write_order_types(results, out, fmt=fmt)
    print(f"written: {written}  to: {out}")


# ---------------------------------------------------------------------------
# beautify-coords  — gradient descent via binary search on each coordinate
# ---------------------------------------------------------------------------

def beautify_coords(
    ot: SmallLambda,
    max_iter: int = 100,
    good_bits: int = 0,
) -> SmallLambda:
    """Beautify coordinates via gradient descent with binary search.

    For each coordinate, binary-searches the largest safe range [zl, zu]
    and moves to the midpoint.  Converges towards a "centred" realization
    with smaller spread.  Runs for at most max_iter iterations or until
    convergence.

    If good_bits > current bits, the point set is scaled up first so that
    scipy has more room to work.

    Ported from old pyotlib PointSetOptimizer.beautifyPointSet.
    Requires ot.realization to be set.
    """
    from fractions import Fraction
    from copy import copy

    pts = _require_realization(ot, "beautify-coords")
    n = ot.n
    pts = _pts_normalize(pts)
    cur_bits = _pts_get_bits(pts)

    # scale up if requested
    if good_bits and cur_bits < good_bits:
        scale = 2 ** (good_bits - cur_bits)
        pts = [(x * scale, y * scale) for x, y in pts]
        cur_bits = good_bits

    # build orientation lookup: only positive-oriented triples (a<b, a<c)
    bl = ot.to_big_lambda()
    o = bl.o
    orientations = [
        (a, b, c)
        for a in range(n) for b in range(n) for c in range(n)
        if a < b and a < c and int(o[a, b, c]) > 0
    ]
    # per-coordinate index: orientations involving coordinate i (x) or i+n (y)
    ori_by_coord = {k: [] for k in range(2 * n)}
    for a, b, c in orientations:
        for idx in (a, b, c):
            ori_by_coord[idx].append((a, b, c))
            ori_by_coord[idx + n].append((a, b, c))

    # Z = [x0, x1, ..., x_{n-1}, y0, y1, ..., y_{n-1}]
    Z = [x for x, y in pts] + [y for x, y in pts]

    def o3(Z, a, b, c):
        return (Z[a] * Z[b + n] - Z[a + n] * Z[b]
                + Z[b] * Z[c + n] - Z[b + n] * Z[c]
                + Z[c] * Z[a + n] - Z[c + n] * Z[a])

    def is_valid_coord(Z, coord):
        for a, b, c in ori_by_coord[coord]:
            if o3(Z, a, b, c) < 1:
                return False
        return True

    ot_str = ot.to_string()
    min_diff = 2 * n * 2 ** cur_bits
    stop_timer = 5

    for it in range(1, max_iter + 1):
        Z0 = copy(Z)
        grad = []

        for coord in range(2 * n):
            z0 = Z[coord]
            zl = zu = z0
            for bit in reversed(range(cur_bits + 1)):
                Z[coord] = zl - 2 ** bit
                if is_valid_coord(Z, coord):
                    zl = Z[coord]
                Z[coord] = zu + 2 ** bit
                if is_valid_coord(Z, coord):
                    zu = Z[coord]
            Z[coord] = z0
            grad.append((zl + zu) // 2 - z0)

        convergence = Fraction(10, 10 + it)
        grad_f = [g * convergence for g in grad]

        factor = Fraction(1)
        # backtrack until valid
        while factor > Fraction(1, 2 ** 20):
            for k in range(2 * n):
                Z[k] = Z0[k] + int(factor * grad_f[k])
            if all(o3(Z, a, b, c) > 0 for a, b, c in orientations):
                break
            factor /= 2
        else:
            Z = Z0
            break

        # try to extrapolate further (cap at 2^20 to avoid infinite loop when grad=0)
        df = Fraction(10, 9)
        while factor < Fraction(2 ** 20):
            factor *= df
            for k in range(2 * n):
                Z[k] = Z0[k] + int(factor * grad_f[k])
            if not all(o3(Z, a, b, c) > 0 for a, b, c in orientations):
                factor /= df
                break
        if factor > 1:
            factor = Fraction(1)
        for k in range(2 * n):
            Z[k] = Z0[k] + int(factor * grad_f[k])

        diffs = sum(abs(Z[k] - Z0[k]) for k in range(2 * n))
        if diffs < min_diff:
            min_diff = diffs
        else:
            stop_timer = 8
        stop_timer -= 1
        if diffs == 0 or stop_timer < 0:
            break

    # reconstruct point set
    new_pts = list(zip(Z[:n], Z[n:]))
    new_pts = _pts_normalize(new_pts)
    from pyotlib2.core.point_set import PointSet
    ps = PointSet(n, new_pts)
    if ps.to_small_lambda().to_string() == ot_str:
        from pyotlib2.core.small_lambda import SmallLambda as SL
        return SL(n, ot.get_l().copy(), realization=new_pts)
    # fallback: return original
    return ot


def _register_beautify_coords(sub) -> None:
    p = sub.add_parser(
        "beautify-coords",
        help="Beautify coordinates via gradient descent or Nelder-Mead",
        description=(
            "Optimize point set coordinates while preserving the order type.\n\n"
            "  --method gd  (default) Binary-search gradient descent: for each\n"
            "               coordinate, find the largest safe range and move to\n"
            "               its midpoint.  Fast, no external dependencies.\n\n"
            "  --method nm  Nelder-Mead simplex (scipy): minimizes a quality\n"
            "               function over exit-edge constraints and spread.\n"
            "               Convex hull points are fixed; result is rounded to\n"
            "               integers after each outer iteration.\n\n"
            "Input must be a concrete format (b08/b16/asc/json)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_input_args(p)
    add_output_args(p, ".beauty")
    p.add_argument("--method", choices=["gd", "nm"], default="gd",
                   help="Optimization method: gd=gradient descent, nm=Nelder-Mead (default: gd)")
    p.add_argument("--max-iter", type=int, default=100,
                   help="Max iterations for gd (default: 100)")
    p.add_argument("--iter1", type=int, default=5,
                   help="Outer iterations for nm (default: 5)")
    p.add_argument("--iter2", type=int, default=2000,
                   help="Max scipy evaluations per outer iter for nm (default: 2000)")
    p.add_argument("--good-bits", type=int, default=0,
                   help="Scale up to this bit-width before optimizing (default: 0 = no scale)")
    p.add_argument("--no-minimize", action="store_false", dest="minimize",
                   help="(nm only) Skip minimize-coords after each scipy run")
    p.add_argument("--norm", type=int, default=1,
                   help="(nm only) Lp norm for quality function (default: 1)")
    p.add_argument("--epsilon", type=float, default=1e-4,
                   help="(nm only) scipy convergence tolerance (default: 1e-4)")
    p.set_defaults(func=_cmd_beautify_coords, minimize=True)


def _cmd_beautify_coords(args) -> None:
    written = 0
    out, fmt = resolve_output(args, ".beauty.lt")
    results = []
    for ot in open_input(args):
        try:
            if args.method == "gd":
                result = beautify_coords(ot, max_iter=args.max_iter, good_bits=args.good_bits)
            else:
                result = beautify2_coords(
                    ot,
                    iter1=args.iter1,
                    iter2=args.iter2,
                    good_bits=args.good_bits,
                    minimize=args.minimize,
                    norm=args.norm,
                    epsilon=args.epsilon,
                )
            b0 = _pts_get_bits(list(ot.realization)) if ot.realization else "?"
            b1 = _pts_get_bits(result.realization)
            print(f"bits: {b0} → {b1}")
            results.append(result)
            written += 1
        except ValueError as e:
            print(f"skip: {e}", file=sys.stderr)
    write_order_types(results, out, fmt=fmt)
    print(f"written: {written}  to: {out}")


# ---------------------------------------------------------------------------
# beautify2-coords  — scipy Nelder-Mead with exit-edge constraint function
# ---------------------------------------------------------------------------

def beautify2_coords(
    ot: SmallLambda,
    iter1: int = 5,
    iter2: int = 2000,
    good_bits: int = 0,
    minimize: bool = True,
    norm: int = 1,
    epsilon: float = 1e-4,
) -> SmallLambda:
    """Beautify coordinates using scipy Nelder-Mead optimization.

    Quality function (minimized):
      error1 = (Σ 1/[o3(a,b,c)·sign / perimeter]^norm)^(1/norm)  [constraint]
      error2 = (Σ d²·Σ 1/d²)^(1/norm)                             [spread]
      error3 = |log(max|coord|)|                                    [scale]
      error  = error1 · error2 · error3

    Only exit-edge triples are used for error1 (much cheaper than all C(n,3)).
    Convex hull points are fixed during scipy optimization.
    After each scipy run the result is rounded to integers.
    Optionally followed by minimize_coords.

    Ported from old pyotlib PointSetOptimizer.beautify2.
    Requires ot.realization to be set.
    """
    import math
    from copy import copy

    try:
        import scipy.optimize
    except ImportError:
        raise RuntimeError("scipy is required for beautify2-coords: pip install scipy")

    pts = _require_realization(ot, "beautify2-coords")
    n = ot.n
    pts = _pts_normalize(pts)
    ot_str = ot.to_string()

    bl = ot.to_big_lambda()
    o = bl.o

    from pyotlib2.algorithms.exit_edges import filter_exit_edges
    edges, witnesses = filter_exit_edges(ot, return_witnesses=True)
    exit_trps = [(a, b, c) for (a, b), ws in witnesses.items() for c in ws]

    hull = ot.get_extremal_points()

    cur_bits = _pts_get_bits(pts)
    if good_bits and cur_bits < good_bits:
        scale = 2 ** (good_bits - cur_bits)
        pts = [(x * scale, y * scale) for x, y in pts]

    def quality(Z):
        # fix hull points
        for i in hull:
            Z[i] = Z0[i]
            Z[i + n] = Z0[i + n]
        X = Z[:n]
        Y = Z[n:]
        zmax = max(abs(z) for z in Z) or 1

        def o3(a, b, c):
            return (X[a] * Y[b] - Y[a] * X[b]
                    + X[b] * Y[c] - Y[b] * X[c]
                    + X[c] * Y[a] - Y[c] * X[a])

        def d2(a, b):
            return (X[a] - X[b]) ** 2 + (Y[a] - Y[b]) ** 2

        def perim(a, b, c):
            return math.sqrt(d2(a, b)) + math.sqrt(d2(b, c)) + math.sqrt(d2(c, a))

        F3 = []
        for a, b, c in exit_trps:
            sign = int(o[a, b, c])
            p = perim(a, b, c)
            v = sign * o3(a, b, c) / p if p > 0 else 0
            F3.append(v)

        if not F3 or min(F3) <= 0:
            return float("inf")

        D2 = [d2(a, b) for a in range(n) for b in range(a + 1, n)]
        if not D2 or min(D2) <= 0:
            return float("inf")

        error1 = sum(1.0 / v ** norm for v in F3) ** (1.0 / norm)
        error2 = (sum(v ** norm for v in D2) * sum(1.0 / v ** norm for v in D2)) ** (1.0 / norm)
        error3 = abs(math.log(zmax))
        return error1 * error2 * (error3 + 1e-9)

    def to_integer(Z):
        scale = max(abs(z) for z in Z) or 1
        factor = 2 ** 20 / scale
        return [int(round(z * factor)) for z in Z]

    for _it in range(iter1):
        X0, Y0 = zip(*pts)
        Z0 = list(X0) + list(Y0)

        sol = scipy.optimize.fmin(
            quality, Z0,
            xtol=epsilon, ftol=epsilon,
            maxiter=iter2, maxfun=50 * iter2,
            disp=False,
        )
        # restore hull
        for i in hull:
            sol[i] = Z0[i]
            sol[i + n] = Z0[i + n]

        Zi = to_integer(list(sol))
        new_pts = list(zip(Zi[:n], Zi[n:]))
        new_pts = _pts_normalize(new_pts)

        # verify OT preserved
        from pyotlib2.core.point_set import PointSet
        ps = PointSet(n, new_pts)
        if ps.to_small_lambda().to_string() == ot_str:
            pts = new_pts
            if minimize:
                from pyotlib2.core.small_lambda import SmallLambda as SL
                tmp = SL(n, ot.get_l().copy(), realization=pts)
                tmp = minimize_coords(tmp, trials=10, randomize=10)
                pts = list(tmp.realization)

    from pyotlib2.core.small_lambda import SmallLambda as SL
    final_pts = _pts_normalize(pts)
    ps = PointSet(n, final_pts)
    if ps.to_small_lambda().to_string() == ot_str:
        return SL(n, ot.get_l().copy(), realization=final_pts)
    return ot  # fallback


# ===========================================================================
# Literal ports of the old pyotlib coordinate optimizers (CLI wrappers)
#
#   minimize-integer-coords  <- scripts/pyotlib.MinimizeCoords.py (method=minimize)
#   beautify-pointset        <- scripts/pyotlib.BeautifyCoords.py
#   beautify2                <- scripts/pyotlib.Beautify2Script.py
#   minimize-pattern-search  <- scripts/pyotlib.MinimizePatternSearch.py
#
# Library code: pyotlib2.algorithms.coord_optimizer.  Defaults are the old
# scripts' defaults.  Independent of minimize-coords / beautify-coords above.
# ===========================================================================

def _literal_out(args, suffix: str) -> tuple:
    """Output path: -o if given, else input path + suffix + input extension
    (like the old PYOTScript, e.g. foo.psz -> foo.psz.min.psz)."""
    from pathlib import Path
    out, fmt = resolve_output(args, suffix)
    if not args.output:
        out = Path(str(args.input) + suffix + Path(args.input).suffix)
    return out, fmt


def _normalized_ots(ots: list) -> list:
    """The old PointSetTextWriter normalized realizations on write (min x = min y = 0)."""
    for ot in ots:
        if ot.realization is not None:
            ot.realization = _pts_normalize(list(ot.realization))
    return ots


def _register_minimize_integer_coords(sub) -> None:
    p = sub.add_parser(
        "minimize-integer-coords",
        help="Literal port of old pyotlib MinimizeCoords (minimizeIntegerCoordinates)",
        description=(
            "Minimize the coordinates of realized order types (order type "
            "preserved).  Literal port of old pyotlib "
            "scripts/pyotlib.MinimizeCoords.py with method=minimize, i.e. "
            "PointSetOptimizer.minimizeIntegerCoordinates.  Defaults are the "
            "old script's defaults.  If --tolerance > 0, order types still "
            "above the tolerance are also written to <output>.bad.psz, the "
            "others to <output>.good.psz."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".min")
    p.add_argument("--maxtrials", type=int, default=1,
                   help="Max. number of trials (default: 1)")
    p.add_argument("--tolerance", type=int, default=0,
                   help="Number of bits tolerated (not to reduce anymore) (default: 0)")
    p.add_argument("--randomize", type=int, default=10,
                   help="Number of randomization trials per phase (default: 10)")
    p.add_argument("--sheering", type=int, default=0,
                   help="Also try sheerings (slower) (default: 0)")
    p.add_argument("--minimize-diameter", type=int, default=0,
                   help="Minimize the diameter, not just the number of bits (default: 0)")
    p.add_argument("--symmetric", type=int, default=0,
                   help="k-fold rotational symmetry to keep (default: 0 = none)")
    p.add_argument("--prop", default="",
                   help="Property to keep when minimizing (old code asserts this off; default: none)")
    p.add_argument("--transformation", default="[1,0,0,1]",
                   help="Linear map [a,b,c,d]: (x,y) -> (a*x+b*y, c*x+d*y) applied first "
                        "(default: [1,0,0,1])")
    p.add_argument("--lazy", type=int, default=1,
                   help="Check cocircular points only between phases (default: 1)")
    p.add_argument("--debug", type=int, default=0,
                   help="1 = progress, 2 = more debug output (default: 0)")
    p.set_defaults(func=_cmd_minimize_integer_coords)


def _cmd_minimize_integer_coords(args) -> None:
    from ast import literal_eval
    from pyotlib2.algorithms import coord_optimizer
    from pyotlib2.algorithms.symmetry import compute_symmetric_ps
    from pyotlib2.core.point_set import PointSet

    transformation = literal_eval(args.transformation)
    coord_optimizer.DEBUG = args.debug - 1 if args.debug > 1 else False
    coord_optimizer.PROVIDE_ALL = False
    coord_optimizer.LAZY_CHECK_COCIRC = args.lazy != 0

    out, fmt = _literal_out(args, ".min")
    results, bad, good = [], [], []
    max_in_bits = max_out_bits = 0
    improved = not_improved = not_tolerated = 0
    stats: dict = {}
    count = 0
    if not args.debug:
        print("NOTE: use '--debug 1' to see progress or '--debug 2' for even more debug output")

    for OT in open_input(args):
        _require_realization(OT, "minimize-integer-coords")
        count += 1
        PS = PointSet(OT.n, OT.realization, OT.coloring)
        print("START", PS.points)

        start_bits = PS.bits()
        if start_bits > max_in_bits:
            max_in_bits = start_bits

        if start_bits > args.tolerance:
            if args.debug:
                print("Start diam:", PS.diameter())
            OTstr0 = OT.to_string()

            a, b, c, d = transformation
            PS.points = [(a * x + b * y, c * x + d * y) for (x, y) in PS.points]

            PS = coord_optimizer.minimize_integer_coordinates(
                PS, symmetric=args.symmetric, randomize=args.randomize,
                maxtrials=args.maxtrials, tolerance=args.tolerance, sheering=args.sheering,
                minimize_diameter=args.minimize_diameter, ensure_property=args.prop)

            if args.symmetric:
                PS.points = compute_symmetric_ps(PS.points, args.symmetric)

            if PS.diameter() < PointSet(OT.n, OT.realization).diameter():
                OT.realization = PS.points
            else:
                PS.points = OT.realization  # backup

            OTstr = PointSet(OT.n, OT.realization).to_small_lambda().to_string()
            assert OTstr == OTstr0

        end_bits = PS.bits()

        if end_bits > max_out_bits:
            max_out_bits = end_bits
        stats[end_bits] = stats.get(end_bits, 0) + 1

        if not args.symmetric:
            PS = PointSet(OT.n, OT.realization)
            PS = PS.normalized()
            OT.realization = PS.points

        print("END", PS.points)

        results.append(OT)
        if args.tolerance > 0:
            (bad if end_bits > args.tolerance else good).append(OT)

        if start_bits > end_bits:
            improved += 1
        else:
            not_improved += 1

        if args.tolerance and end_bits > args.tolerance:
            not_tolerated += 1

        if (args.tolerance and end_bits > args.tolerance) or (start_bits > end_bits):
            if args.debug:
                print("OT#", count, ":", start_bits, "->", end_bits)

    # old: the main writer normalizes unless symmetric; the bad/good writers always
    # normalize and are named after the input path (outfilepath + ".bad.psz").
    write_order_types(results if args.symmetric else _normalized_ots(results), out, fmt=fmt)
    if args.tolerance > 0:
        from copy import copy
        base = args.output or args.input
        write_order_types(_normalized_ots([copy(ot) for ot in bad]), base + ".bad.psz", fmt="psz")
        write_order_types(_normalized_ots([copy(ot) for ot in good]), base + ".good.psz", fmt="psz")

    print("done.")
    print("improved       :", improved)
    print("not improved   :", not_improved)
    print("max in  bits   :", max_in_bits)
    print("max out bits   :", max_out_bits)
    if args.tolerance:
        print("not tolerated  :", not_tolerated)
    print(80 * "-")
    print("statistic:")
    for i in stats:
        print("\t", i, "\t", stats[i], "\t", ((100 * stats[i]) // count), "%")
    print(f"written: {len(results)}  to: {out}")


def _register_beautify_pointset(sub) -> None:
    p = sub.add_parser(
        "beautify-pointset",
        help="Literal port of old pyotlib BeautifyCoords (beautifyPointSet)",
        description=(
            "Beautify realized order types (order type preserved) by the "
            "binary-search gradient method.  Literal port of old pyotlib "
            "scripts/pyotlib.BeautifyCoords.py / "
            "PointSetOptimizer.beautifyPointSet.  Defaults are the old "
            "script's defaults."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".beauty")
    p.add_argument("--maxiter", type=int, default=100,
                   help="Max. number of iterations (default: 100)")
    p.add_argument("--goodbits", type=int, default=0,
                   help="Number of bits to use (default: 0 = current bits)")
    p.set_defaults(func=_cmd_beautify_pointset)


def _cmd_beautify_pointset(args) -> None:
    from pyotlib2.algorithms import coord_optimizer
    from pyotlib2.core.point_set import PointSet

    goodbits = args.goodbits if args.goodbits != 0 else None
    out, fmt = _literal_out(args, ".beauty")
    results = []
    for OT in open_input(args):
        _require_realization(OT, "beautify-pointset")
        PS = PointSet(OT.n, OT.realization)
        slstr = PS.to_small_lambda().to_string()

        coord_optimizer.DEBUG = True
        PS = coord_optimizer.beautify_point_set(PS, args.maxiter, goodbits)

        assert slstr == PS.to_small_lambda().to_string()

        OT.realization = PS.points
        results.append(OT)
    write_order_types(_normalized_ots(results), out, fmt=fmt)
    print("done.")
    print(f"written: {len(results)}  to: {out}")


def _register_beautify2(sub) -> None:
    p = sub.add_parser(
        "beautify2",
        help="Literal port of old pyotlib Beautify2Script (beautify2, scipy)",
        description=(
            "Beautify realized order types (order type preserved) with "
            "scipy's Nelder-Mead (fmin).  Literal port of old pyotlib "
            "scripts/pyotlib.Beautify2Script.py / PointSetOptimizer.beautify2.  "
            "Defaults are the old script's defaults."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".beauty")
    p.add_argument("--minimize", type=int, default=1,
                   help="Run minimizeIntegerCoordinates before/after each scipy run (default: 1)")
    p.add_argument("--eps", type=float, default=0.0001,
                   help="Epsilon for convergence (xtol/ftol) (default: 0.0001)")
    p.add_argument("--norm", type=float, default=1,
                   help="Norm of the quality function (default: 1)")
    p.add_argument("--iter1", type=int, default=10,
                   help="Max number of iterations for outer loop (default: 10)")
    p.add_argument("--iter2", type=int, default=2000,
                   help="Max number of iterations for inner scipy fmin (default: 2000)")
    p.add_argument("--goodbits", type=int, default=30,
                   help="Scale point set up to this many bits (default: 30)")
    p.add_argument("--bound", type=int, default=0,
                   help="Diameter bound (default: 0 = none); requires 2^goodbits <= bound")
    p.set_defaults(func=_cmd_beautify2)


def _cmd_beautify2(args) -> None:
    from pyotlib2.algorithms import coord_optimizer

    if args.bound > 0:
        assert 2 ** args.goodbits <= args.bound
    out, fmt = _literal_out(args, f".beauty{args.iter1}-{args.iter2}")
    results = []
    for OT in open_input(args):
        _require_realization(OT, "beautify2")
        OT = coord_optimizer.beautify2(OT, args.iter1, args.iter2, goodbits=args.goodbits,
                                       minimize=args.minimize, diam_bound=args.bound,
                                       norm=args.norm, epsilon=args.eps)
        results.append(OT)
    write_order_types(_normalized_ots(results), out, fmt=fmt)
    print("done.")
    print(f"written: {len(results)}  to: {out}")


def _register_minimize_pattern_search(sub) -> None:
    p = sub.add_parser(
        "minimize-pattern-search",
        help="Literal port of old pyotlib MinimizePatternSearch",
        description=(
            "Try to minimize the diameter of realized order types (order type "
            "preserved) by an integer pattern search.  Literal port of old "
            "pyotlib scripts/pyotlib.MinimizePatternSearch.py.  Defaults are "
            "the old script's defaults.  As in the original, the --blowup "
            "budget is shared by all order types of the input file."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".min")
    p.add_argument("--maxtrials", type=int, default=2000,
                   help="Max. number of trials (default: 2000)")
    p.add_argument("--scale", type=int, default=1,
                   help="How to scale the point set first (default: 1)")
    p.add_argument("--blowup", type=int, default=0,
                   help="Number of 2x blow-ups allowed when stuck (default: 0)")
    p.add_argument("--tolerance", type=int, default=0,
                   help="Stop when bits <= tolerance (default: 0 = off)")
    p.add_argument("--debug", type=int, default=0,
                   help="Print progress every 10 iterations (default: 0)")
    p.set_defaults(func=_cmd_minimize_pattern_search)


def _cmd_minimize_pattern_search(args) -> None:
    from pyotlib2.algorithms import coord_optimizer

    out, fmt = _literal_out(args, ".min")
    results = []
    blowup = args.blowup
    for OT in open_input(args):
        _require_realization(OT, "minimize-pattern-search")
        OT, blowup = coord_optimizer.minimize_pattern_search(
            OT, maxtrials=args.maxtrials, scale=args.scale, blowup=blowup,
            tolerance=args.tolerance, DEBUG=args.debug)
        results.append(OT)
    write_order_types(_normalized_ots(results), out, fmt=fmt)
    print("done.")
    print(f"written: {len(results)}  to: {out}")




# ---------------------------------------------------------------------------
# plot  — visualize order types as point set drawings
# ---------------------------------------------------------------------------

def plot_ot(
    ot: SmallLambda,
    ax,
    *,
    title: Optional[str] = None,
    labels: bool = True,
    hull: bool = True,
    edges: bool = False,
    crossings: bool = False,
    point_size: int = 60,
) -> None:
    """Draw a single order type on a matplotlib Axes.

    If ot has a concrete realization it is used directly; otherwise a
    realization is computed via scipy (raises ValueError if it fails).

    Parameters
    ----------
    ot:         order type to draw
    ax:         matplotlib Axes to draw on
    title:      optional title string
    labels:     whether to annotate points with their index
    hull:       whether to draw the convex hull polygon
    edges:      whether to draw all n*(n-1)/2 connecting segments
    crossings:  highlight crossing pairs of segments (implies edges=True)
    point_size: scatter marker size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.collections import LineCollection
    except ImportError:
        raise RuntimeError(
            "matplotlib is required for plotting.\n"
            "Install with:  pip install 'pyotlib2[vis]'"
        )

    # --- get coordinates ---
    pts = ot.realization
    if pts is None:
        raise ValueError(
            "OT has no coordinates — use a concrete input format (b08, b16, asc, json)"
        )

    xs = [float(x) for x, y in pts]
    ys = [float(y) for x, y in pts]
    n = ot.n

    # --- normalize to [0,1]² for stable display ---
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    span = max(xhi - xlo, yhi - ylo) or 1.0
    margin = 0.12 * span
    xs_n = [(x - xlo) / span for x in xs]
    ys_n = [(y - ylo) / span for y in ys]

    bl = ot.to_big_lambda()

    # --- all edges (optional) ---
    if edges or crossings:
        segs = [((xs_n[i], ys_n[i]), (xs_n[j], ys_n[j]))
                for i in range(n) for j in range(i + 1, n)]
        if crossings:
            cross_pairs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(n):
                        for l in range(k + 1, n):
                            if len({i, j, k, l}) == 4 and bl.edges_cross((i, j), (k, l)):
                                cross_pairs.add((min(i,j), max(i,j)))
                                cross_pairs.add((min(k,l), max(k,l)))
            colors = [
                "#e74c3c" if (min(i,j), max(i,j)) in cross_pairs else "#bdc3c7"
                for i in range(n) for j in range(i + 1, n)
            ]
            lc = LineCollection(segs, colors=colors, linewidths=0.8, zorder=1)
        else:
            lc = LineCollection(segs, colors="#bdc3c7", linewidths=0.8, zorder=1)
        ax.add_collection(lc)

    # --- convex hull polygon ---
    if hull:
        hull_pts = ot.get_extremal_points()
        # order hull points by angle around centroid
        cx = sum(xs_n[i] for i in hull_pts) / len(hull_pts)
        cy = sum(ys_n[i] for i in hull_pts) / len(hull_pts)
        import math
        hull_pts_sorted = sorted(hull_pts,
                                 key=lambda i: math.atan2(ys_n[i] - cy, xs_n[i] - cx))
        hx = [xs_n[i] for i in hull_pts_sorted] + [xs_n[hull_pts_sorted[0]]]
        hy = [ys_n[i] for i in hull_pts_sorted] + [ys_n[hull_pts_sorted[0]]]
        ax.fill(hx, hy, alpha=0.08, color="#3498db", zorder=0)
        ax.plot(hx, hy, color="#3498db", linewidth=1.2, zorder=2)

    # --- points ---
    ax.scatter(xs_n, ys_n, s=point_size, color="#2c3e50", zorder=3)

    # --- labels ---
    if labels:
        offset = 0.04
        for i, (x, y) in enumerate(zip(xs_n, ys_n)):
            ax.annotate(
                str(i),
                (x, y),
                xytext=(x + offset, y + offset),
                fontsize=8,
                color="#2c3e50",
                zorder=4,
            )

    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, pad=4)


def plot(
    ots: Iterable[SmallLambda],
    output: Optional[str] = None,
    *,
    cols: int = 4,
    labels: bool = True,
    hull: bool = True,
    edges: bool = False,
    crossings: bool = False,
    point_size: int = 60,
    dpi: int = 150,
) -> None:
    """Plot order types as point set drawings.

    Arranges multiple OTs in a grid.  If output is None, opens an
    interactive matplotlib window.

    Parameters
    ----------
    ots:        iterable of order types
    output:     file path to save (png/pdf/svg/…); None → interactive
    cols:       number of columns in the grid
    labels:     annotate points with their index
    hull:       draw convex hull polygon
    edges:      draw all connecting segments
    crossings:  highlight crossing segment pairs (implies edges)
    point_size: scatter marker size
    dpi:        output resolution (for raster formats)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(
            "matplotlib is required for plotting.\n"
            "Install with:  pip install 'pyotlib2[vis]'"
        )
    import math

    ot_list = list(ots)
    if not ot_list:
        print("no order types to plot", file=sys.stderr)
        return

    n_ots = len(ot_list)
    ncols = min(cols, n_ots)
    nrows = math.ceil(n_ots / ncols)
    fig_w = 2.5 * ncols
    fig_h = 2.5 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                              squeeze=False)
    fig.patch.set_facecolor("white")

    for idx, ot in enumerate(ot_list):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        title = f"OT {idx+1}  (n={ot.n})"
        try:
            plot_ot(ot, ax, title=title, labels=labels, hull=hull,
                    edges=edges, crossings=crossings, point_size=point_size)
        except ValueError as e:
            ax.text(0.5, 0.5, f"[no realization]\n{e}",
                    ha="center", va="center", fontsize=7, color="gray",
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=9, pad=4)
            ax.axis("off")

    # hide unused axes
    for idx in range(n_ots, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout(pad=0.5)

    if output:
        plt.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"saved: {output}")
    else:
        plt.show()
    plt.close(fig)


def _register_plot(sub) -> None:
    p = sub.add_parser(
        "plot",
        help="Visualize order types as point set drawings",
        description=(
            "Plot order types as 2D point set drawings using matplotlib.  "
            "Requires a concrete input format with coordinates: b08, b16, asc, json.  "
            "Abstract formats (lt, blt) are not supported — realization is a hard problem.  "
            "Without -o, an interactive window is opened."
        ),
    )
    add_input_args(p)
    p.add_argument("-o", "--output", default=None,
                   help="Output file (png, pdf, svg, …). Default: interactive window.")
    p.add_argument("--cols", type=int, default=4,
                   help="Number of columns in the plot grid (default: 4)")
    p.add_argument("--no-labels", action="store_false", dest="labels",
                   help="Don't annotate points with their index")
    p.add_argument("--no-hull", action="store_false", dest="hull",
                   help="Don't draw the convex hull")
    p.add_argument("--edges", action="store_true",
                   help="Draw all connecting segments between points")
    p.add_argument("--crossings", action="store_true",
                   help="Highlight crossing segment pairs (implies --edges)")
    p.add_argument("--point-size", type=int, default=60,
                   help="Scatter marker size (default: 60)")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output resolution for raster formats (default: 150)")
    p.set_defaults(func=_cmd_plot, labels=True, hull=True)


def _cmd_plot(args) -> None:
    plot(
        open_input(args),
        output=args.output,
        cols=args.cols,
        labels=args.labels,
        hull=args.hull,
        edges=args.edges,
        crossings=args.crossings,
        point_size=args.point_size,
        dpi=args.dpi,
    )


# ---------------------------------------------------------------------------
# editor  — interactive PySide6 point set editor
# ---------------------------------------------------------------------------

def _register_editor(sub) -> None:
    p = sub.add_parser(
        "editor",
        help="Interactive point set editor (requires PySide6)",
        description=(
            "Open an interactive editor to drag points and explore order types.  "
            "Requires PySide6:  pip install -e '.[vis]'  "
            "With a file argument, loads the first order type from that file "
            "(must have concrete coordinates: b08, b16, asc, json).  "
            "Without a file, opens a default 6-point convex configuration."
        ),
    )
    p.add_argument("file", nargs="?", default=None,
                   help="Input file with point coordinates (optional)")
    p.add_argument("-n", dest="n", type=int, default=None,
                   help="Number of points (required for binary formats)")
    p.add_argument("--fmt", default=None,
                   help="File format (auto-detected from extension if omitted)")
    p.set_defaults(func=_cmd_editor)


def _cmd_editor(args) -> None:
    try:
        from pyotlib2.vis.editor.app import run_editor
    except ImportError:
        import sys
        print(
            "Error: PySide6 is required for the editor.\n"
            "Install with:  pip install -e '.[vis]'",
            file=sys.stderr,
        )
        sys.exit(1)
    run_editor(args.file, n=args.n, fmt=args.fmt)


# ===========================================================================
# Shared helper: property counting by name
# ===========================================================================

_PROP_HELP = (
    "  crossings        rectilinear crossing number\n"
    "  empty-triangles  number of empty triangles\n"
    "  empty-kgons-K    number of empty convex K-gons (e.g. empty-kgons-5)\n"
    "  kgons-K          number of convex K-gons\n"
    "  hull             number of convex hull points\n"
    "  onion-layers     number of convex layers\n"
    "  triangulations   number of triangulations"
)


def _count_property(ot: SmallLambda, prop: str) -> int:
    """Count a single named property on an order type."""
    from pyotlib2.algorithms.polygon_count import (
        count_polygons, count_crossings, count_empty_kgons,
    )
    if prop == "crossings":
        return count_crossings(ot)
    elif prop == "hull":
        return len(ot.get_extremal_points())
    elif prop == "onion-layers":
        return len(ot.big_lambda.get_onion())
    elif prop == "empty-triangles":
        return count_empty_kgons(ot, k=3)
    elif prop.startswith("empty-kgons-"):
        k = int(prop.removeprefix("empty-kgons-"))
        return count_empty_kgons(ot, k)
    elif prop.startswith("kgons-"):
        k = int(prop.removeprefix("kgons-"))
        return count_polygons(ot.big_lambda, k, empty_only=False)
    elif prop == "triangulations":
        from pyotlib2.algorithms.triangulations import count_triangulations
        return count_triangulations(ot)
    else:
        raise ValueError(f"Unknown property: {prop!r}. Available:\n{_PROP_HELP}")


def _output_path(input_path: str, suffix: str, index: int = 0) -> str:
    """Build output path: input.suffix or input.suffix.001 input.suffix.002 ..."""
    from pathlib import Path
    base = str(Path(input_path).with_suffix("")) + suffix
    if index == 0:
        return base
    return f"{base}.{index:03d}"


# ===========================================================================
# walk-points  — local search in coordinate space (DFS or random)
# ===========================================================================

def walk_points(
    ot: SmallLambda,
    prop: str,
    *,
    random_walk: bool = False,
    trials: int = 0,
    walk_range: int = 0,
    trace: int = 10000,
    good_bits: int = 30,
    max_steps: int = 0,
    stop_if_improved: bool = False,
    siman: bool = False,
    T0: float = 0.1,
    dT: float = 0.999,
    verbose: bool = True,
    maximize: bool = False,
) -> Iterator[SmallLambda]:
    """Local search in coordinate space to minimize a property
    (or to maximize it, with ``maximize=True``: the search then minimizes the
    negated value; simulated annealing only applies when minimizing).

    Two modes (selected via random_walk):

    DFS mode (random_walk=False):
        For each configuration on the stack: try moving each point in 16
        rational directions via binary search to the OT boundary, then step
        one unit past it to reach a neighboring OT.  Each unseen neighbor
        is pushed onto the DFS stack.

    Random mode (random_walk=True):
        At each step pick a random point and move it by a random offset in
        [-walk_range, +walk_range]².  Accept if the property does not
        increase (greedy descent, no backtracking).

    Both modes yield each strictly improving OT found.
    Requires ot.realization to be set.
    """
    import collections
    import random
    from math import exp
    from pyotlib2.core.point_set import PointSet

    def score(sl) -> int:
        c = _count_property(sl, prop)
        return -c if maximize else c

    pts = _require_realization(ot, "walk-points")
    n = ot.n

    # scale up for numerical stability (DFS mode)
    cur_bits = _pts_get_bits(pts)
    if not random_walk and good_bits and cur_bits < good_bits:
        scale = 2 ** (good_bits - cur_bits)
        pts = [(x * scale, y * scale) for x, y in pts]
    pts = _pts_normalize(pts)

    start_count = score(PointSet(n, pts).to_small_lambda())
    min_count = start_count
    temperature = T0

    if verbose:
        print(f"start: {prop}={abs(start_count)}")

    # ---- random walk mode ----
    if random_walk:
        if walk_range <= 0:
            diam = max(abs(x) for x, y in pts) + max(abs(y) for x, y in pts)
            walk_range = 5 + diam // 3
        if verbose:
            print(f"  range={walk_range}")

        _trial = 0
        while trials == 0 or _trial < trials:
            _trial += 1
            i = random.randrange(n)
            x, y = pts[i]
            dx = random.randint(-walk_range, walk_range)
            dy = random.randint(-walk_range, walk_range)
            pts[i] = (x + dx, y + dy)

            if len(set(pts)) < n:
                pts[i] = (x, y)
                continue
            ps = PointSet(n, pts)
            if ps.has_collinear_points():
                pts[i] = (x, y)
                continue

            new_ot = ps.to_small_lambda()
            new_count = score(new_ot)

            if new_count > min_count:
                pts[i] = (x, y)
                continue

            if new_count < min_count:
                min_count = new_count
                pts = _pts_normalize(pts)
                if verbose:
                    print(f"  improved: {prop}={abs(min_count)}")
                yield new_ot
                if stop_if_improved:
                    return
            # equal: keep the move
        return

    # ---- DFS mode ----
    directions = set()
    for k, l in [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 6)]:
        for dk, dl in [(k, l), (l, k), (-l, k), (-k, l),
                       (-k, -l), (-l, -k), (l, -k), (k, -l)]:
            directions.add((dk, dl))
    directions = list(directions)

    def ot_key(pts):
        return PointSet(n, pts).to_small_lambda().sl_hash()

    def same_ot(pts_a, pts_b, i):
        ps_a = PointSet(n, pts_a)
        ps_b = PointSet(n, pts_b)
        for a in range(n):
            for b in range(a + 1, n):
                if a == i or b == i:
                    continue
                if ps_a.orientation(a, b, i) != ps_b.orientation(a, b, i):
                    return False
        return True

    seen = set()

    class Frame:
        __slots__ = ("pts", "count", "todo", "cur_i", "dirs")
        def __init__(self, pts, count):
            self.pts = pts
            self.count = count
            todo = list(range(n))
            random.shuffle(todo)
            self.todo = todo
            self.cur_i = None
            self.dirs = []

    key0 = ot_key(pts)
    seen.add(key0)
    stack = collections.deque([Frame(pts, start_count)])
    _steps = 0

    while stack:
        if max_steps and _steps >= max_steps:
            break
        _steps += 1
        frame = stack[-1]

        while not frame.dirs:
            if not frame.todo:
                stack.pop()
                break
            frame.cur_i = frame.todo.pop()
            frame.dirs = list(directions)
            random.shuffle(frame.dirs)

        if not stack:
            break
        frame = stack[-1]
        if not frame.dirs:
            continue

        i = frame.cur_i
        dx, dy = frame.dirs.pop()
        pts0 = list(frame.pts)
        x0, y0 = pts0[i]

        pts_tmp = list(pts0)
        step = 1
        x1, y1 = x0, y0
        grow = True
        diam = max(abs(x) for x, y in pts0) + max(abs(y) for x, y in pts0) + 1

        while step > 0:
            pts_tmp[i] = (x1 + step * dx, y1 + step * dy)
            if same_ot(pts0, pts_tmp, i):
                x1, y1 = pts_tmp[i]
                if grow:
                    step *= 2
            else:
                step //= 2
                grow = False
            if step > 100 * diam:
                break
        if step > 100 * diam:
            continue

        pts_new = list(pts0)
        for attempt in range(10):
            pts_new[i] = (x1 + (attempt + 1) * dx, y1 + (attempt + 1) * dy)
            if len(set(pts_new)) == n:
                ps = PointSet(n, pts_new)
                if not ps.has_collinear_points():
                    break
        else:
            continue

        new_sl = PointSet(n, pts_new).to_small_lambda()
        new_count = score(new_sl)

        accept = new_count <= frame.count
        if not accept and siman:
            temperature *= dT
            if frame.count > 0:
                prob = exp(-(new_count - frame.count) / (frame.count * temperature))
                accept = random.random() < prob

        if not accept:
            continue

        if new_count < min_count:
            min_count = new_count
            if verbose:
                print(f"  improved: {prop}={abs(min_count)}")
            yield new_sl
            if stop_if_improved:
                return

        key = ot_key(pts_new)
        if key in seen:
            continue
        seen.add(key)

        stack.append(Frame(_pts_normalize(pts_new), new_count))
        if len(stack) > trace:
            stack.popleft()


def _register_walk_points(sub) -> None:
    p = sub.add_parser(
        "walk-points",
        help="Local search in coordinate space to minimize a property",
        description=(
            "Search for order types with a smaller property value by moving\n"
            "points in coordinate space.  The OT changes at each step.\n"
            "Requires a concrete input format (b08/b16/asc/json).\n\n"
            "  default     DFS: moves each point in 16 rational directions via\n"
            "              binary search to the OT boundary, steps past it.\n"
            "              Each new OT neighbor is pushed onto the DFS stack.\n\n"
            "  --random    Random walk: pick a random point, move it by a random\n"
            "              offset; accept if property does not increase.\n\n"
            "Available properties:\n" + _PROP_HELP
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_input_args(p)
    p.add_argument("prop", help="Property to minimize (see above)")
    p.add_argument("--random", action="store_true", dest="random_walk",
                   help="Random walk mode instead of DFS")
    p.add_argument("--trials", type=int, default=0,
                   help="(--random) Max random steps; 0 = run forever (default: 0)")
    p.add_argument("--range", type=int, default=0, dest="walk_range",
                   help="(--random) Max displacement per step (default: 5 + diam/3)")
    p.add_argument("--trace", type=int, default=10000,
                   help="(DFS) Stack size limit (default: 10000)")
    p.add_argument("--max-steps", type=int, default=0,
                   help="(DFS) Max DFS steps; 0 = run forever (default: 0)")
    p.add_argument("--good-bits", type=int, default=30,
                   help="(DFS) Scale coords to this bit-width (default: 30)")
    p.add_argument("--stop-if-improved", action="store_true",
                   help="Stop after the first improvement")
    p.add_argument("--siman", action="store_true",
                   help="(DFS) Simulated annealing to accept worse neighbors")
    p.add_argument("--T0", type=float, default=0.1,
                   help="(DFS+siman) Initial temperature (default: 0.1)")
    p.add_argument("--dT", type=float, default=0.999,
                   help="(DFS+siman) Temperature decay per step (default: 0.999)")
    p.add_argument("-o", "--output", default=None,
                   help="Output file (default: input.walk.NNN.lt)")
    p.set_defaults(func=_cmd_walk_points)


def _cmd_walk_points(args) -> None:
    idx = 0
    for ot in open_input(args):
        for improved in walk_points(
            ot, args.prop,
            random_walk=args.random_walk,
            trials=args.trials,
            walk_range=args.walk_range,
            trace=args.trace,
            max_steps=args.max_steps,
            good_bits=args.good_bits,
            stop_if_improved=args.stop_if_improved,
            siman=args.siman,
            T0=args.T0,
            dT=args.dT,
        ):
            idx += 1
            out = args.output or _output_path(args.input, ".walk.lt", idx)
            write_order_types([improved], out, fmt="lt")
            print(f"written: {out}")


# ===========================================================================
# walk-abstract  — DFS (or random) on the flip graph (no coordinates)
# ===========================================================================

def walk_abstract(
    ot: SmallLambda,
    prop: str,
    *,
    random_walk: bool = False,
    trials: int = 0,
    trace: int = 10000,
    max_steps: int = 0,
    stop_if_improved: bool = False,
    siman: bool = False,
    T0: float = 0.1,
    dT: float = 0.999,
    verbose: bool = True,
) -> Iterator[SmallLambda]:
    """Local search on the flip graph of exit-edge triples to minimize a property.

    Two modes (selected via random_walk):

    DFS mode (random_walk=False):
        For each abstract OT on the stack: enumerate all exit-edge triples
        (a,b,c), flip each one, evaluate the property, push improving
        (or equal) neighbors onto the DFS stack.

    Random mode (random_walk=True):
        At each step: pick a random exit-edge triple and flip it; accept
        if the property does not increase.

    Works purely on abstract order types — no coordinates needed.
    Yields each strictly improving OT found (lex-min representative).
    """
    import collections
    import random
    from math import exp
    from pyotlib2.algorithms.exit_edges import filter_exit_edges

    start_count = _count_property(ot, prop)
    min_count = start_count
    temperature = T0

    def ot_key(ot):
        return ot.sl_hash()

    if verbose:
        print(f"start: {prop}={start_count}")

    # ---- random walk mode ----
    if random_walk:
        cur_ot = ot
        cur_count = start_count
        _trial = 0
        while trials == 0 or _trial < trials:
            _trial += 1
            _, witnesses = filter_exit_edges(cur_ot, return_witnesses=True)
            triples = [(a, b, c) for (a, b), ws in witnesses.items() for c in ws]
            if not triples:
                break
            a, b, c = random.choice(triples)
            bl = cur_ot.to_big_lambda()
            new_bl = bl.flip_triple(a, b, c)
            try:
                new_sl = new_bl.to_small_lambda()
            except Exception:
                continue
            new_count = _count_property(new_sl, prop)
            if new_count > cur_count:
                continue
            if new_count < min_count:
                min_count = new_count
                lex = new_sl.get_lex_min()
                if verbose:
                    print(f"  improved: {prop}={min_count}")
                yield lex
                if stop_if_improved:
                    return
            cur_ot = new_sl
            cur_count = new_count
        return

    # ---- DFS mode ----
    seen = set()

    class Frame:
        __slots__ = ("ot", "count", "todo")
        def __init__(self, ot, count):
            self.ot = ot
            self.count = count
            _, witnesses = filter_exit_edges(ot, return_witnesses=True)
            todo = [(a, b, c) for (a, b), ws in witnesses.items() for c in ws]
            random.shuffle(todo)
            self.todo = todo

    key0 = ot_key(ot)
    seen.add(key0)
    stack = collections.deque([Frame(ot, start_count)])
    _steps = 0

    while stack:
        if max_steps and _steps >= max_steps:
            break
        _steps += 1
        frame = stack[-1]

        if not frame.todo:
            stack.pop()
            continue

        a, b, c = frame.todo.pop()
        bl = frame.ot.to_big_lambda()
        new_bl = bl.flip_triple(a, b, c)

        try:
            new_sl = new_bl.to_small_lambda()
        except Exception:
            continue

        new_count = _count_property(new_sl, prop)

        accept = new_count <= frame.count
        if not accept and siman:
            temperature *= dT
            if frame.count > 0:
                prob = exp(-(new_count - frame.count) / (frame.count * temperature))
                accept = random.random() < prob

        if not accept:
            continue

        if new_count < min_count:
            min_count = new_count
            lex = new_sl.get_lex_min()
            if verbose:
                print(f"  improved: {prop}={min_count}")
            yield lex
            if stop_if_improved:
                return

        key = ot_key(new_sl)
        if key in seen:
            continue
        seen.add(key)

        stack.append(Frame(new_sl, new_count))
        if len(stack) > trace:
            stack.popleft()


def _register_walk_abstract(sub) -> None:
    p = sub.add_parser(
        "walk-abstract",
        help="Local search on the flip graph to minimize a property (no coordinates)",
        description=(
            "Search for order types with a smaller property value by flipping\n"
            "exit-edge triples in the chirotope.  Works on abstract OTs —\n"
            "no coordinates needed (.lt/.blt input).\n\n"
            "  default     DFS on the flip graph: enumerate all exit-edge triples\n"
            "              and flip each to reach a neighboring OT.\n\n"
            "  --random    Random flip walk: at each step pick a random exit-edge\n"
            "              triple and flip it; accept if property does not increase.\n\n"
            "Available properties:\n" + _PROP_HELP
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_input_args(p)
    p.add_argument("prop", help="Property to minimize (see above)")
    p.add_argument("--random", action="store_true", dest="random_walk",
                   help="Random flip walk mode instead of DFS")
    p.add_argument("--trials", type=int, default=0,
                   help="(--random) Max random flips; 0 = run forever (default: 0)")
    p.add_argument("--trace", type=int, default=10000,
                   help="(DFS) Stack size limit (default: 10000)")
    p.add_argument("--max-steps", type=int, default=0,
                   help="(DFS) Max DFS steps; 0 = run forever (default: 0)")
    p.add_argument("--stop-if-improved", action="store_true",
                   help="Stop after the first improvement")
    p.add_argument("--siman", action="store_true",
                   help="(DFS) Simulated annealing to accept worse neighbors")
    p.add_argument("--T0", type=float, default=0.1,
                   help="(DFS+siman) Initial temperature (default: 0.1)")
    p.add_argument("--dT", type=float, default=0.999,
                   help="(DFS+siman) Temperature decay per step (default: 0.999)")
    p.add_argument("-o", "--output", default=None,
                   help="Output file (default: input.abswalk.NNN.lt)")
    p.set_defaults(func=_cmd_walk_abstract)


def _cmd_walk_abstract(args) -> None:
    idx = 0
    for ot in open_input(args):
        for improved in walk_abstract(
            ot, args.prop,
            random_walk=args.random_walk,
            trials=args.trials,
            trace=args.trace,
            max_steps=args.max_steps,
            stop_if_improved=args.stop_if_improved,
            siman=args.siman,
            T0=args.T0,
            dT=args.dT,
        ):
            idx += 1
            out = args.output or _output_path(args.input, ".abswalk.lt", idx)
            write_order_types([improved], out, fmt="lt")
            print(f"written: {out}")


# ===========================================================================
# extend-abstract  — enumerate all n+1 extensions
# ===========================================================================

def extend_abstract(
    ot: SmallLambda,
    *,
    unify_pcs: bool = False,
    method: str = "recursive",
    auto_relabel: bool = True,
) -> Iterator[SmallLambda]:
    """Enumerate all abstract OTs on n+1 points extending ot.

    Parameters
    ----------
    ot : SmallLambda
        Input order type on n points.
    unify_pcs : bool
        If True, deduplicate at projective-class level.
    method : str
        ``"recursive"`` (default) — fast backtracking with signotope pruning,
        requires natural labeling (point 0 on convex hull, all others to the
        right).  Ported from supplemental-universal-sets (Scheucher 2020),
        see https://doi.org/10.7155/jgaa.00529.

        ``"sat"`` — enumerate via SAT solver (CaDiCaL/python-sat).
        Slower but does not require natural labeling.
        Requires:  pip install pyotlib2[sat]
    auto_relabel : bool
        Only relevant for ``method="recursive"``.  If True (default), apply
        natural labeling automatically before extending.  If False, the input
        must already be in natural labeling (point 0 on hull with all others
        to its right); an AssertionError is raised otherwise.
    """
    if method == "recursive":
        yield from _extend_abstract_recursive(ot, unify_pcs=unify_pcs, auto_relabel=auto_relabel)
    elif method == "sat":
        yield from _extend_abstract_sat(ot, unify_pcs=unify_pcs)
    else:
        raise ValueError(f"unknown method {method!r}, choose 'recursive' or 'sat'")


def _extend_abstract_recursive(
    ot: SmallLambda,
    *,
    unify_pcs: bool = False,
    auto_relabel: bool = True,
) -> Iterator[SmallLambda]:
    """Recursive backtracking extension — port of extend_order_type.cpp.

    Requires natural labeling: point 0 must be on the convex hull and
    o[0,j,k] == -1 for all 0 < j < k < n  (all other points are to the
    right of the ray 0→…).
    """
    import numpy as np
    from pyotlib2.core.big_lambda import BigLambda

    seen: set = set()

    hull_pts = ot.get_extremal_points()

    for h in hull_pts:
        # Apply natural labeling so that hull point h becomes point 0
        lab = ot.get_natural_labeling(h)
        if auto_relabel:
            ot_nat = ot.relabeled(lab)
        else:
            # Verify natural labeling already in place
            assert lab[0] == 0, (
                "Input is not in natural labeling (point 0 not on hull). "
                "Pass auto_relabel=True or relabel the input first."
            )
            ot_nat = ot

        n = ot_nat.n
        n1 = n + 1  # size after extension (new point index = n)
        o_base = ot_nat.to_big_lambda().o  # shape (n, n, n)

        # Verify natural labeling: o[0,j,k] == -1 for all 0<j<k<n
        for j in range(1, n):
            for k in range(j + 1, n):
                assert int(o_base[0, j, k]) == -1, (
                    f"Not in natural labeling: o[0,{j},{k}] = {o_base[0,j,k]}"
                )

        # o_ext: mutable n1×n1×n1 array, initialized from o_base, new entries = 0
        o_ext = np.zeros((n1, n1, n1), dtype=np.int8)
        o_ext[:n, :n, :n] = o_base

        def set_triple(i, j, d, val):
            o_ext[i, j, d] = o_ext[j, d, i] = o_ext[d, i, j] = val
            o_ext[i, d, j] = o_ext[d, j, i] = o_ext[j, i, d] = -val

        def check_signotope(i, j):
            """Check signotope axiom for the newly assigned triple (i,j,n).

            For every k != i,j: form sorted 4-tuple (a,b,c,d=n) where
            {a,b,c} = {i,j,k} sorted.  The sign sequence
            χ(a,b,c), χ(a,b,n), χ(a,c,n), χ(b,c,n)
            must have at most one sign change.
            """
            d = n  # new point index
            for k in range(n):
                if k == i or k == j:
                    continue
                if k < i:
                    a, b, c = k, i, j
                elif k < j:
                    a, b, c = i, k, j
                else:
                    a, b, c = i, j, k
                s_abc = int(o_ext[a, b, c])
                s_abd = int(o_ext[a, b, d])
                s_acd = int(o_ext[a, c, d])
                s_bcd = int(o_ext[b, c, d])
                changes = (
                    (s_abc * s_abd == -1)
                    + (s_abd * s_acd == -1)
                    + (s_acd * s_bcd == -1)
                )
                if changes > 1:
                    return False
            return True

        def recurse(i, j):
            # Try both orientations; if i==0 only try o=-1 (natural labeling fix)
            orientations = [-1] if i == 0 else [-1, +1]
            for o_val in orientations:
                set_triple(i, j, n, o_val)
                if not check_signotope(i, j):
                    continue
                # Advance to next pair
                if j == n - 1:
                    if i == n - 2:
                        # All pairs assigned — record this extension
                        new_sl = BigLambda(n1, o_ext.copy()).to_small_lambda()
                        # Natural labeling around new point n, then lex-min
                        lab2 = new_sl.get_natural_labeling(n)
                        if lab2[0] != n:
                            continue  # new point not on hull → skip
                        new_sl_rel = new_sl.relabeled(lab2)
                        new_ot = new_sl_rel.get_lex_min()
                        if unify_pcs:
                            from pyotlib2.algorithms.projective_class import ProjectiveClass
                            new_ot = ProjectiveClass(new_ot).representer
                        key = new_ot.to_string()
                        if key not in seen:
                            seen.add(key)
                            yield new_ot
                    else:
                        yield from recurse(i + 1, i + 2)
                else:
                    yield from recurse(i, j + 1)
            set_triple(i, j, n, 0)  # reset

        yield from recurse(0, 1)


def _extend_abstract_sat(
    ot: SmallLambda,
    *,
    unify_pcs: bool = False,
) -> Iterator[SmallLambda]:
    """SAT-based extension — port of old pyotlib ExtendAbstract.py."""
    try:
        from pysat.solvers import Cadical195 as Solver
    except ImportError:
        raise RuntimeError(
            "python-sat is required: install with  pip install python-sat\n"
            "or  pip install pyotlib2[sat]"
        )
    from itertools import combinations
    import numpy as np
    from pyotlib2.core.big_lambda import BigLambda

    n = ot.n + 1  # new size (new point appended at end)
    o_old = ot.to_big_lambda().o  # shape (ot.n, ot.n, ot.n)

    all_variables = list(combinations(range(n), 3))
    var_of = {t: i + 1 for i, t in enumerate(all_variables)}

    def var(a, b, c):
        return var_of[(a, b, c)]

    seen: set = set()

    for i0 in range(n):
        constraints = []

        for I3 in combinations(range(n), 3):
            if i0 in I3:
                continue
            I4 = tuple(sorted(I3 + (i0,)))
            I4_triples = list(combinations(I4, 3))
            for t1, t2, t3 in combinations(I4_triples, 3):
                for sgn in (+1, -1):
                    constraints.append([sgn * var(*t1), -sgn * var(*t2), sgn * var(*t3)])

        for a, b, c in combinations(range(n - 1), 3):
            a2 = a if a < i0 else a + 1
            b2 = b if b < i0 else b + 1
            c2 = c if c < i0 else c + 1
            constraints.append([int(o_old[a, b, c]) * var(a2, b2, c2)])

        for sol in _iter_sat(Solver, constraints):
            o_new = [[[0] * n for _ in range(n)] for _ in range(n)]
            for a, b, c in all_variables:
                oabc = +1 if sol[var(a, b, c) - 1] > 0 else -1
                o_new[a][b][c] = o_new[b][c][a] = o_new[c][a][b] = +oabc
                o_new[a][c][b] = o_new[b][a][c] = o_new[c][b][a] = -oabc

            o_arr = np.array(o_new, dtype=np.int8)
            new_sl = BigLambda(n, o_arr).to_small_lambda()
            new_ot = new_sl.get_lex_min()

            if unify_pcs:
                from pyotlib2.algorithms.projective_class import ProjectiveClass
                new_ot = ProjectiveClass(new_ot).representer

            key = new_ot.to_string()
            if key not in seen:
                seen.add(key)
                yield new_ot


def _iter_sat(Solver, constraints):
    """Iterate all SAT solutions, blocking each after yielding."""
    n_vars = max(abs(lit) for clause in constraints for lit in clause)
    with Solver(bootstrap_with=constraints) as solver:
        while solver.solve():
            sol = solver.get_model()
            yield sol
            block = [-lit for lit in sol if 1 <= abs(lit) <= n_vars]
            solver.add_clause(block)


def _register_extend_abstract(sub) -> None:
    p = sub.add_parser(
        "extend-abstract",
        help="Enumerate all n+1 abstract OT extensions",
        description=(
            "For each input OT of n points, enumerate all abstract order types "
            "on n+1 points whose restriction to the original n points equals the "
            "input.  Outputs lex-min representatives.\n\n"
            "method=recursive (default): fast backtracking with signotope pruning "
            "(Scheucher 2020, https://doi.org/10.7155/jgaa.00529).\n"
            "method=sat: SAT solver (CaDiCaL); requires pip install pyotlib2[sat]"
        ),
    )
    add_input_args(p)
    add_output_args(p, ".ext")
    p.add_argument("--unify-pcs", action="store_true",
                   help="Deduplicate at projective-class level (slower)")
    p.add_argument("--method", choices=["recursive", "sat"], default="recursive",
                   help="Extension method (default: recursive)")
    p.add_argument("--no-auto-relabel", action="store_true",
                   help="Do not auto-apply natural labeling (recursive method only)")


def _cmd_extend_abstract(args) -> None:
    out, fmt = resolve_output(args, ".ext.lt")
    results = []
    for ot in open_input(args):
        n_before = len(results)
        for new_ot in extend_abstract(
            ot,
            unify_pcs=args.unify_pcs,
            method=args.method,
            auto_relabel=not args.no_auto_relabel,
        ):
            results.append(new_ot)
        print(f"  n={ot.n} → {len(results) - n_before} extensions")
    write_order_types(results, out, fmt="lt")
    print(f"total: {len(results)}  written to: {out}")


# ===========================================================================
# extend-random  — random point placement for n+1 extensions
# ===========================================================================

def extend_random(
    ot: SmallLambda,
    *,
    trials: int = 1000,
    expand: float = 1.0,
) -> Iterator[SmallLambda]:
    """Extend a realized OT by randomly placing an additional point.

    Tries up to `trials` random placements in a box expanded by `expand`
    times the diameter around the existing point set.  Each valid,
    non-collinear placement defines a new OT; duplicates are suppressed.

    Requires ot.realization to be set.
    Yields lex-min SmallLambda for each distinct extension found.
    """
    import random
    from pyotlib2.core.point_set import PointSet

    pts = _require_realization(ot, "extend-random")
    n = ot.n

    xs = [x for x, y in pts]
    ys = [y for x, y in pts]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    dx = max(xhi - xlo, 1) * expand
    dy = max(yhi - ylo, 1) * expand

    seen: set = set()

    for _ in range(trials):
        x = int(random.uniform(xlo - dx, xhi + dx))
        y = int(random.uniform(ylo - dy, yhi + dy))
        new_pts = pts + [(x, y)]

        ps = PointSet(n + 1, new_pts)
        if ps.has_collinear_points():
            continue

        new_sl = ps.to_small_lambda().get_lex_min()
        key = new_sl.to_string()
        if key in seen:
            continue
        seen.add(key)
        yield new_sl


def _register_extend_random(sub) -> None:
    p = sub.add_parser(
        "extend-random",
        help="Extend realized OTs by randomly placing one additional point",
        description=(
            "For each input OT (with concrete coordinates), try placing an "
            "extra point at random positions in an expanded bounding box.  "
            "Each non-collinear placement defines an n+1-point OT; distinct "
            "ones are output as lex-min representatives.  "
            "Requires a concrete input format (b08/b16/asc/json)."
        ),
    )
    add_input_args(p)
    add_output_args(p, ".ext")
    p.add_argument("--trials", type=int, default=1000,
                   help="Number of random placements to try (default: 1000)")
    p.add_argument("--expand", type=float, default=1.0,
                   help="Expand bounding box by this factor (default: 1.0)")
    p.set_defaults(func=_cmd_extend_random)


def _cmd_extend_random(args) -> None:
    out, fmt = resolve_output(args, ".ext.lt")
    results = []
    for ot in open_input(args):
        n_before = len(results)
        for new_ot in extend_random(ot, trials=args.trials, expand=args.expand):
            results.append(new_ot)
        print(f"  n={ot.n} → {len(results) - n_before} extensions found")
    write_order_types(results, out, fmt="lt")
    print(f"total: {len(results)}  written to: {out}")

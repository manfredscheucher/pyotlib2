"""pyotlib2 CLI entry point.

Usage:
    pyotlib2 <command> [options]

OT-level operations:
    unify-ot       Remove duplicate order types
    lex-min-ot     Relabel order types to lex-min representative
    sort           Sort order types lexicographically
    shuffle        Shuffle order types randomly
    enum-sub-ot    Enumerate k-point sub-order-types
    count-sub-ot   Count distinct k-point sub-order-types per OT
    find-sub-ot    Find order types containing specific sub-OTs

Projective class (PC) operations:
    unify-pc       Remove duplicate projective classes
    lex-min-pc     Relabel order types to their PC representer
    enum-pc        Enumerate all OTs in each projective class

Properties:
    kgons          Count empty/convex k-gons
    properties     Compute combinatorial properties

Realization:
    realize        Test realizability of order types
    realize-pc     Test realizability via projective class search
    smart-realize  Realize OT point-by-point (etherealization)

Coordinate minimization (OT is preserved; only coordinates change):
    minimize-coords   Minimize coordinate size (move-to-even / halve)
    beautify-coords   Beautify coordinates (--method gd|nm)

Literal ports of the old pyotlib coordinate optimizers (OT is preserved):
    minimize-integer-coords  MinimizeCoords / minimizeIntegerCoordinates
    beautify-pointset        BeautifyCoords / beautifyPointSet
    beautify2                Beautify2Script / beautify2 (scipy fmin)
    minimize-pattern-search  MinimizePatternSearch

Property minimization / local search (OT changes; looks for better OTs):
    walk-points    DFS or random walk in coordinate space (needs coords)
    walk-abstract  DFS or random walk on the flip graph (abstract only)

Extension:
    extend-abstract   Enumerate all n+1 abstract extensions via SAT
    extend-random     Random n+1 extensions by point placement

Visualization:
    plot              Visualize order types as point set drawings (matplotlib)
    editor            Interactive point set editor (PySide6)

All commands can also be used as Python functions:
    from pyotlib2.cli.commands import unify_ot, lex_min_ot, ...
"""

import argparse
import sys
from pyotlib2.cli import commands


class _HelpOnErrorParser(argparse.ArgumentParser):
    """Subparser that prints full help (not just usage) on error."""

    def error(self, message):
        self.print_help(sys.stderr)
        sys.stderr.write(f"\nerror: {message}\n")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(
        prog="pyotlib2",
        description="Python Order Type Library 2 — abstract order type computations",
        epilog="Run 'pyotlib2 help' or 'pyotlib2 <command> --help' for details.",
    )
    sub = parser.add_subparsers(
        dest="command",
        metavar="command",
        parser_class=_HelpOnErrorParser,
    )
    sub.required = False

    commands.register_all(sub)

    # add a 'help' pseudo-command
    sub.add_parser("help", help="list all available commands")

    args = parser.parse_args()

    if args.command is None or args.command == "help":
        _print_manpage()
        sys.exit(0)

    args.func(args)


def _print_manpage():
    """Print a rich man-page-style help summary."""
    print("""\
NAME
    pyotlib2 — Python Order Type Library 2

SYNOPSIS
    pyotlib2 <command> [options]

DESCRIPTION
    Compute, enumerate, and visualize abstract order types, projective
    classes, and combinatorial properties of point sets.

COMMANDS
  OT-level operations:
    unify-ot           Remove duplicate order types
    lex-min-ot         Relabel order types to lex-min representative
    sort               Sort order types lexicographically
    shuffle            Shuffle order types randomly
    enum-sub-ot        Enumerate k-point sub-order-types
    count-sub-ot       Count distinct k-point sub-order-types per OT
    find-sub-ot        Find order types containing specific sub-OTs

  Projective class (PC) operations:
    unify-pc           Remove duplicate projective classes
    lex-min-pc         Relabel order types to their PC representer
    enum-pc            Enumerate all OTs in each projective class

  Properties:
    kgons              Count empty/convex k-gons
    properties         Compute combinatorial properties

  Realization:
    realize            Test realizability of order types
    realize-pc         Test realizability via projective class search
    smart-realize      Realize OT point-by-point (etherealization)

  Coordinate minimization (OT preserved, coordinates change):
    minimize-coords    Minimize coordinate size (move-to-even / halve)
    beautify-coords    Beautify coordinates (--method gd|nm)

  Property minimization / local search (OT changes):
    walk-points        DFS or random walk in coordinate space
    walk-abstract      DFS or random walk on the flip graph

  Extension:
    extend-abstract    Enumerate all n+1 abstract extensions via SAT
    extend-random      Random n+1 extensions by point placement

  Visualization:
    plot               Visualize order types (matplotlib)
    editor             Interactive point set editor (PySide6)

DATA STRUCTURES
    SmallLambda (.sl)    Compact binary; abstract OT, fixed-width
    BigLambda (.bl)      Variable-width binary; abstract OT
    PointSet (.pts)      Integer coordinates (one point per line)
    FelsnerMatrix (.fm)  Crossing-family matrix representation

FILE FORMATS
    Format   Ext     Content
    ───────  ──────  ──────────────────────────────────────
    SL       .sl     SmallLambda binary (n ≤ ~20)
    BL       .bl     BigLambda binary (arbitrary n)
    PTS      .pts    Integer point coordinates
    FM       .fm     Felsner matrix
    JSON     .json   JSON point sets

EXAMPLES
    pyotlib2 properties -n 8 input.sl -o results.csv
    pyotlib2 realize input.sl -o output.pts
    pyotlib2 minimize-coords input.pts -o small.pts
    pyotlib2 walk-abstract -n 9 --property cr --minimize
    pyotlib2 editor input.pts
    pyotlib2 plot input.sl -n 7

SEE ALSO
    pyotlib2 <command> --help    Per-command options and details
""")


if __name__ == "__main__":
    main()

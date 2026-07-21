#!/usr/bin/env python3
"""Parse the output of a Mathematica realization script (``--export-mathematica``).

The exported ``.wls`` script (see
:mod:`pyotlib2.realization.mathematica_tester`) writes, next to the coordinate
output file ``<out>``:

  * ``<out>``            -- the coordinates, one ``x y`` pair per line (Table);
  * ``<out>.objective``  -- the single minimised penalty value ``fmin``.

A realization was found iff ``fmin`` is essentially zero.  The exported script
uses a precision-derived threshold ``eps = 10^(-digits/2)`` (``digits`` =
Mathematica WorkingPrecision, i.e. ``ceil(precision_bits / log2(10))``); this
parser recomputes the SAME threshold from ``--precision`` so its verdict matches
the script's, with no hardcoded tolerance.

Usage:
    .venv/bin/python scripts/parse_mathematica_result.py <out-file> [--precision BITS]

``<out-file>`` is the coordinate file passed to ``NMinimize`` export (the
``.objective`` sibling is read automatically).  ``--precision`` must match the
value given to ``--export-mathematica`` (default 0 = MachinePrecision).

Exit code: 0 if a realization was found, 1 if not, 2 on a read/parse error.
Prints a JSON summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def verdict_threshold(precision_bits: int) -> float:
    """The REALIZED threshold for a given --precision (in BITS).

    Mirrors the exported script exactly: digits = ceil(bits / log2(10)); the
    threshold is 10^(-digits/2), or 10^-7 at MachinePrecision (bits <= 0).
    """
    if precision_bits and precision_bits > 0:
        digits = math.ceil(precision_bits / math.log2(10))
        return 10.0 ** (-(digits // 2))
    return 10.0 ** -7


def read_objective(objective_path: str) -> float:
    """Read the single ``fmin`` value from the ``.objective`` file."""
    with open(objective_path) as f:
        tokens = f.read().split()
    if not tokens:
        raise ValueError(f"{objective_path!r} is empty")
    return float(tokens[0])


def read_coords(out_path: str) -> list[tuple[float, ...]]:
    """Read the coordinate table (one ``x y`` pair per line)."""
    coords: list[tuple[float, ...]] = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            coords.append(tuple(float(t) for t in line.split()))
    return coords


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_file",
                    help="coordinate output file (its .objective sibling is read too)")
    ap.add_argument("--precision", type=int, default=0,
                    help="precision BITS used for --export-mathematica "
                         "(0 = MachinePrecision; must match the export)")
    args = ap.parse_args(argv)

    objective_path = args.out_file + ".objective"
    try:
        fmin = read_objective(objective_path)
    except (OSError, ValueError) as e:
        print(json.dumps({"error": f"could not read objective: {e}"}))
        return 2
    try:
        coords = read_coords(args.out_file)
    except OSError as e:
        print(json.dumps({"error": f"could not read coords: {e}"}))
        return 2

    eps = verdict_threshold(args.precision)
    realized = fmin < eps

    print(json.dumps({
        "realized": realized,
        "objective": fmin,
        "threshold": eps,
        "precision_bits": args.precision,
        "num_points": len(coords),
        "coords": coords,
    }, indent=2))
    return 0 if realized else 1


if __name__ == "__main__":
    sys.exit(main())

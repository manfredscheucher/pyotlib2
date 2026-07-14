#!/usr/bin/env python3
"""Convert a .color signotope file to pyotlib2 BigLambda text (.blt) format.

Input format (.color)
----------------------
One line per triple (a, b, c) with a < b < c, 1-indexed points:

    a b c COLOR

where COLOR is 'B' (blue, +) or 'R' (red, -).  The lines must list every
triple of an n-point ground set in lexicographic order of (a, b, c).

Output format (.blt)
--------------------
BigLambda text (see pyotlib2 file-formats doc): a single line of C(n,3)
'+'/'-' characters, one per triple (a, b, c) with a < b < c in lexicographic
order.  Sign convention: B -> '+', R -> '-'.

Because both formats enumerate triples in the same lexicographic order, the
conversion is a straight per-line color-to-sign mapping (after verifying the
triples really are the complete, correctly ordered set).

The auto-detected conventions (n, index base, sign mapping) are written as a
leading '#' comment header; the blt reader skips those lines, so the file
stays loadable while keeping the conventions attached -- handy when the same
object is shared between authors with differing habits.  (This script is kept
self-contained -- no pyotlib2 import -- so it runs from the data directory
without installing the package.  The same '#'-header convention is supported
natively by pyotlib2's write_order_types(..., header=...).)

Usage
-----
    python3 convert_color_to_blt.py test24.color test24.blt
    python3 convert_color_to_blt.py test24.color          # -> test24.blt
"""

from __future__ import annotations

import sys
from itertools import combinations
from math import comb
from pathlib import Path

COLOR_TO_SIGN = {"B": "+", "R": "-"}


def convert(color_path: Path) -> tuple[str, dict]:
    """Read a .color file, validate it, and return (blt_line, params).

    ``params`` reports the auto-detected input conventions (n, index base,
    triple count) so they can be logged / kept — with several authors these
    are just conventions and worth making explicit rather than assuming.
    """
    entries: list[tuple[int, int, int, str]] = []
    with open(color_path) as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) != 4:
                raise ValueError(
                    f"{color_path}:{lineno}: expected 'a b c COLOR', got {raw!r}"
                )
            a, b, c, color = int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
            if not (a < b < c):
                raise ValueError(
                    f"{color_path}:{lineno}: triple not sorted a<b<c: {a} {b} {c}"
                )
            if color not in COLOR_TO_SIGN:
                raise ValueError(
                    f"{color_path}:{lineno}: unknown color {color!r} "
                    f"(expected one of {sorted(COLOR_TO_SIGN)})"
                )
            entries.append((a, b, c, color))

    if not entries:
        raise ValueError(f"{color_path}: no triples found")

    # --- Auto-detect the index convention (0-based vs 1-based) and n. ---
    # Points span [lo, hi]; a valid ground set is contiguous, so n = hi-lo+1.
    lo = min(a for a, _, _, _ in entries)
    hi = max(c for _, _, c, _ in entries)
    if lo == 0:
        base = 0  # points 0 .. n-1
    elif lo == 1:
        base = 1  # points 1 .. n
    else:
        raise ValueError(
            f"unexpected smallest index {lo}: expected 0 (0-based) or 1 (1-based)"
        )
    n = hi - base + 1

    expected = comb(n, 3)
    if len(entries) != expected:
        raise ValueError(
            f"expected C({n},3) = {expected} triples for n={n}, got {len(entries)}"
        )

    # Verify the triples are exactly all triples of [base, base+n) in lex order.
    for got, exp in zip(entries, combinations(range(base, base + n), 3)):
        if (got[0], got[1], got[2]) != exp:
            raise ValueError(
                f"triple order mismatch: file has {got[:3]}, expected {exp}. "
                "Input must list all triples in lexicographic order."
            )

    signs = "".join(COLOR_TO_SIGN[color] for *_, color in entries)
    params = {
        "n": n,
        "index_base": base,
        "point_range": f"{lo}..{hi}",
        "num_triples": len(entries),
        "sign_convention": "B=+ (blue), R=- (red)",
    }
    return signs, params


def main(argv: list[str]) -> int:
    if not (2 <= len(argv) <= 3):
        print(__doc__)
        print(f"error: expected 1 or 2 arguments, got {len(argv) - 1}", file=sys.stderr)
        return 2
    color_path = Path(argv[1])
    out_path = Path(argv[2]) if len(argv) == 3 else color_path.with_suffix(".blt")

    signs, params = convert(color_path)

    # Attach the detected conventions as a '#' comment header.  The blt reader
    # skips lines starting with '#', so the file stays loadable.
    header = [f"# converted from {color_path.name} by convert_color_to_blt.py"]
    header += [f"# {key}: {value}" for key, value in params.items()]
    out_path.write_text("\n".join(header) + "\n" + signs + "\n")

    print(f"detected parameters for {color_path}:")
    for key, value in params.items():
        print(f"  {key:16} {value}")
    print(f"wrote {out_path} ({len(signs)} sign chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

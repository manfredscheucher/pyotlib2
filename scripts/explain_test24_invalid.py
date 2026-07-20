#!/usr/bin/env python3
"""Check whether a .color order-type file is a signotope and a chirotope.

Prints one explicit witness for each violated axiom (with color + line number
for the signotope witness).

Run:  .venv/bin/python scripts/explain_test24_invalid.py [FILE.color]
      (default: data_jelena/test24.color)
"""

import os
import sys
from itertools import combinations, permutations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyotlib2.core.big_lambda import BigLambda

DEFAULT = os.path.join(os.path.dirname(__file__), "..", "data_jelena", "test24.color")
COLOR_TO_SIGN = {"B": 1, "R": -1}


def load_color(path):
    """Read a .color file -> (n, o array, {(a,b,c) 1-indexed: (line, color)})."""
    entries, cidx = [], {}
    for lineno, raw in enumerate(open(path), 1):
        parts = raw.split()
        if len(parts) == 4 and parts[3] in COLOR_TO_SIGN:
            a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
            entries.append((a, b, c, parts[3]))
            cidx[(a, b, c)] = (lineno, parts[3])
    lo = min(a for a, _, _, _ in entries)
    hi = max(c for _, _, c, _ in entries)
    n = hi - lo + 1
    assert len(entries) == n * (n - 1) * (n - 2) // 6, "sign count is not C(n,3)"
    o = np.zeros((n, n, n), dtype=np.int8)
    for a, b, c, col in entries:
        v = np.int8(COLOR_TO_SIGN[col])
        a, b, c = a - lo, b - lo, c - lo  # 0-indexed
        o[a, b, c] = o[b, c, a] = o[c, a, b] = v
        o[a, c, b] = o[b, a, c] = o[c, b, a] = -v
    return n, o, cidx, lo


def s(o, a, b, c):
    return "+" if o[a, b, c] > 0 else "-"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    n, o, cidx, base = load_color(path)
    bl = BigLambda(n, o)
    print(f"{os.path.basename(path)}: n={n}  "
          f"is_signotope={bl.is_signotope()}  is_chirotope={bl.is_chirotope()}")

    # signotope: first 4-tuple with >1 sign change
    for a, b, c, d in combinations(range(n), 4):
        w = [int(o[a, b, c]), int(o[a, b, d]), int(o[a, c, d]), int(o[b, c, d])]
        if sum(1 for x, y in zip(w, w[1:]) if x != y) > 1:
            print(f"not a signotope: 4-tuple ({a+base},{b+base},{c+base},{d+base}), "
                  f"sign word {''.join('+' if x > 0 else '-' for x in w)} (>1 change):")
            for (x, y, z) in [(a, b, c), (a, b, d), (a, c, d), (b, c, d)]:
                t = (x + base, y + base, z + base)
                lineno, col = cidx.get(t, ("?", "?"))
                print(f"    {t[0]} {t[1]} {t[2]} {col}   (line {lineno})")
            break
    else:
        print("signotope: OK")

    # chirotope: first Theorem 3(iii) violation
    for a3 in range(n):
        for a1, a2, b1, b2 in permutations(range(n), 4):
            if a3 in (a1, a2, b1, b2):
                continue
            if (o[b1, a2, a3] * o[a1, b2, a3] >= 0
                    and o[b2, a2, a3] * o[b1, a1, a3] >= 0
                    and o[a1, a2, a3] * o[b1, b2, a3] < 0):
                print(f"not a chirotope: Thm 3(iii) at a1={a1+base} a2={a2+base} "
                      f"a3={a3+base} b1={b1+base} b2={b2+base}; conclusion "
                      f"{s(o,a1,a2,a3)}*{s(o,b1,b2,a3)} < 0")
                return
    print("chirotope: OK")


if __name__ == "__main__":
    main()

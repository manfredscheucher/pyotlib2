#!/usr/bin/env python3
"""List order-type data files in the repo with their point count n.

Scans for readable order-type files (.blt, .b08, .b16, .b32, .lt), reads the
first order type from each, and prints (n, count, path).  Use --min-n to filter.

Run:  .venv/bin/python scripts/list_order_type_files.py --min-n 11
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyotlib2.io.readers import read_order_types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTS = (".blt", ".b08", ".b16", ".b32", ".lt")


def first_ot_n(path: str):
    """Return (n, num_ots_capped) for a file, or None if unreadable."""
    try:
        it = read_order_types(path)
        first = next(iter(it), None)
        if first is None:
            return None
        n = first.n
        # count a few more (cheap cap) just to show single vs. many
        count = 1
        for _ in it:
            count += 1
            if count >= 1000:
                break
        return n, count
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-n", type=int, default=0, help="only show files with n >= this")
    ap.add_argument("--root", default=_ROOT, help="directory to scan (default: repo root)")
    args = ap.parse_args()

    rows = []
    for dirpath, dirs, files in os.walk(args.root):
        if ".venv" in dirpath or "/.git" in dirpath:
            continue
        for f in files:
            if f.lower().endswith(_EXTS):
                path = os.path.join(dirpath, f)
                info = first_ot_n(path)
                if info is None:
                    continue
                n, count = info
                if n >= args.min_n:
                    rows.append((n, count, os.path.relpath(path, args.root)))

    rows.sort(key=lambda r: (-r[0], r[2]))
    print(f"{'n':>4}  {'#ots':>6}  path")
    print("-" * 60)
    for n, count, path in rows:
        cap = f"{count}+" if count >= 1000 else str(count)
        print(f"{n:>4}  {cap:>6}  {path}")
    print(f"\n{len(rows)} files with n >= {args.min_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

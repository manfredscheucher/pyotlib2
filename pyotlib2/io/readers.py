"""File readers for all pyotlib2 order-type formats.

Supported formats
=================

SmallLambda
-----------
slt         SmallLambda text: n rows of n space-separated integers, one blank
            line between order types.  (.txt is a legacy alias.)
slb04       SmallLambda binary, 4 bits/entry, n² entries packed MSB-first,
            zero-padded to a whole number of bytes.  Values 0..n-2; n ≤ 16.
slb08       SmallLambda binary, 8 bits/entry (1 byte), n² entries.  n ≤ 256.
slb16       SmallLambda binary, 16 bits/entry (2 bytes, little-endian), n²
            entries.  n ≤ 65536.
slbc04      SmallLambda compressed binary, 4 bits/entry, C(n-1,2) entries.
            Requires natural labeling (asserts L[0,j]=j-1).  n ≤ 16.
slbc08      SmallLambda compressed binary, 8 bits/entry.  n ≤ 256.
slbc16      SmallLambda compressed binary, 16 bits/entry.  n ≤ 65536.

            Compressed format stores only the upper-triangle entries with both
            indices ≥ 1, i.e. L[i,j] for 1 ≤ i < j ≤ n-1, in row-major
            order.  Row 0 (L[0,j]=j-1) and the lower triangle
            (L[i,j]=n-2-L[j,i]) are reconstructed automatically.

BigLambda
---------
blt         BigLambda text: one line of C(n,3) '+'/'-' characters per order
            type, for all triples (a,b,c) with a<b<c in lexicographic order.
blb         BigLambda binary, 1 bit/entry, C(n,3) bits packed MSB-first,
            zero-padded to a whole number of bytes.  Same triple order as blt.
blbc        BigLambda compressed binary, 1 bit/entry, C(n-1,3) bits.
            Requires natural labeling.  Stores only triples (a,b,c) with
            1 ≤ a < b < c ≤ n-1; the orientations involving point 0 are
            reconstructed from the signotope axioms via to_small_lambda().

FelsnerMatrix
-------------
fmt         FelsnerMatrix text: n rows of n space-separated bits (0/1).
            Requires natural labeling.
fmb         FelsnerMatrix binary, 1 bit/entry, n² bits packed MSB-first,
            zero-padded to a whole number of bytes.  Row-major order.
            Requires natural labeling.

Points
------
pb04        Point set binary, 4 bits/coordinate (x then y per point), values
            0..15, packed MSB-first, zero-padded to whole bytes.
pb08        Point set binary, 8 bits/coordinate (1 byte unsigned), values
            0..255.
pb16        Point set binary, 16 bits/coordinate (2 bytes, little-endian
            unsigned), values 0..65535.
pb32        Point set binary, 32 bits/coordinate (4 bytes, little-endian
            unsigned), values 0..2³²-1.
b08         Legacy alias for pb08 (Aichholzer OTDB files).
b16         Legacy alias for pb16.
asc / psz   ASCII point set text (n on first line, then n lines of "x y").
json        JSON: one line per order type, list of [x,y] pairs.

Usage
=====
    for ot in read_order_types("otypes06.pb08", n=6):
        print(ot.to_string())

    # legacy OTDB files still work:
    for ot in read_order_types("otypes06.b08", n=6):
        ...
"""

from __future__ import annotations
import math
import struct
import json
from itertools import combinations
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.core.big_lambda import BigLambda
from pyotlib2.core.point_set import PointSet
from pyotlib2.core.utils import binomial


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_order_types(
    filepath: str | Path,
    fmt: Optional[str] = None,
    n: Optional[int] = None,
) -> Iterator[SmallLambda]:
    """Yield SmallLambda objects from *filepath*.

    Parameters
    ----------
    filepath:
        Path to the input file.
    fmt:
        Format string (see module docstring).  If ``None``, inferred from the
        file extension (everything after the last dot, lowercased).
    n:
        Number of points per order type.  Required for all binary formats
        unless it can be inferred from the filename (e.g. ``otypes05.pb08``).
    """
    filepath = Path(filepath)
    if fmt is None:
        fmt = filepath.suffix.lstrip(".").lower()

    # legacy OTDB aliases
    if fmt == "b08":
        fmt = "pb08"
    elif fmt == "b16":
        fmt = "pb16"

    # legacy text aliases (SmallLambda text)
    if fmt in ("txt", "lt"):
        fmt = "slt"

    if n is None and fmt in (
        "pb04", "pb08", "pb16", "pb32",
        "slb04", "slb08", "slb16",
        "slbc04", "slbc08", "slbc16",
        "blb", "blbc", "fmb",
    ):
        n = _infer_n_from_filename(filepath)

    dispatch = {
        "slt":    lambda: _read_small_lambda_text(filepath, n),
        "slb04":  lambda: _read_slb(filepath, n, bits=4, compressed=False),
        "slb08":  lambda: _read_slb(filepath, n, bits=8, compressed=False),
        "slb16":  lambda: _read_slb(filepath, n, bits=16, compressed=False),
        "slbc04": lambda: _read_slb(filepath, n, bits=4, compressed=True),
        "slbc08": lambda: _read_slb(filepath, n, bits=8, compressed=True),
        "slbc16": lambda: _read_slb(filepath, n, bits=16, compressed=True),
        "blt":    lambda: _read_big_lambda_text(filepath, n),
        "blb":    lambda: _read_blb(filepath, n, compressed=False),
        "blbc":   lambda: _read_blb(filepath, n, compressed=True),
        "fmt":    lambda: _read_felsner_text(filepath, n),
        "fmb":    lambda: _read_felsner_binary(filepath, n),
        "pb04":   lambda: _read_point_binary(filepath, n, bits=4),
        "pb08":   lambda: _read_point_binary(filepath, n, bits=8),
        "pb16":   lambda: _read_point_binary(filepath, n, bits=16),
        "pb32":   lambda: _read_point_binary(filepath, n, bits=32),
        "asc":    lambda: _read_point_text_asc(filepath, n),
        "psz":    lambda: _read_point_text_asc(filepath, n),
        "json":   lambda: _read_point_text_json(filepath, n),
    }
    if fmt not in dispatch:
        raise ValueError(
            f"Unknown format {fmt!r}.  Supported: "
            + ", ".join(sorted(dispatch))
        )
    yield from dispatch[fmt]()


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------

def _pack_bits(bits: list[int]) -> bytes:
    """Pack a list of 0/1 ints MSB-first into bytes, zero-padded."""
    n_bytes = math.ceil(len(bits) / 8)
    result = bytearray(n_bytes)
    for k, b in enumerate(bits):
        if b:
            result[k // 8] |= 1 << (7 - k % 8)
    return bytes(result)


def _unpack_bits(data: bytes, count: int) -> list[int]:
    """Unpack *count* bits MSB-first from *data*."""
    bits = []
    for k in range(count):
        bits.append((data[k // 8] >> (7 - k % 8)) & 1)
    return bits


def _require_n(n: Optional[int], fmt: str) -> int:
    if n is None:
        raise ValueError(
            f"n (number of points) is required for format {fmt!r} — "
            f"pass n=N or name the file e.g. otypes05.{fmt}"
        )
    return n


# ---------------------------------------------------------------------------
# SmallLambda text
# ---------------------------------------------------------------------------

def _read_small_lambda_text(
    filepath: Path, n: Optional[int]
) -> Iterator[SmallLambda]:
    with open(filepath, "r") as f:
        if n is None:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                n = len(line.split())
                break
            f.seek(0)

        while True:
            lines = []
            for _ in range(n):
                line = f.readline()
                if line == "":
                    if lines:
                        break
                    return
                while not line.strip() or line.strip().startswith("#"):
                    line = f.readline()
                    if line == "":
                        return
                lines.append(line)
            if len(lines) != n:
                return
            yield SmallLambda.from_string(n, "".join(lines))


# ---------------------------------------------------------------------------
# SmallLambda binary (slb / slbc)
# ---------------------------------------------------------------------------

def _read_slb(
    filepath: Path,
    n: Optional[int],
    bits: int,
    compressed: bool,
) -> Iterator[SmallLambda]:
    fmt_name = f"slb{'c' if compressed else ''}{bits:02d}"
    n = _require_n(n, fmt_name)

    if bits == 4:
        assert n <= 16, f"{fmt_name}: n={n} > 16, values don't fit in 4 bits"
    elif bits == 8:
        assert n <= 256, f"{fmt_name}: n={n} > 256, values don't fit in 8 bits"
    elif bits == 16:
        assert n <= 65536, f"{fmt_name}: n={n} > 65536, values don't fit in 16 bits"

    # number of entries per OT
    if compressed:
        # upper triangle with both indices >= 1: pairs (i,j) with 1<=i<j<=n-1
        count = (n - 1) * (n - 2) // 2
    else:
        count = n * n

    if bits == 4:
        n_bytes = math.ceil(count * 4 / 8)
    else:
        n_bytes = count * (bits // 8)

    with open(filepath, "rb") as f:
        while True:
            data = f.read(n_bytes)
            if len(data) < n_bytes:
                return
            yield _decode_slb(n, data, bits, compressed)


def _decode_slb(n: int, data: bytes, bits: int, compressed: bool) -> SmallLambda:
    """Decode one SmallLambda from raw bytes."""
    # read entries
    if bits == 4:
        count = (n - 1) * (n - 2) // 2 if compressed else n * n
        raw = []
        for k in range(count):
            byte = data[k // 2]
            raw.append((byte >> 4) & 0xF if k % 2 == 0 else byte & 0xF)
    elif bits == 8:
        raw = list(data)
    else:  # 16
        raw = [struct.unpack_from("<H", data, k * 2)[0]
               for k in range(len(data) // 2)]

    L = np.zeros((n, n), dtype=np.int32)

    if compressed:
        # row 0: natural labeling
        for j in range(1, n):
            L[0, j] = j - 1
            L[j, 0] = n - 2 - (j - 1)
        # upper triangle i>=1, j>i
        idx = 0
        for i in range(1, n):
            for j in range(i + 1, n):
                L[i, j] = raw[idx]
                L[j, i] = n - 2 - raw[idx]
                idx += 1
    else:
        idx = 0
        for i in range(n):
            for j in range(n):
                L[i, j] = raw[idx]
                idx += 1

    return SmallLambda(n, L)


# ---------------------------------------------------------------------------
# BigLambda text
# ---------------------------------------------------------------------------

def _read_big_lambda_text(
    filepath: Path, n: Optional[int]
) -> Iterator[SmallLambda]:
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chars = [c for c in line if c in "+-"]
            if n is None:
                cur_n = 3
                while binomial(cur_n, 3) < len(chars):
                    cur_n += 1
                assert binomial(cur_n, 3) == len(chars)
            else:
                cur_n = n
            yield _decode_blt_chars(cur_n, chars)


def _decode_blt_chars(n: int, chars: list[str]) -> SmallLambda:
    o = np.zeros((n, n, n), dtype=np.int8)
    for idx, (a, b, c) in enumerate(combinations(range(n), 3)):
        val = np.int8(1) if chars[idx] == "+" else np.int8(-1)
        o[a, b, c] = o[b, c, a] = o[c, a, b] = val
        o[a, c, b] = o[b, a, c] = o[c, b, a] = -val
    return BigLambda(n, o).to_small_lambda()


# ---------------------------------------------------------------------------
# BigLambda binary (blb / blbc)
# ---------------------------------------------------------------------------

def _read_blb(
    filepath: Path,
    n: Optional[int],
    compressed: bool,
) -> Iterator[SmallLambda]:
    fmt_name = "blbc" if compressed else "blb"
    n = _require_n(n, fmt_name)

    count = binomial(n - 1, 3) if compressed else binomial(n, 3)
    n_bytes = math.ceil(count / 8)

    with open(filepath, "rb") as f:
        while True:
            data = f.read(n_bytes)
            if len(data) < n_bytes:
                return
            bits = _unpack_bits(data, count)
            yield _decode_blb(n, bits, compressed)


def _decode_blb(n: int, bits: list[int], compressed: bool) -> SmallLambda:
    if not compressed:
        o = np.zeros((n, n, n), dtype=np.int8)
        for idx, (a, b, c) in enumerate(combinations(range(n), 3)):
            val = np.int8(1) if bits[idx] else np.int8(-1)
            o[a, b, c] = o[b, c, a] = o[c, a, b] = val
            o[a, c, b] = o[b, a, c] = o[c, b, a] = -val
        return BigLambda(n, o).to_small_lambda()

    # compressed: triples (a,b,c) with 1<=a<b<c<=n-1.
    # Reconstruct L directly:
    #   - natural labeling: L[0,j]=j-1, L[j,0]=n-1-j  for j>=1
    #   - L[i,j] for i,j>=1: count k in {0,1..n-1}\{i,j} with o(i,j,k)=+1
    #     = #{k>=1, k!=i,j : o_sub(i,j,k)=+1} + (1 if o(i,j,0)==+1 else 0)
    #   Point-0 contribution: o(i,j,0)=+1 iff i>j  (verified from natural labeling)
    #
    # Triples (a,b,c) with 1<=a<b<c<=n-1 give us the chirotope on
    # points 1..n-1.  Build the full L matrix using natural labeling
    # for row/column 0 (L[0,j]=j-1, L[j,0]=n-1-j), and recover
    # L[i,j] for i,j>=1 from the BigLambda of points 1..n-1.
        #
    o_sub = np.zeros((n, n, n), dtype=np.int8)
    for idx, (a, b, c) in enumerate(combinations(range(1, n), 3)):
        val = np.int8(1) if bits[idx] else np.int8(-1)
        o_sub[a, b, c] = o_sub[b, c, a] = o_sub[c, a, b] = val
        o_sub[a, c, b] = o_sub[b, a, c] = o_sub[c, b, a] = -val

    L = np.zeros((n, n), dtype=np.int32)
    for j in range(1, n):
        L[0, j] = j - 1
        L[j, 0] = n - 1 - j

    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            cnt = 0
            for k in range(1, n):
                if k == i or k == j:
                    continue
                a, b, c = sorted([i, j, k])
                base = int(o_sub[a, b, c])
                perm = [i, j, k]
                sign = (1 if (perm == [a, b, c] or perm == [b, c, a]
                              or perm == [c, a, b]) else -1)
                if base * sign == 1:
                    cnt += 1
            # point-0 contribution: o(i,j,0)=+1 iff i>j (natural labeling)
            if i > j:
                cnt += 1
            L[i, j] = cnt

    return SmallLambda(n, L)


# ---------------------------------------------------------------------------
# FelsnerMatrix text / binary
# ---------------------------------------------------------------------------

def _read_felsner_text(
    filepath: Path, n: Optional[int]
) -> Iterator[SmallLambda]:
    from pyotlib2.core.felsner_matrix import FelsnerMatrix
    n = _require_n(n, "fmt")
    with open(filepath, "r") as f:
        while True:
            lines = []
            while len(lines) < n:
                line = f.readline()
                if line == "":
                    if lines:
                        break
                    return
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    lines.append(stripped)
            if len(lines) < n:
                return
            block = "\n".join(lines)
            fm = FelsnerMatrix.from_fmt_string(n, block)
            yield fm.to_small_lambda()


def _read_felsner_binary(
    filepath: Path, n: Optional[int]
) -> Iterator[SmallLambda]:
    from pyotlib2.core.felsner_matrix import FelsnerMatrix
    n = _require_n(n, "fmb")
    n_bytes = math.ceil(n * n / 8)
    with open(filepath, "rb") as f:
        while True:
            data = f.read(n_bytes)
            if len(data) < n_bytes:
                return
            fm = FelsnerMatrix.from_fmb_bytes(n, data)
            yield fm.to_small_lambda()


# ---------------------------------------------------------------------------
# Point set binary (pb04 / pb08 / pb16 / pb32)
# ---------------------------------------------------------------------------

def _read_point_binary(
    filepath: Path,
    n: Optional[int],
    bits: int,
) -> Iterator[SmallLambda]:
    fmt_name = f"pb{bits:02d}"
    n = _require_n(n, fmt_name)

    if bits == 4:
        # 2n coordinates packed 4 bits each → ceil(2n*4/8) = n bytes per OT
        n_bytes = math.ceil(2 * n * 4 / 8)
        with open(filepath, "rb") as f:
            while True:
                data = f.read(n_bytes)
                if len(data) < n_bytes:
                    return
                coords = []
                for k in range(2 * n):
                    byte = data[k // 2]
                    coords.append((byte >> 4) & 0xF if k % 2 == 0 else byte & 0xF)
                pts = [(coords[2*i], coords[2*i+1]) for i in range(n)]
                yield PointSet(n, pts).to_small_lambda()
    else:
        struct_fmt = {8: "<B", 16: "<H", 32: "<I"}[bits]
        size = bits // 8
        with open(filepath, "rb") as f:
            while True:
                coords = []
                ok = True
                for _ in range(n):
                    bx = f.read(size)
                    by = f.read(size)
                    if len(bx) < size or len(by) < size:
                        ok = False
                        break
                    coords.append((
                        struct.unpack(struct_fmt, bx)[0],
                        struct.unpack(struct_fmt, by)[0],
                    ))
                if not ok:
                    return
                yield PointSet(n, coords).to_small_lambda()


# ---------------------------------------------------------------------------
# ASCII / PSZ text point set
# ---------------------------------------------------------------------------

def _read_point_text_asc(
    filepath: Path, n: Optional[int]
) -> Iterator[SmallLambda]:
    with open(filepath, "r") as f:
        lines = f.readlines()
    i = 0

    def _skip() -> None:
        # Skip blank lines and '#' comments, matching the other text readers.
        nonlocal i
        while i < len(lines) and (
            not lines[i].strip() or lines[i].strip().startswith("#")
        ):
            i += 1

    while i < len(lines):
        _skip()
        if i >= len(lines):
            break
        cur_n = int(lines[i].strip())
        i += 1
        if n is not None:
            assert cur_n == n
        coords = []
        coloring: Optional[dict] = None
        for idx in range(cur_n):
            _skip()
            parts = lines[i].split()
            # "x y" or, for bicolored point sets (PoSeZo c2), "x y COLOR".
            x, y = int(parts[0]), int(parts[1])
            coords.append((x, y))
            if len(parts) >= 3:
                if coloring is None:
                    coloring = {}
                coloring[idx] = parts[2]
            i += 1
        yield PointSet(cur_n, coords, coloring=coloring).to_small_lambda()


# ---------------------------------------------------------------------------
# JSON point set
# ---------------------------------------------------------------------------

def _read_point_text_json(
    filepath: Path, n: Optional[int]
) -> Iterator[SmallLambda]:
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pts = json.loads(line)
            cur_n = len(pts)
            if n is not None:
                assert cur_n == n
            yield PointSet(cur_n, [tuple(p) for p in pts]).to_small_lambda()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_n_from_filename(filepath: Path) -> Optional[int]:
    """Try to extract n from filenames like otypes05.pb08 or myfile_n7.slb08."""
    import re
    stem = filepath.stem
    m = re.search(r'(?<!\d)(\d{1,2})(?!\d)', stem)
    if m:
        return int(m.group(1))
    return None

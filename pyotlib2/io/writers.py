"""File writers for all pyotlib2 order-type formats.

See ``pyotlib2.io.readers`` for the full format documentation.
"""

from __future__ import annotations
import json
import math
import struct
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.core.utils import binomial


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Text formats whose readers skip lines starting with '#'.  Only these
# support an optional comment header.
_COMMENTABLE = frozenset({"slt", "blt", "fmt", "asc", "psz"})


def write_order_types(
    ots: Iterable[SmallLambda],
    filepath: str | Path,
    fmt: Optional[str] = None,
    header: Optional[str | Iterable[str]] = None,
) -> int:
    """Write order types to *filepath*.  Returns the number of OTs written.

    Parameters
    ----------
    ots:
        Iterable of SmallLambda objects.
    filepath:
        Destination path.
    fmt:
        Format string (see ``pyotlib2.io.readers`` module docstring).
        If ``None``, inferred from the file extension.
    header:
        Optional comment header written at the top of the file.  A string
        (possibly multi-line) or an iterable of lines; each line is prefixed
        with ``# `` if not already starting with ``#``.  The matching readers
        skip these lines, so the header is inert on read.  Only supported for
        text formats that honour ``#`` comments (``slt``, ``blt``, ``fmt``,
        ``asc``, ``psz``); requesting a header for any other format raises
        ``ValueError``.
    """
    filepath = Path(filepath)
    if fmt is None:
        fmt = filepath.suffix.lstrip(".").lower()

    # legacy aliases
    if fmt == "b08":
        fmt = "pb08"
    elif fmt == "b16":
        fmt = "pb16"
    if fmt == "txt":
        fmt = "slt"

    if header is not None and fmt not in _COMMENTABLE:
        raise ValueError(
            f"header is only supported for comment-aware text formats "
            f"({', '.join(sorted(_COMMENTABLE))}), not {fmt!r}."
        )

    dispatch = {
        "slt":    lambda: _write_small_lambda_text(ots, filepath, header),
        "slb04":  lambda: _write_slb(ots, filepath, bits=4, compressed=False),
        "slb08":  lambda: _write_slb(ots, filepath, bits=8, compressed=False),
        "slb16":  lambda: _write_slb(ots, filepath, bits=16, compressed=False),
        "slbc04": lambda: _write_slb(ots, filepath, bits=4, compressed=True),
        "slbc08": lambda: _write_slb(ots, filepath, bits=8, compressed=True),
        "slbc16": lambda: _write_slb(ots, filepath, bits=16, compressed=True),
        "blt":    lambda: _write_big_lambda_text(ots, filepath, header),
        "blb":    lambda: _write_blb(ots, filepath, compressed=False),
        "blbc":   lambda: _write_blb(ots, filepath, compressed=True),
        "fmt":    lambda: _write_felsner_text(ots, filepath, header),
        "fmb":    lambda: _write_felsner_binary(ots, filepath),
        "pb04":   lambda: _write_point_binary(ots, filepath, bits=4),
        "pb08":   lambda: _write_point_binary(ots, filepath, bits=8),
        "pb16":   lambda: _write_point_binary(ots, filepath, bits=16),
        "pb32":   lambda: _write_point_binary(ots, filepath, bits=32),
        "asc":    lambda: _write_point_text_asc(ots, filepath, header),
        "psz":    lambda: _write_point_text_asc(ots, filepath, header),
        "json":   lambda: _write_point_text_json(ots, filepath),
    }
    if fmt not in dispatch:
        raise ValueError(
            f"Unknown format {fmt!r}.  Supported: "
            + ", ".join(sorted(dispatch))
        )
    return dispatch[fmt]()


def _write_header(f, header: Optional[str | Iterable[str]]) -> None:
    """Write a '#'-prefixed comment header to open text file *f* (if any)."""
    if header is None:
        return
    lines = header.splitlines() if isinstance(header, str) else list(header)
    for line in lines:
        line = line.rstrip("\n")
        f.write(line if line.startswith("#") else f"# {line}")
        f.write("\n")


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


def _assert_natural(sl: SmallLambda, fmt: str) -> None:
    L = sl.get_l()
    n = sl.n
    for j in range(1, n):
        if int(L[0, j]) != j - 1:
            raise ValueError(
                f"{fmt}: order type is not in natural labeling "
                f"(L[0,{j}]={int(L[0,j])}, expected {j-1}). "
                f"Use sl.relabeled(sl.get_natural_labeling(p0)) first."
            )


# ---------------------------------------------------------------------------
# SmallLambda text
# ---------------------------------------------------------------------------

def _write_small_lambda_text(
    ots: Iterable[SmallLambda], filepath: Path,
    header: Optional[str | Iterable[str]] = None,
) -> int:
    cnt = 0
    with open(filepath, "w") as f:
        _write_header(f, header)
        for ot in ots:
            f.write(ot.to_string())
            f.write("\n")
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# SmallLambda binary (slb / slbc)
# ---------------------------------------------------------------------------

def _write_slb(
    ots: Iterable[SmallLambda],
    filepath: Path,
    bits: int,
    compressed: bool,
) -> int:
    fmt_name = f"slb{'c' if compressed else ''}{bits:02d}"
    cnt = 0
    with open(filepath, "wb") as f:
        for ot in ots:
            L = ot.get_l()
            n = ot.n
            if bits == 4:
                assert n <= 16, f"{fmt_name}: n={n} > 16"
            elif bits == 8:
                assert n <= 256, f"{fmt_name}: n={n} > 256"
            elif bits == 16:
                assert n <= 65536, f"{fmt_name}: n={n} > 65536"
            if compressed:
                _assert_natural(ot, fmt_name)
            f.write(_encode_slb(n, L, bits, compressed))
            cnt += 1
    return cnt


def _encode_slb(n: int, L: np.ndarray, bits: int, compressed: bool) -> bytes:
    if compressed:
        # upper triangle with both indices >= 1
        entries = [int(L[i, j]) for i in range(1, n) for j in range(i + 1, n)]
    else:
        entries = [int(L[i, j]) for i in range(n) for j in range(n)]

    if bits == 4:
        # pack two nibbles per byte, MSB first
        n_bytes = math.ceil(len(entries) * 4 / 8)
        result = bytearray(n_bytes)
        for k, v in enumerate(entries):
            if k % 2 == 0:
                result[k // 2] |= (v & 0xF) << 4
            else:
                result[k // 2] |= v & 0xF
        return bytes(result)
    elif bits == 8:
        return bytes(entries)
    else:  # 16
        return b"".join(struct.pack("<H", v) for v in entries)


# ---------------------------------------------------------------------------
# BigLambda text
# ---------------------------------------------------------------------------

def _write_big_lambda_text(
    ots: Iterable[SmallLambda], filepath: Path,
    header: Optional[str | Iterable[str]] = None,
) -> int:
    cnt = 0
    with open(filepath, "w") as f:
        _write_header(f, header)
        for ot in ots:
            f.write(ot.to_big_lambda().to_string() + "\n")
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# BigLambda binary (blb / blbc)
# ---------------------------------------------------------------------------

def _write_blb(
    ots: Iterable[SmallLambda],
    filepath: Path,
    compressed: bool,
) -> int:
    fmt_name = "blbc" if compressed else "blb"
    cnt = 0
    with open(filepath, "wb") as f:
        for ot in ots:
            if compressed:
                _assert_natural(ot, fmt_name)
            o = ot.to_big_lambda().o
            n = ot.n
            if compressed:
                triples = list(combinations(range(1, n), 3))
            else:
                triples = list(combinations(range(n), 3))
            bits = [1 if int(o[a, b, c]) > 0 else 0 for a, b, c in triples]
            f.write(_pack_bits(bits))
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# FelsnerMatrix text / binary
# ---------------------------------------------------------------------------

def _write_felsner_text(
    ots: Iterable[SmallLambda], filepath: Path,
    header: Optional[str | Iterable[str]] = None,
) -> int:
    from pyotlib2.core.felsner_matrix import FelsnerMatrix
    cnt = 0
    with open(filepath, "w") as f:
        _write_header(f, header)
        for ot in ots:
            fm = FelsnerMatrix.from_small_lambda(ot)
            f.write(fm.to_fmt_string())
            f.write("\n")
            cnt += 1
    return cnt


def _write_felsner_binary(ots: Iterable[SmallLambda], filepath: Path) -> int:
    from pyotlib2.core.felsner_matrix import FelsnerMatrix
    cnt = 0
    with open(filepath, "wb") as f:
        for ot in ots:
            fm = FelsnerMatrix.from_small_lambda(ot)
            f.write(fm.to_fmb_bytes())
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# Point set binary (pb04 / pb08 / pb16 / pb32)
# ---------------------------------------------------------------------------

def _write_point_binary(
    ots: Iterable[SmallLambda],
    filepath: Path,
    bits: int,
) -> int:
    fmt_name = f"pb{bits:02d}"
    cnt = 0
    with open(filepath, "wb") as f:
        for ot in ots:
            assert ot.realization is not None, (
                f"{fmt_name}: realization required for point binary output"
            )
            if bits == 4:
                coords = []
                for x, y in ot.realization:
                    coords += [int(x), int(y)]
                n_bytes = math.ceil(len(coords) * 4 / 8)
                result = bytearray(n_bytes)
                for k, v in enumerate(coords):
                    if k % 2 == 0:
                        result[k // 2] |= (v & 0xF) << 4
                    else:
                        result[k // 2] |= v & 0xF
                f.write(bytes(result))
            else:
                struct_fmt = {8: "<B", 16: "<H", 32: "<I"}[bits]
                for x, y in ot.realization:
                    f.write(struct.pack(struct_fmt, int(x)))
                    f.write(struct.pack(struct_fmt, int(y)))
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# ASCII point set text
# ---------------------------------------------------------------------------

def _write_point_text_asc(
    ots: Iterable[SmallLambda], filepath: Path,
    header: Optional[str | Iterable[str]] = None,
) -> int:
    cnt = 0
    with open(filepath, "w") as f:
        _write_header(f, header)
        for ot in ots:
            assert ot.realization is not None
            f.write(f"{ot.n}\n")
            for i, (x, y) in enumerate(ot.realization):
                # bicolored point sets (PoSeZo c2): "x y COLOR", as read back
                color = (ot.coloring or {}).get(i)
                f.write(f"{x} {y}\n" if color is None else f"{x} {y} {color}\n")
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# JSON point set
# ---------------------------------------------------------------------------

def _write_point_text_json(ots: Iterable[SmallLambda], filepath: Path) -> int:
    cnt = 0
    with open(filepath, "w") as f:
        for ot in ots:
            assert ot.realization is not None
            f.write(json.dumps([[int(x), int(y)] for x, y in ot.realization]) + "\n")
            cnt += 1
    return cnt

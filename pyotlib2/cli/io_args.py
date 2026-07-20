"""Shared argparse arguments and I/O helpers for CLI commands."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterator

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.io.readers import read_order_types
from pyotlib2.io.writers import write_order_types


def add_input_args(parser) -> None:
    parser.add_argument("input", help="Input file path")
    parser.add_argument(
        "--fmt", "--ft",
        dest="fmt",
        default=None,
        help="File format (lt, blt, b08, b16, b32, b64, asc, json). "
             "Default: inferred from extension.",
    )
    parser.add_argument(
        "-n", "--points",
        dest="n",
        type=int,
        default=None,
        help="Number of points per order type (required for binary formats).",
    )
    parser.add_argument(
        "--hash-length",
        dest="hash_length",
        type=int,
        default=6,
        help="Number of hex digits shown for order-type hashes (default: 6).",
    )


def add_output_args(parser, default_suffix: str = ".out") -> None:
    parser.add_argument(
        "-o", "--output",
        dest="output",
        default=None,
        help="Output file path. Default: input path + suffix.",
    )
    parser.add_argument(
        "--ofmt", "--oft",
        dest="ofmt",
        default=None,
        help="Output format. Default: same as input.",
    )


def open_input(args) -> Iterator[SmallLambda]:
    return read_order_types(args.input, fmt=args.fmt, n=args.n)


def resolve_output(args, suffix: str = ".out") -> tuple:
    """Return (output_path, output_fmt)."""
    if args.output:
        out = Path(args.output)
    else:
        out = Path(args.input).with_suffix("")
        out = Path(str(out) + suffix)
    fmt = args.ofmt or (args.fmt or Path(args.input).suffix.lstrip("."))
    return out, fmt


def progress_iter(it: Iterator, label: str = "processing") -> Iterator:
    """Wrap an iterator with simple stderr progress output."""
    from datetime import datetime
    from pyotlib2.logging import log
    start = datetime.now()
    show = 1
    for cnt, item in enumerate(it):
        if cnt == show:
            elapsed = datetime.now() - start
            log(f"{label}: {cnt} ({elapsed})", stream=sys.stderr)
            show = min(show * 2, show + 1000) if show < 1000 else show + 1000
        yield item
    log(f"{label}: done", stream=sys.stderr)

"""Uniform logging / timestamping for pyotlib2.

One place that decides how progress lines look, so every command logs the same
way.  Format of a timestamp: ``[YYYY-MM-DD HH:MM:SS.mmm]`` (date, time, and
milliseconds) — matching the pyotlib1 convention but with millisecond precision.

Use :func:`log` for a plain timestamped line, or :func:`log_fields` for the
compact column layout used by the realizer (time, then space-separated fields).
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional, TextIO


def timestamp() -> str:
    """Return the current time as ``[YYYY-MM-DD HH:MM:SS.mmm]``."""
    now = datetime.now()
    return f"[{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d}]"


def log(*parts: object, stream: Optional[TextIO] = None) -> None:
    """Print a timestamped log line: ``[ts] part part ...`` (space-joined)."""
    (stream or sys.stdout).write(
        timestamp() + " " + " ".join(str(p) for p in parts) + "\n"
    )
    (stream or sys.stdout).flush()


def log_fields(fields: list, widths: Optional[list] = None,
               stream: Optional[TextIO] = None) -> None:
    """Print a timestamped line of columns: ``[ts] f0  f1  f2 ...``.

    ``widths`` left-justifies each field to the given width for tabular output;
    missing/short widths fall back to the raw string.
    """
    cells = []
    for i, f in enumerate(fields):
        s = str(f)
        if widths and i < len(widths) and widths[i]:
            s = s.ljust(widths[i])
        cells.append(s)
    (stream or sys.stdout).write(timestamp() + "  " + "  ".join(cells) + "\n")
    (stream or sys.stdout).flush()

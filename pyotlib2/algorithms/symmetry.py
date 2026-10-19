"""Rotationally symmetric point sets.

Literal port of old pyotlib ``Symmetry.computeSymmetricPS``
(old/pyotlib/pyotlib/Symmetry.py:8).
"""

from __future__ import annotations
from copy import copy
from math import cos, floor, sin, pi


def _round_py2(v) -> int:
    """Python-2 round(): halves are rounded away from zero (Python 3 rounds to even)."""
    return int(floor(v + 0.5)) if v >= 0 else -int(floor(-v + 0.5))


def compute_symmetric_ps(P0: list, k: int) -> list:
    """Make a point list k-fold rotationally symmetric around the origin.

    Points are processed in blocks of k: point i (i = 0, k, 2k, ...) is kept
    and points i+1 .. i+k-1 are replaced by p_i rotated by 2*j*pi/k
    (j = 1..k-1), each coordinate rounded to the nearest integer.  If
    n % k == 1, the last point is set to the origin (0, 0).
    """
    P = copy(P0)
    n = len(P)

    assert n % k in [0, 1]
    if n % k == 1:
        P[-1] = (0, 0)

    def _rotate_point(p, alpha):
        return (_round_py2(cos(alpha) * p[0] - sin(alpha) * p[1]),
                _round_py2(sin(alpha) * p[0] + cos(alpha) * p[1]))

    for i in range(0, n, k):
        for j in range(1, k):
            if i + j < n:
                P[i + j] = _rotate_point(P[i], 2 * j * pi / k)

    return P

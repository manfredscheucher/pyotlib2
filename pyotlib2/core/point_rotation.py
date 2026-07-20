"""Realization transport for BigLambda.from_flips.

``BigLambda.from_flips`` applies a flip vector ``F`` by sign-flipping the
chirotope::

    o'[i,j,k] = G_i G_j G_k o[i,j,k]        with  G_i = 1 - 2 F_i  (∈ {+1,-1}).

In homogeneous coordinates ``v_i = (x_i, y_i, 1)`` we have
``o[i,j,k] = sign det[v_i, v_j, v_k]``, so ``o'`` is exactly the chirotope of the
*signed* homogeneous vectors ``w_i = G_i v_i``.  Flipping an extremal point is a
**projective** operation (it moves the point across the line at infinity), NOT a
Euclidean reflection through the origin: negating only the affine part
``(x,y) -> (-x,-y)`` keeps the homogeneous ``1`` and therefore contributes the
wrong sign (``(-1)(-1)(+1) = +1`` instead of ``-1`` per flipped index).

To turn the projective configuration ``{[w_i]}`` back into finite affine points
we send an appropriate line at infinity to infinity: pick a covector ``c`` with
``c · w_i > 0`` for every ``i`` (a strict separator — it exists because the
flipped order type is itself realizable, so ``{[w_i]}`` lies in some affine
chart), then map through a chart ``T`` whose third row is ``c``.  ``T`` is chosen
with ``det T > 0`` so orientations are preserved *exactly*, not merely up to a
global sign.
"""

from __future__ import annotations

import numpy as np


def rotate_by_flip(points: list, F: list) -> list:
    """Return an affine realization of the flip-transformed point set.

    ``points`` is a list of ``(x, y)`` realizing some chirotope ``o``; the result
    realizes ``o'[i,j,k] = G_i G_j G_k o[i,j,k]`` with ``G_i = 1 - 2 F_i``.

    Raises ``ValueError`` if no affine chart exists for the flipped configuration
    (should not happen for a genuinely realizable flip result).
    """
    if not any(F):
        return [(x, y) for (x, y) in points]

    from scipy.optimize import linprog

    n = len(points)
    G = np.array([1.0 - 2.0 * int(f) for f in F])
    # signed homogeneous vectors w_i = G_i * (x_i, y_i, 1)
    W = np.array([[G[i] * x, G[i] * y, G[i]] for i, (x, y) in enumerate(points)],
                 dtype=float)

    # Strict separator c: W @ c >= 1  (all w_i strictly on the positive side).
    res = linprog(
        c=np.zeros(3),
        A_ub=-W,
        b_ub=-np.ones(n),
        bounds=[(None, None)] * 3,
        method="highs",
    )
    if not res.success:
        raise ValueError("rotate_by_flip: flipped configuration has no affine chart")
    c = np.asarray(res.x, dtype=float)
    c = c / np.linalg.norm(c)

    # Chart T with third row = c and det T > 0, so (T w_i)[2] = c·w_i > 0 and
    # orientations are preserved exactly.  Complete c to a right-handed basis.
    tmp = np.eye(3)[int(np.argmin(np.abs(c)))]
    e1 = np.cross(c, tmp)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(c, e1)  # already unit length and orthogonal to c and e1
    T = np.vstack([e1, e2, c])  # det = |e1|^2 |c|^2 > 0

    Q = W @ T.T  # (n, 3); Q[:,2] = W @ c > 0
    return [(float(Q[i, 0] / Q[i, 2]), float(Q[i, 1] / Q[i, 2])) for i in range(n)]

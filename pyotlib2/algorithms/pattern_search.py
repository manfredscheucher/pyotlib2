"""Generic pattern search (coordinate-wise direct search).

Literal port of old pyotlib ``PatternSearch.py``
(old/pyotlib/pyotlib/PatternSearch.py).  In the old library it was used by
ScipyRealizationTester (method ``my_pattern_search``), not by the coordinate
minimizers; it is ported here as a standalone optimizer.
"""

from __future__ import annotations
from copy import copy


def my_pattern_search_step(f, X0, s0):
    """One pattern-search step with step size s0.  Returns (X0, s0)."""
    n = len(X0)
    f0 = f(X0)

    Xlast = copy(X0)
    flast = f0

    for i in range(len(X0)):
        Xnew = copy(Xlast)
        for sign in [+1, -1]:
            Xnew[i] = Xlast[i] + sign * s0
            fnew = f(Xnew)
            if fnew < flast:
                Xlast = Xnew
                flast = fnew
                break

    flast = f(Xlast)
    if flast < f0:
        X0 = [2 * Xlast[i] - X0[i] for i in range(n)]
        f0 = f(X0)
        if f0 > flast:
            X0, f0 = Xlast, flast
        s0 *= 1.1
    else:
        s0 = s0 / 2.0

    return X0, s0


def my_pattern_search(f, X0, s0, minstep=10 ** -10, maxiter=None, DEBUG=False):
    """Iterate my_pattern_search_step until f0 == 0, s0 < minstep or maxiter.

    Returns {'X0': X0, 'f0': f(X0), 'iterations': i}.
    """
    i = 0
    while i != maxiter:
        i += 1
        f0 = f(X0)

        if DEBUG:
            if DEBUG > 1:
                print("it", i)
                print("X0", X0)
                print("f0", f0)
                print("s0", s0)
                print()
            else:
                print("it", i, "\tf0", f0, "\ts0", s0)

        if f0 == 0:
            break
        if s0 < minstep:
            break
        assert s0 > 0
        X0, s0 = my_pattern_search_step(f, X0, s0)

    return {'X0': X0, 'f0': f(X0), 'iterations': i}

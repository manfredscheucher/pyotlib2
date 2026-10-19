"""Simulated-annealing helpers from PyDCG (no external dependency).

Literal port of the pieces of PyDCG ``heuristics.py``
(old/PyDCG/PyDCG/heuristics.py, Copyright (C) 2015 Ruy Fabila Monroy, GPLv2)
that the old pyotlib simanMinimize relies on.

NOT ported: ``simmulated_annealing_iterator`` -- old pyotlib calls it
(PointSetOptimizer.simanMinimize, scripts/pyotlib.PolygonMinimization.py),
but it does not exist in the PyDCG source shipped in old/PyDCG, so there
is nothing to port literally.  ``greedy`` is not ported either (its default
objective is PyDCG's crossing.count_crossings, and nothing in pyotlib uses it).
"""

from __future__ import annotations
import math
import random
import time


def kirkpatrick_cooling(start_temp, alpha):
    """Infinite generator start_temp, start_temp*alpha, start_temp*alpha^2, ..."""
    T = start_temp
    while True:
        yield T
        T = alpha * T


def rand_move(p, t=1000000):
    """Moves point uniformly on a kxk square randomly to a new location"""
    l = 1.0 / max(float(t), 1)
    tx = random.expovariate(l)
    l = 1.0 / max(float(t), 1)
    ty = random.expovariate(l)
    tx = int(tx)
    ty = int(ty)
    if random.randint(0, 1) == 0:
        tx = -tx
    if random.randint(0, 1) == 0:
        ty = -ty
    p[0] = p[0] + tx
    p[1] = p[1] + ty


def P(vcurrent, vnew, T, minimize=True):
    """Computes the probability of accepting the new value.
       It also runs the corresponding probabilitic trial.
       Returns True if the solution should be accepted
       false otherwise."""

    df = vcurrent - vnew
    if not minimize:
        df = -df
    if df >= 0:
        return True

    df = float(df)
    p = math.exp(df / float(T))
    if random.random() <= p:
        return True
    return False


def simmulated_annealing(n=10, pts=[], run_time=10, k=10000000,
                         k_f=kirkpatrick_cooling(10000000, 0.999),
                         f=[], T=kirkpatrick_cooling(100, 0.999),
                         rand_move=rand_move, minimize=True,
                         print_function=None):
    """Implementation of a simulated annealing algorithm to search for good point sets.

    Note (kept from the original): the default ``pts``, ``k_f`` and ``T``
    objects are created once at definition time and shared between calls.
    """

    for i in range(len(pts), n):
        pts.append([random.randint(-k, k), random.randint(-k, k)])

    n = len(pts)
    start_time = time.time()
    vcurrent = f(pts)
    while time.time() - start_time < run_time:
        idxp = random.randint(0, n - 1)
        p = pts[idxp]
        q = p[:]
        rand_move(p, int(next(k_f)))
        vnew = f(pts)
        if P(vcurrent, vnew, next(T), minimize=minimize):
            if vnew != vcurrent:
                if print_function is None:
                    print(vnew)
                else:
                    print_function(vnew)
            vcurrent = vnew
        else:
            p[0] = q[0]
            p[1] = q[1]

    return pts

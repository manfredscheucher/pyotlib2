"""Coordinate minimization / beautification (order type preserved).

Literal Python-3 port of old pyotlib ``PointSetOptimizer.py``
(old/pyotlib/pyotlib/PointSetOptimizer.py) and of the ``optimize`` routine of
the script ``scripts/pyotlib.MinimizePatternSearch.py``.

Same algorithms, constants, defaults and control flow as the originals; only
Python-2 idioms are translated (``print`` statements, integer ``/`` -> ``//``,
``zip`` -> ``list(zip(...))``, ``gen.next()`` -> ``next(gen)``) and the old
API names are mapped to pyotlib2 (``getBits`` -> ``bits``, ``getDiameter`` ->
``diameter``, ``calcOrientation`` -> ``orientation``, ...).  Quirks of the
originals are kept on purpose (see the "NOTE (original)" comments).

These functions are independent of the (non-literal) ports
``minimize_coords`` / ``beautify_coords`` / ``beautify2_coords`` in
``pyotlib2.cli.commands``.

NOT ported:
  * ``Beautify350Wrapper.call_beautify350``: calls the external executable
    ``beautify350`` whose source is not available.
"""

from __future__ import annotations
import math
import sys
from copy import copy
from fractions import Fraction
from itertools import combinations
from math import gcd, log, sqrt
from random import randint

from pyotlib2.algorithms.exit_edges import filter_exit_edges
from pyotlib2.algorithms.symmetry import compute_symmetric_ps
from pyotlib2.core.point_set import PointSet
from pyotlib2.core.utils import sign

DEBUG = False

PROVIDE_ALL = False
LAZY_CHECK_COCIRC = True

# Module globals exactly as in the original (set by
# minimize_integer_coordinates_private, read by same_ot_as_bl_).
PS0 = None
COMBS = None


# ---------------------------------------------------------------------------
# helpers ported from other old modules
# ---------------------------------------------------------------------------

def _to_integer(D_orig):
    """Port of ScipyRealizationTester._toInteger: scale exactly to integers.

    Converts every value to an exact Fraction and repeatedly multiplies all
    values by the first denominator > 1 until all are integers.
    """
    D = [Fraction(d) for d in D_orig]
    while True:
        c = 1
        for d in D:
            if d.denominator > c:
                c = d.denominator
                break

        if c == 1:
            break
        D = [d * c for d in D]

    return [d.numerator for d in D]


def _line(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    dx = x2 - x1
    dy = y2 - y1
    a = dx
    b = dy
    c1 = a * x1 + b * y1
    c2 = a * x2 + b * y2
    return (2 * a, 2 * b, -(c1 + c2))


def _cross(v1, v2):
    (x1, y1, h1), (x2, y2, h2) = v1, v2
    return (y1 * h2 - y2 * h1, h1 * x2 - h2 * x1, x1 * y2 - x2 * y1)


def _lexmin_vec(v):
    x, y, h = v
    g = gcd(gcd(x, y), h) * sign(h)
    assert g != 0  # collinear points!
    return (x // g, y // g, h // g)


def _circumcenter(A, B, C):
    """Port of mygeometry.circumcenter (homogeneous, normalized)."""
    AB = _line(A, B)
    AC = _line(A, C)
    X = _lexmin_vec(_cross(AB, AC))
    assert X != (0, 0, 0)
    return X


def has_cocircular_points(P) -> bool:
    """Port of mygeometry.hasCoCircularPoints: two triples share a circumcenter."""
    n = len(P)
    L = set()
    for i, j, k in combinations(range(n), 3):
        C = _circumcenter(P[i], P[j], P[k])
        if C in L:
            return True
        L.add(C)
    return False


# ---------------------------------------------------------------------------
# minimizeIntegerCoordinates  (PointSetOptimizer.py:32-304)
# ---------------------------------------------------------------------------

def _balance_points(points, BlowUpConstant=1):
    # balance X and Y coords (scale smaller coords)
    x_max = max([x for (x, y) in points])
    y_max = max([y for (x, y) in points])

    Mx = max(BlowUpConstant, (BlowUpConstant * y_max) // x_max)
    My = max(BlowUpConstant, (BlowUpConstant * x_max) // y_max)
    # both dimensions should have same size ...

    return [(x * Mx, y * My) for (x, y) in points]


def minimize_integer_coordinates(PS, BL=None, SLstr=None, maxtrials=1, randomize=10,
                                 tolerance=None, sheering=False, minimize_diameter=False,
                                 ensure_property=None, symmetric=0):
    """Port of minimizeIntegerCoordinates.  PS is a pyotlib2 PointSet; returns a PointSet."""
    if tolerance == 0:
        tolerance = None
    if PROVIDE_ALL:
        assert not has_cocircular_points(PS.points)

    if minimize_diameter:
        quality_function = lambda PS: PS.diameter()
    else:
        quality_function = lambda PS: PS.bits()

    if not symmetric:
        PS = PS.normalized()
    best_bits = start_bits = quality_function(PS)
    PS_best = PS
    if tolerance and start_bits <= tolerance:
        return PS

    if BL is None:
        BL = PS.to_big_lambda()
    if SLstr is None:
        SLstr = BL.to_small_lambda().to_string()

    dx_ = [0]
    dy_ = [0]
    if sheering:
        dx_ += [-1, +1]
        dy_ += [-1, +1]

    for dx in dx_:
        for dy in dy_:
            if dx * dy != 0:
                continue
            if DEBUG:
                print("TRY SHEERING:", dx, dy)

            PS1 = PointSet(PS.n, [(x + y * dx, y + x * dy) for (x, y) in PS_best.points], PS.coloring)
            PS2 = minimize_integer_coordinates_private(quality_function, PS1, BL, SLstr, maxtrials,
                                                       randomize, tolerance, ensure_property, symmetric)

            if PROVIDE_ALL and has_cocircular_points(PS2.points):
                continue

            end_bits = quality_function(PS2)
            # NOTE (original): best_bits is never updated here, so every
            # sheering result better than the *start* replaces PS_best.
            if end_bits < best_bits:
                PS_best = PS2
            if tolerance and end_bits <= tolerance:
                break

    if PROVIDE_ALL:
        assert not has_cocircular_points(PS_best.points)
    return PS_best


def minimize_integer_coordinates_private(quality_function, PS, BL=None, SLstr=None, maxtrials=1,
                                         randomize=10, tolerance=None, ensure_property=None,
                                         symmetric=0):
    """Port of minimizeIntegerCoordinates_private: repeat the inner minimizer while it improves."""
    global PS0
    PS0 = copy(PS)

    if not symmetric:
        PS = PS.normalized()
    start_bits = quality_function(PS)
    if tolerance and start_bits <= tolerance:
        return PS

    if BL is None:
        BL = PS.to_big_lambda()

    SL = BL.to_small_lambda()
    if SLstr is None:
        SLstr = SL.to_string()

    if DEBUG:
        print("[D] calculate exit-edges")
    edges, witnesses = filter_exit_edges(SL, return_witnesses=True)

    if DEBUG:
        print("[D] exit-edges edge")

    exit_triples = []
    for a, b in edges:
        for c in witnesses[(a, b)]:
            exit_triples.append((a, b, c))

    global COMBS
    COMBS = {i: set() for i in range(SL.n)}
    for a, b, c in exit_triples:
        COMBS[a].add(tuple(sorted((b, c))))
        COMBS[b].add(tuple(sorted((a, c))))
        COMBS[c].add(tuple(sorted((a, b))))

    while True:
        cur_bits = quality_function(PS)
        if tolerance and cur_bits <= tolerance:
            break

        PS2 = _minimize_integer_coordinates_priv(quality_function, PS, BL, SLstr, maxtrials,
                                                 randomize, tolerance, ensure_property, symmetric)
        new_bits = quality_function(PS2)

        if new_bits >= cur_bits:
            break
        PS = PS2

    return PS


def same_ot_(PS1, PS2, i) -> bool:
    """Port of sameOT_: all triples (a, b, i) have equal orientation in PS1 and PS2."""
    for (a, b) in combinations(range(PS1.n), 2):
        if a == i or b == i:
            continue
        if PS1.orientation(a, b, i) != PS2.orientation(a, b, i):
            return False
    return True


def same_ot_as_bl_(BL, PS, i) -> bool:
    """Port of sameOTasBL_: check only the exit-triples through point i (global COMBS)."""
    # 2. same ordertype (implies no collinear)
    for (a, b) in COMBS[i]:
        if BL.o[a][b][i] != PS.orientation(a, b, i):
            return False
    return True


def _is_valid_point_set(n, BL, PS, i, ensure_property) -> bool:
    # 1. INTEGER -> check

    # 2. same ordertype (implies no collinear)
    if not same_ot_as_bl_(BL, PS, i):
        if not ensure_property:
            return False
        else:
            # NOTE (original): the old code disabled this branch with
            # assert(not ensure_property) ("TODO: implement faster, currently
            # really too slow"); the assert is dropped here, the rest is literal.
            if PS.has_collinear_points():
                return False
            from pyotlib2.cli.commands import _count_property
            ct = _count_property(PS.to_small_lambda(), ensure_property)
            property_threshold = _count_property(BL.to_small_lambda(), ensure_property)
            if ct > property_threshold:
                return False

    if PROVIDE_ALL:
        # 3. different X/Y-coords
        X = set([x for (x, y) in PS.points])
        if len(X) < PS.n:
            return False

        Y = set([y for (x, y) in PS.points])
        if len(Y) < PS.n:
            return False

        # 4. euclidean distance squared > 16.
        # scaling input pointset with other properties by factor 4 provides this property

        if not LAZY_CHECK_COCIRC:
            # 5. noncyclic points
            if has_cocircular_points(PS.points):
                return False

    # 6. Lexmin Labeling -> check

    return True


def _move_point_to_even(n, PS, i, BL, ensure_property) -> bool:
    (x, y) = p0 = PS.points[i]

    if x % 2:
        x += 1
    if y % 2:
        y += 1

    if (x, y) == p0:
        return True
    # NOTE (original): the rounded-up point (x, y) itself is NOT tried
    # (commented out in the original); only these three candidates are.
    for p in [(x - 2, y - 2), (x - 2, y), (x, y - 2)]:
        PS.points[i] = p
        if _is_valid_point_set(n, BL, PS, i, ensure_property):
            return True

    PS.points[i] = p0
    return False


def _randomize_point(n, PS, i, BL, bits, ensure_property) -> None:
    (x, y) = p0 = PS.points[i]

    for e in range(bits, 0, -1):
        M = 2 ** e
        for iteration in range(10):
            PS.points[i] = (x + randint(-M, M), y + randint(-M, M))
            if _is_valid_point_set(n, BL, PS, i, ensure_property):
                return

    PS.points[i] = p0


def _minimize_integer_coordinates_priv(quality_function, PS, BL, SLstr, maxtrials, randomize,
                                       tolerance, ensure_property, symmetric):
    """Port of _minimizeIntegerCoordinatesPriv (one even/halve/randomize/blow-up run)."""
    n = PS.n
    start_bits = quality_function(PS)
    PS_tmp = PointSet(n, copy(PS.points), PS.coloring)

    if not symmetric:
        PS_tmp = PS_tmp.normalized()
    PS_tmp.points = _balance_points(PS_tmp.points)
    assert PS_tmp.bits() == PS.bits()

    best_bits = start_bits
    best_pts = copy(PS.points)

    if DEBUG:
        print(" + minimize: ", "(", start_bits, ")", "its:", maxtrials)

    iter = 0

    while iter < maxtrials:
        if DEBUG > 1:
            print("phase 1: Minimize", PS_tmp.points)
        for it in range(1):
            ok = True
            for i in range(n):
                if not _move_point_to_even(n, PS_tmp, i, BL, ensure_property):
                    ok = False
            if ok:
                break

        if ok:
            if DEBUG > 1:
                print("phase 2A: Half")
            backup = copy(PS_tmp.points)
            PS_tmp.points = [(x // 2, y // 2)
                             for (x, y) in (PS_tmp.normalized() if not symmetric else PS_tmp).points]

            if symmetric:
                PS_tmp.points = compute_symmetric_ps(PS_tmp.points, symmetric)
                if PS_tmp.to_small_lambda().to_string() != SLstr:
                    PS_tmp.points = backup  # backup old solution
                    ok = False

            if ok:
                bits = quality_function(PS_tmp)
                if bits < best_bits:
                    best_bits = bits
                    best_pts = copy(PS_tmp.points)
                    if DEBUG:
                        print("->", best_bits, end=" ")
                    sys.stdout.flush()
                    iter = 0

                if tolerance and bits <= tolerance:
                    break

        if not ok:
            if randomize:
                if DEBUG > 1:
                    print("phase 2B: Randomize")

                bits = quality_function(PS_tmp)
                for it in range(randomize):
                    for i in range(n):
                        _randomize_point(n, PS_tmp, i, BL, randint(0, PS_tmp.bits()), ensure_property)

            iter += 1
            if iter == maxtrials // 2:
                if DEBUG > 1:
                    print("phase 2C: Blow Up")

                PS_tmp.points = _balance_points(
                    (PS_tmp.normalized() if not symmetric else PS_tmp).points, BlowUpConstant=3)

    PS_new = PointSet(PS.n, best_pts, PS.coloring)
    if not symmetric:
        PS_new = PS_new.normalized()
    bits_new = quality_function(PS_new)

    if DEBUG:
        print(" - minimized:", "(", bits_new, ")          => -", start_bits - bits_new)

    if PROVIDE_ALL and LAZY_CHECK_COCIRC:
        # 5. noncyclic points
        if has_cocircular_points(PS_new.points):
            if DEBUG:
                print("GOT smaller coords but cocircular points... :/")
            return PS

    # NOTE: with ensure_property the order type may change, so the original's
    # order-type assert only applies without it.
    assert ensure_property or PS_new.to_small_lambda().to_string() == SLstr  # optimization failed?!
    return PS_new


# ---------------------------------------------------------------------------
# beautifyPointSet  (PointSetOptimizer.py:308)
# ---------------------------------------------------------------------------

def beautify_point_set(PS0, maxiter=100, goodbits=None, retries=2):
    """Port of beautifyPointSet.  PS0 is a pyotlib2 PointSet; returns a PointSet.

    NOTE (original): ``retries`` is accepted but not used by the algorithm.
    """
    n = PS0.n
    BL = PS0.to_big_lambda()
    SL = BL.to_small_lambda()

    if DEBUG:
        print("start with", PS0.normalized().points)

    assert len(BL.o) == n

    orientations = dict([((a, b, c), BL.o[a][b][c]) for a in range(n) for b in range(n)
                         for c in range(n) if a < b and a < c and BL.o[a][b][c] > 0])

    orientations2 = dict([(a0, dict([(abc, orientations[abc]) for abc in orientations if a0 in abc]))
                          for a0 in range(n)])

    def isvalid(Z, i=None):
        X = Z[:n]
        Y = Z[n:]

        o2 = lambda a, b: X[a] * Y[b] - Y[a] * X[b]
        o3 = lambda a, b, c: o2(a, b) + o2(b, c) + o2(c, a)

        for (a, b, c) in (orientations2[i % n] if i is not None else orientations):
            if o3(a, b, c) < 1:
                return False
        return True

    curbits = PS0.normalized().bits()
    if goodbits is None:
        goodbits = curbits

    X, Y = zip(*PS0.normalized().points)
    Z = list(X + Y)

    if curbits < goodbits:
        Z = [z * 2 ** (goodbits - curbits) for z in Z]
        curbits = goodbits

    min_diff = 2 * n * 2 ** goodbits
    stop_timer = 5

    for it in range(1, maxiter + 1):
        diffs = 0
        grad = []
        Z0 = copy(Z)

        for i in range(2 * n):
            z0 = Z[i]

            zl = z0
            zu = z0
            for bit in reversed(range(curbits)):
                Z[i] = zl - 2 ** bit
                if isvalid(Z, i):
                    zl = Z[i]
                Z[i] = zu + 2 ** bit
                if isvalid(Z, i):
                    zu = Z[i]
            # NOTE (original): Z[i] is not restored to z0 here; it keeps the
            # last probed value while the following coordinates are probed.

            z1 = (zl + zu) // 2
            grad.append(z1 - z0)
            diffs += abs(z1 - z0)

            continue

        convergence_factor = Fraction(10, 10 + it)
        grad = [g * convergence_factor for g in grad]

        factor = Fraction(1)

        assert isvalid(Z0)

        while True:
            for i in range(2 * n):
                Z[i] = Z0[i] + int(factor * grad[i])

            if isvalid(Z):
                break
            factor /= 2

        df = Fraction(10, 9)
        # NOTE (original): no iteration cap.  If Z0 + factor*grad stays valid
        # for every factor (e.g. grad == 0), this loop does not terminate.
        while True:
            factor *= df
            for i in range(2 * n):
                Z[i] = Z0[i] + int(factor * grad[i])

            if not isvalid(Z):
                factor /= df
                break

        if factor > 1:
            factor = 1

        while True:
            for i in range(2 * n):
                Z[i] = Z0[i] + int(factor * grad[i])

            if isvalid(Z):
                break
            else:
                factor /= df

        diffs = sum(abs(Z[i] - Z0[i]) for i in range(2 * n))
        diffs /= convergence_factor

        if DEBUG:
            print(" it", it, "diffs:", math.log(diffs + 1))

        if diffs < min_diff:
            min_diff = diffs
        else:
            stop_timer = 8
        stop_timer -= 1
        if diffs == 0 or stop_timer < 0:
            break

    X = Z[:n]
    Y = Z[n:]
    P = list(zip(X, Y))
    PS = PointSet(len(P), P, PS0.coloring).normalized()

    if PS.to_small_lambda().to_string() == SL.to_string():
        if DEBUG:
            print("beautifed!", PS.points)
        return PS
    else:
        if DEBUG:
            print("did not work!")
        return PS0


# ---------------------------------------------------------------------------
# beautify2  (PointSetOptimizer.py:440)
# ---------------------------------------------------------------------------

def beautify2(OT, iter1=1, iter2=1000, goodbits=0, minimize=False, diam_bound=None, norm=1,
              epsilon=0.0001):
    """Port of beautify2.  OT is a realized SmallLambda; returns a SmallLambda.

    Returns the new SmallLambda (with realization) if the order type is
    unchanged and (if diam_bound is set) diameter <= diam_bound, else OT.
    """
    import scipy
    import scipy.optimize
    points = OT.realization[:]
    n = OT.n

    BL = OT.to_big_lambda()
    assert len(BL.o) == n

    # NOTE (original): computed and then immediately replaced by exit_triples.
    orientations = dict([((a, b, c), BL.o[a][b][c]) for a in range(n) for b in range(n)
                         for c in range(n) if a < b and a < c and BL.o[a][b][c] > 0])

    edges, witnesses = filter_exit_edges(OT, return_witnesses=True)

    exit_triples = []
    for a, b in edges:
        for c in witnesses[(a, b)]:
            exit_triples.append((a, b, c))

    orientations = exit_triples

    def quality(Z):
        for i in CH0:
            Z[i], Z[i + n] = Z0[i], Z0[i + n]  # fix convex hull
        X = Z[:n]
        Y = Z[n:]
        if diam_bound and max(max(X) - min(X), max(Y) - min(Y)) > diam_bound:
            return float('inf')

        zmax = max([abs(z) for z in Z])

        o2 = lambda a, b: X[a] * Y[b] - Y[a] * X[b]
        o3 = lambda a, b, c: (o2(a, b) + o2(b, c) + o2(c, a))

        d2 = lambda a, b: (X[a] - X[b]) ** 2 + (Y[a] - Y[b]) ** 2
        d1 = lambda a, b: sqrt(d2(a, b))

        D2 = [d2(a, b) for (a, b) in combinations(range(n), 2)]
        F3 = [o3(a, b, c) * BL.o[a][b][c] / float(d1(a, b) + d1(b, c) + d1(c, a))
              for (a, b, c) in orientations]

        if min(F3) <= 0:
            return float('inf')

        error1 = sum(1. / v ** norm for v in F3) ** (1. / norm)
        error2 = (sum(v ** norm for v in D2) * sum(1. / v ** norm for v in D2)) ** (1. / norm)
        error3 = abs(log(zmax))

        error = error1 * error2 * error3
        return error

    PS0 = PS = PointSet(len(points), points).normalized()
    CH0 = OT.get_extremal_points()
    print("extremal", CH0)

    if minimize:
        PS = minimize_integer_coordinates(PS)

    for it in range(iter1):
        dbits = goodbits - PS.bits()
        if dbits > 0:
            print("scale point set!")
            PS.points = [(x * 2 ** dbits, y * 2 ** dbits) for (x, y) in PS.points]

        X0, Y0 = zip(*PS.points)
        Z0 = list(X0 + Y0)
        D0 = max([abs(z) for z in Z0])

        sol = scipy.optimize.fmin(quality, Z0, xtol=epsilon, ftol=epsilon, maxiter=iter2,
                                  maxfun=50 * iter2, disp=True)

        Z = sol
        for i in CH0:
            Z[i], Z[i + n] = Z0[i], Z0[i + n]  # fix convex hull
        Z = _to_integer(Z)
        X = Z[:n]
        Y = Z[n:]
        P = list(zip(X, Y))
        PS = PointSet(len(P), P)
        PS = PS.normalized()
        if minimize:
            PS = minimize_integer_coordinates(PS)
        print("PS", PS.points)

    SL = PS.to_small_lambda()
    if SL.to_string() == OT.to_string() and (not diam_bound or PS.diameter() <= diam_bound):
        return SL
    else:
        return OT  # backup


# ---------------------------------------------------------------------------
# MinimizePatternSearch  (scripts/pyotlib.MinimizePatternSearch.py)
# ---------------------------------------------------------------------------

def _ps_calc_determinante(points, i, j, k):
    ix, iy = points[i]
    jx, jy = points[j]
    kx, ky = points[k]
    return ix * (jy - ky) + jx * (ky - iy) + kx * (iy - jy)


def _ps_same_order_type(points1, points2, exit_triples=None) -> bool:
    n = len(points1)
    triples = exit_triples if exit_triples else combinations(range(n), 3)
    for a, b, c in triples:
        if _ps_calc_determinante(points1, a, b, c) * _ps_calc_determinante(points2, a, b, c) <= 0:
            return False
    return True


def _ps_normalized(points):
    xmin = min([x for (x, _) in points])
    ymin = min([y for (_, y) in points])
    return [(x - xmin, y - ymin) for (x, y) in points]


def _ps_get_diameter(points):
    """(max X - min X) + (max Y - min Y) -- note: sum, not max."""
    X = [x for (x, _) in points]
    Y = [y for (_, y) in points]
    return (max(X) - min(X)) + (max(Y) - min(Y))


def _ps_get_diameter2(points):
    """Sum over points of |n*x - sum X| + |n*y - sum Y| (L1 spread around centroid, times n)."""
    X = [x for (x, _) in points]
    Y = [y for (_, y) in points]
    xmid = sum(X)
    ymid = sum(Y)
    n = len(points)
    return sum((abs(n * x - xmid) + abs(n * y - ymid)) for (x, y) in points)


def pattern_search_optimize(points_old, maxtrials=2000, scale=1, blowup=0, tolerance=0,
                            DEBUG=0, points0=None):
    """Port of MinimizeCoordsScript.actionInner.optimize (MinimizePatternSearch.py:81).

    Returns (points_bst, blowup_left).  The original decremented the script's
    instance attribute ``self.blowup``, so the remaining blow-up budget carried
    over to the next order type of the input file; callers that want that
    behaviour pass ``blowup_left`` on.  ``points0`` (the unscaled input
    realization) is only used by the DEBUG consistency assert.
    """
    from copy import deepcopy

    M = scale
    points_old = [[M * x, M * y] for (x, y) in points_old]
    n = len(points_old)

    SL = PointSet(n, points_old).to_small_lambda()

    def _quality(points):
        if not _ps_same_order_type(points_old, points):
            return _ps_get_diameter2(points_old) * 10000
        else:
            return _ps_get_diameter2(points)

    f_bst = _quality(points_old)
    s = 1
    points_cur = points_old  # copy

    for it in range(maxtrials):
        points_new = deepcopy(points_cur)
        points_bst = deepcopy(points_cur)
        f_old = f_bst

        # search
        anything_improved = False
        for i in range(n):
            for j in [0, 1]:  # x or y coordinate
                improved = False
                for d in [+s, -s]:
                    points_new[i][j] = points_cur[i][j] + d
                    f_new = _quality(points_new)

                    if f_new < f_bst:
                        anything_improved = True
                        improved = True
                        f_bst = f_new
                        points_bst = deepcopy(points_new)

                if not improved:
                    points_new[i][j] = points_cur[i][j]  # backup

        if anything_improved:
            s += 1 + s // 2
            if tolerance > 0:
                if PointSet(n, points_new).bits() <= tolerance:
                    break
        else:
            s //= 4 if s > 6 else 2

        # extrapolate
        for i in range(n):
            for j in [0, 1]:  # x or y coordinate
                diff = points_cur[i][j] - points_old[i][j]
                points_new[i][j] = points_bst[i][j] + diff

        f_new = _quality(points_new)

        if f_new < f_bst:
            points_bst = deepcopy(points_new)
            f_bst = f_new

        diff2 = sum((points_bst[i][j] - points_cur[i][j]) ** 2 for i in range(n) for j in [0, 1])

        if diff2 != 0:
            points_old = deepcopy(points_cur)
            points_cur = deepcopy(points_bst)

        bits = log(_ps_get_diameter(points_cur)) / log(2)
        if DEBUG and it % 10 == 0:
            print("iter", it, "\t", float(f_old), "\t", bits, "\t", float(f_bst - f_old), "\t",
                  sqrt(diff2), s, "\t\t\t")
            assert PointSet(n, points0).to_small_lambda().to_string() == SL.to_string()

        if diff2 == 0 and s == 0:
            if blowup:
                if DEBUG:
                    print("BLOW UP!")
                blowup -= 1
                s = 1
                points_cur = points_old = [[2 * x, 2 * y] for (x, y) in points_cur]
                f_bst = _quality(points_old)
            else:
                if DEBUG:
                    print("STOP! NO CHANGES!")
                break

        if s == 0:
            s = 1
    return points_bst, blowup


def minimize_pattern_search(OT, maxtrials=2000, scale=1, blowup=0, tolerance=0, DEBUG=0):
    """Port of MinimizeCoordsScript.actionInner (MinimizePatternSearch.py:43), minus the writer.

    Returns (SmallLambda with the new realization, blowup_left).
    """
    from pyotlib2.core.small_lambda import SmallLambda

    points = points0 = OT.realization
    n = OT.n
    d0 = _ps_get_diameter(points0)
    bits0 = PointSet(n, points0).bits()

    if not tolerance or bits0 > tolerance:
        SL0 = PointSet(n, points0).to_small_lambda()
        if DEBUG:
            print("\tquality:", d0)

        # optimization here
        points, blowup = pattern_search_optimize(points0, maxtrials=maxtrials, scale=scale,
                                                 blowup=blowup, tolerance=tolerance, DEBUG=DEBUG,
                                                 points0=points0)
        points = _ps_normalized(points)
        SL = PointSet(n, points).to_small_lambda()
        bits = PointSet(n, points).bits()

        assert SL.to_string() == SL0.to_string()

        if DEBUG:
            print("\tquality:", _ps_get_diameter(points), "( started with", d0, ")", "\tbits:",
                  bits0, "->", bits)
        if DEBUG:
            print()

    return SmallLambda(n, OT.l, realization=points, coloring=OT.coloring), blowup

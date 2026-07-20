"""Realizability via a *projective-class* wrapper.

Port of the pyotlib1 ``PCRealizationTester``.

An abstract order type ``ot`` sits inside a *projective class*: the set of all
order types reachable from it by reflecting extremal / convex-hull points
("flips").  Every member of that class is projectively equivalent, but some are
geometrically far less contorted and therefore much easier for a numeric solver
to realize.  This wrapper exploits that:

    1. enumerate the projective class of ``ot``;
    2. hand a (possibly randomly chosen) CLASS MEMBER to an inner solver;
    3. transform the resulting concrete point set BACK so that it induces the
       ORIGINAL ``ot`` -- element for element in the GIVEN point order.

Step 3 is the crux.  We do *not* relabel the final answer relative to ``ot``:
the returned ``ot.realization`` must reproduce ``ot``'s chirotope exactly
(``PointSet(realization).to_big_lambda().o`` element-for-element equal to
``ot.to_big_lambda().o``), which the base-class ``_verify`` enforces.

Back-transform strategy
-----------------------
:class:`~pyotlib2.algorithms.projective_class.ProjectiveClass` records, for each
member ``m``, a tuple ``(flip_vec F, labeling lab, mirror)`` such that::

    m == start.from_flips(F).relabeled(lab)[.mirrored() if mirror]   # start = ot.get_lex_min()

The ``mirror`` bit is essential: each BFS ``get_lex_min`` step may mirror to
reach canonical form, so its cumulative parity is part of the member↔start
relation.  Omitting it leaves the reconstruction off by a global orientation
reversal for members reached across an odd number of mirroring flips.

Every primitive (``from_flips``, ``relabeled``, ``mirrored``) transforms an
*attached realization* consistently with the chirotope -- ``from_flips`` applies
the correct *projective* transform for a convex-hull flip (a sign flip of the
homogeneous vectors, re-charted to finite affine points; see
:func:`pyotlib2.core.point_rotation.rotate_by_flip`) and ``relabeled`` /
``mirrored`` permute / swap coordinates.  So once a solver hands back a realized
member, we replay the inverse chain on the realization-carrying representation
and land on ``ot``'s own coordinates.

Because manual sign / labeling bookkeeping is error-prone, every reconstructed
candidate is GUARDED: it is accepted only when its BigLambda ``.o`` array-equals
``ot.to_big_lambda().o``.  That gate makes the result provably correct
regardless of any slip in the inverse chain; a wrong reconstruction is simply
discarded and the next member tried.
"""

from __future__ import annotations
import random
from typing import Optional

import numpy as np

from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.core.point_set import PointSet
from pyotlib2.realization.base import RealizationTester, Undecided


class PCRealizationTester(RealizationTester):
    """Projective-class wrapper around an inner realization tester.

    Realizes an easier member of ``ot``'s projective class with ``inner`` and
    transports that realization back so it induces ``ot`` in the given point
    order.
    """

    name = "pc"

    def __init__(
        self,
        inner: RealizationTester,
        parent: Optional[RealizationTester] = None,
        max_trials: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(parent)
        self.inner = inner
        self.max_trials = max_trials
        self.shuffle = shuffle
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def _test(self, ot: SmallLambda) -> Optional[bool]:
        from pyotlib2.algorithms.projective_class import ProjectiveClass

        o_target = ot.to_big_lambda().o

        # Enumerate the projective class of ot.  The recorded (flip_vec, lab)
        # relate each member to the lex-min "start" of ot's class, and
        # get_lex_min relates that "start" back to ot itself.
        pc = ProjectiveClass(ot, calc_lex_min=False)
        start, lab0, mir0 = ot.get_lex_min(return_labeling=True)
        inv_lab0 = _invert(lab0)

        members = list(pc._ot_dict.values())
        if self.shuffle:
            self._rng.shuffle(members)

        trials = 0
        for member, flip_vec, labeling, member_mir in members:
            if self.max_trials and trials >= self.max_trials:
                break
            trials += 1

            # Realize the class member abstractly (fresh, no carried-over
            # realization) with the inner solver.
            m_abs = SmallLambda.from_string(member.n, member.to_string())
            try:
                if self.inner.is_realizable(m_abs) is not True:
                    continue
            except Undecided:
                continue
            if m_abs.realization is None:
                continue

            realization = self._back_transform(
                m_abs, flip_vec, labeling, member_mir, mir0, inv_lab0
            )
            if realization is None:
                continue

            # Guard: only accept coordinates that induce ot exactly
            # (element-for-element chirotope, no relabeling).
            ps = PointSet(ot.n, realization)
            if ps.has_collinear_points():
                continue
            if np.array_equal(ps.to_small_lambda(lazy=False).to_big_lambda().o, o_target):
                ot.realization = list(realization)
                return True

        raise Undecided("no projective-class member yielded a realization of ot")

    # ------------------------------------------------------------------
    # Pick a random projective-class member + a back-transform for it
    # ------------------------------------------------------------------

    def pick_member(self, ot: SmallLambda):
        """Choose a random member of ``ot``'s projective class.

        Returns ``(member_abstract, back_transform)`` where ``member_abstract``
        is a fresh abstract SmallLambda of a projectively-equivalent order type
        and ``back_transform(realization) -> realization_of_ot`` maps a
        realization of the member back onto ``ot`` exactly (or None if it does
        not induce ``ot``).  Use this to realize an *easier* member and lift the
        result back to the original order type.
        """
        from pyotlib2.algorithms.projective_class import ProjectiveClass

        o_target = ot.to_big_lambda().o
        pc = ProjectiveClass(ot, calc_lex_min=False)
        _, lab0, mir0 = ot.get_lex_min(return_labeling=True)
        inv_lab0 = _invert(lab0)

        members = list(pc._ot_dict.values())
        if self.shuffle:
            self._rng.shuffle(members)
        member, flip_vec, labeling, member_mir = members[0]
        m_abs = SmallLambda.from_string(member.n, member.to_string())

        def back(realization):
            m_real = SmallLambda(m_abs.n, None, realization=realization)
            r = self._back_transform(
                m_real, flip_vec, labeling, member_mir, mir0, inv_lab0)
            if r is None:
                return None
            ps = PointSet(ot.n, r)
            if ps.has_collinear_points():
                return None
            if not np.array_equal(
                    ps.to_small_lambda(lazy=False).to_big_lambda().o, o_target):
                return None
            return r

        return m_abs, back

    # ------------------------------------------------------------------
    # Back-transform
    # ------------------------------------------------------------------

    @staticmethod
    def _back_transform(
        m_abs: SmallLambda,
        flip_vec: list,
        labeling: list,
        member_mir: bool,
        mir0: bool,
        inv_lab0: list,
    ) -> Optional[list]:
        """Transport a realized member back to the original ot's coordinates.

        Inverts the two composition steps recorded by ProjectiveClass /
        get_lex_min, carrying the realization along every primitive.  Note the
        trailing ``.mirrored()`` (``member_mir``) recorded by ProjectiveClass:
        each BFS ``get_lex_min`` may mirror to reach canonical form, and that
        cumulative parity is part of the member↔start relation.  Undo the LAST
        applied op first (mirror, then relabel, then flip):

            m == start.from_flips(F).relabeled(lab)[.mirrored() if member_mir]
                =>  start = m[.mirrored()].relabeled(inv(lab)).from_flips(F)
                    (mirror and flip are each involutions)
            start == ot.relabeled(lab0)[.mirrored() if mir0]
                =>  ot = start[.mirrored()].relabeled(inv(lab0))
        """
        # Realization-carrying representation of the realized member.
        m_real = SmallLambda(m_abs.n, None, realization=m_abs.realization)

        cur = m_real.mirrored() if member_mir else m_real
        start_real = cur.relabeled(_invert(labeling)).from_flips(flip_vec)
        cur = start_real.mirrored() if mir0 else start_real
        ot_real = cur.relabeled(inv_lab0)

        realization = ot_real.realization
        if realization is None or len(realization) != m_abs.n:
            return None
        return realization


def _invert(perm: list) -> list:
    """Return the inverse of a permutation given as a list."""
    inv = [0] * len(perm)
    for i, v in enumerate(perm):
        inv[v] = i
    return inv

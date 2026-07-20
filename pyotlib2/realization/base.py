"""Abstract base class for realization testers.

Port of the pyotlib1 ``RealizationTester`` chain-of-responsibility contract.

A tester answers one question about an abstract order type ``ot`` (a
:class:`~pyotlib2.core.small_lambda.SmallLambda`):

    is it realizable by a concrete point set?

Three outcomes:

* ``True``  – realizable.  The tester has set ``ot.realization`` to concrete
  coordinates and that realization has been re-verified to induce ``ot``.
* ``False`` – *proven* non-realizable (only the Grassmann-Plücker tester ever
  returns this).
* ``None``  – "don't know": this tester gave up.  The base class then delegates
  to ``self.parent`` (the fallback tester), or returns ``None`` if there is none.

Subclasses implement :meth:`_test`, returning ``True``/``False`` or raising
:class:`Undecided` (equivalently returning ``None``) to defer to the parent.

Warm starting: a tester that supports it reads ``ot.subrealization`` — a point
set realizing the sub-order-type on the first ``n-1`` points — and uses it as an
initial guess so only the newly added point still has to be placed.  This is the
key to the etherealization algorithm scaling past ~10 points.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from pyotlib2.core.small_lambda import SmallLambda


#: Every abstract order type on at most this many points is realizable.  (In
#: rank 3 the smallest non-realizable order type is the non-Pappus / non-Desargues
#: configuration on 9 points, so all n <= 8 order types are realizable.)  Used as
#: a sanity guard: a non-realizability verdict for n <= MAX_REALIZABLE_N means the
#: model or solver is broken, and a numerical realizer that fails on an
#: n <= MAX_REALIZABLE_N (sub-)order type is likewise suspect.
MAX_REALIZABLE_N = 8


class Undecided(Exception):
    """Raised when a tester cannot determine realizability."""
    pass


class RealizationTester(ABC):
    """Base class for realizability testers.

    Chain-of-responsibility: if a tester cannot decide, it delegates to a
    parent tester.  Build a chain inner-most first::

        tester = Z3RealizationTester(
            parent=None)
        tester = ScipyRealizationTester(parent=tester)   # tries scipy first,
                                                          # falls back to z3
    """

    #: human-readable name used in benchmark output / logging
    name: str = "realization-tester"

    def __init__(self, parent: Optional["RealizationTester"] = None):
        self.parent = parent

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def is_realizable(self, ot: SmallLambda) -> Optional[bool]:
        """Return True if realizable, False if proven non-realizable, None if unknown.

        If ``ot`` already carries a realization it is verified and ``True``
        returned without invoking any solver.
        """
        if ot.realization is not None:
            if self._verify(ot):
                return True
            # a bogus pre-set realization: drop it and actually test
            ot.realization = None

        try:
            result = self._test(ot)
        except Undecided:
            result = None

        if result is None:
            if self.parent is not None:
                return self.parent.is_realizable(ot)
            return None

        if result is True:
            # subclasses set ot.realization; double-check it really fits
            if ot.realization is not None and not self._verify(ot):
                ot.realization = None
                if self.parent is not None:
                    return self.parent.is_realizable(ot)
                return None
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _verify(ot: SmallLambda) -> bool:
        """Check that ``ot.realization`` really induces the order type ``ot``.

        The check is done *without any relabeling*: the realization must
        reproduce ``ot`` in exactly the given point order, i.e. its chirotope
        (BigLambda ``o`` array) must be element-for-element identical to
        ``ot``'s.  This is stronger than lex-min / up-to-isomorphism equality
        and is the invariant we require of every tester: ``realization ->
        to_big_lambda()`` must equal the original ``to_big_lambda()``.
        """
        import numpy as np
        from pyotlib2.core.point_set import PointSet
        if ot.realization is None or len(ot.realization) != ot.n:
            return False
        try:
            ps = PointSet(ot.n, ot.realization)
            if ps.has_collinear_points():
                return False
            o_real = ps.to_small_lambda(lazy=False).to_big_lambda().o
            return bool(np.array_equal(o_real, ot.to_big_lambda().o))
        except Exception:
            return False

    @abstractmethod
    def _test(self, ot: SmallLambda) -> Optional[bool]:
        """Subclass-specific test.

        Return ``True`` (and set ``ot.realization``), ``False`` (proven
        non-realizable), or ``None`` / raise :class:`Undecided` to defer to the
        parent tester.
        """
        ...

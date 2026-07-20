"""Realizability testers for abstract order types.

A *realization tester* decides whether an abstract order type can be drawn as a
concrete point set, and if so produces integer coordinates.  Testers share the
:class:`~pyotlib2.realization.base.RealizationTester` chain-of-responsibility
contract, so they can be composed into fallback chains.

Available solvers (selectable by name via :func:`make_tester`):

    ``scipy``   nonlinear optimization (hinge penalty) + global search
    ``octave``  external Octave minimiser (needs the ``octave`` binary); the
                per-method variants ``octave.fminsearch`` / ``octave.fminunc`` /
                ``octave.sqp`` select the Octave solver
    ``sage``    external SageMath ``minimize`` (needs the ``sage`` binary); the
                per-method variants ``sage.simplex`` / ``sage.powell``
    ``mathematica``  external Mathematica ``NMinimize`` (needs ``wolframscript``);
                arbitrary-precision realization
    ``mpmath``  in-process arbitrary-precision Nelder-Mead (needs ``mpmath``);
                set the mantissa bits via ``precision`` (``--precision``)
    ``z3``      SMT over the reals; can also *prove* non-realizability
    ``grid``    randomized integer-grid backtracking (good only up to ~10 points)
    ``gp``      Grassmann-Plücker LP via GLPK; proves non-realizability only

Use :func:`make_chain` to build a fallback chain from a list of names, e.g.
``make_chain(["scipy", "z3"])`` tries scipy first and falls back to z3.
"""

from __future__ import annotations
from typing import Callable, Optional

from pyotlib2.realization.base import RealizationTester, Undecided


def _make_scipy(parent=None, **kw):
    from pyotlib2.realization.scipy_tester import ScipyRealizationTester
    return ScipyRealizationTester(parent=parent, **kw)


def _make_scipy_method(method):
    def factory(parent=None, **kw):
        from pyotlib2.realization.scipy_tester import ScipyMethodTester
        return ScipyMethodTester(method, parent=parent, **kw)
    return factory


def _make_octave(parent=None, **kw):
    from pyotlib2.realization.octave_tester import OctaveRealizationTester
    return OctaveRealizationTester(parent=parent, **kw)


def _make_octave_method(method):
    def factory(parent=None, **kw):
        from pyotlib2.realization.octave_tester import OctaveRealizationTester
        return OctaveRealizationTester(parent=parent, method=method, **kw)
    return factory


def _make_z3(parent=None, **kw):
    from pyotlib2.realization.z3_tester import Z3RealizationTester
    return Z3RealizationTester(parent=parent, **kw)


def _make_grid(parent=None, **kw):
    from pyotlib2.realization.grid_search import GridSearchTester
    return GridSearchTester(parent=parent, **kw)


def _make_gp(parent=None, **kw):
    from pyotlib2.realization.gp_tester import GPRealizationTester
    return GPRealizationTester(parent=parent, **kw)


def _make_sage(parent=None, **kw):
    from pyotlib2.realization.sage_tester import SageRealizationTester
    return SageRealizationTester(parent=parent, **kw)


def _make_mathematica(parent=None, **kw):
    from pyotlib2.realization.mathematica_tester import MathematicaRealizationTester
    return MathematicaRealizationTester(parent=parent, **kw)


def _make_mpmath(parent=None, **kw):
    from pyotlib2.realization.mpmath_tester import MpmathRealizationTester
    return MpmathRealizationTester(parent=parent, **kw)


def _make_sage_method(method):
    def factory(parent=None, **kw):
        from pyotlib2.realization.sage_tester import SageRealizationTester
        return SageRealizationTester(parent=parent, method=method, **kw)
    return factory


#: registry of solver name -> factory(parent, **kwargs)
SOLVERS: dict[str, Callable[..., RealizationTester]] = {
    "scipy": _make_scipy,
    "octave": _make_octave,
    "z3": _make_z3,
    "grid": _make_grid,
    "gp": _make_gp,
    "sage": _make_sage,
    "mathematica": _make_mathematica,
    "mpmath": _make_mpmath,
}

# Per-method scipy variants, registered as ``scipy.<method>`` so each scipy
# optimizer can be selected and benchmarked individually.
try:
    from pyotlib2.realization.scipy_tester import scipy_methods as _scipy_methods
    for _m in _scipy_methods():
        SOLVERS[f"scipy.{_m.replace('-', '')}"] = _make_scipy_method(_m)
except Exception:  # scipy not installed
    pass


def _make_nlopt_method(method):
    def factory(parent=None, **kw):
        from pyotlib2.realization.nlopt_tester import NloptMethodTester
        return NloptMethodTester(method, parent=parent, **kw)
    return factory


# Per-algorithm NLopt variants, registered as ``nlopt.<ALGO>``.
try:
    from pyotlib2.realization.nlopt_tester import nlopt_methods as _nlopt_methods
    for _m in _nlopt_methods():
        SOLVERS[f"nlopt.{_m}"] = _make_nlopt_method(_m)
except Exception:  # nlopt not installed
    pass

# Per-minimiser Octave variants, registered as ``octave.<method>`` (fminsearch /
# fminunc / sqp).  Registration is unconditional (the binary is only required
# when a tester is actually instantiated), mirroring plain ``octave``.
try:
    from pyotlib2.realization.octave_tester import octave_methods as _octave_methods
    for _m in _octave_methods():
        SOLVERS[f"octave.{_m}"] = _make_octave_method(_m)
except Exception:  # pragma: no cover
    pass

# Per-algorithm SageMath variants, registered as ``sage.<method>`` (simplex /
# powell).  Registration is unconditional (the ``sage`` binary is only required
# when a tester is actually instantiated), mirroring plain ``sage``.
try:
    from pyotlib2.realization.sage_tester import sage_methods as _sage_methods
    for _m in _sage_methods():
        SOLVERS[f"sage.{_m}"] = _make_sage_method(_m)
except Exception:  # pragma: no cover
    pass

#: names that require an external binary / optional package to be installed
SOLVER_REQUIRES = {
    "octave": "octave binary",
    "gp": "glpsol (GLPK)",
    "z3": "z3-solver package",
    "nlopt": "nlopt package",
    "sage": "sage binary (SageMath)",
    "mathematica": "wolframscript (Mathematica)",
    "mpmath": "mpmath package",
}


def available_solvers() -> list[str]:
    """Return the primary solver-family names in preferred display order.

    Per-method variants (``scipy.*``, ``nlopt.*``, ``octave.*``, ``sage.*``) are
    additionally registered in :data:`SOLVERS`; these are the compact families.
    """
    families = ["scipy", "nlopt", "z3", "octave", "sage", "mathematica", "mpmath",
                "grid", "gp"]
    # a family is available if it (or any of its per-method variants) is registered
    return [f for f in families
            if f in SOLVERS or any(k.startswith(f + ".") for k in SOLVERS)]


def make_tester(name: str, parent: Optional[RealizationTester] = None, **kwargs) -> RealizationTester:
    """Instantiate a single tester by name (see :data:`SOLVERS`)."""
    try:
        factory = SOLVERS[name]
    except KeyError:
        raise ValueError(
            f"unknown solver {name!r}; choose from {sorted(SOLVERS)}"
        )
    return factory(parent=parent, **kwargs)


def make_chain(names: list[str], **kwargs) -> RealizationTester:
    """Build a fallback chain: ``names[0]`` is tried first, then ``names[1]`` …

    Extra keyword arguments are passed to *every* tester factory that accepts
    them; unknown-argument errors are the caller's responsibility.
    """
    if not names:
        raise ValueError("make_chain needs at least one solver name")
    # build inner-most (last) first so the parent chain points the right way
    tester = None
    for name in reversed(names):
        tester = make_tester(name, parent=tester, **kwargs)
    return tester


__all__ = [
    "RealizationTester",
    "Undecided",
    "SOLVERS",
    "available_solvers",
    "make_tester",
    "make_chain",
]

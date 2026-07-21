"""Tests for the Mathematica realization tester.

The ``wolframscript`` binary is absent on most machines (including CI), so the
solving tests skip cleanly.  Importing the module and generating a script must
work regardless, so those are always exercised.
"""

import shutil

import pytest

from pyotlib2.realization.mathematica_tester import MathematicaRealizationTester
from pyotlib2.realization.qcp import RealizationQCP
from tests.point_configs import one_interior, abstract_only

_HAS_WOLFRAM = shutil.which("wolframscript") is not None


def test_module_imports():
    # The class must be importable even without wolframscript.
    assert MathematicaRealizationTester.name == "mathematica"


def test_write_script_cold_start(tmp_path):
    # write_script is a pure formatter; run it WITHOUT instantiating the tester
    # (which would require the binary) via __new__ + explicit attributes.
    exporter = MathematicaRealizationTester.__new__(MathematicaRealizationTester)
    exporter.precision = 0
    exporter.logsum = False
    exporter.method = "NelderMead"

    ot = abstract_only(one_interior(5))
    qcp = RealizationQCP(ot)
    script = tmp_path / "qcp.m"
    exporter.write_script(str(script), str(tmp_path / "qcp.out"), qcp, z0=None)

    text = script.read_text()
    assert "NMinimize[" in text
    assert "q[t_] := Abs[t] - t" in text
    # default (non-logsum) uses the squared hinge q[...]^2
    assert "]^2" in text
    assert "Export[" in text
    # the found/not-found signal: fmin (the minimised penalty) is captured,
    # written to a sibling .objective file, and a REALIZED/NOT REALIZED verdict
    # plus the coordinates are printed to stdout.
    assert "fmin = First[result];" in text
    assert ".objective" in text
    assert "REALIZED" in text
    assert "NOT REALIZED" in text
    assert 'Print["coords = ", sol];' in text


def test_write_script_precision_and_logsum(tmp_path):
    import math
    exporter = MathematicaRealizationTester.__new__(MathematicaRealizationTester)
    exporter.precision = 100  # BITS
    exporter.logsum = True
    exporter.method = "NelderMead"

    ot = abstract_only(one_interior(5))
    qcp = RealizationQCP(ot, logsum=True)
    script = tmp_path / "qcp.m"
    exporter.write_script(str(script), str(tmp_path / "qcp.out"), qcp, z0=None)

    text = script.read_text()
    # precision is in BITS; Mathematica WorkingPrecision counts DECIMAL digits.
    digits = math.ceil(100 / math.log2(10))  # = 31
    assert f"WorkingPrecision -> {digits}" in text
    assert f"AccuracyGoal -> {digits}" in text
    # the REALIZED threshold is derived from the digits (half), not hardcoded
    assert f"eps = 10^-{digits // 2};" in text
    # q is always the plain hinge; logsum wraps each term in Log[1 + q[...]]
    assert "q[t_] := Abs[t] - t" in text
    assert "Log[1 + q[" in text


def test_verdict_threshold_scales_with_precision(tmp_path):
    """The REALIZED threshold is derived from --precision, never hardcoded."""
    import math
    ot = abstract_only(one_interior(5))
    qcp = RealizationQCP(ot)

    # high precision -> tight threshold
    hi = MathematicaRealizationTester.__new__(MathematicaRealizationTester)
    hi.precision, hi.logsum, hi.method = 200, False, "NelderMead"
    s_hi = tmp_path / "hi.m"
    hi.write_script(str(s_hi), str(tmp_path / "hi.out"), qcp, z0=None)
    d_hi = math.ceil(200 / math.log2(10))  # 61
    assert f"eps = 10^-{d_hi // 2};" in s_hi.read_text()  # 10^-30
    assert "10^-6" not in s_hi.read_text()  # the old hardcoded value is gone

    # no precision (MachinePrecision) -> fallback threshold 10^-7, no WorkingPrecision
    lo = MathematicaRealizationTester.__new__(MathematicaRealizationTester)
    lo.precision, lo.logsum, lo.method = 0, False, "NelderMead"
    s_lo = tmp_path / "lo.m"
    lo.write_script(str(s_lo), str(tmp_path / "lo.out"), qcp, z0=None)
    text_lo = s_lo.read_text()
    assert "eps = 10^-7;" in text_lo
    assert "WorkingPrecision" not in text_lo


def test_parse_scientific_notation():
    # Mathematica's a*^b scientific form must parse.
    assert MathematicaRealizationTester._to_float("1.5*^3") == pytest.approx(1500.0)
    assert MathematicaRealizationTester._to_float("2.0`20") == pytest.approx(2.0)
    assert MathematicaRealizationTester._to_float("-3.25") == pytest.approx(-3.25)


def test_init_requires_binary():
    if _HAS_WOLFRAM:
        pytest.skip("wolframscript present; cannot test the missing-binary path")
    with pytest.raises(RuntimeError):
        MathematicaRealizationTester()


@pytest.mark.skipif(not _HAS_WOLFRAM, reason="wolframscript not installed")
def test_realizes_small_ot():
    ot = abstract_only(one_interior(6))
    assert MathematicaRealizationTester().is_realizable(ot) is True

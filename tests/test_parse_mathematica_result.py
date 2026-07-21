"""Tests for scripts/parse_mathematica_result.py (Mathematica output parser)."""

import importlib.util
import math
import os

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts",
                       "parse_mathematica_result.py")
_spec = importlib.util.spec_from_file_location("parse_mm", _SCRIPT)
parse_mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parse_mm)


def test_threshold_matches_exported_script():
    """verdict_threshold mirrors the .wls: 10^-(digits/2), digits=ceil(bits/log2 10)."""
    for bits in (100, 200, 400):
        digits = math.ceil(bits / math.log2(10))
        assert parse_mm.verdict_threshold(bits) == 10.0 ** (-(digits // 2))
    # MachinePrecision fallback
    assert parse_mm.verdict_threshold(0) == 10.0 ** -7
    assert parse_mm.verdict_threshold(-1) == 10.0 ** -7


def _write(tmp_path, coords_text, objective_text):
    out = tmp_path / "qcp.out"
    out.write_text(coords_text)
    (tmp_path / "qcp.out.objective").write_text(objective_text)
    return str(out)


def test_realized_case(tmp_path, capsys):
    out = _write(tmp_path, "0 0\n4 1\n2 5\n", "1.2e-45\n")
    rc = parse_mm.main([out, "--precision", "200"])
    assert rc == 0
    import json
    d = json.loads(capsys.readouterr().out)
    assert d["realized"] is True
    assert d["num_points"] == 3
    assert d["threshold"] == 10.0 ** -30  # 200 bits -> 61 digits -> 10^-30


def test_not_realized_case(tmp_path, capsys):
    out = _write(tmp_path, "0 0\n1 0\n0 1\n", "3.4\n")
    rc = parse_mm.main([out, "--precision", "200"])
    assert rc == 1
    import json
    d = json.loads(capsys.readouterr().out)
    assert d["realized"] is False
    assert d["objective"] == 3.4


def test_missing_objective_is_error(tmp_path, capsys):
    out = tmp_path / "qcp.out"
    out.write_text("0 0\n")  # no .objective sibling
    rc = parse_mm.main([str(out)])
    assert rc == 2
    assert "error" in capsys.readouterr().out

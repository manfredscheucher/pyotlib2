"""Unit tests for I/O readers and writers (all formats)."""

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyotlib2.core.point_set import PointSet
from pyotlib2.core.small_lambda import SmallLambda
from pyotlib2.io.readers import read_order_types
from pyotlib2.io.writers import write_order_types
from pyotlib2.io.readers import _pack_bits, _unpack_bits
from pyotlib2.algorithms.unify import unify


SAMPLE_POINTS_4  = [(0, 0), (3, 0), (3, 3), (0, 3)]
SAMPLE_POINTS_4B = [(0, 0), (4, 0), (2, 3), (0, 3)]
SAMPLE_POINTS_5  = [(0, 0), (4, 0), (4, 4), (0, 4), (1, 2)]


def make_sl(pts):
    return PointSet(len(pts), pts).to_small_lambda(lazy=False)


def to_natural(ot):
    p0 = ot.get_extremal_points()[0]
    return ot.relabeled(ot.get_natural_labeling(p0))


def unique_ots(n, fname):
    path = f"tests/otdb/otypes/{fname}"
    if not Path(path).exists():
        pytest.skip(f"{path} not found — run tests/otdb/download.py first")
    return list(unify(list(read_order_types(path, n=n))))


def roundtrip(ots, fmt, *, n=None, extra_read_kwargs=None):
    """Write ots to a temp file in fmt, read back, return list of SmallLambda."""
    suffix = f".{fmt}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        path = Path(f.name)
    try:
        write_order_types(ots, path, fmt=fmt)
        kw = {"fmt": fmt}
        if n is not None:
            kw["n"] = n
        if extra_read_kwargs:
            kw.update(extra_read_kwargs)
        return list(read_order_types(path, **kw))
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SmallLambda text  (.slt)
# ---------------------------------------------------------------------------

class TestSmallLambdaText:
    def test_roundtrip(self):
        sl = make_sl(SAMPLE_POINTS_4)
        [sl2] = roundtrip([sl], "slt")
        assert sl.compare(sl2) == 0

    def test_multiple(self):
        ots = [make_sl(SAMPLE_POINTS_4), make_sl(SAMPLE_POINTS_4B)]
        loaded = roundtrip(ots, "slt")
        assert len(loaded) == 2

    def test_txt_alias(self):
        sl = make_sl(SAMPLE_POINTS_4)
        [sl2] = roundtrip([sl], "txt")   # legacy alias
        assert sl.compare(sl2) == 0


# ---------------------------------------------------------------------------
# SmallLambda binary  (slb08, slb04, slb16)
# ---------------------------------------------------------------------------

class TestSmallLambdaBinary:
    def _check(self, fmt, ots):
        n = ots[0].n
        loaded = roundtrip(ots, fmt, n=n)
        assert len(loaded) == len(ots)
        for a, b in zip(ots, loaded):
            assert np.array_equal(a.get_l(), b.get_l()), f"{fmt} roundtrip failed"

    def test_slb08_n6(self):
        ots = unique_ots(6, "otypes06.b08")
        self._check("slb08", ots)

    def test_slb04_n6(self):
        ots = unique_ots(6, "otypes06.b08")
        self._check("slb04", ots)

    def test_slb16_n6(self):
        ots = unique_ots(6, "otypes06.b08")
        self._check("slb16", ots)

    def test_slb08_n7(self):
        ots = unique_ots(7, "otypes07.b08")
        self._check("slb08", ots)


# ---------------------------------------------------------------------------
# SmallLambda compressed binary  (slbc08, slbc04)
# ---------------------------------------------------------------------------

class TestSmallLambdaCompressedBinary:
    def _check(self, fmt, ots):
        nat = [to_natural(ot) for ot in ots]
        n = nat[0].n
        loaded = roundtrip(nat, fmt, n=n)
        assert len(loaded) == len(nat)
        for a, b in zip(nat, loaded):
            assert np.array_equal(a.get_l(), b.get_l()), f"{fmt} roundtrip failed"

    def test_slbc08_n6(self):
        self._check("slbc08", unique_ots(6, "otypes06.b08"))

    def test_slbc04_n6(self):
        self._check("slbc04", unique_ots(6, "otypes06.b08"))

    def test_slbc08_n7(self):
        self._check("slbc08", unique_ots(7, "otypes07.b08"))

    def test_slbc04_n7(self):
        self._check("slbc04", unique_ots(7, "otypes07.b08"))

    def test_non_natural_raises(self):
        sl = make_sl(SAMPLE_POINTS_5)
        # make sure it's NOT in natural labeling artificially
        with pytest.raises((AssertionError, ValueError)):
            roundtrip([sl], "slbc08", n=sl.n)


# ---------------------------------------------------------------------------
# BigLambda text  (.blt)
# ---------------------------------------------------------------------------

class TestBigLambdaText:
    def test_roundtrip(self):
        sl = make_sl(SAMPLE_POINTS_4)
        [sl2] = roundtrip([sl], "blt", n=sl.n)
        assert (sl.to_big_lambda().to_string()
                == sl2.to_big_lambda().to_string())

    def test_n6(self):
        ots = unique_ots(6, "otypes06.b08")
        loaded = roundtrip(ots, "blt", n=6)
        assert len(loaded) == len(ots)


# ---------------------------------------------------------------------------
# BigLambda binary  (.blb)
# ---------------------------------------------------------------------------

class TestBigLambdaBinary:
    def test_blb_n6(self):
        ots = unique_ots(6, "otypes06.b08")
        loaded = roundtrip(ots, "blb", n=6)
        assert len(loaded) == len(ots)
        for a, b in zip(ots, loaded):
            assert (a.to_big_lambda().to_string()
                    == b.to_big_lambda().to_string())

    def test_blb_n7(self):
        ots = unique_ots(7, "otypes07.b08")
        loaded = roundtrip(ots, "blb", n=7)
        assert len(loaded) == len(ots)


# ---------------------------------------------------------------------------
# BigLambda compressed binary  (.blbc)
# ---------------------------------------------------------------------------

class TestBigLambdaCompressedBinary:
    def test_blbc_n6(self):
        ots = [to_natural(ot) for ot in unique_ots(6, "otypes06.b08")]
        loaded = roundtrip(ots, "blbc", n=6)
        assert len(loaded) == len(ots)
        for a, b in zip(ots, loaded):
            assert (a.to_big_lambda().to_string()
                    == b.to_big_lambda().to_string())

    def test_blbc_n7(self):
        ots = [to_natural(ot) for ot in unique_ots(7, "otypes07.b08")]
        loaded = roundtrip(ots, "blbc", n=7)
        assert len(loaded) == len(ots)


# ---------------------------------------------------------------------------
# FelsnerMatrix text / binary  (.fmt / .fmb)
# ---------------------------------------------------------------------------

class TestFelsnerIO:
    def test_fmt_n6(self):
        ots = [to_natural(ot) for ot in unique_ots(6, "otypes06.b08")]
        loaded = roundtrip(ots, "fmt", n=6)
        assert len(loaded) == len(ots)
        for a, b in zip(ots, loaded):
            assert (a.to_big_lambda().to_string()
                    == b.to_big_lambda().to_string())

    def test_fmb_n6(self):
        ots = [to_natural(ot) for ot in unique_ots(6, "otypes06.b08")]
        loaded = roundtrip(ots, "fmb", n=6)
        assert len(loaded) == len(ots)
        for a, b in zip(ots, loaded):
            assert (a.to_big_lambda().to_string()
                    == b.to_big_lambda().to_string())

    def test_fmt_n7(self):
        ots = [to_natural(ot) for ot in unique_ots(7, "otypes07.b08")]
        loaded = roundtrip(ots, "fmt", n=7)
        assert len(loaded) == len(ots)

    def test_fmb_n7(self):
        ots = [to_natural(ot) for ot in unique_ots(7, "otypes07.b08")]
        loaded = roundtrip(ots, "fmb", n=7)
        assert len(loaded) == len(ots)


# ---------------------------------------------------------------------------
# Point binary  (pb08, pb16, pb04)
# ---------------------------------------------------------------------------

class TestPointBinary:
    def _make_pts_sl(self, pts):
        ps = PointSet(len(pts), pts)
        sl = ps.to_small_lambda(lazy=False)
        sl.realization = list(pts)
        return sl

    def test_pb08_roundtrip(self):
        sl = self._make_pts_sl(SAMPLE_POINTS_4)
        [sl2] = roundtrip([sl], "pb08", n=4)
        assert sl2.n == 4

    def test_pb16_roundtrip(self):
        sl = self._make_pts_sl(SAMPLE_POINTS_4)
        [sl2] = roundtrip([sl], "pb16", n=4)
        assert sl2.n == 4

    def test_pb04_roundtrip(self):
        pts = [(0, 0), (3, 0), (3, 3), (0, 3)]   # all coords < 16
        sl = self._make_pts_sl(pts)
        [sl2] = roundtrip([sl], "pb04", n=4)
        assert sl2.n == 4

    def test_b08_legacy_alias(self):
        """b08 is a legacy alias for pb08 (OTDB files)."""
        path = "tests/otdb/otypes/otypes04.b08"
        if not Path(path).exists():
            pytest.skip(f"{path} not found — run tests/otdb/download.py first")
        ots = list(read_order_types(path, n=4))
        assert len(ots) == 2

    def test_multiple_pb08(self):
        ots = [self._make_pts_sl(SAMPLE_POINTS_4),
               self._make_pts_sl(SAMPLE_POINTS_4B)]
        loaded = roundtrip(ots, "pb08", n=4)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

class TestJsonFormat:
    def test_roundtrip(self):
        pts = SAMPLE_POINTS_4
        ps = PointSet(len(pts), pts)
        sl = ps.to_small_lambda()
        sl.realization = list(pts)
        [sl2] = roundtrip([sl], "json", n=len(pts))
        assert sl2.n == len(pts)


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------

class TestBitPacking:
    def test_pack_unpack_roundtrip(self):
        for nbits in [1, 7, 8, 9, 15, 16, 20, 35, 56]:
            import random
            bits = [random.randint(0, 1) for _ in range(nbits)]
            assert _unpack_bits(_pack_bits(bits), nbits) == bits

    def test_padding(self):
        # 9 bits → 2 bytes; last 7 bits of byte 1 are zero-padded
        bits = [1] * 9
        data = _pack_bits(bits)
        assert len(data) == 2
        assert data[1] == 0b10000000

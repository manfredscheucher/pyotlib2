"""Tests for FelsnerMatrix encoding (Felsner 1997 replace matrix).

Round-trip: SmallLambda (natural labeling) → FelsnerMatrix → SmallLambda.
Comparison by chirotope (BigLambda), not L matrix — two different L matrices
can represent the same order type.

Bijection: all unique OTs for n produce distinct replace matrices.
"""

from pathlib import Path

import numpy as np
import pytest

from pyotlib2.core.felsner_matrix import FelsnerMatrix
from pyotlib2.io.readers import read_order_types
from pyotlib2.algorithms.unify import unify


def to_natural(ot):
    """Relabel OT to natural labeling starting from its first extremal point."""
    p0 = ot.get_extremal_points()[0]
    lab = ot.get_natural_labeling(p0)
    return ot.relabeled(lab)


def unique_ots(n, fname):
    path = f"tests/otdb/otypes/{fname}"
    if not Path(path).exists():
        pytest.skip(f"{path} not found — run tests/otdb/download.py first")
    ots = list(read_order_types(path, n=n))
    return list(unify(ots))


# ------------------------------------------------------------------
# Round-trip: encode → decode, compare by chirotope
# ------------------------------------------------------------------

class TestFelsnerRoundTrip:
    def _check_roundtrip(self, sl_nat):
        fm = FelsnerMatrix.from_small_lambda(sl_nat)
        sl2 = fm.to_small_lambda()
        bl1 = sl_nat.to_big_lambda()
        bl2 = sl2.to_big_lambda()
        assert np.array_equal(bl1.o, bl2.o), (
            f"round-trip chirotope mismatch for n={sl_nat.n}:\n"
            f"original L:\n{sl_nat.get_l()}\n"
            f"decoded L:\n{sl2.get_l()}"
        )

    def test_n3(self):
        for ot in unique_ots(3, "otypes03.b08"):
            self._check_roundtrip(to_natural(ot))

    def test_n4(self):
        for ot in unique_ots(4, "otypes04.b08"):
            self._check_roundtrip(to_natural(ot))

    def test_n5(self):
        for ot in unique_ots(5, "otypes05.b08"):
            self._check_roundtrip(to_natural(ot))

    def test_n6(self):
        for ot in unique_ots(6, "otypes06.b08"):
            self._check_roundtrip(to_natural(ot))

    def test_n7(self):
        for ot in unique_ots(7, "otypes07.b08"):
            self._check_roundtrip(to_natural(ot))

    @pytest.mark.slow
    def test_n8(self):
        for ot in unique_ots(8, "otypes08.b08"):
            self._check_roundtrip(to_natural(ot))


# ------------------------------------------------------------------
# Bijection: distinct OTs → distinct matrices
# ------------------------------------------------------------------

class TestFelsnerBijection:
    def _check_bijection(self, n, fname):
        ots = unique_ots(n, fname)
        matrices = set()
        for ot in ots:
            fm = FelsnerMatrix.from_small_lambda(to_natural(ot))
            matrices.add(fm.m.tobytes())
        assert len(matrices) == len(ots), (
            f"n={n}: {len(ots)} OTs but only {len(matrices)} distinct matrices"
        )

    def test_bijection_n3(self):
        self._check_bijection(3, "otypes03.b08")

    def test_bijection_n4(self):
        self._check_bijection(4, "otypes04.b08")

    def test_bijection_n5(self):
        self._check_bijection(5, "otypes05.b08")

    def test_bijection_n6(self):
        self._check_bijection(6, "otypes06.b08")

    def test_bijection_n7(self):
        self._check_bijection(7, "otypes07.b08")

    @pytest.mark.slow
    def test_bijection_n8(self):
        self._check_bijection(8, "otypes08.b08")


# ------------------------------------------------------------------
# Replace matrix properties: row sums and M[i,j] >= M[j,i] for i<j
# ------------------------------------------------------------------

class TestFelsnerMatrixProperties:
    def _check_properties(self, n, fname):
        for ot in unique_ots(n, fname):
            fm = FelsnerMatrix.from_small_lambda(to_natural(ot))
            m = fm.m
            # row sums: sum_j M[i,j] = n-1-i
            for i in range(n):
                assert int(m[i, :].sum()) == n - 1 - i, (
                    f"n={n}: row {i} sum {int(m[i,:].sum())} != {n-1-i}"
                )
            # M[i,j] >= M[j,i] for i < j
            for i in range(n):
                for j in range(i + 1, n):
                    assert int(m[i, j]) >= int(m[j, i]), (
                        f"n={n}: M[{i},{j}]={m[i,j]} < M[{j},{i}]={m[j,i]}"
                    )

    def test_n3(self):
        self._check_properties(3, "otypes03.b08")

    def test_n4(self):
        self._check_properties(4, "otypes04.b08")

    def test_n5(self):
        self._check_properties(5, "otypes05.b08")

    def test_n6(self):
        self._check_properties(6, "otypes06.b08")

    def test_n7(self):
        self._check_properties(7, "otypes07.b08")

    @pytest.mark.slow
    def test_n8(self):
        self._check_properties(8, "otypes08.b08")


# ------------------------------------------------------------------
# Serialisation round-trips: .ft and .fb
# ------------------------------------------------------------------

class TestFelsnerSerialisation:
    def test_fmt_roundtrip_n6(self):
        for ot in unique_ots(6, "otypes06.b08"):
            fm = FelsnerMatrix.from_small_lambda(to_natural(ot))
            s = fm.to_fmt_string()
            fm2 = FelsnerMatrix.from_fmt_string(6, s)
            assert fm == fm2

    def test_fmb_roundtrip_n6(self):
        for ot in unique_ots(6, "otypes06.b08"):
            fm = FelsnerMatrix.from_small_lambda(to_natural(ot))
            b = fm.to_fmb_bytes()
            fm2 = FelsnerMatrix.from_fmb_bytes(6, b)
            assert fm == fm2

    def test_fmt_roundtrip_n7(self):
        for ot in unique_ots(7, "otypes07.b08"):
            fm = FelsnerMatrix.from_small_lambda(to_natural(ot))
            s = fm.to_fmt_string()
            fm2 = FelsnerMatrix.from_fmt_string(7, s)
            assert fm == fm2

    def test_fmb_roundtrip_n7(self):
        for ot in unique_ots(7, "otypes07.b08"):
            fm = FelsnerMatrix.from_small_lambda(to_natural(ot))
            b = fm.to_fmb_bytes()
            fm2 = FelsnerMatrix.from_fmb_bytes(7, b)
            assert fm == fm2

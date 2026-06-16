"""Tests for triangulation counting algorithms.

Verifies count_triangulations_btfc and count_triangulations_modelcount
against known values (Catalan numbers for convex position, etc.)
and against each other on OTDB data.
"""

import time
import pytest

from tests.point_configs import (
    convex_position_moment_curve as _convex,
    one_interior,
    abstract_only,
)
from pyotlib2.algorithms.triangulations import (
    count_triangulations,
    count_triangulations_btfc,
    count_triangulations_modelcount,
)


# Catalan numbers C(n-2): triangulations of convex n-gon
#   n=3 → C(1)=1, n=4 → C(2)=2, n=5 → C(3)=5, n=6 → C(4)=14,
#   n=7 → C(5)=42, n=8 → C(6)=132
CATALAN = {3: 1, 4: 2, 5: 5, 6: 14, 7: 42, 8: 132}


# ---------------------------------------------------------------------------
# BTFC
# ---------------------------------------------------------------------------

class TestTriangulationsBTFC:
    @pytest.mark.parametrize("n,expected", list(CATALAN.items()))
    def test_convex_catalan(self, n, expected):
        """Convex n-gon has C(n-2) triangulations."""
        sl = _convex(n)
        assert count_triangulations_btfc(sl) == expected

    def test_abstract_only(self):
        """Works on abstract-only SmallLambda (no coordinates)."""
        sl = abstract_only(_convex(6))
        assert count_triangulations_btfc(sl) == CATALAN[6]

    def test_one_interior_n5(self):
        """one_interior(5): quadrilateral hull (k=4) + 1 interior = 3 triangulations."""
        sl = one_interior(5)
        val = count_triangulations_btfc(sl)
        assert val == 3

    def test_one_interior_n6(self):
        """Hexagon + 1 interior point: count must be positive and reasonable."""
        sl = one_interior(6)
        val = count_triangulations_btfc(sl)
        assert val > 0

    def test_api_default(self):
        """count_triangulations defaults to btfc."""
        sl = _convex(5)
        assert count_triangulations(sl) == count_triangulations_btfc(sl)


# ---------------------------------------------------------------------------
# ModelCount
# ---------------------------------------------------------------------------

class TestTriangulationsModelCount:
    @pytest.mark.parametrize("n,expected", [
        (3, 1), (4, 2), (5, 5), (6, 14), (7, 42),
    ])
    def test_convex_catalan(self, n, expected):
        """ModelCount gives Catalan numbers for convex n-gon."""
        sl = _convex(n)
        assert count_triangulations_modelcount(sl) == expected

    def test_agrees_with_btfc_n3_to_n6(self):
        """BTFC and ModelCount agree on convex n=3..6."""
        for n in range(3, 7):
            sl = _convex(n)
            assert count_triangulations_modelcount(sl) == count_triangulations_btfc(sl), \
                f"Mismatch at n={n}"

    def test_agrees_with_btfc_interior(self):
        """BTFC and ModelCount agree on one_interior(5) and one_interior(6)."""
        for n in (5, 6):
            sl = one_interior(n)
            assert count_triangulations_modelcount(sl) == count_triangulations_btfc(sl), \
                f"Mismatch at one_interior(n={n})"

    def test_api_modelcount(self):
        """count_triangulations(method='modelcount') routes correctly."""
        sl = _convex(5)
        assert count_triangulations(sl, method="modelcount") == CATALAN[5]


# ---------------------------------------------------------------------------
# OTDB
# ---------------------------------------------------------------------------

class TestTriangulationsOTDB:
    def _load_otdb(self, n):
        from pathlib import Path
        from pyotlib2.io.readers import read_order_types
        path = f"tests/otdb/otypes/otypes{n:02d}.b08"
        if not Path(path).exists():
            pytest.skip(f"{path} not found — run tests/otdb/download.py first")
        return list(read_order_types(path, fmt="pb08", n=n))

    def test_otdb_n6_sum(self):
        """Sum of triangulation counts over all 16 OTs for n=6."""
        ots = self._load_otdb(6)
        assert len(ots) == 16
        total = sum(count_triangulations_btfc(ot) for ot in ots)
        # All 16 OTs for n=6; sum of triangulation counts is a fixed value.
        # The minimum is 14 (all-convex hexagon) and maximum is higher for
        # interior-point configurations.  We just check positivity and convex case.
        assert total >= 16 * 1   # at least 1 per OT
        # The all-convex OT (first in OTDB) has exactly 14 triangulations
        convex_count = count_triangulations_btfc(ots[0])
        assert convex_count == CATALAN[6]

    def test_otdb_n6_btfc_eq_modelcount(self):
        """BTFC == ModelCount for all 16 OTs n=6."""
        ots = self._load_otdb(6)
        for i, ot in enumerate(ots):
            b = count_triangulations_btfc(ot)
            m = count_triangulations_modelcount(ot)
            assert b == m, f"Mismatch on OT #{i}: btfc={b} modelcount={m}"

    @pytest.mark.slow
    def test_otdb_n7_spot_check(self):
        """BTFC == ModelCount for first 10 OTs of n=7."""
        ots = self._load_otdb(7)
        for ot in ots[:10]:
            assert count_triangulations_btfc(ot) == count_triangulations_modelcount(ot)


# ---------------------------------------------------------------------------
# Timing table (informational, not a correctness check)
# ---------------------------------------------------------------------------

class TestTriangulationsTiming:
    def test_timing_table(self, capsys):
        """Print timing comparison table for BTFC vs ModelCount."""
        print()
        print(f"{'n':>3}  {'config':<12}  {'btfc_time':>10}  {'mc_time':>10}  {'count':>8}")
        print("-" * 55)
        for n in range(3, 8):
            sl = _convex(n)
            t0 = time.perf_counter()
            cnt_b = count_triangulations_btfc(sl)
            t_b = time.perf_counter() - t0

            t0 = time.perf_counter()
            cnt_m = count_triangulations_modelcount(sl)
            t_m = time.perf_counter() - t0

            assert cnt_b == cnt_m, f"n={n}: btfc={cnt_b} != mc={cnt_m}"
            print(f"{n:>3}  {'convex':<12}  {t_b:>9.4f}s  {t_m:>9.4f}s  {cnt_b:>8}")

        with capsys.disabled():
            pass  # output already captured by print above

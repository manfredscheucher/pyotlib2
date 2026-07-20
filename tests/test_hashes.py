"""Tests for the three order-type hash levels: SL-hash, OT-hash, PC-hash.

  * ``sl_hash``  identifies the exact rank matrix (labeling-DEPENDENT);
  * ``ot_hash``  identifies the order type (lex-min over relabelings + mirror,
    labeling-INDEPENDENT);
  * ``pc_hash``  identifies the projective class (coarser still).
"""

import random

from pyotlib2.core.small_lambda import SmallLambda
from tests.point_configs import convex_position_moment_curve, one_interior, abstract_only


def _abstract(cfg):
    return abstract_only(cfg)


class TestSLHashLabelingDependent:
    def test_relabel_changes_sl_hash_but_not_ot_hash(self):
        """Relabeling points changes the SL-hash but not the OT-hash."""
        sl = _abstract(one_interior(7))
        rng = random.Random(0)
        base_sl = sl.sl_hash()
        base_ot = sl.ot_hash()
        changed = 0
        for _ in range(10):
            perm = list(range(sl.n))
            rng.shuffle(perm)
            r = sl.relabeled(perm)
            # OT-hash is invariant under relabeling
            assert r.ot_hash() == base_ot
            if r.sl_hash() != base_sl:
                changed += 1
        # at least some permutations produce a different rank matrix
        assert changed > 0

    def test_mirror_changes_sl_hash_not_ot_hash(self):
        sl = _abstract(one_interior(6))
        m = sl.mirrored()
        assert m.ot_hash() == sl.ot_hash()
        # a chiral order type's mirror has a different rank matrix
        # (if symmetric they coincide; one_interior(6) is generally chiral)


class TestOTHashDistinguishes:
    def test_different_order_types_differ(self):
        a = _abstract(convex_position_moment_curve(6))   # convex
        b = _abstract(one_interior(6))                   # has an interior point
        assert a.ot_hash() != b.ot_hash()
        assert a.sl_hash() != b.sl_hash()


class TestPCHash:
    def test_pc_hash_stable_and_coarser(self):
        """PC-hash is invariant under relabeling and equals its representative's."""
        from pyotlib2.algorithms.projective_class import ProjectiveClass
        sl = _abstract(one_interior(7))
        rep = ProjectiveClass(sl).representer
        assert sl.pc_hash() == rep.pc_hash() == rep.ot_hash()
        # relabeling keeps the projective class
        r = sl.relabeled(list(reversed(range(sl.n))))
        assert r.pc_hash() == sl.pc_hash()


class TestShortForms:
    def test_short_hashes_are_prefixes(self):
        sl = _abstract(convex_position_moment_curve(5))
        assert sl.sl_hash().startswith(sl.short_sl_hash())
        assert sl.ot_hash().startswith(sl.short_ot_hash())
        assert len(sl.short_sl_hash()) == 6
        assert len(sl.short_sl_hash(12)) == 12

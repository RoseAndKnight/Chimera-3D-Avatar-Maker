"""Perfect Sync transfer on a made-up face, built here: no model and no donor file are needed.

    <python> -I tools/vroid/tests/test_perfect_sync.py

The face is two eye openings (a hexagon each, ringed by skin). The donor's eyeBlinkLeft/Right bring the upper lid
exactly onto the lower one. The face to dress has the same vertices in another order, larger eyes (lids at +-0.6
instead of +-0.5), a vertex the donor does not have, and the two eyes share their UVs (as the irises do).
"""
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE)]
import perfect_sync as PS     # noqa: E402

LID = np.array([(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0), (2.0, -1.0), (1.0, -1.0)])   # corner, upper, corner, lower


def eye(sx, half):
    """One eye on side sx (+1: character's left, +x): loop then outer ring; faces; UV (the same on both sides)."""
    loop = LID * [1.0, half]
    ring = (LID - [1.5, 0.0]) * 2.0 + [1.5, 0.0]
    xy = np.concatenate([loop, ring])
    pos = np.column_stack([sx * (xy[:, 0] + 1.0), xy[:, 1] + 18.0, np.zeros(len(xy))])
    uv = np.column_stack([(LID[:, 0] + 1.0) / 10.0, (LID[:, 1] + 2.0) / 10.0])
    uv = np.concatenate([uv, uv + [0.5, 0.0]])
    F = []
    for i in range(6):
        k = (i + 1) % 6
        F += [(i, k, 6 + k), (i, 6 + k, 6 + i)]
    return pos, uv, np.array(F)


def face(half):
    (pl, ul, fl), (pr, ur, fr) = eye(1.0, half), eye(-1.0, half)
    return np.concatenate([pl, pr]), np.concatenate([ul, ur]), np.concatenate([fl, fr + len(pl)])


def blink(pos, sx):
    """Upper lid of the eye on side sx brought onto the lower lid (donor: exactly closed)."""
    d = np.zeros_like(pos)
    for i in (1, 2):
        v = i if sx > 0 else 12 + i
        d[v, 1] = -2.0 * 0.5
    return d


class PerfectSync(unittest.TestCase):

    def setUp(self):
        P, UV, F = face(0.5)
        self.folder = tempfile.mkdtemp()
        targets = {"eyeBlinkLeft": blink(P, 1.0), "eyeBlinkRight": blink(P, -1.0),
                   "jawOpen": np.tile([0.0, -0.1, 0.0], (len(P), 1)), "Fcl_EYE_Close": blink(P, 1.0) + blink(P, -1.0)}
        Dn = {"pos": P, "uv": UV, "key": np.full(len(P), PS.SKIN, object), "faces": {PS.SKIN: F}, "targets": targets,
              "path": "donor.vrm", "sort": "female", "edges": {}}
        PS._donors[os.path.join(self.folder, PS.DONOR_FILES["female"])] = Dn
        # the face to dress: larger eyes, vertices shuffled, one extra vertex without a donor twin
        Q, UQ, FQ = face(0.6)
        order = np.random.default_rng(1).permutation(len(Q))
        back = np.argsort(order)
        self.pos = np.concatenate([Q[order], [[0.0, 20.0, 0.0]]])
        self.uv = np.concatenate([UQ[order], [[0.9, 0.9]]])
        self.faces = back[FQ]
        self.order = order

    def run_add(self):
        rows = np.arange(len(self.pos))
        return PS.add(self.pos, self.uv, rows, np.full(len(rows), PS.SKIN, object), self.faces, {"Fcl_EYE_Close"},
                      "female", folder=self.folder, log=lambda x: None)

    def test_match_by_uv_and_side(self):
        out, rep = self.run_add()
        self.assertEqual(rep["unmatched"], 1)
        self.assertEqual(rep["unique"] + rep["ambiguous"], len(self.pos) - 1)
        self.assertEqual(rep["ambiguous"], len(self.pos) - 1)      # every UV is on both sides: told apart by position
        m, _ = PS.match(PS._donors[os.path.join(self.folder, PS.DONOR_FILES["female"])], self.pos[:-1], self.uv[:-1],
                        np.full(len(self.pos) - 1, PS.SKIN, object))
        self.assertTrue(np.array_equal(m, self.order))

    def test_absent_shapes_added_present_kept(self):
        out, rep = self.run_add()
        self.assertEqual(sorted(out), ["eyeBlinkLeft", "eyeBlinkRight", "jawOpen"])
        idx, d = out["jawOpen"]
        self.assertNotIn(len(self.pos) - 1, idx)                    # the unmatched vertex stays still
        self.assertTrue(np.allclose(d, [0.0, -0.1, 0.0]))           # copied as is

    def test_blink_calibrated_to_close(self):
        out, rep = self.run_add()
        for name, sx in (("eyeBlinkLeft", 1.0), ("eyeBlinkRight", -1.0)):
            self.assertAlmostEqual(rep["calibration"][name]["factor"], 1.2, places=3)
            idx, d = out[name]
            P = self.pos.copy()
            P[idx] += d
            self.assertTrue(np.allclose(np.abs(d[:, 1]), 1.2))    # 2 x 0.6: the upper lid lands on the lower one
            self.assertTrue(np.all(np.sign(self.pos[idx, 0]) == sx))

    def test_no_donor_nothing_added(self):
        rows = np.arange(len(self.pos))
        out, rep = PS.add(self.pos, self.uv, rows, np.full(len(rows), PS.SKIN, object), self.faces, set(), "male",
                          folder=self.folder, log=lambda x: None)
        self.assertEqual(out, {})
        self.assertIn("skipped", rep)


if __name__ == "__main__":
    unittest.main()

"""Tests of the VRoid base that need no model.
    python3 -I tools/vroid_base/tests/test_vroid_base.py"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.dirname(HERE), TOOLS]
import build as FA        # noqa: E402
import skin as PE             # noqa: E402

SKIN_RVB = (0.95, 0.80, 0.72)


def texture(n=128):
    """Plain skin, a blue square (a painted garment) and a transparent square (skin erased under the outfit)."""
    t = np.ones((n, n, 4))
    t[..., :3] = SKIN_RVB
    t[20:60, 20:60, :3] = (0.2, 0.3, 0.8)
    t[70:110, 70:110, 3] = 0
    return t


class Skin(unittest.TestCase):
    def test_garment_and_transparency_repainted_in_skin(self):
        t = texture()
        face = np.ones((16, 16, 4))
        face[..., :3] = SKIN_RVB
        own, known = PE.clean(t, face)
        part = known.mean()
        self.assertEqual(own[..., 3].min(), 1.0)
        for zone in (own[30:50, 30:50, :3], own[80:100, 80:100, :3]):
            self.assertLess(np.abs(zone - SKIN_RVB).max(), 0.02)
        self.assertTrue(0.5 < part < 0.9, part)

    def test_mask_refused_border_and_islands(self):
        t = texture(256)
        t[200:203, 10:13, :3] = (0.2, 0.3, 0.8)            # tiny garment
        self.assertTrue(PE.mask_skin(t, SKIN_RVB)[201, 30])
        m = PE.mask_skin(t, SKIN_RVB)
        self.assertFalse(m[40, 40])                         # garment
        self.assertFalse(m[19, 40])                         # eroded edge (1 texel at this size)
        self.assertTrue(m[10, 200])
        self.assertFalse(m[90, 90])                         # transparent

    def test_seams_matched_without_touch_skin_kept(self):
        n = 256
        t = np.ones((n, n, 4))
        t[:, :128, :3] = (0.9, 0.7, 0.6)                    # "leg" island, repainted
        t[:, 128:, :3] = (0.8, 0.6, 0.5)                    # "foot" island, repainted
        known = np.zeros((n, n), bool)
        known[:, 250:] = True                               # a little kept skin, far from the seam
        v = (np.arange(40) + 100.5) / n
        uv_a = np.stack([np.full(40, 100.5 / n), v], 1)     # edge of the leg
        uv_b = np.stack([np.full(40, 150.5 / n), v], 1)     # the same edge, foot side
        out = PE.match_seams(t, known, uv_a, uv_b)
        self.assertLess(np.abs(out[120, 100, :3] - out[120, 150, :3]).max(), 0.01)
        self.assertTrue(np.array_equal(out[known], t[known]))
        self.assertLess(np.abs(out[120, 5, :3] - t[120, 5, :3]).max(), 0.05)    # far from the seam: almost nothing

    def test_fill_keep_known(self):
        rgb = np.random.default_rng(0).random((33, 17, 3))
        known = np.zeros((33, 17), bool)
        known[::3, ::2] = True
        out = PE.fill(rgb, known)
        self.assertTrue(np.array_equal(out[known], rgb[known]))
        self.assertTrue(np.isfinite(out).all())
        with self.assertRaises(ValueError):
            PE.fill(rgb, np.zeros((33, 17), bool))


def pmx_toy():
    """Two "肌" triangles and one "颜", three bones, an expression that touches both materials."""
    pos = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (5, 5, 0), (6, 5, 0), (5, 6, 0)]
    n = len(pos)
    s = {"pos": pos, "nor": [(0, 0, 1)] * n, "uv": [(0, 0)] * n, "uv_add": [[]] * n, "type": [0] * n,
         "bones": [(0,)] * n, "weights": [(1.0,)] * n, "sdef": [None] * n, "contour": [1.0] * n}
    mat = lambda name, k, t: {"name": name, "name_en": name, "texture": t, "n_indices": 3 * k}   # noqa: E731
    return {"vertices": s, "faces": [(4, 5, 6), (0, 1, 2), (1, 3, 2)], "textures": ["a.png", "b.png"],
            "materials": [mat("颜", 1, 0), mat("肌", 2, 1)], "bones": [{"name": "頭"}],
            "morphs": [{"name": "あ", "type": 1, "offsets": [(5, (0, 1, 0)), (1, (0, 0, 1))]}],
            "frames": [], "rigid_bodies": [1], "joints": [1]}


def arrays(P):
    mf = np.concatenate([[i] * (m["n_indices"] // 3) for i, m in enumerate(P["materials"])]).astype(int)
    return {"pos": np.array(P["vertices"]["pos"], float), "faces": np.array(P["faces"]), "mat_face": mf}


class UnderPmx(unittest.TestCase):
    def test_body_alone_then_head_in_same_order(self):
        P = pmx_toy()
        T = arrays(P)
        body, nv = FA.under_pmx(P, T, {"肌"})
        self.assertEqual(len(body["vertices"]["pos"]), 4)
        self.assertEqual(body["textures"], ["b.png"])
        self.assertEqual(body["materials"][0]["texture"], 0)
        self.assertEqual([m["name"] for m in body["materials"]], ["肌"])
        self.assertEqual(body["rigid_bodies"], [])
        self.assertEqual(body["morphs"][0]["offsets"], [(1, (0, 0, 1))])
        first = np.where(nv >= 0)[0][np.argsort(nv[nv >= 0])]
        head, nv2 = FA.under_pmx(P, T, {"肌", "颜"}, first)
        self.assertEqual(head["vertices"]["pos"][:4], body["vertices"]["pos"])       # join_skin: same order
        skin_t = [f for m, f in zip(arrays(head)["mat_face"], head["faces"]) if head["materials"][m]["name"] == "肌"]
        self.assertEqual(skin_t, body["faces"])
        self.assertEqual(len(head["morphs"][0]["offsets"]), 2)


class Seams(unittest.TestCase):
    def test_borders_of_seams(self):
        # two triangles that share an edge at the same spot, but through split vertices with different uv
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 1, 0), (-1, 0, 0)], float)
        uv = np.array([(0.1, 0.1), (0.2, 0.1), (0.1, 0.2), (0.7, 0.7), (0.7, 0.8), (0.6, 0.7)])
        T = {"pos": pos, "uv": uv, "faces": np.array([(0, 1, 2), (3, 4, 5)]), "mat_face": np.array([0, 0])}
        a, b = FA.borders_of_seams(T, 0, stage=4)
        self.assertEqual(len(a), 4)
        if a[0, 0] > 0.5:
            a, b = b, a                                     # the order of the two edges does not matter
        self.assertTrue(np.allclose(a[:, 0], 0.1) and np.allclose(b[:, 0], 0.7))
        self.assertTrue(np.allclose(a[:, 1] - 0.1, b[:, 1] - 0.7))   # same point of the edge on both sides
        T["uv"] = np.array([(0.1, 0.1), (0.2, 0.1), (0.1, 0.2), (0.1, 0.1), (0.1, 0.2), (0.0, 0.1)])
        self.assertEqual(len(FA.borders_of_seams(T, 0)[0]), 0)    # same uv: not a seam


class Checks(unittest.TestCase):
    def test_openings(self):
        # a closed tetrahedron, then with one face removed
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], float)
        F = np.array([(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)])
        self.assertEqual(FA.openings(pos, F), [])
        self.assertEqual(FA.openings(pos, F[:3])[0][0], 3)
        # a seam: split vertices at the same point, no hole
        pos2 = np.vstack([pos, pos[[0]]])
        F2 = F.copy()
        F2[3] = (4, 3, 2)
        self.assertEqual(FA.openings(pos2, F2), [])

    def test_angle_arm(self):
        o = [{"name": "左腕", "pos": (1, 10, 0)}, {"name": "左ひじ", "pos": (2, 9, 0)},
             {"name": "右腕", "pos": (-1, 10, 0)}, {"name": "右ひじ", "pos": (-2, 10, 0)}]
        a = FA.angle_arm(o)
        self.assertAlmostEqual(a[0], 45.0)
        self.assertAlmostEqual(a[1], 0.0)

    def test_origin_warnings(self):
        m = {k: None for k in FA.META}
        m.update(title="X", allowedUserName="OnlyAuthor", licenseName="Redistribution_Prohibited", commercialUssageName="Disallow")
        pj, md = FA.origin(m, "/x/y.vrm", "y", "0" * 64)
        self.assertEqual(len(pj["warnings"]), 3)
        self.assertIn("Warning", md)
        self.assertNotIn("/x/", md)                       # no machine path in the provenance
        m.update(allowedUserName="Everyone", licenseName="CC0", commercialUssageName="Allow")
        self.assertEqual(FA.origin(m, "/x/y.vrm", "y", "0" * 64)[0]["warnings"], [])

    def test_origin_vrm_1(self):
        m = {k: None for k in FA.META_1}
        m.update(vrm="1.0", title="X", name="X", licenseUrl="https://vrm.dev/licenses/1.0/")
        pj, md = FA.origin(m, "/x/y.vrm", "y", "0" * 64)
        self.assertEqual(len(pj["warnings"]), 4)           # absent keys take the strictest 1.0 defaults
        self.assertIn("VRM 1.0 meta", md)
        m.update(avatarPermission="everyone", allowRedistribution=True, commercialUsage="corporation", modification="allowModification")
        pj, md = FA.origin(m, "/x/y.vrm", "y", "0" * 64)
        self.assertEqual(pj["warnings"], [])
        self.assertIn("`commercialUsage` : corporation", md)


class BaseHeadBuilt(unittest.TestCase):
    def test_setting_then_base_then_description(self):
        with tempfile.TemporaryDirectory() as t:
            os.makedirs(os.path.join(t, "vroid-base"))
            with open(os.path.join(t, "vroid-base", "base.json"), "w") as f:
                json.dump({"base_head": {"name": "Model", "eye_gap": 1.06}}, f)
            regl = os.path.join(t, "settings.json")
            with open(regl, "w") as f:
                json.dump({"paths": {"data": t}, "workshop": {"body": "{data}/vroid-base/body.pmx"},
                           "games": {"vroid": {"base_head": {"eye_gap": 2.0}}}}, f)
            old = os.environ.get("CHIMERA_SETTINGS")
            os.environ["CHIMERA_SETTINGS"] = regl
            import settings as R
            import games as games
            try:
                R.forget()
                J = games.Game("vroid")
                self.assertEqual(J.value("base_head.name"), "Model")
                self.assertEqual(J.value("base_head.eye_gap"), 2.0)
                self.assertEqual(J.value("base_head.tint"), "face")
                self.assertIsNone(J.value("base_head.eyes_center", None))
                os.remove(os.path.join(t, "vroid-base", "base.json"))
                self.assertIsNone(J.value("base_head.name", None))
            finally:
                if old is None:
                    os.environ.pop("CHIMERA_SETTINGS", None)
                else:
                    os.environ["CHIMERA_SETTINGS"] = old
                R.forget()


class CatalogueOnBase(unittest.TestCase):
    def test_base_open_to_models_ingested(self):
        """A VRoid base (base.json: base_head) offers its head then those of the imported models, except the
        character it is made from; MakeHuman's free base offers only itself."""
        from unittest import mock
        sys.path.insert(0, os.path.join(TOOLS, "workshop"))
        import catalog as K
        game = mock.Mock(value=lambda the_key, default=None: "all" if the_key == "catalog.heads" else default)
        own_rig = mock.Mock(value=lambda the_key, default=None: default)
        with mock.patch.object(K, "base_free", return_value=True), \
             mock.patch.object(K, "hairstyles_of_game", return_value=["D", "E", "F"]), \
             mock.patch.object(K, "OWN_RIG", own_rig), mock.patch.object(K.C, "GAME", game):
            with mock.patch.object(K, "BASE_JSON", {"base_head": {"name": "D"}}):
                self.assertTrue(K.base_open())
                self.assertEqual(K.catalog("heads", False), [K.FREE, "E", "F"])
                self.assertEqual(K.key_base(), K.FREE)
            with mock.patch.object(K, "BASE_JSON", {}):
                self.assertFalse(K.base_open())
                self.assertEqual(K.catalog("heads", False), [K.FREE])


if __name__ == "__main__":
    unittest.main()

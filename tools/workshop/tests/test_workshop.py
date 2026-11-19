"""Basic tests of the workshop (numpy only):
    python3 -I tools/workshop/tests/test_workshop.py
"""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common as C  # noqa: E402
import outfit as CB  # noqa: E402
import glb as GLB  # noqa: E402

import geom as G  # noqa: E402
import coverage as CV  # noqa: E402
import vrm as VR  # noqa: E402


def sheet(z, n=20, width=0.3):
    """Square of n x n vertices in the plane z, normals towards +z."""
    xs = np.linspace(-width / 2, width / 2, n)
    X, Y = np.meshgrid(xs, xs)
    pos = np.c_[X.ravel(), Y.ravel(), np.full(n * n, z)].astype(float)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            F += [(a, a + 1, a + n), (a + 1, a + n + 1, a + n)]
    F = np.array(F)
    nor = np.tile([0.0, 0.0, 1.0], (n * n, 1))
    rep, inv = G.weld(pos, 1e-5)
    return SimpleNamespace(id="f", name="sheet", pos=pos, nor=nor, F=F, rep=rep, inv=inv, meta={"rank": 1}, flags=None)


def cylinder(r, n=48, m=12, h=0.3):
    """Vertical tube of radius r, normals and faces pointing outwards (like a skirt)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ys = np.linspace(0, h, m)
    pos = np.array([(r * np.cos(t), y, r * np.sin(t)) for y in ys for t in a])
    nor = np.array([(np.cos(t), 0.0, np.sin(t)) for y in ys for t in a])
    F = []
    for j in range(m - 1):
        for i in range(n):
            a0, b0 = j * n + i, j * n + (i + 1) % n
            a1, b1 = a0 + n, b0 + n
            F += [(a0, a1, b0), (b0, a1, b1)]
    F = np.array(F)
    rep, inv = G.weld(pos, 1e-5)
    return SimpleNamespace(id="c", name="tube", pos=pos, nor=nor, F=F, rep=rep, inv=inv, meta={"rank": 1}, flags=None)


def layer(p, meta=None):
    if meta:
        p.meta = meta
    return CB.Layer(p)


class TestGlb(unittest.TestCase):
    def test_go_back_and_orientation(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        nor = np.array([[0, 0, -1.0]] * 3)   # in MMD: -Z, so +Z once in glTF
        p, n, f = GLB.to_gltf(pos, nor, np.array([[0, 2, 1]]))
        a, b, c = p[f[0]]
        self.assertGreater(float(np.dot(np.cross(b - a, c - a), n[0])), 0)
        self.assertAlmostEqual(float(p[1, 0]), 0.08, places=6)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.glb")
            GLB.write(path, p, n, np.zeros((3, 2)), [{"faces": f, "material": 0}],
                       [{"name": "m", "image": "/data/tex/a.webp", "diffuse": (1, 1, 1, 1)}])
            doc, b_ = GLB.read(path)
            self.assertEqual(doc["asset"]["version"], "2.0")
            self.assertEqual(doc["images"][0]["uri"], "/data/tex/a.webp")
            self.assertEqual(doc["accessors"][0]["count"], 3)
            self.assertEqual(len(b_) % 4, 0)

    def test_face_a_l_reverse_flipped(self):
        # strip of four faces; the third is wound backwards and its normals are wrong too
        pos = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]], float)
        nor = np.tile([0, 0, 1.0], (6, 1))
        F = GLB.a_l_upright(pos, nor, [[0, 2, 1], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
        g = np.cross(pos[F[:, 1]] - pos[F[:, 0]], pos[F[:, 2]] - pos[F[:, 0]])
        self.assertTrue((g[:, 2] > 0).all())
        self.assertEqual(F[0].tolist(), [0, 2, 1])

    def test_shoe_placed_d_one_block(self):
        # two feet rotated and enlarged by fitting, a tenth of the vertices gone elsewhere
        import repair as RP
        rng = np.random.default_rng(0)
        S = rng.uniform(-0.05, 0.05, (200, 3)) + np.where(np.arange(200)[:, None] < 100, [-0.1, 0, 0], [0.1, 0, 0])
        c, s_ = np.cos(0.2), np.sin(0.2)
        exact = 1.2 * S @ np.array([[1, 0, 0], [0, c, s_], [0, -s_, c]]) + [0, 0.03, 0.01]
        Z = exact.copy()
        Z[::10] += rng.uniform(0.03, 0.08, (20, 3))
        self.assertLess(np.abs(RP.place(S, Z) - exact).max(), 1e-6)

    def test_island_of_skin_on_fabric_cache(self):
        # two skin faces facing +z, apart; a square of fabric under the first one only
        import skin as PE
        V = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0.2, 0, 0], [0.21, 0, 0], [0.2, 0.01, 0]], float)
        Fp = np.array([[0, 1, 2], [3, 4, 5]])
        N = np.tile([0, 0, 1.0], (6, 1))
        old = PE._skin, CB.skin
        PE._skin = lambda: (V[Fp], np.tile([0, 0, 1.0], (2, 1)), np.array([0.5, 0.5]))
        CB.skin = lambda: (V, Fp, N, None)
        try:
            F = np.array([[0, 1, 2], [1, 3, 2]])
            square = np.array([[-0.02, -0.02, 0], [0.03, -0.02, 0], [-0.02, 0.03, 0], [0.03, 0.03, 0]], float)
            nothing = np.zeros(0, np.int64)
            self.assertEqual(PE.sunk(square + [0, 0, -0.004], F, nothing).tolist(), [True, False])
            self.assertFalse(PE.sunk(square + [0, 0, 0.004], F, nothing).any())     # fabric in front: nothing more to hide
            self.assertFalse(PE.sunk(square + [0, 0, -0.004], F, np.array([0])).any())   # already hidden
        finally:
            PE._skin, CB.skin = old

    def test_part_rereads_one_real_glb(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
        nor = np.array([[0, 0, 1.0]] * 4)
        F = np.array([[0, 1, 2], [1, 3, 2]])
        with tempfile.TemporaryDirectory() as d:
            old, C.CACHE = C.CACHE, d
            try:
                os.makedirs(os.path.join(d, "parts", "p"))
                C.write_json(os.path.join(d, "parts", "p", "meta.json"), {"type_name": "attempt"})
                GLB.write(os.path.join(d, "parts", "p", "part.glb"), pos, nor, np.zeros((4, 2)),
                           [{"faces": F, "material": 0}], [{"name": "m", "image": None, "diffuse": (1, 1, 1, 1)}])
                p = CB.Part("p")
            finally:
                C.CACHE = old
        self.assertEqual(p.F.tolist(), F.tolist())
        self.assertEqual(len(p.pos), 4)


class TestCommon(unittest.TestCase):
    def test_key_sure(self):
        self.assertNotIn("/", C.the_key("a/b", "c d"))
        self.assertEqual(C.the_key("Anby", "skirt-1"), "Anby__skirt-1")

    def test_order_of_layers(self):
        r = lambda t: C.CLOTHES[t][2]  # noqa: E731
        self.assertLess(r("underwear"), r("top"))
        self.assertLess(r("top"), r("skirt"))
        self.assertLess(r("skirt"), r("jacket"))
        self.assertLess(r("jacket"), C.ACCESSORIES["belt"][0])


class TestLayers(unittest.TestCase):
    def test_push_sort_layer_of_over(self):
        under, over = sheet(0.0), sheet(-0.002)   # the upper one is 2 mm inside
        d = CB.push(over, over.pos, [(under.pos, under.F, under.nor)])
        Z = over.pos + d
        center = np.abs(Z[:, :2]).max(1) < 0.1
        self.assertTrue(np.all(Z[center, 2] >= CB.MARGIN - 1e-6))

    def test_crossing_flagged(self):
        top = layer(cylinder(0.15))
        bottom = layer(cylinder(0.156))   # the lower one pokes out by 6 mm
        r = CB.crosses(bottom, top)
        self.assertIsNotNone(r)
        self.assertGreater(r["fraction"], 0.5)
        own = layer(cylinder(0.144))
        r2 = CB.crosses(own, top)
        self.assertTrue(r2 is None or r2["fraction"] == 0)

    def test_faces_masked_not_count_plus(self):
        top = layer(cylinder(0.15))
        bottom = layer(cylinder(0.156))
        bottom.mask[:] = True
        self.assertIsNone(CB.crosses(bottom, top))


class TestRulesOfLayers(unittest.TestCase):
    def test_underwear_removed_under_one_bottom_closed(self):
        for bottom in ("shorts", "trousers"):
            r = CB.removed({"c": {"type": "underwear", "type_name": "underwear"}, "b": {"type": bottom, "type_name": bottom}})
            self.assertEqual(list(r), ["c"])

    def test_underwear_kept_under_one_skirt(self):
        self.assertEqual(CB.removed({"c": {"type": "underwear"}, "j": {"type": "skirt"}}), {})

    def test_tuck_passed_under_without_inflate_over(self):
        over, under = layer(cylinder(0.15)), layer(cylinder(0.156))
        before = over.Z.copy()
        self.assertGreater(CB.tuck(under, over), 0)
        self.assertTrue(np.array_equal(over.Z, before))
        self.assertLess(float(np.hypot(under.Z[:, 0], under.Z[:, 2]).max()), 0.15)

    def test_mask_this_which_is_cache_and_held(self):
        over, under = layer(cylinder(0.15)), layer(cylinder(0.145))
        CB.hide(under, over)
        self.assertTrue(under.mask.all())

    def test_not_mask_step_under_one_part_which_floats(self):
        over, under = layer(cylinder(0.20)), layer(cylinder(0.145))
        CB.hide(under, over)
        self.assertFalse(under.mask.any())

    def test_not_mask_step_under_one_fabric_perforated(self):
        c = cylinder(0.15)
        c.flags = np.full(len(c.F), CB.OUTFIT | CB.OUTER, np.uint8)    # held, but not opaque
        over, under = layer(c), layer(cylinder(0.145))
        CB.hide(under, over)
        self.assertFalse(under.mask.any())

    def test_top_whose_l_hem_sort_rest_outside(self):
        top, bottom = layer(cylinder(0.16), {"type": "top", "rank": 20}), layer(cylinder(0.15), {"type": "trousers", "rank": 30})
        order, notes = CB.order([top, bottom])
        self.assertEqual([c.type for c in order], ["trousers", "top"])
        self.assertTrue(notes)

    def test_top_tucked_rest_under(self):
        top, bottom = layer(cylinder(0.145), {"type": "top", "rank": 20}), layer(cylinder(0.15), {"type": "trousers", "rank": 30})
        order, notes = CB.order([top, bottom])
        self.assertEqual([c.type for c in order], ["top", "trousers"])
        self.assertFalse(notes)

    def test_trousers_tucked_in_boot(self):
        old, CB.body = CB.body, lambda: None
        try:
            trousers = cylinder(0.156)
            trousers.id, trousers.meta = "p", {"type": "trousers", "tab": "bottom", "rank": 30}
            boot = cylinder(0.15)
            boot.id, boot.meta = "b", {"type": "boots", "tab": "shoes", "rank": 60}
            layers, _ = CB.resolve([boot, trousers])
        finally:
            CB.body = old
        self.assertEqual([c.id for c in layers], ["p", "b"])
        self.assertTrue(layers[0].mask.all() or float(np.hypot(layers[0].Z[:, 0], layers[0].Z[:, 2]).max()) < 0.15)

    def test_belt_placed_on_one_top_a_wide_stitches(self):
        old, CB.body = CB.body, lambda: None
        try:
            top = cylinder(0.16)
            top.id, top.meta = "h", {"type": "top", "tab": "top", "rank": 20}
            belt = cylinder(0.15)
            belt.id, belt.meta = "c", {"type": "belt", "tab": "accessories", "rank": 70}
            layers, _ = CB.resolve([belt, top])
        finally:
            CB.body = old
        per_id = {c.id: c for c in layers}
        self.assertTrue(np.array_equal(per_id["h"].Z, top.pos))
        self.assertGreater(float(np.hypot(per_id["c"].Z[:, 0], per_id["c"].Z[:, 2]).min()), 0.16)

    def test_high_sock_under_underwear_but_step_tights(self):
        underwear = cylinder(0.15)
        bottom, tights = cylinder(0.15), cylinder(0.15)
        bottom.pos = bottom.pos - [0.0, float(bottom.pos[:, 1].max() - bottom.pos[:, 1].min()) / 2, 0.0]
        tights.pos = tights.pos + [0.0, 0.02, 0.0]
        def order(ch):
            cs, _ = CB.order([layer(underwear, {"type": "underwear", "rank": 10}), layer(ch, {"type": "socks", "rank": 15})])
            return [c.type for c in cs]
        self.assertEqual(order(bottom), ["socks", "underwear"])
        self.assertEqual(order(tights), ["underwear", "socks"])

    def test_crossing_hidden_per_one_other_part(self):
        self.assertFalse(CB.crossing_visible(0.08, 0.06))
        self.assertTrue(CB.crossing_visible(0.08, 0.02))
        self.assertFalse(CB.crossing_visible(0.03, 0.0))

    def test_shoe_without_over(self):
        foot = np.array([(x, 0.05, z) for x in np.linspace(-0.04, 0.04, 9) for z in np.linspace(-0.1, 0.1, 21)], float)
        n = np.tile([0.0, 1.0, 0.0], (len(foot), 1))
        sole = sheet(-0.0, width=0.4)            # flat, 5 cm below the top of the foot: too far
        sole.pos = sole.pos[:, [0, 2, 1]]
        over = sheet(0.0, width=0.4)
        over.pos = over.pos[:, [0, 2, 1]] + [0.0, 0.06, 0.0]
        self.assertEqual(CV.instep(sole.pos, np.tile([0.0, -1.0, 0.0], (len(sole.pos), 1)), sole.F, foot, n), 0.0)
        self.assertGreater(CV.instep(over.pos, np.tile([0.0, 1.0, 0.0], (len(over.pos), 1)), over.F, foot, n), CV.FOOT_MIN)


class TestLids(unittest.TestCase):
    """The backing that closes a sleeve around the arm is recognised: closed fan, crossed by the axis."""

    def fan(self, center, n=10, open_=False):
        c = np.asarray(center, float)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        P = np.vstack([c, c + 0.5 * np.c_[np.zeros(n), np.cos(t), np.sin(t)]])
        F = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n - (1 if open_ else 0))])
        return P, F

    def test_alone_background_crosses_per_arm(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "clothes"))
        import wear as PO
        bones = [{"name": "左腕", "pos": [1, 0, 0]}, {"name": "左ひじ", "pos": [3, 0, 0]}]
        for center, open_, expected in (([2, 0, 0], False, True),      # around the arm
                                        ([2, 3, 0], False, False),     # beside the arm: ordinary fabric
                                        ([3.2, 0, 0], False, False),   # at the end of the bone: a visible end
                                        ([2, 0, 0], True, False)):     # open: not a backing
            P, F = self.fan(center, open_=open_)
            centers, faces, links = PO.lids(P, F, bones)
            self.assertEqual(len(links), int(expected))
            self.assertEqual(bool(centers[0]), expected, (center, open_))
            self.assertEqual(bool(faces.all()), expected)
            self.assertFalse(centers[1:].any())


class TestNeck(unittest.TestCase):
    def test_neck_of_game_removes_sphere_skull(self):
        import game_neck as CJ
        # Body (mat 0): a torso triangle (y=15..18.3) + a separate skull triangle (y=19.0..19.5)
        # Face (mat 1): a triangle at y=18.8
        pos = np.array([
            [0.0, 15.0, 0.0], [0.2, 18.3, 0.0], [-0.2, 18.3, 0.0],
            [0.0, 19.0, 0.0], [0.3, 19.5, 0.0], [-0.3, 19.5, 0.0],
            [0.0, 18.8, -0.3], [0.2, 19.2, -0.3], [-0.2, 19.2, -0.3],
        ])
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        mat_face = np.array([0, 0, 1])
        neck_head = np.array([[0.2 * np.cos(a), 18.25, 0.2 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 36)])
        remove, _, _, assessment = CJ.cut(pos, faces, mat_face, 0, [1], neck_head, (0.0, 0.0))
        self.assertEqual(assessment["faces_skull"], 1)
        self.assertEqual(remove.tolist(), [False, True, False])


def small_pmx():
    """Minimal humanoid skeleton, arms in A-pose, two triangles per useful bone."""
    names = {"下半身": (-1, (0, 10, 0)), "上半身": (0, (0, 11, 0)), "上半身1": (1, (0, 12, 0)), "上半身2": (2, (0, 13, 0)), "首": (3, (0, 15, 0)), "頭": (4, (0, 16, 0)),
            "左肩": (3, (0.5, 14.5, 0)), "左腕": (6, (1, 14.5, 0)), "左ひじ": (7, (3, 12.5, 0)), "左手首": (8, (5, 10.5, 0)), "左中指１": (9, (5.5, 10, 0)),
            "右肩": (3, (-0.5, 14.5, 0)), "右腕": (11, (-1, 14.5, 0)), "右ひじ": (12, (-3, 12.5, 0)), "右手首": (13, (-5, 10.5, 0)), "右中指１": (14, (-5.5, 10, 0)),
            "左足": (0, (1, 9, 0)), "左ひざ": (16, (1, 5, 0)), "左足首": (17, (1, 1, 0)), "右足": (0, (-1, 9, 0)), "右ひざ": (19, (-1, 5, 0)), "右足首": (20, (-1, 1, 0))}
    os_ = [{"name": n, "parent": p, "pos": tuple(float(x) for x in c)} for n, (p, c) in names.items()]
    pos, nor, uv, fs, links = [], [], [], [], []
    for k, o in enumerate(os_):
        i = len(pos)
        x, y, z = o["pos"]
        pos += [(x, y, z), (x + 0.1, y, z), (x, y + 0.1, z)]
        nor += [(0.0, 0.0, -1.0)] * 3
        uv += [(0.0, 0.0)] * 3
        fs.append((i, i + 2, i + 1))
        links += [(k, -1, -1, -1)] * 3
    weight = [(1.0, 0.0, 0.0, 0.0)] * len(pos)
    return {"bones": os_, "vertices": {"pos": pos, "nor": nor, "uv": uv, "bones": links, "weights": weight}, "faces": fs, "textures": [],
            "materials": [{"name": "skin", "n_indices": len(fs) * 3, "texture": -1, "diffuse": (1, 1, 1, 1), "flags": 1, "outline_size": 0.0, "outline_color": (0, 0, 0, 1)}],
            "morphs": [{"name": "まばたき", "type": 1, "offsets": [(0, (0.0, 0.1, 0.0))]}]}


class TestVrm(unittest.TestCase):
    def test_cutout_only_if__texture_a_of_holes(self):
        m = {"diffuse": [1, 1, 1, 1], "flags": 0, "outline_size": 0, "outline_color": [0, 0, 0, 1]}
        self.assertNotIn("alphaMode", VR._mtoon(m, 0, False, "hair", cutout=False))
        self.assertEqual(VR._mtoon(m, 0, False, "hair", cutout=True)["alphaMode"], "MASK")
        self.assertEqual(VR._mtoon(m, 0, False, "hair+", cutout=False)["alphaMode"], "BLEND")

    def test_written_one_vrm_valid(self):
        with tempfile.TemporaryDirectory() as d:
            ch = os.path.join(d, "x.vrm")
            r = VR.write_vrm(small_pmx(), ch, {})
            self.assertEqual(VR.validate(ch), [])
            doc, _ = VR.read(ch)
        self.assertEqual(r["expressions"], 1)
        self.assertIn("VRMC_vrm", doc["extensions"])
        self.assertTrue(all("VRMC_materials_mtoon" in m["extensions"] for m in doc["materials"]))

    def test_arm_a_l_horizontal(self):
        S = VR.Skeleton(small_pmx())
        Pw, _ = S.tpose()
        ep, co = Pw[S.human["leftUpperArm"]], Pw[S.human["leftHand"]]
        self.assertAlmostEqual(float(ep[1]), float(co[1]), places=5)
        self.assertGreater(float(co[0]), float(ep[0]))

    def test_spine_under_hips(self):
        S = VR.Skeleton(small_pmx())
        self.assertEqual(S.parent[S.human["spine"]], S.human["hips"])


class TestStacking(unittest.TestCase):
    def test_without_os_zero(self):
        P = {"vertices": {"bones": [(138, 5, 0, 0), (7, -1, -1, -1), (3, 4, 9, 1)],
                         "weights": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0)]}}
        self.assertEqual(C.without_os_zero(P), 2)
        self.assertEqual(P["vertices"]["bones"], [(138, 138, 138, 138), (7, -1, -1, -1), (3, 4, 3, 3)])

    def test_enveloping(self):
        self.assertTrue(C.enveloping({"tab": "bottom", "type": "skirt"}))
        self.assertTrue(C.enveloping({"tab": "accessories", "type": "belt"}))
        self.assertFalse(C.enveloping({"tab": "accessories", "type": "necklace"}))


class TestSettingsLayers(unittest.TestCase):
    """The layers come from the settings (layers.default.json, then settings.json)."""

    # The tables and thresholds as they were written in the code before moving to the settings (15/11).
    CLOTHES = {
        "underwear": ("bottom", "under", 10), "socks": ("shoes", "socks", 15), "top": ("top", "top", 20),
        "sleeves": ("top", "sleeves", 26), "shorts": ("bottom", "bottom", 30), "trousers": ("bottom", "bottom", 30),
        "skirt": ("bottom", "skirt", 32), "dress": ("bottom", "dress", 34), "jumpsuit": ("bottom", "dress", 34),
        "shoes": ("shoes", "shoes", 40), "boots": ("shoes", "shoes", 42),
        "gloves": ("accessories", "hands", 44), "collar": ("top", "collar", 48), "waistcoat": ("top", "waistcoat", 52),
        "jacket": ("top", "outer", 55), "coat": ("top", "outer", 60), "hood": ("top", "hood", 62)}
    OCCUPIES = {"dress": ["skirt"], "jumpsuit": ["bottom", "skirt"]}
    ACCESSORIES = {
        "belt": (70, True), "harness": (72, True), "garter": (66, False), "anklet": (67, False),
        "armband": (68, False), "bracelet": (69, False), "ring": (69, False), "tie": (75, True),
        "scarf": (76, True), "necklace": (78, True), "jewel": (78, False), "bag": (80, True),
        "glasses": (84, True), "mask": (84, True), "headwear": (86, True), "headphones": (86, True),
        "tail": (88, True), "wings": (88, True), "ears": (88, True)}
    FR_TYPE = {
        "underwear": "underwear", "socks": "socks", "top": "top", "sleeves": "sleeves", "shorts": "shorts",
        "trousers": "trousers", "skirt": "skirt", "dress": "dress", "jumpsuit": "jumpsuit", "shoes": "shoes",
        "boots": "boots", "gloves": "gloves", "collar": "collar", "waistcoat": "waistcoat", "jacket": "jacket", "coat": "coat",
        "hood": "hood", "garter": "garter", "anklet": "anklet", "armband": "armband",
        "bracelet": "bracelet", "ring": "ring", "tie": "tie", "scarf": "scarf", "necklace": "necklace",
        "jewel": "jewel", "bag": "bag", "glasses": "glasses", "mask": "mask", "headwear": "headwear",
        "headphones": "headphones", "tail": "tail", "wings": "wings", "ears": "ears",
        "belt": "belt", "harness": "harness"}
    THRESHOLDS = {
        "MARGIN": 0.003, "MARGIN_TUCKED": 0.0015, "REACH_TUCKED": 0.05, "FLOOR": 0.001, "REACH_RADIUS": 0.06,
        "REACH_SKIN": 0.06, "FREE_MAX": 0.025, "MATCH": 0.3, "END": 0.0002, "OUTPUT_MAX": 0.008, "RADIUS_LIMB": 0.05,
        "ARROW_MAX": 0.003, "REACH_MASK": 0.03, "TOLERANCE_MASK": 0.006, "PASSES": 4, "FINGERPRINT": 0.8,
        "CONTACT": 0.005, "REACH": 0.025, "THRESHOLD": 0.0005, "PART_CROSSES": 0.01, "ORDER_OUTSIDE": 0.003,
        "CELL": 0.08, "VOXEL": 0.04, "VOXEL_BODY": 0.012}

    def defaults(self):
        """The couches section of the public files only (without the machine's settings.json)."""
        import json
        with open(os.path.join(C.REPO, "layers.default.json"), encoding="utf-8") as f:
            return json.load(f)["layers"]

    def test_defaults_equal_a_front(self):
        V, O, A, Ce, F = C.tables(self.defaults())
        self.assertEqual(V, self.CLOTHES)
        self.assertEqual(O, self.OCCUPIES)
        self.assertEqual(A, self.ACCESSORIES)
        self.assertEqual(Ce, {"belt"})
        self.assertEqual(F, self.FR_TYPE)
        T = self.defaults()["outfit"]
        self.assertEqual((T["ranks_in_outfit"], T["closed_bottoms"], T["tops"], T["bottoms"]),
                         ({"belt": 36}, ["shorts", "trousers"], ["top"], ["shorts", "trousers", "skirt"]))     # settings: format 2

    def test_loads_equal_a_front(self):
        # what the code loaded (with no layer override in the machine's settings.json): exact equality
        if (C.R.all_().get("layers") or {}) != self.defaults():
            self.skipTest("the machine's settings.json overrides the layers")
        self.assertEqual((C.CLOTHES, C.OCCUPIES, C.ACCESSORIES, C.FR_TYPE), (self.CLOTHES, self.OCCUPIES, self.ACCESSORIES, self.FR_TYPE))
        for name, v in self.THRESHOLDS.items():
            self.assertEqual(getattr(CB, name), v, name)
            self.assertIs(type(getattr(CB, name)), type(v), name)
        self.assertEqual((CB.RANKS, CB.BOTTOM_CLOSED, CB.TOPS, CB.BOTTOM),
                         ({"belt": 36}, ("shorts", "trousers"), ("top", ), ("shorts", "trousers", "skirt")))

    def test_override_locale(self):
        """A settings.json that changes a rank, adds a type and a kind, removes a type and changes a threshold,
        without copying the tables: read by a fresh Python, like the server at start-up."""
        import json
        import subprocess
        local = {"layers": {
            "garments": {"skirt": {"rank": 33}, "hood": None,
                         "kimono": {"tab": "top", "slot": "outer", "rank": 57, "name": "kimono"}},
            "accessories": {"broche": {"rank": 79}, "corset": {"rank": 37, "exclusive": True, "belt_like": True}},
            "outfit": {"closed_bottoms": ["trousers"]},
            "thresholds": {"max_outward_mm": 10}}}
        code = ("import json, sys; sys.path.insert(0, %r); import common as C, outfit as CB; "
                "print(json.dumps({'skirt': C.CLOTHES['skirt'], 'jacket': C.CLOTHES['jacket'], 'kimono': C.CLOTHES.get('kimono'), "
                "'hood': 'hood' in C.CLOTHES, 'broche': C.ACCESSORIES.get('broche'), 'name_broche': C.FR_TYPE.get('broche'), "
                "'belts': sorted(C.BELTS), 'corset_env': C.enveloping({'tab': 'accessories', 'type': 'corset'}), "
                "'range': C.arrange({'class': 'garment', 'type': 'skirt', 'tab': 'bottom', 'slot': 'bottom', 'rank': 32}), "
                "'output': CB.OUTPUT_MAX, 'margin': CB.MARGIN_TUCKED, 'bottom_closed': CB.BOTTOM_CLOSED}))"
                % C.HERE)
        with tempfile.TemporaryDirectory() as d:
            ch = os.path.join(d, "settings.json")
            with open(ch, "w", encoding="utf-8") as f:
                json.dump(local, f)
            env = dict(os.environ, CHIMERA_SETTINGS=ch)
            r = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads(r.stdout)
        self.assertEqual(v["skirt"], ["bottom", "skirt", 33])          # rank changed, tab and slot kept
        self.assertEqual(v["jacket"], ["top", "outer", 55])     # the rest of the table is there
        self.assertEqual(v["kimono"], ["top", "outer", 57])    # type added
        self.assertFalse(v["hood"])                           # null removes a type
        self.assertEqual(v["broche"], [79, False])               # kind added, not exclusive by default
        self.assertEqual(v["name_broche"], "broche")              # no name: its own
        self.assertEqual(v["belts"], ["belt", "corset"])
        self.assertTrue(v["corset_env"])
        self.assertEqual(v["range"]["rank"], 33)                 # a record prepared earlier takes today's rank
        self.assertEqual(v["output"], 0.01)
        self.assertEqual(v["margin"], 0.0015)
        self.assertEqual(v["bottom_closed"], ["trousers"])


class TestTranslationMaterials(unittest.TestCase):
    """A game's material names translated into the common language by its description ("materials" table), at the single
    point where PMX files are read (atlas/pmx.lire)."""

    @staticmethod
    def pmx_minimal(names):
        """A PMX with one triangle per material, with the given names."""
        n = len(names)
        pos = [(float(i), float(j), 0.0) for i in range(n) for j in range(3)]
        return {"version": 2.0, "encoding": "UTF-16", "n_uv_add": 0, "name": "attempt", "name_en": "attempt",
                "vertices": {"pos": pos, "nor": [(0.0, 0.0, 1.0)] * len(pos), "uv": [(0.0, 0.0)] * len(pos),
                            "uv_add": [[]] * len(pos), "type": [0] * len(pos), "bones": [(0,)] * len(pos),
                            "weights": [(1.0,)] * len(pos), "contour": [1.0] * len(pos)},
                "faces": [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(n)], "textures": [],
                "materials": [{"name": x, "name_en": x, "diffuse": (1.0, 1.0, 1.0, 1.0), "specular": (0.0, 0.0, 0.0),
                               "brightness": 5.0, "ambient": (0.5, 0.5, 0.5), "flags": 0, "outline_color": (0.0, 0.0, 0.0, 1.0),
                               "outline_size": 0.0, "texture": -1, "sphere": -1, "sphere_mode": 0, "toon_sharing": 1,
                               "toon": 0, "memo": "", "n_indices": 3} for x in names],
                "bones": [{"name": "センター", "name_en": "center", "pos": (0.0, 0.0, 0.0), "parent": -1, "layer": 0,
                        "flags": 0x1E, "tail_bone": None, "tail_offset": (0.0, 1.0, 0.0)}],
                "morphs": [], "frames": [], "rigid_bodies": [], "joints": []}

    def test_zzz_identity(self):
        import games as games
        Z = games.load("zzz")
        self.assertEqual(Z.materials, [])
        for n in ("颜", "肌", "目", "髮2", "体", "外套+", "M_2000001_FACE", "Hair_2"):
            self.assertEqual(Z.translate(n), n)
        # a PMX from the ZZZ sources is tied to no table: pmx.lire returns it as is
        self.assertIsNone(games.game_of_file(os.path.join(Z.path("mmd_extracted"), "Hugo", "雨果.pmx")))

    def test_outerplane_roles_like_front(self):
        import games as games
        OP = games.load("outerplane")
        # what outerplane._role used to return hard-coded (prefix M_<id>_ removed, case ignored)
        expected = {"M_2000001_FACE": "颜", "M_2000001_FACE_EYE": "眉睫", "M_2000001_HAIR": "髮", "M_2000014_HAIR_CLEAR": "髮",
                   "M_2000001_BODY_02": "体", "face": "颜", "M_1_face": "颜", "M_2000001_WEAPON": None,
                   "M_2000001_FACE\n": None, "M__FACE": None, "M_2000001_FACEX": None, "Default-Material": None}
        for n, t in expected.items():
            self.assertEqual(OP.translate(n, otherwise=None), t, n)

    def test_game_fake_translated_a_reading(self):
        import games as games
        import pmx as PMX
        import pmx_write as pmx_write
        names = ["Skin", "Face", "Eye_L", "Hair_2", "Hair", "Coat", "肌"]
        with tempfile.TemporaryDirectory() as d:
            desc = os.path.join(d, "description")
            os.makedirs(desc)
            with open(os.path.join(desc, "game.json"), "w", encoding="utf-8") as f:
                f.write(r'''{"title": "Dummy game", "format": "pmx",
                    "sources": {"models": "%s"},
                    "_materials": "the first matching pattern wins",
                    "materials": [["(?i)^skin$", "肌"], ["(?i)^face$", "颜"], ["(?i)^eye_[lr]$", "目"],
                                  ["(?i)^hair_?(\\d*)$", "髮\\1"]]}''' % os.path.join(d, "models"))
            try:
                J = games.load("dummy-trial", folder=desc)
                self.assertEqual([J.translate(n) for n in names], ["肌", "颜", "目", "髮2", "髮", "Coat", "肌"])
                in_ = os.path.join(d, "models", "character", "character.pmx")
                out_of = os.path.join(d, "elsewhere", "character.pmx")
                for c in (in_, out_of):
                    pmx_write.write(self.pmx_minimal(names), c)
                self.assertIs(games.game_of_file(in_), J)
                self.assertIsNone(games.game_of_file(out_of))
                lu = PMX.read(in_)["materials"]
                self.assertEqual([m["name"] for m in lu], ["肌", "颜", "目", "髮2", "髮", "Coat", "肌"])
                self.assertEqual([m.get("name_game") for m in lu], ["Skin", "Face", "Eye_L", "Hair_2", "Hair", None, None])
                self.assertEqual([m["name_en"] for m in lu], names)       # the rest of the material is not touched
                self.assertEqual([m["name"] for m in PMX.read(out_of)["materials"]], names)   # outside its sources: as is
            finally:
                games._loads.pop("dummy-trial", None)


class TestStartAEmpty(unittest.TestCase):
    """New machine: no body, no catalogue. The server says what to do instead of opening a page that would fail."""
    EMPTY = {"heads": [], "hairstyles": [], "parts": [], "refusals": []}
    FULL = {"heads": [{"id": "t"}], "hairstyles": [{"id": "c"}], "parts": [], "refusals": []}

    def test_without_body(self):
        import catalog as K
        with tempfile.TemporaryDirectory() as d:
            m = K.a_make(self.EMPTY, body=os.path.join(d, "body.pmx"), cache=os.path.join(d, "cache"))
            self.assertEqual(len(m), 2)
            self.assertIn("workshop.body", m[0])
            self.assertIn("Empty catalogue", m[1])

    def test_body_step_prepare(self):
        import catalog as K
        with tempfile.TemporaryDirectory() as d:
            body = os.path.join(d, "body.pmx")
            open(body, "wb").close()
            m = K.a_make(self.FULL, body=body, cache=d)
            self.assertEqual(len(m), 1)
            self.assertIn("catalog.py prepare", m[0])

    def test_all_is(self):
        import catalog as K
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "body"))
            open(os.path.join(d, "body", "body.glb"), "wb").close()
            self.assertEqual(K.a_make(self.FULL, body=os.path.join(d, "absent.pmx"), cache=d), [])
            self.assertEqual(len(K.a_make(dict(self.FULL, hairstyles=[]), cache=d)), 1)

    def test_home_escapes(self):
        import server as S
        h = S.home(["<b>x</b> & y"]).decode("utf-8")
        self.assertIn("&lt;b&gt;x&lt;/b&gt; &amp; y", h)
        self.assertIn("INSTALLATION.md", h)


class TestKeepOfServer(unittest.TestCase):
    """The server launches Blender and writes into the cache: it answers only its own page."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import ThreadingHTTPServer
        import server as S
        cls.S, cls.before = S, S.CATALOGUE
        S.CATALOGUE = {"parts": [{"id": "A__top"}], "hairstyles": [{"id": "c"}], "heads": [{"id": "t"}]}
        cls.addClassCleanup(setattr, S, "CATALOGUE", cls.before)
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), S.Management)
        cls.addClassCleanup(cls.srv.server_close)     # cleanups run even if the class set-up fails later on
        cls.port = cls.srv.server_address[1]
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls.thread.join, 10)
        cls.addClassCleanup(cls.srv.shutdown)

    def request(self, method, path, body=None, **headers):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(method, path, body=body, headers=headers)
        r = c.getresponse()
        r.read()
        c.close()
        return r.status

    def test_other_site_refused(self):
        here = "http://127.0.0.1:%d" % self.port
        self.assertEqual(self.request("GET", "/api/formats", Host="evil.example:%d" % self.port), 403)     # domain name pointing at 127.0.0.1
        self.assertEqual(self.request("GET", "/api/formats", Origin="https://evil.example"), 403)
        self.assertEqual(self.request("POST", "/api/open", b"{}", Origin="https://evil.example", **{"Content-Type": "application/json"}), 403)
        self.assertEqual(self.request("PUT", "/api/thumbnail/parts/A__top", b"x", Origin="https://evil.example"), 403)
        self.assertNotEqual(self.request("GET", "/api/catalog", Origin=here), 403)

    def test_post_without_json_declared_refused(self):
        # a form from another site is sent without a permission request: it is not read
        self.assertEqual(self.request("POST", "/api/assemble", b'{"parts": []}', **{"Content-Type": "text/plain"}), 415)

    def test_ids_outside_catalogue_refused(self):
        j = {"Content-Type": "application/json"}
        for body in (b'{"parts": ["../../etc"]}', b'{"parts": ["A__top"], "hairstyle": "../x"}',
                      b'{"parts": ["A__top"], "head": "../x"}', b'{"parts": "A__top"}', b'[]', b'{'):
            self.assertEqual(self.request("POST", "/api/combination", body, **j), 400, body)
        self.assertEqual(self.request("POST", "/api/assemble", b'{"parts": [], "tint": 3}', **j), 400)
        self.assertEqual(self.request("POST", "/api/open", b'{"id": "19990101-000000"}', **j), 400)

    def test_folders_d_assembly_distinct(self):
        S = self.S
        with tempfile.TemporaryDirectory() as d:
            import types
            before, module = S.ASSEMBLIES, S.threading
            S.ASSEMBLIES = d

            class Nothing:
                def __init__(self, *a, **k):
                    pass

                def start(self):
                    pass
            # only the server module sees the fake Thread: the test server keeps its own threads
            S.threading = types.SimpleNamespace(Thread=Nothing, Lock=module.Lock)
            try:
                ids = {S.assemble({"parts": []}) for _ in range(3)}
            finally:
                S.ASSEMBLIES, S.threading = before, module
            self.assertEqual(len(ids), 3)

    def test_launch_impossible_finishes_in_error(self):
        """Any failure of the job (here the launcher itself) ends in an error state the page reads."""
        import time
        S = self.S
        with tempfile.TemporaryDirectory() as d:
            before, launcher = S.ASSEMBLIES, S.C.launcher
            S.ASSEMBLIES = d

            def failure():
                raise OSError("no launcher")
            S.C.launcher = failure
            try:
                ident = S.assemble({"parts": []})
                for _ in range(100):
                    e = C.read_json(os.path.join(d, ident, "state.json"), {})
                    if e.get("done"):
                        break
                    time.sleep(0.05)
            finally:
                S.ASSEMBLIES, S.C.launcher = before, launcher
            self.assertEqual((e.get("done"), e.get("ok"), e.get("step")), (True, False, "error"))
            self.assertIn("no launcher", e.get("error", ""))
            self.assertNotIn(ident, S._actives)

    def test_work_orphan_read_in_error(self):
        """A job left unfinished by an earlier server is read as interrupted, not as running forever."""
        import http.client
        import json
        S = self.S
        with tempfile.TemporaryDirectory() as d:
            before = S.ASSEMBLIES
            S.ASSEMBLIES = d
            os.makedirs(os.path.join(d, "20200101-000000"))
            C.write_json(os.path.join(d, "20200101-000000", "state.json"), {"step": "graft", "done": False})
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
                c.request("GET", "/api/jobs/20200101-000000")
                e = json.loads(c.getresponse().read())
                c.close()
            finally:
                S.ASSEMBLIES = before
        self.assertEqual((e["done"], e["ok"], e["step"]), (True, False, "error"))
        self.assertIn("interrupted", e["error"])


class TestFailureAssembly(unittest.TestCase):
    """assemble_blender.py ends every failure in state.json (step error, finished, reason), SystemExit included:
    head_graft refuses that way, and the page would otherwise wait on its last step."""

    def launch(self, d):
        import runpy
        argv = sys.argv
        sys.argv = ["blender", "--", "--dossier", d]
        try:
            with self.assertRaises(SystemExit):
                runpy.run_path(os.path.join(C.HERE, "assemble_blender.py"), run_name="__main__")
        finally:
            sys.argv = argv
        return C.read_json(os.path.join(d, "state.json"))

    def test_part_unknown_refused_front_all_work(self):
        with tempfile.TemporaryDirectory() as d:
            C.write_json(os.path.join(d, "selection.json"), {"head": "t", "hairstyle": "c", "parts": [""], "tint": None})
            e = self.launch(d)
        self.assertEqual((e["done"], e["ok"], e["step"]), (True, False, "error"))
        self.assertIn("unknown pieces", e["error"])

    def test_refusals_per_systemexit_readable(self):
        read = C.read_json

        def refusals(path, *a, **k):
            if path.endswith("selection.json"):
                raise SystemExit("neck too different: test")
            return read(path, *a, **k)
        C.read_json = refusals
        try:
            with tempfile.TemporaryDirectory() as d:
                e = self.launch(d)
        finally:
            C.read_json = read
        self.assertEqual((e["done"], e["ok"], e["step"], e["error"]), (True, False, "error", "neck too different: test"))


class TestGapSkeleton(unittest.TestCase):
    """head_graft measures the skeleton under the grafted vertices: hair on 頭 ignores a longer body neck,
    neck skin does not."""

    def test_hair_on_head_and_skin_of_neck(self):
        import head_graft as GT
        os_ = lambda name, pos, parent: {"name": name, "pos": pos, "parent": parent}  # noqa: E731
        H = [os_("首", (0.0, 9.2, 0.0), -1), os_("頭", (0.0, 10.0, 0.0), 0), os_("髪1", (0.0, 10.5, -0.3), 1)]
        B = [os_("首", (0.0, 11.0, 0.0), -1), os_("頭", (0.0, 12.0, 0.0), 0)]   # neck 1.0 long instead of 0.8
        t = np.array(B[1]["pos"]) - np.array(H[1]["pos"])
        idx = {o["name"]: i for i, o in enumerate(B)}
        os_v = np.array([[2, -1, -1, -1], [1, 2, -1, -1], [0, -1, -1, -1], [0, 1, -1, -1]])
        w = np.array([[1, 0, 0, 0], [0.5, 0.5, 0, 0], [1, 0, 0, 0], [0.5, 0.5, 0, 0]], float)
        offset = GT.offsets_os(B, H, idx, t)
        np.testing.assert_allclose(offset[2], 0.0, atol=1e-9)       # hair bone: offset of 頭, its nearest shared ancestor
        e = GT.gap_skeleton(offset, os_v, w)
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), [0.0, 0.0, 0.2, 0.1], atol=1e-9)
        np.testing.assert_allclose(e[2], [0.0, -0.2, 0.0], atol=1e-9)   # neck skin follows the body's lower 首


class TestTrackPerModel(unittest.TestCase):
    """catalog.Track: during an extraction the suite goes model by model, and each model is announced usable
    (« finished ») as soon as its last item is done, the shared work first."""

    def test_order_and_end_per_model(self):
        import contextlib
        import io
        import json
        import catalog as K
        s = K.Track(["Belle", "Anby"])
        suite = [{"kind": "part", "id": C.the_key("Anby", "boots")}, {"kind": "hairstyle", "id": "Belle"},
                 {"kind": "part", "id": C.the_key("Belle", "gloves")}, {"kind": "hairstyle", "id": "Anby"},
                 {"kind": "hairstyle_free", "id": "Libre"}, {"kind": "hairstyle", "id": "Jane"}]   # fr-ok: "Libre" is the id of the free head and hairstyle, a folder name in installed caches
        s.sort(suite)
        self.assertEqual([it["id"] for it in suite][:2], ["Libre", "Jane"])   # fr-ok: "Libre" is the id of the free head and hairstyle, a folder name in installed caches; shared work first
        self.assertEqual([it["id"] for it in suite][2:], ["Belle", C.the_key("Belle", "gloves"), "Anby", C.the_key("Anby", "boots")])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            s.start()
            s.lot_done(suite[:3])
            s.lot_done(suite[3:5])
        lines = [json.loads(l[3:]) for l in output.getvalue().splitlines()]
        finished = [l["model_done"] for l in lines if "model_done" in l]
        self.assertEqual(finished, ["Belle"])
        self.assertEqual([l for l in lines if "model" in l][-1], {"model": "Anby", "stage": "parts"})
        with contextlib.redirect_stdout(output):
            s.lot_done(suite[5:])
        self.assertIn('"model_done": "Anby"', output.getvalue())


class TestPushOfHair(unittest.TestCase):
    def test_layers_keep_their_order(self):
        """Two hair layers 3 mm apart, the inner one pushed by 10 mm: both move, the gap stays."""
        import hairstyles as CF
        a, b = sheet(0.0, n=10), sheet(0.003, n=10)
        P = np.r_[a.pos, b.pos]
        u = np.zeros_like(P)
        u[45] = [0, 0, 0.01]
        d = CF.field_of_push(P, u, 0.02)
        self.assertAlmostEqual(d[45, 2], 0.01)
        self.assertAlmostEqual(d[100 + 45, 2], 0.01 * (1 - (0.003 / 0.02) ** 2) ** 2)
        self.assertTrue((d[100:, 2] >= d[:100, 2] - 1e-3).all())
        self.assertEqual(d[0].tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()

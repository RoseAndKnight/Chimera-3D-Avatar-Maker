"""The VRoid importer on a fake VRM file, built here: no model is needed.

    <python> -I tools/vroid/tests/test_importer.py

The fake file has a VRM 0.x humanoid skeleton in T-pose, a chest bone, a tube of skin from the
pelvis to the head, a face triangle with a shape (Fcl_EYE_Close), a triangle held by the hand, one
by the chest bone and a top. Its VRM 1.0 twin is the same character written as VRoid Hub writes it: bones
with a rotation, inverse bind matrices, the mesh on a node of its own (as UniVRM re-exports), the thumb named
one segment down, the shape split over a sparse accessor.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.dirname(HERE), TOOLS, os.path.join(TOOLS, "atlas")]
import importer as I          # noqa: E402
import pmx as PMX             # noqa: E402

# (name, parent, world position in metres, humanoid role)
OS = [("Root", -1, (0, 0, 0), None), ("J_Bip_C_Hips", 0, (0, 1.0, 0), "hips"), ("J_Bip_C_Spine", 1, (0, 1.1, 0), "spine"),
      ("J_Bip_C_Chest", 2, (0, 1.2, 0), "chest"), ("J_Bip_C_Neck", 3, (0, 1.4, 0), "neck"), ("J_Bip_C_Head", 4, (0, 1.5, 0), "head"),
      ("J_Sec_L_Bust1", 3, (0.05, 1.25, 0.05), None),
      ("J_Bip_L_UpperArm", 3, (0.15, 1.35, 0), "leftUpperArm"), ("J_Bip_L_LowerArm", 7, (0.4, 1.35, 0), "leftLowerArm"),
      ("J_Bip_L_Hand", 8, (0.6, 1.35, 0), "leftHand"),
      ("J_Bip_R_UpperArm", 3, (-0.15, 1.35, 0), "rightUpperArm"), ("J_Bip_R_LowerArm", 10, (-0.4, 1.35, 0), "rightLowerArm"),
      ("J_Bip_R_Hand", 11, (-0.6, 1.35, 0), "rightHand"),
      ("J_Bip_L_UpperLeg", 1, (0.08, 0.9, 0), "leftUpperLeg"), ("J_Bip_L_Foot", 13, (0.08, 0.05, 0), "leftFoot"),
      ("J_Bip_L_ToeBase", 14, (0.08, 0.0, 0.1), "leftToes"),
      ("J_Bip_L_Thumb1", 9, (0.63, 1.34, 0.02), "leftThumbProximal")]
OFFSET = (0.0, 1.2, 0.05)                 # VRM 1.0 twin: the mesh lies on a node placed here
RANK = {n: i for i, (n, _, _, _) in enumerate(OS)}
LOOPS = (1.0, 1.2, 1.35, 1.45, 1.5)      # heights of the skin tube; the neck cut falls at 1.43
SIDES = 6


def quaternion(axis, deg):
    a = np.radians(deg) / 2
    return np.concatenate([np.sin(a) * np.asarray(axis, float) / np.linalg.norm(axis), [np.cos(a)]])


def vrm_fake(path, version=0, faces_vroid=True):
    """version 0: VRoid Studio's VRM 0.x; 1: the same character in VRM 1.0; None: a glTF without VRM.
    faces_vroid False: the materials lose VRoid Studio's names (a VRM made with another tool)."""
    buffer, views, accs = bytearray(), [], []

    def accessory(a, comp, typ):
        a = np.ascontiguousarray(a)
        while len(buffer) % 4:
            buffer.append(0)
        views.append({"buffer": 0, "byteOffset": len(buffer), "byteLength": a.nbytes})
        buffer.extend(a.tobytes())
        accs.append({"bufferView": len(views) - 1, "componentType": comp, "count": len(a), "type": typ})
        return len(accs) - 1

    def accessory_hollow(base, values):       # sparse accessor: zeros except at the listed vertices
        values = np.asarray(values, "<f4")
        idx = np.where(np.abs(values).max(1) > 0)[0].astype("<u4")
        k = accessory(idx, 5125, "SCALAR")
        v = accessory(values[idx], 5126, "VEC3")
        accs.append({"componentType": 5126, "count": len(values), "type": "VEC3",
                     "sparse": {"count": len(idx), "indices": {"bufferView": accs[k]["bufferView"], "componentType": 5125},
                                "values": {"bufferView": accs[v]["bufferView"]}}})
        return len(accs) - 1

    def primitive(pos, faces, os_, material, shape=None):
        pos = np.asarray(pos, float) - (OFFSET if version == 1 else 0.0)
        pos = np.asarray(pos, "<f4")
        j = np.zeros((len(pos), 4), "<u2")
        j[:, 0] = os_
        w = np.tile(np.array([1, 0, 0, 0], "<f4"), (len(pos), 1))
        p = {"attributes": {"POSITION": accessory(pos, 5126, "VEC3"), "JOINTS_0": accessory(j, 5123, "VEC4"), "WEIGHTS_0": accessory(w, 5126, "VEC4")},
             "indices": accessory(np.asarray(faces, "<u4").ravel(), 5125, "SCALAR"), "material": material}
        if shape is not None and version == 1:
            p["targets"] = [{"POSITION": accessory_hollow(None, shape)}]
            p["extras"] = {"targetNames": ["Fcl_EYE_Close"]}
        elif shape is not None:
            p["targets"] = [{"POSITION": accessory(np.asarray(shape, "<f4"), 5126, "VEC3")}]
            p["extras"] = {"targetNames": ["Face.M_F00_000_00_Fcl_EYE_Close"]}
        return p
    # skin tube: each ring follows the closest bone below
    a = np.arange(SIDES) * 2 * np.pi / SIDES
    tube = np.array([(0.05 * np.cos(t), y, 0.05 * np.sin(t)) for y in LOOPS for t in a])
    os_tube = np.repeat([RANK["J_Bip_C_Hips"], RANK["J_Bip_C_Chest"], RANK["J_Bip_C_Chest"], RANK["J_Bip_C_Neck"], RANK["J_Bip_C_Head"]], SIDES)
    f_tube = []
    for k in range(len(LOOPS) - 1):
        for i in range(SIDES):
            p, q = k * SIDES + i, k * SIDES + (i + 1) % SIDES
            f_tube += [(p, q, p + SIDES), (q, q + SIDES, p + SIDES)]
    tri = lambda c: [(c[0], c[1], c[2]), (c[0] + 0.02, c[1], c[2]), (c[0], c[1] + 0.02, c[2])]     # noqa: E731
    prims = [primitive(tube, f_tube, os_tube, 0),
             primitive(tri((0, 1.55, 0.06)), [(0, 1, 2)], RANK["J_Bip_C_Head"], 1, shape=[(0, -0.01, 0), (0, 0, 0), (0, 0, 0)]),
             primitive(tri((0.6, 1.35, 0)), [(0, 1, 2)], RANK["J_Bip_L_Hand"], 0),
             primitive(tri((0.05, 1.25, 0.08)), [(0, 1, 2)], RANK["J_Sec_L_Bust1"], 0),
             primitive(tri((0, 1.25, 0.07)), [(0, 1, 2)], RANK["J_Bip_C_Chest"], 2)]
    # each bone turned by its own rotation in VRM 1.0 (the world positions stay those of OS)
    Q = [quaternion((1, 0.3 * (i % 3), 0.2), 7.0 * (i + 1)) if version == 1 else np.array([0, 0, 0, 1.0]) for i in range(len(OS))]
    nodes, W = [], []
    for i, (name, parent, pos, _) in enumerate(OS):
        Rw = I.matrix({"rotation": Q[i].tolist()})[:3, :3]
        Rp = W[parent][:3, :3] if parent >= 0 else np.eye(3)
        base = OS[parent][2] if parent >= 0 else (0, 0, 0)
        n = {"name": name, "translation": (Rp.T @ (np.subtract(pos, base))).tolist()}
        if version == 1:
            q = I.matrix({"rotation": (Q[parent] if parent >= 0 else [0, 0, 0, 1])})[:3, :3].T @ Rw
            n["rotation"] = rotation_in_quaternion(q).tolist()
        children = [k for k, o in enumerate(OS) if o[1] == i]
        if children:
            n["children"] = children
        nodes.append(n)
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = Rw, pos
        W.append(M)
    stitch = np.eye(4)
    if version == 1:
        stitch[:3, 3] = OFFSET
    ibm = np.array([np.linalg.inv(w) @ stitch for w in W])
    skin = {"joints": list(range(len(OS))), "inverseBindMatrices": accessory(ibm.transpose(0, 2, 1).reshape(-1, 16).astype("<f4"), 5126, "MAT4")}
    nodes.append({"name": "Body", "mesh": 0, "skin": 0} | ({"translation": list(OFFSET)} if version == 1 else {}))
    mats = ["N00_000_00_Body_00_SKIN", "N00_000_00_Face_00_SKIN", "N00_001_01_Tops_01_CLOTH"]
    if version == 1:
        mats = [m + " (Instance)" for m in mats]
    if not faces_vroid:
        mats = ["Body_SKIN", "Face_SKIN", "Tops_CLOTH"]
    if version == 0:
        ext = {"VRM": {"meta": {"title": "fake", "author": "test", "licenseName": "CC0"},
                       "humanoid": {"humanBones": [{"bone": r, "node": i} for i, (_, _, _, r) in enumerate(OS) if r]}}}
    elif version == 1:
        un = {"leftThumbProximal": "leftThumbMetacarpal"}
        ext = {"VRMC_vrm": {"specVersion": "1.0",
                            "meta": {"name": "fake", "authors": ["test"], "licenseUrl": "https://vrm.dev/licenses/1.0/",
                                     "commercialUsage": "personalNonProfit", "modification": "prohibited",
                                     "allowRedistribution": False, "creditNotation": "required", "avatarPermission": "onlyAuthor"},
                            "humanoid": {"humanBones": {un.get(r, r): {"node": i} for i, (_, _, _, r) in enumerate(OS) if r}}}}
    else:
        ext = {}
    j = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(buffer)}], "bufferViews": views, "accessors": accs,
         "nodes": nodes, "scenes": [{"nodes": [0, len(OS)]}], "skins": [skin],
         "meshes": [{"name": "Body", "primitives": prims}],
         "materials": [{"name": m} for m in mats], "extensions": ext}
    text = json.dumps(j).encode()
    text += b" " * (-len(text) % 4)
    while len(buffer) % 4:
        buffer.append(0)
    with open(path, "wb") as f:
        f.write(b"glTF" + struct.pack("<II", 2, 28 + len(text) + len(buffer)))
        f.write(struct.pack("<I", len(text)) + b"JSON" + text)
        f.write(struct.pack("<I", len(buffer)) + b"BIN\0" + bytes(buffer))


def rotation_in_quaternion(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w])


class Ingester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.vrm = os.path.join(cls.tmp.name, "Dummy.vrm")
        vrm_fake(cls.vrm)
        cls.assessment = I.convert(cls.vrm, os.path.join(cls.tmp.name, "mmd", "Dummy"), log=lambda *_: None)
        cls.R = PMX.read(cls.assessment["pmx"])
        cls.T = PMX.to_arrays(cls.R)
        cls.os = [o["name"] for o in cls.R["bones"]]
        cls.mats = [m["name"] for m in cls.R["materials"]]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def faces(self, name):
        k = self.mats.index(name)
        return self.T["faces"][self.T["mat_face"] == k]

    def test_os_to_names_mmd(self):
        for name in ("全ての親", "センター", "下半身", "上半身", "上半身1", "首", "頭", "左腕", "左ひじ", "左手首", "右腕", "左足首"):
            self.assertIn(name, self.os)
        parent = lambda n: self.os[self.R["bones"][self.os.index(n)]["parent"]]     # noqa: E731
        self.assertEqual(parent("上半身"), "センター")
        self.assertEqual(parent("下半身"), "センター")

    def test_os_of_chest_blended_in_son_parent(self):
        self.assertFalse([n for n in self.os if "Bust" in n])
        # the triangle the chest bone held (skin, at 1.25 m in front) is held by 上半身1
        P, held = self.T["pos"], self.T["bones"][:, 0]
        v = np.where((np.abs(P[:, 1] - 1.25 * I.SAMP) < 0.3) & (np.abs(P[:, 2]) > 0.9) & (P[:, 0] > 0.3))[0]
        self.assertTrue(len(v))
        self.assertEqual({self.os[k] for k in held[v]}, {"上半身1"})

    def test_arm_drops(self):
        o = {n: np.array(self.R["bones"][self.os.index(n)]["pos"]) for n in ("左腕", "左手首", "右腕", "右手首")}
        for side in "左右":
            d = o[side + "手首"] - o[side + "腕"]
            angle = np.degrees(np.arctan2(-d[1], abs(d[0])))
            self.assertAlmostEqual(angle, I.description()["pose"]["arm"], places=3)
        self.assertGreater(o["左手首"][0], 0)      # the character's left at +x
        # the triangle held by the hand has followed
        P = self.T["pos"][self.T["bones"][:, 0] == self.os.index("左手首")]
        self.assertLess(np.abs(P - o["左手首"]).max(), 0.5)

    def test_landmark_and_size(self):
        P = self.T["pos"]
        self.assertAlmostEqual(P[:, 1].max(), 1.57 * I.SAMP, places=3)
        self.assertLess(P[self.faces("颜").ravel(), 2].max(), 0)     # the face, in front: -z

    def test_materials_translated(self):
        self.assertEqual(self.mats, ["颜", "肌", "上衣"])
        self.assertEqual(self.R["materials"][1]["name_en"], "N00_000_00_Body_00_SKIN")

    def test_skin_cut_in_bottom_of_neck(self):
        D = I.description()
        y0 = (1.4 + D["pose"]["neck"] * 0.1) * I.SAMP
        F, P = self.faces("肌"), self.T["pos"]
        top = (P[F][:, :, 1] >= y0 - 1e-4).all(1)
        self.assertEqual(int(top.sum()), 2 * SIDES)           # the only band above the cut
        # no face below shares a vertex with the neck skin, neither by index nor by position
        self.assertFalse(set(F[top].ravel()) & set(F[~top].ravel()))
        the_key = lambda v: {tuple(x) for x in np.round(P[np.unique(v)], 4).tolist()}     # noqa: E731
        self.assertFalse(the_key(F[top]) & the_key(F[~top]))
        # neck edge flat, at the height of the cut; the body's just below
        bottom_of_neck = P[np.unique(F[top])][:, 1].min()
        self.assertAlmostEqual(bottom_of_neck, y0, places=4)
        under = P[np.unique(F[~top])][:, 1].max()
        self.assertAlmostEqual(under, y0 - I.CUT, places=4)

    def test_shape_of_face_in_morph_mmd(self):
        m = {x["name"]: x for x in self.R["morphs"]}
        self.assertIn("まばたき", m)
        self.assertEqual(m["まばたき"]["name_en"], "Fcl_EYE_Close")
        self.assertEqual(len(m["まばたき"]["offsets"]), 1)
        self.assertAlmostEqual(m["まばたき"]["offsets"][0][1][1], -0.01 * I.SAMP, places=4)

    def test_faces_erased_per_texture(self):
        a = np.zeros((8, 8), np.float32)
        a[:4] = 1.0                                             # top half drawn
        uv = np.array([[0.1, 0.1], [0.4, 0.1], [0.1, 0.4],      # at the very top: visible
                       [0.1, 0.6], [0.4, 0.6], [0.1, 0.9],      # at the very bottom: erased
                       [0.1, 0.3], [0.4, 0.3], [0.1, 0.9]])     # straddling: visible
        F = np.arange(9).reshape(3, 3)
        self.assertEqual(I.invisible_ones(a, uv, F).tolist(), [False, True, False])

    def test_vrm_1_gives_same_person(self):
        # the same character written as VRM 1.0 (rotated bones, mesh on its own node, thumb renamed, sparse shape):
        # same bones, same vertices, same morph as from the VRM 0.x
        ch = os.path.join(self.tmp.name, "Un.vrm")
        vrm_fake(ch, version=1)
        assessment = I.convert(ch, os.path.join(self.tmp.name, "mmd", "Un"), log=lambda *_: None)
        R = PMX.read(assessment["pmx"])
        T = PMX.to_arrays(R)
        self.assertEqual(assessment["vrm"], "1.0")
        self.assertEqual([o["name"] for o in R["bones"]], self.os)
        self.assertIn("左親指０", self.os)        # 1.0 thumbMetacarpal = 0.x thumbProximal
        for a, b in zip(R["bones"], self.R["bones"]):
            self.assertLess(np.abs(np.subtract(a["pos"], b["pos"])).max(), 1e-3, a["name"])
        self.assertLess(np.abs(T["pos"] - self.T["pos"]).max(), 1e-3)
        self.assertLess(np.abs(T["nor"] - self.T["nor"]).max(), 1e-3)
        self.assertEqual([m["name"] for m in R["materials"]], self.mats)
        m = {x["name"]: x for x in R["morphs"]}["まばたき"]["offsets"]
        self.assertEqual(len(m), 1)
        self.assertAlmostEqual(m[0][1][1], -0.01 * I.SAMP, places=3)
        # the licence of the 1.0 meta is reported as the 0.x one is
        self.assertIn("vrm.dev/licenses/1.0", R["comment"])
        self.assertIn("modification: prohibited", R["comment"])
        self.assertIn("Declared licence: CC0", self.R["comment"])

    def test_vrm_which_n_is_step_vroid_refused(self):
        for version in (0, 1):
            ch = os.path.join(self.tmp.name, "Other%d.vrm" % version)
            vrm_fake(ch, version=version, faces_vroid=False)
            with self.assertRaisesRegex(I.Refusals, "not a VRoid character"):
                I.Model(ch)
            r = subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(HERE), "importer.py"), ch, "--output",
                                os.path.join(self.tmp.name, "refusals")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertTrue(r.stderr.strip().startswith("refused: the file is not a VRoid character"), r.stderr)
        ch = os.path.join(self.tmp.name, "Empty.vrm")
        vrm_fake(ch, version=None)
        with self.assertRaisesRegex(I.Refusals, "no VRM extension"):
            I.Model(ch)

    def test_family_whose_each_brings_its_models(self):
        # models "*": the <Name>/<Name>.pmx folders of mmd_extracted; a value of the description is overridden
        # by the setting games.<game>.<key>
        settings_file = os.path.join(self.tmp.name, "settings.json")
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump({"games": {"active": "vroid", "vroid": {"mmd_extracted": os.path.join(self.tmp.name, "mmd"),
                                                           "base_head": {"name": "Dummy"}}}}, f)
        code = ("import sys, json; sys.path.insert(0, %r); import games as games; J = games.active(); "
                "print(json.dumps([J.name, J.models(), J.value('base_head.name'), J.value('base_head.tint')]))" % TOOLS)
        r = subprocess.run([sys.executable, "-I", "-c", code], env=dict(os.environ, CHIMERA_SETTINGS=settings_file),
                           capture_output=True, text=True, check=True)
        name, models, head, tint = json.loads(r.stdout)
        self.assertEqual(name, "vroid")
        self.assertEqual(models, {"Dummy": os.path.join("Dummy", "Dummy.pmx")})
        self.assertEqual(head, "Dummy")
        self.assertEqual(tint, "face")


if __name__ == "__main__":
    unittest.main()

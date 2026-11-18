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

ICI = os.path.dirname(os.path.abspath(__file__))
OUTILS = os.path.dirname(os.path.dirname(ICI))
sys.path[:0] = [os.path.dirname(ICI), OUTILS, os.path.join(OUTILS, "atlas")]
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
DECALAGE = (0.0, 1.2, 0.05)                 # VRM 1.0 twin: the mesh lies on a node placed here
RANG = {n: i for i, (n, _, _, _) in enumerate(OS)}
ANNEAUX = (1.0, 1.2, 1.35, 1.45, 1.5)      # heights of the skin tube; the neck cut falls at 1.43
COTES = 6


def quaternion(axe, deg):
    a = np.radians(deg) / 2
    return np.concatenate([np.sin(a) * np.asarray(axe, float) / np.linalg.norm(axe), [np.cos(a)]])


def vrm_factice(chemin, version=0, faces_vroid=True):
    """version 0: VRoid Studio's VRM 0.x; 1: the same character in VRM 1.0; None: a glTF without VRM.
    faces_vroid False: the materials lose VRoid Studio's names (a VRM made with another tool)."""
    tampon, vues, accs = bytearray(), [], []

    def acc(a, comp, typ):
        a = np.ascontiguousarray(a)
        while len(tampon) % 4:
            tampon.append(0)
        vues.append({"buffer": 0, "byteOffset": len(tampon), "byteLength": a.nbytes})
        tampon.extend(a.tobytes())
        accs.append({"bufferView": len(vues) - 1, "componentType": comp, "count": len(a), "type": typ})
        return len(accs) - 1

    def acc_creux(base, valeurs):       # sparse accessor: zeros except at the listed vertices
        valeurs = np.asarray(valeurs, "<f4")
        ou = np.where(np.abs(valeurs).max(1) > 0)[0].astype("<u4")
        k = acc(ou, 5125, "SCALAR")
        v = acc(valeurs[ou], 5126, "VEC3")
        accs.append({"componentType": 5126, "count": len(valeurs), "type": "VEC3",
                     "sparse": {"count": len(ou), "indices": {"bufferView": accs[k]["bufferView"], "componentType": 5125},
                                "values": {"bufferView": accs[v]["bufferView"]}}})
        return len(accs) - 1

    def primitive(pos, faces, os_, materiau, forme=None):
        pos = np.asarray(pos, float) - (DECALAGE if version == 1 else 0.0)
        pos = np.asarray(pos, "<f4")
        j = np.zeros((len(pos), 4), "<u2")
        j[:, 0] = os_
        w = np.tile(np.array([1, 0, 0, 0], "<f4"), (len(pos), 1))
        p = {"attributes": {"POSITION": acc(pos, 5126, "VEC3"), "JOINTS_0": acc(j, 5123, "VEC4"), "WEIGHTS_0": acc(w, 5126, "VEC4")},
             "indices": acc(np.asarray(faces, "<u4").ravel(), 5125, "SCALAR"), "material": materiau}
        if forme is not None and version == 1:
            p["targets"] = [{"POSITION": acc_creux(None, forme)}]
            p["extras"] = {"targetNames": ["Fcl_EYE_Close"]}
        elif forme is not None:
            p["targets"] = [{"POSITION": acc(np.asarray(forme, "<f4"), 5126, "VEC3")}]
            p["extras"] = {"targetNames": ["Face.M_F00_000_00_Fcl_EYE_Close"]}
        return p
    # skin tube: each ring follows the closest bone below
    a = np.arange(COTES) * 2 * np.pi / COTES
    tube = np.array([(0.05 * np.cos(t), y, 0.05 * np.sin(t)) for y in ANNEAUX for t in a])
    os_tube = np.repeat([RANG["J_Bip_C_Hips"], RANG["J_Bip_C_Chest"], RANG["J_Bip_C_Chest"], RANG["J_Bip_C_Neck"], RANG["J_Bip_C_Head"]], COTES)
    f_tube = []
    for k in range(len(ANNEAUX) - 1):
        for i in range(COTES):
            p, q = k * COTES + i, k * COTES + (i + 1) % COTES
            f_tube += [(p, q, p + COTES), (q, q + COTES, p + COTES)]
    tri = lambda c: [(c[0], c[1], c[2]), (c[0] + 0.02, c[1], c[2]), (c[0], c[1] + 0.02, c[2])]     # noqa: E731
    prims = [primitive(tube, f_tube, os_tube, 0),
             primitive(tri((0, 1.55, 0.06)), [(0, 1, 2)], RANG["J_Bip_C_Head"], 1, forme=[(0, -0.01, 0), (0, 0, 0), (0, 0, 0)]),
             primitive(tri((0.6, 1.35, 0)), [(0, 1, 2)], RANG["J_Bip_L_Hand"], 0),
             primitive(tri((0.05, 1.25, 0.08)), [(0, 1, 2)], RANG["J_Sec_L_Bust1"], 0),
             primitive(tri((0, 1.25, 0.07)), [(0, 1, 2)], RANG["J_Bip_C_Chest"], 2)]
    # each bone turned by its own rotation in VRM 1.0 (the world positions stay those of OS)
    Q = [quaternion((1, 0.3 * (i % 3), 0.2), 7.0 * (i + 1)) if version == 1 else np.array([0, 0, 0, 1.0]) for i in range(len(OS))]
    noeuds, W = [], []
    for i, (nom, parent, pos, _) in enumerate(OS):
        Rw = I.matrice({"rotation": Q[i].tolist()})[:3, :3]
        Rp = W[parent][:3, :3] if parent >= 0 else np.eye(3)
        base = OS[parent][2] if parent >= 0 else (0, 0, 0)
        n = {"name": nom, "translation": (Rp.T @ (np.subtract(pos, base))).tolist()}
        if version == 1:
            q = I.matrice({"rotation": (Q[parent] if parent >= 0 else [0, 0, 0, 1])})[:3, :3].T @ Rw
            n["rotation"] = rotation_en_quaternion(q).tolist()
        enfants = [k for k, o in enumerate(OS) if o[1] == i]
        if enfants:
            n["children"] = enfants
        noeuds.append(n)
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = Rw, pos
        W.append(M)
    maille = np.eye(4)
    if version == 1:
        maille[:3, 3] = DECALAGE
    ibm = np.array([np.linalg.inv(w) @ maille for w in W])
    peau = {"joints": list(range(len(OS))), "inverseBindMatrices": acc(ibm.transpose(0, 2, 1).reshape(-1, 16).astype("<f4"), 5126, "MAT4")}
    noeuds.append({"name": "Body", "mesh": 0, "skin": 0} | ({"translation": list(DECALAGE)} if version == 1 else {}))
    mats = ["N00_000_00_Body_00_SKIN", "N00_000_00_Face_00_SKIN", "N00_001_01_Tops_01_CLOTH"]
    if version == 1:
        mats = [m + " (Instance)" for m in mats]
    if not faces_vroid:
        mats = ["Body_SKIN", "Face_SKIN", "Tops_CLOTH"]
    if version == 0:
        ext = {"VRM": {"meta": {"title": "factice", "author": "test", "licenseName": "CC0"},
                       "humanoid": {"humanBones": [{"bone": r, "node": i} for i, (_, _, _, r) in enumerate(OS) if r]}}}
    elif version == 1:
        un = {"leftThumbProximal": "leftThumbMetacarpal"}
        ext = {"VRMC_vrm": {"specVersion": "1.0",
                            "meta": {"name": "factice", "authors": ["test"], "licenseUrl": "https://vrm.dev/licenses/1.0/",
                                     "commercialUsage": "personalNonProfit", "modification": "prohibited",
                                     "allowRedistribution": False, "creditNotation": "required", "avatarPermission": "onlyAuthor"},
                            "humanoid": {"humanBones": {un.get(r, r): {"node": i} for i, (_, _, _, r) in enumerate(OS) if r}}}}
    else:
        ext = {}
    j = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(tampon)}], "bufferViews": vues, "accessors": accs,
         "nodes": noeuds, "scenes": [{"nodes": [0, len(OS)]}], "skins": [peau],
         "meshes": [{"name": "Body", "primitives": prims}],
         "materials": [{"name": m} for m in mats], "extensions": ext}
    texte = json.dumps(j).encode()
    texte += b" " * (-len(texte) % 4)
    while len(tampon) % 4:
        tampon.append(0)
    with open(chemin, "wb") as f:
        f.write(b"glTF" + struct.pack("<II", 2, 28 + len(texte) + len(tampon)))
        f.write(struct.pack("<I", len(texte)) + b"JSON" + texte)
        f.write(struct.pack("<I", len(tampon)) + b"BIN\0" + bytes(tampon))


def rotation_en_quaternion(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w])


class Importeur(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.vrm = os.path.join(cls.tmp.name, "Factice.vrm")
        vrm_factice(cls.vrm)
        cls.bilan = I.convertir(cls.vrm, os.path.join(cls.tmp.name, "mmd", "Factice"), journal=lambda *_: None)
        cls.R = PMX.lire(cls.bilan["pmx"])
        cls.T = PMX.vers_tableaux(cls.R)
        cls.os = [o["nom"] for o in cls.R["os"]]
        cls.mats = [m["nom"] for m in cls.R["materiaux"]]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def faces(self, nom):
        k = self.mats.index(nom)
        return self.T["faces"][self.T["mat_face"] == k]

    def test_os_aux_noms_mmd(self):
        for nom in ("全ての親", "センター", "下半身", "上半身", "上半身1", "首", "頭", "左腕", "左ひじ", "左手首", "右腕", "左足首"):
            self.assertIn(nom, self.os)
        parent = lambda n: self.os[self.R["os"][self.os.index(n)]["parent"]]     # noqa: E731
        self.assertEqual(parent("上半身"), "センター")
        self.assertEqual(parent("下半身"), "センター")

    def test_os_de_poitrine_fondu_dans_son_parent(self):
        self.assertFalse([n for n in self.os if "Bust" in n])
        # the triangle the chest bone held (skin, at 1.25 m in front) is held by 上半身1
        P, tenu = self.T["pos"], self.T["os"][:, 0]
        v = np.where((np.abs(P[:, 1] - 1.25 * I.ECH) < 0.3) & (np.abs(P[:, 2]) > 0.9) & (P[:, 0] > 0.3))[0]
        self.assertTrue(len(v))
        self.assertEqual({self.os[k] for k in tenu[v]}, {"上半身1"})

    def test_bras_baisses(self):
        o = {n: np.array(self.R["os"][self.os.index(n)]["pos"]) for n in ("左腕", "左手首", "右腕", "右手首")}
        for cote in "左右":
            d = o[cote + "手首"] - o[cote + "腕"]
            angle = np.degrees(np.arctan2(-d[1], abs(d[0])))
            self.assertAlmostEqual(angle, I.description()["pose"]["bras"], places=3)
        self.assertGreater(o["左手首"][0], 0)      # the character's left at +x
        # the triangle held by the hand has followed
        P = self.T["pos"][self.T["os"][:, 0] == self.os.index("左手首")]
        self.assertLess(np.abs(P - o["左手首"]).max(), 0.5)

    def test_repere_et_taille(self):
        P = self.T["pos"]
        self.assertAlmostEqual(P[:, 1].max(), 1.57 * I.ECH, places=3)
        self.assertLess(P[self.faces("颜").ravel(), 2].max(), 0)     # the face, in front: -z

    def test_materiaux_traduits(self):
        self.assertEqual(self.mats, ["颜", "肌", "上衣"])
        self.assertEqual(self.R["materiaux"][1]["nom_en"], "N00_000_00_Body_00_SKIN")

    def test_peau_coupee_en_bas_du_cou(self):
        D = I.description()
        y0 = (1.4 + D["pose"]["cou"] * 0.1) * I.ECH
        F, P = self.faces("肌"), self.T["pos"]
        haut = (P[F][:, :, 1] >= y0 - 1e-4).all(1)
        self.assertEqual(int(haut.sum()), 2 * COTES)           # the only band above the cut
        # no face below shares a vertex with the neck skin, neither by index nor by position
        self.assertFalse(set(F[haut].ravel()) & set(F[~haut].ravel()))
        cle = lambda v: {tuple(x) for x in np.round(P[np.unique(v)], 4).tolist()}     # noqa: E731
        self.assertFalse(cle(F[haut]) & cle(F[~haut]))
        # neck edge flat, at the height of the cut; the body's just below
        bas_du_cou = P[np.unique(F[haut])][:, 1].min()
        self.assertAlmostEqual(bas_du_cou, y0, places=4)
        dessous = P[np.unique(F[~haut])][:, 1].max()
        self.assertAlmostEqual(dessous, y0 - I.COUPE, places=4)

    def test_forme_du_visage_en_morphe_mmd(self):
        m = {x["nom"]: x for x in self.R["morphes"]}
        self.assertIn("まばたき", m)
        self.assertEqual(m["まばたき"]["nom_en"], "Fcl_EYE_Close")
        self.assertEqual(len(m["まばたき"]["decalages"]), 1)
        self.assertAlmostEqual(m["まばたき"]["decalages"][0][1][1], -0.01 * I.ECH, places=4)

    def test_faces_effacees_par_la_texture(self):
        a = np.zeros((8, 8), np.float32)
        a[:4] = 1.0                                             # top half drawn
        uv = np.array([[0.1, 0.1], [0.4, 0.1], [0.1, 0.4],      # at the very top: visible
                       [0.1, 0.6], [0.4, 0.6], [0.1, 0.9],      # at the very bottom: erased
                       [0.1, 0.3], [0.4, 0.3], [0.1, 0.9]])     # straddling: visible
        F = np.arange(9).reshape(3, 3)
        self.assertEqual(I.invisibles(a, uv, F).tolist(), [False, True, False])

    def test_vrm_1_donne_le_meme_personnage(self):
        # the same character written as VRM 1.0 (rotated bones, mesh on its own node, thumb renamed, sparse shape):
        # same bones, same vertices, same morph as from the VRM 0.x
        ch = os.path.join(self.tmp.name, "Un.vrm")
        vrm_factice(ch, version=1)
        bilan = I.convertir(ch, os.path.join(self.tmp.name, "mmd", "Un"), journal=lambda *_: None)
        R = PMX.lire(bilan["pmx"])
        T = PMX.vers_tableaux(R)
        self.assertEqual(bilan["vrm"], "1.0")
        self.assertEqual([o["nom"] for o in R["os"]], self.os)
        self.assertIn("左親指０", self.os)        # 1.0 thumbMetacarpal = 0.x thumbProximal
        for a, b in zip(R["os"], self.R["os"]):
            self.assertLess(np.abs(np.subtract(a["pos"], b["pos"])).max(), 1e-3, a["nom"])
        self.assertLess(np.abs(T["pos"] - self.T["pos"]).max(), 1e-3)
        self.assertLess(np.abs(T["nor"] - self.T["nor"]).max(), 1e-3)
        self.assertEqual([m["nom"] for m in R["materiaux"]], self.mats)
        m = {x["nom"]: x for x in R["morphes"]}["まばたき"]["decalages"]
        self.assertEqual(len(m), 1)
        self.assertAlmostEqual(m[0][1][1], -0.01 * I.ECH, places=3)
        # the licence of the 1.0 meta is reported as the 0.x one is
        self.assertIn("vrm.dev/licenses/1.0", R["commentaire"])
        self.assertIn("modification: prohibited", R["commentaire"])
        self.assertIn("Declared licence: CC0", self.R["commentaire"])

    def test_vrm_qui_n_est_pas_vroid_refusee(self):
        for version in (0, 1):
            ch = os.path.join(self.tmp.name, "Autre%d.vrm" % version)
            vrm_factice(ch, version=version, faces_vroid=False)
            with self.assertRaisesRegex(I.Refus, "not a VRoid character"):
                I.Modele(ch)
            r = subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(ICI), "importer.py"), ch, "--output",
                                os.path.join(self.tmp.name, "refus")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertTrue(r.stderr.strip().startswith("refused: the file is not a VRoid character"), r.stderr)
        ch = os.path.join(self.tmp.name, "Rien.vrm")
        vrm_factice(ch, version=None)
        with self.assertRaisesRegex(I.Refus, "no VRM extension"):
            I.Modele(ch)

    def test_famille_dont_chacun_apporte_ses_modeles(self):
        # modeles "*": the <Name>/<Name>.pmx folders of mmd_extrait; a value of the description is overridden
        # by the setting games.<game>.<key>
        regl = os.path.join(self.tmp.name, "settings.json")
        with open(regl, "w", encoding="utf-8") as f:
            json.dump({"games": {"active": "vroid", "vroid": {"mmd_extrait": os.path.join(self.tmp.name, "mmd"),
                                                           "tete_base": {"nom": "Factice"}}}}, f)
        code = ("import sys, json; sys.path.insert(0, %r); import games as jeux; J = jeux.actif(); "
                "print(json.dumps([J.nom, J.modeles(), J.valeur('tete_base.nom'), J.valeur('tete_base.teinte')]))" % OUTILS)
        r = subprocess.run([sys.executable, "-I", "-c", code], env=dict(os.environ, CHIMERA_SETTINGS=regl),
                           capture_output=True, text=True, check=True)
        nom, modeles, tete, teinte = json.loads(r.stdout)
        self.assertEqual(nom, "vroid")
        self.assertEqual(modeles, {"Factice": os.path.join("Factice", "Factice.pmx")})
        self.assertEqual(tete, "Factice")
        self.assertEqual(teinte, "visage")


if __name__ == "__main__":
    unittest.main()

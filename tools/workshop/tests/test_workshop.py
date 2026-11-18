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


def feuille(z, n=20, largeur=0.3):
    """Square of n x n vertices in the plane z, normals towards +z."""
    xs = np.linspace(-largeur / 2, largeur / 2, n)
    X, Y = np.meshgrid(xs, xs)
    pos = np.c_[X.ravel(), Y.ravel(), np.full(n * n, z)].astype(float)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            F += [(a, a + 1, a + n), (a + 1, a + n + 1, a + n)]
    F = np.array(F)
    nor = np.tile([0.0, 0.0, 1.0], (n * n, 1))
    rep, inv = G.souder(pos, 1e-5)
    return SimpleNamespace(id="f", nom="feuille", pos=pos, nor=nor, F=F, rep=rep, inv=inv, meta={"rang": 1}, drapeaux=None)


def cylindre(r, n=48, m=12, h=0.3):
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
    rep, inv = G.souder(pos, 1e-5)
    return SimpleNamespace(id="c", nom="tube", pos=pos, nor=nor, F=F, rep=rep, inv=inv, meta={"rang": 1}, drapeaux=None)


def couche(p, meta=None):
    if meta:
        p.meta = meta
    return CB.Couche(p)


class TestGLB(unittest.TestCase):
    def test_aller_retour_et_orientation(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        nor = np.array([[0, 0, -1.0]] * 3)   # in MMD: -Z, so +Z once in glTF
        p, n, f = GLB.vers_gltf(pos, nor, np.array([[0, 2, 1]]))
        a, b, c = p[f[0]]
        self.assertGreater(float(np.dot(np.cross(b - a, c - a), n[0])), 0)
        self.assertAlmostEqual(float(p[1, 0]), 0.08, places=6)
        with tempfile.TemporaryDirectory() as d:
            chemin = os.path.join(d, "x.glb")
            GLB.ecrire(chemin, p, n, np.zeros((3, 2)), [{"faces": f, "materiau": 0}],
                       [{"nom": "m", "image": "/data/tex/a.webp", "diffus": (1, 1, 1, 1)}])
            doc, b_ = GLB.lire(chemin)
            self.assertEqual(doc["asset"]["version"], "2.0")
            self.assertEqual(doc["images"][0]["uri"], "/data/tex/a.webp")
            self.assertEqual(doc["accessors"][0]["count"], 3)
            self.assertEqual(len(b_) % 4, 0)

    def test_face_a_l_envers_retournee(self):
        # strip of four faces; the third is wound backwards and its normals are wrong too
        pos = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]], float)
        nor = np.tile([0, 0, 1.0], (6, 1))
        F = GLB.a_l_endroit(pos, nor, [[0, 2, 1], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
        g = np.cross(pos[F[:, 1]] - pos[F[:, 0]], pos[F[:, 2]] - pos[F[:, 0]])
        self.assertTrue((g[:, 2] > 0).all())
        self.assertEqual(F[0].tolist(), [0, 2, 1])

    def test_chaussure_posee_d_un_bloc(self):
        # two feet rotated and enlarged by fitting, a tenth of the vertices gone elsewhere
        import repair as RP
        rng = np.random.default_rng(0)
        S = rng.uniform(-0.05, 0.05, (200, 3)) + np.where(np.arange(200)[:, None] < 100, [-0.1, 0, 0], [0.1, 0, 0])
        c, s_ = np.cos(0.2), np.sin(0.2)
        juste = 1.2 * S @ np.array([[1, 0, 0], [0, c, s_], [0, -s_, c]]) + [0, 0.03, 0.01]
        Z = juste.copy()
        Z[::10] += rng.uniform(0.03, 0.08, (20, 3))
        self.assertLess(np.abs(RP.poser(S, Z) - juste).max(), 1e-6)

    def test_ilot_de_peau_sur_le_tissu_cache(self):
        # two skin faces facing +z, apart; a square of fabric under the first one only
        import skin as PE
        V = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0.2, 0, 0], [0.21, 0, 0], [0.2, 0.01, 0]], float)
        Fp = np.array([[0, 1, 2], [3, 4, 5]])
        N = np.tile([0, 0, 1.0], (6, 1))
        anciens = PE._peau, CB.peau
        PE._peau = lambda: (V[Fp], np.tile([0, 0, 1.0], (2, 1)), np.array([0.5, 0.5]))
        CB.peau = lambda: (V, Fp, N, None)
        try:
            F = np.array([[0, 1, 2], [1, 3, 2]])
            carre = np.array([[-0.02, -0.02, 0], [0.03, -0.02, 0], [-0.02, 0.03, 0], [0.03, 0.03, 0]], float)
            rien = np.zeros(0, np.int64)
            self.assertEqual(PE.enfoncee(carre + [0, 0, -0.004], F, rien).tolist(), [True, False])
            self.assertFalse(PE.enfoncee(carre + [0, 0, 0.004], F, rien).any())     # fabric in front: nothing more to hide
            self.assertFalse(PE.enfoncee(carre + [0, 0, -0.004], F, np.array([0])).any())   # already hidden
        finally:
            PE._peau, CB.peau = anciens

    def test_piece_relit_un_vrai_glb(self):
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
        nor = np.array([[0, 0, 1.0]] * 4)
        F = np.array([[0, 1, 2], [1, 3, 2]])
        with tempfile.TemporaryDirectory() as d:
            ancien, C.CACHE = C.CACHE, d
            try:
                os.makedirs(os.path.join(d, "pieces", "p"))
                C.ecrire_json(os.path.join(d, "pieces", "p", "meta.json"), {"type_fr": "essai"})
                GLB.ecrire(os.path.join(d, "pieces", "p", "piece.glb"), pos, nor, np.zeros((4, 2)),
                           [{"faces": F, "materiau": 0}], [{"nom": "m", "image": None, "diffus": (1, 1, 1, 1)}])
                p = CB.Piece("p")
            finally:
                C.CACHE = ancien
        self.assertEqual(p.F.tolist(), F.tolist())
        self.assertEqual(len(p.pos), 4)


class TestCommun(unittest.TestCase):
    def test_cle_sure(self):
        self.assertNotIn("/", C.cle("a/b", "c d"))
        self.assertEqual(C.cle("Anby", "jupe-1"), "Anby__jupe-1")

    def test_ordre_des_couches(self):
        r = lambda t: C.VETEMENTS[t][2]  # noqa: E731
        self.assertLess(r("culotte"), r("haut"))
        self.assertLess(r("haut"), r("jupe"))
        self.assertLess(r("jupe"), r("veste"))
        self.assertLess(r("veste"), C.ACCESSOIRES["ceinture"][0])


class TestCouches(unittest.TestCase):
    def test_poussee_sort_la_couche_du_dessus(self):
        dessous, dessus = feuille(0.0), feuille(-0.002)   # the upper one is 2 mm inside
        d = CB.pousser(dessus, dessus.pos, [(dessous.pos, dessous.F, dessous.nor)])
        Z = dessus.pos + d
        centre = np.abs(Z[:, :2]).max(1) < 0.1
        self.assertTrue(np.all(Z[centre, 2] >= CB.MARGE - 1e-6))

    def test_traversee_signalee(self):
        haut = couche(cylindre(0.15))
        bas = couche(cylindre(0.156))   # the lower one pokes out by 6 mm
        r = CB.traverse(bas, haut)
        self.assertIsNotNone(r)
        self.assertGreater(r["part"], 0.5)
        propre = couche(cylindre(0.144))
        r2 = CB.traverse(propre, haut)
        self.assertTrue(r2 is None or r2["part"] == 0)

    def test_faces_masquees_ne_comptent_plus(self):
        haut = couche(cylindre(0.15))
        bas = couche(cylindre(0.156))
        bas.masque[:] = True
        self.assertIsNone(CB.traverse(bas, haut))


class TestReglesDeCouches(unittest.TestCase):
    def test_culotte_retiree_sous_un_bas_ferme(self):
        for bas in ("short", "pantalon"):
            r = CB.retirees({"c": {"type": "culotte", "type_fr": "culotte"}, "b": {"type": bas, "type_fr": bas}})
            self.assertEqual(list(r), ["c"])

    def test_culotte_gardee_sous_une_jupe(self):
        self.assertEqual(CB.retirees({"c": {"type": "culotte"}, "j": {"type": "jupe"}}), {})

    def test_rentrer_passe_dessous_sans_gonfler_le_dessus(self):
        dessus, dessous = couche(cylindre(0.15)), couche(cylindre(0.156))
        avant = dessus.Z.copy()
        self.assertGreater(CB.rentrer(dessous, dessus), 0)
        self.assertTrue(np.array_equal(dessus.Z, avant))
        self.assertLess(float(np.hypot(dessous.Z[:, 0], dessous.Z[:, 2]).max()), 0.15)

    def test_masque_ce_qui_est_cache_et_tenu(self):
        dessus, dessous = couche(cylindre(0.15)), couche(cylindre(0.145))
        CB.masquer(dessous, dessus)
        self.assertTrue(dessous.masque.all())

    def test_ne_masque_pas_sous_une_piece_qui_flotte(self):
        dessus, dessous = couche(cylindre(0.20)), couche(cylindre(0.145))
        CB.masquer(dessous, dessus)
        self.assertFalse(dessous.masque.any())

    def test_ne_masque_pas_sous_un_tissu_ajoure(self):
        c = cylindre(0.15)
        c.drapeaux = np.full(len(c.F), CB.TENUE | CB.EXTERIEUR, np.uint8)    # held, but not opaque
        dessus, dessous = couche(c), couche(cylindre(0.145))
        CB.masquer(dessous, dessus)
        self.assertFalse(dessous.masque.any())

    def test_haut_dont_l_ourlet_sort_reste_dehors(self):
        haut, bas = couche(cylindre(0.16), {"type": "haut", "rang": 20}), couche(cylindre(0.15), {"type": "pantalon", "rang": 30})
        ordre, notes = CB.ordonner([haut, bas])
        self.assertEqual([c.type for c in ordre], ["pantalon", "haut"])
        self.assertTrue(notes)

    def test_haut_rentre_reste_dessous(self):
        haut, bas = couche(cylindre(0.145), {"type": "haut", "rang": 20}), couche(cylindre(0.15), {"type": "pantalon", "rang": 30})
        ordre, notes = CB.ordonner([haut, bas])
        self.assertEqual([c.type for c in ordre], ["haut", "pantalon"])
        self.assertFalse(notes)

    def test_pantalon_rentre_dans_la_botte(self):
        ancien, CB.corps = CB.corps, lambda: None
        try:
            pantalon = cylindre(0.156)
            pantalon.id, pantalon.meta = "p", {"type": "pantalon", "onglet": "bas", "rang": 30}
            botte = cylindre(0.15)
            botte.id, botte.meta = "b", {"type": "bottes", "onglet": "chaussures", "rang": 60}
            couches, _ = CB.resoudre([botte, pantalon])
        finally:
            CB.corps = ancien
        self.assertEqual([c.id for c in couches], ["p", "b"])
        self.assertTrue(couches[0].masque.all() or float(np.hypot(couches[0].Z[:, 0], couches[0].Z[:, 2]).max()) < 0.15)

    def test_ceinture_posee_sur_un_haut_a_larges_mailles(self):
        ancien, CB.corps = CB.corps, lambda: None
        try:
            haut = cylindre(0.16)
            haut.id, haut.meta = "h", {"type": "haut", "onglet": "haut", "rang": 20}
            ceinture = cylindre(0.15)
            ceinture.id, ceinture.meta = "c", {"type": "ceinture", "onglet": "accessoires", "rang": 70}
            couches, _ = CB.resoudre([ceinture, haut])
        finally:
            CB.corps = ancien
        par_id = {c.id: c for c in couches}
        self.assertTrue(np.array_equal(par_id["h"].Z, haut.pos))
        self.assertGreater(float(np.hypot(par_id["c"].Z[:, 0], par_id["c"].Z[:, 2]).min()), 0.16)

    def test_haute_chaussette_sous_la_culotte_mais_pas_le_collant(self):
        culotte = cylindre(0.15)
        bas, collant = cylindre(0.15), cylindre(0.15)
        bas.pos = bas.pos - [0.0, float(bas.pos[:, 1].max() - bas.pos[:, 1].min()) / 2, 0.0]
        collant.pos = collant.pos + [0.0, 0.02, 0.0]
        def ordre(ch):
            cs, _ = CB.ordonner([couche(culotte, {"type": "culotte", "rang": 10}), couche(ch, {"type": "chaussettes", "rang": 15})])
            return [c.type for c in cs]
        self.assertEqual(ordre(bas), ["chaussettes", "culotte"])
        self.assertEqual(ordre(collant), ["culotte", "chaussettes"])

    def test_traversee_cachee_par_une_autre_piece(self):
        self.assertFalse(CB.traversee_visible(0.08, 0.06))
        self.assertTrue(CB.traversee_visible(0.08, 0.02))
        self.assertFalse(CB.traversee_visible(0.03, 0.0))

    def test_chaussure_sans_dessus(self):
        pied = np.array([(x, 0.05, z) for x in np.linspace(-0.04, 0.04, 9) for z in np.linspace(-0.1, 0.1, 21)], float)
        n = np.tile([0.0, 1.0, 0.0], (len(pied), 1))
        semelle = feuille(-0.0, largeur=0.4)            # flat, 5 cm below the top of the foot: too far
        semelle.pos = semelle.pos[:, [0, 2, 1]]
        dessus = feuille(0.0, largeur=0.4)
        dessus.pos = dessus.pos[:, [0, 2, 1]] + [0.0, 0.06, 0.0]
        self.assertEqual(CV.dessus_du_pied(semelle.pos, np.tile([0.0, -1.0, 0.0], (len(semelle.pos), 1)), semelle.F, pied, n), 0.0)
        self.assertGreater(CV.dessus_du_pied(dessus.pos, np.tile([0.0, 1.0, 0.0], (len(dessus.pos), 1)), dessus.F, pied, n), CV.PIED_MIN)


class TestCouvercles(unittest.TestCase):
    """The backing that closes a sleeve around the arm is recognised: closed fan, crossed by the axis."""

    def eventail(self, centre, n=10, ouvert=False):
        c = np.asarray(centre, float)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        P = np.vstack([c, c + 0.5 * np.c_[np.zeros(n), np.cos(t), np.sin(t)]])
        F = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n - (1 if ouvert else 0))])
        return P, F

    def test_seul_le_fond_traverse_par_le_bras(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "clothes"))
        import wear as PO
        oss = [{"nom": "左腕", "pos": [1, 0, 0]}, {"nom": "左ひじ", "pos": [3, 0, 0]}]
        for centre, ouvert, attendu in (([2, 0, 0], False, True),      # around the arm
                                        ([2, 3, 0], False, False),     # beside the arm: ordinary fabric
                                        ([3.2, 0, 0], False, False),   # at the end of the bone: a visible end
                                        ([2, 0, 0], True, False)):     # open: not a backing
            P, F = self.eventail(centre, ouvert=ouvert)
            centres, faces, liens = PO.couvercles(P, F, oss)
            self.assertEqual(len(liens), int(attendu))
            self.assertEqual(bool(centres[0]), attendu, (centre, ouvert))
            self.assertEqual(bool(faces.all()), attendu)
            self.assertFalse(centres[1:].any())


class TestCou(unittest.TestCase):
    def test_cou_du_jeu_retire_sphere_crane(self):
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
        cou_tete = np.array([[0.2 * np.cos(a), 18.25, 0.2 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 36)])
        retirer, _, _, bilan = CJ.couper(pos, faces, mat_face, 0, [1], cou_tete, (0.0, 0.0))
        self.assertEqual(bilan["faces_crane"], 1)
        self.assertEqual(retirer.tolist(), [False, True, False])


def petit_pmx():
    """Minimal humanoid skeleton, arms in A-pose, two triangles per useful bone."""
    noms = {"下半身": (-1, (0, 10, 0)), "上半身": (0, (0, 11, 0)), "上半身1": (1, (0, 12, 0)), "上半身2": (2, (0, 13, 0)), "首": (3, (0, 15, 0)), "頭": (4, (0, 16, 0)),
            "左肩": (3, (0.5, 14.5, 0)), "左腕": (6, (1, 14.5, 0)), "左ひじ": (7, (3, 12.5, 0)), "左手首": (8, (5, 10.5, 0)), "左中指１": (9, (5.5, 10, 0)),
            "右肩": (3, (-0.5, 14.5, 0)), "右腕": (11, (-1, 14.5, 0)), "右ひじ": (12, (-3, 12.5, 0)), "右手首": (13, (-5, 10.5, 0)), "右中指１": (14, (-5.5, 10, 0)),
            "左足": (0, (1, 9, 0)), "左ひざ": (16, (1, 5, 0)), "左足首": (17, (1, 1, 0)), "右足": (0, (-1, 9, 0)), "右ひざ": (19, (-1, 5, 0)), "右足首": (20, (-1, 1, 0))}
    os_ = [{"nom": n, "parent": p, "pos": tuple(float(x) for x in c)} for n, (p, c) in noms.items()]
    pos, nor, uv, fs, liens = [], [], [], [], []
    for k, o in enumerate(os_):
        i = len(pos)
        x, y, z = o["pos"]
        pos += [(x, y, z), (x + 0.1, y, z), (x, y + 0.1, z)]
        nor += [(0.0, 0.0, -1.0)] * 3
        uv += [(0.0, 0.0)] * 3
        fs.append((i, i + 2, i + 1))
        liens += [(k, -1, -1, -1)] * 3
    poids = [(1.0, 0.0, 0.0, 0.0)] * len(pos)
    return {"os": os_, "sommets": {"pos": pos, "nor": nor, "uv": uv, "os": liens, "poids": poids}, "faces": fs, "textures": [],
            "materiaux": [{"nom": "peau", "n_indices": len(fs) * 3, "texture": -1, "diffus": (1, 1, 1, 1), "drapeaux": 1, "contour_taille": 0.0, "contour_couleur": (0, 0, 0, 1)}],
            "morphes": [{"nom": "まばたき", "type": 1, "decalages": [(0, (0.0, 0.1, 0.0))]}]}


class TestVRM(unittest.TestCase):
    def test_decoupe_seulement_si_la_texture_a_des_trous(self):
        m = {"diffus": [1, 1, 1, 1], "drapeaux": 0, "contour_taille": 0, "contour_couleur": [0, 0, 0, 1]}
        self.assertNotIn("alphaMode", VR._mtoon(m, 0, False, "cheveux", decoupe=False))
        self.assertEqual(VR._mtoon(m, 0, False, "cheveux", decoupe=True)["alphaMode"], "MASK")
        self.assertEqual(VR._mtoon(m, 0, False, "cheveux+", decoupe=False)["alphaMode"], "BLEND")

    def test_ecrit_un_vrm_valide(self):
        with tempfile.TemporaryDirectory() as d:
            ch = os.path.join(d, "x.vrm")
            r = VR.ecrire_vrm(petit_pmx(), ch, {})
            self.assertEqual(VR.valider(ch), [])
            doc, _ = VR.lire(ch)
        self.assertEqual(r["expressions"], 1)
        self.assertIn("VRMC_vrm", doc["extensions"])
        self.assertTrue(all("VRMC_materials_mtoon" in m["extensions"] for m in doc["materials"]))

    def test_bras_a_l_horizontale(self):
        S = VR.Squelette(petit_pmx())
        Pw, _ = S.tpose()
        ep, co = Pw[S.humain["leftUpperArm"]], Pw[S.humain["leftHand"]]
        self.assertAlmostEqual(float(ep[1]), float(co[1]), places=5)
        self.assertGreater(float(co[0]), float(ep[0]))

    def test_colonne_sous_les_hanches(self):
        S = VR.Squelette(petit_pmx())
        self.assertEqual(S.parent[S.humain["spine"]], S.humain["hips"])


class TestEmpilement(unittest.TestCase):
    def test_sans_os_nuls(self):
        P = {"sommets": {"os": [(138, 5, 0, 0), (7, -1, -1, -1), (3, 4, 9, 1)],
                         "poids": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0)]}}
        self.assertEqual(C.sans_os_nuls(P), 2)
        self.assertEqual(P["sommets"]["os"], [(138, 138, 138, 138), (7, -1, -1, -1), (3, 4, 3, 3)])

    def test_enveloppant(self):
        self.assertTrue(C.enveloppant({"onglet": "bas", "type": "jupe"}))
        self.assertTrue(C.enveloppant({"onglet": "accessoires", "type": "ceinture"}))
        self.assertFalse(C.enveloppant({"onglet": "accessoires", "type": "collier"}))


class TestReglagesCouches(unittest.TestCase):
    """The layers come from the settings (layers.default.json, then settings.json)."""

    # The tables and thresholds as they were written in the code before moving to the settings (15/11).
    VETEMENTS = {
        "culotte": ("bas", "dessous", 10), "chaussettes": ("chaussures", "chaussettes", 15), "haut": ("haut", "haut", 20),
        "manches": ("haut", "manches", 26), "short": ("bas", "bas", 30), "pantalon": ("bas", "bas", 30),
        "jupe": ("bas", "jupe", 32), "robe": ("bas", "robe", 34), "combinaison": ("bas", "robe", 34),
        "chaussures": ("chaussures", "chaussures", 40), "bottes": ("chaussures", "chaussures", 42),
        "gants": ("accessoires", "mains", 44), "col": ("haut", "col", 48), "gilet": ("haut", "gilet", 52),
        "veste": ("haut", "dessus", 55), "manteau": ("haut", "dessus", 60), "capuche": ("haut", "capuche", 62)}
    OCCUPE = {"robe": ["jupe"], "combinaison": ["bas", "jupe"]}
    ACCESSOIRES = {
        "ceinture": (70, True), "harnais": (72, True), "jarretiere": (66, False), "chevillere": (67, False),
        "brassard": (68, False), "bracelet": (69, False), "bague": (69, False), "cravate": (75, True),
        "echarpe": (76, True), "collier": (78, True), "bijou": (78, False), "sac": (80, True),
        "lunettes": (84, True), "masque": (84, True), "couvre_chef": (86, True), "casque_audio": (86, True),
        "queue": (88, True), "ailes": (88, True), "oreilles": (88, True)}
    FR_TYPE = {
        "culotte": "underwear", "chaussettes": "socks", "haut": "top", "manches": "sleeves", "short": "shorts",
        "pantalon": "trousers", "jupe": "skirt", "robe": "dress", "combinaison": "jumpsuit", "chaussures": "shoes",
        "bottes": "boots", "gants": "gloves", "col": "collar", "gilet": "waistcoat", "veste": "jacket", "manteau": "coat",
        "capuche": "hood", "jarretiere": "garter", "chevillere": "anklet", "brassard": "armband",
        "bracelet": "bracelet", "bague": "ring", "cravate": "tie", "echarpe": "scarf", "collier": "necklace",
        "bijou": "jewel", "sac": "bag", "lunettes": "glasses", "masque": "mask", "couvre_chef": "headwear",
        "casque_audio": "headphones", "queue": "tail", "ailes": "wings", "oreilles": "ears",
        "ceinture": "belt", "harnais": "harness"}
    SEUILS = {
        "MARGE": 0.003, "MARGE_RENTREE": 0.0015, "PORTEE_RENTREE": 0.05, "PLANCHER": 0.001, "PORTEE_RAYON": 0.06,
        "PORTEE_PEAU": 0.06, "LIBRE_MAX": 0.025, "ACCORD": 0.3, "FIN": 0.0002, "SORTIE_MAX": 0.008, "RAYON_MEMBRE": 0.05,
        "FLECHE_MAX": 0.003, "PORTEE_MASQUE": 0.03, "TOLERANCE_MASQUE": 0.006, "PASSES": 4, "EMPREINTE": 0.8,
        "CONTACT": 0.005, "PORTEE": 0.025, "SEUIL": 0.0005, "PART_TRAVERSE": 0.01, "ORDRE_DEHORS": 0.003,
        "CELLULE": 0.08, "VOXEL": 0.04, "VOXEL_CORPS": 0.012}

    def defauts(self):
        """The couches section of the public files only (without the machine's settings.json)."""
        import json
        with open(os.path.join(C.DEPOT, "layers.default.json"), encoding="utf-8") as f:
            return json.load(f)["layers"]

    def test_defauts_egaux_a_avant(self):
        V, O, A, Ce, F = C.tables(self.defauts())
        self.assertEqual(V, self.VETEMENTS)
        self.assertEqual(O, self.OCCUPE)
        self.assertEqual(A, self.ACCESSOIRES)
        self.assertEqual(Ce, {"ceinture"})
        self.assertEqual(F, self.FR_TYPE)
        T = self.defauts()["outfit"]
        self.assertEqual((T["ranks_in_outfit"], T["closed_bottoms"], T["tops"], T["bottoms"]),
                         ({"ceinture": 36}, ["short", "pantalon"], ["haut"], ["short", "pantalon", "jupe"]))

    def test_charges_egaux_a_avant(self):
        # what the code loaded (with no layer override in the machine's settings.json): exact equality
        if (C.R.tout().get("layers") or {}) != self.defauts():
            self.skipTest("the machine's settings.json overrides the layers")
        self.assertEqual((C.VETEMENTS, C.OCCUPE, C.ACCESSOIRES, C.FR_TYPE), (self.VETEMENTS, self.OCCUPE, self.ACCESSOIRES, self.FR_TYPE))
        for nom, v in self.SEUILS.items():
            self.assertEqual(getattr(CB, nom), v, nom)
            self.assertIs(type(getattr(CB, nom)), type(v), nom)
        self.assertEqual((CB.RANGS, CB.BAS_FERMES, CB.HAUTS, CB.BAS),
                         ({"ceinture": 36}, ("short", "pantalon"), ("haut", ), ("short", "pantalon", "jupe")))

    def test_surcharge_locale(self):
        """A settings.json that changes a rank, adds a type and a kind, removes a type and changes a threshold,
        without copying the tables: read by a fresh Python, like the server at start-up."""
        import json
        import subprocess
        local = {"layers": {
            "garments": {"jupe": {"rank": 33}, "capuche": None,
                         "kimono": {"tab": "haut", "slot": "dessus", "rank": 57, "name": "kimono"}},
            "accessories": {"broche": {"rank": 79}, "corset": {"rank": 37, "exclusive": True, "belt_like": True}},
            "outfit": {"closed_bottoms": ["pantalon"]},
            "thresholds": {"max_outward_mm": 10}}}
        code = ("import json, sys; sys.path.insert(0, %r); import common as C, outfit as CB; "
                "print(json.dumps({'jupe': C.VETEMENTS['jupe'], 'veste': C.VETEMENTS['veste'], 'kimono': C.VETEMENTS.get('kimono'), "
                "'capuche': 'capuche' in C.VETEMENTS, 'broche': C.ACCESSOIRES.get('broche'), 'nom_broche': C.FR_TYPE.get('broche'), "
                "'ceintures': sorted(C.CEINTURES), 'corset_env': C.enveloppant({'onglet': 'accessoires', 'type': 'corset'}), "
                "'range': C.ranger({'classe': 'vetement', 'type': 'jupe', 'onglet': 'bas', 'emplacement': 'bas', 'rang': 32}), "
                "'sortie': CB.SORTIE_MAX, 'marge': CB.MARGE_RENTREE, 'bas_fermes': CB.BAS_FERMES}))"
                % C.ICI)
        with tempfile.TemporaryDirectory() as d:
            ch = os.path.join(d, "settings.json")
            with open(ch, "w", encoding="utf-8") as f:
                json.dump(local, f)
            env = dict(os.environ, CHIMERA_SETTINGS=ch)
            r = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads(r.stdout)
        self.assertEqual(v["jupe"], ["bas", "jupe", 33])          # rank changed, tab and slot kept
        self.assertEqual(v["veste"], ["haut", "dessus", 55])     # the rest of the table is there
        self.assertEqual(v["kimono"], ["haut", "dessus", 57])    # type added
        self.assertFalse(v["capuche"])                           # null removes a type
        self.assertEqual(v["broche"], [79, False])               # kind added, not exclusive by default
        self.assertEqual(v["nom_broche"], "broche")              # no name: its own
        self.assertEqual(v["ceintures"], ["ceinture", "corset"])
        self.assertTrue(v["corset_env"])
        self.assertEqual(v["range"]["rang"], 33)                 # a record prepared earlier takes today's rank
        self.assertEqual(v["sortie"], 0.01)
        self.assertEqual(v["marge"], 0.0015)
        self.assertEqual(v["bas_fermes"], ["pantalon"])


class TestTraductionMateriaux(unittest.TestCase):
    """A game's material names translated into the common language by its description ("materiaux" table), at the single
    point where PMX files are read (atlas/pmx.lire)."""

    @staticmethod
    def pmx_minimal(noms):
        """A PMX with one triangle per material, with the given names."""
        n = len(noms)
        pos = [(float(i), float(j), 0.0) for i in range(n) for j in range(3)]
        return {"version": 2.0, "encodage": "UTF-16", "n_uv_add": 0, "nom": "essai", "nom_en": "essai",
                "sommets": {"pos": pos, "nor": [(0.0, 0.0, 1.0)] * len(pos), "uv": [(0.0, 0.0)] * len(pos),
                            "uv_add": [[]] * len(pos), "type": [0] * len(pos), "os": [(0,)] * len(pos),
                            "poids": [(1.0,)] * len(pos), "contour": [1.0] * len(pos)},
                "faces": [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(n)], "textures": [],
                "materiaux": [{"nom": x, "nom_en": x, "diffus": (1.0, 1.0, 1.0, 1.0), "speculaire": (0.0, 0.0, 0.0),
                               "brillance": 5.0, "ambiant": (0.5, 0.5, 0.5), "drapeaux": 0, "contour_couleur": (0.0, 0.0, 0.0, 1.0),
                               "contour_taille": 0.0, "texture": -1, "sphere": -1, "sphere_mode": 0, "toon_partage": 1,
                               "toon": 0, "memo": "", "n_indices": 3} for x in noms],
                "os": [{"nom": "センター", "nom_en": "center", "pos": (0.0, 0.0, 0.0), "parent": -1, "couche": 0,
                        "drapeaux": 0x1E, "queue_os": None, "queue_dec": (0.0, 1.0, 0.0)}],
                "morphes": [], "cadres": [], "rigides": [], "joints": []}

    def test_zzz_identite(self):
        import games as jeux
        Z = jeux.charger("zzz")
        self.assertEqual(Z.materiaux, [])
        for n in ("颜", "肌", "目", "髮2", "体", "外套+", "M_2000001_FACE", "Hair_2"):
            self.assertEqual(Z.traduire(n), n)
        # a PMX from the ZZZ sources is tied to no table: pmx.lire returns it as is
        self.assertIsNone(jeux.jeu_du_fichier(os.path.join(Z.chemin("mmd_extrait"), "Hugo", "雨果.pmx")))

    def test_outerplane_roles_comme_avant(self):
        import games as jeux
        OP = jeux.charger("outerplane")
        # what outerplane._role used to return hard-coded (prefix M_<id>_ removed, case ignored)
        attendu = {"M_2000001_FACE": "颜", "M_2000001_FACE_EYE": "眉睫", "M_2000001_HAIR": "髮", "M_2000014_HAIR_CLEAR": "髮",
                   "M_2000001_BODY_02": "体", "face": "颜", "M_1_face": "颜", "M_2000001_WEAPON": None,
                   "M_2000001_FACE\n": None, "M__FACE": None, "M_2000001_FACEX": None, "Default-Material": None}
        for n, t in attendu.items():
            self.assertEqual(OP.traduire(n, sinon=None), t, n)

    def test_jeu_factice_traduit_a_la_lecture(self):
        import games as jeux
        import pmx as PMX
        import pmx_write as pmx_ecrire
        noms = ["Skin", "Face", "Eye_L", "Hair_2", "Hair", "Coat", "肌"]
        with tempfile.TemporaryDirectory() as d:
            desc = os.path.join(d, "description")
            os.makedirs(desc)
            with open(os.path.join(desc, "game.json"), "w", encoding="utf-8") as f:
                f.write(r'''{"titre": "Dummy game", "format": "pmx",
                    "sources": {"modeles": "%s"},
                    "_materiaux": "the first matching pattern wins",
                    "materiaux": [["(?i)^skin$", "肌"], ["(?i)^face$", "颜"], ["(?i)^eye_[lr]$", "目"],
                                  ["(?i)^hair_?(\\d*)$", "髮\\1"]]}''' % os.path.join(d, "modeles"))
            try:
                J = jeux.charger("factice-essai", dossier=desc)
                self.assertEqual([J.traduire(n) for n in noms], ["肌", "颜", "目", "髮2", "髮", "Coat", "肌"])
                dans = os.path.join(d, "modeles", "perso", "perso.pmx")
                hors = os.path.join(d, "ailleurs", "perso.pmx")
                for c in (dans, hors):
                    pmx_ecrire.ecrire(self.pmx_minimal(noms), c)
                self.assertIs(jeux.jeu_du_fichier(dans), J)
                self.assertIsNone(jeux.jeu_du_fichier(hors))
                lu = PMX.lire(dans)["materiaux"]
                self.assertEqual([m["nom"] for m in lu], ["肌", "颜", "目", "髮2", "髮", "Coat", "肌"])
                self.assertEqual([m.get("nom_jeu") for m in lu], ["Skin", "Face", "Eye_L", "Hair_2", "Hair", None, None])
                self.assertEqual([m["nom_en"] for m in lu], noms)       # the rest of the material is not touched
                self.assertEqual([m["nom"] for m in PMX.lire(hors)["materiaux"]], noms)   # outside its sources: as is
            finally:
                jeux._charges.pop("factice-essai", None)


class TestDepartAVide(unittest.TestCase):
    """New machine: no body, no catalogue. The server says what to do instead of opening a page that would fail."""
    VIDE = {"tetes": [], "coiffures": [], "pieces": [], "refus": []}
    PLEIN = {"tetes": [{"id": "t"}], "coiffures": [{"id": "c"}], "pieces": [], "refus": []}

    def test_sans_corps(self):
        import catalog as K
        with tempfile.TemporaryDirectory() as d:
            m = K.a_faire(self.VIDE, corps=os.path.join(d, "corps.pmx"), cache=os.path.join(d, "cache"))
            self.assertEqual(len(m), 2)
            self.assertIn("workshop.body", m[0])
            self.assertIn("Empty catalogue", m[1])

    def test_corps_pas_prepare(self):
        import catalog as K
        with tempfile.TemporaryDirectory() as d:
            corps = os.path.join(d, "corps.pmx")
            open(corps, "wb").close()
            m = K.a_faire(self.PLEIN, corps=corps, cache=d)
            self.assertEqual(len(m), 1)
            self.assertIn("catalog.py prepare", m[0])

    def test_tout_est_la(self):
        import catalog as K
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "corps"))
            open(os.path.join(d, "corps", "corps.glb"), "wb").close()
            self.assertEqual(K.a_faire(self.PLEIN, corps=os.path.join(d, "absent.pmx"), cache=d), [])
            self.assertEqual(len(K.a_faire(dict(self.PLEIN, coiffures=[]), cache=d)), 1)

    def test_accueil_echappe(self):
        import server as S
        h = S.accueil(["<b>x</b> & y"]).decode("utf-8")
        self.assertIn("&lt;b&gt;x&lt;/b&gt; &amp; y", h)
        self.assertIn("INSTALLATION.md", h)


class TestGardeDuServeur(unittest.TestCase):
    """The server launches Blender and writes into the cache: it answers only its own page."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import ThreadingHTTPServer
        import server as S
        cls.S, cls.avant = S, S.CATALOGUE
        S.CATALOGUE = {"pieces": [{"id": "A__haut"}], "coiffures": [{"id": "c"}], "tetes": [{"id": "t"}]}
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), S.Gestion)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.S.CATALOGUE = cls.avant

    def demande(self, methode, chemin, corps=None, **entetes):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(methode, chemin, body=corps, headers=entetes)
        r = c.getresponse()
        r.read()
        c.close()
        return r.status

    def test_autre_site_refuse(self):
        ici = "http://127.0.0.1:%d" % self.port
        self.assertEqual(self.demande("GET", "/api/formats", Host="evil.example:%d" % self.port), 403)     # domain name pointing at 127.0.0.1
        self.assertEqual(self.demande("GET", "/api/formats", Origin="https://evil.example"), 403)
        self.assertEqual(self.demande("POST", "/api/ouvrir", b"{}", Origin="https://evil.example", **{"Content-Type": "application/json"}), 403)
        self.assertEqual(self.demande("PUT", "/api/vignette/pieces/A__haut", b"x", Origin="https://evil.example"), 403)
        self.assertNotEqual(self.demande("GET", "/api/catalogue", Origin=ici), 403)

    def test_post_sans_json_declare_refuse(self):
        # a form from another site is sent without a permission request: it is not read
        self.assertEqual(self.demande("POST", "/api/assembler", b'{"pieces": []}', **{"Content-Type": "text/plain"}), 415)

    def test_identifiants_hors_catalogue_refuses(self):
        j = {"Content-Type": "application/json"}
        for corps in (b'{"pieces": ["../../etc"]}', b'{"pieces": ["A__haut"], "coiffure": "../x"}',
                      b'{"pieces": ["A__haut"], "tete": "../x"}', b'{"pieces": "A__haut"}', b'[]', b'{'):
            self.assertEqual(self.demande("POST", "/api/combinaison", corps, **j), 400, corps)
        self.assertEqual(self.demande("POST", "/api/assembler", b'{"pieces": [], "teinte": 3}', **j), 400)
        self.assertEqual(self.demande("POST", "/api/ouvrir", b'{"id": "19990101-000000"}', **j), 400)

    def test_dossiers_d_assemblage_distincts(self):
        S = self.S
        with tempfile.TemporaryDirectory() as d:
            avant, lancer = S.ASSEMBLAGES, S.threading.Thread
            S.ASSEMBLAGES = d

            class Rien:
                def __init__(self, *a, **k):
                    pass

                def start(self):
                    pass
            S.threading.Thread = Rien
            try:
                ids = {S.assembler({"pieces": []}) for _ in range(3)}
            finally:
                S.ASSEMBLAGES, S.threading.Thread = avant, lancer
            self.assertEqual(len(ids), 3)

    def test_lancement_impossible_finit_en_erreur(self):
        """Any failure of the job (here the launcher itself) ends in an error state the page reads."""
        import time
        S = self.S
        with tempfile.TemporaryDirectory() as d:
            avant, lanceur = S.ASSEMBLAGES, S.C.lanceur
            S.ASSEMBLAGES = d

            def panne():
                raise OSError("no launcher")
            S.C.lanceur = panne
            try:
                ident = S.assembler({"pieces": []})
                for _ in range(100):
                    e = C.lire_json(os.path.join(d, ident, "etat.json"), {})
                    if e.get("fini"):
                        break
                    time.sleep(0.05)
            finally:
                S.ASSEMBLAGES, S.C.lanceur = avant, lanceur
            self.assertEqual((e.get("fini"), e.get("ok"), e.get("etape")), (True, False, "erreur"))
            self.assertIn("no launcher", e.get("erreur", ""))
            self.assertNotIn(ident, S._actifs)

    def test_travail_orphelin_lu_en_erreur(self):
        """A job left unfinished by an earlier server is read as interrupted, not as running forever."""
        import http.client
        import json
        S = self.S
        with tempfile.TemporaryDirectory() as d:
            avant = S.ASSEMBLAGES
            S.ASSEMBLAGES = d
            os.makedirs(os.path.join(d, "20200101-000000"))
            C.ecrire_json(os.path.join(d, "20200101-000000", "etat.json"), {"etape": "greffe", "fini": False})
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
                c.request("GET", "/api/travaux/20200101-000000")
                e = json.loads(c.getresponse().read())
                c.close()
            finally:
                S.ASSEMBLAGES = avant
        self.assertEqual((e["fini"], e["ok"], e["etape"]), (True, False, "erreur"))
        self.assertIn("interrupted", e["erreur"])


class TestEchecAssemblage(unittest.TestCase):
    """assemble_blender.py ends every failure in etat.json (etape erreur, fini, reason), SystemExit included:
    greffe_tete refuses that way, and the page would otherwise wait on its last step."""

    def lancer(self, d):
        import runpy
        argv = sys.argv
        sys.argv = ["blender", "--", "--dossier", d]
        try:
            with self.assertRaises(SystemExit):
                runpy.run_path(os.path.join(C.ICI, "assemble_blender.py"), run_name="__main__")
        finally:
            sys.argv = argv
        return C.lire_json(os.path.join(d, "etat.json"))

    def test_piece_inconnue_refusee_avant_tout_travail(self):
        with tempfile.TemporaryDirectory() as d:
            C.ecrire_json(os.path.join(d, "selection.json"), {"tete": "t", "coiffure": "c", "pieces": [""], "teinte": None})
            e = self.lancer(d)
        self.assertEqual((e["fini"], e["ok"], e["etape"]), (True, False, "erreur"))
        self.assertIn("unknown pieces", e["erreur"])

    def test_refus_par_systemexit_lisible(self):
        lire = C.lire_json

        def refus(chemin, *a, **k):
            if chemin.endswith("selection.json"):
                raise SystemExit("neck too different: test")
            return lire(chemin, *a, **k)
        C.lire_json = refus
        try:
            with tempfile.TemporaryDirectory() as d:
                e = self.lancer(d)
        finally:
            C.lire_json = lire
        self.assertEqual((e["fini"], e["ok"], e["etape"], e["erreur"]), (True, False, "erreur", "neck too different: test"))


class TestEcartSquelette(unittest.TestCase):
    """greffe_tete measures the skeleton under the grafted vertices: hair on 頭 ignores a longer body neck,
    neck skin does not."""

    def test_cheveux_sur_la_tete_et_peau_du_cou(self):
        import head_graft as GT
        os_ = lambda nom, pos, parent: {"nom": nom, "pos": pos, "parent": parent}  # noqa: E731
        H = [os_("首", (0.0, 9.2, 0.0), -1), os_("頭", (0.0, 10.0, 0.0), 0), os_("髪1", (0.0, 10.5, -0.3), 1)]
        B = [os_("首", (0.0, 11.0, 0.0), -1), os_("頭", (0.0, 12.0, 0.0), 0)]   # neck 1.0 long instead of 0.8
        t = np.array(B[1]["pos"]) - np.array(H[1]["pos"])
        idx = {o["nom"]: i for i, o in enumerate(B)}
        os_v = np.array([[2, -1, -1, -1], [1, 2, -1, -1], [0, -1, -1, -1], [0, 1, -1, -1]])
        w = np.array([[1, 0, 0, 0], [0.5, 0.5, 0, 0], [1, 0, 0, 0], [0.5, 0.5, 0, 0]], float)
        dec = GT.decalages_os(B, H, idx, t)
        np.testing.assert_allclose(dec[2], 0.0, atol=1e-9)       # hair bone: offset of 頭, its nearest shared ancestor
        e = GT.ecart_squelette(dec, os_v, w)
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), [0.0, 0.0, 0.2, 0.1], atol=1e-9)
        np.testing.assert_allclose(e[2], [0.0, -0.2, 0.0], atol=1e-9)   # neck skin follows the body's lower 首


class TestSuiviParModele(unittest.TestCase):
    """catalogue.Suivi: during an extraction the suite goes model by model, and each model is announced usable
    (« fini ») as soon as its last item is done, the shared work first."""

    def test_ordre_et_fin_par_modele(self):
        import contextlib
        import io
        import json
        import catalog as K
        s = K.Suivi(["Belle", "Anby"])
        suite = [{"genre": "piece", "id": C.cle("Anby", "bottes")}, {"genre": "coiffure", "id": "Belle"},
                 {"genre": "piece", "id": C.cle("Belle", "gants")}, {"genre": "coiffure", "id": "Anby"},
                 {"genre": "coiffure_libre", "id": "Libre"}, {"genre": "coiffure", "id": "Jane"}]
        s.trier(suite)
        self.assertEqual([it["id"] for it in suite][:2], ["Libre", "Jane"])        # shared work first
        self.assertEqual([it["id"] for it in suite][2:], ["Belle", C.cle("Belle", "gants"), "Anby", C.cle("Anby", "bottes")])
        sortie = io.StringIO()
        with contextlib.redirect_stdout(sortie):
            s.debut()
            s.lot_fini(suite[:3])
            s.lot_fini(suite[3:5])
        lignes = [json.loads(l[3:]) for l in sortie.getvalue().splitlines()]
        finis = [l["fini"] for l in lignes if "fini" in l]
        self.assertEqual(finis, ["Belle"])
        self.assertEqual([l for l in lignes if "modele" in l][-1], {"modele": "Anby", "pas": "pieces"})
        with contextlib.redirect_stdout(sortie):
            s.lot_fini(suite[5:])
        self.assertIn('"fini": "Anby"', sortie.getvalue())


class TestPousseeDesCheveux(unittest.TestCase):
    def test_couches_gardent_leur_ordre(self):
        """Two hair layers 3 mm apart, the inner one pushed by 10 mm: both move, the gap stays."""
        import hairstyles as CF
        a, b = feuille(0.0, n=10), feuille(0.003, n=10)
        P = np.r_[a.pos, b.pos]
        u = np.zeros_like(P)
        u[45] = [0, 0, 0.01]
        d = CF.champ_de_poussee(P, u, 0.02)
        self.assertAlmostEqual(d[45, 2], 0.01)
        self.assertAlmostEqual(d[100 + 45, 2], 0.01 * (1 - (0.003 / 0.02) ** 2) ** 2)
        self.assertTrue((d[100:, 2] >= d[:100, 2] - 1e-3).all())
        self.assertEqual(d[0].tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()

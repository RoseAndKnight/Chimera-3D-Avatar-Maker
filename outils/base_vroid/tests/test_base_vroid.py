"""Tests of the VRoid base that need no model.
    python3 -I outils/base_vroid/tests/test_base_vroid.py"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
OUTILS = os.path.dirname(os.path.dirname(ICI))
sys.path[:0] = [os.path.dirname(ICI), OUTILS]
import fabriquer as FA        # noqa: E402
import peau as PE             # noqa: E402

PEAU_RVB = (0.95, 0.80, 0.72)


def texture(n=128):
    """Plain skin, a blue square (a painted garment) and a transparent square (skin erased under the outfit)."""
    t = np.ones((n, n, 4))
    t[..., :3] = PEAU_RVB
    t[20:60, 20:60, :3] = (0.2, 0.3, 0.8)
    t[70:110, 70:110, 3] = 0
    return t


class Peau(unittest.TestCase):
    def test_vetement_et_transparence_repeints_en_peau(self):
        t = texture()
        visage = np.ones((16, 16, 4))
        visage[..., :3] = PEAU_RVB
        propre, connu = PE.nettoyer(t, visage)
        part = connu.mean()
        self.assertEqual(propre[..., 3].min(), 1.0)
        for zone in (propre[30:50, 30:50, :3], propre[80:100, 80:100, :3]):
            self.assertLess(np.abs(zone - PEAU_RVB).max(), 0.02)
        self.assertTrue(0.5 < part < 0.9, part)

    def test_masque_refuse_le_bord_et_les_ilots(self):
        t = texture(256)
        t[200:203, 10:13, :3] = (0.2, 0.3, 0.8)            # tiny garment
        self.assertTrue(PE.masque_peau(t, PEAU_RVB)[201, 30])
        m = PE.masque_peau(t, PEAU_RVB)
        self.assertFalse(m[40, 40])                         # garment
        self.assertFalse(m[19, 40])                         # eroded edge (1 texel at this size)
        self.assertTrue(m[10, 200])
        self.assertFalse(m[90, 90])                         # transparent

    def test_coutures_accordees_sans_toucher_la_peau_gardee(self):
        n = 256
        t = np.ones((n, n, 4))
        t[:, :128, :3] = (0.9, 0.7, 0.6)                    # "leg" island, repainted
        t[:, 128:, :3] = (0.8, 0.6, 0.5)                    # "foot" island, repainted
        connu = np.zeros((n, n), bool)
        connu[:, 250:] = True                               # a little kept skin, far from the seam
        v = (np.arange(40) + 100.5) / n
        uv_a = np.stack([np.full(40, 100.5 / n), v], 1)     # edge of the leg
        uv_b = np.stack([np.full(40, 150.5 / n), v], 1)     # the same edge, foot side
        out = PE.accorder_coutures(t, connu, uv_a, uv_b)
        self.assertLess(np.abs(out[120, 100, :3] - out[120, 150, :3]).max(), 0.01)
        self.assertTrue(np.array_equal(out[connu], t[connu]))
        self.assertLess(np.abs(out[120, 5, :3] - t[120, 5, :3]).max(), 0.05)    # far from the seam: almost nothing

    def test_remplir_garde_le_connu(self):
        rgb = np.random.default_rng(0).random((33, 17, 3))
        connu = np.zeros((33, 17), bool)
        connu[::3, ::2] = True
        out = PE.remplir(rgb, connu)
        self.assertTrue(np.array_equal(out[connu], rgb[connu]))
        self.assertTrue(np.isfinite(out).all())
        with self.assertRaises(ValueError):
            PE.remplir(rgb, np.zeros((33, 17), bool))


def pmx_jouet():
    """Two "肌" triangles and one "颜", three bones, an expression that touches both materials."""
    pos = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (5, 5, 0), (6, 5, 0), (5, 6, 0)]
    n = len(pos)
    s = {"pos": pos, "nor": [(0, 0, 1)] * n, "uv": [(0, 0)] * n, "uv_add": [[]] * n, "type": [0] * n,
         "os": [(0,)] * n, "poids": [(1.0,)] * n, "sdef": [None] * n, "contour": [1.0] * n}
    mat = lambda nom, k, t: {"nom": nom, "nom_en": nom, "texture": t, "n_indices": 3 * k}   # noqa: E731
    return {"sommets": s, "faces": [(4, 5, 6), (0, 1, 2), (1, 3, 2)], "textures": ["a.png", "b.png"],
            "materiaux": [mat("颜", 1, 0), mat("肌", 2, 1)], "os": [{"nom": "頭"}],
            "morphes": [{"nom": "あ", "type": 1, "decalages": [(5, (0, 1, 0)), (1, (0, 0, 1))]}],
            "cadres": [], "rigides": [1], "joints": [1]}


def tableaux(P):
    mf = np.concatenate([[i] * (m["n_indices"] // 3) for i, m in enumerate(P["materiaux"])]).astype(int)
    return {"pos": np.array(P["sommets"]["pos"], float), "faces": np.array(P["faces"]), "mat_face": mf}


class SousPmx(unittest.TestCase):
    def test_corps_seul_puis_tete_dans_le_meme_ordre(self):
        P = pmx_jouet()
        T = tableaux(P)
        corps, nv = FA.sous_pmx(P, T, {"肌"})
        self.assertEqual(len(corps["sommets"]["pos"]), 4)
        self.assertEqual(corps["textures"], ["b.png"])
        self.assertEqual(corps["materiaux"][0]["texture"], 0)
        self.assertEqual([m["nom"] for m in corps["materiaux"]], ["肌"])
        self.assertEqual(corps["rigides"], [])
        self.assertEqual(corps["morphes"][0]["decalages"], [(1, (0, 0, 1))])
        premiers = np.where(nv >= 0)[0][np.argsort(nv[nv >= 0])]
        tete, nv2 = FA.sous_pmx(P, T, {"肌", "颜"}, premiers)
        self.assertEqual(tete["sommets"]["pos"][:4], corps["sommets"]["pos"])       # raccord_peau: same order
        peau_t = [f for m, f in zip(tableaux(tete)["mat_face"], tete["faces"]) if tete["materiaux"][m]["nom"] == "肌"]
        self.assertEqual(peau_t, corps["faces"])
        self.assertEqual(len(tete["morphes"][0]["decalages"]), 2)


class Coutures(unittest.TestCase):
    def test_bords_de_coutures(self):
        # two triangles that share an edge at the same spot, but through split vertices with different uv
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 1, 0), (-1, 0, 0)], float)
        uv = np.array([(0.1, 0.1), (0.2, 0.1), (0.1, 0.2), (0.7, 0.7), (0.7, 0.8), (0.6, 0.7)])
        T = {"pos": pos, "uv": uv, "faces": np.array([(0, 1, 2), (3, 4, 5)]), "mat_face": np.array([0, 0])}
        a, b = FA.bords_de_coutures(T, 0, pas=4)
        self.assertEqual(len(a), 4)
        if a[0, 0] > 0.5:
            a, b = b, a                                     # the order of the two edges does not matter
        self.assertTrue(np.allclose(a[:, 0], 0.1) and np.allclose(b[:, 0], 0.7))
        self.assertTrue(np.allclose(a[:, 1] - 0.1, b[:, 1] - 0.7))   # same point of the edge on both sides
        T["uv"] = np.array([(0.1, 0.1), (0.2, 0.1), (0.1, 0.2), (0.1, 0.1), (0.1, 0.2), (0.0, 0.1)])
        self.assertEqual(len(FA.bords_de_coutures(T, 0)[0]), 0)    # same uv: not a seam


class Controles(unittest.TestCase):
    def test_ouvertures(self):
        # a closed tetrahedron, then with one face removed
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], float)
        F = np.array([(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)])
        self.assertEqual(FA.ouvertures(pos, F), [])
        self.assertEqual(FA.ouvertures(pos, F[:3])[0][0], 3)
        # a seam: split vertices at the same point, no hole
        pos2 = np.vstack([pos, pos[[0]]])
        F2 = F.copy()
        F2[3] = (4, 3, 2)
        self.assertEqual(FA.ouvertures(pos2, F2), [])

    def test_angle_bras(self):
        o = [{"nom": "左腕", "pos": (1, 10, 0)}, {"nom": "左ひじ", "pos": (2, 9, 0)},
             {"nom": "右腕", "pos": (-1, 10, 0)}, {"nom": "右ひじ", "pos": (-2, 10, 0)}]
        a = FA.angle_bras(o)
        self.assertAlmostEqual(a[0], 45.0)
        self.assertAlmostEqual(a[1], 0.0)

    def test_provenance_alertes(self):
        m = {k: None for k in FA.META}
        m.update(title="X", allowedUserName="OnlyAuthor", licenseName="Redistribution_Prohibited", commercialUssageName="Disallow")
        pj, md = FA.provenance(m, "/x/y.vrm", "y", "0" * 64)
        self.assertEqual(len(pj["alertes"]), 3)
        self.assertIn("Warning", md)
        self.assertNotIn("/x/", md)                       # no machine path in the provenance
        m.update(allowedUserName="Everyone", licenseName="CC0", commercialUssageName="Allow")
        self.assertEqual(FA.provenance(m, "/x/y.vrm", "y", "0" * 64)[0]["alertes"], [])

    def test_provenance_vrm_1(self):
        m = {k: None for k in FA.META_1}
        m.update(vrm="1.0", title="X", name="X", licenseUrl="https://vrm.dev/licenses/1.0/")
        pj, md = FA.provenance(m, "/x/y.vrm", "y", "0" * 64)
        self.assertEqual(len(pj["alertes"]), 4)           # absent keys take the strictest 1.0 defaults
        self.assertIn("VRM 1.0 meta", md)
        m.update(avatarPermission="everyone", allowRedistribution=True, commercialUsage="corporation", modification="allowModification")
        pj, md = FA.provenance(m, "/x/y.vrm", "y", "0" * 64)
        self.assertEqual(pj["alertes"], [])
        self.assertIn("`commercialUsage` : corporation", md)


class TeteBaseFabriquee(unittest.TestCase):
    def test_reglage_puis_base_puis_description(self):
        with tempfile.TemporaryDirectory() as t:
            os.makedirs(os.path.join(t, "base-vroid"))
            with open(os.path.join(t, "base-vroid", "base.json"), "w") as f:
                json.dump({"tete_base": {"nom": "Modele", "ecart_yeux": 1.06}}, f)
            regl = os.path.join(t, "reglages.json")
            with open(regl, "w") as f:
                json.dump({"chemins": {"donnees": t}, "atelier": {"corps": "{donnees}/base-vroid/corps.pmx"},
                           "jeux": {"vroid": {"tete_base": {"ecart_yeux": 2.0}}}}, f)
            ancien = os.environ.get("ATELIER_REGLAGES")
            os.environ["ATELIER_REGLAGES"] = regl
            import reglages as R
            import jeux
            try:
                R.oublier()
                J = jeux.Jeu("vroid")
                self.assertEqual(J.valeur("tete_base.nom"), "Modele")
                self.assertEqual(J.valeur("tete_base.ecart_yeux"), 2.0)
                self.assertEqual(J.valeur("tete_base.teinte"), "visage")
                self.assertIsNone(J.valeur("tete_base.centre_yeux", None))
                os.remove(os.path.join(t, "base-vroid", "base.json"))
                self.assertIsNone(J.valeur("tete_base.nom", None))
            finally:
                if ancien is None:
                    os.environ.pop("ATELIER_REGLAGES", None)
                else:
                    os.environ["ATELIER_REGLAGES"] = ancien
                R.oublier()


class CatalogueSurLaBase(unittest.TestCase):
    def test_base_ouverte_aux_modeles_importes(self):
        """A VRoid base (base.json: tete_base) offers its head then those of the imported models, except the
        character it is made from; MakeHuman's free base offers only itself."""
        from unittest import mock
        sys.path.insert(0, os.path.join(OUTILS, "atelier"))
        import catalogue as K
        jeu = mock.Mock(valeur=lambda cle, defaut=None: "toutes" if cle == "catalogue.tetes" else defaut)
        maison = mock.Mock(valeur=lambda cle, defaut=None: defaut)
        with mock.patch.object(K, "base_libre", return_value=True), \
             mock.patch.object(K, "coiffures_du_jeu", return_value=["D", "E", "F"]), \
             mock.patch.object(K, "MAISON", maison), mock.patch.object(K.C, "JEU", jeu):
            with mock.patch.object(K, "BASE_JSON", {"tete_base": {"nom": "D"}}):
                self.assertTrue(K.base_ouverte())
                self.assertEqual(K.catalogue("tetes", False), [K.LIBRE, "E", "F"])
                self.assertEqual(K.cle_base(), K.LIBRE)
            with mock.patch.object(K, "BASE_JSON", {}):
                self.assertFalse(K.base_ouverte())
                self.assertEqual(K.catalogue("tetes", False), [K.LIBRE])


if __name__ == "__main__":
    unittest.main()

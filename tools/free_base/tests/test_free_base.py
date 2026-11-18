"""Tests of the free base body that need neither a download nor a long computation.
    python3 -I tools/free_base/tests/test_free_base.py"""
import os
import sys
import tempfile
import unittest

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ICI, "..")] + [os.path.join(ICI, "..", "..", d)
                                            for d in ("", "custom_body", "rebuilt_body", "atlas", "head")]
import build as FA        # noqa: E402
import makehuman as MH        # noqa: E402
import free_head as TE       # noqa: E402
import free_face as VL     # noqa: E402


class Silhouette(unittest.TestCase):
    def test_cibles_nommees_par_leur_chemin(self):
        d = MH.doses()
        self.assertTrue(d)
        for c, w in d.items():
            self.assertTrue(c.endswith(".target") and "/" in c and not c.startswith("/"), c)
            self.assertTrue(0 < w <= 1.0, c)

    def test_rien_a_telecharger_quand_tout_est_la(self):
        with tempfile.TemporaryDirectory() as t:
            for f in list(FA.FICHIERS) + [os.path.join("targets", c) for c in MH.doses()] + list(VL.YEUX.values()):
                os.makedirs(os.path.dirname(os.path.join(t, f)), exist_ok=True)
                open(os.path.join(t, f), "w").close()
            vu = []
            self.assertEqual(FA.telecharger(t, vu.append), os.path.join(t, "base.obj"))
            self.assertEqual(vu, [])

    def test_lecture_et_cible(self):
        with tempfile.TemporaryDirectory() as t:
            with open(os.path.join(t, "base.obj"), "w") as f:
                f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\ng body\nf 1 2 3\ng joint-a\nf 1 2 3\n")
            with open(os.path.join(t, "c.target"), "w") as f:
                f.write("# cible\n1 0.5 0 0\n")
            V = MH.lire(os.path.join(t, "base.obj"), {os.path.join(t, "c.target"): 0.5})[0]
            self.assertAlmostEqual(float(V[1, 0]), 1.25)


class Coupe(unittest.TestCase):
    def test_coupe_nette(self):
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0]], float)
        F = np.array([[0, 1, 2], [1, 3, 2]])
        P, T = TE._couper(V, F, V[:, 1] - 1.0)[:2]
        self.assertTrue(len(T) and (P[np.unique(T)][:, 1] >= 1.0 - 1e-9).all())
        aire = sum(0.5 * np.linalg.norm(np.cross(P[b] - P[a], P[c] - P[a])) for a, b, c in T)
        self.assertAlmostEqual(aire, 1.0)


class Visage(unittest.TestCase):
    def test_mhclo_un_sommet_par_ligne(self):
        with tempfile.TemporaryDirectory() as t:
            c = os.path.join(t, "y.mhclo")
            with open(c, "w") as f:
                f.write("name yeux\nverts 0\n 14598\n 14599\n\n# fin\n 7\nfaces 3\n")
            self.assertEqual(VL._indices_mhclo(c).tolist(), [14598, 14599, 7])

    def test_triangles_retournes(self):
        self.assertEqual(VL._triangles([(0, 1, 2, 3)]), [(3, 2, 1), (3, 1, 0)])


class ModeLibre(unittest.TestCase):
    """Chimera's catalogue, without any game model: a free body, its face, its hairstyle."""

    def test_catalogue_libre(self):
        import subprocess
        code = ("import sys; sys.path.insert(0, %r); import catalog as K; import common as C; "
                "print(K.base_libre(), K.catalogue('tetes', False), K.catalogue('coiffures', False), "
                "bool([m for m in K.manque_pour_preparer() if 'head' in m.lower()]))" % os.path.join(ICI, "..", "..", "workshop"))
        with tempfile.TemporaryDirectory() as t:
            base = os.path.join(t, "base-libre")
            os.makedirs(base)
            env = dict(os.environ, CHIMERA_DATA=t, CHIMERA_BODY=os.path.join(base, "corps.pmx"), CHIMERA_CACHE=os.path.join(t, "cache"))
            lancer = lambda: subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True)   # noqa: E731
            r = lancer()
            self.assertEqual(r.stdout.split()[0], "False", r.stderr)          # nothing built: no free mode
            for f in ("corps.pmx", "tete.pmx", "coiffure.pmx"):
                open(os.path.join(base, f), "w").close()
            r = lancer()
            self.assertEqual(r.stdout.strip(), "True ['Libre'] ['Libre'] False", r.stderr)


if __name__ == "__main__":
    unittest.main()

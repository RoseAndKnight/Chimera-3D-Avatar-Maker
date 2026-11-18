"""Grey contact sheets of the free body (forme.pkl), without Blender: whole, arm, hand, foot, upper trunk.
    python3 -I regarder.py <folder>"""
import os
import pickle
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI] + [os.path.join(ICI, "..", d) for d in ("corps_maison", "corps_refait", "atlas")]
import poids as PO            # noqa: E402
import squelette as SQ        # noqa: E402
import vue as VU              # noqa: E402

d = sys.argv[1]
g = pickle.load(open(os.path.join(d, "forme.pkl"), "rb"))
sq = SQ.lire(os.path.join(d, "maison", "squelette.json"))
T = PO.triangles(g["quads"])
P = g["P"]
if "tete" in g:
    T = np.concatenate([T, g["tete"]["tris"] + len(P), g["tete"]["pont"]])
    P = np.concatenate([P, g["tete"]["P"]])
VU.planche(P, T, [VU.VUES[k] for k in ("face", "trois", "profil", "dos")], os.path.join(d, "forme.png"), T=700)
m, c, p = (sq.tetes[sq.i[n]] for n in ("main_G", "avant_bras_G", "pied_G"))
VU.planche(P, T, [VU.VUES[k] for k in ("face", "dessus", "dos")], os.path.join(d, "bras.png"), T=700,
           boite=[8, m[1] - 4, -14, m[0] + 4, 150, 14], fil=True)
VU.planche(P, T, [VU.VUES[k] for k in ("face", "dessus", "profil")], os.path.join(d, "main.png"), T=700,
           boite=[m[0] - 4, m[1] - 20, -12, m[0] + 20, m[1] + 4, 12], fil=True)
VU.planche(P, T, [VU.VUES[k] for k in ("dessus", "profil", "face")], os.path.join(d, "pied.png"), T=700,
           boite=[p[0] - 8, -1, -24, p[0] + 8, 14, 12], fil=True)
h = sq.tetes[sq.i["tete"]]
VU.planche(P, T, [VU.VUES[k] for k in ("face", "trois", "profil", "dos")], os.path.join(d, "tete.png"), T=700,
           boite=[-16, h[1] - 16, -16, 16, h[1] + 18, 16], fil=True)

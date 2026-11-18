"""Grey contact sheets of the base head stitched onto the body (Chimera's cache): whole and close-up of the neck.
    python3 -I view_base.py <resultat.pmx> <contact sheet folder>"""
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI] + [os.path.join(ICI, "..", d) for d in ("custom_body", "rebuilt_body", "atlas", "head")]
import pmx as PMX             # noqa: E402
import view as VU              # noqa: E402

B = PMX.lire(sys.argv[1])
t = PMX.vers_tableaux(B)
P, T = np.array(t["pos"]) * 8.0, np.array(t["faces"])
h = next(np.array(b["pos"]) * 8.0 for b in B["os"] if b["nom"] == "頭")
vues = [VU.VUES[k] for k in ("face", "trois", "profil", "dos")]
VU.planche(P, T, vues, os.path.join(sys.argv[2], "base-entier.png"), T=700)
VU.planche(P, T, vues, os.path.join(sys.argv[2], "base-tete.png"), T=700, boite=[-18, h[1] - 14, -18, 18, h[1] + 26, 18], fil=True)

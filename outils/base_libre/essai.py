"""The body grid, laid on MakeHuman's neutral skin (base.obj without shape targets).

    python3 -I essai.py <MakeHuman base.obj> <output folder>      writes <folder>/forme-makehuman.pkl

The silhouette (silhouette.json) comes afterwards, in corps.py: the grid follows it without being rebuilt.
"""
import os
import pickle
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI, os.path.join(ICI, "..", "corps_refait"), os.path.join(ICI, "..", "tete"), os.path.join(ICI, "..", "atlas")]
import reference as RL        # noqa: E402  (base_libre: comes before corps_refait)
import geometrie as GE        # noqa: E402

R = RL.Reference(sys.argv[1], {})
R.reperes = RL.reperes(R)
g = GE.construire(R)
os.makedirs(sys.argv[2], exist_ok=True)
with open(os.path.join(sys.argv[2], "forme-makehuman.pkl"), "wb") as f:
    pickle.dump(g, f)
print("SHAPE DONE", len(g["P"]), "vertices", len(g["quads"]), "quads")

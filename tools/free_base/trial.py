"""The body grid, laid on MakeHuman's neutral skin (base.obj without shape targets).

    python3 -I trial.py <MakeHuman base.obj> <output folder>      writes <folder>/forme-makehuman.pkl

The silhouette (silhouette.json) comes afterwards, in body.py: the grid follows it without being rebuilt.
"""
import os
import pickle
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI, os.path.join(ICI, "..", "rebuilt_body"), os.path.join(ICI, "..", "head"), os.path.join(ICI, "..", "atlas")]
import reference as RL        # noqa: E402  (base_libre: comes before corps_refait)
import geometry as GE        # noqa: E402

R = RL.Reference(sys.argv[1], {})
R.reperes = RL.reperes(R)
g = GE.construire(R)
os.makedirs(sys.argv[2], exist_ok=True)
with open(os.path.join(sys.argv[2], "forme-makehuman.pkl"), "wb") as f:
    pickle.dump(g, f)
print("SHAPE DONE", len(g["P"]), "vertices", len(g["quads"]), "quads")

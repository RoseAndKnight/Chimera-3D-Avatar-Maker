"""The body grid, laid on MakeHuman's neutral skin (base.obj without shape targets).

    python3 -I trial.py <MakeHuman base.obj> <output folder>      writes <folder>/shape-makehuman.pkl

The silhouette (silhouette.json) comes afterwards, in body.py: the grid follows it without being rebuilt.
"""
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "rebuilt_body"), os.path.join(HERE, "..", "head"), os.path.join(HERE, "..", "atlas")]
import reference as RL        # noqa: E402  (free_base: comes before rebuilt_body)
import geometry as GE        # noqa: E402

R = RL.Reference(sys.argv[1], {})
R.landmarks = RL.landmarks(R)
g = GE.build(R)
os.makedirs(sys.argv[2], exist_ok=True)
with open(os.path.join(sys.argv[2], "shape-makehuman.pkl"), "wb") as f:
    pickle.dump(g, f)
print("SHAPE DONE", len(g["P"]), "vertices", len(g["quads"]), "quads")

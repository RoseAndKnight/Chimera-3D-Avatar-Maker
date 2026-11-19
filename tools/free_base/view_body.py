"""Grey contact sheets of the free body (shape.pkl), without Blender: whole, arm, hand, foot, upper trunk.
    python3 -I view_body.py <folder>"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(HERE, "..", d) for d in ("custom_body", "rebuilt_body", "atlas")]
import weights as PO            # noqa: E402
import skeleton as SQ        # noqa: E402
import view as VU              # noqa: E402

d = sys.argv[1]
g = pickle.load(open(os.path.join(d, "shape.pkl"), "rb"))
sq = SQ.read(os.path.join(d, "own-rig", "skeleton.json"))
T = PO.triangles(g["quads"])
P = g["P"]
if "head" in g:
    T = np.concatenate([T, g["head"]["tris"] + len(P), g["head"]["bridge"]])
    P = np.concatenate([P, g["head"]["P"]])
VU.sheet(P, T, [VU.VIEWS[k] for k in ("face", "three_quarter", "profile", "back")], os.path.join(d, "shape.png"), T=700)
m, c, p = (sq.heads[sq.i[n]] for n in ("hand_L", "forearm_L", "foot_L"))
VU.sheet(P, T, [VU.VIEWS[k] for k in ("face", "over", "back")], os.path.join(d, "arm.png"), T=700,
           box=[8, m[1] - 4, -14, m[0] + 4, 150, 14], thread=True)
VU.sheet(P, T, [VU.VIEWS[k] for k in ("face", "over", "profile")], os.path.join(d, "hand.png"), T=700,
           box=[m[0] - 4, m[1] - 20, -12, m[0] + 20, m[1] + 4, 12], thread=True)
VU.sheet(P, T, [VU.VIEWS[k] for k in ("over", "profile", "face")], os.path.join(d, "foot.png"), T=700,
           box=[p[0] - 8, -1, -24, p[0] + 8, 14, 12], thread=True)
h = sq.heads[sq.i["head"]]
VU.sheet(P, T, [VU.VIEWS[k] for k in ("face", "three_quarter", "profile", "back")], os.path.join(d, "head.png"), T=700,
           box=[-16, h[1] - 16, -16, 16, h[1] + 18, 16], thread=True)

"""Grey contact sheets of the base head stitched onto the body (Chimera's cache): whole and close-up of the neck.
    python3 -I view_base.py <result.pmx> <contact sheet folder>"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(HERE, "..", d) for d in ("custom_body", "rebuilt_body", "atlas", "head")]
import pmx as PMX             # noqa: E402
import view as VU              # noqa: E402

B = PMX.read(sys.argv[1])
t = PMX.to_arrays(B)
P, T = np.array(t["pos"]) * 8.0, np.array(t["faces"])
h = next(np.array(b["pos"]) * 8.0 for b in B["bones"] if b["name"] == "頭")
views = [VU.VIEWS[k] for k in ("face", "three_quarter", "profile", "back")]
VU.sheet(P, T, views, os.path.join(sys.argv[2], "base-full.png"), T=700)
VU.sheet(P, T, views, os.path.join(sys.argv[2], "base-head.png"), T=700, box=[-18, h[1] - 14, -18, 18, h[1] + 26, 18], thread=True)

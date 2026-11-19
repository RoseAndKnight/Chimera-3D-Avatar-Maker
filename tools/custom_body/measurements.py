"""Read the grid of the rebuilt body (shape.pkl) as tubes of rings, and take measurements on it.

Each quad of the rebuilt body carries its map (tube name, corner (u, v)): u goes around the tube, v goes up it.
A ring is the set of vertices sharing the same v. The bones are placed on these rings.
"""
import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import settings as R  # noqa: E402

OUTFIT = os.path.join(R.path("paths.data"), "clothes", "outfit")   # the owner's outfit folder
# Chimera's body (same rule as tools/workshop/common.py: workshop.body of the settings, CHIMERA_BODY).
BODY = R.path("workshop.body")
SHAPE = os.path.join(os.path.dirname(BODY), "shape.pkl")
# The house skeleton of this body: <body folder>/own-rig; that of the rebuilt body stayed in a folder of the outfit (owner's data).
OWN_RIG = (os.path.join(OUTFIT, "own-rig-body") if os.path.basename(os.path.dirname(BODY)) == "rebuilt-body"
          else os.path.join(os.path.dirname(BODY), "own-rig"))


def load(path=SHAPE):
    with open(path, "rb") as f:
        g = pickle.load(f)
    grids = {}
    for q, c in zip(g["quads"], g["maps"]):
        if c is None:
            continue
        for s, (u, v) in zip(q, c[1]):
            grids.setdefault(c[0], {})[(int(round(u)), int(round(v)))] = int(s)
    g["grids"] = grids
    return g


def loop(g, map_, v):
    """Vertices of ring v of a map, in the order of u."""
    gr = g["grids"][map_]
    return np.array([s for (u, vv), s in sorted(gr.items()) if vv == v])


def ranks(g, map_):
    return sorted({v for (_, v) in g["grids"][map_]})


def center(g, map_, v):
    """Centre of a ring, weighted by the length of its edges (insensitive to how tightly the vertices are packed)."""
    p = g["P"][loop(g, map_, v)]
    q = np.roll(p, -1, 0)
    L = np.linalg.norm(q - p, axis=1)
    if len(p) < 6 or L.sum() < 1e-6:          # closed tip of a finger: a few vertices
        return p.mean(0)
    return ((p + q) / 2 * L[:, None]).sum(0) / L.sum()


def perimeter(g, map_, v):
    p = g["P"][loop(g, map_, v)]
    return float(np.linalg.norm(np.roll(p, -1, 0) - p, axis=1).sum())


def line(g, map_):
    """Centres and perimeters of all the rings of a map (closed tube)."""
    r = ranks(g, map_)
    return r, np.array([center(g, map_, v) for v in r]), np.array([perimeter(g, map_, v) for v in r])


def elbow(C):
    """Rank where a line of centres bends: the point farthest from the straight line joining its two ends."""
    a, b = C[0], C[-1]
    d = (b - a) / np.linalg.norm(b - a)
    h = (C - a) - np.outer((C - a) @ d, d)
    e = np.linalg.norm(h, axis=1)
    return int(e.argmax()), float(e.max())

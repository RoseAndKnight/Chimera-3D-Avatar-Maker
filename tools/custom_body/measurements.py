"""Read the grid of the rebuilt body (forme.pkl) as tubes of rings, and take measurements on it.

Each quad of the rebuilt body carries its map (tube name, corner (u, v)): u goes around the tube, v goes up it.
A ring is the set of vertices sharing the same v. The bones are placed on these rings.
"""
import os
import pickle
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import settings as R  # noqa: E402

TENUE = os.path.join(R.chemin("paths.data"), "vetements", "tenue")
# Chimera's body (same rule as tools/workshop/common.py: workshop.body of the settings, CHIMERA_BODY).
CORPS = R.chemin("workshop.body")
FORME = os.path.join(os.path.dirname(CORPS), "forme.pkl")
# The house skeleton of this body: <body folder>/maison; that of the rebuilt body stayed in tenue/corps_maison.
MAISON = (os.path.join(TENUE, "corps_maison") if os.path.basename(os.path.dirname(CORPS)) == "corps_refait"
          else os.path.join(os.path.dirname(CORPS), "maison"))


def charger(chemin=FORME):
    with open(chemin, "rb") as f:
        g = pickle.load(f)
    grilles = {}
    for q, c in zip(g["quads"], g["cartes"]):
        if c is None:
            continue
        for s, (u, v) in zip(q, c[1]):
            grilles.setdefault(c[0], {})[(int(round(u)), int(round(v)))] = int(s)
    g["grilles"] = grilles
    return g


def anneau(g, carte, v):
    """Vertices of ring v of a map, in the order of u."""
    gr = g["grilles"][carte]
    return np.array([s for (u, vv), s in sorted(gr.items()) if vv == v])


def rangs(g, carte):
    return sorted({v for (_, v) in g["grilles"][carte]})


def centre(g, carte, v):
    """Centre of a ring, weighted by the length of its edges (insensitive to how tightly the vertices are packed)."""
    p = g["P"][anneau(g, carte, v)]
    q = np.roll(p, -1, 0)
    L = np.linalg.norm(q - p, axis=1)
    if len(p) < 6 or L.sum() < 1e-6:          # closed tip of a finger: a few vertices
        return p.mean(0)
    return ((p + q) / 2 * L[:, None]).sum(0) / L.sum()


def perimetre(g, carte, v):
    p = g["P"][anneau(g, carte, v)]
    return float(np.linalg.norm(np.roll(p, -1, 0) - p, axis=1).sum())


def ligne(g, carte):
    """Centres and perimeters of all the rings of a map (closed tube)."""
    r = rangs(g, carte)
    return r, np.array([centre(g, carte, v) for v in r]), np.array([perimetre(g, carte, v) for v in r])


def coude(C):
    """Rank where a line of centres bends: the point farthest from the straight line joining its two ends."""
    a, b = C[0], C[-1]
    d = (b - a) / np.linalg.norm(b - a)
    h = (C - a) - np.outer((C - a) @ d, d)
    e = np.linalg.norm(h, axis=1)
    return int(e.argmax()), float(e.max())

"""The head of the free body: MakeHuman's (CC0), kept as it is above the neck of the grid.

The grid (corps_refait) stops at the top of the neck on a ring. Above it, MakeHuman's faces are kept
(jaw, face, skull, ears), and a band of triangles joins their edge to the ring. A game head
stitched on afterwards (vetements/tete_entiere.py) covers the face; the skull stays underneath.
"""
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ICI, "..", d) for d in ("tete", "corps_maison")]
import geom as G              # noqa: E402
import poids as PO            # noqa: E402

MARGE = 0.3                   # cm between the neck ring and the cut of MakeHuman's head
FONDU = 5.0                   # cm above the ring over which the weights go from those of the ring to the head bone


def anneau(P, T, haut):
    """The highest boundary loop (haut) or the lowest."""
    b = G.boucles_bord(T)
    y = [P[x][:, 1].mean() for x in b]
    return np.array(b[int(np.argmax(y) if haut else np.argmin(y))])


def _pont(Pa, a, Pb, b):
    """Band of triangles between two closed loops a (below) and b (above), paired by the angle around
    the vertical axis: returns triplets ((0|1, index), ...) facing outwards."""
    c = 0.5 * (Pa[a].mean(0) + Pb[b].mean(0))

    def tourner(P, l):
        ang = np.arctan2(P[l][:, 0] - c[0], P[l][:, 2] - c[2])
        if ((np.diff(np.unwrap(np.r_[ang, ang[0]]))).sum() < 0):
            l, ang = l[::-1], ang[::-1]
        k = int(np.argmin(np.abs(ang)))
        return np.roll(l, -k), np.unwrap(np.roll(ang, -k))
    a, ta = tourner(Pa, a)
    b, tb = tourner(Pb, b)
    ta, tb = ta - ta[0], tb - tb[0]
    ta, tb = np.r_[ta, 2 * np.pi], np.r_[tb, 2 * np.pi]
    i = j = 0
    tris = []
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and ta[i + 1] <= tb[j + 1]):
            tris.append(((0, a[i]), (0, a[(i + 1) % len(a)]), (1, b[j % len(b)])))
            i += 1
        else:
            tris.append(((0, a[i % len(a)]), (1, b[(j + 1) % len(b)]), (1, b[j])))
            j += 1
    # winding: the normal of the first triangle must point away from the axis
    pt = lambda s: (Pa if s[0] == 0 else Pb)[s[1]]      # noqa: E731
    t = tris[0]
    n = np.cross(pt(t[1]) - pt(t[0]), pt(t[2]) - pt(t[0]))
    if n @ ((pt(t[0]) - c) * [1.0, 0.0, 1.0]) < 0:
        tris = [(x, z, y) for x, y, z in tris]
    return tris


def _couper(V, F, h):
    """The triangles where h > 0 (h: one value per vertex), cut cleanly at h = 0: (vertices, triangles)."""
    pts, cle = [], {}

    def garde(i):
        if ("v", i) not in cle:
            cle[("v", i)] = len(pts)
            pts.append(V[i])
        return cle[("v", i)]

    def coupe(i, j):
        k = ("e", min(i, j), max(i, j))
        if k not in cle:
            cle[k] = len(pts)
            pts.append(V[i] + h[i] / (h[i] - h[j]) * (V[j] - V[i]))
        return cle[k]
    T = []
    for f in F[(h[F] > 0).any(1)]:
        poly = []
        for a, b in zip(f, np.roll(f, -1)):
            if h[a] > 0:
                poly.append(garde(a))
            if (h[a] > 0) != (h[b] > 0):
                poly.append(coupe(a, b))
        T += [(poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1)]
    return np.array(pts), np.array(T)


def faire(g, R, dy, journal=print):
    """g: grid in rest pose; R: MakeHuman's silhouette (same frame, dy cm lower).
    Returns {P, tris (local indices), pont (triangles of global indices: grid then head), anneau}."""
    an = anneau(g["P"], PO.triangles(g["quads"]), True)
    A = g["P"][an]
    c = A.mean(0)
    V = R.V + [0.0, dy, 0.0]
    # the cut follows the ring: at each angle around the neck, MARGE cm above it
    ang = np.arctan2(A[:, 0] - c[0], A[:, 2] - c[2])
    o = np.argsort(ang)
    haut = np.interp(np.arctan2(V[:, 0] - c[0], V[:, 2] - c[2]), ang[o], A[o, 1], period=2 * np.pi)
    y0 = float(A[:, 1].min()) + MARGE
    P, Tt = _couper(V, R.F, V[:, 1] - haut - MARGE)
    comp = np.arange(len(P))                       # the part that goes up to the top of the skull
    while True:
        avant = comp.copy()
        m = comp[Tt].min(1)
        for k in range(3):
            np.minimum.at(comp, Tt[:, k], m)
        if (comp == avant).all():
            break
    Tt = Tt[comp[Tt[:, 0]] == comp[int(P[:, 1].argmax())]]
    ids = np.unique(Tt)
    loc = -np.ones(len(P), int)
    loc[ids] = np.arange(len(ids))
    P, Tt = P[ids], loc[Tt]
    bord = anneau(P, Tt, False)
    assert np.ptp(P[bord][:, 1]) < np.ptp(A[:, 1]) + 0.5, "the bottom edge of the head is not the neck cut"
    pont = _pont(g["P"], an, P, bord)
    n = len(g["P"])
    pont = np.array([[s[1] + (n if s[0] else 0) for s in t] for t in pont])
    journal("head: %d vertices, %d MakeHuman triangles above the neck (from y = %.1f cm); bridge of %d triangles "
            "between the neck ring (%d vertices) and the cut (%d vertices)" % (len(P), len(Tt), y0, len(pont), len(an), len(bord)))
    return {"P": P, "tris": Tt, "pont": pont, "anneau": an, "y0": y0, "dy": dy}


def poids(g, tete, w, i_tete):
    """Weights (on our skeleton) of the head vertices: those of the closest vertex of the neck ring,
    blended towards the head bone over FONDU cm."""
    an = tete["anneau"]
    A = g["P"][an]
    k = np.linalg.norm(tete["P"][:, None] - A[None], axis=2).argmin(1)
    s = np.clip((tete["P"][:, 1] - tete["y0"]) / FONDU, 0, 1)
    s = s * s * (3 - 2 * s)
    out = w[an][k] * (1 - s)[:, None]
    out[:, i_tete] += s
    return out

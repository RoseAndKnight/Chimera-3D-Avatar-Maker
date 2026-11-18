"""Planar cuts of a mesh and resampling of the contours (numpy only).

A row of the grid is a cut of the reference body by a plane perpendicular to the bone:
the contour around the axis, resampled at equal steps along its length.
"""
import numpy as np


def coupe_plan(V, F, p0, n):
    """Contours of the cut of the mesh (V, F welded) by the plane (p0, n).
    Returns a list of (points (k, 3), closed)."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    d = (V - p0) @ n
    d = np.where(d == 0, 1e-9, d)
    s = d[F] > 0
    mixte = s.any(1) & ~s.all(1)
    Fm = F[mixte]
    if not len(Fm):
        return []
    pts = {}
    lien = {}
    for f in Fm:
        cles = []
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if (d[a] > 0) != (d[b] > 0):
                k = (int(a), int(b)) if a < b else (int(b), int(a))
                if k not in pts:
                    t = d[k[0]] / (d[k[0]] - d[k[1]])
                    pts[k] = V[k[0]] + t * (V[k[1]] - V[k[0]])
                cles.append(k)
        if len(cles) == 2:
            lien.setdefault(cles[0], []).append(cles[1])
            lien.setdefault(cles[1], []).append(cles[0])
    vus = set()
    out = []
    # open chains first (ends with a single link), then loops
    for passe in (1, 2):
        for k0 in lien:
            if k0 in vus or (passe == 1 and len(lien[k0]) != 1):
                continue
            ch = [k0]
            vus.add(k0)
            cur = k0
            while True:
                suiv = [x for x in lien[cur] if x not in vus]
                if not suiv:
                    break
                cur = suiv[0]
                vus.add(cur)
                ch.append(cur)
            ferme = passe == 2 and len(ch) > 2 and ch[0] in lien[ch[-1]]
            out.append((np.array([pts[k] for k in ch]), ferme))
    return out


def repere(n, ref):
    """Two axes of the plane with normal n: e1 = ref brought into the plane, e2 = n x e1."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    e1 = np.asarray(ref, float) - (np.asarray(ref, float) @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def tours(P, c, e1, e2):
    """Winding number of the closed contour P around c in the plane (e1, e2)."""
    x, y = (P - c) @ e1, (P - c) @ e2
    a = np.arctan2(y, x)
    da = np.diff(np.concatenate([a, a[:1]]))
    da = (da + np.pi) % (2 * np.pi) - np.pi
    return int(round(da.sum() / (2 * np.pi)))


def longueur(P, ferme=True):
    Q = np.concatenate([P, P[:1]]) if ferme else P
    return float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())


def contour_autour(V, F, c, n, ref):
    """The closed contour of the cut that surrounds c (the shortest if there are several), oriented in the
    direction e1 -> e2. Returns (points, e1, e2) or None."""
    e1, e2 = repere(n, ref)
    mieux = None
    for P, ferme in coupe_plan(V, F, c, n):
        if not ferme:
            continue
        t = tours(P, c, e1, e2)
        if t == 0:
            continue
        if t < 0:
            P = P[::-1]
        L = longueur(P)
        if mieux is None or L < mieux[1]:
            mieux = (P, L)
    return None if mieux is None else (mieux[0], e1, e2)


def reechantillonner(P, k, c, e1, e2, angle0=0.0):
    """k points at equal steps along the closed contour P (oriented e1 -> e2), the first where the contour
    crosses the half-line at angle angle0 (radians, from e1 towards e2) starting at c."""
    x, y = (P - c) @ e1, (P - c) @ e2
    a = (np.arctan2(y, x) - angle0 + np.pi) % (2 * np.pi) - np.pi     # relative angle, in ]-pi, pi]
    m = len(P)
    # crossing from negative to positive, the farthest from c if there are several
    i = np.arange(m)
    j = (i + 1) % m
    cand = np.where((a[i] <= 0) & (a[j] > 0) & (a[j] - a[i] < np.pi))[0]
    if not len(cand):
        cand = np.array([int(np.abs(a).argmin())])
    r = np.hypot(x, y)
    i0 = cand[np.argmax(r[cand])]
    j0 = (i0 + 1) % m
    t = -a[i0] / max(a[j0] - a[i0], 1e-12)
    dep = P[i0] + t * (P[j0] - P[i0])
    Q = np.concatenate([[dep], np.roll(P, -j0, axis=0), [dep]])
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.arange(k) * cum[-1] / k
    out = np.stack([np.interp(s, cum, Q[:, d]) for d in range(3)], 1)
    return out


def anneau(V, F, c, n, ref, k, angle0=0.0):
    """Row of k points: cut around c, resampled. Returns (points, perimeter) or None."""
    r = contour_autour(V, F, c, n, ref)
    if r is None:
        return None
    P, e1, e2 = r
    return reechantillonner(P, k, c, e1, e2, angle0), longueur(P)

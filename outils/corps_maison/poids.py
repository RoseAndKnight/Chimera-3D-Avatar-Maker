"""The skin weights on the house skeleton, computed by us (nothing is copied from another model).

Method: bone heat (Baran and Popović, "Automatic rigging and animation of 3D characters", 2007).
Each vertex receives the heat of the closest bone it sees (the segment vertex -> bone stays inside the
body: a flank does not see the arm bone through the armpit, a finger does not see the neighbour's bone), then
the heat spreads over the skin:  (L + A H) w_bone = A H p_bone , H = 1 / distance², L the Laplacian of the
grid, A the area around the vertex. Then the share of a twist bone is split between its steps
according to the position of the vertex along the bone, and left and right are made equal.
"""
import numpy as np

EPS_DEDANS = 0.05        # cm: start of the ray, just under the skin
CANDIDATS = 6            # bones tried per vertex, from closest to farthest
ITERATIONS = 600
SERRAGE = 1.0            # H = SERRAGE / distance²: 1 in Baran and Popović; larger, the heat spreads less


def triangles(quads):
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def normales(P, quads):
    n = np.zeros_like(P)
    for a, b, c in ((0, 1, 3), (1, 2, 0), (2, 3, 1), (3, 0, 2)):
        f = np.cross(P[quads[:, b]] - P[quads[:, a]], P[quads[:, c]] - P[quads[:, a]])
        np.add.at(n, quads[:, a], f)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def laplacien(P, quads):
    """Edges (i, j) and their weight: for a quad with sides a and b, the edge of length a weighs b / (2a)."""
    I, Jc, W = [], [], []
    for k in range(4):
        i, j = quads[:, k], quads[:, (k + 1) % 4]
        a = np.linalg.norm(P[i] - P[j], axis=1)
        b = 0.5 * (np.linalg.norm(P[j] - P[quads[:, (k + 2) % 4]], axis=1) + np.linalg.norm(P[quads[:, (k + 3) % 4]] - P[i], axis=1))
        I.append(i), Jc.append(j), W.append(0.5 * b / np.maximum(a, 1e-6))
    I, Jc, W = np.concatenate(I), np.concatenate(Jc), np.concatenate(W)
    aire = np.zeros(len(P))
    A = 0.5 * (np.linalg.norm(np.cross(P[quads[:, 1]] - P[quads[:, 0]], P[quads[:, 2]] - P[quads[:, 0]]), axis=1)
               + np.linalg.norm(np.cross(P[quads[:, 2]] - P[quads[:, 0]], P[quads[:, 3]] - P[quads[:, 0]]), axis=1))
    for k in range(4):
        np.add.at(aire, quads[:, k], A / 4)
    return I, Jc, W, aire


def proche(P, segs):
    """Closest point of each vertex on a bone made of several segments: (distance, point)."""
    best = np.full(len(P), np.inf)
    pt = np.zeros_like(P)
    for a, b in segs:
        a, b = np.asarray(a), np.asarray(b)
        d = b - a
        t = np.clip((P - a) @ d / max(d @ d, 1e-12), 0, 1)
        c = a + t[:, None] * d
        e = np.linalg.norm(P - c, axis=1)
        m = e < best
        best[m], pt[m] = e[m], c[m]
    return best, pt


def coupe(O, C, T0, E1, E2, lot=48):
    """True for each segment O -> C that crosses the skin (Möller-Trumbore, in batches)."""
    out = np.zeros(len(O), bool)
    lo, hi = np.minimum(np.minimum(T0, T0 + E1), T0 + E2), np.maximum(np.maximum(T0, T0 + E1), T0 + E2)
    for s in range(0, len(O), lot):
        o, c = O[s:s + lot], C[s:s + lot]
        b0, b1 = np.minimum(o, c).min(0), np.maximum(o, c).max(0)
        sel = np.all((hi >= b0) & (lo <= b1), 1)
        if not sel.any():
            continue
        t0, e1, e2 = T0[sel], E1[sel], E2[sel]
        d = (c - o)[:, None, :]
        h = np.cross(d, e2[None])
        a = (e1[None] * h).sum(2)
        ok = np.abs(a) > 1e-12
        f = 1.0 / np.where(ok, a, 1.0)
        sv = o[:, None, :] - t0[None]
        u = f * (sv * h).sum(2)
        q = np.cross(sv, e1[None])
        v = f * (d * q).sum(2)
        t = f * (e2[None] * q).sum(2)
        out[s:s + lot] = (ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < 1 - 1e-6)).any(1)
    return out


def sources(P, quads, sym, sq, journal=print):
    """p (n, bones): share of heat each vertex receives from each bone, and its distance to the heating bone."""
    n = len(P)
    peau = [k for k, o in enumerate(sq.os) if o["segments"]]
    D = np.zeros((n, len(peau)))
    C = np.zeros((n, len(peau), 3))
    for c, k in enumerate(peau):
        D[:, c], C[:, c] = proche(P, sq.os[k]["segments"])
    N = normales(P, quads)
    tr = triangles(quads)
    T0, E1, E2 = P[tr[:, 0]], P[tr[:, 1]] - P[tr[:, 0]], P[tr[:, 2]] - P[tr[:, 0]]
    moi = np.flatnonzero(P[:, 0] >= -1e-6)                    # the left and the middle; the right mirrored
    ordre = np.argsort(D[moi], axis=1)[:, :CANDIDATS]
    choix = -np.ones(len(moi), int)
    for r in range(CANDIDATS):
        att = np.flatnonzero(choix < 0)
        if not len(att):
            break
        c = ordre[att, r]
        cache = coupe(P[moi[att]] - EPS_DEDANS * N[moi[att]], C[moi[att], c], T0, E1, E2)
        choix[att[~cache]] = c[~cache]
        journal("  bones seen at rank %d: %d vertices, %d remain" % (r + 1, int((~cache).sum()), int(cache.sum())))
    perdus = choix < 0
    choix[perdus] = ordre[perdus, 0]
    p = np.zeros((n, len(sq.os)))
    d = np.zeros(n)
    dm = D[moi, choix]
    egal = D[moi] <= dm[:, None] * 1.0001 + 1e-6              # at a joint, the two bones share
    egal[np.arange(len(moi)), choix] = True
    p[np.ix_(moi, peau)] = egal / egal.sum(1, keepdims=True)
    d[moi] = dm
    mir = sq.miroir()
    autres = np.flatnonzero(P[:, 0] < -1e-6)
    p[autres] = p[sym[autres]][:, mir]
    d[autres] = d[sym[autres]]
    milieu = np.flatnonzero(np.abs(P[:, 0]) <= 1e-6)
    p[milieu] = 0.5 * (p[milieu] + p[milieu][:, mir])
    return p, d, int(perdus.sum())


def chaleur(P, quads, p, d, journal=print):
    """Solves (L + A H) w = A H p for all bones at once (conjugate gradient, diagonal preconditioner)."""
    I, Jc, W, aire = laplacien(P, quads)
    n = len(P)
    H = SERRAGE * aire / np.maximum(d, 0.05) ** 2
    deg = np.zeros(n)
    np.add.at(deg, I, W)
    np.add.at(deg, Jc, W)
    diag = deg + H

    def mul(X):
        Y = diag[:, None] * X
        np.subtract.at(Y, I, W[:, None] * X[Jc])
        np.subtract.at(Y, Jc, W[:, None] * X[I])
        return Y
    actifs = np.flatnonzero(p.sum(0) > 0)
    B = H[:, None] * p[:, actifs]
    X = p[:, actifs].copy()
    R = B - mul(X)
    Zr = R / diag[:, None]
    D = Zr.copy()
    rz = (R * Zr).sum(0)
    for it in range(ITERATIONS):
        AD = mul(D)
        al = rz / np.maximum((D * AD).sum(0), 1e-30)
        X += al * D
        R -= al * AD
        Zr = R / diag[:, None]
        rz2 = (R * Zr).sum(0)
        D = Zr + (rz2 / np.maximum(rz, 1e-30)) * D
        rz = rz2
        if np.abs(R).max() < 1e-7:
            break
    journal("  heat: %d iterations, residual %.1e" % (it + 1, np.abs(R).max()))
    w = np.zeros_like(p)
    w[:, actifs] = np.clip(X, 0, None)
    return w / w.sum(1, keepdims=True)


def partager_torsion(P, w, sq):
    """The share of a twist bone goes to its steps, according to the position of the vertex along the bone."""
    w = w.copy()
    groupes = {}
    for k, o in enumerate(sq.os):
        if o["torsion"]:
            groupes.setdefault(sq.parents[k], []).append(k)
    for parent, enfants in groupes.items():
        a = sq.tetes[parent]
        b = np.array(sq.os[parent]["segments"][0][1])
        t = np.clip((P - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        retient = sq.os[enfants[0]]["torsion"]["sorte"] == "retient"
        noeuds = [(sq.os[k]["torsion"]["part"], k) for k in enfants] + [((1.0, parent) if retient else (0.0, parent))]
        noeuds.sort()
        part = w[:, parent].copy()
        w[:, parent] = 0
        for i, (x, k) in enumerate(noeuds):
            f = np.zeros(len(P))
            if i > 0:
                x0 = noeuds[i - 1][0]
                m = (t >= x0) & (t <= x)
                f[m] = (t[m] - x0) / (x - x0)
            if i < len(noeuds) - 1:
                x1 = noeuds[i + 1][0]
                m = (t > x) & (t <= x1)
                f[m] = (x1 - t[m]) / (x1 - x)
            if i == 0:
                f[t <= x] = 1
            if i == len(noeuds) - 1:
                f[t >= x] = 1
            w[:, k] += part * f
    return w


def quatre(w):
    """Four bones per vertex at most (what a PMX and a game engine can carry), without breaks: from each vertex
    the value of its fifth weight is subtracted, then the sum is brought back to 1. Cutting off the smallest ones
    would give two neighbouring vertices different bones, and the skin would break between them."""
    tri = -np.sort(-w, axis=1)
    w = np.clip(w - tri[:, 4:5], 0, None)
    idx = np.argsort(-w, axis=1)[:, :4]
    pw = np.take_along_axis(w, idx, 1)
    pw /= pw.sum(1, keepdims=True)
    idx[pw == 0] = -1
    print("  four bones per vertex: the removed fifth weight is at worst %.2f" % tri[:, 4].max())
    return idx, pw


def calculer(g, sq, journal=print):
    P, quads, sym = g["P"], g["quads"], g["sym"]
    p, d, perdus = sources(P, quads, sym, sq, journal)
    journal("  %d vertices see none of their %d closest bones: they take the closest" % (perdus, CANDIDATS))
    w = chaleur(P, quads, p, d, journal)
    mir = sq.miroir()
    w = 0.5 * (w + w[sym][:, mir])
    w = partager_torsion(P, w, sq)
    return w

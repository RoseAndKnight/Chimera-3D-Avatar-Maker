"""Places the free vertices of the grid (the junctions) on the reference.

1. interpolation: each free vertex at the average of its neighbours (linear system, anchored ones fixed);
2. relaxation: average of the neighbours then back to the closest point of the reference, several times;
3. the right side stays the mirror of the left.
"""
import numpy as np

import geom as G

MIR = np.array([-1.0, 1.0, 1.0])


def voisins_quads(n, Qd):
    A = [set() for _ in range(n)]
    for q in Qd:
        for a, b in zip(q, np.roll(q, -1)):
            A[a].add(int(b))
            A[b].add(int(a))
    return [sorted(a) for a in A]


def triangles(Qd):
    return np.concatenate([Qd[:, [0, 1, 2]], Qd[:, [0, 2, 3]]])


def interpoler(P, libre, vois):
    """Free vertices: average of their neighbours (exact solve)."""
    idx = np.flatnonzero(libre)
    rang = -np.ones(len(P), int)
    rang[idx] = np.arange(len(idx))
    A = np.zeros((len(idx), len(idx)))
    B = np.zeros((len(idx), 3))
    for r, i in enumerate(idx):
        A[r, r] = len(vois[i])
        for j in vois[i]:
            if libre[j]:
                A[r, rang[j]] -= 1.0
            else:
                B[r] += P[j]
    P = P.copy()
    P[idx] = np.linalg.solve(A, B)
    return P


def symetrique(P, sym):
    return 0.5 * (P + P[sym] * MIR)


def detendre(P, bouge, vois, sym, cibles, tours=40, pas=0.6, journal=print):
    """cibles: list of (vertex mask, V, F): which surface each group lands on."""
    P = P.copy()
    idx = np.flatnonzero(bouge)
    # only the faces near the moving vertices serve as targets
    prep = []
    for m, V, F in cibles:
        k = np.flatnonzero(m & bouge)
        if len(k):
            lo, hi = P[k].min(0) - 6.0, P[k].max(0) + 6.0
            c = V[F].mean(1)
            Fp = F[((c > lo) & (c < hi)).all(1)]
            prep.append((k, V, Fp, V[Fp].mean(1)))
    for t in range(tours):
        moy = np.array([P[vois[i]].mean(0) for i in idx])
        P[idx] = (1 - pas) * P[idx] + pas * moy
        for k, V, F, Cf in prep:
            P[k] = proche_rapide(P[k], V, F, Cf)
        P = symetrique(P, sym)
    return P


def ecart(P, Qd, V, F, masque=None):
    """Distance from the reference to the new mesh (mm): median, 95th percentile, maximum."""
    T = triangles(Qd)
    S = V[np.unique(F)]
    if masque is not None:
        S = S[masque(S)]
    d = G.plus_proche(S, P, T)[3] * 10
    return float(np.median(d)), float(np.percentile(d, 95)), float(d.max())


def proche_rapide(Q, V, F, C=None, k=20, bloc=256, marge=3.0):
    """Closest point on (V, F) among the k faces with the closest centre, searched in the neighbourhood
    of each batch of points: enough for a point already near a surface with small faces, and much
    faster than the exhaustive search."""
    if C is None:
        C = V[F].mean(1)
    out = np.zeros_like(Q)
    ordre = np.lexsort((Q[:, 0] > 0, np.round(Q[:, 1] / 4.0)))
    for s in range(0, len(Q), bloc):
        ii = ordre[s:s + bloc]
        q = Q[ii]
        m = marge
        while True:
            sel = np.flatnonzero(((C > q.min(0) - m) & (C < q.max(0) + m)).all(1))
            if len(sel) >= 1 or m > 100:
                break
            m *= 2
        Cs, Fs = C[sel], F[sel]
        kk = min(k, len(sel))
        d2 = ((q[:, None, :] - Cs[None, :, :]) ** 2).sum(-1)
        cand = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        pq, _ = G._pp_triangle(q[:, None, :], V[Fs[cand, 0]], V[Fs[cand, 1]], V[Fs[cand, 2]])
        j = np.linalg.norm(pq - q[:, None, :], axis=2).argmin(1)
        out[ii] = pq[np.arange(len(q)), j]
    return out


def dilater(masque, vois, rangs):
    m = masque.copy()
    for _ in range(rangs):
        k = np.flatnonzero(m)
        for i in k:
            m[vois[i]] = True
    return m


def adoucir(P, T, libre, vois, sym, seuil=38.0, passes=12, tours=4, pas=0.5, rangs=1, journal=print):
    """Removes sharp edges: the vertices of edges over `seuil` degrees and their neighbours are smoothed
    (average of the neighbours, without going back to the reference), until none are left."""
    P = P.copy()
    for p in range(passes):
        ang, ar = G.diedres(P, T)
        vif = ar[ang > seuil]
        m = np.zeros(len(P), bool)
        m[vif.ravel()] = True
        m = dilater(m, vois, rangs) & libre
        m |= m[sym]
        if not m.any():
            break
        idx = np.flatnonzero(m)
        for _ in range(tours):
            moy = np.array([P[vois[i]].mean(0) for i in idx])
            P[idx] = (1 - pas) * P[idx] + pas * moy
            P = symetrique(P, sym)
    ang, ar = G.diedres(P, T)
    reste = ar[ang > seuil]
    reste = reste[libre[reste].all(1)]
    journal(f"smoothing: {p + 1} passes, {len(reste)} edges over {seuil:.0f}° remain outside hands and feet")
    return P

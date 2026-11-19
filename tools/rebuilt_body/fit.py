"""Places the free vertices of the grid (the junctions) on the reference.

1. interpolation: each free vertex at the average of its neighbours (linear system, anchored ones fixed);
2. relaxation: average of the neighbours then back to the closest point of the reference, several times;
3. the right side stays the mirror of the left.
"""
import numpy as np

import geom as G

MIR = np.array([-1.0, 1.0, 1.0])


def neighbors_quads(n, Qd):
    A = [set() for _ in range(n)]
    for q in Qd:
        for a, b in zip(q, np.roll(q, -1)):
            A[a].add(int(b))
            A[b].add(int(a))
    return [sorted(a) for a in A]


def triangles(Qd):
    return np.concatenate([Qd[:, [0, 1, 2]], Qd[:, [0, 2, 3]]])


def interpolate(P, free, neigh):
    """Free vertices: average of their neighbours (exact solve)."""
    idx = np.flatnonzero(free)
    rank = -np.ones(len(P), int)
    rank[idx] = np.arange(len(idx))
    A = np.zeros((len(idx), len(idx)))
    B = np.zeros((len(idx), 3))
    for r, i in enumerate(idx):
        A[r, r] = len(neigh[i])
        for j in neigh[i]:
            if free[j]:
                A[r, rank[j]] -= 1.0
            else:
                B[r] += P[j]
    P = P.copy()
    P[idx] = np.linalg.solve(A, B)
    return P


def symmetric(P, sym):
    return 0.5 * (P + P[sym] * MIR)


def relax(P, moves, neigh, sym, targets, perimeters=40, stage=0.6, log=print):
    """targets: list of (vertex mask, V, F): which surface each group lands on."""
    P = P.copy()
    idx = np.flatnonzero(moves)
    # only the faces near the moving vertices serve as targets
    prep = []
    for m, V, F in targets:
        k = np.flatnonzero(m & moves)
        if len(k):
            lo, hi = P[k].min(0) - 6.0, P[k].max(0) + 6.0
            c = V[F].mean(1)
            Fp = F[((c > lo) & (c < hi)).all(1)]
            prep.append((k, V, Fp, V[Fp].mean(1)))
    for t in range(perimeters):
        avg = np.array([P[neigh[i]].mean(0) for i in idx])
        P[idx] = (1 - stage) * P[idx] + stage * avg
        for k, V, F, Cf in prep:
            P[k] = near_fast(P[k], V, F, Cf)
        P = symmetric(P, sym)
    return P


def gap(P, Qd, V, F, mask=None):
    """Distance from the reference to the new mesh (mm): median, 95th percentile, maximum."""
    T = triangles(Qd)
    S = V[np.unique(F)]
    if mask is not None:
        S = S[mask(S)]
    d = G.plus_near(S, P, T)[3] * 10
    return float(np.median(d)), float(np.percentile(d, 95)), float(d.max())


def near_fast(Q, V, F, C=None, k=20, block=256, margin=3.0):
    """Closest point on (V, F) among the k faces with the closest centre, searched in the neighbourhood
    of each batch of points: enough for a point already near a surface with small faces, and much
    faster than the exhaustive search."""
    if C is None:
        C = V[F].mean(1)
    out = np.zeros_like(Q)
    order = np.lexsort((Q[:, 0] > 0, np.round(Q[:, 1] / 4.0)))
    for s in range(0, len(Q), block):
        ii = order[s:s + block]
        q = Q[ii]
        m = margin
        while True:
            sel = np.flatnonzero(((C > q.min(0) - m) & (C < q.max(0) + m)).all(1))
            if len(sel) >= 1 or m > 100:
                break
            m *= 2
        Cs, Fs = C[sel], F[sel]
        kk = min(k, len(sel))
        d2 = ((q[:, None, :] - Cs[None, :, :]) ** 2).sum(-1)
        candidate = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        pq, _ = G._pp_triangle(q[:, None, :], V[Fs[candidate, 0]], V[Fs[candidate, 1]], V[Fs[candidate, 2]])
        j = np.linalg.norm(pq - q[:, None, :], axis=2).argmin(1)
        out[ii] = pq[np.arange(len(q)), j]
    return out


def dilate(mask, neigh, ranks):
    m = mask.copy()
    for _ in range(ranks):
        k = np.flatnonzero(m)
        for i in k:
            m[neigh[i]] = True
    return m


def soften(P, T, free, neigh, sym, threshold=38.0, passes=12, perimeters=4, stage=0.5, ranks=1, log=print):
    """Removes sharp edges: the vertices of edges over `threshold` degrees and their neighbours are smoothed
    (average of the neighbours, without going back to the reference), until none are left."""
    P = P.copy()
    for p in range(passes):
        ang, ar = G.dihedrals(P, T)
        vif = ar[ang > threshold]
        m = np.zeros(len(P), bool)
        m[vif.ravel()] = True
        m = dilate(m, neigh, ranks) & free
        m |= m[sym]
        if not m.any():
            break
        idx = np.flatnonzero(m)
        for _ in range(perimeters):
            avg = np.array([P[neigh[i]].mean(0) for i in idx])
            P[idx] = (1 - stage) * P[idx] + stage * avg
            P = symmetric(P, sym)
    ang, ar = G.dihedrals(P, T)
    rest = ar[ang > threshold]
    rest = rest[free[rest].all(1)]
    log(f"smoothing: {p + 1} passes, {len(rest)} edges over {threshold:.0f}° remain outside hands and feet")
    return P

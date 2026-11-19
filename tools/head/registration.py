"""Correspondence between two face skins with different topologies.

1. Pairs of computed landmarks (corners of the eyes and mouth, nose, chin, forehead, ears,
   eyebrows) and pairs along the rims of the eye and mouth holes (same normalised arc
   length, starting at the outer corner).
2. Initial thin-plate (TPS) deformation on these pairs.
3. Non-rigid ICP: at each step, target = closest point on B (rejected if too far, if the
   normals diverge or if the point falls on the boundary of B); we solve
   min Σ w_i |x_i - c_i|² + λ Σ_landmarks |x_l - b_l|² + β |L (x - X0)|²
   with decreasing β (stiff then soft, in the manner of Amberg and co-authors, 2007, but on a
   displacement field regularised by the Laplacian rather than local affines).
4. Final projection onto B: φ(v) = point of B, face and barycentric coordinates.
"""
import numpy as np

import geom as G


def _abscissa(P, closed=True):
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1) if closed else np.r_[np.linalg.norm(np.diff(P, axis=0), axis=1), 0]
    s = np.r_[0, np.cumsum(d)[:-1]]
    return s / d.sum()


def pairs_loop(VA, ba, ia0, VB, bb, ib0, direction_a, direction_b):
    """For each vertex of loop A, the point of loop B at the same arc length.
    ia0/ib0: starting vertex (outer corner), direction: makes the loop orders comparable."""
    def prepare(b, i0, direction):
        b = list(b)
        k = b.index(i0)
        b = b[k:] + b[:k]
        if direction < 0:
            b = [b[0]] + b[1:][::-1]
        return np.array(b)
    a = prepare(ba, ia0, direction_a)
    b = prepare(bb, ib0, direction_b)
    sa = _abscissa(VA[a])
    sb = _abscissa(VB[b])
    PB = VB[b]
    PB2 = np.vstack([PB, PB[:1]])
    sb2 = np.r_[sb, 1.0]
    tgt = np.stack([np.interp(sa, sb2, PB2[:, k]) for k in range(3)], 1)
    return a, tgt


def direction_loop(V, b, cx):
    """+1 if the loop runs counter-clockwise seen from the front (x, y), -1 otherwise."""
    P = V[b][:, :2]
    a = 0.5 * float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))
    return 1 if a > 0 else -1


def pairs_landmarks(A, XA, B, XB):
    """Pairs (vertex index in A, target position on B), in the host frame."""
    ia, target = [], []
    for k, va in A.landmarks.items():
        ia.append(va)
        target.append(XB[B.landmarks[k]])
    # eyebrows: skin vertex closest to the eyebrow point, on each side
    for k, p in A.points_eyebrows.items():
        pa = A._host(p)
        pb = B._host(B.points_eyebrows[k])
        j = int(np.argmin(np.linalg.norm(XA - pa, axis=1)))
        jb = int(np.argmin(np.linalg.norm(XB - pb, axis=1)))
        ia.append(j)
        target.append(XB[jb])
    ia_r = np.array(ia)
    target_r = np.array(target)
    # hole rims
    ib, cb = [], []
    for side in ("g", "d"):
        ba, bb = A.loop_eye[side], B.loop_eye[side]
        a, t = pairs_loop(XA, ba, A.landmarks["eye_%s_ext" % side], XB, bb,
                             B.landmarks["eye_%s_ext" % side], direction_loop(XA, ba, 0),
                             direction_loop(XB, bb, 0))
        ib.append(a)
        cb.append(t)
    a, t = pairs_loop(XA, A.loop_mouth, A.landmarks["mouth_l"], XB, B.loop_mouth,
                         B.landmarks["mouth_l"], direction_loop(XA, A.loop_mouth, 0),
                         direction_loop(XB, B.loop_mouth, 0))
    ib.append(a)
    cb.append(t)
    return ia_r, target_r, np.concatenate(ib), np.vstack(cb)


ANCHORING = 1e-4


def register(A, XA, B, XB, log=None, steps=(30.0, 10.0, 3.0, 1.0, 0.3), iters=4, on_place=False):
    """Registers the skin of A (positions XA, topology A.F) onto the skin of B (XB, B.F).
    on_place: the landmarks of A do not go to those of B but to the point of B right beneath
    them (A's face keeps the place it was given on B's head).
    Returns a dict: X (A registered), phi (projection onto B), face, bary, distances, log."""
    n = len(XA)
    il, tl, ib, tb = pairs_landmarks(A, XA, B, XB)
    if on_place:
        tl = G.plus_near(XA[il], XB, B.F)[0]
        tb = G.plus_near(XA[ib], XB, B.F)[0]
    src = np.vstack([XA[il], XA[ib]])
    dst = np.vstack([tl, tb])
    # TPS: deduplicate the sources
    _, u = np.unique(np.round(src, 6), axis=0, return_index=True)
    X0 = G.tps(src[u], dst[u], XA, smooth=1e-3)
    L = G.laplacian(n, A.F)
    LtL = G.ltl(L)
    nB = G.normals_vertices(XB, B.F)
    nfB = G.normals_faces(XB, B.F)
    borderB = np.zeros(len(XB), dtype=bool)
    for b in B.loops:
        borderB[b] = True
    ue, _, _ = G.edges(B.F)
    lm_avg = float(np.linalg.norm(XB[ue[:, 0]] - XB[ue[:, 1]], axis=1).mean())
    PBb = XB[borderB]
    # landmark weights
    M = np.zeros(n)
    T = np.zeros((n, 3))
    np.add.at(M, il, 1.0)
    np.add.at(T, il, tl)
    np.add.at(M, ib, 0.5)
    np.add.at(T, ib, 0.5 * tb)
    Tm = np.where(M[:, None] > 0, T / np.maximum(M, 1e-9)[:, None], 0)
    X = X0.copy()
    track = []
    for beta in steps:
        for _ in range(iters):
            c, f, bary, d = G.plus_near(X, XB, B.F)
            nA = G.normals_vertices(X, A.F)
            ok = (nA * nfB[f]).sum(1) > 0.5
            ok &= d < max(0.25 * A.iod, 6 * lm_avg)
            # reject points that fall on the boundary of B (areas B does not cover)
            db = np.linalg.norm(c[:, None] - PBb[None], axis=2).min(1)
            ok &= db > 0.5 * lm_avg
            w = ok.astype(float)
            lam = 10.0
            # weak anchoring to X0: a detached skin island (eyelid strip) with no point kept
            # otherwise has nothing holding it (the Laplacian leaves its translation free)
            # and drifts away; this way it keeps the place the landmarks give it
            Amat = np.diag(w + lam * M + ANCHORING) + beta * LtL
            rhs = w[:, None] * c + lam * M[:, None] * Tm + ANCHORING * X0 + beta * (LtL @ X0)
            X = np.linalg.solve(Amat, rhs)
            track.append({"beta": beta, "kept_points": int(ok.sum()), "dist_mean": round(float(d[ok].mean()), 5),
                          "dist_p95": round(float(np.percentile(d[ok], 95)), 5)})
    c, f, bary, d = G.plus_near(X, XB, B.F)
    if log:
        log("registration: %d steps, final mean distance %.4f (p95 %.4f)" % (
            len(track), d.mean(), np.percentile(d, 95)))
    return {"X": X, "X0": X0, "phi": c, "face": f, "bary": bary, "dist": d, "track": track,
            "mean_edge_B": lm_avg}

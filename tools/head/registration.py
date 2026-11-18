"""Correspondence between two face skins with different topologies.

1. Pairs of computed landmarks (corners of the eyes and mouth, nose, chin, forehead, ears,
   eyebrows) and pairs along the rims of the eye and mouth holes (same normalised arc
   length, starting at the outer corner).
2. Initial thin-plate (TPS) deformation on these pairs.
3. Non-rigid ICP: at each step, target = closest point on B (rejected if too far, if the
   normals diverge or if the point falls on the boundary of B); we solve
   min Σ w_i |x_i - c_i|² + λ Σ_landmarks |x_l - b_l|² + β |L (x - X0)|²
   with decreasing β (stiff then soft, in the manner of Amberg et al. 2007, but on a
   displacement field regularised by the Laplacian rather than local affines).
4. Final projection onto B: φ(v) = point of B, face and barycentric coordinates.
"""
import numpy as np

import geom as G


def _abscisse(P, ferme=True):
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1) if ferme else np.r_[np.linalg.norm(np.diff(P, axis=0), axis=1), 0]
    s = np.r_[0, np.cumsum(d)[:-1]]
    return s / d.sum()


def paires_boucle(VA, ba, ia0, VB, bb, ib0, sens_a, sens_b):
    """For each vertex of loop A, the point of loop B at the same arc length.
    ia0/ib0: starting vertex (outer corner), sens: makes the loop orders comparable."""
    def preparer(b, i0, sens):
        b = list(b)
        k = b.index(i0)
        b = b[k:] + b[:k]
        if sens < 0:
            b = [b[0]] + b[1:][::-1]
        return np.array(b)
    a = preparer(ba, ia0, sens_a)
    b = preparer(bb, ib0, sens_b)
    sa = _abscisse(VA[a])
    sb = _abscisse(VB[b])
    PB = VB[b]
    PB2 = np.vstack([PB, PB[:1]])
    sb2 = np.r_[sb, 1.0]
    tgt = np.stack([np.interp(sa, sb2, PB2[:, k]) for k in range(3)], 1)
    return a, tgt


def sens_boucle(V, b, cx):
    """+1 if the loop runs counter-clockwise seen from the front (x, y), -1 otherwise."""
    P = V[b][:, :2]
    a = 0.5 * float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1)))
    return 1 if a > 0 else -1


def paires_reperes(A, XA, B, XB):
    """Pairs (vertex index in A, target position on B), in the host frame."""
    ia, cible = [], []
    for k, va in A.reperes.items():
        ia.append(va)
        cible.append(XB[B.reperes[k]])
    # eyebrows: skin vertex closest to the eyebrow point, on each side
    for k, p in A.points_sourcils.items():
        pa = A._hote(p)
        pb = B._hote(B.points_sourcils[k])
        j = int(np.argmin(np.linalg.norm(XA - pa, axis=1)))
        jb = int(np.argmin(np.linalg.norm(XB - pb, axis=1)))
        ia.append(j)
        cible.append(XB[jb])
    ia_r = np.array(ia)
    cible_r = np.array(cible)
    # hole rims
    ib, cb = [], []
    for cote in ("g", "d"):
        ba, bb = A.boucle_oeil[cote], B.boucle_oeil[cote]
        a, t = paires_boucle(XA, ba, A.reperes["oeil_%s_ext" % cote], XB, bb,
                             B.reperes["oeil_%s_ext" % cote], sens_boucle(XA, ba, 0),
                             sens_boucle(XB, bb, 0))
        ib.append(a)
        cb.append(t)
    a, t = paires_boucle(XA, A.boucle_bouche, A.reperes["bouche_g"], XB, B.boucle_bouche,
                         B.reperes["bouche_g"], sens_boucle(XA, A.boucle_bouche, 0),
                         sens_boucle(XB, B.boucle_bouche, 0))
    ib.append(a)
    cb.append(t)
    return ia_r, cible_r, np.concatenate(ib), np.vstack(cb)


ANCRAGE = 1e-4


def recaler(A, XA, B, XB, journal=None, etapes=(30.0, 10.0, 3.0, 1.0, 0.3), iters=4, sur_place=False):
    """Registers the skin of A (positions XA, topology A.F) onto the skin of B (XB, B.F).
    sur_place: the landmarks of A do not go to those of B but to the point of B right beneath
    them (A's face keeps the place it was given on B's head).
    Returns a dict: X (A registered), phi (projection onto B), face, bary, distances, log."""
    n = len(XA)
    il, tl, ib, tb = paires_reperes(A, XA, B, XB)
    if sur_place:
        tl = G.plus_proche(XA[il], XB, B.F)[0]
        tb = G.plus_proche(XA[ib], XB, B.F)[0]
    src = np.vstack([XA[il], XA[ib]])
    dst = np.vstack([tl, tb])
    # TPS: deduplicate the sources
    _, u = np.unique(np.round(src, 6), axis=0, return_index=True)
    X0 = G.tps(src[u], dst[u], XA, lisse=1e-3)
    L = G.laplacien(n, A.F)
    LtL = G.ltl(L)
    nB = G.normales_sommets(XB, B.F)
    nfB = G.normales_faces(XB, B.F)
    bordB = np.zeros(len(XB), dtype=bool)
    for b in B.boucles:
        bordB[b] = True
    ue, _, _ = G.aretes(B.F)
    lm_moy = float(np.linalg.norm(XB[ue[:, 0]] - XB[ue[:, 1]], axis=1).mean())
    PBb = XB[bordB]
    # landmark weights
    M = np.zeros(n)
    T = np.zeros((n, 3))
    np.add.at(M, il, 1.0)
    np.add.at(T, il, tl)
    np.add.at(M, ib, 0.5)
    np.add.at(T, ib, 0.5 * tb)
    Tm = np.where(M[:, None] > 0, T / np.maximum(M, 1e-9)[:, None], 0)
    X = X0.copy()
    suivi = []
    for beta in etapes:
        for _ in range(iters):
            c, f, bary, d = G.plus_proche(X, XB, B.F)
            nA = G.normales_sommets(X, A.F)
            ok = (nA * nfB[f]).sum(1) > 0.5
            ok &= d < max(0.25 * A.iod, 6 * lm_moy)
            # reject points that fall on the boundary of B (areas B does not cover)
            db = np.linalg.norm(c[:, None] - PBb[None], axis=2).min(1)
            ok &= db > 0.5 * lm_moy
            w = ok.astype(float)
            lam = 10.0
            # weak anchoring to X0: a detached skin island (eyelid strip) with no point kept
            # otherwise has nothing holding it (the Laplacian leaves its translation free)
            # and drifts away; this way it keeps the place the landmarks give it
            Amat = np.diag(w + lam * M + ANCRAGE) + beta * LtL
            rhs = w[:, None] * c + lam * M[:, None] * Tm + ANCRAGE * X0 + beta * (LtL @ X0)
            X = np.linalg.solve(Amat, rhs)
            suivi.append({"beta": beta, "retenus": int(ok.sum()), "dist_moy": round(float(d[ok].mean()), 5),
                          "dist_p95": round(float(np.percentile(d[ok], 95)), 5)})
    c, f, bary, d = G.plus_proche(X, XB, B.F)
    if journal:
        journal("registration: %d steps, final mean distance %.4f (p95 %.4f)" % (
            len(suivi), d.mean(), np.percentile(d, 95)))
    return {"X": X, "X0": X0, "phi": c, "face": f, "bary": bary, "dist": d, "suivi": suivi,
            "arete_moy_B": lm_moy}

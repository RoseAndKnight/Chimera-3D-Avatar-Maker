"""Basic geometry for the head retargeter (numpy only, no scipy).

All functions work on welded triangle meshes (V: (n, 3), F: (m, 3)).
"""
import heapq

import numpy as np


# ------------------------------------------------------------------ welding and topology

def souder(pos, tol=1e-5):
    """Classes of coincident vertices. Returns (rep, inv): rep[k] = one raw vertex of class k,
    inv[i] = class of raw vertex i."""
    q = np.round(np.asarray(pos) / tol).astype(np.int64)
    _, rep, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return rep, inv.ravel()


def sous_maillage(F_global, garder_faces):
    """Selected faces -> (vertices used (global indices), reindexed faces)."""
    Fs = F_global[garder_faces]
    vs, inv = np.unique(Fs, return_inverse=True)
    return vs, inv.reshape(-1, 3)


def aretes(F):
    """Unique (sorted) edges, number of faces per edge, and for each face its 3 edges."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    Es = np.sort(E, axis=1)
    u, inv, c = np.unique(Es, axis=0, return_inverse=True, return_counts=True)
    fe = inv.ravel().reshape(3, -1).T
    return u, c, fe


def voisins(n, F):
    u, _, _ = aretes(F)
    adj = [[] for _ in range(n)]
    for a, b in u:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def boucles_bord(F):
    """Boundary loops oriented like the faces (half-edges without an opposite), each an
    ordered list of vertices. "Butterfly" vertices (two loops) are tolerated."""
    he = {}
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            he[(int(u), int(v))] = True
    suiv = {}
    for (u, v) in he:
        if (v, u) not in he:
            suiv.setdefault(u, []).append(v)
    vus = set()
    boucles = []
    for u0 in list(suiv):
        for v0 in suiv[u0]:
            if (u0, v0) in vus:
                continue
            b = [u0]
            u, v = u0, v0
            while (u, v) not in vus:
                vus.add((u, v))
                if v == u0:
                    break
                b.append(v)
                cand = [w for w in suiv.get(v, []) if (v, w) not in vus]
                if not cand:
                    break
                u, v = v, cand[0]
            boucles.append(b)
    return boucles


def composantes_faces(F, garder):
    """Connected components (through edges) of the faces where garder is true. Returns an array
    of labels (-1 outside the selection)."""
    _, _, fe = aretes(F)
    idx = np.where(garder)[0]
    lab = np.full(len(F), -1)
    par_arete = {}
    for f in idx:
        for e in fe[f]:
            par_arete.setdefault(int(e), []).append(int(f))
    k = 0
    for f0 in idx:
        if lab[f0] >= 0:
            continue
        pile = [f0]
        lab[f0] = k
        while pile:
            f = pile.pop()
            for e in fe[f]:
                for g in par_arete[int(e)]:
                    if lab[g] < 0:
                        lab[g] = k
                        pile.append(g)
        k += 1
    return lab


def remplir_disque(F, sel):
    """Face selection -> its largest component, plus everything it encloses
    (components of the complement other than the largest)."""
    lab = composantes_faces(F, sel)
    if lab.max() < 0:
        return sel.copy()
    k = np.bincount(lab[lab >= 0]).argmax()
    s = lab == k
    lc = composantes_faces(F, ~s)
    if lc.max() >= 0:
        dehors = np.bincount(lc[lc >= 0]).argmax()
        s = s | ((lc >= 0) & (lc != dehors))
    return s


# ------------------------------------------------------------------ normals and quality

def normales_faces(V, F, unitaires=True):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if unitaires:
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    return n


def normales_sommets(V, F):
    nf = normales_faces(V, F, unitaires=False)
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], nf)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def diedres(V, F):
    """Angle (degrees) between the normals of the two faces of each interior edge, and the edges."""
    u, c, fe = aretes(F)
    nf = normales_faces(V, F)
    f_de = [[] for _ in range(len(u))]
    for f in range(len(F)):
        for e in fe[f]:
            f_de[e].append(f)
    inter = np.array([len(x) == 2 for x in f_de])
    a = np.array([x[0] for x, i in zip(f_de, inter) if i], dtype=np.int64)
    b = np.array([x[1] for x, i in zip(f_de, inter) if i], dtype=np.int64)
    cosv = np.clip((nf[a] * nf[b]).sum(1), -1, 1)
    return np.degrees(np.arccos(cosv)), u[inter]


def qualite(V, F):
    """Project metrics: dihedral q99, share of edges > 15°, share of valence ≠ 6 (interior
    vertices), boundary edges and non-manifold edges."""
    u, c, _ = aretes(F)
    d, _ = diedres(V, F)
    bord = np.zeros(len(V), dtype=bool)
    bord[u[c == 1].ravel()] = True
    val = np.bincount(u.ravel(), minlength=len(V))
    utiles = np.zeros(len(V), dtype=bool)
    utiles[F.ravel()] = True
    inter = utiles & ~bord
    return {
        "diedre_q99_deg": round(float(np.percentile(d, 99)), 2) if len(d) else 0.0,
        "part_aretes_sup15": round(float((d > 15).mean()), 4) if len(d) else 0.0,
        "part_valence_diff6": round(float((val[inter] != 6).mean()), 4) if inter.any() else 0.0,
        "aretes_bord": int((c == 1).sum()),
        "aretes_non_varietes": int((c > 2).sum()),
        "n_sommets": int(utiles.sum()),
        "n_faces": int(len(F)),
    }


def _tac(V, F, ef, fe, E):
    """For each interior edge: its faces, apexes, the current local total absolute curvature and the one
    after a swap (see basculer_diagonales). Edges that cannot be swapped get an infinite swapped cost."""
    inter = np.flatnonzero(ef[:, 1] >= 0)
    fa, fb = ef[inter, 0], ef[inter, 1]
    a, b = E[inter, 0], E[inter, 1]
    sens = np.zeros(len(inter), bool)                       # fa holds a -> b ?
    for k in range(3):
        sens |= (F[fa, k] == a) & (F[fa, (k + 1) % 3] == b)
    A, B = np.where(sens, a, b), np.where(sens, b, a)       # fa = (A, B, c), fb = (B, A, d)
    c = F[fa].sum(1) - a - b
    d = F[fb].sum(1) - a - b

    def n_(p, q, r):
        n = np.cross(V[q] - V[p], V[r] - V[p])
        return n, np.linalg.norm(n, axis=1)
    nf, af = n_(F[:, 0], F[:, 1], F[:, 2])
    nf = nf / np.maximum(af, 1e-30)[:, None]

    def ang(n1, n2):
        return np.arccos(np.clip((n1 * n2).sum(1), -1, 1))

    def lg(p, q):
        return np.linalg.norm(V[p] - V[q], axis=1)

    def voisine(f, p, q):
        """Face across edge (p, q) of face f (-1 on a boundary), via the edge ids of f."""
        out = np.full(len(f), -1)
        for k in range(3):
            u, v = F[f, k], F[f, (k + 1) % 3]
            m = ((u == p) & (v == q)) | ((u == q) & (v == p))
            e = fe[f, k][m]
            g = ef[e]
            out[m] = np.where(g[:, 0] == f[m], g[:, 1], g[:, 0])
        return out

    def terme(n, g, p, q):
        ok = g >= 0
        t = np.zeros(len(n))
        t[ok] = ang(n[ok], nf[g[ok]]) * lg(p[ok], q[ok])
        return t
    g1, g2, g3, g4 = voisine(fa, B, c), voisine(fa, c, A), voisine(fb, A, d), voisine(fb, d, B)
    na, nb = nf[fa], nf[fb]
    act = (ang(na, nb) * lg(a, b) + terme(na, g1, B, c) + terme(na, g2, c, A)
           + terme(nb, g3, A, d) + terme(nb, g4, d, B))
    n1, s1 = n_(c, A, d)                                    # swapped faces, same winding as fa and fb
    n2, s2 = n_(d, B, c)
    n1, n2 = n1 / np.maximum(s1, 1e-30)[:, None], n2 / np.maximum(s2, 1e-30)[:, None]
    per = (ang(n1, n2) * lg(c, d) + terme(n2, g1, B, c) + terme(n1, g2, c, A)
           + terme(n1, g3, A, d) + terme(n2, g4, d, B))
    # a swap must not fold, flip or thin a face: both new faces face the way the old pair faced, their
    # crease stays under 90°, and their smallest angle stays above 20° or above the old pair's smallest
    def angle_min(p, q, r):
        out = np.full(len(p), np.pi)
        for u, v, w in ((p, q, r), (q, r, p), (r, p, q)):
            e1, e2 = V[v] - V[u], V[w] - V[u]
            cs = (e1 * e2).sum(1) / np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-30)
            out = np.minimum(out, np.arccos(np.clip(cs, -1, 1)))
        return out
    avant = np.minimum(angle_min(A, B, c), angle_min(B, A, d))
    apres = np.minimum(angle_min(c, A, d), angle_min(d, B, c))
    moy = na + nb
    bon = ((c != d) & ((n1 * moy).sum(1) > 0) & ((n2 * moy).sum(1) > 0) & ((n1 * n2).sum(1) > 0)
           & (apres >= np.minimum(avant, np.radians(20.0))))
    per[~bon] = np.inf
    voisines = np.stack([fa, fb, g1, g2, g3, g4], 1)
    return inter, A, B, c, d, act, per, voisines


def basculer_diagonales(V, F, gain_relatif=1e-3, gain_min=0.0, passes=60):
    """Swap the diagonals that cross a fold instead of following it.

    A quad split along the wrong diagonal, or two triangles whose shared edge runs across a crease,
    show a sharp crease and a zigzag in grazing light or toon shading, although the vertices are right.
    The edge (a, b) between (A, B, c) and (B, A, d) is swapped to (c, d) when that lowers the local total
    absolute mean curvature (sum of |dihedral| × length over the edge and its four neighbours; Dyn,
    Hormann, Kim & Levin 2001, "Optimizing 3D triangulations using discrete curvature analysis") by more
    than `gain_relatif` of its value and by more than `gain_min` (units of V × radians). Vertices are not
    moved: only the connectivity changes, with each face keeping its winding. Greedy, best gain first,
    on independent edges, repeated until nothing improves.

    F uses raw indices: an edge whose two sides use different vertices (UV seam) is a boundary and stays.
    Returns (new faces in the same order, number of swaps)."""
    F = np.array(F, dtype=np.int64)
    V = np.asarray(V, float)
    total = 0
    for _ in range(passes):
        E, c_, fe = aretes(F)
        # the two faces of each edge (-1 for a boundary; non-manifold edges are left alone)
        ef = -np.ones((len(E), 2), np.int64)
        ordre = np.argsort(fe.ravel(), kind="stable")
        faces_de = np.repeat(np.arange(len(F)), 3)[ordre]
        aret = fe.ravel()[ordre]
        i1 = np.flatnonzero(np.r_[True, aret[1:] != aret[:-1]])
        ef[aret[i1], 0] = faces_de[i1]
        i2 = i1 + 1
        i2 = i2[i2 < len(aret)]
        i2 = i2[aret[i2] == aret[i2 - 1]]
        ef[aret[i2], 1] = faces_de[i2]
        ef[c_ > 2] = -1
        inter, A, B, c, d, act, per, vois = _tac(V, F, ef, fe, E)
        gain = act - per
        cand = np.flatnonzero((gain > gain_relatif * act) & (gain > gain_min) & np.isfinite(per))
        if not len(cand):
            break
        existe = set(map(tuple, E.tolist()))
        pris = np.zeros(len(F), bool)
        n = 0
        for k in cand[np.argsort(-gain[cand])]:
            v = vois[k]
            v = v[v >= 0]
            if pris[v].any():
                continue
            cd = (min(c[k], d[k]), max(c[k], d[k]))
            if cd in existe:                  # c and d already joined: the swap would make a double edge
                continue
            fa, fb = vois[k, 0], vois[k, 1]
            F[fa] = (c[k], A[k], d[k])
            F[fb] = (d[k], B[k], c[k])
            existe.add(cd)
            pris[v] = True
            n += 1
        total += n
        if not n:
            break
    return F, total


# ------------------------------------------------------------------ closest point

def _pp_triangle(p, a, b, c):
    """Closest point to p on triangle abc (vectorised, Ericson 5.1.5).
    Returns (q, bary)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den
    w = vc / den
    bary = np.stack([1 - v - w, v, w], -1)
    # regions
    m = (d1 <= 0) & (d2 <= 0)
    bary[m] = (1, 0, 0)
    m2 = (d3 >= 0) & (d4 <= d3)
    bary[m2] = (0, 1, 0)
    m3 = (d6 >= 0) & (d5 <= d6)
    bary[m3] = (0, 0, 1)
    m4 = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~(m | m2 | m3)
    t = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3)
    bary[m4] = np.stack([1 - t[m4], t[m4], 0 * t[m4]], -1)
    m5 = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~(m | m2 | m3 | m4)
    t = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6)
    bary[m5] = np.stack([1 - t[m5], 0 * t[m5], t[m5]], -1)
    m6 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~(m | m2 | m3 | m4 | m5)
    t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1e-30, (d4 - d3) + (d5 - d6))
    bary[m6] = np.stack([0 * t[m6], 1 - t[m6], t[m6]], -1)
    q = bary[..., 0:1] * a + bary[..., 1:2] * b + bary[..., 2:3] * c
    return q, bary


def _groupes(Q, taille):
    """Indices of Q cut into spatially compact groups of at most `taille` points (median splits on the
    longest axis)."""
    pile, res = [np.arange(len(Q))], []
    while pile:
        idx = pile.pop()
        if len(idx) <= taille:
            res.append(idx)
            continue
        P = Q[idx]
        ax = int(np.argmax(P.max(0) - P.min(0)))
        m = len(idx) // 2
        o = np.argpartition(P[:, ax], m)
        pile += [idx[o[:m]], idx[o[m:]]]
    return res


def plus_proche(Q, V, F, k=24, bloc=256):
    """For each point of Q: closest point on the mesh (V, F).
    Returns (point, face, bary, distance). Exact: the points are taken by spatially compact groups;
    for a group, a first pass on the faces nearest to its box gives an upper bound of every point's
    distance, then only the faces whose bounding sphere comes within that bound of the box are searched
    (_plus_proche_toutes), instead of every face of the mesh for every point."""
    Q = np.asarray(Q, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    if len(Q) == 0 or len(F) <= 4 * max(k, 64):
        return _plus_proche_toutes(Q, V, F, k, bloc)
    C = V[F].mean(1)
    R = np.linalg.norm(V[F] - C[:, None, :], axis=2).max(1)
    out_q = np.zeros_like(Q)
    out_f = np.zeros(len(Q), dtype=np.int64)
    out_b = np.zeros((len(Q), 3))
    out_d = np.zeros(len(Q))
    n0 = 4 * max(k, 64)
    for idx in _groupes(Q, bloc):
        q = Q[idx]
        lo, hi = q.min(0), q.max(0)
        # distance from each face centre to the group's box: a face is at least that minus its radius away
        dbox = np.linalg.norm(np.maximum(0.0, np.maximum(lo - C, C - hi)), axis=1) - R
        pres = np.argpartition(dbox, n0 - 1)[:n0]
        _, _, _, d0 = _plus_proche_toutes(q, V, F[pres], k, bloc)
        tau = d0.max()
        sel = np.where(dbox <= tau * (1 + 1e-9) + 1e-12)[0]
        pq, f, b, d = _plus_proche_toutes(q, V, F[sel], k, bloc)
        out_q[idx], out_f[idx], out_b[idx], out_d[idx] = pq, sel[f], b, d
    return out_q, out_f, out_b, out_d


def _plus_proche_toutes(Q, V, F, k=24, bloc=256):
    """plus_proche against every face of (V, F): candidates are the k faces with the closest centre,
    then every face whose bounding sphere could beat the best candidate."""
    Q = np.asarray(Q, dtype=np.float64)
    C = V[F].mean(1)
    k = min(k, len(F))
    out_q = np.zeros_like(Q)
    out_f = np.zeros(len(Q), dtype=np.int64)
    out_b = np.zeros((len(Q), 3))
    out_d = np.zeros(len(Q))
    # radius of each face (centre -> farthest vertex) for a safe bound
    R = np.linalg.norm(V[F] - C[:, None, :], axis=2).max(1)
    for s in range(0, len(Q), bloc):
        q = Q[s:s + bloc]
        d2 = ((q[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        dc = np.sqrt(d2)
        cand = np.argpartition(dc, k - 1, axis=1)[:, :k]
        # bound: any face whose centre - radius < k-th distance may be better
        dk = np.take_along_axis(dc, cand, 1).max(1)
        a, b, c = V[F[cand, 0]], V[F[cand, 1]], V[F[cand, 2]]
        pq, bb = _pp_triangle(q[:, None, :], a, b, c)
        dd = np.linalg.norm(pq - q[:, None, :], axis=2)
        j = dd.argmin(1)
        r = np.arange(len(q))
        best = dd[r, j]
        # check: faces outside the candidates that could beat best
        sus = (dc - R[None, :]) < best[:, None]
        sus[r[:, None], cand] = False
        bq, bf, bba = pq[r, j], cand[r, j], bb[r, j]
        # every (point, face) pair at once, by slices: per point, the closest face (the lowest index among
        # equals) replaces the best candidate when strictly closer
        ii, ff = np.nonzero(sus)
        for t in range(0, len(ii), 65536):
            i_, f_ = ii[t:t + 65536], ff[t:t + 65536]
            p2, b2 = _pp_triangle(q[i_], V[F[f_, 0]], V[F[f_, 1]], V[F[f_, 2]])
            d2_ = np.linalg.norm(p2 - q[i_], axis=1)
            o = np.lexsort((f_, d2_, i_))
            prem = o[np.r_[True, i_[o][1:] != i_[o][:-1]]]
            mieux = prem[d2_[prem] < best[i_[prem]]]
            i2 = i_[mieux]
            best[i2], bq[i2], bf[i2], bba[i2] = d2_[mieux], p2[mieux], f_[mieux], b2[mieux]
        out_q[s:s + bloc], out_f[s:s + bloc], out_b[s:s + bloc], out_d[s:s + bloc] = bq, bf, bba, best
    return out_q, out_f, out_b, out_d


def hausdorff(VA, FA, VB, FB, selA=None, selB=None):
    """Surface distances A->B and B->A (vertices against triangles). Returns a dict."""
    qa = VA if selA is None else VA[selA]
    qb = VB if selB is None else VB[selB]
    _, _, _, dab = plus_proche(qa, VB, FB)
    _, _, _, dba = plus_proche(qb, VA, FA)
    return {"a_vers_b_max": float(dab.max()), "a_vers_b_p95": float(np.percentile(dab, 95)),
            "a_vers_b_moy": float(dab.mean()), "b_vers_a_max": float(dba.max()),
            "b_vers_a_p95": float(np.percentile(dba, 95)), "b_vers_a_moy": float(dba.mean()),
            "hausdorff": float(max(dab.max(), dba.max()))}


# ------------------------------------------------------------------ graphs and smoothing

def dijkstra(V, adj, sources, limite=np.inf):
    d = np.full(len(V), np.inf)
    h = []
    for s in sources:
        d[s] = 0.0
        h.append((0.0, int(s)))
    heapq.heapify(h)
    while h:
        x, u = heapq.heappop(h)
        if x > d[u] or x > limite:
            continue
        for v in adj[u]:
            nx = x + float(np.linalg.norm(V[u] - V[v]))
            if nx < d[v]:
                d[v] = nx
                heapq.heappush(h, (nx, v))
    return d


def laplacien(n, F, poids=None):
    """Dense uniform Laplacian L = I - D^-1 A."""
    u, _, _ = aretes(F)
    A = np.zeros((n, n))
    A[u[:, 0], u[:, 1]] = 1
    A[u[:, 1], u[:, 0]] = 1
    deg = A.sum(1)
    L = np.eye(n) - A / np.maximum(deg, 1)[:, None]
    L[deg == 0] = 0
    return L


def ltl(L):
    """L.T @ L for a dense matrix with few non-zero values per row (a Laplacian), summed over those values
    only: the dense product costs n³ and takes tens of seconds with a NumPy built without an optimised BLAS
    (Blender's), for a few milliseconds here."""
    r, c = np.nonzero(L)
    n = L.shape[1]
    res = np.zeros((n, n))
    if not len(r):
        return res
    cnt = np.bincount(r, minlength=L.shape[0])
    m = int(cnt.max())
    debut = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = np.arange(len(r)) - debut[r]          # rank of each value in its row (np.nonzero is row-major)
    idx = np.zeros((L.shape[0], m), dtype=np.int64)
    val = np.zeros((L.shape[0], m))
    idx[r, pos], val[r, pos] = c, L[r, c]
    for a in range(m):
        for b in range(m):
            np.add.at(res, (idx[:, a], idx[:, b]), val[:, a] * val[:, b])
    return res


def lisser(n_f):
    """Smooth step 0 -> 1 (smoothstep)."""
    t = np.clip(n_f, 0, 1)
    return t * t * (3 - 2 * t)


def rotation_min(a, b):
    """Minimal rotations (n, 3, 3) taking unit vectors a onto b."""
    v = np.cross(a, b)
    c = (a * b).sum(1)
    s2 = (v * v).sum(1)
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    f = np.where(s2 > 1e-14, (1 - c) / np.maximum(s2, 1e-14), 0.5)
    R = np.eye(3)[None] + K + (K @ K) * f[:, None, None]
    return R


def tps(src, dst, pts, lisse=0.0):
    """3D thin-plate interpolation (kernel r) mapping src onto dst, applied to pts."""
    n = len(src)
    K = np.linalg.norm(src[:, None] - src[None], axis=2)
    P = np.hstack([np.ones((n, 1)), src])
    A = np.zeros((n + 4, n + 4))
    A[:n, :n] = K + lisse * np.eye(n)
    A[:n, n:] = P
    A[n:, :n] = P.T
    b = np.zeros((n + 4, 3))
    b[:n] = dst
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    Kp = np.linalg.norm(pts[:, None] - src[None], axis=2)
    return Kp @ sol[:n] + np.hstack([np.ones((len(pts), 1)), pts]) @ sol[n:]


def shepard(src, val, pts, k=12, p=2.0):
    """Average of the values of the k closest sources, weights 1/d^p."""
    out = np.zeros((len(pts),) + val.shape[1:])
    for s in range(0, len(pts), 2048):
        q = pts[s:s + 2048]
        d = np.linalg.norm(q[:, None] - src[None], axis=2)
        kk = min(k, len(src))
        j = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        dj = np.take_along_axis(d, j, 1)
        w = 1.0 / np.maximum(dj, 1e-9) ** p
        w /= w.sum(1, keepdims=True)
        out[s:s + 2048] = np.einsum("ij,ij...->i...", w, val[j])
    return out


def aire_polygone_xy(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def enveloppe_convexe_2d(P):
    P = np.unique(np.round(P, 9), axis=0)
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in P[::-1]:
        while len(hi) >= 2 and cr(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1])


def distance_signee_polygone(Q, H):
    """Signed distance (negative inside) from 2D points to a convex CCW polygon."""
    a = H
    b = np.roll(H, -1, axis=0)
    ab = b - a
    t = np.clip((((Q[:, None] - a[None]) * ab[None]).sum(-1)) / (ab * ab).sum(-1)[None], 0, 1)
    pr = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(Q[:, None] - pr, axis=2).min(1)
    cr = ab[None, :, 0] * (Q[:, None, 1] - a[None, :, 1]) - ab[None, :, 1] * (Q[:, None, 0] - a[None, :, 0])
    dedans = (cr >= 0).all(1)
    return np.where(dedans, -d, d)

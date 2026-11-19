"""Head retargeter, fully automatic.

    python3 -I tools/head/retarget_head.py shape --source A.pmx --target B.pmx --output <folder>
    python3 -I tools/head/retarget_head.py face  --source A.pmx --target B.pmx --output <folder>
        [--no-sheet] [--no-tint]

shape : A's head takes B's skull shape; A's face (eyes, nose, mouth, eyelids, eyebrows,
         their proportions, all its morphs) stays A's, on A's body. Output: A modified
         (same topology, same materials).
face  : A's face replaces B's, on B's head and body. Output: B with its face area cut out,
         A's face (skin, eyes, lashes, eyebrows, teeth, tongue) sewn in its place, A's face
         morphs carried over.

Python: Blender's (numpy). Outputs: <output>/result.pmx, measures.json, registration.npz,
sheet.png (Workbench render through the Blender queue), texture_sheet.png (textured
render). In face mode, B's skin is recoloured to the tint of A's face (tint.py: textures
copied into <output>/tint/, difference in measures.json "tint").
"""
import argparse
import copy
import json
import os
import subprocess
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))
sys.path.append(os.path.join(HERE, ".."))
import settings as R  # noqa: E402

import geom as G  # noqa: E402
import pmx_write as pmx_write  # noqa: E402
import refine as RF  # noqa: E402
import registration as RC  # noqa: E402
import head as TT  # noqa: E402
import checks as VF  # noqa: E402

# Face area: hull (front view) of the eyes, eyebrows, nose and mouth, dilated by MARGE,
# transition to the skull over BAND (in inter-ocular distances).
MARGIN = 0.06
BAND = 0.22
# Neck seam held in shape mode: return to A's shape over this radius (geodesic).
RADIUS_NECK = 0.45
# shape mode: A's hair and ears stay in place. Held skin: skin from which an outward ray
# hits hair within DISTANCE_HAIR, and skin within RADIUS_EAR (geodesic) of the most
# lateral point on each side.
DISTANCE_HAIR = 0.6
RADIUS_EAR = 0.4
# Face: A's face fades out at (LOOPS_BORDER_B + 0.5) mean edges from B's border.
LOOPS_BORDER_B = 4
# Face: morphs interpolated (zero Laplacian) over this number of rings around the seam.
LOOPS_HARMONY = 3
# Face: the patch border is carried onto the hole border; the displacement fades out over
# this number of rings of the patch.
# minimum gap between the patch border and the hole border, in edges of B
GAP_SEAM = 0.5
# local refinement of B near the seam: levels, radius (in edge lengths of B)
LEVELS_REFINEMENT = 1
R_REFINEMENT = 2.5
# Displacement smoothing (shape mode): weight of the Laplacian against fidelity to the projected point.
SMOOTHING = 2.0
# run through bash: the queue script may not be executable where the tool runs (agent sandbox)
FILE_BLENDER = os.path.join(R.REPO, "bin", "blender-queue")   # queue for Blender runs


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_json(path, d):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, path)


def _center_x0(c):
    c = np.array(c, dtype=float)
    c[0] = 0.0
    return c


def scale_face(A, B):
    """Uniform scale (no rotation) that puts A's face landmarks onto B's."""
    PA, PB = A.points_face(), B.points_face()
    k = sorted(set(PA) & set(PB))
    a = np.array([PA[x] for x in k])
    b = np.array([PB[x] for x in k])
    a = a - a.mean(0)
    b = b - b.mean(0)
    return float((a * b).sum() / (a * a).sum())


def smooth_displacement(X0, X, F, fixed, mu=SMOOTHING):
    """Smoothed displacement X - X0 (min |D - D0|² + mu |L D|², fixed vertices held): the head
    keeps its own details (small folds, edges) and takes only the overall shape from B."""
    n = len(X0)
    L = G.laplacian(n, F)
    c = np.where(fixed, 1e4, 1.0)
    D0 = X - X0
    D = np.linalg.solve(np.diag(c) + mu * G.ltl(L), c[:, None] * D0)
    return X0 + D


def rotations_local(XA, X, F):
    """Minimal rotation of each vertex between its normal before and after."""
    return G.rotation_min(G.normals_vertices(XA, F), G.normals_vertices(X, F))


# =================================================================== shared core

def projection_phong(q, f, bary, VB, FB, alpha=0.75):
    """Point projected onto B, carried onto the smooth Phong surface (Boubekeur and Alexa 2008):
    keeps the head from taking on B's facets where its edges cross B's."""
    nB = G.normals_vertices(VB, FB)
    tri = FB[f]
    out = np.zeros_like(q)
    for k in range(3):
        v = VB[tri[:, k]]
        n = nB[tri[:, k]]
        pi = q - ((q - v) * n).sum(1, keepdims=True) * n
        out += bary[:, k:k + 1] * pi
    return (1 - alpha) * q + alpha * out


def place_face(A, XA, B, XB, edge_B=False, on_place=False):
    """Registers A's skin onto B, then blends: face = A (placed), skull = B (projected).
    edge_B: A's face fades out before the outer border of B's skin (hairline, temples), so
    that the seam falls where A already lies on B."""
    rec = RC.register(A, XA, B, XB, log=log, on_place=on_place)
    rec["phi_flat"] = rec["phi"]
    rec["phi"] = projection_phong(rec["phi"], rec["face"], rec["bary"], XB, B.F)
    w = A.weight_face(MARGIN * A.iod, BAND * A.iod)
    if edge_B:
        dB = G.dijkstra(XB, B.adj, B.loop_ext)
        _, j = _plus_near_vertex(XB, rec["phi"])
        lm = rec["mean_edge_B"]
        band = BAND * A.iod * A.scale
        w = w * G.smooth((dB[j] - (LOOPS_BORDER_B + 0.5) * lm) / band)
        # likewise before the border of A's skin
        dA = G.dijkstra(XA, A.adj, A.loop_ext)
        w = w * G.smooth((dA - 2.5 * lm) / band)
    X = w[:, None] * XA + (1 - w[:, None]) * rec["phi"]
    return rec, w, X


# =================================================================== shape mode

def put_shape(A, B):
    """Shape: B scaled to A's head size (width × chin-to-skull-top height, area kept), head
    midpoints superimposed, then A's face follows the head: stretched in width and height as
    the head goes from A to B, around the middle of the head (the eyes stay at the same
    relative height, the forehead and chin keep their share of the head). Returns B's scale,
    the face's diagonal matrix K and its centre."""
    s = float(np.sqrt(A.W * A.H_crane / (B.W * B.H_crane)))
    yA = 0.5 * (A.y_skull + A.V[A.landmarks["chin"], 1])
    yB = 0.5 * (B.y_skull + B.V[B.landmarks["chin"], 1])
    cA, cB = _center_x0(A.center_face()), _center_x0(B.center_face())
    cA[1], cB[1] = yA, yB
    K = np.array([s * B.W / A.W, s * B.H_crane / A.H_crane, 1.0])
    w = A.weight_face(MARGIN * A.iod, BAND * A.iod) > 0.5
    c0 = cA.copy()
    for _ in range(3):  # B moved forward or back until it touches the stretched face
        B.put(s, cB, cA)
        Q = c0 + K * (A.V[w] - c0)
        c = G.plus_near(Q, B._host(B.V), B.F)[0]
        cA = cA - np.array([0.0, 0.0, float(np.median(c[:, 2] - Q[:, 2]))])
    B.put(s, cB, cA)
    return s, K, c0


def _rays_touch(O, D, T, dmax):
    """Does each point's ray (O, D) hit one of the triangles T (n, 3, 3) before dmax?"""
    hit = np.zeros(len(O), dtype=bool)
    e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    for k in range(0, len(O), 64):
        o, d = O[k:k + 64, None], D[k:k + 64, None]
        p = np.cross(d, e2[None])
        det = (e1[None] * p).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = 1.0 / np.where(ok, det, 1.0)
        t0 = o - T[None, :, 0]
        u = (t0 * p).sum(-1) * inv
        q = np.cross(t0, e1[None])
        v = (d * q).sum(-1) * inv
        t = (e2[None] * q).sum(-1) * inv
        hit[k:k + 64] = (ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4) & (t < dmax)).any(1)
    return hit


def vertices_hair(A):
    ih = [i for i, n in enumerate(A.names_mat) if "髮" in n or "发" in n]
    return np.unique(A.faces[np.isin(A.mat_face, ih)]), A.faces[np.isin(A.mat_face, ih)]


def zone_fixed(A):
    """Weight (1 = does not move) of A's skin hidden under its hair (an outward ray hits hair
    within DISTANCE_HAIR eye distances) and of its ears (RADIUS_EAR around the most
    lateral point), smooth transition over BAND."""
    _, Fh = vertices_hair(A)
    N = G.normals_vertices(A.V, A.F)
    # outward direction: the one moving away from the centre of the head
    if ((A.V - A.V.mean(0)) * N).sum(1).mean() < 0:
        N = -N
    under = (_rays_touch(A.V, N, A.pos[Fh], DISTANCE_HAIR * A.iod) if len(Fh)
            else np.zeros(A.n, dtype=bool))
    ear = np.zeros(A.n, dtype=bool)
    for side in ("g", "d"):
        ear |= G.dijkstra(A.V, A.adj, [A.landmarks["ear" + side]]) < RADIUS_EAR * A.iod
    src = np.where(under | ear)[0]
    if not len(src):
        return np.zeros(A.n), {"under_hair": 0, "ears": 0}
    d = G.dijkstra(A.V, A.adj, src)
    return 1 - G.smooth(d / (BAND * A.iod)), {"under_hair": int(under.sum()), "ears": int(ear.sum())}


def mode_shape(A, B, output):
    s, K, c0 = put_shape(A, B)
    XA = A.V.copy()
    XB = B._host(B.V)
    XK = c0 + K * (XA - c0)
    rec, w, X = place_face(A, XK, B, XB, on_place=True)
    # A's hair and ears do not move: the skin under the hair (B's shape would not be seen
    # there and would pierce the hair) and the ears keep their place on A
    q, n_fixed = zone_fixed(A)
    q = q * (1 - w)
    X = q[:, None] * XA + (1 - q[:, None]) * X
    # neck seam: gradual return to A
    dcou = G.dijkstra(XA, A.adj, A.neck) if len(A.neck) else np.full(A.n, np.inf)
    p = 1 - G.smooth(dcou / (RADIUS_NECK * A.iod))
    X = p[:, None] * XA + (1 - p[:, None]) * X
    X = smooth_displacement(XA, X, A.F, fixed=(w >= 0.999) | (p >= 0.999) | (q >= 0.999))
    D = X - XA
    Rv = rotations_local(XA, X, A.F)

    # --- output: copy of A, with modified positions, normals and morphs
    P = copy.deepcopy(A.pmx)
    P["textures"] = [_texture_rel(x, os.path.dirname(A.path), output) for x in P["textures"]]
    pos = np.asarray(A.pos).copy()
    nor = np.asarray(A.t["nor"]).copy()
    vb = A.skin_raw
    k = A.b2s[vb]
    pos[vb] = X[k]
    nor[vb] = np.einsum("nij,nj->ni", Rv[k], nor[vb])
    # vertices that follow the head (face parts, accessories); the hair stays
    others = np.where((A.weight_head > 0) & (A.b2s < 0))[0]
    others = np.setdiff1d(others, vertices_hair(A)[0])
    Dsh = G.shepard(XA, D, A.pos[others], k=16)
    pos[others] += Dsh * A.weight_head[others][:, None]
    _, jn = _plus_near_vertex(XA, A.pos[others])
    nor[others] = np.einsum("nij,nj->ni", Rv[jn], nor[others])
    # rotation per raw vertex, for the morphs, after the face stretch (weighted by w)
    Kv = 1 + w[:, None] * (K - 1)
    Rb = {}
    for i, kk in zip(vb, k):
        Rb[int(i)] = Rv[kk] * Kv[kk]
    for i, j in zip(others, jn):
        Rb[int(i)] = Rv[j] * Kv[j]
    for m in P["morphs"]:
        if m["type"] != 1:
            continue
        m["offsets"] = [(i, tuple(Rb[i] @ np.asarray(d)) if i in Rb else d) for i, d in m["offsets"]]
    P["vertices"]["pos"] = [tuple(x) for x in pos]
    P["vertices"]["nor"] = [tuple(x) for x in nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)]
    P["comment"] = (P.get("comment") or "") + "\r\n[retarget_head shape: skull of %s]" % B.name
    path = os.path.join(output, "result.pmx")
    pmx_write.write(P, path)
    np.savez_compressed(os.path.join(output, "registration.npz"), X=X, XA=XA, XB=XB, FA=A.F, FB=B.F,
                        phi=rec["phi"], w=w, p_neck=p, X0=rec["X0"], Xicp=rec["X"])
    eyes = 0.5 * (X[A.loop_eye["g"]].mean(0) + X[A.loop_eye["d"]].mean(0))[1]
    eyes0 = 0.5 * (XA[A.loop_eye["g"]].mean(0) + XA[A.loop_eye["d"]].mean(0))[1]
    chin, skull = X[A.landmarks["chin"], 1], B._host(np.array([[0.0, B.y_skull, 0.0]]))[0, 1]
    info = {"scale_target": s, "stretch_face": K.round(4).tolist(), "zone_fixed": n_fixed,
            "height_eyes_on_head": {"A": round(float((eyes0 - A.V[A.landmarks["chin"], 1]) / A.H_crane), 4),
                                      "result": round(float((eyes - chin) / (skull - chin)), 4)},
            "icp_track": rec["track"], "vertices_followers": int(len(others)),
            "shift_max_followers": float(np.linalg.norm(Dsh, axis=1).max()) if len(others) else 0.0,
            "zone": {"face_w>0.5": int((w > 0.5).sum()), "band": int(((w > 0.02) & (w < 0.98)).sum()),
                     "crane_w<0.02": int((w <= 0.02).sum())}}
    return path, info, {"w": w, "p_neck": p, "q": q, "XB": XB, "rec": rec, "K": K,
                          "idxA": np.arange(len(A.pos))}


def _plus_near_vertex(V, Q):
    j = np.zeros(len(Q), dtype=np.int64)
    d = np.zeros(len(Q))
    for s in range(0, len(Q), 2048):
        dd = np.linalg.norm(Q[s:s + 2048, None] - V[None], axis=2)
        j[s:s + 2048] = dd.argmin(1)
        d[s:s + 2048] = dd.min(1)
    return d, j


# =================================================================== visage mode

def _map_os(A, B):
    """A's bones -> B's bones by name (mouth synonyms), otherwise B's 頭."""
    nb = {n: i for i, n in enumerate(B.names_os)}
    syn = {"口左": "左口", "口右": "右口", "口中": "口", "左口": "口左", "右口": "口右", "口": "口中"}
    head = nb.get("頭", 0)
    out = []
    for n in A.names_os:
        if n in nb:
            out.append(nb[n])
        elif syn.get(n) in nb:
            out.append(nb[syn[n]])
        else:
            out.append(head)
    return np.array(out)


def _texture_rel(path_tex, folder_model, output):
    p = os.path.normpath(os.path.join(folder_model, path_tex.replace("\\", "/")))
    return os.path.relpath(p, output).replace("/", "\\")


def _distance_polyline(P, Q):
    """Distance from each point of P to the closed polyline Q."""
    Q2 = np.vstack([Q, Q[:1]])
    a, ab = Q2[:-1], Q2[1:] - Q2[:-1]
    out = np.empty(len(P))
    for k in range(0, len(P), 512):
        p = P[k:k + 512]
        t = np.clip((((p[:, None] - a[None]) * ab[None]).sum(2)) / np.maximum((ab * ab).sum(1), 1e-18)[None], 0, 1)
        out[k:k + 512] = np.linalg.norm(p[:, None] - (a[None] + t[..., None] * ab[None]), axis=2).min(1)
    return out


def _contains_edge(F, u, v):
    """True if a face of F carries the oriented edge u -> v."""
    return bool(((F[:, 0] == u) & (F[:, 1] == v)).any() or ((F[:, 1] == u) & (F[:, 2] == v)).any()
                or ((F[:, 2] == u) & (F[:, 0] == v)).any())


def band_between_loops(Pa, Pb, Nb):
    """Triangulates the strip between two neighbouring closed loops with the same orientation.
    Pa: patch border, walked in the direction of its faces; Pb: hole border, walked against
    the direction of its faces; Nb: B's normals at Pb. Optimal triangulation by dynamic
    programming (Fuchs, Kedem and Uselton 1977): sum of diagonals, plus a heavy penalty for
    each triangle flipped relative to B. Returns triangles [(("a"|"b"), index) × 3]."""
    na, nb = len(Pa), len(Pb)
    j0 = int(np.linalg.norm(Pb - Pa[0], axis=1).argmin())
    A_ = lambda i: Pa[i % na]
    B_ = lambda j: Pb[(j0 + j) % nb]
    NB_ = lambda j: Nb[(j0 + j) % nb]

    def cost(p, q, r, n, diag):
        c = np.cross(q - p, r - p)
        return diag + (1e3 if np.dot(c, n) <= 0 else 0.0)

    INF = np.inf
    C = np.full((na + 1, nb + 1), INF)
    stage = np.zeros((na + 1, nb + 1), dtype=np.int8)
    C[0, 0] = 0.0
    for i in range(na + 1):
        for j in range(nb + 1):
            if i == 0 and j == 0:
                continue
            best, k = INF, 0
            if i > 0 and C[i - 1, j] < INF:  # triangle (a_i, a_{i-1}, b_j)
                v = C[i - 1, j] + cost(A_(i), A_(i - 1), B_(j), NB_(j), np.linalg.norm(A_(i) - B_(j)))
                if v < best:
                    best, k = v, 1
            if j > 0 and C[i, j - 1] < INF:  # triangle (b_{j-1}, b_j, a_i)
                v = C[i, j - 1] + cost(B_(j - 1), B_(j), A_(i), NB_(j), np.linalg.norm(B_(j) - A_(i)))
                if v < best:
                    best, k = v, 2
            C[i, j], stage[i, j] = best, k
    tri = []
    i, j = na, nb
    while i > 0 or j > 0:
        if stage[i, j] == 1:
            tri.append([("a", i % na), ("a", (i - 1) % na), ("b", (j0 + j) % nb)])
            i -= 1
        else:
            tri.append([("b", (j0 + j - 1) % nb), ("b", (j0 + j) % nb), ("a", i % na)])
            j -= 1
    tri = tri[::-1]
    P = lambda c, q: Pa[q] if c == "a" else Pb[q]
    ret = []
    for t in tri:
        p = [P(c, q) for c, q in t]
        jb = [q for c, q in t if c == "b"][0]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), Nb[jb]) <= 0:
            ret.append(t)
    return tri, ret

def rings(n, adj, sources):
    """Distance in rings (edges) from source vertices."""
    d = np.full(n, 10 ** 9)
    d[list(sources)] = 0
    front = list(sources)
    k = 0
    while front:
        k += 1
        nf = []
        for u in front:
            for v in adj[u]:
                if d[v] > k:
                    d[v] = k
                    nf.append(v)
        front = nf
    return d


def harmonize_morphs(P, mats_skin, seeds, loops_k=LOOPS_HARMONY):
    """Morphs on both sides of the seam: the offsets of A (patch) and of B (the rest) do not
    join up on the strip. On skin vertices within `loops_k` rings of the strip, the offsets
    are replaced by the harmonic interpolation (zero Laplacian) of the values held beyond:
    the seam follows both sides without tearing."""
    pos = np.asarray(P["vertices"]["pos"], dtype=float)
    F = np.asarray(P["faces"], dtype=np.int64)
    mf = np.repeat(np.arange(len(P["materials"])), [m["n_indices"] // 3 for m in P["materials"]])
    Fs = F[np.isin(mf, mats_skin)]
    vb = np.unique(Fs)
    rep, inv = G.weld(pos[vb])
    b2s = np.full(len(pos), -1)
    b2s[vb] = inv
    n = len(rep)
    Fw = b2s[Fs]
    Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
    adj = G.neighbors(n, Fw)
    d = rings(n, adj, np.unique(b2s[seeds]))
    free = d <= loops_k
    held = ~free
    L = G.laplacian(n, Fw)
    Lff = L[np.ix_(free, free)]
    Lft = L[np.ix_(free, held)]
    il = np.where(free)[0]
    selvage = np.array([held[j] and any(free[x] for x in adj[j]) for j in range(n)])
    raw_of = [[] for _ in range(n)]
    for i, k in zip(vb, inv):
        raw_of[k].append(int(i))
    touched = 0
    for m in P["morphs"]:
        if m["type"] != 1 or not m["offsets"]:
            continue
        D = np.zeros((n, 3))
        c = np.zeros(n)
        others = []
        for v, x in m["offsets"]:
            k = b2s[v]
            if k >= 0:
                D[k] += x
                c[k] += 1
            else:
                others.append((v, x))
        D /= np.maximum(c, 1)[:, None]
        if not np.abs(D[free | selvage]).max() > 0:
            continue
        Df = np.linalg.solve(Lff, -Lft @ D[held])
        D[il] = Df
        offset = list(others)
        for k in np.where(np.abs(D).max(1) > 1e-7)[0]:
            for v in raw_of[k]:
                offset.append((v, tuple(D[k])))
        m["offsets"] = offset
        touched += 1
    return {"free_vertices": int(free.sum()), "harmonized_morphs": touched}


def regularize_border(F, sel, iters=20):
    """Border of a face selection without spikes or notches: selected faces with only one
    selected neighbour are removed, neighbouring faces with two or three are added, until
    stable."""
    _, _, fe = G.edges(F)
    # two faces per interior edge
    n_e = int(fe.max()) + 1
    f1 = np.full(n_e, -1)
    f2 = np.full(n_e, -1)
    for f in range(len(F)):
        for e in fe[f]:
            if f1[e] < 0:
                f1[e] = f
            else:
                f2[e] = f
    neigh = np.where(f1[fe] == np.arange(len(F))[:, None], f2[fe], f1[fe])
    s = sel.copy()
    for _ in range(iters):
        nv = np.where(neigh >= 0, s[np.maximum(neigh, 0)], False).sum(1)
        addition = ~s & (nv >= 2)
        inset = s & (nv <= 1)
        if not addition.any() and not inset.any():
            break
        s = (s | addition) & ~inset
    return s


def refine_near_seam(A, X, Fp, B, XB):
    """Marks B's skin faces near the patch border (radius R_REFINEMENT edges of B) whose longest
    edge exceeds 1.3 times A's mean edge, then splits them."""
    ua, _, _ = G.edges(Fp)
    eA = float(np.linalg.norm(X[ua[:, 0]] - X[ua[:, 1]], axis=1).mean())
    holes_A = set()
    for b in A.loops:
        holes_A.update(int(x) for x in b)
    border = [np.array(b) for b in G.border_loops(Fp) if not all(int(x) in holes_A for x in b)]
    if len(border) != 1:
        return None
    Pb = X[border[0]]
    T = XB[B.F]
    lg = np.max(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), axis=1)
    c = T.mean(1)
    d = np.min(np.linalg.norm(c[:, None] - Pb[None], axis=2), axis=1)
    mark = (d < R_REFINEMENT * lg) & (lg > 1.3 * eA)
    if not mark.any():
        return None
    P = B.pmx
    info = {"marked_faces": int(mark.sum()), "edge_A": round(eA, 4)}
    for m in B.i_skin:
        P, i = RF.refine(P, m, B.F_brut[mark])
        for k, x in i.items():
            info[k] = info.get(k, 0) + x
    return TT.Head(B.path, skin=B.mats_skin, pmx=P), info


def put_proportions_source(A, B):
    """Places A on B while keeping the proportions of A's face relative to its head: A's
    chin-to-skull-top height becomes B's, and the middle of one goes onto the middle of the
    other (the eyes land at the same relative height as on A, the forehead keeps its length
    in eye distances). The skull top is taken under the hair (Head.y_crane): the skin alone
    stops higher or lower depending on the model. The depth is set so that A's face touches B."""
    yA = 0.5 * (A.y_skull + A.V[A.landmarks["chin"], 1])
    yB = 0.5 * (B.y_skull + B.V[B.landmarks["chin"], 1])
    cA, cB = _center_x0(A.center_face()), _center_x0(B.center_face())
    cA[1], cB[1] = yA, yB
    s = B.H_crane / A.H_crane
    w = A.weight_face(MARGIN * A.iod, BAND * A.iod) > 0.5
    for _ in range(3):
        A.put(s, cA, cB)
        Q = A._host(A.V[w])
        c = G.plus_near(Q, B.V, B.F)[0]
        cB = cB + np.array([0.0, 0.0, float(np.median(c[:, 2] - Q[:, 2]))])
    A.put(s, cA, cB)
    return s


def mode_face(A, B, output, proportions="target"):
    on_place = proportions == "source"
    if on_place:
        s = put_proportions_source(A, B)
    else:
        # A brought onto B's face landmarks (uniform scale, translation, no rotation)
        s = scale_face(A, B)
        A.put(s, _center_x0(A.center_face()), _center_x0(B.center_face()))
    XA = A._host(A.V)
    XB = B.V.copy()
    refinement = []
    for passed in range(LEVELS_REFINEMENT + 1):
        rec, w, X = place_face(A, XA, B, XB, edge_B=True, on_place=on_place)
        # --- patch: A's faces of the face and the strip, as a disc, away from A's outer border
        wf = w[A.F].mean(1)
        # the patch also covers B's own face (there, A already lies on B: w ≈ 0)
        wB = B.weight_face(MARGIN * B.iod, BAND * B.iod)
        wBphi = (wB[B.F[rec["face"]]] * rec["bary"]).sum(1)
        sel = (wf >= 0.05) | (wBphi[A.F].mean(1) >= 0.3)
        sel = G.fill_disk(A.F, sel)
        Fp = A.F[sel]
        if passed == LEVELS_REFINEMENT:
            break
        # --- B refined near the patch border, down to the size of A's edges: the seam then
        # joins two borders with the same spacing (B's surface does not move)
        Bnew = refine_near_seam(A, X, Fp, B, XB)
        if Bnew is None:
            break
        B, info_r = Bnew
        XB = B.V.copy()
        refinement.append(info_r)
        log("B refined near the seam: %s" % info_r)
    # --- patch border
    bp = [np.array(b) for b in G.border_loops(Fp)]
    holes_A = set()
    for b in A.loops:
        holes_A.update(int(x) for x in b)
    loops_a = [b for b in bp if not all(int(x) in holes_A for x in b)]
    if len(loops_a) != 1:
        raise RuntimeError("patch border: %d loops instead of one" % len(loops_a))
    loop_A = np.asarray(loops_a[0])
    loop_A_raw = loops_a[0]
    # --- hole in B: every vertex of B under the patch (closest point inside the patch: zero
    # tangential gap) or within δ of the patch border is removed with its faces; the hole
    # border thus stays entirely outside the patch, at a gap of δ to δ + one edge
    lm = rec["mean_edge_B"]
    pq, fq, _, dq = G.plus_near(XB, X, Fp)
    rq = XB - pq
    nq = G.normals_faces(X, Fp)[fq]
    tang = np.linalg.norm(rq - (rq * nq).sum(1, keepdims=True) * nq, axis=1)
    dL = _distance_polyline(XB, X[loop_A])
    eB = np.linalg.norm(XB[B.F] - np.roll(XB[B.F], 1, axis=1), axis=2).mean(1)
    near = _distance_polyline(XB[B.F].mean(1), X[loop_A]) < 3 * eB
    delta = GAP_SEAM * float(np.median(eB[near]))
    covered = ((dq < lm) & (tang < 0.05 * lm)) | (dL < delta)
    # B's eye and mouth holes: to be removed (the fill takes what they enclose)
    traits = np.zeros(B.n, dtype=bool)
    for b in (B.loop_eye["g"], B.loop_eye["d"], B.loop_mouth):
        traits[b] = True
    base = (covered | traits)[B.F].any(1)
    hole = G.fill_disk(B.F, base)
    hole = regularize_border(B.F, hole) | base
    hole = G.fill_disk(B.F, hole)
    extB = np.zeros(B.n, dtype=bool)
    extB[B.loop_ext] = True
    if extB[B.F[hole]].any():
        log("hole: %d faces touch B's outer border" % int(extB[B.F[hole]].any(1).sum()))
        raise RuntimeError("the face hole touches the outer border of B's skin")
    # --- loops to sew, then the strip; a notch in the hole border that the strip cannot cover
    # without flipping is removed with the hole (a few passes at most)
    if not _contains_edge(Fp, loop_A[0], loop_A[1]):
        loop_A = loop_A[::-1].copy()
    na = len(loop_A)
    holes_B = set()
    for b in B.loops:
        holes_B.update(int(x) for x in b)
    nBv = G.normals_vertices(XB, B.F)
    notches = 0
    for passed in range(5):
        Fr = B.F[~hole]
        br = [np.array(b) for b in G.border_loops(Fr)]
        lb = [b for b in br if not all(int(x) in holes_B for x in b)]
        if len(lb) != 1:
            raise RuntimeError("hole border: %d loops instead of one" % len(lb))
        Lb = np.asarray(lb[0])
        if _contains_edge(Fr, Lb[0], Lb[1]):
            Lb = Lb[::-1].copy()
        nb = len(Lb)
        band, flipped = band_between_loops(X[loop_A], XB[Lb], nBv[Lb])
        if not flipped or passed == 4:
            break
        addition = np.zeros(len(B.F), dtype=bool)
        for t in flipped:
            vb_ = [int(Lb[q]) for c, q in t if c == "b"]
            addition |= ~hole & np.isin(B.F, vb_).sum(1).astype(bool) if len(vb_) == 1 else \
                ~hole & (np.isin(B.F, vb_).sum(1) == 2)
        hole = G.fill_disk(B.F, hole | addition)
        notches += int(addition.sum())
        if extB[B.F[hole]].any():
            raise RuntimeError("the face hole touches the outer border of B's skin")
    band_ret = len(flipped)
    gap = _distance_polyline(XB[Lb], X[loop_A])
    log("seam: patch %d faces (border %d), hole %d faces (border %d), strip of %d triangles, "
            "border gap median %.4f max %.4f (δ = %.4f), %d notch faces removed, %d flipped" % (
                len(Fp), na, int(hole.sum()), nb, len(band), np.median(gap), gap.max(), delta,
                notches, band_ret))

    # =============================================================== PMX assembly
    Pa, Pb = A.pmx, B.pmx
    sa_, sb_ = Pa["vertices"], Pb["vertices"]
    nuv = Pb["n_uv_add"]
    map_ = _map_os(A, B)
    pieces_B = set(B.i_parts)
    i_skinB = B.i_skin[0]
    # A already composed (颜 + 颜_A): each patch face keeps its material, hence its texture;
    # the face is its 颜_A, the rest of its skin becomes 颜_A2
    i_visA = A.names_mat.index(TT.SKIN_SEWN) if TT.SKIN_SEWN in A.names_mat else A.i_skin[0]
    # B's raw faces kept
    fb_hole = set(int(x) for x in B.F_brut[hole])
    kept_fB = np.array([not (B.mat_face[f] in pieces_B or f in fb_hole) for f in range(len(B.faces))])
    # B's raw faces duplicated in the welded skin (degenerate faces): they follow their face
    vertsB_kept = np.unique(B.faces[kept_fB])
    # A's vertices used: patch skin and parts
    fA_patch = A.F_brut[sel]
    fA_pieces = np.where(np.isin(A.mat_face, A.i_parts))[0]
    vertsA = np.unique(np.concatenate([A.faces[fA_patch].ravel(), A.faces[fA_pieces].ravel()]))
    nB = len(vertsB_kept)
    idxB = np.full(len(B.pos), -1)
    idxB[vertsB_kept] = np.arange(nB)
    idxA = np.full(len(A.pos), -1)
    idxA[vertsA] = nB + np.arange(len(vertsA))
    # A's positions: skin -> X, parts -> displacement of the nearby skin
    posA = A._host(A.pos[vertsA])
    norA = np.asarray(A.t["nor"])[vertsA].copy()
    Rv = rotations_local(XA, X, A.F)
    is_skin = A.b2s[vertsA] >= 0
    kpe = A.b2s[vertsA[is_skin]]
    D = X - XA
    posA[is_skin] = X[kpe]
    norA[is_skin] = np.einsum("nij,nj->ni", Rv[kpe], norA[is_skin])
    pi = ~is_skin
    Dsh = G.shepard(XA, D, posA[pi], k=16)
    posA[pi] += Dsh
    _, jn = _plus_near_vertex(XA, posA[pi])
    norA[pi] = np.einsum("nij,nj->ni", Rv[jn], norA[pi])
    RvA = np.zeros((len(vertsA), 3, 3))
    RvA[is_skin] = Rv[kpe]
    RvA[pi] = Rv[jn]
    n_dup0 = nB + len(vertsA)
    # B's raw vertex for each welded vertex of the hole border: the one from the kept faces
    rep_b = {}
    for f in np.where(kept_fB & (B.mat_face == i_skinB))[0]:
        for v in B.faces[f]:
            k = B.b2s[v]
            if k >= 0 and k not in rep_b:
                rep_b[int(k)] = int(v)

    # --- vertex arrays
    S = {"pos": [], "nor": [], "uv": [], "uv_add": [], "type": [], "bones": [], "weights": [],
         "sdef": [], "contour": []}

    def add_B(v):
        for key in S:
            S[key].append(sb_[key][v])

    def uva(src, v):
        u = list(src["uv_add"][v])[:nuv]
        return u + [(0.0, 0.0, 0.0, 0.0)] * (nuv - len(u))

    def add_A(v, p, n, uv=None):
        t = sa_["type"][v]
        o = [int(map_[x]) if x >= 0 else -1 for x in sa_["bones"][v]]
        w = list(sa_["weights"][v])
        if t == 3:
            t = 1
        if t == 4:
            t = 2
        if t == 0:
            o = [o[0], -1, -1, -1]
        S["pos"].append(tuple(p))
        S["nor"].append(tuple(n / max(np.linalg.norm(n), 1e-12)))
        S["uv"].append(tuple(uv) if uv is not None else sa_["uv"][v])
        S["uv_add"].append(uva(sa_, v))
        S["type"].append(t)
        S["bones"].append(tuple(o))
        S["weights"].append(tuple(w))
        S["sdef"].append(None)
        S["contour"].append(sa_["contour"][v])

    for v in vertsB_kept:
        add_B(int(v))
    for j, v in enumerate(vertsA):
        add_A(int(v), posA[j], norA[j])
    # seam normals: B's, interpolated along the hole border
    norB = np.asarray(sb_["nor"], dtype=float)
    uvB = np.asarray(sb_["uv"], dtype=float)
    # on the patch border, B's normals and UVs at the projected point (B's skin carries on)
    fbr = B.faces[B.F_brut[rec["face"][loop_A]]]
    br_ = rec["bary"][loop_A]
    nor_loop_A = (br_[:, :, None] * norB[fbr]).sum(1)
    uv_loop_A = (br_[:, :, None] * uvB[fbr]).sum(1)
    for i in range(na):
        for v in A.s2b[int(loop_A[i])]:
            j = idxA[v]
            if j >= 0:
                S["nor"][j] = tuple(nor_loop_A[i] / max(np.linalg.norm(nor_loop_A[i]), 1e-12))
    dupA_de = {}  # A's raw vertex (patch border) -> duplicates with B's material
    dupB_de = {}  # B's raw vertex (hole border) -> duplicates with A's material

    def new():
        return len(S["pos"]) - 1


    # --- faces per material, order: B's materials up to 颜, then A's, then the rest
    faces_per_mat = {}
    for f in np.where(kept_fB)[0]:
        faces_per_mat.setdefault(("B", int(B.mat_face[f])), []).append(tuple(int(idxB[v]) for v in B.faces[f]))
    for f in fA_patch:
        faces_per_mat.setdefault(("A", int(A.mat_face[f])), []).append(tuple(int(idxA[v]) for v in A.faces[f]))
    for f in fA_pieces:
        faces_per_mat.setdefault(("A", int(A.mat_face[f])), []).append(tuple(int(idxA[v]) for v in A.faces[f]))
    # seam strip, with B's material: copies of the patch border vertices (A's weights,
    # B's UVs), and the hole border vertices
    copy_A = {}
    for i in range(na):
        va_ = A.s2b[int(loop_A[i])][0]
        add_A(int(va_), X[loop_A[i]], nor_loop_A[i], uv=uv_loop_A[i])
        copy_A[i] = new()
        dupA_de.setdefault(int(va_), []).append(new())
    for t in band:
        faces_per_mat.setdefault(("B", i_skinB), []).append(tuple(
            copy_A[q] if c == "a" else int(idxB[rep_b[int(Lb[q])]]) for c, q in t))
    order = []
    for i in range(len(Pb["materials"])):
        if i in pieces_B:
            continue
        order.append(("B", i))
        if i == i_skinB:
            order += [("A", j) for j in A.i_skin if j == i_visA or ("A", j) in faces_per_mat]
            order += [("A", j) for j in A.i_parts]
    P = copy.deepcopy(Pb)
    off_tex = len(Pb["textures"])
    P["textures"] = [_texture_rel(x, os.path.dirname(B.path), output) for x in Pb["textures"]] + [_texture_rel(x, os.path.dirname(A.path), output) for x in Pa["textures"]]
    mats, faces, new_mat = [], [], {}
    for the_key in order:
        src, i = the_key
        m = copy.deepcopy((Pb if src == "B" else Pa)["materials"][i])
        if src == "A":
            for kk in ("texture", "sphere"):
                if m[kk] >= 0:
                    m[kk] += off_tex
            if m["toon_sharing"] == 0 and m["toon"] >= 0:
                m["toon"] += off_tex
            if i in A.i_skin:
                suffix = "" if i == i_visA else "2"
                m["name"] = TT.SKIN_SEWN + suffix
                m["name_en"] = (m["name_en"] or "face").removesuffix("_A") + "_A" + suffix
        fl = faces_per_mat.get(the_key, [])
        m["n_indices"] = 3 * len(fl)
        new_mat[the_key] = len(mats)
        mats.append(m)
        faces += fl
    P["materials"] = mats
    P["faces"] = faces
    P["vertices"] = S

    # --- morphs: B's are kept (on B's kept vertices); for a name that A also has, A's offsets
    # (rotated into the local frame, scaled) are added on the face
    morA = {}
    for m in Pa["morphs"]:
        if m["type"] == 1:
            morA[m["name"]] = m
    posA_de = {int(v): j for j, v in enumerate(vertsA)}
    dup_of = dupA_de
    n_dup = len(S["pos"]) - n_dup0

    def offsets_A(m):
        out = []
        for v, d in m["offsets"]:
            j = posA_de.get(int(v))
            if j is None:
                continue
            dd = tuple(A.scale * (RvA[j] @ np.asarray(d)))
            out.append((nB + j, dd))
            for jd in dup_of.get(int(v), []):
                out.append((jd, dd))
        return out

    mor = []
    seen = set()
    MoB = None
    for m in Pb["morphs"]:
        m = copy.deepcopy(m)
        t = m["type"]
        if t in (0, 2) and m["name"] in morA:
            # B's bone (or group) morph replaced by a vertex morph of A: its effect on B's
            # kept vertices becomes offsets, and A's are added to them (a single morph with
            # this name)
            import morphs as MO
            MoB = MoB or MO.Model(pmx=Pb)
            dd = MoB.displacement(m["name"])
            m["type"] = 1
            m["offsets"] = [(int(idxB[v]), tuple(dd[v])) for v in np.where(np.abs(dd).max(1) > 1e-7)[0]
                              if idxB[v] >= 0] + offsets_A(morA[m["name"]])
            seen.add(m["name"])
            mor.append(m)
            continue
        if t == 1 or 3 <= t <= 7:
            offset = [(int(idxB[v]), d) for v, d in m["offsets"] if idxB[v] >= 0]
            if t == 1:
                offset += [(jd, d) for v, d in m["offsets"] for jd in dupB_de.get(int(v), [])]
            if t == 1 and m["name"] in morA:
                offset += offsets_A(morA[m["name"]])
                seen.add(m["name"])
            m["offsets"] = offset
        elif t == 8:
            offset = []
            for d in m["offsets"]:
                if d["material"] < 0:
                    offset.append(d)
                elif ("B", d["material"]) in new_mat:
                    d = dict(d)
                    d["material"] = new_mat[("B", d["material"])]
                    offset.append(d)
            m["offsets"] = offset
        mor.append(m)
    addition = []
    for name, m in morA.items():
        if name in seen:
            continue
        m = copy.deepcopy(m)
        m["offsets"] = offsets_A(m)
        addition.append(len(mor))
        mor.append(m)
    P["morphs"] = mor
    harm = harmonize_morphs(P, [new_mat[("B", i_skinB)]] + [new_mat[("A", j)] for j in A.i_skin if ("A", j) in new_mat],
                              np.arange(n_dup0, n_dup0 + n_dup))
    for c in P["frames"]:
        if c["name"] == "表情":
            c["elements"] = list(c["elements"]) + [(1, k) for k in addition]
    P["comment"] = (P.get("comment") or "") + "\r\n[retarget_head visage: face of %s]" % A.name
    path = os.path.join(output, "result.pmx")
    pmx_write.write(P, path)
    np.savez_compressed(os.path.join(output, "registration.npz"), X=X, XA=XA, XB=XB, FA=A.F, FB=B.F,
                        phi=rec["phi"], w=w, patch=sel, hole=hole, X0=rec["X0"], Xicp=rec["X"])
    info = {"source_scale": s, "icp_track": rec["track"], "first_duplicate": int(n_dup0),
            "patch_faces": int(sel.sum()), "hole_faces": int(hole.sum()),
            "patch_border": int(na), "hole_border": int(nb), "seam_band_triangles": len(band),
            "seam_band_flipped": band_ret, "seam_notches_removed": notches,
            "seam_gap_median": float(np.median(gap)), "seam_gap_max": float(gap.max()),
            "seam_delta": delta,
            "morphs_from_A_added": [mor[k]["name"] for k in addition], "seam_harmony": harm,
            "refinement_B": refinement,
            "zone": {"face_w>0.5": int((w > 0.5).sum()), "band": int(((w > 0.02) & (w < 0.98)).sum())}}
    return path, info, {"B_refined": B, "w": w, "XB": XB, "rec": rec, "sel": sel, "hole": hole,
                          "loop_A": loop_A, "Lb": Lb, "idxA": idxA, "vertsB_kept": vertsB_kept,
                          "n_dup0": n_dup0, "n_dup": n_dup, "border_expected":
                          sum(len(x) for x in bp if x is not loop_A_raw) + sum(len(x) for x in br if x is not lb[0])}


# =================================================================== main

def sheet(mode, source, target, result, output):
    script = os.path.join(HERE, "head_sheet.py")
    cmd = ["bash", FILE_BLENDER, "-b", "--factory-startup", "--python", script, "--", mode, source, target,
           result, output]
    log("contact sheet: Blender queue")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("contact sheet: Blender failed")
    return os.path.join(output, "sheet.png")


def tint(result):
    """visage mode: B's skin takes the colour of A's face (recoloured textures)."""
    cmd = ["bash", FILE_BLENDER, "-b", "--factory-startup", "--python", os.path.join(HERE, "tint.py"), "--", result]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or "tint:" not in r.stdout:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("tint: Blender failed")
    with open(os.path.join(os.path.dirname(result), "tint.json"), encoding="utf-8") as fh:
        return json.load(fh)


def sheet_textures(source, target, result, output):
    cmd = ["bash", FILE_BLENDER, "-b", "--factory-startup", "--python", os.path.join(HERE, "texture_sheet.py"), "--",
           os.path.join(output, "texture_sheet.png"), source, os.path.dirname(source),
           target, os.path.dirname(target), result, output]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("textured contact sheet: Blender failed")
    return os.path.join(output, "texture_sheet.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("shape", "face"))
    ap.add_argument("--source", required=True, help="A: the face that is kept")
    ap.add_argument("--target", dest="target", required=True, help="B: the head shape (shape mode) or the host head (face mode)")
    ap.add_argument("--output", dest="output", required=True)
    ap.add_argument("--no-sheet", dest="without_sheet", action="store_true")
    ap.add_argument("--proportions", choices=("target", "source"), default="target",
                    help="face: face size and position taken from B (target) or kept from A relative to its head (source)")
    ap.add_argument("--no-tint", dest="without_tint", action="store_true", help="face: keep B's skin colour")
    a = ap.parse_args(argv)
    output = os.path.abspath(os.path.expanduser(a.output))
    os.makedirs(output, exist_ok=True)
    t0 = time.time()
    A = TT.Head(os.path.abspath(a.source))
    B = TT.Head(os.path.abspath(a.target))
    log("A = %s (%d skin vertices), B = %s (%d)" % (A.name, A.n, B.name, B.n))
    if a.mode == "shape":
        path, info, inner = mode_shape(A, B, output)
    else:
        path, info, inner = mode_face(A, B, output, a.proportions)
        B = inner.pop("B_refined")
    log("result written: %s" % path)
    measures = VF.check(a.mode, A, B, path, inner, log=log)
    measures["entries"] = {"mode": a.mode, "source": A.summary(), "target": B.summary()}
    measures["parameters"] = {"proportions": a.proportions, "margin_iod": MARGIN, "band_iod": BAND, "neck_radius_iod": RADIUS_NECK, "smoothing": SMOOTHING}
    measures["construction"] = info
    if a.mode == "face" and not a.without_tint:
        measures["tint"] = tint(path)
        log("tint: ΔE %.1f -> %.1f" % (measures["tint"]["dE_before"], measures["tint"]["dE_after"]))
    measures["duration_s"] = round(time.time() - t0, 1)
    write_json(os.path.join(output, "measures.json"), measures)
    log("measurements: %s" % ("ALL PASS" if measures["verdict"]["passed"] else "FAILURES: " + ", ".join(measures["verdict"]["failures"])))
    if not a.without_sheet:
        p = sheet(a.mode, A.path, B.path, path, output)
        log("contact sheet: %s" % p)
        log("textured contact sheet: %s" % sheet_textures(A.path, B.path, path, output))
    return 0 if measures["verdict"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

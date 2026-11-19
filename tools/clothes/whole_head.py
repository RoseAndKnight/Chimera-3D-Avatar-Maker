"""Whole studio head (face, ears and neck in a single piece) stitched onto the common body.

    blender -b --factory-startup --python tools/clothes/whole_head.py -- \
        --body BODY.pmx --head HEAD.pmx --output FOLDER
    (without Blender, python3 -I does everything except the tint, which reads and writes textures)

On a studio head (Yuzuha), the face 颜 and the skin 肌 of the ears and neck are welded into a single
surface, which ends at the bottom of the neck on a horizontal cut, and at the back on an open edge that
the hair covers. The rest of 肌 (body, V-shaped collar piece) is not taken. This piece is kept as is
(the underside of the jaw stays the studio's) and the body is stitched to it:

 - graft: `head_graft --keep-skull` places the piece by the 頭 bone, shares the bones, carries the morphs;
 - the body goes away under the piece: any face with a vertex seen through the piece from its centre, and
   above the neck cut; two edges remain, below the neck cut and around the scalp;
 - the body adjusts to the piece, which is left untouched: at the bottom, radius around the neck axis,
   faded out over `RADIUS_SHAPE_MM`; at the top, the body edge is laid on the scalp edge by arc length,
   the whole scalp follows through a harmonic field (zero at the neck), and the rows behind that edge are
   smoothed (Taubin) over `RADIUS_SMOOTHING_MM`;
 - a strip of faces (body material) links each body edge to the piece edge: its vertices on the piece
   side are copies of the piece's (position, normal, weights, morphs), so nothing opens when the model
   moves;
 - neck weights: the body edge takes those of the ring; below it, those of the head's original neck
   (projected onto its triangles) up to `RADIUS_WEIGHT_NECK_MM[0]`, the body's beyond the second radius,
   a harmonic field in between; scalp weights and normals are blended towards the piece edge over
   `RADIUS_WEIGHT_MM`;
 - closed head (whole scalp, no open edge at the back: VRoid): no seam at the top, the body loses
   everything it has above the neck cut;
 - small holes in the piece (edges of 6 vertices at most, behind the ears) are filled;
 - tint: the body skin is repainted, each texel taking the piece's colour at the nearest edge,
   blended over `RADIUS_TINT_MM` towards that of the bottom of the neck (--base-tint visage: towards
   the face's, for a family that paints a shadow on the neck, VRoid); the UVs of the piece edge are pulled
   in by `INSET_TEXELS` (dark band at the edge of the studio's UV islands).

Output: <sortie>/result.pmx (textures in tex/), seam.json. Animation test:
tools/clothes/head_motion_test.py.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))
sys.path.insert(0, os.path.join(HERE, "..", "head"))
sys.path.insert(0, HERE)
import pmx as PMX  # noqa: E402
import pmx_write as pmx_write  # noqa: E402
import head_graft as graft_head  # noqa: E402

MM = 1.0 / 80.0
PART = "肌_tete"
FACE = ("颜", "口舌", "齿", "眉", "睫", "白目", "目", "目影", "目光", "二重", "口线", "颜_A", "睫眉", "颜2")
RADIUS_SHAPE_MM = 40.0
RADIUS_WEIGHT_MM = 15.0
RADIUS_WEIGHT_NECK_MM = (10.0, 25.0)   # head neck weights up to the 1st, body weights after the 2nd, harmonic in between
RADIUS_TINT_MM = 30.0
RADIUS_SMOOTHING_MM = 20.0
INSET_TEXELS = 7.0      # 4-texel dark band at the edge of the piece's UV islands, plus filtering
HOLE_MAX = 6
SKULL_MIN_MM = 60.0           # a scalp edge spans at least this much; smaller: it is not one
SLIT_MAX = 0.01              # area / perimeter² below which an edge is a slit (circle: 0.08)


def args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", dest="body", required=True)
    p.add_argument("--head", dest="head", required=True)
    p.add_argument("--output", dest="output", required=True)
    p.add_argument("--skin", dest="skin", default="肌")
    p.add_argument("--max-neck-gap", dest="neck_gap_max", default="10", help="tolerated offset between the 首 bones of the body and of the head "
                   "placed by 頭; the body with our own shape has a longer neck than Yuzuha's head: 20")
    p.add_argument("--base-tint", dest="tint_base", default="loop", choices=("loop", "face"),
                   help="body skin tint away from the join: that of the neck ring, or that of the face "
                   "when the head's neck carries a painted shadow")
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return p.parse_args(argv)


# ---------------------------------------------------------------------- surfaces

def weld(pos, tol=1e-4):
    _, inv = np.unique(np.round(pos / tol).astype(np.int64), axis=0, return_inverse=True)
    return inv.ravel()


def components(F, sub):
    """Component (through welded vertices) of each face."""
    n = int(sub.max()) + 1
    per = np.arange(n)

    def root(x):
        while per[x] != x:
            per[x] = per[per[x]]
            x = per[x]
        return x
    for a, b, c in sub[F]:
        ra, rb, rc = root(a), root(b), root(c)
        per[rb] = ra
        per[root(rc)] = ra
    return np.array([root(x) for x in sub[F[:, 0]]])


def loops(F, sub):
    """Ordered edges (raw indices, one per welded position), in face winding order."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    S = np.sort(sub[E], axis=1)
    _, inv, cnt = np.unique(S, axis=0, return_inverse=True, return_counts=True)
    B = E[cnt[inv.ravel()] == 1]
    next_ = {int(sub[a]): (int(sub[b]), int(a), int(b)) for a, b in B}
    raw = {}
    for a, b in B:
        raw.setdefault(int(sub[a]), int(a))
    seen, out = set(), []
    for s in next_:
        if s in seen:
            continue
        L, x = [], s
        while x not in seen and x in next_:
            seen.add(x)
            L.append(raw[x])
            x = next_[x][0]
        if x == s and len(L) >= 3:
            out.append(np.array(L))
    return out


def crosses(o, d, T):
    """For each ray (o, d), true if it hits one of the triangles T (Moller-Trumbore)."""
    v0, e1, e2 = T[:, 0], T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    res = np.zeros(len(d), bool)
    for k in range(0, len(d), 256):
        D = d[k:k + 256, None, :]
        p = np.cross(D, e2[None])
        det = (e1[None] * p).sum(2)
        ok = np.abs(det) > 1e-12
        inv = 1.0 / np.where(ok, det, 1.0)
        s = o - v0
        u = (s[None] * p).sum(2) * inv
        q = np.cross(s, e1)
        v = (D * q[None]).sum(2) * inv
        t = (e2 * q).sum(1)[None] * inv
        res[k:k + 256] = (ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)).any(1)
    return res


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def point_polyline(q, L):
    """Closest point of each q on the closed loop L: (point, i, j, u) with point = L[i] + u (L[j] - L[i])."""
    a, b = L, np.roll(L, -1, axis=0)
    ab = b - a
    u = np.clip(((q[:, None] - a) * ab).sum(2) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    pts = a + u[..., None] * ab
    d = ((q[:, None] - pts) ** 2).sum(2)
    k = d.argmin(1)
    r = np.arange(len(q))
    return pts[r, k], k, (k + 1) % len(L), u[r, k], np.sqrt(d[r, k])


def angles(P, center, axis):
    """Angle of each point around the axis (frame of the perpendicular plane)."""
    axis = axis / np.linalg.norm(axis)
    e1 = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    Q = P - center
    return np.arctan2(Q @ e2, Q @ e1)


def closure(A, B, PA, PB, center, axis, params=None):
    """Triangles closing the strip between two closed loops A and B (indices), walked by increasing angle
    around the axis (or by `params`, precomputed angles); each triangle has two vertices from one loop
    and one from the other."""
    ta, tb = params if params is not None else (angles(PA, center, axis), angles(PB, center, axis))

    def order(L, t):
        # increasing direction: most angle steps are positive
        dt = np.angle(np.exp(1j * (np.roll(t, -1) - t)))
        if (dt > 0).sum() < len(dt) / 2:
            L, t = L[::-1], t[::-1]
        k = int(np.argmin(np.mod(t, 2 * np.pi)))
        L, t = np.roll(L, -k), np.roll(t, -k)
        t = np.mod(t - t[0], 2 * np.pi)
        # unwrapped, increasing angle (a local step back is flattened)
        t = np.maximum.accumulate(t)
        return L, t
    A, ta = order(np.asarray(A), ta)
    B, tb = order(np.asarray(B), tb)
    # aligned starts: B starts at the point closest in angle to A's start
    ta2 = np.append(ta, 2 * np.pi)
    tb2 = np.append(tb, 2 * np.pi)
    A2, B2 = np.append(A, A[0]), np.append(B, B[0])
    i = j = 0
    tri = []
    while i < len(A) or j < len(B):
        if j >= len(B) or (i < len(A) and ta2[i + 1] <= tb2[j + 1]):
            tri.append((A2[i], A2[i + 1], B2[j]))
            i += 1
        else:
            tri.append((A2[i], B2[j + 1], B2[j]))
            j += 1
    return np.array(tri, np.int64)


def harmonic(n, edges, fixed, val, iterations=6000):
    """Extend `val` (imposed where `fixed`) over the graph by neighbour averaging."""
    x = val.copy()
    a, b = edges[:, 0], edges[:, 1]
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    free = ~fixed & (deg > 0)
    for _ in range(iterations):
        total = np.stack([np.bincount(a, x[b, k], n) + np.bincount(b, x[a, k], n) for k in range(x.shape[1])], 1)
        nv = total[free] / deg[free][:, None]
        gap = np.abs(nv - x[free]).max() if free.any() else 0.0
        x[free] = nv
        if gap < 1e-7:
            break
    return x


def place_on(A, B):
    """Points of the closed loop B at the arc lengths of the vertices of loop A (same start and direction
    as B: start nearest to B[0], best-fitting direction); also returns those arc lengths and those of
    B's vertices, in [0, 1)."""
    def abscissas(X):
        l = np.linalg.norm(np.roll(X, -1, 0) - X, axis=1)
        return np.concatenate([[0], np.cumsum(l)[:-1]]) / l.sum(), l.sum()
    sb, _ = abscissas(B)
    Bf = np.vstack([B, B[:1]])
    sbf = np.append(sb, 1.0)

    def on_b(u):
        return np.stack([np.interp(u % 1.0, sbf, Bf[:, k]) for k in range(3)], 1)
    k0 = int(np.argmin(np.linalg.norm(A - B[0], axis=1)))
    best = None
    for direction in (1, -1):
        Ar = np.roll(A[::direction], -(k0 if direction == 1 else len(A) - 1 - k0), 0)
        sa, _ = abscissas(Ar)
        Q = on_b(sa)
        e = np.linalg.norm(Q - Ar, axis=1).mean()
        if best is None or e < best[0]:
            best = (e, direction, Q, sa)
    _, direction, Q, sa = best
    # back in A's order, with each point's arc length on B
    out = np.empty_like(A)
    s_out = np.empty(len(A))
    order = np.roll(np.arange(len(A))[::direction], -(k0 if direction == 1 else len(A) - 1 - k0))
    out[order] = Q
    s_out[order] = sa
    return out, s_out, sb


# ---------------------------------------------------------------------- head piece

def part_of_head(path, folder, skin="肌"):
    """Temporary PMX: the skin material keeps only the faces welded to the face, renamed PIECE;
    textures as absolute paths (the PMX no longer sits next to them)."""
    H = PMX.read(path)
    t = PMX.to_arrays(H)
    names = [m["name"] for m in H["materials"]]
    iv, ip = names.index("颜"), names.index(skin)
    sel = np.isin(t["mat_face"], [iv, ip])
    F = t["faces"][sel]
    comp = components(F, weld(t["pos"]))
    of_face = np.isin(comp, np.unique(comp[t["mat_face"][sel] == iv]))
    keep_skin = np.zeros(len(t["faces"]), bool)
    keep_skin[np.where(sel)[0][of_face & (t["mat_face"][sel] == ip)]] = True
    faces, mats = [], []
    for i, m in enumerate(H["materials"]):
        Fi = t["faces"][(t["mat_face"] == i) & (keep_skin if i == ip else True)]
        mm = dict(m)
        if i == ip:
            mm["name"] = PART
        mm["n_indices"] = 3 * len(Fi)
        mats.append(mm)
        faces.append(Fi)
    H["materials"] = mats
    H["faces"] = np.vstack(faces).tolist()
    base = os.path.dirname(path)
    H["textures"] = [os.path.normpath(os.path.join(base, x.replace("\\", "/"))) for x in H["textures"]]
    out = os.path.join(folder, "head_part.pmx")
    pmx_write.write(H, out + ".tmp")
    os.replace(out + ".tmp", out)
    return out, int(keep_skin.sum())


# ---------------------------------------------------------------------- weights

SKIN_SOURCE = ("颜", "肌", "体", "体2")   # skin and body of the original head (no hair, no skirt)


def plus_near_triangle(p, T):
    """Barycentric coordinates (k, 3) of the point of each triangle T (k, 3, 3) closest to p, and
    distance."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    n = np.cross(b - a, c - a)
    nn = np.maximum((n * n).sum(1), 1e-30)
    q = p - ((p - a) * n).sum(1)[:, None] * n / nn[:, None]
    def bary(x):
        v0, v1, v2 = b - a, c - a, x - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        return np.stack([1 - w1 - w2, w1, w2], 1)
    bc = bary(q)
    inside = (bc >= 0).all(1)
    best_bc = bc.copy()
    best = np.where(inside, np.linalg.norm(q - p, axis=1), np.inf)
    for i0, i1 in ((0, 1), (1, 2), (2, 0)):
        e0, e1 = T[:, i0], T[:, i1]
        e = e1 - e0
        tt = np.clip(((p - e0) * e).sum(1) / np.maximum((e * e).sum(1), 1e-30), 0, 1)
        x = e0 + tt[:, None] * e
        dd = np.linalg.norm(x - p, axis=1)
        m = dd < best
        b_ = np.zeros((len(T), 3))
        b_[:, i0], b_[:, i1] = 1 - tt, tt
        best_bc[m], best[m] = b_[m], dd[m]
    return best_bc, best

def weight_dict(os_, pw):
    d = {}
    for b, w in zip(os_, pw):
        if b >= 0 and w > 0:
            d[int(b)] = d.get(int(b), 0.0) + float(w)
    return d


def weight_write(d):
    it = sorted(d.items(), key=lambda x: -x[1])[:4]
    s = sum(w for _, w in it)
    it = [(b, w / s) for b, w in it]
    while len(it) < 4:
        it.append((-1, 0.0))
    return 2, tuple(b for b, _ in it), tuple(w for _, w in it)


def blend(d1, d2, s):
    out = {b: (1 - s) * w for b, w in d1.items()}
    for b, w in d2.items():
        out[b] = out.get(b, 0.0) + s * w
    return out


# ---------------------------------------------------------------------- seam

def sew(R, skin, log, source=None):
    s = R["vertices"]
    t = PMX.to_arrays(R)
    P = t["pos"].copy()
    names = [m["name"] for m in R["materials"]]
    i_skin = names.index(skin)
    i_part = [names.index(n) for n in ("颜", PART) if n in names]
    mf = t["mat_face"]
    F = t["faces"]
    sub = weld(P)
    Fp = F[np.isin(mf, i_part)]

    # piece edges: neck cut (lowest), scalp edge (tallest in height), small holes
    L = loops(Fp, sub)
    L.sort(key=lambda b: P[b, 1].mean())
    loop = L[0]
    others = [b for b in L[1:]]
    # closed head (whole scalp, like VRoid's): no edge goes around the scalp; the body then has nothing to
    # stitch at the top, and everything it has above the neck cut goes away
    def slit(b):       # edge with no opening: two poorly welded lips (near-zero area for its perimeter)
        Q = P[b]
        Qc = Q - Q.mean(0)
        area = np.linalg.norm(np.cross(Qc, np.roll(Qc, -1, axis=0)).sum(0)) / 2
        return area < SLIT_MAX * np.linalg.norm(Q - np.roll(Q, -1, axis=0), axis=1).sum() ** 2
    wide = [b for b in others if float(np.ptp(P[b], axis=0).max()) >= SKULL_MIN_MM * MM and not slit(b)]
    skull = max(wide, key=lambda b: np.ptp(P[b, 1]) * len(b)) if wide else None
    closed = skull is None
    if closed:
        skull = np.zeros(0, int)
    holes = [b for b in others if len(b) <= HOLE_MAX and b is not skull]
    y_neck = float(P[loop, 1].mean())
    c_neck = P[loop].mean(0)
    center = P[np.unique(Fp)].mean(0)
    log["loop"] = {"vertices": len(loop), "height": round(y_neck, 4),
                         "gap_height_mm": round(float(np.ptp(P[loop, 1]) / MM), 2)}
    log["border_skull"] = len(skull)
    log["head_closed"] = bool(closed)

    # holes filled as a fan, in the winding of the neighbouring faces (edge walked backwards)
    mouth = []
    for b in holes:
        for k in range(1, len(b) - 1):
            mouth.append((b[0], b[k + 1], b[k]))
    log["holes_bouches"] = len(holes)

    # body skin under the piece
    Fc_all = np.where(mf == i_skin)[0]
    Fc = F[Fc_all]
    vc = np.unique(Fc)
    nearest = vc[P[vc, 1] > y_neck - 0.5]
    d = P[nearest] - center
    vu = crosses(center, d / np.linalg.norm(d, axis=1, keepdims=True), P[Fp])
    under = np.zeros(len(P), bool)
    under[nearest[vu & (P[nearest, 1] > y_neck - 0.2 * MM)]] = True
    # at the neck, everything above the cut inside the tube goes away too
    r_neck = np.linalg.norm((P[loop] - c_neck)[:, [0, 2]], axis=1).max()
    tube = (np.linalg.norm((P - c_neck)[:, [0, 2]], axis=1) < 1.6 * r_neck) & (P[:, 1] > y_neck) & (P[:, 1] < y_neck + 0.6)
    under |= tube
    removed = under[Fc].any(1)
    log["body_faces_removed"] = int(removed.sum())
    Fc_g = Fc[~removed]

    # body edges: the one below the ring and the one around the scalp edge
    Lc = loops(Fc_g, sub)

    def near_of(b, target):
        return point_polyline(P[b], P[target])[4].mean()
    bottom = min(Lc, key=lambda b: near_of(b, loop))
    # the scalp edge runs along the piece's from end to end, both ways: a small edge left close by
    # (ear hole of a body that has a face) is not one
    top = np.zeros(0, int) if closed else min([b for b in Lc if b is not bottom], key=lambda b: max(near_of(b, skull), near_of(skull, b)))
    # a body that has a face keeps leftovers of it under the piece (mouth, eye sockets): islands above the
    # cut, attached neither to the body nor to the scalp, removed
    island = components(Fc_g, sub)
    holds = [island[np.isin(Fc_g, b).any(1)][0] for b in (bottom, top) if len(b)]
    rest = ~np.isin(island, holds) & (P[Fc_g][:, :, 1].min(1) > y_neck)
    rest = np.isin(island, np.setdiff1d(island[rest], island[~rest]))
    if rest.any():
        log["body_faces_islands"] = int(rest.sum())
        Fc_g = Fc_g[~rest]
    log["border_body_bottom"] = {"vertices": len(bottom), "gap_average_mm": round(float(near_of(bottom, loop) / MM), 2)}
    if not closed:
        log["border_body_top"] = {"vertices": len(top), "gap_average_mm": round(float(near_of(top, skull) / MM), 2)}

    # shape: the body adjusts to the piece near both edges
    keep_v = np.unique(Fc_g)
    radius = RADIUS_SHAPE_MM * MM
    displacement = np.zeros_like(P)
    # bottom: radius around the neck axis, taken from the ring by angle
    ax = np.array([0.0, 1.0, 0.0])
    ta = angles(P[loop], c_neck, ax)
    ra = np.linalg.norm((P[loop] - c_neck)[:, [0, 2]], axis=1)
    o = np.argsort(ta)
    ta_s, ra_s = ta[o], ra[o]
    v_bottom = keep_v[(P[keep_v, 1] <= y_neck + 1e-6) & (P[keep_v, 1] > y_neck - radius)
                    & (np.linalg.norm((P[keep_v] - c_neck)[:, [0, 2]], axis=1) < 2.0 * r_neck)]
    tb = angles(P[bottom], c_neck, ax)
    rb = np.linalg.norm((P[bottom] - c_neck)[:, [0, 2]], axis=1)
    target_b = np.interp(tb, ta_s, ra_s, period=2 * np.pi)
    db = target_b - rb                                   # radius offset at the edge, by angle
    o2 = np.argsort(tb)
    tv = angles(P[v_bottom], c_neck, ax)
    dv = np.interp(tv, tb[o2], db[o2], period=2 * np.pi)
    hv = np.clip((P[bottom, 1].mean() - P[v_bottom, 1]) / radius, 0, 1)
    radial = (P[v_bottom] - c_neck) * np.array([1.0, 0.0, 1.0])
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    displacement[v_bottom] += radial * (dv * (1 - smooth(hv)))[:, None]
    log["adjustment_neck_mm"] = {"max": round(float(np.abs(db).max() / MM), 2),
                                    "average": round(float(np.abs(db).mean() / MM), 2)}
    # top: the body edge is laid on the scalp edge by arc length (order kept, no fold when several
    # vertices target the same point), the rest of the scalp follows through a harmonic field on the
    # mesh, zero at the neck; a point-by-point fit crumpled the edge
    E = np.vstack([Fc_g[:, [0, 1]], Fc_g[:, [1, 2]], Fc_g[:, [2, 0]]])
    E = np.unique(np.sort(sub[E], 1), axis=0)
    E = E[E[:, 0] != E[:, 1]]
    s_top = s_skull = None
    if not closed:
        target_h, s_top, s_skull = place_on(P[top], P[skull])
        dh = np.linalg.norm(target_h - center, axis=1) - np.linalg.norm(P[top] - center, axis=1)
        fixed = np.zeros(len(P), bool)
        val = np.zeros_like(P)
        fixed[sub[keep_v[P[keep_v, 1] < y_neck + 0.2]]] = True
        fixed[sub[top]] = True
        val[sub[top]] = target_h - P[top]
        field = harmonic(len(P), E, fixed, val)
        v_top = keep_v[(P[keep_v, 1] >= y_neck + 0.2) | np.isin(sub[keep_v], sub[top])]
        displacement[v_top] += field[sub[v_top]]
        log["adjustment_skull_mm"] = {"max": round(float(np.abs(dh).max() / MM), 2),
                                          "average": round(float(np.abs(dh).mean() / MM), 2)}
    # (the welded copies of a vertex share the same position, hence the same displacement)
    P += displacement
    if not closed:
        # body rows behind the scalp edge: the old stair-stepped edge leaves teeth there, smoothed
        # (Taubin, without shrinking) over RADIUS_SMOOTHING_MM, the laid edge staying fixed
        Pw = np.zeros_like(P)
        Pw[sub] = P
        Pw[sub[keep_v]] = P[keep_v]                       # a vertex of another material welded there does not count
        d_border = np.linalg.norm(Pw[:, None] - P[top][None], axis=2).min(1)
        zone = (d_border < RADIUS_SMOOTHING_MM * MM) & ~np.isin(np.arange(len(P)), sub[top])
        zone &= np.isin(np.arange(len(P)), sub[keep_v[P[keep_v, 1] > y_neck + 0.2]])
        a_, b_ = E[:, 0], E[:, 1]
        deg = np.bincount(a_, minlength=len(P)) + np.bincount(b_, minlength=len(P))
        zone &= deg > 0
        for _ in range(30):
            for f_ in (0.5, -0.53):
                m = np.stack([np.bincount(a_, Pw[b_, k], len(P)) + np.bincount(b_, Pw[a_, k], len(P)) for k in range(3)], 1)
                m = m[zone] / deg[zone][:, None]
                Pw[zone] += f_ * (m - Pw[zone])
        P[keep_v] = Pw[sub[keep_v]]
        log["smoothing_border_skull"] = int(zone.sum())
        _, _, _, _, gap_h = point_polyline(P[top], P[skull])
        _, _, _, _, gap_h = point_polyline(P[top], P[skull])
        log["gap_border_skull_mm"] = round(float(gap_h.max() / MM), 4)

    # body weights and normals blended towards the piece edge
    N = t["nor"].copy()
    W = [weight_dict(s["bones"][i], s["weights"][i]) for i in range(len(P))]
    rp = RADIUS_WEIGHT_MM * MM
    new_type = {}
    # neck: below the ring, the weights of the head's original neck (made for that ring), projected onto its
    # triangles up to RADIUS_WEIGHT_NECK_MM[0], the body's beyond RADIUS_WEIGHT_NECK_MM[1], and in between
    # the harmonic field (the smoothest transition). A short blend towards the ring let the 頭 weight
    # drop from 0.8 to 0.1 within a few mm at the front, and the skin collapsed there when the head turned.
    r_in, r_out = (r * MM for r in RADIUS_WEIGHT_NECK_MM)
    _, _, _, _, d_an = point_polyline(P[keep_v], P[loop])
    d_cr = np.full(len(keep_v), np.inf) if closed else point_polyline(P[keep_v], P[skull])[4]
    near = d_an < d_cr
    twins = {}
    for v in keep_v.tolist():
        twins.setdefault(int(sub[v]), []).append(v)
    lab_bottom = set(sub[bottom].tolist())
    interior = keep_v[(d_an < r_in) & near & ~np.isin(sub[keep_v], list(lab_bottom))]
    free = keep_v[(d_an >= r_in) & (d_an < r_out) & near]
    q, i, j, u, _ = point_polyline(P[bottom], P[loop])
    for v, a_, b_, uu in zip(bottom.tolist(), loop[i].tolist(), loop[j].tolist(), u.tolist()):
        wb = blend(W[a_], W[b_], uu)
        for x in twins.get(int(sub[v]), [v]):     # the body edge has duplicated vertices (UV seams)
            W[x] = dict(wb)
            new_type[x] = True
    n_src = 0
    if source is not None:
        Hs = PMX.read(source[0])
        ts = PMX.to_arrays(Hs)
        Ps = ts["pos"] + np.asarray(source[1]) * MM
        ns = [m["name"] for m in Hs["materials"]]
        Fs_ = ts["faces"][np.isin(ts["mat_face"], [k for k, n in enumerate(ns) if n in SKIN_SOURCE])]
        g_ = Ps[Fs_].mean(1)
        Fs_ = Fs_[(g_[:, 1] < y_neck + 0.3) & (np.linalg.norm((g_ - c_neck)[:, [0, 2]], axis=1) < 2 * r_neck)]
        name_os = {o["name"]: k for k, o in enumerate(R["bones"])}
        os_src = [name_os.get(o["name"], -1) for o in Hs["bones"]]
        Ws = [{os_src[b]: w for b, w in weight_dict(ts["bones"][k], ts["weights"][k]).items() if os_src[b] >= 0}
              for k in range(len(Ps))]
        G = Ps[Fs_].mean(1)
        for v in interior.tolist():
            candidate = np.where(np.linalg.norm(G - P[v], axis=1) < 20 * MM)[0]
            if not len(candidate):
                continue
            bc, dist = plus_near_triangle(P[v], Ps[Fs_[candidate]])
            k = int(np.argmin(dist))
            if dist[k] > 8 * MM:
                continue
            d = {}
            for vert, c in zip(Fs_[candidate[k]], bc[k]):
                for b, w in Ws[vert].items():
                    d[b] = d.get(b, 0.0) + c * w
            tot = sum(d.values())
            if tot > 0:
                W[v] = {b: w / tot for b, w in d.items()}
                new_type[v] = True
                n_src += 1
    zone = np.r_[interior, free, bottom]
    os_neck = sorted({b for v in zone.tolist() for b in W[v]})
    collar = {b: k for k, b in enumerate(os_neck)}
    Wm = np.zeros((len(P), len(os_neck)))
    for v in keep_v.tolist():
        for b, w in W[v].items():
            if b in collar:
                Wm[sub[v], collar[b]] = w
    fixed_w = np.zeros(len(P), bool)
    fixed_w[sub[keep_v]] = True
    fixed_w[sub[free]] = False
    Wm = harmonic(len(P), E, fixed_w, Wm)
    for v in free.tolist():
        d = {os_neck[k]: float(w) for k, w in enumerate(Wm[sub[v]]) if w > 1e-4}
        d.update({b: w for b, w in W[v].items() if b not in collar})
        W[v] = d
        new_type[v] = True
    zone_c = np.r_[interior, free]
    log["weight_neck"] = {"vertices": int(len(zone_c)), "resumed_of_head": n_src,
                            "harmoniques": int(len(free)), "rays_mm": list(RADIUS_WEIGHT_NECK_MM)}
    for border_c, border_p in ((bottom, loop), (top, skull))[:1 if closed else 2]:
        vz = keep_v[np.linalg.norm(P[keep_v][:, None] - P[border_c][None], axis=2).min(1) < rp] if len(keep_v) else []
        q, i, j, u, dist = point_polyline(P[vz], P[border_p])
        for v, a, b, uu, dd_ in zip(vz.tolist(), border_p[i].tolist(), border_p[j].tolist(), u.tolist(), dist.tolist()):
            sx = 1 - smooth(dd_ / rp)
            if border_p is skull:
                W[v] = blend(W[v], blend(W[a], W[b], uu), sx)
            nb = (1 - uu) * N[a] + uu * N[b]
            n_ = (1 - sx) * N[v] + sx * nb
            N[v] = n_ / max(np.linalg.norm(n_), 1e-9)
            new_type[v] = True

    # strips: copies of the piece's edge vertices, body material
    copies = {}
    uv = t["uv"]

    def copied(v, uv_body):
        if v in copies:
            return copies[v]
        k = len(s["pos"])
        for key in ("pos", "nor", "type", "bones", "weights", "sdef", "contour", "uv_add"):
            s[key].append(s[key][v])
        s["uv"].append(tuple(uv_body))
        copies[v] = k
        return k
    bands = []
    for border_c, border_p, axis_, cen in ((bottom, loop, np.array([0.0, 1.0, 0.0]), c_neck),
                                       (top, skull, None, None))[:1 if closed else 2]:
        if axis_ is None:
            # scalp edge: axis = mean normal of the edge plane
            Q = P[border_p] - P[border_p].mean(0)
            axis_ = np.linalg.svd(Q)[2][2]
            cen = P[border_p].mean(0)
        _, kk, _, _, _ = point_polyline(P[border_p], P[border_c])
        cp = [copied(int(v), uv[border_c[kk[n]]]) for n, v in enumerate(border_p)]
        params = None
        if border_p is skull:
            # the body edge is laid on the scalp's: seam by the arc lengths of that placement (a nearest
            # point picks the wrong segment where the edge zigzags)
            params = (2 * np.pi * s_top, 2 * np.pi * s_skull)
        T = closure(border_c, np.array(cp), P[border_c], P[border_p], cen, axis_, params)
        # winding: that of the neighbouring body faces
        idx = {int(c): n for n, c in enumerate(cp)}
        def pos_(x):
            return P[x] if x < len(P) else P[border_p[idx[x]]]
        nt = np.array([np.cross(pos_(b) - pos_(a), pos_(c) - pos_(a)) for a, b, c in T])
        ext = np.array([pos_(a) - (center if border_p is skull else c_neck) for a, b, c in T])
        if border_p is loop:
            ext[:, 1] = 0
        if ((nt * ext).sum(1) < 0).mean() > 0.5:
            T = T[:, [0, 2, 1]]
        bands.append(T)
    bands = np.vstack(bands)
    log["faces_bands"] = int(len(bands))

    # write positions, normals, weights
    for v in range(len(P)):
        s["pos"][v] = tuple(P[v].tolist())
    for v in new_type:
        s["nor"][v] = tuple(N[v].tolist())
        ty, b, w = weight_write(W[v])
        s["type"][v], s["bones"][v], s["weights"][v], s["sdef"][v] = ty, b, w, None
    for v, k in copies.items():
        s["pos"][k] = s["pos"][v]

    # morphs: the copies follow their original
    for m in R["morphs"]:
        if m["type"] != 1:
            continue
        addition = [(copies[i], d) for i, d in m["offsets"] if i in copies]
        m["offsets"] = list(m["offsets"]) + addition

    # faces: body without the part under the piece, plus the strips; piece plus its filled holes
    faces = []
    for i, m in enumerate(R["materials"]):
        Fi = F[mf == i]
        if i == i_skin:
            Fi = np.vstack([Fc_g, bands])
        elif names[i] == PART and mouth:
            Fi = np.vstack([Fi, np.array(mouth)])
        m["n_indices"] = 3 * len(Fi)
        faces.append(Fi)
    R["faces"] = np.vstack(faces).tolist()
    return {"loop": loop, "skull": skull, "copies": copies, "bottom": bottom, "top": top}


# ---------------------------------------------------------------------- tint

def tint(R, folder, log, borders, copies=(), radius=RADIUS_TINT_MM * MM, base_of="loop"):
    """Body skin painted: each texel takes the piece's colour at the nearest edge (neck ring, scalp edge),
    blended towards the tint of the bottom of the neck at `rayon`. The UVs of the piece edge are pulled in
    by seven texels: the band at the UV island edge is a darker bleed, which drew a fringe."""
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("tint: must run in Blender; nothing changed")
        log["tint"] = None
        return
    s = R["vertices"]
    t = PMX.to_arrays(R)
    names = [m["name"] for m in R["materials"]]
    i_skin = names.index("肌")
    idx = [names.index(n) for n in ("颜", PART) if n in names]

    def path(i):
        r = R["textures"][R["materials"][i]["texture"]]
        return r if os.path.isabs(r) else os.path.join(folder, r)
    images = {i: graft_head.read_image(path(i)) for i in idx}

    def texel(img, uv):
        h, w = img.shape[:2]
        return img[np.clip((uv[:, 1] % 1) * h, 0, h - 1).astype(int), np.clip((uv[:, 0] % 1) * w, 0, w - 1).astype(int), :3]

    # edge UVs pulled seven texels into the piece (along the edge, nothing moves)
    all_ = np.concatenate([np.asarray(b) for b in borders])
    is_border = np.zeros(len(t["pos"]), bool)
    is_border[all_] = True
    to = {v: [] for v in all_.tolist()}
    faces_border = {}
    for i in idx:
        Fi = t["faces"][t["mat_face"] == i]
        faces_border[i] = Fi[is_border[Fi].any(1)]
        for f in faces_border[i]:
            for k in range(3):
                v = int(f[k])
                if v in to:
                    to[v] += [(i, t["uv"][o] - t["uv"][v]) for o in (f[(k + 1) % 3], f[(k + 2) % 3]) if not is_border[o]]
    mat_of = {}
    for i in idx:
        for v in faces_border[i].ravel().tolist():
            mat_of.setdefault(v, i)
    seam = np.array([v for v in all_.tolist() if v in mat_of])
    uv = t["uv"].copy()
    for v in seam.tolist():
        if not to[v]:
            continue
        i, _ = to[v][0]
        h, w = images[i].shape[:2]
        d = np.mean([x for _, x in to[v]], 0)
        n = np.linalg.norm(d)
        if n > 1e-9:
            uv[v] = uv[v] + d / n * INSET_TEXELS / max(h, w)
            s["uv"][v] = tuple(uv[v])
    # edge colour: the one shown at the piece edge, read at the pulled-in UVs (read closer to the island
    # edge, it picked up the dark bleed and painted it on the body: fringe on the nape)
    collar = np.array([texel(images[mat_of[v]], uv[v][None])[0] for v in seam])
    # points along the edge's edges (the ring has only 16 vertices)
    pts, cpts, of_loop = [], [], []
    in_loop = np.zeros(len(t["pos"]), bool)
    in_loop[np.asarray(borders[0])] = True
    for i in idx:
        for f in faces_border[i][is_border[faces_border[i]].sum(1) >= 2]:
            for k in range(3):
                a_, b_ = f[k], f[(k + 1) % 3]
                if not (is_border[a_] and is_border[b_]):
                    continue
                for u in np.linspace(0.1, 0.9, 5):
                    pts.append((1 - u) * t["pos"][a_] + u * t["pos"][b_])
                    cpts.append(texel(images[i], ((1 - u) * uv[a_] + u * uv[b_])[None])[0])
                    of_loop.append(in_loop[a_] and in_loop[b_])
    pos_b = np.vstack([t["pos"][seam], np.array(pts).reshape(-1, 3)])
    collar_b = np.vstack([collar, np.array(cpts).reshape(-1, 3)])
    loop = np.concatenate([np.isin(seam, np.asarray(borders[0])), np.array(of_loop, bool)])
    base = np.median(collar_b[loop], 0)
    if base_of == "face" and "颜" in names:
        # neck with a painted shadow: the ring is in shadow, the body skin takes the face's tint
        i = names.index("颜")
        base = np.median(texel(images[i], t["uv"][np.unique(t["faces"][t["mat_face"] == i])]), 0)

    # colour per body skin vertex: weighted mean of the 8 nearest edge points, blended towards the base
    Fb = t["faces"][t["mat_face"] == i_skin]
    vb = np.unique(Fb)
    P = t["pos"]
    D = np.linalg.norm(P[vb][:, None] - pos_b[None], axis=2)
    k = np.argsort(D, 1)[:, :8]
    dk = np.take_along_axis(D, k, 1)
    wk = 1.0 / (dk ** 2 + (0.5 * MM) ** 2)
    nearest = (wk[..., None] * collar_b[k]).sum(1) / wk.sum(1)[:, None]
    a = smooth(np.clip(dk[:, 0] / radius, 0, 1))[:, None]
    cv = np.zeros((len(P), 3))
    cv[vb] = (1 - a) * nearest + a * base

    # painting: skin triangles rasterised into a new texture, colour interpolated from the vertices
    size = 1024
    img = np.tile(np.append(base, 1.0), (size, size, 1)).astype(np.float32)
    uvb = np.mod(t["uv"], 1.0) * size
    # strip faces painted first: their copies take the UVs of the nearest body vertex, some stretch in UV
    # over the nape; the body faces then repaint their own texels
    of_band = np.zeros(len(P), bool)
    of_band[np.asarray(copies, int)] = True
    for f in Fb[np.argsort(~of_band[Fb].any(1), kind="stable")]:
        p = uvb[f]
        x0, y0 = np.maximum(np.floor(p.min(0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(p.max(0)).astype(int) + 2, size)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        v0, v1 = p[1] - p[0], p[2] - p[0]
        det = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(det) < 1e-9:
            continue
        l1 = ((xs - p[0, 0]) * v1[1] - (ys - p[0, 1]) * v1[0]) / det
        l2 = ((ys - p[0, 1]) * v0[0] - (xs - p[0, 0]) * v0[1]) / det
        l0 = 1 - l1 - l2
        margin = -1.5 / max(abs(det) ** 0.5, 1.0)     # one and a half texel overlap: no fringe between triangles
        inside = (l0 >= margin) & (l1 >= margin) & (l2 >= margin)
        c = l0[..., None] * cv[f[0]] + l1[..., None] * cv[f[1]] + l2[..., None] * cv[f[2]]
        img[y0:y1, x0:x1, :3][inside] = c[inside]
    name = "body_skin.png"
    graft_head.write_image(os.path.join(folder, "tex", name), img)
    R["textures"].append("tex/" + name)
    R["materials"][i_skin]["texture"] = len(R["textures"]) - 1
    lab = graft_head.srgb_to_lab(collar_b)
    log["tint"] = {"base": [round(float(x), 4) for x in base], "points_of_border": len(collar_b),
                         "dispersion_lab_borders": round(float(np.median(np.linalg.norm(lab - np.median(lab, 0), axis=1))), 2),
                         "gap_lab_max_borders": round(float(np.linalg.norm(lab - graft_head.srgb_to_lab(base), axis=1).max()), 2),
                         "radius_mm": round(radius / MM, 1), "base_of": base_of}


def main(argv=None):
    a = args(argv)
    os.makedirs(a.output, exist_ok=True)
    tmp = os.path.join(a.output, "_graft")
    os.makedirs(tmp, exist_ok=True)
    part_name, n_skin = part_of_head(a.head, tmp, a.skin)
    H = PMX.read(part_name)
    present = [m["name"] for m in H["materials"] if m["n_indices"]]
    take = [n for n in FACE if n in present] + [PART]
    graft_head.main(["--body", a.body, "--head", part_name, "--output", tmp, "--keep-skull", "--no-tint",
                      "--take", ",".join(take), "--skin", a.skin, "--max-neck-gap", a.neck_gap_max])
    R = PMX.read(os.path.join(tmp, "result.pmx"))
    log = {"body": a.body, "head": a.head, "skin_faces_head": n_skin}
    if os.path.isdir(os.path.join(a.output, "tex")):
        shutil.rmtree(os.path.join(a.output, "tex"))
    shutil.copytree(os.path.join(tmp, "tex"), os.path.join(a.output, "tex"))
    tr = json.load(open(os.path.join(tmp, "graft.json"), encoding="utf-8"))["translation_mm"]
    info = sew(R, a.skin, log, (a.head, tr))
    log["indices"] = {"loop": [int(x) for x in info["loop"]], "skull": [int(x) for x in info["skull"]],
                          "copies": {str(k): int(v) for k, v in info["copies"].items()}}
    tint(R, a.output, log, [b for b in (info["loop"], info["skull"]) if len(b)], list(info["copies"].values()),
           base_of=a.tint_base)
    path = os.path.join(a.output, "result.pmx")
    pmx_write.write(R, path + ".tmp")
    os.replace(path + ".tmp", path)
    log["vertices"] = len(R["vertices"]["pos"])
    log["faces"] = len(R["faces"])
    with open(os.path.join(a.output, "seam.json") + ".tmp", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    os.replace(os.path.join(a.output, "seam.json") + ".tmp", os.path.join(a.output, "seam.json"))
    print(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Weld the edge of the grafted face to the body skin, as the game models do.

    bin/blender-queue -b --factory-startup --python tools/clothes/neck_seam.py -- \
        --pmx MODEL.pmx --output FOLDER [--skin 肌] [--face 颜,颜_A]
    (without Blender, python3 -I does everything except the tint)

On Jane, Evelyn, Ellen, the face and the neck are a single surface: the skin ends exactly on the face
edge, under the jaw, with the same positions, the same normals, the same weights (all on 頭) and the
outline at 1 everywhere. The jaw line is then nothing but the silhouette of a continuous surface. A
skin-completed body, on the contrary, has a closed scalp ball whose bottom runs 5 to
15 mm under the face edge: a face laid on top of it reads as a disc.

What the tool does, on the skin material only (`head_graft.main` calls it by default):
 - if the skin is made of a closed scalp ball and a neck tube going through it, the two are first
   welded into a single surface (`join_tube`): otherwise their intersection edge and the tube's outline
   shell draw a boundary and dashes under the chin;
 - the skin under the face goes away: any face with a vertex under the face's footprint (ray from the
   head centre -> vertex that crosses the face);
 - the skin edge thus freed is glued onto the outer edge of the face: each face corner receives a skin
   vertex, the other vertices are projected onto the edges;
 - the rest of the skin follows through a harmonic displacement that fades out at `radius` along the
   surface: a gentle slope instead of a step;
 - normals: the face's on the weld, blended towards the skin's over `radius_normals`;
 - weights: the face's on the weld, blended over `radius_weight`; morphs that move the face edge carry the
   weld along. The face itself is not touched;
 - scalp: the body skin is pulled in behind the front silhouette of the face (`tuck_skull`);
 - line: the face outline thins from the mouth to the tip of the chin (`thin_chin`);
 - tint: along the weld, the skin texture takes the colour of the face edge, and returns to its own
   over `RADIUS_TINT_MM`: no more colour step. Must run in Blender (texture reading).

No effect if no skin edge is within reach (`reach`) of the face, or if that edge has fewer vertices
than the face has corners (summary: `skin_too_loose`): it is a neck stump hidden under the head.
Command-line output: <sortie>/result.pmx (tex/ textures copied over), seam.json.
"""
import argparse
import heapq
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))
sys.path.insert(0, os.path.join(HERE, "..", "head"))
import pmx as PMX  # noqa: E402
import pmx_write as pmx_write  # noqa: E402

MM = 1.0 / 80.0
RADIUS_MM = 40.0
RADIUS_NORMALS_MM = 15.0
RADIUS_WEIGHT_MM = 8.0
REACH_MM = 30.0
RADIUS_RELAXATION_MM = 6.0
RADIUS_TINT_MM = 25.0
STEP_SILHOUETTE_MM = 3.0     # height slices for the front silhouette of the face
NECK_UNDER_CHIN_MM = 20.0   # where the neck width is measured, under the chin
RADIUS_SKULL_MM = 12.0       # the scalp tightening fades out at this distance from the weld
TRAIT_CHIN_BOTTOM = 0.0       # outline weight at the tip of the chin (1 = full line)
TRAIT_CHIN_HEIGHT = 0.15  # mouth, as a fraction of the face height above the chin
PART_CORNERS_MAX = 0.1
RADIUS_TUBE_MM = 20.0
RADIUS_NORMALS_TUBE_MM = 12.0
RADIUS_WEIGHT_TUBE_MM = 10.0
GAP_MAX_TUBE_MM = 25.0
UNDER_FACE_MM = 3.0        # the body skin stays at least this deep under the face
REACH_UNDER_MM = 8.0        # beyond this, the skin is not under the face
BORDER_UNDER_MM = 4.0          # distance from the face edge at which the skin reaches that depth
RADIUS_TEST = np.array([0.6123, 0.4771, 0.6303])     # arbitrary direction: runs along no edge
RADIUS_HORIZONTAL = np.array([0.9998, 0.0011, 0.0173])


def weld(pos, tol=1e-4):
    """Welded index of each vertex (positions equal within tol)."""
    _, inv = np.unique(np.round(pos / tol).astype(np.int64), axis=0, return_inverse=True)
    return inv.ravel()


def edges_of_border(F, sub):
    """Edges (raw indices) that touch only one face, once the vertices are welded."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    S = np.sort(sub[E], axis=1)
    _, inv, cnt = np.unique(S, axis=0, return_inverse=True, return_counts=True)
    return E[cnt[inv.ravel()] == 1]


def point_segment(q, a, b):
    """Closest point of each q on the best segment (a, b): (point, index, distance)."""
    ab = b - a
    t = np.clip(((q[:, None] - a) * ab).sum(2) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    pts = a + t[..., None] * ab
    d = ((q[:, None] - pts) ** 2).sum(2)
    k = d.argmin(1)
    return pts[np.arange(len(q)), k], k, np.sqrt(d[np.arange(len(q)), k])


def border_outer(E, sub, pos):
    """Edges of the longest boundary loop: the outline of the face, without the eye and mouth holes."""
    s = sub[E]
    lab = np.arange(sub.max() + 1)
    while True:
        m = np.minimum(lab[s[:, 0]], lab[s[:, 1]])
        nv = lab.copy()
        np.minimum.at(nv, s[:, 0], m)
        np.minimum.at(nv, s[:, 1], m)
        nv = nv[nv]
        if np.array_equal(nv, lab):
            break
        lab = nv
    groups = lab[s[:, 0]]
    length = np.linalg.norm(pos[E[:, 0]] - pos[E[:, 1]], axis=1)
    total = np.bincount(groups, length, minlength=len(lab))
    return E[groups == int(total.argmax())]


def rays_face(C, q, tri):
    """Distance from C to the first crossing of the surface `tri` along the ray C -> q; inf if none."""
    A, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    d = q - C
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    res = np.full(len(q), np.inf)
    for s in range(0, len(q), 256):
        dd = d[s:s + 256]
        h = np.cross(dd[:, None], e2[None])
        a = (e1[None] * h).sum(2)
        on = np.abs(a) > 1e-12
        f = np.where(on, 1.0 / np.where(on, a, 1.0), 0.0)
        w = C[None, None] - A[None]
        u = f * (w * h).sum(2)
        qq = np.cross(w, e1[None])
        v = f * (dd[:, None] * qq).sum(2)
        t = f * (e2[None] * qq).sum(2)
        ok = on & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 1e-6)
        res[s:s + 256] = np.where(ok, t, np.inf).min(1)
    return res


def groups_edges(E, sub):
    """Loop number (connected component over welded vertices) of each boundary edge."""
    s = sub[E]
    lab = np.arange(sub.max() + 1)
    while True:
        m = np.minimum(lab[s[:, 0]], lab[s[:, 1]])
        nv = lab.copy()
        np.minimum.at(nv, s[:, 0], m)
        np.minimum.at(nv, s[:, 1], m)
        nv = nv[nv]
        if np.array_equal(nv, lab):
            return lab[s[:, 0]]
        lab = nv


def choose_loops(pos, E, groups, corners, reach):
    """Skin boundary loops facing the face edge, from most to least covering."""
    remaining = np.ones(len(corners), bool)
    chosen = []
    uniques = np.unique(groups)
    covers = {}
    for g in uniques:
        v = np.unique(E[groups == g])
        d = np.sqrt(((pos[corners][:, None] - pos[v][None]) ** 2).sum(2)).min(1)
        covers[int(g)] = d < reach
    while True:
        g = max(covers, key=lambda k: int((covers[k] & remaining).sum()), default=None)
        if g is None or (covers[g] & remaining).sum() < max(2, 0.05 * len(corners)):
            return chosen
        chosen.append(g)
        remaining &= ~covers.pop(g)


def distances(n, edges, lengths, sources):
    """Distance along the surface to the nearest source, and that source (welded graph)."""
    neighbors = [[] for _ in range(n)]
    for (a, b), l in zip(edges.tolist(), lengths.tolist()):
        neighbors[a].append((b, l))
        neighbors[b].append((a, l))
    dist = np.full(n, np.inf)
    src = np.full(n, -1)
    pile = []
    for s in sources:
        dist[s] = 0.0
        src[s] = s
        pile.append((0.0, s))
    heapq.heapify(pile)
    while pile:
        d, a = heapq.heappop(pile)
        if d > dist[a]:
            continue
        for b, l in neighbors[a]:
            if d + l < dist[b]:
                dist[b] = d + l
                src[b] = src[a]
                heapq.heappush(pile, (d + l, b))
    return dist, src


def harmonic(n, edges, fixed, val, iterations=4000):
    """Extend `val` (imposed where `fixed`) over the graph by neighbour averaging."""
    x = val.copy()
    a, b = edges[:, 0], edges[:, 1]
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    free = ~fixed & (deg > 0)
    for _ in range(iterations):
        total = np.stack([np.bincount(a, x[b, k], n) + np.bincount(b, x[a, k], n) for k in range(3)], 1)
        nv = total[free] / deg[free][:, None]
        gap = np.abs(nv - x[free]).max() if free.any() else 0.0
        x[free] = nv
        if gap < 1e-7:
            break
    return x


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def blend_weight(os_a, p_a, os_b, p_b, t):
    """Blend (1-t) a + t b of two weight sets, four bones at most: (type, bones, weights)."""
    d = {}
    for o, w, f in list(zip(os_a, p_a, [1 - t] * 4)) + list(zip(os_b, p_b, [t] * 4)):
        if o >= 0 and w > 0:
            d[int(o)] = d.get(int(o), 0.0) + float(w) * f
    taken = sorted(d.items(), key=lambda kv: -kv[1])[:4]
    s = sum(w for _, w in taken)
    o = [k for k, _ in taken] + [-1] * (4 - len(taken))
    w = [v / s for _, v in taken] + [0.0] * (4 - len(taken))
    return {1: 0, 2: 1}.get(len(taken), 2), tuple(o), tuple(w)


def crossings(q, d, tri):
    """Number of triangles crossed by the ray q + t d (t > 0), for each q."""
    A, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    a = (e1 * h).sum(1)
    on = np.abs(a) > 1e-12
    A, e1, e2, h, f = A[on], e1[on], e2[on], h[on], 1.0 / a[on]
    n = np.zeros(len(q), int)
    for s in range(0, len(q), 128):
        w = q[s:s + 128, None] - A[None]
        u = f * (w * h[None]).sum(2)
        qq = np.cross(w, e1[None])
        v = f * (qq * d).sum(2)
        t = f * (qq * e2[None]).sum(2)
        n[s:s + 128] = ((u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)).sum(1)
    return n


def graph(F, sub):
    """Welded graph of faces F: (groups, number per group, representative, live vertices, edges)."""
    alive = np.unique(F)
    gs = np.unique(sub[alive])
    num = np.full(sub.max() + 1, -1)
    num[gs] = np.arange(len(gs))
    rep = np.zeros(len(gs), int)
    rep[num[sub[alive]]] = alive
    Eg = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    Eg = np.unique(np.sort(num[sub[Eg]], axis=1), axis=0)
    return gs, num, rep, alive, Eg[Eg[:, 0] != Eg[:, 1]]


def relax(x, Eg, free, iterations=12):
    """Smooth the positions x (one per graph node) over the free nodes."""
    a, b = Eg[:, 0], Eg[:, 1]
    n = len(x)
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    free = free & (deg > 0)
    for _ in range(iterations):
        total = np.stack([np.bincount(a, x[b, k], n) + np.bincount(b, x[a, k], n) for k in range(3)], 1)
        x[free] = 0.5 * x[free] + 0.5 * total[free] / deg[free][:, None]
    return x


def blend_normals(P, nor, F, num, sub, rep, n, v, k, d, src, n_weld, radius_normals, radius):
    """Weld normals, then new geometry, then original normals further away."""
    Fg = num[sub[F]]
    e1, e2 = P[rep[Fg[:, 1]]] - P[rep[Fg[:, 0]]], P[rep[Fg[:, 2]]] - P[rep[Fg[:, 0]]]
    nf = np.cross(e1, e2)
    n_geo = np.stack([np.bincount(Fg.ravel(), np.repeat(nf[:, c], 3), n) for c in range(3)], 1)
    n_geo /= np.maximum(np.linalg.norm(n_geo, axis=1, keepdims=True), 1e-12)
    n_src = np.zeros((n, 3))
    for g, x in n_weld.items():
        n_src[int(num[g])] = x
    n_src = n_src[np.maximum(src, 0)]
    # the PMX normal direction is authoritative: the geometry is flipped if its sign is reversed
    sign = np.sign((n_geo[k] * nor[v]).sum(1, keepdims=True))
    sign[sign == 0] = 1
    nv = n_geo[k] * sign
    w = smooth((d - radius_normals) / max(radius - radius_normals, 1e-9))[:, None]
    nv = nv * (1 - w) + nor[v] * w
    w = (1 - smooth(d / radius_normals))[:, None]
    nv = nv * (1 - w) + n_src[k] * w
    return nv / np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)


def join_tube(pos, nor, faces, mat_face, i_skin, os_, weight, lo, hi, radius=RADIUS_TUBE_MM * MM,
              radius_normals=RADIUS_NORMALS_TUBE_MM * MM, radius_weight=RADIUS_WEIGHT_TUBE_MM * MM):
    """Weld the neck tube to the closed scalp ball it goes through; None if the skin is not built that way.

    The part of the tube inside the ball and the bottom of the ball inside the tube go away; the edge of the
    ball's hole (denser) folds onto the corners of the tube edge: the edges coincide, with no gap.
    """
    idx = np.where(mat_face == i_skin)[0]
    F = faces[idx]
    sub = weld(pos)
    comp = groups_edges(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), sub)[:len(F)]
    ids = np.unique(comp).tolist()
    c = pos[F].mean(1)
    footprint = np.all((c >= lo) & (c <= hi), axis=1)
    ball, best = None, 99
    for i in ids:
        m = comp == i
        nb = int((m & footprint).sum())
        if nb > best and len(edges_of_border(F[m], sub)) <= 0.03 * m.sum():
            ball, best = i, nb
    if ball is None:
        return None
    F_b = F[comp == ball]
    tri_b = pos[F_b]
    bottom_b = tri_b[..., 1].min()
    tube, best = None, 7
    for i in ids:
        m = comp == i
        if i == ball or m.sum() < 50:
            continue
        v = np.unique(F[m])
        if pos[v, 1].min() > bottom_b - 30 * MM or pos[v, 1].max() < bottom_b:
            continue
        inside = crossings(pos[v], RADIUS_TEST, tri_b) % 2 == 1
        if inside.sum() > best and (~inside).sum() >= 8:
            tube, best = i, int(inside.sum())
    if tube is None:
        return None
    F_t = F[comp == tube]
    v_t = np.unique(F_t)
    in_b = np.zeros(len(pos), bool)
    in_b[v_t] = crossings(pos[v_t], RADIUS_TEST, tri_b) % 2 == 1
    v_b = np.unique(F_b)
    v_b = v_b[pos[v_b, 1] < pos[v_t, 1].max()]
    in_t = np.zeros(len(pos), bool)
    in_t[v_b] = crossings(pos[v_b], RADIUS_HORIZONTAL, pos[F_t]) % 2 == 1
    r_b, r_t = in_t[F_b].any(1), in_b[F_t].any(1)
    G_b, G_t = F_b[~r_b], F_t[~r_t]
    if not r_b.any() or not r_t.any() or not len(G_b) or not len(G_t):
        return None

    def new_border(G, F0):
        E = edges_of_border(G, sub)
        before = {tuple(x) for x in np.sort(sub[edges_of_border(F0, sub)], axis=1).tolist()}
        return border_outer(E[[tuple(x) not in before for x in np.sort(sub[E], axis=1).tolist()]], sub, pos)

    def uniques(E):
        v = np.unique(E)
        return v[np.unique(sub[v], return_index=True)[1]]
    corners = uniques(new_border(G_t, F_t))
    border = uniques(new_border(G_b, F_b))
    d = np.sqrt(((pos[corners][:, None] - pos[border][None]) ** 2).sum(2))
    taken = {}
    for c_ in np.argsort(d.min(1)).tolist():
        for j in np.argsort(d[c_]).tolist():
            if j not in taken:
                taken[j] = c_
                break
    to = d.argmin(0)
    for j, c_ in taken.items():
        to[j] = c_
    gap = np.linalg.norm(pos[corners[to]] - pos[border], axis=1).max()
    if gap > GAP_MAX_TUBE_MM * MM:
        return None

    # --- the ball's hole goes down onto the tube edge; the ball follows, the weld stays fixed
    gs, num, rep, alive, Eg = graph(G_b, sub)
    lengths = np.linalg.norm(pos[rep[Eg[:, 0]]] - pos[rep[Eg[:, 1]]], axis=1)
    sources = sorted(set(num[sub[border]].tolist()))
    dist, src = distances(len(gs), Eg, lengths, sources)
    fixed = dist >= radius
    val = np.zeros((len(gs), 3))
    k_b = num[sub[border]]
    fixed[k_b] = True
    val[k_b] = pos[corners[to]] - pos[border]
    disp = harmonic(len(gs), Eg, fixed, val)
    x = relax(pos[rep] + disp, Eg, (dist < RADIUS_RELAXATION_MM * MM) & ~np.isin(np.arange(len(gs)), sources))
    P = pos.copy()
    k_v = num[sub[alive]]
    P[alive] = x[k_v]

    # --- edge faces collapsed onto a single corner are removed
    sou2 = weld(P)
    flat = np.array([len(set(t)) < 3 for t in sou2[G_b].tolist()], bool)
    guard = np.ones(len(faces), bool)
    sel_b = idx[comp == ball]
    sel_t = idx[comp == tube]
    guard[sel_b[r_b]] = False
    guard[sel_t[r_t]] = False
    guard[sel_b[~r_b][flat]] = False

    # --- normals: mean of the two surfaces on the weld, blended on each side
    def average(n):
        return n / max(np.linalg.norm(n), 1e-12)
    n_corners = {}
    for c_ in range(len(corners)):
        n_b = nor[border[to == c_]]
        n_corners[c_] = average(nor[corners[c_]] + average(n_b.sum(0)) if len(n_b) else nor[corners[c_]])
    n_s_b = {int(sub[v]): n_corners[int(c_)] for v, c_ in zip(border.tolist(), to.tolist())}
    N = nor.copy()
    N[alive] = blend_normals(P, nor, G_b[~flat], num, sub, rep, len(gs), alive, k_v, dist[k_v], src, n_s_b, radius_normals, radius)
    gt, num_t, rep_t, viv_t, E_t = graph(G_t, sub)
    k_t = num_t[sub[viv_t]]
    long_t = np.linalg.norm(pos[rep_t[E_t[:, 0]]] - pos[rep_t[E_t[:, 1]]], axis=1)
    dist_t, src_t = distances(len(gt), E_t, long_t, sorted(set(num_t[sub[corners]].tolist())))
    n_s_t = {int(sub[corners[c_]]): n_corners[c_] for c_ in range(len(corners))}
    N[viv_t] = blend_normals(P, nor, G_t, num_t, sub, rep_t, len(gt), viv_t, k_t, dist_t[k_t], src_t, n_s_t, radius_normals, radius)

    # --- weights: the tube's on the weld, blended over the ball
    os2, p2, new = os_.copy(), weight.copy(), {}
    for_k = {int(num[sub[v]]): (os_[corners[c_]], weight[corners[c_]]) for v, c_ in zip(border.tolist(), to.tolist())}
    for v, k, dd in zip(alive.tolist(), k_v.tolist(), dist[k_v].tolist()):
        if dd < radius_weight:
            o_s, w_s = for_k[int(src[k])]
            ty, o, w = blend_weight(os_[v], weight[v], o_s, w_s, float(1 - smooth(dd / radius_weight)))
            new[v] = (ty, o, w)
            os2[v], p2[v] = o, w
    return dict(pos=P, nor=N, bones=os2, weights=p2, guard=guard, weight_modified=new,
                assessment={"faces_ball_removed": int(r_b.sum() + flat.sum()), "faces_tube_removed": int(r_t.sum()),
                       "corners_tube": len(corners), "vertices_ball_welded": len(border), "gap_max_mm": round(float(gap / MM), 2)})


def join_neck(pos, nor, faces, mat_face, i_skin, i_face, os_, weight, radius=RADIUS_MM * MM,
                radius_normals=RADIUS_NORMALS_MM * MM, radius_weight=RADIUS_WEIGHT_MM * MM, reach=REACH_MM * MM,
                radius_relaxation=RADIUS_RELAXATION_MM * MM):
    """Returns a dict: guard, pos, nor (new arrays), weights {v: (type, bones, weights)}, weld {v: (a, b, t)}, assessment."""
    mask_skin = mat_face == i_skin
    F_vis = faces[np.isin(mat_face, i_face)]
    res = {"guard": np.ones(len(faces), bool), "pos": pos.copy(), "nor": nor.copy(), "weights": {}, "weld": {},
           "assessment": {"skin_faces": int(mask_skin.sum()), "removed": 0, "welded": 0}}
    if not len(F_vis) or not mask_skin.any():
        return res
    tri = pos[F_vis]
    lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    C = 0.5 * (lo + hi)
    union = join_tube(pos, nor, faces, mat_face, i_skin, os_, weight, lo, hi)
    kept0, weight_tube = np.ones(len(faces), bool), {}
    if union is not None:
        pos, nor, os_, weight, kept0, weight_tube = union["pos"], union["nor"], union["bones"], union["weights"], union["guard"], union["weight_modified"]
        mask_skin = mask_skin & kept0
    F_skin = faces[mask_skin]
    sub = weld(pos)

    # --- the skin under the face goes away
    candidates = np.unique(F_skin[pos[F_skin].mean(1)[:, 1] > lo[1]])
    in_ = np.zeros(len(pos), bool)
    if len(candidates):
        in_[candidates[np.isfinite(rays_face(C, pos[candidates], tri))]] = True
    remove = mask_skin.copy()
    remove[mask_skin] = in_[F_skin].any(1)

    # --- face edge and skin edges within its reach
    border_vis = border_outer(edges_of_border(F_vis, sub), sub, pos)
    F_rest = faces[mask_skin & ~remove]
    E = edges_of_border(F_rest, sub)
    if not len(E):
        return res
    groups = groups_edges(E, sub)
    corners = np.unique(border_vis)
    _, ic = np.unique(sub[corners], return_index=True)
    corners = corners[ic]
    chosen = choose_loops(pos, E, groups, corners, reach)
    if not chosen:
        res["assessment"]["outside_reach"] = True
        return res
    v_border = np.unique(E[np.isin(groups, chosen)])
    _, iu = np.unique(sub[v_border], return_index=True)
    v_border = v_border[iu]
    a, b = border_vis[:, 0], border_vis[:, 1]
    proj, ka, d_border = point_segment(pos[v_border], pos[a], pos[b])
    taken = d_border < reach
    v_border, proj, ka = v_border[taken], proj[taken], ka[taken]
    if not len(v_border):
        res["assessment"]["outside_reach"] = True
        return res

    # each face corner receives the nearest skin vertex: no gap in the corners
    d_corner = np.sqrt(((pos[corners][:, None] - pos[v_border][None]) ** 2).sum(2))
    order = np.argsort(d_corner.min(1))
    taken_corner = {}
    without_corner = 0
    for c in order.tolist():
        for j in np.argsort(d_corner[c]).tolist():
            if j not in taken_corner:
                if d_corner[c, j] < reach:
                    taken_corner[j] = c
                else:
                    without_corner += 1
                break

    # skin looser than the face (neck stump of a game body, hidden under the head) is not welded:
    # its few vertices would each travel far, and it is not visible
    if without_corner > PART_CORNERS_MAX * len(corners):
        res["assessment"].update(skin_too_loose=True, corners=len(corners), corners_without_vertex=without_corner)
        return res
    res["guard"] = ~remove & kept0
    res["pos"], res["nor"], res["weights"] = pos.copy(), nor.copy(), dict(weight_tube)
    removed_tube = int((~kept0).sum())
    res["assessment"]["removed"] = int(remove.sum()) + removed_tube
    if union is not None:
        res["assessment"]["tube"] = union["assessment"]

    # --- weld: position, normal, weights of the face edge
    nor_corner = {}
    for v in np.unique(F_vis).tolist():
        nor_corner.setdefault(int(sub[v]), []).append(nor[v])
    def n_border(v):
        n = np.mean(nor_corner[int(sub[v])], 0)
        return n / max(np.linalg.norm(n), 1e-12)
    n_weld = {}
    target = {}
    for j, v in enumerate(v_border.tolist()):
        if j in taken_corner:
            c = int(corners[taken_corner[j]])
            target[int(sub[v])] = pos[c].copy()
            n_weld[int(sub[v])] = n_border(c)
            res["weld"][v] = (c, c, 0.0)
        else:
            k = int(ka[j])
            ab = pos[b[k]] - pos[a[k]]
            t = float(np.clip((proj[j] - pos[a[k]]) @ ab / max(ab @ ab, 1e-12), 0, 1))
            target[int(sub[v])] = proj[j].copy()
            n = (1 - t) * n_border(int(a[k])) + t * n_border(int(b[k]))
            n_weld[int(sub[v])] = n / max(np.linalg.norm(n), 1e-12)
            res["weld"][v] = (int(a[k]), int(b[k]), t)

    # --- harmonic displacement of the skin around the weld (welded graph of the kept skin)
    alive = np.unique(F_rest)
    gs = np.unique(sub[alive])
    num = np.full(sub.max() + 1, -1)
    num[gs] = np.arange(len(gs))
    rep = np.zeros(len(gs), int)
    rep[num[sub[alive]]] = alive
    Eg = np.vstack([F_rest[:, [0, 1]], F_rest[:, [1, 2]], F_rest[:, [2, 0]]])
    Eg = np.unique(np.sort(num[sub[Eg]], axis=1), axis=0)
    Eg = Eg[Eg[:, 0] != Eg[:, 1]]
    lengths = np.linalg.norm(pos[rep[Eg[:, 0]]] - pos[rep[Eg[:, 1]]], axis=1)
    sources = sorted(int(num[g]) for g in target)
    dist, src = distances(len(gs), Eg, lengths, sources)
    fixed = dist >= radius
    val = np.zeros((len(gs), 3))
    for g, c in target.items():
        k = int(num[g])
        fixed[k] = True
        val[k] = c - pos[rep[k]]
    disp = harmonic(len(gs), Eg, fixed, val)
    res["assessment"].update(welded=len(target), corners=len(corners), corners_without_vertex=without_corner,
                        shift_max_mm=round(float(np.linalg.norm(val, axis=1).max() / MM), 2),
                        displacement_median_mm=round(float(np.median(np.linalg.norm(val[sources], axis=1)) / MM), 2),
                        skin_moved=int((dist < radius).sum()))
    P = res["pos"]
    moves = np.where(dist < radius)[0]
    mask_g = np.zeros(sub.max() + 1, bool)
    mask_g[gs[moves]] = True
    v_touched = alive[mask_g[sub[alive]]]
    k_touched = num[sub[v_touched]]
    P[v_touched] = pos[v_touched] + disp[k_touched]

    # --- relax the first rings: gluing a sawtooth edge wrinkles the skin touching it
    src_seam = np.zeros(len(gs), bool)
    src_seam[sources] = True
    x = relax(P[rep].copy(), Eg, (dist < radius_relaxation) & ~src_seam)
    P[v_touched] = x[k_touched]

    # --- normals: imposed on the weld, then new geometry, back to the original normals further away
    d = dist[k_touched]
    res["nor"][v_touched] = blend_normals(P, nor, F_rest, num, sub, rep, len(gs), v_touched, k_touched, d, src, n_weld,
                                            radius_normals, radius)[:]

    # --- weights: the face's on the weld, blended around it
    for_v = {}
    for v, (va, vb, t) in res["weld"].items():
        ty, o, w = blend_weight(os_[va], weight[va], os_[vb], weight[vb], t)
        for_v[int(num[sub[v]])] = (o, w)
    for v, k, dd in zip(v_touched.tolist(), k_touched.tolist(), d.tolist()):
        if dd >= radius_weight:
            continue
        o_s, w_s = for_v[int(src[k])]
        f = float(1 - smooth(dd / radius_weight))
        if {x for x, y in zip(os_[v], weight[v]) if x >= 0 and y > 0} == {x for x, y in zip(o_s, w_s) if x >= 0 and y > 0} and f < 1:
            if abs(np.sort(weight[v]) - np.sort(w_s)).max() < 1e-3:
                continue
        res["weights"][v] = blend_weight(os_[v], weight[v], o_s, w_s, f)

    # --- duplicated welded vertices (skin UV seams) follow their weld
    per_g = {int(sub[v]): s for v, s in res["weld"].items()}
    for v in v_touched.tolist():
        if int(sub[v]) in per_g:
            res["weld"][v] = per_g[int(sub[v])]

    # --- faces collapsed by the gluing
    guard = res["guard"]
    idx = np.where(mask_skin & guard)[0]
    tr = P[faces[idx]]
    area = np.linalg.norm(np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0]), axis=1)
    flat = idx[area < 1e-9]
    guard[flat] = False
    # a face resting on the weld that flips over is a gluing fold: its neighbours cover its place
    nf = np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0])
    nm = res["nor"][faces[idx]].mean(1)
    cos = (nf * nm).sum(1) / np.maximum(area * np.linalg.norm(nm, axis=1), 1e-30)
    direction = np.sign(np.median(cos))
    on_weld = (src_seam[num[sub[faces[idx]]]].sum(1) >= 2)
    folds = idx[on_weld & (cos * direction < 0.1)]
    guard[folds] = False
    res["assessment"].update(removed=int(remove.sum()) + len(flat) + len(folds) + removed_tube, faces_flat=len(flat), folds=len(folds))
    res["graph"] = dict(num=num, sub=sub, rep=rep, Eg=Eg, dist=dist, n=len(gs), F_rest=F_rest)
    return res


def read_image(path):
    """sRGB RGBA, rows top to bottom (Blender's Python)."""
    import bpy
    im = bpy.data.images.load(path, check_existing=False)
    im.colorspace_settings.name = "Non-Color"
    w, h = im.size
    px = np.empty(4 * w * h, np.float32)
    im.pixels.foreach_get(px)
    bpy.data.images.remove(im)
    return px.reshape(h, w, 4)[::-1].copy()


def write_image(path, px):
    import bpy
    h, w = px.shape[:2]
    im = bpy.data.images.new("tmp", w, h, alpha=True)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(np.clip(px[::-1], 0, 1).astype(np.float32).ravel())
    tmp = path + ".tmp.png"
    im.filepath_raw = tmp
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)
    os.replace(tmp, path)


def texel(img, uv):
    h, w = img.shape[:2]
    u = np.mod(uv, 1.0)
    return img[(u[:, 1] * h).astype(int).clip(0, h - 1), (u[:, 0] * w).astype(int).clip(0, w - 1), :3]


def paint_tint(folder, textures, materials, uv, mat_vertex, i_skin, r, radius=RADIUS_TINT_MM * MM):
    """The skin texture takes the colour of the face edge along the weld; summary, or None outside Blender."""
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("join: tint must run in Blender (texture reading); nothing changed")
        return None
    def path_tex(i_mat):
        t = materials[i_mat]["texture"]
        return os.path.join(folder, textures[t].replace("\\", "/")) if 0 <= t < len(textures) else None
    mats = {i_skin} | {int(mat_vertex[x]) for v in r["weld"].values() for x in v[:2]}
    absent = [i for i in mats if path_tex(i) is None or not os.path.isfile(path_tex(i))]
    if absent:
        print("join: texture missing for materials", absent, "; tint not matched")
        return None
    g = r["graph"]
    num, sub, Eg, dist, n = g["num"], g["sub"], g["Eg"], g["dist"], g["n"]
    images = {}

    def image(i_mat):
        t = materials[i_mat]["texture"]
        if t not in images:
            images[t] = read_image(os.path.join(folder, textures[t].replace("\\", "/")))
        return images[t]
    img_skin = image(i_skin)
    fixed = dist >= radius
    val = np.zeros((n, 3))
    seen = {}
    for v, (a, b, t) in r["weld"].items():
        k = int(num[sub[v]])
        ca = texel(image(int(mat_vertex[a])), uv[[a]])[0]
        cb = texel(image(int(mat_vertex[b])), uv[[b]])[0]
        seen.setdefault(k, []).append((1 - t) * ca + t * cb - texel(img_skin, uv[[v]])[0])
    for k, d in seen.items():
        fixed[k] = True
        val[k] = np.mean(d, 0)
    delta = harmonic(n, Eg, fixed, val)

    # painting: skin triangles of the strip, rasterised into the texture
    h, w = img_skin.shape[:2]
    F = g["F_rest"]
    Fg = num[sub[F]]
    band = (dist[Fg] < radius).any(1)
    F, Fg = F[band], Fg[band]
    correction = np.zeros((h, w, 3), np.float32)
    covered = np.zeros((h, w), bool)
    for f, fg in zip(F.tolist(), Fg.tolist()):
        p = np.mod(uv[f], 1.0) * [w, h]
        x0, y0 = np.maximum(np.floor(p.min(0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(p.max(0)).astype(int) + 2, [w, h])
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
        margin = -1.0 / max(abs(det) ** 0.5, 1.0)     # one texel of overlap: no fringe between two triangles
        m = (l0 >= margin) & (l1 >= margin) & (l2 >= margin)
        if not m.any():
            continue
        c = np.clip(l0[m], 0, 1)[:, None] * delta[fg[0]] + np.clip(l1[m], 0, 1)[:, None] * delta[fg[1]] + np.clip(l2[m], 0, 1)[:, None] * delta[fg[2]]
        correction[y0:y1, x0:x1][m] = c
        covered[y0:y1, x0:x1] |= m
    res = img_skin.copy()
    res[..., :3] = np.clip(res[..., :3] + correction, 0, 1)
    write_image(os.path.join(folder, textures[materials[i_skin]["texture"]].replace("\\", "/")), res)
    return {"texels": int(covered.sum()), "delta_median_srgb": [round(float(x) * 255, 1) for x in np.median(val[list(seen)], 0)]}


def tuck_skull(s, faces, mat_face, i_skin, i_face, weld=(), stage=STEP_SILHOUETTE_MM * MM, neck=NECK_UNDER_CHIN_MM * MM):
    """The body skin does not stick out, seen from the front, beyond the face silhouette; returns the number of vertices pulled in.

    The scalp of a skin-completed body is wider than the face: from the front, it surrounds it from the
    eyes to the chin with a band of skin that extends the jaw in a long V down to the neck (long pointed chin, no
    jaw corner). On the game models, that space belongs to the back hair, and the neck joins the jaw only from
    below. At each face height, each side is bounded by the face edge on that side, without going below the
    neck half-width measured `neck` under the chin. The weld does not move.
    """
    pos = np.asarray(s["pos"], float)
    v = np.unique(faces[np.isin(mat_face, list(i_face))])
    vp = np.setdiff1d(np.unique(faces[mat_face == i_skin]), np.asarray(list(weld), int))
    if not len(v) or not len(vp):
        return 0
    xc = pos[v, 0].mean()
    yc, yh = pos[v, 1].min(), pos[v, 1].max()
    bins = np.arange(yc, yh + stage, stage)
    dx = pos[v, 0] - xc
    slice_ = [np.abs(pos[v, 1] - b) < stage for b in bins]
    left = np.array([(-dx[m]).max(initial=0.0) for m in slice_])
    right = np.array([dx[m].max(initial=0.0) for m in slice_])
    q = pos[vp]
    collar = q[(np.abs(q[:, 1] - (yc - neck)) < stage) & (q[:, 2] > pos[v, 2].min())]
    hw_neck = np.abs(collar[:, 0] - xc).max() if len(collar) else 0.0
    # Snapping each vertex onto the limit flattened the scalp into a wall with `pas` steps along the face
    # edge (jagged ridge on the cheek, in 3/4 view): instead the scalp tightens by a factor per height and
    # per side, smoothed over height, which fades out near the weld
    in_ = vp[(q[:, 1] >= yc - stage) & (q[:, 1] <= yh + stage)]
    if not len(in_):
        return 0
    kernel = np.hanning(7)[1:-1]
    kernel /= kernel.sum()
    sub = np.asarray(list(weld), int)
    d_sub = np.sqrt(((pos[in_][:, None] - pos[sub][None]) ** 2).sum(2)).min(1) if len(sub) else np.full(len(in_), np.inf)
    n = 0
    for side, limit_v in ((1, right), (-1, left)):
        lim = np.maximum(limit_v, hw_neck)
        d = side * (pos[in_, 0] - xc)
        b = np.clip(np.round((pos[in_, 1] - yc) / stage).astype(int), 0, len(bins) - 1)
        large = np.zeros(len(bins))
        np.maximum.at(large, b[d > 0], d[d > 0])
        f = np.where(large > lim, lim / np.maximum(large, 1e-12), 1.0)
        f = np.convolve(np.pad(f, 2, mode="edge"), kernel, mode="valid")
        fi = np.interp(pos[in_, 1], bins, f)
        w = np.clip(d_sub / RADIUS_SKULL_MM / MM, 0, 1)
        w = w * w * (3 - 2 * w)
        k = (d > 0) & (fi < 1)
        x = xc + (pos[in_, 0] - xc) * (1 - w * (1 - fi))
        for i, xi in zip(in_[k].tolist(), x[k].tolist()):
            s["pos"][i] = (float(xi), float(pos[i, 1]), float(pos[i, 2]))
        n += int(k.sum())
    return n


def sink_under_face(s, faces, mat_face, i_skin, i_face, weld=(), margin=UNDER_FACE_MM * MM,
                         reach=REACH_UNDER_MM * MM, border=BORDER_UNDER_MM * MM):
    """The body skin hidden by the face goes down to `marge` beneath it; returns the number of vertices pushed in.

    The join leaves skin vertices under the face flush with its surface, sometimes above it: they poke
    through it (spots of body tint), and the body's outline shell, inflated by 1 to 2 mm, draws dark dashes
    there along the cheek edge: the skin rising towards the weld passes over the face before joining it.
    Each vertex whose nearest point lies in the face goes down, along that face's normal (or, just past
    the edge, under the extension of the nearest face), to a depth growing from 0 on the face outline to
    `marge` at `border` from it.
    """
    pos = np.asarray(s["pos"], float)
    nor = np.asarray(s["nor"], float)
    Fv = faces[np.isin(mat_face, list(i_face))]
    vp = np.setdiff1d(np.unique(faces[mat_face == i_skin]), np.asarray(list(weld), int))
    if not len(Fv) or not len(vp):
        return 0
    sub = globals()["weld"](pos)  # the `weld` parameter shadows the function
    girth = np.unique(border_outer(edges_of_border(Fv, sub), sub, pos))
    T = pos[Fv]
    lo, hi = T.reshape(-1, 3).min(0) - reach, T.reshape(-1, 3).max(0) + reach
    vp = vp[((pos[vp] > lo) & (pos[vp] < hi)).all(1)]
    if not len(vp) or not len(girth):
        return 0
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n *= np.sign((n * nor[Fv].sum(1)).sum(1))[:, None] + ((n * nor[Fv].sum(1)).sum(1) == 0)[:, None]
    c = T.mean(1)
    k = 0
    for i in vp.tolist():
        q = pos[i]
        j = np.argsort(((c - q) ** 2).sum(1))[:6]
        # nearest point on each of the candidate faces
        a, b, cc = T[j, 0], T[j, 1], T[j, 2]
        sd = ((q - a) * n[j]).sum(1)
        p = q - sd[:, None] * n[j]
        e0, e1, r = b - a, cc - a, p - a
        d00, d01, d11 = (e0 * e0).sum(1), (e0 * e1).sum(1), (e1 * e1).sum(1)
        d20, d21 = (r * e0).sum(1), (r * e1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        inside = (v >= 0) & (w >= 0) & (v + w <= 1)
        # outside the face, just past its edge, the skin must pass under the extension of its surface
        m = np.where(inside, np.abs(sd), np.inf).argmin() if inside.any() else 0
        d_perimeter = np.sqrt(((pos[girth] - q) ** 2).sum(1)).min()
        if not inside.any() and d_perimeter > reach:
            continue
        # target depth: zero on the outline (the skin is welded there), `marge` at `border` from it
        depth = margin * min(d_perimeter / border, 1.0)
        if abs(sd[m]) > reach or sd[m] <= -depth:
            continue
        s["pos"][i] = tuple(float(x) for x in q + (-depth - sd[m]) * n[j[m]])
        k += 1
    return k


def thin_chin(s, faces, mat_face, i_face, weld=(), bottom=TRAIT_CHIN_BOTTOM, height=TRAIT_CHIN_HEIGHT):
    """Outline weight of the lower face: 1 at mouth height, `bottom` at the tip of the chin.

    At full thickness, the face's outline shell runs along the whole jaw and lies on the bare neck:
    a long black V that lengthens the chin. The mouth is at `TRAIT_CHIN_HEIGHT` of the face height above the chin (0.15 Maelys, 0.16 Grace).
    Skin vertices welded to the face edge take the same weight. Returns the number of thinned vertices.
    """
    pos = np.asarray(s["pos"], float)
    v = np.unique(faces[np.isin(mat_face, list(i_face))])
    if not len(v):
        return 0
    yc, yh = pos[v, 1].min(), pos[v, 1].max()
    yb = yc + height * (yh - yc)
    c = np.asarray(s["contour"], float)
    for w in np.concatenate([v, np.asarray(list(weld), int)]).tolist():
        t = min(max((pos[w, 1] - yc) / (yb - yc), 0.0), 1.0)
        c[w] = min(c[w], bottom + (1 - bottom) * t * t * (3 - 2 * t))
    s["contour"] = [float(x) for x in c]
    return int((c[v] < 0.999).sum())


def apply(s, faces, mat_face, materials, morphs, i_skin, i_face, folder=None, textures=None):
    """Apply the join to the vertex table `s`; returns (kept faces, summary).

    With `dossier` and `textures` (paths relative to `dossier`), the skin texture is matched there as well.
    """
    os_ = np.asarray(s["bones"], np.int32)
    weight = np.asarray(s["weights"], float)
    r = join_neck(np.array(s["pos"], float), np.array(s["nor"], float), faces, mat_face, i_skin, i_face, os_, weight)
    if not r["weld"]:
        r["assessment"]["trait_chin"] = thin_chin(s, faces, mat_face, i_face)
        return faces, r["assessment"]
    change = np.where(np.any(r["pos"] != np.array(s["pos"], float), axis=1) | np.any(r["nor"] != np.array(s["nor"], float), axis=1))[0]
    for v in change.tolist():
        s["pos"][v] = tuple(float(x) for x in r["pos"][v])
        s["nor"][v] = tuple(float(x) for x in r["nor"][v])
    for v, (ty, o, w) in r["weights"].items():
        s["type"][v], s["bones"][v], s["weights"][v], s["sdef"][v] = ty, o, w, None
    # morphs that move the face edge carry the weld along
    for m in morphs:
        if m["type"] != 1:
            continue
        offset = dict((int(i), np.asarray(d, float)) for i, d in m["offsets"])
        additions = []
        for v, (a, b, t) in r["weld"].items():
            if a in offset or b in offset:
                d = (1 - t) * offset.get(a, np.zeros(3)) + t * offset.get(b, np.zeros(3))
                if np.abs(d).max() > 1e-7:
                    additions.append((v, tuple(float(x) for x in d)))
        if additions:
            m["offsets"] = list(m["offsets"]) + additions
            r["assessment"]["morphs_tracked"] = r["assessment"].get("morphs_tracked", 0) + 1
    g = r["guard"]
    r["assessment"]["skull_tucked"] = tuck_skull(s, faces[g], mat_face[g], i_skin, i_face, r["weld"])
    r["assessment"]["under_face"] = sink_under_face(s, faces[g], mat_face[g], i_skin, i_face, r["weld"])
    r["assessment"]["trait_chin"] = thin_chin(s, faces, mat_face, i_face, r["weld"])
    if folder is not None:
        mat_vertex = np.full(len(s["pos"]), -1)
        mat_vertex[faces.ravel()] = np.repeat(mat_face, 3)
        r["assessment"]["tint"] = paint_tint(folder, textures, materials, np.array(s["uv"], float), mat_vertex, i_skin, r)
    guard = r["guard"]
    fm = mat_face[guard]
    for i, m in enumerate(materials):
        m["n_indices"] = int(3 * (fm == i).sum())
    return faces[guard], r["assessment"]


def args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pmx", required=True)
    p.add_argument("--output", dest="output", required=True)
    p.add_argument("--skin", dest="skin", default="肌")
    p.add_argument("--face", dest="face", default="颜,颜_A")
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return p.parse_args(argv)


def copy_textures(R, folder_pmx, output):
    """Copy the model's textures into `sortie`; those stored outside the PMX folder go into tex/."""
    os.makedirs(output, exist_ok=True)
    for i, t in enumerate(R["textures"]):
        src = os.path.normpath(os.path.join(folder_pmx, t.replace("\\", "/")))
        if not os.path.isfile(src):
            continue
        rel = os.path.relpath(src, folder_pmx)
        if rel.startswith(".."):
            rel = os.path.join("tex", "%d_%s" % (i, os.path.basename(src)))
        dst = os.path.join(output, rel)
        os.makedirs(os.path.dirname(dst) or output, exist_ok=True)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        R["textures"][i] = rel.replace(os.sep, "\\")


def main(argv=None):
    a = args(argv)
    R = PMX.read(a.pmx)
    t = PMX.to_arrays(R)
    names = [m["name"] for m in R["materials"]]
    i_vis = [names.index(n) for n in a.face.split(",") if n in names]
    copy_textures(R, os.path.dirname(os.path.abspath(a.pmx)), a.output)
    F, assessment = apply(R["vertices"], t["faces"], t["mat_face"], R["materials"], R["morphs"], names.index(a.skin), i_vis,
                         folder=a.output, textures=R["textures"])
    R["faces"] = F.tolist()
    path = os.path.join(a.output, "result.pmx")
    pmx_write.write(R, path + ".tmp")
    os.replace(path + ".tmp", path)
    with open(os.path.join(a.output, "seam.json"), "w", encoding="utf-8") as f:
        json.dump(assessment, f, ensure_ascii=False, indent=1)
    print("neck join:", assessment, "->", path)


if __name__ == "__main__":
    main()

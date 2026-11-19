"""Stress bench of method v2 (§2.3, §2.5): poses P13-P15 and amplitude variants
added to P-08, left-right floor P-12, analytic lighting P-13,
indicators D1 (share out of bounds), D3 (area), D4 (signed), D6 to D10.

Numpy only, except D4 and D10 (mathutils.bvhtree, hence Blender's Python).
`primitives.py` and `kinematics.py` are read, never modified: the poses of A
(P0-P12) stay those of `primitives.pose`; this module only adds the new ones.

Conventions (those of `primitives`): Blender frame, facing -Y, +Z up,
side 左 at +X; local bone rotations in the world frame at rest.
"""
import numpy as np

import kinematics as kin
import primitives as P

FRONT, TOP, BOTTOM = P.FRONT, P.TOP, P.BOTTOM
LATERAL = {"L": np.array([1.0, 0.0, 0.0]), "R": np.array([-1.0, 0.0, 0.0])}

# ======================================================================= P-08 extended

# Amplitude variants of D §4 (method v2 §2.3, P-08): pose of A -> changed angles
VARIANTS = {
    "P1@90": "arms raised by 90° (P1 at 150°)",
    "P4@90": "elbow bent by 90° (P4 at 130°)",
    "P4@140": "elbow bent by 140°",
    "P5@60": "left thigh raised by 60° (P5 at 90°)",
    "P6@90": "left knee bent by 90° (P6 at 120°)",
    "P6@140": "left knee bent by 140°",
    "P7@D": "crouch of D (P9 of D): hips 110°, knees 140°, ankles 30°",
}
NEW = {
    "P13": "shoulder shrug of 20°, then arms crossed in front",
    "P14": "T-pose",
    "P15": "forearm in pronation of -90° (P3 in the opposite direction)",
}
# Full bench of E-18: P-08 of A (without the rest pose), new poses, variants
BENCH = ["P1", "P1@90", "P2", "P3", "P15", "P4", "P4@90", "P4@140", "P5", "P5@60", "P6",
        "P6@90", "P6@140", "P7", "P7@D", "P8", "P9", "P10", "P11", "P12", "P13", "P14"]


def _name(c, os_):
    return ("左" if c == "L" else "右") + os_


def _rot_to(sq, name, to, target, degrees=None):
    """Index and rotation that brings the direction of `name` (towards the head of `to`) towards
    `target`, by `degrees` (None: all the way to the target). None if a bone is missing."""
    if not sq.a(name) or not sq.a(to):
        return None
    b = sq.index(name)
    d, _ = sq.direction(b, to)
    c = np.asarray(target, dtype=float)
    c = c / np.linalg.norm(c)
    if degrees is None:
        degrees = float(np.degrees(np.arccos(np.clip(d @ c, -1.0, 1.0))))
    axis = np.cross(d, c)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.cross(d, FRONT)
    return b, kin.q_axe_angle(axis, degrees)


def _add(r, x):
    if x is None:
        return
    b, q = x
    r[b] = kin.q_mul(r.get(b, kin.IDENTITY), q)


def _rot_parent(sq, r, b):
    """World matrix of the parent of b under the rotations r (to give a target in
    the rest frame of b when its ancestors are already rotated)."""
    R, _ = sq.kinematics(r)
    p = sq.parents[b]
    return R[p] if p >= 0 else np.eye(3)


def pose_bench(name, sq, sides=("L", "R")):
    """Local rotations (bone index -> quaternion) of a bench pose.

    P0-P12: `primitives.pose` (A), unchanged. P13-P15 and variants `Pk@x`: here.
    Angles counted from the delivered rest pose, as in A."""
    if name in P.POSES:
        return P.pose(name, sq, sides)
    r = {}
    if name == "P13":
        for c in sides:
            # shrug: 肩 raised by 20° upwards
            _add(r, _rot_to(sq, _name(c, "肩"), _name(c, "腕"), TOP, 20))
            # arms in front, a little low, inwards; elbow bent by 110° inwards
            med = -LATERAL[c]
            _add(r, _rot_to(sq, _name(c, "腕"), _name(c, "ひじ"), FRONT + 0.5 * BOTTOM + 0.4 * med))
            if sq.a(_name(c, "ひじ")):
                b = sq.index(_name(c, "ひじ"))
                Rp = _rot_parent(sq, r, b)
                _add(r, _rot_to(sq, _name(c, "ひじ"), _name(c, "手首"), Rp.T @ med, 110))
    elif name == "P14":
        for c in sides:
            _add(r, _rot_to(sq, _name(c, "腕"), _name(c, "ひじ"), LATERAL[c]))
    elif name == "P15":
        os_ = "手捩" if sq.a("左手捩") else "手首"
        for c in sides:
            n, v = _name(c, "ひじ"), _name(c, "手首")
            if not sq.a(n) or not sq.a(v) or not sq.a(_name(c, os_)):
                continue
            d, _ = sq.direction(sq.index(n), v)
            _add(r, (sq.index(_name(c, os_)), kin.q_axe_angle(d, -90 if c == "L" else 90)))
    elif name == "P1@90":
        for c in sides:
            _add(r, _rot_to(sq, _name(c, "腕"), _name(c, "ひじ"), TOP, 90))
    elif name in ("P4@90", "P4@140"):
        a = float(name.split("@")[1])
        for c in sides:
            _add(r, _rot_to(sq, _name(c, "ひじ"), _name(c, "手首"), FRONT, a))
    elif name == "P5@60":
        _add(r, _rot_to(sq, "左足", "左ひざ", FRONT, 60))
    elif name in ("P6@90", "P6@140"):
        a = float(name.split("@")[1])
        _add(r, _rot_to(sq, "左ひざ", "左足首", -FRONT, a))
    elif name == "P7@D":
        for c in sides:
            _add(r, _rot_to(sq, _name(c, "足"), _name(c, "ひざ"), FRONT, 110))
            _add(r, _rot_to(sq, _name(c, "ひざ"), _name(c, "足首"), -FRONT, 140))
            _add(r, _rot_to(sq, _name(c, "足首"), _name(c, "つま先"), TOP, 30))
    elif name == "ASSIS":      # sitting on a chair (outside the bench): hips and knees at 90°
        for c in sides:
            _add(r, _rot_to(sq, _name(c, "足"), _name(c, "ひざ"), FRONT, 90))
            _add(r, _rot_to(sq, _name(c, "ひざ"), _name(c, "足首"), -FRONT, 90))
    else:
        raise ValueError("unknown bench pose: %s" % name)
    return r


def description_pose(name):
    return VARIANTS.get(name) or NEW.get(name) or "P-08 of A (%s)" % name


# ======================================================================= P-12

index_mirror_os = P.mirror_os  # by name; `tol` adds the position


def mirror_rotations(rotations, sq, im=None):
    """Mirror pose (P-12): each rotation moves to the symmetric bone, reflected through x = 0."""
    im = index_mirror_os(sq) if im is None else im
    out = {}
    for b, q in rotations.items():
        mb = int(im[b])
        out[mb] = kin.q_mul(out.get(mb, kin.IDENTITY), kin.q_mirror_x(q))
    return out


def sides_vertices(pos, H, tol=1e-3):
    """Side of each vertex: 'L' (x > tol·H), 'R' (x < -tol·H), 'M' otherwise."""
    x = pos[:, 0]
    return np.where(x > tol * H, "L", np.where(x < -tol * H, "R", "M"))


def floor(m_left_p, m_right_mirror):
    """P-12: P(z, p) = |m(z_G, p) - m(z_D, mirror(p))|. None if either is missing."""
    if m_left_p is None or m_right_mirror is None:
        return None
    return float(abs(m_left_p - m_right_mirror))


# ======================================================================= P-13

ELEVATIONS = (0.0, 30.0, -20.0)


def lights(stage=5.0, elevations=ELEVATIONS):
    """P-13: 72 azimuths × 3 elevations = 216 directions (towards the light), unit length.
    Azimuth 0 = light from the front (-Y), counted towards +X (side 左)."""
    az = np.radians(np.arange(0.0, 360.0, stage))
    out = []
    for e in np.radians(elevations):
        out.append(np.stack([np.sin(az) * np.cos(e), -np.cos(az) * np.cos(e),
                             np.full_like(az, np.sin(e))], axis=1))
    return np.concatenate(out)


def illumination(N, L):
    """N·L, (n_lights, n_vertices)."""
    N = np.asarray(N, dtype=float)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return np.asarray(L) @ N.T


def shade(N, L, threshold=0.5):
    """Vertex in shadow if N·L < 0.5 (analytic toon, no render)."""
    return illumination(N, L) < threshold


def normals_vertices(pos, faces):
    """Normals recomputed from the geometry (angle-weighted average)."""
    pos = np.asarray(pos, dtype=float)
    f = np.asarray(faces)
    nf, _ = P.normals_faces(pos, f)
    N = np.zeros_like(pos)
    for k in range(3):
        a, b, c = f[:, k], f[:, (k + 1) % 3], f[:, (k + 2) % 3]
        u = pos[b] - pos[a]
        v = pos[c] - pos[a]
        cu = np.einsum("ij,ij->i", u, v) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-20)
        ang = np.arccos(np.clip(cu, -1, 1))
        np.add.at(N, a, nf * ang[:, None])
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


# ======================================================================= D1

BOUNDS_D1 = (0.7, 1.4)


def d1_reports(pos0, pos1, edges):
    """r = ‖e_p‖ / ‖e_rest‖ per edge (nan for a zero-length edge)."""
    l0 = np.linalg.norm(pos0[edges[:, 0]] - pos0[edges[:, 1]], axis=1)
    l1 = np.linalg.norm(pos1[edges[:, 0]] - pos1[edges[:, 1]], axis=1)
    r = np.full(len(edges), np.nan)
    ok = l0 > 1e-9
    r[ok] = l1[ok] / l0[ok]
    return r


def d1_zone(r, mask=None, bounds=BOUNDS_D1):
    """D1 on a zone: share outside [0.7, 1.4] (D), percentiles 1 and 99 (A), min, max,
    standard deviation (B03). Breaks: share > 2 %."""
    x = r if mask is None else r[mask]
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    out_of = float(((x < bounds[0]) | (x > bounds[1])).mean())
    return {"part_outside": out_of, "c01": float(np.percentile(x, 1)), "c99": float(np.percentile(x, 99)),
            "min": float(x.min()), "max": float(x.max()), "gap_type": float(x.std()),
            "n": int(len(x)), "broken": out_of > 0.02}


def d1_per_vertex(r, edges, n):
    """Strongest stretch or squash of the edges of each vertex: max(r, 1/r)."""
    s = np.where(np.isfinite(r), np.maximum(r, 1.0 / np.maximum(r, 1e-12)), 1.0)
    out = np.ones(n)
    np.maximum.at(out, edges[:, 0], s)
    np.maximum.at(out, edges[:, 1], s)
    return out


# ======================================================================= D2

def d2_mask(pos0, pos1, faces, R_dominant):
    """Flipped faces (v2: posed normal · rotation of the dominant bone × rest normal < 0)."""
    n0, a0 = P.normals_faces(pos0, faces)
    n1, a1 = P.normals_faces(pos1, faces)
    att = np.einsum("mij,mj->mi", R_dominant, n0)
    return (np.einsum("mi,mi->m", att, n1) < 0) & (a0 > 1e-12) & (a1 > 1e-12)


# ======================================================================= D3

THRESHOLDS_D3 = (0.85, 1.15)


def area_loop(points, normal):
    """Area of the polygon (points in loop order) projected onto its plane."""
    p = np.asarray(points, dtype=float)
    c = p.mean(axis=0)
    q = p - c
    s = np.cross(q, np.roll(q, -1, axis=0)).sum(axis=0)
    return float(abs(s @ np.asarray(normal, dtype=float)) / 2.0)


def d3_area(pos0, pos1, faces, sq, os_, to, R, t, fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
            masks=None, cache0=None, loops_band=3, jaccard_min=0.5):
    """D3 of D: posed / rest cross-section area along the bone (P-04 on both).

    masques: list of face masks tried in order; the first one that gives a closed
    loop at rest is kept, and the same one for the pose. cache0: dict to keep
    the rest cuts from one call to the next. Returns a list per fraction and the
    minimum / maximum of the ratio. Breaks: < 0.85 or > 1.15.

    The posed cut is searched in a band around the rest loop (faces within
    `loops_band` neighbourhood rings of the edges cut at rest): without it, the
    posed cut catches another loop (sleeve, neighbouring thigh) and the ratio no
    longer measures the same section."""
    b = sq.index(os_)
    d0, L = sq.direction(b, to)
    d1 = R[b] @ d0
    masks = masks or [None]
    out = []
    for f in fractions:
        the_key = (os_, to, f)
        if cache0 is not None and the_key in cache0:
            k, a0 = cache0[the_key]
        else:
            k, a0 = None, None
            for i, mk in enumerate(masks):
                a = P.loop(pos0, faces, sq.heads[b], d0, L, f, mk)
                if a and a.get("closed"):
                    k, a0 = i, a
                    break
            if a0 is not None:
                a0 = dict(a0)
                a0["band"] = band_faces(faces, np.array(a0["edges"]).ravel(), loops_band, masks[k])
            if cache0 is not None:
                cache0[the_key] = (k, a0)
        if a0 is None:
            out.append({"t": f, "report": None, "reason": "no closed loop at rest"})
            continue
        a1 = P.loop(pos1, faces, t[b], d1, L, f, a0["band"])
        if not a1 or not a1.get("closed"):
            out.append({"t": f, "report": None, "reason": "no closed loop in the pose"})
            continue
        e0 = {tuple(x) for x in a0["edges"]}
        e1 = {tuple(x) for x in a1["edges"]}
        jac = len(e0 & e1) / max(len(e0 | e1), 1)
        s0 = area_loop(a0["points"], a0["normal"])
        s1 = area_loop(a1["points"], a1["normal"])
        o = {"t": f, "area0": s0, "area1": s1, "report": s1 / s0 if s0 > 0 else None,
             "n0": a0["n_vertices"], "n1": a1["n_vertices"], "jaccard": jac}
        if jac < jaccard_min:
            o["report_raw"], o["report"] = o["report"], None
            o["reason"] = "another loop in the pose (shared cut edges: %.2f)" % jac
        out.append(o)
    # rest loop that is not a section of the segment (joint: cut through the trunk)
    n0s = [o["n0"] for o in out if o.get("n0")]
    if n0s:
        med = float(np.median(n0s))
        for o in out:
            if o.get("n0") and o["n0"] > 2 * med and o.get("report") is not None:
                o["report_raw"], o["report"] = o["report"], None
                o["reason"] = "rest loop outside the segment (%d edges, median %.0f)" % (o["n0"], med)
    rs = [o["report"] for o in out if o.get("report") is not None]
    res = {"profile": out, "min": min(rs) if rs else None, "max": max(rs) if rs else None}
    res["broken"] = bool(rs) and (res["min"] < THRESHOLDS_D3[0] or res["max"] > THRESHOLDS_D3[1])
    res["discarded"] = sum(1 for o in out if o.get("report") is None)
    return res


def band_faces(faces, vertices, n, mask=None):
    """Faces within n neighbourhood rings (by faces) of the given vertices, inside `masque`."""
    f = np.asarray(faces)
    v = np.zeros(int(f.max()) + 1, dtype=bool)
    v[np.asarray(vertices, dtype=np.int64)] = True
    sel = v[f].any(axis=1)
    for _ in range(n):
        v[f[sel].ravel()] = True
        sel = v[f].any(axis=1)
    return sel if mask is None else sel & np.asarray(mask, dtype=bool)


# ======================================================================= D4

def d4_sign(pos_a, pos_b, faces_b, limit):
    """D4 of D: signed distance of each point of A to the surface of B (BVH, sign from
    the normal of the closest face of B). nan beyond `limit`."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    out = np.full(len(pos_a), np.nan)
    if len(faces_b) == 0 or len(pos_a) == 0:
        return out
    bvh = BVHTree.FromPolygons(np.asarray(pos_b).tolist(), np.asarray(faces_b).tolist(),
                               all_triangles=True)
    for k, p in enumerate(np.asarray(pos_a)):
        loc, nor, _, d = bvh.find_nearest(Vector(p), limit)
        if loc is None:
            continue
        out[k] = float((Vector(p) - loc).dot(nor))
    return out


def components_mask(n, edges, mask):
    """Size of each connected component of the vertices of `masque` (edge graph)."""
    sel = mask[edges[:, 0]] & mask[edges[:, 1]]
    lab = np.arange(n)
    e = edges[sel]
    if len(e):
        while True:
            m = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
            old = lab.copy()
            np.minimum.at(lab, e[:, 0], m)
            np.minimum.at(lab, e[:, 1], m)
            lab = lab[lab]
            if np.array_equal(lab, old):
                break
    l = lab[mask]
    if len(l) == 0:
        return lab, np.array([], dtype=np.int64)
    _, sizes = np.unique(l, return_counts=True)
    return lab, sizes


def d4_zone(s_pose, s_rest, H, edges_a=None, n_a=None, depth=0.002):
    """D4 on the points of a zone of A: share beyond -0.002 H in the pose that were not
    there at rest, largest connected island (if edges_a, local indices).
    Breaks: share > 0.5 % or island > 5 vertices."""
    lim = -depth * H
    new = (s_pose < lim) & ~(s_rest < lim)
    new = np.where(np.isfinite(s_pose), new, False)
    n = len(s_pose)
    if n == 0:
        return None
    island = 0
    if edges_a is not None and new.any():
        _, t = components_mask(n_a or n, edges_a, new)
        island = int(t.max()) if len(t) else 0
    part = float(new.mean())
    return {"fraction": part, "n": int(new.sum()), "island_max": island,
            "depth_max_H": float(-np.nanmin(np.where(new, s_pose, np.nan)) / H) if new.any() else 0.0,
            "broken": part > 0.005 or island > 5}


# ======================================================================= D6, D7

def d6_monotony(N, N_ref, edges, mask, L=None, delta=0.05, threshold=0.5):
    """D6 (D5 of D): monotonicity of the shadow under P-13.

    - Vertex in violation for a light: in shadow (N·L < 0.5) while a
      deeper neighbour according to the reference (N_ref·L smaller by `delta`) is lit.
    - Shadow area A(θ) of the zone along each elevation: its direction of change
      reverses where the reference does not change that way ("no lighting reason").
    N_ref = normals of the geometry (normals_vertices): the "lighting reason".
    Breaks: more than 0.5 % of the zone's vertices in violation under the same light."""
    L = lights() if L is None else L
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    e = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
    s = illumination(N[idx], L)
    sr = illumination(N_ref[idx], L)
    loc = -np.ones(len(N), dtype=np.int64)
    loc[idx] = np.arange(len(idx))
    a, b = loc[e[:, 0]], loc[e[:, 1]]
    om = s < threshold
    viol = np.zeros(om.shape, dtype=bool)
    # a in shadow, b lit and deeper according to the reference (and the reverse)
    v1 = om[:, a] & ~om[:, b] & (sr[:, b] < sr[:, a] - delta)
    v2 = om[:, b] & ~om[:, a] & (sr[:, a] < sr[:, b] - delta)
    for k in range(len(L)):
        viol[k, a[v1[k]]] = True
        viol[k, b[v2[k]]] = True
    part_lum = viol.mean(axis=1)
    # shadow area along the azimuth
    n_el = len(L) // 72 if len(L) % 72 == 0 else 1
    A = om.reshape(n_el, -1, len(idx)).mean(axis=2)
    Ar = (sr < threshold).reshape(n_el, -1, len(idx)).mean(axis=2)
    dA = np.diff(np.concatenate([A, A[:, :1]], axis=1), axis=1)
    dAr = np.diff(np.concatenate([Ar, Ar[:, :1]], axis=1), axis=1)
    tol = 1.0 / len(idx)
    inversions = int(((np.sign(dA) * np.sign(dAr) < 0) & (np.abs(dA) > tol)).sum())
    k = int(np.argmax(part_lum))
    return {"part_max": float(part_lum[k]), "light": k, "part_average": float(part_lum.mean()),
            "vertices_touched": float(viol.any(axis=0).mean()), "inversions_area": inversions,
            "broken": bool(part_lum[k] > 0.005)}


def area_vertices(pos, faces):
    """Area assigned to each vertex (a third of the neighbouring faces)."""
    _, a = P.normals_faces(pos, faces)
    out = np.zeros(len(pos))
    for k in range(3):
        np.add.at(out, faces[:, k], a / 3.0)
    return out


def d7_stains(N, edges, mask, area_v, L=None, part_area=0.01, part_lum=0.05, threshold=0.5):
    """D7 (D6 of D): shadow spots. Per light, connected components of the vertices of the
    zone in shadow; island = component of less than 1 % of the zone's area.
    Breaks: at least one island in more than 5 % of the lights."""
    L = lights() if L is None else L
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    area_z = float(area_v[idx].sum())
    om = shade(N[idx], L, threshold)
    loc = -np.ones(len(N), dtype=np.int64)
    loc[idx] = np.arange(len(idx))
    e = edges[mask[edges[:, 0]] & mask[edges[:, 1]]]
    e = np.stack([loc[e[:, 0]], loc[e[:, 1]]], axis=1)
    av = area_v[idx]
    n_islands = np.zeros(len(L), dtype=np.int64)
    for k in range(len(L)):
        if not om[k].any():
            continue
        lab, _ = components_mask(len(idx), e, om[k])
        l = lab[om[k]]
        u, inv = np.unique(l, return_inverse=True)
        a = np.bincount(inv, weights=av[om[k]])
        n_islands[k] = int((a < part_area * area_z).sum())
    with_ = float((n_islands > 0).mean())
    return {"part_lights_with_island": with_, "islands_max": int(n_islands.max()),
            "broken": with_ > part_lum}


# ======================================================================= D8

def d8_outline(pos, nor, s_v, e_mat, b2s, draws=None, scale=0.08):
    """D8 (D7 of D): break of the inverted-hull outline. Offset position = vertex + e·s_v·N_v
    (e = outline size of the material, in MMD units × scale). At split vertices
    (same welded vertex b2s), largest distance between offset positions, relative to e.
    draws: mask of the vertices whose material draws the outline.
    Breaks: gap > 0.25 × e."""
    pos = np.asarray(pos, dtype=float)
    nor = np.asarray(nor, dtype=float)
    nor = nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)
    e = np.asarray(e_mat, dtype=float) * scale
    offset = pos + (e * np.asarray(s_v, dtype=float))[:, None] * nor
    sel = np.ones(len(pos), dtype=bool) if draws is None else np.asarray(draws, dtype=bool)
    order = np.argsort(b2s, kind="stable")
    order = order[sel[order]]
    g = np.asarray(b2s)[order]
    start = np.r_[0, np.where(np.diff(g) != 0)[0] + 1]
    end = np.r_[start[1:], len(g)]
    reports = []
    vertices = []
    for d, f in zip(start, end):
        if f - d < 2:
            continue
        v = order[d:f]
        D = np.linalg.norm(offset[v][:, None, :] - offset[v][None, :, :], axis=2)
        em = np.maximum(e[v][:, None], e[v][None, :])
        r = np.where(em > 0, D / np.maximum(em, 1e-12), 0.0)
        reports.append(float(r.max()))
        vertices.append(int(g[d]))
    reports = np.array(reports)
    if len(reports) == 0:
        return {"n_seams": 0, "max": 0.0, "part_broken": 0.0, "broken": False}
    return {"n_seams": int(len(reports)), "max": float(reports.max()),
            "c99": float(np.percentile(reports, 99)),
            "n_broken": int((reports > 0.25).sum()),
            "part_broken": float((reports > 0.25).mean()),
            "broken": bool((reports > 0.25).any()),
            "_per_welded_vertex": dict(zip(vertices, reports.tolist()))}


# ======================================================================= D9

def d9_opening(pos0, pos1, top, bottom):
    """D9: eye opening at the middle of the eyelid (pairs of vertices top[i] / bottom[i]),
    posed / rest. Breaks on blink: opening > 10 %."""
    o0 = np.linalg.norm(pos0[top] - pos0[bottom], axis=1).mean()
    o1 = np.linalg.norm(pos1[top] - pos1[bottom], axis=1).mean()
    r = float(o1 / o0) if o0 > 0 else None
    return {"opening": r, "broken_blink": r is not None and r > 0.10}


def d9_angle_corner(pos, corner, a, b):
    """D9: angle at the corner of the eye between the directions towards a and b (degrees).
    Breaks: angle < 170° where the corner must stay open (threshold of D)."""
    u = pos[a] - pos[corner]
    v = pos[b] - pos[corner]
    c = float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-20))
    ang = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return {"angle": ang, "broken": ang < 170.0}


# ======================================================================= D10

def directions_cone(axis, n=8):
    """n directions perpendicular to `axe`, evenly spaced (cone of E-11)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    t = np.cross(a, [0.0, 0.0, 1.0])
    if np.linalg.norm(t) < 1e-6:
        t = np.cross(a, [1.0, 0.0, 0.0])
    t /= np.linalg.norm(t)
    u = np.cross(a, t)
    return [np.cos(2 * np.pi * k / n) * t + np.sin(2 * np.pi * k / n) * u for k in range(n)]


def d10_margin(positions_a, touched, amplitudes):
    """D10: first amplitude (in the given order) where `touched(positions)` is true.

    positions_a(amplitude) -> posed positions; touched(positions) -> bool (D4 against the
    body). Returns {"margin": amplitude or None if nothing touches, "derniere_free": the last amplitude without contact}.
    Quantity without a threshold, compared between intact and ablation."""
    free = None
    for a in amplitudes:
        if touched(positions_a(a)):
            return {"margin": float(a), "derniere_free": free}
        free = float(a)
    return {"margin": None, "derniere_free": free}


def rotation_chain(sq, root, axe_rotation, degrees, rotations=None):
    """Rotations that turn the root of a chain by `degrees` around `axe_rotation`
    (world frame at rest), added to an existing pose."""
    r = dict(rotations or {})
    b = sq.index(root)
    r[b] = kin.q_mul(r.get(b, kin.IDENTITY), kin.q_axe_angle(axe_rotation, degrees))
    return r

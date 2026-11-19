"""Transport of a part from one body to another, body to body.

The common frame is the body itself, not the skeleton alone: around each segment (thigh, shin,
upper arm, forearm, pelvis, torso, neck), a point is located by its position along the segment,
its angle around it and its distance to the skin. A segment runs from one named joint to the next
(hip-knee, knee-ankle...): it does not depend on the "tail" the PMX gives the bone.

For each segment, rays cast outward from the axis measure:
- on the target, the radius of the skin;
- on the source, the first surface hit: the skin if it is there (exact body radius); another
  garment under the part stands in for the body (the part is fitted alone on the target, at the
  offset it had from what it covered); the part itself only gives a bound (the body lies
  beneath). Where the source body is unknown, its radius is the target's multiplied by the ratio
  of the two bodies measured where it is known, never exceeding the part; a tight-fitting part
  counts as the body. Fabric seen edge-on (its normal more than 60° from the ray) does not
  surround the body: it is the bottom that closes a hem from below (a cone under Wise's shirt,
  taken for a 2 cm radius body, pushed the hem out by 10 cm); the ray goes through it.

A vertex keeps its position along the segment (as a fraction; along the trunk, as a landmark height), its angle, and its offset from the skin
(multiplied by the size ratio): target radius = target skin + offset. More than 4 cm from the body, the
"skin" is the convex hull of the slice (loose fabric does not follow hollows). Under the source skin, it
keeps its relative depth. No bound: the target body may be twice as thick or as thin.
Segments form chains (trunk, arms, legs). A vertex held by a bone of a chain is located on the
segment of that chain where it actually lies (a coat panel held by the chest but hanging at the
hips is located on the pelvis), with a 3 cm blend at the joints.
A limb is measured only on its own skin: near the armpit or the groin, the ray of the arm or the
thigh crosses the torso welded to the limb and exits on the other side; a surface belonging to
another chain (by its weights) does not count, and the missing radius is filled in along the segment.
A ray starting from a point outside the body (under the crotch) enters the skin before leaving it:
the exit is what counts, on the source too, whether the thigh is bare or covered there (tights,
shorts, panties): the last surface under the part is taken. Vertices held by a hand, a foot or the
head stay on the per-bone fit (they are not cylinders).

Along the trunk, a fraction between joints is not an anatomical position: each rig puts its spine
bones elsewhere (VRoid's upper chest bone is 11 cm under the neck, a ZZZ model's under the breasts:
Jane's bust, at 36 % of that segment, landed 7 cm above a VRoid body's bust). The trunk segments
therefore carry heights through the landmarks measured on both surfaces (`landmarks.py`: crotch, hip
joints, waist, underbust, bust apex, shoulders, then the neck and head joints): the garment's bust
goes on the body's bust, its waist on the waist, its crotch on the crotch. If the landmarks cannot be
read, the fraction between joints is kept.
"""
import math

import numpy as np

import model as MD
import landmarks as RP

STEP = 0.15            # step along the segment (MMD units: 1.2 cm)
NS = 24               # sectors around the segment
OVERFLOW = 2.5          # the grid extends 20 cm past each end of the segment
REACH = 5.0          # maximum radius searched (40 cm)
BLENDED = 0.4           # blend between two segments of a chain (3 cm)
MM = 80.0             # mm per MMD unit
WIDE = (0.1, 0.5)    # offset from the body hull over which fabric goes from the local skin (0.8 cm) to the hull (4 cm)
OWN = 0.3          # share of a face's weights on the segment's chain for it to count as "its" skin
GRAZING = 0.5          # |normal · ray| below which fabric is seen edge-on (more than 60° from the skin it would surround: bottom closing a hem)
TRUNK = ("pelvis", "belly", "bust", "neck")   # segments placed by the trunk's landmark heights
ANCHORS = ("C", "hip", "W", "U", "B", "S", "首", "頭")
SLOPE = (0.4, 2.5)    # plausible range of the height map's slopes; outside it, the landmarks are not trusted
KNOWN_MIN = 12        # number of source skin cells below which a segment takes the overall ratio

# segment: (head joint, tail joint, bones attached to it)
def _side(c):
    return {
        c + "thigh": ((c + "足",), (c + "ひざ",), (c + "足", c + "足D")),
        c + "leg": ((c + "ひざ",), (c + "足首",), (c + "ひざ", c + "ひざD")),
        c + "arm": ((c + "腕",), (c + "ひじ",), (c + "腕", c + "腕捩")),
        c + "forearm": ((c + "ひじ",), (c + "手首",), (c + "ひじ", c + "手捩")),
    }


SEGMENTS = {
    "pelvis": (("下半身",), ("左足", "右足"), ("下半身",)),
    "belly": (("上半身",), ("上半身2",), ("上半身", "上半身1")),
    "bust": (("上半身2",), ("首",), ("上半身2", "上半身3", "左肩", "右肩", "左肩P", "右肩P")),
    "neck": (("首",), ("頭",), ("首",)),
    **_side("左"), **_side("右"),
}
# held by these bones (and their descendants): no cylinder, per-bone fit
CHAINS = (("neck", "bust", "belly", "pelvis"), ("左arm", "左forearm"), ("右arm", "右forearm"),
           ("左thigh", "左leg"), ("右thigh", "右leg"))
TIPS = tuple(c[-1] for c in CHAINS)      # last segment of each chain: beyond it, the axis leaves the body
RIGID_BODIES = ("頭", "左手首", "右手首", "左足首", "右足首", "左足首D", "右足首D")


def _joint(bones, names):
    idx = {o["name"]: i for i, o in enumerate(bones)}
    p = [np.asarray(bones[idx[n]]["pos"], float) for n in names if n in idx]
    return np.mean(p, 0) if len(p) == len(names) else None


class Landmark:
    def __init__(self, h, q):
        self.h, self.q = h, q
        self.L = float(np.linalg.norm(q - h))
        self.u = (q - h) / self.L
        e1 = np.cross(self.u, [0, 0, 1.0])
        if np.linalg.norm(e1) < 0.3:
            e1 = np.cross(self.u, [1.0, 0, 0])
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.u, self.e1)

    def coords(self, X):
        rel = X - self.h
        t = rel @ self.u
        rv = rel - t[:, None] * self.u
        return t, np.arctan2(rv @ self.e2, rv @ self.e1), np.linalg.norm(rv, axis=1)

    def direction(self, ang):
        return np.cos(ang)[:, None] * self.e1 + np.sin(ang)[:, None] * self.e2


def _to_target(t, Ls, Lt, s):
    """Position along the segment: as a fraction between the joints, as a length beyond them."""
    return np.where(t < 0, t * s, np.where(t > Ls, Lt + (t - Ls) * s, t * Lt / Ls))


def _to_source(tt, Ls, Lt, s):
    return np.where(tt < 0, tt / s, np.where(tt > Lt, Ls + (tt - Lt) / s, tt * Ls / Lt))


def _landmarks_body(model):
    """Landmark heights of a body (innermost surface: skin and garments, no face or hair), plus its
    neck and head joints. None if they cannot be read."""
    P = np.asarray(model["vertices"]["pos"], float)
    F = np.asarray(model["faces"], np.int64).reshape(-1, 3)
    mf, d = np.zeros(len(F), np.int64), 0
    for i, m in enumerate(model["materials"]):
        mf[d:d + m["n_indices"] // 3] = i
        d += m["n_indices"] // 3
    guard = np.array([not MD.NON_GARMENT.match(m["name"]) for m in model["materials"]], bool)[mf]
    try:
        r = RP.landmarks(P, F[guard], model["bones"])
    except (ValueError, IndexError, TypeError):
        return None
    for n in ("首", "頭"):
        j = _joint(model["bones"], (n,))
        r[n] = float(j[1]) if j is not None else np.nan
    return r


def map_trunk(src, target):
    """Height map of the trunk, source to target, and its inverse; None if the landmarks of either body
    cannot be read or give an implausible map (then the fraction between joints is used)."""
    rs, rc = _landmarks_body(src), _landmarks_body(target)
    if rs is None or rc is None:
        return None
    c = RP.map_(rs, rc, ANCHORS)
    if c is None:
        return None
    ys = np.array([rs[n] for n in ANCHORS if np.isfinite(rs.get(n, np.nan)) and np.isfinite(rc.get(n, np.nan))])
    ys = np.sort(ys)
    yt = c[0](ys)
    slope = np.diff(yt) / np.maximum(np.diff(ys), 1e-6)
    for n in ("C", "W", "U", "B"):          # the bust and the crotch must be among the anchors
        if not (np.isfinite(rs.get(n, np.nan)) and np.isfinite(rc.get(n, np.nan))):
            return None
    if len(ys) < 4 or (slope[np.diff(ys) > 1e-3] < SLOPE[0]).any() or (slope[np.diff(ys) > 1e-3] > SLOPE[1]).any():
        return None
    return c


def _fill(a):
    """Fills the NaNs of a grid (slices, sectors) with the nearest valid value along the segment."""
    a = a.copy()
    for s in range(a.shape[1]):
        v = np.where(np.isfinite(a[:, s]))[0]
        if len(v):
            a[:, s] = a[v[np.abs(np.arange(a.shape[0])[:, None] - v[None]).argmin(1)], s]
    if np.isnan(a).any():                      # empty sector: mean of the slice
        m = np.nanmean(a, 1)
        a = np.where(np.isnan(a), m[:, None], a)
    return a


def _hull(G, ang):
    """Convex hull of each slice (radii per sector around the axis): what loose fabric rests on,
    since it does not follow hollows (gap between the thighs, between the breasts). Returns the radii."""
    out = G.copy()
    c, sn = np.cos(ang), np.sin(ang)
    n = len(ang)
    for j in range(G.shape[0]):
        P = np.stack([G[j] * c, G[j] * sn], 1)
        guard = list(range(n))
        change = True
        while change and len(guard) > 3:      # remove reflex vertices until none are left
            change = False
            for i in range(len(guard)):
                a_, b_, c_ = P[guard[i - 1]], P[guard[i]], P[guard[(i + 1) % len(guard)]]
                if (b_[0] - a_[0]) * (c_[1] - a_[1]) - (b_[1] - a_[1]) * (c_[0] - a_[0]) < 0:
                    del guard[i]
                    change = True
                    break
        for i in range(len(guard)):           # between two kept vertices: the chord, read along the ray
            i0, i1 = guard[i], guard[(i + 1) % len(guard)]
            a_, b_ = P[i0], P[i1]
            k = (i0 + 1) % n
            while k != i1:
                d = np.array([c[k], sn[k]])
                den = d[0] * (b_[1] - a_[1]) - d[1] * (b_[0] - a_[0])
                if abs(den) > 1e-9:
                    out[j, k] = max(G[j, k], (a_[0] * (b_[1] - a_[1]) - a_[1] * (b_[0] - a_[0])) / den)
                k = (k + 1) % n
    return out


def _read(g, ft, fs):
    """Bilinear lookup (slices clamped, sectors cyclic)."""
    nb = g.shape[0]
    ft = np.clip(ft, 0, nb - 1)
    j0 = np.floor(ft).astype(int)
    j1 = np.minimum(j0 + 1, nb - 1)
    a = ft - j0
    s0 = np.floor(fs).astype(int)
    b = fs - s0
    s0 %= NS
    s1 = (s0 + 1) % NS
    return (1 - a) * ((1 - b) * g[j0, s0] + b * g[j0, s1]) + a * ((1 - b) * g[j1, s0] + b * g[j1, s1])


def _segments_os(bones, own_ones=()):
    """Segment of each bone (itself or its first attached ancestor); -2 = rigid, -1 = none."""
    seg_of_name, names_seg = {}, list(SEGMENTS)
    for k, n in enumerate(names_seg):
        for o in SEGMENTS[n][2]:
            seg_of_name[o] = k
    ns = len(bones)
    seg = np.full(ns, -1)
    for i in range(ns):
        k, n = i, 0
        while 0 <= k < ns and n < 256:
            nm = bones[k]["name"]
            if nm in RIGID_BODIES:
                seg[i] = -2
                break
            if nm in seg_of_name and nm not in own_ones:
                seg[i] = seg_of_name[nm]
                break
            k, n = bones[k]["parent"], n + 1
    return seg


def _part_chains(model, faces, own_ones=()):
    """Share of each chain (trunk, arms, legs) in the weights of each face: (faces, chains)."""
    seg = _segments_os(model["bones"], own_ones)
    names_seg = list(SEGMENTS)
    ch_of_seg = np.full(len(names_seg), -1)
    for c, chain in enumerate(CHAINS):
        for n in chain:
            ch_of_seg[names_seg.index(n)] = c
    ch_os = np.where(seg >= 0, ch_of_seg[np.maximum(seg, 0)], -1)
    I = np.asarray(model["vertices"]["bones"], np.int64).reshape(len(model["vertices"]["pos"]), -1)
    Wm = np.asarray(model["vertices"]["weights"], float).reshape(I.shape)
    Wm = np.where(I >= 0, Wm, 0.0)
    ch = np.where(I >= 0, ch_os[np.clip(I, 0, len(ch_os) - 1)], -1)
    part = np.zeros((len(I), len(CHAINS)))
    for c in range(len(CHAINS)):
        part[:, c] = (Wm * (ch == c)).sum(1)
    part /= np.maximum(Wm.sum(1), 1e-9)[:, None]
    return part[np.asarray(faces, np.int64)].mean(1)


def transport(X, Is, W, src, target, own_ones, tree_target, Y, s, part_faces, tight, direction_target=1.0, faces_target=None):
    """X: vertices of the part in the source; Is, W: bones (indices in the source) and weights;
    Y: the same vertices fitted per bone (fallback); tree_target: surface under the part on the target;
    piece_faces: faces of the part in the source (indices into X); tight: the part counts as the body.
    direction_target: +1 if the normals of the target surface point outward; faces_target: the faces of tree_target
    (indices into the target's vertices), to know which limb each one belongs to. Returns (vertices on the target, figures per segment)."""
    from mathutils.bvhtree import BVHTree
    bones, osc = src["bones"], target["bones"]
    ns = len(bones)
    # ---- segment of each source bone (itself or its first attached ancestor); -2 = rigid
    names_seg = list(SEGMENTS)
    seg = _segments_os(bones, own_ones)
    A = np.where(Is >= 0, seg[np.maximum(Is, 0)], -1)
    # ---- source surfaces: everything except face and hair; the part itself separately
    Ps = np.asarray(src["vertices"]["pos"], float)
    Fs = np.asarray(src["faces"], np.int64).reshape(-1, 3)
    mf, d = np.zeros(len(Fs), np.int64), 0
    for i, m in enumerate(src["materials"]):
        mf[d:d + m["n_indices"] // 3] = i
        d += m["n_indices"] // 3
    names_m = [m["name"] for m in src["materials"]]
    guard = np.array([not MD.NON_GARMENT.match(n) for n in names_m])[mf]
    is_skin = np.array([bool(MD.SKIN.match(n)) for n in names_m])[mf][guard]
    Fg = Fs[guard]
    nfs = np.cross(Ps[Fg[:, 1]] - Ps[Fg[:, 0]], Ps[Fg[:, 2]] - Ps[Fg[:, 0]])
    if is_skin.any():          # orientation of the source skin normals: vote against the body's vertical axis
        cg = Ps[Fg[is_skin]].mean(1) - [np.median(Ps[:, 0]), 0, np.median(Ps[:, 2])]
        cg[:, 1] = 0
        nfs *= 1.0 if float((nfs[is_skin] * cg).sum()) > 0 else -1.0
    tree_s = BVHTree.FromPolygons([tuple(p) for p in Ps], [tuple(int(x) for x in f) for f in Fs[guard]], all_triangles=True)
    tree_p = BVHTree.FromPolygons([tuple(p) for p in X], [tuple(int(x) for x in f) for f in part_faces], all_triangles=True)
    # the source without the part (its vertices are at the same places): what it has underneath, seen from outside
    keys = {tuple(v) for v in np.round(X, 4)}
    of_part = np.array([tuple(v) in keys for v in np.round(Ps, 4)])
    without = np.where(~of_part[Fg].all(1))[0]
    tree_n = BVHTree.FromPolygons([tuple(p) for p in Ps], [tuple(int(x) for x in f) for f in Fg[without]], all_triangles=True) if len(without) else None

    def shot(tree, O, D, between=None, outside=None, its_own=None, shaved=None):
        """First surface hit; `between(face, normal, direction)` true = the ray enters the skin
        (it started outside): keep going until the exit. `outside` (array, filled in): the ray
        started outside the body. `its_own(ray, face)` false = the surface hit belongs to another
        limb (the arm's ray, near the armpit, crosses the torso welded to the arm and exits on the
        other side): this limb has no surface of its own in that direction. `shaved(face, normal,
        direction)` true = fabric seen edge-on (the disc closing a hem from below): it does not
        surround the body, the ray goes through it."""
        outside = np.zeros(len(O), bool) if outside is None else outside
        r = np.full(len(O), np.nan)
        f = np.full(len(O), -1)
        outside[:] = False
        for i in range(len(O)):
            done, o, between_ = 0.0, O[i], False
            for _ in range(6):
                loc, n, k, dist = tree.ray_cast(tuple(o), tuple(D[i]), REACH - done)
                if loc is None:
                    break
                e = between is not None and between(k, n, D[i])
                if e or (shaved is not None and shaved(k, n, D[i])):
                    between_ |= e
                    done += dist + 1e-3
                    o = O[i] + D[i] * done
                    continue
                if its_own is None or its_own(i, k):
                    r[i], f[i] = done + dist, k
                break
            outside[i] = between_
        return r, f

    def last(tree, O, D, lim):
        """Last surface hit before distance `lim` (NaN: ray not cast)."""
        r = np.full(len(O), np.nan)
        f = np.full(len(O), -1)
        if tree is None:
            return r, f
        for i in np.where(np.isfinite(lim))[0]:
            done = 0.0
            for _ in range(40):                # doubled layers, folds: every face crossed counts
                loc, n, k, dist = tree.ray_cast(tuple(O[i] + D[i] * done), tuple(D[i]), lim[i] - 1e-3 - done)
                if loc is None:
                    break
                r[i], f[i] = done + dist, k
                done += dist + 1e-3
        return r, f

    def between_target(k, n, d):
        return float(np.asarray(n) @ d) * direction_target < 0

    def between_source(k, n, d):
        return bool(is_skin[k]) and float(nfs[k] @ d) < 0

    def shaved_source(k, n, d):
        return not is_skin[k] and abs(float(np.asarray(n) @ d)) < GRAZING

    def shaved_part(k, n, d):
        return abs(float(np.asarray(n) @ d)) < GRAZING

    # which limb each face belongs to (share of each chain in its weights)
    part_s = _part_chains(src, Fg, own_ones)
    part_c = _part_chains(target, faces_target) if faces_target is not None else None
    ch_of = {n: c for c, chain in enumerate(CHAINS) for n in chain}

    trunk = [names_seg.index(n) for n in TRUNK]
    heights = map_trunk(src, target) if np.isin(A, trunk).any() else None
    ang = (np.arange(NS) + 0.5) / NS * 2 * np.pi - np.pi
    fields, digits, reports = {}, {}, []
    for k in np.unique(A[A >= 0]):
        head, queue_, _ = SEGMENTS[names_seg[k]]
        hs, qs, ht, qt = _joint(bones, head), _joint(bones, queue_), _joint(osc, head), _joint(osc, queue_)
        if any(v is None for v in (hs, qs, ht, qt)) or np.linalg.norm(qs - hs) < 0.05 or np.linalg.norm(qt - ht) < 0.05:
            continue
        rs, rt = Landmark(hs, qs), Landmark(ht, qt)
        if heights is not None and names_seg[k] in TRUNK and min(abs(rs.u[1]), abs(rt.u[1])) > 0.5:
            # along the trunk: the same anatomical height, whatever the bones (see the module's docstring)
            rs.to_target = (lambda t, rs=rs, rt=rt: (heights[0](rs.h[1] + t * rs.u[1]) - rt.h[1]) / rt.u[1])
            rs.to_source = (lambda tt, rs=rs, rt=rt: (heights[1](rt.h[1] + tt * rt.u[1]) - rs.h[1]) / rs.u[1])
        else:
            rs.to_target = (lambda t, rs=rs, rt=rt: _to_target(t, rs.L, rt.L, s))
            rs.to_source = (lambda tt, rs=rs, rt=rt: _to_source(tt, rs.L, rt.L, s))
        tt = np.arange(-OVERFLOW, rt.L + OVERFLOW + STEP, STEP)
        ts = rs.to_source(tt)
        nb = len(tt)
        Dt, Ds = rt.direction(ang), rs.direction(ang)
        Ot = np.repeat(rt.h[None] + tt[:, None] * rt.u, NS, 0)
        Os = np.repeat(rs.h[None] + ts[:, None] * rs.u, NS, 0)
        out_of = np.zeros(nb * NS, bool)
        # past the end of a chain, the axis is outside the body: any surface counts (the thighs, for
        # the pelvis); elsewhere a limb is measured only on its own skin
        c_k = ch_of[names_seg[k]]
        free = np.repeat(tt > rt.L, NS) if names_seg[k] in TIPS else np.zeros(nb * NS, bool)
        own_c = (lambda i, f: free[i] or part_c[f, c_k] >= OWN) if part_c is not None else None
        Rt, _ = shot(tree_target, Ot, np.tile(Dt, (nb, 1)), between_target, out_of, own_c)
        R1, f1 = shot(tree_s, Os, np.tile(Ds, (nb, 1)), between_source, None, lambda i, f: free[i] or part_s[f, c_k] >= OWN, shaved_source)
        Rp, _ = shot(tree_p, Os, np.tile(Ds, (nb, 1)), shaved=shaved_part)
        # where the axis is outside the body, the source is read like the target, bare or covered
        # skin: the last surface under the outer layer of the part (a lining near the axis is not the
        # body). Outside the body: past the end of a chain (under the crotch, the pelvis axis runs
        # between the thighs, even if the target's thighs touch), or when the target says so (its
        # ray entered the skin before leaving it)
        out_of |= free
        Dss = np.tile(Ds, (nb, 1))
        Rpl, _ = last(tree_p, Os, Dss, np.where(out_of, REACH, np.nan))
        Rd, fd = last(tree_n, Os, Dss, Rpl)
        lm = np.isfinite(Rpl)
        R1 = np.where(lm, np.where(fd >= 0, Rd, Rpl), R1)
        f1 = np.where(lm, np.where(fd >= 0, without[np.maximum(fd, 0)], -1), f1)
        Rp = np.where(lm, Rpl, Rp)
        # seen from outside, a surface twice as close to the axis as the target skin is not the body
        # the part surrounds (panty gusset, leftover skin: games remove hidden skin): unknown
        false = out_of & np.isfinite(Rt) & (R1 < 0.5 * Rt / s)
        R1, f1 = np.where(false, np.nan, R1), np.where(false, -1, f1)
        Rt, R1, f1, Rp = Rt.reshape(nb, NS), R1.reshape(nb, NS), f1.reshape(nb, NS), Rp.reshape(nb, NS)
        if np.isnan(Rt).all():
            continue
        skin = (f1 >= 0) & is_skin[np.maximum(f1, 0)] & np.isfinite(Rt)
        # another garment under the part (shirt under the jacket) stands in for the body: the part is
        # fitted alone on the target, at the offset it had from what it covered. Only the part
        # itself, when it is the first surface, gives no more than a bound on the body.
        is_part = np.isfinite(Rp) & (Rp <= R1 + 1e-4)
        known = skin | ((f1 >= 0) & ~is_part & np.isfinite(Rt))
        fields[k] = [rs, rt, tt[0], Rt, R1, Rp, known, float(skin.mean())]
        if known.sum() >= KNOWN_MIN:
            reports.append(float(np.median(R1[known] / Rt[known])))
    general = float(np.median(reports)) if reports else 1.0 / s
    for k, (rs, rt, t0, Rt, R1, Rp, known, fraction_skin) in fields.items():
        kseg = float(np.median(R1[known] / Rt[known])) if known.sum() >= KNOWN_MIN else general
        Rtc = _fill(Rt)
        # source body / target body ratio: exact where the source skin is present, diffused elsewhere
        rap = np.where(known, R1 / Rtc, kseg)
        for _ in range(12):
            m = (np.roll(rap, 1, 1) + np.roll(rap, -1, 1) + np.r_[rap[:1], rap[:-1]] + np.r_[rap[1:], rap[-1:]]) / 4
            rap = np.where(known, rap, 0.5 * m + 0.5 * kseg)
        body = rap * Rtc
        bound = np.where(np.isfinite(R1), R1, np.inf)          # the body lies under the first fabric
        body = np.where(known, body, np.minimum(body, bound))
        if tight:                                             # the part counts as the body where it is the first surface
            lm = np.isfinite(Rp) & (Rp <= bound + 1e-4)
            body = np.where(lm & ~known, Rp, body)
        fields[k] = (rs, rt, t0, Rtc, body, _hull(Rtc, ang), _hull(body, ang))
        digits[names_seg[k]] = {"source_skin_known": round(fraction_skin, 2),
                                 "source_body_known": round(float(known.mean()), 2), "report": round(kseg, 3),
                                 "source_body_mm": round(float(np.median(body)) * MM, 1),
                                 "target_body_mm": round(float(np.median(Rtc)) * MM, 1)}
    # ---- vertices: per chain, on the segment where the vertex lies
    Z = np.zeros_like(X)
    part = np.zeros(len(X))
    for chain in CHAINS:
        ks = [names_seg.index(n) for n in chain if names_seg.index(n) in fields]
        if not ks:
            continue
        w = (W * np.isin(A, [names_seg.index(n) for n in chain])).sum(1)
        rows = np.where(w > 0)[0]
        if not len(rows):
            continue
        pos, out_of = [], []
        for k in ks:
            rs, rt, t0, Rtc, body, Rte, corpse = fields[k]
            t, a, r = rs.coords(X[rows])
            tc = rs.to_target(t)
            ft, fs = (tc - t0) / STEP, (a + np.pi) / (2 * np.pi) * NS - 0.5
            pt, cs = _read(Rtc, ft, fs), _read(body, ft, fs)
            # fabric far from the body rests on its hull, not on every hollow: where the source has
            # a gap between the thighs and the target does not, the offset to the local skin would jump by 15 cm
            pe, ce = _read(Rte, ft, fs), _read(corpse, ft, fs)
            am = np.clip((r - ce - WIDE[0]) / (WIDE[1] - WIDE[0]), 0, 1)
            pt, cs = pt + am * (pe - pt), cs + am * (ce - cs)
            r2 = np.where(r >= cs, pt + (r - cs) * s, pt * r / np.maximum(cs, 1e-6))
            pos.append(rt.h[None] + tc[:, None] * rt.u + r2[:, None] * rt.direction(a))
            out_of.append(np.maximum(np.maximum(-t, t - rs.L), 0))
        out_of = np.array(out_of)
        om = np.maximum(0, 1 - out_of / BLENDED)
        far = om.sum(0) <= 0                     # beyond the ends of the chain: the nearest segment
        om[out_of.argmin(0)[far], np.where(far)[0]] = 1
        om /= om.sum(0)
        Z[rows] += w[rows, None] * sum(o[:, None] * p_ for o, p_ in zip(om, pos))
        part[rows] += w[rows]
    # the rest of the weights (hands, feet, head, bones without a segment) follows the per-bone fit
    Z += (1 - part)[:, None] * Y
    digits["fraction_transported"] = round(float(part.mean()), 3)
    if np.isin(A, trunk).any():
        digits["torso_heights"] = "landmarks" if heights is not None else "fractions"
    return Z, digits

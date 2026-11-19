"""Fit a part from the garment library onto another ZZZ body.

    bin/blender-queue -b --factory-startup --python tools/clothes/wear.py -- \
        --part <library>/<Character>/<part> --body BODY.pmx --output DIR [--name NAME]
        [--margin-mm 3] [--skin keep|remove] [--render]

The body may be a PMX that is already dressed (output of a previous fitting): the part is then
placed over the garment (a belt over a skirt). Output: DIR/NAME.pmx (body + part, textures in
DIR/tex, relative paths), DIR/portage.json (materials of the part, type for measure.py,
fitting figures), and with --render DIR/render.png (front, back, side).

Steps (MMD units and frame, 1 unit = 8 cm):
1. skeletons: for each body bone present on both sides (same name), transform
   source head -> target head, minimal rotation of the head->tail direction, scale along the
   bone (length ratio, clamped to 0.85-1.2) and across it (size ratio). For a shoe, a tail that is a bone
   found on both sides is taken at that bone's head on the target too (a VRoid knee's own tail
   stops mid-shin), so two bones meet at their joint on the target as on the source; a deformation
   duplicate without a tail (ひざD) turns and scales like the bone it copies. This per-bone fit
   now only places the hands, the feet, the head and the part's bones (see 3). The part's own
   bones (physics chains) and the bones missing from the target follow their first fitted
   ancestor, except a deformation duplicate (足首D copies 足首's rotation from its head), which
   is placed with the bone it copies.
2. linear skinning of the part's vertices with their original weights.
3. body-to-body transport (`transport.py`): around each body segment (thigh, arm,
   pelvis, chest...), each vertex keeps its position along the segment, its angle and its offset
   from the skin; the source body is its skin where it is visible, the garment underneath
   elsewhere, the part itself for a tight-fitting part. No bound. Hands, feet and head stay on
   the per-bone fit of step 2.
4. push out of the surface underneath (body, or body + garments already fitted): each vertex
   inside, or closer than the margin (vertex, or face centre and edge midpoint less than
   15 mm inside), is moved out along the normal; the displacement is spread to neighbours within
   3 cm, until nothing is left inside (at most 12 passes; `stays_inside` counts what is
   left). `under_skin` then counts the vertices inside the body with no depth limit
   (ray parity): this is the check that rejects a part in the workshop.
5. weights: near the body (< 3 cm), the part keeps its original weights, corrected by the
   difference between the target skin and what it had underneath on its source (a panel hanging
   from the chest next to the arm does not take the arm's weights); with nothing underneath on
   the source, it takes those of the target skin. The part's own bones keep their share.
6. hidden skin (garments only): skin faces whose outgoing ray hits the part within 5 cm,
   on an area held by the body (not by a physics bone), are removed.
7. the part's rigid bodies and joints follow their bone (position and rotation).
"""
import copy
import json
import math
import os
import re
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "atlas"))
sys.path.insert(0, os.path.join(ROOT, "head"))
import model as MD  # noqa: E402
import foot as PD  # noqa: E402
import pmx as PMXL  # noqa: E402

MM = 1 / 80.0          # MMD units per mm
UNIT = 0.08           # metres per MMD unit
LACE_FOOT = 20.0      # degrees: a shoe turns at most this much about the vertical to the target foot's heading
TYPES_FOOT = ("shoes", "boots")
SOLE_PASSES = 3      # a posed shoe is lowered until the foot no longer goes through its sole
TYPES_MEASURE = {"jacket": "jacket_open", "waistcoat": "jacket_open", "coat": "coat_long",
                "top": "loose_top", "jumpsuit": "loose_top", "dress": "skirt", "skirt": "skirt",
                "trousers": "trousers", "shorts": "shorts", "socks": "trousers"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ====================================================================== rotations

def rot_mmd(r):
    """Rotation of a PMX rigid body or joint (YXZ order: R = Ry · Rx · Rz)."""
    x, y, z = r
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def euler_mmd(R):
    x = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    return (x, math.atan2(R[0, 2], R[2, 2]), math.atan2(R[1, 0], R[1, 1]))


def minimal_rotation(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -0.999:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


# ====================================================================== skeletons

def queue(bones, i):
    o = bones[i]
    if o["tail_bone"] is not None:
        q = o["tail_bone"]
        return np.asarray(bones[q]["pos"], float) if 0 <= q < len(bones) and q != i else None
    d = np.asarray(o["tail_offset"], float)
    return np.asarray(o["pos"], float) + d if np.linalg.norm(d) > 1e-6 else None


def height(pmx):
    p = np.asarray(pmx["vertices"]["pos"], float)
    return float(p[:, 1].max() - p[:, 1].min())


class Registration:
    """Per-bone transforms, source -> target: x' = h_t + R · A · (x - h_s)."""

    def __init__(self, src, target, s_global, feet_rigid_bodies=False, shoe=False):
        """shoe: a shoe's fit. Its bones are then measured between the same joints on both
        bodies (_queue_target) and a tail-less duplicate (ひざD) turns like the bone it copies. Every
        other part keeps the plain per-bone fit: its limbs are placed by the transport, and those
        two rules moved the normals and the off-segment bones of shorts and socks for nothing."""
        self.src, self.target = src["bones"], target["bones"]
        self.shoe = shoe
        self._in_progress = set()
        self.name_c = {o["name"]: i for i, o in enumerate(self.target)}
        n = len(self.src)
        self.R = [None] * n
        self.A = [None] * n
        self.hs = [None] * n
        self.ht = [None] * n
        self.direct = np.zeros(n, bool)
        self.angles = {}
        self.duplicates = {}
        for i, o in enumerate(self.src):
            j = self.name_c.get(o["name"])
            if j is None:
                j = self._double(i)
            if j is None:
                continue
            hs = np.asarray(o["pos"], float)
            ht = np.asarray(self.target[j]["pos"], float)
            self.hs[i], self.ht[i] = hs, ht
            self.direct[i] = True
            qs, qt = queue(self.src, i), (self._queue_target(i, j) if shoe else queue(self.target, j))
            if qs is None or qt is None:
                continue
            ds, dt = qs - hs, qt - ht
            ls, lt = np.linalg.norm(ds), np.linalg.norm(dt)
            if ls < 0.05 or lt < 0.05:
                continue
            ang = math.degrees(math.acos(max(-1, min(1, ds @ dt / ls / lt))))
            if ang > 30:          # tails defined differently: the direction is not trusted
                continue
            self.angles[o["name"]] = round(ang, 1)
            u = ds / ls
            sa = min(1.2, max(0.85, lt / ls))
            self.R[i] = minimal_rotation(ds, dt)
            self.A[i] = s_global * np.eye(3) + (sa - s_global) * np.outer(u, u)
        # a deformation duplicate with no tail of its own (ひざD) moves like the bone it copies: inheriting
        # its parent's (足D, the thigh) would map the shin away from the ankle, and a boot shaft weighted
        # between the knee and the ankle would be torn apart by the gap between the two
        for i in range(n if shoe else 0):
            k = self._copy(i)
            if self.R[i] is None and k is not None and self.R[k] is not None and self.hs[i] is not None:
                self.R[i], self.A[i] = self.R[k], self.A[k]
        if feet_rigid_bodies:
            self._feet_rigid_bodies(s_global)
        # inheritance: rotation/scale of the first ancestor that has one, head of the first fitted ancestor
        self.s_global = s_global
        for i in range(n):
            self._resolve(i)

    def _feet_rigid_bodies(self, s_global):
        """A shoe keeps the shape it has on its source: every bone of the source foot moves it as one
        block around the ankle's head (scale s_global, turned only about the vertical to the target
        foot's heading). The target foot is posed into it (foot.py), not the other way round."""
        cs, ct = PD.chains(self.src), PD.chains(self.target)
        for name, c in cs.items():
            if name not in ct:
                continue
            a, b = c["ht"] - c["ha"], ct[name]["ht"] - ct[name]["ha"]
            a[1] = b[1] = 0.0
            lace = math.degrees(math.atan2(np.cross(a, b)[1], float(a @ b)))
            R = PD.rotation(np.array([0.0, 1.0, 0.0]), max(-LACE_FOOT, min(LACE_FOOT, lace)))
            for i in c["foot"]:
                self.hs[i], self.ht[i] = c["ha"], ct[name]["ha"]
                self.R[i], self.A[i] = R, s_global * np.eye(3)
                self.direct[i] = True

    def _copy(self, i):
        """Source bone whose rotation bone i copies whole (grant factor 1) from the same head, or None."""
        o = self.src[i]
        a = o.get("addition")
        k = a["source"] if a else -1
        if not (0 <= k < len(self.src)) or k == i or not a.get("rotation") or abs(a.get("factor", 0) - 1) > 1e-3:
            return None
        if np.linalg.norm(np.asarray(o["pos"], float) - np.asarray(self.src[k]["pos"], float)) > 1e-3:
            return None
        return k

    def _queue_target(self, i, j):
        """Target end of source bone i (target bone j): when the source's tail is another bone found on
        the target (same name, or the bone a missing duplicate copies), that bone's head, so that both
        sides measure the bone between the same joints; otherwise the target bone's own tail."""
        q = self.src[i]["tail_bone"]
        if q is not None and 0 <= q < len(self.src) and q != i:
            jq = self.name_c.get(self.src[q]["name"])
            if jq is None:
                jq = self._double(q)
            if jq is not None and jq != j:
                return np.asarray(self.target[jq]["pos"], float)
        return queue(self.target, j)

    def _double(self, i):
        """Target bone standing for source bone i when the target lacks it: the bone whose rotation i copies
        whole (grant factor 1) from the same head. That is a deformation duplicate (MMD 足首D of 足首): a
        target without duplicates (a VRoid body) moves the duplicate's vertices with the original. Without
        it, the duplicate would follow its first fitted ancestor (下半身) and a shoe would land at the hips' offset."""
        k = self._copy(i)
        if k is None or i in self._in_progress:       # two bones copying each other: no original
            return None
        j = self.name_c.get(self.src[k]["name"])
        if j is None:
            self._in_progress.add(i)
            j = self._double(k)
            self._in_progress.discard(i)
        if j is not None:
            self.duplicates[self.src[i]["name"]] = self.target[j]["name"]
        return j

    def _resolve(self, i):
        if self.R[i] is not None and self.hs[i] is not None:
            return
        p = self.src[i]["parent"]
        if 0 <= p < len(self.src) and p != i:
            self._resolve(p)
            if self.hs[i] is None:      # bone missing from the target: follows its parent (rigidly)
                self.hs[i], self.ht[i] = self.hs[p], self.ht[p]
            if self.R[i] is None:
                self.R[i], self.A[i] = self.R[p], self.A[p]
        else:
            if self.hs[i] is None:
                self.hs[i] = self.ht[i] = np.zeros(3)
            if self.R[i] is None:
                self.R[i], self.A[i] = np.eye(3), self.s_global * np.eye(3)

    def point(self, i, x):
        return self.ht[i] + self.R[i] @ (self.A[i] @ (np.asarray(x, float) - self.hs[i]))

    def points(self, I, W, X):
        """LBS of X (n, 3) with indices I (n, 4) and weights W (n, 4)."""
        out = np.zeros_like(X)
        for b in np.unique(I[I >= 0]):
            m = (I == b) & (W > 0)
            rows = np.where(m.any(1))[0]
            w = (W * m).sum(1)[rows]
            Y = self.ht[b] + ((X[rows] - self.hs[b]) @ self.A[b].T) @ self.R[b].T
            out[rows] += w[:, None] * Y
        return out

    def vectors(self, I, W, N):
        out = np.zeros_like(N)
        for b in np.unique(I[I >= 0]):
            m = (I == b) & (W > 0)
            rows = np.where(m.any(1))[0]
            w = (W * m).sum(1)[rows]
            out[rows] += w[:, None] * (N[rows] @ self.R[b].T)
        return out / np.maximum(np.linalg.norm(out, axis=1), 1e-12)[:, None]


# ====================================================================== surfaces (BVH)

def bvh(P, F):
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons([tuple(p) for p in P], [tuple(int(x) for x in f) for f in F], all_triangles=True)


def normals_faces(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-20)[:, None]


def sign_outside(P, F, nf):
    """+1 if the face winding gives outward normals (MMD frame), otherwise -1: vote of the faces
    against the body's vertical axis."""
    c = P[F].mean(1)
    axis = np.array([np.median(P[:, 0]), 0, np.median(P[:, 2])])
    r = c - axis
    r[:, 1] = 0
    return 1.0 if float((nf * r).sum(1).mean()) > 0 else -1.0


def under_skin(Z, tree, mm, sg=1.0):
    """Vertices inside the body, with no depth limit: parity of the surface crossings along three
    rays (majority of the three, since the body has openings). A short ray along the normal does
    not see a part sunk several centimetres deep; parity does.
    Parity is wrong near openings: a vertex is only counted if the nearest face also sees it from
    behind (two independent clues).
    Counting is done by position: the centre of a fan-shaped cap (the waist bottom of a skirt,
    hidden in the body as on its source) is a single point repeated about twenty times,
    and twenty buried "vertices" got the part rejected for a single point."""
    Z = np.asarray(Z, float)
    if len(Z):
        Z = Z[np.unique(np.round(Z / (0.1 * mm)).astype(np.int64), axis=0, return_index=True)[1]]
    dirs = np.array([(0.577, 0.577, 0.578), (-0.8, 0.1, 0.59), (0.3, -0.2, -0.933)])
    votes = np.zeros(len(Z), int)
    for D in dirs:
        for i, z in enumerate(Z):
            n, o = 0, np.asarray(z, float)
            for _ in range(64):
                loc, _n, _k, dist = tree.ray_cast(tuple(o), tuple(D))
                if loc is None:
                    break
                n += 1
                o = o + D * (dist + 1e-4)
            votes[i] += n % 2
    depth = np.zeros(len(Z))
    for i in np.where(votes >= 2)[0]:
        loc, _n, _k, dist = tree.find_nearest(tuple(Z[i]))
        if loc is not None and sg * float(np.dot(np.asarray(Z[i]) - np.asarray(loc), _n)) < 0:
            depth[i] = dist
    d = depth > 1 * mm
    return {"fraction": round(float(d.mean()), 4) if len(Z) else 0.0, "vertices": int(d.sum()),
            "depth_p50_mm": round(float(np.median(depth[d]) / mm), 1) if d.any() else 0.0,
            "depth_max_mm": round(float(depth.max() / mm), 1) if d.any() else 0.0}


def lids(X, Fp, bones):
    """Caps of the part: a closed fan of faces around a centre, crossed by a body bone.
    This is the bottom that closes an opening around the body (armhole of a jacket, end of a
    sleeve, top of a sock): the bone continues on each side, so the body passes through it, the
    bottom is hidden inside on its source and stays hidden on the target. A bone that ends inside
    the part (hand in a glove, foot in a shoe) does not count: that end is visible.
    Returns (centre vertices, cap faces) as booleans and, per cap, (centre vertices,
    rim vertices). The centre may be a single vertex or a point repeated on each face."""
    import transport as TR
    the_key = np.unique(np.round(X / (0.1 * MM)).astype(np.int64), axis=0, return_inverse=True)[1].ravel()
    Fw = the_key[Fp]
    real = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 2] != Fw[:, 0])
    nb = np.bincount(Fw[real].ravel(), minlength=int(the_key.max()) + 1)
    axes = []
    for a, b in LID_OS:
        h, q = TR._joint(bones, (a,)), TR._joint(bones, (b,))
        if h is not None and q is not None and np.linalg.norm(q - h) > 0.05:
            axes.append((h, q))
    centers, faces, links = np.zeros(len(X), bool), np.zeros(len(Fp), bool), []
    for c in np.where(nb >= LID_MIN)[0]:
        fs = np.where(real & (Fw == c).any(1))[0]
        girth = Fw[fs][Fw[fs] != c]
        if len(girth) != 2 * len(fs) or (np.bincount(girth)[np.unique(girth)] != 2).any():
            continue                                    # open fan: not a bottom
        T = X[Fp[fs]]
        e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
        pierced = False
        for a, b in axes:                               # segment against triangles (Moller-Trumbore)
            d = b - a
            pv = np.cross(d, e2)
            det = (e1 * pv).sum(1)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            tv = a - T[:, 0]
            uu = (tv * pv).sum(1) * inv
            qv = np.cross(tv, e1)
            vv = (qv @ d) * inv
            tt = (e2 * qv).sum(1) * inv
            if (ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt >= 0) & (tt <= 1)).any():
                pierced = True
                break
        if pierced:
            centers[the_key == c] = True
            faces[fs] = True
            links.append((np.where(the_key == c)[0], np.where(np.isin(the_key, np.unique(girth)))[0]))
    return centers, faces, links


# ====================================================================== settings

PASSES_MAX = 12
LID_MIN = 8         # faces around the centre of a cap
LID_OS = tuple(     # body bones that continue on each side of what they cross
    [("下半身", "上半身"), ("上半身", "上半身2"), ("上半身2", "首"), ("首", "頭")]
    + [(c + a, c + b) for c in "左右" for a, b in (("肩", "腕"), ("腕", "ひじ"), ("ひじ", "手首"), ("足", "ひざ"), ("ひざ", "足首"))]
    + [("上半身2", c + "肩") for c in "左右"] + [("下半身", c + "足") for c in "左右"])
TIGHT = ("socks", "tights")     # the part counts as the body: it is placed on the skin


# ====================================================================== rings attached to their carrier

LOOP_SECTORS = 24
LOOP_SURROUNDS_MIN = 0.5    # share of sectors where the ring has something between its centre and itself (surrounds)
LOOP_PERIMETER_MIN = 0.75       # share of sectors covered by the part around its axis
BORDER_PERIMETER_MIN = 0.6          # share of sectors where a garment has a measurable (top or bottom) edge
BORDER_GAP = 40 * MM         # distance (along the axis) between the ring and the edge of the source carrier
GAP_MAX = 20 * MM          # clearance kept between the ring and what it surrounds (source), clamped
SUPPORT_MIN = 0.5              # share of hits under the ring that land on fabric, below which it slides onto its carrier


def landmark_loop(X):
    """Centre, axis (normal of the mean plane, pointing up or away from the body), two axes
    of the plane, and share of sectors covered around the axis."""
    c = X.mean(0)
    _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
    n = Vt[2]
    if abs(n[1]) >= 0.3:
        n = n if n[1] > 0 else -n
    else:
        o = np.array([c[0], 0, c[2]])
        n = n if n @ o >= 0 else -n
    e1 = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(e1) < 0.3:
        e1 = np.cross(n, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    rel = X - c
    ang = np.arctan2(rel @ e2, rel @ e1)
    sect = ((ang + np.pi) / (2 * np.pi) * LOOP_SECTORS).astype(int) % LOOP_SECTORS
    return c, n, e1, e2, len(np.unique(sect)) / LOOP_SECTORS


def borders(V, c, n, e1, e2, rmax, hmax):
    """Top and bottom edges of a garment (vertices V) around the axis (c, n): per sector, the
    highest vertex ("top" edge: the garment is below, waistband of a skirt) or the lowest
    ("bottom" edge: hem of a top). Returns {direction: (centre, normal, coverage, height)} for
    edges that are complete enough; centre and normal of a plane fitted to the edge points."""
    rel = V - c
    h = rel @ n
    rv = rel - h[:, None] * n
    r = np.linalg.norm(rv, axis=1)
    m = (r < rmax) & (np.abs(h) < hmax) & (r > 1e-6)
    if m.sum() < 8:
        return {}
    idx = np.where(m)[0]
    ang = np.arctan2(rv[idx] @ e2, rv[idx] @ e1)
    sect = ((ang + np.pi) / (2 * np.pi) * LOOP_SECTORS).astype(int) % LOOP_SECTORS
    out = {}
    for direction, choice in (("top", np.argmax), ("bottom", np.argmin)):
        pts = []
        for s in np.unique(sect):
            k = idx[sect == s]
            pts.append(V[k[choice(h[k])]])
        cov = len(pts) / LOOP_SECTORS
        if cov < BORDER_PERIMETER_MIN:
            continue
        pts = np.array(pts)
        cb = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - cb, full_matrices=False)
        # a sector with no edge vertex (sparse mesh, slit) gives a point much lower down
        # (Anby, skirt: 3 sectors out of 24 at -50..-200 mm) that tilts the plane: the worst point is
        # dropped and the plane refitted, as long as it clearly stands apart from the others
        while len(pts) > 6:
            e = np.abs((pts - cb) @ Vt[2])
            if e.max() <= max(10 * MM, 3 * np.median(e)):
                break
            pts = np.delete(pts, np.argmax(e), 0)
            cb = pts.mean(0)
            _, _, Vt = np.linalg.svd(pts - cb, full_matrices=False)
        nb = Vt[2] if Vt[2] @ n >= 0 else -Vt[2]
        if nb @ n < 0.7:            # edge tilted too much: not a garment waistband
            nb = n
        out[direction] = (cb, nb, cov, float((cb - c) @ n))
    return out


def rays_surface(tree, c, n, e1, e2, hs, r_ext, fabric=None):
    """Per sector, radius of the outermost surface under the ring (ray cast from r_ext
    towards the axis, at heights hs); nan if nothing is hit. With `fabric` (boolean per tree face),
    also returns the share of hits that land on fabric (not on skin)."""
    out = np.full(LOOP_SECTORS, np.nan)
    nt = nk = 0
    for s in range(LOOP_SECTORS):
        a = (s + 0.5) / LOOP_SECTORS * 2 * np.pi - np.pi
        d = math.cos(a) * e1 + math.sin(a) * e2
        best = np.nan
        for h in hs:
            o = c + h * n + r_ext[s] * d
            loc, _, fi, dist = tree.ray_cast(tuple(o), tuple(-d), float(r_ext[s]))
            if loc is not None:
                rs = r_ext[s] - dist
                best = rs if np.isnan(best) else max(best, rs)
                if fabric is not None:
                    nk += 1
                    nt += bool(fabric[fi])
        out[s] = best
    if fabric is not None:
        return out, (nt / nk if nk else 0.0)
    return out


def surrounds(tree, c, n, e1, e2, hs, rin):
    """Share of sectors where, going from the centre towards the ring, one meets what it surrounds
    (a limb, the waist, the neck, the head) before reaching its inner edge. A plate or a pendant
    lying next to the body also covers the full turn around its centre, but surrounds nothing."""
    k = t = 0
    for s in range(LOOP_SECTORS):
        if np.isnan(rin[s]):
            continue
        a = (s + 0.5) / LOOP_SECTORS * 2 * np.pi - np.pi
        d = math.cos(a) * e1 + math.sin(a) * e2
        k += 1
        t += any(tree.ray_cast(tuple(c + h * n), tuple(d), float(rin[s]) + 5 * MM)[0] is not None for h in hs)
    return t / k if k else 0.0


def profile_loop(X, c, n, e1, e2):
    """Per sector: inner (min) radius of the ring, min/max heights; sector of each vertex."""
    rel = X - c
    h = rel @ n
    rv = rel - h[:, None] * n
    r = np.linalg.norm(rv, axis=1)
    ang = np.arctan2(rv @ e2, rv @ e1)
    sect = ((ang + np.pi) / (2 * np.pi) * LOOP_SECTORS).astype(int) % LOOP_SECTORS
    rin = np.full(LOOP_SECTORS, np.nan)
    for s in np.unique(sect):
        rin[s] = r[sect == s].min()
    return rin, h, r, ang, rv


def fill(v):
    """Empty sectors (nan): interpolated circularly between the known neighbours."""
    v = np.asarray(v, float).copy()
    ok = ~np.isnan(v)
    if not ok.any():
        return np.zeros_like(v)
    k = np.arange(len(v))
    kk = np.concatenate([k[ok] - len(v), k[ok], k[ok] + len(v)])
    vv = np.tile(v[ok], 3)
    return np.interp(k, kk, vv)


UNDER_NEAR = 5 * MM     # below this, the direction to the surface is not reliable
UNDER_COS = 0.7         # beyond 45° from the normal, the nearby point is beside, not underneath
REACH_FAR = 15 * 10     # mm: reach of the target skin when what lies under the source is known


def hook_loop(card, P, src, Y, s_glob, C, Fc, mfc, fdes, tree_under, margin, on_fabric=True, adjust=True):
    """A ring-shaped accessory (belt, garter, armband...) is fitted onto the garment worn by the
    target: on the source, its position is measured relative to the edge of the garment it
    surrounds (height along the axis, relative tilt of the plane, clearance between the ring and
    the fabric); on the target, the edge of the same direction (top of a skirt, of shorts) nearest
    to the part fitted by bones is found, the ring is placed there (same relative height, plane
    parallel to the edge), then its circumference is adjusted sector by sector onto the worn
    surface (body + garments), at the source's clearance. With no carrying edge, only the
    circumference is adjusted. Returns None if the part is not a ring, otherwise (Z, rigid
    transform (Rm, t), info)."""
    Xs = np.asarray(P["vertices"]["pos"], float)
    c_s, n_s, e1_s, e2_s, girth = landmark_loop(Xs)
    if girth < LOOP_PERIMETER_MIN:
        return None
    info = {"perimeter_source": round(girth, 2)}
    rin_s, h_s, _, _, _ = profile_loop(Xs, c_s, n_s, e1_s, e2_s)
    rmed = float(np.nanmedian(rin_s))
    # ---- source: garments from the cutting, surface under the ring (everything except the part)
    back = os.path.dirname(os.path.dirname(os.path.join(card["_dossier"], "")))
    offset = json.load(open(os.path.join(back, "cutout.json"), encoding="utf-8"))
    Ps = np.asarray(src["vertices"]["pos"], float)
    Fs = np.asarray(src["faces"], np.int64).reshape(-1, 3)
    a_me = set(offset["parts"][card["id"]]["faces"]) | set(offset["parts"][card["id"]]["layer_faces"])
    surf = [f for f in offset["skin"]["faces"]]
    carrier_s = None
    for pid, pc in offset["parts"].items():
        if pid == card["id"]:
            continue
        surf.extend(pc["faces"])
        fch = os.path.join(back, pid, "record.json")
        if not os.path.exists(fch) or json.load(open(fch, encoding="utf-8"))["class"] != "garment":
            continue
        V = Ps[np.unique(Fs[pc["faces"]])]
        for direction, (cb, nb, cov, hb) in borders(V, c_s, n_s, e1_s, e2_s, 2.5 * rmed, 40 * 10 * MM).items():
            if h_s.min() - BORDER_GAP <= hb <= h_s.max() + BORDER_GAP:
                if carrier_s is None or abs(hb) < abs(carrier_s[2][3]):
                    carrier_s = (pid, direction, (cb, nb, cov, hb))
    surf = np.array(sorted(set(surf) - a_me), np.int64)
    tree_s = bvh(Ps, Fs[surf])
    fabric_s = ~np.isin(surf, offset["skin"]["faces"])
    r_ext = np.nan_to_num(rin_s, nan=rmed) + 60 * MM
    hs = [float(np.percentile(h_s, q)) for q in (15, 50, 85)]
    rsurf_s, support_s = rays_surface(tree_s, c_s, n_s, e1_s, e2_s, hs, r_ext, fabric_s)
    game = np.clip(fill(rin_s - rsurf_s), 0, GAP_MAX)
    info["game_source_mm"] = round(float(np.median(game)) / MM, 1)
    info["support_fabric_source"] = round(support_s, 2)
    info["surrounds_source"] = round(surrounds(tree_s, c_s, n_s, e1_s, e2_s, hs, rin_s), 2)
    # a ring surrounds something: a wing, a bag, a harness also cover the full turn around
    # their centre, and adjusting their "circumference" onto the body stretched them around it
    # (Phionie's wings: 7.5 cm; Koleda's harness: 7.3 cm). For those, `porter` also tries placing
    # them as one block without adjusting the circumference (`adjust=False`) and the transport.
    info["loop"] = bool(info["surrounds_source"] >= LOOP_SURROUNDS_MIN)
    # ---- target: frame of the part fitted by bones, edges of the garments worn
    c_y, n_y, e1_y, e2_y, _ = landmark_loop(Y)
    names_c = [m["name"] for m in C["materials"]]
    Pc = np.asarray(C["vertices"]["pos"], float)
    fabric_c = ~np.array([bool(MD.SKIN.match(nm)) for nm in names_c])[mfc[fdes]]
    Q = minimal_rotation(n_s, n_y)
    c_t = c_y
    border_t = None
    if carrier_s is not None and on_fabric:
        pid, direction, (cb_s, nb_s, _, _) = carrier_s
        info["carrier_source"] = "%s (edge %s)" % (pid, direction)
        best = None
        for i, nm in enumerate(names_c):
            if MD.NON_GARMENT.match(nm) or MD.SKIN.match(nm) or C["materials"][i]["n_indices"] == 0:
                continue
            V = Pc[np.unique(Fc[mfc == i])]
            b = borders(V, c_y, n_y, e1_y, e2_y, 2.5 * rmed * s_glob, 40 * 10 * MM).get(direction)
            if b is not None and (best is None or abs(b[3]) < abs(best[1][3])):
                best = (nm, b)
        if best is not None and abs(best[1][3]) < 30 * 10 * MM:
            nm, (cb_t, nb_t, cov_t, hb_t) = best
            # centre: same position relative to the edge as on the source; plane: that of the carrying edge.
            # The source's relative tilt is not kept: it was measured against another garment
            # (Lucy: belt tilted 7° on the edge of her shorts), and carried over to another edge
            # it would tilt the ring on it; a ring held by an edge follows that edge.
            c_t = cb_t + s_glob * (minimal_rotation(nb_s, nb_t) @ (c_s - cb_s))
            Q = minimal_rotation(n_s, nb_t)
            info["carrier_target"] = "%s (edge %s, %d %% of the circumference)" % (nm, direction, round(100 * cov_t))
            info["displacement_border_mm"] = round(float(np.linalg.norm(c_t - c_y)) / MM, 1)
            info["tilt_border_deg"] = round(math.degrees(math.acos(max(-1, min(1, nb_s @ nb_t)))), 1)
            border_t = (cb_t, nb_t, direction)
    n_t, e1_t, e2_t = Q @ n_s, Q @ e1_s, Q @ e2_s
    hs_t = [s_glob * h for h in hs]

    def support(ct):
        Zt = Xs @ (s_glob * Q).T + (ct - s_glob * Q @ c_s)
        rin = profile_loop(Zt, ct, n_t, e1_t, e2_t)[0]
        return rays_surface(tree_under, ct, n_t, e1_t, e2_t, hs_t,
                              np.nan_to_num(rin, nan=rmed * s_glob) + 80 * MM, fabric_c)[1]
    # a fitted accessory rests on its carrier: if the ring lands mostly on skin on the target
    # (bare waist above a low skirt), it slides along the axis towards the carrier until it is
    # entirely on it (its top edge at the garment's top edge, or its bottom edge at the hem).
    # This holds even if the source wore it on the skin (Lucy: belt on the bare belly,
    # support 0.18): once fitted, it holds to the garment.
    if border_t is not None:
        a0 = support(c_t)
        info["support_fabric_target"] = round(a0, 2)
        cb_t, nb_t, direction = border_t
        h_rel = s_glob * h_s - s_glob * (h_s.mean())          # heights of the ring around its centre
        top_loop = float((c_t - cb_t) @ nb_t) + (np.percentile(h_rel, 97) if direction == "top" else np.percentile(h_rel, 3))
        if a0 < SUPPORT_MIN and ((direction == "top" and top_loop > 0) or (direction == "bottom" and top_loop < 0)):
            offset_h = -(top_loop + (2 * MM if direction == "top" else -2 * MM))
            c_t = c_t + offset_h * nb_t
            info["slide_on_carrier_mm"] = round(float(offset_h) / MM, 1)
            info["support_fabric_target"] = round(support(c_t), 2)
    # rigid placement: x' = c_t + s Q (x - c_s)
    Rm = s_glob * Q
    t = c_t - Rm @ c_s
    Z = Xs @ Rm.T + t
    if not adjust:
        return Z, (Rm, t, Q), info
    # ---- circumference: per sector, the ring rests on the worn surface, at the source's clearance
    rin_t, h_t, _, ang_t, rv_t = profile_loop(Z, c_t, n_t, e1_t, e2_t)
    r_ext = np.nan_to_num(rin_t, nan=rmed * s_glob) + 80 * MM
    rsurf_t = rays_surface(tree_under, c_t, n_t, e1_t, e2_t, hs_t, r_ext)
    wanted = rsurf_t + np.maximum(game * s_glob, margin)
    delta = fill(wanted - rin_t)
    delta = (np.roll(delta, 1) + 2 * delta + np.roll(delta, -1)) / 4
    fs = (ang_t + np.pi) / (2 * np.pi) * LOOP_SECTORS - 0.5
    s0 = np.floor(fs).astype(int)
    b = fs - s0
    dv = (1 - b) * delta[s0 % LOOP_SECTORS] + b * delta[(s0 + 1) % LOOP_SECTORS]
    ru = rv_t / np.maximum(np.linalg.norm(rv_t, axis=1), 1e-9)[:, None]
    Z = Z + dv[:, None] * ru
    info["perimeter_mm"] = {"min": round(float(delta.min()) / MM, 1), "median": round(float(np.median(delta)) / MM, 1),
                       "max": round(float(delta.max()) / MM, 1)}
    return Z, (Rm, t, Q), info


# ====================================================================== fitting

SOLE_GROUND = 3 * 10 * MM     # a sole touches the ground: the bottom of the part is closer than this to y = 0 on its source
SOLE_AREA = 40.0           # cm² of flat faces (vertical normal) in the bottom 4 cm: a sock has none


def a_sole(P):
    """The part has a sole: it reaches the ground and its underside is flat. Measured on the
    library: 0 to 4 cm² of flat area for socks, 72 to 580 for shoes and boots (only one boot
    under 72, with no sole edge)."""
    X = np.asarray(P["vertices"]["pos"], float)
    F = np.asarray(P["faces"], np.int64).reshape(-1, 3)
    if not len(F) or X[:, 1].min() > SOLE_GROUND:
        return False
    T = X[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    a = np.linalg.norm(n, axis=1) / 2
    flat = (T[:, :, 1].max(1) < X[:, 1].min() + 40 * MM) & (np.abs(n[:, 1]) > 0.9 * 2 * a)
    return float(a[flat].sum()) / (10 * MM) ** 2 >= SOLE_AREA


def load_part(folder):
    if folder.endswith(".pmx"):
        folder = os.path.dirname(folder)
    card = json.load(open(os.path.join(folder, "record.json"), encoding="utf-8"))
    card["_dossier"] = os.path.abspath(folder)
    # a garment is fitted with its trimmings (buttons, pockets, straps: library.trim_character)
    g = card.get("trimmed")
    path = os.path.join(folder, card["pmx"])
    if g and os.path.exists(os.path.join(folder, g["pmx"])):
        card["own_bones"] = g["own_bones"]
        path = os.path.join(folder, g["pmx"])
    # the cutting names "socks" whatever dresses the leg when the character has no
    # separate shoes: with a sole, it is a boot (rigid object, not a second skin)
    if card.get("type") == "socks" and a_sole(PMXL.read(path)):
        card["type"] = "boots"
    # it also sometimes separates the sole and heel of a shoe from its upper, and names them
    # "anklet" (ankle accessory): it is a shoe, fitted and checked as such
    elif card.get("class") == "accessory" and card.get("sort") == "anklet" and a_sole(PMXL.read(path)):
        card["class"], card["type"] = "garment", "shoes"
    return card, path


def inner_layers(X, N, Ny, Z, Fp, gap_max_mm=30.0, keep_mm=1.5):
    """A single part can have several layers (lining under the outer layer of a skirt). The body
    pushes back the one touching it; without this it pokes out through the outer one. Each vertex
    that had its own part above it in the original model (X, along its normal N) must stay that
    way: it is the outer layer that moves away, so as not to push the lining back into the skin.
    Returns (corrected Z, number of vertices moved)."""
    from mathutils import Vector
    from mathutils.kdtree import KDTree
    a = bvh(X, Fp)
    links = []
    for v in range(len(X)):
        n0 = np.linalg.norm(N[v])
        n1 = np.linalg.norm(Ny[v])
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        n = N[v] / n0
        hit, nf, f, dist = a.ray_cast(Vector(X[v] + n * 0.3 * MM), Vector(n), gap_max_mm * MM)
        if hit is None or dist < keep_mm * MM or v in Fp[f]:
            continue
        # a layer lying above is hit on its back side; hit on its front side, it is the facing
        # fabric (the other leg at the crotch, a neighbouring finger), not a layer
        if float(np.dot(nf, n)) <= 0:
            continue
        links.append((v, Fp[f], barycenter(np.asarray(hit), X[Fp[f]]), Ny[v] / n1, min(dist, keep_mm * MM)))
    # the other way round: a vertex of the outer layer under which the lining passes (coarse mesh:
    # the lining can poke through between the vertices of the outer layer)
    under = []
    for v in range(len(X)):
        n0 = np.linalg.norm(N[v])
        n1 = np.linalg.norm(Ny[v])
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        n = -N[v] / n0
        hit, nf, f, dist = a.ray_cast(Vector(X[v] + n * 0.3 * MM), Vector(n), gap_max_mm * MM)
        if hit is None or dist < keep_mm * MM or v in Fp[f]:
            continue
        if float(np.dot(nf, n)) >= 0:      # the lining underneath is hit on its front side
            continue
        under.append((v, Fp[f], barycenter(np.asarray(hit), X[Fp[f]]), Ny[v] / n1, min(dist, keep_mm * MM)))
    moves = set()
    rr = 2 * 10 * MM
    for _ in range(3):
        d0, depth = np.zeros_like(Z), np.zeros(len(Z))
        for v, tri, bc, d, guard in links:
            e = float((Z[v] - (bc[:, None] * Z[tri]).sum(0)) @ d) + guard
            if e > 0:
                for i in tri:
                    if e > depth[i]:
                        depth[i], d0[i] = e, e * d
        for v, tri, bc, d, guard in under:
            e = guard - float((Z[v] - (bc[:, None] * Z[tri]).sum(0)) @ d)
            if e > depth[v]:
                depth[v], d0[v] = e, e * d
        inside = np.where(depth > 0)[0]
        if not len(inside):
            break
        kd = KDTree(len(Z))
        for i, z in enumerate(Z):
            kd.insert(tuple(z), i)
        kd.balance()
        dlt, mag = np.zeros_like(Z), np.zeros(len(Z))
        nyu = Ny / np.maximum(np.linalg.norm(Ny, axis=1), 1e-12)[:, None]
        for i in inside:
            for _, j, dd in kd.find_range(tuple(Z[i]), rr):
                if j != i and float(d0[i] @ nyu[j]) <= 0:
                    continue
                f = (1 - dd / rr) ** 2
                if depth[i] * f > mag[j]:
                    mag[j], dlt[j] = depth[i] * f, d0[i] * f
        Z = Z + dlt
        moves.update(int(i) for i in inside)
    return Z, len(moves)


ELONGATED = 1.3             # an edge longer than this (times its scaled length) counts as stretched
LIMB_GAIN = 0.01     # stretch share a limb must gain to be fitted differently from the rest of the part
RIGID_MIN = 8            # vertices: below this, a piece has no shape to keep
RIGID_MATCH = 0.5       # share of transported vertices that must agree with a one-block placement
RIGID_SCALE = (0.85, 1.2)


def chunks(X, Fp):
    """Piece number of each vertex: faces that touch, coincident vertices merged."""
    _, w = np.unique(np.round(X, 4), axis=0, return_inverse=True)
    w = w.ravel()
    e = np.arange(int(w.max()) + 1)
    Fw = w[Fp]
    for _ in range(500):
        before = e.copy()
        m = e[Fw].min(1)
        for k in range(3):
            np.minimum.at(e, Fw[:, k], m)
        e = e[e]
        if (e == before).all():
            break
    return e[w]


LIMB_OF_TIPS = {"頭": 0, "左手首": 1, "右手首": 2, "左足首": 3, "右足首": 4, "左足首D": 3, "右足首D": 4}


def limbs(X, Fp, Is, W, src):
    """Limb of each vertex (0 trunk and head, 1-2 arms, 3-4 legs; -1 none), the same for a whole
    piece: the one that weighs most in the piece's weights, the part's own bones counted for the
    limb they are attached to."""
    import transport as TR
    bones = src["bones"]
    seg = TR._segments_os(bones)
    names_seg = list(TR.SEGMENTS)
    ch_of_seg = np.full(len(names_seg), -1)
    for c, chain in enumerate(TR.CHAINS):
        for n in chain:
            ch_of_seg[names_seg.index(n)] = c
    ch_os = np.where(seg >= 0, ch_of_seg[np.maximum(seg, 0)], -1)
    for i in np.where(seg == -2)[0]:        # hand, foot, head: the limb that carries them
        k, n = int(i), 0
        while 0 <= k < len(bones) and n < 256 and bones[k]["name"] not in LIMB_OF_TIPS:
            k, n = bones[k]["parent"], n + 1
        if 0 <= k < len(bones):
            ch_os[i] = LIMB_OF_TIPS.get(bones[k]["name"], -1)
    ch = np.where(Is >= 0, ch_os[np.clip(Is, 0, len(ch_os) - 1)], -1)
    c = chunks(X, Fp)
    out = np.full(len(X), -1)
    for z in np.unique(c):
        v = c == z
        weight = [float((W[v] * (ch[v] == m)).sum()) for m in range(len(TR.CHAINS))]
        if max(weight) > 0:
            out[v] = int(np.argmax(weight))
    return out


def chunks_rigid_bodies(X, Fp, I, W):
    """Pieces of the part (faces that touch, coincident vertices merged) whose vertices are all
    held by one and the same bone, unshared: on its source such a piece never deforms
    (backpack, buckle, spike, plate). Returns the list of vertex index arrays."""
    c = chunks(X, Fp)
    k = W.argmax(1)
    os1 = I[np.arange(len(I)), k]
    alone = W.max(1) > 0.999
    out = []
    for z in np.unique(c):
        v = np.where(c == z)[0]
        if len(v) >= RIGID_MIN and alone[v].all() and (os1[v] == os1[v[0]]).all():
            out.append(v)
    return out


def place_rigid(A, B):
    """Similarity transform (rotation, clamped scale, translation) that brings A onto B, fitted on
    the points that agree (the others set aside): returns (A placed, share of agreeing points)."""
    ok = np.ones(len(A), bool)
    for _ in range(8):
        a, b = A[ok] - A[ok].mean(0), B[ok] - B[ok].mean(0)
        U, sv, Vt = np.linalg.svd(a.T @ b)
        d = np.sign(np.linalg.det(U @ Vt)) or 1.0
        D = np.diag([1.0, 1.0, d])
        samp = min(RIGID_SCALE[1], max(RIGID_SCALE[0], (sv * np.diag(D)).sum() / max((a * a).sum(), 1e-12)))
        Pz = (A - A[ok].mean(0)) @ (samp * (U @ D @ Vt)) + B[ok].mean(0)
        r = np.linalg.norm(Pz - B, axis=1)
        ok2 = r <= max(2.5 * np.median(r), 3 * MM)
        if ok2.sum() < 4 or (ok2 == ok).all():
            break
        ok = ok2
    return Pz, float(ok.mean()), U @ D @ Vt


def weight_np(pmx):
    I = np.asarray(pmx["vertices"]["bones"], np.int64)
    W = np.asarray(pmx["vertices"]["weights"], float)
    W = np.where(I >= 0, W, 0)
    W = W / np.maximum(W.sum(1), 1e-12)[:, None]
    return I, W


def faces_mats(pmx):
    F = np.asarray(pmx["faces"], np.int64).reshape(-1, 3)
    mf = np.zeros(len(F), np.int64)
    d = 0
    for i, m in enumerate(pmx["materials"]):
        k = m["n_indices"] // 3
        mf[d:d + k] = i
        d += k
    return F, mf


SHOE_SIZES = tuple(1.0 + 0.03 * i for i in range(11))   # a shoe grows at most 30 % to fit the foot (Promeia on the house body: 1.3, nothing smaller holds the foot)
SHOE_SIZE_FREE = 1.15     # beyond this, a size is only tried when none up to it holds the foot without lowering
                          # the sole, and only taken if it holds the foot (nothing inside, sole not lowered)
SOLE_GAME = 5 * MM      # lowering the shoe by more than this onto its sole means it does not hold the foot
FOOT_FLAT = 8.0           # degrees: a foot posed less than this in its shoe (no heel) stays at rest in it
SHOE_SIZE_INSIDE = 0.0     # share of the foot part's vertices allowed inside the foot
WEAR_GAP = 0.1      # units of height per unit of shift: the smallest move wins a tie
STEM_BLENDED = 80 * MM      # above the ankle, a boot's shaft goes from sized with the foot to left as it is
                          # (over 4 cm, a shoe grown 15 % sheared its collar apart)


def _sole(Z, m, pushes, threshold, count=None):
    """Lowers the vertices m of a shoe as a whole until the foot no longer goes through its sole
    (the vertical drop that brings each vertex pushed downwards out, a few times since the nearest
    skin changes as it goes down). Only the vertices `count` (default m: the foot part, a boot's
    shaft is fitted around the leg afterwards) are checked.
    Returns (Z[m] lowered, drop, share of `count` more than `threshold` deep, margin included)."""
    count = m if count is None else count
    Z2 = Z.copy()
    dy = 0.0
    for _ in range(SOLE_PASSES):
        d0, depth = pushes(Z2, faces=False)
        n = d0 / np.maximum(np.linalg.norm(d0, axis=1), 1e-12)[:, None]
        under = (depth > 0) & (n[:, 1] < -0.5) & count
        if not under.any():
            break
        d = float((depth[under] / -n[under, 1]).max())
        Z2[m, 1] -= d
        dy += d
    _, depth = pushes(Z2, faces=False)
    return Z2[m], dy, float(((depth > threshold) & count).sum() / max(count.sum(), 1))


ADVANCES = tuple(x * 5 * MM * k for x in range(7) for k in ((1,) if x == 0 else (-1, 1)))  # up to 3 cm


SHEATH_STEP = 10 * MM        # height of the bands a boot's shaft is fitted by
SHEATH_BLENDED = 50 * MM      # a band's fit fades out over this height above and below it
SHEATH_LEG = 90 * MM      # skin this close to the leg's axis is the leg's (not the other leg's)
SHEATH_SECTORS = 12        # directions around the leg a shaft's band is widened by
SHEATH_ANGLE = np.pi / 3    # a direction's widening fades out over this angle on each side
SHEATH_PASSES = 4
SHEATH_WALL = 0.5          # vertices nearer the leg's centre than this share of their band's median
                           # are inner parts (lining, sole, straps' backs), not the shaft's wall


def _smooth(t, tb, values, weight_min=1e-6):
    """Per-band values (one row per band centre tb) spread over the heights t with a tent of
    half-width SHEATH_BLENDED, as a weighted mean (bands without a value carry weight 0)."""
    w = np.clip(1.0 - np.abs(t[:, None] - tb[None, :]) / SHEATH_BLENDED, 0.0, 1.0)
    return (w @ values) / np.maximum(w.sum(1), weight_min)[:, None]


def sheathe(Z, stem, axes, pushes, skin):
    """A boot's shaft fitted around the leg like a sleeve, band by band along the leg's axis:
    first each band is moved so that its centre is the leg's centre at that height (the source's
    legs are further apart or closer), then its wall is moved outwards from that centre by an
    offset that depends only on height and direction (SHEATH_SECTORS directions), the depth by
    which the leg goes through the wall there, spread smoothly over SHEATH_BLENDED and SHEATH_ANGLE and
    faded in from the ankle by `stem` (0 on the foot part, 1 from STEM_BLENDED above the ankle).
    Pushing the shaft vertex by vertex tore it into spikes and flaps; one widening factor per band
    ballooned an oval shaft on its long side. axes: [(vertices of one side's shaft, ankle head,
    unit axis up to the knee)]; skin: the body's skin points.
    Returns (Z, {"recentred" (m), "passes", "widened": the largest outward offset (m)})."""
    Z = Z.copy()
    centers, moved, widened, n = [], 0.0, 0.0, 0
    for m, h, u in axes:
        V = Z[m] - h
        t = V @ u
        R = V - t[:, None] * u
        nb = int(np.floor((t.max() - t.min()) / SHEATH_STEP)) + 1
        k = np.minimum(np.floor((t - t.min()) / SHEATH_STEP).astype(int), nb - 1)
        tb = t.min() + (np.arange(nb) + 0.5) * SHEATH_STEP
        Q = skin - h
        tq = Q @ u
        Rq = Q - tq[:, None] * u
        nearest = np.linalg.norm(Rq, axis=1) < SHEATH_LEG
        cj, cs, a = np.zeros((nb, 3)), np.zeros((nb, 3)), np.zeros((nb, 1))
        for b in range(nb):
            j = nearest & (np.abs(tq - tb[b]) <= SHEATH_STEP / 2)
            v = k == b
            if j.sum() >= 8 and v.any():
                cj[b] = Rq[j].mean(0)
                cs[b] = (R[v].min(0) + R[v].max(0)) / 2      # middle of the band, straps and all
                a[b] = 1.0
        if a.any():
            offset = _smooth(t, tb, (cj - cs) * a) * stem[m][:, None]
            Z[m] = Z[m] + offset
            moved = max(moved, float(np.linalg.norm(offset, axis=1).max()))
        else:
            cj[:] = 0.0
        center = np.zeros((nb, 3)) + (cj * a).sum(0) / max(a.sum(), 1.0)
        center[a[:, 0] > 0] = cj[a[:, 0] > 0]
        centers.append((m, h, u, t, k, tb, center))
    for n in range(1, SHEATH_PASSES + 1):
        _, depth = pushes(Z)
        moves = False
        for m, h, u, t, k, tb, center in centers:
            c = h + t[:, None] * u + _smooth(t, tb, center)
            R = Z[m] - c
            R = R - (R @ u)[:, None] * u
            r = np.maximum(np.linalg.norm(R, axis=1), 1e-9)
            e1 = np.cross(u, [0.0, 0.0, 1.0])
            e1 = e1 / np.linalg.norm(e1)
            a = np.arctan2(R @ np.cross(u, e1), R @ e1)
            sa = np.floor((a + np.pi) / (2 * np.pi) * SHEATH_SECTORS).astype(int) % SHEATH_SECTORS
            need = np.zeros((len(tb), SHEATH_SECTORS))
            for b in range(len(tb)):
                v = k == b
                if v.any():
                    wall = v & (r >= SHEATH_WALL * np.median(r[v]))
                    np.maximum.at(need[b], sa[wall], depth[m][wall])
            if need.max() <= 0.2 * MM:
                continue
            ac = (np.arange(SHEATH_SECTORS) + 0.5) / SHEATH_SECTORS * 2 * np.pi - np.pi
            wt = np.clip(1.0 - np.abs(t[:, None] - tb[None, :]) / SHEATH_BLENDED, 0.0, 1.0)
            da = np.abs((a[:, None] - ac[None, :] + np.pi) % (2 * np.pi) - np.pi)
            wa = np.clip(1.0 - da / SHEATH_ANGLE, 0.0, 1.0)
            d = (wt[:, :, None] * wa[:, None, :] * need[None]).max((1, 2)) * stem[m]
            if d.max() > 0.2 * MM:
                moves = True
                widened = max(widened, float(d.max()))
                Z[m] = Z[m] + R / r[:, None] * d[:, None]
        if not moves:
            break
    return Z, {"recentred": round(moved * UNIT, 4), "passes": n, "widened": round(widened * UNIT, 4)}


def put_on_shoes(Z, m, center, before, ankle_y, pushes, threshold):
    """Size and place of a shoe (vertices m of Z) for a foot of another body: the shoe is the
    source's foot, smaller or larger than this one, its toes further forward or back from the
    ankle. Among the uniform scales about the foot's centre (SHOE_SIZES) and the shifts along the
    foot (ADVANCES, `before`: horizontal forward) with which the foot no longer goes through the shoe
    once it is lowered onto its sole (_sole), the smallest size that holds the foot without
    lowering more than SOLE_GAME (a larger shoe than the foot needs looks like a clog around the
    ankle), then the shift whose bottom ends highest (the body is raised least); if none does,
    the one whose bottom ends highest: the foot rests on the sole instead of being held up by its toes against the
    front. Scaling and moving keep the shoe's shape, where pushing vertex by vertex tears it.
    Only the foot part (up to the ankle, `ankle_y`) is sized and checked; a boot's shaft fades
    from it to unscaled over STEM_BLENDED, moves with it, and is fitted around the leg afterwards.
    Returns (Z[m], {"shoe_size", "progress" (metres), "sole" (metres)}, share of each vertex sized with the foot)."""
    w = np.clip(1.0 - (Z[:, 1] - ankle_y) / STEM_BLENDED, 0.0, 1.0)
    foot = m & (w >= 1.0)
    attempts = []
    for s in SHOE_SIZES:
        if s > SHOE_SIZE_FREE + 1e-9 and not min(attempts, key=lambda e: e[:5])[2]:
            break
        k = (1.0 + w[m] * (s - 1.0))[:, None]
        for dz in ADVANCES:
            Zs = Z.copy()
            Zs[m] = center + k * (Z[m] - center) + dz * before
            Zm, dy, inside = _sole(Zs, m, pushes, threshold, count=foot)
            cost = -float(Zm[w[m] >= 1.0, 1].min()) + WEAR_GAP * abs(dz)
            too = inside > SHOE_SIZE_INSIDE
            sunk = dy > SOLE_GAME      # the foot only fits by pushing the sole down: too small
            if s > SHOE_SIZE_FREE + 1e-9 and (too or sunk):
                continue
            attempts.append((too, inside if too else 0.0, sunk, cost if sunk else s, cost, Zm,
                           {"shoe_size": round(s, 3), "progress": round(dz * UNIT, 4), "sole": round(dy * UNIT, 4)}))
    *_, Zm, info = min(attempts, key=lambda e: e[:5])
    return Zm, info, w


def deposit_output(out, C0, n0, nb_c, nr0, ch, pose):
    """Fitted on the posed body, written in the rest frame: the body's vertices and bones are those of the
    rest body C0, the part's vertices, bones, rigid bodies and joints are un-posed with their weights."""
    S = out["vertices"]
    S["pos"][:n0] = [tuple(p) for p in C0["vertices"]["pos"]]
    S["nor"][:n0] = [tuple(p) for p in C0["vertices"]["nor"]]
    for o, r in zip(out["bones"][:nb_c], C0["bones"]):
        for k in ("pos", "tail_offset", "fixed_axis", "local_axes"):
            o[k] = copy.deepcopy(r[k])
    T = PD.transforms(ch, pose)
    cho = PD.chains(out["bones"])       # rest skeleton plus the part's bones, hung under the feet
    for name in cho:
        if name in ch:
            cho[name].update({k: ch[name][k] for k in ("ha", "ht", "axis")})
    I = np.asarray(S["bones"][n0:], np.int64)
    Wt = np.asarray(S["weights"][n0:], float)
    Wt = np.where(I >= 0, Wt, 0)
    Wt = Wt / np.maximum(Wt.sum(1), 1e-12)[:, None]
    wa, wt, side = PD.weight(cho, I, Wt)
    Zp, Np = np.asarray(S["pos"][n0:], float), np.asarray(S["nor"][n0:], float)
    S["pos"][n0:] = [tuple(float(v) for v in x) for x in PD.deposit(Zp, wa, wt, side, T)]
    S["nor"][n0:] = [tuple(float(v) for v in x) for x in PD.deposit(Np, wa, wt, side, T, vectors=True)]
    tos = PD.transform_os(cho, T, len(out["bones"]))
    for i in range(nb_c, len(out["bones"])):
        R, t = tos[i]
        o = out["bones"][i]
        o["pos"] = tuple(float(x) for x in R.T @ (np.asarray(o["pos"], float) - t))
        for k in ("tail_offset", "fixed_axis"):
            if o.get(k) is not None and (k != "tail_offset" or o.get("tail_bone") is None):
                o[k] = tuple(float(x) for x in R.T @ np.asarray(o[k], float))
        if o.get("local_axes") is not None:
            o["local_axes"] = tuple(tuple(float(x) for x in R.T @ np.asarray(a, float)) for a in o["local_axes"])
    for r in out["rigid_bodies"][nr0:]:
        R, t = tos[r["bones"]] if 0 <= r["bones"] < len(tos) else (np.eye(3), np.zeros(3))
        r["pos"] = tuple(float(x) for x in R.T @ (np.asarray(r["pos"], float) - t))
        r["rot"] = euler_mmd(R.T @ rot_mmd(r["rot"]))
    for j in out["joints"]:
        if j["a"] < nr0:
            continue
        b = out["rigid_bodies"][j["a"]]["bones"]
        R, t = tos[b] if 0 <= b < len(tos) else (np.eye(3), np.zeros(3))
        j["pos"] = tuple(float(x) for x in R.T @ (np.asarray(j["pos"], float) - t))
        j["rot"] = euler_mmd(R.T @ rot_mmd(j["rot"]))


def wear(part_dir, body, output, name=None, margin_mm=3.0, skin="remove", surface="all", foot=None):
    """pied (shoes with a sole only): "search" finds the pose of the body's feet inside the shoe,
    a pose dict (foot.py) reuses one; the part is fitted on the posed body and written back in the
    rest frame (res["pied"]: the pose and the height to raise the model by). None: flat foot.
    "flat": flat foot, but the shoe's foot part stays as the bones carry it (a heel sinks into the
    ground and the foot goes through the sole, the shoe stays whole); only a boot's shaft is
    pushed out of the leg."""
    card, path_part = load_part(part_dir)
    name = name or "%s-%s" % (card["character"], card["id"])
    P = PMXL.read(path_part)
    C = PMXL.read(body)
    src_complete = PMXL.read(card["source"]["pmx"])
    # part bones -> bones of the full source (reliable tails)
    name_src = {o["name"]: i for i, o in enumerate(src_complete["bones"])}
    s_glob = min(1.15, max(0.87, height(C) / height(src_complete)))
    po = [name_src.get(o["name"], -1) for o in P["bones"]]          # part bone -> source bone
    own_ones = set(card.get("own_bones") or [])
    X = np.asarray(P["vertices"]["pos"], float)
    N = np.asarray(P["vertices"]["nor"], float)
    I, W = weight_np(P)
    Is = np.where(I >= 0, np.asarray(po)[np.maximum(I, 0)], -1)
    # ---------------------------------------------------------------- 0. foot posed in the shoe
    C0, pose_foot, ch_foot = None, None, None
    flat = (foot == "flat" and card.get("type") in TYPES_FOOT and a_sole(P)
            and bool(PD.chains(C["bones"])) and bool(PD.chains(src_complete["bones"])))
    if flat:
        ch_foot = PD.chains(C["bones"])
    elif (foot not in (None, "flat") and card.get("type") in TYPES_FOOT and a_sole(P)
            and PD.chains(C["bones"]) and PD.chains(src_complete["bones"])):
        if isinstance(foot, dict):
            pose_foot = {k: {"ankle": v["ankle"], "toes": v["toes"]}
                         for k, v in foot.items() if isinstance(v, dict)}
            ch_foot = PD.chains(C["bones"])
        else:
            sys.path.insert(0, os.path.join(ROOT, "workshop"))
            import rays as rays
            Y0 = Registration(src_complete, C, s_glob, feet_rigid_bodies=True, shoe=True).points(Is, W, X)
            pose_foot, ch_foot = PD.search(C, Y0, faces_mats(P)[0], UNIT, rays)
        C0, C = C, copy.deepcopy(C)
        if all(max(abs(v["ankle"]), abs(v["toes"])) < FOOT_FLAT for v in pose_foot.values()):
            # a flat shoe: the foot already sits in it and stays at rest, but the shoe is still placed
            # in one block (pushing it out of the foot vertex by vertex tears it to shards)
            pose_foot = {k: {"ankle": 0.0, "toes": 0.0} for k in pose_foot}
        else:
            PD.place_pmx(C, pose_foot, ch=ch_foot)
    rc = Registration(src_complete, C, s_glob, feet_rigid_bodies=pose_foot is not None,
                  shoe=card.get("type") in TYPES_FOOT and a_sole(P))
    # ---------------------------------------------------------------- 1-2. LBS
    Y = rc.points(Is, W, X)
    Ny = rc.vectors(Is, W, N)
    # ---------------------------------------------------------------- surface underneath
    Fc, mfc = faces_mats(C)
    Pc = np.asarray(C["vertices"]["pos"], float)
    names_c = [m["name"] for m in C["materials"]]
    # surface="skin": the part is placed on the skin alone, as if it were worn alone;
    # the outfit's layers are then adjusted together (tools/workshop/outfit.py)
    keep_surface = (lambda n: bool(MD.SKIN.match(n))) if surface == "skin" else (lambda n: not MD.NON_GARMENT.match(n))
    under = np.array([keep_surface(n) and C["materials"][i]["n_indices"] > 0
                        for i, n in enumerate(names_c)])[mfc]
    fdes = np.where(under)[0]
    nfc = normals_faces(Pc, Fc)
    sg = sign_outside(Pc, Fc[fdes], nfc[fdes])
    nfc = nfc * sg
    tree = bvh(Pc, Fc[fdes])
    margin = margin_mm * MM
    Fp, mfp = faces_mats(P)

    background_s, background_f, background_links = lids(X, Fp, src_complete["bones"])

    def pushes(Z, faces=True):
        d0 = np.zeros_like(Z)
        depth = np.zeros(len(Z))
        for i, z in enumerate(Z):
            if background_s[i]:               # centre of a cap: hidden in the body, as on its source
                continue
            loc, _, k, dist = tree.find_nearest(tuple(z), 4 * margin + 0.4)
            if loc is None:
                continue
            n = nfc[fdes[k]]
            s = float((z - np.asarray(loc)) @ n)
            if s < margin and dist < 0.4:
                d0[i] = (margin - s) * n
                depth[i] = margin - s
        # inside the faces: centre and edge midpoints (a wide fold can pass under the skin
        # between vertices that are outside); a point inside pushes the three vertices of its face
        T = Z[Fp]
        samp = [T.mean(1), (T[:, 0] + T[:, 1]) / 2, (T[:, 1] + T[:, 2]) / 2, (T[:, 2] + T[:, 0]) / 2]
        for E in (samp if faces else []):
            for k, z in enumerate(E):
                if background_f[k]:
                    continue
                loc, _, kk, dist = tree.find_nearest(tuple(z), 4 * margin + 0.4)
                if loc is None or dist >= 0.4:
                    continue
                n = nfc[fdes[kk]]
                sv = float((z - np.asarray(loc)) @ n)
                # between vertices that are outside, a face only sinks by its sag (a few mm):
                # any deeper, it is a wrongly oriented body normal, so it is ignored
                if margin - 15 * MM < sv < margin:
                    for i in Fp[k]:
                        if margin - sv > depth[i]:
                            d0[i] = (margin - sv) * n
                            depth[i] = margin - sv
        return d0, depth

    from mathutils.kdtree import KDTree
    stats = {}
    hook = None
    if card["class"] == "accessory":
        hook = hook_loop(card, P, src_complete, Y, s_glob, C, Fc, mfc, fdes, tree, margin, on_fabric=surface != "skin")
    def finish(hook, Ny):
        stats = {}
        Ny = Ny.copy()
        if hook == "bones":        # the part follows the target's bones, as it follows those of its source
            Z = Y.copy()
        elif hook is not None:
            Z, rigid, stats["loop"] = hook
            Ny = N @ rigid[2].T
        elif pose_foot is not None:
            # a shoe is rigid: it stays as placed around the posed foot, lowered as a whole until
            # the foot no longer goes through its sole. Pushing the sole vertex by vertex drags the
            # whole sole down pass after pass (a solid platform), and tears it
            Z = Y.copy()
            wa_c, wt_c, side_c = PD.weight_pmx(C, ch_foot)
            stats["on_foot"] = {}
            stem = np.zeros(len(Z))
            for k_c, (name_c, _, sg_c) in enumerate(PD.SIDES):
                # the side is the source's: a foot set close to the other one has its inner edge
                # at x = 0 once worn, and a vertex across it would be moved with the other shoe
                m = sg_c * X[:, 0] > 0
                foot_c = Pc[(side_c == k_c) & (wa_c + wt_c > 0.5)]
                if not m.any() or not len(foot_c) or name_c not in ch_foot:
                    continue
                av = ch_foot[name_c]["ht"] - ch_foot[name_c]["ha"]
                av[1] = 0.0
                Z[m], stats["on_foot"][name_c], w_c = put_on_shoes(Z, m, foot_c.mean(0), av / np.linalg.norm(av),
                                                              ch_foot[name_c]["ha"][1], pushes, margin + 1 * MM)
                stem[m] = 1.0 - w_c[m]
        else:
            import transport as TR
            tight = card.get("type") in TIGHT or all(re.match(r"体", m["name"]) for m in P["materials"])
            Z, stats["transport"] = TR.transport(X, Is, W, src_complete, C, own_ones, tree, Y, s_glob, Fp, tight, sg, faces_target=Fc[fdes])
            if flat:
                # the foot part stays as carried; a boot's shaft is pushed, fading in above the ankle
                stem = np.zeros(len(Z))
                for name_c, _, sg_c in PD.SIDES:
                    m = sg_c * X[:, 0] > 0
                    if name_c in ch_foot:
                        stem[m] = np.clip((Y[m, 1] - ch_foot[name_c]["ha"][1]) / STEM_BLENDED, 0.0, 1.0)
        # until nothing is left inside (at most PASSES_MAX); what remains is counted, not hidden
        stats["stays_inside"] = 0
        Z_before = Z.copy()
        # a shoe's foot part was placed as a whole (put_on_shoes, or carried flat) and stays so; a boot's
        # shaft is widened around the leg like a sleeve (sheathe), never pushed vertex by vertex
        on_foot = (pose_foot is not None or flat) and hook is None
        if on_foot:
            axes = []
            idx_c = {o["name"]: i for i, o in enumerate(C["bones"])}
            for name_c, pre_c, sg_c in PD.SIDES:
                m = (sg_c * X[:, 0] > 0) & (stem > 0)
                if name_c not in ch_foot or not m.any():
                    continue
                h = ch_foot[name_c]["ha"]
                g = idx_c.get(pre_c + "ひざ")
                u = (np.asarray(C["bones"][g]["pos"], float) - h) if g is not None else np.array([0.0, 1.0, 0.0])
                axes.append((m, h, u / np.linalg.norm(u)))
            Z, stats["sheath"] = sheathe(Z, stem, axes, pushes, Pc[np.unique(Fc[fdes])])
            stats["stays_inside"] = int(((pushes(Z)[1] > 1 * MM) & (stem > 0)).sum())
        for passed in range(0 if on_foot else PASSES_MAX):
            d0, depth = pushes(Z)
            inside = np.where(depth > 0)[0]
            stats["stays_inside"] = int((depth > 1 * MM).sum())
            stats["pass_%d" % passed] = {"vertices": int(len(inside)),
                                         "depth_max_mm": round(float(depth.max() / MM), 1) if len(inside) else 0}
            if not len(inside):
                break
            kd = KDTree(len(Z))
            for i, z in enumerate(Z):
                kd.insert(tuple(z), i)
            kd.balance()
            rr = 3 * 10 * MM if passed == 0 else 1.0 * 10 * MM
            dlt = np.zeros_like(Z)
            mag = np.zeros(len(Z))
            nyu = Ny / np.maximum(np.linalg.norm(Ny, axis=1), 1e-12)[:, None]
            for i in inside:
                for _, j, d in kd.find_range(tuple(Z[i]), rr):
                    # a vertex's push only drags neighbours facing the same way: between the
                    # thighs or the fingers, the facing fabric is close and moves out the opposite way
                    if j != i and float(d0[i] @ nyu[j]) <= 0:
                        continue
                    f = (1 - d / rr) ** 2
                    m = depth[i] * f
                    if m > mag[j]:
                        mag[j] = m
                        dlt[j] = d0[i] * f
            dlt[inside] = np.where((np.linalg.norm(d0[inside], axis=1) > mag[inside])[:, None], d0[inside], dlt[inside])
            Z = Z + dlt
        for center, girth in background_links:         # the bottom follows the rim it closes, without leaving the body
            Z[center] = Z_before[center] + (Z[girth] - Z_before[girth]).mean(0)
        if hook is None:
            Z, stats["inner_layers"] = inner_layers(X, N, Ny, Z, Fp)
            # what is rigid on the source stays rigid: the transport places each vertex according to
            # the body and twists a backpack or a buckle like fabric. Each piece held by a single bone
            # takes back its shape, at the position and orientation the transport gave it, then
            # moves out of the skin as one block; if it ends up deeper than before, the transport is kept.
            def sunk(Zc):
                worst, dv = 0.0, np.zeros(3)
                for z in Zc:
                    loc, _, k, dist = tree.find_nearest(tuple(z), 4 * margin + 0.4)
                    if loc is None or dist >= 0.4:
                        continue
                    n = nfc[fdes[k]]
                    pr = margin - float((z - np.asarray(loc)) @ n)
                    if pr > worst:
                        worst, dv = pr, pr * n
                return worst, dv

            poses = 0
            # a shoe was not transported: its foot part is already one block, its shaft kept its shape
            for v in ([] if on_foot else chunks_rigid_bodies(X, Fp, I, W)):
                Pz, match_, Rz = place_rigid(X[v], Z[v])
                if match_ < RIGID_MATCH:
                    continue
                before = sunk(Z[v])[0]
                for _ in range(6):
                    worst, dv = sunk(Pz)
                    if worst <= 0:
                        break
                    Pz = Pz + dv
                if sunk(Pz)[0] > before + 1 * MM:
                    continue
                Z[v] = Pz
                Ny[v] = N[v] @ Rz
                poses += len(v)
            stats["vertex_rigid_bodies"] = int(poses)
        return Z, Ny, stats

    # what does not clearly surround something: placed as one block (tail, wings, far from the body),
    # transported like the rest (harness, tie, bag against the body) or adjusted as a ring; none
    # works everywhere, so the one whose edges stretch least once the part is out of the skin
    # is kept
    attempts = [("loop", hook)]
    if hook is not None:
        block = hook_loop(card, P, src_complete, Y, s_glob, C, Fc, mfc, fdes, tree, margin, on_fabric=surface != "skin", adjust=False)
        attempts = [("block", block), ("transport", None), ("loop", hook)]
        if hook[2]["loop"]:           # on a tie, what clearly surrounds something stays a ring
            attempts = [attempts[2], attempts[0], attempts[1]]
    done = []
    for name_e, a in attempts:
        Z, Nya, st = finish(a, Ny)
        E_ = np.concatenate([Fp[:, [0, 1]], Fp[:, [1, 2]], Fp[:, [2, 0]]])
        l0 = s_glob * np.linalg.norm(X[E_[:, 0]] - X[E_[:, 1]], axis=1)
        l1 = np.linalg.norm(Z[E_[:, 0]] - Z[E_[:, 1]], axis=1)
        st["elongation"] = round(float(np.maximum(l1 - ELONGATED * l0, 0).sum() / max(l0.sum(), 1e-9)), 4)
        done.append((st["elongation"], len(done), Z, Nya, st, a, name_e))
    _, _, Z, Ny, st, hook, name_e = min(done, key=lambda f: f[:2])
    if len(done) > 1:
        st["attempts"] = {f[6]: f[0] for f in done}
        # a part attached to several limbs (a tail shipped with its arm ornaments) is not fitted
        # in a single way: the tail is placed as one block, what is attached to the arms follows them.
        # Each limb takes, among the chosen placement, the other family (block or ring / transport)
        # and the per-bone placement (what is attached to an arm follows it as on its source), the one
        # that stretches its edges least.
        mb = limbs(X, Fp, Is, W, src_complete)
        if len(np.unique(mb[mb >= 0])) > 1:
            blocks = [f for f in done if f[5] is not None]
            others = [min(blocks, key=lambda f: f[:2]) if hook is None else next(f for f in done if f[5] is None)]
            Zo, Nyo, _ = finish("bones", Ny)
            others.append((0, 0, Zo, Nyo, None, None, "bones"))
            Z, Ny = Z.copy(), Ny.copy()
            em = mb[E_[:, 0]]
            per = {}
            for m in np.unique(mb[mb >= 0]):
                k, v = em == m, mb == m

                def elongated(Zc):
                    l1 = np.linalg.norm(Zc[E_[k, 0]] - Zc[E_[k, 1]], axis=1)
                    return float(np.maximum(l1 - ELONGATED * l0[k], 0).sum() / max(l0[k].sum(), 1e-9))
                here = elongated(Z)
                per[str(int(m))] = {name_e: round(here, 4)}
                for f in others:
                    elong = elongated(f[2])
                    per[str(int(m))][f[6]] = round(elong, 4)
                    if elong < here - LIMB_GAIN or (f[6] == "bones" and elong < here):     # roughly equal: what is attached to a limb follows its bones
                        Z[v], Ny[v], here = f[2][v], f[3][v], elong
                        per[str(int(m))]["taken"] = f[6]
            if any("taken" in x for x in per.values()):
                st["per_limb"] = per
                if hook is None and any(x.get("taken") == others[0][6] for x in per.values()):
                    hook = others[0][5]        # a limb placed as one block: its own bones follow that block
    stats.update(st)
    stats["extent_mm"] = round(float(np.linalg.norm(Z.max(0) - Z.min(0)) / MM), 1)
    # what the part has underneath on its source (body or underlying garment, excluding face, hair
    # and the part itself), with its weights mapped to the target's bones
    Ps_ = np.asarray(src_complete["vertices"]["pos"], float)
    Fs_, mfs_ = faces_mats(src_complete)
    mats_p = {m["name"] for m in P["materials"]}
    gs_ = np.array([not MD.NON_GARMENT.match(m["name"]) and m["name"] not in mats_p
                    for m in src_complete["materials"]])[mfs_]
    Fs_ = Fs_[gs_]
    tree_src = bvh(Ps_, Fs_) if len(Fs_) else None
    stats["under_skin"] = under_skin(Z[~background_s], tree, MM, sg)
    stats["lids"] = int(background_f.sum())
    stats["rigid_pose"] = hook is not None
    # a shoe's foot part kept in one block (put_on_shoes, or carried flat) holds the foot inside it by nature
    stats["foot_as_block"] = (pose_foot is not None or flat) and hook is None
    stats["shift_mm"] = {"p50": round(float(np.median(np.linalg.norm(Z - Y, axis=1)) / MM), 1),
                               "p90": round(float(np.percentile(np.linalg.norm(Z - Y, axis=1), 90) / MM), 1),
                               "max": round(float(np.linalg.norm(Z - Y, axis=1).max() / MM), 1)}
    # ---------------------------------------------------------------- 3. output bones
    out = copy.deepcopy(C)
    nb_c = len(out["bones"])
    name_target = {o["name"]: i for i, o in enumerate(out["bones"])}
    map_ = {}                    # part bone -> output bone
    added = []
    for i, o in enumerate(P["bones"]):
        if o["name"] in name_target and o["name"] not in own_ones:
            map_[i] = name_target[o["name"]]
    names_taken = set(name_target)
    for i, o in enumerate(P["bones"]):
        if i in map_:
            continue
        map_[i] = nb_c + len(added)
        added.append(i)
    for i in added:
        o = copy.deepcopy(P["bones"][i])
        b = po[i]
        if hook is not None:
            Rm, tr, _ = hook[1]
            o["pos"] = tuple(float(v) for v in Rm @ np.asarray(o["pos"], float) + tr)
            M3 = Rm
        elif b >= 0:
            o["pos"] = tuple(float(v) for v in rc.point(b, o["pos"]))
            M3 = rc.R[b] @ rc.A[b]
        else:
            M3 = np.eye(3)
        nm = o["name"]
        while nm in names_taken:
            nm += "'"
        names_taken.add(nm)
        o["name"] = nm
        o["parent"] = map_.get(o["parent"], -1) if o["parent"] >= 0 else -1
        if o["tail_bone"] is not None:
            o["tail_bone"] = map_.get(o["tail_bone"], -1) if o["tail_bone"] >= 0 else -1
        elif o["tail_offset"] is not None:
            o["tail_offset"] = tuple(float(v) for v in M3 @ np.asarray(o["tail_offset"], float))
        if o["addition"]:
            o["addition"]["source"] = map_.get(o["addition"]["source"], -1)
            if o["addition"]["source"] < 0:
                o["addition"] = None
                o["flags"] &= ~0x0300
        if o["ik"]:
            o["ik"]["target"] = map_.get(o["ik"]["target"], -1)
            for l in o["ik"]["links"]:
                l["bones"] = map_.get(l["bones"], -1)
        if o.get("fixed_axis") is not None and b >= 0:
            v = rc.R[b] @ np.asarray(o["fixed_axis"], float)
            o["fixed_axis"] = tuple(float(x) for x in v / max(np.linalg.norm(v), 1e-12))
        if o.get("local_axes") is not None and b >= 0:
            o["local_axes"] = tuple(tuple(float(x) for x in rc.R[b] @ np.asarray(a, float)) for a in o["local_axes"])
        out["bones"].append(o)
    # ---------------------------------------------------------------- 4. weights
    Io = np.where(I >= 0, np.vectorize(lambda b: map_.get(int(b), -1))(np.maximum(I, 0)), -1)
    Io[I < 0] = -1
    Wo = W.copy()
    body_out = np.zeros(len(out["bones"]), bool)
    body_out[:nb_c] = True
    for i in added:      # source body bones missing from the target: counted as body
        if P["bones"][i]["name"] not in own_ones:
            body_out[map_[i]] = True
    Ic, Wc = weight_np(C)
    Isrc, Wsrc = weight_np(src_complete)
    to_c = np.array([name_target.get(o["name"], -1) for o in src_complete["bones"]])

    def blend(arb, z, T, Pt, It, Wt, reach, to=None, under=False):
        if arb is None:
            return None
        loc, nrm, k, dist = arb.find_nearest(tuple(z), reach)
        if loc is None:
            return None
        if under and dist > UNDER_NEAR:
            # "underneath" = along the surface's axis. A nearest point reached at a slant is the
            # edge of a hole (skin removed under the garment) or the neighbouring skin: nothing underneath.
            if abs(float(np.dot(np.asarray(z) - np.asarray(loc), nrm))) < UNDER_COS * dist:
                return None
        f = T[k]
        cw = {}
        for a_, t in zip(f, barycenter(np.asarray(loc), Pt[f])):
            for b_, w in zip(It[a_], Wt[a_]):
                if b_ >= 0 and w > 0:
                    b_ = int(b_) if to is None else int(to[b_])
                    if b_ >= 0:
                        cw[b_] = cw.get(b_, 0) + t * w
        tot = sum(cw.values())
        return {b_: w / tot for b_, w in cw.items()} if tot > 1e-6 else None

    resumed = fixed = 0
    for v in range(len(Z)):
        sw = blend(tree_src, X[v], Fs_, Ps_, Isrc, Wsrc, 6 * 10 * MM, to_c, under=True)
        # with nothing known underneath on the source, the target skin only counts up close; with it,
        # the target − source difference stays meaningful far from the body (a flared panel that followed the chest)
        cw = blend(tree, Z[v], Fc[fdes], Pc, Ic, Wc, (REACH_FAR if sw is not None else 3 * 10) * MM)
        if cw is None:
            continue
        iv, wv = Io[v], Wo[v]
        own = {int(b): float(w) for b, w in zip(iv, wv) if b >= 0 and w > 0 and not body_out[b]}
        part_body = 1 - sum(own.values())
        if part_body < 0.05:
            continue
        # the garment does not necessarily follow the nearest skin (a panel hanging from the chest
        # next to the arm): its original weights are kept, corrected by the difference between the
        # target skin and what it had underneath on its source. Nothing underneath on the source (skin
        # removed under a closed garment): the garment stood in for the body there, it takes the target skin.
        if sw is not None:
            g = {int(b): float(w) / part_body for b, w in zip(iv, wv) if b >= 0 and w > 0 and body_out[b]}
            cw = {b: max(0.0, g.get(b, 0) + cw.get(b, 0) - sw.get(b, 0)) for b in set(g) | set(cw) | set(sw)}
            fixed += 1
        lst = sorted(own.items(), key=lambda x: -x[1])[:4]
        rest = 4 - len(lst)
        cs = [x for x in sorted(cw.items(), key=lambda x: -x[1])[:rest] if x[1] > 0]
        tot = sum(w for _, w in cs)
        if not cs or tot <= 0:
            continue
        lst += [(b, part_body * w / tot) for b, w in cs]
        lst += [(-1, 0.0)] * (4 - len(lst))
        Io[v] = [b for b, _ in lst]
        Wo[v] = [w for _, w in lst]
        resumed += 1
    stats["weights_from_body"] = resumed
    stats["weights_fixed_by_source"] = fixed
    # ---------------------------------------------------------------- vertices / faces / materials
    S = out["vertices"]
    n0 = len(S["pos"])
    k_add = out["n_uv_add"]
    for v in range(len(Z)):
        S["pos"].append(tuple(float(x) for x in Z[v]))
        S["nor"].append(tuple(float(x) for x in Ny[v]))
        S["uv"].append(P["vertices"]["uv"][v])
        ua = list(P["vertices"]["uv_add"][v]) if P["n_uv_add"] else []
        S["uv_add"].append(tuple((ua + [(0.0, 0.0, 0.0, 0.0)] * k_add)[:k_add]))
        nz = int((Wo[v] > 0).sum())
        if nz <= 1:
            b = int(Io[v][np.argmax(Wo[v])])
            S["type"].append(0)
            S["bones"].append((b, -1, -1, -1))
            S["weights"].append((1.0, 0.0, 0.0, 0.0))
        else:
            S["type"].append(2)
            S["bones"].append(tuple(int(b) if b >= 0 else 0 for b in Io[v]))
            S["weights"].append(tuple(float(w) for w in Wo[v]))
        S["sdef"].append(None)
        S["contour"].append(P["vertices"]["contour"][v])
    # textures: everything in sortie/tex (hard links), relative paths
    os.makedirs(os.path.join(output, "tex"), exist_ok=True)
    tex_out, tmap = [], {}

    def link_tex(path_rel, base_dir):
        src = os.path.normpath(os.path.join(base_dir, path_rel.replace("\\", "/")))
        if src in tmap:
            return tmap[src]
        b = os.path.basename(src)
        names = {os.path.basename(t.replace("\\", "/")) for t in tex_out}
        root, ext = os.path.splitext(b)
        k = 1
        while b in names:
            b = "%s_%d%s" % (root, k, ext)
            k += 1
        if os.path.exists(src):
            dst = os.path.join(output, "tex", b)
            if not os.path.exists(dst):
                try:
                    os.link(src, dst + ".tmp")
                except OSError:
                    import shutil
                    shutil.copyfile(src, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
        tmap[src] = len(tex_out)
        tex_out.append("tex\\" + b)
        return tmap[src]
    dir_c, dir_p = os.path.dirname(os.path.abspath(body)), os.path.dirname(os.path.abspath(path_part))

    def retex(m, textures, base):
        for k in ("texture", "sphere"):
            if 0 <= m[k] < len(textures):
                m[k] = link_tex(textures[m[k]], base)
            else:
                m[k] = -1
        if m.get("toon_sharing", 1) == 0:
            m["toon"] = link_tex(textures[m["toon"]], base) if 0 <= m["toon"] < len(textures) else -1
    for m in out["materials"]:
        retex(m, C["textures"], dir_c)
    names_mats = {m["name"] for m in out["materials"]}
    mats_part = []
    for m in P["materials"]:
        m = copy.deepcopy(m)
        retex(m, P["textures"], dir_p)
        nm = m["name"]
        while nm in names_mats:
            nm += "'"
        names_mats.add(nm)
        m["name"] = nm
        mats_part.append(m)
    out["textures"] = tex_out
    # ---------------------------------------------------------------- 5. hidden skin
    Fcorps = [tuple(int(x) for x in f) for f in Fc]
    keep_c = np.ones(len(Fc), bool)
    if card["class"] == "garment" and skin in ("remove", "note"):
        # areas held by the body (not the physics panels)
        own_part = np.array([sum(w for b, w in zip(Io[v], Wo[v]) if b >= 0 and not body_out[b]) for v in range(len(Z))])
        fp_outfit = np.where(own_part[Fp].max(1) < 0.5)[0]
        if len(fp_outfit):
            tree_p = bvh(Z, Fp[fp_outfit])
            roles_skin = np.array([bool(MD.SKIN.match(n)) for n in names_c])[mfc]
            for k in np.where(roles_skin)[0]:
                f = Fc[k]
                n = nfc[k]
                pts = list(Pc[f]) + [Pc[f].mean(0)]
                if all(tree_p.ray_cast(tuple(p + n * 1e-4), tuple(n), 50 * MM)[0] is not None for p in pts):
                    keep_c[k] = False
    stats["skin_faces_removed"] = int((~keep_c).sum())
    # skin="note": the skin stays whole (the next part is placed on it), the hidden faces are returned
    hidden_skin = [int(k) for k in np.where(~keep_c)[0]]
    if skin == "note":
        keep_c[:] = True
    # body faces kept, per material
    Fout = []
    for i, m in enumerate(out["materials"]):
        sel = (mfc == i) & keep_c
        m["n_indices"] = int(3 * sel.sum())
        Fout.extend(Fcorps[k] for k in np.where(sel)[0])
    for i, m in enumerate(mats_part):
        sel = mfp == i
        Fout.extend(tuple(int(x) + n0 for x in f) for f in Fp[sel])
        out["materials"].append(m)
    out["faces"] = Fout
    # ---------------------------------------------------------------- 6. rigid bodies and joints
    nr0 = len(out["rigid_bodies"])
    name_rig = {r["name"] for r in out["rigid_bodies"]}
    for r in P["rigid_bodies"]:
        r = copy.deepcopy(r)
        b = po[r["bones"]] if 0 <= r["bones"] < len(po) else -1
        if hook is not None and r["bones"] in added:
            Rm, tr, Qa = hook[1]
            r["pos"] = tuple(float(x) for x in Rm @ np.asarray(r["pos"], float) + tr)
            r["rot"] = euler_mmd(Qa @ rot_mmd(r["rot"]))
        elif b >= 0:
            r["pos"] = tuple(float(x) for x in rc.point(b, r["pos"]))
            r["rot"] = euler_mmd(rc.R[b] @ rot_mmd(r["rot"]))
        r["bones"] = map_.get(r["bones"], -1)
        while r["name"] in name_rig:
            r["name"] += "'"
        name_rig.add(r["name"])
        out["rigid_bodies"].append(r)
    for j in P["joints"]:
        j = copy.deepcopy(j)
        ra = P["rigid_bodies"][j["a"]]
        b = po[ra["bones"]] if 0 <= ra["bones"] < len(po) else -1
        if hook is not None and ra["bones"] in added:
            Rm, tr, Qa = hook[1]
            j["pos"] = tuple(float(x) for x in Rm @ np.asarray(j["pos"], float) + tr)
            j["rot"] = euler_mmd(Qa @ rot_mmd(j["rot"]))
        elif b >= 0:
            j["pos"] = tuple(float(x) for x in rc.point(b, j["pos"]))
            j["rot"] = euler_mmd(rc.R[b] @ rot_mmd(j["rot"]))
        j["a"], j["b"] = j["a"] + nr0, j["b"] + nr0
        out["joints"].append(j)
    # display frame for the added bones
    if added:
        out["frames"].append({"name": card["id"][:15], "name_en": card["id"][:15], "special": 0,
                              "elements": [(0, map_[i]) for i in added]})
    if pose_foot is not None:
        # the shoe's bottom brought to where the bare body stands (its lowest point, not y = 0:
        # a body may stand a little above its origin)
        ground = float(np.asarray(C0["vertices"]["pos"], float)[:, 1].min())
        lift = (ground - float(Z[:, 1].min())) * UNIT
        stats["foot"] = {**pose_foot, "lift": round(lift, 4)}
        deposit_output(out, C0, n0, nb_c, nr0, ch_foot, pose_foot)
    out["name"] = name
    out["comment"] = "%s + %s from %s (wear.py)" % (C.get("name", ""), card["id"], card["character"])
    import pmx_write as pmx_write
    os.makedirs(output, exist_ok=True)
    path = os.path.join(output, name + ".pmx")
    pmx_write.write(out, path + ".tmp.pmx")
    os.replace(path + ".tmp.pmx", path)
    res = {"part_name": card["character"] + "/" + card["id"], "class": card["class"],
           "type": card.get("type"), "sort": card.get("sort"),
           "measure_type": TYPES_MEASURE.get(card.get("type"), "loose_top"),
           "body": os.path.abspath(body), "pmx": path, "mats": [m["name"] for m in mats_part],
           "global_scale": round(s_glob, 3), "registered_bones": int(rc.direct.sum()),
           "added_bones": len(added), "angles_deg_max": max(rc.angles.values()) if rc.angles else 0,
           "rigid_bodies": len(P["rigid_bodies"]), "joints": len(P["joints"]), "margin_mm": margin_mm, **stats}
    with open(os.path.join(output, "portage.json.tmp"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    os.replace(os.path.join(output, "portage.json.tmp"), os.path.join(output, "fitting.json"))
    log("fitted", name, json.dumps(stats, ensure_ascii=False))
    res["hidden_skin"] = hidden_skin
    return res


def barycenter(p, T):
    a, b, c = T
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-20:
        return np.array([1 / 3, 1 / 3, 1 / 3])
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bc = np.clip(np.array([1 - v - w, v, w]), 0, 1)
    return bc / bc.sum()


# ====================================================================== render

def render(res, path_png):
    import bpy
    import importlib.util
    import tempfile
    import contact_sheet as PL
    # the garment library, loaded by file: tools/head has one with the same name, earlier in sys.path
    sp = importlib.util.spec_from_file_location("library_clothes", os.path.join(HERE, "library.py"))
    BI = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(BI)
    tmp = tempfile.mkdtemp(prefix="rendus-porter-")
    M = MD.Model(os.path.basename(res["pmx"]), path=res["pmx"])
    PL.empty()
    BI._bpy_scene()
    part_name = np.isin(M.mf, [i for i, n in enumerate(M.names_mat) if n in res["mats"]])
    all_ = np.where(~M.faces_role(["empty"]))[0]
    PL.obj(M, all_, "all", {})
    lo, hi = M.pos.min(0), M.pos.max(0)
    vs = np.unique(M.F[part_name])
    plo, phi = M.pos[vs].min(0) - 0.05, M.pos[vs].max(0) + 0.05
    line = []
    for ang, nm, (a, b) in ((0, "face", (lo, hi)), (180, "back", (lo, hi)), (90, "profile", (lo, hi)),
                            (0, "close front", (plo, phi)), (180, "close back", (plo, phi)), (35, "close 3/4", (plo, phi))):
        f = os.path.join(tmp, "%s_%d_%s.png" % (os.path.basename(res["pmx"]), ang, nm.replace(" ", "_").replace("/", "")))
        BI._render(a, b, ang, f, 300, 480, text=nm if ang or nm != "face" else res["part_name"])
        line.append(f)
    BI._glue([line], path_png)


def main(argv):
    val = lambda o, d=None: argv[argv.index(o) + 1] if o in argv else d  # noqa: E731
    res = wear(val("--part"), val("--body"), val("--output"), val("--name"),
                 float(val("--margin-mm", 3.0)), val("--skin", "remove"))
    if "--render" in argv:
        render(res, os.path.join(val("--output"), "render.png"))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(a)

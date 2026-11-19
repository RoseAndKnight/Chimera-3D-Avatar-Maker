"""Shared primitives P-01 to P-11 (method §2.3).

Interfaces: every function takes numpy arrays (Blender frame,
metres) and returns arrays or plain dicts. Only P-07 and D4/D5 of P-09
need `mathutils` (Blender); the rest runs with numpy alone.

Usual input data (see data.py):
  pos (n, 3)         positions of `welded` (or of `brut`)
  faces (m, 3)       triangles
  edges (e, 2)      unique edges of `welded`
  sq                 Skeleton (names, heads, parents, additions)
"""
import json
import os
import re
import sys
import unicodedata

import numpy as np

import kinematics as kin

HERE = os.path.dirname(os.path.abspath(__file__))

# ======================================================================= P-01

def nfkc(name):
    return unicodedata.normalize("NFKC", name)


_SIDE = [
    (re.compile(r"^左(.+)$"), "L"), (re.compile(r"^右(.+)$"), "R"),
    (re.compile(r"^(.+)左$"), "L"), (re.compile(r"^(.+)右$"), "R"),
    (re.compile(r"^(.+)[._]L$"), "L"), (re.compile(r"^(.+)[._]R$"), "R"),
    (re.compile(r"^L[._](.+)$"), "L"), (re.compile(r"^R[._](.+)$"), "R"),
]

_SIDE_MIDDLE = re.compile(r"^(.+?)_([LR])_(.+)$")


def separer_side(name):
    """'左腕' -> ('腕', 'L') ; '腰キャンセル右' -> ('腰キャンセル', 'R') ; '頭' -> ('頭', None)."""
    n = nfkc(name)
    for rx, c in _SIDE:
        m = rx.match(n)
        if m:
            return m.group(1), c
    # side in the middle (Ctr_/Skn_ bones of Unity ports: Ctr_L_Hair_01 -> Ctr_Hair_01, L)
    m = _SIDE_MIDDLE.match(n)
    if m:
        return m.group(1) + "_" + m.group(3), m.group(2)
    return n, None


def identifier(name):
    """Normalized identifier of §6.1: _L / _R instead of 左 / 右, NFKC digits."""
    base, c = separer_side(name)
    return base + ("_" + c if c else "")


def name_mirror(name):
    """PMX name of the symmetric element (左 <-> 右, _L_ <-> _R_), or None."""
    if "左" in name:
        return name.replace("左", "右")
    if "右" in name:
        return name.replace("右", "左")
    m = _SIDE_MIDDLE.match(nfkc(name))
    if m and separer_side(name)[1] is not None and not any(rx.match(nfkc(name)) for rx, _ in _SIDE):
        return "%s_%s_%s" % (m.group(1), "R" if m.group(2) == "L" else "L", m.group(3))
    return None


# Bone classes of SQ-01 (names without side, NFKC)
STANDARD_MMD = {"全ての親", "センター", "グルーブ", "腰", "上半身", "上半身2", "首", "頭", "両目", "目",
                "肩", "腕", "ひじ", "手首", "親指1", "親指2", "下半身", "足", "ひざ", "足首", "つま先",
                "足IK", "つま先IK"} | {d + str(k) for d in ("人指", "中指", "薬指", "小指") for k in (1, 2, 3)}
SEMI_STANDARD_MMD = {"操作中心", "足IK親", "肩P", "肩C", "腕捩", "腕捩1", "腕捩2", "腕捩3", "手捩",
                     "手捩1", "手捩2", "手捩3", "ダミー", "親指0", "腰キャンセル", "足D", "ひざD",
                     "足首D", "足先EX", "上半身1", "上半身3"}


def class__os(name, sheet=False):
    """SQ-01: standard_mmd, semi_standard_mmd or own. The "先" end bones
    (leaves whose name ends with 先) are semi-standard."""
    base, _ = separer_side(name)
    if base in STANDARD_MMD:
        return "standard_mmd"
    if base in SEMI_STANDARD_MMD:
        return "semi_standard_mmd"
    if sheet and base.endswith("先"):
        return "semi_standard_mmd"
    return "own"


# Order matters: the first pattern that matches wins.
FAMILIES = [
    ("root", r"^(全ての親|操作中心|センター|グルーブ)$"),
    ("accessory", r"坠|珠|刀|扣|结|穗|发带|武器|錘?鞘"),
    ("garment", r"外套|衣|スカート|裙|领带|腰环|袖"),
    ("hair", r"髪|发|刘海|发髻"),
    ("face", r"目|眉|齿|舌|口|あご|顎"),
    ("buttocks", r"臀|尻"),
    ("chest", r"胸"),
    ("hand", r"指"),
    ("arm", r"肩|腕|ひじ|手首|捩|手先|ダミー"),
    ("leg", r"足|ひざ|膝|つま先"),
    ("trunk", r"上半身|下半身|腰|首|頭"),
]
_FAMILIES = [(f, re.compile(rx)) for f, rx in FAMILIES]

# Body sub-region given by the bone with the largest weight (P-02). The pattern that covers the
# most characters of the name wins (足首 over 首 and 足, 手首 over 首); at equal length,
# the order of the list (足指: 指 before 足).
REGIONS_OS = [
    ("finger", r"指"),
    ("hand", r"手首|手先|ダミー"),
    ("forearm", r"ひじ|手捩"),
    ("arm", r"腕"),
    ("trunk_top", r"肩|上半身2|上半身3"),
    ("chest", r"胸"),
    ("buttocks", r"臀|尻"),
    ("belly", r"上半身1?$"),
    ("pelvis", r"下半身|腰|^(全ての親|操作中心|センター|グルーブ)$"),
    ("neck", r"首"),
    ("head", r"頭"),
    ("foot", r"足首|つま先|足先"),
    ("leg", r"ひざ|膝"),
    ("thigh", r"足"),
]
_REGIONS_OS = [(r, re.compile(rx)) for r, rx in REGIONS_OS]


_FAMILIES_JSON = {}


def _games():
    tools = os.path.dirname(HERE)
    if tools not in sys.path:
        sys.path.append(tools)
    import games as games
    return games


def load_families():
    """Hand-kept classifications for the models of the active game: families.json of its description
    (tools/games/<game>/families.json); no classification if the game has none."""
    p = _games().active().file("families.json")
    if p is None:
        _FAMILIES_JSON.update({"d": {}, "mtime": None, "patterns": []})
        return _FAMILIES_JSON["d"]
    m = os.path.getmtime(p)
    if _FAMILIES_JSON.get("mtime") != m:
        with open(p, encoding="utf-8") as f:
            _FAMILIES_JSON["d"] = json.load(f)
        _FAMILIES_JSON["mtime"] = m
        _FAMILIES_JSON["patterns"] = [(re.compile(e["pattern"]), e)
                                    for e in _FAMILIES_JSON["d"].get("patterns_os", [])]
    return _FAMILIES_JSON["d"]


def _input(e):
    return {"family": e} if isinstance(e, str) else e


def input_os(name, manual=None):
    """Hand-written classification (families.json) of a bone: dict {family, region?} or None.
    `manuel`: "os" dict instead of families.json (the patterns are then ignored if
    manuel is not None and does not come from the file)."""
    d = load_families()
    file = manual is None
    manual = d.get("bones", {}) if manual is None else manual
    for the_key in (name, identifier(name)):
        if the_key in manual:
            return _input(manual[the_key])
    if file or manual is d.get("bones"):
        n = nfkc(name)
        for rx, e in _FAMILIES_JSON["patterns"]:
            if rx.search(n):
                return e
    return None


def family_os(name, manual=None):
    """P-01: family of a bone: written classification (families.json: "os" then "motifs_os"),
    otherwise patterns of FAMILIES, otherwise None (to be classified in families.json)."""
    e = input_os(name, manual)
    if e is not None:
        return e["family"]
    base, _ = separer_side(name)
    for f, rx in _FAMILIES:
        if rx.search(base):
            return f
    return None


def region_os(name, family):
    """Body region of a bone (for P-02), or the family if the bone is not part of the body."""
    if family in ("hair", "garment", "accessory"):
        return {"garment": "garment_free"}.get(family, family)
    if family == "face":
        return "head"
    e = input_os(name)
    if e is not None and e.get("region"):
        return e["region"]
    base, _ = separer_side(name)
    best, n = None, 0
    for r, rx in _REGIONS_OS:
        t = rx.search(base)
        if t and t.end() - t.start() > n:
            best, n = r, t.end() - t.start()
    return best


# ======================================================================= P-02

GROUPS_MATERIAL = {
    "eyes": [r"^白目$", r"^目$", r"^目光", r"^睫$", r"^眉睫影$", r"^目影$"],
    "face": [r"^颜$", r"^痣$", r"^眉$", r"^口$", r"^舌$", r"^齿$"],
    "hair": [r"^发$", r"^侧发$", r"^前发$"],
    "skin": [r"^肌$"],
}


def group_material(name, accessories=None):
    accessories = accessories if accessories is not None else load_families().get("materials_accessory", [])
    for g, patterns in GROUPS_MATERIAL.items():
        if any(re.search(m, name) for m in patterns):
            return g
    if name in accessories:
        return "accessory"
    return "outfit"


def table_materials(model):
    """ZZ-04 table of a model (families.json "materials/per_model"), or None."""
    pm = load_families().get("materials", {}).get("per_model", {})
    t = pm.get(model)
    for _ in range(5):
        if isinstance(t, str) and t.startswith("="):
            t = pm.get(t[1:])
    return t if isinstance(t, dict) else None


def classes_materials(model, names):
    """ZZ-04 then P-02: for each material, dict {class_, group, de?, default?}.
    class_ None = unclassified (reported, group "non_class"). Without a table for this
    model: old classification by Evelyn's names (class_ None, source "names")."""
    d = load_families().get("materials", {})
    to_group = d.get("groups", {})
    t = table_materials(model)
    if t is None:
        accessory = load_families().get("materials_accessory", [])
        return [{"class": None, "group": group_material(n, accessory), "source": "names"} for n in names]
    out = []
    for n in names:
        e = t.get(n)
        e = {"class": e} if isinstance(e, str) else dict(e or {"class": None})
        c = e["class"]
        if c == "doublet":
            b = t.get(e.get("from"))
            b = b if isinstance(b, str) else (b or {}).get("class")
            e["group"] = to_group.get(b, "non_class")
        elif c == "mixed":
            e["group"] = "mixed"
        else:
            e["group"] = to_group.get(c, "non_class")
        e["source"] = "table"
        out.append(e)
    return out


def mat_vertex(faces, mat_face, n):
    """Material of each vertex (that of its first face)."""
    m = np.full(n, -1, dtype=np.int64)
    f = np.asarray(faces)
    for k in (2, 1, 0):
        m[f[:, k]] = mat_face
    return m


_FAMILY_TO_GROUP = {"hair": "hair", "accessory": "accessory", "garment": "outfit"}
WITHOUT_FACE = "without_face"


def groups_welded(classes, mat_face, faces, n, idx, weight, families_os):
    """P-02: group of each face and each vertex of `welded`. "mixed" materials
    are resolved per geometric island (P-05): majority of the dominant bone families
    (hair, accessory, garment) of the material's vertices in the island,
    otherwise the material's "default" group."""
    gm = np.array([c["group"] for c in classes], dtype=object)
    gf = gm[mat_face]
    ms = mat_vertex(faces, mat_face, n)
    gs = gm[ms]
    gs[ms < 0] = WITHOUT_FACE  # vertex used by no face: neither material nor group
    mixed = [i for i, c in enumerate(classes) if c.get("class") == "mixed"]
    if not mixed:
        return gf, gs
    ar, _ = edges_of(faces)
    il = islands_geometric(n, ar)
    dom = os_dominant(idx, weight)
    gdom = np.array([_FAMILY_TO_GROUP.get(families_os[b]) if b >= 0 else None for b in dom],
                    dtype=object)
    for i in mixed:
        default = classes[i].get("default", "outfit")
        # group of an island dominated by clothing bones when the material is not part of the outfit
        # (ZhuYuan: 髮 also carries a brooch and an armband, under bones classified garment)
        to = {"outfit": classes[i].get("garment", "outfit")}
        fm = np.where(mat_face == i)[0]
        vm = np.unique(faces[fm])
        choice = {}
        for k in np.unique(il[vm]):
            v = vm[il[vm] == k]
            g = [x for x in gdom[v] if x is not None]
            if g:
                u, c = np.unique(np.array(g, dtype=object), return_counts=True)
                choice[k] = to.get(u[np.argmax(c)], u[np.argmax(c)])
            else:
                choice[k] = default
        gf[fm] = [choice[k] for k in il[faces[fm, 0]]]
        sv = np.where(ms == i)[0]
        gs[sv] = [choice[k] for k in il[sv]]
    return gf, gs


GROUPS = ["skin", "outfit", "accessory", "face", "eyes", "hair", WITHOUT_FACE]
REGIONS = ["trunk_top", "chest", "belly", "pelvis", "buttocks", "neck", "head", "arm",
           "forearm", "hand", "finger", "thigh", "leg", "foot", "face", "eyes",
           "hair", "garment_free", "accessory", "without", WITHOUT_FACE]

# "chest" region of P-02: rigs with shared weights (NangongYu, Qingyi, Trigger) leave
# almost no vertex dominated by a chest bone (胸上2 tops out at 0.51-0.68).
# So the sum of the chest bone weights is read, at half of its maximum on the
# model (the full width at half maximum assumes no absolute weight value).
REGIONS_TO_CHEST = ("trunk_top", "belly", "chest")


def os_dominant(idx, weight):
    """Bone with the largest weight for each vertex (-1 if none)."""
    w = np.where(idx >= 0, weight, -1.0)
    k = np.argmax(w, axis=1)
    b = idx[np.arange(len(idx)), k]
    b[w[np.arange(len(idx)), k] <= 0] = -1
    return b


def regions(group_vertex, idx, weight, names_os, families_os):
    """P-02: region of each vertex. group_vertex: array of group names
    (material first). Returns an array of region names (REGIONS).

    A vertex without a face belongs to no body region (SANS_FACE). The chest
    is taken from the sum of weights (REGIONS_TO_CHEST), not from the dominant bone."""
    dom = os_dominant(idx, weight)
    reg_os = [region_os(n, f) for n, f in zip(names_os, families_os)]
    out = np.empty(len(group_vertex), dtype=object)
    for i, (g, b) in enumerate(zip(group_vertex, dom)):
        if g in ("eyes", "face", "hair", WITHOUT_FACE):
            out[i] = g
        elif b < 0 or reg_os[b] is None:
            out[i] = "without"
        else:
            out[i] = reg_os[b]
    is_p = np.array([r == "chest" for r in reg_os], dtype=bool)
    if is_p.any():
        ok = idx >= 0
        s = (np.where(ok, weight, 0.0) * is_p[np.maximum(idx, 0)]).sum(axis=1)
        seen = np.isin(out, REGIONS_TO_CHEST)
        if seen.any() and s[seen].max() > 0:
            out[seen & (s >= 0.5 * s[seen].max())] = "chest"
    return out


# ======================================================================= utilities

def edges_of(faces):
    """Unique sorted edges (e, 2), and for each face its 3 edge indices."""
    f = np.asarray(faces)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.reshape(3, -1).T
    return uniq, inv


def faces_per_edge(face_edges, n_edges):
    """Number of faces per edge."""
    return np.bincount(face_edges.ravel(), minlength=n_edges)


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def found(self, i):
        p = self.p
        r = i
        while p[r] != r:
            r = p[r]
        while p[i] != r:
            p[i], i = r, p[i]
        return r

    def join(self, a, b):
        ra, rb = self.found(a), self.found(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def labels(self):
        lab = np.array([self.found(i) for i in range(len(self.p))])
        _, lab = np.unique(lab, return_inverse=True)
        return lab


def components(n, edges):
    """Connected components through the edges (labels 0..k-1), in numpy."""
    lab = np.arange(n)
    a, b = np.asarray(edges).T
    while True:
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def normals_faces(pos, faces):
    v = pos[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = n / (2 * area[:, None])
    return np.nan_to_num(nn), area


def plan_svd(points):
    """Fitted plane: (centre, normal, RMS deviation)."""
    c = points.mean(axis=0)
    _, s, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[-1]
    rms = np.sqrt(np.mean(((points - c) @ n) ** 2))
    return c, n, rms


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return None

    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        # average ranks for ties
        xs = x[o]
        deb = 0
        for i in range(1, len(x) + 1):
            if i == len(x) or xs[i] != xs[deb]:
                r[o[deb:i]] = (deb + i - 1) / 2
                deb = i
        return r

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ======================================================================= P-03

def mirror(pos, H, tol=1e-3):
    """P-03: m(v) = nearest neighbour of (-x, y, z) within tol·H, otherwise -1."""
    from mathutils.kdtree import KDTree
    kd = KDTree(len(pos))
    for i, p in enumerate(pos):
        kd.insert(p, i)
    kd.balance()
    out = np.full(len(pos), -1, dtype=np.int64)
    lim = tol * H
    for i, p in enumerate(pos):
        _, j, d = kd.find((-p[0], p[1], p[2]))
        if d <= lim:
            out[i] = j
    return out


def mirror_os(sq, tol=None):
    """Mirror of a bone (σ): symmetric bone of each bone, by name (左/右, _L_/_R_), otherwise
    by position (mirrored head within `tol`, bone off the median plane, name without a side:
    numbered skirt bones); the bone itself if it has none."""
    out = np.arange(len(sq))
    per_name = np.zeros(len(sq), dtype=bool)
    for i, n in enumerate(sq.names_pmx):
        mn = name_mirror(n)
        if mn:
            j = sq.index(mn, -1)
            if j >= 0:
                out[i] = j
                per_name[i] = True
    if tol is not None:
        t = sq.heads
        free = np.where(~per_name & (np.abs(t[:, 0]) > tol))[0]
        if len(free):
            mt = t[free] * np.array([-1.0, 1, 1])
            for k, i in enumerate(free):
                d = np.linalg.norm(t[free] - mt[k], axis=1)
                j = int(np.argmin(d))
                if d[j] <= tol and free[j] != i:
                    out[i] = free[j]
    return out


def pairs_mirror(mirror_vertices):
    """Pairs (a, b) a < b of P-03 from m(v); vertices of the median plane (m(v) = v)
    and those without a counterpart (−1) are not included; nor is a pair that is not reciprocal."""
    m = np.asarray(mirror_vertices)
    a = np.where((m >= 0) & (m != np.arange(len(m))))[0]
    a = a[m[m[a]] == a]
    a = a[a < m[a]]
    return np.c_[a, m[a]]


# ======================================================================= alignment

def procrustes(a, b, scale=False):
    """Rotation R (proper), translation t and scale s that bring `a` onto `b` (paired
    points, Kabsch-Umeyama): b ≈ s·R·a + t. Returns (s, R, t)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ca, cb = a.mean(0), b.mean(0)
    A, B = a - ca, b - cb
    U, S, Vt = np.linalg.svd(A.T @ B)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / max((A * A).sum(), 1e-30)) if scale else 1.0
    return s, R, cb - s * R @ ca


def plus_near(a, b, block=2048):
    """Nearest neighbour of each point of a among b (blockwise brute force, numpy only):
    (indices, distances)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.empty(len(a), dtype=np.int64)
    dist = np.empty(len(a))
    bb = (b * b).sum(1)
    for i in range(0, len(a), block):
        x = a[i:i + block]
        d2 = (x * x).sum(1)[:, None] - 2 * x @ b.T + bb[None, :]
        k = np.argmin(d2, axis=1)
        idx[i:i + block] = k
        dist[i:i + block] = np.sqrt(np.maximum(d2[np.arange(len(x)), k], 0))
    return idx, dist


def icp_point_plan(a, b, normals_b, iterations=20, R0=None, t0=None):
    """Point-to-plane ICP: brings `a` onto `b` (normals of `b` given) by small linearized
    rotations, without scale. Returns (R, t, final distances to the nearest point)."""
    a, b, nb = np.asarray(a, float), np.asarray(b, float), np.asarray(normals_b, float)
    R = np.eye(3) if R0 is None else np.asarray(R0, float)
    t = np.zeros(3) if t0 is None else np.asarray(t0, float)
    for _ in range(iterations):
        q = a @ R.T + t
        j, _d = plus_near(q, b)
        n = nb[j]
        M = np.c_[np.cross(q, n), n]
        y = ((b[j] - q) * n).sum(1)
        x, *_ = np.linalg.lstsq(M, y, rcond=None)
        w, dt = x[:3], x[3:]
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        U, _s, Vt = np.linalg.svd(np.eye(3) + K)  # project back onto the rotations
        dR = U @ Vt
        R = dR @ R
        t = dR @ t + dt
    return R, t, plus_near(a @ R.T + t, b)[1]


# ======================================================================= P-04

def _loop_empty(n_loops):
    return {"n_vertices": 0, "radius": None, "center": None, "normal": None, "rms_plan": None,
            "angle_axe": None, "n_loops": n_loops, "n_closed": 0, "dist_axe": None,
            "points": np.zeros((0, 3)), "edges": [], "closed": False}


def loop(pos, faces, origin, direction, length=1.0, t=0.0, mask_faces=None, choice="nearest"):
    """P-04: cut of the mesh by the plane perpendicular to `direction` going
    through origin + t·longueur·direction. Returns the closed loop closest to
    the axis: {closed, n_vertices, rayon, centre, normale, rms_plan, angle_axe, n_loops,
    n_closed, dist_axe, points, edges (cut edge of each point, (a, b) with a < b)}.
    The shape is always the same: without a closed loop, closed is False, n_vertices 0,
    edges and points are empty, the geometric quantities are None.

    Computed in numpy (same cut as bmesh.ops.bisect_plane); n_vertices is the
    number of mesh edges crossed by the loop.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    c = np.asarray(origin, dtype=float) + t * length * d
    f = np.asarray(faces)
    if mask_faces is not None:
        f = f[mask_faces]
    s = (pos - c) @ d
    s = np.where(s == 0, 1e-12, s)
    sf = s[f] > 0
    crossed = sf.any(axis=1) & ~sf.all(axis=1)
    f = f[crossed]
    if len(f) == 0:
        return _loop_empty(0)
    # each crossed face gives a segment between two cut edges
    nodes = {}
    pts = []
    segs = []
    for tri in f:
        keys = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if (s[a] > 0) != (s[b] > 0):
                k = (min(a, b), max(a, b))
                if k not in nodes:
                    nodes[k] = len(pts)
                    u = s[a] / (s[a] - s[b])
                    pts.append(pos[a] + u * (pos[b] - pos[a]))
                keys.append(nodes[k])
        if len(keys) == 2:
            segs.append(keys)
    pts = np.array(pts)
    keys_pts = [None] * len(pts)
    for k, i in nodes.items():
        keys_pts[i] = k
    neighbors = [[] for _ in range(len(pts))]
    for a, b in segs:
        neighbors[a].append(b)
        neighbors[b].append(a)
    vu = np.zeros(len(pts), dtype=bool)
    loops = []
    for i in range(len(pts)):
        if vu[i]:
            continue
        pile, comp = [i], []
        vu[i] = True
        while pile:
            k = pile.pop()
            comp.append(k)
            for v in neighbors[k]:
                if not vu[v]:
                    vu[v] = True
                    pile.append(v)
        closed = all(len(neighbors[k]) == 2 for k in comp) and len(comp) >= 3
        loops.append((comp, closed))
    closed_ones = [b for b, ok in loops if ok]
    if not closed_ones:
        return _loop_empty(len(loops))
    infos = []
    for comp in closed_ones:
        P = pts[comp]
        cen, nor, rms = plan_svd(P)
        r = np.linalg.norm(P - cen, axis=1).mean()
        dist_axe = np.linalg.norm(cen - c)
        infos.append((dist_axe, comp, cen, nor, rms, r))
    infos.sort(key=lambda x: x[0] if choice == "nearest" else -len(x[1]))
    dist_axe, comp, cen, nor, rms, r = infos[0]
    ang = np.degrees(np.arccos(min(1.0, abs(float(nor @ d)))))
    return {"n_vertices": len(comp), "radius": float(r), "center": cen, "normal": nor,
            "rms_plan": float(rms), "angle_axe": float(ang), "n_loops": len(loops),
            "n_closed": len(closed_ones), "dist_axe": float(dist_axe), "points": pts[comp],
            "edges": [keys_pts[i] for i in comp], "closed": True}


# ======================================================================= UV channels (NO-08, IN-05)

def r2_lineaire(X, Y):
    """R² (minimum over the columns of Y) of a linear regression with a constant."""
    if len(X) < 10:
        return None
    A = np.c_[X, np.ones(len(X))]
    r2 = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        v = y.var()
        if v < 1e-18:
            r2.append(1.0)
            continue
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2.append(1 - ((A @ c - y) ** 2).mean() / v)
    return float(min(r2))


def coverage_square(triangles_uv, n=64):
    """Share of the centres of an n × n grid of the square [0, 1]² covered by at least one triangle
    (triangles_uv: (t, 3, 2)). Used by NO-08 and IN-05: an SDF channel fills its square."""
    gx, gy = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
    pts = np.c_[gx.ravel(), gy.ravel()]
    covered = np.zeros(len(pts), bool)
    T = np.asarray(triangles_uv)
    for i0 in range(0, len(T), 512):
        tt = T[i0:i0 + 512]
        a0, a1, a2 = tt[:, 0], tt[:, 1], tt[:, 2]
        v0, v1 = a1 - a0, a2 - a0
        den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        okd = np.abs(den) > 1e-14
        if not okd.any():
            continue
        a0, v0, v1, den = a0[okd], v0[okd], v1[okd], den[okd]
        w = pts[None, :, :] - a0[:, None, :]
        l1 = (w[..., 0] * v1[:, None, 1] - w[..., 1] * v1[:, None, 0]) / den[:, None]
        l2 = (v0[:, None, 0] * w[..., 1] - v0[:, None, 1] * w[..., 0]) / den[:, None]
        covered |= ((l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)).any(0)
    return float(covered.mean())


# ======================================================================= P-05

def islands_geometric(n_vertices, edges):
    """P-05: union-find on the edges of `welded`."""
    return components(n_vertices, edges)


def islands_uv(faces_raw, b2s, uv, tol=1e-6):
    """P-05: UV islands on `brut`; two triangles are linked if they share an
    edge (welded vertices) at the same UV coordinates. Returns one label per face."""
    f = np.asarray(faces_raw)
    m = len(f)
    q = np.round(uv / tol).astype(np.int64)
    keys = {}
    uf = UnionFind(m)
    for fi in range(m):
        for a, b in ((f[fi, 0], f[fi, 1]), (f[fi, 1], f[fi, 2]), (f[fi, 2], f[fi, 0])):
            sa, sb = b2s[a], b2s[b]
            if sa > sb:
                sa, sb, a, b = sb, sa, b, a
            k = (sa, sb, q[a, 0], q[a, 1], q[b, 0], q[b, 1])
            if k in keys:
                uf.join(fi, keys[k])
            else:
                keys[k] = fi
    return uf.labels()


def label_islands(lab_vertices, mat_vertex, pos):
    """Label of an island: majority material, box, number of vertices."""
    out = []
    for k in range(lab_vertices.max() + 1):
        sel = np.where(lab_vertices == k)[0]
        mats, cnt = np.unique(mat_vertex[sel], return_counts=True)
        P = pos[sel]
        out.append({"island": int(k), "n_vertices": int(len(sel)), "material": int(mats[np.argmax(cnt)]),
                    "box_min": P.min(0).tolist(), "box_max": P.max(0).tolist()})
    return out


# ======================================================================= P-06

def density(pos, edges, mask_vertices, H):
    """P-06: mean length (in H) of the edges of `welded` touching the zone, and
    density = 1 / length. Returns (longueur_H, density, n_edges)."""
    m = mask_vertices[edges[:, 0]] | mask_vertices[edges[:, 1]]
    if not m.any():
        return None, None, 0
    L = np.linalg.norm(pos[edges[m, 0]] - pos[edges[m, 1]], axis=1).mean() / H
    return float(L), float(1.0 / L), int(m.sum())


# ======================================================================= P-07

def icosphere(level=2):
    """Vertices of a unit icosphere (12, 42, 162… vertices)."""
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t),
         (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, dtype=float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache = {}
        nf = []

        def middle(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]

        for a, b, c in f:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v)


def cameras(pos, distance=3.0, level=2):
    """P-07: 162 camera positions at `distance` m from the centre of the box."""
    center = (pos.min(0) + pos.max(0)) / 2
    return center + distance * icosphere(level)


def faces_views(pos, faces, cams=None, mask_faces=None):
    """P-07: a face is seen if a ray from a camera reaches its centre
    unobstructed. Returns a boolean array (m,). Needs mathutils."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    pos = np.asarray(pos, dtype=float)
    faces = np.asarray(faces)
    if cams is None:
        cams = cameras(pos)
    bvh = BVHTree.FromPolygons(pos.tolist(), faces.tolist(), all_triangles=True, epsilon=0.0)
    centers = pos[faces].mean(axis=1)
    view = np.zeros(len(faces), dtype=bool)
    a_tester = np.arange(len(faces)) if mask_faces is None else np.where(mask_faces)[0]
    for cam in cams:
        o = Vector(cam)
        rest = a_tester[~view[a_tester]]
        if len(rest) == 0:
            break
        dirs = centers[rest] - cam
        dist = np.linalg.norm(dirs, axis=1)
        dirs /= dist[:, None]
        for fi, dv, di in zip(rest, dirs, dist):
            loc, nor, idx, d = bvh.ray_cast(o, Vector(dv), di + 1e-4)
            if idx is not None and (idx == fi or d >= di - 1e-5):
                view[fi] = True
    return view


# ======================================================================= P-08

class Skeleton:
    """Bones of the PMX in the Blender frame. Heads are in metres."""

    def __init__(self, names_pmx, heads, tails, parents, addition_source, addition_factor,
                 addition_rot, addition_disp):
        self.names_pmx = list(names_pmx)
        self.ids = [identifier(n) for n in self.names_pmx]
        self.heads = np.asarray(heads, dtype=float)
        self.tails = np.asarray(tails, dtype=float)
        self.parents = np.asarray(parents, dtype=np.int64)
        self.addition_source = np.asarray(addition_source, dtype=np.int64)
        self.addition_factor = np.asarray(addition_factor, dtype=float)
        self.addition_rot = np.asarray(addition_rot, dtype=bool)
        self.addition_disp = np.asarray(addition_disp, dtype=bool)
        self._index = {}
        for i, (n, k) in enumerate(zip(self.names_pmx, self.ids)):
            self._index.setdefault(n, i)
            self._index.setdefault(k, i)
            self._index.setdefault(nfkc(n), i)

    def __len__(self):
        return len(self.names_pmx)

    def index(self, name, default=None):
        i = self._index.get(name, self._index.get(nfkc(name)))
        if i is None:
            if default is not None:
                return default
            raise KeyError(name)
        return i

    def a(self, name):
        return self.index(name, -1) >= 0

    def children(self, b):
        return np.where(self.parents == b)[0]

    def direction(self, b, to=None):
        """Direction of the bone: towards the head of `to` if given, otherwise towards its tail."""
        target = self.heads[self.index(to)] if to is not None else self.tails[b]
        v = target - self.heads[b]
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.array([0.0, 0.0, 1.0]), n

    def kinematics(self, rotations=None, additions_actives=None):
        return kin.kinematics(self.heads, self.parents, self.addition_source, self.addition_factor,
                               self.addition_rot, self.addition_disp, rotations, None, additions_actives)


FRONT = np.array([0.0, -1.0, 0.0])
TOP = np.array([0.0, 0.0, 1.0])
BOTTOM = -TOP

# Segment of each limb bone: bone -> bone whose head ends the segment
SEGMENTS = {"腕": "ひじ", "ひじ": "手首", "手首": "中指１", "足": "ひざ", "ひざ": "足首",
            "足首": "つま先", "首": "頭"}
POSES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"]


def _bend(sq, os_, to, target, degrees):
    """Rotation that brings the direction of bone `os_` towards `target` by `degrees`."""
    d, _ = sq.direction(sq.index(os_), to)
    axis = np.cross(d, target)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.cross(d, FRONT)
    return kin.q_axe_angle(axis, degrees)


def pose(number, sq, sides=("L", "R")):
    """P-08: local rotations (dict bone index -> quaternion) of the test pose.

    Angles counted from the rest pose (the delivered A-pose). The rotations
    apply to the standard bones; the D bones, twist bones and 腰キャンセル follow through
    their additions (kinematics). Right side = mirror of the left side."""
    r = {}

    def put(name, q, side):
        n = ("左" if side == "L" else "右") + name
        b = sq.index(n, -1)
        if b < 0:
            return
        r[b] = kin.q_mul(r.get(b, kin.IDENTITY), q)

    def symmetric(name, built, which=sides):
        for c in which:
            put(name, built(c), c)

    def for_(c, os_, to, target, deg):
        name = ("左" if c == "L" else "右") + os_
        v = ("左" if c == "L" else "右") + to
        if not sq.a(name) or not sq.a(v):
            return kin.IDENTITY
        return _bend(sq, name, v, target, deg)

    def around(c, os_, to, deg):
        name = ("左" if c == "L" else "右") + os_
        v = ("左" if c == "L" else "右") + to
        if not sq.a(name) or not sq.a(v):
            return kin.IDENTITY
        d, _ = sq.direction(sq.index(name), v)
        return kin.q_axe_angle(d, deg if c == "L" else -deg)

    n = number.upper()
    if n == "P0":
        pass
    elif n == "P1":  # arms raised by 150°
        symmetric("腕", lambda c: for_(c, "腕", "ひじ", TOP, 150))
    elif n == "P2":  # arms in front at 90°
        symmetric("腕", lambda c: for_(c, "腕", "ひじ", FRONT, 90))
    elif n == "P3":  # forearm in pronation of 90° (wrist twist bone)
        os_ = "手捩" if sq.a("左手捩") else "手首"
        symmetric(os_, lambda c: around(c, "ひじ", "手首", 90))
    elif n == "P4":  # elbow bent by 130°
        symmetric("ひじ", lambda c: for_(c, "ひじ", "手首", FRONT, 130))
    elif n == "P5":  # left thigh raised by 90°
        symmetric("足", lambda c: for_(c, "足", "ひざ", FRONT, 90), ("L",))
    elif n == "P6":  # left knee bent by 120°
        symmetric("ひざ", lambda c: for_(c, "ひざ", "足首", -FRONT, 120), ("L",))
    elif n == "P7":  # crouch: hips 110°, knees 130°
        symmetric("足", lambda c: for_(c, "足", "ひざ", FRONT, 110))
        symmetric("ひざ", lambda c: for_(c, "ひざ", "足首", -FRONT, 130))
    elif n == "P8":  # torso turned by 45°
        b = sq.index("上半身", -1)
        if b >= 0:
            r[b] = kin.q_axe_angle(TOP, 45)
    elif n == "P9":  # head turned by 60° and raised by 30°
        b = sq.index("頭", -1)
        if b >= 0:
            r[b] = kin.q_mul(kin.q_axe_angle(TOP, 60), kin.q_axe_angle([-1, 0, 0], 30))
    elif n == "P10":  # wrist bent by 70° (towards the palm, grip downwards)
        symmetric("手首", lambda c: for_(c, "手首", "中指１", BOTTOM, 70))
    elif n == "P11":  # fingers closed
        for finger in ("人指", "中指", "薬指", "小指"):
            for k, deg in (("１", 70), ("２", 90), ("３", 60)):
                suite = {"１": "２", "２": "３", "３": None}[k]
                for c in sides:
                    name = ("左" if c == "L" else "右") + finger + k
                    if not sq.a(name):
                        continue
                    b = sq.index(name)
                    v = (("左" if c == "L" else "右") + finger + suite) if suite else None
                    d, _ = sq.direction(b, v) if v and sq.a(v) else sq.direction(b)
                    axis = np.cross(d, BOTTOM)
                    r[b] = kin.q_axe_angle(axis, deg)
        for k, suite, deg in (("１", "２", 30), ("２", None, 45)):
            for c in sides:
                name = ("左" if c == "L" else "右") + "親指" + k
                if not sq.a(name):
                    continue
                b = sq.index(name)
                v = (("左" if c == "L" else "右") + "親指" + suite) if suite else None
                d, _ = sq.direction(b, v) if v and sq.a(v) else sq.direction(b)
                r[b] = kin.q_axe_angle(np.cross(d, BOTTOM), deg)
    elif n == "P12":  # walking step: left thigh +30° in front, right 30° behind
        symmetric("足", lambda c: for_(c, "足", "ひざ", FRONT, 30), ("L",))
        symmetric("足", lambda c: for_(c, "足", "ひざ", -FRONT, 30), ("R",))
    else:
        raise ValueError("unknown pose: %s" % number)
    return r


def deform(pos, idx, weight, sq, number=None, rotations=None, additions_actives=None,
             normalize=True):
    """Positions deformed by the pose (numpy linear skinning, checked against
    Blender in §2.1). Returns (positions, R, t)."""
    if rotations is None:
        rotations = pose(number, sq) if number else {}
    R, t = sq.kinematics(rotations, additions_actives)
    return kin.skinning(pos, idx, weight, sq.heads, R, t, normalize), R, t


# ======================================================================= P-09

def d1_lengths(pos0, pos1, edges, mask_edges=None):
    """D1: ratio of edge lengths, percentiles 1 and 99."""
    e = edges if mask_edges is None else edges[mask_edges]
    l0 = np.linalg.norm(pos0[e[:, 0]] - pos0[e[:, 1]], axis=1)
    l1 = np.linalg.norm(pos1[e[:, 0]] - pos1[e[:, 1]], axis=1)
    ok = l0 > 1e-9
    r = l1[ok] / l0[ok]
    if len(r) == 0:
        return None
    return {"c01": float(np.percentile(r, 1)), "c99": float(np.percentile(r, 99)),
            "n": int(len(r))}


def d2_flipped(pos0, pos1, faces, R_dominant, mask_faces=None):
    """D2: flipped faces. The deformed normal is compared to the rest normal
    rotated by the dominant bone of the face (otherwise every face of a limb
    rotated by more than 90° would count). R_dominant: (m, 3, 3)."""
    f = faces if mask_faces is None else faces[mask_faces]
    Rd = R_dominant if mask_faces is None else R_dominant[mask_faces]
    n0, a0 = normals_faces(pos0, f)
    n1, a1 = normals_faces(pos1, f)
    expected = np.einsum("mij,mj->mi", Rd, n0)
    ok = (a0 > 1e-12) & (a1 > 1e-12)
    return int(((np.einsum("mi,mi->m", expected, n1) < 0) & ok).sum())


def rotation_dominant_faces(faces, idx, weight, R):
    """Rotation of the bone with the largest total weight on each face (for D2)."""
    dom = os_dominant(idx, weight)
    fd = dom[faces]
    # most frequent bone of the three vertices, otherwise that of the first
    b = np.where(fd[:, 1] == fd[:, 2], fd[:, 1], fd[:, 0])
    b = np.where(b < 0, 0, b)
    return R[b]


def d3_profile(pos0, pos1, faces, sq, os_, to, R, t, fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
              mask_faces=None):
    """D3: radius profile r(t) along the bone, posed / rest (P-04 on both)."""
    b = sq.index(os_)
    d0, L = sq.direction(b, to)
    h1 = t[b]
    d1 = R[b] @ d0
    out = []
    for f in fractions:
        a0 = loop(pos0, faces, sq.heads[b], d0, L, f, mask_faces)
        a1 = loop(pos1, faces, h1, d1, L, f, mask_faces)
        if a0 and a1 and a0.get("closed") and a1.get("closed"):
            out.append({"t": f, "r0": a0["radius"], "r1": a1["radius"],
                        "report": a1["radius"] / a0["radius"]})
        else:
            out.append({"t": f, "r0": None, "r1": None, "report": None})
    return out


def d4_interpenetrations(pos, faces, mask_a, mask_b):
    """D4: pairs of intersecting faces between two non-neighbouring sets of faces
    (BVHTree.overlap). Needs mathutils."""
    from mathutils.bvhtree import BVHTree
    fa = faces[mask_a]
    fb = faces[mask_b]
    if len(fa) == 0 or len(fb) == 0:
        return 0
    ta = BVHTree.FromPolygons(pos.tolist(), fa.tolist(), all_triangles=True)
    tb = BVHTree.FromPolygons(pos.tolist(), fb.tolist(), all_triangles=True)
    pairs = ta.overlap(tb)
    # discard the pairs that share a vertex (neighbours)
    n = 0
    for i, j in pairs:
        if not set(fa[i]) & set(fb[j]):
            n += 1
    return n


def d5_hausdorff(pos_a, pos_b, faces_b=None, mask_vertices=None):
    """D5: one-sided Hausdorff distance from A to B. Without faces_b (same
    topology), the largest vertex-to-vertex displacement (upper bound)."""
    sel = slice(None) if mask_vertices is None else mask_vertices
    if faces_b is None:
        d = np.linalg.norm(pos_a[sel] - pos_b[sel], axis=1)
        return float(d.max()) if len(d) else 0.0
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromPolygons(pos_b.tolist(), np.asarray(faces_b).tolist(), all_triangles=True)
    m = 0.0
    for p in pos_a[sel]:
        loc, _, _, d = bvh.find_nearest(Vector(p))
        if d is not None and d > m:
            m = d
    return float(m)


def indicators(pos0, pos1, faces, edges, idx, weight, R, mask_vertices):
    """P-09 combined for a region (D1, D2, D5; D3 and D4 are requested separately
    because they need a bone or two regions)."""
    ma = mask_vertices[edges[:, 0]] & mask_vertices[edges[:, 1]]
    mf = mask_vertices[faces].all(axis=1)
    Rd = rotation_dominant_faces(faces, idx, weight, R)
    return {"D1": d1_lengths(pos0, pos1, edges, ma),
            "D2": d2_flipped(pos0, pos1, faces, Rd, mf),
            "D5_vertices": d5_hausdorff(pos1, pos0, None, mask_vertices)}


# ======================================================================= P-10

def vertices_morph(offsets, H, eps=1e-4):
    """P-10: vertices whose displacement exceeds ε = 1e-4 H. offsets: (n, 3)."""
    return np.where(np.linalg.norm(offsets, axis=1) > eps * H)[0]


# ======================================================================= P-11

def _border_loops(faces_sel):
    """Boundary loops of a set of faces (edges used only once)."""
    e, fa = edges_of(faces_sel)
    cnt = faces_per_edge(fa, len(e))
    border = e[cnt == 1]
    neigh = {}
    for a, b in border:
        neigh.setdefault(a, []).append(b)
        neigh.setdefault(b, []).append(a)
    vu, loops = set(), []
    for s in neigh:
        if s in vu:
            continue
        pile, comp = [s], []
        vu.add(s)
        while pile:
            k = pile.pop()
            comp.append(k)
            for v in neigh[k]:
                if v not in vu:
                    vu.add(v)
                    pile.append(v)
        loops.append(np.array(comp))
    return loops


def landmarks_face(pos, faces, group_face, name_mat_face, weight_head, H_tol=0.002,
                   mats_iris=("目",), mat_face="颜"):
    """P-11: face landmarks on `welded`.

    pos, faces       : `welded`
    group_face      : group of each face (P-02, material)
    name_mat_face     : material name of each face
    weight_head       : weight of 頭 per vertex
    mats_iris        : materials of class "iris" (ZZ-04); empty = model without eyes
    mat_visage       : largest material of class "visage" (ZZ-04)
    Returns a dict of points (Blender frame) and H, d_eyes. Without iris, the points that
    depend on them (iris, d_eyes, eye sockets, symmetry) are None and the bounds that relied
    on the eyes switch to the skull (nose) and to 0.5 H (mouth width).
    """
    out = {}
    name_mat_face = np.asarray(name_mat_face)
    # iris: barycenter of the vertices of the iris materials, split by the sign of x
    fi = faces[np.isin(name_mat_face, list(mats_iris))] if len(mats_iris) else faces[:0]
    vi = np.unique(fi)
    P = pos[vi]
    left = P[P[:, 0] > 0]
    right = P[P[:, 0] < 0]
    a_iris = len(left) > 0 and len(right) > 0
    if len(mats_iris) and not a_iris:
        raise ValueError("iris not found (materials %s)" % ", ".join(mats_iris))
    if a_iris:
        out["iris_L"] = left.mean(0)
        out["iris_R"] = right.mean(0)
        out["d_eyes"] = float(np.linalg.norm(out["iris_L"] - out["iris_R"]))
        out["radius_iris"] = float(np.abs(left[:, 0] - out["iris_L"][0]).max())
    else:
        out["iris_L"] = out["iris_R"] = out["d_eyes"] = out["radius_iris"] = None
    # top of the skull: max z of the vertices with weight > 0.5 on 頭, hair and accessories excluded
    excluded = np.zeros(len(pos), dtype=bool)
    for g in ("hair", "accessory"):
        excluded[np.unique(faces[group_face == g])] = True
    guard = (weight_head > 0.5) & ~excluded
    if not guard.any():
        raise ValueError("no skull vertex")
    k = np.where(guard)[0]
    tol = max(H_tol, 0.01 * out["d_eyes"]) if a_iris else H_tol
    # the landmark lives on the median plane (Trigger: the highest point is an isolated vertex
    # of the face material, with no mirror counterpart, at x = −0.319 H); without a median vertex, the highest
    k_med = k[np.abs(pos[k, 0]) < tol]
    k_cr = k_med if len(k_med) else k
    skull = pos[k_cr[np.argmax(pos[k_cr, 2])]]
    # face material
    fv = faces[name_mat_face == mat_face]
    vv = np.unique(fv)
    Pv = pos[vv]
    med = vv[np.abs(Pv[:, 0]) < tol]
    if len(med) == 0:
        raise ValueError("no face vertex on the median plane")
    z_iris = (out["iris_L"][2] + out["iris_R"][2]) / 2 if a_iris else skull[2]
    # chin: lowest point of the face material on the median plane
    out["chin"] = pos[med[np.argmin(pos[med, 2])]]
    width_max = 1.5 * out["d_eyes"] if a_iris else 0.5 * float(skull[2] - out["chin"][2])
    # edge of the face: boundary loops of the face material
    loops = _border_loops(fv)
    # eye sockets: boundary loop whose centre is closest to each iris
    for c in ("L", "R"):
        iris = out["iris_" + c]
        if iris is None:
            out["orbit_" + c] = None
            continue
        best = min(loops, key=lambda b: np.linalg.norm(pos[b].mean(0) - iris))
        Pb = pos[best]
        out["orbit_" + c] = {"n_vertices": int(len(best)),
                              "corner_int": Pb[np.argmin(np.abs(Pb[:, 0]))],
                              "corner_ext": Pb[np.argmax(np.abs(Pb[:, 0]))]}
    # mouth: boundary loop centred on the median plane, below the eyes, above the chin
    candidate = []
    for b in loops:
        Pb = pos[b]
        cen = Pb.mean(0)
        wid = Pb[:, 0].max() - Pb[:, 0].min()
        if abs(cen[0]) < tol * 3 and out["chin"][2] < cen[2] < z_iris and \
                wid < width_max and (Pb[:, 0].max() > 0 > Pb[:, 0].min()):
            candidate.append((wid, b))
    if candidate:
        wid, b = max(candidate, key=lambda x: x[0])
        Pb = pos[b]
        out["commissure_L"] = Pb[np.argmax(Pb[:, 0])]
        out["commissure_R"] = Pb[np.argmin(Pb[:, 0])]
        out["mouth"] = Pb.mean(0)
        out["mouth_n_vertices"] = int(len(b))
    else:
        out["mouth"] = None
    # nose: smallest y on the median plane, above the mouth and below the eyes
    zb = out["mouth"][2] if out["mouth"] is not None else out["chin"][2]
    sel = med[(pos[med, 2] > zb) & (pos[med, 2] < z_iris)]
    if len(sel) == 0:
        raise ValueError("nose not found")
    out["nose"] = pos[sel[np.argmin(pos[sel, 1])]]
    out["skull"] = skull
    out["H"] = float(out["skull"][2] - out["chin"][2])
    out["symetrie_iris_H"] = (float(np.linalg.norm(out["iris_L"] * [-1, 1, 1] - out["iris_R"]) / out["H"])
                              if a_iris else None)
    return out


def mats_landmarks(classes, names, mat_face):
    """Reference materials of P-11 taken from ZZ-04: (iris, visage). Visage = the material
    of class "visage" that has the most faces; falls back on Evelyn's names."""
    names = list(names)
    iris = tuple(n for n, c in zip(names, classes) if c.get("class") == "iris")
    vis = [i for i, c in enumerate(classes) if c.get("class") == "face"]
    if not any(c.get("class") for c in classes):
        return ("目",), "颜"
    if not vis:
        raise ValueError("no material of class visage")
    nf = np.bincount(np.asarray(mat_face), minlength=len(names))
    return iris, names[max(vis, key=lambda i: nf[i])]

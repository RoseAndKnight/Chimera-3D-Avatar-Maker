"""Measures how an official ZZZ hairstyle is built: mesh, UVs, texture, physics.

    bin/blender-queue -b --factory-startup --python tools/hair/measure.py -- \
        --library <data>/face-library \
        --output <data>/hair [--only Anton,Jane] [--all]
    <python> -I tools/hair/measure.py --table <sortie>
        (only rebuilds the table; needs numpy, hence Blender's Python)

Inputs: the head library (<biblio>/<P>/fiche.json, coiffure.npz from hairstyle.py: the
hairstyle's faces and the bones that weight only it). The official PMX is re-read (read-only).
Outputs (outside the repository): <sortie>/<P>.json per character, <sortie>/ensemble.json and
<sortie>/tableau.md. Blender is only used to read textures; without bpy, the texture part is
skipped and everything else is computed.

Frame: the PMX one (MMD units, Y up, face towards -Z, character's left at +X).
Lengths in H = head height (top of skull - chin, head.Head.H_crane). C = centre of the sphere
fitted to the head skin above the eyes (the skull).

Cutting units, from coarsest to finest:
  material  role "hair", "texture_layer" (overlay: a copy laid on the hair, same vertices
            within 1 % of H: the "髮+" highlight layer), "accessory" (everything else
            in the hairstyle);
  part      connected component of the hair faces once coincident vertices are welded;
  island    component of the raw faces (the PMX splits vertices at UV seams): one UV island;
  strand    island of at least MIN_ILE vertices, analysed along its abscissa s (0 root, 1 tip,
            geodesic distance from the vertices closest to C);
  chain     hairstyle bones starting from a bone outside the hairstyle (頭); links = longest path.
"""
import argparse
import heapq
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "head"), os.path.join(HERE, "..", "atlas")]
sys.path.append(os.path.join(HERE, ".."))
import settings as R  # noqa: E402
import games as games  # noqa: E402

import geom as G  # noqa: E402
import pmx as PMX  # noqa: E402
import head as TT  # noqa: E402

try:
    import bpy  # noqa: F401
except ImportError:
    bpy = None

# hairstyles studied: chosen in the active game (description in tools/games/<game>/game.json, "studies.hair")
CHOICE = tuple(games.active().value("studies.hair", ()))   # a game with no study: no hairstyle chosen
HAIR = ("髮", "髪", "发", "毛")
ACCESSORY = ("饰", "飾", "带", "帽", "巾", "镜", "环", "耳", "宝", "金属", "机", "罩", "裙", "服", "工具", "下摆", "新規")
MIN_ISLAND = 12            # minimum vertices for an island to count as a strand
TEXTURE_LAYER_TOL = 0.01       # H: an overlay vertex lies on the hair if closer than this
TEXTURE_LAYER_PART = 0.8       # share of laid vertices for a material to be an overlay
SLICES = 20           # slices along s
SYM_TOL = 0.01          # H: a vertex has its X mirror within this distance
RIG_DYN = (1, 2)        # simulated rigid-body modes
SAMPLE_TEX = 24         # texture samples per face (drawn in proportion to UV area)
OS_BODY = ("頭", "首", "上半身", "上半身1", "上半身2", "上半身3", "右肩", "左肩")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=_json_def)
    os.replace(tmp, path)


def _json_def(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def r3(x, n=3):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r3(v, n) for v in x]
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def q(v, ps=(10, 50, 90), n=3):
    """Quantiles of a series (short dict); None if empty."""
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return {"n": int(len(v)), **{"p%d" % p: r3(np.percentile(v, p), n) for p in ps}}


# =================================================================== geometry helpers

def kd_plus_near(ref, req):
    """Distance to, and index of, the closest point of ref for each point of req."""
    ref, req = np.asarray(ref, float), np.asarray(req, float)
    try:
        from mathutils.kdtree import KDTree
        kd = KDTree(len(ref))
        for i, p in enumerate(ref):
            kd.insert(p.tolist(), i)
        kd.balance()
        d = np.empty(len(req))
        j = np.empty(len(req), dtype=np.int64)
        for i, p in enumerate(req):
            _, k, dist = kd.find(p.tolist())
            d[i], j[i] = dist, k
        return d, j
    except ImportError:
        d = np.empty(len(req))
        j = np.empty(len(req), dtype=np.int64)
        for a in range(0, len(req), 256):
            qq = req[a:a + 256]
            dd = ((qq[:, None, :] - ref[None]) ** 2).sum(-1)
            k = dd.argmin(1)
            j[a:a + 256] = k
            d[a:a + 256] = np.sqrt(dd[np.arange(len(qq)), k])
        return d, j


def sphere(points):
    p = np.asarray(points, float)
    A = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3]
    return c, float(np.sqrt(max(x[3] + c @ c, 0.0)))


def adjacency(n, F, V=None):
    """Neighbour lists (and edge lengths if V is given)."""
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
    adj = [[] for _ in range(n)]
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) if V is not None else np.ones(len(E))
    for (a, b), l in zip(E, L):
        adj[a].append((b, l))
        adj[b].append((a, l))
    return adj


def dijkstra(adj, sources):
    d = np.full(len(adj), np.inf)
    h = []
    for s in sources:
        d[s] = 0.0
        h.append((0.0, int(s)))
    heapq.heapify(h)
    while h:
        d_u, u = heapq.heappop(h)
        if d_u > d[u]:
            continue
        for v, l in adj[u]:
            nd = d_u + l
            if nd < d[v]:
                d[v] = nd
                heapq.heappush(h, (nd, v))
    return d


def bfs(adj, sources):
    d = np.full(len(adj), -1)
    file = list(sources)
    for s in file:
        d[s] = 0
    i = 0
    while i < len(file):
        u = file[i]
        i += 1
        for v, _ in adj[u]:
            if d[v] < 0:
                d[v] = d[u] + 1
                file.append(v)
    return d


def edges_counts(F):
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, inv, c = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    return u, c, inv.ravel().reshape(3, -1).T


def rot_mmd(r):
    """Rotation matrix of a PMX rigid body or joint (angles in radians, MMD's YXZ order:
    R = Ry · Rx · Rz)."""
    x, y, z = r
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def segment_rigid(rg):
    """Rigid body -> (a, b, radius): inner segment and radius (capsule; sphere: a = b;
    box: bounding capsule along the longest side, radius = diagonal of the cross-section)."""
    c = np.asarray(rg["pos"], float)
    R = rot_mmd(rg["rot"])
    t = rg["size"]
    if rg["shape"] == 0:
        return c, c, float(t[0])
    if rg["shape"] == 2:
        ax = R[:, 1] * t[1] / 2
        return c - ax, c + ax, float(t[0])
    k = int(np.argmax(t))
    others = [t[i] for i in range(3) if i != k]
    ax = R[:, k] * max(t[k] - min(others), 0)
    return c - ax, c + ax, float(np.hypot(*others))


def dist_segments(p1, q1, p2, q2):
    """Distance between two segments (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def dist_point_segment(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def collide(ra, rb):
    """Bullet: two bodies collide if each accepts the other's group (bit set)."""
    return bool((ra["mask"] >> rb["group"]) & 1) and bool((rb["mask"] >> ra["group"]) & 1)


def ignores(rg):
    return [g for g in range(16) if not (rg["mask"] >> g) & 1]


# =================================================================== loading

class Hairstyle:
    def __init__(self, lib, name):
        self.name = name
        with open(os.path.join(lib, name, "record.json"), encoding="utf-8") as fh:
            self.path = os.path.join(lib, name, json.load(fh)["pmx"])  # relative (Outerplane) or absolute
        z = np.load(os.path.join(lib, name, "hairstyle.npz"))
        self.mask_faces, self.mask_os = z["hairstyle"], z["bones"]
        self.D = D = TT.Head(self.path)
        self.p, self.t = D.pmx, D.t
        self.pos, self.F, self.mf = D.pos, D.faces, D.mat_face
        self.names_os = D.names_os
        self.per = np.array([o["parent"] for o in self.p["bones"]])
        # head frame
        eyes = 0.5 * (D.center_eye["g"][1] + D.center_eye["d"][1])
        cr = D.V[D.V[:, 1] > eyes]
        self.C, self.R = sphere(cr)
        self.H = float(D.H_crane)
        self.y_eyes = float(eyes)
        self.chin = D.V[D.landmarks["chin"]]
        self._roles()

    def _roles(self):
        p, mf = self.p, self.mf
        mats = sorted(set(np.unique(mf[self.mask_faces]).tolist()))
        tex = lambda m: p["textures"][m["texture"]] if m["texture"] >= 0 else ""
        fur = []
        for i in mats:
            m = p["materials"][i]
            n, tx = m["name"], os.path.basename(tex(m).replace("\\", "/"))
            if "+" in n:
                continue
            name_ch = any(c in n for c in HAIR) and not any(a in n for a in ACCESSORY)
            tex_ch = any(c in tx for c in HAIR) and not any(a in n for a in ACCESSORY)
            if name_ch or tex_ch:
                fur.append(i)
        vh = np.unique(self.F[np.isin(mf, fur) & self.mask_faces])
        self.roles = {}
        for i in mats:
            fk = np.where((mf == i) & self.mask_faces)[0]
            if i in fur:
                self.roles[i] = "hair"
                continue
            vk = np.unique(self.F[fk])
            d, _ = kd_plus_near(self.pos[vh], self.pos[vk]) if len(vh) else (np.full(len(vk), np.inf), None)
            pose = float((d < TEXTURE_LAYER_TOL * self.H).mean())
            self.roles[i] = "texture_layer" if pose >= TEXTURE_LAYER_PART else "accessory"
        self.mats_hair = [i for i in mats if self.roles[i] == "hair"]
        self.f_ch = np.where(np.isin(mf, self.mats_hair) & self.mask_faces)[0]
        self.i_head = self.names_os.index("頭") if "頭" in self.names_os else -1


# =================================================================== GLB input (our models)

# hair materials of a GLB: name containing one of these patterns (lower case), except eyebrows,
# eyelashes and accessories; the "+" copies and overlays are then found by geometry
# names some models write in French (materials, bones): the model's own words, matched as they are
MODEL_FRENCH = {"hair": ("mèche",), "head": ("tête",), "face": ("œil", "oeil", "cil", "dent", "langue", "bouche")}   # fr-ok: names written in the models
PATTERNS_HAIR_GLB = ("hair", "strand", "lock", "fringe", "hairstyle", "髮", "髪", "发", "毛") + MODEL_FRENCH["hair"]
EXCLUDED_HAIR_GLB = ("brow", "lash", "eyebrow", "lashes", "眉", "睫")
PATTERNS_HEAD_OS = ("頭", "head") + MODEL_FRENCH["head"]
EXCLUDED_HEAD_OS = ("top", "end", "tip", "_fin", "tip", "nu")   # 頭先, HeadTop_End…


def _choose_names(names, patterns, excluded=()):
    """Indices of the names containing a pattern ("=name": exact name) and no excluded pattern."""
    ok = lambda n, m: n.lower() == m[1:] if m.startswith("=") else m in n.lower()
    return [i for i, n in enumerate(names) if any(ok(n, m) for m in patterns)
            and not any(e in n.lower() for e in excluded)]


class HairstyleGlb(Hairstyle):
    """A hairstyle read from one or more GLBs (the model, plus any separate strands in the
    same frame and on the same skeleton, merged by bone name). Same attributes as Coiffure:
    measurements A to F run unchanged.

    - Positions: bind pose (mesh vertices, node world matrix); bones: origin of the mesh world
      matrix · inverse(IBM), i.e. the bind pose even if the GLB was exported posed.
    - Frame: the PMX one, via head_glb.HeadGlb (eyes and face), units scaled to the ZZZ
      inter-eye distance. H and the skull as for PMX.
    - Hairstyle: hair materials (PATTERNS_HAIR_GLB or --hair-mats), then the overlays
      (>= 80 % of vertices lying on the hair), then the bones that weight only them (rule from
      hairstyle.py, head excluded). Ornaments are not counted.
    - No rigid bodies or joints: physics comes from Godot springs (springs.py)."""

    def __init__(self, paths, name, mats_hair=None, skin=None, eyes=None):
        import head_glb as TG
        self.name = name
        self.path = paths[0]
        self.paths = list(paths)
        self._read(paths, TG)
        self._landmark(TG, skin, eyes)
        self._isolate(mats_hair)
        self._roles()
        self.i_head = self._os_head()
        self._skull_fallback()

    # -------------------------------------------------------------- reading
    def _read(self, paths, TG):
        pos, nor, uv, uva, jo, wo, faces, mf = [], [], [], [], [], [], [], []
        mats, textures, self._images = [], [], []
        os_names, os_per, os_M = [], [], []
        morphs = []
        base = 0
        for kf, ch in enumerate(paths):
            j, binary = TG.read_glb(ch)
            worlds = TG._world(j)
            parent = {c: i for i, nd in enumerate(j.get("nodes", [])) for c in nd.get("children", [])}
            # images and textures (glTF index -> index in self.p["textures"])
            t0 = len(textures)
            for ki, im in enumerate(j.get("images", [])):
                textures.append("%s#%d:%s" % (os.path.basename(ch), ki, im.get("name") or im.get("uri") or ""))
                self._images.append((j, binary, ch, im))
            tex_img = [tx.get("source", -1) for tx in j.get("textures", [])]
            m0 = len(mats)
            for km, m in enumerate(j.get("materials", [])):
                pbr = m.get("pbrMetallicRoughness", {})
                bt = pbr.get("baseColorTexture", {}).get("index", -1)
                img = tex_img[bt] if 0 <= bt < len(tex_img) else -1
                name_m = m.get("name", "mat%d" % km)
                if kf:
                    name_m += " [%s]" % os.path.splitext(os.path.basename(ch))[0]
                mats.append({"name": name_m, "texture": t0 + img if img >= 0 else -1,
                             "flags": PMX.MAT_DOUBLE_FACE if m.get("doubleSided") else 0,
                             "outline_size": None, "outline_color": None,
                             "diffuse": list(pbr.get("baseColorFactor", (1, 1, 1, 1))),
                             "alpha_mode": m.get("alphaMode", "OPAQUE"),
                             "sphere": -1, "sphere_mode": 0, "toon": -1, "toon_sharing": 0})
            # bones: merged by name. They are the skin joints and everything hanging below them
            # that is not a mesh (weightless bones, which the export does not always put in
            # "joints": chain ends, roots); parent = closest ancestor that is a bone
            nds = j.get("nodes", [])
            jts = {jt for sk in j.get("skins", []) for jt in sk["joints"]}
            roots_sq = {x for x in jts if not any(a in jts for a in self._ancestors(parent, x))}
            is_os, pile = set(), list(roots_sq)
            while pile:
                x = pile.pop()
                if "mesh" in nds[x] or x in is_os:
                    continue
                is_os.add(x)
                pile += nds[x].get("children", [])
            order = sorted(is_os, key=lambda x: len(self._ancestors(parent, x)))
            for nd_i in order:
                nm = nds[nd_i].get("name", "os%d" % nd_i)
                if nm in os_names:
                    continue
                p_ = parent.get(nd_i)
                while p_ is not None and p_ not in is_os:
                    p_ = parent.get(p_)
                os_names.append(nm)
                os_per.append(nds[p_].get("name") if p_ is not None else None)
                os_M.append(None)
            for i, nd in enumerate(j.get("nodes", [])):
                if "mesh" not in nd:
                    continue
                M = worlds[i]
                sk = j["skins"][nd["skin"]] if "skin" in nd else None
                map_ = None
                if sk is not None:
                    names_sk = [j["nodes"][x].get("name", "os%d" % x) for x in sk["joints"]]
                    map_ = np.array([os_names.index(n) for n in names_sk])
                    ibm = (TG.accessor(j, binary, sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
                           if "inverseBindMatrices" in sk else None)
                    for kk, g in enumerate(map_):
                        if os_M[g] is None:
                            os_M[g] = M @ np.linalg.inv(ibm[kk]) if ibm is not None else worlds[sk["joints"][kk]]
                mesh = j["meshes"][nd["mesh"]]
                names_targets = mesh.get("extras", {}).get("targetNames", [])
                for pr in mesh["primitives"]:
                    if pr.get("mode", 4) != 4:
                        continue
                    at = pr["attributes"]
                    accessory = lambda k: TG.accessor(j, binary, at[k]).astype(np.float64)
                    P = accessory("POSITION")
                    n = len(P)
                    P = P @ M[:3, :3].T + M[:3, 3]
                    Nn = accessory("NORMAL") @ np.linalg.inv(M[:3, :3]) if "NORMAL" in at else np.zeros((n, 3))
                    Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-12)
                    pos.append(P)
                    nor.append(Nn)
                    uv.append(accessory("TEXCOORD_0") if "TEXCOORD_0" in at else np.zeros((n, 2)))
                    upper = []
                    for k in range(1, 8):
                        if "TEXCOORD_%d" % k not in at:
                            break
                        u = accessory("TEXCOORD_%d" % k)
                        upper.append(np.c_[u, np.zeros((n, 2))])
                    uva.append(upper)
                    if map_ is not None and "JOINTS_0" in at:
                        J = TG.accessor(j, binary, at["JOINTS_0"]).astype(np.int64)
                        W = accessory("WEIGHTS_0")
                        W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
                        J = np.where(W > 0, map_[J], -1)
                    else:
                        J, W = np.full((n, 4), -1), np.zeros((n, 4))
                    jo.append(J)
                    wo.append(W)
                    I = (TG.accessor(j, binary, pr["indices"]).astype(np.int64).reshape(-1, 3)
                         if "indices" in pr else np.arange(n).reshape(-1, 3))
                    faces.append(I + base)
                    mf.append(np.full(len(I), m0 + pr.get("material", 0) if "material" in pr else -1))
                    for kt, tg in enumerate(pr.get("targets", [])):
                        if "POSITION" not in tg:
                            continue
                        d = TG.accessor(j, binary, tg["POSITION"]).astype(np.float64)
                        idx = np.where(np.linalg.norm(d, axis=1) > 1e-6)[0]
                        if len(idx):
                            morphs.append({"type": 1, "name": names_targets[kt] if kt < len(names_targets) else "target%d" % kt,
                                            "offsets": [(int(base + x), d[x]) for x in idx]})
                    base += n
            # bind pose of bones outside "joints": the parent's, then the local transform
            for nd_i in order:
                g = os_names.index(nds[nd_i].get("name", "os%d" % nd_i))
                if os_M[g] is None:
                    p_ = parent.get(nd_i)
                    gp = os_names.index(nds[p_].get("name", "os%d" % p_)) if p_ in is_os else None
                    os_M[g] = (os_M[gp] @ TG._matrix_node(nds[nd_i]) if gp is not None and os_M[gp] is not None
                               else worlds[nd_i])
        n_upper = max((len(s) for s in uva), default=0)
        uva = [np.stack(s + [np.zeros((len(p_), 4))] * (n_upper - len(s)), 1) if n_upper else np.zeros((len(p_), 0, 4))
               for s, p_ in zip(uva, pos)]
        self._raw = dict(pos=np.vstack(pos), nor=np.vstack(nor), uv=np.vstack(uv), uv_add=np.concatenate(uva),
                          bones=np.vstack(jo), weights=np.vstack(wo), faces=np.vstack(faces), mat_face=np.concatenate(mf))
        self._weld_like_pmx(morphs)
        self._os = (os_names, os_per, os_M)
        self._mats, self._textures, self._morphs = mats, textures, morphs

    @staticmethod
    def _ancestors(parent, x):
        out = []
        while x in parent:
            x = parent[x]
            out.append(x)
        return out

    def _weld_like_pmx(self, morphs):
        """A Blender glTF export splits a vertex wherever its corners differ (UV *or* normal);
        a PMX only splits it at UV seams. To measure like the ZZZ models (UV islands, M7 normal
        seams), vertices with the same position, UVs and weights are merged; their normal is
        the average. The share of merged vertices whose normals differed by more than 1 deg is
        recorded: these are hard edges outside UV seams."""
        b = self._raw
        the_key = np.c_[np.round(b["pos"], 6), np.round(b["uv"], 6), b["bones"], np.round(b["weights"], 4)]
        _, rep, inv = np.unique(the_key, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        n = len(rep)
        N = np.zeros((n, 3))
        np.add.at(N, inv, b["nor"])
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        dev = np.degrees(np.arccos(np.clip((b["nor"] * N[inv]).sum(1), -1, 1)))
        ecl = np.zeros(n)
        np.maximum.at(ecl, inv, dev)
        multi = np.bincount(inv, minlength=n) > 1
        self.weld = {"vertices_raw": int(len(inv)), "vertices_welded": int(n),
                        "merged_normals_different_part": r3((ecl[multi] > 1).mean()) if multi.any() else 0.0,
                        "shine_normals_deg": q(ecl[multi])}
        self._ecl_weld = ecl
        for k in ("pos", "uv", "uv_add", "bones", "weights"):
            b[k] = b[k][rep]
        b["nor"] = N
        b["faces"] = inv[b["faces"]]
        for m in morphs:
            vu = {}
            for i, d in m["offsets"]:
                vu.setdefault(int(inv[i]), d)
            m["offsets"] = list(vu.items())

    # -------------------------------------------------------------- frame
    def _landmark(self, TG, skin, eyes):
        b = self._raw
        names_mat = [m["name"] for m in self._mats]
        ip = _choose_names(names_mat, skin) if skin else None
        iy = _choose_names(names_mat, eyes) if eyes else None
        D = TG.HeadGlb(self.path, name=self.name, skin=ip, eyes=iy, mesh_=(b["pos"], b["faces"], b["mat_face"], names_mat))
        R, u = D.rotation, D.unit
        P0 = b["pos"]
        e = (P0 - (D.pos / u) @ R).mean(0)
        self.D, self.rotation, self.unit, self.origin = D, R, u, e
        tr = lambda P: (P - e) @ R.T * u
        pos = tr(P0)
        nor = b["nor"] @ R.T
        names, per, Ms = self._os
        O = []
        for k, nm in enumerate(names):
            M = Ms[k] if Ms[k] is not None else np.eye(4)
            O.append({"name": nm, "parent": names.index(per[k]) if per[k] in names else -1,
                      "pos": tr(M[:3, 3][None])[0], "base": R @ (M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)),
                      "scale_os": float(np.linalg.norm(M[:3, :3], axis=0).mean()),
                      "tail_offset": None})
        n = len(pos)
        self.t = {"pos": pos, "nor": nor, "uv": b["uv"], "uv_add": b["uv_add"], "bones": b["bones"], "weights": b["weights"],
                  "contour": np.full(n, np.nan), "weight_type": np.full(n, 2, dtype=np.int8),
                  "faces": b["faces"], "mat_face": b["mat_face"]}
        self.p = {"materials": self._mats, "textures": self._textures, "bones": O, "rigid_bodies": [], "joints": [],
                  "morphs": [{"type": 1, "name": m["name"], "offsets": m["offsets"]} for m in self._morphs]}
        self.pos, self.F, self.mf = pos, b["faces"], b["mat_face"]
        self.names_os = names
        self.per = np.array([o["parent"] for o in O], dtype=np.int64)
        eyes_y = 0.5 * (D.center_eye["g"][1] + D.center_eye["d"][1])
        self.y_eyes = float(eyes_y)
        self.H = float(D.H_crane)
        self.chin = D.V[D.landmarks["chin"]]
        # skull: the face skin above the eyes, as for PMX; if it only covers the forehead (face
        # separate from the skull), the sphere is badly placed: we then take all the skin
        # weighted to the head bone above the eyes (excluding hair, eyes, eyebrows)
        cr = D.V[D.V[:, 1] > eyes_y]
        self.C, self.R = sphere(cr)
        self.landmark_skull = "face skin"

    def _skull_fallback(self):
        """If the face-skin sphere does not have a skull radius (ZZZ: 0.36-0.45 H), refit it on
        all the skin weighted to the head bone above the eyes, excluding the hairstyle and the
        eye, eyebrow and eyelash materials."""
        if 0.3 <= self.R / self.H <= 0.55 or self.i_head < 0:
            return
        names = [m["name"] for m in self.p["materials"]]
        excluded = set(_choose_names(names, ("eye", "eyes", "brow", "lash", "eyebrow", "teeth", "tongue", "mouth",
                                          "目", "眉", "睫", "齿", "舌") + MODEL_FRENCH["face"]))
        excluded |= set(np.unique(self.mf[self.mask_faces]).tolist())
        vk = np.unique(self.F[~np.isin(self.mf, list(excluded))])
        dom = self.t["bones"][vk, np.argmax(self.t["weights"][vk], 1)]
        P = self.pos[vk[(dom == self.i_head) & (self.pos[vk, 1] > self.y_eyes)]]
        if len(P) < 30:
            return
        C, R = sphere(P)
        log("%s: skull refitted on the head-bone skin (R/H %.2f -> %.2f)" % (self.name, self.R / self.H, R / self.H))
        self.C, self.R = C, R
        self.landmark_skull = "skin weighted to the head bone"

    def _os_head(self):
        names = self.names_os
        candidate = _choose_names(names, PATTERNS_HEAD_OS, EXCLUDED_HEAD_OS)
        if "頭" in names:
            return names.index("頭")
        # among the candidates, the one most hairstyle bones descend from
        if candidate:
            oc = np.where(self.mask_os)[0]
            return max(candidate, key=lambda i: sum(1 for o in oc if self._ancestor(o, i)))
        roots = [int(self.per[o]) for o in np.where(self.mask_os)[0] if self.per[o] >= 0 and not self.mask_os[self.per[o]]]
        return max(set(roots), key=roots.count) if roots else -1

    def _ancestor(self, o, a):
        while o >= 0:
            if o == a:
                return True
            o = int(self.per[o])
        return False

    # -------------------------------------------------------------- hairstyle
    def _isolate(self, patterns):
        names = [m["name"] for m in self._mats]
        fur = (_choose_names(names, patterns, ()) if patterns else
                 [i for i in _choose_names(names, PATTERNS_HAIR_GLB, EXCLUDED_HAIR_GLB)
                  if not any(a in names[i] for a in ACCESSORY)])
        if not fur:
            raise ValueError("%s: no hair material among %s" % (self.name, names))
        mf, F = self.mf, self.F
        hairdo = np.isin(mf, fur)
        vh = np.unique(F[hairdo])
        # overlays: materials with >= 80 % of their vertices lying on the hair
        for i in sorted(set(np.unique(mf).tolist()) - set(fur)):
            vk = np.unique(F[mf == i])
            d, _ = kd_plus_near(self.pos[vh], self.pos[vk])
            if (d < TEXTURE_LAYER_TOL * self.H).mean() >= TEXTURE_LAYER_PART:
                hairdo |= mf == i
        # hairstyle bones (rule from hairstyle.py): they weight it and almost nothing else
        n_os = len(self.names_os)

        def mass(s):
            o, w = self.t["bones"][s], self.t["weights"][s]
            m = np.zeros(n_os)
            ok = o >= 0
            np.add.at(m, o[ok], w[ok])
            return m
        per = self.per
        head_name = lambda k: any(m in self.names_os[k].lower() for m in PATTERNS_HEAD_OS)
        dom_all = self.t["bones"][np.arange(len(self.pos)), np.argmax(self.t["weights"], 1)]

        head = np.array([head_name(k) for k in range(n_os)], bool)

        def os_of(hairdo):
            vc = np.unique(F[hairdo])
            out_of = np.setdiff1d(np.arange(len(self.pos)), vc)
            Wc, Wn = mass(vc), mass(out_of)
            seeds = (Wc > 0) & (Wn <= 0.02 * Wc) & ~head
            without_weight = (Wc + Wn) <= 0
            os_c = seeds.copy()
            for _ in range(128):
                nd = os_c | (without_weight & (per >= 0) & os_c[np.maximum(per, 0)])
                if (nd == os_c).all():
                    break
                os_c = nd
            for i in np.where(seeds)[0]:
                p_ = per[i]
                while p_ >= 0 and without_weight[p_] and not os_c[p_] and not head_name(p_):
                    os_c[p_] = True
                    p_ = per[p_]
            return os_c, Wc
        # Ornaments are not added: a round-trip GLB may mix ornaments, belt and badge in a single
        # material (Qingyi: 饰1, weighted mostly by Ctr_Belt and Ctr_Badge). The bones weighting
        # these ornaments therefore leave the hairstyle (Qingyi: Ctr_HairBD, Ctr_HairBA_01); a
        # known difference from the PMX library, where 饰1 counts as part of the hairstyle.
        os_c, _ = os_of(hairdo)
        self.mask_faces, self.mask_os = hairdo, os_c
        # for _roles: hair materials matched by pattern are "hair" even without HAIR
        self._fur_glb = fur

    def _roles(self):
        p, mf = self.p, self.mf
        mats = sorted(set(np.unique(mf[self.mask_faces]).tolist()))
        fur = [i for i in self._fur_glb if "+" not in p["materials"][i]["name"]]
        vh = np.unique(self.F[np.isin(mf, fur) & self.mask_faces])
        self.roles = {}
        for i in mats:
            if i in fur:
                self.roles[i] = "hair"
                continue
            vk = np.unique(self.F[(mf == i) & self.mask_faces])
            d, _ = kd_plus_near(self.pos[vh], self.pos[vk])
            self.roles[i] = "texture_layer" if (d < TEXTURE_LAYER_TOL * self.H).mean() >= TEXTURE_LAYER_PART else "accessory"
        self.mats_hair = [i for i in mats if self.roles[i] == "hair"]
        self.f_ch = np.where(np.isin(mf, self.mats_hair) & self.mask_faces)[0]

    # -------------------------------------------------------------- embedded textures
    def image(self, name):
        """Image of a GLB texture (embedded or alongside), as an (h, w, 4) array; None without bpy."""
        if bpy is None or name not in self.p["textures"]:
            return None
        j, binary, ch, im = self._images[self.p["textures"].index(name)]
        if "bufferView" in im:
            bv = j["bufferViews"][im["bufferView"]]
            bytes_ = binary[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
            ext = ".png" if "png" in im.get("mimeType", "png") else ".jpg"
            d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "hair-measure-images")
            os.makedirs(d, exist_ok=True)
            file_ = os.path.join(d, "%s-%d%s" % (self.name, self.p["textures"].index(name), ext))
            with open(file_ + ".tmp", "wb") as fh:
                fh.write(bytes_)
            os.replace(file_ + ".tmp", file_)
        elif "uri" in im and not im["uri"].startswith("data:"):
            file_ = os.path.join(os.path.dirname(ch), im["uri"])
        else:
            return None
        try:
            img = bpy.data.images.load(file_)
            w, h = img.size
            px = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(px)
            bpy.data.images.remove(img)
            return px.reshape(h, w, 4)
        except Exception as e:
            log("unreadable texture", file_, e)
            return None


# =================================================================== A. mesh

def measure_mesh(c):
    H, C = c.H, c.C
    F = c.F[c.f_ch]
    vh = np.unique(F)
    P = c.pos[vh]
    loc = np.full(len(c.pos), -1)
    loc[vh] = np.arange(len(vh))
    Fl = loc[F]
    rep, inv = G.weld(P, 1e-5)
    Fw = inv[Fl]
    ok = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
    Vw = P[rep]
    E, cnt, fe = edges_counts(Fw[ok])
    border_w = np.zeros(len(Vw), bool)
    border_w[E[cnt == 1].ravel()] = True
    lab = G.components_faces(Fw, ok)
    parts = []
    dist_cr = (np.linalg.norm(Vw - C, axis=1) - c.R) / H
    for k in range(lab.max() + 1):
        fk = np.where(lab == k)[0]
        vk = np.unique(Fw[fk])
        Ek, ck, _ = edges_counts(Fw[fk])
        parts.append({"faces": int(len(fk)), "vertices": int(len(vk)),
                       "border_edges": int((ck == 1).sum()), "edges_non_manifold": int((ck > 2).sum()),
                       "part_border": r3((ck == 1).sum() / max(len(Ek), 1)),
                       "euler": int(len(vk) - len(Ek) + len(fk)),
                       "dist_crane_med_H": r3(np.median(dist_cr[vk])),
                       "bottom_H": r3((c.C[1] - Vw[vk, 1].min()) / H),
                       "materials": sorted({c.p["materials"][c.mf[c.f_ch[f]]]["name"] for f in fk})})
    parts.sort(key=lambda x: -x["faces"])
    # cap: vertices above the eyes within 0.12 H of the skull
    top = Vw[:, 1] > c.y_eyes
    cal = top & (dist_cr > -0.03) & (dist_cr < 0.12)
    # pairs of opposite faces (explicit double-sided)
    Fs = np.sort(Fw[ok], 1)
    _, cc = np.unique(Fs, axis=0, return_counts=True)
    # faces flipped relative to the skull: normal towards C (on the cap only)
    Fo = Fw[ok]
    nf = G.normals_faces(Vw, Fo)
    cf = Vw[Fo].mean(1)
    on_cal = cal[Fo].all(1)
    to_c = ((nf * (cf - C)).sum(1) < 0)
    # PMX normals: compared with the welded geometric normal and the radial direction
    nw = G.normals_vertices(Vw, Fo)
    Np = c.t["nor"][vh]
    Np = Np / np.maximum(np.linalg.norm(Np, axis=1, keepdims=True), 1e-12)
    ang_geo = np.degrees(np.arccos(np.clip((Np * nw[inv]).sum(1), -1, 1)))
    rad = P - C
    rad /= np.linalg.norm(rad, axis=1, keepdims=True)
    ang_rad = np.degrees(np.arccos(np.clip((Np * rad).sum(1), -1, 1)))
    # normal seams: coincident vertices whose normals differ by more than 1 deg
    shine = np.zeros(len(Vw))
    groups = np.bincount(inv)
    n_mean = np.zeros((len(Vw), 3))
    np.add.at(n_mean, inv, Np)
    n_mean /= np.maximum(np.linalg.norm(n_mean, axis=1, keepdims=True), 1e-12)
    dev = np.degrees(np.arccos(np.clip((Np * n_mean[inv]).sum(1), -1, 1)))
    np.maximum.at(shine, inv, dev)
    multi = groups > 1
    # symmetry: X mirror of each welded vertex
    d_sym, j_sym = kd_plus_near(Vw, Vw * [-1, 1, 1])
    # budget
    tot_v, tot_f = len(c.pos), len(c.F)
    roles = {}
    for i, rl in c.roles.items():
        fk = np.where((c.mf == i) & c.mask_faces)[0]
        m = c.p["materials"][i]
        roles[m["name"]] = {"role": rl, "faces": int(len(fk)), "vertices": int(len(np.unique(c.F[fk]))),
                           "double_sided": bool(m["flags"] & PMX.MAT_DOUBLE_FACE),
                           "contour": bool(m["flags"] & PMX.MAT_OUTLINE),
                           "outline_size": r3(m["outline_size"]), "outline_color": r3(m["outline_color"]),
                           "alpha": r3(m["diffuse"][3]),
                           "texture": c.p["textures"][m["texture"]] if m["texture"] >= 0 else None,
                           "sphere": c.p["textures"][m["sphere"]] if m["sphere"] >= 0 else None,
                           "sphere_mode": m["sphere_mode"],
                           "toon": ("shared %d" % m["toon"]) if m["toon_sharing"] else (
                               c.p["textures"][m["toon"]] if m["toon"] >= 0 else None)}
    # overlays: offset along the normal relative to the hair
    texture_layers = {}
    for i, rl in c.roles.items():
        if rl != "texture_layer":
            continue
        vk = np.unique(c.F[(c.mf == i) & c.mask_faces])
        d, j = kd_plus_near(P, c.pos[vk])
        offset = ((c.pos[vk] - P[j]) * Np[j]).sum(1)
        texture_layers[c.p["materials"][i]["name"]] = {
            "vertices": int(len(vk)), "part_vertices_hair_covered": r3(len(np.unique(inv[j][d < 1e-4 * H])) / len(Vw)),
            "dist_med_H": r3(np.median(d) / H), "normal_offset_med_H": r3(np.median(offset) / H),
            "same_uv": r3(float((np.linalg.norm(c.t["uv"][vk] - c.t["uv"][vh][j], axis=1) < 1e-4).mean())),
            "y_bottom_H": r3((C[1] - c.pos[vk, 1].min()) / H)}
    outline_v = c.t["contour"][vh]
    out = {
        "budget": {"vertices_model": tot_v, "faces_model": tot_f, "vertices_hair": int(len(vh)),
                   "vertices_hair_welded": int(len(Vw)), "faces_hair": int(len(F)),
                   "part_faces": r3(len(F) / tot_f), "hairstyle_faces_whole": int(c.mask_faces.sum()),
                   "faces_texture_layers": int(sum(v["faces"] for v in roles.values() if v["role"] == "texture_layer")),
                   "faces_accessories": int(sum(v["faces"] for v in roles.values() if v["role"] == "accessory")),
                   "vertices_per_face": r3(len(Vw) / max(len(F), 1))},
        "materials": roles,
        "texture_layers": texture_layers,
        "parts": {"count": len(parts), "faces_plus_large": parts[0]["faces"] if parts else 0,
                   "part_plus_large": r3(parts[0]["faces"] / len(F)) if parts else None,
                   "closed_ones": int(sum(1 for x in parts if x["border_edges"] == 0)),
                   "non_manifold": int(sum(x["edges_non_manifold"] for x in parts)),
                   "sizes": [x["faces"] for x in parts[:40]], "detail": parts[:25]},
        "border_edges_part": r3((cnt == 1).sum() / len(E)),
        "faces_degenerate": int((~ok).sum()),
        "pairs_faces_opposite": int((cc > 1).sum()),
        "cap": {"vertices": int(cal.sum()), "part_top": r3(cal[top].mean()) if top.any() else None,
                    "faces_to_skull_part": r3(to_c[on_cal].mean()) if on_cal.any() else None,
                    "dist_crane_H": q(dist_cr[top])},
        "normals": {"angle_geometric_deg": q(ang_geo), "angle_radial_deg": q(ang_rad),
                     "seams_part": r3((shine[multi] > 1).mean()) if multi.any() else 0.0,
                     "vertices_duplicates_part": r3(multi.mean())},
        "outline_vertices": q(outline_v),
        "symmetry": {"part_1pc": r3((d_sym < SYM_TOL * H).mean()), "part_3pc": r3((d_sym < 3 * SYM_TOL * H).mean()),
                     "hausdorff_H": r3(d_sym.max() / H)},
    }
    c.Vw, c.inv_w, c.vh, c.Fw, c.ok_w, c.border_w, c.Np = Vw, inv, vh, Fw, ok, border_w, Np
    c.d_sym, c.j_sym = d_sym, j_sym
    return out


# =================================================================== B. islands and strands

def measure_locks(c):
    H, C = c.H, c.C
    F = c.F[c.f_ch]
    loc = np.full(len(c.pos), -1)
    loc[c.vh] = np.arange(len(c.vh))
    Fl = loc[F]
    lab = G.components_faces(Fl, np.ones(len(Fl), bool))
    uv = c.t["uv"][c.vh]
    P = c.pos[c.vh]
    W_os, W_p = c.t["bones"][c.vh], c.t["weights"][c.vh]
    dom = W_os[np.arange(len(W_os)), np.argmax(W_p, 1)]
    locks = []
    s_glob = np.full(len(c.vh), np.nan)
    island_of = np.full(len(c.vh), -1)
    for k in range(lab.max() + 1):
        fk = np.where(lab == k)[0]
        vk = np.unique(Fl[fk])
        island_of[vk] = k
        if len(vk) < MIN_ISLAND:
            continue
        m = _lock(c, k, fk, vk, Fl, P, uv, dom)
        if m is None:
            continue
        s_glob[vk] = m.pop("_s")
        locks.append(m)
    c.s_glob, c.island_of, c.dom = s_glob, island_of, dom
    c.locks = locks
    return locks


def _lock(c, k, fk, vk, Fl, P, uv, dom):
    H, C = c.H, c.C
    loc = np.full(len(P), -1)
    loc[vk] = np.arange(len(vk))
    Fk = loc[Fl[fk]]
    X = P[vk]
    adj = adjacency(len(vk), Fk, X)
    dC = np.linalg.norm(X - C, axis=1)
    src = np.where(dC <= dC.min() + 0.03 * H)[0]
    g = dijkstra(adj, src)
    if not np.isfinite(g).all() or g.max() <= 0:
        return None
    s = g / g.max()
    hop = bfs(adj, src)
    rings = int(hop.max() + 1)
    # slices along s
    b = np.minimum((s * SLICES).astype(int), SLICES - 1)
    cent = [X[b == i].mean(0) if (b == i).sum() >= 2 else None for i in range(SLICES)]
    wid, ep = [np.nan] * SLICES, [np.nan] * SLICES
    cent_ok = [i for i in range(SLICES) if cent[i] is not None]
    for i in range(SLICES):
        if cent[i] is None:
            continue
        j0 = max([j for j in cent_ok if j < i], default=i)
        j1 = min([j for j in cent_ok if j > i], default=i)
        ax = cent[j1] - cent[j0]
        if np.linalg.norm(ax) < 1e-9:
            ax = X[s.argmax()] - X[s.argmin()]
        ax = ax / max(np.linalg.norm(ax), 1e-12)
        Q = X[b == i] - cent[i]
        Q = Q - np.outer(Q @ ax, ax)
        _, sv, vt = np.linalg.svd(Q, full_matrices=False)
        pr = Q @ vt.T
        wid[i] = np.ptp(pr[:, 0]) / H
        ep[i] = (np.ptp(pr[:, 1]) / H) if pr.shape[1] > 1 else 0.0
    cc = np.array([cent[i] for i in cent_ok])
    long_H = float(np.linalg.norm(np.diff(cc, axis=0), axis=1).sum() / H) if len(cc) > 1 else 0.0
    wid, ep = np.array(wid), np.array(ep)
    lmed = np.nanmedian(wid)
    root, spike = X[src].mean(0), X[np.argmax(s)]
    # ring spacing: mean g per level
    gm = np.array([g[hop == h].mean() for h in range(rings)])
    esp = np.diff(gm) / H
    n4 = max(len(esp) // 4, 1)
    # UV: orientation of V along s
    U = uv[vk]
    cv = np.corrcoef(U[:, 1], s)[0, 1] if np.ptp(U[:, 1]) > 0 else np.nan
    cu = np.corrcoef(U[:, 0], s)[0, 1] if np.ptp(U[:, 0]) > 0 else np.nan
    Uc = U - U.mean(0)
    _, _, vtu = np.linalg.svd(Uc, full_matrices=False)
    ang_v = float(np.degrees(np.arccos(min(abs(vtu[0, 1]), 1.0))))
    t3 = X[Fk]
    a3 = 0.5 * np.linalg.norm(np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0]), axis=1).sum()
    tu = U[Fk]
    auv = 0.5 * np.abs((tu[:, 1, 0] - tu[:, 0, 0]) * (tu[:, 2, 1] - tu[:, 0, 1])
                       - (tu[:, 2, 0] - tu[:, 0, 0]) * (tu[:, 1, 1] - tu[:, 0, 1])).sum()
    # free edges (welded mesh) in the island
    vw = c.inv_w[vk]
    border = float(c.border_w[vw].mean())
    # faces turned towards the strand axis (back of a ribbon, tucked-in rim)
    nf = G.normals_faces(X, Fk)
    cf = t3.mean(1)
    sf = s[Fk].mean(1)
    bi = np.minimum((sf * SLICES).astype(int), SLICES - 1)
    axe_pt = np.array([cent[i] if cent[i] is not None else X.mean(0) for i in range(SLICES)])[bi]
    to_axe = float(((nf * (cf - axe_pt)).sum(1) < 0).mean())
    # dominant bones
    od = dom[vk]
    names = c.names_os
    head = c.i_head
    part_head = float((od == head).mean())
    vals, cnts = np.unique(od, return_counts=True)
    mats = sorted({c.p["materials"][c.mf[c.f_ch[f]]]["name"] for f in fk})
    rel = lambda p: (p - C) / H
    rr, pp = rel(root), rel(spike)
    # position category (MMD frame: front is -Z)
    if pp[1] < -1.0:
        cat = "long"
    elif rr[2] < -0.15 and pp[2] < 0 and abs(rr[0]) < 0.35:
        cat = "fringe"
    elif abs(rr[0]) >= 0.35 or abs(pp[0]) >= 0.45:
        cat = "side"
    elif rr[2] > 0.15:
        cat = "back"
    else:
        cat = "outer"
    return {"island": int(k), "vertices": int(len(vk)), "faces": int(len(fk)), "materials": mats, "category": cat,
            "root_H": r3(rr), "tip_H": r3(pp), "length_H": r3(long_H),
            "width_med_H": r3(lmed), "thickness_med_H": r3(np.nanmedian(ep)),
            "flattening": r3(np.nanmedian(ep) / lmed) if lmed > 0 else None,
            "width_tip_rel": r3(np.nanmean(wid[-3:]) / lmed) if lmed > 0 else None,
            "width_root_rel": r3(np.nanmean(wid[:3]) / lmed) if lmed > 0 else None,
            "rings": rings, "vertices_per_loop": r3(len(vk) / rings, 1),
            "spacing_med_H": r3(np.median(esp)) if len(esp) else None,
            "spacing_tip_rel": r3(np.mean(esp[-n4:]) / np.median(esp)) if len(esp) > 3 and np.median(esp) > 0 else None,
            "corr_v_s": r3(cv), "corr_u_s": r3(cu), "angle_axe_uv_v_deg": r3(ang_v, 1),
            "uv_box": r3([np.ptp(U[:, 0]), np.ptp(U[:, 1])]),
            "density_texel": r3(np.sqrt(auv / a3) * c.H) if a3 > 0 else None,
            "border_free_part": r3(border), "faces_to_axe_part": r3(to_axe),
            "os_head_part": r3(part_head), "os_dominant": names[int(vals[np.argmax(cnts)])],
            "os_dominant_part": r3(cnts.max() / len(vk)), "_s": s}


# =================================================================== C. UV (overall)

def measure_uv(c):
    uv = c.t["uv"][c.vh]
    F = c.F[c.f_ch]
    loc = np.full(len(c.pos), -1)
    loc[c.vh] = np.arange(len(c.vh))
    Fl = loc[F]
    out_of = float(((uv < -1e-3) | (uv > 1 + 1e-3)).any(1).mean())
    # stacking: pixels (256²) covered by more than one island
    N = 256
    rng = np.random.default_rng(0)
    cnt = {}
    for k in np.unique(c.island_of):
        if k < 0:
            continue
        fk = np.where(c.island_of[Fl[:, 0]] == k)[0]
        T = uv[Fl[fk]]
        a = 0.5 * np.abs((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))
        n = int(min(max(a.sum() * N * N * 3, 50), 200000))
        if a.sum() <= 0:
            continue
        j = rng.choice(len(fk), n, p=a / a.sum())
        r1, r2 = rng.random(n), rng.random(n)
        sq = np.sqrt(r1)
        bb = np.stack([1 - sq, sq * (1 - r2), sq * r2], 1)
        p = np.einsum("nk,nkd->nd", bb, T[j])
        px = set(map(tuple, np.clip((np.mod(p, 1) * N).astype(int), 0, N - 1)))
        for x in px:
            cnt[x] = cnt.get(x, 0) + 1
    v = np.array(list(cnt.values())) if cnt else np.zeros(1)
    # UV symmetry: for vertices mirrored within 1 % of H, are the UVs identical?
    m = c.d_sym < SYM_TOL * c.H
    rep_raw = np.zeros(len(c.Vw), dtype=np.int64)
    rep_raw[c.inv_w] = np.arange(len(c.vh))
    pairs = np.where(m)[0]
    pairs = pairs[c.j_sym[pairs] != pairs]
    d_u = np.linalg.norm(uv[rep_raw[pairs]] - uv[rep_raw[c.j_sym[pairs]]], axis=1) if len(pairs) else np.zeros(0)
    # extra UV layers: nature of each channel
    layers = []
    ua = c.t["uv_add"][c.vh] if c.t["uv_add"].shape[1] else np.zeros((len(c.vh), 0, 4))
    N_ = c.Np
    for k in range(ua.shape[1]):
        u = ua[:, k]
        n3 = np.linalg.norm(u[:, :3], axis=1)
        nature = "other"
        if np.ptp(u, 0).max() < 1e-6:
            nature = "constant"
        elif np.allclose(n3, 1, atol=1e-2) and np.isin(np.round(u[:, 3]), [-1, 1]).all():
            perp = float(np.median(np.abs((u[:, :3] * N_).sum(1))))
            nature = "tangent" if perp < 0.05 else "unit vector"
        elif (np.abs(u[:, 2:]).max() < 1e-6):
            nature = "uv2" if (u[:, :2].min() > -0.01 and u[:, :2].max() < 1.01) else "uv2 outside [0,1]"
        elif u.min() >= -1e-3 and u.max() <= 1 + 1e-3:
            nature = "colour [0,1]"
        corr = np.nan
        if np.isfinite(c.s_glob).any():
            ok = np.isfinite(c.s_glob)
            vals = [abs(np.corrcoef(u[ok, d], c.s_glob[ok])[0, 1]) for d in range(4) if np.ptp(u[ok, d]) > 1e-9]
            corr = max(vals) if vals else np.nan
        layers.append({"canal": k + 1, "nature": nature, "min": r3(u.min(0)), "max": r3(u.max(0)),
                        "values_distinct": int(len(np.unique(np.round(u, 3), axis=0))),
                        "corr_max_with_s": r3(corr)})
    return {"outside_0_1_part": r3(out_of), "pixels_covered_part": r3(len(cnt) / N / N),
            "pixels_stacked_part": r3((v > 1).sum() / max(len(v), 1)),
            "mirror_uv_identical_part": r3((d_u < 1e-3).mean()) if len(d_u) else None,
            "mirror_pairs": int(len(d_u)), "extra_layers": layers}


# =================================================================== D. texture

def _folder_pmx(c):
    return os.path.dirname(c.path)


def _read_texture(c, name):
    if bpy is None or not name:
        return None
    if hasattr(c, "image"):          # GLB: embedded textures
        return c.image(name)
    ch = os.path.join(_folder_pmx(c), name.replace("\\", "/"))
    if not os.path.exists(ch):
        return None
    try:
        im = bpy.data.images.load(ch)
        w, h = im.size
        px = np.empty(w * h * 4, dtype=np.float32)
        im.pixels.foreach_get(px)
        bpy.data.images.remove(im)
        return px.reshape(h, w, 4)
    except Exception as e:  # unreadable texture: logged, not fatal
        log("unreadable texture", ch, e)
        return None


def _L(rgb):
    """CIE lightness L* of an sRGB [0,1] colour."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(Y > 216 / 24389, np.cbrt(Y), (24389 / 27 * Y + 16) / 116)
    return 116 * f - 16


def _sample(img, uv, Fl, value, rng, n_per_face=SAMPLE_TEX):
    """Points drawn in each face (random barycentric coordinates): colour (RGBA) and an
    interpolated value (for example s)."""
    T = uv[Fl]
    n = len(Fl) * n_per_face
    j = np.repeat(np.arange(len(Fl)), n_per_face)
    r1, r2 = rng.random(n), rng.random(n)
    sq = np.sqrt(r1)
    bb = np.stack([1 - sq, sq * (1 - r2), sq * r2], 1)
    p = np.einsum("nk,nkd->nd", bb, T[j])
    val = (bb * value[Fl[j]]).sum(1)
    h, w = img.shape[:2]
    x = (np.mod(p[:, 0], 1) * w).astype(int).clip(0, w - 1)
    y = ((1 - np.mod(p[:, 1], 1)) * h).astype(int).clip(0, h - 1)
    return img[y, x], val, j


def measure_texture(c):
    if bpy is None:
        return {"lu": False, "reason": "without Blender"}
    rng = np.random.default_rng(1)
    loc = np.full(len(c.pos), -1)
    loc[c.vh] = np.arange(len(c.vh))
    uv = c.t["uv"][c.vh]
    out = {"lu": True, "per_material": {}}
    s = c.s_glob.copy()
    profiles = []
    for i in c.mats_hair:
        m = c.p["materials"][i]
        texname = c.p["textures"][m["texture"]] if m["texture"] >= 0 else None
        img = _read_texture(c, texname)
        if img is None:
            out["per_material"][m["name"]] = {"texture": texname, "read": False}
            continue
        fk = np.where((c.mf == i) & c.mask_faces)[0]
        Fl = loc[c.F[fk]]
        okf = np.isfinite(s[Fl]).all(1)
        collar, sv, jf = _sample(img, uv, Fl[okf], np.nan_to_num(s), rng)
        if not len(collar):
            out["per_material"][m["name"]] = {"texture": texname, "read": True, "samples": 0}
            continue
        L = _L(np.clip(collar[:, :3], 0, 1))
        A = collar[:, 3]
        island = c.island_of[Fl[okf][jf, 0]]
        bins = np.minimum((sv * 10).astype(int), 9)
        prof_L = [r3(np.median(L[bins == b]), 1) if (bins == b).any() else None for b in range(10)]
        prof_A = [r3(np.mean(A[bins == b]), 2) if (bins == b).any() else None for b in range(10)]
        # per island: root -> tip slope and light bump (painted highlight)
        slopes, bumps = [], []
        for k in np.unique(island):
            mk = island == k
            if mk.sum() < 200:
                continue
            bk = bins[mk]
            pr = np.array([np.median(L[mk][bk == b]) if (bk == b).sum() >= 5 else np.nan for b in range(10)])
            okb = np.isfinite(pr)
            if okb.sum() < 5:
                continue
            xs = np.arange(10)[okb]
            a_, b_ = np.polyfit(xs, pr[okb], 1)
            slopes.append(pr[okb][-1] - pr[okb][0])
            bumps.append(float((pr[okb] - (a_ * xs + b_)).max()))
            profiles.append(pr.tolist())
        # lines painted on the texture: lightness difference between neighbours (whole texture, under the UVs)
        h, w = img.shape[:2]
        Lt = _L(np.clip(img[..., :3], 0, 1))
        grad = np.hypot(np.diff(Lt, axis=0)[:, :-1], np.diff(Lt, axis=1)[:-1, :])
        px = (np.clip((np.mod(uv[np.unique(Fl)], 1) * [w, h]).astype(int), 0, [w - 1, h - 1]))
        out["per_material"][m["name"]] = {
            "texture": texname, "read": True, "size": [int(w), int(h)],
            "L_med": r3(np.median(L), 1), "L_p5_p95": r3(np.percentile(L, [5, 95]), 1),
            "alpha_avg": r3(A.mean()), "alpha_under_0_9_part": r3((A < 0.9).mean()),
            "profile_L_s": prof_L, "profile_alpha_s": prof_A,
            "slope_L_root_tip": q(slopes, n=1), "bright_bump_L": q(bumps, n=1),
            "islands_tip_plus_sombre_part": r3(np.mean(np.array(slopes) < -3)) if slopes else None,
            "gradient_L_texture_p90": r3(np.percentile(grad, 90), 1)}
    # overlays (highlight): texture, opaque coverage, position along s
    for i, rl in c.roles.items():
        if rl != "texture_layer":
            continue
        m = c.p["materials"][i]
        texname = c.p["textures"][m["texture"]] if m["texture"] >= 0 else None
        img = _read_texture(c, texname)
        fk = np.where((c.mf == i) & c.mask_faces)[0]
        vk = np.unique(c.F[fk])
        d, j = kd_plus_near(c.pos[c.vh], c.pos[vk])
        sv = np.full(len(c.pos), np.nan)
        sv[vk] = s[j]
        info = {"texture": texname, "sphere": c.p["textures"][m["sphere"]] if m["sphere"] >= 0 else None,
                "sphere_mode": m["sphere_mode"]}
        if img is not None:
            loc2 = np.full(len(c.pos), -1)
            loc2[vk] = np.arange(len(vk))
            Fl2 = loc2[c.F[fk]]
            collar, s2, _ = _sample(img, c.t["uv"][vk], Fl2, np.nan_to_num(sv[vk]), rng, 8)
            A = collar[:, 3]
            L = _L(np.clip(collar[:, :3], 0, 1))
            vis = A > 0.5
            info.update({"size": list(img.shape[1::-1]), "alpha_avg": r3(A.mean()), "visible_part": r3(vis.mean()),
                         "L_visible_med": r3(np.median(L[vis]), 1) if vis.any() else None,
                         "s_visible": q(s2[vis]) if vis.any() else None})
        out.setdefault("texture_layers", {})[m["name"]] = info
    out["textures_model"] = sorted(set(c.p["textures"]))
    return out


# =================================================================== E. physics

def measure_physics(c):
    H, C = c.H, c.C
    p = c.p
    O, RG, J = p["bones"], p["rigid_bodies"], p["joints"]
    names = c.names_os
    oc_all = c.mask_os
    lest = np.array([("錘" in n or "锤" in n) for n in names])   # "髮1錘", "Ctr_HairA_01錘_L"
    oc = oc_all & ~lest          # links: hairstyle bones except the weights
    children = {}
    for i, o in enumerate(O):
        children.setdefault(o["parent"], []).append(i)
    rig_os = {}
    for k, r in enumerate(RG):
        rig_os.setdefault(r["bones"], []).append(k)
    j_per_b = {}
    for jj in J:
        j_per_b.setdefault(jj["b"], []).append(jj)
    roots = [i for i in np.where(oc)[0] if O[i]["parent"] < 0 or not oc[O[i]["parent"]]]
    W_os, W_p = c.t["bones"][c.vh], c.t["weights"][c.vh]
    dom = c.dom
    body = [k for k, r in enumerate(RG) if r["mode"] == 0 and r["bones"] >= 0 and names[r["bones"]] in OS_BODY]
    head_rg = [k for k in body if names[RG[k]["bones"]] == "頭"]
    seg_body = {k: segment_rigid(RG[k]) for k in body}

    def plus_long(i):
        ch = [i]
        while True:
            e = [x for x in children.get(ch[-1], []) if oc[x]]
            if not e:
                return ch
            ch.append(max(e, key=lambda x: len(plus_long(x))))

    chains = []
    for r0 in roots:
        ch = plus_long(r0)
        under = [r0]
        k = 0
        while k < len(under):
            under += [x for x in children.get(under[k], []) if oc_all[x]]
            k += 1
        ballasts = [x for x in under if lest[x]]
        m_lest = [RG[rig_os[x][0]]["mass"] for x in ballasts if rig_os.get(x)]
        under_chain_links = [x for x in under if not lest[x]]
        pos = np.array([O[i]["pos"] for i in ch])
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1) / H
        rgs = [rig_os.get(i, []) for i in ch]
        r1 = [RG[x[0]] if x else None for x in rgs]
        dyn = [rr is not None and rr["mode"] in RIG_DYN for rr in r1]
        first = next((i for i, d_ in enumerate(dyn) if d_), None)
        masses = [rr["mass"] if rr else None for rr in r1]
        rapp = [masses[i + 1] / masses[i] for i in range(len(masses) - 1)
                if masses[i] and masses[i + 1] and dyn[i] and dyn[i + 1]]
        # joints attached to the chain's bodies
        jts = []
        for rr in rgs:
            if not rr:
                continue
            for jj in j_per_b.get(rr[0], []):
                jts.append(jj)
        lim = [[np.degrees(jj["rot_min"][a]), np.degrees(jj["rot_max"][a])] for jj in jts for a in range(3)]
        # vertices carried by the chain (dominant bone in the subtree)
        under_set = np.isin(dom, under)
        n_v = int(under_set.sum())
        # body radius / strand half-width at that bone
        reports, gaps = [], []
        for i, rr in zip(ch, r1):
            vv = c.pos[c.vh][dom == i]
            if rr is None or len(vv) < 6:
                continue
            a = np.asarray(O[i]["pos"])
            e = [x for x in children.get(i, []) if oc[x]]
            b = np.asarray(O[e[0]]["pos"]) if e else (a + np.asarray(O[i]["tail_offset"] or (0, -1e-3, 0)))
            ax = b - a
            if np.linalg.norm(ax) < 1e-9:
                continue
            ax /= np.linalg.norm(ax)
            rel = vv - a
            perp = rel - np.outer(rel @ ax, ax)
            dp = np.linalg.norm(perp, axis=1)
            half = np.percentile(dp, 90)
            _, _, radius = segment_rigid(rr)
            if half > 0:
                reports.append(radius / half)
            gaps.append(np.linalg.norm(perp.mean(0)) / H)
        # collisions with the head and the body
        coll_head = [collide(rr, RG[k]) for rr in r1 if rr is not None and rr["mode"] in RIG_DYN for k in head_rg]
        # at rest: overlap of the chain's bodies with those of the head and body
        rec, rec_free = [], []
        for rr in r1:
            if rr is None or rr["mode"] not in RIG_DYN:
                continue
            a, b, ra = segment_rigid(rr)
            for k in body:
                a2, b2, r2 = seg_body[k]
                d = (dist_segments(a, b, a2, b2) - ra - r2) / H
                (rec if collide(rr, RG[k]) else rec_free).append(d)
        # chain root relative to the skull; reach of the carried vertices along the chain
        root_skull = (np.linalg.norm(pos[0] - C) - c.R) / H
        vp = c.pos[c.vh][under_set]
        reach = float(np.linalg.norm(vp - pos[0], axis=1).max() / H) if len(vp) else None
        # worst pose within the joint limits: each link fully rotated, in 8 directions
        worst = _worst_pose(c, ch, r1, jts, seg_body, head_rg)
        chains.append({
            "racine": names[r0], "parent": names[O[r0]["parent"]] if O[r0]["parent"] >= 0 else None,
            "chain_links": len(ch), "os_under_tree": len(under_chain_links), "branched": len(under_chain_links) > len(ch),
            "ballasts": len(ballasts), "mass_ballasts": r3(sum(m_lest)) if m_lest else 0.0,
            "length_H": r3(seg.sum()), "spacing_H": r3(seg), "spacing_cv": r3(seg.std() / seg.mean()) if len(seg) and seg.mean() > 0 else None,
            "first_dynamic": first, "modes": [rr["mode"] if rr else None for rr in r1],
            "shapes": [rr["shape"] if rr else None for rr in r1],
            "masses": r3(masses), "report_mass_med": r3(np.median(rapp)) if rapp else None,
            "damping_disp": r3([rr["damping_disp"] if rr else None for rr in r1]),
            "damping_rot": r3([rr["damping_rot"] if rr else None for rr in r1]),
            "restitution": r3(sorted({rr["restitution"] for rr in r1 if rr})), "friction": r3(sorted({rr["friction"] for rr in r1 if rr})),
            "radii_H": r3([segment_rigid(rr)[2] / H if rr else None for rr in r1]),
            "groups": sorted({rr["group"] for rr in r1 if rr}),
            "ignores": sorted({tuple(ignores(rr)) for rr in r1 if rr}),
            "touched_head": (sum(coll_head) / len(coll_head)) if coll_head else None,
            "joints": len(jts), "limits_rot_deg": r3(lim, 1) if lim else [],
            "amplitude_max_deg": r3([max(np.degrees(jj["rot_max"][a] - jj["rot_min"][a]) for jj in jts) for a in range(3)], 1) if jts else None,
            "joint_rot_zero": all(np.allclose(jj["rot"], 0, atol=1e-3) for jj in jts) if jts else None,
            "springs_rot": r3(sorted({tuple(np.round(jj["spring_rot"], 2)) for jj in jts})) if jts else [],
            "springs_disp": r3(sorted({tuple(np.round(jj["spring_disp"], 2)) for jj in jts})) if jts else [],
            "translation_free": any(np.ptp([jj["disp_min"], jj["disp_max"]], axis=0).max() > 1e-6 for jj in jts) if jts else None,
            "vertices_worn": n_v, "root_dist_skull_H": r3(root_skull), "reach_vertices_H": r3(reach),
            "chain_on_reach": r3(seg.sum() / reach) if reach else None,
            "overlap_rest_free_min_H": r3(min(rec_free)) if rec_free else None, "radius_on_half_width": q(reports, (10, 50, 90), 2),
            "bone_center_gap_H": q(gaps), "overlap_rest_min_H": r3(min(rec)) if rec else None,
            "worst_pose": worst})
    # weights of the hair vertices
    nb = (W_p > 1e-3).sum(1)
    head = c.i_head
    static = (nb == 1) & (W_os[:, 0] == head)
    typ = c.t["weight_type"][c.vh]
    # two influences: parent and child?
    pe = []
    for i in np.where(nb == 2)[0]:
        a, b = W_os[i][np.argsort(-W_p[i])[:2]]
        pe.append(O[a]["parent"] == b or O[b]["parent"] == a)
    # strands and chains: monotonicity of the dominant bone's rank along s
    # rank within the whole subtree (depth from the root)
    depth = np.full(len(O), -1)
    for r0 in roots:
        pile = [(r0, 0)]
        while pile:
            i, d_ = pile.pop()
            depth[i] = d_
            pile += [(x, d_ + 1) for x in children.get(i, []) if oc[x]]
    mono = []
    for m in c.locks:
        vk = np.where(c.island_of == m["island"])[0]
        pr = depth[dom[vk]]
        okp = pr >= 0
        if okp.sum() < 10 or np.ptp(pr[okp]) == 0:
            continue
        sv = c.s_glob[vk][okp]
        rk = pr[okp]
        rs = np.argsort(np.argsort(sv))
        rr_ = np.argsort(np.argsort(rk + 1e-6 * sv))
        mono.append(float(np.corrcoef(rs, rr_)[0, 1]))
        m["correlation_s_depth_os"] = r3(mono[-1])
        m["chain_part"] = r3(okp.mean())
    # collision bodies of the head and upper body
    collar = []
    for k in body:
        r = RG[k]
        a, b, rad = seg_body[k]
        collar.append({"name": r["name"], "bones": names[r["bones"]], "shape": PMX.SHAPES_RIGID[r["shape"]],
                    "rayon_H": r3(rad / H), "length_H": r3(np.linalg.norm(b - a) / H), "group": r["group"],
                    "ignores": ignores(r)})
    # hair inside the head collision bodies at rest (the skull itself)
    Ph = c.pos[c.vh]
    inside_head = None
    if head_rg:
        a, b, rad = seg_body[head_rg[0]]
        dp = dist_point_segment(Ph, a, b)
        inside_head = float((dp < rad).mean())
        # gap between the head body and the underside of the cap
        top = Ph[:, 1] > c.y_eyes
        gap_cal = q((dp[top] - rad) / H)
    else:
        gap_cal = None
    dyn_rg = [r for r in RG if r["bones"] >= 0 and oc[r["bones"]] and r["mode"] in RIG_DYN]
    lest_rg = [r for r in RG if r["bones"] >= 0 and oc_all[r["bones"]] and lest[r["bones"]]]
    groups = {}
    for r in dyn_rg:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    itself = [not ((r["mask"] >> r["group"]) & 1) for r in dyn_rg]
    return {
        "hairstyle_bones": int(oc.sum()), "chains": chains, "count_chains": len(chains),
        "rigid_bodies_dynamic": len(dyn_rg), "groups_hair": groups,
        "ballasts": {"count": len(lest_rg), "modes": sorted({r["mode"] for r in lest_rg}),
                  "masks": sorted({r["mask"] for r in lest_rg}), "masses": q([r["mass"] for r in lest_rg])},
        "ignore_son_group_part": r3(np.mean(itself)) if itself else None,
        "body_collision": collar, "hair_in_rigid_head_part": r3(inside_head),
        "cap_gap_rigid_head_H": gap_cal,
        "weights": {"influences": {str(k): int((nb == k).sum()) for k in range(1, 5)},
                  "types": {PMX.TYPES_WEIGHT[k]: int((typ == k).sum()) for k in np.unique(typ)},
                  "static_head_part": r3(static.mean()),
                  "two_os_parent_child_part": r3(np.mean(pe)) if pe else None,
                  "monotony_s_os": q(mono)}}


def _worst_pose(c, ch, r1, jts, seg_body, head_rg):
    """Chain rotated to the maximum of its limits (X then Z in the world frame, joints at zero
    angle) in 8 directions: smallest gap (in H) between its bodies and the head body, and
    between its bodies and the upper body. Kinematic approximation: no gravity, no contacts;
    it only tells whether the limits are enough to keep the strand out of the skull."""
    if not head_rg or not jts:
        return None
    O = c.p["bones"]
    pos = np.array([O[i]["pos"] for i in ch], float)
    limits = []
    for jj in jts[:len(ch)]:
        limits.append((np.array(jj["rot_min"]), np.array(jj["rot_max"])))
    while len(limits) < len(ch):
        limits.append(limits[-1])
    res = {"head_min_H": np.inf, "body_min_H": np.inf}
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        dx, dz = np.cos(ang), np.sin(ang)
        P = pos.copy()
        Join = np.eye(3)
        for k in range(len(ch) - 1):
            lo, hi = limits[k]
            ax = hi[0] if dx >= 0 else lo[0]
            az = hi[2] if dz >= 0 else lo[2]
            Rk = rot_mmd((ax * abs(dx), 0.0, az * abs(dz)))
            Join = Join @ Rk
            off = pos[k + 1:] - pos[k]
            P[k + 1:] = P[k] + (Join @ (pos[k + 1:] - pos[k]).T).T if k == 0 else P[k] + (Rk @ (P[k + 1:] - P[k]).T).T
        for k in range(len(ch) - 1):
            rr = r1[k]
            if rr is None or rr["mode"] not in RIG_DYN:
                continue
            ra = segment_rigid(rr)[2]
            for kk, (a2, b2, r2) in seg_body.items():
                d = (dist_segments(P[k], P[k + 1], a2, b2) - ra - r2) / c.H
                if not collide(rr, c.p["rigid_bodies"][kk]):
                    the_key = "sans_contact_min_H"
                else:
                    the_key = "head_min_H" if kk in head_rg else "body_min_H"
                res[the_key] = min(res.get(the_key, np.inf), d)
    return {k: r3(v) for k, v in res.items()}


# =================================================================== F. morphs

def measure_morphs(c):
    vh = set(c.vh.tolist())
    out = []
    for m in c.p["morphs"]:
        if m["type"] != 1 or not m["offsets"]:
            continue
        idx = [d[0] for d in m["offsets"] if d[0] in vh and np.linalg.norm(d[1]) > 1e-6]
        if idx:
            out.append({"name": m["name"], "vertices_hair": len(idx)})
    return out


# =================================================================== per-character summary

def resume_locks(locks):
    def s(the_key, filter_=None):
        return q([m[the_key] for m in locks if (filter_ is None or filter_(m)) and m.get(the_key) is not None])
    cats = {}
    for m in locks:
        cats[m["category"]] = cats.get(m["category"], 0) + 1
    dyn = lambda m: m["os_head_part"] is not None and m["os_head_part"] < 0.5
    return {"count": len(locks), "categories": cats,
            "dynamic": int(sum(1 for m in locks if dyn(m))),
            "length_H": s("length_H"), "width_H": s("width_med_H"), "thickness_H": s("thickness_med_H"),
            "flattening": s("flattening"), "tip_rel": s("width_tip_rel"),
            "rings": s("rings"), "vertices_per_loop": s("vertices_per_loop"),
            "spacing_H": s("spacing_med_H"), "spacing_tip_rel": s("spacing_tip_rel"),
            "corr_v_s": s("corr_v_s"), "corr_u_s": s("corr_u_s"), "angle_axe_uv_v_deg": s("angle_axe_uv_v_deg"),
            "density_texel_cv": r3(np.std([m["density_texel"] for m in locks if m["density_texel"]]) /
                                   np.mean([m["density_texel"] for m in locks if m["density_texel"]])) if locks else None,
            "border_free": s("border_free_part"), "faces_to_axe": s("faces_to_axe_part"),
            "os_dominant_part_dyn": s("os_dominant_part", dyn),
            "monotony_s_os": s("correlation_s_depth_os", dyn)}


def measure(lib, name, c=None, springs=None):
    """Measurements of a hairstyle: PMX from the library (library, name) or an already loaded
    hairstyle (c, for example CoiffureGLB); springs: Godot scenes (.tscn) or JSON of the bone
    springs."""
    t0 = time.time()
    c = c if c is not None else Hairstyle(lib, name)
    out = {"person": name, "pmx": c.path, "H": r3(c.H, 4), "iod_H": r3(c.D.iod / c.H),
           "rayon_crane_H": r3(c.R / c.H), "landmark": "MMD: Y up, face towards -Z; lengths in H"}
    out["mesh"] = measure_mesh(c)
    locks = measure_locks(c)
    out["uv"] = measure_uv(c)
    out["physics"] = measure_physics(c)
    out["locks_resume"] = resume_locks(locks)
    out["locks"] = locks
    out["texture"] = measure_texture(c)
    out["morphs"] = measure_morphs(c)
    if isinstance(c, HairstyleGlb):
        out["glb"] = {"files": c.paths, "unit_to_mmd": r3(c.unit, 5), "skull": c.landmark_skull,
                      "head_bone": c.names_os[c.i_head] if c.i_head >= 0 else None,
                      "materials_hair": [c.p["materials"][i]["name"] for i in c.mats_hair],
                      "roles": {c.p["materials"][i]["name"]: r for i, r in c.roles.items()},
                      "hairstyle_bones": [c.names_os[i] for i in np.where(c.mask_os)[0]],
                      "weld": c.weld}
    if springs:
        import springs as RS
        out["springs"] = RS.measure(c, springs)
    out["duration_s"] = round(time.time() - t0, 1)
    return out


# =================================================================== overall table

# Rows of the overall table: (label, value extracted from a character's JSON). The French labels
# are written as keys in ensemble.json and as row titles in tableau.md: they are identifiers,
# do not translate them. Meaning, in order: hair faces, share of the model, overlay faces, parts,
# share of the largest part, boundary edges (share), strands (islands >= 12 vertices), median
# length (H), median width (H), median flatness, median tip/width, median vertices per ring,
# median V-s correlation, normals: median geometric angle (deg), normals: median radial angle
# (deg), symmetry (share within 1 % H), identical mirrored UVs, chains, median links, median bone
# spacing (H), dynamic rigid bodies, weights (錘), median max X range (deg), max Y range (deg),
# median child/parent mass ratio, median rigid radius / half-width, min rest overlap (H,
# colliding pairs), worst pose min head gap (H), median s / bone depth monotonicity, ignores its
# own group, static vertices (頭 only).

LINES = [
    ("hair faces", lambda r: r["mesh"]["budget"]["faces_hair"]),
    ("share of the model", lambda r: r["mesh"]["budget"]["part_faces"]),
    ("overlay faces", lambda r: r["mesh"]["budget"]["faces_texture_layers"]),
    ("pieces", lambda r: r["mesh"]["parts"]["count"]),
    ("share of the largest piece", lambda r: r["mesh"]["parts"]["part_plus_large"]),
    ("border edges (share)", lambda r: r["mesh"]["border_edges_part"]),
    ("locks (islands >= 12 v.)", lambda r: r["locks_resume"]["count"]),
    ("median length (H)", lambda r: _p(r["locks_resume"]["length_H"])),
    ("median width (H)", lambda r: _p(r["locks_resume"]["width_H"])),
    ("median flattening", lambda r: _p(r["locks_resume"]["flattening"])),
    ("median tip/width", lambda r: _p(r["locks_resume"]["tip_rel"])),
    ("median vertices/loop", lambda r: _p(r["locks_resume"]["vertices_per_loop"])),
    ("median corr V-s", lambda r: _p(r["locks_resume"]["corr_v_s"])),
    ("normals: median geometric angle (deg)", lambda r: _p(r["mesh"]["normals"]["angle_geometric_deg"])),
    ("normals: median radial angle (deg)", lambda r: _p(r["mesh"]["normals"]["angle_radial_deg"])),
    ("symmetry (share at 1 % H)", lambda r: r["mesh"]["symmetry"]["part_1pc"]),
    ("identical mirror UVs", lambda r: r["uv"]["mirror_uv_identical_part"]),
    ("chains", lambda r: r["physics"]["count_chains"]),
    ("median links", lambda r: _med([ch["chain_links"] for ch in r["physics"]["chains"]])),
    ("median bone spacing (H)", lambda r: _med([e for ch in r["physics"]["chains"] for e in (ch["spacing_H"] or [])])),
    ("dynamic rigid bodies", lambda r: r["physics"]["rigid_bodies_dynamic"]),
    ("weights (錘)", lambda r: r["physics"]["ballasts"]["count"]),
    ("median max X range (deg)", lambda r: _med([ch["amplitude_max_deg"][0] for ch in r["physics"]["chains"] if ch["amplitude_max_deg"]])),
    ("max Y range (deg)", lambda r: max([ch["amplitude_max_deg"][1] for ch in r["physics"]["chains"] if ch["amplitude_max_deg"]], default=None)),
    ("median child/parent mass ratio", lambda r: _med([ch["report_mass_med"] for ch in r["physics"]["chains"]])),
    ("median rigid radius / half-width", lambda r: _med([ch["radius_on_half_width"]["p50"] for ch in r["physics"]["chains"] if ch["radius_on_half_width"]])),
    ("min rest overlap (H, colliding pairs)", lambda r: min([ch["overlap_rest_min_H"] for ch in r["physics"]["chains"] if ch["overlap_rest_min_H"] is not None], default=None)),
    ("worst pose min head gap (H)", lambda r: min([ch["worst_pose"]["head_min_H"] for ch in r["physics"]["chains"] if ch["worst_pose"] and ch["worst_pose"].get("head_min_H") is not None], default=None)),
    ("median s / bone depth monotonicity", lambda r: _p(r["physics"]["weights"]["monotony_s_os"])),
    ("ignores its own group", lambda r: r["physics"]["ignore_son_group_part"]),
    ("static vertices (頭 only)", lambda r: r["physics"]["weights"]["static_head_part"]),
]


def _p(d):
    return d["p50"] if d else None


def _med(v):
    v = [x for x in v if x is not None]
    return r3(np.median(v)) if v else None


def array(output):
    res = []
    for f in sorted(os.listdir(output)):
        if f.endswith(".json") and f != "ensemble.json":
            with open(os.path.join(output, f), encoding="utf-8") as fh:
                res.append(json.load(fh))
    names = [r["person"] for r in res]
    lines = ["| measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    ens = {}
    for title, fn in LINES:
        vals = []
        for r in res:
            try:
                vals.append(fn(r))
            except (KeyError, TypeError, ValueError):
                vals.append(None)
        ens[title] = dict(zip(names, vals))
        lines.append("| %s | %s |" % (title, " | ".join("—" if v is None else str(v) for v in vals)))
    write_json(os.path.join(output, "ensemble.json"), ens)
    tmp = os.path.join(output, "tableau.md.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, os.path.join(output, "tableau.md"))
    log("table:", len(res), "characters")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", dest="lib", default=os.path.join(R.path("paths.data"), "face-library"))
    ap.add_argument("--output", dest="output", default=os.path.join(R.path("paths.data"), "hair"))
    ap.add_argument("--only", dest="alone", default=None)
    ap.add_argument("--all", dest="all_", action="store_true", help="the whole library (index.json)")
    ap.add_argument("--table", dest="array", default=None, help="only rebuild the table of this folder")
    ap.add_argument("--glb", action="append", default=None,
                    help="model GLB (repeat for separate strands, same skeleton): measures this model only")
    ap.add_argument("--name", dest="name", default=None, help="name of the GLB character (file <name>.json)")
    ap.add_argument("--hair-mats", dest="mats_hair", default=None, help="hair material patterns (comma-separated; default: PATTERNS_HAIR_GLB)")
    ap.add_argument("--skin", dest="skin", default=None, help="face material patterns (default: those of head_glb)")
    ap.add_argument("--eyes", dest="eyes", default=None, help="eye material patterns (default: those of head_glb)")
    ap.add_argument("--springs", dest="springs", action="append", default=None,
                    help="Godot scene (.tscn) or JSON of the SpringBoneSimulator3D nodes (repeatable)")
    a = ap.parse_args(argv)
    if a.array:
        array(a.array)
        return
    os.makedirs(a.output, exist_ok=True)
    if a.glb:
        name = a.name or os.path.splitext(os.path.basename(a.glb[0]))[0]
        offset = lambda x: tuple(m.strip().lower() for m in x.split(",")) if x else None
        c = HairstyleGlb(a.glb, name, offset(a.mats_hair), offset(a.skin), offset(a.eyes))
        r = measure(None, name, c, a.springs)
        write_json(os.path.join(a.output, name + ".json"), r)
        log(name, "done in %.0f s" % r["duration_s"], "->", os.path.join(a.output, name + ".json"))
        return
    if a.all_:
        with open(os.path.join(a.lib, "index.json"), encoding="utf-8") as fh:
            names = sorted(json.load(fh)["hairstyle"])
    elif a.alone:
        names = a.alone.split(",")
    else:
        names = list(CHOICE)
    for name in names:
        try:
            r = measure(a.lib, name)
        except Exception as e:  # one failing character does not stop the others
            import traceback
            traceback.print_exc()
            log(name, "FAILED", e)
            continue
        write_json(os.path.join(a.output, name + ".json"), r)
        log(name, "done in %.0f s" % r["duration_s"])
    array(a.output)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

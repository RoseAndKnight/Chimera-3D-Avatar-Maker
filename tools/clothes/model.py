"""Loads an official PMX for garment measurement: Blender frame in metres, welding,
parts, bone regions, isolation of a garment from its accessories.

Frame: MMD (x, y, z) -> (x, z, y) × 0.08 m,
facing -Y, +Z up, character's left on +X; faces flipped to [0, 2, 1] (swapping
axes changes handedness). Lengths in metres in computations, reported in mm.

Numpy only; mathutils (Blender's Python) for BVHs, imported on demand.
"""
import json
import os
import re
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATLAS = os.path.join(HERE, "..", "atlas")
if ATLAS not in sys.path:
    sys.path.insert(0, ATLAS)
sys.path.append(os.path.join(HERE, ".."))
import settings as R  # noqa: E402
import games as games  # noqa: E402

import pmx as PMX  # noqa: E402
import primitives as P  # noqa: E402

SCALE = 0.08
LIBRARY = os.path.join(R.path("paths.data"), "face-library")

# Materials that are never clothing: face, eyes, hair (first character or full name)
NON_GARMENT = re.compile(r"^(颜|口|舌|齿|目|白目|眉|睫|二重|瞳|痣|表情|眼角|髮|髪|发|前发|后发|前髮|后髮|前髪|后髪|侧发)")
SKIN = re.compile(r"^(肌|手$|手2$|脖$|指甲)")
# Accessory materials (surveyed on the library's 50 PMX files, inventory of 2026-11-07):
# ornaments 饰, metal 金属, weapons, glasses, jewellery, screens, wings, tail, plush toys, bags, tools…
ACCESSORY_MAT = re.compile(r"饰|金属|武器|刀|墨镜|镜|宝石|珠宝|屏幕|翅膀|翼|尾|熊|狸猫|邦布|电池|耳机|包|工具|手雷|"
                            r"面具|面罩|眼罩|光环|绳|锁链|拉链|耳|戒指|指甲|頭巾|头饰|发饰|髮飾|髮饰|带$|^带|挂|"
                            r"Weapon|Sticker|透明罩|新規材質|帽|角|翅")
# Accessory bones (in addition to the "accessory" family of primitives.FAMILIES): any bone that carries an object
ACCESSORY_OS = re.compile(r"坠|珠|刀|扣|结|穗|武器|鞘|錘|铃|挂|链|饰|包|环$|铃|耳机|电池|领带|带\d|尾")


def to_blender(p):
    p = np.asarray(p, dtype=np.float64)
    return p[..., [0, 2, 1]] * SCALE


SOURCES = games.active().path("mmd_extracted")   # official PMX files of the active game (alternate outfits outside the library)


def model_of(character, pmx=None):
    """Model of a library character, or of an official PMX outside the library (alternate
    outfits): `pmx` relative to SOURCES, or an absolute path."""
    return Model(character, path=os.path.join(SOURCES, pmx) if pmx else None)


def path_pmx(character, lib=LIBRARY):
    with open(os.path.join(lib, character, "record.json"), encoding="utf-8") as fh:
        return json.load(fh)["pmx"]


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
        return np.array([self.found(i) for i in range(len(self.p))])


def components(n, edges):
    """Connected-component label of each vertex (vectorised union-find by propagation)."""
    lab = np.arange(n)
    a, b = edges[:, 0], edges[:, 1]
    while True:
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def edges_faces(F):
    """Unique (sorted) edges, face count per edge, edge of each corner (f, 3)."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, inv, c = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    return u, c, inv.ravel().reshape(3, -1).T


def normals_faces(pos, F):
    n = np.cross(pos[F[:, 1]] - pos[F[:, 0]], pos[F[:, 2]] - pos[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-20)[:, None], a / 2


class Model:
    """An official PMX in the Blender frame (metres)."""

    def __init__(self, character, lib=LIBRARY, path=None):
        self.character = character
        self.path = path or path_pmx(character, lib)
        self.p = p = PMX.read(self.path)
        t = PMX.to_arrays(p)
        self.pos = to_blender(t["pos"])
        self.nor = t["nor"][:, [0, 2, 1]]
        self.uv = t["uv"]
        self.uv_add = t["uv_add"]
        self.type_weight = t["weight_type"]
        self.idx = t["bones"]
        self.weight = t["weights"]
        self.outline = t["contour"]
        self.F = t["faces"][:, [0, 2, 1]]
        self.mf = t["mat_face"]
        self.mats = p["materials"]
        self.names_mat = [m["name"] for m in p["materials"]]
        self.textures = p["textures"]
        bones = p["bones"]
        self.names_os = [o["name"] for o in bones]
        self.per = np.array([o["parent"] for o in bones])
        heads = to_blender([o["pos"] for o in bones])
        tails = np.zeros_like(heads)
        for i, o in enumerate(bones):
            if o["tail_bone"] is not None:
                q = o["tail_bone"]
                tails[i] = heads[q] if 0 <= q < len(bones) else heads[i]
            else:
                tails[i] = heads[i] + to_blender(o["tail_offset"])
        self.heads, self.tails = heads, tails
        src = np.array([o["addition"]["source"] if o["addition"] else -1 for o in bones])
        fac = np.array([o["addition"]["factor"] if o["addition"] else 0.0 for o in bones])
        arot = np.array([bool(o["addition"] and o["addition"]["rotation"]) for o in bones])
        adep = np.array([bool(o["addition"] and o["addition"]["displacement"]) for o in bones])
        self.sq = P.Skeleton(self.names_os, heads, tails, self.per, src, fac, arot, adep)
        family_man = P.load_families().get("bones", {})
        self.family_os = [P.family_os(n, family_man) for n in self.names_os]
        self.region_os = [P.region_os(n, f) for n, f in zip(self.names_os, self.family_os)]
        self.rigid_bodies = p["rigid_bodies"]
        self.joints = p["joints"]
        # welding: coincident vertices (UV seams, split normals)
        the_key = np.round(self.pos / 1e-6).astype(np.int64)
        _, self.sv, inv = np.unique(the_key, axis=0, return_index=True, return_inverse=True)
        self.s = inv.ravel()                       # raw vertex -> welded vertex
        self.pos_s = self.pos[self.sv]
        self.Fs = self.s[self.F]                   # welded faces
        self.nf, self.area = normals_faces(self.pos, self.F)
        self.dom = P.os_dominant(self.idx, self.weight)
        self.H_corps = float(self.pos[:, 2].max() - self.pos[:, 2].min())

    # ------------------------------------------------------------------ material roles

    def role_material(self, i):
        n = self.names_mat[i]
        if self.mats[i]["n_indices"] == 0:
            return "empty"
        if NON_GARMENT.match(n):
            return "head"
        if SKIN.match(n):
            return "skin"
        if n.endswith("+") or "+" in n:
            return "texture_layer"
        if ACCESSORY_MAT.search(n):
            return "accessory"
        return "outfit"

    def faces_role(self, roles):
        r = np.array([self.role_material(i) for i in range(len(self.mats))])
        return np.isin(r[self.mf], roles)

    def os_accessory(self, b):
        if b < 0:
            return False
        n = self.names_os[b]
        return self.family_os[b] == "accessory" or bool(ACCESSORY_OS.search(n))

    def region_ancestor(self, b):
        """Body region of a bone; for a clothing or accessory bone, that of the first
        ancestor that belongs to the body."""
        body = ("trunk_top", "chest", "belly", "pelvis", "buttocks", "neck", "head", "arm",
                 "forearm", "hand", "finger", "thigh", "leg", "foot")
        k = b
        while k >= 0:
            if self.region_os[k] in body:
                return self.region_os[k]
            k = self.per[k]
        return "without"

    # ------------------------------------------------------------------ parts

    def parts(self, mask_faces):
        """Welded parts of a set of faces: list of arrays of face indices."""
        fi = np.where(mask_faces)[0]
        if not len(fi):
            return []
        Fs = self.Fs[fi]
        E = np.concatenate([Fs[:, [0, 1]], Fs[:, [1, 2]]])
        lab = components(len(self.pos_s), E)
        lf = lab[Fs[:, 0]]
        o = np.argsort(lf, kind="stable")
        u, start = np.unique(lf[o], return_index=True)
        groups = np.split(fi[o], start[1:])
        return sorted(groups, key=lambda g: -len(g))

    def describe_part(self, fk):
        """Size, closedness, dominant bone and region of a part."""
        Fs = self.Fs[fk]
        E, c, _ = edges_faces(Fs)
        vs = np.unique(Fs)
        p = self.pos_s[vs]
        vb = np.unique(self.F[fk])
        dom = self.dom[vb]
        u, n = np.unique(dom[dom >= 0], return_counts=True)
        o = np.argsort(-n)
        os_dom = [(self.names_os[u[k]], round(float(n[k] / n.sum()), 2)) for k in o[:3]]
        accessory = np.array([self.os_accessory(b) for b in dom])
        return {"faces": int(len(fk)), "vertices": int(len(vs)),
                "border_edges": int((c == 1).sum()), "edges": int(len(E)),
                "closed": bool((c == 1).sum() == 0),
                "box_mm": np.round((p.max(0) - p.min(0)) * 1000, 1).tolist(),
                "center_mm": np.round(p.mean(0) * 1000, 1).tolist(),
                "diag_mm": round(float(np.linalg.norm(p.max(0) - p.min(0)) * 1000), 1),
                "os_dominants": os_dom,
                "part_os_accessory": round(float(accessory.mean()), 3),
                "region": self.region_ancestor(int(u[o[0]])) if len(u) else "without",
                "materials": sorted({self.names_mat[i] for i in np.unique(self.mf[fk])})}


# ====================================================================== isolation

def in_uv(M, boxes, exclude=None):
    """Faces whose UV centre falls inside one of the boxes [[umin,vmin],[umax,vmax]],
    and outside the "exclude" boxes."""
    c = M.uv[M.F].mean(1)
    def in_(bs):
        m = np.zeros(len(M.F), bool)
        for lo, hi in bs or []:
            m |= ((c >= np.asarray(lo)) & (c <= np.asarray(hi))).all(1)
        return m
    return in_(boxes) & ~in_(exclude)


def clarity_uv(M):
    """Texture lightness (0-1) at the UV centre of each face; NaN when no readable texture.
    Reads the images through bpy (Blender's Python only)."""
    import bpy
    L = np.full(len(M.F), np.nan)
    c = M.uv[M.F].mean(1)
    for mi in np.unique(M.mf):
        t = M.mats[mi]["texture"]
        if not 0 <= t < len(M.textures):
            continue
        f = os.path.normpath(os.path.join(os.path.dirname(M.path), M.textures[t].replace("\\", "/")))
        if not os.path.exists(f):
            continue
        im = bpy.data.images.load(f)
        w, h = im.size
        px = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(px)
        bpy.data.images.remove(im)
        px = px.reshape(h, w, 4)                   # rows from bottom to top; PMX v points down
        fi = np.where(M.mf == mi)[0]
        u = np.clip((c[fi, 0] % 1) * w, 0, w - 1).astype(int)
        v = np.clip((1 - c[fi, 1] % 1) * h, 0, h - 1).astype(int)
        L[fi] = px[v, u, :3] @ [0.299, 0.587, 0.114]
    return L


def zone_uv(M, spec):
    """Faces kept by the record's UV cutting (uv_boxes, uv_exclude, uv_clarity_min)."""
    m = in_uv(M, spec["uv_boxes"], spec.get("uv_exclude"))
    if spec.get("uv_clarity_min") is not None:
        m &= ~(clarity_uv(M) < spec["uv_clarity_min"])
    return m


def isolate(M, spec):
    """Faces of the garment described by spec, and parts set aside as accessories (with the reason).

    spec: {"mats": [names], "regions": [kept regions] (optional), "exclude_os": pattern of
    dominant bones to set aside (optional), "keep_os": pattern of required dominant bones (optional),
    "box": [[xmin,ymin,zmin],[xmax,ymax,zmax]] in m, tested on the part's centre (optional),
    "exclude_parts": [part indices] (optional), "keep_parts": [indices]: only
    these (optional), "uv_boxes": [[[umin,vmin],[umax,vmax]], …] (and "uv_exclude", same boxes, removed;
    "uv_clarity_min": removes faces whose texture is darker at the UV centre):
    only faces whose UV centre falls in one of them (optional; when a single material carries the whole
    welded outfit of one part, the garment's texture area cuts it out)}.

    A part is an accessory (detected without manual input) if:
      bone   : more than half of its vertices have an accessory dominant bone (ACCESSORY_OS);
      button : closed part with a diagonal under ACC_DIAG_MM (button, rivet, buckle);
      small  : fewer than ACC_PART of the garment's faces and a diagonal under ACC_DIAG_MM.
    """
    mats = [i for i, n in enumerate(M.names_mat) if n in spec["mats"]]
    missing = set(spec["mats"]) - {M.names_mat[i] for i in mats}
    if missing:
        raise KeyError("materials missing from %s: %s" % (M.character, sorted(missing)))
    mask = np.isin(M.mf, mats)
    if spec.get("uv_boxes"):
        mask &= zone_uv(M, spec)
    parts = M.parts(mask)
    total = sum(len(g) for g in parts)
    kept, discarded = [], []
    rx_ex = re.compile(spec["exclude_os"]) if spec.get("exclude_os") else None
    rx_ga = re.compile(spec["keep_os"]) if spec.get("keep_os") else None
    for k, fk in enumerate(parts):
        d = M.describe_part(fk)
        d["index"] = k
        reason = None
        names_dom = [n for n, _ in d["os_dominants"]]
        if k in spec.get("exclude_parts", []):
            reason = "set aside manually (spec)"
        elif "keep_parts" in spec and k not in spec["keep_parts"]:
            reason = "not among the kept parts (spec)"
        elif spec.get("regions") and d["region"] not in spec["regions"]:
            reason = "outside the regions (%s)" % d["region"]
        elif rx_ga and not (names_dom and rx_ga.search(names_dom[0])):
            reason = "dominant bone not part of the garment (%s)" % (names_dom[0] if names_dom else "-")
        elif rx_ex and names_dom and rx_ex.search(names_dom[0]):
            reason = "excluded dominant bone (%s)" % names_dom[0]
        elif spec.get("box"):
            lo, hi = np.asarray(spec["box"][0]), np.asarray(spec["box"][1])
            c = np.asarray(d["center_mm"]) / 1000
            if not ((c >= lo).all() and (c <= hi).all()):
                reason = "outside the box"
        if reason is None:
            if d["part_os_accessory"] > 0.5:
                reason = "accessory: accessory bone (%s)" % names_dom[0]
            elif d["closed"] and d["diag_mm"] < ACCESSORY_DIAG_MM:
                reason = "accessory: small closed part (button, rivet, buckle)"
            elif d["faces"] < ACCESSORY_PART * total and d["diag_mm"] < ACCESSORY_DIAG_MM:
                reason = "accessory: small part"
        d["reason"] = reason
        (discarded if reason else kept).append((fk, d))
    sel = np.zeros(len(M.F), bool)
    for fk, _ in kept:
        sel[fk] = True
    return sel, [d for _, d in kept], [d for _, d in discarded]


ACCESSORY_DIAG_MM = 60.0   # above this, a part is no longer a button or a buckle
ACCESSORY_PART = 0.02      # share of the garment's faces below which a small part is an accessory

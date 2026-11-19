"""ZZZ clothing library: each outfit cut into separate garments and accessories,
each part written as a standalone PMX that can be fitted on another body (wear.py).

    python3.13 -I library.py extract --output DOSSIER [--only Anby,Belle]
        cuts every outfit of the face library, writes <Character>/<part>/{<part>.pmx,
        fiche.json}, <Character>/cutout.json (check), check.json and the index.
    bin/blender-queue -b --factory-startup --python library.py -- sheets --output DOSSIER [--only …]
        thumbnail of each part and contact sheet of each character (exploded, front and back).
    python3.13 -I library.py index --output DOSSIER
        index/clothes/<type>.json, index/accessories/<sort>.json, index.json.

Source files are only ever read.
"""
import json
import os
import re
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as MD  # noqa: E402
from model import P, PMX  # noqa: E402

try:
    import OpenImageIO as OIIO
except ImportError:  # inside Blender: bpy reads the images
    OIIO = None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ====================================================================== textures

_CACHE_TEX = {}


def read_texture(path):
    """Float RGB image (h, w, 3), rows from top to bottom (PMX v points down), or None."""
    if path in _CACHE_TEX:
        return _CACHE_TEX[path]
    im = None
    if os.path.exists(path):
        if OIIO is not None:
            b = OIIO.ImageInput.open(path)
            if b:
                a = b.read_image(format=OIIO.FLOAT)
                b.close()
                if a is not None:
                    if a.ndim == 2:
                        a = a[:, :, None]
                    if a.shape[2] == 1:
                        a = np.repeat(a, 3, 2)
                    im = np.ascontiguousarray(a[:, :, :3], dtype=np.float32)
        else:
            import bpy
            i = bpy.data.images.load(path)
            w, h = i.size
            px = np.empty(w * h * 4, np.float32)
            i.pixels.foreach_get(px)
            bpy.data.images.remove(i)
            im = px.reshape(h, w, 4)[::-1, :, :3].copy()
    if len(_CACHE_TEX) > 6:
        _CACHE_TEX.pop(next(iter(_CACHE_TEX)))
    _CACHE_TEX[path] = im
    return im


def path_texture(M, mi):
    t = M.mats[mi]["texture"]
    if not 0 <= t < len(M.textures):
        return None
    return os.path.normpath(os.path.join(os.path.dirname(M.path), M.textures[t].replace("\\", "/")))


def lab(rgb):
    """sRGB (0-1) -> CIELab (D65)."""
    c = np.clip(np.asarray(rgb, np.float64), 0, 1)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    X = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    X = X / np.array([0.95047, 1.0, 1.08883])
    f = np.where(X > 0.008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def sample(M, faces, uvs):
    """Lab colour of each face's material texture at the given UVs (n, 2); NaN when there is no
    texture."""
    out = np.full((len(faces), 3), np.nan)
    mf = M.mf[faces]
    for mi in np.unique(mf):
        ch = path_texture(M, mi)
        im = read_texture(ch) if ch else None
        k = np.where(mf == mi)[0]
        if im is None:
            d = M.mats[mi]["diffuse"]
            out[k] = lab(np.array(d[:3]))[None]
            continue
        h, w = im.shape[:2]
        u = np.clip((uvs[k, 0] % 1) * w, 0, w - 1).astype(int)
        v = np.clip((uvs[k, 1] % 1) * h, 0, h - 1).astype(int)
        out[k] = lab(im[v, u])
    return out


# ====================================================================== atoms of an outfit

REGIONS = ("head", "neck", "trunk_top", "chest", "belly", "pelvis", "buttocks", "arm", "forearm",
           "hand", "finger", "thigh", "leg", "foot")
RI = {r: i for i, r in enumerate(REGIONS)}
# body zones (groups of regions): used to type a garment
ZONES = {"head": ("head",), "neck": ("neck",), "torso": ("trunk_top", "chest", "belly"),
         "arm": ("arm",), "forearm": ("forearm",), "hand": ("hand", "finger"),
         "pelvis": ("pelvis", "buttocks"), "thigh": ("thigh",), "leg": ("leg",), "foot": ("foot",)}
ZN = tuple(ZONES)


def edges_welded(Fs):
    """Welded edges: (E (m, 2), faces of each edge (per-edge list via sorting))."""
    nf = len(Fs)
    E = np.sort(np.concatenate([Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]]), 1)
    fid = np.tile(np.arange(nf), 3)
    corner = np.repeat(np.arange(3), nf)
    u, inv = np.unique(E, axis=0, return_inverse=True)
    return u, inv.ravel(), fid, corner


SKIN_IN_PLUS = re.compile(r"^首$")


class Outfit:
    """The outfit of an official PMX (everything except head and hair), cut into atoms: islands of
    the raw mesh (UV seams and split normals included) within a single material."""

    def __init__(self, character, M=None, path=None):
        self.character = character
        self.M = M = M or MD.Model(character, path=path)
        self.roles = np.array([M.role_material(i) for i in range(len(M.mats))])
        # skin named in Japanese, missing from model.SKIN: 首 (Yanagi's neck and ears)
        self.roles[[i for i, n in enumerate(M.names_mat) if SKIN_IN_PLUS.match(n)]] = "skin"
        rf = self.roles[M.mf]
        self.base = np.isin(rf, ["outfit", "accessory", "skin"])
        self.texture_layer = rf == "texture_layer"
        fi = np.where(self.base)[0]
        self.fi = fi
        F = M.F[fi]
        # colour of each face (UV centre)
        self.lab_face = np.full((len(M.F), 3), np.nan)
        self.lab_face[fi] = sample(M, fi, M.uv[F].mean(1))
        nv = len(M.pos)

        def islands(sheet):
            """Islands (faces of fi) of the raw mesh, split by material and by flag."""
            Fd = F + nv * sheet[:, None]
            E = np.concatenate([Fd[:, [0, 1]], Fd[:, [1, 2]]])
            lab = MD.components(2 * nv, E)
            return np.unique(lab[Fd[:, 0]].astype(np.int64) * 4096 + M.mf[fi], return_inverse=True)[1]

        # skin painted in the same island as a garment (Qingyi's shorts and thighs): in a
        # generic material, skin-coloured faces forming a region of at least 40 faces and
        # 15 % of their island become a separate island (skin, for the cutting)
        sheet = np.zeros(len(fi), np.int64)
        self.skin_lab = color_skin_roles(self)
        gen_f = np.array([mat_generic(n) for n in M.names_mat])[M.mf[fi]] & (self.roles[M.mf[fi]] == "outfit")
        if self.skin_lab is not None and gen_f.any():
            de = np.linalg.norm(self.lab_face[fi] - self.skin_lab, axis=1)
            candidate = gen_f & (de < OF_SKIN)
            if candidate.any():
                at0 = islands(np.zeros(len(fi), np.int64))
                sheet = candidate.astype(np.int64)
                at1 = islands(sheet)
                n0 = np.bincount(at0)
                n1 = np.bincount(at1)
                ok = candidate & (n1[at1] >= 40) & (n1[at1] >= 0.15 * n0[at0]) & (n1[at1] < 0.97 * n0[at0])
                # skin-coloured bits welded to the skin, whatever their size: shoulder edges
                # of Qingyi's top and thigh tops of her leggings, Nicole's ears and neck
                # (in pieces of fewer than 40 faces along the fabric edge); clear-cut colour
                # only (median ΔE < OF_SKIN / 2): the cream fabrics of Aria, Cissia or
                # YeShunguang come close to the threshold (ΔE 8 to 10) and are not skin
                vp = np.zeros(len(M.pos_s), bool)
                vp[M.Fs[self.roles[M.mf] == "skin"].ravel()] = True
                vp[M.Fs[fi[ok]].ravel()] = True
                glue = candidate & ~ok & vp[M.Fs[fi]].any(1) & (n1[at1] < 0.97 * n0[at0])
                for k in np.unique(at1[glue]):
                    m = at1 == k
                    if np.median(de[m]) < OF_SKIN / 2:
                        ok |= m
                sheet = ok.astype(np.int64)
        self.skin_per_faces = int(sheet.sum())
        at = islands(sheet)
        u = np.unique(at)
        self.atom = np.full(len(M.F), -1)
        self.atom[fi] = at
        self.na = na = len(u)
        self.faces_atom = [fi[at == k] for k in range(na)] if na < 4000 else None
        o = np.argsort(at, kind="stable")
        _, deb = np.unique(at[o], return_index=True)
        self.faces_atom = np.split(fi[o], deb[1:])
        self.mat_atom = np.array([M.mf[g[0]] for g in self.faces_atom])
        self.role_atom = self.roles[self.mat_atom]
        self.nf_atom = np.array([len(g) for g in self.faces_atom])
        self.area_atom = np.array([M.area[g].sum() for g in self.faces_atom])
        w = M.area[fi]
        self.lab_atom = np.zeros((na, 3))
        for c in range(3):
            self.lab_atom[:, c] = np.bincount(at, weights=np.nan_to_num(self.lab_face[fi, c]) * w, minlength=na) / \
                np.maximum(np.bincount(at, weights=w, minlength=na), 1e-20)
        self._regions()
        self._seams()
        self.seg = segments_body(M)
        self.axe_s = os_near(M.pos_s, self.seg)

    # ------------------------------------------------------------------ body regions
    def _regions(self):
        M = self.M
        nb = len(M.names_os)
        reg_os = np.array([RI.get(M.region_ancestor(b), -1) for b in range(nb)])
        self.reg_os = reg_os
        # own bones: neither from the standard body (root family / body regions) nor from the head
        body = np.array([M.family_os[b] not in ("garment", "accessory", "hair", None) or
                          M.names_os[b] in P.STANDARD_MMD or M.names_os[b] in P.SEMI_STANDARD_MMD
                          for b in range(nb)])
        self.os_body = body
        self.os_accessory = np.array([M.os_accessory(b) for b in range(nb)])
        nv = len(M.pos)
        R = np.zeros((nv, len(REGIONS)))
        own = np.zeros(nv)
        accessory = np.zeros(nv)
        for k in range(M.idx.shape[1]):
            b = M.idx[:, k]
            w = M.weight[:, k]
            ok = (b >= 0) & (w > 0)
            r = reg_os[np.where(ok, b, 0)]
            m = ok & (r >= 0)
            np.add.at(R, (np.where(m)[0], r[m]), w[m])
            own[ok] += w[ok] * ~body[b[ok]]
            accessory[ok] += w[ok] * self.os_accessory[b[ok]]
        s = np.maximum(R.sum(1), 1e-9)
        self.R_v = R / s[:, None]
        self.own_v = own
        self.accessory_v = accessory
        na = self.na
        self.R_atom = np.zeros((na, len(REGIONS)))
        self.own_atom = np.zeros(na)
        self.accessory_atom = np.zeros(na)
        for k, g in enumerate(self.faces_atom):
            v = M.F[g].ravel()
            wa = np.repeat(M.area[g], 3)
            self.R_atom[k] = (self.R_v[v] * wa[:, None]).sum(0) / max(wa.sum(), 1e-20)
            self.own_atom[k] = (own[v] * wa).sum() / max(wa.sum(), 1e-20)
            self.accessory_atom[k] = (accessory[v] * wa).sum() / max(wa.sum(), 1e-20)

    def zones(self, R):
        """Region vector -> share per zone (dict)."""
        return {z: float(sum(R[..., RI[r]] for r in rs)) for z, rs in ZONES.items()}

    # ------------------------------------------------------------------ welded seams between atoms
    def _seams(self):
        """For each pair of sewn atoms (shared welded edge): number of edges, length,
        median colour difference across the seam (sampled near the edge)."""
        M = self.M
        fi = self.fi
        Fs = M.Fs[fi]
        u, inv, fid, corner = edges_welded(Fs)
        o = np.argsort(inv, kind="stable")
        inv_o, fid_o, corner_o = inv[o], fid[o], corner[o]
        # edges with exactly two faces (manifold) between two different atoms
        _, deb, cnt = np.unique(inv_o, return_index=True, return_counts=True)
        m2 = cnt == 2
        i1, i2 = deb[m2], deb[m2] + 1
        f1, f2 = fi[fid_o[i1]], fi[fid_o[i2]]
        a1, a2 = self.atom[f1], self.atom[f2]
        diff = a1 != a2
        f1, f2, c1, c2 = f1[diff], f2[diff], corner_o[i1][diff], corner_o[i2][diff]
        a1, a2 = a1[diff], a2[diff]
        e = u[inv_o[i1][diff]]
        L = np.linalg.norm(M.pos_s[e[:, 0]] - M.pos_s[e[:, 1]], axis=1)

        def near(f, c):
            # UV of a point near the middle of the edge (corner c, c+1), a quarter of the way to the 3rd vertex
            tri = M.uv[M.F[f]]
            a = tri[np.arange(len(f)), c]
            b = tri[np.arange(len(f)), (c + 1) % 3]
            t = tri[np.arange(len(f)), (c + 2) % 3]
            return 0.75 * (a + b) / 2 + 0.25 * t
        l1 = sample(M, f1, near(f1, c1))
        l2 = sample(M, f2, near(f2, c2))
        dE = np.linalg.norm(l1 - l2, axis=1)
        lo, hi = np.minimum(a1, a2), np.maximum(a1, a2)
        the_key = lo.astype(np.int64) * 100000 + hi
        uc, ic = np.unique(the_key, return_inverse=True)
        self.seams = {}
        for k, c in enumerate(uc):
            m = ic == k
            self.seams[(int(c // 100000), int(c % 100000))] = {
                "edges": int(m.sum()), "length": float(L[m].sum()),
                "dE": float(np.median(np.nan_to_num(dE[m], nan=50.0)))}
        self.border_sewn = np.zeros(self.na)
        for (a, b), d in self.seams.items():
            self.border_sewn[a] += d["length"]
            self.border_sewn[b] += d["length"]


# ====================================================================== grouping

GENERIC = re.compile(r"^(体|衣|服|足|新規|毛)\d*([-_].*)?$")


def mat_generic(name):
    """"Texture sheet" material (体, 体2, 衣1…) that can carry several garments, as opposed
    to a material named after a garment (裤, 袜, 外套…)."""
    return bool(GENERIC.match(name))


def color_skin_roles(T):
    """Median skin Lab: faces of the skin materials, otherwise of the face (颜)."""
    M = T.M
    f = np.where(T.roles[M.mf] == "skin")[0]
    if len(f) < 50:
        f = np.where(np.array([n.startswith("颜") for n in M.names_mat])[M.mf])[0]
        if not len(f):
            return None
        return np.nanmedian(sample(M, f, M.uv[M.F[f]].mean(1)), 0)
    return np.nanmedian(T.lab_face[f], 0)


def color_skin(T):
    return T.skin_lab


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def t(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def u(self, a, b):
        a, b = self.t(a), self.t(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)



    def groups(self, ids):
        out = {}
        for i in ids:
            out.setdefault(self.t(i), []).append(i)
        return list(out.values())


# ====================================================================== geometry of a set of faces

SEG_EXCLUDED = ("finger", "head")


# IK handles (足ＩＫ, 足IK親, つま先ＩＫ): placed at the foot, but filed as "thigh" by their name
# (足…). A ring that used them as its axis was called a garter although it sits at the ankle
# (Anby). Their region is read from where they are: that of the body bone whose head is closest.
OS_IK = re.compile(r"IK|ＩＫ")


def region_axe(T, b):
    M = T.M
    if not OS_IK.search(M.names_os[b]):
        return M.region_ancestor(b)
    others = [i for i in T.seg[0] if not OS_IK.search(M.names_os[i])]
    j = others[int(np.argmin(np.linalg.norm(M.heads[others] - M.heads[b], axis=1)))]
    return M.region_ancestor(j)


def segments_body(M):
    """Body bones (head-tail segments) used as axes: (indices, heads, tails)."""
    body = ("trunk_top", "chest", "belly", "pelvis", "buttocks", "neck", "arm", "forearm",
             "hand", "thigh", "leg", "foot")
    b = [i for i in range(len(M.names_os)) if M.region_os[i] in body
         and np.linalg.norm(M.tails[i] - M.heads[i]) > 0.01]
    return np.array(b), M.heads[b], M.tails[b]


def os_near(Pv, seg):
    """For each point: nearest body bone segment, parameter t (m along the bone),
    angle around the bone, radial distance."""
    b, A, C = seg
    best = np.full(len(Pv), np.inf)
    k = np.zeros(len(Pv), int)
    tt = np.zeros(len(Pv))
    for j in range(len(b)):
        ab = C[j] - A[j]
        L2 = ab @ ab
        t = np.clip(((Pv - A[j]) @ ab) / L2, 0, 1)
        d = np.linalg.norm(Pv - (A[j] + t[:, None] * ab), axis=1)
        m = d < best
        best[m], k[m], tt[m] = d[m], j, t[m] * np.sqrt(L2)
    ax = (C - A)[k]
    ax /= np.linalg.norm(ax, axis=1)[:, None]
    ref = np.where(np.abs(ax[:, 2:3]) > 0.8, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
    e1 = np.cross(ax, ref)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(ax, e1)
    rel = Pv - A[k]
    ang = np.arctan2((rel * e2).sum(1), (rel * e1).sum(1))
    return k, tt, ang, best


class Geo:
    """Geometric measurements of a set of faces (indices into M.F)."""

    def __init__(self, T, faces):
        M = T.M
        self.faces = faces
        Fs = M.Fs[faces]
        self.vs = vs = np.unique(Fs)
        p = M.pos_s[vs]
        self.p = p
        self.area = float(M.area[faces].sum())
        self.lo, self.hi = p.min(0), p.max(0)
        self.diag = float(np.linalg.norm(self.hi - self.lo))
        self.center = (p * 1.0).mean(0)
        E, c, _ = MD.edges_faces(Fs)
        self.border = int((c == 1).sum())
        self.closed = self.border == 0
        # ring: around its dominant bone, short extent along the bone, almost a full turn
        k, t, ang, r = T.axe_s[0][vs], T.axe_s[1][vs], T.axe_s[2][vs], T.axe_s[3][vs]
        kd = np.bincount(k).argmax()
        m = k == kd
        self.os_axe = int(T.seg[0][kd])
        A, C = T.seg[1][kd], T.seg[2][kd]
        ax = (C - A) / np.linalg.norm(C - A)
        h = (p - A) @ ax        # all vertices, not only the bone's: a glove extends past the wrist
        self.extent_axe = float(np.percentile(h, 97) - np.percentile(h, 3))
        self.girth = float(len(np.unique(((ang[m] + np.pi) / (2 * np.pi) * 24).astype(int) % 24)) / 24)
        self.part_axe = float(m.mean())
        self.radius = float(np.median(r))
        # width of a band: area / length (diagonal)
        self.width = self.area / max(self.diag, 1e-6)


# ====================================================================== cutting

SLOTS = {"top": ("torso", "arm", "forearm"), "bottom": ("pelvis", "thigh"), "leg": ("leg",),
         "foot": ("foot",), "hand": ("hand",), "neck": ("neck",), "head": ("head",)}
SLOTS_NEIGHBORS = {frozenset(x) for x in (("top", "bottom"), ("bottom", "leg"), ("leg", "foot"), ("top", "neck"),
                                        ("top", "hand"))}
ACCESSORY_MAT_IN_PLUS = re.compile(r"配件|腿环|饰")   # in addition to model.ACCESSORY_MAT: "attached parts", thigh ring
OF_SAME_FABRIC = 15.0      # colour difference (Lab) below which two sewn panels are the same fabric
OF_SKIN_DROP = 12.0      # skin offcut left as an accessory (skin-coloured area share >= 0.5)
OF_SKIN = 10.0            # difference from the skin colour below which a generic-material panel is skin
ACCESSORY_DIAG = 0.06           # m: smaller part = accessory (button, rivet, buckle)
LOOP_TOP = 0.07        # m: ring narrower than this along its bone = belt, bracelet, necklace, garter
LOOP_PERIMETER = 0.75        # share of the full turn covered
STRAP_WIDTH = 0.03     # m: narrower band = strap
PLACED_AREA = 0.03         # share of the outfit's area below which a shell lying on a garment is an accessory
PLACED_DIAG = 0.25         # m: and diagonal (pocket, patch, pouch)
BORDER_AGAINST_MIN = 0.5     # share of the smaller panel's border touching the other one for them to be sewn back together
SMALL_PART = 0.012       # share of the outfit's faces below which a small shell (diag < 15 cm) is an accessory
CHAIN_PART = 0.5         # share of a shell's weights on bones that only it uses: above this, a separate object

# accessory kinds read from material or bone names (first pattern found)
KINDS_NAME = [(r"武器|刀|Weapon|拳套", "weapon"), (r"墨镜|镜片|护目镜|眼镜|镜", "glasses"), (r"面具|面罩|眼罩", "mask"),
              (r"耳机", "headphones"), (r"帽|头饰|頭巾|发饰|髮飾|髮饰|光环|角", "headwear"), (r"翅膀|翼|翅", "wings"),
              (r"尾", "tail"), (r"^耳|_耳|耳$", "ears"), (r"耳饰|耳环", "loop_ear"), (r"包|Backpack", "bag"),
              (r"熊|狸猫|邦布|玩偶", "plush"), (r"电池|工具|手雷|屏幕", "object"), (r"戒指", "ring"),
              (r"宝石|珠宝|项链|颈饰|吊坠|坠|珠", "jewel"), (r"领带|Tie|Bowtie", "tie"), (r"拉链", "closure"),
              (r"锁链|链", "chain"), (r"绳", "rope"), (r"徽章|Badge", "badge"), (r"扣", "loop"), (r"腿环|腿圈|LegRing|Garter", "garter"),
              (r"腰带|皮带|Belt", "belt"),
              (r"围巾|领巾|Scarf", "scarf"), (r"Strap|带", "strap")]
_KINDS_NAME = [(re.compile(rx), s) for rx, s in KINDS_NAME]


KINDS_OBJ = {"weapon", "glasses", "mask", "headphones", "wings", "tail", "ears", "bag", "plush", "object"}


def kind_name(name):
    for rx, s in _KINDS_NAME:
        if rx.search(name):
            return s
    return None


def d_one_side(T, g):
    """Share of vertices on one side (front or back) of the torso's frontal plane: a top wraps around
    the torso, a backpack or a breastplate stays on one side."""
    M = T.M
    names = {n: i for i, n in enumerate(M.names_os)}
    if "上半身2" not in names:
        return 0.0
    y0 = M.heads[names["上半身2"]][1]
    back = (g.p[:, 1] > y0 + 0.01).mean()
    dev = (g.p[:, 1] < y0 - 0.01).mean()
    return float(max(back, dev))


def border_against(T, fa, fb, grid_cell=0.008):
    """Share of fa's border vertices that have a vertex of fb within one cell: two panels
    of the same garment touch along a seam, two garments at a few points only."""
    M = T.M
    E, c, _ = MD.edges_faces(M.Fs[fa])
    vb = np.unique(E[c == 1])
    if len(vb) == 0:
        return 0.0
    kb = set(keys_voxels(M.pos_s[np.unique(M.Fs[fb])], grid_cell).tolist())
    p = M.pos_s[vb]
    ok = np.zeros(len(vb), bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k = keys_voxels(p + np.array([dx, dy, dz]) * grid_cell, grid_cell)
                ok |= np.isin(k, list(kb))
    return float(ok.mean())


def slot_of(T, R):
    z = T.zones(R)
    sc = {s: sum(z[x] for x in zs) for s, zs in SLOTS.items()}
    return max(sc, key=sc.get)


def keys_voxels(p, grid_cell):
    q = np.floor(p / grid_cell).astype(np.int64) + 1 << 20
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


NEAR = 0.02   # m: below this, a surface is "touched" by an accessory
WORN_PART = 0.1   # share of an accessory's points that must touch a part for that part to carry it


def _dist_triangles(p, A, B, C):
    """Distance from each point p (n,3) to each triangle (m,3) -> (n, m)."""
    def seg(a, e):
        v = p[:, None] - a[None]
        t = np.clip((v * e[None]).sum(2) / np.maximum((e * e).sum(1), 1e-20)[None], 0, 1)
        return np.linalg.norm(v - t[:, :, None] * e[None], axis=2)
    AB, AC = B - A, C - A
    N = np.cross(AB, AC)
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-20)[:, None]
    v = p[:, None] - A[None]
    h = (v * N[None]).sum(2)
    q = v - h[:, :, None] * N[None]
    d00, d01, d11 = (AB * AB).sum(1), (AB * AC).sum(1), (AC * AC).sum(1)
    d20, d21 = (q * AB[None]).sum(2), (q * AC[None]).sum(2)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-20)[None]
    b = (d11[None] * d20 - d01[None] * d21) / den
    c = (d00[None] * d21 - d01[None] * d20) / den
    border = np.minimum(np.minimum(seg(A, AB), seg(A, AC)), seg(B, C - B))
    return np.where((b >= 0) & (c >= 0) & (b + c <= 1), np.abs(h), border)


def plus_near_surface(T, faces, groups, reach=NEAR, points_max=150, over=False):
    """What a small part really touches: for each of its points (vertices and face centres),
    the group of faces whose surface is closest, within `reach`. Returns the
    number of points per group. The cells of `contacts` only see vertices: a button
    in the middle of a large jacket face touches nothing there. `over`: a point only counts for a
    group if it lies on the outer side of its surface (a tie touches the shirt and the
    jacket, but it lies on the shirt and goes under the jacket)."""
    M = T.M
    tri = M.pos_s[M.Fs[faces]]
    p = np.concatenate([np.unique(tri.reshape(-1, 3), axis=0), tri.mean(1)])
    if len(p) > points_max:
        p = p[np.linspace(0, len(p) - 1, points_max).astype(int)]
    lo, hi = p.min(0) - reach, p.max(0) + reach
    D = np.full((len(p), len(groups)), np.inf)
    for k, f in enumerate(groups):
        if not len(f):
            continue
        t = M.pos_s[M.Fs[f]]
        t = t[((t.max(1) >= lo) & (t.min(1) <= hi)).all(1)]
        for i in range(0, len(t), 4000):
            x = t[i:i + 4000]
            d = _dist_triangles(p, x[:, 0], x[:, 1], x[:, 2])
            if over:
                kk = d.argmin(1)
                nn = np.cross(x[kk, 1] - x[kk, 0], x[kk, 2] - x[kk, 0])
                d = np.where(((p - x[kk, 0]) * nn).sum(1) > 0, d.min(1), np.inf)
            else:
                d = d.min(1)
            D[:, k] = np.minimum(D[:, k], d)
    k = D.argmin(1)
    return np.bincount(k[D.min(1) <= reach], minlength=len(groups))


def contacts(T, groups, grid_cell):
    """Pairs of groups (lists of faces) that have vertices in the same cell of
    `grid_cell` m: {(i, j): number of shared cells}."""
    if not len(groups):
        return {}       # an outfit without garment shells (accessories only, or nothing): no pair
    M = T.M
    ks, gs = [], []
    for i, f in enumerate(groups):
        vs = np.unique(M.Fs[f])
        k = np.unique(keys_voxels(M.pos_s[vs], grid_cell))
        ks.append(k)
        gs.append(np.full(len(k), i))
    k = np.concatenate(ks)
    g = np.concatenate(gs)
    o = np.argsort(k, kind="stable")
    k, g = k[o], g[o]
    _, deb, cnt = np.unique(k, return_index=True, return_counts=True)
    out = {}
    for d, c in zip(deb[cnt > 1], cnt[cnt > 1]):
        gg = g[d:d + c]
        for a in range(len(gg)):
            for b in range(a + 1, len(gg)):
                the_key = (int(min(gg[a], gg[b])), int(max(gg[a], gg[b])))
                out[the_key] = out.get(the_key, 0) + 1
    return out


def cases_body(T, faces, step_t=0.025, n_ang=12):
    """Cells (bone, slice along the bone, angular sector) covered by a set of
    faces, with the median radius in each cell: {cell: radius}."""
    vs = np.unique(T.M.Fs[faces])
    k, t, ang, r = (a[vs] for a in T.axe_s)
    c = (k.astype(np.int64) * 1000 + (t / step_t).astype(np.int64)) * 100 + \
        ((ang + np.pi) / (2 * np.pi) * n_ang).astype(np.int64) % n_ang
    u, inv = np.unique(c, return_inverse=True)
    rm = np.bincount(inv, weights=r) / np.bincount(inv)
    return dict(zip(u.tolist(), rm.tolist()))


def mirrors(T, ga, gb):
    """Are two sets of faces left-right symmetric (mirrored centres, similar sizes)?"""
    M = T.M
    ca = M.pos[np.unique(M.F[ga])].mean(0)
    cb = M.pos[np.unique(M.F[gb])].mean(0)
    cb = cb * np.array([-1, 1, 1])
    na, nb = len(ga), len(gb)
    return np.linalg.norm(ca - cb) < 0.03 and 0.8 < na / nb < 1.25 and abs(ca[0]) > 0.02


def slice_(T):
    """Cuts an outfit into parts. Returns (pieces, info); each part: dict with "class_"
    (garment | accessory), "faces" (indices into M.F, overlay layers included), "atoms", reasons."""
    M = T.M
    names = M.names_mat
    na = T.na
    gen = np.array([mat_generic(names[m]) for m in T.mat_atom])
    skin_lab = color_skin(T)
    is_skin = T.role_atom == "skin"
    if skin_lab is not None:
        is_skin = is_skin | (gen & (np.linalg.norm(T.lab_atom - skin_lab, axis=1) < OF_SKIN))
    mat_accessory = np.array([T.roles[m] == "accessory" or bool(ACCESSORY_MAT_IN_PLUS.search(names[m])) for m in range(len(names))])
    accessory_at = ~is_skin & (mat_accessory[T.mat_atom] | (T.accessory_atom > 0.5))
    slot_at = [slot_of(T, T.R_atom[a]) for a in range(na)]
    reason_at = {a: ("accessory material (%s)" % names[T.mat_atom[a]] if mat_accessory[T.mat_atom[a]]
                     else "accessory bone") for a in np.where(accessory_at)[0]}
    garment_at = ~is_skin & ~accessory_at
    # --- 1. shells: sewn garment atoms, same material or two generic materials
    uf = UF(na)
    for (a, b), c in T.seams.items():
        if garment_at[a] and garment_at[b]:
            if T.mat_atom[a] == T.mat_atom[b] or (gen[a] and gen[b]):
                uf.u(a, b)
        elif accessory_at[a] and accessory_at[b]:
            uf.u(a, b)
    shells = uf.groups([a for a in range(na) if not is_skin[a]])
    total_f = sum(T.nf_atom[a] for a in range(na) if garment_at[a]) or 1
    area_outfit = sum(T.area_atom[a] for a in range(na) if not is_skin[a]) or 1
    faces_of = lambda ats: np.concatenate([T.faces_atom[a] for a in ats])  # noqa: E731
    # --- 2. garment shells that are accessories (shape, size, lying on a garment)
    cq_garment = [c for c in shells if garment_at[c[0]]]
    cq_accessory = [c for c in shells if accessory_at[c[0]]]
    geo = [Geo(T, faces_of(c)) for c in cq_garment]
    big = [i for i, c in enumerate(cq_garment) if geo[i].area >= PLACED_AREA * area_outfit]
    support = contacts(T, [faces_of(c) for c in cq_garment], 0.02)
    # material named after a garment (靴, 手套…) and large enough: its shells belong to it, except small ones
    nf_mat = np.bincount(T.mat_atom[garment_at], weights=T.nf_atom[garment_at], minlength=len(names))
    named_large = [not gen[c[0]] and nf_mat[T.mat_atom[c[0]]] > 0.03 * total_f for c in cq_garment]
    # bone chain of its own for a shell: clothing bones almost all of whose weight (area-weighted)
    # falls in this single shell. A material named after a garment tells on which texture sheet
    # the shell is painted, not what it is: an isolated, small shell driven by
    # its own chain is a separate object (Hugo's tie, painted in 外套 with the jacket)
    mass = np.zeros((len(cq_garment), len(M.names_os)))
    for i, c in enumerate(cq_garment):
        f = faces_of(c)
        v = M.F[f].ravel()
        wa = np.repeat(M.area[f], 3)
        for k in range(M.idx.shape[1]):
            ob, w = M.idx[v, k], M.weight[v, k]
            ok = (ob >= 0) & (w > 0)
            np.add.at(mass[i], ob[ok], w[ok] * wa[ok])
    a_itself = (mass >= 0.95 * np.maximum(mass.sum(0), 1e-20)) & ~T.os_body
    chain = (mass * a_itself).sum(1) / np.maximum(mass.sum(1), 1e-20)
    accessory_cq = {}
    for i, (c, g) in enumerate(zip(cq_garment, geo)):
        nf = sum(T.nf_atom[a] for a in c)
        placed = any(((min(i, j), max(i, j)) in support) for j in big if j != i)
        r = None
        if g.diag < ACCESSORY_DIAG:
            r = "small part (%.0f mm)" % (g.diag * 1000)
        elif named_large[i] and chain[i] >= CHAIN_PART and g.area < PLACED_AREA * area_outfit:
            r = "object driven by its own bone chain (%.0f %% of its weights)" % (100 * chain[i])
        elif g.extent_axe < LOOP_TOP and g.girth >= LOOP_PERIMETER and g.part_axe > 0.6 and g.diag < 0.5:
            r = "ring around %s (%.0f mm high)" % (M.names_os[g.os_axe], g.extent_axe * 1000)
        elif g.width < STRAP_WIDTH and g.diag > 0.08 and g.area < PLACED_AREA * area_outfit:
            r = "narrow band (%.0f mm)" % (g.width * 1000)
        elif nf < SMALL_PART * total_f and g.diag < 0.15:
            r = "small shell (%d faces)" % nf
        elif not named_large[i] and g.diag < 0.45 and d_one_side(T, g) >= 0.9 and \
                slot_of(T, (T.R_atom[c] * T.area_atom[c, None]).sum(0) / T.area_atom[c].sum()) == "top":
            r = "volume on one side of the torso only (bag, breastplate)"
        elif placed and g.area < PLACED_AREA * area_outfit and g.diag < PLACED_DIAG and not named_large[i]:
            r = "lying on a garment (%.1f %% of the area)" % (100 * g.area / area_outfit)
        if r:
            accessory_cq[i] = r
    # --- 3. within each garment shell: sewn panels grouped into garments
    garments = []      # lists of atoms
    for i, c in enumerate(cq_garment):
        if i in accessory_cq:
            continue
        if not gen[c[0]]:
            garments.append(list(c))
            continue
        cs = set(c)
        u2 = UF(na)
        for (a, b), d in T.seams.items():
            if a in cs and b in cs:
                sa, sb = slot_at[a], slot_at[b]
                if sa == sb or (frozenset((sa, sb)) in SLOTS_NEIGHBORS and d["dE"] < OF_SAME_FABRIC):
                    u2.u(a, b)
        garments.extend(u2.groups(c))
    # --- 4. garments from different shells: same named material, or generic ones that touch
    #        without overlapping, same slot, similar colour
    def slot_g(ats):
        f = faces_of(ats)
        return slot_of(T, (T.R_atom[ats] * T.area_atom[ats, None]).sum(0) / T.area_atom[ats].sum())
    log_merge = T.log_merge = []
    for _ in range(3):
        n0 = len(garments)
        fl = [faces_of(v) for v in garments]
        sl = [slot_g(v) for v in garments]
        labv = [(T.lab_atom[v] * T.area_atom[v, None]).sum(0) / T.area_atom[v].sum() for v in garments]
        cases = [cases_body(T, f) for f in fl]
        touched = contacts(T, fl, 0.006)
        u3 = UF(len(garments))
        for i in range(len(garments)):
            for j in range(i + 1, len(garments)):
                mi, mj = T.mat_atom[garments[i][0]], T.mat_atom[garments[j][0]]
                if mi == mj and not gen[garments[i][0]]:
                    # same named material: one garment, unless two distant slots, or neighbouring ones
                    # with different colours (trousers and sneakers in 裤 for Manato, briefs and boots
                    # in 靴 for Corin)
                    if sl[i] == sl[j] or (frozenset((sl[i], sl[j])) in SLOTS_NEIGHBORS
                                          and np.linalg.norm(labv[i] - labv[j]) <= 2 * OF_SAME_FABRIC):
                        u3.u(i, j)
                    continue
                if (i, j) not in touched or sl[i] != sl[j] or not (gen[garments[i][0]] and gen[garments[j][0]]):
                    continue
                if np.linalg.norm(labv[i] - labv[j]) > 2 * OF_SAME_FABRIC:
                    continue
                com = set(cases[i]) & set(cases[j])
                if len(com) > 0.25 * min(len(cases[i]), len(cases[j])):
                    continue        # one covers the other: two layers
                small, large = (fl[i], fl[j]) if len(fl[i]) <= len(fl[j]) else (fl[j], fl[i])
                bc = border_against(T, small, large)
                log_merge.append((len(fl[i]), len(fl[j]), round(bc, 3)))
                if bc < BORDER_AGAINST_MIN:
                    continue        # only a few contact points: two neighbouring garments
                u3.u(i, j)
        garments = [sum((garments[k] for k in g), []) for g in u3.groups(range(len(garments)))]
        if len(garments) == n0:
            break
    # --- 5. small leftovers sewn to a garment: absorbed (piping, lapel); otherwise accessories
    size = np.array([sum(T.nf_atom[a] for a in v) for v in garments])
    guard = []
    remainders = []
    for k, v in enumerate(garments):
        if size[k] < 0.02 * total_f:
            remainders.append(k)
        else:
            guard.append(k)
    at_garment = {}
    for k in guard:
        for a in garments[k]:
            at_garment[a] = k
    accessory_rest = []
    for k in sorted(remainders, key=lambda k: -size[k]):
        best, lb = None, 0
        for a in garments[k]:
            for (x, y), d in T.seams.items():
                o = y if x == a else x if y == a else None
                if o is not None and o in at_garment and d["length"] > lb:
                    best, lb = at_garment[o], d["length"]
        if best is not None:
            garments[best] = garments[best] + garments[k]
            for a in garments[k]:
                at_garment[a] = best
        else:
            accessory_rest.append(k)
    parts = []
    for k in guard:
        parts.append({"class": "garment", "atoms": sorted(garments[k])})
    # --- 6. accessories: accessory atoms, accessory shells, leftovers; those that touch each other
    #        (a buckle on its belt) become one
    accessory_groups = [list(c) for c in cq_accessory] + [list(cq_garment[i]) for i in accessory_cq] + [garments[k] for k in accessory_rest]
    accessory_reason = [reason_at[c[0]] for c in cq_accessory] + [accessory_cq[i] for i in accessory_cq] + ["small isolated leftover"] * len(accessory_rest)
    if accessory_groups:
        fl = [faces_of(g) for g in accessory_groups]
        touched = contacts(T, fl, 0.003)
        u4 = UF(len(accessory_groups))
        small = [Geo(T, f).diag < 2 * ACCESSORY_DIAG for f in fl]
        for (i, j), n in touched.items():
            if n >= 2 and (small[i] or small[j]):
                u4.u(i, j)
        # named objects (tool kit, weapon, plush toy…): nearby pieces of the same material
        # are a single object
        mat_g = [T.mat_atom[g[0]] for g in accessory_groups]
        obj_g = [kind_name(names[m]) in KINDS_OBJ for m in mat_g]
        if any(obj_g):
            nearest = contacts(T, fl, 0.03)
            for (i, j) in nearest:
                if obj_g[i] and mat_g[i] == mat_g[j]:
                    u4.u(i, j)
        for g in u4.groups(range(len(accessory_groups))):
            ats = sorted(sum((accessory_groups[k] for k in g), []))
            parts.append({"class": "accessory", "atoms": ats,
                           "reason": sorted({accessory_reason[k] for k in g})})
    for pc in parts:
        pc["faces"] = np.sort(faces_of(pc["atoms"]))
    # skin offcuts: an accessory entirely in generic material, at least half of whose area
    # is skin-coloured (ΔE < 12, colour per face) -> skin
    # ; enclosed in the skin (all its vertices welded to the skin, none to another part) and of
    # the colour of the surrounding skin (median ΔE < 12 to the neighbouring skin, < 24 to the
    # overall skin): fingertips sticking out of gloves (YeShunguang), ears and neck (Nicole),
    # edge between skin and fabric (Qingyi); the pink pattern on Aria's ankle (ΔE 27 to
    # her neighbouring skin) stays an accessory
    if skin_lab is not None:
        v_skin = np.zeros(len(M.pos_s), bool)
        f_skin = faces_of([a for a in range(na) if is_skin[a]]) if is_skin.any() else np.zeros(0, int)
        v_skin[M.Fs[f_skin].ravel()] = True
        n_pc = np.zeros(len(M.pos_s), int)
        for pc in parts:
            n_pc[np.unique(M.Fs[pc["faces"]])] += 1
        rest = []
        for pc in parts:
            fa = pc["faces"]
            if pc["class"] == "accessory" and all(gen[a] for a in pc["atoms"]):
                de = np.linalg.norm(T.lab_face[fa] - skin_lab, axis=1)
                ok = ~np.isnan(de)
                w = M.area[fa][ok]
                vs = np.unique(M.Fs[fa])
                enclave = v_skin[vs].all() and (n_pc[vs] == 1).all() and ok.any() \
                    and np.median(de[ok]) < 2 * OF_SKIN_DROP
                if enclave:
                    mv = np.zeros(len(M.pos_s), bool)
                    mv[vs] = True
                    lv = T.lab_face[f_skin[mv[M.Fs[f_skin]].any(1)]]
                    lv = lv[~np.isnan(lv).any(1)]
                    lf = T.lab_face[fa][ok]
                    enclave = len(lv) > 0 and np.median(np.linalg.norm(lf - np.median(lv, 0), axis=1)) < OF_SKIN_DROP
                if w.sum() > 0 and ((w * (de[ok] < OF_SKIN_DROP)).sum() / w.sum() >= 0.5 or enclave):
                    is_skin[pc["atoms"]] = True
                    continue
            rest.append(pc)
        parts[:] = rest
    skin_faces = faces_of([a for a in range(na) if is_skin[a]]) if is_skin.any() else np.zeros(0, int)
    T.is_skin = is_skin
    return parts, {"skin_faces": skin_faces, "color_skin": None if skin_lab is None else skin_lab.tolist()}


# ====================================================================== types, kinds, layers

def zones_part(T, faces):
    v = T.M.F[faces].ravel()
    w = np.repeat(T.M.area[faces], 3)
    R = (T.R_v[v] * w[:, None]).sum(0) / max(w.sum(), 1e-20)
    return T.zones(R)


def legs_separate(T, faces):
    """Does the garment wrap each leg separately (trousers, shorts) or both together
    (skirt, dress)? On thigh slices, counts the vertices on the inner side of each thigh
    (within ±25° of the direction of the other leg, within 10 cm of the bone, or within 1.1 times
    half the distance between the thighs for wide trousers on a large build). Trousers have some, a
    skirt does not (it passes in front of and behind the crotch). None if the part does not reach the thighs."""
    M = T.M
    names = {n: i for i, n in enumerate(M.names_os)}
    p = M.pos_s[np.unique(M.Fs[faces])]
    tot = inn = 0
    for c, o, g in (("左足", "右足", "左ひざ"), ("右足", "左足", "右ひざ")):
        if c not in names or g not in names or o not in names:
            return None
        a, b, ao = M.heads[names[c]], M.heads[names[g]], M.heads[names[o]]
        go = M.heads[names[g.replace(c[0], o[0])]] if g.replace(c[0], o[0]) in names else b * [-1, 1, 1]
        side = np.sign(ao[0] - a[0])
        for t in (0.1, 0.2, 0.3, 0.45, 0.6):
            q = a + (b - a) * t
            half = 0.5 * np.linalg.norm((ao + (go - ao) * t)[:2] - q[:2])
            m = np.abs(p[:, 2] - q[2]) < 0.02
            d = p[m, :2] - q[:2]
            r = np.linalg.norm(d, axis=1)
            ang = np.degrees(np.arctan2(d[:, 1], d[:, 0] * side))
            tot += int((r < 0.25).sum())
            inn += int(((np.abs(ang) < 25) & (r < max(0.10, 1.1 * half)) & (r > 0.005)).sum())
    if tot < 10:
        return None
    return inn >= 3 and inn / tot >= 0.02


WIDE_TIBIA = 0.10   # m: median radius around the shin above which the lower garment hangs loose (a calf is 5 to 6 cm)


def magnitude_tibia(T, faces):
    """Median radius (m) of the part around the shin bones, between knee and ankle. None if it
    does not reach that far down."""
    M = T.M
    names = {n: i for i, n in enumerate(M.names_os)}
    p = M.pos_s[np.unique(M.Fs[faces])]
    rs = []
    for g, c in (("左ひざ", "左足首"), ("右ひざ", "右足首")):
        if g not in names or c not in names:
            continue
        a, b = M.heads[names[g]], M.heads[names[c]]
        u = b - a
        t = ((p - a) @ u) / max(float(u @ u), 1e-20)
        m = (t > 0.2) & (t < 0.8) & (np.sign(p[:, 0] - 0.5 * (M.heads[names["左ひざ"]][0] + M.heads[names["右ひざ"]][0]))
                                     == np.sign(a[0] - 0.5 * (M.heads[names["左ひざ"]][0] + M.heads[names["右ひざ"]][0]))) \
            if "左ひざ" in names and "右ひざ" in names else (t > 0.2) & (t < 0.8)
        if m.any():
            rs.append(np.linalg.norm((p[m] - a) - t[m, None] * u, axis=1))
    return float(np.median(np.concatenate(rs))) if rs else None


def perimeter_of_skull(T, faces):
    """Share of the 24 sectors around the head bone (4-10 cm above its head, within
    15 cm of the axis) where the part has vertices. A hood or a head of hair wraps around the skull;
    pigtails or strands hang down the sides."""
    M = T.M
    names = {n: i for i, n in enumerate(M.names_os)}
    if "頭" not in names:
        return None
    c = M.heads[names["頭"]]
    p = M.pos_s[np.unique(M.Fs[faces])] - c
    m = (p[:, 2] > 0.04) & (p[:, 2] < 0.10) & (np.linalg.norm(p[:, :2], axis=1) < 0.15)
    if not m.any():
        return 0.0
    sect = ((np.arctan2(p[m, 1], p[m, 0]) + np.pi) / (2 * np.pi) * 24).astype(int) % 24
    return len(np.unique(sect)) / 24


HAIR_OS = re.compile(r"[Hh]air|髪|髮|发")


def under_hips(T, faces):
    """How far (m) the part goes below the heads of the thigh bones."""
    M = T.M
    names = {n: i for i, n in enumerate(M.names_os)}
    zs = [M.heads[names[n]][2] for n in ("左足", "右足") if n in names]
    if not zs:
        return 0.0
    return float(min(zs) - M.pos_s[np.unique(M.Fs[faces]), 2].min())


def part_chest(T, faces):
    """Share of the part's area above the chest (8 cm above the head of 上半身2):
    a skirt never goes up there, a short dress does (its lower half outweighs its upper half)."""
    M = T.M
    names = {n: i for i, n in enumerate(M.names_os)}
    if "上半身2" not in names:
        return 0.0
    zc = M.pos_s[M.Fs[faces], 2].mean(1)
    w = M.area[faces]
    return float((w * (zc > M.heads[names["上半身2"]][2] + 0.08)).sum() / max(w.sum(), 1e-20))


COLLAR_CHEST = 0.85   # share of the area above the chest beyond which a torso garment is a collar


def type_garment(T, faces, layer, own, on_top=None):
    """Type of a garment from the body zones it covers; `layer`: number of garments
    it covers + 1; `on_top`: does it cover another top (jacket, vest, coat)?"""
    if on_top is None:
        on_top = layer >= 2
    z = zones_part(T, faces)
    top = z["torso"] + z["arm"] + z["forearm"] + z["neck"]
    bottom = z["pelvis"] + z["thigh"] + z["leg"]
    sleeves = z["arm"] + z["forearm"]
    arm = z["arm"] + z["forearm"] + z["hand"]
    if z["hand"] >= 0.5 or (arm >= 0.8 and z["hand"] >= 0.15):
        return "gloves"
    if z["foot"] >= 0.35:
        return "boots" if z["leg"] + z["thigh"] >= 0.25 else "shoes"
    # a long lower garment also weighs mostly on the legs: a sock wraps each leg
    # closely; whatever reaches the pelvis and hangs loose around the shin is wide trousers or a skirt
    if (z["leg"] + z["foot"] >= 0.5 or (z["leg"] + z["thigh"] + z["foot"] >= 0.75 and z["pelvis"] < 0.1)) \
            and legs_separate(T, faces) is not False \
            and not (z["pelvis"] >= 0.05 and (magnitude_tibia(T, faces) or 0) >= WIDE_TIBIA):
        return "socks"
    if z["neck"] >= 0.5 or (z["torso"] + z["neck"] >= 0.5 and part_chest(T, faces) >= COLLAR_CHEST):
        # sailor collar, fur collar, scarf, neck warmer: on the torso but entirely above the
        # chest (a vest or a top goes below it: ≤ 0.54 measured)
        return "collar"
    if z["head"] >= 0.5:
        return "hood"
    if sleeves >= 0.5 and z["torso"] < 0.1:
        # arms only, no torso: sleeves (Corin's 袖, Remielle's 袖子), not a top
        return "sleeves"
    if top >= 0.45 or (top >= 0.35 and bottom >= 0.35) or (bottom >= 0.3 and part_chest(T, faces) >= 0.04):
        long_ = under_hips(T, faces) > 0.15
        if bottom >= 0.25 or long_:
            if (on_top or sleeves >= 0.15) and legs_separate(T, faces) is not True:
                return "coat"
            return "jumpsuit" if legs_separate(T, faces) else "dress"
        if on_top:
            return "jacket" if sleeves >= 0.15 else "waistcoat"
        return "top"
    if sleeves >= 0.5:
        return "sleeves"
    if bottom >= 0.5:
        if legs_separate(T, faces) is False or own >= 0.3:
            return "skirt"
        return "trousers" if z["leg"] >= 0.15 else "shorts"
    return "other"


# kinds read from a name that say where the part is worn: kept only if the part's dominant zone
# is there ("Belt" also names arm straps, "Tie" hip ribbons)
ZONES_KIND = {"belt": ("torso", "pelvis"), "tie": ("neck", "torso"), "headwear": ("head",),
               "glasses": ("head",), "mask": ("head", "neck"), "headphones": ("head", "neck"),
               "ears": ("head",), "loop_ear": ("head",), "ring": ("hand",), "necklace": ("neck", "torso"),
               "garter": ("thigh", "leg"), "scarf": ("neck", "torso")}

# animal ear bones (Bdy_L_Ear_01, Ctr_Tea_R_Ear_01, MY_Bone_Ear_L); not Earring, Earphone, EarHair
EAR_OS = re.compile(r"(^|_)Ear(_|\d|$)")


def perimeter_seeded(T, faces, os_axe):
    """Share of the 24 sectors around `os_axe` covered by points seeded on the faces
    (vertices, edge midpoints, centres) closest to this bone. A band of large quads does not
    have a vertex in every sector: Nicole's thigh strap covers 0.62 of the turn by
    its vertices, 0.79 by its faces."""
    M = T.M
    P3 = M.pos_s[M.Fs[faces]]
    pts = np.concatenate([P3.reshape(-1, 3), ((P3 + np.roll(P3, 1, 1)) / 2).reshape(-1, 3), P3.mean(1)])
    k, _, ang, _ = os_near(pts, T.seg)
    m = T.seg[0][k] == os_axe
    if not m.any():
        return 0.0
    return len(np.unique(((ang[m] + np.pi) / (2 * np.pi) * 24).astype(int) % 24)) / 24


def kind_accessory(T, pc, g):
    M = T.M
    z = zones_part(T, pc["faces"])
    zd = max(z, key=z.get) if z else None
    for what, names in (("material", pc["materials"]), ("bone", [n for n, _ in pc["own_bones"][:3]])):
        for n in names:
            s = kind_name(n)
            if s and (s not in ZONES_KIND or zd in ZONES_KIND[s]):
                return s, "%s name (%s)" % (what, n)
    reg = region_axe(T, g.os_axe)
    # seeded turn: only around the limbs and the neck, where bands of large quads fool the
    # per-vertex turn; around the trunk it would take flaps and clips for rings
    # (Nicole strap-12, Soldat11 ornament-21, Remielle strap-15: seen on the thumbnails)
    girth = g.girth
    if reg in ("thigh", "leg", "foot", "arm", "forearm", "hand", "neck") and girth < LOOP_PERIMETER:
        girth = max(girth, perimeter_seeded(T, pc["faces"], g.os_axe))
    loop = g.extent_axe < LOOP_TOP * 1.5 and girth >= LOOP_PERIMETER
    if loop:
        s = {"pelvis": "belt", "buttocks": "belt", "belly": "belt", "thigh": "garter",
             "leg": "garter", "arm": "armband", "forearm": "bracelet", "hand": "bracelet",
             "neck": "necklace", "head": "headwear", "foot": "anklet"}.get(reg, "loop")
        # the leg bone (ひざ) runs from knee to ankle: a ring on its lower half is at the
        # ankle (Yuzuha, Zhao), not a garter
        if reg == "leg":
            seg = T.seg
            j = int(np.flatnonzero(seg[0] == g.os_axe)[0])
            L = np.linalg.norm(seg[2][j] - seg[1][j])
            k, t, _, _ = os_near(M.pos_s[M.Fs[pc["faces"]]].mean(1), seg)
            if (k == j).any() and np.median(t[k == j]) > 0.5 * L:
                s = "anklet"
        # a ring entirely on the head is headwear, even if its axis is the neck:
        # Orphie's beret (体2, resting on 頭) turns around the extension of 首
        if zd == "head":
            s = "headwear"
        return s, "ring around %s" % M.names_os[g.os_axe]
    # animal ears: head part carried by its own ear bones, neither tiny (earring)
    # nor a narrow band (Soukaku, Jane, Pulchra, Chinatsu: 体, 毛2, 饰 with no ear name)
    if pc["own_bones"] and EAR_OS.search(pc["own_bones"][0][0]) and zd == "head" \
            and g.diag >= ACCESSORY_DIAG and g.width >= STRAP_WIDTH * 1.3:
        return "ears", "ear bone (%s)" % pc["own_bones"][0][0]
    if any(x.startswith("hair") for x in pc.get("reason", [])):
        return "hair", pc["reason"][0]
    if any(x.startswith("coiffe") for x in pc.get("reason", [])):
        return "headwear", pc["reason"][0]
    if any(x.startswith("bands") for x in pc.get("reason", [])):
        return ("harness" if reg in ("trunk_top", "chest", "belly", "pelvis") else "strap"), "bands"
    if g.diag < ACCESSORY_DIAG:
        return "fastener", "small part"
    if g.width < STRAP_WIDTH * 1.3:
        return ("harness" if reg in ("trunk_top", "chest", "belly") and g.diag > 0.3 else "strap"), "narrow band"
    ep = np.sort(g.hi - g.lo)
    if reg in ("pelvis", "buttocks", "thigh") and ep[0] > 0.02:
        return "pocket", "volume lying on the hip or thigh"
    if ep[0] < 0.012 and g.diag < 0.2:
        return "crest", "flat part"
    if reg == "head":
        return "headwear", "on the head"
    return "ornament", "other"


def layers(T, parts):
    """Layer of each garment: 1 + number of garments it covers (larger radius
    around the same bones, on at least 10 % of the shared cells)."""
    cases = [cases_body(T, pc["faces"]) for pc in parts]
    n = len(parts)
    over = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            com = set(cases[i]) & set(cases[j])
            if len(com) < max(4, 0.1 * min(len(cases[i]), len(cases[j]))):
                continue
            d = np.array([cases[i][c] - cases[j][c] for c in com])
            if (d > 0.003).mean() > 0.6:
                over[i].add(j)
            elif (d < -0.003).mean() > 0.6:
                over[j].add(i)
    return [1 + len(s) for s in over], over


PAIRS = ("gloves", "shoes", "boots", "socks", "sleeves")
NAMES_FR = {"top": "top", "jacket": "jacket", "waistcoat": "waistcoat", "coat": "coat", "dress": "dress",
           "jumpsuit": "jumpsuit", "skirt": "skirt", "trousers": "trousers", "shorts": "shorts",
           "socks": "socks", "shoes": "shoes", "boots": "boots", "gloves": "gloves",
           "collar": "collar", "hood": "hood", "sleeves": "sleeves", "other": "garment"}


FILLING_MIN = 0.75   # area / area of the body cells crossed: below this threshold, bands (harness)


def filling(T, faces):
    cs = cases_body(T, faces)
    r = np.maximum(np.array(list(cs.values())), 0.02)
    return float(T.M.area[faces].sum() / (0.025 * 2 * np.pi * r / 12).sum())


def os_of(T, faces):
    """Bones carrying the faces' vertices: [(name, weight share)], body bones and own bones."""
    M = T.M
    v = np.unique(M.F[faces])
    w = np.zeros(len(M.names_os))
    for k in range(M.idx.shape[1]):
        b, p = M.idx[v, k], M.weight[v, k]
        ok = (b >= 0) & (p > 0)
        np.add.at(w, b[ok], p[ok])
    w /= max(w.sum(), 1e-20)
    o = np.argsort(-w)
    body = [(M.names_os[b], round(float(w[b]), 3)) for b in o if w[b] > 0.005 and T.os_body[b]]
    own_ones = [(M.names_os[b], round(float(w[b]), 4)) for b in o if w[b] > 0 and not T.os_body[b]]
    return body, own_ones, float(sum(w[b] for b in range(len(w)) if not T.os_body[b]))


SLEEVE_OF_REST = 20.0    # minimum ΔE between an arm cluster and the rest of its garment
SLEEVE_OF_TOP = 10.0     # maximum ΔE between the cluster and the sleeveless top that receives it
SLEEVE_DIST = 0.04        # m: maximum cluster-top distance
SLEEVE_AREA = 0.005       # m²: minimum area of the cluster
SLEEVE_CASES = 0.1        # maximum share of the cluster's body cells that the top already covers


def _lab_average(T, ats):
    ats = np.asarray(sorted(ats))
    w = T.area_atom[ats]
    return (T.lab_atom[ats] * w[:, None]).sum(0) / max(w.sum(), 1e-20)


def move_sleeves(T, garment):
    """Jacket sleeves left in the garment underneath (ZhuYuan: the jacket's blue sleeves
    sewn to the armholes of the grey jumpsuit, with no seam to the jacket body):
    a cluster of arm atoms (upper arm + forearm ≥ 0.5, sewn to each other) whose colour is
    clearly different from the rest of its garment (ΔE ≥ 20) moves to a sleeveless top
    (torso ≥ 0.5, arms < 0.05) of its colour (ΔE ≤ 10), within 4 cm, that does not already
    cover it (< 10 % shared cells); atoms of the same colour sewn to the cluster (armholes)
    follow it. Decisions are made on the prior state, then the moves are applied. Measured on the 49 outfits:
    only ZhuYuan triggers it."""
    M = T.M
    faces_of = lambda ats: np.concatenate([T.faces_atom[a] for a in sorted(ats)])  # noqa: E731
    zq = [zones_part(T, pc["faces"]) for pc in garment]
    lab_v = [_lab_average(T, pc["atoms"]) for pc in garment]
    arm_at = lambda a: T.zones(T.R_atom[a])["arm"] + T.zones(T.R_atom[a])["forearm"]  # noqa: E731
    decisions = []
    for ip, P in enumerate(garment):
        ats = set(P["atoms"])
        armat = [a for a in ats if T.nf_atom[a] >= 4 and arm_at(a) >= 0.5]
        if not armat:
            continue
        uf = UF(T.na)
        sa = set(armat)
        for (x, y) in T.seams:
            if x in sa and y in sa:
                uf.u(x, y)
        for G in uf.groups(armat):
            G = set(G)
            rest = ats - G
            if T.area_atom[sorted(G)].sum() < SLEEVE_AREA or not rest:
                continue
            labG = _lab_average(T, G)
            if np.linalg.norm(labG - _lab_average(T, rest)) < SLEEVE_OF_REST:
                continue
            fg = faces_of(G)
            pg = M.pos_s[np.unique(M.Fs[fg])]
            cg = None
            best = None
            for iq, Q in enumerate(garment):
                z = zq[iq]
                if iq == ip or z["torso"] < 0.5 or z["arm"] + z["forearm"] >= 0.05:
                    continue
                de = float(np.linalg.norm(labG - lab_v[iq]))
                if de > SLEEVE_OF_TOP:
                    continue
                pq = M.pos_s[np.unique(M.Fs[Q["faces"]])]
                dm = min(np.linalg.norm(pg[i:i + 400, None] - pq[None], axis=2).min() for i in range(0, len(pg), 400))
                if dm > SLEEVE_DIST:
                    continue
                cg = cg if cg is not None else cases_body(T, fg)
                if len(set(cg) & set(cases_body(T, Q["faces"]))) >= SLEEVE_CASES * max(len(cg), 1):
                    continue
                if best is None or de < best[0]:
                    best = (de, iq)
            if best is None:
                continue
            g = set(G)
            finished = False
            while not finished:
                finished = True
                for (x, y), d in T.seams.items():
                    for a, o in ((x, y), (y, x)):
                        if a in g and o in rest and o not in g and d["length"] > 0.01 \
                                and np.linalg.norm(T.lab_atom[o] - labG) <= SLEEVE_OF_TOP:
                            g.add(o)
                            finished = False
            decisions.append((ip, best[1], g))
    for ip, iq, g in decisions:
        for k, plus in ((ip, False), (iq, True)):
            pc = garment[k]
            pc["atoms"] = sorted((set(pc["atoms"]) | g) if plus else (set(pc["atoms"]) - g))
            pc["faces"] = np.sort(faces_of(pc["atoms"]))
            pc["materials"] = sorted({M.names_mat[m] for m in np.unique(M.mf[pc["faces"]])})
            pc["os_body"], pc["own_bones"], pc["part_own_bones"] = os_of(T, pc["faces"])
        garment[iq].setdefault("reason", []).append("same-colour sleeves taken back from the garment underneath")
    return decisions


def describe(T, parts, info):
    """Completes the parts: materials, bones, type or kind, layer, carrier, hidden skin, overlay layers,
    identifier; merges left-right pairs of the same type."""
    M = T.M
    for pc in parts:
        pc["materials"] = sorted({M.names_mat[m] for m in np.unique(M.mf[pc["faces"]])})
        pc["os_body"], pc["own_bones"], pc["part_own_bones"] = os_of(T, pc["faces"])
    # braces, harness, scarf: a "garment" that covers only a small share of the body cells
    # it crosses is made of bands (only named material affected, measured on the 49
    # outfits: Lighter's scarf 围巾, 0.45; all other garments ≥ 0.75)
    for pc in parts:
        if pc["class"] == "garment" and "type_force" not in pc:
            r = filling(T, pc["faces"])
            if r < FILLING_MIN:
                pc["class"] = "accessory"
                pc["reason"] = ["bands (fill ratio %.2f)" % r]
    # head part that is not a hood: in 毛 material, on hair bones, or not wrapping around
    # the skull. It is hair (Pulchra's 毛2, Nicole's 新規 pigtails) if the material or
    # the bones say so, or if it is in generic material with no ear bone; otherwise a headdress
    # (material named after a fabric: Corin's frilled headdress 裙) or ears (Ear bones)
    for pc in parts:
        if pc["class"] != "garment" or "type_force" in pc or zones_part(T, pc["faces"])["head"] < 0.5:
            continue
        tc = perimeter_of_skull(T, pc["faces"])
        name_ch = (any(n.startswith("毛") for n in pc["materials"])
                  or any(HAIR_OS.search(n) for n, _ in pc["own_bones"][:5]))
        if not name_ch and not (tc is not None and tc < 0.5):
            continue
        pc["class"] = "accessory"
        ear = bool(pc["own_bones"]) and bool(EAR_OS.search(pc["own_bones"][0][0]))
        tcs = None if tc is None else round(tc, 2)
        # in generic material, with no hair name: strands move on their own bones
        # (Nicole's pigtails); a rigid part, entirely on the head bone, is a worn object
        # (Orphie's earmuffs): headwear
        if name_ch or (all(mat_generic(n) for n in pc["materials"]) and not ear and pc["own_bones"]):
            pc["reason"] = ["hair (skull coverage %s)" % tcs]
        elif ear:
            pc["reason"] = ["ear bones (skull coverage %s)" % tcs]
        elif all(mat_generic(n) for n in pc["materials"]):
            pc["reason"] = ["coiffe: rigid on the head, no own bones, does not wrap around the skull (%s)" % tcs]
        else:
            pc["reason"] = ["coiffe: named material, does not wrap around the skull (%s)" % tcs]
    garment = [pc for pc in parts if pc["class"] == "garment"]
    accessory = [pc for pc in parts if pc["class"] == "accessory"]
    cs, _ = layers(T, garment)
    for pc, c in zip(garment, cs):
        pc["type"] = type_garment(T, pc["faces"], c, pc["part_own_bones"], False)
    # left-right pairs of the same type: one part
    finished = False
    while not finished:
        finished = True
        for i in range(len(garment)):
            for j in range(i + 1, len(garment)):
                a, b = garment[i], garment[j]
                if a["type"] == b["type"] and a["type"] in PAIRS and mirrors(T, a["faces"], b["faces"]):
                    a["atoms"] = sorted(a["atoms"] + b["atoms"])
                    a["faces"] = np.sort(np.concatenate([a["faces"], b["faces"]]))
                    a["materials"] = sorted(set(a["materials"]) | set(b["materials"]))
                    a["os_body"], a["own_bones"], a["part_own_bones"] = os_of(T, a["faces"])
                    garment.pop(j)
                    finished = False
                    break
            if not finished:
                break
    cs, over = layers(T, garment)
    move_sleeves(T, garment)
    zh = [zones_part(T, pc["faces"]) for pc in garment]
    # a covered garment counts as a top if it covers the torso and arms, or if it is a
    # jumpsuit or a dress (ZhuYuan: her jumpsuit, without the jacket's sleeves, only has
    # 0.26 of top left, but the jacket covers it)
    is_top = [z["torso"] + z["arm"] + z["forearm"] >= 0.45 or pc["type"] in ("jumpsuit", "dress")
                for pc, z in zip(garment, zh)]
    for k, (pc, c) in enumerate(zip(garment, cs)):
        pc["layer"] = c
        pc["type"] = pc.get("type_force") or type_garment(T, pc["faces"], c, pc["part_own_bones"],
                                                           any(is_top[j] for j in over[k]))
        pc["covers"] = sorted(over[k])
    # carrier of an accessory: the garment it touches most (2 cm cells)
    all_ = garment + accessory
    cont = contacts(T, [pc["faces"] for pc in all_], 0.02)
    for k, pc in enumerate(accessory):
        i = len(garment) + k
        best, nb = None, 0
        for j in range(len(garment)):
            n = cont.get((min(i, j), max(i, j)), 0)
            if n > nb:
                best, nb = j, n
        pc["carrier_index"] = best
        g = Geo(T, pc["faces"])
        pc["sort"], pc["kind_reason"] = kind_accessory(T, pc, g)
        pc["layer"] = (garment[best]["layer"] + 1) if best is not None else 1
    # no garment in its cells: whatever its surface touches most closely, a garment, another
    # accessory (a belt's buckle, a strap's button) or skin (worn directly on the body)
    groups = [pc["faces"] for pc in all_] + [info["skin_faces"]]
    for k, pc in enumerate(accessory):
        if pc["carrier_index"] is not None:
            continue
        i = len(garment) + k
        nb = plus_near_surface(T, pc["faces"], groups[:i] + [np.zeros(0, int)] + groups[i + 1:])
        # skin carries nothing: it is only underneath. A part lying flat on the skin that
        # also touches another part (the strap end of a garter) belongs to that part
        j = int(nb[:len(all_)].argmax())
        if nb[j] < max(2, WORN_PART * nb.sum()):
            continue
        if j < len(garment):
            pc["carrier_index"] = j
            pc["layer"] = garment[j]["layer"] + 1
        else:
            pc["carrier_accessory"] = all_[j]
    # accessory chain: a trim lying on another trim follows that trim's carrier;
    # lying on a selectable accessory (belt, bag, harness), it belongs to that accessory
    for pc in accessory:
        x, seen_ = pc.get("carrier_accessory"), 0
        while x is not None and x["sort"] in TRIMS and seen_ < 8:
            if x["carrier_index"] is not None:
                pc["carrier_index"] = x["carrier_index"]
                pc["layer"] = garment[x["carrier_index"]]["layer"] + 1
                x = None
                break
            x, seen_ = x.get("carrier_accessory"), seen_ + 1
        pc["carrier_accessory"] = x if (x is not None and x["sort"] not in TRIMS and x is not pc) else None
    # left-right pairs of accessories of the same kind (two bracelets, two garters): one part
    k = 0
    while k < len(accessory):
        a = accessory[k]
        for j in range(k + 1, len(accessory)):
            b = accessory[j]
            if a["sort"] == b["sort"] and mirrors(T, a["faces"], b["faces"]):
                a["atoms"] = sorted(a["atoms"] + b["atoms"])
                a["faces"] = np.sort(np.concatenate([a["faces"], b["faces"]]))
                a["materials"] = sorted(set(a["materials"]) | set(b["materials"]))
                a["os_body"], a["own_bones"], a["part_own_bones"] = os_of(T, a["faces"])
                a["reason"] = sorted(set(a.get("reason", [])) | set(b.get("reason", [])))
                a["pair"] = True
                b["blended_in"] = a
                accessory.pop(j)
                break
        k += 1
    all_ = garment + accessory
    # hidden skin: body zones covered; skin-tight if sewn to the skin
    skin = T.is_skin
    for pc in all_:
        z = zones_part(T, pc["faces"])
        pc["zones"] = {k: round(v, 3) for k, v in z.items() if v >= 0.01}
        ats = set(pc["atoms"])
        sewn = sum(d["length"] for (a, b), d in T.seams.items()
                    if (a in ats and skin[b]) or (b in ats and skin[a]))
        pc["sewn_skin_mm"] = round(sewn * 1000, 1)
    # overlay layers ("+"): each face joins the part that carries the same welded face (or the skin);
    # with no exact twin, the part that carries most of its welded vertices
    cal = np.where(T.texture_layer)[0]
    carriers = [pc["faces"] for pc in all_] + [info["skin_faces"]]
    pc_of = np.full(len(M.F), -1)
    for k, f in enumerate(carriers):
        pc_of[f] = k
    skin_cal = []
    info["texture_layers_without_twin"] = 0
    if len(cal):
        the_key = lambda F: np.sort(F, 1) @ np.array([1, 1 << 20, 1 << 40], dtype=np.int64)  # noqa: E731
        base = np.where(pc_of >= 0)[0]
        kb = the_key(M.Fs[base])
        o = np.argsort(kb)
        kc = the_key(M.Fs[cal])
        pos = np.clip(np.searchsorted(kb[o], kc), 0, len(kb) - 1)
        found = kb[o][pos] == kc
        info["texture_layers_without_twin"] = int((~found).sum())
        pc_v = np.full(len(M.pos_s), -1)
        for k, f in enumerate(carriers):
            pc_v[M.Fs[f].ravel()] = k
        for f, ok, pp in zip(cal, found, pos):
            if ok:
                k = pc_of[base[o][pp]]
            else:
                c = pc_v[M.Fs[f]]
                c = c[c >= 0]
                k = np.bincount(c).argmax() if len(c) else len(all_)
            if k >= len(all_):
                skin_cal.append(int(f))
            else:
                all_[k].setdefault("texture_layers", []).append(int(f))
    info["skin_texture_layers"] = np.array(skin_cal, int)
    for pc in all_:
        pc["layer_faces"] = np.array(sorted(pc.pop("texture_layers", [])), int)
    # identifiers
    seen = {}
    for pc in garment + accessory:
        name = NAMES_FR.get(pc.get("type"), None) or pc["sort"]
        seen[name] = seen.get(name, 0) + 1
        pc["id"] = name if seen[name] == 1 else "%s-%d" % (name, seen[name])
    for pc in garment + accessory:
        n = seen[pc["id"].split("-")[0]]
        if n > 1 and "-" not in pc["id"]:
            pc["id"] = pc["id"] + "-1"
    for pc in accessory:
        pc["carrier"] = garment[pc["carrier_index"]]["id"] if pc["carrier_index"] is not None else None
        x = pc.pop("carrier_accessory", None)
        while x is not None and "blended_in" in x:
            x = x["blended_in"]
        if pc["carrier"] is None and x is not None and x is not pc:
            pc["carrier"] = x["id"]
    for pc in garment:
        pc["covers"] = [garment[j]["id"] for j in pc["covers"]]
    return garment + accessory


# ====================================================================== hidden skin

def hidden_skin(T, faces, skin_faces, margin=0.001):
    """Skin faces under the part: their three vertices fall in cells (bone, slice,
    sector) covered by the part, closer to the bone than the part."""
    if len(skin_faces) == 0:
        return np.zeros(0, int)
    cp = cases_body(T, faces)
    M = T.M
    vs = np.unique(M.Fs[skin_faces])
    k, t, ang, r = (a[vs] for a in T.axe_s)
    c = (k.astype(np.int64) * 1000 + (t / 0.025).astype(np.int64)) * 100 + \
        ((ang + np.pi) / (2 * np.pi) * 12).astype(np.int64) % 12
    rp = np.array([cp.get(int(x), -1.0) for x in c])
    under = np.zeros(len(M.pos_s), bool)
    under[vs] = r < rp + margin
    return skin_faces[under[M.Fs[skin_faces]].all(1)]


# ====================================================================== PMX writing

sys.path.insert(0, os.path.join(HERE, "..", "head"))


def _write_pmx_module():
    import pmx_write as pmx_write  # tools/head (read-only: imported, not modified)
    return pmx_write


def _link(src, dst):
    """Copies a texture: hard link if possible (same disk), otherwise a copy."""
    if os.path.exists(dst):
        return
    tmp = dst + ".tmp"
    try:
        os.link(src, tmp)
    except OSError:
        import shutil
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def os_kept(M, verts, own_body):
    """Bones to keep for some vertices: those carrying weight, their ancestors, the descendants of
    own bones (ends of physics chains), append sources; rigid bodies on own bones and
    their anchors (rigid bodies linked by a joint), joints between kept rigid bodies."""
    p = M.p
    bones = p["bones"]
    n = len(bones)
    guard = np.zeros(n, bool)
    b = M.idx[verts].ravel()
    w = M.weight[verts].ravel()
    guard[np.unique(b[(b >= 0) & (w > 0)])] = True
    children = [[] for _ in range(n)]
    for i, o in enumerate(bones):
        if 0 <= o["parent"] < n:
            children[o["parent"]].append(i)
    own = ~own_body
    pile = [i for i in np.where(guard & own)[0]]
    while pile:
        i = pile.pop()
        for e in children[i]:
            if own[e] and not guard[e]:
                guard[e] = True
                pile.append(e)
    rig = p["rigid_bodies"]
    jts = p["joints"]
    rg = np.zeros(len(rig), bool)
    for k, r in enumerate(rig):
        if 0 <= r["bones"] < n and guard[r["bones"]] and own[r["bones"]]:
            rg[k] = True
    for j in jts:
        a, c = j["a"], j["b"]
        if 0 <= a < len(rig) and 0 <= c < len(rig) and (rg[a] or rg[c]):
            for x in (a, c):
                if not rg[x]:
                    rg[x] = True
                    if 0 <= rig[x]["bones"] < n:
                        guard[rig[x]["bones"]] = True
    jg = [k for k, j in enumerate(jts) if 0 <= j["a"] < len(rig) and 0 <= j["b"] < len(rig)
          and rg[j["a"]] and rg[j["b"]]]
    # body collision rigid bodies (static, on a body bone) that the part's dynamic rigid bodies
    # hit within 12 cm: without them the fitted skirt clips through the legs.
    dyn = [k for k in np.where(rg)[0] if rig[k]["mode"] in (1, 2)]
    if dyn:
        cd = np.array([rig[k]["pos"] for k in dyn], float)
        for k, r in enumerate(rig):
            if rg[k] or r["mode"] != 0 or not (0 <= r["bones"] < n) or own[r["bones"]]:
                continue
            hit = [kk for kk in dyn if (r["mask"] >> rig[kk]["group"]) & 1 and (rig[kk]["mask"] >> r["group"]) & 1]
            if hit and np.linalg.norm(cd[[dyn.index(x) for x in hit]] - np.asarray(r["pos"], float), axis=1).min() \
                    - max(r["size"]) < 1.5:
                rg[k] = True
                guard[r["bones"]] = True
    while True:
        before = guard.sum()
        for i in np.where(guard)[0]:
            o = bones[i]
            if 0 <= o["parent"] < n:
                guard[o["parent"]] = True
            if o["addition"] and 0 <= o["addition"]["source"] < n:
                guard[o["addition"]["source"]] = True
        if guard.sum() == before:
            break
    return np.where(guard)[0], np.where(rg)[0], jg


def part_pmx(T, faces, name, folder_tex_rel, textures_copied):
    """Standalone PMX dict of the given faces (indices into M.F): vertices, faces, materials,
    textures (relative paths), bones, rigid bodies, joints, display frames. No morphs."""
    import copy
    M = T.M
    p = M.p
    faces = np.asarray(faces)
    faces = faces[np.argsort(M.mf[faces], kind="stable")]
    raw = np.asarray(p["faces"])[faces]
    vs = np.unique(raw)
    nv = np.full(len(M.pos), -1)
    nv[vs] = np.arange(len(vs))
    ob, rg, jg = os_kept(M, vs, np.array(T.os_body))
    no = np.full(len(p["bones"]), -1)
    no[ob] = np.arange(len(ob))
    nr = np.full(len(p["rigid_bodies"]), -1)
    nr[rg] = np.arange(len(rg))
    S0 = p["vertices"]
    S = {k: [S0[k][i] for i in vs] for k in S0}
    S["bones"] = [tuple(int(no[b]) if b >= 0 and no[b] >= 0 else (0 if b >= 0 else -1) for b in o) for o in S["bones"]]
    # bones
    bones = []
    for i in ob:
        o = copy.deepcopy(p["bones"][i])
        o["parent"] = int(no[o["parent"]]) if 0 <= o["parent"] < len(no) else -1
        if o["tail_bone"] is not None:
            o["tail_bone"] = int(no[o["tail_bone"]]) if 0 <= o["tail_bone"] < len(no) else -1
        if o["addition"]:
            s = o["addition"]["source"]
            if 0 <= s < len(no) and no[s] >= 0:
                o["addition"]["source"] = int(no[s])
            else:
                o["addition"] = None
                o["flags"] &= ~0x0300
        if o["ik"]:
            ik = o["ik"]
            ok = 0 <= ik["target"] < len(no) and no[ik["target"]] >= 0 and all(no[l["bones"]] >= 0 for l in ik["links"])
            if ok:
                ik["target"] = int(no[ik["target"]])
                for l in ik["links"]:
                    l["bones"] = int(no[l["bones"]])
            else:
                o["ik"] = None
                o["flags"] &= ~0x0020
        bones.append(o)
    rig = []
    for k in rg:
        r = copy.deepcopy(p["rigid_bodies"][k])
        r["bones"] = int(no[r["bones"]]) if 0 <= r["bones"] < len(no) else -1
        rig.append(r)
    jts = []
    for k in jg:
        j = copy.deepcopy(p["joints"][k])
        j["a"], j["b"] = int(nr[j["a"]]), int(nr[j["b"]])
        jts.append(j)
    # materials and textures
    mats, tex, tmap = [], [], {}
    folder = os.path.dirname(M.path)

    def t_index(t):
        if t < 0 or t >= len(p["textures"]):
            return -1
        if t not in tmap:
            src = os.path.normpath(os.path.join(folder, p["textures"][t].replace("\\", "/")))
            base = os.path.basename(src)
            if os.path.exists(src):
                textures_copied[base] = src
            tmap[t] = len(tex)
            tex.append((folder_tex_rel + "/" + base).replace("/", "\\"))
        return tmap[t]
    Fout = []
    for m in np.unique(M.mf[faces]):
        sel = M.mf[faces] == m
        mm = copy.deepcopy(p["materials"][int(m)])
        mm["texture"] = t_index(mm["texture"])
        mm["sphere"] = t_index(mm["sphere"])
        if mm["toon_sharing"] == 0:
            mm["toon"] = t_index(mm["toon"])
        mm["n_indices"] = int(3 * sel.sum())
        mats.append(mm)
        Fout.extend(tuple(int(nv[v]) for v in f) for f in raw[sel])
    # display frames: root, expressions (empty), all bones; rigid bodies are not in them
    frames = [{"name": "Root", "name_en": "Root", "special": 1, "elements": [(0, 0)]},
              {"name": "表情", "name_en": "Exp", "special": 1, "elements": []},
              {"name": "bones", "name_en": "bones", "special": 0, "elements": [(0, i) for i in range(1, len(bones))]}]
    out = {k: p[k] for k in ("version", "encoding", "n_uv_add")}
    out.update({"sizes_index": p.get("sizes_index"), "name": name, "name_en": name,
                "comment": "Part of %s (ZZZ clothing library)" % T.character,
                "comment_en": "", "vertices": S, "faces": Fout, "textures": tex, "materials": mats,
                "bones": bones, "morphs": [], "frames": frames, "rigid_bodies": rig, "joints": jts,
                "bytes_remaining": b""})
    return out, {"vertices": len(vs), "bones": len(ob), "rigid_bodies": len(rg), "joints": len(jg),
                 "os_physiques": int(sum(1 for r in rig if r["mode"] != 0))}


# ====================================================================== automatic check

INLAY_MIN = 0.9    # share of an atom's border sewn to the garment above which it is an inlaid panel


def part_sewn(T, a):
    """Length sewn to other atoms / length of the atom's (welded) border."""
    M = T.M
    Fs = M.Fs[T.faces_atom[a]]
    E = np.sort(np.concatenate([Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    b = u[c == 1]
    Lb = np.linalg.norm(M.pos_s[b[:, 0]] - M.pos_s[b[:, 1]], axis=1).sum()
    Lc = sum(d["length"] for (x, y), d in T.seams.items() if a in (x, y))
    return Lc / max(Lb, 1e-9)


def check(T, parts, info):
    """Three checks per outfit. Returns {"lost_faces", "duplicate_faces", "two_garments",
    "accessory_left"} (lists of anomalies) and the face tally."""
    M = T.M
    outfit = np.array([T.roles[m] not in ("head", "empty") for m in M.mf])
    count = np.zeros(len(M.F), int)
    for pc in parts:
        count[pc["faces"]] += 1
        count[pc["layer_faces"]] += 1
    count[info["skin_faces"]] += 1
    count[info.get("skin_texture_layers", np.zeros(0, int))] += 1
    lost = np.where(outfit & (count == 0))[0]
    duplicates = np.where(count > 1)[0]
    two, rest, inner_layers, inlay = [], [], [], []
    for pc in parts:
        if pc["class"] != "garment":
            continue
        z = pc["zones"]
        sl = {s: sum(z.get(x, 0) for x in zs) for s, zs in SLOTS.items()}
        strong = [s for s, v in sl.items() if v >= 0.15]
        for i in range(len(strong)):
            for j in range(i + 1, len(strong)):
                if frozenset((strong[i], strong[j])) not in SLOTS_NEIGHBORS and \
                        {strong[i], strong[j]} != {"top", "bottom"} and pc["type"] not in ("dress", "coat", "jumpsuit"):
                    two.append((pc["id"], "distant slots %s+%s" % (strong[i], strong[j])))
        # two layers in one part: panels (sewn groups of atoms) of >= 10 % that overlap
        ats = pc["atoms"]
        if len(ats) > 1:
            u = UF(T.na)
            sa = set(ats)
            for (a, b), d in T.seams.items():
                if a in sa and b in sa:
                    u.u(a, b)
            gr = [g for g in u.groups(ats) if sum(T.nf_atom[a] for a in g) >= 0.1 * len(pc["faces"])]
            if len(gr) > 1:
                fl = [np.concatenate([T.faces_atom[a] for a in g]) for g in gr]
                _, dess = layers(T, [{"faces": f} for f in fl])
                if any(dess):
                    mats_gr = [{T.mat_atom[a] for a in g} for g in gr]
                    one_name = len(set().union(*mats_gr)) == 1 and not mat_generic(M.names_mat[next(iter(mats_gr[0]))])
                    if one_name:
                        # a single named material (下摆, 外套): lapels, flaps, lining of the same garment
                        inner_layers.append((pc["id"], "%d overlapping panels, material %s" % (len(gr), pc["materials"][0])))
                    else:
                        two.append((pc["id"], "two layers in the part (%d panels)" % len(gr)))
        # accessory left inside: small or ring-shaped atom, with a colour very different from its sewn neighbours
        for a in ats:
            g = Geo(T, T.faces_atom[a])
            small = g.diag < ACCESSORY_DIAG or (g.extent_axe < LOOP_TOP and g.girth >= LOOP_PERIMETER and g.diag < 0.5)
            if not small:
                continue
            dEs = [d["dE"] for (x, y), d in T.seams.items() if a in (x, y) and (y if x == a else x) in sa]
            if dEs and min(dEs) > 2 * OF_SAME_FABRIC:
                # border fully sewn: inlay (a coloured panel in the fabric, removing it
                # would leave a hole); border partly free: a part laid on and welded, to report
                if part_sewn(T, a) >= INLAY_MIN:
                    inlay.append((pc["id"], a))
                    continue
                rest.append((pc["id"], "atom %d (%d faces, %.0f mm, ΔE %.0f)" % (a, T.nf_atom[a], g.diag * 1000, min(dEs))))
    return {"outfit_faces": int(outfit.sum()), "part_faces": int(sum(len(pc["faces"]) + len(pc["layer_faces"]) for pc in parts)),
            "skin_faces": int(len(info["skin_faces"]) + len(info.get("skin_texture_layers", []))),
            "lost_faces": int(len(lost)), "duplicate_faces": int(len(duplicates)),
            "two_garments": two, "accessory_left": rest, "allowed_inner_layers": inner_layers,
            "allowed_inlays": len(inlay),
            "passed": len(lost) == 0 and len(duplicates) == 0 and not two and not rest}


# ====================================================================== extracting a character

def _json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, path)


def zones_skin(T, faces):
    if len(faces) == 0:
        return {}
    z = zones_part(T, faces)
    return {k: round(v, 3) for k, v in z.items() if v >= 0.01}


# ====================================================================== hard cases (named)
# Characters whose mesh does not separate a garment (e.g. a cardigan welded to the dress): described in the
# game description (people.<character>.exception_garment: spec = a record of measure.CLOTHES, cut by
# its texture area; type; why), with no new record.
EXCEPTIONS = MD.games.case_("exception_garment")


def clarity_uv_oiio(M):
    """Like model.clarity_uv, with OpenImageIO (outside Blender)."""
    L = np.full(len(M.F), np.nan)
    c = M.uv[M.F].mean(1)
    for mi in np.unique(M.mf):
        ch = path_texture(M, mi)
        im = read_texture(ch) if ch else None
        if im is None:
            continue
        h, w = im.shape[:2]
        fi = np.where(M.mf == mi)[0]
        u = np.clip((c[fi, 0] % 1) * w, 0, w - 1).astype(int)
        v = np.clip((c[fi, 1] % 1) * h, 0, h - 1).astype(int)
        L[fi] = im[v, u, :3] @ [0.299, 0.587, 0.114]
    return L


def apply_exception(T, parts, ex):
    """Splits off from a garment the faces that the measure.py record assigns to it."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("measure_garment", os.path.join(HERE, "measure.py"))
    mes = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mes)
    spec = next(v for v in mes.CLOTHES if v["id"] == ex["spec"])
    old = MD.clarity_uv
    MD.clarity_uv = clarity_uv_oiio
    try:
        sel = MD.isolate(T.M, spec)[0]
    finally:
        MD.clarity_uv = old
    garments = [pc for pc in parts if pc["class"] == "garment"]
    pc = max(garments, key=lambda x: sel[x["faces"]].sum())
    inside = sel[pc["faces"]]
    if inside.all() or not inside.any():
        return parts
    a, b = pc["faces"][inside], pc["faces"][~inside]
    # pieces of the remainder carried by the arms (dark cuffs set aside by the measurement): go to the cardigan
    mb = np.zeros(len(T.M.F), bool)
    mb[b] = True
    for g in T.M.parts(mb):
        z = zones_part(T, g)
        if z["arm"] + z["forearm"] + z["hand"] >= 0.5:
            mb[g] = False
    a = np.sort(np.concatenate([a, b[~mb[b]]]))
    b = b[mb[b]]
    ats_a = [x for x in pc["atoms"] if sel[T.faces_atom[x]].mean() >= 0.5]
    ats_b = [x for x in pc["atoms"] if x not in ats_a]
    pc["faces"], pc["atoms"] = b, ats_b
    parts.append({"class": "garment", "faces": np.sort(a), "atoms": ats_a, "type_force": ex["type"],
                   "exception": ex["why"]})
    return parts


def analyze(character, path=None):
    T = Outfit(character, path=path)
    parts, info = slice_(T)
    if character in EXCEPTIONS and path is None:
        parts = apply_exception(T, parts, EXCEPTIONS[character])
    parts = describe(T, parts, info)
    return T, parts, info, check(T, parts, info)


def put_of_side(src, dst):
    """Moves a folder (rename); if the file system refuses to rename a folder
    (overlay: EXDEV), moves the files one by one then removes the emptied folders."""
    try:
        os.rename(src, dst)
        return
    except OSError:
        pass
    for root, folders, files in os.walk(src, topdown=False):
        target = os.path.join(dst, os.path.relpath(root, src))
        os.makedirs(target, exist_ok=True)
        for f in files:
            os.rename(os.path.join(root, f), os.path.join(target, f))
        os.rmdir(root)


def extract_character(character, output, path=None):
    """Cuts a character and writes its parts: <sortie>/<character>/<id>/<id>.pmx + fiche.json,
    shared textures in <sortie>/<character>/tex/, full cutting in cutout.json."""
    ecr = _write_pmx_module()
    T, parts, info, ctrl = analyze(character, path)
    M = T.M
    base = os.path.join(output, character)
    if os.path.exists(base):
        # previous cutting: moved aside (not erased), so that no stale parts are kept
        corb = os.path.join(output, ".corbeille")   # same disk as the output: a plain rename
        os.makedirs(corb, exist_ok=True)
        put_of_side(base, os.path.join(corb, "%s-%d" % (character, time.time() * 1000)))
    os.makedirs(os.path.join(base, "tex"), exist_ok=True)
    copied = {}
    cards = []
    for pc in parts:
        d = os.path.join(base, pc["id"])
        os.makedirs(d, exist_ok=True)
        faces = np.concatenate([pc["faces"], pc["layer_faces"]]).astype(int)
        pmx, st = part_pmx(T, faces, "%s %s" % (character, pc["id"]), "../tex", copied)
        tmp = os.path.join(d, pc["id"] + ".pmx.tmp")
        ecr.write(pmx, tmp)
        os.replace(tmp, os.path.join(d, pc["id"] + ".pmx"))
        hidden = hidden_skin(T, pc["faces"], info["skin_faces"])
        fi = {"character": character, "id": pc["id"], "class": pc["class"],
              "type": pc.get("type"), "sort": pc.get("sort"),
              "why": pc.get("reason") or pc.get("exception"), "sort_reason": pc.get("kind_reason"),
              "layer": pc["layer"], "covers": pc.get("covers"), "carrier": pc.get("carrier"),
              "pmx": pc["id"] + ".pmx", "thumbnail": "thumbnail.png",
              "faces": int(len(pc["faces"])), "layer_faces": int(len(pc["layer_faces"])),
              "vertices": st["vertices"], "materials": pc["materials"],
              "attach_bones": [[n, w] for n, w in pc["os_body"][:6]],
              "own_bones": [n for n, _ in pc["own_bones"]],
              "physics": {"bones": st["bones"], "own_bones": len(pc["own_bones"]), "rigid_bodies": st["rigid_bodies"],
                           "physics_rigid_bodies": st["os_physiques"], "joints": st["joints"]},
              "zones": pc["zones"],
              "hidden_skin": {"faces": int(len(hidden)), "zones": zones_skin(T, hidden)},
              "sewn_to_skin_mm": pc["sewn_skin_mm"],
              "source": {"pmx": M.path, "faces": "../cutout.json"}}
        _json(fi, os.path.join(d, "record.json"))
        cards.append(fi)
    for name, src in copied.items():
        _link(src, os.path.join(base, "tex", name))
    offset = {"character": character, "pmx": M.path, "skin_color_lab": info["color_skin"],
           "check": ctrl, "texture_layers_without_twin": info.get("texture_layers_without_twin"),
           "skin": {"faces": info["skin_faces"].tolist(), "texture_layers": info["skin_texture_layers"].tolist()},
           "parts": {pc["id"]: {"faces": pc["faces"].tolist(), "layer_faces": pc["layer_faces"].tolist(),
                                 "atoms": pc["atoms"]} for pc in parts}}
    _json(offset, os.path.join(base, "cutout.json"))
    trim_character(character, output, T)
    return cards, ctrl


# trims: accessories that only exist on their garment (buttons, pockets, straps, ornaments).
# They cannot be selected on their own; without them, the garment is worn bare. The other accessories
# (belt, tie, bag…) are selected separately.
TRIMS = {"strap", "fastener", "ornament", "pocket", "crest", "badge", "chain", "rope",
              "closure", "loop"}


def chunks(T, faces):
    """Connected components (welded vertices) of a set of faces -> list of face arrays."""
    F = T.M.Fs[faces]
    u, inv = np.unique(F, return_inverse=True)
    inv = inv.reshape(F.shape)
    per = np.arange(len(u))

    def r(x):
        while per[x] != x:
            per[x] = per[per[x]]
            x = per[x]
        return x
    for a_, b_, c_ in inv:
        ra, rb, rc = r(a_), r(b_), r(c_)
        per[rb] = ra
        per[r(rc)] = ra
    rt = np.array([r(x) for x in inv[:, 0]])
    return [np.asarray(faces)[rt == k] for k in np.unique(rt)]


def distribute_trims(T, offset, cards):
    """A trim is often a cluster of scattered pieces grouped by their material (ornaments on
    both cuffs, the back and the waist): the cluster has a single carrier, but each piece belongs to
    what it touches. Returns {carrier id: [(trim id, faces of the piece)]}: each
    piece goes to the selectable part (garment, belt, bag…) its surface touches most
    closely; if it touches none (lying on another trim), it follows the cluster's carrier."""
    targets = [fi for fi in cards.values() if fi["class"] == "garment" or fi.get("sort") not in TRIMS]
    groups = [np.asarray(offset["parts"][fi["id"]]["faces"], int) for fi in targets]
    out = {}
    for fi in cards.values():
        if fi["class"] != "accessory" or fi.get("sort") not in TRIMS:
            continue
        po = cards.get(fi.get("carrier"))
        default = po["id"] if po and (po["class"] == "garment" or po.get("sort") not in TRIMS) else None
        f = np.concatenate([np.asarray(offset["parts"][fi["id"]][k], int) for k in ("faces", "layer_faces")])
        ms = chunks(T, f)
        for m in ms:
            target = default
            if len(ms) > 1 and groups:
                nb = plus_near_surface(T, m, groups, over=True)
                j = int(nb.argmax())
                if nb[j] >= max(2, WORN_PART * nb.sum()):
                    target = targets[j]["id"]
            if target is not None:
                out.setdefault(target, []).append((fi["id"], m))
    return out


def trim_character(character, output, T=None):
    """For each already extracted garment that carries trims: writes <id>+trims.pmx next to
    its PMX (the garment and its trims as a single object) and records it in its fiche
    ("trimmed": pmx, parts, faces, own bones). Reads cutout.json, cuts nothing again."""
    base = os.path.join(output, character)
    with open(os.path.join(base, "cutout.json"), encoding="utf-8") as fh:
        offset = json.load(fh)
    cards = {}
    for pid in offset["parts"]:
        f = os.path.join(base, pid, "record.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                cards[pid] = json.load(fh)
    trimmed = any(fi["class"] == "accessory" and fi.get("sort") in TRIMS and fi.get("carrier") for fi in cards.values())
    if trimmed:
        T = T or Outfit(character)
    reaches = distribute_trims(T, offset, cards) if trimmed else {}
    ecr = _write_pmx_module() if reaches else None
    copied = {}
    n = 0
    for pid, fi in cards.items():
        d = os.path.join(base, pid)
        g = sorted(reaches.get(pid, []), key=lambda x: x[0])
        if not g or (fi["class"] == "accessory" and fi.get("sort") in TRIMS):
            if fi.pop("trimmed", None) is not None:
                _json(fi, os.path.join(d, "record.json"))
            continue
        own = np.concatenate([np.asarray(offset["parts"][pid][k], int) for k in ("faces", "layer_faces")])
        faces = np.unique(np.concatenate([own] + [m for _, m in g]))
        pmx, st = part_pmx(T, faces, "%s %s garni" % (character, pid), "../tex", copied)
        name = pid + "+trims.pmx"
        ecr.write(pmx, os.path.join(d, name + ".tmp"))
        os.replace(os.path.join(d, name + ".tmp"), os.path.join(d, name))
        ids = sorted({x for x, _ in g})
        fi["trimmed"] = {"pmx": name, "parts": ids, "faces": int(len(faces)),
                        "own_bones": sorted(set(fi.get("own_bones") or []).union(*[cards[x].get("own_bones") or [] for x in ids]))}
        _json(fi, os.path.join(d, "record.json"))
        n += 1
    for name, src in copied.items():
        _link(src, os.path.join(base, "tex", name))
    return n


def characters_library():
    return sorted(d for d in os.listdir(MD.LIBRARY)
                  if os.path.exists(os.path.join(MD.LIBRARY, d, "record.json")))


def index(output):
    """index.json per garment type and per accessory kind, plus a summary."""
    all_ = []
    for p in sorted(os.listdir(output)):
        d = os.path.join(output, p)
        if not os.path.exists(os.path.join(d, "cutout.json")):
            continue
        for pc in sorted(os.listdir(d)):
            f = os.path.join(d, pc, "record.json")
            if os.path.exists(f):
                with open(f, encoding="utf-8") as fh:
                    all_.append(json.load(fh))
    os.makedirs(os.path.join(output, "index", "clothes"), exist_ok=True)
    os.makedirs(os.path.join(output, "index", "accessories"), exist_ok=True)
    per = {}
    for fi in all_:
        the_key = ("clothes", fi["type"]) if fi["class"] == "garment" else ("accessories", fi["sort"])
        per.setdefault(the_key, []).append({"character": fi["character"], "id": fi["id"],
                                        "pmx": "../../%s/%s/%s" % (fi["character"], fi["id"], fi["pmx"]),
                                        "thumbnail": "../../%s/%s/thumbnail.png" % (fi["character"], fi["id"]),
                                        "faces": fi["faces"], "layer": fi["layer"],
                                        "carrier": fi.get("carrier"),
                                        "physics": fi["physics"]["physics_rigid_bodies"] > 0})
    for (c, t), l in per.items():
        _json(l, os.path.join(output, "index", {"clothes": "clothes", "accessories": "accessories"}[c], "%s.json" % t))
    assessment = {"characters": len({fi["character"] for fi in all_}), "parts": len(all_),
             "clothes": sum(1 for fi in all_ if fi["class"] == "garment"),
             "accessories": sum(1 for fi in all_ if fi["class"] == "accessory"),
             "clothes_by_type": {t: len(l) for (c, t), l in sorted(per.items()) if c == "clothes"},
             "accessories_by_sort": {t: len(l) for (c, t), l in sorted(per.items()) if c == "accessories"}}
    _json(assessment, os.path.join(output, "index.json"))
    return assessment


# ====================================================================== renders (Blender)

def _bpy_scene():
    import bpy
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "FLAT"
    sh.color_type = "TEXTURE"
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    w = bpy.data.worlds.new("background") if not bpy.data.worlds else bpy.data.worlds[0]
    sc.world = w
    w.color = (0.93, 0.93, 0.93)
    sh.background_type = "WORLD"
    sc.display.render_aa = "8"
    return sc


def _render(lo, hi, angle, path, tx, ty, text=None):
    """Orthographic camera framed on [lo, hi] (tx × ty tile), rotated by `angle` degrees around Z."""
    import math
    import bpy
    from mathutils import Vector
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = tx, ty
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = "ORTHO"
    c = (lo + hi) / 2
    h = max(hi[2] - lo[2], 1e-3)
    l = max(hi[0] - lo[0], hi[1] - lo[1], 1e-3)
    margin = (1.12 + 0.12 * text.count("\n")) if text else 1.06
    if ty >= tx:
        cam_d.sensor_fit = "VERTICAL"
        cam_d.ortho_scale = margin * max(h, l * ty / tx)
        top = cam_d.ortho_scale
    else:
        cam_d.sensor_fit = "HORIZONTAL"
        cam_d.ortho_scale = margin * max(l, h * tx / ty)
        top = cam_d.ortho_scale * ty / tx
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    a = math.radians(angle)
    d = 5.0
    cz = c[2] - ((0.05 + 0.06 * text.count("\n")) * top if text else 0)
    cam.location = Vector((c[0] + d * math.sin(a), c[1] - d * math.cos(a), cz))
    cam.rotation_euler = (math.pi / 2, 0, a)
    cam_d.clip_end = 20
    sc.camera = cam
    to = None
    if text:
        cu = bpy.data.curves.new("txt", "FONT")
        cu.body = text
        cu.size = (0.055 if ty > 300 else 0.085) * top
        cu.align_x = "CENTER"
        to = bpy.data.objects.new("txt", cu)
        m = bpy.data.materials.new("txt")
        m.diffuse_color = (0.05, 0.05, 0.05, 1)
        cu.materials.append(m)
        sc.collection.objects.link(to)
        av = 1.0
        nl = text.count("\n")
        to.location = Vector((c[0] + av * math.sin(a), c[1] - av * math.cos(a),
                              cz - 0.47 * top + (nl + 0.2) * cu.size * 1.15))
        to.rotation_euler = (math.pi / 2, 0, a)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)
    if to is not None:
        bpy.data.objects.remove(to, do_unlink=True)


def _glue(lines, path, background=(1, 1, 1, 1)):
    """Assembles rows of PNGs (any size) into one contact sheet."""
    import bpy
    ims = []
    for line in lines:
        r = []
        for f in line:
            im = bpy.data.images.load(f)
            px = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)
            r.append(px)
            bpy.data.images.remove(im)
        ims.append(r)
    W = max(sum(p.shape[1] for p in r) for r in ims)
    H = sum(max(p.shape[0] for p in r) for r in ims) + 2 * (len(ims) - 1)
    out = np.ones((H, W, 4), np.float32) * np.array(background, np.float32)
    y = H
    for r in ims:
        hr = max(p.shape[0] for p in r)
        x = 0
        for p in r:
            out[y - p.shape[0]:y, x:x + p.shape[1]] = p
            x += p.shape[1]
        y -= hr
        if y > 0:
            out[y - 2:y] = (0.3, 0.3, 0.3, 1)
            y -= 2
    img = bpy.data.images.new("sheet", W, H, alpha=True)
    img.pixels.foreach_set(out.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def sheet_character(output, character, tmp):
    """Thumbnail of each part (<part>/vignette.png, from the front) and the character's contact sheet
    (<character>/sheet.png): whole outfit, garments only, accessories only, each garment,
    from the front then from the back; then each accessory from the front."""
    import bpy
    import contact_sheet as PL
    PL.empty()
    _bpy_scene()
    base = os.path.join(output, character)
    cards = []
    for pc in sorted(os.listdir(base)):
        f = os.path.join(base, pc, "record.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                cards.append(json.load(fh))
    cards.sort(key=lambda fi: (fi["class"] != "garment", -fi["faces"]))
    cache, objs, boxes = {}, {}, {}
    for fi in cards:
        M = MD.Model(fi["id"], path=os.path.join(base, fi["id"], fi["pmx"]))
        o = PL.obj(M, np.arange(len(M.F)), fi["id"], cache)
        objs[fi["id"]] = o
        boxes[fi["id"]] = (M.pos.min(0), M.pos.max(0))

    def show(ids):
        for k, o in objs.items():
            o.hide_render = k not in ids
        lo = np.min([boxes[k][0] for k in ids], 0)
        hi = np.max([boxes[k][1] for k in ids], 0)
        return lo, hi
    garment = [fi["id"] for fi in cards if fi["class"] == "garment"]
    accessory = [fi["id"] for fi in cards if fi["class"] == "accessory"]
    TXg, TYg = 220, 380
    lines = []
    groups = [("outfit", garment + accessory), ("clothes", garment), ("accessories", accessory)] + [(v, [v]) for v in garment]
    for ang, view in ((0, "face"), (180, "back")):
        line = []
        for name, ids in groups:
            if not ids:
                continue
            fi = next((x for x in cards if x["id"] == name), None)
            lab = "%s\n%s c%d" % (name, fi["type"], fi["layer"]) if fi else "%s (%s)" % (name, view)
            f = os.path.join(tmp, "pl_%s_%s_%s.png" % (character, name, view))
            lo, hi = show(ids)
            _render(lo, hi, ang, f, TXg, TYg, lab)
            line.append(f)
        lines.append(line)
    # thumbnails (all parts, from the front, no text); accessory tiles with text
    line = []
    for fi in cards:
        lo, hi = show([fi["id"]])
        _render(lo, hi, 0, os.path.join(base, fi["id"], "thumbnail.png"), 256, 256)
        if fi["class"] == "accessory":
            f = os.path.join(tmp, "pl_%s_%s_acc.png" % (character, fi["id"]))
            _render(lo, hi, 0, f, 165, 165, "%s\n→ %s" % (fi["id"], fi.get("carrier") or "body"))
            line.append(f)
            if len(line) == 8:
                lines.append(line)
                line = []
    if line:
        lines.append(line)
    _glue(lines, os.path.join(base, "sheet.png"))
    log("contact sheet", character, len(cards), "parts")


# ====================================================================== command line

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="library.py")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("extract", help="cut and write the parts (plain Python)")
    a.add_argument("--output", dest="output", required=True)
    a.add_argument("--only", dest="alone", help="comma-separated characters (default: the whole face library)")
    a = sp.add_parser("trim", help="garments with their trims, from the cutting already written (plain Python)")
    a.add_argument("--output", dest="output", required=True)
    a.add_argument("--only", dest="alone")
    a = sp.add_parser("sheets", help="thumbnails and contact sheets (in Blender)")
    a.add_argument("--output", dest="output", required=True)
    a.add_argument("--only", dest="alone")
    a = sp.add_parser("index", help="index by type and kind")
    a.add_argument("--output", dest="output", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "extract":
        characters = args.alone.split(",") if args.alone else characters_library()
        assessment = {}
        for p in characters:
            t = time.time()
            try:
                fi, c = extract_character(p, args.output)
            except Exception as e:  # a failing character does not stop the others; it is named
                import traceback
                traceback.print_exc()
                assessment[p] = {"error": repr(e)}
                continue
            assessment[p] = {"parts": len(fi), "clothes": sum(1 for x in fi if x["class"] == "garment"),
                      "check": {k: v for k, v in c.items()}}
            log(p, len(fi), "parts", "check", "passed" if c["passed"] else "to review", "%.0fs" % (time.time() - t))
        prev = {}
        f = os.path.join(args.output, "check.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                prev = json.load(fh)
        prev.update(assessment)
        _json(prev, f)
        print(json.dumps(index(args.output), ensure_ascii=False, indent=1))
    elif args.cmd == "trim":
        characters = args.alone.split(",") if args.alone else sorted(
            d for d in os.listdir(args.output) if os.path.exists(os.path.join(args.output, d, "cutout.json")))
        for p in characters:
            log(p, trim_character(p, args.output), "garments with trims")
    elif args.cmd == "sheets":
        tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "rendus-biblio")
        os.makedirs(tmp, exist_ok=True)
        characters = args.alone.split(",") if args.alone else sorted(
            d for d in os.listdir(args.output) if os.path.exists(os.path.join(args.output, d, "cutout.json")))
        for p in characters:
            sheet_character(args.output, p, tmp)
    elif args.cmd == "index":
        print(json.dumps(index(args.output), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])

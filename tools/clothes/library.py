"""ZZZ clothing library: each outfit cut into separate garments and accessories,
each part written as a standalone PMX that can be fitted on another body (wear.py).

    python3.13 -I library.py extract --output DOSSIER [--only Anby,Belle]
        cuts every outfit of the face library, writes <Character>/<part>/{<part>.pmx,
        fiche.json}, <Character>/decoupe.json (check), controle.json and the index.
    bin/blender-queue -b --factory-startup --python library.py -- planches --output DOSSIER [--only …]
        thumbnail of each part and contact sheet of each character (exploded, front and back).
    python3.13 -I library.py index --output DOSSIER
        index/vetements/<type>.json, index/accessoires/<sorte>.json, index.json.

Source files are only ever read.
"""
import json
import os
import re
import sys
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import model as MD  # noqa: E402
from model import P, PMX  # noqa: E402

try:
    import OpenImageIO as OIIO
except ImportError:  # inside Blender: bpy reads the images
    OIIO = None


def journal(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ====================================================================== textures

_CACHE_TEX = {}


def lire_texture(chemin):
    """Float RGB image (h, w, 3), rows from top to bottom (PMX v points down), or None."""
    if chemin in _CACHE_TEX:
        return _CACHE_TEX[chemin]
    im = None
    if os.path.exists(chemin):
        if OIIO is not None:
            b = OIIO.ImageInput.open(chemin)
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
            i = bpy.data.images.load(chemin)
            w, h = i.size
            px = np.empty(w * h * 4, np.float32)
            i.pixels.foreach_get(px)
            bpy.data.images.remove(i)
            im = px.reshape(h, w, 4)[::-1, :, :3].copy()
    if len(_CACHE_TEX) > 6:
        _CACHE_TEX.pop(next(iter(_CACHE_TEX)))
    _CACHE_TEX[chemin] = im
    return im


def chemin_texture(M, mi):
    t = M.mats[mi]["texture"]
    if not 0 <= t < len(M.textures):
        return None
    return os.path.normpath(os.path.join(os.path.dirname(M.chemin), M.textures[t].replace("\\", "/")))


def lab(rgb):
    """sRGB (0-1) -> CIELab (D65)."""
    c = np.clip(np.asarray(rgb, np.float64), 0, 1)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    X = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    X = X / np.array([0.95047, 1.0, 1.08883])
    f = np.where(X > 0.008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def echantillonner(M, faces, uvs):
    """Lab colour of each face's material texture at the given UVs (n, 2); NaN when there is no
    texture."""
    out = np.full((len(faces), 3), np.nan)
    mf = M.mf[faces]
    for mi in np.unique(mf):
        ch = chemin_texture(M, mi)
        im = lire_texture(ch) if ch else None
        k = np.where(mf == mi)[0]
        if im is None:
            d = M.mats[mi]["diffus"]
            out[k] = lab(np.array(d[:3]))[None]
            continue
        h, w = im.shape[:2]
        u = np.clip((uvs[k, 0] % 1) * w, 0, w - 1).astype(int)
        v = np.clip((uvs[k, 1] % 1) * h, 0, h - 1).astype(int)
        out[k] = lab(im[v, u])
    return out


# ====================================================================== atoms of an outfit

REGIONS = ("tete", "cou", "tronc_haut", "poitrine", "ventre", "bassin", "fesses", "bras", "avant_bras",
           "main", "doigt", "cuisse", "jambe", "pied")
RI = {r: i for i, r in enumerate(REGIONS)}
# body zones (groups of regions): used to type a garment
ZONES = {"tete": ("tete",), "cou": ("cou",), "torse": ("tronc_haut", "poitrine", "ventre"),
         "bras": ("bras",), "avant_bras": ("avant_bras",), "main": ("main", "doigt"),
         "bassin": ("bassin", "fesses"), "cuisse": ("cuisse",), "jambe": ("jambe",), "pied": ("pied",)}
ZN = tuple(ZONES)


def aretes_soudees(Fs):
    """Welded edges: (E (m, 2), faces of each edge (per-edge list via sorting))."""
    nf = len(Fs)
    E = np.sort(np.concatenate([Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]]), 1)
    fid = np.tile(np.arange(nf), 3)
    coin = np.repeat(np.arange(3), nf)
    u, inv = np.unique(E, axis=0, return_inverse=True)
    return u, inv.ravel(), fid, coin


PEAU_EN_PLUS = re.compile(r"^首$")


class Tenue:
    """The outfit of an official PMX (everything except head and hair), cut into atoms: islands of
    the raw mesh (UV seams and split normals included) within a single material."""

    def __init__(self, perso, M=None, chemin=None):
        self.perso = perso
        self.M = M = M or MD.Modele(perso, chemin=chemin)
        self.roles = np.array([M.role_materiau(i) for i in range(len(M.mats))])
        # skin named in Japanese, missing from modele.PEAU: 首 (Yanagi's neck and ears)
        self.roles[[i for i, n in enumerate(M.noms_mat) if PEAU_EN_PLUS.match(n)]] = "peau"
        rf = self.roles[M.mf]
        self.base = np.isin(rf, ["tenue", "accessoire", "peau"])
        self.calque = rf == "calque"
        fi = np.where(self.base)[0]
        self.fi = fi
        F = M.F[fi]
        # colour of each face (UV centre)
        self.lab_face = np.full((len(M.F), 3), np.nan)
        self.lab_face[fi] = echantillonner(M, fi, M.uv[F].mean(1))
        nv = len(M.pos)

        def ilots(drap):
            """Islands (faces of fi) of the raw mesh, split by material and by flag."""
            Fd = F + nv * drap[:, None]
            E = np.concatenate([Fd[:, [0, 1]], Fd[:, [1, 2]]])
            lab = MD.composantes(2 * nv, E)
            return np.unique(lab[Fd[:, 0]].astype(np.int64) * 4096 + M.mf[fi], return_inverse=True)[1]

        # skin painted in the same island as a garment (Qingyi's shorts and thighs): in a
        # generic material, skin-coloured faces forming a region of at least 40 faces and
        # 15 % of their island become a separate island (skin, for the cutting)
        drap = np.zeros(len(fi), np.int64)
        self.peau_lab = couleur_peau_roles(self)
        gen_f = np.array([mat_generique(n) for n in M.noms_mat])[M.mf[fi]] & (self.roles[M.mf[fi]] == "tenue")
        if self.peau_lab is not None and gen_f.any():
            de = np.linalg.norm(self.lab_face[fi] - self.peau_lab, axis=1)
            cand = gen_f & (de < DE_PEAU)
            if cand.any():
                at0 = ilots(np.zeros(len(fi), np.int64))
                drap = cand.astype(np.int64)
                at1 = ilots(drap)
                n0 = np.bincount(at0)
                n1 = np.bincount(at1)
                ok = cand & (n1[at1] >= 40) & (n1[at1] >= 0.15 * n0[at0]) & (n1[at1] < 0.97 * n0[at0])
                # skin-coloured bits welded to the skin, whatever their size: shoulder edges
                # of Qingyi's top and thigh tops of her leggings, Nicole's ears and neck
                # (in pieces of fewer than 40 faces along the fabric edge); clear-cut colour
                # only (median ΔE < DE_PEAU / 2): the cream fabrics of Aria, Cissia or
                # YeShunguang come close to the threshold (ΔE 8 to 10) and are not skin
                vp = np.zeros(len(M.pos_s), bool)
                vp[M.Fs[self.roles[M.mf] == "peau"].ravel()] = True
                vp[M.Fs[fi[ok]].ravel()] = True
                colle = cand & ~ok & vp[M.Fs[fi]].any(1) & (n1[at1] < 0.97 * n0[at0])
                for k in np.unique(at1[colle]):
                    m = at1 == k
                    if np.median(de[m]) < DE_PEAU / 2:
                        ok |= m
                drap = ok.astype(np.int64)
        self.peau_par_faces = int(drap.sum())
        at = ilots(drap)
        u = np.unique(at)
        self.atome = np.full(len(M.F), -1)
        self.atome[fi] = at
        self.na = na = len(u)
        self.faces_atome = [fi[at == k] for k in range(na)] if na < 4000 else None
        o = np.argsort(at, kind="stable")
        _, deb = np.unique(at[o], return_index=True)
        self.faces_atome = np.split(fi[o], deb[1:])
        self.mat_atome = np.array([M.mf[g[0]] for g in self.faces_atome])
        self.role_atome = self.roles[self.mat_atome]
        self.nf_atome = np.array([len(g) for g in self.faces_atome])
        self.aire_atome = np.array([M.aire[g].sum() for g in self.faces_atome])
        w = M.aire[fi]
        self.lab_atome = np.zeros((na, 3))
        for c in range(3):
            self.lab_atome[:, c] = np.bincount(at, weights=np.nan_to_num(self.lab_face[fi, c]) * w, minlength=na) / \
                np.maximum(np.bincount(at, weights=w, minlength=na), 1e-20)
        self._regions()
        self._coutures()
        self.seg = segments_corps(M)
        self.axe_s = os_proche(M.pos_s, self.seg)

    # ------------------------------------------------------------------ body regions
    def _regions(self):
        M = self.M
        nb = len(M.noms_os)
        reg_os = np.array([RI.get(M.region_ancetre(b), -1) for b in range(nb)])
        self.reg_os = reg_os
        # own bones: neither from the standard body (root family / body regions) nor from the head
        corps = np.array([M.famille_os[b] not in ("vetement", "accessoire", "cheveux", None) or
                          M.noms_os[b] in P.STANDARD_MMD or M.noms_os[b] in P.SEMI_STANDARD_MMD
                          for b in range(nb)])
        self.os_corps = corps
        self.os_acc = np.array([M.os_accessoire(b) for b in range(nb)])
        nv = len(M.pos)
        R = np.zeros((nv, len(REGIONS)))
        propre = np.zeros(nv)
        acc = np.zeros(nv)
        for k in range(M.idx.shape[1]):
            b = M.idx[:, k]
            w = M.poids[:, k]
            ok = (b >= 0) & (w > 0)
            r = reg_os[np.where(ok, b, 0)]
            m = ok & (r >= 0)
            np.add.at(R, (np.where(m)[0], r[m]), w[m])
            propre[ok] += w[ok] * ~corps[b[ok]]
            acc[ok] += w[ok] * self.os_acc[b[ok]]
        s = np.maximum(R.sum(1), 1e-9)
        self.R_v = R / s[:, None]
        self.propre_v = propre
        self.acc_v = acc
        na = self.na
        self.R_atome = np.zeros((na, len(REGIONS)))
        self.propre_atome = np.zeros(na)
        self.acc_atome = np.zeros(na)
        for k, g in enumerate(self.faces_atome):
            v = M.F[g].ravel()
            wa = np.repeat(M.aire[g], 3)
            self.R_atome[k] = (self.R_v[v] * wa[:, None]).sum(0) / max(wa.sum(), 1e-20)
            self.propre_atome[k] = (propre[v] * wa).sum() / max(wa.sum(), 1e-20)
            self.acc_atome[k] = (acc[v] * wa).sum() / max(wa.sum(), 1e-20)

    def zones(self, R):
        """Region vector -> share per zone (dict)."""
        return {z: float(sum(R[..., RI[r]] for r in rs)) for z, rs in ZONES.items()}

    # ------------------------------------------------------------------ welded seams between atoms
    def _coutures(self):
        """For each pair of sewn atoms (shared welded edge): number of edges, length,
        median colour difference across the seam (sampled near the edge)."""
        M = self.M
        fi = self.fi
        Fs = M.Fs[fi]
        u, inv, fid, coin = aretes_soudees(Fs)
        o = np.argsort(inv, kind="stable")
        inv_o, fid_o, coin_o = inv[o], fid[o], coin[o]
        # edges with exactly two faces (manifold) between two different atoms
        _, deb, cnt = np.unique(inv_o, return_index=True, return_counts=True)
        m2 = cnt == 2
        i1, i2 = deb[m2], deb[m2] + 1
        f1, f2 = fi[fid_o[i1]], fi[fid_o[i2]]
        a1, a2 = self.atome[f1], self.atome[f2]
        diff = a1 != a2
        f1, f2, c1, c2 = f1[diff], f2[diff], coin_o[i1][diff], coin_o[i2][diff]
        a1, a2 = a1[diff], a2[diff]
        e = u[inv_o[i1][diff]]
        L = np.linalg.norm(M.pos_s[e[:, 0]] - M.pos_s[e[:, 1]], axis=1)

        def pres(f, c):
            # UV of a point near the middle of the edge (corner c, c+1), a quarter of the way to the 3rd vertex
            tri = M.uv[M.F[f]]
            a = tri[np.arange(len(f)), c]
            b = tri[np.arange(len(f)), (c + 1) % 3]
            t = tri[np.arange(len(f)), (c + 2) % 3]
            return 0.75 * (a + b) / 2 + 0.25 * t
        l1 = echantillonner(M, f1, pres(f1, c1))
        l2 = echantillonner(M, f2, pres(f2, c2))
        dE = np.linalg.norm(l1 - l2, axis=1)
        lo, hi = np.minimum(a1, a2), np.maximum(a1, a2)
        cle = lo.astype(np.int64) * 100000 + hi
        uc, ic = np.unique(cle, return_inverse=True)
        self.coutures = {}
        for k, c in enumerate(uc):
            m = ic == k
            self.coutures[(int(c // 100000), int(c % 100000))] = {
                "aretes": int(m.sum()), "longueur": float(L[m].sum()),
                "dE": float(np.median(np.nan_to_num(dE[m], nan=50.0)))}
        self.bord_cousu = np.zeros(self.na)
        for (a, b), d in self.coutures.items():
            self.bord_cousu[a] += d["longueur"]
            self.bord_cousu[b] += d["longueur"]


# ====================================================================== grouping

GENERIQUE = re.compile(r"^(体|衣|服|足|新規|毛)\d*([-_].*)?$")


def mat_generique(nom):
    """"Texture sheet" material (体, 体2, 衣1…) that can carry several garments, as opposed
    to a material named after a garment (裤, 袜, 外套…)."""
    return bool(GENERIQUE.match(nom))


def couleur_peau_roles(T):
    """Median skin Lab: faces of the skin materials, otherwise of the face (颜)."""
    M = T.M
    f = np.where(T.roles[M.mf] == "peau")[0]
    if len(f) < 50:
        f = np.where(np.array([n.startswith("颜") for n in M.noms_mat])[M.mf])[0]
        if not len(f):
            return None
        return np.nanmedian(echantillonner(M, f, M.uv[M.F[f]].mean(1)), 0)
    return np.nanmedian(T.lab_face[f], 0)


def couleur_peau(T):
    return T.peau_lab


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



    def groupes(self, ids):
        out = {}
        for i in ids:
            out.setdefault(self.t(i), []).append(i)
        return list(out.values())


# ====================================================================== geometry of a set of faces

SEG_EXCLUS = ("doigt", "tete")


# IK handles (足ＩＫ, 足IK親, つま先ＩＫ): placed at the foot, but filed as "cuisse" (thigh) by their name
# (足…). A ring that used them as its axis was called a garter although it sits at the ankle
# (Anby). Their region is read from where they are: that of the body bone whose head is closest.
OS_IK = re.compile(r"IK|ＩＫ")


def region_axe(T, b):
    M = T.M
    if not OS_IK.search(M.noms_os[b]):
        return M.region_ancetre(b)
    autres = [i for i in T.seg[0] if not OS_IK.search(M.noms_os[i])]
    j = autres[int(np.argmin(np.linalg.norm(M.tetes[autres] - M.tetes[b], axis=1)))]
    return M.region_ancetre(j)


def segments_corps(M):
    """Body bones (head-tail segments) used as axes: (indices, heads, tails)."""
    corps = ("tronc_haut", "poitrine", "ventre", "bassin", "fesses", "cou", "bras", "avant_bras",
             "main", "cuisse", "jambe", "pied")
    b = [i for i in range(len(M.noms_os)) if M.region_os[i] in corps
         and np.linalg.norm(M.queues[i] - M.tetes[i]) > 0.01]
    return np.array(b), M.tetes[b], M.queues[b]


def os_proche(Pv, seg):
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
        self.aire = float(M.aire[faces].sum())
        self.lo, self.hi = p.min(0), p.max(0)
        self.diag = float(np.linalg.norm(self.hi - self.lo))
        self.centre = (p * 1.0).mean(0)
        E, c, _ = MD.aretes_faces(Fs)
        self.bord = int((c == 1).sum())
        self.fermee = self.bord == 0
        # ring: around its dominant bone, short extent along the bone, almost a full turn
        k, t, ang, r = T.axe_s[0][vs], T.axe_s[1][vs], T.axe_s[2][vs], T.axe_s[3][vs]
        kd = np.bincount(k).argmax()
        m = k == kd
        self.os_axe = int(T.seg[0][kd])
        A, C = T.seg[1][kd], T.seg[2][kd]
        ax = (C - A) / np.linalg.norm(C - A)
        h = (p - A) @ ax        # all vertices, not only the bone's: a glove extends past the wrist
        self.etendue_axe = float(np.percentile(h, 97) - np.percentile(h, 3))
        self.tour = float(len(np.unique(((ang[m] + np.pi) / (2 * np.pi) * 24).astype(int) % 24)) / 24)
        self.part_axe = float(m.mean())
        self.rayon = float(np.median(r))
        # width of a band: area / length (diagonal)
        self.largeur = self.aire / max(self.diag, 1e-6)


# ====================================================================== cutting

SLOTS = {"haut": ("torse", "bras", "avant_bras"), "bas": ("bassin", "cuisse"), "jambe": ("jambe",),
         "pied": ("pied",), "main": ("main",), "cou": ("cou",), "tete": ("tete",)}
SLOTS_VOISINS = {frozenset(x) for x in (("haut", "bas"), ("bas", "jambe"), ("jambe", "pied"), ("haut", "cou"),
                                        ("haut", "main"))}
ACC_MAT_EN_PLUS = re.compile(r"配件|腿环|饰")   # in addition to modele.ACCESSOIRE_MAT: "attached parts", thigh ring
DE_MEME_TISSU = 15.0      # colour difference (Lab) below which two sewn panels are the same fabric
DE_PEAU_CHUTE = 12.0      # skin offcut left as an accessory (skin-coloured area share >= 0.5)
DE_PEAU = 10.0            # difference from the skin colour below which a generic-material panel is skin
ACC_DIAG = 0.06           # m: smaller part = accessory (button, rivet, buckle)
ANNEAU_HAUT = 0.07        # m: ring narrower than this along its bone = belt, bracelet, necklace, garter
ANNEAU_TOUR = 0.75        # share of the full turn covered
SANGLE_LARGEUR = 0.03     # m: narrower band = strap
POSEE_AIRE = 0.03         # share of the outfit's area below which a shell lying on a garment is an accessory
POSEE_DIAG = 0.25         # m: and diagonal (pocket, patch, pouch)
BORD_CONTRE_MIN = 0.5     # share of the smaller panel's border touching the other one for them to be sewn back together
PETITE_PART = 0.012       # share of the outfit's faces below which a small shell (diag < 15 cm) is an accessory
CHAINE_PART = 0.5         # share of a shell's weights on bones that only it uses: above this, a separate object

# accessory kinds read from material or bone names (first pattern found)
SORTES_NOM = [(r"武器|刀|Weapon|拳套", "arme"), (r"墨镜|镜片|护目镜|眼镜|镜", "lunettes"), (r"面具|面罩|眼罩", "masque"),
              (r"耳机", "casque_audio"), (r"帽|头饰|頭巾|发饰|髮飾|髮饰|光环|角", "couvre_chef"), (r"翅膀|翼|翅", "ailes"),
              (r"尾", "queue"), (r"^耳|_耳|耳$", "oreilles"), (r"耳饰|耳环", "boucle_oreille"), (r"包|Backpack", "sac"),
              (r"熊|狸猫|邦布|玩偶", "peluche"), (r"电池|工具|手雷|屏幕", "objet"), (r"戒指", "bague"),
              (r"宝石|珠宝|项链|颈饰|吊坠|坠|珠", "bijou"), (r"领带|Tie|Bowtie", "cravate"), (r"拉链", "fermeture"),
              (r"锁链|链", "chaine"), (r"绳", "corde"), (r"徽章|Badge", "badge"), (r"扣", "boucle"), (r"腿环|腿圈|LegRing|Garter", "jarretiere"),
              (r"腰带|皮带|Belt", "ceinture"),
              (r"围巾|领巾|Scarf", "echarpe"), (r"Strap|带", "sangle")]
_SORTES_NOM = [(re.compile(rx), s) for rx, s in SORTES_NOM]


SORTES_OBJET = {"arme", "lunettes", "masque", "casque_audio", "ailes", "queue", "oreilles", "sac", "peluche", "objet"}


def sorte_nom(nom):
    for rx, s in _SORTES_NOM:
        if rx.search(nom):
            return s
    return None


def d_un_cote(T, g):
    """Share of vertices on one side (front or back) of the torso's frontal plane: a top wraps around
    the torso, a backpack or a breastplate stays on one side."""
    M = T.M
    noms = {n: i for i, n in enumerate(M.noms_os)}
    if "上半身2" not in noms:
        return 0.0
    y0 = M.tetes[noms["上半身2"]][1]
    dos = (g.p[:, 1] > y0 + 0.01).mean()
    dev = (g.p[:, 1] < y0 - 0.01).mean()
    return float(max(dos, dev))


def bord_contre(T, fa, fb, cellule=0.008):
    """Share of fa's border vertices that have a vertex of fb within one cell: two panels
    of the same garment touch along a seam, two garments at a few points only."""
    M = T.M
    E, c, _ = MD.aretes_faces(M.Fs[fa])
    vb = np.unique(E[c == 1])
    if len(vb) == 0:
        return 0.0
    kb = set(cles_voxels(M.pos_s[np.unique(M.Fs[fb])], cellule).tolist())
    p = M.pos_s[vb]
    ok = np.zeros(len(vb), bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k = cles_voxels(p + np.array([dx, dy, dz]) * cellule, cellule)
                ok |= np.isin(k, list(kb))
    return float(ok.mean())


def slot_de(T, R):
    z = T.zones(R)
    sc = {s: sum(z[x] for x in zs) for s, zs in SLOTS.items()}
    return max(sc, key=sc.get)


def cles_voxels(p, cellule):
    q = np.floor(p / cellule).astype(np.int64) + 1 << 20
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


PROCHE = 0.02   # m: below this, a surface is "touched" by an accessory
PORTE_PART = 0.1   # share of an accessory's points that must touch a part for that part to carry it


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
    bord = np.minimum(np.minimum(seg(A, AB), seg(A, AC)), seg(B, C - B))
    return np.where((b >= 0) & (c >= 0) & (b + c <= 1), np.abs(h), bord)


def plus_proche_surface(T, faces, groupes, portee=PROCHE, points_max=150, dessus=False):
    """What a small part really touches: for each of its points (vertices and face centres),
    the group of faces whose surface is closest, within `portee`. Returns the
    number of points per group. The cells of `contacts` only see vertices: a button
    in the middle of a large jacket face touches nothing there. `dessus`: a point only counts for a
    group if it lies on the outer side of its surface (a tie touches the shirt and the
    jacket, but it lies on the shirt and goes under the jacket)."""
    M = T.M
    tri = M.pos_s[M.Fs[faces]]
    p = np.concatenate([np.unique(tri.reshape(-1, 3), axis=0), tri.mean(1)])
    if len(p) > points_max:
        p = p[np.linspace(0, len(p) - 1, points_max).astype(int)]
    lo, hi = p.min(0) - portee, p.max(0) + portee
    D = np.full((len(p), len(groupes)), np.inf)
    for k, f in enumerate(groupes):
        if not len(f):
            continue
        t = M.pos_s[M.Fs[f]]
        t = t[((t.max(1) >= lo) & (t.min(1) <= hi)).all(1)]
        for i in range(0, len(t), 4000):
            x = t[i:i + 4000]
            d = _dist_triangles(p, x[:, 0], x[:, 1], x[:, 2])
            if dessus:
                kk = d.argmin(1)
                nn = np.cross(x[kk, 1] - x[kk, 0], x[kk, 2] - x[kk, 0])
                d = np.where(((p - x[kk, 0]) * nn).sum(1) > 0, d.min(1), np.inf)
            else:
                d = d.min(1)
            D[:, k] = np.minimum(D[:, k], d)
    k = D.argmin(1)
    return np.bincount(k[D.min(1) <= portee], minlength=len(groupes))


def contacts(T, groupes, cellule):
    """Pairs of groups (lists of faces) that have vertices in the same cell of
    `cellule` m: {(i, j): number of shared cells}."""
    if not len(groupes):
        return {}       # an outfit without garment shells (accessories only, or nothing): no pair
    M = T.M
    ks, gs = [], []
    for i, f in enumerate(groupes):
        vs = np.unique(M.Fs[f])
        k = np.unique(cles_voxels(M.pos_s[vs], cellule))
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
                cle = (int(min(gg[a], gg[b])), int(max(gg[a], gg[b])))
                out[cle] = out.get(cle, 0) + 1
    return out


def cases_corps(T, faces, pas_t=0.025, n_ang=12):
    """Cells (bone, slice along the bone, angular sector) covered by a set of
    faces, with the median radius in each cell: {cell: radius}."""
    vs = np.unique(T.M.Fs[faces])
    k, t, ang, r = (a[vs] for a in T.axe_s)
    c = (k.astype(np.int64) * 1000 + (t / pas_t).astype(np.int64)) * 100 + \
        ((ang + np.pi) / (2 * np.pi) * n_ang).astype(np.int64) % n_ang
    u, inv = np.unique(c, return_inverse=True)
    rm = np.bincount(inv, weights=r) / np.bincount(inv)
    return dict(zip(u.tolist(), rm.tolist()))


def miroirs(T, ga, gb):
    """Are two sets of faces left-right symmetric (mirrored centres, similar sizes)?"""
    M = T.M
    ca = M.pos[np.unique(M.F[ga])].mean(0)
    cb = M.pos[np.unique(M.F[gb])].mean(0)
    cb = cb * np.array([-1, 1, 1])
    na, nb = len(ga), len(gb)
    return np.linalg.norm(ca - cb) < 0.03 and 0.8 < na / nb < 1.25 and abs(ca[0]) > 0.02


def decouper(T):
    """Cuts an outfit into parts. Returns (pieces, info); each part: dict with "classe"
    (vetement | accessoire), "faces" (indices into M.F, overlay layers included), "atomes", reasons."""
    M = T.M
    noms = M.noms_mat
    na = T.na
    gen = np.array([mat_generique(noms[m]) for m in T.mat_atome])
    peau_lab = couleur_peau(T)
    est_peau = T.role_atome == "peau"
    if peau_lab is not None:
        est_peau = est_peau | (gen & (np.linalg.norm(T.lab_atome - peau_lab, axis=1) < DE_PEAU))
    mat_acc = np.array([T.roles[m] == "accessoire" or bool(ACC_MAT_EN_PLUS.search(noms[m])) for m in range(len(noms))])
    acc_at = ~est_peau & (mat_acc[T.mat_atome] | (T.acc_atome > 0.5))
    slot_at = [slot_de(T, T.R_atome[a]) for a in range(na)]
    raison_at = {a: ("accessory material (%s)" % noms[T.mat_atome[a]] if mat_acc[T.mat_atome[a]]
                     else "accessory bone") for a in np.where(acc_at)[0]}
    vet_at = ~est_peau & ~acc_at
    # --- 1. shells: sewn garment atoms, same material or two generic materials
    uf = UF(na)
    for (a, b), c in T.coutures.items():
        if vet_at[a] and vet_at[b]:
            if T.mat_atome[a] == T.mat_atome[b] or (gen[a] and gen[b]):
                uf.u(a, b)
        elif acc_at[a] and acc_at[b]:
            uf.u(a, b)
    coques = uf.groupes([a for a in range(na) if not est_peau[a]])
    total_f = sum(T.nf_atome[a] for a in range(na) if vet_at[a]) or 1
    aire_tenue = sum(T.aire_atome[a] for a in range(na) if not est_peau[a]) or 1
    faces_de = lambda ats: np.concatenate([T.faces_atome[a] for a in ats])  # noqa: E731
    # --- 2. garment shells that are accessories (shape, size, lying on a garment)
    cq_vet = [c for c in coques if vet_at[c[0]]]
    cq_acc = [c for c in coques if acc_at[c[0]]]
    geo = [Geo(T, faces_de(c)) for c in cq_vet]
    grandes = [i for i, c in enumerate(cq_vet) if geo[i].aire >= POSEE_AIRE * aire_tenue]
    support = contacts(T, [faces_de(c) for c in cq_vet], 0.02)
    # material named after a garment (靴, 手套…) and large enough: its shells belong to it, except small ones
    nf_mat = np.bincount(T.mat_atome[vet_at], weights=T.nf_atome[vet_at], minlength=len(noms))
    nomme_grand = [not gen[c[0]] and nf_mat[T.mat_atome[c[0]]] > 0.03 * total_f for c in cq_vet]
    # bone chain of its own for a shell: clothing bones almost all of whose weight (area-weighted)
    # falls in this single shell. A material named after a garment tells on which texture sheet
    # the shell is painted, not what it is: an isolated, small shell driven by
    # its own chain is a separate object (Hugo's tie, painted in 外套 with the jacket)
    masse = np.zeros((len(cq_vet), len(M.noms_os)))
    for i, c in enumerate(cq_vet):
        f = faces_de(c)
        v = M.F[f].ravel()
        wa = np.repeat(M.aire[f], 3)
        for k in range(M.idx.shape[1]):
            ob, w = M.idx[v, k], M.poids[v, k]
            ok = (ob >= 0) & (w > 0)
            np.add.at(masse[i], ob[ok], w[ok] * wa[ok])
    a_soi = (masse >= 0.95 * np.maximum(masse.sum(0), 1e-20)) & ~T.os_corps
    chaine = (masse * a_soi).sum(1) / np.maximum(masse.sum(1), 1e-20)
    acc_cq = {}
    for i, (c, g) in enumerate(zip(cq_vet, geo)):
        nf = sum(T.nf_atome[a] for a in c)
        posee = any(((min(i, j), max(i, j)) in support) for j in grandes if j != i)
        r = None
        if g.diag < ACC_DIAG:
            r = "small part (%.0f mm)" % (g.diag * 1000)
        elif nomme_grand[i] and chaine[i] >= CHAINE_PART and g.aire < POSEE_AIRE * aire_tenue:
            r = "object driven by its own bone chain (%.0f %% of its weights)" % (100 * chaine[i])
        elif g.etendue_axe < ANNEAU_HAUT and g.tour >= ANNEAU_TOUR and g.part_axe > 0.6 and g.diag < 0.5:
            r = "ring around %s (%.0f mm high)" % (M.noms_os[g.os_axe], g.etendue_axe * 1000)
        elif g.largeur < SANGLE_LARGEUR and g.diag > 0.08 and g.aire < POSEE_AIRE * aire_tenue:
            r = "narrow band (%.0f mm)" % (g.largeur * 1000)
        elif nf < PETITE_PART * total_f and g.diag < 0.15:
            r = "small shell (%d faces)" % nf
        elif not nomme_grand[i] and g.diag < 0.45 and d_un_cote(T, g) >= 0.9 and \
                slot_de(T, (T.R_atome[c] * T.aire_atome[c, None]).sum(0) / T.aire_atome[c].sum()) == "haut":
            r = "volume on one side of the torso only (bag, breastplate)"
        elif posee and g.aire < POSEE_AIRE * aire_tenue and g.diag < POSEE_DIAG and not nomme_grand[i]:
            r = "lying on a garment (%.1f %% of the area)" % (100 * g.aire / aire_tenue)
        if r:
            acc_cq[i] = r
    # --- 3. within each garment shell: sewn panels grouped into garments
    vets = []      # lists of atoms
    for i, c in enumerate(cq_vet):
        if i in acc_cq:
            continue
        if not gen[c[0]]:
            vets.append(list(c))
            continue
        cs = set(c)
        u2 = UF(na)
        for (a, b), d in T.coutures.items():
            if a in cs and b in cs:
                sa, sb = slot_at[a], slot_at[b]
                if sa == sb or (frozenset((sa, sb)) in SLOTS_VOISINS and d["dE"] < DE_MEME_TISSU):
                    u2.u(a, b)
        vets.extend(u2.groupes(c))
    # --- 4. garments from different shells: same named material, or generic ones that touch
    #        without overlapping, same slot, similar colour
    def slot_g(ats):
        f = faces_de(ats)
        return slot_de(T, (T.R_atome[ats] * T.aire_atome[ats, None]).sum(0) / T.aire_atome[ats].sum())
    journal_fusion = T.journal_fusion = []
    for _ in range(3):
        n0 = len(vets)
        fl = [faces_de(v) for v in vets]
        sl = [slot_g(v) for v in vets]
        labv = [(T.lab_atome[v] * T.aire_atome[v, None]).sum(0) / T.aire_atome[v].sum() for v in vets]
        cases = [cases_corps(T, f) for f in fl]
        touche = contacts(T, fl, 0.006)
        u3 = UF(len(vets))
        for i in range(len(vets)):
            for j in range(i + 1, len(vets)):
                mi, mj = T.mat_atome[vets[i][0]], T.mat_atome[vets[j][0]]
                if mi == mj and not gen[vets[i][0]]:
                    # same named material: one garment, unless two distant slots, or neighbouring ones
                    # with different colours (trousers and sneakers in 裤 for Manato, briefs and boots
                    # in 靴 for Corin)
                    if sl[i] == sl[j] or (frozenset((sl[i], sl[j])) in SLOTS_VOISINS
                                          and np.linalg.norm(labv[i] - labv[j]) <= 2 * DE_MEME_TISSU):
                        u3.u(i, j)
                    continue
                if (i, j) not in touche or sl[i] != sl[j] or not (gen[vets[i][0]] and gen[vets[j][0]]):
                    continue
                if np.linalg.norm(labv[i] - labv[j]) > 2 * DE_MEME_TISSU:
                    continue
                com = set(cases[i]) & set(cases[j])
                if len(com) > 0.25 * min(len(cases[i]), len(cases[j])):
                    continue        # one covers the other: two layers
                petit, grand = (fl[i], fl[j]) if len(fl[i]) <= len(fl[j]) else (fl[j], fl[i])
                bc = bord_contre(T, petit, grand)
                journal_fusion.append((len(fl[i]), len(fl[j]), round(bc, 3)))
                if bc < BORD_CONTRE_MIN:
                    continue        # only a few contact points: two neighbouring garments
                u3.u(i, j)
        vets = [sum((vets[k] for k in g), []) for g in u3.groupes(range(len(vets)))]
        if len(vets) == n0:
            break
    # --- 5. small leftovers sewn to a garment: absorbed (piping, lapel); otherwise accessories
    taille = np.array([sum(T.nf_atome[a] for a in v) for v in vets])
    garde = []
    restes = []
    for k, v in enumerate(vets):
        if taille[k] < 0.02 * total_f:
            restes.append(k)
        else:
            garde.append(k)
    at_vet = {}
    for k in garde:
        for a in vets[k]:
            at_vet[a] = k
    acc_rest = []
    for k in sorted(restes, key=lambda k: -taille[k]):
        best, lb = None, 0
        for a in vets[k]:
            for (x, y), d in T.coutures.items():
                o = y if x == a else x if y == a else None
                if o is not None and o in at_vet and d["longueur"] > lb:
                    best, lb = at_vet[o], d["longueur"]
        if best is not None:
            vets[best] = vets[best] + vets[k]
            for a in vets[k]:
                at_vet[a] = best
        else:
            acc_rest.append(k)
    pieces = []
    for k in garde:
        pieces.append({"classe": "vetement", "atomes": sorted(vets[k])})
    # --- 6. accessories: accessory atoms, accessory shells, leftovers; those that touch each other
    #        (a buckle on its belt) become one
    acc_groupes = [list(c) for c in cq_acc] + [list(cq_vet[i]) for i in acc_cq] + [vets[k] for k in acc_rest]
    acc_raison = [raison_at[c[0]] for c in cq_acc] + [acc_cq[i] for i in acc_cq] + ["small isolated leftover"] * len(acc_rest)
    if acc_groupes:
        fl = [faces_de(g) for g in acc_groupes]
        touche = contacts(T, fl, 0.003)
        u4 = UF(len(acc_groupes))
        petit = [Geo(T, f).diag < 2 * ACC_DIAG for f in fl]
        for (i, j), n in touche.items():
            if n >= 2 and (petit[i] or petit[j]):
                u4.u(i, j)
        # named objects (tool kit, weapon, plush toy…): nearby pieces of the same material
        # are a single object
        mat_g = [T.mat_atome[g[0]] for g in acc_groupes]
        objet_g = [sorte_nom(noms[m]) in SORTES_OBJET for m in mat_g]
        if any(objet_g):
            proche = contacts(T, fl, 0.03)
            for (i, j) in proche:
                if objet_g[i] and mat_g[i] == mat_g[j]:
                    u4.u(i, j)
        for g in u4.groupes(range(len(acc_groupes))):
            ats = sorted(sum((acc_groupes[k] for k in g), []))
            pieces.append({"classe": "accessoire", "atomes": ats,
                           "raison": sorted({acc_raison[k] for k in g})})
    for pc in pieces:
        pc["faces"] = np.sort(faces_de(pc["atomes"]))
    # skin offcuts: an accessory entirely in generic material, at least half of whose area
    # is skin-coloured (ΔE < 12, colour per face) -> skin
    # ; enclosed in the skin (all its vertices welded to the skin, none to another part) and of
    # the colour of the surrounding skin (median ΔE < 12 to the neighbouring skin, < 24 to the
    # overall skin): fingertips sticking out of gloves (YeShunguang), ears and neck (Nicole),
    # edge between skin and fabric (Qingyi); the pink pattern on Aria's ankle (ΔE 27 to
    # her neighbouring skin) stays an accessory
    if peau_lab is not None:
        v_peau = np.zeros(len(M.pos_s), bool)
        f_peau = faces_de([a for a in range(na) if est_peau[a]]) if est_peau.any() else np.zeros(0, int)
        v_peau[M.Fs[f_peau].ravel()] = True
        n_pc = np.zeros(len(M.pos_s), int)
        for pc in pieces:
            n_pc[np.unique(M.Fs[pc["faces"]])] += 1
        reste = []
        for pc in pieces:
            fa = pc["faces"]
            if pc["classe"] == "accessoire" and all(gen[a] for a in pc["atomes"]):
                de = np.linalg.norm(T.lab_face[fa] - peau_lab, axis=1)
                ok = ~np.isnan(de)
                w = M.aire[fa][ok]
                vs = np.unique(M.Fs[fa])
                enclave = v_peau[vs].all() and (n_pc[vs] == 1).all() and ok.any() \
                    and np.median(de[ok]) < 2 * DE_PEAU_CHUTE
                if enclave:
                    mv = np.zeros(len(M.pos_s), bool)
                    mv[vs] = True
                    lv = T.lab_face[f_peau[mv[M.Fs[f_peau]].any(1)]]
                    lv = lv[~np.isnan(lv).any(1)]
                    lf = T.lab_face[fa][ok]
                    enclave = len(lv) > 0 and np.median(np.linalg.norm(lf - np.median(lv, 0), axis=1)) < DE_PEAU_CHUTE
                if w.sum() > 0 and ((w * (de[ok] < DE_PEAU_CHUTE)).sum() / w.sum() >= 0.5 or enclave):
                    est_peau[pc["atomes"]] = True
                    continue
            reste.append(pc)
        pieces[:] = reste
    peau_faces = faces_de([a for a in range(na) if est_peau[a]]) if est_peau.any() else np.zeros(0, int)
    T.est_peau = est_peau
    return pieces, {"peau_faces": peau_faces, "couleur_peau": None if peau_lab is None else peau_lab.tolist()}


# ====================================================================== types, kinds, layers

def zones_piece(T, faces):
    v = T.M.F[faces].ravel()
    w = np.repeat(T.M.aire[faces], 3)
    R = (T.R_v[v] * w[:, None]).sum(0) / max(w.sum(), 1e-20)
    return T.zones(R)


def jambes_separees(T, faces):
    """Does the garment wrap each leg separately (trousers, shorts) or both together
    (skirt, dress)? On thigh slices, counts the vertices on the inner side of each thigh
    (within ±25° of the direction of the other leg, within 10 cm of the bone, or within 1.1 times
    half the distance between the thighs for wide trousers on a large build). Trousers have some, a
    skirt does not (it passes in front of and behind the crotch). None if the part does not reach the thighs."""
    M = T.M
    noms = {n: i for i, n in enumerate(M.noms_os)}
    p = M.pos_s[np.unique(M.Fs[faces])]
    tot = inn = 0
    for c, o, g in (("左足", "右足", "左ひざ"), ("右足", "左足", "右ひざ")):
        if c not in noms or g not in noms or o not in noms:
            return None
        a, b, ao = M.tetes[noms[c]], M.tetes[noms[g]], M.tetes[noms[o]]
        go = M.tetes[noms[g.replace(c[0], o[0])]] if g.replace(c[0], o[0]) in noms else b * [-1, 1, 1]
        cote = np.sign(ao[0] - a[0])
        for t in (0.1, 0.2, 0.3, 0.45, 0.6):
            q = a + (b - a) * t
            demi = 0.5 * np.linalg.norm((ao + (go - ao) * t)[:2] - q[:2])
            m = np.abs(p[:, 2] - q[2]) < 0.02
            d = p[m, :2] - q[:2]
            r = np.linalg.norm(d, axis=1)
            ang = np.degrees(np.arctan2(d[:, 1], d[:, 0] * cote))
            tot += int((r < 0.25).sum())
            inn += int(((np.abs(ang) < 25) & (r < max(0.10, 1.1 * demi)) & (r > 0.005)).sum())
    if tot < 10:
        return None
    return inn >= 3 and inn / tot >= 0.02


AMPLE_TIBIA = 0.10   # m: median radius around the shin above which the lower garment hangs loose (a calf is 5 to 6 cm)


def ampleur_tibia(T, faces):
    """Median radius (m) of the part around the shin bones, between knee and ankle. None if it
    does not reach that far down."""
    M = T.M
    noms = {n: i for i, n in enumerate(M.noms_os)}
    p = M.pos_s[np.unique(M.Fs[faces])]
    rs = []
    for g, c in (("左ひざ", "左足首"), ("右ひざ", "右足首")):
        if g not in noms or c not in noms:
            continue
        a, b = M.tetes[noms[g]], M.tetes[noms[c]]
        u = b - a
        t = ((p - a) @ u) / max(float(u @ u), 1e-20)
        m = (t > 0.2) & (t < 0.8) & (np.sign(p[:, 0] - 0.5 * (M.tetes[noms["左ひざ"]][0] + M.tetes[noms["右ひざ"]][0]))
                                     == np.sign(a[0] - 0.5 * (M.tetes[noms["左ひざ"]][0] + M.tetes[noms["右ひざ"]][0]))) \
            if "左ひざ" in noms and "右ひざ" in noms else (t > 0.2) & (t < 0.8)
        if m.any():
            rs.append(np.linalg.norm((p[m] - a) - t[m, None] * u, axis=1))
    return float(np.median(np.concatenate(rs))) if rs else None


def tour_du_crane(T, faces):
    """Share of the 24 sectors around the head bone (4-10 cm above its head, within
    15 cm of the axis) where the part has vertices. A hood or a head of hair wraps around the skull;
    pigtails or strands hang down the sides."""
    M = T.M
    noms = {n: i for i, n in enumerate(M.noms_os)}
    if "頭" not in noms:
        return None
    c = M.tetes[noms["頭"]]
    p = M.pos_s[np.unique(M.Fs[faces])] - c
    m = (p[:, 2] > 0.04) & (p[:, 2] < 0.10) & (np.linalg.norm(p[:, :2], axis=1) < 0.15)
    if not m.any():
        return 0.0
    sect = ((np.arctan2(p[m, 1], p[m, 0]) + np.pi) / (2 * np.pi) * 24).astype(int) % 24
    return len(np.unique(sect)) / 24


CHEVEUX_OS = re.compile(r"[Hh]air|髪|髮|发")


def sous_hanches(T, faces):
    """How far (m) the part goes below the heads of the thigh bones."""
    M = T.M
    noms = {n: i for i, n in enumerate(M.noms_os)}
    zs = [M.tetes[noms[n]][2] for n in ("左足", "右足") if n in noms]
    if not zs:
        return 0.0
    return float(min(zs) - M.pos_s[np.unique(M.Fs[faces]), 2].min())


def part_poitrine(T, faces):
    """Share of the part's area above the chest (8 cm above the head of 上半身2):
    a skirt never goes up there, a short dress does (its lower half outweighs its upper half)."""
    M = T.M
    noms = {n: i for i, n in enumerate(M.noms_os)}
    if "上半身2" not in noms:
        return 0.0
    zc = M.pos_s[M.Fs[faces], 2].mean(1)
    w = M.aire[faces]
    return float((w * (zc > M.tetes[noms["上半身2"]][2] + 0.08)).sum() / max(w.sum(), 1e-20))


COL_POITRINE = 0.85   # share of the area above the chest beyond which a torso garment is a collar


def type_vetement(T, faces, couche, propre, sur_haut=None):
    """Type of a garment from the body zones it covers; `couche`: number of garments
    it covers + 1; `sur_haut`: does it cover another top (jacket, vest, coat)?"""
    if sur_haut is None:
        sur_haut = couche >= 2
    z = zones_piece(T, faces)
    haut = z["torse"] + z["bras"] + z["avant_bras"] + z["cou"]
    bas = z["bassin"] + z["cuisse"] + z["jambe"]
    manches = z["bras"] + z["avant_bras"]
    bras = z["bras"] + z["avant_bras"] + z["main"]
    if z["main"] >= 0.5 or (bras >= 0.8 and z["main"] >= 0.15):
        return "gants"
    if z["pied"] >= 0.35:
        return "bottes" if z["jambe"] + z["cuisse"] >= 0.25 else "chaussures"
    # a long lower garment also weighs mostly on the legs: a sock wraps each leg
    # closely; whatever reaches the pelvis and hangs loose around the shin is wide trousers or a skirt
    if (z["jambe"] + z["pied"] >= 0.5 or (z["jambe"] + z["cuisse"] + z["pied"] >= 0.75 and z["bassin"] < 0.1)) \
            and jambes_separees(T, faces) is not False \
            and not (z["bassin"] >= 0.05 and (ampleur_tibia(T, faces) or 0) >= AMPLE_TIBIA):
        return "chaussettes"
    if z["cou"] >= 0.5 or (z["torse"] + z["cou"] >= 0.5 and part_poitrine(T, faces) >= COL_POITRINE):
        # sailor collar, fur collar, scarf, neck warmer: on the torso but entirely above the
        # chest (a vest or a top goes below it: ≤ 0.54 measured)
        return "col"
    if z["tete"] >= 0.5:
        return "capuche"
    if manches >= 0.5 and z["torse"] < 0.1:
        # arms only, no torso: sleeves (Corin's 袖, Remielle's 袖子), not a top
        return "manches"
    if haut >= 0.45 or (haut >= 0.35 and bas >= 0.35) or (bas >= 0.3 and part_poitrine(T, faces) >= 0.04):
        long_ = sous_hanches(T, faces) > 0.15
        if bas >= 0.25 or long_:
            if (sur_haut or manches >= 0.15) and jambes_separees(T, faces) is not True:
                return "manteau"
            return "combinaison" if jambes_separees(T, faces) else "robe"
        if sur_haut:
            return "veste" if manches >= 0.15 else "gilet"
        return "haut"
    if manches >= 0.5:
        return "manches"
    if bas >= 0.5:
        if jambes_separees(T, faces) is False or propre >= 0.3:
            return "jupe"
        return "pantalon" if z["jambe"] >= 0.15 else "short"
    return "autre"


# kinds read from a name that say where the part is worn: kept only if the part's dominant zone
# is there ("Belt" also names arm straps, "Tie" hip ribbons)
ZONES_SORTE = {"ceinture": ("torse", "bassin"), "cravate": ("cou", "torse"), "couvre_chef": ("tete",),
               "lunettes": ("tete",), "masque": ("tete", "cou"), "casque_audio": ("tete", "cou"),
               "oreilles": ("tete",), "boucle_oreille": ("tete",), "bague": ("main",), "collier": ("cou", "torse"),
               "jarretiere": ("cuisse", "jambe"), "echarpe": ("cou", "torse")}

# animal ear bones (Bdy_L_Ear_01, Ctr_Tea_R_Ear_01, MY_Bone_Ear_L); not Earring, Earphone, EarHair
OREILLE_OS = re.compile(r"(^|_)Ear(_|\d|$)")


def tour_seme(T, faces, os_axe):
    """Share of the 24 sectors around `os_axe` covered by points seeded on the faces
    (vertices, edge midpoints, centres) closest to this bone. A band of large quads does not
    have a vertex in every sector: Nicole's thigh strap covers 0.62 of the turn by
    its vertices, 0.79 by its faces."""
    M = T.M
    P3 = M.pos_s[M.Fs[faces]]
    pts = np.concatenate([P3.reshape(-1, 3), ((P3 + np.roll(P3, 1, 1)) / 2).reshape(-1, 3), P3.mean(1)])
    k, _, ang, _ = os_proche(pts, T.seg)
    m = T.seg[0][k] == os_axe
    if not m.any():
        return 0.0
    return len(np.unique(((ang[m] + np.pi) / (2 * np.pi) * 24).astype(int) % 24)) / 24


def sorte_accessoire(T, pc, g):
    M = T.M
    z = zones_piece(T, pc["faces"])
    zd = max(z, key=z.get) if z else None
    for quoi, noms in (("material", pc["materiaux"]), ("bone", [n for n, _ in pc["os_propres"][:3]])):
        for n in noms:
            s = sorte_nom(n)
            if s and (s not in ZONES_SORTE or zd in ZONES_SORTE[s]):
                return s, "%s name (%s)" % (quoi, n)
    reg = region_axe(T, g.os_axe)
    # seeded turn: only around the limbs and the neck, where bands of large quads fool the
    # per-vertex turn; around the trunk it would take flaps and clips for rings
    # (Nicole sangle-12, Soldat11 ornement-21, Remielle sangle-15: seen on the thumbnails)
    tour = g.tour
    if reg in ("cuisse", "jambe", "pied", "bras", "avant_bras", "main", "cou") and tour < ANNEAU_TOUR:
        tour = max(tour, tour_seme(T, pc["faces"], g.os_axe))
    anneau = g.etendue_axe < ANNEAU_HAUT * 1.5 and tour >= ANNEAU_TOUR
    if anneau:
        s = {"bassin": "ceinture", "fesses": "ceinture", "ventre": "ceinture", "cuisse": "jarretiere",
             "jambe": "jarretiere", "bras": "brassard", "avant_bras": "bracelet", "main": "bracelet",
             "cou": "collier", "tete": "couvre_chef", "pied": "chevillere"}.get(reg, "anneau")
        # the leg bone (ひざ) runs from knee to ankle: a ring on its lower half is at the
        # ankle (Yuzuha, Zhao), not a garter
        if reg == "jambe":
            seg = T.seg
            j = int(np.flatnonzero(seg[0] == g.os_axe)[0])
            L = np.linalg.norm(seg[2][j] - seg[1][j])
            k, t, _, _ = os_proche(M.pos_s[M.Fs[pc["faces"]]].mean(1), seg)
            if (k == j).any() and np.median(t[k == j]) > 0.5 * L:
                s = "chevillere"
        # a ring entirely on the head is headwear, even if its axis is the neck:
        # Orphie's beret (体2, resting on 頭) turns around the extension of 首
        if zd == "tete":
            s = "couvre_chef"
        return s, "ring around %s" % M.noms_os[g.os_axe]
    # animal ears: head part carried by its own ear bones, neither tiny (earring)
    # nor a narrow band (Soukaku, Jane, Pulchra, Chinatsu: 体, 毛2, 饰 with no ear name)
    if pc["os_propres"] and OREILLE_OS.search(pc["os_propres"][0][0]) and zd == "tete" \
            and g.diag >= ACC_DIAG and g.largeur >= SANGLE_LARGEUR * 1.3:
        return "oreilles", "ear bone (%s)" % pc["os_propres"][0][0]
    if any(x.startswith("cheveux") for x in pc.get("raison", [])):
        return "cheveux", pc["raison"][0]
    if any(x.startswith("coiffe") for x in pc.get("raison", [])):
        return "couvre_chef", pc["raison"][0]
    if any(x.startswith("bandes") for x in pc.get("raison", [])):
        return ("harnais" if reg in ("tronc_haut", "poitrine", "ventre", "bassin") else "sangle"), "bandes"
    if g.diag < ACC_DIAG:
        return "boucle_bouton", "small part"
    if g.largeur < SANGLE_LARGEUR * 1.3:
        return ("harnais" if reg in ("tronc_haut", "poitrine", "ventre") and g.diag > 0.3 else "sangle"), "narrow band"
    ep = np.sort(g.hi - g.lo)
    if reg in ("bassin", "fesses", "cuisse") and ep[0] > 0.02:
        return "poche", "volume lying on the hip or thigh"
    if ep[0] < 0.012 and g.diag < 0.2:
        return "ecusson", "flat part"
    if reg == "tete":
        return "couvre_chef", "on the head"
    return "ornement", "other"


def couches(T, pieces):
    """Layer of each garment: 1 + number of garments it covers (larger radius
    around the same bones, on at least 10 % of the shared cells)."""
    cases = [cases_corps(T, pc["faces"]) for pc in pieces]
    n = len(pieces)
    dessus = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            com = set(cases[i]) & set(cases[j])
            if len(com) < max(4, 0.1 * min(len(cases[i]), len(cases[j]))):
                continue
            d = np.array([cases[i][c] - cases[j][c] for c in com])
            if (d > 0.003).mean() > 0.6:
                dessus[i].add(j)
            elif (d < -0.003).mean() > 0.6:
                dessus[j].add(i)
    return [1 + len(s) for s in dessus], dessus


PAIRES = ("gants", "chaussures", "bottes", "chaussettes", "manches")
NOMS_FR = {"haut": "haut", "veste": "veste", "gilet": "gilet", "manteau": "manteau", "robe": "robe",
           "combinaison": "combinaison", "jupe": "jupe", "pantalon": "pantalon", "short": "short",
           "chaussettes": "chaussettes", "chaussures": "chaussures", "bottes": "bottes", "gants": "gants",
           "col": "col", "capuche": "capuche", "manches": "manches", "autre": "vetement"}


REMPLISSAGE_MIN = 0.75   # area / area of the body cells crossed: below this threshold, bands (harness)


def remplissage(T, faces):
    cs = cases_corps(T, faces)
    r = np.maximum(np.array(list(cs.values())), 0.02)
    return float(T.M.aire[faces].sum() / (0.025 * 2 * np.pi * r / 12).sum())


def os_de(T, faces):
    """Bones carrying the faces' vertices: [(name, weight share)], body bones and own bones."""
    M = T.M
    v = np.unique(M.F[faces])
    w = np.zeros(len(M.noms_os))
    for k in range(M.idx.shape[1]):
        b, p = M.idx[v, k], M.poids[v, k]
        ok = (b >= 0) & (p > 0)
        np.add.at(w, b[ok], p[ok])
    w /= max(w.sum(), 1e-20)
    o = np.argsort(-w)
    corps = [(M.noms_os[b], round(float(w[b]), 3)) for b in o if w[b] > 0.005 and T.os_corps[b]]
    propres = [(M.noms_os[b], round(float(w[b]), 4)) for b in o if w[b] > 0 and not T.os_corps[b]]
    return corps, propres, float(sum(w[b] for b in range(len(w)) if not T.os_corps[b]))


MANCHE_DE_RESTE = 20.0    # minimum ΔE between an arm cluster and the rest of its garment
MANCHE_DE_HAUT = 10.0     # maximum ΔE between the cluster and the sleeveless top that receives it
MANCHE_DIST = 0.04        # m: maximum cluster-top distance
MANCHE_AIRE = 0.005       # m²: minimum area of the cluster
MANCHE_CASES = 0.1        # maximum share of the cluster's body cells that the top already covers


def _lab_moyen(T, ats):
    ats = np.asarray(sorted(ats))
    w = T.aire_atome[ats]
    return (T.lab_atome[ats] * w[:, None]).sum(0) / max(w.sum(), 1e-20)


def deplacer_manches(T, vet):
    """Jacket sleeves left in the garment underneath (ZhuYuan: the jacket's blue sleeves
    sewn to the armholes of the grey jumpsuit, with no seam to the jacket body):
    a cluster of arm atoms (upper arm + forearm ≥ 0.5, sewn to each other) whose colour is
    clearly different from the rest of its garment (ΔE ≥ 20) moves to a sleeveless top
    (torso ≥ 0.5, arms < 0.05) of its colour (ΔE ≤ 10), within 4 cm, that does not already
    cover it (< 10 % shared cells); atoms of the same colour sewn to the cluster (armholes)
    follow it. Decisions are made on the prior state, then the moves are applied. Measured on the 49 outfits:
    only ZhuYuan triggers it."""
    M = T.M
    faces_de = lambda ats: np.concatenate([T.faces_atome[a] for a in sorted(ats)])  # noqa: E731
    zq = [zones_piece(T, pc["faces"]) for pc in vet]
    lab_v = [_lab_moyen(T, pc["atomes"]) for pc in vet]
    bras_at = lambda a: T.zones(T.R_atome[a])["bras"] + T.zones(T.R_atome[a])["avant_bras"]  # noqa: E731
    decisions = []
    for ip, P in enumerate(vet):
        ats = set(P["atomes"])
        armat = [a for a in ats if T.nf_atome[a] >= 4 and bras_at(a) >= 0.5]
        if not armat:
            continue
        uf = UF(T.na)
        sa = set(armat)
        for (x, y) in T.coutures:
            if x in sa and y in sa:
                uf.u(x, y)
        for G in uf.groupes(armat):
            G = set(G)
            reste = ats - G
            if T.aire_atome[sorted(G)].sum() < MANCHE_AIRE or not reste:
                continue
            labG = _lab_moyen(T, G)
            if np.linalg.norm(labG - _lab_moyen(T, reste)) < MANCHE_DE_RESTE:
                continue
            fg = faces_de(G)
            pg = M.pos_s[np.unique(M.Fs[fg])]
            cg = None
            best = None
            for iq, Q in enumerate(vet):
                z = zq[iq]
                if iq == ip or z["torse"] < 0.5 or z["bras"] + z["avant_bras"] >= 0.05:
                    continue
                de = float(np.linalg.norm(labG - lab_v[iq]))
                if de > MANCHE_DE_HAUT:
                    continue
                pq = M.pos_s[np.unique(M.Fs[Q["faces"]])]
                dm = min(np.linalg.norm(pg[i:i + 400, None] - pq[None], axis=2).min() for i in range(0, len(pg), 400))
                if dm > MANCHE_DIST:
                    continue
                cg = cg if cg is not None else cases_corps(T, fg)
                if len(set(cg) & set(cases_corps(T, Q["faces"]))) >= MANCHE_CASES * max(len(cg), 1):
                    continue
                if best is None or de < best[0]:
                    best = (de, iq)
            if best is None:
                continue
            g = set(G)
            fini = False
            while not fini:
                fini = True
                for (x, y), d in T.coutures.items():
                    for a, o in ((x, y), (y, x)):
                        if a in g and o in reste and o not in g and d["longueur"] > 0.01 \
                                and np.linalg.norm(T.lab_atome[o] - labG) <= MANCHE_DE_HAUT:
                            g.add(o)
                            fini = False
            decisions.append((ip, best[1], g))
    for ip, iq, g in decisions:
        for k, plus in ((ip, False), (iq, True)):
            pc = vet[k]
            pc["atomes"] = sorted((set(pc["atomes"]) | g) if plus else (set(pc["atomes"]) - g))
            pc["faces"] = np.sort(faces_de(pc["atomes"]))
            pc["materiaux"] = sorted({M.noms_mat[m] for m in np.unique(M.mf[pc["faces"]])})
            pc["os_corps"], pc["os_propres"], pc["part_os_propres"] = os_de(T, pc["faces"])
        vet[iq].setdefault("raison", []).append("same-colour sleeves taken back from the garment underneath")
    return decisions


def decrire(T, pieces, info):
    """Completes the parts: materials, bones, type or kind, layer, carrier, hidden skin, overlay layers,
    identifier; merges left-right pairs of the same type."""
    M = T.M
    for pc in pieces:
        pc["materiaux"] = sorted({M.noms_mat[m] for m in np.unique(M.mf[pc["faces"]])})
        pc["os_corps"], pc["os_propres"], pc["part_os_propres"] = os_de(T, pc["faces"])
    # braces, harness, scarf: a "garment" that covers only a small share of the body cells
    # it crosses is made of bands (only named material affected, measured on the 49
    # outfits: Lighter's scarf 围巾, 0.45; all other garments ≥ 0.75)
    for pc in pieces:
        if pc["classe"] == "vetement" and "type_force" not in pc:
            r = remplissage(T, pc["faces"])
            if r < REMPLISSAGE_MIN:
                pc["classe"] = "accessoire"
                pc["raison"] = ["bandes (fill ratio %.2f)" % r]
    # head part that is not a hood: in 毛 material, on hair bones, or not wrapping around
    # the skull. It is hair (Pulchra's 毛2, Nicole's 新規 pigtails) if the material or
    # the bones say so, or if it is in generic material with no ear bone; otherwise a headdress
    # (material named after a fabric: Corin's frilled headdress 裙) or ears (Ear bones)
    for pc in pieces:
        if pc["classe"] != "vetement" or "type_force" in pc or zones_piece(T, pc["faces"])["tete"] < 0.5:
            continue
        tc = tour_du_crane(T, pc["faces"])
        nom_ch = (any(n.startswith("毛") for n in pc["materiaux"])
                  or any(CHEVEUX_OS.search(n) for n, _ in pc["os_propres"][:5]))
        if not nom_ch and not (tc is not None and tc < 0.5):
            continue
        pc["classe"] = "accessoire"
        oreille = bool(pc["os_propres"]) and bool(OREILLE_OS.search(pc["os_propres"][0][0]))
        tcs = None if tc is None else round(tc, 2)
        # in generic material, with no hair name: strands move on their own bones
        # (Nicole's pigtails); a rigid part, entirely on the head bone, is a worn object
        # (Orphie's earmuffs): headwear
        if nom_ch or (all(mat_generique(n) for n in pc["materiaux"]) and not oreille and pc["os_propres"]):
            pc["raison"] = ["cheveux (skull coverage %s)" % tcs]
        elif oreille:
            pc["raison"] = ["ear bones (skull coverage %s)" % tcs]
        elif all(mat_generique(n) for n in pc["materiaux"]):
            pc["raison"] = ["coiffe: rigid on the head, no own bones, does not wrap around the skull (%s)" % tcs]
        else:
            pc["raison"] = ["coiffe: named material, does not wrap around the skull (%s)" % tcs]
    vet = [pc for pc in pieces if pc["classe"] == "vetement"]
    acc = [pc for pc in pieces if pc["classe"] == "accessoire"]
    cs, _ = couches(T, vet)
    for pc, c in zip(vet, cs):
        pc["type"] = type_vetement(T, pc["faces"], c, pc["part_os_propres"], False)
    # left-right pairs of the same type: one part
    fini = False
    while not fini:
        fini = True
        for i in range(len(vet)):
            for j in range(i + 1, len(vet)):
                a, b = vet[i], vet[j]
                if a["type"] == b["type"] and a["type"] in PAIRES and miroirs(T, a["faces"], b["faces"]):
                    a["atomes"] = sorted(a["atomes"] + b["atomes"])
                    a["faces"] = np.sort(np.concatenate([a["faces"], b["faces"]]))
                    a["materiaux"] = sorted(set(a["materiaux"]) | set(b["materiaux"]))
                    a["os_corps"], a["os_propres"], a["part_os_propres"] = os_de(T, a["faces"])
                    vet.pop(j)
                    fini = False
                    break
            if not fini:
                break
    cs, dessus = couches(T, vet)
    deplacer_manches(T, vet)
    zh = [zones_piece(T, pc["faces"]) for pc in vet]
    # a covered garment counts as a top if it covers the torso and arms, or if it is a
    # jumpsuit or a dress (ZhuYuan: her jumpsuit, without the jacket's sleeves, only has
    # 0.26 of top left, but the jacket covers it)
    est_haut = [z["torse"] + z["bras"] + z["avant_bras"] >= 0.45 or pc["type"] in ("combinaison", "robe")
                for pc, z in zip(vet, zh)]
    for k, (pc, c) in enumerate(zip(vet, cs)):
        pc["couche"] = c
        pc["type"] = pc.get("type_force") or type_vetement(T, pc["faces"], c, pc["part_os_propres"],
                                                           any(est_haut[j] for j in dessus[k]))
        pc["recouvre"] = sorted(dessus[k])
    # carrier of an accessory: the garment it touches most (2 cm cells)
    tout = vet + acc
    cont = contacts(T, [pc["faces"] for pc in tout], 0.02)
    for k, pc in enumerate(acc):
        i = len(vet) + k
        best, nb = None, 0
        for j in range(len(vet)):
            n = cont.get((min(i, j), max(i, j)), 0)
            if n > nb:
                best, nb = j, n
        pc["porteur_indice"] = best
        g = Geo(T, pc["faces"])
        pc["sorte"], pc["sorte_raison"] = sorte_accessoire(T, pc, g)
        pc["couche"] = (vet[best]["couche"] + 1) if best is not None else 1
    # no garment in its cells: whatever its surface touches most closely, a garment, another
    # accessory (a belt's buckle, a strap's button) or skin (worn directly on the body)
    groupes = [pc["faces"] for pc in tout] + [info["peau_faces"]]
    for k, pc in enumerate(acc):
        if pc["porteur_indice"] is not None:
            continue
        i = len(vet) + k
        nb = plus_proche_surface(T, pc["faces"], groupes[:i] + [np.zeros(0, int)] + groupes[i + 1:])
        # skin carries nothing: it is only underneath. A part lying flat on the skin that
        # also touches another part (the strap end of a garter) belongs to that part
        j = int(nb[:len(tout)].argmax())
        if nb[j] < max(2, PORTE_PART * nb.sum()):
            continue
        if j < len(vet):
            pc["porteur_indice"] = j
            pc["couche"] = vet[j]["couche"] + 1
        else:
            pc["porteur_acc"] = tout[j]
    # accessory chain: a trim lying on another trim follows that trim's carrier;
    # lying on a selectable accessory (belt, bag, harness), it belongs to that accessory
    for pc in acc:
        x, vus_ = pc.get("porteur_acc"), 0
        while x is not None and x["sorte"] in GARNITURES and vus_ < 8:
            if x["porteur_indice"] is not None:
                pc["porteur_indice"] = x["porteur_indice"]
                pc["couche"] = vet[x["porteur_indice"]]["couche"] + 1
                x = None
                break
            x, vus_ = x.get("porteur_acc"), vus_ + 1
        pc["porteur_acc"] = x if (x is not None and x["sorte"] not in GARNITURES and x is not pc) else None
    # left-right pairs of accessories of the same kind (two bracelets, two garters): one part
    k = 0
    while k < len(acc):
        a = acc[k]
        for j in range(k + 1, len(acc)):
            b = acc[j]
            if a["sorte"] == b["sorte"] and miroirs(T, a["faces"], b["faces"]):
                a["atomes"] = sorted(a["atomes"] + b["atomes"])
                a["faces"] = np.sort(np.concatenate([a["faces"], b["faces"]]))
                a["materiaux"] = sorted(set(a["materiaux"]) | set(b["materiaux"]))
                a["os_corps"], a["os_propres"], a["part_os_propres"] = os_de(T, a["faces"])
                a["raison"] = sorted(set(a.get("raison", [])) | set(b.get("raison", [])))
                a["paire"] = True
                b["fondu_dans"] = a
                acc.pop(j)
                break
        k += 1
    tout = vet + acc
    # hidden skin: body zones covered; skin-tight if sewn to the skin
    peau = T.est_peau
    for pc in tout:
        z = zones_piece(T, pc["faces"])
        pc["zones"] = {k: round(v, 3) for k, v in z.items() if v >= 0.01}
        ats = set(pc["atomes"])
        cousu = sum(d["longueur"] for (a, b), d in T.coutures.items()
                    if (a in ats and peau[b]) or (b in ats and peau[a]))
        pc["cousu_peau_mm"] = round(cousu * 1000, 1)
    # overlay layers ("+"): each face joins the part that carries the same welded face (or the skin);
    # with no exact twin, the part that carries most of its welded vertices
    cal = np.where(T.calque)[0]
    porteurs = [pc["faces"] for pc in tout] + [info["peau_faces"]]
    pc_de = np.full(len(M.F), -1)
    for k, f in enumerate(porteurs):
        pc_de[f] = k
    peau_cal = []
    info["calques_sans_jumelle"] = 0
    if len(cal):
        cle = lambda F: np.sort(F, 1) @ np.array([1, 1 << 20, 1 << 40], dtype=np.int64)  # noqa: E731
        base = np.where(pc_de >= 0)[0]
        kb = cle(M.Fs[base])
        o = np.argsort(kb)
        kc = cle(M.Fs[cal])
        pos = np.clip(np.searchsorted(kb[o], kc), 0, len(kb) - 1)
        trouve = kb[o][pos] == kc
        info["calques_sans_jumelle"] = int((~trouve).sum())
        pc_v = np.full(len(M.pos_s), -1)
        for k, f in enumerate(porteurs):
            pc_v[M.Fs[f].ravel()] = k
        for f, ok, pp in zip(cal, trouve, pos):
            if ok:
                k = pc_de[base[o][pp]]
            else:
                c = pc_v[M.Fs[f]]
                c = c[c >= 0]
                k = np.bincount(c).argmax() if len(c) else len(tout)
            if k >= len(tout):
                peau_cal.append(int(f))
            else:
                tout[k].setdefault("calques", []).append(int(f))
    info["peau_calques"] = np.array(peau_cal, int)
    for pc in tout:
        pc["faces_calque"] = np.array(sorted(pc.pop("calques", [])), int)
    # identifiers
    vus = {}
    for pc in vet + acc:
        nom = NOMS_FR.get(pc.get("type"), None) or pc["sorte"]
        vus[nom] = vus.get(nom, 0) + 1
        pc["id"] = nom if vus[nom] == 1 else "%s-%d" % (nom, vus[nom])
    for pc in vet + acc:
        n = vus[pc["id"].split("-")[0]]
        if n > 1 and "-" not in pc["id"]:
            pc["id"] = pc["id"] + "-1"
    for pc in acc:
        pc["porteur"] = vet[pc["porteur_indice"]]["id"] if pc["porteur_indice"] is not None else None
        x = pc.pop("porteur_acc", None)
        while x is not None and "fondu_dans" in x:
            x = x["fondu_dans"]
        if pc["porteur"] is None and x is not None and x is not pc:
            pc["porteur"] = x["id"]
    for pc in vet:
        pc["recouvre"] = [vet[j]["id"] for j in pc["recouvre"]]
    return vet + acc


# ====================================================================== hidden skin

def peau_cachee(T, faces, peau_faces, marge=0.001):
    """Skin faces under the part: their three vertices fall in cells (bone, slice,
    sector) covered by the part, closer to the bone than the part."""
    if len(peau_faces) == 0:
        return np.zeros(0, int)
    cp = cases_corps(T, faces)
    M = T.M
    vs = np.unique(M.Fs[peau_faces])
    k, t, ang, r = (a[vs] for a in T.axe_s)
    c = (k.astype(np.int64) * 1000 + (t / 0.025).astype(np.int64)) * 100 + \
        ((ang + np.pi) / (2 * np.pi) * 12).astype(np.int64) % 12
    rp = np.array([cp.get(int(x), -1.0) for x in c])
    dessous = np.zeros(len(M.pos_s), bool)
    dessous[vs] = r < rp + marge
    return peau_faces[dessous[M.Fs[peau_faces]].all(1)]


# ====================================================================== PMX writing

sys.path.insert(0, os.path.join(ICI, "..", "head"))


def _ecrire_pmx_module():
    import pmx_write as pmx_ecrire  # tools/head (read-only: imported, not modified)
    return pmx_ecrire


def _lier(src, dst):
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


def os_gardes(M, verts, propres_corps):
    """Bones to keep for some vertices: those carrying weight, their ancestors, the descendants of
    own bones (ends of physics chains), append sources; rigid bodies on own bones and
    their anchors (rigid bodies linked by a joint), joints between kept rigid bodies."""
    p = M.p
    oss = p["os"]
    n = len(oss)
    garde = np.zeros(n, bool)
    b = M.idx[verts].ravel()
    w = M.poids[verts].ravel()
    garde[np.unique(b[(b >= 0) & (w > 0)])] = True
    enfants = [[] for _ in range(n)]
    for i, o in enumerate(oss):
        if 0 <= o["parent"] < n:
            enfants[o["parent"]].append(i)
    propre = ~propres_corps
    pile = [i for i in np.where(garde & propre)[0]]
    while pile:
        i = pile.pop()
        for e in enfants[i]:
            if propre[e] and not garde[e]:
                garde[e] = True
                pile.append(e)
    rig = p["rigides"]
    jts = p["joints"]
    rg = np.zeros(len(rig), bool)
    for k, r in enumerate(rig):
        if 0 <= r["os"] < n and garde[r["os"]] and propre[r["os"]]:
            rg[k] = True
    for j in jts:
        a, c = j["a"], j["b"]
        if 0 <= a < len(rig) and 0 <= c < len(rig) and (rg[a] or rg[c]):
            for x in (a, c):
                if not rg[x]:
                    rg[x] = True
                    if 0 <= rig[x]["os"] < n:
                        garde[rig[x]["os"]] = True
    jg = [k for k, j in enumerate(jts) if 0 <= j["a"] < len(rig) and 0 <= j["b"] < len(rig)
          and rg[j["a"]] and rg[j["b"]]]
    # body collision rigid bodies (static, on a body bone) that the part's dynamic rigid bodies
    # hit within 12 cm: without them the fitted skirt clips through the legs.
    dyn = [k for k in np.where(rg)[0] if rig[k]["mode"] in (1, 2)]
    if dyn:
        cd = np.array([rig[k]["pos"] for k in dyn], float)
        for k, r in enumerate(rig):
            if rg[k] or r["mode"] != 0 or not (0 <= r["os"] < n) or propre[r["os"]]:
                continue
            hit = [kk for kk in dyn if (r["masque"] >> rig[kk]["groupe"]) & 1 and (rig[kk]["masque"] >> r["groupe"]) & 1]
            if hit and np.linalg.norm(cd[[dyn.index(x) for x in hit]] - np.asarray(r["pos"], float), axis=1).min() \
                    - max(r["taille"]) < 1.5:
                rg[k] = True
                garde[r["os"]] = True
    while True:
        avant = garde.sum()
        for i in np.where(garde)[0]:
            o = oss[i]
            if 0 <= o["parent"] < n:
                garde[o["parent"]] = True
            if o["ajout"] and 0 <= o["ajout"]["source"] < n:
                garde[o["ajout"]["source"]] = True
        if garde.sum() == avant:
            break
    return np.where(garde)[0], np.where(rg)[0], jg


def piece_pmx(T, faces, nom, dossier_tex_rel, textures_copiees):
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
    ob, rg, jg = os_gardes(M, vs, np.array(T.os_corps))
    no = np.full(len(p["os"]), -1)
    no[ob] = np.arange(len(ob))
    nr = np.full(len(p["rigides"]), -1)
    nr[rg] = np.arange(len(rg))
    S0 = p["sommets"]
    S = {k: [S0[k][i] for i in vs] for k in S0}
    S["os"] = [tuple(int(no[b]) if b >= 0 and no[b] >= 0 else (0 if b >= 0 else -1) for b in o) for o in S["os"]]
    # bones
    oss = []
    for i in ob:
        o = copy.deepcopy(p["os"][i])
        o["parent"] = int(no[o["parent"]]) if 0 <= o["parent"] < len(no) else -1
        if o["queue_os"] is not None:
            o["queue_os"] = int(no[o["queue_os"]]) if 0 <= o["queue_os"] < len(no) else -1
        if o["ajout"]:
            s = o["ajout"]["source"]
            if 0 <= s < len(no) and no[s] >= 0:
                o["ajout"]["source"] = int(no[s])
            else:
                o["ajout"] = None
                o["drapeaux"] &= ~0x0300
        if o["ik"]:
            ik = o["ik"]
            ok = 0 <= ik["cible"] < len(no) and no[ik["cible"]] >= 0 and all(no[l["os"]] >= 0 for l in ik["liens"])
            if ok:
                ik["cible"] = int(no[ik["cible"]])
                for l in ik["liens"]:
                    l["os"] = int(no[l["os"]])
            else:
                o["ik"] = None
                o["drapeaux"] &= ~0x0020
        oss.append(o)
    rig = []
    for k in rg:
        r = copy.deepcopy(p["rigides"][k])
        r["os"] = int(no[r["os"]]) if 0 <= r["os"] < len(no) else -1
        rig.append(r)
    jts = []
    for k in jg:
        j = copy.deepcopy(p["joints"][k])
        j["a"], j["b"] = int(nr[j["a"]]), int(nr[j["b"]])
        jts.append(j)
    # materials and textures
    mats, tex, tmap = [], [], {}
    dossier = os.path.dirname(M.chemin)

    def t_index(t):
        if t < 0 or t >= len(p["textures"]):
            return -1
        if t not in tmap:
            src = os.path.normpath(os.path.join(dossier, p["textures"][t].replace("\\", "/")))
            base = os.path.basename(src)
            if os.path.exists(src):
                textures_copiees[base] = src
            tmap[t] = len(tex)
            tex.append((dossier_tex_rel + "/" + base).replace("/", "\\"))
        return tmap[t]
    Fout = []
    for m in np.unique(M.mf[faces]):
        sel = M.mf[faces] == m
        mm = copy.deepcopy(p["materiaux"][int(m)])
        mm["texture"] = t_index(mm["texture"])
        mm["sphere"] = t_index(mm["sphere"])
        if mm["toon_partage"] == 0:
            mm["toon"] = t_index(mm["toon"])
        mm["n_indices"] = int(3 * sel.sum())
        mats.append(mm)
        Fout.extend(tuple(int(nv[v]) for v in f) for f in raw[sel])
    # display frames: root, expressions (empty), all bones; rigid bodies are not in them
    cadres = [{"nom": "Root", "nom_en": "Root", "special": 1, "elements": [(0, 0)]},
              {"nom": "表情", "nom_en": "Exp", "special": 1, "elements": []},
              {"nom": "os", "nom_en": "bones", "special": 0, "elements": [(0, i) for i in range(1, len(oss))]}]
    out = {k: p[k] for k in ("version", "encodage", "n_uv_add")}
    out.update({"tailles_index": p.get("tailles_index"), "nom": nom, "nom_en": nom,
                "commentaire": "Part of %s (ZZZ clothing library)" % T.perso,
                "commentaire_en": "", "sommets": S, "faces": Fout, "textures": tex, "materiaux": mats,
                "os": oss, "morphes": [], "cadres": cadres, "rigides": rig, "joints": jts,
                "octets_restants": b""})
    return out, {"sommets": len(vs), "os": len(ob), "rigides": len(rg), "joints": len(jg),
                 "os_physiques": int(sum(1 for r in rig if r["mode"] != 0))}


# ====================================================================== automatic check

INCRUSTATION_MIN = 0.9    # share of an atom's border sewn to the garment above which it is an inlaid panel


def part_cousue(T, a):
    """Length sewn to other atoms / length of the atom's (welded) border."""
    M = T.M
    Fs = M.Fs[T.faces_atome[a]]
    E = np.sort(np.concatenate([Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    b = u[c == 1]
    Lb = np.linalg.norm(M.pos_s[b[:, 0]] - M.pos_s[b[:, 1]], axis=1).sum()
    Lc = sum(d["longueur"] for (x, y), d in T.coutures.items() if a in (x, y))
    return Lc / max(Lb, 1e-9)


def controler(T, pieces, info):
    """Three checks per outfit. Returns {"faces_perdues", "faces_en_double", "deux_vetements",
    "accessoire_reste"} (lists of anomalies) and the face tally."""
    M = T.M
    tenue = np.array([T.roles[m] not in ("tete", "vide") for m in M.mf])
    compte = np.zeros(len(M.F), int)
    for pc in pieces:
        compte[pc["faces"]] += 1
        compte[pc["faces_calque"]] += 1
    compte[info["peau_faces"]] += 1
    compte[info.get("peau_calques", np.zeros(0, int))] += 1
    perdues = np.where(tenue & (compte == 0))[0]
    doubles = np.where(compte > 1)[0]
    deux, reste, couches_internes, incrust = [], [], [], []
    for pc in pieces:
        if pc["classe"] != "vetement":
            continue
        z = pc["zones"]
        sl = {s: sum(z.get(x, 0) for x in zs) for s, zs in SLOTS.items()}
        forts = [s for s, v in sl.items() if v >= 0.15]
        for i in range(len(forts)):
            for j in range(i + 1, len(forts)):
                if frozenset((forts[i], forts[j])) not in SLOTS_VOISINS and \
                        {forts[i], forts[j]} != {"haut", "bas"} and pc["type"] not in ("robe", "manteau", "combinaison"):
                    deux.append((pc["id"], "distant slots %s+%s" % (forts[i], forts[j])))
        # two layers in one part: panels (sewn groups of atoms) of >= 10 % that overlap
        ats = pc["atomes"]
        if len(ats) > 1:
            u = UF(T.na)
            sa = set(ats)
            for (a, b), d in T.coutures.items():
                if a in sa and b in sa:
                    u.u(a, b)
            gr = [g for g in u.groupes(ats) if sum(T.nf_atome[a] for a in g) >= 0.1 * len(pc["faces"])]
            if len(gr) > 1:
                fl = [np.concatenate([T.faces_atome[a] for a in g]) for g in gr]
                _, dess = couches(T, [{"faces": f} for f in fl])
                if any(dess):
                    mats_gr = [{T.mat_atome[a] for a in g} for g in gr]
                    un_nom = len(set().union(*mats_gr)) == 1 and not mat_generique(M.noms_mat[next(iter(mats_gr[0]))])
                    if un_nom:
                        # a single named material (下摆, 外套): lapels, flaps, lining of the same garment
                        couches_internes.append((pc["id"], "%d overlapping panels, material %s" % (len(gr), pc["materiaux"][0])))
                    else:
                        deux.append((pc["id"], "two layers in the part (%d panels)" % len(gr)))
        # accessory left inside: small or ring-shaped atom, with a colour very different from its sewn neighbours
        for a in ats:
            g = Geo(T, T.faces_atome[a])
            petit = g.diag < ACC_DIAG or (g.etendue_axe < ANNEAU_HAUT and g.tour >= ANNEAU_TOUR and g.diag < 0.5)
            if not petit:
                continue
            dEs = [d["dE"] for (x, y), d in T.coutures.items() if a in (x, y) and (y if x == a else x) in sa]
            if dEs and min(dEs) > 2 * DE_MEME_TISSU:
                # border fully sewn: inlay (a coloured panel in the fabric, removing it
                # would leave a hole); border partly free: a part laid on and welded, to report
                if part_cousue(T, a) >= INCRUSTATION_MIN:
                    incrust.append((pc["id"], a))
                    continue
                reste.append((pc["id"], "atom %d (%d faces, %.0f mm, ΔE %.0f)" % (a, T.nf_atome[a], g.diag * 1000, min(dEs))))
    return {"faces_tenue": int(tenue.sum()), "faces_pieces": int(sum(len(pc["faces"]) + len(pc["faces_calque"]) for pc in pieces)),
            "faces_peau": int(len(info["peau_faces"]) + len(info.get("peau_calques", []))),
            "faces_perdues": int(len(perdues)), "faces_en_double": int(len(doubles)),
            "deux_vetements": deux, "accessoire_reste": reste, "couches_internes_admises": couches_internes,
            "incrustations_admises": len(incrust),
            "passe": len(perdues) == 0 and len(doubles) == 0 and not deux and not reste}


# ====================================================================== extracting a character

def _json(obj, chemin):
    tmp = chemin + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, chemin)


def zones_peau(T, faces):
    if len(faces) == 0:
        return {}
    z = zones_piece(T, faces)
    return {k: round(v, 3) for k, v in z.items() if v >= 0.01}


# ====================================================================== hard cases (named)
# Characters whose mesh does not separate a garment (e.g. a cardigan welded to the dress): described in the
# game description (personnages.<perso>.exception_vetement: spec = a record of mesure.VETEMENTS, cut by
# its texture area; type; why), with no new record.
EXCEPTIONS = MD.jeux.cas("exception_vetement")


def clarte_uv_oiio(M):
    """Like modele.clarte_uv, with OpenImageIO (outside Blender)."""
    L = np.full(len(M.F), np.nan)
    c = M.uv[M.F].mean(1)
    for mi in np.unique(M.mf):
        ch = chemin_texture(M, mi)
        im = lire_texture(ch) if ch else None
        if im is None:
            continue
        h, w = im.shape[:2]
        fi = np.where(M.mf == mi)[0]
        u = np.clip((c[fi, 0] % 1) * w, 0, w - 1).astype(int)
        v = np.clip((c[fi, 1] % 1) * h, 0, h - 1).astype(int)
        L[fi] = im[v, u, :3] @ [0.299, 0.587, 0.114]
    return L


def appliquer_exception(T, pieces, ex):
    """Splits off from a garment the faces that the measure.py record assigns to it."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("mesure_vet", os.path.join(ICI, "measure.py"))
    mes = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mes)
    spec = next(v for v in mes.VETEMENTS if v["id"] == ex["spec"])
    ancien = MD.clarte_uv
    MD.clarte_uv = clarte_uv_oiio
    try:
        sel = MD.isoler(T.M, spec)[0]
    finally:
        MD.clarte_uv = ancien
    vets = [pc for pc in pieces if pc["classe"] == "vetement"]
    pc = max(vets, key=lambda x: sel[x["faces"]].sum())
    dedans = sel[pc["faces"]]
    if dedans.all() or not dedans.any():
        return pieces
    a, b = pc["faces"][dedans], pc["faces"][~dedans]
    # pieces of the remainder carried by the arms (dark cuffs set aside by the measurement): go to the cardigan
    mb = np.zeros(len(T.M.F), bool)
    mb[b] = True
    for g in T.M.pieces(mb):
        z = zones_piece(T, g)
        if z["bras"] + z["avant_bras"] + z["main"] >= 0.5:
            mb[g] = False
    a = np.sort(np.concatenate([a, b[~mb[b]]]))
    b = b[mb[b]]
    ats_a = [x for x in pc["atomes"] if sel[T.faces_atome[x]].mean() >= 0.5]
    ats_b = [x for x in pc["atomes"] if x not in ats_a]
    pc["faces"], pc["atomes"] = b, ats_b
    pieces.append({"classe": "vetement", "faces": np.sort(a), "atomes": ats_a, "type_force": ex["type"],
                   "exception": ex["pourquoi"]})
    return pieces


def analyser(perso, chemin=None):
    T = Tenue(perso, chemin=chemin)
    pieces, info = decouper(T)
    if perso in EXCEPTIONS and chemin is None:
        pieces = appliquer_exception(T, pieces, EXCEPTIONS[perso])
    pieces = decrire(T, pieces, info)
    return T, pieces, info, controler(T, pieces, info)


def mettre_de_cote(src, dst):
    """Moves a folder (rename); if the file system refuses to rename a folder
    (overlay: EXDEV), moves the files one by one then removes the emptied folders."""
    try:
        os.rename(src, dst)
        return
    except OSError:
        pass
    for racine, dossiers, fichiers in os.walk(src, topdown=False):
        cible = os.path.join(dst, os.path.relpath(racine, src))
        os.makedirs(cible, exist_ok=True)
        for f in fichiers:
            os.rename(os.path.join(racine, f), os.path.join(cible, f))
        os.rmdir(racine)


def extraire_perso(perso, sortie, chemin=None):
    """Cuts a character and writes its parts: <sortie>/<perso>/<id>/<id>.pmx + fiche.json,
    shared textures in <sortie>/<perso>/tex/, full cutting in decoupe.json."""
    ecr = _ecrire_pmx_module()
    T, pieces, info, ctrl = analyser(perso, chemin)
    M = T.M
    base = os.path.join(sortie, perso)
    if os.path.exists(base):
        # previous cutting: moved aside (not erased), so that no stale parts are kept
        corb = os.path.join(sortie, ".corbeille")   # same disk as the output: a plain rename
        os.makedirs(corb, exist_ok=True)
        mettre_de_cote(base, os.path.join(corb, "%s-%d" % (perso, time.time() * 1000)))
    os.makedirs(os.path.join(base, "tex"), exist_ok=True)
    copiees = {}
    fiches = []
    for pc in pieces:
        d = os.path.join(base, pc["id"])
        os.makedirs(d, exist_ok=True)
        faces = np.concatenate([pc["faces"], pc["faces_calque"]]).astype(int)
        pmx, st = piece_pmx(T, faces, "%s %s" % (perso, pc["id"]), "../tex", copiees)
        tmp = os.path.join(d, pc["id"] + ".pmx.tmp")
        ecr.ecrire(pmx, tmp)
        os.replace(tmp, os.path.join(d, pc["id"] + ".pmx"))
        cachee = peau_cachee(T, pc["faces"], info["peau_faces"])
        fi = {"perso": perso, "id": pc["id"], "classe": pc["classe"],
              "type": pc.get("type"), "sorte": pc.get("sorte"),
              "pourquoi": pc.get("raison") or pc.get("exception"), "sorte_pourquoi": pc.get("sorte_raison"),
              "couche": pc["couche"], "recouvre": pc.get("recouvre"), "porteur": pc.get("porteur"),
              "pmx": pc["id"] + ".pmx", "vignette": "vignette.png",
              "faces": int(len(pc["faces"])), "faces_calque": int(len(pc["faces_calque"])),
              "sommets": st["sommets"], "materiaux": pc["materiaux"],
              "os_attache": [[n, w] for n, w in pc["os_corps"][:6]],
              "os_propres": [n for n, _ in pc["os_propres"]],
              "physique": {"os": st["os"], "os_propres": len(pc["os_propres"]), "rigides": st["rigides"],
                           "rigides_physiques": st["os_physiques"], "joints": st["joints"]},
              "zones": pc["zones"],
              "peau_cachee": {"faces": int(len(cachee)), "zones": zones_peau(T, cachee)},
              "cousu_a_la_peau_mm": pc["cousu_peau_mm"],
              "source": {"pmx": M.chemin, "faces": "../decoupe.json"}}
        _json(fi, os.path.join(d, "fiche.json"))
        fiches.append(fi)
    for nom, src in copiees.items():
        _lier(src, os.path.join(base, "tex", nom))
    dec = {"perso": perso, "pmx": M.chemin, "couleur_peau_lab": info["couleur_peau"],
           "controle": ctrl, "calques_sans_jumelle": info.get("calques_sans_jumelle"),
           "peau": {"faces": info["peau_faces"].tolist(), "calques": info["peau_calques"].tolist()},
           "pieces": {pc["id"]: {"faces": pc["faces"].tolist(), "faces_calque": pc["faces_calque"].tolist(),
                                 "atomes": pc["atomes"]} for pc in pieces}}
    _json(dec, os.path.join(base, "decoupe.json"))
    garnir_perso(perso, sortie, T)
    return fiches, ctrl


# trims: accessories that only exist on their garment (buttons, pockets, straps, ornaments).
# They cannot be selected on their own; without them, the garment is worn bare. The other accessories
# (belt, tie, bag…) are selected separately.
GARNITURES = {"sangle", "boucle_bouton", "ornement", "poche", "ecusson", "badge", "chaine", "corde",
              "fermeture", "boucle"}


def morceaux(T, faces):
    """Connected components (welded vertices) of a set of faces -> list of face arrays."""
    F = T.M.Fs[faces]
    u, inv = np.unique(F, return_inverse=True)
    inv = inv.reshape(F.shape)
    par = np.arange(len(u))

    def r(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a_, b_, c_ in inv:
        ra, rb, rc = r(a_), r(b_), r(c_)
        par[rb] = ra
        par[r(rc)] = ra
    rac = np.array([r(x) for x in inv[:, 0]])
    return [np.asarray(faces)[rac == k] for k in np.unique(rac)]


def repartir_garnitures(T, dec, fiches):
    """A trim is often a cluster of scattered pieces grouped by their material (ornaments on
    both cuffs, the back and the waist): the cluster has a single carrier, but each piece belongs to
    what it touches. Returns {carrier id: [(trim id, faces of the piece)]}: each
    piece goes to the selectable part (garment, belt, bag…) its surface touches most
    closely; if it touches none (lying on another trim), it follows the cluster's carrier."""
    cibles = [fi for fi in fiches.values() if fi["classe"] == "vetement" or fi.get("sorte") not in GARNITURES]
    groupes = [np.asarray(dec["pieces"][fi["id"]]["faces"], int) for fi in cibles]
    out = {}
    for fi in fiches.values():
        if fi["classe"] != "accessoire" or fi.get("sorte") not in GARNITURES:
            continue
        po = fiches.get(fi.get("porteur"))
        defaut = po["id"] if po and (po["classe"] == "vetement" or po.get("sorte") not in GARNITURES) else None
        f = np.concatenate([np.asarray(dec["pieces"][fi["id"]][k], int) for k in ("faces", "faces_calque")])
        ms = morceaux(T, f)
        for m in ms:
            cible = defaut
            if len(ms) > 1 and groupes:
                nb = plus_proche_surface(T, m, groupes, dessus=True)
                j = int(nb.argmax())
                if nb[j] >= max(2, PORTE_PART * nb.sum()):
                    cible = cibles[j]["id"]
            if cible is not None:
                out.setdefault(cible, []).append((fi["id"], m))
    return out


def garnir_perso(perso, sortie, T=None):
    """For each already extracted garment that carries trims: writes <id>+garnitures.pmx next to
    its PMX (the garment and its trims as a single object) and records it in its fiche
    ("garnie": pmx, parts, faces, own bones). Reads decoupe.json, cuts nothing again."""
    base = os.path.join(sortie, perso)
    with open(os.path.join(base, "decoupe.json"), encoding="utf-8") as fh:
        dec = json.load(fh)
    fiches = {}
    for pid in dec["pieces"]:
        f = os.path.join(base, pid, "fiche.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                fiches[pid] = json.load(fh)
    garnis = any(fi["classe"] == "accessoire" and fi.get("sorte") in GARNITURES and fi.get("porteur") for fi in fiches.values())
    if garnis:
        T = T or Tenue(perso)
    portees = repartir_garnitures(T, dec, fiches) if garnis else {}
    ecr = _ecrire_pmx_module() if portees else None
    copiees = {}
    n = 0
    for pid, fi in fiches.items():
        d = os.path.join(base, pid)
        g = sorted(portees.get(pid, []), key=lambda x: x[0])
        if not g or (fi["classe"] == "accessoire" and fi.get("sorte") in GARNITURES):
            if fi.pop("garnie", None) is not None:
                _json(fi, os.path.join(d, "fiche.json"))
            continue
        propre = np.concatenate([np.asarray(dec["pieces"][pid][k], int) for k in ("faces", "faces_calque")])
        faces = np.unique(np.concatenate([propre] + [m for _, m in g]))
        pmx, st = piece_pmx(T, faces, "%s %s garni" % (perso, pid), "../tex", copiees)
        nom = pid + "+garnitures.pmx"
        ecr.ecrire(pmx, os.path.join(d, nom + ".tmp"))
        os.replace(os.path.join(d, nom + ".tmp"), os.path.join(d, nom))
        ids = sorted({x for x, _ in g})
        fi["garnie"] = {"pmx": nom, "pieces": ids, "faces": int(len(faces)),
                        "os_propres": sorted(set(fi.get("os_propres") or []).union(*[fiches[x].get("os_propres") or [] for x in ids]))}
        _json(fi, os.path.join(d, "fiche.json"))
        n += 1
    for nom, src in copiees.items():
        _lier(src, os.path.join(base, "tex", nom))
    return n


def persos_biblio():
    return sorted(d for d in os.listdir(MD.BIBLIO)
                  if os.path.exists(os.path.join(MD.BIBLIO, d, "fiche.json")))


def indexer(sortie):
    """index.json per garment type and per accessory kind, plus a summary."""
    tout = []
    for p in sorted(os.listdir(sortie)):
        d = os.path.join(sortie, p)
        if not os.path.exists(os.path.join(d, "decoupe.json")):
            continue
        for pc in sorted(os.listdir(d)):
            f = os.path.join(d, pc, "fiche.json")
            if os.path.exists(f):
                with open(f, encoding="utf-8") as fh:
                    tout.append(json.load(fh))
    os.makedirs(os.path.join(sortie, "index", "vetements"), exist_ok=True)
    os.makedirs(os.path.join(sortie, "index", "accessoires"), exist_ok=True)
    par = {}
    for fi in tout:
        cle = ("vetements", fi["type"]) if fi["classe"] == "vetement" else ("accessoires", fi["sorte"])
        par.setdefault(cle, []).append({"perso": fi["perso"], "id": fi["id"],
                                        "pmx": "../../%s/%s/%s" % (fi["perso"], fi["id"], fi["pmx"]),
                                        "vignette": "../../%s/%s/vignette.png" % (fi["perso"], fi["id"]),
                                        "faces": fi["faces"], "couche": fi["couche"],
                                        "porteur": fi.get("porteur"),
                                        "physique": fi["physique"]["rigides_physiques"] > 0})
    for (c, t), l in par.items():
        _json(l, os.path.join(sortie, "index", c, "%s.json" % t))
    bilan = {"persos": len({fi["perso"] for fi in tout}), "pieces": len(tout),
             "vetements": sum(1 for fi in tout if fi["classe"] == "vetement"),
             "accessoires": sum(1 for fi in tout if fi["classe"] == "accessoire"),
             "vetements_par_type": {t: len(l) for (c, t), l in sorted(par.items()) if c == "vetements"},
             "accessoires_par_sorte": {t: len(l) for (c, t), l in sorted(par.items()) if c == "accessoires"}}
    _json(bilan, os.path.join(sortie, "index.json"))
    return bilan


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
    w = bpy.data.worlds.new("fond") if not bpy.data.worlds else bpy.data.worlds[0]
    sc.world = w
    w.color = (0.93, 0.93, 0.93)
    sh.background_type = "WORLD"
    sc.display.render_aa = "8"
    return sc


def _rendre(lo, hi, angle, chemin, tx, ty, texte=None):
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
    marge = (1.12 + 0.12 * texte.count("\n")) if texte else 1.06
    if ty >= tx:
        cam_d.sensor_fit = "VERTICAL"
        cam_d.ortho_scale = marge * max(h, l * ty / tx)
        haut = cam_d.ortho_scale
    else:
        cam_d.sensor_fit = "HORIZONTAL"
        cam_d.ortho_scale = marge * max(l, h * tx / ty)
        haut = cam_d.ortho_scale * ty / tx
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    a = math.radians(angle)
    d = 5.0
    cz = c[2] - ((0.05 + 0.06 * texte.count("\n")) * haut if texte else 0)
    cam.location = Vector((c[0] + d * math.sin(a), c[1] - d * math.cos(a), cz))
    cam.rotation_euler = (math.pi / 2, 0, a)
    cam_d.clip_end = 20
    sc.camera = cam
    to = None
    if texte:
        cu = bpy.data.curves.new("txt", "FONT")
        cu.body = texte
        cu.size = (0.055 if ty > 300 else 0.085) * haut
        cu.align_x = "CENTER"
        to = bpy.data.objects.new("txt", cu)
        m = bpy.data.materials.new("txt")
        m.diffuse_color = (0.05, 0.05, 0.05, 1)
        cu.materials.append(m)
        sc.collection.objects.link(to)
        av = 1.0
        nl = texte.count("\n")
        to.location = Vector((c[0] + av * math.sin(a), c[1] - av * math.cos(a),
                              cz - 0.47 * haut + (nl + 0.2) * cu.size * 1.15))
        to.rotation_euler = (math.pi / 2, 0, a)
    sc.render.filepath = chemin
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)
    if to is not None:
        bpy.data.objects.remove(to, do_unlink=True)


def _coller(lignes, chemin, fond=(1, 1, 1, 1)):
    """Assembles rows of PNGs (any size) into one contact sheet."""
    import bpy
    ims = []
    for ligne in lignes:
        r = []
        for f in ligne:
            im = bpy.data.images.load(f)
            px = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)
            r.append(px)
            bpy.data.images.remove(im)
        ims.append(r)
    W = max(sum(p.shape[1] for p in r) for r in ims)
    H = sum(max(p.shape[0] for p in r) for r in ims) + 2 * (len(ims) - 1)
    out = np.ones((H, W, 4), np.float32) * np.array(fond, np.float32)
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
    img = bpy.data.images.new("planche", W, H, alpha=True)
    img.pixels.foreach_set(out.ravel())
    img.filepath_raw = chemin
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def planche_perso(sortie, perso, tmp):
    """Thumbnail of each part (<part>/vignette.png, from the front) and the character's contact sheet
    (<perso>/planche.png): whole outfit, garments only, accessories only, each garment,
    from the front then from the back; then each accessory from the front."""
    import bpy
    import contact_sheet as PL
    PL.vider()
    _bpy_scene()
    base = os.path.join(sortie, perso)
    fiches = []
    for pc in sorted(os.listdir(base)):
        f = os.path.join(base, pc, "fiche.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                fiches.append(json.load(fh))
    fiches.sort(key=lambda fi: (fi["classe"] != "vetement", -fi["faces"]))
    cache, objs, boites = {}, {}, {}
    for fi in fiches:
        M = MD.Modele(fi["id"], chemin=os.path.join(base, fi["id"], fi["pmx"]))
        o = PL.objet(M, np.arange(len(M.F)), fi["id"], cache)
        objs[fi["id"]] = o
        boites[fi["id"]] = (M.pos.min(0), M.pos.max(0))

    def montrer(ids):
        for k, o in objs.items():
            o.hide_render = k not in ids
        lo = np.min([boites[k][0] for k in ids], 0)
        hi = np.max([boites[k][1] for k in ids], 0)
        return lo, hi
    vet = [fi["id"] for fi in fiches if fi["classe"] == "vetement"]
    acc = [fi["id"] for fi in fiches if fi["classe"] == "accessoire"]
    TXg, TYg = 220, 380
    lignes = []
    groupes = [("tenue", vet + acc), ("vetements", vet), ("accessoires", acc)] + [(v, [v]) for v in vet]
    for ang, vue in ((0, "face"), (180, "dos")):
        ligne = []
        for nom, ids in groupes:
            if not ids:
                continue
            fi = next((x for x in fiches if x["id"] == nom), None)
            lab = "%s\n%s c%d" % (nom, fi["type"], fi["couche"]) if fi else "%s (%s)" % (nom, vue)
            f = os.path.join(tmp, "pl_%s_%s_%s.png" % (perso, nom, vue))
            lo, hi = montrer(ids)
            _rendre(lo, hi, ang, f, TXg, TYg, lab)
            ligne.append(f)
        lignes.append(ligne)
    # thumbnails (all parts, from the front, no text); accessory tiles with text
    ligne = []
    for fi in fiches:
        lo, hi = montrer([fi["id"]])
        _rendre(lo, hi, 0, os.path.join(base, fi["id"], "vignette.png"), 256, 256)
        if fi["classe"] == "accessoire":
            f = os.path.join(tmp, "pl_%s_%s_acc.png" % (perso, fi["id"]))
            _rendre(lo, hi, 0, f, 165, 165, "%s\n→ %s" % (fi["id"], fi.get("porteur") or "body"))
            ligne.append(f)
            if len(ligne) == 8:
                lignes.append(ligne)
                ligne = []
    if ligne:
        lignes.append(ligne)
    _coller(lignes, os.path.join(base, "planche.png"))
    journal("contact sheet", perso, len(fiches), "parts")


# ====================================================================== command line

def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="library.py")
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("extract", help="cut and write the parts (plain Python)")
    a.add_argument("--output", dest="sortie", required=True)
    a.add_argument("--only", dest="seuls", help="comma-separated characters (default: the whole face library)")
    a = sp.add_parser("trim", help="garments with their trims, from the cutting already written (plain Python)")
    a.add_argument("--output", dest="sortie", required=True)
    a.add_argument("--only", dest="seuls")
    a = sp.add_parser("sheets", help="thumbnails and contact sheets (in Blender)")
    a.add_argument("--output", dest="sortie", required=True)
    a.add_argument("--only", dest="seuls")
    a = sp.add_parser("index", help="index by type and kind")
    a.add_argument("--output", dest="sortie", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "extract":
        persos = args.seuls.split(",") if args.seuls else persos_biblio()
        bil = {}
        for p in persos:
            t = time.time()
            try:
                fi, c = extraire_perso(p, args.sortie)
            except Exception as e:  # a failing character does not stop the others; it is named
                import traceback
                traceback.print_exc()
                bil[p] = {"erreur": repr(e)}
                continue
            bil[p] = {"pieces": len(fi), "vetements": sum(1 for x in fi if x["classe"] == "vetement"),
                      "controle": {k: v for k, v in c.items()}}
            journal(p, len(fi), "parts", "check", "passed" if c["passe"] else "to review", "%.0fs" % (time.time() - t))
        prec = {}
        f = os.path.join(args.sortie, "controle.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                prec = json.load(fh)
        prec.update(bil)
        _json(prec, f)
        print(json.dumps(indexer(args.sortie), ensure_ascii=False, indent=1))
    elif args.cmd == "trim":
        persos = args.seuls.split(",") if args.seuls else sorted(
            d for d in os.listdir(args.sortie) if os.path.exists(os.path.join(args.sortie, d, "decoupe.json")))
        for p in persos:
            journal(p, garnir_perso(p, args.sortie), "garments with trims")
    elif args.cmd == "sheets":
        tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "rendus-biblio")
        os.makedirs(tmp, exist_ok=True)
        persos = args.seuls.split(",") if args.seuls else sorted(
            d for d in os.listdir(args.sortie) if os.path.exists(os.path.join(args.sortie, d, "decoupe.json")))
        for p in persos:
            planche_perso(args.sortie, p, tmp)
    elif args.cmd == "index":
        print(json.dumps(indexer(args.sortie), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])

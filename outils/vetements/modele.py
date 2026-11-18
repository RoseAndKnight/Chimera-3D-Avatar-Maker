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

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
ATLAS = os.path.join(ICI, "..", "atlas")
if ATLAS not in sys.path:
    sys.path.insert(0, ATLAS)
sys.path.append(os.path.join(ICI, ".."))
import reglages as R  # noqa: E402
import jeux  # noqa: E402

import pmx as PMX  # noqa: E402
import primitives as P  # noqa: E402

ECHELLE = 0.08
BIBLIO = os.path.join(R.chemin("chemins.donnees"), "bibliotheque-visage")

# Materials that are never clothing: face, eyes, hair (first character or full name)
NON_VETEMENT = re.compile(r"^(颜|口|舌|齿|目|白目|眉|睫|二重|瞳|痣|表情|眼角|髮|髪|发|前发|后发|前髮|后髮|前髪|后髪|侧发)")
PEAU = re.compile(r"^(肌|手$|手2$|脖$|指甲)")
# Accessory materials (surveyed on the library's 50 PMX files, inventory of 2026-11-07):
# ornaments 饰, metal 金属, weapons, glasses, jewellery, screens, wings, tail, plush toys, bags, tools…
ACCESSOIRE_MAT = re.compile(r"饰|金属|武器|刀|墨镜|镜|宝石|珠宝|屏幕|翅膀|翼|尾|熊|狸猫|邦布|电池|耳机|包|工具|手雷|"
                            r"面具|面罩|眼罩|光环|绳|锁链|拉链|耳|戒指|指甲|頭巾|头饰|发饰|髮飾|髮饰|带$|^带|挂|"
                            r"Weapon|Sticker|透明罩|新規材質|帽|角|翅")
# Accessory bones (in addition to the "accessoire" family of primitives.FAMILLES): any bone that carries an object
ACCESSOIRE_OS = re.compile(r"坠|珠|刀|扣|结|穗|武器|鞘|錘|铃|挂|链|饰|包|环$|铃|耳机|电池|领带|带\d|尾")


def vers_blender(p):
    p = np.asarray(p, dtype=np.float64)
    return p[..., [0, 2, 1]] * ECHELLE


SOURCES = jeux.actif().chemin("mmd_extrait")   # official PMX files of the active game (alternate outfits outside the library)


def modele_de(perso, pmx=None):
    """Model of a library character, or of an official PMX outside the library (alternate
    outfits): `pmx` relative to SOURCES, or an absolute path."""
    return Modele(perso, chemin=os.path.join(SOURCES, pmx) if pmx else None)


def chemin_pmx(perso, biblio=BIBLIO):
    with open(os.path.join(biblio, perso, "fiche.json"), encoding="utf-8") as fh:
        return json.load(fh)["pmx"]


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def trouve(self, i):
        p = self.p
        r = i
        while p[r] != r:
            r = p[r]
        while p[i] != r:
            p[i], i = r, p[i]
        return r

    def unir(self, a, b):
        ra, rb = self.trouve(a), self.trouve(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def etiquettes(self):
        return np.array([self.trouve(i) for i in range(len(self.p))])


def composantes(n, aretes):
    """Connected-component label of each vertex (vectorised union-find by propagation)."""
    lab = np.arange(n)
    a, b = aretes[:, 0], aretes[:, 1]
    while True:
        m = np.minimum(lab[a], lab[b])
        nouveau = lab.copy()
        np.minimum.at(nouveau, a, m)
        np.minimum.at(nouveau, b, m)
        nouveau = nouveau[nouveau]
        if np.array_equal(nouveau, lab):
            return lab
        lab = nouveau


def aretes_faces(F):
    """Unique (sorted) edges, face count per edge, edge of each corner (f, 3)."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, inv, c = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    return u, c, inv.ravel().reshape(3, -1).T


def normales_faces(pos, F):
    n = np.cross(pos[F[:, 1]] - pos[F[:, 0]], pos[F[:, 2]] - pos[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-20)[:, None], a / 2


class Modele:
    """An official PMX in the Blender frame (metres)."""

    def __init__(self, perso, biblio=BIBLIO, chemin=None):
        self.perso = perso
        self.chemin = chemin or chemin_pmx(perso, biblio)
        self.p = p = PMX.lire(self.chemin)
        t = PMX.vers_tableaux(p)
        self.pos = vers_blender(t["pos"])
        self.nor = t["nor"][:, [0, 2, 1]]
        self.uv = t["uv"]
        self.uv_add = t["uv_add"]
        self.type_poids = t["type_poids"]
        self.idx = t["os"]
        self.poids = t["poids"]
        self.contour = t["contour"]
        self.F = t["faces"][:, [0, 2, 1]]
        self.mf = t["mat_face"]
        self.mats = p["materiaux"]
        self.noms_mat = [m["nom"] for m in p["materiaux"]]
        self.textures = p["textures"]
        oss = p["os"]
        self.noms_os = [o["nom"] for o in oss]
        self.par = np.array([o["parent"] for o in oss])
        tetes = vers_blender([o["pos"] for o in oss])
        queues = np.zeros_like(tetes)
        for i, o in enumerate(oss):
            if o["queue_os"] is not None:
                q = o["queue_os"]
                queues[i] = tetes[q] if 0 <= q < len(oss) else tetes[i]
            else:
                queues[i] = tetes[i] + vers_blender(o["queue_dec"])
        self.tetes, self.queues = tetes, queues
        src = np.array([o["ajout"]["source"] if o["ajout"] else -1 for o in oss])
        fac = np.array([o["ajout"]["facteur"] if o["ajout"] else 0.0 for o in oss])
        arot = np.array([bool(o["ajout"] and o["ajout"]["rotation"]) for o in oss])
        adep = np.array([bool(o["ajout"] and o["ajout"]["deplacement"]) for o in oss])
        self.sq = P.Squelette(self.noms_os, tetes, queues, self.par, src, fac, arot, adep)
        fam_man = P.charger_familles().get("os", {})
        self.famille_os = [P.famille_os(n, fam_man) for n in self.noms_os]
        self.region_os = [P.region_os(n, f) for n, f in zip(self.noms_os, self.famille_os)]
        self.rigides = p["rigides"]
        self.joints = p["joints"]
        # welding: coincident vertices (UV seams, split normals)
        cle = np.round(self.pos / 1e-6).astype(np.int64)
        _, self.sv, inv = np.unique(cle, axis=0, return_index=True, return_inverse=True)
        self.s = inv.ravel()                       # raw vertex -> welded vertex
        self.pos_s = self.pos[self.sv]
        self.Fs = self.s[self.F]                   # welded faces
        self.nf, self.aire = normales_faces(self.pos, self.F)
        self.dom = P.os_dominant(self.idx, self.poids)
        self.H_corps = float(self.pos[:, 2].max() - self.pos[:, 2].min())

    # ------------------------------------------------------------------ material roles

    def role_materiau(self, i):
        n = self.noms_mat[i]
        if self.mats[i]["n_indices"] == 0:
            return "vide"
        if NON_VETEMENT.match(n):
            return "tete"
        if PEAU.match(n):
            return "peau"
        if n.endswith("+") or "+" in n:
            return "calque"
        if ACCESSOIRE_MAT.search(n):
            return "accessoire"
        return "tenue"

    def faces_role(self, roles):
        r = np.array([self.role_materiau(i) for i in range(len(self.mats))])
        return np.isin(r[self.mf], roles)

    def os_accessoire(self, b):
        if b < 0:
            return False
        n = self.noms_os[b]
        return self.famille_os[b] == "accessoire" or bool(ACCESSOIRE_OS.search(n))

    def region_ancetre(self, b):
        """Body region of a bone; for a clothing or accessory bone, that of the first
        ancestor that belongs to the body."""
        corps = ("tronc_haut", "poitrine", "ventre", "bassin", "fesses", "cou", "tete", "bras",
                 "avant_bras", "main", "doigt", "cuisse", "jambe", "pied")
        k = b
        while k >= 0:
            if self.region_os[k] in corps:
                return self.region_os[k]
            k = self.par[k]
        return "sans"

    # ------------------------------------------------------------------ parts

    def pieces(self, masque_faces):
        """Welded parts of a set of faces: list of arrays of face indices."""
        fi = np.where(masque_faces)[0]
        if not len(fi):
            return []
        Fs = self.Fs[fi]
        E = np.concatenate([Fs[:, [0, 1]], Fs[:, [1, 2]]])
        lab = composantes(len(self.pos_s), E)
        lf = lab[Fs[:, 0]]
        o = np.argsort(lf, kind="stable")
        u, debut = np.unique(lf[o], return_index=True)
        groupes = np.split(fi[o], debut[1:])
        return sorted(groupes, key=lambda g: -len(g))

    def decrire_piece(self, fk):
        """Size, closedness, dominant bone and region of a part."""
        Fs = self.Fs[fk]
        E, c, _ = aretes_faces(Fs)
        vs = np.unique(Fs)
        p = self.pos_s[vs]
        vb = np.unique(self.F[fk])
        dom = self.dom[vb]
        u, n = np.unique(dom[dom >= 0], return_counts=True)
        o = np.argsort(-n)
        os_dom = [(self.noms_os[u[k]], round(float(n[k] / n.sum()), 2)) for k in o[:3]]
        acc = np.array([self.os_accessoire(b) for b in dom])
        return {"faces": int(len(fk)), "sommets": int(len(vs)),
                "aretes_bord": int((c == 1).sum()), "aretes": int(len(E)),
                "fermee": bool((c == 1).sum() == 0),
                "boite_mm": np.round((p.max(0) - p.min(0)) * 1000, 1).tolist(),
                "centre_mm": np.round(p.mean(0) * 1000, 1).tolist(),
                "diag_mm": round(float(np.linalg.norm(p.max(0) - p.min(0)) * 1000), 1),
                "os_dominants": os_dom,
                "part_os_accessoire": round(float(acc.mean()), 3),
                "region": self.region_ancetre(int(u[o[0]])) if len(u) else "sans",
                "materiaux": sorted({self.noms_mat[i] for i in np.unique(self.mf[fk])})}


# ====================================================================== isolation

def dans_uv(M, boites, exclure=None):
    """Faces whose UV centre falls inside one of the boxes [[umin,vmin],[umax,vmax]],
    and outside the "exclure" boxes."""
    c = M.uv[M.F].mean(1)
    def dans(bs):
        m = np.zeros(len(M.F), bool)
        for lo, hi in bs or []:
            m |= ((c >= np.asarray(lo)) & (c <= np.asarray(hi))).all(1)
        return m
    return dans(boites) & ~dans(exclure)


def clarte_uv(M):
    """Texture lightness (0-1) at the UV centre of each face; NaN when no readable texture.
    Reads the images through bpy (Blender's Python only)."""
    import bpy
    L = np.full(len(M.F), np.nan)
    c = M.uv[M.F].mean(1)
    for mi in np.unique(M.mf):
        t = M.mats[mi]["texture"]
        if not 0 <= t < len(M.textures):
            continue
        f = os.path.normpath(os.path.join(os.path.dirname(M.chemin), M.textures[t].replace("\\", "/")))
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
    """Faces kept by the record's UV cutting (uv_boites, uv_exclure, uv_clarte_min)."""
    m = dans_uv(M, spec["uv_boites"], spec.get("uv_exclure"))
    if spec.get("uv_clarte_min") is not None:
        m &= ~(clarte_uv(M) < spec["uv_clarte_min"])
    return m


def isoler(M, spec):
    """Faces of the garment described by spec, and parts set aside as accessories (with the reason).

    spec: {"mats": [names], "regions": [kept regions] (optional), "exclure_os": pattern of
    dominant bones to set aside (optional), "garder_os": pattern of required dominant bones (optional),
    "boite": [[xmin,ymin,zmin],[xmax,ymax,zmax]] in m, tested on the part's centre (optional),
    "exclure_pieces": [part indices] (optional), "garder_pieces": [indices]: only
    these (optional), "uv_boites": [[[umin,vmin],[umax,vmax]], …] (and "uv_exclure", same boxes, removed;
    "uv_clarte_min": removes faces whose texture is darker at the UV centre):
    only faces whose UV centre falls in one of them (optional; when a single material carries the whole
    welded outfit of one part, the garment's texture area cuts it out)}.

    A part is an accessory (detected without manual input) if:
      bone   : more than half of its vertices have an accessory dominant bone (ACCESSOIRE_OS);
      button : closed part with a diagonal under ACC_DIAG_MM (button, rivet, buckle);
      small  : fewer than ACC_PART of the garment's faces and a diagonal under ACC_DIAG_MM.
    """
    mats = [i for i, n in enumerate(M.noms_mat) if n in spec["mats"]]
    manque = set(spec["mats"]) - {M.noms_mat[i] for i in mats}
    if manque:
        raise KeyError("materials missing from %s: %s" % (M.perso, sorted(manque)))
    masque = np.isin(M.mf, mats)
    if spec.get("uv_boites"):
        masque &= zone_uv(M, spec)
    pieces = M.pieces(masque)
    total = sum(len(g) for g in pieces)
    gardees, ecartees = [], []
    rx_ex = re.compile(spec["exclure_os"]) if spec.get("exclure_os") else None
    rx_ga = re.compile(spec["garder_os"]) if spec.get("garder_os") else None
    for k, fk in enumerate(pieces):
        d = M.decrire_piece(fk)
        d["indice"] = k
        raison = None
        noms_dom = [n for n, _ in d["os_dominants"]]
        if k in spec.get("exclure_pieces", []):
            raison = "set aside manually (spec)"
        elif "garder_pieces" in spec and k not in spec["garder_pieces"]:
            raison = "not among the kept parts (spec)"
        elif spec.get("regions") and d["region"] not in spec["regions"]:
            raison = "outside the regions (%s)" % d["region"]
        elif rx_ga and not (noms_dom and rx_ga.search(noms_dom[0])):
            raison = "dominant bone not part of the garment (%s)" % (noms_dom[0] if noms_dom else "-")
        elif rx_ex and noms_dom and rx_ex.search(noms_dom[0]):
            raison = "excluded dominant bone (%s)" % noms_dom[0]
        elif spec.get("boite"):
            lo, hi = np.asarray(spec["boite"][0]), np.asarray(spec["boite"][1])
            c = np.asarray(d["centre_mm"]) / 1000
            if not ((c >= lo).all() and (c <= hi).all()):
                raison = "outside the box"
        if raison is None:
            if d["part_os_accessoire"] > 0.5:
                raison = "accessory: accessory bone (%s)" % noms_dom[0]
            elif d["fermee"] and d["diag_mm"] < ACC_DIAG_MM:
                raison = "accessory: small closed part (button, rivet, buckle)"
            elif d["faces"] < ACC_PART * total and d["diag_mm"] < ACC_DIAG_MM:
                raison = "accessory: small part"
        d["raison"] = raison
        (ecartees if raison else gardees).append((fk, d))
    sel = np.zeros(len(M.F), bool)
    for fk, _ in gardees:
        sel[fk] = True
    return sel, [d for _, d in gardees], [d for _, d in ecartees]


ACC_DIAG_MM = 60.0   # above this, a part is no longer a button or a buckle
ACC_PART = 0.02      # share of the garment's faces below which a small part is an accessory

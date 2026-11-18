"""Measures how ZZZ garments (official PMX files) are built, modelled on
outils/cheveux/mesure.py. Accessories excluded (pockets, buttons, straps, buckles...).

    bin/blender-file -b --factory-startup \
        --python outils/vetements/mesure.py -- [--sortie DIR] [--seuls id1,id2] [--sans-poses]
    ... -- --pmx CANDIDATE.pmx --mats A,B --type haut_moulant|haut_ample|veste_ouverte|... [--id NAME]
        [--calques X] --sortie DIR: measures a manufactured garment and prints the checks.

Writes DIR/<id>.json per garment (with "controles": guardrails and contracts,
see controler), DIR/ensemble.json and DIR/tableau.md.
Default DIR: <data>/vetements (chemins.donnees in the settings), otherwise $TMPDIR/vetements.

Blender is used for the BVHs (mathutils) and for reading textures (bpy). The PMX files stay
read-only. Frame and units: see modele.py (metres; outputs are in mm).

Each measurement block states what it changes for manufacturing (docstring "-> manufacturing").
"""
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import modele as MD  # noqa: E402
from modele import P, PMX  # noqa: E402
import banc as BANC  # noqa: E402  (outils/atlas, put on the path by modele)

_spec = importlib.util.spec_from_file_location("mesure_cheveux", os.path.join(ICI, "..", "cheveux", "mesure.py"))
MC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MC)
r3, q, ecrire_json, journal = MC.r3, MC.q, MC.ecrire_json, MC.journal
segment_rigide, dist_segments, collisionnent, ignores = MC.segment_rigide, MC.dist_segments, MC.collisionnent, MC.ignores

try:
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
except ImportError:  # outside Blender: nothing geometric
    bpy = None

# ====================================================================== the garments studied
# Chosen in the active game (description outils/jeux/<jeu>/jeu.json, "etudes.vetements"): one garment per requested
# type, on different characters when possible, contrasting constructions (single sheet, closed shell,
# double-sided, garment = body), then real sweaters; the description keeps the reason for each.
VETEMENTS = MD.jeux.actif().valeur("etudes.vetements", [])   # a game with no study: no garment chosen

RAYON_DESSOUS = 0.15     # m: beyond this, "nothing underneath"
TRAVERSE_MAX = 0.010     # m: an outgoing ray hitting the back of a surface closer than this = clipping
GLISSE_MAX = 0.005       # m: sliding along the surface beyond which the plane of the tracked face no longer holds
TRAVERSE_SEUIL = 0.0005  # m: depth below which a tracked pair is counted as clipping in a pose
CLASSES_ECART = [(0.0, 0.001, "colle_lt1mm"), (0.001, 0.003, "1_3mm"), (0.003, 0.010, "3_10mm"),
                 (0.010, 0.030, "10_30mm"), (0.030, RAYON_DESSOUS, "30mm_plus")]
CORPS_REG = ("tronc_haut", "poitrine", "ventre", "bassin", "fesses", "cou", "tete", "bras",
             "avant_bras", "main", "doigt", "cuisse", "jambe", "pied")
ARTICULATIONS = [("epaule", "左腕", "左肩"), ("coude", "左ひじ", "左腕"), ("hanche", "左足D", "下半身"),
                 ("genou", "左ひざD", "左足D")]
RIG_DYN = (1, 2)


def mm(x):
    return None if x is None else r3(np.asarray(x) * 1000, 1)


def qmm(v, ps=(10, 50, 90)):
    v = np.asarray(v, float)
    return q(v * 1000, ps, 2)


def part(m, poids=None):
    m = np.asarray(m, bool)
    if not m.size:
        return None
    if poids is None:
        return r3(m.mean())
    return r3((np.asarray(poids) * m).sum() / max(np.asarray(poids).sum(), 1e-20))


# ====================================================================== BVH

def bvh(pos, F):
    return BVHTree.FromPolygons([Vector(p) for p in pos], [tuple(int(x) for x in f) for f in F], all_triangles=True)


def rayons(arbre, O, D, dmax):
    """First hit: (face index or -1, distance, normal of the face hit)."""
    n = len(O)
    idx, dist, nor = np.full(n, -1), np.full(n, np.inf), np.zeros((n, 3))
    for i in range(n):
        h = arbre.ray_cast(Vector(O[i]), Vector(D[i]), dmax)
        if h[0] is not None:
            idx[i], dist[i], nor[i] = h[2], h[3], h[1]
    return idx, dist, nor


def barycentres(p, A, B, C):
    v0, v1, v2 = B - A, C - A, p - A
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - v - w, v, w], 1)


# ====================================================================== context of a garment

class Vetement:
    def __init__(self, M, spec):
        self.M, self.spec = M, spec
        self.sel, self.gardees, self.ecartees = MD.isoler(M, spec)
        cal = np.zeros(len(M.F), bool)
        if spec.get("calques"):
            sc = dict(spec, mats=spec["calques"])
            cal, _, _ = MD.isoler(M, sc)
        self.calque = cal
        self.fg = np.where(self.sel)[0]
        self.Fg = M.F[self.fg]
        self.Fgs = M.Fs[self.fg]
        self.vs = np.unique(self.Fgs)                 # welded vertices of the garment
        self.vr = np.unique(self.Fg)                  # raw vertices
        loc = np.full(len(M.pos_s), -1)
        loc[self.vs] = np.arange(len(self.vs))
        self.loc = loc
        self.Fl = loc[self.Fgs]                       # faces as local welded indices
        self.P = M.pos_s[self.vs]
        self.nf = M.nf[self.fg]
        self.aire = M.aire[self.fg]
        # geometric normals per welded vertex (area-weighted)
        ng = np.zeros_like(self.P)
        for k in range(3):
            np.add.at(ng, self.Fl[:, k], self.nf * self.aire[:, None])
        self.ng = ng / np.maximum(np.linalg.norm(ng, axis=1), 1e-20)[:, None]
        # body bones and "outward" direction: from the nearest body bone
        corps = np.array([(M.region_os[b] in CORPS_REG or M.famille_os[b] == "racine") for b in range(len(M.noms_os))])
        self.os_corps = corps
        seg = [(b, M.tetes[b], M.queues[b]) for b in np.where(corps)[0]
               if np.linalg.norm(M.queues[b] - M.tetes[b]) > 0.01 and M.region_os[b] not in ("doigt", "tete")]
        best = np.full(len(self.P), np.inf)
        self.dehors = np.zeros_like(self.P)
        self.os_proche = np.full(len(self.P), -1)
        for b, a, c in seg:
            ab = c - a
            t = np.clip(((self.P - a) @ ab) / (ab @ ab), 0, 1)
            pr = a + t[:, None] * ab
            d = np.linalg.norm(self.P - pr, axis=1)
            m = d < best
            best[m], self.dehors[m], self.os_proche[m] = d[m], (self.P - pr)[m], b
        self.dehors /= np.maximum(np.linalg.norm(self.dehors, axis=1), 1e-20)[:, None]
        self.feuille_ext = (self.ng * self.dehors).sum(1) > 0
        # "underneath": everything else, except empty, head (hair), accessories, overlay layers and the
        # other parts made of the same materials (pockets placed on top)
        roles = np.array([M.role_materiau(i) for i in range(len(M.mats))])
        memes = np.isin(M.mf, [i for i, n in enumerate(M.noms_mat) if n in spec["mats"] + spec.get("calques", [])])
        if spec.get("uv_boites"):  # material carrying the whole outfit: only the cut-out garment counts
            memes &= MD.zone_uv(M, spec)
        self.fd = np.where(np.isin(roles[M.mf], ["peau", "tenue"]) & ~self.sel & ~memes)[0]
        self.peau = np.isin(roles[M.mf], ["peau"])
        # layers lying over the garment (jacket over a t-shirt): removed from the underneath.
        # Vote per welded part: a +n ray that sees its back within 30 mm says "over",
        # a -n ray that sees its front says "under".
        self.couches_dessus = []
        if len(self.fd):
            pcs = M.pieces(np.isin(np.arange(len(M.F)), self.fd))
            piece_de = np.full(len(M.F), -1)
            for k, g in enumerate(pcs):
                piece_de[g] = k
            arb0 = bvh(M.pos, M.F[self.fd])
            ext = self.feuille_ext
            iu, _, nu = rayons(arb0, self.P[ext] + self.ng[ext] * 1e-5, self.ng[ext], 0.03)
            idn, _, nd_ = rayons(arb0, self.P[ext] + self.ng[ext] * 1e-4, -self.ng[ext], 0.03)
            dessus = np.zeros(len(pcs))
            dessous = np.zeros(len(pcs))
            ngx = self.ng[ext]
            for i in np.where((iu >= 0) & ((nu * ngx).sum(1) > 0))[0]:
                dessus[piece_de[self.fd[iu[i]]]] += 1
            for i in np.where((idn >= 0) & ((nd_ * ngx).sum(1) > 0))[0]:
                dessous[piece_de[self.fd[idn[i]]]] += 1
            haut = np.where((dessus > dessous) & (dessus >= 5))[0]
            for k in haut:
                self.couches_dessus.append({"materiaux": sorted({M.noms_mat[i] for i in np.unique(M.mf[pcs[k]])}),
                                            "faces": int(len(pcs[k])), "votes_dessus": int(dessus[k]),
                                            "votes_dessous": int(dessous[k])})
            if len(haut):
                retire = np.concatenate([pcs[k] for k in haut])
                self.fd = np.setdiff1d(self.fd, retire)
        # region of each welded vertex
        dom = M.dom[M.sv[self.vs]]
        self.region = np.array([M.region_ancetre(int(b)) if b >= 0 else "sans" for b in dom])
        self.arbre = bvh(self.P, self.Fl)
        if len(self.fd):
            self.arbre_d = bvh(M.pos, M.F[self.fd])
        else:
            self.arbre_d = None
        # welded edges of the garment
        E, c, coin = MD.aretes_faces(self.Fl)
        self.E, self.Ec, self.coin = E, c, coin
        adj = [[] for _ in range(len(self.P))]
        for a, b in E:
            adj[a].append(b)
            adj[b].append(a)
        self.adj = adj

    def anneaux(self, sources, n):
        """Distance in edges (BFS) from the source vertices, up to n."""
        d = np.full(len(self.P), 99)
        d[sources] = 0
        front = list(sources)
        for k in range(1, n + 1):
            nv = []
            for a in front:
                for b in self.adj[a]:
                    if d[b] > k:
                        d[b] = k
                        nv.append(b)
            front = nv
        return d


# ====================================================================== A. mesh

def mesurer_maillage(V):
    """-> manufacturing: kind of sheet to model (single, thick, shell), edge finish,
    face budget and density relative to the skin, folds to model or to paint, normals,
    outline."""
    M = V.M
    E, c = V.E, V.Ec
    bord = c == 1
    # thickness: -n ray from each face, hit on the garment with an opposite normal
    cen = V.P[V.Fl].mean(1)
    n = V.nf
    O = cen - n * 1e-5
    idx, dist, nor = rayons(V.arbre, O, -n, 0.02)
    paire = (idx >= 0) & ((nor * n).sum(1) < -0.5)
    # vertices near an edge (3 rings)
    vb = np.unique(E[bord])
    an = V.anneaux(vb, 3) if len(vb) else np.full(len(V.P), 99)
    f_pres_bord = (an[V.Fl] <= 3).any(1)
    # parts
    pcs = [d["faces"] for d in V.gardees]
    fermees = sum(1 for d in V.gardees if d["fermee"])
    # edge lengths: garment, skin, underneath
    L = np.linalg.norm(V.P[E[:, 0]] - V.P[E[:, 1]], axis=1)

    def longueurs(fi):
        if not len(fi):
            return None
        Ed, _, _ = MD.aretes_faces(M.Fs[fi])
        return np.linalg.norm(M.pos_s[Ed[:, 0]] - M.pos_s[Ed[:, 1]], axis=1)
    Lp = longueurs(np.where(M.peau_faces)[0]) if hasattr(M, "peau_faces") else longueurs(np.where(V.peau)[0])
    Ld = longueurs(V.fd)
    aire = V.aire.sum()
    # geometric folds: offset from a smoothed surface (Taubin, edges fixed), along the normal
    lisse = taubin(V, 30)
    dev = ((V.P - lisse) * V.ng).sum(1)
    regions = {}
    for r in np.unique(V.region):
        m = V.region == r
        if m.sum() >= 30:
            regions[r] = {"sommets": int(m.sum()), "abs_p90_mm": r3(np.percentile(np.abs(dev[m]), 90) * 1000, 2),
                          "part_gt2mm": r3((np.abs(dev[m]) > 0.002).mean())}
    # interior dihedral angles
    f2 = np.where(c == 2)[0]
    # faces of each interior edge
    fa = np.full((len(E), 2), -1)
    cnt = np.zeros(len(E), int)
    for k in range(3):
        for fi, e in enumerate(V.coin[:, k]):
            if cnt[e] < 2:
                fa[e, cnt[e]] = fi
            cnt[e] += 1
    ok = (c == 2)
    dih = np.degrees(np.arccos(np.clip((V.nf[fa[ok, 0]] * V.nf[fa[ok, 1]]).sum(1), -1, 1)))
    # stored normals against geometric and smoothed normals
    nr = M.nor[V.vr]
    nr = nr / np.maximum(np.linalg.norm(nr, axis=1), 1e-20)[:, None]
    ls = V.loc[M.s[V.vr]]
    ang_geo = np.degrees(np.arccos(np.clip((nr * V.ng[ls]).sum(1), -1, 1)))
    nl = normales_sommets(lisse, V.Fl)
    ang_lisse = np.degrees(np.arccos(np.clip((nr * nl[ls]).sum(1), -1, 1)))
    # split normals: a welded vertex carries several stored normals
    moy = np.zeros_like(V.P)
    np.add.at(moy, ls, nr)
    moy /= np.maximum(np.linalg.norm(moy, axis=1), 1e-20)[:, None]
    ecart = np.degrees(np.arccos(np.clip((nr * moy[ls]).sum(1), -1, 1)))
    coupe = np.zeros(len(V.P), bool)
    np.logical_or.at(coupe, ls, ecart > 5)
    uvs = M.uv[V.vr]
    uv_moy = np.zeros((len(V.P), 2))
    np.add.at(uv_moy, ls, uvs)
    nb = np.bincount(ls, minlength=len(V.P))
    uv_moy /= np.maximum(nb, 1)[:, None]
    couture_uv = np.zeros(len(V.P), bool)
    np.logical_or.at(couture_uv, ls, np.abs(uvs - uv_moy[ls]).max(1) > 1e-5)
    # outline
    mats = sorted({int(i) for i in np.unique(M.mf[V.fg])})
    cont = M.contour[V.vr]
    peau_mats = [i for i in range(len(M.mats)) if M.role_materiau(i) == "peau"]
    reg_raw = V.region[ls]
    cont_reg = {r: r3(np.median(cont[reg_raw == r])) for r in np.unique(reg_raw) if (reg_raw == r).sum() >= 20}
    return {
        "faces": int(len(V.fg)), "sommets_soudes": int(len(V.P)), "sommets_bruts": int(len(V.vr)),
        "faces_modele": int(len(M.F)), "part_faces_modele": r3(len(V.fg) / len(M.F)),
        "pieces": len(pcs), "faces_par_piece": pcs[:12], "pieces_fermees": fermees,
        "aretes_bord": int(bord.sum()), "aretes_non_manifold": int((c > 2).sum()),
        "part_aretes_bord": r3(bord.mean()),
        "double_face_drapeau": [bool(M.mats[i]["drapeaux"] & 1) for i in mats],
        "epaisseur": {"part_faces_en_vis_a_vis": part(paire, V.aire),
                      "distance_mm": qmm(dist[paire]) if paire.any() else None,
                      "part_pres_bord": part(paire[f_pres_bord], V.aire[f_pres_bord]) if f_pres_bord.any() else None,
                      "part_loin_bord": part(paire[~f_pres_bord], V.aire[~f_pres_bord]) if (~f_pres_bord).any() else None},
        "feuille_exterieure_part": part(V.feuille_ext),
        "densite": {"arete_mm": qmm(L), "arete_peau_mm": qmm(Lp) if Lp is not None and len(Lp) else None,
                    "arete_dessous_mm": qmm(Ld) if Ld is not None and len(Ld) else None,
                    "rapport_arete_peau": r3(np.median(L) / np.median(Lp)) if Lp is not None and len(Lp) else None,
                    "rapport_arete_dessous": r3(np.median(L) / np.median(Ld)) if Ld is not None and len(Ld) else None,
                    "aire_dm2": r3(aire * 100, 2), "faces_par_dm2": r3(len(V.fg) / (aire * 100), 1)},
        "plis_geometriques": {"ecart_lisse_abs_mm": q(np.abs(dev) * 1000, (50, 90, 99), 2),
                              "max_mm": r3(np.abs(dev).max() * 1000, 1),
                              "part_gt2mm": r3((np.abs(dev) > 0.002).mean()),
                              "part_gt5mm": r3((np.abs(dev) > 0.005).mean()),
                              "par_region": regions,
                              "diedre_deg": q(dih, (50, 90, 99), 1),
                              "part_diedre_gt30": r3((dih > 30).mean()), "part_diedre_gt60": r3((dih > 60).mean())},
        "normales": {"angle_geo_deg": q(ang_geo, (50, 90, 99), 1),
                     "angle_lisse_deg": q(ang_lisse, (50, 90, 99), 1),
                     "plus_proche_du_lisse_part": r3((ang_lisse < ang_geo - 2).mean()),
                     "sommets_coupes_part": r3(coupe.mean()),
                     "coupes_sur_couture_uv_part": r3(couture_uv[coupe].mean()) if coupe.any() else None,
                     "coupes_hors_couture_uv": int((coupe & ~couture_uv).sum())},
        "contour": {"materiaux": [{"nom": M.noms_mat[i], "trace": bool(M.mats[i]["drapeaux"] & 0x10),
                                   "taille": r3(M.mats[i]["contour_taille"]),
                                   "couleur": r3(M.mats[i].get("contour_couleur")) if M.mats[i].get("contour_couleur") is not None else None}
                                  for i in mats],
                    "taille_peau": sorted({r3(M.mats[i]["contour_taille"]) for i in peau_mats}),
                    "sommet": q(cont, (1, 10, 50, 90)), "part_sommet_ne_1": r3((np.abs(cont - 1) > 1e-3).mean()),
                    "part_sommet_0": r3((cont < 1e-3).mean()), "par_region_med": cont_reg},
    }


def normales_sommets(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    out = np.zeros_like(P)
    for k in range(3):
        np.add.at(out, F[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1), 1e-20)[:, None]


def taubin(V, it, lam=0.5, mu=-0.53):
    """Taubin smoothing (no shrinkage) of the welded vertices, edges fixed."""
    P = V.P.copy()
    E = V.E
    bord = np.zeros(len(P), bool)
    bord[V.E[V.Ec == 1].ravel()] = True
    deg = np.zeros(len(P))
    np.add.at(deg, E[:, 0], 1)
    np.add.at(deg, E[:, 1], 1)
    deg = np.maximum(deg, 1)
    for k in range(it * 2):
        s = np.zeros_like(P)
        np.add.at(s, E[:, 0], P[E[:, 1]])
        np.add.at(s, E[:, 1], P[E[:, 0]])
        lap = s / deg[:, None] - P
        f = lam if k % 2 == 0 else mu
        P[~bord] += f * lap[~bord]
    return P


# ====================================================================== B. fit

def mesurer_ajustement(V, poses=True):
    """-> manufacturing: offset to give the fabric per zone, keep or remove the hidden skin,
    edges to watch (pinching) and poses where skinning causes clipping."""
    M = V.M
    res = {}
    if V.arbre_d is None:
        return {"dessous": "no surface underneath"}
    # inward offset (-n ray from the outer sheet)
    ext = V.feuille_ext
    O = V.P - V.ng * (-1e-4)          # start 0.1 mm outside
    idx, dist, nor = rayons(V.arbre_d, O, -V.ng, RAYON_DESSOUS)
    face_ok = (idx >= 0) & ((nor * V.ng).sum(1) > 0)      # surface underneath seen from the front
    d = np.where(face_ok, dist - 1e-4, np.inf)
    classe = {}
    for lo, hi, nom in CLASSES_ECART:
        classe[nom] = part(ext & (d >= lo) & (d < hi)) if ext.any() else None
    classe["rien_dessous"] = part(ext & ~np.isfinite(d))
    # materials hit underneath
    fd = V.fd
    mat_dessous = {}
    for i in np.where(ext & np.isfinite(d))[0]:
        n = M.noms_mat[M.mf[fd[idx[i]]]]
        mat_dessous[n] = mat_dessous.get(n, 0) + 1
    tot = max(sum(mat_dessous.values()), 1)
    par_region = {}
    for r in np.unique(V.region):
        m = (V.region == r) & ext
        if m.sum() < 20:
            continue
        dd = d[m]
        par_region[r] = {"sommets": int(m.sum()), "rien_dessous": r3((~np.isfinite(dd)).mean()),
                         "ecart_mm": qmm(dd[np.isfinite(dd)]) if np.isfinite(dd).any() else None}
    res["ecart"] = {"classes_part": classe, "ecart_mm": qmm(d[ext & np.isfinite(d)]) if (ext & np.isfinite(d)).any() else None,
                    "par_region": par_region,
                    "materiaux_dessous": {k: r3(v / tot) for k, v in sorted(mat_dessous.items(), key=lambda x: -x[1])[:6]}}
    # clipping at rest: a +n ray hits the back of a surface underneath within TRAVERSE_MAX
    O2 = V.P + V.ng * 1e-5
    idx2, dist2, nor2 = rayons(V.arbre_d, O2, V.ng, TRAVERSE_MAX)
    trav = (idx2 >= 0) & ((nor2 * V.ng).sum(1) > 0)
    tm = {}
    for i in np.where(trav)[0]:
        n = M.noms_mat[M.mf[fd[idx2[i]]]]
        tm[n] = tm.get(n, 0) + 1
    res["traversee_repos"] = {"part": part(trav), "profondeur_mm": qmm(dist2[trav]) if trav.any() else None,
                              "par_materiau": tm,
                              "regions": {r: int((trav & (V.region == r)).sum()) for r in np.unique(V.region[trav])}}
    # skin and underneath hidden by the garment: +n ray from the faces underneath
    Fd = M.F[fd]
    cen = M.pos[Fd].mean(1)
    nd = M.nf[fd]
    idx3, dist3, nor3 = rayons(V.arbre, cen + nd * 1e-5, nd, 0.05)
    cach = idx3 >= 0
    ad = M.aire[fd]
    par_mat = {}
    for i in np.where(cach)[0]:
        n = M.noms_mat[M.mf[fd[i]]]
        par_mat[n] = par_mat.get(n, 0) + ad[i]
    pe = V.peau[fd]
    res["cache"] = {"aire_cachee_sur_aire_vetement": r3(ad[cach].sum() / V.aire.sum()),
                    "aire_peau_cachee_sur_aire_vetement": r3(ad[cach & pe].sum() / V.aire.sum()),
                    "ecart_peau_cachee_mm": qmm(dist3[cach & pe]) if (cach & pe).any() else None,
                    "par_materiau_dm2": {k: r3(v * 100, 3) for k, v in sorted(par_mat.items(), key=lambda x: -x[1])[:6]}}
    # edges: free or joined, pinching
    res["bords"] = bords(V, d)
    # poses
    if poses:
        res["poses"] = mesurer_poses(V, idx, d, trav)
    return res


def bords(V, d):
    """Edge loops of the garment: joined to another surface (vertices shared after
    welding) or free; offset at the hem against the offset 3 to 5 rings further in (pinching)."""
    M = V.M
    E, c = V.E, V.Ec
    eb = E[c == 1]
    if not len(eb):
        return {"boucles": 0}
    # joined: the welded edge exists in a face outside the garment (not empty). Copies of the
    # garment's faces ("+" overlay, flipped copy) do not count: they border the whole garment.
    autres = np.where(~V.sel & ~M.faces_role(["vide"]))[0]
    copies = set(map(tuple, np.sort(M.Fs[V.sel], 1).tolist()))
    autres = autres[[tuple(x) not in copies for x in np.sort(M.Fs[autres], 1).tolist()]]
    Ea, _, _ = MD.aretes_faces(M.Fs[autres])
    ga = V.vs[eb]
    ga = np.sort(ga, 1)
    cle_a = set(map(tuple, Ea.tolist()))
    racc = np.array([tuple(x) in cle_a for x in ga.tolist()])
    # role of the other face
    lab = MD.composantes(len(V.P), eb)
    boucles = []
    an_tous = None
    for l in np.unique(lab[eb[:, 0]]):
        m = lab[eb[:, 0]] == l
        vb = np.unique(eb[m])
        if len(vb) < 4:
            continue
        an = V.anneaux(vb, 5)
        anneau = (an >= 3) & (an <= 5)
        g0 = d[vb][np.isfinite(d[vb])]
        g3 = d[anneau][np.isfinite(d[anneau])]
        long_ = np.linalg.norm(V.P[eb[m, 0]] - V.P[eb[m, 1]], axis=1).sum()
        reg, nreg = np.unique(V.region[vb], return_counts=True)
        e0 = float(np.median(g0)) if len(g0) >= 3 else None
        e3 = float(np.median(g3)) if len(g3) >= 3 else None
        boucles.append({"aretes": int(m.sum()), "longueur_mm": r3(long_ * 1000, 1),
                        "region": str(reg[np.argmax(nreg)]), "raccordee_part": r3(racc[m].mean()),
                        "rien_dessous_part": r3((~np.isfinite(d[vb])).mean()),
                        "ecart_ourlet_mm": mm(e0), "ecart_anneau3_5_mm": mm(e3),
                        "pince": bool(racc[m].mean() <= 0.5 and e0 is not None and e3 is not None
                                      and e0 < 0.5 * e3 and e0 < 0.003)})
    boucles.sort(key=lambda b: -b["longueur_mm"])
    return {"boucles": len(boucles), "raccordees": sum(b["raccordee_part"] > 0.5 for b in boucles),
            "libres": sum(b["raccordee_part"] <= 0.5 for b in boucles),
            "pincees": sum(b["pince"] for b in boucles), "detail": boucles[:12]}


def mesurer_poses(V, idx, d, trav_repos):
    """Bench of 22 poses (outils/atlas/banc.py), linear skinning (SDEF read as BDEF2).
    1) tracking: each outer vertex and the surface underneath it sees at rest (barycentric
       point); clipping if the signed offset drops below -0.5 mm.
    2) new clipping: 10 mm +n ray towards the back of a deformed surface underneath,
       minus those already present at rest."""
    M = V.M
    ok = np.isfinite(d) & V.feuille_ext
    if not ok.any():
        return {"paires": 0, "note": "nothing underneath: no pair to track"}
    fd = V.fd
    fi = fd[idx[ok]]
    Fq = M.F[fi]
    rep = M.sv[V.vs[ok]]                         # representative raw vertex
    P0 = M.pos[rep]
    bc = barycentres(P0 - V.ng[ok] * d[ok][:, None], M.pos[Fq[:, 0]], M.pos[Fq[:, 1]], M.pos[Fq[:, 2]])
    s0 = d[ok]
    rep_tous = M.sv[V.vs]
    # vertices carried (even slightly) by the garment's own bones (physics): physics moves them
    # in game, skinning alone says nothing about them
    Wv, Iv = M.poids[rep], M.idx[rep]
    libre = (np.where((Iv >= 0) & ~V.os_corps[np.maximum(Iv, 0)], Wv, 0).sum(1) > 0.05)
    res = {"paires": int(ok.sum()), "paires_corps_seul": int((~libre).sum()), "paires_os_propres_part": r3(libre.mean()),
           "skinning": "LBS, SDEF ignored, physics bones left at rest", "par_pose": {}}
    pires = []
    for nom in BANC.BANC:
        try:
            rot = BANC.pose_banc(nom, M.sq)
        except Exception as e:  # missing bone
            res["par_pose"][nom] = {"erreur": str(e)}
            continue
        Pd, _, _ = P.deformer(M.pos, M.idx, M.poids, M.sq, rotations=rot)
        A, B, C = Pd[Fq[:, 0]], Pd[Fq[:, 1]], Pd[Fq[:, 2]]
        qd = bc[:, :1] * A + bc[:, 1:2] * B + bc[:, 2:] * C
        nq = np.cross(B - A, C - A)
        nq /= np.maximum(np.linalg.norm(nq, axis=1), 1e-20)[:, None]
        s1 = ((Pd[rep] - qd) * nq).sum(1)
        tr = s1 < -TRAVERSE_SEUIL
        arb = bvh(Pd, M.F[fd])
        # the plane of the tracked face only holds if the vertex stayed above it: a garment
        # sliding around a curved surface (a belt turning with the chest) goes below that plane
        # without clipping anything. Once it has slid, clipping is confirmed against the
        # deformed surface itself: under its nearest face, and along its axis.
        glisse = np.linalg.norm((Pd[rep] - qd) - s1[:, None] * nq, axis=1) > GLISSE_MAX
        for j in np.where(tr & glisse)[0]:
            loc, nf, _, dist = arb.find_nearest(Vector(Pd[rep[j]]))
            sd = float((Vector(Pd[rep[j]]) - loc).dot(nf)) if loc is not None else 0.0
            if sd < -TRAVERSE_SEUIL and -sd >= 0.7 * dist:
                s1[j] = sd
            else:
                tr[j] = False
        # new clipping (all surfaces underneath)
        Pg = Pd[rep_tous]
        ngp = normales_sommets(Pg, V.Fl)
        i2, d2, n2 = rayons(arb, Pg + ngp * 1e-5, ngp, TRAVERSE_MAX)
        tn = (i2 >= 0) & ((n2 * ngp).sum(1) > 0) & ~trav_repos
        reg = V.region[ok][tr]
        u, n_ = np.unique(reg, return_counts=True)
        reg2 = V.region[tn]
        u2, n2_ = np.unique(reg2, return_counts=True)
        res["par_pose"][nom] = {"suivi_traverse_part": r3(tr.mean(), 4),
                                # share of the whole part, not of the tracked points only: a skirt with physics
                                # has only a handful of points held by the body, and 5 out of 14 got it rejected
                                "suivi_corps_seul_part": r3(tr[~libre].sum() / max(len(rep_tous), 1), 4) if (~libre).any() else None,
                                "suivi_profondeur_mm": qmm(-s1[tr], (50, 90, 100)) if tr.any() else None,
                                "suivi_regions": {str(a): int(b) for a, b in zip(u, n_)},
                                "nouvelles_part": r3(tn.mean(), 4),
                                "nouvelles_regions": {str(a): int(b) for a, b in zip(u2, n2_)}}
        pires.append((tr.mean() + tn.mean(), nom))
    pires.sort(reverse=True)
    vals = [v for v in res["par_pose"].values() if "suivi_traverse_part" in v]
    res["suivi_max_part"] = r3(max(v["suivi_traverse_part"] for v in vals), 4) if vals else None
    res["nouvelles_max_part"] = r3(max(v["nouvelles_part"] for v in vals), 4) if vals else None
    cs = [v["suivi_corps_seul_part"] for v in vals if v.get("suivi_corps_seul_part") is not None]
    res["suivi_corps_seul_max_part"] = r3(max(cs), 4) if cs else None
    res["poses_sans_traversee"] = sum(1 for v in vals if v["suivi_traverse_part"] == 0 and v["nouvelles_part"] == 0)
    res["poses_mesurees"] = len(vals)
    res["pires"] = [n for _, n in pires[:4]]
    return res


# ====================================================================== C. UV and texture

def mesurer_uv(V):
    """-> manufacturing: UV cutting (where to place seams), texel density to aim for, shared
    symmetry, what is painted (folds, shadows, pattern) and auxiliary textures."""
    M = V.M
    F = V.Fg
    uv = M.uv
    # islands (raw vertices, unwelded)
    Er = np.concatenate([F[:, [0, 1]], F[:, [1, 2]]])
    lab = MD.composantes(len(M.pos), Er)
    il = lab[F[:, 0]]
    u, cnt = np.unique(il, return_counts=True)
    # seams: interior welded edge whose two faces do not share the same raw edge
    Ebr = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), 2)   # (f,3,2) raw
    ebr = Ebr.reshape(-1, 2)
    cle_s = V.coin.reshape(-1)         # welded edge of each corner (f*3); coin is (f,3) -> order [01,12,20]
    # V.coin comes from aretes_faces(Fl), which stacks [01],[12],[20]: same order as Ebr
    couture = np.zeros(len(V.E), bool)
    uvd = np.zeros(len(V.E), bool)
    premier = np.full((len(V.E), 2), -1)
    for k in range(len(ebr)):
        e = cle_s[k]
        if premier[e, 0] < 0:
            premier[e] = ebr[k]
        elif (premier[e] != ebr[k]).any():
            couture[e] = True
            a, b = premier[e]
            c_, d_ = ebr[k]
            # same position, different UVs?
            pa, pc = M.s[a], M.s[c_]
            u1 = (uv[a], uv[b]) if pa == pc else (uv[b], uv[a])
            if np.abs(np.array(u1) - np.array([uv[c_], uv[d_]])).max() > 1e-5:
                uvd[e] = True
    cs = couture & uvd
    Lc = np.linalg.norm(V.P[V.E[:, 0]] - V.P[V.E[:, 1]], axis=1)
    # position of the seams around the nearest body bone
    mid = (V.P[V.E[:, 0]] + V.P[V.E[:, 1]]) / 2
    secteurs = {}
    for e in np.where(cs)[0]:
        b = V.os_proche[V.E[e, 0]]
        if b < 0:
            continue
        a, c_ = M.tetes[b], M.queues[b]
        ax = (c_ - a) / max(np.linalg.norm(c_ - a), 1e-9)
        r = mid[e] - a
        r = r - (r @ ax) * ax
        av = np.array([0, -1.0, 0]) - (np.array([0, -1.0, 0]) @ ax) * ax
        av /= max(np.linalg.norm(av), 1e-9)
        ang = np.degrees(np.arccos(np.clip(r @ av / max(np.linalg.norm(r), 1e-9), -1, 1)))
        fam = famille_region(M.region_os[b])
        sec = "avant" if ang < 30 else "avant_cote" if ang < 75 else "cote" if ang < 105 else "arriere_cote" if ang < 150 else "arriere"
        if fam == "bras" and np.linalg.norm(r) > 0:
            sec = "dessous" if r[2] < -0.5 * np.linalg.norm(r) else "dessus" if r[2] > 0.5 * np.linalg.norm(r) else sec
        secteurs.setdefault(fam, {}).setdefault(sec, 0.0)
        secteurs[fam][sec] += Lc[e]
    for fam in secteurs:
        t = sum(secteurs[fam].values())
        secteurs[fam] = {k: r3(v / t) for k, v in sorted(secteurs[fam].items(), key=lambda x: -x[1])}
    centre = cs & (np.abs(mid[:, 0]) < 0.003)
    # shared symmetry: identical UV triangles
    T = np.round(uv[F] * 1e4).astype(np.int64).tolist()
    cles = [tuple(sorted(map(tuple, t))) for t in T]
    from collections import Counter
    cc = Counter(cles)
    partage = np.array([cc[k] > 1 for k in cles])
    # texel density
    tailles = {}
    out = {"iles": int(len(u)), "plus_grande_ile_part": r3(cnt.max() / cnt.sum()),
           "iles_part_faces": r3(np.sort(cnt)[::-1][:8] / cnt.sum()),
           "aretes_couture_uv": int(cs.sum()), "longueur_couture_mm": r3(Lc[cs].sum() * 1000, 0),
           "couture_sur_longueur_bord": r3(Lc[cs].sum() / max(Lc[V.Ec == 1].sum(), 1e-9)),
           "coupes_normales_seules": int((couture & ~uvd).sum()),
           "coutures_par_secteur": secteurs, "couture_ligne_centre_part": r3(Lc[centre].sum() / max(Lc[cs].sum(), 1e-9)),
           "uv_partages_part": part(partage, V.aire)}
    out["texels"] = densite_texels(V, tailles)
    out["texture"] = mesurer_texture(V, tailles)
    return out


def aire_uv(T):
    a, b = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2


def famille_region(r):
    if r in ("bras", "avant_bras", "main", "doigt"):
        return "bras"
    if r in ("cuisse", "jambe", "pied"):
        return "jambe"
    return "tronc"


def charger_image(M, t, cache):
    if t < 0 or t >= len(M.textures):
        return None
    f = os.path.normpath(os.path.join(os.path.dirname(M.chemin), M.textures[t].replace("\\", "/")))
    if f in cache:
        return cache[f]
    img = None
    if bpy is not None and os.path.exists(f):
        try:
            im = bpy.data.images.load(f)
            w, h = im.size
            px = np.empty(w * h * 4, np.float32)
            im.pixels.foreach_get(px)
            bpy.data.images.remove(im)
            img = {"nom": M.textures[t], "w": w, "h": h, "px": px.reshape(h, w, 4)}
        except Exception as e:
            journal("unreadable texture", f, e)
    cache[f] = img
    return img


def densite_texels(V, cache):
    """px per metre (square root of the area ratio); garment, skin, underneath."""
    M = V.M

    def dens(fi):
        vals, poids = [], []
        for mi in np.unique(M.mf[fi]):
            img = charger_image(M, M.mats[mi]["texture"], cache)
            if img is None:
                continue
            ff = fi[M.mf[fi] == mi]
            T = M.uv[M.F[ff]]
            auv = aire_uv(T) * img["w"] * img["h"]
            a3 = M.aire[ff]
            ok = a3 > 1e-12
            vals.append(np.sqrt(auv[ok] / a3[ok]))
            poids.append(a3[ok])
        if not vals or not sum(len(x) for x in vals):
            return None
        v, p = np.concatenate(vals), np.concatenate(poids)
        o = np.argsort(v)
        c = np.cumsum(p[o]) / p.sum()
        return {"p10": r3(v[o][np.searchsorted(c, 0.1)], 0), "p50": r3(v[o][np.searchsorted(c, 0.5)], 0),
                "p90": r3(v[o][np.searchsorted(c, 0.9)], 0)}
    g = dens(V.fg)
    pe = dens(np.where(V.peau & ~M.faces_role(["vide"]))[0])
    de = dens(V.fd)
    tex = sorted({(c["nom"], c["w"], c["h"]) for c in cache.values() if c})
    return {"px_par_m": g, "peau_px_par_m": pe, "dessous_px_par_m": de,
            "rapport_peau": r3(g["p50"] / pe["p50"]) if g and pe else None,
            "rapport_dessous": r3(g["p50"] / de["p50"]) if g and de else None,
            "textures_vues": [list(t) for t in tex]}


def flou(L, r):
    """Square moving average of radius r (cumulative sums), edges repeated."""
    if r < 1:
        return L
    P_ = np.pad(L, r + 1, mode="edge")
    S = P_.cumsum(0).cumsum(1)
    k = 2 * r + 1
    out = S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]
    return out[:L.shape[0], :L.shape[1]] / (k * k)


def mesurer_texture(V, cache):
    """Samples drawn over the garment's UV surface: lightness, painted lines and shadows
    (offset from the local mean), fine pattern (offset from a 2 px blur), transparency."""
    M = V.M
    rng = np.random.default_rng(1)
    out = {"par_materiau": {}}
    for mi in np.unique(M.mf[V.fg]):
        m = M.mats[mi]
        img = charger_image(M, m["texture"], cache)
        ff = V.fg[M.mf[V.fg] == mi]
        info = {"texture": M.textures[m["texture"]] if 0 <= m["texture"] < len(M.textures) else None,
                "sphere": M.textures[m["sphere"]] if 0 <= m.get("sphere", -1) < len(M.textures) else None,
                "sphere_mode": m.get("sphere_mode"),
                "toon": ("toon%02d.bmp (shared)" % (m["toon"] + 1) if m.get("toon_partage") == 1 else
                         M.textures[m["toon"]] if 0 <= m.get("toon", -1) < len(M.textures) else None),
                "diffus": r3(m["diffus"]), "speculaire": r3(m.get("speculaire")), "ambiant": r3(m.get("ambiant"))}
        if img is not None:
            h, w = img["h"], img["w"]
            px = img["px"]
            Lm = MC._L(np.clip(px[..., :3], 0, 1))
            r = max(2, int(round(w / 128)))
            Lb = flou(Lm, r)
            L2 = flou(Lm, 2)
            T = M.uv[M.F[ff]]
            auv = aire_uv(T)
            n = 60000
            j = rng.choice(len(ff), n, p=auv / auv.sum()) if auv.sum() > 0 else rng.integers(0, len(ff), n)
            r1, r2 = rng.random(n), rng.random(n)
            sq = np.sqrt(r1)
            bb = np.stack([1 - sq, sq * (1 - r2), sq * r2], 1)
            p = np.einsum("nk,nkd->nd", bb, T[j])
            x = (np.mod(p[:, 0], 1) * w).astype(int).clip(0, w - 1)
            y = ((1 - np.mod(p[:, 1], 1)) * h).astype(int).clip(0, h - 1)
            Ls, Lbs, L2s, a = Lm[y, x], Lb[y, x], L2[y, x], px[y, x, 3]
            info.update({"taille": [w, h], "L": q(Ls, (5, 50, 95), 1),
                         "trait_sombre_part": r3((Ls - Lbs < -10).mean()),
                         "reflet_part": r3((Ls - Lbs > 10).mean()),
                         "ecart_local_p5_p95": [r3(np.percentile(Ls - Lbs, 5), 1), r3(np.percentile(Ls - Lbs, 95), 1)],
                         "motif_fin_p90": r3(np.percentile(np.abs(Ls - L2s), 90), 2),
                         "transparent_part": r3((a < 0.5).mean()), "rayon_flou_px": r})
        out["par_materiau"][M.noms_mat[mi]] = info
    # additional UVs and per-vertex outline on the garment's vertices
    ua = M.uv_add[V.vr] if M.uv_add.shape[1] else None
    if ua is not None:
        out["uv_add"] = [{"min": r3(ua[:, k].min(0)), "max": r3(ua[:, k].max(0)),
                          "constante": bool(np.ptp(ua[:, k], 0).max() < 1e-6)} for k in range(ua.shape[1])]
    else:
        out["uv_add"] = []
    return out


# ====================================================================== D. weights and physics

def mesurer_poids(V):
    """-> manufacturing: copy the body's weights or not, number of bones per vertex, width of
    the transitions at the joints, the garment's own bones."""
    M = V.M
    vr = V.vr
    W, I = M.poids[vr], M.idx[vr]
    actif = (W > 1e-3) & (I >= 0)
    nb = actif.sum(1)
    typ = M.type_poids[vr]
    corps = V.os_corps
    propre = np.where(actif & ~corps[np.maximum(I, 0)], W, 0).sum(1)
    # copied weights: nearest vertex underneath (within 20 mm)
    dr = np.unique(M.F[V.fd]) if len(V.fd) else np.array([], int)
    copie = None
    if len(dr):
        dist, j = MC.kd_plus_proche(M.pos[dr], M.pos[vr])
        prox = dist < 0.02
        L1 = []
        for a, b in zip(np.where(prox)[0], dr[j[prox]]):
            wa = {int(i): float(w) for i, w in zip(I[a], W[a]) if i >= 0 and w > 1e-4}
            wb = {int(i): float(w) for i, w in zip(M.idx[b], M.poids[b]) if i >= 0 and w > 1e-4}
            L1.append(sum(abs(wa.get(k, 0) - wb.get(k, 0)) for k in set(wa) | set(wb)))
        L1 = np.array(L1)
        copie = {"sommets_proches_part": r3(prox.mean()), "L1": q(L1, (10, 50, 90), 3),
                 "identiques_part": r3((L1 < 0.05).mean()) if len(L1) else None,
                 "proches_part": r3((L1 < 0.3).mean()) if len(L1) else None}
    # transitions at the joints: width (along the limb) of 0.1 < w_child < 0.9
    trans = {}
    nonv = np.setdiff1d(np.unique(M.F[~M.faces_role(["vide", "tete"])]), vr)
    for nom, j_os, p_os in ARTICULATIONS:
        if j_os not in M.noms_os or p_os not in M.noms_os:
            continue
        b, pb = M.noms_os.index(j_os), M.noms_os.index(p_os)
        sous = sous_arbre(M, b)
        ax = M.tetes[b] - M.tetes[pb]
        ax /= np.linalg.norm(ax)

        def largeur(vv):
            P_ = M.pos[vv]
            pr = (P_ - M.tetes[b]) @ ax
            radial = np.linalg.norm((P_ - M.tetes[b]) - np.outer(pr, ax), axis=1)
            pres = (np.abs(pr) < 0.12) & (radial < 0.12)
            wc = np.where(np.isin(M.idx[vv], sous), M.poids[vv], 0).sum(1)
            t = pres & (wc > 0.1) & (wc < 0.9)
            if t.sum() < 5:
                return None
            return {"sommets": int(t.sum()), "largeur_mm": r3((np.percentile(pr[t], 95) - np.percentile(pr[t], 5)) * 1000, 1),
                    "centre_mm": r3(np.median(pr[t]) * 1000, 1)}
        trans[nom] = {"vetement": largeur(vr), "reste_du_modele": largeur(nonv)}
    return {"influences": {str(k): int((nb == k).sum()) for k in range(1, 5)},
            "influences_moy": r3(nb.mean(), 2),
            "types": {PMX.TYPES_POIDS[int(k)]: int((typ == k).sum()) for k in np.unique(typ)},
            "poids_os_propres": {"part_sommets": r3((propre > 1e-3).mean()), "moyenne": r3(propre.mean())},
            "copie_du_dessous": copie, "transitions": trans}


def sous_arbre(M, b):
    out, pile = [], [b]
    enfants = {}
    for i, p in enumerate(M.par):
        enfants.setdefault(int(p), []).append(i)
    while pile:
        x = pile.pop()
        out.append(x)
        pile += enfants.get(x, [])
    return out


def mesurer_physique(V):
    """-> manufacturing: bone chains of the free panels, rigid bodies (shape, mass, damping),
    joints, groups and collisions with the legs. Units: mm (rigid bodies read in MMD units × 80)."""
    M = V.M
    O, RG, J = M.p["os"], M.rigides, M.joints
    vr = V.vr
    W, I = M.poids[vr], M.idx[vr]
    portes = np.unique(I[(W > 0.05) & (I >= 0)])
    propres = [int(b) for b in portes if not V.os_corps[b]]
    if not propres:
        return {"os_propres": 0}
    sp = set(propres)
    # extend to non-body descendants (leaf bones without weights are part of the chain)
    enfants = {}
    for i, o in enumerate(O):
        enfants.setdefault(o["parent"], []).append(i)
    tout = set(sp)
    pile = list(sp)
    while pile:
        x = pile.pop()
        for e in enfants.get(x, []):
            if not V.os_corps[e] and e not in tout:
                tout.add(e)
                pile.append(e)
    # walk up to non-body ancestors
    for b in list(tout):
        p = O[b]["parent"]
        while p >= 0 and not V.os_corps[p] and p not in tout:
            tout.add(p)
            p = O[p]["parent"]
    racines = [b for b in tout if O[b]["parent"] < 0 or O[b]["parent"] not in tout]

    def plus_long(i):
        ch = [i]
        while True:
            e = [x for x in enfants.get(ch[-1], []) if x in tout]
            if not e:
                return ch
            ch.append(max(e, key=lambda x: len(plus_long(x))))
    rig_os = {}
    for k, r in enumerate(RG):
        rig_os.setdefault(r["os"], []).append(k)
    rg_v = [k for b in tout for k in rig_os.get(b, [])]
    dyn = [k for k in rg_v if RG[k]["mode"] in RIG_DYN]
    jambes = [k for k, r in enumerate(RG) if r["os"] >= 0 and r["mode"] == 0 and V.os_corps[r["os"]]
              and M.region_os[r["os"]] in ("cuisse", "jambe", "bassin", "fesses")]
    tronc = [k for k, r in enumerate(RG) if r["os"] >= 0 and r["mode"] == 0 and V.os_corps[r["os"]]
             and M.region_os[r["os"]] in ("tronc_haut", "ventre", "poitrine", "bras", "avant_bras")]
    seg = {k: segment_rigide(RG[k]) for k in set(dyn) | set(jambes) | set(tronc)}
    E_ = 80.0  # mm per MMD unit

    def contacts(cible):
        heurte, rec = [], []
        for k in dyn:
            a, b, ra = seg[k]
            for kk in cible:
                a2, b2, r2 = seg[kk]
                dd = (dist_segments(a, b, a2, b2) - ra - r2) * E_
                if collisionnent(RG[k], RG[kk]):
                    heurte.append(kk)
                    rec.append(dd)
        return heurte, rec
    hj, recj = contacts(jambes)
    ht, rect = contacts(tronc)
    sd = set(dyn)
    jts = [j for j in J if j["a"] in sd or j["b"] in sd]
    # cross joints: between two dynamic bodies of different chains
    chaine_de = {}
    chaines = []
    for r0 in sorted(racines):
        ch = plus_long(r0)
        sous = sous_arbre_set(r0, enfants, tout)
        for b in sous:
            for k in rig_os.get(b, []):
                chaine_de[k] = r0
        rr = [RG[rig_os[b][0]] if rig_os.get(b) else None for b in ch]
        pos = np.array([O[b]["pos"] for b in ch])
        L = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() * E_ if len(ch) > 1 else 0.0
        chaines.append({"racine": M.noms_os[r0], "parent": M.noms_os[O[r0]["parent"]] if O[r0]["parent"] >= 0 else None,
                        "maillons": len(ch), "os": len(sous), "longueur_mm": r3(L, 1),
                        "modes": [r["mode"] if r else None for r in rr],
                        "masses": r3([r["masse"] if r else None for r in rr]),
                        "amort_dep": r3([r["amort_dep"] if r else None for r in rr]),
                        "amort_rot": r3([r["amort_rot"] if r else None for r in rr])})
    transv = [j for j in jts if j["a"] in chaine_de and j["b"] in chaine_de and chaine_de[j["a"]] != chaine_de[j["b"]]]
    long_ = [j for j in jts if j not in transv]

    def limites(js):
        if not js:
            return None
        lo = np.degrees(np.array([j["rot_min"] for j in js]))
        hi = np.degrees(np.array([j["rot_max"] for j in js]))
        amp = hi - lo
        dep = np.array([np.subtract(j["dep_max"], j["dep_min"]) for j in js]) * E_
        return {"n": len(js), "rot_min_deg_med": r3(np.median(lo, 0), 1), "rot_max_deg_med": r3(np.median(hi, 0), 1),
                "amplitude_deg_p90": r3(np.percentile(amp, 90, axis=0), 1),
                "bloques_part": r3((amp.max(1) < 0.01).mean()),
                "translation_libre_part": r3((dep.max(1) > 1e-3).mean()),
                "ressort_rot": sorted({tuple(r3(j["ressort_rot"], 1)) for j in js})[:4],
                "ressort_dep": sorted({tuple(r3(j["ressort_dep"], 1)) for j in js})[:4]}
    groupes = {}
    for k in dyn:
        groupes[RG[k]["groupe"]] = groupes.get(RG[k]["groupe"], 0) + 1
    rd = [RG[k] for k in dyn]
    return {
        "os_propres": len(tout), "os_portant_poids": len(propres), "chaines": len(chaines),
        "maillons": q([c["maillons"] for c in chaines], (10, 50, 90), 1),
        "longueur_chaine_mm": q([c["longueur_mm"] for c in chaines], (10, 50, 90), 1),
        "rigides": len(rg_v), "rigides_dynamiques": len(dyn),
        "modes": {str(m): sum(1 for k in rg_v if RG[k]["mode"] == m) for m in (0, 1, 2)},
        "formes": {PMX.FORMES_RIGIDE[f]: sum(1 for r in rd if r["forme"] == f) for f in (0, 1, 2)},
        "taille_mm": q([max(r["taille"]) * E_ for r in rd], (10, 50, 90), 1) if rd else None,
        "rayon_mm": q([segment_rigide(r)[2] * E_ for r in rd], (10, 50, 90), 1) if rd else None,
        "masse": q([r["masse"] for r in rd], (10, 50, 90)) if rd else None,
        "amort_dep": q([r["amort_dep"] for r in rd], (10, 50, 90)) if rd else None,
        "amort_rot": q([r["amort_rot"] for r in rd], (10, 50, 90)) if rd else None,
        "rebond": sorted({r3(r["rebond"]) for r in rd}) if rd else None,
        "friction": sorted({r3(r["friction"]) for r in rd}) if rd else None,
        "groupes": groupes,
        "ignore_son_groupe_part": r3(np.mean([not ((r["masque"] >> r["groupe"]) & 1) for r in rd])) if rd else None,
        "ignores": sorted({tuple(ignores(r)) for r in rd})[:4] if rd else None,
        "rigides_jambes": len(jambes),
        "heurte_jambes_part": r3(np.mean([any(collisionnent(RG[k], RG[kk]) for kk in jambes) for k in dyn])) if dyn and jambes else None,
        "heurte_tronc_bras_part": r3(np.mean([any(collisionnent(RG[k], RG[kk]) for kk in tronc) for k in dyn])) if dyn and tronc else None,
        "recouvrement_repos_jambes_mm": q(recj, (0, 10, 50), 1) if recj else None,
        "recouvrement_repos_tronc_mm": q(rect, (0, 10, 50), 1) if rect else None,
        "joints_long": limites(long_), "joints_transverses": limites(transv),
        "detail_chaines": chaines[:8]}


def sous_arbre_set(r0, enfants, tout):
    out, pile = [], [r0]
    while pile:
        x = pile.pop()
        out.append(x)
        pile += [e for e in enfants.get(x, []) if e in tout]
    return out


# ====================================================================== overlay layers

def mesurer_calque(V):
    """"+" material: same geometry (shadow, pattern or outline overlay) or lining?"""
    M = V.M
    fc = np.where(V.calque)[0]
    if not len(fc):
        return None
    Fc = M.Fs[fc]
    cg = set(map(tuple, np.sort(V.Fgs, 1).tolist()))
    meme = np.array([tuple(f) in cg for f in np.sort(Fc, 1).tolist()])
    vv = np.unique(M.F[fc])
    d = []
    for p in M.pos[vv]:
        h = V.arbre.find_nearest(Vector(p))
        d.append(h[3] if h[0] is not None else np.inf)
    d = np.array(d)
    cen = M.pos[M.F[fc]].mean(1)
    nc = M.nf[fc]
    # orientation relative to the garment: same direction or flipped (lining)
    sens = []
    for c_, n_ in zip(cen, nc):
        h = V.arbre.find_nearest(Vector(c_))
        if h[0] is not None:
            sens.append(float(np.dot(n_, np.asarray(h[1]))))
    sens = np.array(sens)
    mats = sorted({int(i) for i in np.unique(M.mf[fc])})
    return {"faces": int(len(fc)), "memes_faces_part": r3(meme.mean()),
            "distance_mm": qmm(d[np.isfinite(d)], (50, 90, 100)),
            "meme_sens_part": r3((sens > 0.5).mean()) if len(sens) else None,
            "retourne_part": r3((sens < -0.5).mean()) if len(sens) else None,
            "materiaux": [{"nom": M.noms_mat[i], "texture": M.textures[M.mats[i]["texture"]] if 0 <= M.mats[i]["texture"] < len(M.textures) else None,
                           "diffus": r3(M.mats[i]["diffus"]), "drapeaux": M.mats[i]["drapeaux"],
                           "contour": bool(M.mats[i]["drapeaux"] & 0x10)} for i in mats]}


# ====================================================================== checks

MOULANTS = ("haut_moulant", "pantalon", "short", "peint_sur_corps")


def controler(r):
    """Guardrails and contracts of a garment, thresholds
    taken from the 10 official ones. Returns a list of {nom, valeur, seuil, ok}; ok None = not applicable.
    -> manufacturing: a "no" on a guardrail makes the garment wrong; on a type contract
    (tight / loose), it departs from every official garment of that type."""
    m, a, u, p, ph = r["maillage"], r["ajustement"], r["uv"], r["poids"], r["physique"]
    C = []

    def c(nom, val, seuil, ok):
        C.append({"nom": nom, "valeur": val, "seuil": seuil, "ok": None if val is None else bool(ok)})
    # guardrails (10/10 on the official ones, except named exceptions)
    c("non_manifold", m["aretes_non_manifold"], "0", m["aretes_non_manifold"] == 0)
    rec = r.get("calque") or {}
    c("double_face", bool(all(m["double_face_drapeau"]) or (rec.get("retourne_part") or 0) > 0.9),
      "flag 0x01 or flipped overlay", all(m["double_face_drapeau"]) or (rec.get("retourne_part") or 0) > 0.9)
    tailles = [x["taille"] for x in _g(m, "contour", "materiaux") or [] if x["trace"]]
    c("contour_taille", min(tailles) if tailles else 0.0, "0.5-0.6 on every material",
      len(tailles) == len(m["contour"]["materiaux"]) and all(0.45 <= t <= 0.65 for t in tailles))
    c("contour_sommet_p1", _g(m, "contour", "sommet", "p1"), "1.0", _g(m, "contour", "sommet", "p1") == 1.0)
    c("budget_part_modele", m["part_faces_modele"], "<= 0.19", m["part_faces_modele"] <= 0.19)
    c("influences_max_4", max(int(k) for k, v in p["influences"].items() if v), "<= 4",
      max(int(k) for k, v in p["influences"].items() if v) <= 4)
    c("sans_sdef", "SDEF" not in p["types"] or p["types"]["SDEF"] == 0, "no SDEF",
      "SDEF" not in p["types"] or p["types"]["SDEF"] == 0)
    c("texels_sur_peau", _g(u, "texels", "rapport_peau"), "0.65-1.6",
      _g(u, "texels", "rapport_peau") is not None and 0.65 <= u["texels"]["rapport_peau"] <= 1.6)
    c("peau_cachee", _g(a, "cache", "aire_peau_cachee_sur_aire_vetement"), "<= 0.16",
      (_g(a, "cache", "aire_peau_cachee_sur_aire_vetement") or 0) <= 0.16)
    c("traversee_repos", _g(a, "traversee_repos", "part"), "<= 0.031",
      (_g(a, "traversee_repos", "part") or 0) <= 0.031)
    c("traversee_repos_p90_mm", _g(a, "traversee_repos", "profondeur_mm", "p90"), "<= 9",
      (_g(a, "traversee_repos", "profondeur_mm", "p90") or 0) <= 9)
    cs = _g(a, "poses", "suivi_corps_seul_max_part")
    c("poses_corps_seul", cs, "<= 0.10 (target 0.01)", cs is not None and cs <= 0.10)
    if ph.get("os_propres"):
        c("phys_ignore_son_groupe", ph.get("ignore_son_groupe_part"), "1", ph.get("ignore_son_groupe_part") == 1)
        c("phys_heurte_jambes", ph.get("heurte_jambes_part"), "1", ph.get("heurte_jambes_part") == 1)
        jl = ph.get("joints_long") or {}
        c("phys_torsion_bloquee", _g(jl, "rot_max_deg_med"), "Y = 0 (median of the chain joints)",
          bool(jl) and jl["rot_min_deg_med"][1] == 0 and jl["rot_max_deg_med"][1] == 0)
        c("phys_ressorts_nuls", _g(jl, "ressort_rot"), "0", bool(jl) and all(v == 0 for t in jl["ressort_rot"] for v in t))
        c("phys_rebond_nul", ph.get("rebond"), "0", ph.get("rebond") == [0.0])
    # contracts per family
    cl = _g(a, "ecart", "classes_part") or {}
    if r["type"] in MOULANTS:
        c("moulant_rien_dessous", cl.get("rien_dessous"), ">= 0.65", (cl.get("rien_dessous") or 0) >= 0.65)
        c("moulant_bord_soude", _g(a, "bords", "raccordees"), ">= 1 loop welded to the skin",
          (_g(a, "bords", "raccordees") or 0) >= 1)
        c("moulant_plis_p90_mm", _g(m, "plis_geometriques", "ecart_lisse_abs_mm", "p90"), "<= 4",
          (_g(m, "plis_geometriques", "ecart_lisse_abs_mm", "p90") or 0) <= 4)
        c("moulant_sans_os_propres", ph.get("os_propres", 0), "0", ph.get("os_propres", 0) == 0)
    else:
        e50 = _g(a, "ecart", "ecart_mm", "p50")
        c("ample_ecart_p50_mm", e50, "2-45", e50 is not None and 2 <= e50 <= 45)
        c("ample_bord_libre", _g(a, "bords", "libres"), ">= 1 free loop", (_g(a, "bords", "libres") or 0) >= 1)
        c("ample_normales_coupees_sur_coutures", _g(m, "normales", "coupes_sur_couture_uv_part"),
          ">= 0.99 if split", (_g(m, "normales", "coupes_sur_couture_uv_part") or 1) >= 0.99)
    return C


# ====================================================================== main

class SansObjet(Exception):
    """The measurement has nothing to measure: every piece of the part is set aside (rigid
    accessory, small object). This is not a failure of the part."""


def mesurer(spec, M, poses=True):
    t0 = time.time()
    V = Vetement(M, spec)
    if not len(V.fg):
        raise SansObjet("no piece kept (%s)" %", ".join(sorted({str(d.get("raison")) for d in V.ecartees}))[:120])
    out = {"id": spec["id"], "perso": spec["perso"], "type": spec["type"], "pourquoi": spec["pourquoi"],
           "materiaux": spec["mats"], "calques": spec.get("calques", []),
           "isolement": {"pieces_gardees": [{k: d[k] for k in ("indice", "faces", "region", "diag_mm", "fermee", "os_dominants")}
                                            for d in V.gardees],
                         "pieces_ecartees": [{k: d[k] for k in ("indice", "faces", "region", "diag_mm", "fermee", "os_dominants", "raison")}
                                             for d in V.ecartees]},
           "couches_dessus": V.couches_dessus}
    journal(spec["id"], "mesh")
    out["maillage"] = mesurer_maillage(V)
    journal(spec["id"], "fit")
    out["ajustement"] = mesurer_ajustement(V, poses)
    journal(spec["id"], "uv")
    out["uv"] = mesurer_uv(V)
    journal(spec["id"], "weights")
    out["poids"] = mesurer_poids(V)
    out["physique"] = mesurer_physique(V)
    out["calque"] = mesurer_calque(V)
    out["controles"] = controler(out)
    out["duree_s"] = r3(time.time() - t0, 1)
    return out


def _g(d, *ch):
    for c in ch:
        if d is None:
            return None
        d = d.get(c) if isinstance(d, dict) else None
    return d


def tableau(resultats, chemin):
    L = ["# ZZZ garments: overview table", "",
         "One row per garment. mm; shares between 0 and 1. See <id>.json for details.", "",
         "| id | type | faces | parts | boundary edges | facing | edge / skin | folds p90 mm | smoothed normals | vertex outline p50 | nothing underneath | stuck <1mm | offset p50 mm | clipping at rest | worst pose (tracked) | worst pose (body only) | UV islands | texels / skin | dark line | mean influences | body copy | own bones | dyn rigid bodies |",
         "|" + "---|" * 23]
    for r in resultats:
        m, a, u, p, ph = r["maillage"], r["ajustement"], r["uv"], r["poids"], r["physique"]
        tex = list(_g(u, "texture", "par_materiau").values())
        L.append("|" + " %s |" * 23 % (
            r["id"], r["type"], m["faces"], m["pieces"], m["aretes_bord"], _g(m, "epaisseur", "part_faces_en_vis_a_vis"),
            _g(m, "densite", "rapport_arete_peau"), _g(m, "plis_geometriques", "ecart_lisse_abs_mm", "p90"),
            _g(m, "normales", "plus_proche_du_lisse_part"), _g(m, "contour", "sommet", "p50"),
            _g(a, "ecart", "classes_part", "rien_dessous"), _g(a, "ecart", "classes_part", "colle_lt1mm"),
            _g(a, "ecart", "ecart_mm", "p50"), _g(a, "traversee_repos", "part"), _g(a, "poses", "suivi_max_part"),
            _g(a, "poses", "suivi_corps_seul_max_part"), u["iles"], _g(u, "texels", "rapport_peau"), tex[0].get("trait_sombre_part") if tex else None,
            p["influences_moy"], _g(p, "copie_du_dessous", "identiques_part"), ph.get("os_propres"),
            ph.get("rigides_dynamiques")))
    tmp = chemin + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    os.replace(tmp, chemin)


def dossier_sortie(arg):
    if arg:
        return arg
    d = os.path.join(MD.R.chemin("chemins.donnees"), "vetements")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    if os.path.isdir(d) and os.access(d, os.W_OK):
        return d
    return os.path.join(os.environ.get("TMPDIR", "/tmp"), "vetements")


def main(argv):
    sortie = dossier_sortie(argv[argv.index("--sortie") + 1] if "--sortie" in argv else None)
    os.makedirs(sortie, exist_ok=True)
    seuls = argv[argv.index("--seuls") + 1].split(",") if "--seuls" in argv else None
    poses = "--sans-poses" not in argv
    modeles = {}
    res = []
    specs = VETEMENTS
    if "--pmx" in argv:
        # manufactured candidate: --pmx PATH --mats A,B [--id NAME] [--type T] [--calques X,Y]
        chemin = argv[argv.index("--pmx") + 1]
        val = lambda o, d=None: argv[argv.index(o) + 1] if o in argv else d  # noqa: E731
        nom = val("--id", os.path.splitext(os.path.basename(chemin))[0])
        modeles[nom] = MD.Modele(nom, chemin=chemin)
        specs = [{"id": nom, "perso": nom, "type": val("--type", "haut_ample"), "mats": val("--mats").split(","),
                  "calques": val("--calques", "").split(",") if val("--calques") else [],
                  "pourquoi": "manufactured candidate (%s)" % chemin}]
        seuls = [nom]
    for spec in specs:
        if seuls and spec["id"] not in seuls:
            continue
        M = modeles.get(spec["perso"]) or MD.modele_de(spec["perso"], spec.get("pmx"))
        modeles[spec["perso"]] = M
        r = mesurer(spec, M, poses)
        ecrire_json(os.path.join(sortie, spec["id"] + ".json"), r)
        journal(spec["id"], "written (%.0f s)" % r["duree_s"])
        res.append(r)
        mauvais = [x for x in r["controles"] if x["ok"] is False]
        journal(spec["id"], "checks: %d/%d passed%s" % (
            sum(1 for x in r["controles"] if x["ok"]), sum(1 for x in r["controles"] if x["ok"] is not None),
            "".join("\n    FAIL %s = %s (threshold %s)" % (x["nom"], x["valeur"], x["seuil"]) for x in mauvais)))
    if not seuls:
        ecrire_json(os.path.join(sortie, "ensemble.json"), {"date": time.strftime("%Y-%m-%d"), "vetements": res})
        tableau(res, os.path.join(sortie, "tableau.md"))
    else:
        tableau(res, os.path.join(sortie, "tableau_partiel.md"))
    print("outputs:", sortie)


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(a)

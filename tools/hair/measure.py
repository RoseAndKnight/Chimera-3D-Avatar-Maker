"""Measures how an official ZZZ hairstyle is built: mesh, UVs, texture, physics.

    bin/blender-queue -b --factory-startup --python tools/hair/measure.py -- \
        --biblio <data>/bibliotheque-visage \
        --sortie <data>/cheveux [--seuls Anton,Jane] [--tous]
    <python> -I tools/hair/measure.py --tableau <sortie>
        (only rebuilds the table; needs numpy, hence Blender's Python)

Inputs: the head library (<biblio>/<P>/fiche.json, coiffure.npz from hairstyle.py: the
hairstyle's faces and the bones that weight only it). The official PMX is re-read (read-only).
Outputs (outside the repository): <sortie>/<P>.json per character, <sortie>/ensemble.json and
<sortie>/tableau.md. Blender is only used to read textures; without bpy, the texture part is
skipped and everything else is computed.

Frame: the PMX one (MMD units, Y up, face towards -Z, character's left at +X).
Lengths in H = head height (top of skull - chin, tete.Tete.H_crane). C = centre of the sphere
fitted to the head skin above the eyes (the skull).

Cutting units, from coarsest to finest:
  material  role "cheveux" (hair), "calque" (overlay: a copy laid on the hair, same vertices
            within 1 % of H: the "髮+" highlight layer), "accessoire" (accessory: everything else
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
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI, os.path.join(ICI, "..", "head"), os.path.join(ICI, "..", "atlas")]
sys.path.append(os.path.join(ICI, ".."))
import settings as R  # noqa: E402
import games as jeux  # noqa: E402

import geom as G  # noqa: E402
import pmx as PMX  # noqa: E402
import head as TT  # noqa: E402

try:
    import bpy  # noqa: F401
except ImportError:
    bpy = None

# hairstyles studied: chosen in the active game (description in tools/games/<game>/game.json, "etudes.cheveux")
CHOIX = tuple(jeux.actif().valeur("etudes.cheveux", ()))   # a game with no study: no hairstyle chosen
CHEVEUX = ("髮", "髪", "发", "毛")
ACCESSOIRE = ("饰", "飾", "带", "帽", "巾", "镜", "环", "耳", "宝", "金属", "机", "罩", "裙", "服", "工具", "下摆", "新規")
MIN_ILE = 12            # minimum vertices for an island to count as a strand
CALQUE_TOL = 0.01       # H: an overlay vertex lies on the hair if closer than this
CALQUE_PART = 0.8       # share of laid vertices for a material to be an overlay
TRANCHES = 20           # slices along s
SYM_TOL = 0.01          # H: a vertex has its X mirror within this distance
RIG_DYN = (1, 2)        # simulated rigid-body modes
ECHANT_TEX = 24         # texture samples per face (drawn in proportion to UV area)
OS_CORPS = ("頭", "首", "上半身", "上半身1", "上半身2", "上半身3", "右肩", "左肩")


def journal(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ecrire_json(chemin, obj):
    tmp = chemin + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=_json_def)
    os.replace(tmp, chemin)


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

def kd_plus_proche(ref, req):
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


def adjacence(n, F, V=None):
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
        du, u = heapq.heappop(h)
        if du > d[u]:
            continue
        for v, l in adj[u]:
            nd = du + l
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


def aretes_comptes(F):
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


def segment_rigide(rg):
    """Rigid body -> (a, b, radius): inner segment and radius (capsule; sphere: a = b;
    box: bounding capsule along the longest side, radius = diagonal of the cross-section)."""
    c = np.asarray(rg["pos"], float)
    R = rot_mmd(rg["rot"])
    t = rg["taille"]
    if rg["forme"] == 0:
        return c, c, float(t[0])
    if rg["forme"] == 2:
        ax = R[:, 1] * t[1] / 2
        return c - ax, c + ax, float(t[0])
    k = int(np.argmax(t))
    autres = [t[i] for i in range(3) if i != k]
    ax = R[:, k] * max(t[k] - min(autres), 0)
    return c - ax, c + ax, float(np.hypot(*autres))


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


def collisionnent(ra, rb):
    """Bullet: two bodies collide if each accepts the other's group (bit set)."""
    return bool((ra["masque"] >> rb["groupe"]) & 1) and bool((rb["masque"] >> ra["groupe"]) & 1)


def ignores(rg):
    return [g for g in range(16) if not (rg["masque"] >> g) & 1]


# =================================================================== loading

class Coiffure:
    def __init__(self, biblio, nom):
        self.nom = nom
        with open(os.path.join(biblio, nom, "fiche.json"), encoding="utf-8") as fh:
            self.chemin = os.path.join(biblio, nom, json.load(fh)["pmx"])  # relative (Outerplane) or absolute
        z = np.load(os.path.join(biblio, nom, "coiffure.npz"))
        self.masque_faces, self.masque_os = z["coiffure"], z["os"]
        self.D = D = TT.Tete(self.chemin)
        self.p, self.t = D.pmx, D.t
        self.pos, self.F, self.mf = D.pos, D.faces, D.mat_face
        self.noms_os = D.noms_os
        self.par = np.array([o["parent"] for o in self.p["os"]])
        # head frame
        yeux = 0.5 * (D.centre_oeil["g"][1] + D.centre_oeil["d"][1])
        cr = D.V[D.V[:, 1] > yeux]
        self.C, self.R = sphere(cr)
        self.H = float(D.H_crane)
        self.y_yeux = float(yeux)
        self.menton = D.V[D.reperes["menton"]]
        self._roles()

    def _roles(self):
        p, mf = self.p, self.mf
        mats = sorted(set(np.unique(mf[self.masque_faces]).tolist()))
        tex = lambda m: p["textures"][m["texture"]] if m["texture"] >= 0 else ""
        poils = []
        for i in mats:
            m = p["materiaux"][i]
            n, tx = m["nom"], os.path.basename(tex(m).replace("\\", "/"))
            if "+" in n:
                continue
            nom_ch = any(c in n for c in CHEVEUX) and not any(a in n for a in ACCESSOIRE)
            tex_ch = any(c in tx for c in CHEVEUX) and not any(a in n for a in ACCESSOIRE)
            if nom_ch or tex_ch:
                poils.append(i)
        vh = np.unique(self.F[np.isin(mf, poils) & self.masque_faces])
        self.roles = {}
        for i in mats:
            fk = np.where((mf == i) & self.masque_faces)[0]
            if i in poils:
                self.roles[i] = "cheveux"
                continue
            vk = np.unique(self.F[fk])
            d, _ = kd_plus_proche(self.pos[vh], self.pos[vk]) if len(vh) else (np.full(len(vk), np.inf), None)
            pose = float((d < CALQUE_TOL * self.H).mean())
            self.roles[i] = "calque" if pose >= CALQUE_PART else "accessoire"
        self.mats_cheveux = [i for i in mats if self.roles[i] == "cheveux"]
        self.f_ch = np.where(np.isin(mf, self.mats_cheveux) & self.masque_faces)[0]
        self.i_tete = self.noms_os.index("頭") if "頭" in self.noms_os else -1


# =================================================================== GLB input (our models)

# hair materials of a GLB: name containing one of these patterns (lower case), except eyebrows,
# eyelashes and accessories; the "+" copies and overlays are then found by geometry
MOTIFS_CHEVEUX_GLB = ("hair", "cheveu", "meche", "mèche", "frange", "coiffure", "髮", "髪", "发", "毛")
EXCLUS_CHEVEUX_GLB = ("brow", "lash", "sourcil", "cils", "眉", "睫")
MOTIFS_TETE_OS = ("頭", "head", "tete", "tête")
EXCLUS_TETE_OS = ("top", "end", "tip", "_fin", "bout", "nu")   # 頭先, HeadTop_End…


def _choisir_noms(noms, motifs, exclus=()):
    """Indices of the names containing a pattern ("=name": exact name) and no excluded pattern."""
    ok = lambda n, m: n.lower() == m[1:] if m.startswith("=") else m in n.lower()
    return [i for i, n in enumerate(noms) if any(ok(n, m) for m in motifs)
            and not any(e in n.lower() for e in exclus)]


class CoiffureGLB(Coiffure):
    """A hairstyle read from one or more GLBs (the model, plus any separate strands in the
    same frame and on the same skeleton, merged by bone name). Same attributes as Coiffure:
    measurements A to F run unchanged.

    - Positions: bind pose (mesh vertices, node world matrix); bones: origin of the mesh world
      matrix · inverse(IBM), i.e. the bind pose even if the GLB was exported posed.
    - Frame: the PMX one, via tete_glb.TeteGLB (eyes and face), units scaled to the ZZZ
      inter-eye distance. H and the skull as for PMX.
    - Hairstyle: hair materials (MOTIFS_CHEVEUX_GLB or --mats-cheveux), then the overlays
      (>= 80 % of vertices lying on the hair), then the bones that weight only them (rule from
      hairstyle.py, head excluded). Ornaments are not counted.
    - No rigid bodies or joints: physics comes from Godot springs (springs.py)."""

    def __init__(self, chemins, nom, mats_cheveux=None, peau=None, yeux=None):
        import head_glb as TG
        self.nom = nom
        self.chemin = chemins[0]
        self.chemins = list(chemins)
        self._lire(chemins, TG)
        self._repere(TG, peau, yeux)
        self._isoler(mats_cheveux)
        self._roles()
        self.i_tete = self._os_tete()
        self._crane_secours()

    # -------------------------------------------------------------- reading
    def _lire(self, chemins, TG):
        pos, nor, uv, uva, jo, wo, faces, mf = [], [], [], [], [], [], [], []
        mats, textures, self._images = [], [], []
        os_noms, os_par, os_M = [], [], []
        morphes = []
        base = 0
        for kf, ch in enumerate(chemins):
            j, binaire = TG.lire_glb(ch)
            mondes = TG._monde(j)
            parent = {c: i for i, nd in enumerate(j.get("nodes", [])) for c in nd.get("children", [])}
            # images and textures (glTF index -> index in self.p["textures"])
            t0 = len(textures)
            for ki, im in enumerate(j.get("images", [])):
                textures.append("%s#%d:%s" % (os.path.basename(ch), ki, im.get("name") or im.get("uri") or ""))
                self._images.append((j, binaire, ch, im))
            tex_img = [tx.get("source", -1) for tx in j.get("textures", [])]
            m0 = len(mats)
            for km, m in enumerate(j.get("materials", [])):
                pbr = m.get("pbrMetallicRoughness", {})
                bt = pbr.get("baseColorTexture", {}).get("index", -1)
                img = tex_img[bt] if 0 <= bt < len(tex_img) else -1
                nom_m = m.get("name", "mat%d" % km)
                if kf:
                    nom_m += " [%s]" % os.path.splitext(os.path.basename(ch))[0]
                mats.append({"nom": nom_m, "texture": t0 + img if img >= 0 else -1,
                             "drapeaux": PMX.MAT_DOUBLE_FACE if m.get("doubleSided") else 0,
                             "contour_taille": None, "contour_couleur": None,
                             "diffus": list(pbr.get("baseColorFactor", (1, 1, 1, 1))),
                             "alpha_mode": m.get("alphaMode", "OPAQUE"),
                             "sphere": -1, "sphere_mode": 0, "toon": -1, "toon_partage": 0})
            # bones: merged by name. They are the skin joints and everything hanging below them
            # that is not a mesh (weightless bones, which the export does not always put in
            # "joints": chain ends, roots); parent = closest ancestor that is a bone
            nds = j.get("nodes", [])
            jts = {jt for sk in j.get("skins", []) for jt in sk["joints"]}
            racines_sq = {x for x in jts if not any(a in jts for a in self._ancetres(parent, x))}
            est_os, pile = set(), list(racines_sq)
            while pile:
                x = pile.pop()
                if "mesh" in nds[x] or x in est_os:
                    continue
                est_os.add(x)
                pile += nds[x].get("children", [])
            ordre = sorted(est_os, key=lambda x: len(self._ancetres(parent, x)))
            for nd_i in ordre:
                nm = nds[nd_i].get("name", "os%d" % nd_i)
                if nm in os_noms:
                    continue
                p_ = parent.get(nd_i)
                while p_ is not None and p_ not in est_os:
                    p_ = parent.get(p_)
                os_noms.append(nm)
                os_par.append(nds[p_].get("name") if p_ is not None else None)
                os_M.append(None)
            for i, nd in enumerate(j.get("nodes", [])):
                if "mesh" not in nd:
                    continue
                M = mondes[i]
                sk = j["skins"][nd["skin"]] if "skin" in nd else None
                carte = None
                if sk is not None:
                    noms_sk = [j["nodes"][x].get("name", "os%d" % x) for x in sk["joints"]]
                    carte = np.array([os_noms.index(n) for n in noms_sk])
                    ibm = (TG.accesseur(j, binaire, sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
                           if "inverseBindMatrices" in sk else None)
                    for kk, g in enumerate(carte):
                        if os_M[g] is None:
                            os_M[g] = M @ np.linalg.inv(ibm[kk]) if ibm is not None else mondes[sk["joints"][kk]]
                mesh = j["meshes"][nd["mesh"]]
                noms_cibles = mesh.get("extras", {}).get("targetNames", [])
                for pr in mesh["primitives"]:
                    if pr.get("mode", 4) != 4:
                        continue
                    at = pr["attributes"]
                    acc = lambda k: TG.accesseur(j, binaire, at[k]).astype(np.float64)
                    P = acc("POSITION")
                    n = len(P)
                    P = P @ M[:3, :3].T + M[:3, 3]
                    Nn = acc("NORMAL") @ np.linalg.inv(M[:3, :3]) if "NORMAL" in at else np.zeros((n, 3))
                    Nn /= np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-12)
                    pos.append(P)
                    nor.append(Nn)
                    uv.append(acc("TEXCOORD_0") if "TEXCOORD_0" in at else np.zeros((n, 2)))
                    sup = []
                    for k in range(1, 8):
                        if "TEXCOORD_%d" % k not in at:
                            break
                        u = acc("TEXCOORD_%d" % k)
                        sup.append(np.c_[u, np.zeros((n, 2))])
                    uva.append(sup)
                    if carte is not None and "JOINTS_0" in at:
                        J = TG.accesseur(j, binaire, at["JOINTS_0"]).astype(np.int64)
                        W = acc("WEIGHTS_0")
                        W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
                        J = np.where(W > 0, carte[J], -1)
                    else:
                        J, W = np.full((n, 4), -1), np.zeros((n, 4))
                    jo.append(J)
                    wo.append(W)
                    I = (TG.accesseur(j, binaire, pr["indices"]).astype(np.int64).reshape(-1, 3)
                         if "indices" in pr else np.arange(n).reshape(-1, 3))
                    faces.append(I + base)
                    mf.append(np.full(len(I), m0 + pr.get("material", 0) if "material" in pr else -1))
                    for kt, tg in enumerate(pr.get("targets", [])):
                        if "POSITION" not in tg:
                            continue
                        d = TG.accesseur(j, binaire, tg["POSITION"]).astype(np.float64)
                        idx = np.where(np.linalg.norm(d, axis=1) > 1e-6)[0]
                        if len(idx):
                            morphes.append({"type": 1, "nom": noms_cibles[kt] if kt < len(noms_cibles) else "cible%d" % kt,
                                            "decalages": [(int(base + x), d[x]) for x in idx]})
                    base += n
            # bind pose of bones outside "joints": the parent's, then the local transform
            for nd_i in ordre:
                g = os_noms.index(nds[nd_i].get("name", "os%d" % nd_i))
                if os_M[g] is None:
                    p_ = parent.get(nd_i)
                    gp = os_noms.index(nds[p_].get("name", "os%d" % p_)) if p_ in est_os else None
                    os_M[g] = (os_M[gp] @ TG._matrice_noeud(nds[nd_i]) if gp is not None and os_M[gp] is not None
                               else mondes[nd_i])
        n_sup = max((len(s) for s in uva), default=0)
        uva = [np.stack(s + [np.zeros((len(p_), 4))] * (n_sup - len(s)), 1) if n_sup else np.zeros((len(p_), 0, 4))
               for s, p_ in zip(uva, pos)]
        self._brut = dict(pos=np.vstack(pos), nor=np.vstack(nor), uv=np.vstack(uv), uv_add=np.concatenate(uva),
                          os=np.vstack(jo), poids=np.vstack(wo), faces=np.vstack(faces), mat_face=np.concatenate(mf))
        self._souder_comme_pmx(morphes)
        self._os = (os_noms, os_par, os_M)
        self._mats, self._textures, self._morphes = mats, textures, morphes

    @staticmethod
    def _ancetres(parent, x):
        out = []
        while x in parent:
            x = parent[x]
            out.append(x)
        return out

    def _souder_comme_pmx(self, morphes):
        """A Blender glTF export splits a vertex wherever its corners differ (UV *or* normal);
        a PMX only splits it at UV seams. To measure like the ZZZ models (UV islands, M7 normal
        seams), vertices with the same position, UVs and weights are merged; their normal is
        the average. The share of merged vertices whose normals differed by more than 1 deg is
        recorded: these are hard edges outside UV seams."""
        b = self._brut
        cle = np.c_[np.round(b["pos"], 6), np.round(b["uv"], 6), b["os"], np.round(b["poids"], 4)]
        _, rep, inv = np.unique(cle, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        n = len(rep)
        N = np.zeros((n, 3))
        np.add.at(N, inv, b["nor"])
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        dev = np.degrees(np.arccos(np.clip((b["nor"] * N[inv]).sum(1), -1, 1)))
        ecl = np.zeros(n)
        np.maximum.at(ecl, inv, dev)
        multi = np.bincount(inv, minlength=n) > 1
        self.soudure = {"sommets_bruts": int(len(inv)), "sommets_soudes": int(n),
                        "reunis_normales_differentes_part": r3((ecl[multi] > 1).mean()) if multi.any() else 0.0,
                        "eclat_normales_deg": q(ecl[multi])}
        self._ecl_soudure = ecl
        for k in ("pos", "uv", "uv_add", "os", "poids"):
            b[k] = b[k][rep]
        b["nor"] = N
        b["faces"] = inv[b["faces"]]
        for m in morphes:
            vu = {}
            for i, d in m["decalages"]:
                vu.setdefault(int(inv[i]), d)
            m["decalages"] = list(vu.items())

    # -------------------------------------------------------------- frame
    def _repere(self, TG, peau, yeux):
        b = self._brut
        noms_mat = [m["nom"] for m in self._mats]
        ip = _choisir_noms(noms_mat, peau) if peau else None
        iy = _choisir_noms(noms_mat, yeux) if yeux else None
        D = TG.TeteGLB(self.chemin, nom=self.nom, peau=ip, yeux=iy, maillage_=(b["pos"], b["faces"], b["mat_face"], noms_mat))
        R, u = D.rotation, D.unite
        P0 = b["pos"]
        e = (P0 - (D.pos / u) @ R).mean(0)
        self.D, self.rotation, self.unite, self.origine = D, R, u, e
        tr = lambda P: (P - e) @ R.T * u
        pos = tr(P0)
        nor = b["nor"] @ R.T
        noms, par, Ms = self._os
        O = []
        for k, nm in enumerate(noms):
            M = Ms[k] if Ms[k] is not None else np.eye(4)
            O.append({"nom": nm, "parent": noms.index(par[k]) if par[k] in noms else -1,
                      "pos": tr(M[:3, 3][None])[0], "base": R @ (M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)),
                      "echelle_os": float(np.linalg.norm(M[:3, :3], axis=0).mean()),
                      "queue_dec": None})
        n = len(pos)
        self.t = {"pos": pos, "nor": nor, "uv": b["uv"], "uv_add": b["uv_add"], "os": b["os"], "poids": b["poids"],
                  "contour": np.full(n, np.nan), "type_poids": np.full(n, 2, dtype=np.int8),
                  "faces": b["faces"], "mat_face": b["mat_face"]}
        self.p = {"materiaux": self._mats, "textures": self._textures, "os": O, "rigides": [], "joints": [],
                  "morphes": [{"type": 1, "nom": m["nom"], "decalages": m["decalages"]} for m in self._morphes]}
        self.pos, self.F, self.mf = pos, b["faces"], b["mat_face"]
        self.noms_os = noms
        self.par = np.array([o["parent"] for o in O], dtype=np.int64)
        yeux_y = 0.5 * (D.centre_oeil["g"][1] + D.centre_oeil["d"][1])
        self.y_yeux = float(yeux_y)
        self.H = float(D.H_crane)
        self.menton = D.V[D.reperes["menton"]]
        # skull: the face skin above the eyes, as for PMX; if it only covers the forehead (face
        # separate from the skull), the sphere is badly placed: we then take all the skin
        # weighted to the head bone above the eyes (excluding hair, eyes, eyebrows)
        cr = D.V[D.V[:, 1] > yeux_y]
        self.C, self.R = sphere(cr)
        self.repere_crane = "face skin"

    def _crane_secours(self):
        """If the face-skin sphere does not have a skull radius (ZZZ: 0.36-0.45 H), refit it on
        all the skin weighted to the head bone above the eyes, excluding the hairstyle and the
        eye, eyebrow and eyelash materials."""
        if 0.3 <= self.R / self.H <= 0.55 or self.i_tete < 0:
            return
        noms = [m["nom"] for m in self.p["materiaux"]]
        exclus = set(_choisir_noms(noms, ("eye", "oeil", "œil", "yeux", "brow", "lash", "sourcil", "cil",
                                          "teeth", "dent", "tongue", "langue", "mouth", "bouche", "目", "眉", "睫", "齿", "舌")))
        exclus |= set(np.unique(self.mf[self.masque_faces]).tolist())
        vk = np.unique(self.F[~np.isin(self.mf, list(exclus))])
        dom = self.t["os"][vk, np.argmax(self.t["poids"][vk], 1)]
        P = self.pos[vk[(dom == self.i_tete) & (self.pos[vk, 1] > self.y_yeux)]]
        if len(P) < 30:
            return
        C, R = sphere(P)
        journal("%s: skull refitted on the head-bone skin (R/H %.2f -> %.2f)" % (self.nom, self.R / self.H, R / self.H))
        self.C, self.R = C, R
        self.repere_crane = "skin weighted to the head bone"

    def _os_tete(self):
        noms = self.noms_os
        cand = _choisir_noms(noms, MOTIFS_TETE_OS, EXCLUS_TETE_OS)
        if "頭" in noms:
            return noms.index("頭")
        # among the candidates, the one most hairstyle bones descend from
        if cand:
            oc = np.where(self.masque_os)[0]
            return max(cand, key=lambda i: sum(1 for o in oc if self._ancetre(o, i)))
        racines = [int(self.par[o]) for o in np.where(self.masque_os)[0] if self.par[o] >= 0 and not self.masque_os[self.par[o]]]
        return max(set(racines), key=racines.count) if racines else -1

    def _ancetre(self, o, a):
        while o >= 0:
            if o == a:
                return True
            o = int(self.par[o])
        return False

    # -------------------------------------------------------------- hairstyle
    def _isoler(self, motifs):
        noms = [m["nom"] for m in self._mats]
        poils = (_choisir_noms(noms, motifs, ()) if motifs else
                 [i for i in _choisir_noms(noms, MOTIFS_CHEVEUX_GLB, EXCLUS_CHEVEUX_GLB)
                  if not any(a in noms[i] for a in ACCESSOIRE)])
        if not poils:
            raise ValueError("%s: no hair material among %s" % (self.nom, noms))
        mf, F = self.mf, self.F
        coif = np.isin(mf, poils)
        vh = np.unique(F[coif])
        # overlays: materials with >= 80 % of their vertices lying on the hair
        for i in sorted(set(np.unique(mf).tolist()) - set(poils)):
            vk = np.unique(F[mf == i])
            d, _ = kd_plus_proche(self.pos[vh], self.pos[vk])
            if (d < CALQUE_TOL * self.H).mean() >= CALQUE_PART:
                coif |= mf == i
        # hairstyle bones (rule from hairstyle.py): they weight it and almost nothing else
        n_os = len(self.noms_os)

        def masse(s):
            o, w = self.t["os"][s], self.t["poids"][s]
            m = np.zeros(n_os)
            ok = o >= 0
            np.add.at(m, o[ok], w[ok])
            return m
        par = self.par
        tete_nom = lambda k: any(m in self.noms_os[k].lower() for m in MOTIFS_TETE_OS)
        dom_all = self.t["os"][np.arange(len(self.pos)), np.argmax(self.t["poids"], 1)]

        tete = np.array([tete_nom(k) for k in range(n_os)], bool)

        def os_de(coif):
            vc = np.unique(F[coif])
            hors = np.setdiff1d(np.arange(len(self.pos)), vc)
            Wc, Wn = masse(vc), masse(hors)
            graines = (Wc > 0) & (Wn <= 0.02 * Wc) & ~tete
            sans_poids = (Wc + Wn) <= 0
            os_c = graines.copy()
            for _ in range(128):
                nd = os_c | (sans_poids & (par >= 0) & os_c[np.maximum(par, 0)])
                if (nd == os_c).all():
                    break
                os_c = nd
            for i in np.where(graines)[0]:
                p_ = par[i]
                while p_ >= 0 and sans_poids[p_] and not os_c[p_] and not tete_nom(p_):
                    os_c[p_] = True
                    p_ = par[p_]
            return os_c, Wc
        # Ornaments are not added: a round-trip GLB may mix ornaments, belt and badge in a single
        # material (Qingyi: 饰1, weighted mostly by Ctr_Belt and Ctr_Badge). The bones weighting
        # these ornaments therefore leave the hairstyle (Qingyi: Ctr_HairBD, Ctr_HairBA_01); a
        # known difference from the PMX library, where 饰1 counts as part of the hairstyle.
        os_c, _ = os_de(coif)
        self.masque_faces, self.masque_os = coif, os_c
        # for _roles: hair materials matched by pattern are "cheveux" even without CHEVEUX
        self._poils_glb = poils

    def _roles(self):
        p, mf = self.p, self.mf
        mats = sorted(set(np.unique(mf[self.masque_faces]).tolist()))
        poils = [i for i in self._poils_glb if "+" not in p["materiaux"][i]["nom"]]
        vh = np.unique(self.F[np.isin(mf, poils) & self.masque_faces])
        self.roles = {}
        for i in mats:
            if i in poils:
                self.roles[i] = "cheveux"
                continue
            vk = np.unique(self.F[(mf == i) & self.masque_faces])
            d, _ = kd_plus_proche(self.pos[vh], self.pos[vk])
            self.roles[i] = "calque" if (d < CALQUE_TOL * self.H).mean() >= CALQUE_PART else "accessoire"
        self.mats_cheveux = [i for i in mats if self.roles[i] == "cheveux"]
        self.f_ch = np.where(np.isin(mf, self.mats_cheveux) & self.masque_faces)[0]

    # -------------------------------------------------------------- embedded textures
    def image(self, nom):
        """Image of a GLB texture (embedded or alongside), as an (h, w, 4) array; None without bpy."""
        if bpy is None or nom not in self.p["textures"]:
            return None
        j, binaire, ch, im = self._images[self.p["textures"].index(nom)]
        if "bufferView" in im:
            bv = j["bufferViews"][im["bufferView"]]
            octets = binaire[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
            ext = ".png" if "png" in im.get("mimeType", "png") else ".jpg"
            d = os.path.join(os.environ.get("TMPDIR", "/tmp"), "mesure-cheveux-images")
            os.makedirs(d, exist_ok=True)
            fic = os.path.join(d, "%s-%d%s" % (self.nom, self.p["textures"].index(nom), ext))
            with open(fic + ".tmp", "wb") as fh:
                fh.write(octets)
            os.replace(fic + ".tmp", fic)
        elif "uri" in im and not im["uri"].startswith("data:"):
            fic = os.path.join(os.path.dirname(ch), im["uri"])
        else:
            return None
        try:
            img = bpy.data.images.load(fic)
            w, h = img.size
            px = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(px)
            bpy.data.images.remove(img)
            return px.reshape(h, w, 4)
        except Exception as e:
            journal("unreadable texture", fic, e)
            return None


# =================================================================== A. mesh

def mesurer_maillage(c):
    H, C = c.H, c.C
    F = c.F[c.f_ch]
    vh = np.unique(F)
    P = c.pos[vh]
    loc = np.full(len(c.pos), -1)
    loc[vh] = np.arange(len(vh))
    Fl = loc[F]
    rep, inv = G.souder(P, 1e-5)
    Fw = inv[Fl]
    ok = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
    Vw = P[rep]
    E, cnt, fe = aretes_comptes(Fw[ok])
    bord_w = np.zeros(len(Vw), bool)
    bord_w[E[cnt == 1].ravel()] = True
    lab = G.composantes_faces(Fw, ok)
    pieces = []
    dist_cr = (np.linalg.norm(Vw - C, axis=1) - c.R) / H
    for k in range(lab.max() + 1):
        fk = np.where(lab == k)[0]
        vk = np.unique(Fw[fk])
        Ek, ck, _ = aretes_comptes(Fw[fk])
        pieces.append({"faces": int(len(fk)), "sommets": int(len(vk)),
                       "aretes_bord": int((ck == 1).sum()), "aretes_non_manifold": int((ck > 2).sum()),
                       "part_bord": r3((ck == 1).sum() / max(len(Ek), 1)),
                       "euler": int(len(vk) - len(Ek) + len(fk)),
                       "dist_crane_med_H": r3(np.median(dist_cr[vk])),
                       "bas_H": r3((c.C[1] - Vw[vk, 1].min()) / H),
                       "materiaux": sorted({c.p["materiaux"][c.mf[c.f_ch[f]]]["nom"] for f in fk})})
    pieces.sort(key=lambda x: -x["faces"])
    # cap: vertices above the eyes within 0.12 H of the skull
    haut = Vw[:, 1] > c.y_yeux
    cal = haut & (dist_cr > -0.03) & (dist_cr < 0.12)
    # pairs of opposite faces (explicit double-sided)
    Fs = np.sort(Fw[ok], 1)
    _, cc = np.unique(Fs, axis=0, return_counts=True)
    # faces flipped relative to the skull: normal towards C (on the cap only)
    Fo = Fw[ok]
    nf = G.normales_faces(Vw, Fo)
    cf = Vw[Fo].mean(1)
    sur_cal = cal[Fo].all(1)
    vers_c = ((nf * (cf - C)).sum(1) < 0)
    # PMX normals: compared with the welded geometric normal and the radial direction
    nw = G.normales_sommets(Vw, Fo)
    Np = c.t["nor"][vh]
    Np = Np / np.maximum(np.linalg.norm(Np, axis=1, keepdims=True), 1e-12)
    ang_geo = np.degrees(np.arccos(np.clip((Np * nw[inv]).sum(1), -1, 1)))
    rad = P - C
    rad /= np.linalg.norm(rad, axis=1, keepdims=True)
    ang_rad = np.degrees(np.arccos(np.clip((Np * rad).sum(1), -1, 1)))
    # normal seams: coincident vertices whose normals differ by more than 1 deg
    eclat = np.zeros(len(Vw))
    groupes = np.bincount(inv)
    nmoy = np.zeros((len(Vw), 3))
    np.add.at(nmoy, inv, Np)
    nmoy /= np.maximum(np.linalg.norm(nmoy, axis=1, keepdims=True), 1e-12)
    dev = np.degrees(np.arccos(np.clip((Np * nmoy[inv]).sum(1), -1, 1)))
    np.maximum.at(eclat, inv, dev)
    multi = groupes > 1
    # symmetry: X mirror of each welded vertex
    d_sym, j_sym = kd_plus_proche(Vw, Vw * [-1, 1, 1])
    # budget
    tot_v, tot_f = len(c.pos), len(c.F)
    roles = {}
    for i, rl in c.roles.items():
        fk = np.where((c.mf == i) & c.masque_faces)[0]
        m = c.p["materiaux"][i]
        roles[m["nom"]] = {"role": rl, "faces": int(len(fk)), "sommets": int(len(np.unique(c.F[fk]))),
                           "double_face": bool(m["drapeaux"] & PMX.MAT_DOUBLE_FACE),
                           "contour": bool(m["drapeaux"] & PMX.MAT_CONTOUR),
                           "contour_taille": r3(m["contour_taille"]), "contour_couleur": r3(m["contour_couleur"]),
                           "alpha": r3(m["diffus"][3]),
                           "texture": c.p["textures"][m["texture"]] if m["texture"] >= 0 else None,
                           "sphere": c.p["textures"][m["sphere"]] if m["sphere"] >= 0 else None,
                           "sphere_mode": m["sphere_mode"],
                           "toon": ("shared %d" % m["toon"]) if m["toon_partage"] else (
                               c.p["textures"][m["toon"]] if m["toon"] >= 0 else None)}
    # overlays: offset along the normal relative to the hair
    calques = {}
    for i, rl in c.roles.items():
        if rl != "calque":
            continue
        vk = np.unique(c.F[(c.mf == i) & c.masque_faces])
        d, j = kd_plus_proche(P, c.pos[vk])
        dec = ((c.pos[vk] - P[j]) * Np[j]).sum(1)
        calques[c.p["materiaux"][i]["nom"]] = {
            "sommets": int(len(vk)), "part_sommets_cheveux_couverts": r3(len(np.unique(inv[j][d < 1e-4 * H])) / len(Vw)),
            "dist_med_H": r3(np.median(d) / H), "decalage_normal_med_H": r3(np.median(dec) / H),
            "meme_uv": r3(float((np.linalg.norm(c.t["uv"][vk] - c.t["uv"][vh][j], axis=1) < 1e-4).mean())),
            "y_bas_H": r3((C[1] - c.pos[vk, 1].min()) / H)}
    contour_v = c.t["contour"][vh]
    out = {
        "budget": {"sommets_modele": tot_v, "faces_modele": tot_f, "sommets_cheveux": int(len(vh)),
                   "sommets_cheveux_soudes": int(len(Vw)), "faces_cheveux": int(len(F)),
                   "part_faces": r3(len(F) / tot_f), "faces_coiffure_entiere": int(c.masque_faces.sum()),
                   "faces_calques": int(sum(v["faces"] for v in roles.values() if v["role"] == "calque")),
                   "faces_accessoires": int(sum(v["faces"] for v in roles.values() if v["role"] == "accessoire")),
                   "sommets_par_face": r3(len(Vw) / max(len(F), 1))},
        "materiaux": roles,
        "calques": calques,
        "pieces": {"nombre": len(pieces), "faces_plus_grande": pieces[0]["faces"] if pieces else 0,
                   "part_plus_grande": r3(pieces[0]["faces"] / len(F)) if pieces else None,
                   "fermees": int(sum(1 for x in pieces if x["aretes_bord"] == 0)),
                   "non_manifold": int(sum(x["aretes_non_manifold"] for x in pieces)),
                   "tailles": [x["faces"] for x in pieces[:40]], "detail": pieces[:25]},
        "aretes_bord_part": r3((cnt == 1).sum() / len(E)),
        "faces_degenerees": int((~ok).sum()),
        "paires_faces_opposees": int((cc > 1).sum()),
        "calotte": {"sommets": int(cal.sum()), "part_haut": r3(cal[haut].mean()) if haut.any() else None,
                    "faces_vers_crane_part": r3(vers_c[sur_cal].mean()) if sur_cal.any() else None,
                    "dist_crane_H": q(dist_cr[haut])},
        "normales": {"angle_geometrique_deg": q(ang_geo), "angle_radial_deg": q(ang_rad),
                     "coutures_part": r3((eclat[multi] > 1).mean()) if multi.any() else 0.0,
                     "sommets_doubles_part": r3(multi.mean())},
        "contour_sommets": q(contour_v),
        "symetrie": {"part_1pc": r3((d_sym < SYM_TOL * H).mean()), "part_3pc": r3((d_sym < 3 * SYM_TOL * H).mean()),
                     "hausdorff_H": r3(d_sym.max() / H)},
    }
    c.Vw, c.inv_w, c.vh, c.Fw, c.ok_w, c.bord_w, c.Np = Vw, inv, vh, Fw, ok, bord_w, Np
    c.d_sym, c.j_sym = d_sym, j_sym
    return out


# =================================================================== B. islands and strands

def mesurer_meches(c):
    H, C = c.H, c.C
    F = c.F[c.f_ch]
    loc = np.full(len(c.pos), -1)
    loc[c.vh] = np.arange(len(c.vh))
    Fl = loc[F]
    lab = G.composantes_faces(Fl, np.ones(len(Fl), bool))
    uv = c.t["uv"][c.vh]
    P = c.pos[c.vh]
    W_os, W_p = c.t["os"][c.vh], c.t["poids"][c.vh]
    dom = W_os[np.arange(len(W_os)), np.argmax(W_p, 1)]
    meches = []
    s_glob = np.full(len(c.vh), np.nan)
    ile_de = np.full(len(c.vh), -1)
    for k in range(lab.max() + 1):
        fk = np.where(lab == k)[0]
        vk = np.unique(Fl[fk])
        ile_de[vk] = k
        if len(vk) < MIN_ILE:
            continue
        m = _meche(c, k, fk, vk, Fl, P, uv, dom)
        if m is None:
            continue
        s_glob[vk] = m.pop("_s")
        meches.append(m)
    c.s_glob, c.ile_de, c.dom = s_glob, ile_de, dom
    c.meches = meches
    return meches


def _meche(c, k, fk, vk, Fl, P, uv, dom):
    H, C = c.H, c.C
    loc = np.full(len(P), -1)
    loc[vk] = np.arange(len(vk))
    Fk = loc[Fl[fk]]
    X = P[vk]
    adj = adjacence(len(vk), Fk, X)
    dC = np.linalg.norm(X - C, axis=1)
    src = np.where(dC <= dC.min() + 0.03 * H)[0]
    g = dijkstra(adj, src)
    if not np.isfinite(g).all() or g.max() <= 0:
        return None
    s = g / g.max()
    hop = bfs(adj, src)
    anneaux = int(hop.max() + 1)
    # slices along s
    b = np.minimum((s * TRANCHES).astype(int), TRANCHES - 1)
    cent = [X[b == i].mean(0) if (b == i).sum() >= 2 else None for i in range(TRANCHES)]
    larg, ep = [np.nan] * TRANCHES, [np.nan] * TRANCHES
    cent_ok = [i for i in range(TRANCHES) if cent[i] is not None]
    for i in range(TRANCHES):
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
        larg[i] = np.ptp(pr[:, 0]) / H
        ep[i] = (np.ptp(pr[:, 1]) / H) if pr.shape[1] > 1 else 0.0
    cc = np.array([cent[i] for i in cent_ok])
    long_H = float(np.linalg.norm(np.diff(cc, axis=0), axis=1).sum() / H) if len(cc) > 1 else 0.0
    larg, ep = np.array(larg), np.array(ep)
    lmed = np.nanmedian(larg)
    racine, pointe = X[src].mean(0), X[np.argmax(s)]
    # ring spacing: mean g per level
    gm = np.array([g[hop == h].mean() for h in range(anneaux)])
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
    bord = float(c.bord_w[vw].mean())
    # faces turned towards the strand axis (back of a ribbon, tucked-in rim)
    nf = G.normales_faces(X, Fk)
    cf = t3.mean(1)
    sf = s[Fk].mean(1)
    bi = np.minimum((sf * TRANCHES).astype(int), TRANCHES - 1)
    axe_pt = np.array([cent[i] if cent[i] is not None else X.mean(0) for i in range(TRANCHES)])[bi]
    vers_axe = float(((nf * (cf - axe_pt)).sum(1) < 0).mean())
    # dominant bones
    od = dom[vk]
    noms = c.noms_os
    tete = c.i_tete
    part_tete = float((od == tete).mean())
    vals, cnts = np.unique(od, return_counts=True)
    mats = sorted({c.p["materiaux"][c.mf[c.f_ch[f]]]["nom"] for f in fk})
    rel = lambda p: (p - C) / H
    rr, pp = rel(racine), rel(pointe)
    # position category (MMD frame: front is -Z)
    if pp[1] < -1.0:
        cat = "longue"
    elif rr[2] < -0.15 and pp[2] < 0 and abs(rr[0]) < 0.35:
        cat = "frange"
    elif abs(rr[0]) >= 0.35 or abs(pp[0]) >= 0.45:
        cat = "cote"
    elif rr[2] > 0.15:
        cat = "arriere"
    else:
        cat = "dessus"
    return {"ile": int(k), "sommets": int(len(vk)), "faces": int(len(fk)), "materiaux": mats, "categorie": cat,
            "racine_H": r3(rr), "pointe_H": r3(pp), "longueur_H": r3(long_H),
            "largeur_med_H": r3(lmed), "epaisseur_med_H": r3(np.nanmedian(ep)),
            "aplatissement": r3(np.nanmedian(ep) / lmed) if lmed > 0 else None,
            "largeur_pointe_rel": r3(np.nanmean(larg[-3:]) / lmed) if lmed > 0 else None,
            "largeur_racine_rel": r3(np.nanmean(larg[:3]) / lmed) if lmed > 0 else None,
            "anneaux": anneaux, "sommets_par_anneau": r3(len(vk) / anneaux, 1),
            "espacement_med_H": r3(np.median(esp)) if len(esp) else None,
            "espacement_pointe_rel": r3(np.mean(esp[-n4:]) / np.median(esp)) if len(esp) > 3 and np.median(esp) > 0 else None,
            "corr_v_s": r3(cv), "corr_u_s": r3(cu), "angle_axe_uv_v_deg": r3(ang_v, 1),
            "uv_boite": r3([np.ptp(U[:, 0]), np.ptp(U[:, 1])]),
            "densite_texel": r3(np.sqrt(auv / a3) * c.H) if a3 > 0 else None,
            "bord_libre_part": r3(bord), "faces_vers_axe_part": r3(vers_axe),
            "os_tete_part": r3(part_tete), "os_dominant": noms[int(vals[np.argmax(cnts)])],
            "os_dominant_part": r3(cnts.max() / len(vk)), "_s": s}


# =================================================================== C. UV (overall)

def mesurer_uv(c):
    uv = c.t["uv"][c.vh]
    F = c.F[c.f_ch]
    loc = np.full(len(c.pos), -1)
    loc[c.vh] = np.arange(len(c.vh))
    Fl = loc[F]
    hors = float(((uv < -1e-3) | (uv > 1 + 1e-3)).any(1).mean())
    # stacking: pixels (256²) covered by more than one island
    N = 256
    rng = np.random.default_rng(0)
    cnt = {}
    for k in np.unique(c.ile_de):
        if k < 0:
            continue
        fk = np.where(c.ile_de[Fl[:, 0]] == k)[0]
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
    paires = np.where(m)[0]
    paires = paires[c.j_sym[paires] != paires]
    du = np.linalg.norm(uv[rep_raw[paires]] - uv[rep_raw[c.j_sym[paires]]], axis=1) if len(paires) else np.zeros(0)
    # extra UV layers: nature of each channel
    couches = []
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
        couches.append({"canal": k + 1, "nature": nature, "min": r3(u.min(0)), "max": r3(u.max(0)),
                        "valeurs_distinctes": int(len(np.unique(np.round(u, 3), axis=0))),
                        "corr_max_avec_s": r3(corr)})
    return {"hors_0_1_part": r3(hors), "pixels_couverts_part": r3(len(cnt) / N / N),
            "pixels_empiles_part": r3((v > 1).sum() / max(len(v), 1)),
            "miroir_uv_identiques_part": r3((du < 1e-3).mean()) if len(du) else None,
            "miroir_paires": int(len(du)), "couches_supplementaires": couches}


# =================================================================== D. texture

def _dossier_pmx(c):
    return os.path.dirname(c.chemin)


def _lire_texture(c, nom):
    if bpy is None or not nom:
        return None
    if hasattr(c, "image"):          # GLB: embedded textures
        return c.image(nom)
    ch = os.path.join(_dossier_pmx(c), nom.replace("\\", "/"))
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
        journal("unreadable texture", ch, e)
        return None


def _L(rgb):
    """CIE lightness L* of an sRGB [0,1] colour."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(Y > 216 / 24389, np.cbrt(Y), (24389 / 27 * Y + 16) / 116)
    return 116 * f - 16


def _echantillonner(img, uv, Fl, valeur, rng, n_par_face=ECHANT_TEX):
    """Points drawn in each face (random barycentric coordinates): colour (RGBA) and an
    interpolated value (for example s)."""
    T = uv[Fl]
    n = len(Fl) * n_par_face
    j = np.repeat(np.arange(len(Fl)), n_par_face)
    r1, r2 = rng.random(n), rng.random(n)
    sq = np.sqrt(r1)
    bb = np.stack([1 - sq, sq * (1 - r2), sq * r2], 1)
    p = np.einsum("nk,nkd->nd", bb, T[j])
    val = (bb * valeur[Fl[j]]).sum(1)
    h, w = img.shape[:2]
    x = (np.mod(p[:, 0], 1) * w).astype(int).clip(0, w - 1)
    y = ((1 - np.mod(p[:, 1], 1)) * h).astype(int).clip(0, h - 1)
    return img[y, x], val, j


def mesurer_texture(c):
    if bpy is None:
        return {"lu": False, "raison": "without Blender"}
    rng = np.random.default_rng(1)
    loc = np.full(len(c.pos), -1)
    loc[c.vh] = np.arange(len(c.vh))
    uv = c.t["uv"][c.vh]
    out = {"lu": True, "par_materiau": {}}
    s = c.s_glob.copy()
    profils = []
    for i in c.mats_cheveux:
        m = c.p["materiaux"][i]
        nomtex = c.p["textures"][m["texture"]] if m["texture"] >= 0 else None
        img = _lire_texture(c, nomtex)
        if img is None:
            out["par_materiau"][m["nom"]] = {"texture": nomtex, "lue": False}
            continue
        fk = np.where((c.mf == i) & c.masque_faces)[0]
        Fl = loc[c.F[fk]]
        okf = np.isfinite(s[Fl]).all(1)
        col, sv, jf = _echantillonner(img, uv, Fl[okf], np.nan_to_num(s), rng)
        if not len(col):
            out["par_materiau"][m["nom"]] = {"texture": nomtex, "lue": True, "echantillons": 0}
            continue
        L = _L(np.clip(col[:, :3], 0, 1))
        A = col[:, 3]
        ile = c.ile_de[Fl[okf][jf, 0]]
        bins = np.minimum((sv * 10).astype(int), 9)
        prof_L = [r3(np.median(L[bins == b]), 1) if (bins == b).any() else None for b in range(10)]
        prof_A = [r3(np.mean(A[bins == b]), 2) if (bins == b).any() else None for b in range(10)]
        # per island: root -> tip slope and light bump (painted highlight)
        pentes, bosses = [], []
        for k in np.unique(ile):
            mk = ile == k
            if mk.sum() < 200:
                continue
            bk = bins[mk]
            pr = np.array([np.median(L[mk][bk == b]) if (bk == b).sum() >= 5 else np.nan for b in range(10)])
            okb = np.isfinite(pr)
            if okb.sum() < 5:
                continue
            xs = np.arange(10)[okb]
            a_, b_ = np.polyfit(xs, pr[okb], 1)
            pentes.append(pr[okb][-1] - pr[okb][0])
            bosses.append(float((pr[okb] - (a_ * xs + b_)).max()))
            profils.append(pr.tolist())
        # lines painted on the texture: lightness difference between neighbours (whole texture, under the UVs)
        h, w = img.shape[:2]
        Lt = _L(np.clip(img[..., :3], 0, 1))
        grad = np.hypot(np.diff(Lt, axis=0)[:, :-1], np.diff(Lt, axis=1)[:-1, :])
        px = (np.clip((np.mod(uv[np.unique(Fl)], 1) * [w, h]).astype(int), 0, [w - 1, h - 1]))
        out["par_materiau"][m["nom"]] = {
            "texture": nomtex, "lue": True, "taille": [int(w), int(h)],
            "L_med": r3(np.median(L), 1), "L_p5_p95": r3(np.percentile(L, [5, 95]), 1),
            "alpha_moy": r3(A.mean()), "alpha_sous_0_9_part": r3((A < 0.9).mean()),
            "profil_L_s": prof_L, "profil_alpha_s": prof_A,
            "pente_L_racine_pointe": q(pentes, n=1), "bosse_claire_L": q(bosses, n=1),
            "iles_pointe_plus_sombre_part": r3(np.mean(np.array(pentes) < -3)) if pentes else None,
            "gradient_L_texture_p90": r3(np.percentile(grad, 90), 1)}
    # overlays (highlight): texture, opaque coverage, position along s
    for i, rl in c.roles.items():
        if rl != "calque":
            continue
        m = c.p["materiaux"][i]
        nomtex = c.p["textures"][m["texture"]] if m["texture"] >= 0 else None
        img = _lire_texture(c, nomtex)
        fk = np.where((c.mf == i) & c.masque_faces)[0]
        vk = np.unique(c.F[fk])
        d, j = kd_plus_proche(c.pos[c.vh], c.pos[vk])
        sv = np.full(len(c.pos), np.nan)
        sv[vk] = s[j]
        info = {"texture": nomtex, "sphere": c.p["textures"][m["sphere"]] if m["sphere"] >= 0 else None,
                "sphere_mode": m["sphere_mode"]}
        if img is not None:
            loc2 = np.full(len(c.pos), -1)
            loc2[vk] = np.arange(len(vk))
            Fl2 = loc2[c.F[fk]]
            col, s2, _ = _echantillonner(img, c.t["uv"][vk], Fl2, np.nan_to_num(sv[vk]), rng, 8)
            A = col[:, 3]
            L = _L(np.clip(col[:, :3], 0, 1))
            vis = A > 0.5
            info.update({"taille": list(img.shape[1::-1]), "alpha_moy": r3(A.mean()), "visible_part": r3(vis.mean()),
                         "L_visible_med": r3(np.median(L[vis]), 1) if vis.any() else None,
                         "s_visible": q(s2[vis]) if vis.any() else None})
        out.setdefault("calques", {})[m["nom"]] = info
    out["textures_modele"] = sorted(set(c.p["textures"]))
    return out


# =================================================================== E. physics

def mesurer_physique(c):
    H, C = c.H, c.C
    p = c.p
    O, RG, J = p["os"], p["rigides"], p["joints"]
    noms = c.noms_os
    oc_tout = c.masque_os
    lest = np.array([("錘" in n or "锤" in n) for n in noms])   # "髮1錘", "Ctr_HairA_01錘_L"
    oc = oc_tout & ~lest          # links: hairstyle bones except the weights
    enfants = {}
    for i, o in enumerate(O):
        enfants.setdefault(o["parent"], []).append(i)
    rig_os = {}
    for k, r in enumerate(RG):
        rig_os.setdefault(r["os"], []).append(k)
    j_par_b = {}
    for jj in J:
        j_par_b.setdefault(jj["b"], []).append(jj)
    racines = [i for i in np.where(oc)[0] if O[i]["parent"] < 0 or not oc[O[i]["parent"]]]
    W_os, W_p = c.t["os"][c.vh], c.t["poids"][c.vh]
    dom = c.dom
    corps = [k for k, r in enumerate(RG) if r["mode"] == 0 and r["os"] >= 0 and noms[r["os"]] in OS_CORPS]
    tete_rg = [k for k in corps if noms[RG[k]["os"]] == "頭"]
    seg_corps = {k: segment_rigide(RG[k]) for k in corps}

    def plus_long(i):
        ch = [i]
        while True:
            e = [x for x in enfants.get(ch[-1], []) if oc[x]]
            if not e:
                return ch
            ch.append(max(e, key=lambda x: len(plus_long(x))))

    chaines = []
    for r0 in racines:
        ch = plus_long(r0)
        sous = [r0]
        k = 0
        while k < len(sous):
            sous += [x for x in enfants.get(sous[k], []) if oc_tout[x]]
            k += 1
        lests = [x for x in sous if lest[x]]
        m_lest = [RG[rig_os[x][0]]["masse"] for x in lests if rig_os.get(x)]
        sous_maillons = [x for x in sous if not lest[x]]
        pos = np.array([O[i]["pos"] for i in ch])
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1) / H
        rgs = [rig_os.get(i, []) for i in ch]
        r1 = [RG[x[0]] if x else None for x in rgs]
        dyn = [rr is not None and rr["mode"] in RIG_DYN for rr in r1]
        premier = next((i for i, d_ in enumerate(dyn) if d_), None)
        masses = [rr["masse"] if rr else None for rr in r1]
        rapp = [masses[i + 1] / masses[i] for i in range(len(masses) - 1)
                if masses[i] and masses[i + 1] and dyn[i] and dyn[i + 1]]
        # joints attached to the chain's bodies
        jts = []
        for rr in rgs:
            if not rr:
                continue
            for jj in j_par_b.get(rr[0], []):
                jts.append(jj)
        lim = [[np.degrees(jj["rot_min"][a]), np.degrees(jj["rot_max"][a])] for jj in jts for a in range(3)]
        # vertices carried by the chain (dominant bone in the subtree)
        sous_set = np.isin(dom, sous)
        n_v = int(sous_set.sum())
        # body radius / strand half-width at that bone
        rapports, ecarts = [], []
        for i, rr in zip(ch, r1):
            vv = c.pos[c.vh][dom == i]
            if rr is None or len(vv) < 6:
                continue
            a = np.asarray(O[i]["pos"])
            e = [x for x in enfants.get(i, []) if oc[x]]
            b = np.asarray(O[e[0]]["pos"]) if e else (a + np.asarray(O[i]["queue_dec"] or (0, -1e-3, 0)))
            ax = b - a
            if np.linalg.norm(ax) < 1e-9:
                continue
            ax /= np.linalg.norm(ax)
            rel = vv - a
            perp = rel - np.outer(rel @ ax, ax)
            dp = np.linalg.norm(perp, axis=1)
            demi = np.percentile(dp, 90)
            _, _, rayon = segment_rigide(rr)
            if demi > 0:
                rapports.append(rayon / demi)
            ecarts.append(np.linalg.norm(perp.mean(0)) / H)
        # collisions with the head and the body
        coll_tete = [collisionnent(rr, RG[k]) for rr in r1 if rr is not None and rr["mode"] in RIG_DYN for k in tete_rg]
        # at rest: overlap of the chain's bodies with those of the head and body
        rec, rec_libre = [], []
        for rr in r1:
            if rr is None or rr["mode"] not in RIG_DYN:
                continue
            a, b, ra = segment_rigide(rr)
            for k in corps:
                a2, b2, r2 = seg_corps[k]
                d = (dist_segments(a, b, a2, b2) - ra - r2) / H
                (rec if collisionnent(rr, RG[k]) else rec_libre).append(d)
        # chain root relative to the skull; reach of the carried vertices along the chain
        racine_crane = (np.linalg.norm(pos[0] - C) - c.R) / H
        vp = c.pos[c.vh][sous_set]
        portee = float(np.linalg.norm(vp - pos[0], axis=1).max() / H) if len(vp) else None
        # worst pose within the joint limits: each link fully rotated, in 8 directions
        pire = _pire_pose(c, ch, r1, jts, seg_corps, tete_rg)
        chaines.append({
            "racine": noms[r0], "parent": noms[O[r0]["parent"]] if O[r0]["parent"] >= 0 else None,
            "maillons": len(ch), "os_sous_arbre": len(sous_maillons), "ramifiee": len(sous_maillons) > len(ch),
            "lests": len(lests), "masse_lests": r3(sum(m_lest)) if m_lest else 0.0,
            "longueur_H": r3(seg.sum()), "espacement_H": r3(seg), "espacement_cv": r3(seg.std() / seg.mean()) if len(seg) and seg.mean() > 0 else None,
            "premier_dynamique": premier, "modes": [rr["mode"] if rr else None for rr in r1],
            "formes": [rr["forme"] if rr else None for rr in r1],
            "masses": r3(masses), "rapport_masse_med": r3(np.median(rapp)) if rapp else None,
            "amort_dep": r3([rr["amort_dep"] if rr else None for rr in r1]),
            "amort_rot": r3([rr["amort_rot"] if rr else None for rr in r1]),
            "rebond": r3(sorted({rr["rebond"] for rr in r1 if rr})), "friction": r3(sorted({rr["friction"] for rr in r1 if rr})),
            "rayons_H": r3([segment_rigide(rr)[2] / H if rr else None for rr in r1]),
            "groupes": sorted({rr["groupe"] for rr in r1 if rr}),
            "ignores": sorted({tuple(ignores(rr)) for rr in r1 if rr}),
            "touche_tete": (sum(coll_tete) / len(coll_tete)) if coll_tete else None,
            "joints": len(jts), "limites_rot_deg": r3(lim, 1) if lim else [],
            "amplitude_max_deg": r3([max(np.degrees(jj["rot_max"][a] - jj["rot_min"][a]) for jj in jts) for a in range(3)], 1) if jts else None,
            "joint_rot_nulle": all(np.allclose(jj["rot"], 0, atol=1e-3) for jj in jts) if jts else None,
            "ressorts_rot": r3(sorted({tuple(np.round(jj["ressort_rot"], 2)) for jj in jts})) if jts else [],
            "ressorts_dep": r3(sorted({tuple(np.round(jj["ressort_dep"], 2)) for jj in jts})) if jts else [],
            "translation_libre": any(np.ptp([jj["dep_min"], jj["dep_max"]], axis=0).max() > 1e-6 for jj in jts) if jts else None,
            "sommets_portes": n_v, "racine_dist_crane_H": r3(racine_crane), "portee_sommets_H": r3(portee),
            "chaine_sur_portee": r3(seg.sum() / portee) if portee else None,
            "recouvrement_repos_sans_contact_min_H": r3(min(rec_libre)) if rec_libre else None, "rayon_sur_demi_largeur": q(rapports, (10, 50, 90), 2),
            "ecart_os_centre_H": q(ecarts), "recouvrement_repos_min_H": r3(min(rec)) if rec else None,
            "pire_pose": pire})
    # weights of the hair vertices
    nb = (W_p > 1e-3).sum(1)
    tete = c.i_tete
    statique = (nb == 1) & (W_os[:, 0] == tete)
    typ = c.t["type_poids"][c.vh]
    # two influences: parent and child?
    pe = []
    for i in np.where(nb == 2)[0]:
        a, b = W_os[i][np.argsort(-W_p[i])[:2]]
        pe.append(O[a]["parent"] == b or O[b]["parent"] == a)
    # strands and chains: monotonicity of the dominant bone's rank along s
    # rank within the whole subtree (depth from the root)
    prof = np.full(len(O), -1)
    for r0 in racines:
        pile = [(r0, 0)]
        while pile:
            i, d_ = pile.pop()
            prof[i] = d_
            pile += [(x, d_ + 1) for x in enfants.get(i, []) if oc[x]]
    mono = []
    for m in c.meches:
        vk = np.where(c.ile_de == m["ile"])[0]
        pr = prof[dom[vk]]
        okp = pr >= 0
        if okp.sum() < 10 or np.ptp(pr[okp]) == 0:
            continue
        sv = c.s_glob[vk][okp]
        rk = pr[okp]
        rs = np.argsort(np.argsort(sv))
        rr_ = np.argsort(np.argsort(rk + 1e-6 * sv))
        mono.append(float(np.corrcoef(rs, rr_)[0, 1]))
        m["correlation_s_profondeur_os"] = r3(mono[-1])
        m["chaine_part"] = r3(okp.mean())
    # collision bodies of the head and upper body
    col = []
    for k in corps:
        r = RG[k]
        a, b, rad = seg_corps[k]
        col.append({"nom": r["nom"], "os": noms[r["os"]], "forme": PMX.FORMES_RIGIDE[r["forme"]],
                    "rayon_H": r3(rad / H), "longueur_H": r3(np.linalg.norm(b - a) / H), "groupe": r["groupe"],
                    "ignores": ignores(r)})
    # hair inside the head collision bodies at rest (the skull itself)
    Ph = c.pos[c.vh]
    dedans_tete = None
    if tete_rg:
        a, b, rad = seg_corps[tete_rg[0]]
        dp = dist_point_segment(Ph, a, b)
        dedans_tete = float((dp < rad).mean())
        # gap between the head body and the underside of the cap
        haut = Ph[:, 1] > c.y_yeux
        ecart_cal = q((dp[haut] - rad) / H)
    else:
        ecart_cal = None
    dyn_rg = [r for r in RG if r["os"] >= 0 and oc[r["os"]] and r["mode"] in RIG_DYN]
    lest_rg = [r for r in RG if r["os"] >= 0 and oc_tout[r["os"]] and lest[r["os"]]]
    groupes = {}
    for r in dyn_rg:
        groupes[r["groupe"]] = groupes.get(r["groupe"], 0) + 1
    soi = [not ((r["masque"] >> r["groupe"]) & 1) for r in dyn_rg]
    return {
        "os_coiffure": int(oc.sum()), "chaines": chaines, "nombre_chaines": len(chaines),
        "rigides_dynamiques": len(dyn_rg), "groupes_cheveux": groupes,
        "lests": {"nombre": len(lest_rg), "modes": sorted({r["mode"] for r in lest_rg}),
                  "masques": sorted({r["masque"] for r in lest_rg}), "masses": q([r["masse"] for r in lest_rg])},
        "ignore_son_groupe_part": r3(np.mean(soi)) if soi else None,
        "corps_collision": col, "cheveux_dans_rigide_tete_part": r3(dedans_tete),
        "ecart_calotte_rigide_tete_H": ecart_cal,
        "poids": {"influences": {str(k): int((nb == k).sum()) for k in range(1, 5)},
                  "types": {PMX.TYPES_POIDS[k]: int((typ == k).sum()) for k in np.unique(typ)},
                  "statiques_tete_part": r3(statique.mean()),
                  "deux_os_parent_enfant_part": r3(np.mean(pe)) if pe else None,
                  "monotonie_s_os": q(mono)}}


def _pire_pose(c, ch, r1, jts, seg_corps, tete_rg):
    """Chain rotated to the maximum of its limits (X then Z in the world frame, joints at zero
    angle) in 8 directions: smallest gap (in H) between its bodies and the head body, and
    between its bodies and the upper body. Kinematic approximation: no gravity, no contacts;
    it only tells whether the limits are enough to keep the strand out of the skull."""
    if not tete_rg or not jts:
        return None
    O = c.p["os"]
    pos = np.array([O[i]["pos"] for i in ch], float)
    lims = []
    for jj in jts[:len(ch)]:
        lims.append((np.array(jj["rot_min"]), np.array(jj["rot_max"])))
    while len(lims) < len(ch):
        lims.append(lims[-1])
    res = {"tete_min_H": np.inf, "corps_min_H": np.inf}
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        dx, dz = np.cos(ang), np.sin(ang)
        P = pos.copy()
        Racc = np.eye(3)
        for k in range(len(ch) - 1):
            lo, hi = lims[k]
            ax = hi[0] if dx >= 0 else lo[0]
            az = hi[2] if dz >= 0 else lo[2]
            Rk = rot_mmd((ax * abs(dx), 0.0, az * abs(dz)))
            Racc = Racc @ Rk
            off = pos[k + 1:] - pos[k]
            P[k + 1:] = P[k] + (Racc @ (pos[k + 1:] - pos[k]).T).T if k == 0 else P[k] + (Rk @ (P[k + 1:] - P[k]).T).T
        for k in range(len(ch) - 1):
            rr = r1[k]
            if rr is None or rr["mode"] not in RIG_DYN:
                continue
            ra = segment_rigide(rr)[2]
            for kk, (a2, b2, r2) in seg_corps.items():
                d = (dist_segments(P[k], P[k + 1], a2, b2) - ra - r2) / c.H
                if not collisionnent(rr, c.p["rigides"][kk]):
                    cle = "sans_contact_min_H"
                else:
                    cle = "tete_min_H" if kk in tete_rg else "corps_min_H"
                res[cle] = min(res.get(cle, np.inf), d)
    return {k: r3(v) for k, v in res.items()}


# =================================================================== F. morphs

def mesurer_morphes(c):
    vh = set(c.vh.tolist())
    out = []
    for m in c.p["morphes"]:
        if m["type"] != 1 or not m["decalages"]:
            continue
        idx = [d[0] for d in m["decalages"] if d[0] in vh and np.linalg.norm(d[1]) > 1e-6]
        if idx:
            out.append({"nom": m["nom"], "sommets_cheveux": len(idx)})
    return out


# =================================================================== per-character summary

def resume_meches(meches):
    def s(cle, filtre=None):
        return q([m[cle] for m in meches if (filtre is None or filtre(m)) and m.get(cle) is not None])
    cats = {}
    for m in meches:
        cats[m["categorie"]] = cats.get(m["categorie"], 0) + 1
    dyn = lambda m: m["os_tete_part"] is not None and m["os_tete_part"] < 0.5
    return {"nombre": len(meches), "categories": cats,
            "dynamiques": int(sum(1 for m in meches if dyn(m))),
            "longueur_H": s("longueur_H"), "largeur_H": s("largeur_med_H"), "epaisseur_H": s("epaisseur_med_H"),
            "aplatissement": s("aplatissement"), "pointe_rel": s("largeur_pointe_rel"),
            "anneaux": s("anneaux"), "sommets_par_anneau": s("sommets_par_anneau"),
            "espacement_H": s("espacement_med_H"), "espacement_pointe_rel": s("espacement_pointe_rel"),
            "corr_v_s": s("corr_v_s"), "corr_u_s": s("corr_u_s"), "angle_axe_uv_v_deg": s("angle_axe_uv_v_deg"),
            "densite_texel_cv": r3(np.std([m["densite_texel"] for m in meches if m["densite_texel"]]) /
                                   np.mean([m["densite_texel"] for m in meches if m["densite_texel"]])) if meches else None,
            "bord_libre": s("bord_libre_part"), "faces_vers_axe": s("faces_vers_axe_part"),
            "os_dominant_part_dyn": s("os_dominant_part", dyn),
            "monotonie_s_os": s("correlation_s_profondeur_os", dyn)}


def mesurer(biblio, nom, c=None, ressorts=None):
    """Measurements of a hairstyle: PMX from the library (biblio, nom) or an already loaded
    hairstyle (c, for example CoiffureGLB); ressorts: Godot scenes (.tscn) or JSON of the bone
    springs."""
    t0 = time.time()
    c = c if c is not None else Coiffure(biblio, nom)
    out = {"personnage": nom, "pmx": c.chemin, "H": r3(c.H, 4), "iod_H": r3(c.D.iod / c.H),
           "rayon_crane_H": r3(c.R / c.H), "repere": "MMD: Y up, face towards -Z; lengths in H"}
    out["maillage"] = mesurer_maillage(c)
    meches = mesurer_meches(c)
    out["uv"] = mesurer_uv(c)
    out["physique"] = mesurer_physique(c)
    out["meches_resume"] = resume_meches(meches)
    out["meches"] = meches
    out["texture"] = mesurer_texture(c)
    out["morphes"] = mesurer_morphes(c)
    if isinstance(c, CoiffureGLB):
        out["glb"] = {"fichiers": c.chemins, "unite_vers_mmd": r3(c.unite, 5), "crane": c.repere_crane,
                      "os_tete": c.noms_os[c.i_tete] if c.i_tete >= 0 else None,
                      "materiaux_cheveux": [c.p["materiaux"][i]["nom"] for i in c.mats_cheveux],
                      "roles": {c.p["materiaux"][i]["nom"]: r for i, r in c.roles.items()},
                      "os_coiffure": [c.noms_os[i] for i in np.where(c.masque_os)[0]],
                      "soudure": c.soudure}
    if ressorts:
        import springs as RS
        out["ressorts"] = RS.mesurer(c, ressorts)
    out["duree_s"] = round(time.time() - t0, 1)
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

LIGNES = [
    ("faces cheveux", lambda r: r["maillage"]["budget"]["faces_cheveux"]),
    ("part du modèle", lambda r: r["maillage"]["budget"]["part_faces"]),
    ("faces calque", lambda r: r["maillage"]["budget"]["faces_calques"]),
    ("pièces", lambda r: r["maillage"]["pieces"]["nombre"]),
    ("part plus grande pièce", lambda r: r["maillage"]["pieces"]["part_plus_grande"]),
    ("arêtes de bord (part)", lambda r: r["maillage"]["aretes_bord_part"]),
    ("mèches (îles ≥ 12 s.)", lambda r: r["meches_resume"]["nombre"]),
    ("longueur méd. (H)", lambda r: _p(r["meches_resume"]["longueur_H"])),
    ("largeur méd. (H)", lambda r: _p(r["meches_resume"]["largeur_H"])),
    ("aplatissement méd.", lambda r: _p(r["meches_resume"]["aplatissement"])),
    ("pointe/largeur méd.", lambda r: _p(r["meches_resume"]["pointe_rel"])),
    ("sommets/anneau méd.", lambda r: _p(r["meches_resume"]["sommets_par_anneau"])),
    ("corr V–s méd.", lambda r: _p(r["meches_resume"]["corr_v_s"])),
    ("normales : angle géom. méd. (°)", lambda r: _p(r["maillage"]["normales"]["angle_geometrique_deg"])),
    ("normales : angle radial méd. (°)", lambda r: _p(r["maillage"]["normales"]["angle_radial_deg"])),
    ("symétrie (part à 1 % H)", lambda r: r["maillage"]["symetrie"]["part_1pc"]),
    ("UV miroir identiques", lambda r: r["uv"]["miroir_uv_identiques_part"]),
    ("chaînes", lambda r: r["physique"]["nombre_chaines"]),
    ("maillons méd.", lambda r: _med([ch["maillons"] for ch in r["physique"]["chaines"]])),
    ("espacement os méd. (H)", lambda r: _med([e for ch in r["physique"]["chaines"] for e in (ch["espacement_H"] or [])])),
    ("rigides dynamiques", lambda r: r["physique"]["rigides_dynamiques"]),
    ("lests (錘)", lambda r: r["physique"]["lests"]["nombre"]),
    ("amplitude X max méd. (°)", lambda r: _med([ch["amplitude_max_deg"][0] for ch in r["physique"]["chaines"] if ch["amplitude_max_deg"]])),
    ("amplitude Y max (°)", lambda r: max([ch["amplitude_max_deg"][1] for ch in r["physique"]["chaines"] if ch["amplitude_max_deg"]], default=None)),
    ("rapport masse enfant/parent méd.", lambda r: _med([ch["rapport_masse_med"] for ch in r["physique"]["chaines"]])),
    ("rayon rigide / demi-largeur méd.", lambda r: _med([ch["rayon_sur_demi_largeur"]["p50"] for ch in r["physique"]["chaines"] if ch["rayon_sur_demi_largeur"]])),
    ("recouvrement repos min (H, paires en contact)", lambda r: min([ch["recouvrement_repos_min_H"] for ch in r["physique"]["chaines"] if ch["recouvrement_repos_min_H"] is not None], default=None)),
    ("pire pose tête min (H)", lambda r: min([ch["pire_pose"]["tete_min_H"] for ch in r["physique"]["chaines"] if ch["pire_pose"] and ch["pire_pose"].get("tete_min_H") is not None], default=None)),
    ("monotonie s / profondeur os méd.", lambda r: _p(r["physique"]["poids"]["monotonie_s_os"])),
    ("ignore son groupe", lambda r: r["physique"]["ignore_son_groupe_part"]),
    ("sommets statiques (頭 seul)", lambda r: r["physique"]["poids"]["statiques_tete_part"]),
]


def _p(d):
    return d["p50"] if d else None


def _med(v):
    v = [x for x in v if x is not None]
    return r3(np.median(v)) if v else None


def tableau(sortie):
    res = []
    for f in sorted(os.listdir(sortie)):
        if f.endswith(".json") and f != "ensemble.json":
            with open(os.path.join(sortie, f), encoding="utf-8") as fh:
                res.append(json.load(fh))
    noms = [r["personnage"] for r in res]
    lignes = ["| measure | " + " | ".join(noms) + " |", "|---|" + "---|" * len(noms)]
    ens = {}
    for titre, fn in LIGNES:
        vals = []
        for r in res:
            try:
                vals.append(fn(r))
            except (KeyError, TypeError, ValueError):
                vals.append(None)
        ens[titre] = dict(zip(noms, vals))
        lignes.append("| %s | %s |" % (titre, " | ".join("—" if v is None else str(v) for v in vals)))
    ecrire_json(os.path.join(sortie, "ensemble.json"), ens)
    tmp = os.path.join(sortie, "tableau.md.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lignes) + "\n")
    os.replace(tmp, os.path.join(sortie, "tableau.md"))
    journal("table:", len(res), "characters")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--biblio", default=os.path.join(R.chemin("paths.data"), "bibliotheque-visage"))
    ap.add_argument("--sortie", default=os.path.join(R.chemin("paths.data"), "cheveux"))
    ap.add_argument("--seuls", default=None)
    ap.add_argument("--tous", action="store_true", help="the whole library (index.json)")
    ap.add_argument("--tableau", default=None, help="only rebuild the table of this folder")
    ap.add_argument("--glb", action="append", default=None,
                    help="model GLB (repeat for separate strands, same skeleton): measures this model only")
    ap.add_argument("--nom", default=None, help="name of the GLB character (file <nom>.json)")
    ap.add_argument("--mats-cheveux", default=None, help="hair material patterns (comma-separated; default: MOTIFS_CHEVEUX_GLB)")
    ap.add_argument("--peau", default=None, help="face material patterns (default: those of tete_glb)")
    ap.add_argument("--yeux", default=None, help="eye material patterns (default: those of tete_glb)")
    ap.add_argument("--ressorts", action="append", default=None,
                    help="Godot scene (.tscn) or JSON of the SpringBoneSimulator3D nodes (repeatable)")
    a = ap.parse_args(argv)
    if a.tableau:
        tableau(a.tableau)
        return
    os.makedirs(a.sortie, exist_ok=True)
    if a.glb:
        nom = a.nom or os.path.splitext(os.path.basename(a.glb[0]))[0]
        dec = lambda x: tuple(m.strip().lower() for m in x.split(",")) if x else None
        c = CoiffureGLB(a.glb, nom, dec(a.mats_cheveux), dec(a.peau), dec(a.yeux))
        r = mesurer(None, nom, c, a.ressorts)
        ecrire_json(os.path.join(a.sortie, nom + ".json"), r)
        journal(nom, "done in %.0f s" % r["duree_s"], "->", os.path.join(a.sortie, nom + ".json"))
        return
    if a.tous:
        with open(os.path.join(a.biblio, "index.json"), encoding="utf-8") as fh:
            noms = sorted(json.load(fh)["coiffure"])
    elif a.seuls:
        noms = a.seuls.split(",")
    else:
        noms = list(CHOIX)
    for nom in noms:
        try:
            r = mesurer(a.biblio, nom)
        except Exception as e:  # one failing character does not stop the others
            import traceback
            traceback.print_exc()
            journal(nom, "FAILED", e)
            continue
        ecrire_json(os.path.join(a.sortie, nom + ".json"), r)
        journal(nom, "done in %.0f s" % r["duree_s"])
    tableau(a.sortie)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)

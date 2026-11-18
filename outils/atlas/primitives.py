"""Shared primitives P-01 to P-11 (method §2.3).

Interfaces: every function takes numpy arrays (Blender frame,
metres) and returns arrays or plain dicts. Only P-07 and D4/D5 of P-09
need `mathutils` (Blender); the rest runs with numpy alone.

Usual input data (see donnees.py):
  pos (n, 3)         positions of `soude` (or of `brut`)
  faces (m, 3)       triangles
  aretes (e, 2)      unique edges of `soude`
  sq                 Skeleton (names, heads, parents, additions)
"""
import json
import os
import re
import sys
import unicodedata

import numpy as np

import cinematique as cin

ICI = os.path.dirname(os.path.abspath(__file__))

# ======================================================================= P-01

def nfkc(nom):
    return unicodedata.normalize("NFKC", nom)


_COTE = [
    (re.compile(r"^左(.+)$"), "L"), (re.compile(r"^右(.+)$"), "R"),
    (re.compile(r"^(.+)左$"), "L"), (re.compile(r"^(.+)右$"), "R"),
    (re.compile(r"^(.+)[._]L$"), "L"), (re.compile(r"^(.+)[._]R$"), "R"),
    (re.compile(r"^L[._](.+)$"), "L"), (re.compile(r"^R[._](.+)$"), "R"),
]

_COTE_MILIEU = re.compile(r"^(.+?)_([LR])_(.+)$")


def separer_cote(nom):
    """'左腕' -> ('腕', 'L') ; '腰キャンセル右' -> ('腰キャンセル', 'R') ; '頭' -> ('頭', None)."""
    n = nfkc(nom)
    for rx, c in _COTE:
        m = rx.match(n)
        if m:
            return m.group(1), c
    # side in the middle (Ctr_/Skn_ bones of Unity ports: Ctr_L_Hair_01 -> Ctr_Hair_01, L)
    m = _COTE_MILIEU.match(n)
    if m:
        return m.group(1) + "_" + m.group(3), m.group(2)
    return n, None


def identifiant(nom):
    """Normalized identifier of §6.1: _L / _R instead of 左 / 右, NFKC digits."""
    base, c = separer_cote(nom)
    return base + ("_" + c if c else "")


def nom_miroir(nom):
    """PMX name of the symmetric element (左 <-> 右, _L_ <-> _R_), or None."""
    if "左" in nom:
        return nom.replace("左", "右")
    if "右" in nom:
        return nom.replace("右", "左")
    m = _COTE_MILIEU.match(nfkc(nom))
    if m and separer_cote(nom)[1] is not None and not any(rx.match(nfkc(nom)) for rx, _ in _COTE):
        return "%s_%s_%s" % (m.group(1), "R" if m.group(2) == "L" else "L", m.group(3))
    return None


# Bone classes of SQ-01 (names without side, NFKC)
STANDARD_MMD = {"全ての親", "センター", "グルーブ", "腰", "上半身", "上半身2", "首", "頭", "両目", "目",
                "肩", "腕", "ひじ", "手首", "親指1", "親指2", "下半身", "足", "ひざ", "足首", "つま先",
                "足IK", "つま先IK"} | {d + str(k) for d in ("人指", "中指", "薬指", "小指") for k in (1, 2, 3)}
SEMI_STANDARD_MMD = {"操作中心", "足IK親", "肩P", "肩C", "腕捩", "腕捩1", "腕捩2", "腕捩3", "手捩",
                     "手捩1", "手捩2", "手捩3", "ダミー", "親指0", "腰キャンセル", "足D", "ひざD",
                     "足首D", "足先EX", "上半身1", "上半身3"}


def classe_os(nom, feuille=False):
    """SQ-01: standard_mmd, semi_standard_mmd or propre. The "先" end bones
    (leaves whose name ends with 先) are semi-standard."""
    base, _ = separer_cote(nom)
    if base in STANDARD_MMD:
        return "standard_mmd"
    if base in SEMI_STANDARD_MMD:
        return "semi_standard_mmd"
    if feuille and base.endswith("先"):
        return "semi_standard_mmd"
    return "propre"


# Order matters: the first pattern that matches wins.
FAMILLES = [
    ("racine", r"^(全ての親|操作中心|センター|グルーブ)$"),
    ("accessoire", r"坠|珠|刀|扣|结|穗|发带|武器|錘?鞘"),
    ("vetement", r"外套|衣|スカート|裙|领带|腰环|袖"),
    ("cheveux", r"髪|发|刘海|发髻"),
    ("visage", r"目|眉|齿|舌|口|あご|顎"),
    ("fesses", r"臀|尻"),
    ("poitrine", r"胸"),
    ("main", r"指"),
    ("bras", r"肩|腕|ひじ|手首|捩|手先|ダミー"),
    ("jambe", r"足|ひざ|膝|つま先"),
    ("tronc", r"上半身|下半身|腰|首|頭"),
]
_FAMILLES = [(f, re.compile(rx)) for f, rx in FAMILLES]

# Body sub-region given by the bone with the largest weight (P-02). The pattern that covers the
# most characters of the name wins (足首 over 首 and 足, 手首 over 首); at equal length,
# the order of the list (足指: 指 before 足).
REGIONS_OS = [
    ("doigt", r"指"),
    ("main", r"手首|手先|ダミー"),
    ("avant_bras", r"ひじ|手捩"),
    ("bras", r"腕"),
    ("tronc_haut", r"肩|上半身2|上半身3"),
    ("poitrine", r"胸"),
    ("fesses", r"臀|尻"),
    ("ventre", r"上半身1?$"),
    ("bassin", r"下半身|腰|^(全ての親|操作中心|センター|グルーブ)$"),
    ("cou", r"首"),
    ("tete", r"頭"),
    ("pied", r"足首|つま先|足先"),
    ("jambe", r"ひざ|膝"),
    ("cuisse", r"足"),
]
_REGIONS_OS = [(r, re.compile(rx)) for r, rx in REGIONS_OS]


_FAMILLES_JSON = {}


def _jeux():
    outils = os.path.dirname(ICI)
    if outils not in sys.path:
        sys.path.append(outils)
    import jeux
    return jeux


def charger_familles():
    """Hand-kept classifications for the models of the active game: familles.json of its description
    (outils/jeux/<jeu>/familles.json); no classification if the game has none."""
    p = _jeux().actif().fichier("familles.json")
    if p is None:
        _FAMILLES_JSON.update({"d": {}, "mtime": None, "motifs": []})
        return _FAMILLES_JSON["d"]
    m = os.path.getmtime(p)
    if _FAMILLES_JSON.get("mtime") != m:
        with open(p, encoding="utf-8") as f:
            _FAMILLES_JSON["d"] = json.load(f)
        _FAMILLES_JSON["mtime"] = m
        _FAMILLES_JSON["motifs"] = [(re.compile(e["motif"]), e)
                                    for e in _FAMILLES_JSON["d"].get("motifs_os", [])]
    return _FAMILLES_JSON["d"]


def _entree(e):
    return {"famille": e} if isinstance(e, str) else e


def entree_os(nom, manuel=None):
    """Hand-written classification (familles.json) of a bone: dict {famille, region?} or None.
    `manuel`: "os" dict instead of familles.json (the patterns are then ignored if
    manuel is not None and does not come from the file)."""
    d = charger_familles()
    fichier = manuel is None
    manuel = d.get("os", {}) if manuel is None else manuel
    for cle in (nom, identifiant(nom)):
        if cle in manuel:
            return _entree(manuel[cle])
    if fichier or manuel is d.get("os"):
        n = nfkc(nom)
        for rx, e in _FAMILLES_JSON["motifs"]:
            if rx.search(n):
                return e
    return None


def famille_os(nom, manuel=None):
    """P-01: family of a bone: written classification (familles.json: "os" then "motifs_os"),
    otherwise patterns of FAMILLES, otherwise None (to be classified in familles.json)."""
    e = entree_os(nom, manuel)
    if e is not None:
        return e["famille"]
    base, _ = separer_cote(nom)
    for f, rx in _FAMILLES:
        if rx.search(base):
            return f
    return None


def region_os(nom, famille):
    """Body region of a bone (for P-02), or the family if the bone is not part of the body."""
    if famille in ("cheveux", "vetement", "accessoire"):
        return {"vetement": "vetement_libre"}.get(famille, famille)
    if famille == "visage":
        return "tete"
    e = entree_os(nom)
    if e is not None and e.get("region"):
        return e["region"]
    base, _ = separer_cote(nom)
    meilleur, n = None, 0
    for r, rx in _REGIONS_OS:
        t = rx.search(base)
        if t and t.end() - t.start() > n:
            meilleur, n = r, t.end() - t.start()
    return meilleur


# ======================================================================= P-02

GROUPES_MATERIAU = {
    "yeux": [r"^白目$", r"^目$", r"^目光", r"^睫$", r"^眉睫影$", r"^目影$"],
    "visage": [r"^颜$", r"^痣$", r"^眉$", r"^口$", r"^舌$", r"^齿$"],
    "cheveux": [r"^发$", r"^侧发$", r"^前发$"],
    "peau": [r"^肌$"],
}


def groupe_materiau(nom, accessoires=None):
    accessoires = accessoires if accessoires is not None else charger_familles().get("materiaux_accessoire", [])
    for g, motifs in GROUPES_MATERIAU.items():
        if any(re.search(m, nom) for m in motifs):
            return g
    if nom in accessoires:
        return "accessoire"
    return "tenue"


def table_materiaux(modele):
    """ZZ-04 table of a model (familles.json "materiaux/par_modele"), or None."""
    pm = charger_familles().get("materiaux", {}).get("par_modele", {})
    t = pm.get(modele)
    for _ in range(5):
        if isinstance(t, str) and t.startswith("="):
            t = pm.get(t[1:])
    return t if isinstance(t, dict) else None


def classes_materiaux(modele, noms):
    """ZZ-04 then P-02: for each material, dict {classe, groupe, de?, defaut?}.
    classe None = unclassified (reported, group "non_classe"). Without a table for this
    model: old classification by Evelyn's names (classe None, source "noms")."""
    d = charger_familles().get("materiaux", {})
    vers_groupe = d.get("groupes", {})
    t = table_materiaux(modele)
    if t is None:
        acc = charger_familles().get("materiaux_accessoire", [])
        return [{"classe": None, "groupe": groupe_materiau(n, acc), "source": "noms"} for n in noms]
    out = []
    for n in noms:
        e = t.get(n)
        e = {"classe": e} if isinstance(e, str) else dict(e or {"classe": None})
        c = e["classe"]
        if c == "doublet":
            b = t.get(e.get("de"))
            b = b if isinstance(b, str) else (b or {}).get("classe")
            e["groupe"] = vers_groupe.get(b, "non_classe")
        elif c == "mixte":
            e["groupe"] = "mixte"
        else:
            e["groupe"] = vers_groupe.get(c, "non_classe")
        e["source"] = "table"
        out.append(e)
    return out


def mat_sommet(faces, mat_face, n):
    """Material of each vertex (that of its first face)."""
    m = np.full(n, -1, dtype=np.int64)
    f = np.asarray(faces)
    for k in (2, 1, 0):
        m[f[:, k]] = mat_face
    return m


_FAMILLE_VERS_GROUPE = {"cheveux": "cheveux", "accessoire": "accessoire", "vetement": "tenue"}
SANS_FACE = "sans_face"


def groupes_soude(classes, mat_face, faces, n, idx, poids, familles_os):
    """P-02: group of each face and each vertex of `soude`. "mixte" materials
    are resolved per geometric island (P-05): majority of the dominant bone families
    (cheveux, accessoire, vetement) of the material's vertices in the island,
    otherwise the material's "defaut" group."""
    gm = np.array([c["groupe"] for c in classes], dtype=object)
    gf = gm[mat_face]
    ms = mat_sommet(faces, mat_face, n)
    gs = gm[ms]
    gs[ms < 0] = SANS_FACE  # vertex used by no face: neither material nor group
    mixtes = [i for i, c in enumerate(classes) if c.get("classe") == "mixte"]
    if not mixtes:
        return gf, gs
    ar, _ = aretes_de(faces)
    il = ilots_geometriques(n, ar)
    dom = os_dominant(idx, poids)
    gdom = np.array([_FAMILLE_VERS_GROUPE.get(familles_os[b]) if b >= 0 else None for b in dom],
                    dtype=object)
    for i in mixtes:
        defaut = classes[i].get("defaut", "tenue")
        # group of an island dominated by clothing bones when the material is not part of the outfit
        # (ZhuYuan: 髮 also carries a brooch and an armband, under bones classified vetement)
        vers = {"tenue": classes[i].get("vetement", "tenue")}
        fm = np.where(mat_face == i)[0]
        vm = np.unique(faces[fm])
        choix = {}
        for k in np.unique(il[vm]):
            v = vm[il[vm] == k]
            g = [x for x in gdom[v] if x is not None]
            if g:
                u, c = np.unique(np.array(g, dtype=object), return_counts=True)
                choix[k] = vers.get(u[np.argmax(c)], u[np.argmax(c)])
            else:
                choix[k] = defaut
        gf[fm] = [choix[k] for k in il[faces[fm, 0]]]
        sv = np.where(ms == i)[0]
        gs[sv] = [choix[k] for k in il[sv]]
    return gf, gs


GROUPES = ["peau", "tenue", "accessoire", "visage", "yeux", "cheveux", SANS_FACE]
REGIONS = ["tronc_haut", "poitrine", "ventre", "bassin", "fesses", "cou", "tete", "bras",
           "avant_bras", "main", "doigt", "cuisse", "jambe", "pied", "visage", "yeux",
           "cheveux", "vetement_libre", "accessoire", "sans", SANS_FACE]

# "poitrine" (chest) region of P-02: rigs with shared weights (NangongYu, Qingyi, Trigger) leave
# almost no vertex dominated by a chest bone (胸上2 tops out at 0.51-0.68).
# So the sum of the chest bone weights is read, at half of its maximum on the
# model (the full width at half maximum assumes no absolute weight value).
REGIONS_VERS_POITRINE = ("tronc_haut", "ventre", "poitrine")


def os_dominant(idx, poids):
    """Bone with the largest weight for each vertex (-1 if none)."""
    w = np.where(idx >= 0, poids, -1.0)
    k = np.argmax(w, axis=1)
    b = idx[np.arange(len(idx)), k]
    b[w[np.arange(len(idx)), k] <= 0] = -1
    return b


def regions(groupe_sommet, idx, poids, noms_os, familles_os):
    """P-02: region of each vertex. groupe_sommet: array of group names
    (material first). Returns an array of region names (REGIONS).

    A vertex without a face belongs to no body region (SANS_FACE). The chest
    is taken from the sum of weights (REGIONS_VERS_POITRINE), not from the dominant bone."""
    dom = os_dominant(idx, poids)
    reg_os = [region_os(n, f) for n, f in zip(noms_os, familles_os)]
    out = np.empty(len(groupe_sommet), dtype=object)
    for i, (g, b) in enumerate(zip(groupe_sommet, dom)):
        if g in ("yeux", "visage", "cheveux", SANS_FACE):
            out[i] = g
        elif b < 0 or reg_os[b] is None:
            out[i] = "sans"
        else:
            out[i] = reg_os[b]
    est_p = np.array([r == "poitrine" for r in reg_os], dtype=bool)
    if est_p.any():
        ok = idx >= 0
        s = (np.where(ok, poids, 0.0) * est_p[np.maximum(idx, 0)]).sum(axis=1)
        vus = np.isin(out, REGIONS_VERS_POITRINE)
        if vus.any() and s[vus].max() > 0:
            out[vus & (s >= 0.5 * s[vus].max())] = "poitrine"
    return out


# ======================================================================= utilities

def aretes_de(faces):
    """Unique sorted edges (e, 2), and for each face its 3 edge indices."""
    f = np.asarray(faces)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    inv = inv.reshape(3, -1).T
    return uniq, inv


def faces_par_arete(face_aretes, n_aretes):
    """Number of faces per edge."""
    return np.bincount(face_aretes.ravel(), minlength=n_aretes)


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
        lab = np.array([self.trouve(i) for i in range(len(self.p))])
        _, lab = np.unique(lab, return_inverse=True)
        return lab


def composantes(n, aretes):
    """Connected components through the edges (labels 0..k-1), in numpy."""
    lab = np.arange(n)
    a, b = np.asarray(aretes).T
    while True:
        m = np.minimum(lab[a], lab[b])
        nouv = lab.copy()
        np.minimum.at(nouv, a, m)
        np.minimum.at(nouv, b, m)
        nouv = nouv[nouv]
        if np.array_equal(nouv, lab):
            break
        lab = nouv
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def normales_faces(pos, faces):
    v = pos[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    aire = np.linalg.norm(n, axis=1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        nn = n / (2 * aire[:, None])
    return np.nan_to_num(nn), aire


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

    def rang(x):
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

    ra, rb = rang(a), rang(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ======================================================================= P-03

def miroir(pos, H, tol=1e-3):
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


def miroir_os(sq, tol=None):
    """Mirror of a bone (σ): symmetric bone of each bone, by name (左/右, _L_/_R_), otherwise
    by position (mirrored head within `tol`, bone off the median plane, name without a side:
    numbered skirt bones); the bone itself if it has none."""
    out = np.arange(len(sq))
    par_nom = np.zeros(len(sq), dtype=bool)
    for i, n in enumerate(sq.noms_pmx):
        mn = nom_miroir(n)
        if mn:
            j = sq.index(mn, -1)
            if j >= 0:
                out[i] = j
                par_nom[i] = True
    if tol is not None:
        t = sq.tetes
        libres = np.where(~par_nom & (np.abs(t[:, 0]) > tol))[0]
        if len(libres):
            mt = t[libres] * np.array([-1.0, 1, 1])
            for k, i in enumerate(libres):
                d = np.linalg.norm(t[libres] - mt[k], axis=1)
                j = int(np.argmin(d))
                if d[j] <= tol and libres[j] != i:
                    out[i] = libres[j]
    return out


def paires_miroir(miroir_sommets):
    """Pairs (a, b) a < b of P-03 from m(v); vertices of the median plane (m(v) = v)
    and those without a counterpart (−1) are not included; nor is a pair that is not reciprocal."""
    m = np.asarray(miroir_sommets)
    a = np.where((m >= 0) & (m != np.arange(len(m))))[0]
    a = a[m[m[a]] == a]
    a = a[a < m[a]]
    return np.c_[a, m[a]]


# ======================================================================= alignment

def procrustes(a, b, echelle=False):
    """Rotation R (proper), translation t and scale s that bring `a` onto `b` (paired
    points, Kabsch-Umeyama): b ≈ s·R·a + t. Returns (s, R, t)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ca, cb = a.mean(0), b.mean(0)
    A, B = a - ca, b - cb
    U, S, Vt = np.linalg.svd(A.T @ B)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / max((A * A).sum(), 1e-30)) if echelle else 1.0
    return s, R, cb - s * R @ ca


def plus_proches(a, b, bloc=2048):
    """Nearest neighbour of each point of a among b (blockwise brute force, numpy only):
    (indices, distances)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.empty(len(a), dtype=np.int64)
    dist = np.empty(len(a))
    bb = (b * b).sum(1)
    for i in range(0, len(a), bloc):
        x = a[i:i + bloc]
        d2 = (x * x).sum(1)[:, None] - 2 * x @ b.T + bb[None, :]
        k = np.argmin(d2, axis=1)
        idx[i:i + bloc] = k
        dist[i:i + bloc] = np.sqrt(np.maximum(d2[np.arange(len(x)), k], 0))
    return idx, dist


def icp_point_plan(a, b, normales_b, iterations=20, R0=None, t0=None):
    """Point-to-plane ICP: brings `a` onto `b` (normals of `b` given) by small linearized
    rotations, without scale. Returns (R, t, final distances to the nearest point)."""
    a, b, nb = np.asarray(a, float), np.asarray(b, float), np.asarray(normales_b, float)
    R = np.eye(3) if R0 is None else np.asarray(R0, float)
    t = np.zeros(3) if t0 is None else np.asarray(t0, float)
    for _ in range(iterations):
        q = a @ R.T + t
        j, _d = plus_proches(q, b)
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
    return R, t, plus_proches(a @ R.T + t, b)[1]


# ======================================================================= P-04

def _anneau_vide(n_boucles):
    return {"n_sommets": 0, "rayon": None, "centre": None, "normale": None, "rms_plan": None,
            "angle_axe": None, "n_boucles": n_boucles, "n_fermees": 0, "dist_axe": None,
            "points": np.zeros((0, 3)), "aretes": [], "fermee": False}


def anneau(pos, faces, origine, direction, longueur=1.0, t=0.0, masque_faces=None, choix="proche"):
    """P-04: cut of the mesh by the plane perpendicular to `direction` going
    through origine + t·longueur·direction. Returns the closed loop closest to
    the axis: {fermee, n_sommets, rayon, centre, normale, rms_plan, angle_axe, n_boucles,
    n_fermees, dist_axe, points, aretes (cut edge of each point, (a, b) with a < b)}.
    The shape is always the same: without a closed loop, fermee is False, n_sommets 0,
    aretes and points are empty, the geometric quantities are None.

    Computed in numpy (same cut as bmesh.ops.bisect_plane); n_sommets is the
    number of mesh edges crossed by the loop.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    c = np.asarray(origine, dtype=float) + t * longueur * d
    f = np.asarray(faces)
    if masque_faces is not None:
        f = f[masque_faces]
    s = (pos - c) @ d
    s = np.where(s == 0, 1e-12, s)
    sf = s[f] > 0
    croise = sf.any(axis=1) & ~sf.all(axis=1)
    f = f[croise]
    if len(f) == 0:
        return _anneau_vide(0)
    # each crossed face gives a segment between two cut edges
    noeuds = {}
    pts = []
    segs = []
    for tri in f:
        cles = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if (s[a] > 0) != (s[b] > 0):
                k = (min(a, b), max(a, b))
                if k not in noeuds:
                    noeuds[k] = len(pts)
                    u = s[a] / (s[a] - s[b])
                    pts.append(pos[a] + u * (pos[b] - pos[a]))
                cles.append(noeuds[k])
        if len(cles) == 2:
            segs.append(cles)
    pts = np.array(pts)
    cles_pts = [None] * len(pts)
    for k, i in noeuds.items():
        cles_pts[i] = k
    voisins = [[] for _ in range(len(pts))]
    for a, b in segs:
        voisins[a].append(b)
        voisins[b].append(a)
    vu = np.zeros(len(pts), dtype=bool)
    boucles = []
    for i in range(len(pts)):
        if vu[i]:
            continue
        pile, comp = [i], []
        vu[i] = True
        while pile:
            k = pile.pop()
            comp.append(k)
            for v in voisins[k]:
                if not vu[v]:
                    vu[v] = True
                    pile.append(v)
        fermee = all(len(voisins[k]) == 2 for k in comp) and len(comp) >= 3
        boucles.append((comp, fermee))
    fermees = [b for b, ok in boucles if ok]
    if not fermees:
        return _anneau_vide(len(boucles))
    infos = []
    for comp in fermees:
        P = pts[comp]
        cen, nor, rms = plan_svd(P)
        r = np.linalg.norm(P - cen, axis=1).mean()
        dist_axe = np.linalg.norm(cen - c)
        infos.append((dist_axe, comp, cen, nor, rms, r))
    infos.sort(key=lambda x: x[0] if choix == "proche" else -len(x[1]))
    dist_axe, comp, cen, nor, rms, r = infos[0]
    ang = np.degrees(np.arccos(min(1.0, abs(float(nor @ d)))))
    return {"n_sommets": len(comp), "rayon": float(r), "centre": cen, "normale": nor,
            "rms_plan": float(rms), "angle_axe": float(ang), "n_boucles": len(boucles),
            "n_fermees": len(fermees), "dist_axe": float(dist_axe), "points": pts[comp],
            "aretes": [cles_pts[i] for i in comp], "fermee": True}


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


def couverture_carre(triangles_uv, n=64):
    """Share of the centres of an n × n grid of the square [0, 1]² covered by at least one triangle
    (triangles_uv: (t, 3, 2)). Used by NO-08 and IN-05: an SDF channel fills its square."""
    gx, gy = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
    pts = np.c_[gx.ravel(), gy.ravel()]
    couvert = np.zeros(len(pts), bool)
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
        couvert |= ((l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)).any(0)
    return float(couvert.mean())


# ======================================================================= P-05

def ilots_geometriques(n_sommets, aretes):
    """P-05: union-find on the edges of `soude`."""
    return composantes(n_sommets, aretes)


def ilots_uv(faces_brut, b2s, uv, tol=1e-6):
    """P-05: UV islands on `brut`; two triangles are linked if they share an
    edge (welded vertices) at the same UV coordinates. Returns one label per face."""
    f = np.asarray(faces_brut)
    m = len(f)
    q = np.round(uv / tol).astype(np.int64)
    cles = {}
    uf = UnionFind(m)
    for fi in range(m):
        for a, b in ((f[fi, 0], f[fi, 1]), (f[fi, 1], f[fi, 2]), (f[fi, 2], f[fi, 0])):
            sa, sb = b2s[a], b2s[b]
            if sa > sb:
                sa, sb, a, b = sb, sa, b, a
            k = (sa, sb, q[a, 0], q[a, 1], q[b, 0], q[b, 1])
            if k in cles:
                uf.unir(fi, cles[k])
            else:
                cles[k] = fi
    return uf.etiquettes()


def etiquette_ilots(lab_sommets, mat_sommet, pos):
    """Label of an island: majority material, box, number of vertices."""
    out = []
    for k in range(lab_sommets.max() + 1):
        sel = np.where(lab_sommets == k)[0]
        mats, cnt = np.unique(mat_sommet[sel], return_counts=True)
        P = pos[sel]
        out.append({"ilot": int(k), "n_sommets": int(len(sel)), "materiau": int(mats[np.argmax(cnt)]),
                    "boite_min": P.min(0).tolist(), "boite_max": P.max(0).tolist()})
    return out


# ======================================================================= P-06

def densite(pos, aretes, masque_sommets, H):
    """P-06: mean length (in H) of the edges of `soude` touching the zone, and
    density = 1 / length. Returns (longueur_H, densite, n_aretes)."""
    m = masque_sommets[aretes[:, 0]] | masque_sommets[aretes[:, 1]]
    if not m.any():
        return None, None, 0
    L = np.linalg.norm(pos[aretes[m, 0]] - pos[aretes[m, 1]], axis=1).mean() / H
    return float(L), float(1.0 / L), int(m.sum())


# ======================================================================= P-07

def icosphere(niveau=2):
    """Vertices of a unit icosphere (12, 42, 162… vertices)."""
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t),
         (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, dtype=float) / np.linalg.norm(p) for p in v]
    for _ in range(niveau):
        cache = {}
        nf = []

        def milieu(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]

        for a, b, c in f:
            ab, bc, ca = milieu(a, b), milieu(b, c), milieu(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v)


def cameras(pos, distance=3.0, niveau=2):
    """P-07: 162 camera positions at `distance` m from the centre of the box."""
    centre = (pos.min(0) + pos.max(0)) / 2
    return centre + distance * icosphere(niveau)


def faces_vues(pos, faces, cams=None, masque_faces=None):
    """P-07: a face is seen if a ray from a camera reaches its centre
    unobstructed. Returns a boolean array (m,). Needs mathutils."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    pos = np.asarray(pos, dtype=float)
    faces = np.asarray(faces)
    if cams is None:
        cams = cameras(pos)
    bvh = BVHTree.FromPolygons(pos.tolist(), faces.tolist(), all_triangles=True, epsilon=0.0)
    centres = pos[faces].mean(axis=1)
    vue = np.zeros(len(faces), dtype=bool)
    a_tester = np.arange(len(faces)) if masque_faces is None else np.where(masque_faces)[0]
    for cam in cams:
        o = Vector(cam)
        reste = a_tester[~vue[a_tester]]
        if len(reste) == 0:
            break
        dirs = centres[reste] - cam
        dist = np.linalg.norm(dirs, axis=1)
        dirs /= dist[:, None]
        for fi, dv, di in zip(reste, dirs, dist):
            loc, nor, idx, d = bvh.ray_cast(o, Vector(dv), di + 1e-4)
            if idx is not None and (idx == fi or d >= di - 1e-5):
                vue[fi] = True
    return vue


# ======================================================================= P-08

class Squelette:
    """Bones of the PMX in the Blender frame. Heads are in metres."""

    def __init__(self, noms_pmx, tetes, queues, parents, ajout_source, ajout_facteur,
                 ajout_rot, ajout_dep):
        self.noms_pmx = list(noms_pmx)
        self.ids = [identifiant(n) for n in self.noms_pmx]
        self.tetes = np.asarray(tetes, dtype=float)
        self.queues = np.asarray(queues, dtype=float)
        self.parents = np.asarray(parents, dtype=np.int64)
        self.ajout_source = np.asarray(ajout_source, dtype=np.int64)
        self.ajout_facteur = np.asarray(ajout_facteur, dtype=float)
        self.ajout_rot = np.asarray(ajout_rot, dtype=bool)
        self.ajout_dep = np.asarray(ajout_dep, dtype=bool)
        self._index = {}
        for i, (n, k) in enumerate(zip(self.noms_pmx, self.ids)):
            self._index.setdefault(n, i)
            self._index.setdefault(k, i)
            self._index.setdefault(nfkc(n), i)

    def __len__(self):
        return len(self.noms_pmx)

    def index(self, nom, defaut=None):
        i = self._index.get(nom, self._index.get(nfkc(nom)))
        if i is None:
            if defaut is not None:
                return defaut
            raise KeyError(nom)
        return i

    def a(self, nom):
        return self.index(nom, -1) >= 0

    def enfants(self, b):
        return np.where(self.parents == b)[0]

    def direction(self, b, vers=None):
        """Direction of the bone: towards the head of `vers` if given, otherwise towards its tail."""
        cible = self.tetes[self.index(vers)] if vers is not None else self.queues[b]
        v = cible - self.tetes[b]
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.array([0.0, 0.0, 1.0]), n

    def cinematique(self, rotations=None, ajouts_actifs=None):
        return cin.cinematique(self.tetes, self.parents, self.ajout_source, self.ajout_facteur,
                               self.ajout_rot, self.ajout_dep, rotations, None, ajouts_actifs)


AVANT = np.array([0.0, -1.0, 0.0])
HAUT = np.array([0.0, 0.0, 1.0])
BAS = -HAUT

# Segment of each limb bone: bone -> bone whose head ends the segment
SEGMENTS = {"腕": "ひじ", "ひじ": "手首", "手首": "中指１", "足": "ひざ", "ひざ": "足首",
            "足首": "つま先", "首": "頭"}
POSES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"]


def _flechir(sq, os_, vers, cible, degres):
    """Rotation that brings the direction of bone `os_` towards `cible` by `degres`."""
    d, _ = sq.direction(sq.index(os_), vers)
    axe = np.cross(d, cible)
    if np.linalg.norm(axe) < 1e-9:
        axe = np.cross(d, AVANT)
    return cin.q_axe_angle(axe, degres)


def pose(numero, sq, cotes=("L", "R")):
    """P-08: local rotations (dict bone index -> quaternion) of the test pose.

    Angles counted from the rest pose (the delivered A-pose). The rotations
    apply to the standard bones; the D bones, twist bones and 腰キャンセル follow through
    their additions (cinematique). Right side = mirror of the left side."""
    r = {}

    def mettre(nom, q, cote):
        n = ("左" if cote == "L" else "右") + nom
        b = sq.index(n, -1)
        if b < 0:
            return
        r[b] = cin.q_mul(r.get(b, cin.IDENTITE), q)

    def symetrique(nom, fabrique, quels=cotes):
        for c in quels:
            mettre(nom, fabrique(c), c)

    def pour(c, os_, vers, cible, deg):
        nom = ("左" if c == "L" else "右") + os_
        v = ("左" if c == "L" else "右") + vers
        if not sq.a(nom) or not sq.a(v):
            return cin.IDENTITE
        return _flechir(sq, nom, v, cible, deg)

    def autour(c, os_, vers, deg):
        nom = ("左" if c == "L" else "右") + os_
        v = ("左" if c == "L" else "右") + vers
        if not sq.a(nom) or not sq.a(v):
            return cin.IDENTITE
        d, _ = sq.direction(sq.index(nom), v)
        return cin.q_axe_angle(d, deg if c == "L" else -deg)

    n = numero.upper()
    if n == "P0":
        pass
    elif n == "P1":  # arms raised by 150°
        symetrique("腕", lambda c: pour(c, "腕", "ひじ", HAUT, 150))
    elif n == "P2":  # arms in front at 90°
        symetrique("腕", lambda c: pour(c, "腕", "ひじ", AVANT, 90))
    elif n == "P3":  # forearm in pronation of 90° (wrist twist bone)
        os_ = "手捩" if sq.a("左手捩") else "手首"
        symetrique(os_, lambda c: autour(c, "ひじ", "手首", 90))
    elif n == "P4":  # elbow bent by 130°
        symetrique("ひじ", lambda c: pour(c, "ひじ", "手首", AVANT, 130))
    elif n == "P5":  # left thigh raised by 90°
        symetrique("足", lambda c: pour(c, "足", "ひざ", AVANT, 90), ("L",))
    elif n == "P6":  # left knee bent by 120°
        symetrique("ひざ", lambda c: pour(c, "ひざ", "足首", -AVANT, 120), ("L",))
    elif n == "P7":  # crouch: hips 110°, knees 130°
        symetrique("足", lambda c: pour(c, "足", "ひざ", AVANT, 110))
        symetrique("ひざ", lambda c: pour(c, "ひざ", "足首", -AVANT, 130))
    elif n == "P8":  # torso turned by 45°
        b = sq.index("上半身", -1)
        if b >= 0:
            r[b] = cin.q_axe_angle(HAUT, 45)
    elif n == "P9":  # head turned by 60° and raised by 30°
        b = sq.index("頭", -1)
        if b >= 0:
            r[b] = cin.q_mul(cin.q_axe_angle(HAUT, 60), cin.q_axe_angle([-1, 0, 0], 30))
    elif n == "P10":  # wrist bent by 70° (towards the palm, grip downwards)
        symetrique("手首", lambda c: pour(c, "手首", "中指１", BAS, 70))
    elif n == "P11":  # fingers closed
        for doigt in ("人指", "中指", "薬指", "小指"):
            for k, deg in (("１", 70), ("２", 90), ("３", 60)):
                suite = {"１": "２", "２": "３", "３": None}[k]
                for c in cotes:
                    nom = ("左" if c == "L" else "右") + doigt + k
                    if not sq.a(nom):
                        continue
                    b = sq.index(nom)
                    v = (("左" if c == "L" else "右") + doigt + suite) if suite else None
                    d, _ = sq.direction(b, v) if v and sq.a(v) else sq.direction(b)
                    axe = np.cross(d, BAS)
                    r[b] = cin.q_axe_angle(axe, deg)
        for k, suite, deg in (("１", "２", 30), ("２", None, 45)):
            for c in cotes:
                nom = ("左" if c == "L" else "右") + "親指" + k
                if not sq.a(nom):
                    continue
                b = sq.index(nom)
                v = (("左" if c == "L" else "右") + "親指" + suite) if suite else None
                d, _ = sq.direction(b, v) if v and sq.a(v) else sq.direction(b)
                r[b] = cin.q_axe_angle(np.cross(d, BAS), deg)
    elif n == "P12":  # walking step: left thigh +30° in front, right 30° behind
        symetrique("足", lambda c: pour(c, "足", "ひざ", AVANT, 30), ("L",))
        symetrique("足", lambda c: pour(c, "足", "ひざ", -AVANT, 30), ("R",))
    else:
        raise ValueError("unknown pose: %s" % numero)
    return r


def deformer(pos, idx, poids, sq, numero=None, rotations=None, ajouts_actifs=None,
             normaliser=True):
    """Positions deformed by the pose (numpy linear skinning, checked against
    Blender in §2.1). Returns (positions, R, t)."""
    if rotations is None:
        rotations = pose(numero, sq) if numero else {}
    R, t = sq.cinematique(rotations, ajouts_actifs)
    return cin.skinning(pos, idx, poids, sq.tetes, R, t, normaliser), R, t


# ======================================================================= P-09

def d1_longueurs(pos0, pos1, aretes, masque_aretes=None):
    """D1: ratio of edge lengths, percentiles 1 and 99."""
    e = aretes if masque_aretes is None else aretes[masque_aretes]
    l0 = np.linalg.norm(pos0[e[:, 0]] - pos0[e[:, 1]], axis=1)
    l1 = np.linalg.norm(pos1[e[:, 0]] - pos1[e[:, 1]], axis=1)
    ok = l0 > 1e-9
    r = l1[ok] / l0[ok]
    if len(r) == 0:
        return None
    return {"c01": float(np.percentile(r, 1)), "c99": float(np.percentile(r, 99)),
            "n": int(len(r))}


def d2_retournees(pos0, pos1, faces, R_dominant, masque_faces=None):
    """D2: flipped faces. The deformed normal is compared to the rest normal
    rotated by the dominant bone of the face (otherwise every face of a limb
    rotated by more than 90° would count). R_dominant: (m, 3, 3)."""
    f = faces if masque_faces is None else faces[masque_faces]
    Rd = R_dominant if masque_faces is None else R_dominant[masque_faces]
    n0, a0 = normales_faces(pos0, f)
    n1, a1 = normales_faces(pos1, f)
    attendu = np.einsum("mij,mj->mi", Rd, n0)
    ok = (a0 > 1e-12) & (a1 > 1e-12)
    return int(((np.einsum("mi,mi->m", attendu, n1) < 0) & ok).sum())


def rotation_dominante_faces(faces, idx, poids, R):
    """Rotation of the bone with the largest total weight on each face (for D2)."""
    dom = os_dominant(idx, poids)
    fd = dom[faces]
    # most frequent bone of the three vertices, otherwise that of the first
    b = np.where(fd[:, 1] == fd[:, 2], fd[:, 1], fd[:, 0])
    b = np.where(b < 0, 0, b)
    return R[b]


def d3_profil(pos0, pos1, faces, sq, os_, vers, R, t, fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
              masque_faces=None):
    """D3: radius profile r(t) along the bone, posed / rest (P-04 on both)."""
    b = sq.index(os_)
    d0, L = sq.direction(b, vers)
    h1 = t[b]
    d1 = R[b] @ d0
    out = []
    for f in fractions:
        a0 = anneau(pos0, faces, sq.tetes[b], d0, L, f, masque_faces)
        a1 = anneau(pos1, faces, h1, d1, L, f, masque_faces)
        if a0 and a1 and a0.get("fermee") and a1.get("fermee"):
            out.append({"t": f, "r0": a0["rayon"], "r1": a1["rayon"],
                        "rapport": a1["rayon"] / a0["rayon"]})
        else:
            out.append({"t": f, "r0": None, "r1": None, "rapport": None})
    return out


def d4_interpenetrations(pos, faces, masque_a, masque_b):
    """D4: pairs of intersecting faces between two non-neighbouring sets of faces
    (BVHTree.overlap). Needs mathutils."""
    from mathutils.bvhtree import BVHTree
    fa = faces[masque_a]
    fb = faces[masque_b]
    if len(fa) == 0 or len(fb) == 0:
        return 0
    ta = BVHTree.FromPolygons(pos.tolist(), fa.tolist(), all_triangles=True)
    tb = BVHTree.FromPolygons(pos.tolist(), fb.tolist(), all_triangles=True)
    paires = ta.overlap(tb)
    # discard the pairs that share a vertex (neighbours)
    n = 0
    for i, j in paires:
        if not set(fa[i]) & set(fb[j]):
            n += 1
    return n


def d5_hausdorff(pos_a, pos_b, faces_b=None, masque_sommets=None):
    """D5: one-sided Hausdorff distance from A to B. Without faces_b (same
    topology), the largest vertex-to-vertex displacement (upper bound)."""
    sel = slice(None) if masque_sommets is None else masque_sommets
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


def indicateurs(pos0, pos1, faces, aretes, idx, poids, R, masque_sommets):
    """P-09 combined for a region (D1, D2, D5; D3 and D4 are requested separately
    because they need a bone or two regions)."""
    ma = masque_sommets[aretes[:, 0]] & masque_sommets[aretes[:, 1]]
    mf = masque_sommets[faces].all(axis=1)
    Rd = rotation_dominante_faces(faces, idx, poids, R)
    return {"D1": d1_longueurs(pos0, pos1, aretes, ma),
            "D2": d2_retournees(pos0, pos1, faces, Rd, mf),
            "D5_sommets": d5_hausdorff(pos1, pos0, None, masque_sommets)}


# ======================================================================= P-10

def sommets_morphe(decalages, H, eps=1e-4):
    """P-10: vertices whose displacement exceeds ε = 1e-4 H. decalages: (n, 3)."""
    return np.where(np.linalg.norm(decalages, axis=1) > eps * H)[0]


# ======================================================================= P-11

def _boucles_bord(faces_sel):
    """Boundary loops of a set of faces (edges used only once)."""
    e, fa = aretes_de(faces_sel)
    cnt = faces_par_arete(fa, len(e))
    bord = e[cnt == 1]
    vois = {}
    for a, b in bord:
        vois.setdefault(a, []).append(b)
        vois.setdefault(b, []).append(a)
    vu, boucles = set(), []
    for s in vois:
        if s in vu:
            continue
        pile, comp = [s], []
        vu.add(s)
        while pile:
            k = pile.pop()
            comp.append(k)
            for v in vois[k]:
                if v not in vu:
                    vu.add(v)
                    pile.append(v)
        boucles.append(np.array(comp))
    return boucles


def reperes_visage(pos, faces, groupe_face, nom_mat_face, poids_tete, H_tol=0.002,
                   mats_iris=("目",), mat_visage="颜"):
    """P-11: face landmarks on `soude`.

    pos, faces       : `soude`
    groupe_face      : group of each face (P-02, material)
    nom_mat_face     : material name of each face
    poids_tete       : weight of 頭 per vertex
    mats_iris        : materials of class "iris" (ZZ-04); empty = model without eyes
    mat_visage       : largest material of class "visage" (ZZ-04)
    Returns a dict of points (Blender frame) and H, d_yeux. Without iris, the points that
    depend on them (iris, d_yeux, eye sockets, symmetry) are None and the bounds that relied
    on the eyes switch to the skull (nose) and to 0.5 H (mouth width).
    """
    out = {}
    nom_mat_face = np.asarray(nom_mat_face)
    # iris: barycentre of the vertices of the iris materials, split by the sign of x
    fi = faces[np.isin(nom_mat_face, list(mats_iris))] if len(mats_iris) else faces[:0]
    vi = np.unique(fi)
    P = pos[vi]
    gauche = P[P[:, 0] > 0]
    droite = P[P[:, 0] < 0]
    a_iris = len(gauche) > 0 and len(droite) > 0
    if len(mats_iris) and not a_iris:
        raise ValueError("iris not found (materials %s)" % ", ".join(mats_iris))
    if a_iris:
        out["iris_L"] = gauche.mean(0)
        out["iris_R"] = droite.mean(0)
        out["d_yeux"] = float(np.linalg.norm(out["iris_L"] - out["iris_R"]))
        out["rayon_iris"] = float(np.abs(gauche[:, 0] - out["iris_L"][0]).max())
    else:
        out["iris_L"] = out["iris_R"] = out["d_yeux"] = out["rayon_iris"] = None
    # top of the skull: max z of the vertices with weight > 0.5 on 頭, hair and accessories excluded
    exclus = np.zeros(len(pos), dtype=bool)
    for g in ("cheveux", "accessoire"):
        exclus[np.unique(faces[groupe_face == g])] = True
    garde = (poids_tete > 0.5) & ~exclus
    if not garde.any():
        raise ValueError("no skull vertex")
    k = np.where(garde)[0]
    tol = max(H_tol, 0.01 * out["d_yeux"]) if a_iris else H_tol
    # the landmark lives on the median plane (Trigger: the highest point is an isolated vertex
    # of the face material, with no mirror counterpart, at x = −0.319 H); without a median vertex, the highest
    k_med = k[np.abs(pos[k, 0]) < tol]
    k_cr = k_med if len(k_med) else k
    crane = pos[k_cr[np.argmax(pos[k_cr, 2])]]
    # face material
    fv = faces[nom_mat_face == mat_visage]
    vv = np.unique(fv)
    Pv = pos[vv]
    med = vv[np.abs(Pv[:, 0]) < tol]
    if len(med) == 0:
        raise ValueError("no face vertex on the median plane")
    z_iris = (out["iris_L"][2] + out["iris_R"][2]) / 2 if a_iris else crane[2]
    # chin: lowest point of the face material on the median plane
    out["menton"] = pos[med[np.argmin(pos[med, 2])]]
    larg_max = 1.5 * out["d_yeux"] if a_iris else 0.5 * float(crane[2] - out["menton"][2])
    # edge of the face: boundary loops of the face material
    boucles = _boucles_bord(fv)
    # eye sockets: boundary loop whose centre is closest to each iris
    for c in ("L", "R"):
        iris = out["iris_" + c]
        if iris is None:
            out["orbite_" + c] = None
            continue
        best = min(boucles, key=lambda b: np.linalg.norm(pos[b].mean(0) - iris))
        Pb = pos[best]
        out["orbite_" + c] = {"n_sommets": int(len(best)),
                              "coin_int": Pb[np.argmin(np.abs(Pb[:, 0]))],
                              "coin_ext": Pb[np.argmax(np.abs(Pb[:, 0]))]}
    # mouth: boundary loop centred on the median plane, below the eyes, above the chin
    cand = []
    for b in boucles:
        Pb = pos[b]
        cen = Pb.mean(0)
        larg = Pb[:, 0].max() - Pb[:, 0].min()
        if abs(cen[0]) < tol * 3 and out["menton"][2] < cen[2] < z_iris and \
                larg < larg_max and (Pb[:, 0].max() > 0 > Pb[:, 0].min()):
            cand.append((larg, b))
    if cand:
        larg, b = max(cand, key=lambda x: x[0])
        Pb = pos[b]
        out["commissure_L"] = Pb[np.argmax(Pb[:, 0])]
        out["commissure_R"] = Pb[np.argmin(Pb[:, 0])]
        out["bouche"] = Pb.mean(0)
        out["bouche_n_sommets"] = int(len(b))
    else:
        out["bouche"] = None
    # nose: smallest y on the median plane, above the mouth and below the eyes
    zb = out["bouche"][2] if out["bouche"] is not None else out["menton"][2]
    sel = med[(pos[med, 2] > zb) & (pos[med, 2] < z_iris)]
    if len(sel) == 0:
        raise ValueError("nose not found")
    out["nez"] = pos[sel[np.argmin(pos[sel, 1])]]
    out["crane"] = crane
    out["H"] = float(out["crane"][2] - out["menton"][2])
    out["symetrie_iris_H"] = (float(np.linalg.norm(out["iris_L"] * [-1, 1, 1] - out["iris_R"]) / out["H"])
                              if a_iris else None)
    return out


def mats_reperes(classes, noms, mat_face):
    """Reference materials of P-11 taken from ZZ-04: (iris, visage). Visage = the material
    of class "visage" that has the most faces; falls back on Evelyn's names."""
    noms = list(noms)
    iris = tuple(n for n, c in zip(noms, classes) if c.get("classe") == "iris")
    vis = [i for i, c in enumerate(classes) if c.get("classe") == "visage"]
    if not any(c.get("classe") for c in classes):
        return ("目",), "颜"
    if not vis:
        raise ValueError("no material of class visage")
    nf = np.bincount(np.asarray(mat_face), minlength=len(noms))
    return iris, noms[max(vis, key=lambda i: nf[i])]

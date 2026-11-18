"""Head retargeter, fully automatic.

    python3 -I tools/head/retarget_head.py forme  --source A.pmx --target B.pmx --output <dossier>
    python3 -I tools/head/retarget_head.py visage --source A.pmx --target B.pmx --output <dossier>
        [--no-sheet] [--sans-teinte]

forme  : A's head takes B's skull shape; A's face (eyes, nose, mouth, eyelids, eyebrows,
         their proportions, all its morphs) stays A's, on A's body. Output: A modified
         (same topology, same materials).
visage : A's face replaces B's, on B's head and body. Output: B with its face area cut out,
         A's face (skin, eyes, lashes, eyebrows, teeth, tongue) sewn in its place, A's face
         morphs carried over.

Python: Blender's (numpy). Outputs: <sortie>/resultat.pmx, mesures.json, recalage.npz,
planche.png (Workbench render through the Blender queue), planche_textures.png (textured
render). In visage mode, B's skin is recoloured to the tint of A's face (tint.py: textures
copied into <sortie>/teinte/, difference in mesures.json "teinte").
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
sys.path.append(os.path.join(ICI, ".."))
import settings as R  # noqa: E402

import geom as G  # noqa: E402
import pmx_write as pmx_ecrire  # noqa: E402
import refine as RF  # noqa: E402
import registration as RC  # noqa: E402
import head as TT  # noqa: E402
import checks as VF  # noqa: E402

# Face area: hull (front view) of the eyes, eyebrows, nose and mouth, dilated by MARGE,
# transition to the skull over BANDE (in inter-ocular distances).
MARGE = 0.06
BANDE = 0.22
# Neck seam held in forme mode: return to A's shape over this radius (geodesic).
RAYON_COU = 0.45
# forme mode: A's hair and ears stay in place. Held skin: skin from which an outward ray
# hits hair within DISTANCE_CHEVEUX, and skin within RAYON_OREILLE (geodesic) of the most
# lateral point on each side.
DISTANCE_CHEVEUX = 0.6
RAYON_OREILLE = 0.4
# Face: A's face fades out at (ANNEAUX_BORD_B + 0.5) mean edges from B's border.
ANNEAUX_BORD_B = 4
# Face: morphs interpolated (zero Laplacian) over this number of rings around the seam.
ANNEAUX_HARMONIE = 3
# Face: the patch border is carried onto the hole border; the displacement fades out over
# this number of rings of the patch.
# minimum gap between the patch border and the hole border, in edges of B
ECART_COUTURE = 0.5
# local refinement of B near the seam: levels, radius (in edge lengths of B)
NIVEAUX_RAFFINAGE = 1
R_RAFFINAGE = 2.5
# Displacement smoothing (forme): weight of the Laplacian against fidelity to the projected point.
LISSAGE = 2.0
# run through bash: the queue script may not be executable where the tool runs (agent sandbox)
FILE_BLENDER = os.path.join(R.DEPOT, "bin", "blender-queue")   # queue for Blender runs


def journal(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ecrire_json(chemin, d):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(chemin), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, chemin)


def _centre_x0(c):
    c = np.array(c, dtype=float)
    c[0] = 0.0
    return c


def echelle_face(A, B):
    """Uniform scale (no rotation) that puts A's face landmarks onto B's."""
    PA, PB = A.points_visage(), B.points_visage()
    k = sorted(set(PA) & set(PB))
    a = np.array([PA[x] for x in k])
    b = np.array([PB[x] for x in k])
    a = a - a.mean(0)
    b = b - b.mean(0)
    return float((a * b).sum() / (a * a).sum())


def lisser_deplacement(X0, X, F, fixe, mu=LISSAGE):
    """Smoothed displacement X - X0 (min |D - D0|² + mu |L D|², fixed vertices held): the head
    keeps its own details (small folds, edges) and takes only the overall shape from B."""
    n = len(X0)
    L = G.laplacien(n, F)
    c = np.where(fixe, 1e4, 1.0)
    D0 = X - X0
    D = np.linalg.solve(np.diag(c) + mu * G.ltl(L), c[:, None] * D0)
    return X0 + D


def rotations_locales(XA, X, F):
    """Minimal rotation of each vertex between its normal before and after."""
    return G.rotation_min(G.normales_sommets(XA, F), G.normales_sommets(X, F))


# =================================================================== shared core

def projection_phong(q, f, bary, VB, FB, alpha=0.75):
    """Point projected onto B, carried onto the smooth Phong surface (Boubekeur and Alexa 2008):
    keeps the head from taking on B's facets where its edges cross B's."""
    nB = G.normales_sommets(VB, FB)
    tri = FB[f]
    out = np.zeros_like(q)
    for k in range(3):
        v = VB[tri[:, k]]
        n = nB[tri[:, k]]
        pi = q - ((q - v) * n).sum(1, keepdims=True) * n
        out += bary[:, k:k + 1] * pi
    return (1 - alpha) * q + alpha * out


def poser_visage(A, XA, B, XB, bord_B=False, sur_place=False):
    """Registers A's skin onto B, then blends: face = A (placed), skull = B (projected).
    bord_B: A's face fades out before the outer border of B's skin (hairline, temples), so
    that the seam falls where A already lies on B."""
    rec = RC.recaler(A, XA, B, XB, journal=journal, sur_place=sur_place)
    rec["phi_plat"] = rec["phi"]
    rec["phi"] = projection_phong(rec["phi"], rec["face"], rec["bary"], XB, B.F)
    w = A.poids_visage(MARGE * A.iod, BANDE * A.iod)
    if bord_B:
        dB = G.dijkstra(XB, B.adj, B.boucle_ext)
        _, j = _plus_proche_sommet(XB, rec["phi"])
        lm = rec["arete_moy_B"]
        bande = BANDE * A.iod * A.echelle
        w = w * G.lisser((dB[j] - (ANNEAUX_BORD_B + 0.5) * lm) / bande)
        # likewise before the border of A's skin
        dA = G.dijkstra(XA, A.adj, A.boucle_ext)
        w = w * G.lisser((dA - 2.5 * lm) / bande)
    X = w[:, None] * XA + (1 - w[:, None]) * rec["phi"]
    return rec, w, X


# =================================================================== forme mode

def placer_forme(A, B):
    """Forme: B scaled to A's head size (width × chin-to-skull-top height, area kept), head
    midpoints superimposed, then A's face follows the head: stretched in width and height as
    the head goes from A to B, around the middle of the head (the eyes stay at the same
    relative height, the forehead and chin keep their share of the head). Returns B's scale,
    the face's diagonal matrix K and its centre."""
    s = float(np.sqrt(A.W * A.H_crane / (B.W * B.H_crane)))
    yA = 0.5 * (A.y_crane + A.V[A.reperes["menton"], 1])
    yB = 0.5 * (B.y_crane + B.V[B.reperes["menton"], 1])
    cA, cB = _centre_x0(A.centre_visage()), _centre_x0(B.centre_visage())
    cA[1], cB[1] = yA, yB
    K = np.array([s * B.W / A.W, s * B.H_crane / A.H_crane, 1.0])
    w = A.poids_visage(MARGE * A.iod, BANDE * A.iod) > 0.5
    c0 = cA.copy()
    for _ in range(3):  # B moved forward or back until it touches the stretched face
        B.placer(s, cB, cA)
        Q = c0 + K * (A.V[w] - c0)
        c = G.plus_proche(Q, B._hote(B.V), B.F)[0]
        cA = cA - np.array([0.0, 0.0, float(np.median(c[:, 2] - Q[:, 2]))])
    B.placer(s, cB, cA)
    return s, K, c0


def _rayons_touchent(O, D, T, dmax):
    """Does each point's ray (O, D) hit one of the triangles T (n, 3, 3) before dmax?"""
    hit = np.zeros(len(O), dtype=bool)
    e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    for k in range(0, len(O), 64):
        o, d = O[k:k + 64, None], D[k:k + 64, None]
        p = np.cross(d, e2[None])
        det = (e1[None] * p).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = 1.0 / np.where(ok, det, 1.0)
        t0 = o - T[None, :, 0]
        u = (t0 * p).sum(-1) * inv
        q = np.cross(t0, e1[None])
        v = (d * q).sum(-1) * inv
        t = (e2[None] * q).sum(-1) * inv
        hit[k:k + 64] = (ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4) & (t < dmax)).any(1)
    return hit


def sommets_cheveux(A):
    ih = [i for i, n in enumerate(A.noms_mat) if "髮" in n or "发" in n]
    return np.unique(A.faces[np.isin(A.mat_face, ih)]), A.faces[np.isin(A.mat_face, ih)]


def zone_fixe(A):
    """Weight (1 = does not move) of A's skin hidden under its hair (an outward ray hits hair
    within DISTANCE_CHEVEUX eye distances) and of its ears (RAYON_OREILLE around the most
    lateral point), smooth transition over BANDE."""
    _, Fh = sommets_cheveux(A)
    N = G.normales_sommets(A.V, A.F)
    # outward direction: the one moving away from the centre of the head
    if ((A.V - A.V.mean(0)) * N).sum(1).mean() < 0:
        N = -N
    sous = (_rayons_touchent(A.V, N, A.pos[Fh], DISTANCE_CHEVEUX * A.iod) if len(Fh)
            else np.zeros(A.n, dtype=bool))
    oreille = np.zeros(A.n, dtype=bool)
    for cote in ("g", "d"):
        oreille |= G.dijkstra(A.V, A.adj, [A.reperes["oreille_" + cote]]) < RAYON_OREILLE * A.iod
    src = np.where(sous | oreille)[0]
    if not len(src):
        return np.zeros(A.n), {"sous_cheveux": 0, "oreilles": 0}
    d = G.dijkstra(A.V, A.adj, src)
    return 1 - G.lisser(d / (BANDE * A.iod)), {"sous_cheveux": int(sous.sum()), "oreilles": int(oreille.sum())}


def mode_forme(A, B, sortie):
    s, K, c0 = placer_forme(A, B)
    XA = A.V.copy()
    XB = B._hote(B.V)
    XK = c0 + K * (XA - c0)
    rec, w, X = poser_visage(A, XK, B, XB, sur_place=True)
    # A's hair and ears do not move: the skin under the hair (B's shape would not be seen
    # there and would pierce the hair) and the ears keep their place on A
    q, nfixe = zone_fixe(A)
    q = q * (1 - w)
    X = q[:, None] * XA + (1 - q[:, None]) * X
    # neck seam: gradual return to A
    dcou = G.dijkstra(XA, A.adj, A.cou) if len(A.cou) else np.full(A.n, np.inf)
    p = 1 - G.lisser(dcou / (RAYON_COU * A.iod))
    X = p[:, None] * XA + (1 - p[:, None]) * X
    X = lisser_deplacement(XA, X, A.F, fixe=(w >= 0.999) | (p >= 0.999) | (q >= 0.999))
    D = X - XA
    Rv = rotations_locales(XA, X, A.F)

    # --- output: copy of A, with modified positions, normals and morphs
    P = copy.deepcopy(A.pmx)
    P["textures"] = [_texture_rel(x, os.path.dirname(A.chemin), sortie) for x in P["textures"]]
    pos = np.asarray(A.pos).copy()
    nor = np.asarray(A.t["nor"]).copy()
    vb = A.peau_bruts
    k = A.b2s[vb]
    pos[vb] = X[k]
    nor[vb] = np.einsum("nij,nj->ni", Rv[k], nor[vb])
    # vertices that follow the head (face parts, accessories); the hair stays
    autres = np.where((A.poids_tete > 0) & (A.b2s < 0))[0]
    autres = np.setdiff1d(autres, sommets_cheveux(A)[0])
    Dsh = G.shepard(XA, D, A.pos[autres], k=16)
    pos[autres] += Dsh * A.poids_tete[autres][:, None]
    _, jn = _plus_proche_sommet(XA, A.pos[autres])
    nor[autres] = np.einsum("nij,nj->ni", Rv[jn], nor[autres])
    # rotation per raw vertex, for the morphs, after the face stretch (weighted by w)
    Kv = 1 + w[:, None] * (K - 1)
    Rb = {}
    for i, kk in zip(vb, k):
        Rb[int(i)] = Rv[kk] * Kv[kk]
    for i, j in zip(autres, jn):
        Rb[int(i)] = Rv[j] * Kv[j]
    for m in P["morphes"]:
        if m["type"] != 1:
            continue
        m["decalages"] = [(i, tuple(Rb[i] @ np.asarray(d)) if i in Rb else d) for i, d in m["decalages"]]
    P["sommets"]["pos"] = [tuple(x) for x in pos]
    P["sommets"]["nor"] = [tuple(x) for x in nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)]
    P["commentaire"] = (P.get("commentaire") or "") + "\r\n[retarget_tete forme: skull of %s]" % B.nom
    chemin = os.path.join(sortie, "resultat.pmx")
    pmx_ecrire.ecrire(P, chemin)
    np.savez_compressed(os.path.join(sortie, "recalage.npz"), X=X, XA=XA, XB=XB, FA=A.F, FB=B.F,
                        phi=rec["phi"], w=w, p_cou=p, X0=rec["X0"], Xicp=rec["X"])
    yeux = 0.5 * (X[A.boucle_oeil["g"]].mean(0) + X[A.boucle_oeil["d"]].mean(0))[1]
    yeux0 = 0.5 * (XA[A.boucle_oeil["g"]].mean(0) + XA[A.boucle_oeil["d"]].mean(0))[1]
    menton, crane = X[A.reperes["menton"], 1], B._hote(np.array([[0.0, B.y_crane, 0.0]]))[0, 1]
    info = {"echelle_cible": s, "etirement_visage": K.round(4).tolist(), "zone_fixe": nfixe,
            "hauteur_yeux_sur_tete": {"A": round(float((yeux0 - A.V[A.reperes["menton"], 1]) / A.H_crane), 4),
                                      "resultat": round(float((yeux - menton) / (crane - menton)), 4)},
            "suivi_icp": rec["suivi"], "sommets_suiveurs": int(len(autres)),
            "deplacement_max_suiveurs": float(np.linalg.norm(Dsh, axis=1).max()) if len(autres) else 0.0,
            "zone": {"visage_w>0.5": int((w > 0.5).sum()), "bande": int(((w > 0.02) & (w < 0.98)).sum()),
                     "crane_w<0.02": int((w <= 0.02).sum())}}
    return chemin, info, {"w": w, "p_cou": p, "q": q, "XB": XB, "rec": rec, "K": K,
                          "idxA": np.arange(len(A.pos))}


def _plus_proche_sommet(V, Q):
    j = np.zeros(len(Q), dtype=np.int64)
    d = np.zeros(len(Q))
    for s in range(0, len(Q), 2048):
        dd = np.linalg.norm(Q[s:s + 2048, None] - V[None], axis=2)
        j[s:s + 2048] = dd.argmin(1)
        d[s:s + 2048] = dd.min(1)
    return d, j


# =================================================================== visage mode

def _carte_os(A, B):
    """A's bones -> B's bones by name (mouth synonyms), otherwise B's 頭."""
    nb = {n: i for i, n in enumerate(B.noms_os)}
    syn = {"口左": "左口", "口右": "右口", "口中": "口", "左口": "口左", "右口": "口右", "口": "口中"}
    tete = nb.get("頭", 0)
    out = []
    for n in A.noms_os:
        if n in nb:
            out.append(nb[n])
        elif syn.get(n) in nb:
            out.append(nb[syn[n]])
        else:
            out.append(tete)
    return np.array(out)


def _texture_rel(chemin_tex, dossier_modele, sortie):
    p = os.path.normpath(os.path.join(dossier_modele, chemin_tex.replace("\\", "/")))
    return os.path.relpath(p, sortie).replace("/", "\\")


def _distance_polyligne(P, Q):
    """Distance from each point of P to the closed polyline Q."""
    Q2 = np.vstack([Q, Q[:1]])
    a, ab = Q2[:-1], Q2[1:] - Q2[:-1]
    out = np.empty(len(P))
    for k in range(0, len(P), 512):
        p = P[k:k + 512]
        t = np.clip((((p[:, None] - a[None]) * ab[None]).sum(2)) / np.maximum((ab * ab).sum(1), 1e-18)[None], 0, 1)
        out[k:k + 512] = np.linalg.norm(p[:, None] - (a[None] + t[..., None] * ab[None]), axis=2).min(1)
    return out


def _contient_arete(F, u, v):
    """True if a face of F carries the oriented edge u -> v."""
    return bool(((F[:, 0] == u) & (F[:, 1] == v)).any() or ((F[:, 1] == u) & (F[:, 2] == v)).any()
                or ((F[:, 2] == u) & (F[:, 0] == v)).any())


def bande_entre_boucles(Pa, Pb, Nb):
    """Triangulates the strip between two neighbouring closed loops with the same orientation.
    Pa: patch border, walked in the direction of its faces; Pb: hole border, walked against
    the direction of its faces; Nb: B's normals at Pb. Optimal triangulation by dynamic
    programming (Fuchs, Kedem and Uselton 1977): sum of diagonals, plus a heavy penalty for
    each triangle flipped relative to B. Returns triangles [(("a"|"b"), index) × 3]."""
    na, nb = len(Pa), len(Pb)
    j0 = int(np.linalg.norm(Pb - Pa[0], axis=1).argmin())
    A_ = lambda i: Pa[i % na]
    B_ = lambda j: Pb[(j0 + j) % nb]
    NB_ = lambda j: Nb[(j0 + j) % nb]

    def cout(p, q, r, n, diag):
        c = np.cross(q - p, r - p)
        return diag + (1e3 if np.dot(c, n) <= 0 else 0.0)

    INF = np.inf
    C = np.full((na + 1, nb + 1), INF)
    pas = np.zeros((na + 1, nb + 1), dtype=np.int8)
    C[0, 0] = 0.0
    for i in range(na + 1):
        for j in range(nb + 1):
            if i == 0 and j == 0:
                continue
            best, k = INF, 0
            if i > 0 and C[i - 1, j] < INF:  # triangle (a_i, a_{i-1}, b_j)
                v = C[i - 1, j] + cout(A_(i), A_(i - 1), B_(j), NB_(j), np.linalg.norm(A_(i) - B_(j)))
                if v < best:
                    best, k = v, 1
            if j > 0 and C[i, j - 1] < INF:  # triangle (b_{j-1}, b_j, a_i)
                v = C[i, j - 1] + cout(B_(j - 1), B_(j), A_(i), NB_(j), np.linalg.norm(B_(j) - A_(i)))
                if v < best:
                    best, k = v, 2
            C[i, j], pas[i, j] = best, k
    tri = []
    i, j = na, nb
    while i > 0 or j > 0:
        if pas[i, j] == 1:
            tri.append([("a", i % na), ("a", (i - 1) % na), ("b", (j0 + j) % nb)])
            i -= 1
        else:
            tri.append([("b", (j0 + j - 1) % nb), ("b", (j0 + j) % nb), ("a", i % na)])
            j -= 1
    tri = tri[::-1]
    P = lambda c, q: Pa[q] if c == "a" else Pb[q]
    ret = []
    for t in tri:
        p = [P(c, q) for c, q in t]
        jb = [q for c, q in t if c == "b"][0]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), Nb[jb]) <= 0:
            ret.append(t)
    return tri, ret

def anneaux(n, adj, sources):
    """Distance in rings (edges) from source vertices."""
    d = np.full(n, 10 ** 9)
    d[list(sources)] = 0
    front = list(sources)
    k = 0
    while front:
        k += 1
        nf = []
        for u in front:
            for v in adj[u]:
                if d[v] > k:
                    d[v] = k
                    nf.append(v)
        front = nf
    return d


def harmoniser_morphes(P, mats_peau, graines, anneaux_k=ANNEAUX_HARMONIE):
    """Morphs on both sides of the seam: the offsets of A (patch) and of B (the rest) do not
    join up on the strip. On skin vertices within `anneaux_k` rings of the strip, the offsets
    are replaced by the harmonic interpolation (zero Laplacian) of the values held beyond:
    the seam follows both sides without tearing."""
    pos = np.asarray(P["sommets"]["pos"], dtype=float)
    F = np.asarray(P["faces"], dtype=np.int64)
    mf = np.repeat(np.arange(len(P["materiaux"])), [m["n_indices"] // 3 for m in P["materiaux"]])
    Fs = F[np.isin(mf, mats_peau)]
    vb = np.unique(Fs)
    rep, inv = G.souder(pos[vb])
    b2s = np.full(len(pos), -1)
    b2s[vb] = inv
    n = len(rep)
    Fw = b2s[Fs]
    Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
    adj = G.voisins(n, Fw)
    d = anneaux(n, adj, np.unique(b2s[graines]))
    libre = d <= anneaux_k
    tenu = ~libre
    L = G.laplacien(n, Fw)
    Lff = L[np.ix_(libre, libre)]
    Lft = L[np.ix_(libre, tenu)]
    il = np.where(libre)[0]
    lisiere = np.array([tenu[j] and any(libre[x] for x in adj[j]) for j in range(n)])
    bruts_de = [[] for _ in range(n)]
    for i, k in zip(vb, inv):
        bruts_de[k].append(int(i))
    touches = 0
    for m in P["morphes"]:
        if m["type"] != 1 or not m["decalages"]:
            continue
        D = np.zeros((n, 3))
        c = np.zeros(n)
        autres = []
        for v, x in m["decalages"]:
            k = b2s[v]
            if k >= 0:
                D[k] += x
                c[k] += 1
            else:
                autres.append((v, x))
        D /= np.maximum(c, 1)[:, None]
        if not np.abs(D[libre | lisiere]).max() > 0:
            continue
        Df = np.linalg.solve(Lff, -Lft @ D[tenu])
        D[il] = Df
        dec = list(autres)
        for k in np.where(np.abs(D).max(1) > 1e-7)[0]:
            for v in bruts_de[k]:
                dec.append((v, tuple(D[k])))
        m["decalages"] = dec
        touches += 1
    return {"sommets_libres": int(libre.sum()), "morphes_harmonises": touches}


def regulariser_bord(F, sel, iters=20):
    """Border of a face selection without spikes or notches: selected faces with only one
    selected neighbour are removed, neighbouring faces with two or three are added, until
    stable."""
    _, _, fe = G.aretes(F)
    # two faces per interior edge
    ne = int(fe.max()) + 1
    f1 = np.full(ne, -1)
    f2 = np.full(ne, -1)
    for f in range(len(F)):
        for e in fe[f]:
            if f1[e] < 0:
                f1[e] = f
            else:
                f2[e] = f
    vois = np.where(f1[fe] == np.arange(len(F))[:, None], f2[fe], f1[fe])
    s = sel.copy()
    for _ in range(iters):
        nv = np.where(vois >= 0, s[np.maximum(vois, 0)], False).sum(1)
        ajout = ~s & (nv >= 2)
        retrait = s & (nv <= 1)
        if not ajout.any() and not retrait.any():
            break
        s = (s | ajout) & ~retrait
    return s


def raffiner_pres_couture(A, X, Fp, B, XB):
    """Marks B's skin faces near the patch border (radius R_RAFFINAGE edges of B) whose longest
    edge exceeds 1.3 times A's mean edge, then splits them."""
    ua, _, _ = G.aretes(Fp)
    eA = float(np.linalg.norm(X[ua[:, 0]] - X[ua[:, 1]], axis=1).mean())
    trous_A = set()
    for b in A.boucles:
        trous_A.update(int(x) for x in b)
    bord = [np.array(b) for b in G.boucles_bord(Fp) if not all(int(x) in trous_A for x in b)]
    if len(bord) != 1:
        return None
    Pb = X[bord[0]]
    T = XB[B.F]
    lg = np.max(np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2), axis=1)
    c = T.mean(1)
    d = np.min(np.linalg.norm(c[:, None] - Pb[None], axis=2), axis=1)
    marque = (d < R_RAFFINAGE * lg) & (lg > 1.3 * eA)
    if not marque.any():
        return None
    P = B.pmx
    info = {"faces_marquees": int(marque.sum()), "arete_A": round(eA, 4)}
    for m in B.i_peau:
        P, i = RF.raffiner(P, m, B.F_brut[marque])
        for k, x in i.items():
            info[k] = info.get(k, 0) + x
    return TT.Tete(B.chemin, peau=B.mats_peau, pmx=P), info


def placer_proportions_source(A, B):
    """Places A on B while keeping the proportions of A's face relative to its head: A's
    chin-to-skull-top height becomes B's, and the middle of one goes onto the middle of the
    other (the eyes land at the same relative height as on A, the forehead keeps its length
    in eye distances). The skull top is taken under the hair (Tete.y_crane): the skin alone
    stops higher or lower depending on the model. The depth is set so that A's face touches B."""
    yA = 0.5 * (A.y_crane + A.V[A.reperes["menton"], 1])
    yB = 0.5 * (B.y_crane + B.V[B.reperes["menton"], 1])
    cA, cB = _centre_x0(A.centre_visage()), _centre_x0(B.centre_visage())
    cA[1], cB[1] = yA, yB
    s = B.H_crane / A.H_crane
    w = A.poids_visage(MARGE * A.iod, BANDE * A.iod) > 0.5
    for _ in range(3):
        A.placer(s, cA, cB)
        Q = A._hote(A.V[w])
        c = G.plus_proche(Q, B.V, B.F)[0]
        cB = cB + np.array([0.0, 0.0, float(np.median(c[:, 2] - Q[:, 2]))])
    A.placer(s, cA, cB)
    return s


def mode_visage(A, B, sortie, proportions="cible"):
    sur_place = proportions == "source"
    if sur_place:
        s = placer_proportions_source(A, B)
    else:
        # A brought onto B's face landmarks (uniform scale, translation, no rotation)
        s = echelle_face(A, B)
        A.placer(s, _centre_x0(A.centre_visage()), _centre_x0(B.centre_visage()))
    XA = A._hote(A.V)
    XB = B.V.copy()
    raffinage = []
    for passe in range(NIVEAUX_RAFFINAGE + 1):
        rec, w, X = poser_visage(A, XA, B, XB, bord_B=True, sur_place=sur_place)
        # --- patch: A's faces of the face and the strip, as a disc, away from A's outer border
        wf = w[A.F].mean(1)
        # the patch also covers B's own face (there, A already lies on B: w ≈ 0)
        wB = B.poids_visage(MARGE * B.iod, BANDE * B.iod)
        wBphi = (wB[B.F[rec["face"]]] * rec["bary"]).sum(1)
        sel = (wf >= 0.05) | (wBphi[A.F].mean(1) >= 0.3)
        sel = G.remplir_disque(A.F, sel)
        Fp = A.F[sel]
        if passe == NIVEAUX_RAFFINAGE:
            break
        # --- B refined near the patch border, down to the size of A's edges: the seam then
        # joins two borders with the same spacing (B's surface does not move)
        Bnew = raffiner_pres_couture(A, X, Fp, B, XB)
        if Bnew is None:
            break
        B, info_r = Bnew
        XB = B.V.copy()
        raffinage.append(info_r)
        journal("B refined near the seam: %s" % info_r)
    # --- patch border
    bp = [np.array(b) for b in G.boucles_bord(Fp)]
    trous_A = set()
    for b in A.boucles:
        trous_A.update(int(x) for x in b)
    la = [b for b in bp if not all(int(x) in trous_A for x in b)]
    if len(la) != 1:
        raise RuntimeError("patch border: %d loops instead of one" % len(la))
    La = np.asarray(la[0])
    La_brut = la[0]
    # --- hole in B: every vertex of B under the patch (closest point inside the patch: zero
    # tangential gap) or within δ of the patch border is removed with its faces; the hole
    # border thus stays entirely outside the patch, at a gap of δ to δ + one edge
    lm = rec["arete_moy_B"]
    pq, fq, _, dq = G.plus_proche(XB, X, Fp)
    rq = XB - pq
    nq = G.normales_faces(X, Fp)[fq]
    tang = np.linalg.norm(rq - (rq * nq).sum(1, keepdims=True) * nq, axis=1)
    dL = _distance_polyligne(XB, X[La])
    eB = np.linalg.norm(XB[B.F] - np.roll(XB[B.F], 1, axis=1), axis=2).mean(1)
    pres = _distance_polyligne(XB[B.F].mean(1), X[La]) < 3 * eB
    delta = ECART_COUTURE * float(np.median(eB[pres]))
    couvert = ((dq < lm) & (tang < 0.05 * lm)) | (dL < delta)
    # B's eye and mouth holes: to be removed (the fill takes what they enclose)
    traits = np.zeros(B.n, dtype=bool)
    for b in (B.boucle_oeil["g"], B.boucle_oeil["d"], B.boucle_bouche):
        traits[b] = True
    base = (couvert | traits)[B.F].any(1)
    trou = G.remplir_disque(B.F, base)
    trou = regulariser_bord(B.F, trou) | base
    trou = G.remplir_disque(B.F, trou)
    extB = np.zeros(B.n, dtype=bool)
    extB[B.boucle_ext] = True
    if extB[B.F[trou]].any():
        journal("hole: %d faces touch B's outer border" % int(extB[B.F[trou]].any(1).sum()))
        raise RuntimeError("the face hole touches the outer border of B's skin")
    # --- loops to sew, then the strip; a notch in the hole border that the strip cannot cover
    # without flipping is removed with the hole (a few passes at most)
    if not _contient_arete(Fp, La[0], La[1]):
        La = La[::-1].copy()
    na = len(La)
    trous_B = set()
    for b in B.boucles:
        trous_B.update(int(x) for x in b)
    nBv = G.normales_sommets(XB, B.F)
    encoches = 0
    for passe in range(5):
        Fr = B.F[~trou]
        br = [np.array(b) for b in G.boucles_bord(Fr)]
        lb = [b for b in br if not all(int(x) in trous_B for x in b)]
        if len(lb) != 1:
            raise RuntimeError("hole border: %d loops instead of one" % len(lb))
        Lb = np.asarray(lb[0])
        if _contient_arete(Fr, Lb[0], Lb[1]):
            Lb = Lb[::-1].copy()
        nb = len(Lb)
        bande, retournes = bande_entre_boucles(X[La], XB[Lb], nBv[Lb])
        if not retournes or passe == 4:
            break
        ajout = np.zeros(len(B.F), dtype=bool)
        for t in retournes:
            vb_ = [int(Lb[q]) for c, q in t if c == "b"]
            ajout |= ~trou & np.isin(B.F, vb_).sum(1).astype(bool) if len(vb_) == 1 else \
                ~trou & (np.isin(B.F, vb_).sum(1) == 2)
        trou = G.remplir_disque(B.F, trou | ajout)
        encoches += int(ajout.sum())
        if extB[B.F[trou]].any():
            raise RuntimeError("the face hole touches the outer border of B's skin")
    bande_ret = len(retournes)
    ecart = _distance_polyligne(XB[Lb], X[La])
    journal("seam: patch %d faces (border %d), hole %d faces (border %d), strip of %d triangles, "
            "border gap median %.4f max %.4f (δ = %.4f), %d notch faces removed, %d flipped" % (
                len(Fp), na, int(trou.sum()), nb, len(bande), np.median(ecart), ecart.max(), delta,
                encoches, bande_ret))

    # =============================================================== PMX assembly
    Pa, Pb = A.pmx, B.pmx
    sa_, sb_ = Pa["sommets"], Pb["sommets"]
    nuv = Pb["n_uv_add"]
    carte = _carte_os(A, B)
    pieces_B = set(B.i_pieces)
    i_peauB = B.i_peau[0]
    # A already composed (颜 + 颜_A): each patch face keeps its material, hence its texture;
    # the face is its 颜_A, the rest of its skin becomes 颜_A2
    i_visA = A.noms_mat.index(TT.PEAU_COUSUE) if TT.PEAU_COUSUE in A.noms_mat else A.i_peau[0]
    # B's raw faces kept
    fb_trou = set(int(x) for x in B.F_brut[trou])
    garde_fB = np.array([not (B.mat_face[f] in pieces_B or f in fb_trou) for f in range(len(B.faces))])
    # B's raw faces duplicated in the welded skin (degenerate faces): they follow their face
    vertsB_gardes = np.unique(B.faces[garde_fB])
    # A's vertices used: patch skin and parts
    fA_patch = A.F_brut[sel]
    fA_pieces = np.where(np.isin(A.mat_face, A.i_pieces))[0]
    vertsA = np.unique(np.concatenate([A.faces[fA_patch].ravel(), A.faces[fA_pieces].ravel()]))
    nB = len(vertsB_gardes)
    idxB = np.full(len(B.pos), -1)
    idxB[vertsB_gardes] = np.arange(nB)
    idxA = np.full(len(A.pos), -1)
    idxA[vertsA] = nB + np.arange(len(vertsA))
    # A's positions: skin -> X, parts -> displacement of the nearby skin
    posA = A._hote(A.pos[vertsA])
    norA = np.asarray(A.t["nor"])[vertsA].copy()
    Rv = rotations_locales(XA, X, A.F)
    estpeau = A.b2s[vertsA] >= 0
    kpe = A.b2s[vertsA[estpeau]]
    D = X - XA
    posA[estpeau] = X[kpe]
    norA[estpeau] = np.einsum("nij,nj->ni", Rv[kpe], norA[estpeau])
    pi = ~estpeau
    Dsh = G.shepard(XA, D, posA[pi], k=16)
    posA[pi] += Dsh
    _, jn = _plus_proche_sommet(XA, posA[pi])
    norA[pi] = np.einsum("nij,nj->ni", Rv[jn], norA[pi])
    RvA = np.zeros((len(vertsA), 3, 3))
    RvA[estpeau] = Rv[kpe]
    RvA[pi] = Rv[jn]
    n_dup0 = nB + len(vertsA)
    # B's raw vertex for each welded vertex of the hole border: the one from the kept faces
    rep_b = {}
    for f in np.where(garde_fB & (B.mat_face == i_peauB))[0]:
        for v in B.faces[f]:
            k = B.b2s[v]
            if k >= 0 and k not in rep_b:
                rep_b[int(k)] = int(v)

    # --- vertex arrays
    S = {"pos": [], "nor": [], "uv": [], "uv_add": [], "type": [], "os": [], "poids": [],
         "sdef": [], "contour": []}

    def ajouter_B(v):
        for key in S:
            S[key].append(sb_[key][v])

    def uva(src, v):
        u = list(src["uv_add"][v])[:nuv]
        return u + [(0.0, 0.0, 0.0, 0.0)] * (nuv - len(u))

    def ajouter_A(v, p, n, uv=None):
        t = sa_["type"][v]
        o = [int(carte[x]) if x >= 0 else -1 for x in sa_["os"][v]]
        w = list(sa_["poids"][v])
        if t == 3:
            t = 1
        if t == 4:
            t = 2
        if t == 0:
            o = [o[0], -1, -1, -1]
        S["pos"].append(tuple(p))
        S["nor"].append(tuple(n / max(np.linalg.norm(n), 1e-12)))
        S["uv"].append(tuple(uv) if uv is not None else sa_["uv"][v])
        S["uv_add"].append(uva(sa_, v))
        S["type"].append(t)
        S["os"].append(tuple(o))
        S["poids"].append(tuple(w))
        S["sdef"].append(None)
        S["contour"].append(sa_["contour"][v])

    for v in vertsB_gardes:
        ajouter_B(int(v))
    for j, v in enumerate(vertsA):
        ajouter_A(int(v), posA[j], norA[j])
    # seam normals: B's, interpolated along the hole border
    norB = np.asarray(sb_["nor"], dtype=float)
    uvB = np.asarray(sb_["uv"], dtype=float)
    # on the patch border, B's normals and UVs at the projected point (B's skin carries on)
    fbr = B.faces[B.F_brut[rec["face"][La]]]
    br_ = rec["bary"][La]
    nor_La = (br_[:, :, None] * norB[fbr]).sum(1)
    uv_La = (br_[:, :, None] * uvB[fbr]).sum(1)
    for i in range(na):
        for v in A.s2b[int(La[i])]:
            j = idxA[v]
            if j >= 0:
                S["nor"][j] = tuple(nor_La[i] / max(np.linalg.norm(nor_La[i]), 1e-12))
    dupA_de = {}  # A's raw vertex (patch border) -> duplicates with B's material
    dupB_de = {}  # B's raw vertex (hole border) -> duplicates with A's material

    def nouveau():
        return len(S["pos"]) - 1


    # --- faces per material, order: B's materials up to 颜, then A's, then the rest
    faces_par_mat = {}
    for f in np.where(garde_fB)[0]:
        faces_par_mat.setdefault(("B", int(B.mat_face[f])), []).append(tuple(int(idxB[v]) for v in B.faces[f]))
    for f in fA_patch:
        faces_par_mat.setdefault(("A", int(A.mat_face[f])), []).append(tuple(int(idxA[v]) for v in A.faces[f]))
    for f in fA_pieces:
        faces_par_mat.setdefault(("A", int(A.mat_face[f])), []).append(tuple(int(idxA[v]) for v in A.faces[f]))
    # seam strip, with B's material: copies of the patch border vertices (A's weights,
    # B's UVs), and the hole border vertices
    copie_La = {}
    for i in range(na):
        va_ = A.s2b[int(La[i])][0]
        ajouter_A(int(va_), X[La[i]], nor_La[i], uv=uv_La[i])
        copie_La[i] = nouveau()
        dupA_de.setdefault(int(va_), []).append(nouveau())
    for t in bande:
        faces_par_mat.setdefault(("B", i_peauB), []).append(tuple(
            copie_La[q] if c == "a" else int(idxB[rep_b[int(Lb[q])]]) for c, q in t))
    ordre = []
    for i in range(len(Pb["materiaux"])):
        if i in pieces_B:
            continue
        ordre.append(("B", i))
        if i == i_peauB:
            ordre += [("A", j) for j in A.i_peau if j == i_visA or ("A", j) in faces_par_mat]
            ordre += [("A", j) for j in A.i_pieces]
    P = copy.deepcopy(Pb)
    off_tex = len(Pb["textures"])
    P["textures"] = [_texture_rel(x, os.path.dirname(B.chemin), sortie) for x in Pb["textures"]] + [_texture_rel(x, os.path.dirname(A.chemin), sortie) for x in Pa["textures"]]
    mats, faces, nouv_mat = [], [], {}
    for cle in ordre:
        src, i = cle
        m = copy.deepcopy((Pb if src == "B" else Pa)["materiaux"][i])
        if src == "A":
            for kk in ("texture", "sphere"):
                if m[kk] >= 0:
                    m[kk] += off_tex
            if m["toon_partage"] == 0 and m["toon"] >= 0:
                m["toon"] += off_tex
            if i in A.i_peau:
                suffixe = "" if i == i_visA else "2"
                m["nom"] = TT.PEAU_COUSUE + suffixe
                m["nom_en"] = (m["nom_en"] or "face").removesuffix("_A") + "_A" + suffixe
        fl = faces_par_mat.get(cle, [])
        m["n_indices"] = 3 * len(fl)
        nouv_mat[cle] = len(mats)
        mats.append(m)
        faces += fl
    P["materiaux"] = mats
    P["faces"] = faces
    P["sommets"] = S

    # --- morphs: B's are kept (on B's kept vertices); for a name that A also has, A's offsets
    # (rotated into the local frame, scaled) are added on the face
    morA = {}
    for m in Pa["morphes"]:
        if m["type"] == 1:
            morA[m["nom"]] = m
    posA_de = {int(v): j for j, v in enumerate(vertsA)}
    dup_de = dupA_de
    n_dup = len(S["pos"]) - n_dup0

    def decalages_A(m):
        out = []
        for v, d in m["decalages"]:
            j = posA_de.get(int(v))
            if j is None:
                continue
            dd = tuple(A.echelle * (RvA[j] @ np.asarray(d)))
            out.append((nB + j, dd))
            for jd in dup_de.get(int(v), []):
                out.append((jd, dd))
        return out

    mor = []
    vus = set()
    MoB = None
    for m in Pb["morphes"]:
        m = copy.deepcopy(m)
        t = m["type"]
        if t in (0, 2) and m["nom"] in morA:
            # B's bone (or group) morph replaced by a vertex morph of A: its effect on B's
            # kept vertices becomes offsets, and A's are added to them (a single morph with
            # this name)
            import morphs as MO
            MoB = MoB or MO.Modele(pmx=Pb)
            dd = MoB.deplacement(m["nom"])
            m["type"] = 1
            m["decalages"] = [(int(idxB[v]), tuple(dd[v])) for v in np.where(np.abs(dd).max(1) > 1e-7)[0]
                              if idxB[v] >= 0] + decalages_A(morA[m["nom"]])
            vus.add(m["nom"])
            mor.append(m)
            continue
        if t == 1 or 3 <= t <= 7:
            dec = [(int(idxB[v]), d) for v, d in m["decalages"] if idxB[v] >= 0]
            if t == 1:
                dec += [(jd, d) for v, d in m["decalages"] for jd in dupB_de.get(int(v), [])]
            if t == 1 and m["nom"] in morA:
                dec += decalages_A(morA[m["nom"]])
                vus.add(m["nom"])
            m["decalages"] = dec
        elif t == 8:
            dec = []
            for d in m["decalages"]:
                if d["materiau"] < 0:
                    dec.append(d)
                elif ("B", d["materiau"]) in nouv_mat:
                    d = dict(d)
                    d["materiau"] = nouv_mat[("B", d["materiau"])]
                    dec.append(d)
            m["decalages"] = dec
        mor.append(m)
    ajout = []
    for nom, m in morA.items():
        if nom in vus:
            continue
        m = copy.deepcopy(m)
        m["decalages"] = decalages_A(m)
        ajout.append(len(mor))
        mor.append(m)
    P["morphes"] = mor
    harm = harmoniser_morphes(P, [nouv_mat[("B", i_peauB)]] + [nouv_mat[("A", j)] for j in A.i_peau if ("A", j) in nouv_mat],
                              np.arange(n_dup0, n_dup0 + n_dup))
    for c in P["cadres"]:
        if c["nom"] == "表情":
            c["elements"] = list(c["elements"]) + [(1, k) for k in ajout]
    P["commentaire"] = (P.get("commentaire") or "") + "\r\n[retarget_tete visage: face of %s]" % A.nom
    chemin = os.path.join(sortie, "resultat.pmx")
    pmx_ecrire.ecrire(P, chemin)
    np.savez_compressed(os.path.join(sortie, "recalage.npz"), X=X, XA=XA, XB=XB, FA=A.F, FB=B.F,
                        phi=rec["phi"], w=w, patch=sel, trou=trou, X0=rec["X0"], Xicp=rec["X"])
    info = {"echelle_source": s, "suivi_icp": rec["suivi"], "premier_doublon": int(n_dup0),
            "patch_faces": int(sel.sum()), "trou_faces": int(trou.sum()),
            "bord_patch": int(na), "bord_trou": int(nb), "couture_bande_triangles": len(bande),
            "couture_bande_retournes": bande_ret, "couture_encoches_retirees": encoches,
            "couture_ecart_median": float(np.median(ecart)), "couture_ecart_max": float(ecart.max()),
            "couture_delta": delta,
            "morphes_de_A_ajoutes": [mor[k]["nom"] for k in ajout], "harmonie_couture": harm,
            "raffinage_B": raffinage,
            "zone": {"visage_w>0.5": int((w > 0.5).sum()), "bande": int(((w > 0.02) & (w < 0.98)).sum())}}
    return chemin, info, {"B_raffine": B, "w": w, "XB": XB, "rec": rec, "sel": sel, "trou": trou,
                          "La": La, "Lb": Lb, "idxA": idxA, "vertsB_gardes": vertsB_gardes,
                          "n_dup0": n_dup0, "n_dup": n_dup, "bord_attendu":
                          sum(len(x) for x in bp if x is not La_brut) + sum(len(x) for x in br if x is not lb[0])}


# =================================================================== main

def planche(mode, source, cible, resultat, sortie):
    script = os.path.join(ICI, "head_sheet.py")
    cmd = ["bash", FILE_BLENDER, "-b", "--factory-startup", "--python", script, "--", mode, source, cible,
           resultat, sortie]
    journal("contact sheet: Blender queue")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("contact sheet: Blender failed")
    return os.path.join(sortie, "planche.png")


def teinte(resultat):
    """visage mode: B's skin takes the colour of A's face (recoloured textures)."""
    cmd = ["bash", FILE_BLENDER, "-b", "--factory-startup", "--python", os.path.join(ICI, "tint.py"), "--", resultat]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or "teinte :" not in r.stdout:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("tint: Blender failed")
    with open(os.path.join(os.path.dirname(resultat), "teinte.json"), encoding="utf-8") as fh:
        return json.load(fh)


def planche_textures(source, cible, resultat, sortie):
    cmd = ["bash", FILE_BLENDER, "-b", "--factory-startup", "--python", os.path.join(ICI, "texture_sheet.py"), "--",
           os.path.join(sortie, "planche_textures.png"), source, os.path.dirname(source),
           cible, os.path.dirname(cible), resultat, sortie]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("textured contact sheet: Blender failed")
    return os.path.join(sortie, "planche_textures.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("forme", "visage"))
    ap.add_argument("--source", required=True, help="A: the face that is kept")
    ap.add_argument("--target", dest="cible", required=True, help="B: the head shape (forme) or the host head (visage)")
    ap.add_argument("--output", dest="sortie", required=True)
    ap.add_argument("--no-sheet", dest="sans_planche", action="store_true")
    ap.add_argument("--proportions", choices=("cible", "source"), default="cible",
                    help="visage: face size and position taken from B (cible) or kept from A relative to its head (source)")
    ap.add_argument("--sans-teinte", action="store_true", help="visage: keep B's skin colour")
    a = ap.parse_args(argv)
    sortie = os.path.abspath(os.path.expanduser(a.sortie))
    os.makedirs(sortie, exist_ok=True)
    t0 = time.time()
    A = TT.Tete(os.path.abspath(a.source))
    B = TT.Tete(os.path.abspath(a.cible))
    journal("A = %s (%d skin vertices), B = %s (%d)" % (A.nom, A.n, B.nom, B.n))
    if a.mode == "forme":
        chemin, info, interne = mode_forme(A, B, sortie)
    else:
        chemin, info, interne = mode_visage(A, B, sortie, a.proportions)
        B = interne.pop("B_raffine")
    journal("result written: %s" % chemin)
    mesures = VF.verifier(a.mode, A, B, chemin, interne, journal=journal)
    mesures["entrees"] = {"mode": a.mode, "source": A.resume(), "cible": B.resume()}
    mesures["parametres"] = {"proportions": a.proportions, "marge_iod": MARGE, "bande_iod": BANDE, "rayon_cou_iod": RAYON_COU, "lissage": LISSAGE}
    mesures["construction"] = info
    if a.mode == "visage" and not a.sans_teinte:
        mesures["teinte"] = teinte(chemin)
        journal("tint: ΔE %.1f -> %.1f" % (mesures["teinte"]["dE_avant"], mesures["teinte"]["dE_apres"]))
    mesures["duree_s"] = round(time.time() - t0, 1)
    ecrire_json(os.path.join(sortie, "mesures.json"), mesures)
    journal("measurements: %s" % ("ALL PASS" if mesures["verdict"]["passe"] else "FAILURES: " + ", ".join(mesures["verdict"]["echecs"])))
    if not a.sans_planche:
        p = planche(a.mode, A.chemin, B.chemin, chemin, sortie)
        journal("contact sheet: %s" % p)
        journal("textured contact sheet: %s" % planche_textures(A.chemin, B.chemin, chemin, sortie))
    return 0 if mesures["verdict"]["passe"] else 1


if __name__ == "__main__":
    sys.exit(main())

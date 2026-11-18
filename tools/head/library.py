"""Library of face parts: eyes, eyebrows, nose, mouth and face base (forehead, cheeks, jaw,
chin: the background on which the other parts sit) of each character, stored on the pivot
head; a face is then composed part by part.

    python3 -I tools/head/library.py extract --output <biblio> [--jobs 4] [--only Jane,Ellen]
    python3 -I tools/head/library.py extract --outerplane --output <biblio> [--only Idith,...]
    python3 -I tools/head/library.py compose --library <biblio> --output <dossier> \
        --base Jane --yeux Ellen --sourcils Miyabi --nez Miyabi --bouche Anby [--onto Belle \
        [--hair Miyabi [--ears Nekomata|aucune]]]

Extract, for each character:
1. the pivot head's skin is registered onto theirs (same registration as pivot.py): same
   mesh for everyone, point to point; each part is a soft mask on this mesh;
2. their skin morphs are sampled at the registered points (blink, vowels, smile...);
3. their skin texture is copied into the pivot head's UVs (for each pivot texel, the
   registered point, then the character's texture at that point); the images are made in
   Blender (library_images.py);
4. their parts (eyeballs, lashes, eyebrows, teeth, tongue...) are sorted by face part, by
   connected piece and by position: eyes, eyebrows, mouth, otherwise base.
Outputs: <biblio>/<Character>/partie.npz, texture.png, fiche.json; <biblio>/index.json
(for each part, the characters that provide it, with the registration reliability; the
parts not provided and why). The index is merged with the one already in <biblio>:
characters that have no folder there keep their entry.

--outerplane: Outerplane heads (outerplane.py) are first converted to a donor PMX in
<biblio>/<Name>/<Name>/tete.pmx (textures next to it), then extracted like the ZZZ ones; the
paths in their record are relative to <biblio>/<Name> (movable folder). A head rejected by
the conversion or by tete.Tete is recorded as not provided, with the reason.

Compose: the shape is the base's, and each part takes its donor's shape in it, moved to the
place it has in the base (without changing size: the donor is already brought to the pivot's
scale); soft masks between parts. Morphs, texture (brought to the base's skin tone) and
parts likewise come from each part's donor. The result is the pivot head (Yuzuha) wearing
the composed face; --onto <character> then places it in one pass of retarget_tete visage.
"""
import argparse
import copy
import heapq
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
sys.path.append(os.path.join(ICI, ".."))
import settings as R  # noqa: E402

import geom as G  # noqa: E402
import morphs as MO  # noqa: E402
import pivot as PV  # noqa: E402
import pmx_write as pmx_ecrire  # noqa: E402
import registration as RC  # noqa: E402
import retarget_head as RT  # noqa: E402
import head as TT  # noqa: E402

PIVOT_JSON = os.path.join(R.chemin("paths.data"), "tete-pivot", "tete-pivot.json")
PARTIES = ("yeux", "sourcils", "nez", "bouche")
BASE = "base"
# masks (in interocular distances): full zone up to the margin, then a transition over the band
MARGE_OEIL, BANDE_OEIL = 0.10, 0.15
MARGE_SOURCIL, BANDE_SOURCIL = 0.06, 0.12
MARGE_BOUCHE, BANDE_BOUCHE = 0.08, 0.15
NEZ_RAYON_HAUT, NEZ_RAYON_BAS, BANDE_NEZ = 0.10, 0.20, 0.12
PROFONDEUR, BANDE_PROFONDEUR = 0.30, 0.20  # beyond this, towards the back, it is no longer the part
# parts: nearest face part (front view) under these distances, otherwise base
PIECE_OEIL, PIECE_SOURCIL, PIECE_BOUCHE = 0.6, 0.6, 0.6
# unambiguous materials; the others (lashes and eyebrows mixed, 颜2) are sorted by position
PAR_MATERIAU = {"yeux": ("目", "白目", "目影", "目光", "目光2", "瞳光", "目色", "二重", "二重影"),
                "sourcils": ("眉", "眉毛"), "bouche": ("齿", "口舌", "口线")}
COTE_RAYON = 0.25  # interocular distances: around a mouth or an eye, upper or lower lip (eyelid)
TEX = 1024  # sampling grid for the donor's UVs (texture rendered at the pivot's size)


def journal(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def charger_pivot():
    with open(PIVOT_JSON, encoding="utf-8") as fh:
        return TT.Tete(json.load(fh)["pmx"])


# =================================================================== part masks

def _tranche(sd, marge, bande):
    return 1 - G.lisser((sd - marge) / bande)


def _profondeur(V, z0, iod):
    # the face looks towards -Z: what lies far behind the part is not part of it
    return 1 - G.lisser((V[:, 2] - z0 - PROFONDEUR * iod) / (BANDE_PROFONDEUR * iod))


def _dist_polyligne_xy(Q, P):
    d = np.full(len(Q), np.inf)
    for a, b in zip(P[:-1], P[1:]):
        ab = b - a
        t = np.clip(((Q - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(Q - (a + t[:, None] * ab), axis=1))
    return d


def masques(A):
    """Part masks on the pivot's welded skin: name -> (n,) in [0, 1]. Eyes and eyebrows have
    one mask per side (yeux_g, yeux_d...). The sum is at most 1; the base takes the rest."""
    V, iod = A.V, A.iod
    xy = V[:, :2]
    brut = {}
    for c in "gd":
        b = A.boucle_oeil[c]
        sd = G.distance_signee_polygone(xy, G.enveloppe_convexe_2d(V[b, :2]))
        brut["yeux_" + c] = _tranche(sd, MARGE_OEIL * iod, BANDE_OEIL * iod) * _profondeur(V, V[b, 2].mean(), iod)
        P = np.array([A.points_sourcils["sourcil_%s_%s" % (c, k)] for k in ("int", "haut", "ext")])
        d = _dist_polyligne_xy(xy, P[:, :2])
        brut["sourcils_" + c] = _tranche(d, MARGE_SOURCIL * iod, BANDE_SOURCIL * iod) * _profondeur(V, P[:, 2].mean(), iod)
    b = A.boucle_bouche
    sd = G.distance_signee_polygone(xy, G.enveloppe_convexe_2d(V[b, :2]))
    brut["bouche"] = _tranche(sd, MARGE_BOUCHE * iod, BANDE_BOUCHE * iod) * _profondeur(V, V[b, 2].mean(), iod)
    # nose: capsule from the bridge (between the eyes) to below the tip, wider at the bottom
    yo = 0.5 * (A.centre_oeil["g"][1] + A.centre_oeil["d"][1])
    pn = V[A.reperes["nez"]]
    h, bas = np.array([0.0, yo - 0.15 * iod]), np.array([0.0, pn[1] - 0.05 * iod])
    ab = bas - h
    t = np.clip(((xy - h) @ ab) / (ab @ ab), 0, 1)
    d = np.linalg.norm(xy - (h + t[:, None] * ab), axis=1)
    r = (NEZ_RAYON_HAUT + t * (NEZ_RAYON_BAS - NEZ_RAYON_HAUT)) * iod
    brut["nez"] = (1 - G.lisser((d - r) / (BANDE_NEZ * iod))) * _profondeur(V, pn[2], iod)
    S = sum(brut.values())
    k = np.maximum(S, 1.0)
    out = {n: m / k for n, m in brut.items()}
    out[BASE] = 1 - np.minimum(S, 1.0)
    return out


def masque_partie(M, partie):
    return sum(m for n, m in M.items() if n == partie or n.startswith(partie + "_"))


# =================================================================== extraction

def _geodesique(T, sources):
    """Distance along the welded skin from source vertices (Dijkstra)."""
    adj = G.voisins(T.n, T.F)
    d = np.full(T.n, np.inf)
    pile = [(0.0, int(v)) for v in sources]
    for _, v in pile:
        d[v] = 0.0
    heapq.heapify(pile)
    while pile:
        dv, v = heapq.heappop(pile)
        if dv > d[v]:
            continue
        for w in adj[v]:
            nd = dv + float(np.linalg.norm(T.V[v] - T.V[w]))
            if nd < d[w]:
                d[w] = nd
                heapq.heappush(pile, (nd, w))
    return d


def cotes(T, rayon):
    """Side of each welded vertex around the openings (mouth, eyes): +1 upper lip or eyelid,
    -1 lower, 0 beyond the radius (along the skin) or at the corners. A closed mouth is an
    almost zero-width slit: without this side, a point of the upper lip projects onto the
    donor's lower lip and takes its motion (the skin folds over the opening)."""
    c = np.zeros(T.n, dtype=np.int8)
    for b in [T.boucle_bouche, T.boucle_oeil["g"], T.boucle_oeil["d"]]:
        b = list(b)
        x = T.V[b, 0]
        i, j = int(np.argmin(x)), int(np.argmax(x))
        if i > j:
            i, j = j, i
        arcs = [b[i + 1:j], b[j + 1:] + b[:i]]
        if min(len(a) for a in arcs) == 0:
            continue
        d0, d1 = _geodesique(T, arcs[0]), _geodesique(T, arcs[1])
        pres = np.minimum(d0, d1) < rayon
        r0, r1 = pres & (d0 < d1), pres & (d1 < d0)
        # closed mouth: both edges of the slit are at the same height; the top is the side
        # whose skin is highest
        sg = 1 if T.V[r0, 1].mean() > T.V[r1, 1].mean() else -1
        c[r0], c[r1] = sg, -sg
    return c


def _composantes(D, faces):
    """Connected pieces (by vertices welded by position, without crossing a material change)
    among raw faces."""
    pos = D.pos
    vb = np.unique(D.faces[faces])
    _, inv = G.souder(pos[vb])
    s = np.full(len(pos), -1)
    s[vb] = inv
    Fw = s[D.faces[faces]]
    par = np.arange(inv.max() + 1)

    def racine(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for a, b, c in Fw:
        ra, rb, rc = racine(a), racine(b), racine(c)
        par[rb] = ra
        par[racine(rc)] = ra
    r = np.array([racine(int(f[0])) for f in Fw])
    return r * (int(D.mat_face.max()) + 1) + D.mat_face[faces]


def ranger_pieces(D):
    """Raw faces of D's parts -> face part (yeux, sourcils, bouche, base) per connected piece,
    from the position (front view, in the host frame) of its centre: distance to each part's
    zone, the one used by the masks (hull of the eye or mouth opening, inner-top-outer eyebrow
    line), in interocular distances.

    The distance to the eye centre (former rule) sorted into the eyebrows the pieces placed at
    the top of the eye, far from its centre: eyelid folds of Yixuan, Alice, Aria, Zhao, Caesar,
    NangongYu, lash tips at the outer corner of Miyabi, Grace, Manato (0.04 to 0.11 from the
    eye opening, 0.08 to 0.30 from the eyebrow line; real eyebrows are at most 0.05 from their
    line, at least 0.08 from the eye). Composed with someone else's eyes, they stayed with the
    eyebrows: Miyabi's lash tip, carried by her eyes' blink, fell below Maëlys's closed eye
    (a spot under the eye)."""
    faces = np.where(np.isin(D.mat_face, D.i_pieces))[0]
    out = {p: [] for p in ("yeux", "sourcils", "bouche", BASE)}
    if not len(faces):
        return {p: np.zeros(0, dtype=np.int64) for p in out}
    lab = _composantes(D, faces)
    iod = D.iod * D.echelle
    yeux = [G.enveloppe_convexe_2d(D._hote(D.V[D.boucle_oeil[c]])[:, :2]) for c in "gd"]
    sourcils = [D._hote(np.array([D.points_sourcils["sourcil_%s_%s" % (c, k)] for k in ("int", "haut", "ext")]))[:, :2]
                for c in "gd"]
    bouche = G.enveloppe_convexe_2d(D._hote(D.V[D.boucle_bouche])[:, :2])
    mat_part = {n: p for p, ns in PAR_MATERIAU.items() for n in ns}
    for l in np.unique(lab):
        fs = faces[lab == l]
        p = mat_part.get(D.noms_mat[int(D.mat_face[fs[0]])])
        if p is not None:
            out[p].append(fs)
            continue
        c = D._hote(D.pos[np.unique(D.faces[fs])].mean(0))[None, :2]
        de = max(min(G.distance_signee_polygone(c, H)[0] for H in yeux), 0.0) / iod
        ds = min(_dist_polyligne_xy(c, S)[0] for S in sourcils) / iod
        dm = max(G.distance_signee_polygone(c, bouche)[0], 0.0) / iod
        choix = [(de / PIECE_OEIL, "yeux"), (ds / PIECE_SOURCIL, "sourcils"), (dm / PIECE_BOUCHE, "bouche")]
        r, p = min(choix)
        out[p if r < 1 else BASE].append(fs)
    return {p: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for p, v in out.items()}


def raster_uv(uv_tri, n):
    """Texels (row 0 at the bottom, like Blender pixels; PMX V points down) covered by the UV
    triangles: triangle index (-1 outside) and barycentrics."""
    tri = np.full((n, n), -1, dtype=np.int64)
    bary = np.zeros((n, n, 3), dtype=np.float32)
    # in grid coordinates: x = u n, y = (1 - v) n
    T = np.stack([uv_tri[..., 0] * n, (1 - uv_tri[..., 1]) * n], -1)
    for i, t in enumerate(T):
        lo = np.maximum(np.floor(t.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(t.max(0)).astype(int), n)
        if (hi <= lo).any():
            continue
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
        p = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
        a, b, c = t
        M = np.array([b - a, c - a]).T
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        l = np.linalg.solve(M, (p - a).T).T
        ok = (l >= -1e-6).all(1) & (l.sum(1) <= 1 + 1e-6)
        x, y = xs.ravel()[ok], ys.ravel()[ok]
        tri[y, x] = i
        bary[y, x] = np.stack([1 - l[ok].sum(1), l[ok, 0], l[ok, 1]], 1)
    return tri, bary


def _rel_dedans(chemin, dossier):
    """Path relative to dossier if it is inside it (movable library), otherwise unchanged."""
    r = os.path.relpath(chemin, dossier)
    return chemin if r.startswith("..") else r


def lire_fiche(biblio, nom):
    """fiche.json with pmx and texture as absolute paths (stored relative to the character folder)."""
    d = os.path.join(biblio, nom)
    with open(os.path.join(d, "fiche.json"), encoding="utf-8") as fh:
        f = json.load(fh)
    for k in ("pmx", "texture"):
        if f.get(k):
            f[k] = os.path.normpath(os.path.join(d, f[k]))
    return f


# 6 axes and 8 diagonals: normal sorting directions (oriented texture sampling)
_DIRS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                 + [(x, y, z) for x in (1, -1) for y in (1, -1) for z in (1, -1)], dtype=np.float64)
_DIRS /= np.linalg.norm(_DIRS, axis=1)[:, None]


def _normales(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    return n / (np.linalg.norm(n, axis=1)[:, None] + 1e-12)


def _reorienter(P3, nA, XD, DF, fD, bD, dD):
    """Texels whose nearest donor face points the wrong way (inside of a lip or of a folded
    eyelid): they are reprojected onto faces turned the same way. Returns the number of
    texels redone."""
    nD = _normales(XD, DF)
    mauvais = np.where((nA * nD[fD]).sum(1) < 0)[0]
    if not len(mauvais):
        return 0
    lab = np.argmax(nA[mauvais] @ _DIRS.T, 1)
    for j in np.unique(lab):
        sel = mauvais[lab == j]
        bon = np.where(nD @ _DIRS[j] > 0.2)[0]
        if not len(bon):
            continue
        _, f2, b2, d2 = G.plus_proche(P3[sel], XD, DF[bon])
        fD[sel], bD[sel], dD[sel] = bon[f2], b2, d2
    return int(len(mauvais))


def extraire_ou_dire(args):
    try:
        return extraire_un(args)
    except Exception as e:  # noqa: BLE001
        return args[0], "%s: %s" % (type(e).__name__, e)


def extraire_un(args):
    """args = (nom, pmx, sortie[, orientee]); orientee: texture sampling limited to the donor
    faces turned like the pivot's (Outerplane: inside of the lips painted red)."""
    nom, chemin, sortie = args[:3]
    orientee = len(args) > 3 and args[3]
    A = charger_pivot()
    D = TT.Tete(chemin)
    if orientee:  # Outerplane: eye rim seen from the front, not the folded slit of the hole
        D.oeil_visible()
    s = float(np.sqrt(A.W * A.H_crane / (D.W * D.H_crane)))
    D.placer(s, D.centre_visage(), A.centre_visage())
    XD = D._hote(D.V)
    rec = RC.recaler(A, A.V, D, XD)
    X = rec["X"]
    bord = np.zeros(D.n, dtype=bool)
    for b in D.boucles:
        bord[b] = True
    fiable = (rec["dist"] < PV.FIABLE_DIST * A.iod) & ~bord[D.F[rec["face"]]].any(1)
    # skin morphs sampled at the registered points, at the pivot's scale
    noms = [n for n in A.morphes]
    delta = np.zeros((len(noms), A.n, 3), dtype=np.float32)
    present = np.zeros(len(noms), dtype=bool)
    # around the openings, each point projects onto the same side on the donor
    face, bary = rec["face"].copy(), rec["bary"].copy()
    cA = cotes(A, COTE_RAYON * A.iod)
    cD = cotes(D, COTE_RAYON * D.iod)
    cF = np.sign(cD[D.F].astype(int).sum(1))
    for sg in (1, -1):
        qa = np.where(cA == sg)[0]
        fd = np.where(cF != -sg)[0]
        if len(qa) and len(fd):
            _, f2, b2, _ = G.plus_proche(X[qa], XD, D.F[fd])
            face[qa], bary[qa] = fd[f2], b2
    tri = D.F[face]
    for k, n in enumerate(noms):
        if n in D.morphes:
            d = D.decalage_peau(n)
            delta[k] = s * (bary[:, :, None] * d[tri]).sum(1)
            present[k] = np.abs(d).max() > 0
    # donor UVs for each texel of the pivot's skin
    uvA = A.t["uv"]
    fa = A.faces[A.faces_peau_brutes]
    tri_t, bary_t = raster_uv(uvA[fa], TEX)
    ok = tri_t >= 0
    wv = A.b2s[fa[tri_t[ok]]]  # pivot's welded vertices
    P3 = (bary_t[ok][:, :, None] * X[wv]).sum(1)
    _, fD, bD, dD = G.plus_proche(P3, XD, D.F)
    # pivot normals at rest: those of the registered pivot (X) flip where the registration
    # folds the skin (mouth corners, eyelid rims) and scattered red texels there
    repris = _reorienter(P3, _normales(A.V, wv), XD, D.F, fD, bD, dD) if orientee else None
    uvD = D.t["uv"][D.faces[D.F_brut[fD]]]
    uv = np.zeros((TEX, TEX, 2), dtype=np.float32)
    uv[ok] = (bD[:, :, None] * uvD).sum(1)
    m = D.pmx["materiaux"][D.i_peau[0]]
    tex = os.path.normpath(os.path.join(os.path.dirname(chemin), D.pmx["textures"][m["texture"]].replace("\\", "/")))
    pieces = ranger_pieces(D)
    d_ = os.path.join(sortie, nom)
    os.makedirs(d_, exist_ok=True)
    np.savez_compressed(os.path.join(d_, "partie.npz"), X=X.astype(np.float32), fiable=fiable,
                        dist=(rec["dist"] / A.iod).astype(np.float32), noms=np.array(noms), delta=delta,
                        present=present, uv=uv, uv_ok=ok, echelle=s, c_src=D.c_src, c_dst=D.c_dst,
                        **{"pieces_" + p: v for p, v in pieces.items()})
    M = masques(A)
    fiche = {"personnage": nom, "pmx": _rel_dedans(chemin, d_), "texture": _rel_dedans(tex, d_), "echelle": s,
             "distance_moyenne_iod": float(np.mean(rec["dist"]) / A.iod),
             "texels_loin": float(np.mean(dD > 0.03 * A.iod)), "parties": {}}
    if repris is not None:
        fiche["texels_reorientes"] = float(repris / len(P3))
    mats = D.noms_mat
    for p in PARTIES + (BASE,):
        w = masque_partie(M, p)
        fs = pieces.get(p, np.zeros(0, dtype=np.int64))
        fiche["parties"][p] = {
            "fiabilite": float((w * fiable).sum() / w.sum()),
            "pieces": sorted(set(mats[i] for i in D.mat_face[fs])),
            "faces_pieces": int(len(fs)),
        }
    fiche["morphes"] = [n for n, ok_ in zip(noms, present) if ok_]
    fiche["morphes_absents"] = [n for n, ok_ in zip(noms, present) if not ok_]
    _ecrire_json(os.path.join(d_, "fiche.json"), fiche)
    return nom, fiche


def _ecrire_json(chemin, obj):
    tmp = chemin + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, chemin)


# face parts of an Outerplane head: the piece required for it to be provided
REQUIS_OP = {"yeux": ("目", "eye plate missing"), "sourcils": ("眉", "eyebrows missing from the geometry")}
# carried blink: fraction of the pivot's eye hole that stays open when the blink sampled on the
# donor is applied to it; ZZZ: 0 to 0.27 (Jane), median 0.03. Beyond that, the donor's eye closes
# by something other than its skin rim (Outerplane: an eyelid sliding in front of a folded
# socket) and its eyes, placed on another head, stay open when blinking.
CLIGNE_MAX = 0.3


def clignement_porte(A, X, d):
    """{side: area of the registered pivot's (X) eye hole closed (X + d) / open area}."""
    aire = lambda V, b: abs(G.aire_polygone_xy(V[b][:, :2]))  # noqa: E731
    return {c: round(float(aire(X + d, b) / max(aire(X, b), 1e-12)), 3) for c, b in A.boucle_oeil.items()}


def extraire_outerplane_un(args):
    """Conversion (outerplane.py) then extraction; (name, record or None, reason)."""
    import outerplane as OP
    nom, dossier, sortie = args
    src = os.path.join(sortie, nom, nom, "tete.pmx")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    try:
        T = OP.convertir(dossier, src)
        _, fiche = extraire_un((nom, src, sortie, True))
    except Exception as e:  # noqa: BLE001  (a rejected head does not block the others)
        # nothing of this head stays in the library (trash, no deletion)
        # gio refuses the trash on /tmp: the head is then renamed _refusee-<nom> in place
        # (the sandbox refuses to move it to another folder: EXDEV)
        d = os.path.join(sortie, nom)
        import shutil
        if os.path.exists(d) and (not shutil.which("gio") or subprocess.run(["gio", "trash", d], capture_output=True).returncode):
            os.replace(d, os.path.join(sortie, "_refusee-" + nom))
        return nom, None, str(e)
    fiche["jeu"] = "Outerplane"
    fiche["dossier_jeu"] = os.path.basename(dossier)
    fiche["conversion"] = {"bouche": T.bouche, "materiaux": T.materiaux}
    fiche["non_fournies"] = {}
    for p, (piece, raison) in REQUIS_OP.items():
        if piece not in fiche["parties"][p]["pieces"]:
            fiche["non_fournies"][p] = raison
    z = np.load(os.path.join(sortie, nom, "partie.npz"))
    noms = list(z["noms"])
    if TT.MORPHE_CLIGNE in noms and z["present"][noms.index(TT.MORPHE_CLIGNE)]:
        cl = clignement_porte(charger_pivot(), z["X"].astype(float), z["delta"][noms.index(TT.MORPHE_CLIGNE)].astype(float))
        fiche["clignement_porte"] = cl
        if max(cl.values()) > CLIGNE_MAX and "yeux" not in fiche["non_fournies"]:
            fiche["non_fournies"]["yeux"] = ("blink not carried: on the pivot, the eye stays %d %% open "
                                             "(ZZZ: 27 %% at most)" % round(100 * max(cl.values())))
    _ecrire_json(os.path.join(sortie, nom, "fiche.json"), fiche)
    return nom, fiche, None


def ecrire_index(sortie, refus=None):
    """index.json: all characters extracted in sortie, plus those of the existing index that
    have no folder there; refus: name -> {jeu, dossier, raison} (heads not provided)."""
    fi = os.path.join(sortie, "index.json")
    ancien = {}
    if os.path.exists(fi):
        with open(fi, encoding="utf-8") as fh:
            ancien = json.load(fh)
    ici = lambda n: os.path.exists(os.path.join(sortie, n, "fiche.json"))  # noqa: E731
    idx = {"pivot": charger_pivot().nom, "parties": {p: {} for p in PARTIES + (BASE,)}}
    for p, d in ancien.get("parties", {}).items():
        for n, v in d.items():
            if not ici(n):
                idx["parties"].setdefault(p, {})[n] = v
    non = {n: v for n, v in ancien.get("non_fournies", {}).items() if not ici(n) and n not in (refus or {})}
    for n in sorted(os.listdir(sortie)):
        if not ici(n):
            continue
        with open(os.path.join(sortie, n, "fiche.json"), encoding="utf-8") as fh:
            f = json.load(fh)
        nf = f.get("non_fournies", {})
        for p, v in f["parties"].items():
            if p in nf:
                continue
            e = {"fiabilite": round(v["fiabilite"], 3), "pieces": v["pieces"]}
            if f.get("jeu"):
                e["jeu"] = f["jeu"]
            idx["parties"][p][n] = e
        if nf:
            non[n] = {"jeu": f.get("jeu", PV.JEU.valeur("abrege", PV.JEU.titre)), "dossier": f.get("dossier_jeu"), "parties": nf}
    for n, r in (refus or {}).items():
        non[n] = dict(r, parties={p: r["raison"] for p in PARTIES + (BASE,)})
    if non:
        idx["non_fournies"] = dict(sorted(non.items()))
    for k in ("coiffure",):
        if k in ancien:
            idx[k] = ancien[k]
    _ecrire_json(fi, idx)
    journal("index: %s" % fi)
    return idx


def images(sortie, noms, travail="texture"):
    cmd = ["bash", RT.FILE_BLENDER, "-b", "--factory-startup", "--python", os.path.join(ICI, "library_images.py"),
           "--", travail, sortie] + list(noms)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or "images : fini" not in r.stdout:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("images: Blender failed")


def _carte_fork(fn, jobs, travaux):
    """map via os.fork, without semaphores: each child handles one job out of every `travaux` and
    writes its results (pickle) to a temporary file; job order is kept."""
    import pickle
    import tempfile
    rep = tempfile.mkdtemp(prefix="carte-")
    pids = []
    for k in range(travaux):
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                out = [(i, fn(jobs[i])) for i in range(k, len(jobs), travaux)]
                with open(os.path.join(rep, "%d.pkl" % k), "wb") as fh:
                    pickle.dump(out, fh)
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                code = 1
            os._exit(code)
        pids.append(pid)
    res = {}
    for k, pid in enumerate(pids):
        os.waitpid(pid, 0)
        f = os.path.join(rep, "%d.pkl" % k)
        if not os.path.exists(f):
            raise RuntimeError("carte: worker %d returned nothing" % k)
        with open(f, "rb") as fh:
            res.update(dict(pickle.load(fh)))
        os.replace(f, f + ".lu")
    return [res[i] for i in range(len(jobs))]


def carte(fn, jobs, travaux):
    """Parallel map; serial if travaux <= 1. A memory-capped launcher (compute.launcher) may refuse
    ProcessPoolExecutor's semaphores: os.fork is used instead."""
    jobs = list(jobs)
    if travaux <= 1:
        for j in jobs:
            yield fn(j)
        return
    try:
        ex = ProcessPoolExecutor(travaux)
    except PermissionError:
        yield from _carte_fork(fn, jobs, travaux)
        return
    with ex:
        yield from ex.map(fn, jobs)


def extraire_outerplane(sortie, travaux, seuls=None):
    import outerplane as OP
    os.makedirs(sortie, exist_ok=True)
    cands, ecartes = OP.candidats(PV.PERSONNAGES)
    jobs = [(n, d, sortie) for n, d in cands if not seuls or n in seuls]
    refus, ok = {}, []
    for (nom, f, raison), (_, d, _) in zip(carte(extraire_outerplane_un, jobs, travaux), jobs):
        if f is None:
            refus[nom] = {"jeu": "Outerplane", "dossier": os.path.basename(d), "raison": raison}
            journal("  %-16s NOT PROVIDED: %s" % (nom, raison))
            continue
        ok.append(nom)
        journal("  %-16s distance %.4f, reliability eyes %.2f nose %.2f mouth %.2f, eye parts %s%s" % (
            nom, f["distance_moyenne_iod"], f["parties"]["yeux"]["fiabilite"], f["parties"]["nez"]["fiabilite"],
            f["parties"]["bouche"]["fiabilite"], ",".join(f["parties"]["yeux"]["pieces"]),
            ("; not provided: %s" % f["non_fournies"]) if f["non_fournies"] else ""))
    if not seuls:
        for d, r in ecartes.items():
            refus[OP.nom_lisible(d) if OP.nom_lisible(d) != d else d] = {"jeu": "Outerplane", "dossier": d,
                                                                         "raison": "excluded: " + r}
    journal("Outerplane: %d heads extracted, %d not provided" % (len(ok), len(refus)))
    if ok:
        journal("textures copied into the pivot's UVs (Blender)")
        images(sortie, ok)
    ecrire_index(sortie, refus)


def extraire(sortie, travaux, seuls=None):
    os.makedirs(sortie, exist_ok=True)
    for n in sorted(set(seuls or ()) - set(PV.PERSONNAGES)):
        journal("  %-12s unknown among the active game's models" % n)
    noms = [n for n in PV.PERSONNAGES if not seuls or n in seuls]
    jobs = [(n, os.path.join(PV.SOURCES, PV.PERSONNAGES[n]), sortie) for n in noms]
    fiches, refus = {}, {}
    with ProcessPoolExecutor(travaux) as ex:
        for nom, f in ex.map(extraire_ou_dire, jobs):
            if isinstance(f, str):      # a head that cannot be sampled does not stop the others
                refus[nom] = {"jeu": PV.jeux.nom_actif(), "dossier": os.path.dirname(PV.PERSONNAGES[nom]), "raison": f}
                journal("  %-12s excluded: %s" % (nom, f))
                continue
            fiches[nom] = f
            journal("  %-12s distance %.4f, reliability eyes %.2f nose %.2f mouth %.2f, eye parts %s" % (
                nom, f["distance_moyenne_iod"], f["parties"]["yeux"]["fiabilite"], f["parties"]["nez"]["fiabilite"],
                f["parties"]["bouche"]["fiabilite"], ",".join(f["parties"]["yeux"]["pieces"])))
    journal("textures copied into the pivot's UVs (Blender)")
    if fiches:
        images(sortie, [n for n in noms if n in fiches])
    ecrire_index(sortie, refus)


# =================================================================== composition

def _charger(biblio, nom):
    z = np.load(os.path.join(biblio, nom, "partie.npz"))
    return dict(z), lire_fiche(biblio, nom)


def forme_composee(A, M, npz, recette):
    """Shape of the composed skin (pivot's frame and weld order), offset of each part
    (zone -> translation) and face weight. npz: donor -> loaded partie.npz."""
    Xb = npz[recette[BASE]]["X"].astype(float)
    # shape: each part (each side for eyes and eyebrows) moved to its place in the base
    X = Xb.copy()
    depl = {}
    for zone, m in M.items():
        if zone == BASE:
            continue
        p = zone.split("_")[0]
        Xd = npz[recette[p]]["X"].astype(float)
        T = (m[:, None] * (Xb - Xd)).sum(0) / m.sum()
        depl[zone] = T
        X += m[:, None] * (Xd + T - Xb)
    # the composed face replaces the pivot head only within the face: skull, ears and neck
    # stay the pivot head's (a retarget_tete visage pass only takes the face)
    wv = A.poids_visage(RT.MARGE * A.iod, RT.BANDE * A.iod)
    return A.V + wv[:, None] * (X - A.V), depl, wv


def composer(biblio, recette, sortie):
    """recette: part -> character (base required). Writes <sortie>/visage.pmx."""
    os.makedirs(sortie, exist_ok=True)
    A = charger_pivot()
    M = masques(A)
    donneurs = {n: _charger(biblio, n) for n in set(recette.values())}
    # a part the library does not provide (record: non_fournies) is not accepted
    for p, n in recette.items():
        nf = donneurs[n][1].get("non_fournies", {})
        if p in nf:
            raise ValueError("%s does not provide %s: %s (take this part from another character)" % (n, p, nf[p]))
    Xf, depl, wv = forme_composee(A, M, {n: z for n, (z, _) in donneurs.items()}, recette)
    # skin morphs: from each part's donor, otherwise from the base, otherwise the pivot's;
    # all face morphs of the donors, even those the pivot lacks (morphs.py)
    dons = MO.Donneurs(biblio, A)
    noms = list(dict.fromkeys(list(A.morphes) + MO.noms_recette(dons, recette, M, wv)))
    delta = {n: MO.delta_compose(dons, recette, M, wv, n) for n in noms}
    chemin = _ecrire_compose(A, Xf, delta, recette, dons, depl, sortie)
    # texture: part weights in the pivot's UVs
    uvA = A.t["uv"]
    fa = A.faces[A.faces_peau_brutes]
    tri_t, bary_t = raster_uv(uvA[fa], TEX)
    ok = tri_t >= 0
    wvs = A.b2s[fa[tri_t[ok]]]
    poids = {}
    for p in PARTIES + (BASE,):
        g = np.zeros((TEX, TEX), dtype=np.float32)
        g[ok] = (bary_t[ok] * masque_partie(M, p)[wvs]).sum(1)
        poids[p] = g
    g = np.zeros((TEX, TEX), dtype=np.float32)
    g[ok] = (bary_t[ok] * wv[wvs]).sum(1)
    m = A.pmx["materiaux"][A.i_peau[0]]
    np.savez_compressed(os.path.join(sortie, "poids_texture.npz"), ok=ok, visage=g,
                        **{"poids_" + p: v for p, v in poids.items()})
    _ecrire_json(os.path.join(sortie, "recette.json"), {
        "recette": recette, "biblio": biblio,
        "texture_pivot": os.path.normpath(os.path.join(os.path.dirname(A.chemin), A.pmx["textures"][m["texture"]].replace("\\", "/")))})
    images(sortie, [], travail="compose")
    return chemin


def _ecrire_compose(A, Xf, delta, recette, dons, depl, sortie):
    P = copy.deepcopy(A.pmx)
    nuv = P["n_uv_add"]
    P["textures"] = [RT._texture_rel(x, os.path.dirname(A.chemin), sortie) for x in P["textures"]]
    S = P["sommets"]
    vb = A.peau_bruts
    k = A.b2s[vb]
    nor = G.normales_sommets(Xf, A.F)
    for i, kk in zip(vb, k):
        S["pos"][i] = tuple(Xf[kk])
        S["nor"][i] = tuple(nor[kk])
    # morphs: offsets per raw vertex, written after the parts (MO.ecrire_morphe)
    peau = set(int(i) for i in vb)
    acc = {n: {} for n in delta}
    # parts: the pivot's removed (materials kept, empty), the donors' added right after the
    # skin
    pieces_A = set(A.i_pieces)
    nouv = []  # (material, faces)
    mixte = len({dons[n][1].get("jeu", PV.JEU.valeur("abrege", PV.JEU.titre)) for n in set(recette.values())}) > 1
    for p in ("yeux", "sourcils", "bouche", BASE):
        nom_d = recette[p]
        z, fiche, D, _ = dons[nom_d]
        fs = dons.pieces(nom_d, p)
        if not len(fs):
            continue
        Pd = D.pmx
        sd = Pd["sommets"]
        carte = RT._carte_os(D, A)
        vs = np.unique(D.faces[fs])
        pos = D._hote(D.pos[vs])
        if p in ("yeux", "sourcils"):
            # each side follows the offset of its eye or eyebrow
            for c, sg in (("g", 1), ("d", -1)):
                cote = (pos[:, 0] * sg) > 0
                pos[cote] += depl["%s_%s" % (p, c)]
        elif p == "bouche":
            pos += depl["bouche"]
        base_v = len(S["pos"])
        idx = {int(v): base_v + j for j, v in enumerate(vs)}
        for j, v in enumerate(vs):
            v = int(v)
            t = sd["type"][v]
            o = [int(carte[x]) if x >= 0 else -1 for x in sd["os"][v]]
            if t == 3:
                t = 1
            if t == 4:
                t = 2
            if t == 0:
                o = [o[0], -1, -1, -1]
            u = list(sd["uv_add"][v])[:nuv]
            S["pos"].append(tuple(pos[j]))
            S["nor"].append(sd["nor"][v])
            S["uv"].append(sd["uv"][v])
            S["uv_add"].append(u + [(0.0, 0.0, 0.0, 0.0)] * (nuv - len(u)))
            S["type"].append(t)
            S["os"].append(tuple(o))
            S["poids"].append(tuple(sd["poids"][v]))
            S["sdef"].append(None)
            S["contour"].append(sd["contour"][v])
        # part morphs (lash blink...): the donor's at the pivot's scale, plus the difference
        # between the composed skin and the donor's skin under the part (the whole difference
        # if the donor lacks the morph: the part follows the skin); the iris does not follow the skin
        suit = ~np.isin(D.mat_face[fs], [i for i, nm in enumerate(D.noms_mat) if nm in MO.FIXES])
        v_suit = np.isin(vs, np.unique(D.faces[fs[suit]]))
        # inside of the mouth and eyes: stays behind the skin (MO.retenir)
        v_int = np.isin(vs, np.unique(D.faces[fs[np.isin(D.mat_face[fs], [i for i, nm in enumerate(D.noms_mat) if nm in MO.INTERIEUR])]]))
        dv = pos - D.pos[vs]  # translation of the part (depl)
        (Fd_, b_d), (Fp_, b_p) = MO.peau_commune(
            D.pos, D.faces[D.faces_peau_brutes], D.pos[vs], Xf, A.F,
            lambda x, vec: x if vec else x + dv)
        # recipe mixing games: the difference is taken at the same pivot point, the one nearest
        # the placed part (composed skin minus this donor's skin carried onto the pivot). The
        # donor point carried then projected landed, for OP, away from the skin touching the
        # part: lashes frozen during the blink on a ZZZ base.
        # OP donor: its 颜2 makeup plates have no morph (motionless on a skin that moves up to
        # 0.15); a part vertex that the donor's morph leaves in place while the donor's skin
        # moves under it follows the composed skin entirely.
        op = fiche.get("jeu") == "Outerplane"
        meme_point = mixte or op
        if meme_point:
            _, f_m, b_m, _ = G.plus_proche(pos, Xf, A.F)
            F_m = A.F[f_m]
        for n in delta:
            dd = dons.dep(nom_d, n)
            d = np.zeros((len(vs), 3)) if dd is None else D.echelle * dd[vs]
            if meme_point:
                sous = (b_m[:, :, None] * delta[n][F_m]).sum(1)
                sk_d = (b_m[:, :, None] * dons.peau(nom_d, n)[F_m]).sum(1) if dd is not None else 0
                e = sous - sk_d
                if op and dd is not None:
                    a_d = np.linalg.norm(sk_d, axis=1)
                    fige = v_suit & (a_d > MO.SEUIL) & (np.linalg.norm(d, axis=1) < MO.FIGEE * a_d)
                    d[fige], e[fige] = 0.0, sous[fige]
            else:
                # composed skin under the part, minus the donor's skin under the part on the donor
                e = (b_p[:, :, None] * delta[n][Fp_]).sum(1)
                if dd is not None:
                    e -= D.echelle * (b_d[:, :, None] * dd[Fd_]).sum(1)
            d[v_suit] += e[v_suit]
            # even when motionless, an inner part stays behind a moving skin
            if v_int.any():
                d[v_int] = MO.retenir(Xf, Xf + delta[n], A.F, pos[v_int], d[v_int])[0]
            for j in np.where(np.abs(d).max(1) > 1e-7)[0]:
                acc[n][idx[int(vs[j])]] = d[j]
        off = len(P["textures"])
        P["textures"] += [RT._texture_rel(x, os.path.dirname(D.chemin), sortie) for x in Pd["textures"]]
        for i in np.unique(D.mat_face[fs]):
            mm = copy.deepcopy(Pd["materiaux"][int(i)])
            for kk in ("texture", "sphere"):
                if mm[kk] >= 0:
                    mm[kk] += off
            if mm["toon_partage"] == 0 and mm["toon"] >= 0:
                mm["toon"] += off
            # name kept (the head tools recognise parts by their name); the donor is noted in
            # the English name
            mm["nom_en"] = "%s (%s)" % (mm["nom_en"] or mm["nom"], nom_d)
            nouv.append((mm, [tuple(idx[int(v)] for v in D.faces[f]) for f in fs[D.mat_face[fs] == i]]))
    dA_os = {}
    for n, d in delta.items():
        m = next((x for x in P["morphes"] if x["nom"] == n), None)
        dec = {}
        if m is not None and m["type"] == 1:
            dec = {int(v): np.asarray(x) for v, x in m["decalages"] if int(v) not in peau}
        elif m is not None:
            # bone (or group) morph of the pivot: becomes a vertex morph, its effect outside the
            # skin kept
            if not dA_os:
                dA_os["Mo"] = MO.Modele(pmx=A.pmx)
            dd = dA_os["Mo"].deplacement(n)
            dec = {int(v): dd[v] for v in np.where(np.abs(dd).max(1) > 1e-7)[0] if int(v) not in peau}
        for i, kk in zip(vb, k):
            if np.abs(d[kk]).max() > 1e-7:
                dec[int(i)] = d[kk]
        dec.update(acc[n])
        if m is None and not any(np.linalg.norm(x) >= MO.SEUIL for x in dec.values()):
            continue
        panneau, nom_en = (m["panneau"], m["nom_en"]) if m is not None else MO.panneau_de(dons, recette, n)
        MO.ecrire_morphe(P, n, dec, panneau, nom_en)
    faces_A = np.asarray(P["faces"])
    mf = np.repeat(np.arange(len(P["materiaux"])), [m["n_indices"] // 3 for m in P["materiaux"]])
    mats, faces, nouvel_indice = [], [], {}
    i_peau = A.i_peau[0]
    for i, m in enumerate(P["materiaux"]):
        m = copy.deepcopy(m)
        fl = [] if i in pieces_A else [tuple(int(x) for x in f) for f in faces_A[mf == i]]
        if i in pieces_A:
            m["nom"] += "_pivot"  # emptied: its parts are replaced by the donors'
        if i == i_peau:
            m["texture"] = len(P["textures"])
        m["n_indices"] = 3 * len(fl)
        nouvel_indice[i] = len(mats)
        mats.append(m)
        faces += fl
        if i == i_peau:
            for mm, fl2 in nouv:
                mm["n_indices"] = 3 * len(fl2)
                mats.append(mm)
                faces += fl2
    P["textures"].append("visage_compose.png")
    for m in P["morphes"]:
        if m["type"] == 8:
            for d in m["decalages"]:
                if d["materiau"] >= 0:
                    d["materiau"] = nouvel_indice[d["materiau"]]
    P["materiaux"] = mats
    P["faces"] = faces
    P["nom"] = "visage-compose"
    P["commentaire"] = (P.get("commentaire") or "") + "\r\n[bibliotheque: %s]" % ", ".join(
        "%s from %s" % (p, n) for p, n in recette.items())
    chemin = os.path.join(sortie, "visage.pmx")
    pmx_ecrire.ecrire(P, chemin)
    return chemin


# =================================================================== principal

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("extract", "compose"))
    ap.add_argument("--output", dest="sortie", required=True)
    ap.add_argument("--library", dest="biblio")
    ap.add_argument("--jobs", dest="travaux", type=int, default=4)
    ap.add_argument("--only", dest="seuls")
    ap.add_argument("--outerplane", action="store_true", help="extraire: the Outerplane heads (outerplane.py)")
    ap.add_argument("--base")
    for p in PARTIES:
        ap.add_argument("--" + p)
    ap.add_argument("--onto", dest="sur", help="character on which to place the composed face")
    ap.add_argument("--hair", dest="cheveux", help="with --onto: character that gives its hairstyle (hairstyle.py)")
    ap.add_argument("--ears", dest="oreilles", help="with --hair: character that gives its animal ears, or 'aucune' (none)")
    a = ap.parse_args(argv)
    if a.action == "extract":
        (extraire_outerplane if a.outerplane else extraire)(a.sortie, a.travaux, a.seuls.split(",") if a.seuls else None)
        return
    recette = {BASE: a.base}
    for p in PARTIES:
        recette[p] = getattr(a, p) or a.base
    chemin = composer(a.biblio, recette, a.sortie)
    journal("composed face: %s" % chemin)
    if a.sur:
        cible = os.path.join(PV.SOURCES, PV.PERSONNAGES[a.sur])
        r = subprocess.run([sys.executable, "-I", os.path.join(ICI, "retarget_head.py"), "visage", "--source", chemin,
                            "--target", cible, "--output", os.path.join(a.sortie, "sur-" + a.sur)])
        res = os.path.join(a.sortie, "sur-" + a.sur, "resultat.pmx")
        if r.returncode and not os.path.exists(res):
            raise RuntimeError("retarget_tete visage failed")
        # a failed measurement (mesures.json) leaves a written result: it is finished anyway
        MO.retenir_pose(res, journal)
        if a.cheveux:
            cmd = [sys.executable, "-I", os.path.join(ICI, "hairstyle.py"), "poser", "--library", a.biblio,
                   "--hair", a.cheveux, "--onto", a.sur, "--target", os.path.join(a.sortie, "sur-" + a.sur, "resultat.pmx"),
                   "--output", os.path.join(a.sortie, "sur-%s-cheveux-%s" % (a.sur, a.cheveux))]
            if a.oreilles:
                cmd += ["--ears", a.oreilles]
            if subprocess.run(cmd).returncode:
                raise RuntimeError("coiffure poser failed")
        if r.returncode:
            raise RuntimeError("retarget_tete visage: failed measurements (see %s); result written and finished"
                               % os.path.join(a.sortie, "sur-" + a.sur, "mesures.json"))


if __name__ == "__main__":
    main()

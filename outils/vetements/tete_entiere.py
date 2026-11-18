"""Whole studio head (face, ears and neck in a single piece) stitched onto the common body.

    blender -b --factory-startup --python outils/vetements/tete_entiere.py -- \
        --corps BODY.pmx --tete HEAD.pmx --sortie FOLDER
    (without Blender, python3 -I does everything except the tint, which reads and writes textures)

On a studio head (Yuzuha), the face 颜 and the skin 肌 of the ears and neck are welded into a single
surface, which ends at the bottom of the neck on a horizontal cut, and at the back on an open edge that
the hair covers. The rest of 肌 (body, V-shaped collar piece) is not taken. This piece is kept as is
(the underside of the jaw stays the studio's) and the body is stitched to it:

 - graft: `greffe_tete --garder-crane` places the piece by the 頭 bone, shares the bones, carries the morphs;
 - the body goes away under the piece: any face with a vertex seen through the piece from its centre, and
   above the neck cut; two edges remain, below the neck cut and around the scalp;
 - the body adjusts to the piece, which is left untouched: at the bottom, radius around the neck axis,
   faded out over `RAYON_FORME_MM`; at the top, the body edge is laid on the scalp edge by arc length,
   the whole scalp follows through a harmonic field (zero at the neck), and the rows behind that edge are
   smoothed (Taubin) over `RAYON_LISSAGE_MM`;
 - a strip of faces (body material) links each body edge to the piece edge: its vertices on the piece
   side are copies of the piece's (position, normal, weights, morphs), so nothing opens when the model
   moves;
 - neck weights: the body edge takes those of the ring; below it, those of the head's original neck
   (projected onto its triangles) up to `RAYON_POIDS_COU_MM[0]`, the body's beyond the second radius,
   a harmonic field in between; scalp weights and normals are blended towards the piece edge over
   `RAYON_POIDS_MM`;
 - closed head (whole scalp, no open edge at the back: VRoid): no seam at the top, the body loses
   everything it has above the neck cut;
 - small holes in the piece (edges of 6 vertices at most, behind the ears) are filled;
 - tint: the body skin is repainted, each texel taking the piece's colour at the nearest edge,
   blended over `RAYON_TEINTE_MM` towards that of the bottom of the neck (--teinte-base visage: towards
   the face's, for a family that paints a shadow on the neck, VRoid); the UVs of the piece edge are pulled
   in by `RETRAIT_TEXELS` (dark band at the edge of the studio's UV islands).

Output: <sortie>/resultat.pmx (textures in tex/), raccord.json. Animation test:
outils/vetements/epreuve_tete.py.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
sys.path.insert(0, os.path.join(ICI, "..", "tete"))
sys.path.insert(0, ICI)
import pmx as PMX  # noqa: E402
import pmx_ecrire  # noqa: E402
import greffe_tete  # noqa: E402

MM = 1.0 / 80.0
PIECE = "肌_tete"
VISAGE = ("颜", "口舌", "齿", "眉", "睫", "白目", "目", "目影", "目光", "二重", "口线", "颜_A", "睫眉", "颜2")
RAYON_FORME_MM = 40.0
RAYON_POIDS_MM = 15.0
RAYON_POIDS_COU_MM = (10.0, 25.0)   # head neck weights up to the 1st, body weights after the 2nd, harmonic in between
RAYON_TEINTE_MM = 30.0
RAYON_LISSAGE_MM = 20.0
RETRAIT_TEXELS = 7.0      # 4-texel dark band at the edge of the piece's UV islands, plus filtering
TROU_MAX = 6
CRANE_MIN_MM = 60.0           # a scalp edge spans at least this much; smaller: it is not one
FENTE_MAX = 0.01              # area / perimeter² below which an edge is a slit (circle: 0.08)


def args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corps", required=True)
    p.add_argument("--tete", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--peau", default="肌")
    p.add_argument("--ecart-cou-max", default="10", help="tolerated offset between the 首 bones of the body and of the head "
                   "placed by 頭; the body with our own shape has a longer neck than Yuzuha's head: 20")
    p.add_argument("--teinte-base", default="anneau", choices=("anneau", "visage"),
                   help="body skin tint away from the join: that of the neck ring, or that of the face "
                   "when the head's neck carries a painted shadow")
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return p.parse_args(argv)


# ---------------------------------------------------------------------- surfaces

def soudure(pos, tol=1e-4):
    _, inv = np.unique(np.round(pos / tol).astype(np.int64), axis=0, return_inverse=True)
    return inv.ravel()


def composantes(F, sou):
    """Component (through welded vertices) of each face."""
    n = int(sou.max()) + 1
    par = np.arange(n)

    def racine(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b, c in sou[F]:
        ra, rb, rc = racine(a), racine(b), racine(c)
        par[rb] = ra
        par[racine(rc)] = ra
    return np.array([racine(x) for x in sou[F[:, 0]]])


def boucles(F, sou):
    """Ordered edges (raw indices, one per welded position), in face winding order."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    S = np.sort(sou[E], axis=1)
    _, inv, cnt = np.unique(S, axis=0, return_inverse=True, return_counts=True)
    B = E[cnt[inv.ravel()] == 1]
    suivant = {int(sou[a]): (int(sou[b]), int(a), int(b)) for a, b in B}
    brut = {}
    for a, b in B:
        brut.setdefault(int(sou[a]), int(a))
    vus, out = set(), []
    for s in suivant:
        if s in vus:
            continue
        L, x = [], s
        while x not in vus and x in suivant:
            vus.add(x)
            L.append(brut[x])
            x = suivant[x][0]
        if x == s and len(L) >= 3:
            out.append(np.array(L))
    return out


def traverse(o, d, T):
    """For each ray (o, d), true if it hits one of the triangles T (Möller-Trumbore)."""
    v0, e1, e2 = T[:, 0], T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    res = np.zeros(len(d), bool)
    for k in range(0, len(d), 256):
        D = d[k:k + 256, None, :]
        p = np.cross(D, e2[None])
        det = (e1[None] * p).sum(2)
        ok = np.abs(det) > 1e-12
        inv = 1.0 / np.where(ok, det, 1.0)
        s = o - v0
        u = (s[None] * p).sum(2) * inv
        q = np.cross(s, e1)
        v = (D * q[None]).sum(2) * inv
        t = (e2 * q).sum(1)[None] * inv
        res[k:k + 256] = (ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)).any(1)
    return res


def lisse(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def point_polyligne(q, L):
    """Closest point of each q on the closed loop L: (point, i, j, u) with point = L[i] + u (L[j] - L[i])."""
    a, b = L, np.roll(L, -1, axis=0)
    ab = b - a
    u = np.clip(((q[:, None] - a) * ab).sum(2) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    pts = a + u[..., None] * ab
    d = ((q[:, None] - pts) ** 2).sum(2)
    k = d.argmin(1)
    r = np.arange(len(q))
    return pts[r, k], k, (k + 1) % len(L), u[r, k], np.sqrt(d[r, k])


def angles(P, centre, axe):
    """Angle of each point around the axis (frame of the perpendicular plane)."""
    axe = axe / np.linalg.norm(axe)
    e1 = np.cross(axe, [0.0, 0.0, 1.0] if abs(axe[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axe, e1)
    Q = P - centre
    return np.arctan2(Q @ e2, Q @ e1)


def fermeture(A, B, PA, PB, centre, axe, params=None):
    """Triangles closing the strip between two closed loops A and B (indices), walked by increasing angle
    around the axis (or by `params`, precomputed angles); each triangle has two vertices from one loop
    and one from the other."""
    ta, tb = params if params is not None else (angles(PA, centre, axe), angles(PB, centre, axe))

    def ordonner(L, t):
        # increasing direction: most angle steps are positive
        dt = np.angle(np.exp(1j * (np.roll(t, -1) - t)))
        if (dt > 0).sum() < len(dt) / 2:
            L, t = L[::-1], t[::-1]
        k = int(np.argmin(np.mod(t, 2 * np.pi)))
        L, t = np.roll(L, -k), np.roll(t, -k)
        t = np.mod(t - t[0], 2 * np.pi)
        # unwrapped, increasing angle (a local step back is flattened)
        t = np.maximum.accumulate(t)
        return L, t
    A, ta = ordonner(np.asarray(A), ta)
    B, tb = ordonner(np.asarray(B), tb)
    # aligned starts: B starts at the point closest in angle to A's start
    ta2 = np.append(ta, 2 * np.pi)
    tb2 = np.append(tb, 2 * np.pi)
    A2, B2 = np.append(A, A[0]), np.append(B, B[0])
    i = j = 0
    tri = []
    while i < len(A) or j < len(B):
        if j >= len(B) or (i < len(A) and ta2[i + 1] <= tb2[j + 1]):
            tri.append((A2[i], A2[i + 1], B2[j]))
            i += 1
        else:
            tri.append((A2[i], B2[j + 1], B2[j]))
            j += 1
    return np.array(tri, np.int64)


def harmonique(n, aretes, fixe, val, iterations=6000):
    """Extend `val` (imposed where `fixe`) over the graph by neighbour averaging."""
    x = val.copy()
    a, b = aretes[:, 0], aretes[:, 1]
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    libre = ~fixe & (deg > 0)
    for _ in range(iterations):
        somme = np.stack([np.bincount(a, x[b, k], n) + np.bincount(b, x[a, k], n) for k in range(x.shape[1])], 1)
        nv = somme[libre] / deg[libre][:, None]
        ecart = np.abs(nv - x[libre]).max() if libre.any() else 0.0
        x[libre] = nv
        if ecart < 1e-7:
            break
    return x


def poser_sur(A, B):
    """Points of the closed loop B at the arc lengths of the vertices of loop A (same start and direction
    as B: start nearest to B[0], best-fitting direction); also returns those arc lengths and those of
    B's vertices, in [0, 1)."""
    def abscisses(X):
        l = np.linalg.norm(np.roll(X, -1, 0) - X, axis=1)
        return np.concatenate([[0], np.cumsum(l)[:-1]]) / l.sum(), l.sum()
    sb, _ = abscisses(B)
    Bf = np.vstack([B, B[:1]])
    sbf = np.append(sb, 1.0)

    def sur_b(u):
        return np.stack([np.interp(u % 1.0, sbf, Bf[:, k]) for k in range(3)], 1)
    k0 = int(np.argmin(np.linalg.norm(A - B[0], axis=1)))
    meilleur = None
    for sens in (1, -1):
        Ar = np.roll(A[::sens], -(k0 if sens == 1 else len(A) - 1 - k0), 0)
        sa, _ = abscisses(Ar)
        Q = sur_b(sa)
        e = np.linalg.norm(Q - Ar, axis=1).mean()
        if meilleur is None or e < meilleur[0]:
            meilleur = (e, sens, Q, sa)
    _, sens, Q, sa = meilleur
    # back in A's order, with each point's arc length on B
    out = np.empty_like(A)
    s_out = np.empty(len(A))
    ordre = np.roll(np.arange(len(A))[::sens], -(k0 if sens == 1 else len(A) - 1 - k0))
    out[ordre] = Q
    s_out[ordre] = sa
    return out, s_out, sb


# ---------------------------------------------------------------------- head piece

def piece_de_tete(chemin, dossier, peau="肌"):
    """Temporary PMX: the skin material keeps only the faces welded to the face, renamed PIECE;
    textures as absolute paths (the PMX no longer sits next to them)."""
    H = PMX.lire(chemin)
    t = PMX.vers_tableaux(H)
    noms = [m["nom"] for m in H["materiaux"]]
    iv, ip = noms.index("颜"), noms.index(peau)
    sel = np.isin(t["mat_face"], [iv, ip])
    F = t["faces"][sel]
    comp = composantes(F, soudure(t["pos"]))
    du_visage = np.isin(comp, np.unique(comp[t["mat_face"][sel] == iv]))
    garde_peau = np.zeros(len(t["faces"]), bool)
    garde_peau[np.where(sel)[0][du_visage & (t["mat_face"][sel] == ip)]] = True
    faces, mats = [], []
    for i, m in enumerate(H["materiaux"]):
        Fi = t["faces"][(t["mat_face"] == i) & (garde_peau if i == ip else True)]
        mm = dict(m)
        if i == ip:
            mm["nom"] = PIECE
        mm["n_indices"] = 3 * len(Fi)
        mats.append(mm)
        faces.append(Fi)
    H["materiaux"] = mats
    H["faces"] = np.vstack(faces).tolist()
    base = os.path.dirname(chemin)
    H["textures"] = [os.path.normpath(os.path.join(base, x.replace("\\", "/"))) for x in H["textures"]]
    out = os.path.join(dossier, "piece_tete.pmx")
    pmx_ecrire.ecrire(H, out + ".tmp")
    os.replace(out + ".tmp", out)
    return out, int(garde_peau.sum())


# ---------------------------------------------------------------------- weights

PEAU_SOURCE = ("颜", "肌", "体", "体2")   # skin and body of the original head (no hair, no skirt)


def plus_proche_triangle(p, T):
    """Barycentric coordinates (k, 3) of the point of each triangle T (k, 3, 3) closest to p, and
    distance."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    n = np.cross(b - a, c - a)
    nn = np.maximum((n * n).sum(1), 1e-30)
    q = p - ((p - a) * n).sum(1)[:, None] * n / nn[:, None]
    def bary(x):
        v0, v1, v2 = b - a, c - a, x - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        return np.stack([1 - w1 - w2, w1, w2], 1)
    bc = bary(q)
    dedans = (bc >= 0).all(1)
    best_bc = bc.copy()
    best = np.where(dedans, np.linalg.norm(q - p, axis=1), np.inf)
    for i0, i1 in ((0, 1), (1, 2), (2, 0)):
        e0, e1 = T[:, i0], T[:, i1]
        e = e1 - e0
        tt = np.clip(((p - e0) * e).sum(1) / np.maximum((e * e).sum(1), 1e-30), 0, 1)
        x = e0 + tt[:, None] * e
        dd = np.linalg.norm(x - p, axis=1)
        m = dd < best
        b_ = np.zeros((len(T), 3))
        b_[:, i0], b_[:, i1] = 1 - tt, tt
        best_bc[m], best[m] = b_[m], dd[m]
    return best_bc, best

def poids_dict(os_, pw):
    d = {}
    for b, w in zip(os_, pw):
        if b >= 0 and w > 0:
            d[int(b)] = d.get(int(b), 0.0) + float(w)
    return d


def poids_ecrire(d):
    it = sorted(d.items(), key=lambda x: -x[1])[:4]
    s = sum(w for _, w in it)
    it = [(b, w / s) for b, w in it]
    while len(it) < 4:
        it.append((-1, 0.0))
    return 2, tuple(b for b, _ in it), tuple(w for _, w in it)


def melange(d1, d2, s):
    out = {b: (1 - s) * w for b, w in d1.items()}
    for b, w in d2.items():
        out[b] = out.get(b, 0.0) + s * w
    return out


# ---------------------------------------------------------------------- seam

def coudre(R, peau, journal, source=None):
    s = R["sommets"]
    t = PMX.vers_tableaux(R)
    P = t["pos"].copy()
    noms = [m["nom"] for m in R["materiaux"]]
    i_peau = noms.index(peau)
    i_piece = [noms.index(n) for n in ("颜", PIECE) if n in noms]
    mf = t["mat_face"]
    F = t["faces"]
    sou = soudure(P)
    Fp = F[np.isin(mf, i_piece)]

    # piece edges: neck cut (lowest), scalp edge (tallest in height), small holes
    L = boucles(Fp, sou)
    L.sort(key=lambda b: P[b, 1].mean())
    anneau = L[0]
    autres = [b for b in L[1:]]
    # closed head (whole scalp, like VRoid's): no edge goes around the scalp; the body then has nothing to
    # stitch at the top, and everything it has above the neck cut goes away
    def fente(b):       # edge with no opening: two poorly welded lips (near-zero area for its perimeter)
        Q = P[b]
        Qc = Q - Q.mean(0)
        aire = np.linalg.norm(np.cross(Qc, np.roll(Qc, -1, axis=0)).sum(0)) / 2
        return aire < FENTE_MAX * np.linalg.norm(Q - np.roll(Q, -1, axis=0), axis=1).sum() ** 2
    larges = [b for b in autres if float(np.ptp(P[b], axis=0).max()) >= CRANE_MIN_MM * MM and not fente(b)]
    crane = max(larges, key=lambda b: np.ptp(P[b, 1]) * len(b)) if larges else None
    ferme = crane is None
    if ferme:
        crane = np.zeros(0, int)
    trous = [b for b in autres if len(b) <= TROU_MAX and b is not crane]
    y_cou = float(P[anneau, 1].mean())
    c_cou = P[anneau].mean(0)
    centre = P[np.unique(Fp)].mean(0)
    journal["anneau"] = {"sommets": len(anneau), "hauteur": round(y_cou, 4),
                         "ecart_hauteur_mm": round(float(np.ptp(P[anneau, 1]) / MM), 2)}
    journal["bord_crane"] = len(crane)
    journal["tete_fermee"] = bool(ferme)

    # holes filled as a fan, in the winding of the neighbouring faces (edge walked backwards)
    bouche = []
    for b in trous:
        for k in range(1, len(b) - 1):
            bouche.append((b[0], b[k + 1], b[k]))
    journal["trous_bouches"] = len(trous)

    # body skin under the piece
    Fc_all = np.where(mf == i_peau)[0]
    Fc = F[Fc_all]
    vc = np.unique(Fc)
    proche = vc[P[vc, 1] > y_cou - 0.5]
    d = P[proche] - centre
    vu = traverse(centre, d / np.linalg.norm(d, axis=1, keepdims=True), P[Fp])
    sous = np.zeros(len(P), bool)
    sous[proche[vu & (P[proche, 1] > y_cou - 0.2 * MM)]] = True
    # at the neck, everything above the cut inside the tube goes away too
    r_cou = np.linalg.norm((P[anneau] - c_cou)[:, [0, 2]], axis=1).max()
    tube = (np.linalg.norm((P - c_cou)[:, [0, 2]], axis=1) < 1.6 * r_cou) & (P[:, 1] > y_cou) & (P[:, 1] < y_cou + 0.6)
    sous |= tube
    enleve = sous[Fc].any(1)
    journal["faces_corps_retirees"] = int(enleve.sum())
    Fc_g = Fc[~enleve]

    # body edges: the one below the ring and the one around the scalp edge
    Lc = boucles(Fc_g, sou)

    def proche_de(b, cible):
        return point_polyligne(P[b], P[cible])[4].mean()
    bas = min(Lc, key=lambda b: proche_de(b, anneau))
    # the scalp edge runs along the piece's from end to end, both ways: a small edge left close by
    # (ear hole of a body that has a face) is not one
    haut = np.zeros(0, int) if ferme else min([b for b in Lc if b is not bas], key=lambda b: max(proche_de(b, crane), proche_de(crane, b)))
    # a body that has a face keeps leftovers of it under the piece (mouth, eye sockets): islands above the
    # cut, attached neither to the body nor to the scalp, removed
    ilot = composantes(Fc_g, sou)
    tient = [ilot[np.isin(Fc_g, b).any(1)][0] for b in (bas, haut) if len(b)]
    reste = ~np.isin(ilot, tient) & (P[Fc_g][:, :, 1].min(1) > y_cou)
    reste = np.isin(ilot, np.setdiff1d(ilot[reste], ilot[~reste]))
    if reste.any():
        journal["faces_corps_ilots"] = int(reste.sum())
        Fc_g = Fc_g[~reste]
    journal["bord_corps_bas"] = {"sommets": len(bas), "ecart_moyen_mm": round(float(proche_de(bas, anneau) / MM), 2)}
    if not ferme:
        journal["bord_corps_haut"] = {"sommets": len(haut), "ecart_moyen_mm": round(float(proche_de(haut, crane) / MM), 2)}

    # shape: the body adjusts to the piece near both edges
    garde_v = np.unique(Fc_g)
    rayon = RAYON_FORME_MM * MM
    deplacement = np.zeros_like(P)
    # bottom: radius around the neck axis, taken from the ring by angle
    ax = np.array([0.0, 1.0, 0.0])
    ta = angles(P[anneau], c_cou, ax)
    ra = np.linalg.norm((P[anneau] - c_cou)[:, [0, 2]], axis=1)
    o = np.argsort(ta)
    ta_s, ra_s = ta[o], ra[o]
    v_bas = garde_v[(P[garde_v, 1] <= y_cou + 1e-6) & (P[garde_v, 1] > y_cou - rayon)
                    & (np.linalg.norm((P[garde_v] - c_cou)[:, [0, 2]], axis=1) < 2.0 * r_cou)]
    tb = angles(P[bas], c_cou, ax)
    rb = np.linalg.norm((P[bas] - c_cou)[:, [0, 2]], axis=1)
    cible_b = np.interp(tb, ta_s, ra_s, period=2 * np.pi)
    db = cible_b - rb                                   # radius offset at the edge, by angle
    o2 = np.argsort(tb)
    tv = angles(P[v_bas], c_cou, ax)
    dv = np.interp(tv, tb[o2], db[o2], period=2 * np.pi)
    hv = np.clip((P[bas, 1].mean() - P[v_bas, 1]) / rayon, 0, 1)
    radial = (P[v_bas] - c_cou) * np.array([1.0, 0.0, 1.0])
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    deplacement[v_bas] += radial * (dv * (1 - lisse(hv)))[:, None]
    journal["ajustement_cou_mm"] = {"max": round(float(np.abs(db).max() / MM), 2),
                                    "moyen": round(float(np.abs(db).mean() / MM), 2)}
    # top: the body edge is laid on the scalp edge by arc length (order kept, no fold when several
    # vertices target the same point), the rest of the scalp follows through a harmonic field on the
    # mesh, zero at the neck; a point-by-point fit crumpled the edge
    E = np.vstack([Fc_g[:, [0, 1]], Fc_g[:, [1, 2]], Fc_g[:, [2, 0]]])
    E = np.unique(np.sort(sou[E], 1), axis=0)
    E = E[E[:, 0] != E[:, 1]]
    s_haut = s_crane = None
    if not ferme:
        cible_h, s_haut, s_crane = poser_sur(P[haut], P[crane])
        dh = np.linalg.norm(cible_h - centre, axis=1) - np.linalg.norm(P[haut] - centre, axis=1)
        fixe = np.zeros(len(P), bool)
        val = np.zeros_like(P)
        fixe[sou[garde_v[P[garde_v, 1] < y_cou + 0.2]]] = True
        fixe[sou[haut]] = True
        val[sou[haut]] = cible_h - P[haut]
        champ = harmonique(len(P), E, fixe, val)
        v_haut = garde_v[(P[garde_v, 1] >= y_cou + 0.2) | np.isin(sou[garde_v], sou[haut])]
        deplacement[v_haut] += champ[sou[v_haut]]
        journal["ajustement_crane_mm"] = {"max": round(float(np.abs(dh).max() / MM), 2),
                                          "moyen": round(float(np.abs(dh).mean() / MM), 2)}
    # (the welded copies of a vertex share the same position, hence the same displacement)
    P += deplacement
    if not ferme:
        # body rows behind the scalp edge: the old stair-stepped edge leaves teeth there, smoothed
        # (Taubin, without shrinking) over RAYON_LISSAGE_MM, the laid edge staying fixed
        Pw = np.zeros_like(P)
        Pw[sou] = P
        Pw[sou[garde_v]] = P[garde_v]                       # a vertex of another material welded there does not count
        d_bord = np.linalg.norm(Pw[:, None] - P[haut][None], axis=2).min(1)
        zone = (d_bord < RAYON_LISSAGE_MM * MM) & ~np.isin(np.arange(len(P)), sou[haut])
        zone &= np.isin(np.arange(len(P)), sou[garde_v[P[garde_v, 1] > y_cou + 0.2]])
        a_, b_ = E[:, 0], E[:, 1]
        deg = np.bincount(a_, minlength=len(P)) + np.bincount(b_, minlength=len(P))
        zone &= deg > 0
        for _ in range(30):
            for f_ in (0.5, -0.53):
                m = np.stack([np.bincount(a_, Pw[b_, k], len(P)) + np.bincount(b_, Pw[a_, k], len(P)) for k in range(3)], 1)
                m = m[zone] / deg[zone][:, None]
                Pw[zone] += f_ * (m - Pw[zone])
        P[garde_v] = Pw[sou[garde_v]]
        journal["lissage_bord_crane"] = int(zone.sum())
        _, _, _, _, ecart_h = point_polyligne(P[haut], P[crane])
        _, _, _, _, ecart_h = point_polyligne(P[haut], P[crane])
        journal["ecart_bord_crane_mm"] = round(float(ecart_h.max() / MM), 4)

    # body weights and normals blended towards the piece edge
    N = t["nor"].copy()
    W = [poids_dict(s["os"][i], s["poids"][i]) for i in range(len(P))]
    rp = RAYON_POIDS_MM * MM
    nouv_type = {}
    # neck: below the ring, the weights of the head's original neck (made for that ring), projected onto its
    # triangles up to RAYON_POIDS_COU_MM[0], the body's beyond RAYON_POIDS_COU_MM[1], and in between
    # the harmonic field (the smoothest transition). A short blend towards the ring let the 頭 weight
    # drop from 0.8 to 0.1 within a few mm at the front, and the skin collapsed there when the head turned.
    r_in, r_out = (r * MM for r in RAYON_POIDS_COU_MM)
    _, _, _, _, d_an = point_polyligne(P[garde_v], P[anneau])
    d_cr = np.full(len(garde_v), np.inf) if ferme else point_polyligne(P[garde_v], P[crane])[4]
    pres = d_an < d_cr
    jumeaux = {}
    for v in garde_v.tolist():
        jumeaux.setdefault(int(sou[v]), []).append(v)
    lab_bas = set(sou[bas].tolist())
    interieur = garde_v[(d_an < r_in) & pres & ~np.isin(sou[garde_v], list(lab_bas))]
    libre = garde_v[(d_an >= r_in) & (d_an < r_out) & pres]
    q, i, j, u, _ = point_polyligne(P[bas], P[anneau])
    for v, a_, b_, uu in zip(bas.tolist(), anneau[i].tolist(), anneau[j].tolist(), u.tolist()):
        wb = melange(W[a_], W[b_], uu)
        for x in jumeaux.get(int(sou[v]), [v]):     # the body edge has duplicated vertices (UV seams)
            W[x] = dict(wb)
            nouv_type[x] = True
    n_src = 0
    if source is not None:
        Hs = PMX.lire(source[0])
        ts = PMX.vers_tableaux(Hs)
        Ps = ts["pos"] + np.asarray(source[1]) * MM
        ns = [m["nom"] for m in Hs["materiaux"]]
        Fs_ = ts["faces"][np.isin(ts["mat_face"], [k for k, n in enumerate(ns) if n in PEAU_SOURCE])]
        g_ = Ps[Fs_].mean(1)
        Fs_ = Fs_[(g_[:, 1] < y_cou + 0.3) & (np.linalg.norm((g_ - c_cou)[:, [0, 2]], axis=1) < 2 * r_cou)]
        nom_os = {o["nom"]: k for k, o in enumerate(R["os"])}
        os_src = [nom_os.get(o["nom"], -1) for o in Hs["os"]]
        Ws = [{os_src[b]: w for b, w in poids_dict(ts["os"][k], ts["poids"][k]).items() if os_src[b] >= 0}
              for k in range(len(Ps))]
        G = Ps[Fs_].mean(1)
        for v in interieur.tolist():
            cand = np.where(np.linalg.norm(G - P[v], axis=1) < 20 * MM)[0]
            if not len(cand):
                continue
            bc, dist = plus_proche_triangle(P[v], Ps[Fs_[cand]])
            k = int(np.argmin(dist))
            if dist[k] > 8 * MM:
                continue
            d = {}
            for som, c in zip(Fs_[cand[k]], bc[k]):
                for b, w in Ws[som].items():
                    d[b] = d.get(b, 0.0) + c * w
            tot = sum(d.values())
            if tot > 0:
                W[v] = {b: w / tot for b, w in d.items()}
                nouv_type[v] = True
                n_src += 1
    zone = np.r_[interieur, libre, bas]
    os_cou = sorted({b for v in zone.tolist() for b in W[v]})
    col = {b: k for k, b in enumerate(os_cou)}
    Wm = np.zeros((len(P), len(os_cou)))
    for v in garde_v.tolist():
        for b, w in W[v].items():
            if b in col:
                Wm[sou[v], col[b]] = w
    fixe_w = np.zeros(len(P), bool)
    fixe_w[sou[garde_v]] = True
    fixe_w[sou[libre]] = False
    Wm = harmonique(len(P), E, fixe_w, Wm)
    for v in libre.tolist():
        d = {os_cou[k]: float(w) for k, w in enumerate(Wm[sou[v]]) if w > 1e-4}
        d.update({b: w for b, w in W[v].items() if b not in col})
        W[v] = d
        nouv_type[v] = True
    zone_c = np.r_[interieur, libre]
    journal["poids_cou"] = {"sommets": int(len(zone_c)), "repris_de_la_tete": n_src,
                            "harmoniques": int(len(libre)), "rayons_mm": list(RAYON_POIDS_COU_MM)}
    for bord_c, bord_p in ((bas, anneau), (haut, crane))[:1 if ferme else 2]:
        vz = garde_v[np.linalg.norm(P[garde_v][:, None] - P[bord_c][None], axis=2).min(1) < rp] if len(garde_v) else []
        q, i, j, u, dist = point_polyligne(P[vz], P[bord_p])
        for v, a, b, uu, dd_ in zip(vz.tolist(), bord_p[i].tolist(), bord_p[j].tolist(), u.tolist(), dist.tolist()):
            sx = 1 - lisse(dd_ / rp)
            if bord_p is crane:
                W[v] = melange(W[v], melange(W[a], W[b], uu), sx)
            nb = (1 - uu) * N[a] + uu * N[b]
            n_ = (1 - sx) * N[v] + sx * nb
            N[v] = n_ / max(np.linalg.norm(n_), 1e-9)
            nouv_type[v] = True

    # strips: copies of the piece's edge vertices, body material
    copies = {}
    uv = t["uv"]

    def copie(v, uv_corps):
        if v in copies:
            return copies[v]
        k = len(s["pos"])
        for key in ("pos", "nor", "type", "os", "poids", "sdef", "contour", "uv_add"):
            s[key].append(s[key][v])
        s["uv"].append(tuple(uv_corps))
        copies[v] = k
        return k
    bandes = []
    for bord_c, bord_p, axe_, cen in ((bas, anneau, np.array([0.0, 1.0, 0.0]), c_cou),
                                       (haut, crane, None, None))[:1 if ferme else 2]:
        if axe_ is None:
            # scalp edge: axis = mean normal of the edge plane
            Q = P[bord_p] - P[bord_p].mean(0)
            axe_ = np.linalg.svd(Q)[2][2]
            cen = P[bord_p].mean(0)
        _, kk, _, _, _ = point_polyligne(P[bord_p], P[bord_c])
        cp = [copie(int(v), uv[bord_c[kk[n]]]) for n, v in enumerate(bord_p)]
        params = None
        if bord_p is crane:
            # the body edge is laid on the scalp's: seam by the arc lengths of that placement (a nearest
            # point picks the wrong segment where the edge zigzags)
            params = (2 * np.pi * s_haut, 2 * np.pi * s_crane)
        T = fermeture(bord_c, np.array(cp), P[bord_c], P[bord_p], cen, axe_, params)
        # winding: that of the neighbouring body faces
        idx = {int(c): n for n, c in enumerate(cp)}
        def pos_(x):
            return P[x] if x < len(P) else P[bord_p[idx[x]]]
        nt = np.array([np.cross(pos_(b) - pos_(a), pos_(c) - pos_(a)) for a, b, c in T])
        ext = np.array([pos_(a) - (centre if bord_p is crane else c_cou) for a, b, c in T])
        if bord_p is anneau:
            ext[:, 1] = 0
        if ((nt * ext).sum(1) < 0).mean() > 0.5:
            T = T[:, [0, 2, 1]]
        bandes.append(T)
    bandes = np.vstack(bandes)
    journal["faces_bandes"] = int(len(bandes))

    # write positions, normals, weights
    for v in range(len(P)):
        s["pos"][v] = tuple(P[v].tolist())
    for v in nouv_type:
        s["nor"][v] = tuple(N[v].tolist())
        ty, b, w = poids_ecrire(W[v])
        s["type"][v], s["os"][v], s["poids"][v], s["sdef"][v] = ty, b, w, None
    for v, k in copies.items():
        s["pos"][k] = s["pos"][v]

    # morphs: the copies follow their original
    for m in R["morphes"]:
        if m["type"] != 1:
            continue
        ajout = [(copies[i], d) for i, d in m["decalages"] if i in copies]
        m["decalages"] = list(m["decalages"]) + ajout

    # faces: body without the part under the piece, plus the strips; piece plus its filled holes
    faces = []
    for i, m in enumerate(R["materiaux"]):
        Fi = F[mf == i]
        if i == i_peau:
            Fi = np.vstack([Fc_g, bandes])
        elif noms[i] == PIECE and bouche:
            Fi = np.vstack([Fi, np.array(bouche)])
        m["n_indices"] = 3 * len(Fi)
        faces.append(Fi)
    R["faces"] = np.vstack(faces).tolist()
    return {"anneau": anneau, "crane": crane, "copies": copies, "bas": bas, "haut": haut}


# ---------------------------------------------------------------------- tint

def teinte(R, dossier, journal, bords, copies=(), rayon=RAYON_TEINTE_MM * MM, base_de="anneau"):
    """Body skin painted: each texel takes the piece's colour at the nearest edge (neck ring, scalp edge),
    blended towards the tint of the bottom of the neck at `rayon`. The UVs of the piece edge are pulled in
    by seven texels: the band at the UV island edge is a darker bleed, which drew a fringe."""
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("tint: must run in Blender; nothing changed")
        journal["teinte"] = None
        return
    s = R["sommets"]
    t = PMX.vers_tableaux(R)
    noms = [m["nom"] for m in R["materiaux"]]
    i_peau = noms.index("肌")
    idx = [noms.index(n) for n in ("颜", PIECE) if n in noms]

    def chemin(i):
        r = R["textures"][R["materiaux"][i]["texture"]]
        return r if os.path.isabs(r) else os.path.join(dossier, r)
    images = {i: greffe_tete.lire_image(chemin(i)) for i in idx}

    def texel(img, uv):
        h, w = img.shape[:2]
        return img[np.clip((uv[:, 1] % 1) * h, 0, h - 1).astype(int), np.clip((uv[:, 0] % 1) * w, 0, w - 1).astype(int), :3]

    # edge UVs pulled seven texels into the piece (along the edge, nothing moves)
    tous = np.concatenate([np.asarray(b) for b in bords])
    est_bord = np.zeros(len(t["pos"]), bool)
    est_bord[tous] = True
    vers = {v: [] for v in tous.tolist()}
    faces_bord = {}
    for i in idx:
        Fi = t["faces"][t["mat_face"] == i]
        faces_bord[i] = Fi[est_bord[Fi].any(1)]
        for f in faces_bord[i]:
            for k in range(3):
                v = int(f[k])
                if v in vers:
                    vers[v] += [(i, t["uv"][o] - t["uv"][v]) for o in (f[(k + 1) % 3], f[(k + 2) % 3]) if not est_bord[o]]
    mat_de = {}
    for i in idx:
        for v in faces_bord[i].ravel().tolist():
            mat_de.setdefault(v, i)
    seam = np.array([v for v in tous.tolist() if v in mat_de])
    uv = t["uv"].copy()
    for v in seam.tolist():
        if not vers[v]:
            continue
        i, _ = vers[v][0]
        h, w = images[i].shape[:2]
        d = np.mean([x for _, x in vers[v]], 0)
        n = np.linalg.norm(d)
        if n > 1e-9:
            uv[v] = uv[v] + d / n * RETRAIT_TEXELS / max(h, w)
            s["uv"][v] = tuple(uv[v])
    # edge colour: the one shown at the piece edge, read at the pulled-in UVs (read closer to the island
    # edge, it picked up the dark bleed and painted it on the body: fringe on the nape)
    col = np.array([texel(images[mat_de[v]], uv[v][None])[0] for v in seam])
    # points along the edge's edges (the ring has only 16 vertices)
    pts, cpts, de_anneau = [], [], []
    dans_anneau = np.zeros(len(t["pos"]), bool)
    dans_anneau[np.asarray(bords[0])] = True
    for i in idx:
        for f in faces_bord[i][est_bord[faces_bord[i]].sum(1) >= 2]:
            for k in range(3):
                a_, b_ = f[k], f[(k + 1) % 3]
                if not (est_bord[a_] and est_bord[b_]):
                    continue
                for u in np.linspace(0.1, 0.9, 5):
                    pts.append((1 - u) * t["pos"][a_] + u * t["pos"][b_])
                    cpts.append(texel(images[i], ((1 - u) * uv[a_] + u * uv[b_])[None])[0])
                    de_anneau.append(dans_anneau[a_] and dans_anneau[b_])
    pos_b = np.vstack([t["pos"][seam], np.array(pts).reshape(-1, 3)])
    col_b = np.vstack([col, np.array(cpts).reshape(-1, 3)])
    anneau = np.concatenate([np.isin(seam, np.asarray(bords[0])), np.array(de_anneau, bool)])
    base = np.median(col_b[anneau], 0)
    if base_de == "visage" and "颜" in noms:
        # neck with a painted shadow: the ring is in shadow, the body skin takes the face's tint
        i = noms.index("颜")
        base = np.median(texel(images[i], t["uv"][np.unique(t["faces"][t["mat_face"] == i])]), 0)

    # colour per body skin vertex: weighted mean of the 8 nearest edge points, blended towards the base
    Fb = t["faces"][t["mat_face"] == i_peau]
    vb = np.unique(Fb)
    P = t["pos"]
    D = np.linalg.norm(P[vb][:, None] - pos_b[None], axis=2)
    k = np.argsort(D, 1)[:, :8]
    dk = np.take_along_axis(D, k, 1)
    wk = 1.0 / (dk ** 2 + (0.5 * MM) ** 2)
    proche = (wk[..., None] * col_b[k]).sum(1) / wk.sum(1)[:, None]
    a = lisse(np.clip(dk[:, 0] / rayon, 0, 1))[:, None]
    cv = np.zeros((len(P), 3))
    cv[vb] = (1 - a) * proche + a * base

    # painting: skin triangles rasterised into a new texture, colour interpolated from the vertices
    taille = 1024
    img = np.tile(np.append(base, 1.0), (taille, taille, 1)).astype(np.float32)
    uvb = np.mod(t["uv"], 1.0) * taille
    # strip faces painted first: their copies take the UVs of the nearest body vertex, some stretch in UV
    # over the nape; the body faces then repaint their own texels
    de_bande = np.zeros(len(P), bool)
    de_bande[np.asarray(copies, int)] = True
    for f in Fb[np.argsort(~de_bande[Fb].any(1), kind="stable")]:
        p = uvb[f]
        x0, y0 = np.maximum(np.floor(p.min(0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(p.max(0)).astype(int) + 2, taille)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        v0, v1 = p[1] - p[0], p[2] - p[0]
        det = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(det) < 1e-9:
            continue
        l1 = ((xs - p[0, 0]) * v1[1] - (ys - p[0, 1]) * v1[0]) / det
        l2 = ((ys - p[0, 1]) * v0[0] - (xs - p[0, 0]) * v0[1]) / det
        l0 = 1 - l1 - l2
        marge = -1.5 / max(abs(det) ** 0.5, 1.0)     # one and a half texel overlap: no fringe between triangles
        dedans = (l0 >= marge) & (l1 >= marge) & (l2 >= marge)
        c = l0[..., None] * cv[f[0]] + l1[..., None] * cv[f[1]] + l2[..., None] * cv[f[2]]
        img[y0:y1, x0:x1, :3][dedans] = c[dedans]
    nom = "peau_corps.png"
    greffe_tete.ecrire_image(os.path.join(dossier, "tex", nom), img)
    R["textures"].append("tex/" + nom)
    R["materiaux"][i_peau]["texture"] = len(R["textures"]) - 1
    lab = greffe_tete.srgb_vers_lab(col_b)
    journal["teinte"] = {"base": [round(float(x), 4) for x in base], "points_de_bord": len(col_b),
                         "dispersion_lab_bords": round(float(np.median(np.linalg.norm(lab - np.median(lab, 0), axis=1))), 2),
                         "ecart_lab_max_bords": round(float(np.linalg.norm(lab - greffe_tete.srgb_vers_lab(base), axis=1).max()), 2),
                         "rayon_mm": round(rayon / MM, 1), "base_de": base_de}


def main(argv=None):
    a = args(argv)
    os.makedirs(a.sortie, exist_ok=True)
    tmp = os.path.join(a.sortie, "_greffe")
    os.makedirs(tmp, exist_ok=True)
    piece, n_peau = piece_de_tete(a.tete, tmp, a.peau)
    H = PMX.lire(piece)
    presents = [m["nom"] for m in H["materiaux"] if m["n_indices"]]
    prendre = [n for n in VISAGE if n in presents] + [PIECE]
    greffe_tete.main(["--corps", a.corps, "--tete", piece, "--sortie", tmp, "--garder-crane", "--sans-teinte",
                      "--prendre", ",".join(prendre), "--peau", a.peau, "--ecart-cou-max", a.ecart_cou_max])
    R = PMX.lire(os.path.join(tmp, "resultat.pmx"))
    journal = {"corps": a.corps, "tete": a.tete, "faces_peau_tete": n_peau}
    if os.path.isdir(os.path.join(a.sortie, "tex")):
        shutil.rmtree(os.path.join(a.sortie, "tex"))
    shutil.copytree(os.path.join(tmp, "tex"), os.path.join(a.sortie, "tex"))
    tr = json.load(open(os.path.join(tmp, "greffe.json"), encoding="utf-8"))["translation_mm"]
    info = coudre(R, a.peau, journal, (a.tete, tr))
    journal["indices"] = {"anneau": [int(x) for x in info["anneau"]], "crane": [int(x) for x in info["crane"]],
                          "copies": {str(k): int(v) for k, v in info["copies"].items()}}
    teinte(R, a.sortie, journal, [b for b in (info["anneau"], info["crane"]) if len(b)], list(info["copies"].values()),
           base_de=a.teinte_base)
    chemin = os.path.join(a.sortie, "resultat.pmx")
    pmx_ecrire.ecrire(R, chemin + ".tmp")
    os.replace(chemin + ".tmp", chemin)
    journal["sommets"] = len(R["sommets"]["pos"])
    journal["faces"] = len(R["faces"])
    with open(os.path.join(a.sortie, "raccord.json") + ".tmp", "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=1)
    os.replace(os.path.join(a.sortie, "raccord.json") + ".tmp", os.path.join(a.sortie, "raccord.json"))
    print(json.dumps(journal, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

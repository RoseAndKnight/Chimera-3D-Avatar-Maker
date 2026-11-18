"""Weld the edge of the grafted face to the body skin, as the game models do.

    bin/blender-queue -b --factory-startup --python tools/clothes/neck_seam.py -- \
        --pmx MODEL.pmx --sortie FOLDER [--peau 肌] [--visage 颜,颜_A]
    (without Blender, python3 -I does everything except the tint)

On Jane, Evelyn, Ellen, the face and the neck are a single surface: the skin ends exactly on the face
edge, under the jaw, with the same positions, the same normals, the same weights (all on 頭) and the
outline at 1 everywhere. The jaw line is then nothing but the silhouette of a continuous surface. A
body coming out of `peau_complete.py`, on the contrary, has a closed scalp ball whose bottom runs 5 to
15 mm under the face edge: a face laid on top of it reads as a disc.

What the tool does, on the skin material only (`greffe_tete.main` calls it by default):
 - if the skin is made of a closed scalp ball and a neck tube going through it, the two are first
   welded into a single surface (`unir_tube`): otherwise their intersection edge and the tube's outline
   shell draw a boundary and dashes under the chin;
 - the skin under the face goes away: any face with a vertex under the face's footprint (ray from the
   head centre -> vertex that crosses the face);
 - the skin edge thus freed is glued onto the outer edge of the face: each face corner receives a skin
   vertex, the other vertices are projected onto the edges;
 - the rest of the skin follows through a harmonic displacement that fades out at `rayon` along the
   surface: a gentle slope instead of a step;
 - normals: the face's on the weld, blended towards the skin's over `rayon_normales`;
 - weights: the face's on the weld, blended over `rayon_poids`; morphs that move the face edge carry the
   weld along. The face itself is not touched;
 - scalp: the body skin is pulled in behind the front silhouette of the face (`rentrer_crane`);
 - line: the face outline thins from the mouth to the tip of the chin (`amincir_menton`);
 - tint: along the weld, the skin texture takes the colour of the face edge, and returns to its own
   over `RAYON_TEINTE_MM`: no more colour step. Must run in Blender (texture reading).

No effect if no skin edge is within reach (`portee`) of the face, or if that edge has fewer vertices
than the face has corners (summary: `peau_trop_lache`): it is a neck stump hidden under the head.
Command-line output: <sortie>/resultat.pmx (tex/ textures copied over), raccord.json.
"""
import argparse
import heapq
import json
import os
import shutil
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
sys.path.insert(0, os.path.join(ICI, "..", "head"))
import pmx as PMX  # noqa: E402
import pmx_write as pmx_ecrire  # noqa: E402

MM = 1.0 / 80.0
RAYON_MM = 40.0
RAYON_NORMALES_MM = 15.0
RAYON_POIDS_MM = 8.0
PORTEE_MM = 30.0
RAYON_DETENTE_MM = 6.0
RAYON_TEINTE_MM = 25.0
PAS_SILHOUETTE_MM = 3.0     # height slices for the front silhouette of the face
COU_SOUS_MENTON_MM = 20.0   # where the neck width is measured, under the chin
RAYON_CRANE_MM = 12.0       # the scalp tightening fades out at this distance from the weld
TRAIT_MENTON_BAS = 0.0       # outline weight at the tip of the chin (1 = full line)
TRAIT_MENTON_HAUTEUR = 0.15  # mouth, as a fraction of the face height above the chin
PART_COINS_MAX = 0.1
RAYON_TUBE_MM = 20.0
RAYON_NORMALES_TUBE_MM = 12.0
RAYON_POIDS_TUBE_MM = 10.0
ECART_MAX_TUBE_MM = 25.0
SOUS_VISAGE_MM = 3.0        # the body skin stays at least this deep under the face
PORTEE_SOUS_MM = 8.0        # beyond this, the skin is not under the face
BORD_SOUS_MM = 4.0          # distance from the face edge at which the skin reaches that depth
RAYON_TEST = np.array([0.6123, 0.4771, 0.6303])     # arbitrary direction: runs along no edge
RAYON_HORIZONTAL = np.array([0.9998, 0.0011, 0.0173])


def soudure(pos, tol=1e-4):
    """Welded index of each vertex (positions equal within tol)."""
    _, inv = np.unique(np.round(pos / tol).astype(np.int64), axis=0, return_inverse=True)
    return inv.ravel()


def aretes_de_bord(F, sou):
    """Edges (raw indices) that touch only one face, once the vertices are welded."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    S = np.sort(sou[E], axis=1)
    _, inv, cnt = np.unique(S, axis=0, return_inverse=True, return_counts=True)
    return E[cnt[inv.ravel()] == 1]


def point_segment(q, a, b):
    """Closest point of each q on the best segment (a, b): (point, index, distance)."""
    ab = b - a
    t = np.clip(((q[:, None] - a) * ab).sum(2) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    pts = a + t[..., None] * ab
    d = ((q[:, None] - pts) ** 2).sum(2)
    k = d.argmin(1)
    return pts[np.arange(len(q)), k], k, np.sqrt(d[np.arange(len(q)), k])


def bord_exterieur(E, sou, pos):
    """Edges of the longest boundary loop: the outline of the face, without the eye and mouth holes."""
    s = sou[E]
    lab = np.arange(sou.max() + 1)
    while True:
        m = np.minimum(lab[s[:, 0]], lab[s[:, 1]])
        nv = lab.copy()
        np.minimum.at(nv, s[:, 0], m)
        np.minimum.at(nv, s[:, 1], m)
        nv = nv[nv]
        if np.array_equal(nv, lab):
            break
        lab = nv
    groupes = lab[s[:, 0]]
    longueur = np.linalg.norm(pos[E[:, 0]] - pos[E[:, 1]], axis=1)
    somme = np.bincount(groupes, longueur, minlength=len(lab))
    return E[groupes == int(somme.argmax())]


def rayons_visage(C, q, tri):
    """Distance from C to the first crossing of the surface `tri` along the ray C -> q; inf if none."""
    A, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    d = q - C
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    res = np.full(len(q), np.inf)
    for s in range(0, len(q), 256):
        dd = d[s:s + 256]
        h = np.cross(dd[:, None], e2[None])
        a = (e1[None] * h).sum(2)
        sur = np.abs(a) > 1e-12
        f = np.where(sur, 1.0 / np.where(sur, a, 1.0), 0.0)
        w = C[None, None] - A[None]
        u = f * (w * h).sum(2)
        qq = np.cross(w, e1[None])
        v = f * (dd[:, None] * qq).sum(2)
        t = f * (e2[None] * qq).sum(2)
        ok = sur & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 1e-6)
        res[s:s + 256] = np.where(ok, t, np.inf).min(1)
    return res


def groupes_aretes(E, sou):
    """Loop number (connected component over welded vertices) of each boundary edge."""
    s = sou[E]
    lab = np.arange(sou.max() + 1)
    while True:
        m = np.minimum(lab[s[:, 0]], lab[s[:, 1]])
        nv = lab.copy()
        np.minimum.at(nv, s[:, 0], m)
        np.minimum.at(nv, s[:, 1], m)
        nv = nv[nv]
        if np.array_equal(nv, lab):
            return lab[s[:, 0]]
        lab = nv


def choisir_boucles(pos, E, groupes, coins, portee):
    """Skin boundary loops facing the face edge, from most to least covering."""
    restants = np.ones(len(coins), bool)
    choisies = []
    uniques = np.unique(groupes)
    couvre = {}
    for g in uniques:
        v = np.unique(E[groupes == g])
        d = np.sqrt(((pos[coins][:, None] - pos[v][None]) ** 2).sum(2)).min(1)
        couvre[int(g)] = d < portee
    while True:
        g = max(couvre, key=lambda k: int((couvre[k] & restants).sum()), default=None)
        if g is None or (couvre[g] & restants).sum() < max(2, 0.05 * len(coins)):
            return choisies
        choisies.append(g)
        restants &= ~couvre.pop(g)


def distances(n, aretes, longueurs, sources):
    """Distance along the surface to the nearest source, and that source (welded graph)."""
    voisins = [[] for _ in range(n)]
    for (a, b), l in zip(aretes.tolist(), longueurs.tolist()):
        voisins[a].append((b, l))
        voisins[b].append((a, l))
    dist = np.full(n, np.inf)
    src = np.full(n, -1)
    tas = []
    for s in sources:
        dist[s] = 0.0
        src[s] = s
        tas.append((0.0, s))
    heapq.heapify(tas)
    while tas:
        d, a = heapq.heappop(tas)
        if d > dist[a]:
            continue
        for b, l in voisins[a]:
            if d + l < dist[b]:
                dist[b] = d + l
                src[b] = src[a]
                heapq.heappush(tas, (d + l, b))
    return dist, src


def harmonique(n, aretes, fixe, val, iterations=4000):
    """Extend `val` (imposed where `fixe`) over the graph by neighbour averaging."""
    x = val.copy()
    a, b = aretes[:, 0], aretes[:, 1]
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    libre = ~fixe & (deg > 0)
    for _ in range(iterations):
        somme = np.stack([np.bincount(a, x[b, k], n) + np.bincount(b, x[a, k], n) for k in range(3)], 1)
        nv = somme[libre] / deg[libre][:, None]
        ecart = np.abs(nv - x[libre]).max() if libre.any() else 0.0
        x[libre] = nv
        if ecart < 1e-7:
            break
    return x


def lisse(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fondre_poids(os_a, p_a, os_b, p_b, t):
    """Blend (1-t) a + t b of two weight sets, four bones at most: (type, bones, weights)."""
    d = {}
    for o, w, f in list(zip(os_a, p_a, [1 - t] * 4)) + list(zip(os_b, p_b, [t] * 4)):
        if o >= 0 and w > 0:
            d[int(o)] = d.get(int(o), 0.0) + float(w) * f
    pris = sorted(d.items(), key=lambda kv: -kv[1])[:4]
    s = sum(w for _, w in pris)
    o = [k for k, _ in pris] + [-1] * (4 - len(pris))
    w = [v / s for _, v in pris] + [0.0] * (4 - len(pris))
    return {1: 0, 2: 1}.get(len(pris), 2), tuple(o), tuple(w)


def traversees(q, d, tri):
    """Number of triangles crossed by the ray q + t d (t > 0), for each q."""
    A, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    a = (e1 * h).sum(1)
    sur = np.abs(a) > 1e-12
    A, e1, e2, h, f = A[sur], e1[sur], e2[sur], h[sur], 1.0 / a[sur]
    n = np.zeros(len(q), int)
    for s in range(0, len(q), 128):
        w = q[s:s + 128, None] - A[None]
        u = f * (w * h[None]).sum(2)
        qq = np.cross(w, e1[None])
        v = f * (qq * d).sum(2)
        t = f * (qq * e2[None]).sum(2)
        n[s:s + 128] = ((u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)).sum(1)
    return n


def graphe(F, sou):
    """Welded graph of faces F: (groups, number per group, representative, live vertices, edges)."""
    vivants = np.unique(F)
    gs = np.unique(sou[vivants])
    num = np.full(sou.max() + 1, -1)
    num[gs] = np.arange(len(gs))
    rep = np.zeros(len(gs), int)
    rep[num[sou[vivants]]] = vivants
    Eg = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    Eg = np.unique(np.sort(num[sou[Eg]], axis=1), axis=0)
    return gs, num, rep, vivants, Eg[Eg[:, 0] != Eg[:, 1]]


def detendre(x, Eg, libres, iterations=12):
    """Smooth the positions x (one per graph node) over the free nodes."""
    a, b = Eg[:, 0], Eg[:, 1]
    n = len(x)
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    libres = libres & (deg > 0)
    for _ in range(iterations):
        somme = np.stack([np.bincount(a, x[b, k], n) + np.bincount(b, x[a, k], n) for k in range(3)], 1)
        x[libres] = 0.5 * x[libres] + 0.5 * somme[libres] / deg[libres][:, None]
    return x


def fondre_normales(P, nor, F, num, sou, rep, n, v, k, d, src, n_soudure, rayon_normales, rayon):
    """Weld normals, then new geometry, then original normals further away."""
    Fg = num[sou[F]]
    e1, e2 = P[rep[Fg[:, 1]]] - P[rep[Fg[:, 0]]], P[rep[Fg[:, 2]]] - P[rep[Fg[:, 0]]]
    nf = np.cross(e1, e2)
    n_geo = np.stack([np.bincount(Fg.ravel(), np.repeat(nf[:, c], 3), n) for c in range(3)], 1)
    n_geo /= np.maximum(np.linalg.norm(n_geo, axis=1, keepdims=True), 1e-12)
    n_src = np.zeros((n, 3))
    for g, x in n_soudure.items():
        n_src[int(num[g])] = x
    n_src = n_src[np.maximum(src, 0)]
    # the PMX normal direction is authoritative: the geometry is flipped if its sign is reversed
    signe = np.sign((n_geo[k] * nor[v]).sum(1, keepdims=True))
    signe[signe == 0] = 1
    nv = n_geo[k] * signe
    w = lisse((d - rayon_normales) / max(rayon - rayon_normales, 1e-9))[:, None]
    nv = nv * (1 - w) + nor[v] * w
    w = (1 - lisse(d / rayon_normales))[:, None]
    nv = nv * (1 - w) + n_src[k] * w
    return nv / np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)


def unir_tube(pos, nor, faces, mat_face, i_peau, os_, poids, lo, hi, rayon=RAYON_TUBE_MM * MM,
              rayon_normales=RAYON_NORMALES_TUBE_MM * MM, rayon_poids=RAYON_POIDS_TUBE_MM * MM):
    """Weld the neck tube to the closed scalp ball it goes through; None if the skin is not built that way.

    The part of the tube inside the ball and the bottom of the ball inside the tube go away; the edge of the
    ball's hole (denser) folds onto the corners of the tube edge: the edges coincide, with no gap.
    """
    idx = np.where(mat_face == i_peau)[0]
    F = faces[idx]
    sou = soudure(pos)
    comp = groupes_aretes(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), sou)[:len(F)]
    ids = np.unique(comp).tolist()
    c = pos[F].mean(1)
    emprise = np.all((c >= lo) & (c <= hi), axis=1)
    boule, meilleur = None, 99
    for i in ids:
        m = comp == i
        nb = int((m & emprise).sum())
        if nb > meilleur and len(aretes_de_bord(F[m], sou)) <= 0.03 * m.sum():
            boule, meilleur = i, nb
    if boule is None:
        return None
    F_b = F[comp == boule]
    tri_b = pos[F_b]
    bas_b = tri_b[..., 1].min()
    tube, meilleur = None, 7
    for i in ids:
        m = comp == i
        if i == boule or m.sum() < 50:
            continue
        v = np.unique(F[m])
        if pos[v, 1].min() > bas_b - 30 * MM or pos[v, 1].max() < bas_b:
            continue
        dedans = traversees(pos[v], RAYON_TEST, tri_b) % 2 == 1
        if dedans.sum() > meilleur and (~dedans).sum() >= 8:
            tube, meilleur = i, int(dedans.sum())
    if tube is None:
        return None
    F_t = F[comp == tube]
    v_t = np.unique(F_t)
    dans_b = np.zeros(len(pos), bool)
    dans_b[v_t] = traversees(pos[v_t], RAYON_TEST, tri_b) % 2 == 1
    v_b = np.unique(F_b)
    v_b = v_b[pos[v_b, 1] < pos[v_t, 1].max()]
    dans_t = np.zeros(len(pos), bool)
    dans_t[v_b] = traversees(pos[v_b], RAYON_HORIZONTAL, pos[F_t]) % 2 == 1
    r_b, r_t = dans_t[F_b].any(1), dans_b[F_t].any(1)
    G_b, G_t = F_b[~r_b], F_t[~r_t]
    if not r_b.any() or not r_t.any() or not len(G_b) or not len(G_t):
        return None

    def nouveau_bord(G, F0):
        E = aretes_de_bord(G, sou)
        avant = {tuple(x) for x in np.sort(sou[aretes_de_bord(F0, sou)], axis=1).tolist()}
        return bord_exterieur(E[[tuple(x) not in avant for x in np.sort(sou[E], axis=1).tolist()]], sou, pos)

    def uniques(E):
        v = np.unique(E)
        return v[np.unique(sou[v], return_index=True)[1]]
    coins = uniques(nouveau_bord(G_t, F_t))
    bord = uniques(nouveau_bord(G_b, F_b))
    d = np.sqrt(((pos[coins][:, None] - pos[bord][None]) ** 2).sum(2))
    pris = {}
    for c_ in np.argsort(d.min(1)).tolist():
        for j in np.argsort(d[c_]).tolist():
            if j not in pris:
                pris[j] = c_
                break
    vers = d.argmin(0)
    for j, c_ in pris.items():
        vers[j] = c_
    ecart = np.linalg.norm(pos[coins[vers]] - pos[bord], axis=1).max()
    if ecart > ECART_MAX_TUBE_MM * MM:
        return None

    # --- the ball's hole goes down onto the tube edge; the ball follows, the weld stays fixed
    gs, num, rep, vivants, Eg = graphe(G_b, sou)
    longueurs = np.linalg.norm(pos[rep[Eg[:, 0]]] - pos[rep[Eg[:, 1]]], axis=1)
    sources = sorted(set(num[sou[bord]].tolist()))
    dist, src = distances(len(gs), Eg, longueurs, sources)
    fixe = dist >= rayon
    val = np.zeros((len(gs), 3))
    k_b = num[sou[bord]]
    fixe[k_b] = True
    val[k_b] = pos[coins[vers]] - pos[bord]
    dep = harmonique(len(gs), Eg, fixe, val)
    x = detendre(pos[rep] + dep, Eg, (dist < RAYON_DETENTE_MM * MM) & ~np.isin(np.arange(len(gs)), sources))
    P = pos.copy()
    k_v = num[sou[vivants]]
    P[vivants] = x[k_v]

    # --- edge faces collapsed onto a single corner are removed
    sou2 = soudure(P)
    plates = np.array([len(set(t)) < 3 for t in sou2[G_b].tolist()], bool)
    garde = np.ones(len(faces), bool)
    sel_b = idx[comp == boule]
    sel_t = idx[comp == tube]
    garde[sel_b[r_b]] = False
    garde[sel_t[r_t]] = False
    garde[sel_b[~r_b][plates]] = False

    # --- normals: mean of the two surfaces on the weld, blended on each side
    def moyenne(n):
        return n / max(np.linalg.norm(n), 1e-12)
    n_coins = {}
    for c_ in range(len(coins)):
        n_b = nor[bord[vers == c_]]
        n_coins[c_] = moyenne(nor[coins[c_]] + moyenne(n_b.sum(0)) if len(n_b) else nor[coins[c_]])
    n_s_b = {int(sou[v]): n_coins[int(c_)] for v, c_ in zip(bord.tolist(), vers.tolist())}
    N = nor.copy()
    N[vivants] = fondre_normales(P, nor, G_b[~plates], num, sou, rep, len(gs), vivants, k_v, dist[k_v], src, n_s_b, rayon_normales, rayon)
    gt, num_t, rep_t, viv_t, Et = graphe(G_t, sou)
    k_t = num_t[sou[viv_t]]
    long_t = np.linalg.norm(pos[rep_t[Et[:, 0]]] - pos[rep_t[Et[:, 1]]], axis=1)
    dist_t, src_t = distances(len(gt), Et, long_t, sorted(set(num_t[sou[coins]].tolist())))
    n_s_t = {int(sou[coins[c_]]): n_coins[c_] for c_ in range(len(coins))}
    N[viv_t] = fondre_normales(P, nor, G_t, num_t, sou, rep_t, len(gt), viv_t, k_t, dist_t[k_t], src_t, n_s_t, rayon_normales, rayon)

    # --- weights: the tube's on the weld, blended over the ball
    os2, p2, nouveaux = os_.copy(), poids.copy(), {}
    pour_k = {int(num[sou[v]]): (os_[coins[c_]], poids[coins[c_]]) for v, c_ in zip(bord.tolist(), vers.tolist())}
    for v, k, dd in zip(vivants.tolist(), k_v.tolist(), dist[k_v].tolist()):
        if dd < rayon_poids:
            o_s, w_s = pour_k[int(src[k])]
            ty, o, w = fondre_poids(os_[v], poids[v], o_s, w_s, float(1 - lisse(dd / rayon_poids)))
            nouveaux[v] = (ty, o, w)
            os2[v], p2[v] = o, w
    return dict(pos=P, nor=N, os=os2, poids=p2, garde=garde, poids_modifies=nouveaux,
                bilan={"faces_boule_retirees": int(r_b.sum() + plates.sum()), "faces_tube_retirees": int(r_t.sum()),
                       "coins_tube": len(coins), "sommets_boule_soudes": len(bord), "ecart_max_mm": round(float(ecart / MM), 2)})


def raccord_cou(pos, nor, faces, mat_face, i_peau, i_visage, os_, poids, rayon=RAYON_MM * MM,
                rayon_normales=RAYON_NORMALES_MM * MM, rayon_poids=RAYON_POIDS_MM * MM, portee=PORTEE_MM * MM,
                rayon_detente=RAYON_DETENTE_MM * MM):
    """Returns a dict: garde, pos, nor (new arrays), poids {v: (type, bones, weights)}, soudure {v: (a, b, t)}, bilan."""
    masque_peau = mat_face == i_peau
    F_vis = faces[np.isin(mat_face, i_visage)]
    res = {"garde": np.ones(len(faces), bool), "pos": pos.copy(), "nor": nor.copy(), "poids": {}, "soudure": {},
           "bilan": {"faces_peau": int(masque_peau.sum()), "retirees": 0, "soudes": 0}}
    if not len(F_vis) or not masque_peau.any():
        return res
    tri = pos[F_vis]
    lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    C = 0.5 * (lo + hi)
    union = unir_tube(pos, nor, faces, mat_face, i_peau, os_, poids, lo, hi)
    garde0, poids_tube = np.ones(len(faces), bool), {}
    if union is not None:
        pos, nor, os_, poids, garde0, poids_tube = union["pos"], union["nor"], union["os"], union["poids"], union["garde"], union["poids_modifies"]
        masque_peau = masque_peau & garde0
    F_peau = faces[masque_peau]
    sou = soudure(pos)

    # --- the skin under the face goes away
    candidats = np.unique(F_peau[pos[F_peau].mean(1)[:, 1] > lo[1]])
    dans = np.zeros(len(pos), bool)
    if len(candidats):
        dans[candidats[np.isfinite(rayons_visage(C, pos[candidats], tri))]] = True
    retirer = masque_peau.copy()
    retirer[masque_peau] = dans[F_peau].any(1)

    # --- face edge and skin edges within its reach
    bord_vis = bord_exterieur(aretes_de_bord(F_vis, sou), sou, pos)
    F_reste = faces[masque_peau & ~retirer]
    E = aretes_de_bord(F_reste, sou)
    if not len(E):
        return res
    groupes = groupes_aretes(E, sou)
    coins = np.unique(bord_vis)
    _, ic = np.unique(sou[coins], return_index=True)
    coins = coins[ic]
    choisies = choisir_boucles(pos, E, groupes, coins, portee)
    if not choisies:
        res["bilan"]["hors_portee"] = True
        return res
    v_bord = np.unique(E[np.isin(groupes, choisies)])
    _, iu = np.unique(sou[v_bord], return_index=True)
    v_bord = v_bord[iu]
    a, b = bord_vis[:, 0], bord_vis[:, 1]
    proj, ka, d_bord = point_segment(pos[v_bord], pos[a], pos[b])
    pris = d_bord < portee
    v_bord, proj, ka = v_bord[pris], proj[pris], ka[pris]
    if not len(v_bord):
        res["bilan"]["hors_portee"] = True
        return res

    # each face corner receives the nearest skin vertex: no gap in the corners
    d_coin = np.sqrt(((pos[coins][:, None] - pos[v_bord][None]) ** 2).sum(2))
    ordre = np.argsort(d_coin.min(1))
    pris_coin = {}
    sans_coin = 0
    for c in ordre.tolist():
        for j in np.argsort(d_coin[c]).tolist():
            if j not in pris_coin:
                if d_coin[c, j] < portee:
                    pris_coin[j] = c
                else:
                    sans_coin += 1
                break

    # skin looser than the face (neck stump of a game body, hidden under the head) is not welded:
    # its few vertices would each travel far, and it is not visible
    if sans_coin > PART_COINS_MAX * len(coins):
        res["bilan"].update(peau_trop_lache=True, coins=len(coins), coins_sans_sommet=sans_coin)
        return res
    res["garde"] = ~retirer & garde0
    res["pos"], res["nor"], res["poids"] = pos.copy(), nor.copy(), dict(poids_tube)
    retirees_tube = int((~garde0).sum())
    res["bilan"]["retirees"] = int(retirer.sum()) + retirees_tube
    if union is not None:
        res["bilan"]["tube"] = union["bilan"]

    # --- weld: position, normal, weights of the face edge
    nor_coin = {}
    for v in np.unique(F_vis).tolist():
        nor_coin.setdefault(int(sou[v]), []).append(nor[v])
    def n_bord(v):
        n = np.mean(nor_coin[int(sou[v])], 0)
        return n / max(np.linalg.norm(n), 1e-12)
    n_soudure = {}
    cible = {}
    for j, v in enumerate(v_bord.tolist()):
        if j in pris_coin:
            c = int(coins[pris_coin[j]])
            cible[int(sou[v])] = pos[c].copy()
            n_soudure[int(sou[v])] = n_bord(c)
            res["soudure"][v] = (c, c, 0.0)
        else:
            k = int(ka[j])
            ab = pos[b[k]] - pos[a[k]]
            t = float(np.clip((proj[j] - pos[a[k]]) @ ab / max(ab @ ab, 1e-12), 0, 1))
            cible[int(sou[v])] = proj[j].copy()
            n = (1 - t) * n_bord(int(a[k])) + t * n_bord(int(b[k]))
            n_soudure[int(sou[v])] = n / max(np.linalg.norm(n), 1e-12)
            res["soudure"][v] = (int(a[k]), int(b[k]), t)

    # --- harmonic displacement of the skin around the weld (welded graph of the kept skin)
    vivants = np.unique(F_reste)
    gs = np.unique(sou[vivants])
    num = np.full(sou.max() + 1, -1)
    num[gs] = np.arange(len(gs))
    rep = np.zeros(len(gs), int)
    rep[num[sou[vivants]]] = vivants
    Eg = np.vstack([F_reste[:, [0, 1]], F_reste[:, [1, 2]], F_reste[:, [2, 0]]])
    Eg = np.unique(np.sort(num[sou[Eg]], axis=1), axis=0)
    Eg = Eg[Eg[:, 0] != Eg[:, 1]]
    longueurs = np.linalg.norm(pos[rep[Eg[:, 0]]] - pos[rep[Eg[:, 1]]], axis=1)
    sources = sorted(int(num[g]) for g in cible)
    dist, src = distances(len(gs), Eg, longueurs, sources)
    fixe = dist >= rayon
    val = np.zeros((len(gs), 3))
    for g, c in cible.items():
        k = int(num[g])
        fixe[k] = True
        val[k] = c - pos[rep[k]]
    dep = harmonique(len(gs), Eg, fixe, val)
    res["bilan"].update(soudes=len(cible), coins=len(coins), coins_sans_sommet=sans_coin,
                        deplacement_max_mm=round(float(np.linalg.norm(val, axis=1).max() / MM), 2),
                        deplacement_median_mm=round(float(np.median(np.linalg.norm(val[sources], axis=1)) / MM), 2),
                        peau_bougee=int((dist < rayon).sum()))
    P = res["pos"]
    bouges = np.where(dist < rayon)[0]
    masque_g = np.zeros(sou.max() + 1, bool)
    masque_g[gs[bouges]] = True
    v_touches = vivants[masque_g[sou[vivants]]]
    k_touches = num[sou[v_touches]]
    P[v_touches] = pos[v_touches] + dep[k_touches]

    # --- relax the first rings: gluing a sawtooth edge wrinkles the skin touching it
    src_seam = np.zeros(len(gs), bool)
    src_seam[sources] = True
    x = detendre(P[rep].copy(), Eg, (dist < rayon_detente) & ~src_seam)
    P[v_touches] = x[k_touches]

    # --- normals: imposed on the weld, then new geometry, back to the original normals further away
    d = dist[k_touches]
    res["nor"][v_touches] = fondre_normales(P, nor, F_reste, num, sou, rep, len(gs), v_touches, k_touches, d, src, n_soudure,
                                            rayon_normales, rayon)[:]

    # --- weights: the face's on the weld, blended around it
    pour_v = {}
    for v, (va, vb, t) in res["soudure"].items():
        ty, o, w = fondre_poids(os_[va], poids[va], os_[vb], poids[vb], t)
        pour_v[int(num[sou[v]])] = (o, w)
    for v, k, dd in zip(v_touches.tolist(), k_touches.tolist(), d.tolist()):
        if dd >= rayon_poids:
            continue
        o_s, w_s = pour_v[int(src[k])]
        f = float(1 - lisse(dd / rayon_poids))
        if {x for x, y in zip(os_[v], poids[v]) if x >= 0 and y > 0} == {x for x, y in zip(o_s, w_s) if x >= 0 and y > 0} and f < 1:
            if abs(np.sort(poids[v]) - np.sort(w_s)).max() < 1e-3:
                continue
        res["poids"][v] = fondre_poids(os_[v], poids[v], o_s, w_s, f)

    # --- duplicated welded vertices (skin UV seams) follow their weld
    par_g = {int(sou[v]): s for v, s in res["soudure"].items()}
    for v in v_touches.tolist():
        if int(sou[v]) in par_g:
            res["soudure"][v] = par_g[int(sou[v])]

    # --- faces collapsed by the gluing
    garde = res["garde"]
    idx = np.where(masque_peau & garde)[0]
    tr = P[faces[idx]]
    aire = np.linalg.norm(np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0]), axis=1)
    plates = idx[aire < 1e-9]
    garde[plates] = False
    # a face resting on the weld that flips over is a gluing fold: its neighbours cover its place
    nf = np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0])
    nm = res["nor"][faces[idx]].mean(1)
    cos = (nf * nm).sum(1) / np.maximum(aire * np.linalg.norm(nm, axis=1), 1e-30)
    sens = np.sign(np.median(cos))
    sur_soudure = (src_seam[num[sou[faces[idx]]]].sum(1) >= 2)
    plis = idx[sur_soudure & (cos * sens < 0.1)]
    garde[plis] = False
    res["bilan"].update(retirees=int(retirer.sum()) + len(plates) + len(plis) + retirees_tube, faces_plates=len(plates), plis=len(plis))
    res["graphe"] = dict(num=num, sou=sou, rep=rep, Eg=Eg, dist=dist, n=len(gs), F_reste=F_reste)
    return res


def lire_image(chemin):
    """sRGB RGBA, rows top to bottom (Blender's Python)."""
    import bpy
    im = bpy.data.images.load(chemin, check_existing=False)
    im.colorspace_settings.name = "Non-Color"
    w, h = im.size
    px = np.empty(4 * w * h, np.float32)
    im.pixels.foreach_get(px)
    bpy.data.images.remove(im)
    return px.reshape(h, w, 4)[::-1].copy()


def ecrire_image(chemin, px):
    import bpy
    h, w = px.shape[:2]
    im = bpy.data.images.new("tmp", w, h, alpha=True)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(np.clip(px[::-1], 0, 1).astype(np.float32).ravel())
    tmp = chemin + ".tmp.png"
    im.filepath_raw = tmp
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)
    os.replace(tmp, chemin)


def texel(img, uv):
    h, w = img.shape[:2]
    u = np.mod(uv, 1.0)
    return img[(u[:, 1] * h).astype(int).clip(0, h - 1), (u[:, 0] * w).astype(int).clip(0, w - 1), :3]


def peindre_teinte(dossier, textures, materiaux, uv, mat_sommet, i_peau, r, rayon=RAYON_TEINTE_MM * MM):
    """The skin texture takes the colour of the face edge along the weld; summary, or None outside Blender."""
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("join: tint must run in Blender (texture reading); nothing changed")
        return None
    def chemin_tex(i_mat):
        t = materiaux[i_mat]["texture"]
        return os.path.join(dossier, textures[t].replace("\\", "/")) if 0 <= t < len(textures) else None
    mats = {i_peau} | {int(mat_sommet[x]) for v in r["soudure"].values() for x in v[:2]}
    absentes = [i for i in mats if chemin_tex(i) is None or not os.path.isfile(chemin_tex(i))]
    if absentes:
        print("join: texture missing for materials", absentes, "; tint not matched")
        return None
    g = r["graphe"]
    num, sou, Eg, dist, n = g["num"], g["sou"], g["Eg"], g["dist"], g["n"]
    images = {}

    def image(i_mat):
        t = materiaux[i_mat]["texture"]
        if t not in images:
            images[t] = lire_image(os.path.join(dossier, textures[t].replace("\\", "/")))
        return images[t]
    img_peau = image(i_peau)
    fixe = dist >= rayon
    val = np.zeros((n, 3))
    vus = {}
    for v, (a, b, t) in r["soudure"].items():
        k = int(num[sou[v]])
        ca = texel(image(int(mat_sommet[a])), uv[[a]])[0]
        cb = texel(image(int(mat_sommet[b])), uv[[b]])[0]
        vus.setdefault(k, []).append((1 - t) * ca + t * cb - texel(img_peau, uv[[v]])[0])
    for k, d in vus.items():
        fixe[k] = True
        val[k] = np.mean(d, 0)
    delta = harmonique(n, Eg, fixe, val)

    # painting: skin triangles of the strip, rasterised into the texture
    h, w = img_peau.shape[:2]
    F = g["F_reste"]
    Fg = num[sou[F]]
    bande = (dist[Fg] < rayon).any(1)
    F, Fg = F[bande], Fg[bande]
    correction = np.zeros((h, w, 3), np.float32)
    couvert = np.zeros((h, w), bool)
    for f, fg in zip(F.tolist(), Fg.tolist()):
        p = np.mod(uv[f], 1.0) * [w, h]
        x0, y0 = np.maximum(np.floor(p.min(0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(p.max(0)).astype(int) + 2, [w, h])
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
        marge = -1.0 / max(abs(det) ** 0.5, 1.0)     # one texel of overlap: no fringe between two triangles
        m = (l0 >= marge) & (l1 >= marge) & (l2 >= marge)
        if not m.any():
            continue
        c = np.clip(l0[m], 0, 1)[:, None] * delta[fg[0]] + np.clip(l1[m], 0, 1)[:, None] * delta[fg[1]] + np.clip(l2[m], 0, 1)[:, None] * delta[fg[2]]
        correction[y0:y1, x0:x1][m] = c
        couvert[y0:y1, x0:x1] |= m
    res = img_peau.copy()
    res[..., :3] = np.clip(res[..., :3] + correction, 0, 1)
    ecrire_image(os.path.join(dossier, textures[materiaux[i_peau]["texture"]].replace("\\", "/")), res)
    return {"texels": int(couvert.sum()), "delta_median_srgb": [round(float(x) * 255, 1) for x in np.median(val[list(vus)], 0)]}


def rentrer_crane(s, faces, mat_face, i_peau, i_visage, soudure=(), pas=PAS_SILHOUETTE_MM * MM, cou=COU_SOUS_MENTON_MM * MM):
    """The body skin does not stick out, seen from the front, beyond the face silhouette; returns the number of vertices pulled in.

    The scalp of a body from `peau_complete.py` is wider than the face: from the front, it surrounds it from the
    eyes to the chin with a band of skin that extends the jaw in a long V down to the neck (long pointed chin, no
    jaw corner). On the game models, that space belongs to the back hair, and the neck joins the jaw only from
    below. At each face height, each side is bounded by the face edge on that side, without going below the
    neck half-width measured `cou` under the chin. The weld does not move.
    """
    pos = np.asarray(s["pos"], float)
    v = np.unique(faces[np.isin(mat_face, list(i_visage))])
    vp = np.setdiff1d(np.unique(faces[mat_face == i_peau]), np.asarray(list(soudure), int))
    if not len(v) or not len(vp):
        return 0
    xc = pos[v, 0].mean()
    yc, yh = pos[v, 1].min(), pos[v, 1].max()
    bins = np.arange(yc, yh + pas, pas)
    dx = pos[v, 0] - xc
    tranche = [np.abs(pos[v, 1] - b) < pas for b in bins]
    gauche = np.array([(-dx[m]).max(initial=0.0) for m in tranche])
    droite = np.array([dx[m].max(initial=0.0) for m in tranche])
    q = pos[vp]
    col = q[(np.abs(q[:, 1] - (yc - cou)) < pas) & (q[:, 2] > pos[v, 2].min())]
    hw_cou = np.abs(col[:, 0] - xc).max() if len(col) else 0.0
    # Snapping each vertex onto the limit flattened the scalp into a wall with `pas` steps along the face
    # edge (jagged ridge on the cheek, in 3/4 view): instead the scalp tightens by a factor per height and
    # per side, smoothed over height, which fades out near the weld
    dans = vp[(q[:, 1] >= yc - pas) & (q[:, 1] <= yh + pas)]
    if not len(dans):
        return 0
    noyau = np.hanning(7)[1:-1]
    noyau /= noyau.sum()
    sou = np.asarray(list(soudure), int)
    d_sou = np.sqrt(((pos[dans][:, None] - pos[sou][None]) ** 2).sum(2)).min(1) if len(sou) else np.full(len(dans), np.inf)
    n = 0
    for cote, lim_v in ((1, droite), (-1, gauche)):
        lim = np.maximum(lim_v, hw_cou)
        d = cote * (pos[dans, 0] - xc)
        b = np.clip(np.round((pos[dans, 1] - yc) / pas).astype(int), 0, len(bins) - 1)
        large = np.zeros(len(bins))
        np.maximum.at(large, b[d > 0], d[d > 0])
        f = np.where(large > lim, lim / np.maximum(large, 1e-12), 1.0)
        f = np.convolve(np.pad(f, 2, mode="edge"), noyau, mode="valid")
        fi = np.interp(pos[dans, 1], bins, f)
        w = np.clip(d_sou / RAYON_CRANE_MM / MM, 0, 1)
        w = w * w * (3 - 2 * w)
        k = (d > 0) & (fi < 1)
        x = xc + (pos[dans, 0] - xc) * (1 - w * (1 - fi))
        for i, xi in zip(dans[k].tolist(), x[k].tolist()):
            s["pos"][i] = (float(xi), float(pos[i, 1]), float(pos[i, 2]))
        n += int(k.sum())
    return n


def enfoncer_sous_visage(s, faces, mat_face, i_peau, i_visage, soudure=(), marge=SOUS_VISAGE_MM * MM,
                         portee=PORTEE_SOUS_MM * MM, bord=BORD_SOUS_MM * MM):
    """The body skin hidden by the face goes down to `marge` beneath it; returns the number of vertices pushed in.

    The join leaves skin vertices under the face flush with its surface, sometimes above it: they poke
    through it (spots of body tint), and the body's outline shell, inflated by 1 to 2 mm, draws dark dashes
    there along the cheek edge: the skin rising towards the weld passes over the face before joining it.
    Each vertex whose nearest point lies in the face goes down, along that face's normal (or, just past
    the edge, under the extension of the nearest face), to a depth growing from 0 on the face outline to
    `marge` at `bord` from it.
    """
    pos = np.asarray(s["pos"], float)
    nor = np.asarray(s["nor"], float)
    Fv = faces[np.isin(mat_face, list(i_visage))]
    vp = np.setdiff1d(np.unique(faces[mat_face == i_peau]), np.asarray(list(soudure), int))
    if not len(Fv) or not len(vp):
        return 0
    sou = globals()["soudure"](pos)  # the `soudure` parameter shadows the function
    tour = np.unique(bord_exterieur(aretes_de_bord(Fv, sou), sou, pos))
    T = pos[Fv]
    lo, hi = T.reshape(-1, 3).min(0) - portee, T.reshape(-1, 3).max(0) + portee
    vp = vp[((pos[vp] > lo) & (pos[vp] < hi)).all(1)]
    if not len(vp) or not len(tour):
        return 0
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n *= np.sign((n * nor[Fv].sum(1)).sum(1))[:, None] + ((n * nor[Fv].sum(1)).sum(1) == 0)[:, None]
    c = T.mean(1)
    k = 0
    for i in vp.tolist():
        q = pos[i]
        j = np.argsort(((c - q) ** 2).sum(1))[:6]
        # nearest point on each of the candidate faces
        a, b, cc = T[j, 0], T[j, 1], T[j, 2]
        sd = ((q - a) * n[j]).sum(1)
        p = q - sd[:, None] * n[j]
        e0, e1, r = b - a, cc - a, p - a
        d00, d01, d11 = (e0 * e0).sum(1), (e0 * e1).sum(1), (e1 * e1).sum(1)
        d20, d21 = (r * e0).sum(1), (r * e1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        dedans = (v >= 0) & (w >= 0) & (v + w <= 1)
        # outside the face, just past its edge, the skin must pass under the extension of its surface
        m = np.where(dedans, np.abs(sd), np.inf).argmin() if dedans.any() else 0
        d_tour = np.sqrt(((pos[tour] - q) ** 2).sum(1)).min()
        if not dedans.any() and d_tour > portee:
            continue
        # target depth: zero on the outline (the skin is welded there), `marge` at `bord` from it
        prof = marge * min(d_tour / bord, 1.0)
        if abs(sd[m]) > portee or sd[m] <= -prof:
            continue
        s["pos"][i] = tuple(float(x) for x in q + (-prof - sd[m]) * n[j[m]])
        k += 1
    return k


def amincir_menton(s, faces, mat_face, i_visage, soudure=(), bas=TRAIT_MENTON_BAS, hauteur=TRAIT_MENTON_HAUTEUR):
    """Outline weight of the lower face: 1 at mouth height, `bas` at the tip of the chin.

    At full thickness, the face's outline shell runs along the whole jaw and lies on the bare neck:
    a long black V that lengthens the chin. The mouth is at `hauteur` of the face height above the chin (0.15 Maëlys, 0.16 Grace).
    Skin vertices welded to the face edge take the same weight. Returns the number of thinned vertices.
    """
    pos = np.asarray(s["pos"], float)
    v = np.unique(faces[np.isin(mat_face, list(i_visage))])
    if not len(v):
        return 0
    yc, yh = pos[v, 1].min(), pos[v, 1].max()
    yb = yc + hauteur * (yh - yc)
    c = np.asarray(s["contour"], float)
    for w in np.concatenate([v, np.asarray(list(soudure), int)]).tolist():
        t = min(max((pos[w, 1] - yc) / (yb - yc), 0.0), 1.0)
        c[w] = min(c[w], bas + (1 - bas) * t * t * (3 - 2 * t))
    s["contour"] = [float(x) for x in c]
    return int((c[v] < 0.999).sum())


def appliquer(s, faces, mat_face, materiaux, morphes, i_peau, i_visage, dossier=None, textures=None):
    """Apply the join to the vertex table `s`; returns (kept faces, summary).

    With `dossier` and `textures` (paths relative to `dossier`), the skin texture is matched there as well.
    """
    os_ = np.asarray(s["os"], np.int32)
    poids = np.asarray(s["poids"], float)
    r = raccord_cou(np.array(s["pos"], float), np.array(s["nor"], float), faces, mat_face, i_peau, i_visage, os_, poids)
    if not r["soudure"]:
        r["bilan"]["trait_menton"] = amincir_menton(s, faces, mat_face, i_visage)
        return faces, r["bilan"]
    change = np.where(np.any(r["pos"] != np.array(s["pos"], float), axis=1) | np.any(r["nor"] != np.array(s["nor"], float), axis=1))[0]
    for v in change.tolist():
        s["pos"][v] = tuple(float(x) for x in r["pos"][v])
        s["nor"][v] = tuple(float(x) for x in r["nor"][v])
    for v, (ty, o, w) in r["poids"].items():
        s["type"][v], s["os"][v], s["poids"][v], s["sdef"][v] = ty, o, w, None
    # morphs that move the face edge carry the weld along
    for m in morphes:
        if m["type"] != 1:
            continue
        dec = dict((int(i), np.asarray(d, float)) for i, d in m["decalages"])
        ajouts = []
        for v, (a, b, t) in r["soudure"].items():
            if a in dec or b in dec:
                d = (1 - t) * dec.get(a, np.zeros(3)) + t * dec.get(b, np.zeros(3))
                if np.abs(d).max() > 1e-7:
                    ajouts.append((v, tuple(float(x) for x in d)))
        if ajouts:
            m["decalages"] = list(m["decalages"]) + ajouts
            r["bilan"]["morphes_suivis"] = r["bilan"].get("morphes_suivis", 0) + 1
    g = r["garde"]
    r["bilan"]["crane_rentre"] = rentrer_crane(s, faces[g], mat_face[g], i_peau, i_visage, r["soudure"])
    r["bilan"]["sous_visage"] = enfoncer_sous_visage(s, faces[g], mat_face[g], i_peau, i_visage, r["soudure"])
    r["bilan"]["trait_menton"] = amincir_menton(s, faces, mat_face, i_visage, r["soudure"])
    if dossier is not None:
        mat_sommet = np.full(len(s["pos"]), -1)
        mat_sommet[faces.ravel()] = np.repeat(mat_face, 3)
        r["bilan"]["teinte"] = peindre_teinte(dossier, textures, materiaux, np.array(s["uv"], float), mat_sommet, i_peau, r)
    garde = r["garde"]
    fm = mat_face[garde]
    for i, m in enumerate(materiaux):
        m["n_indices"] = int(3 * (fm == i).sum())
    return faces[garde], r["bilan"]


def args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pmx", required=True)
    p.add_argument("--sortie", required=True)
    p.add_argument("--peau", default="肌")
    p.add_argument("--visage", default="颜,颜_A")
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    return p.parse_args(argv)


def recopier_textures(R, dossier_pmx, sortie):
    """Copy the model's textures into `sortie`; those stored outside the PMX folder go into tex/."""
    os.makedirs(sortie, exist_ok=True)
    for i, t in enumerate(R["textures"]):
        src = os.path.normpath(os.path.join(dossier_pmx, t.replace("\\", "/")))
        if not os.path.isfile(src):
            continue
        rel = os.path.relpath(src, dossier_pmx)
        if rel.startswith(".."):
            rel = os.path.join("tex", "%d_%s" % (i, os.path.basename(src)))
        dst = os.path.join(sortie, rel)
        os.makedirs(os.path.dirname(dst) or sortie, exist_ok=True)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        R["textures"][i] = rel.replace(os.sep, "\\")


def main(argv=None):
    a = args(argv)
    R = PMX.lire(a.pmx)
    t = PMX.vers_tableaux(R)
    noms = [m["nom"] for m in R["materiaux"]]
    i_vis = [noms.index(n) for n in a.visage.split(",") if n in noms]
    recopier_textures(R, os.path.dirname(os.path.abspath(a.pmx)), a.sortie)
    F, bilan = appliquer(R["sommets"], t["faces"], t["mat_face"], R["materiaux"], R["morphes"], noms.index(a.peau), i_vis,
                         dossier=a.sortie, textures=R["textures"])
    R["faces"] = F.tolist()
    chemin = os.path.join(a.sortie, "resultat.pmx")
    pmx_ecrire.ecrire(R, chemin + ".tmp")
    os.replace(chemin + ".tmp", chemin)
    with open(os.path.join(a.sortie, "raccord.json"), "w", encoding="utf-8") as f:
        json.dump(bilan, f, ensure_ascii=False, indent=1)
    print("neck join:", bilan, "->", chemin)


if __name__ == "__main__":
    main()

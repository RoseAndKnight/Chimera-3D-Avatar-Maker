"""Body skin cut as on a game model: a neck under the face, no scalp.

Why: on a ZZZ model, the body skin stops at the top of the neck, hidden under the face and the hair
cap. The outfit's body comes from `peau_complete.py`, whose skin is closed by a scalp sphere; it
poked out around the grafted face and through the hair. That sphere goes entirely (skin piece lying
wholly above the neck), and so does the body skin above the top of the donor head's neck, never lower
than `SOUS_VISAGE_MM` under the face.

A single join, at the face edge, with no seam on the neck: the neck stays the body's, in one piece,
with its weights. The head brings only its ears (`ile_du_cou`), which follow the head alone, and its
scalp skin (`crane_du_jeu`), which closes the head under the hair; the bottom of that scalp slides
under the body's neck (`_poser_nuque`), with the neck's weights where it touches it. A seam between
the head's neck and the body's would open when the head turns.

The body's neck is never touched. It is the skin vertices that the head (bone 頭 and its children)
carries at less than `COU_POIDS`: neither they nor any face touching one of them move, go away or
lose their outline. The cut only takes from the body's head. `cou_intact.py` (atelier) proves it on
what the page draws, vertex by vertex, face by face and pixel by pixel.
"""
import numpy as np

MM = 1.0 / 80.0
SECTEURS = 36
ZONE_MM = 80.0      # only skin above the bottom of the face minus this margin is considered
RAYON_MM = 140.0    # and within this distance of the neck axis (not the shoulders)
SANS_COU_MM = 30.0  # donor head without a neck: cut at this height above the bottom of the face
SANS_TRAIT_MM = 10.0  # the body's head within this distance of the face draws no outline
SOUS_VISAGE_MM = 12.0  # the body rises at least to this height below the bottom of the face, direction by direction
TETE_SEULE = 0.98   # minimum weight share on the head bone (and its children) for a taken ear vertex
COU_POIDS = 0.5     # a vertex the head carries at less than this belongs to the neck: left untouched
SEMIS_MM = 2.0      # the original neck is measured over its surface, at least one point every this much
AFFLEURE_MM = 3.0   # body head sticking out of the face by less than this is duplicate skin
RENTRE_MM = 0.5     # it goes back under the face by this much
NUQUE_MM = 15.0     # height of the head's lower scalp strip laid on the body's neck
LISSE_MM = 6.0      # the strip follows the edge vertices within this much
PROCHE_MM = 20.0    # a lower scalp edge is laid only if the body skin passes within this much
RECOUVRE_MM = 8.0   # above the laid edge, the body keeps this much skin under the scalp
DANS_VISAGE_MM = 20.0  # scalp skin up to this much under the face is a hidden duplicate


def _secteur(p, c):
    a = np.arctan2(p[:, 2] - c[1], p[:, 0] - c[0])
    return np.floor((a + np.pi) / (2 * np.pi) * SECTEURS).astype(int) % SECTEURS


def surface(pos, faces):
    """Points on the surface of the triangles `faces`, at most `SEMIS_MM` apart, vertices included.

    A donor head's neck has only a few vertex rings: measured at its vertices alone, most cells
    (5 mm slice × 10° sector) stayed empty or saw a single vertex, and the target radius wobbled
    from one body ring to the next (wrinkle under Nicole's jaw)."""
    pos = np.asarray(pos, float)
    faces = np.asarray(faces, int).reshape(-1, 3)
    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    cote = np.max([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], axis=0)
    n = np.maximum(1, np.ceil(cote / (SEMIS_MM * MM)).astype(int))
    points = [pos[np.unique(faces)]]
    for k in np.unique(n):
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1))
        u, v = (i[i + j <= k] / k)[None, :, None], (j[i + j <= k] / k)[None, :, None]
        m = n == k
        points.append((a[m, None] * (1 - u - v) + b[m, None] * u + c[m, None] * v).reshape(-1, 3))
    return np.concatenate(points)


def _composantes(pos, F):
    """Connected-component label of each face (vertices welded by position)."""
    if not len(F):
        return np.zeros(0, int)
    _, rep = np.unique(np.round(pos, 5), axis=0, return_inverse=True)
    rep = rep.ravel()
    par = np.arange(rep.max() + 1)

    def racine(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a, b, c in rep[F].tolist():
        ra = racine(a)
        for x in (b, c):
            rx = racine(x)
            if rx != ra:
                par[rx] = ra
    return np.array([racine(a) for a in rep[F[:, 0]].tolist()])


def _ilots(F):
    """Label of each face: piece connected through vertex indices (a UV island)."""
    if not len(F):
        return np.zeros(0, int)
    par = np.arange(F.max() + 1)

    def racine(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a, b, c in F.tolist():
        ra = racine(a)
        for x in (b, c):
            rx = racine(x)
            if rx != ra:
                par[rx] = ra
    return np.array([racine(a) for a in F[:, 0].tolist()])


def os_de_tete(os_pmx, i_tete):
    """Indices of the head bone and all its descendants."""
    pris = {i_tete}
    for _ in range(len(os_pmx)):
        n = len(pris)
        pris |= {i for i, o in enumerate(os_pmx) if o["parent"] in pris}
        if len(pris) == n:
            break
    return np.array(sorted(pris))




def ile_du_cou(pos, faces, mat_face, i_peau, i_visage, axe, os_s=None, poids=None, tete=None, uv=None, img=None):
    """Donor head skin faces to graft along with the face (the ears): in the first candidate
    material surrounding the head, the pieces welded to the face and the isolated ears, without the
    black cap faces, and only above the bottom of the face, where all their vertices follow the head
    alone (`TETE_SEULE`): nothing there is stitched to the body's neck.

    Only UV islands taken whole remain. Survey of 09/11 on the 16 cached heads: the ears are always
    whole islands; the rest taken was, on each side, a corner of scalp skin behind the jaw, cut into
    a sawtooth by the weight and height filters (22 faces of 15 to 25 mm on Maëlys). Its points hung
    below the ear, through the hair."""
    pos = np.asarray(pos, float)
    pris = np.zeros(len(faces), bool)
    fv = faces[np.isin(mat_face, i_visage)]
    if not len(fv):
        return pris
    vv = np.unique(fv)
    y_bas = pos[vv, 1].min()
    y0 = y_bas - ZONE_MM * MM
    c = np.asarray(axe, float)
    dans = (pos[:, 1] > y0) & (np.hypot(pos[:, 0] - c[0], pos[:, 2] - c[1]) < RAYON_MM * MM)
    cle_v = {tuple(x) for x in np.round(pos[vv] / 1e-4).astype(np.int64)}
    touche_v = np.array([tuple(x) in cle_v for x in np.round(pos / 1e-4).astype(np.int64)])
    part = (np.asarray(poids, float) * np.isin(os_s, tete)).sum(1) if os_s is not None else np.ones(len(pos))
    cands = [int(i_peau)] if isinstance(i_peau, (int, np.integer)) else [int(x) for x in i_peau]
    for im in cands:
        fp = np.where(mat_face == im)[0]
        if not len(fp):
            continue
        F = faces[fp]
        comp = _composantes(pos, F)
        hors = np.unique(comp[~dans[F].all(1)])
        garde = ~np.isin(comp, hors)
        if not garde.any():
            # one-piece body (e.g. Pulchra): cut above the choker
            haut = dans[F].all(1) & (pos[F, 1] > y_bas - 30.0 * MM).all(1)
            if not haut.any():
                continue
            comp_h = _composantes(pos, F[haut])
            tc = np.unique(comp_h[touche_v[F[haut]].any(1)])
            garde = np.zeros(len(fp), bool)
            garde[haut] = np.isin(comp_h, tc)
        else:
            oreille = ((np.abs(pos[F, 0] - c[0]) > 35.0 * MM).any(1)
                       & (pos[F, 1] > y_bas).all(1) & (part[F] >= TETE_SEULE).all(1))
            bons = np.unique(comp[garde & (touche_v[F].any(1) | oreille)])
            garde &= np.isin(comp, bons)
            # black inner cap under the hair at the back of the skull (e.g. Miyabi)
            calotte = ((part[F] >= TETE_SEULE).all(1) & (pos[F, 2] > c[1] + 22.0 * MM).all(1)
                       & (np.abs(pos[F, 0] - c[0]) < 42.0 * MM).all(1) & ~touche_v[F].any(1))
            garde &= ~calotte
        garde &= (part[F] >= TETE_SEULE).all(1) & (pos[F, 1] > y_bas).all(1)
        if img is not None and uv is not None and garde.any():
            h, w = img.shape[:2]
            u = np.mod(np.asarray(uv, float)[F].mean(1), 1.0)
            px = img[(u[:, 1] * h).astype(int).clip(0, h - 1), (u[:, 0] * w).astype(int).clip(0, w - 1), :3]
            noir = (px.max(1) < 0.28) & ((px.max(1) - px.min(1)) < 0.08)
            garde &= ~noir
        if garde.any():
            ilot = _ilots(F)
            coupes = np.unique(ilot[~garde])
            pris[fp[garde & ~np.isin(ilot, coupes)]] = True
            break
    return pris


def _sous_le_menton(pos, Fv, q):
    """Points of `q` that a triangle of `Fv` covers vertically, lying entirely above them."""
    a, b, c = (pos[Fv[:, k]][:, [0, 2]] for k in range(3))
    y_min = pos[Fv, 1].min(1)
    v0, v1 = c - a, b - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den[np.abs(den) < 1e-18] = 1e-18
    out = np.zeros(len(q), bool)
    for k, p in enumerate(q):
        v2 = p[[0, 2]] - a
        d02, d12 = (v0 * v2).sum(1), (v1 * v2).sum(1)
        u, v = (d11 * d02 - d01 * d12) / den, (d00 * d12 - d01 * d02) / den
        out[k] = ((u >= 0) & (v >= 0) & (u + v <= 1) & (y_min > p[1])).any()
    return out


def crane_du_jeu(pos, nor, faces, mat_face, i_peau, i_visage, axe, os_s, poids, tete):
    """Scalp skin faces of the donor head (back, top, sides behind the jaw): in the first candidate
    material that has any, those whose vertices all follow the head at `COU_POIDS` or more, except
    under the chin and under the face (up to `DANS_VISAGE_MM` inside: a hidden duplicate).

    The body has lost its scalp sphere (`couper`) and `ile_du_cou` only takes the ears: the head
    stayed open at the nape, and under the hair the back side of the face showed (survey of 10/11 on
    Pulchra and Grace, wearing their own hair). A game model closes its head with this very skin, to
    which its hair is fitted; its bottom is then laid on the body's neck (`_poser_nuque`)."""
    pos, nor = np.asarray(pos, float), np.asarray(nor, float)
    pris = np.zeros(len(faces), bool)
    Fv = faces[np.isin(mat_face, i_visage)]
    if not len(Fv):
        return pris
    y0 = pos[np.unique(Fv), 1].min() - ZONE_MM * MM
    c = np.asarray(axe, float)
    dans = (pos[:, 1] > y0) & (np.hypot(pos[:, 0] - c[0], pos[:, 2] - c[1]) < RAYON_MM * MM)
    part = (np.asarray(poids, float) * np.isin(os_s, tete)).sum(1)
    cands = [int(i_peau)] if isinstance(i_peau, (int, np.integer)) else [int(x) for x in i_peau]
    for im in cands:
        fp = np.where(mat_face == im)[0]
        F = faces[fp]
        sel = (part[F] >= COU_POIDS).all(1) & dans[F].all(1)
        if not sel.any():
            continue
        k = np.where(sel)[0]
        sel[k[_sous_le_menton(pos, Fv, pos[F[k]].mean(1))]] = False
        k = np.where(sel)[0]
        d, _ = _sous_le_visage(pos, nor, Fv, pos[F[k]].mean(1))
        sel[k[(d < 0) & (d > -DANS_VISAGE_MM * MM)]] = False
        pris[fp[sel]] = True
        break
    return pris


def _lisser_secteurs(val, vus):
    k = np.arange(SECTEURS)
    out = np.interp(k, np.concatenate([k[vus] - SECTEURS, k[vus], k[vus] + SECTEURS]), np.tile(val[vus], 3))
    noyau = np.hanning(7)[1:-1]
    return np.convolve(np.concatenate([out[-2:], out, out[:2]]), noyau / noyau.sum(), mode="valid")



def _crane_du_corps(pos, faces, mat_face, i_peau, y0):
    """The scalp sphere of peau_complete.py: any body skin piece lying entirely above y0."""
    fp_idx = np.where(mat_face == i_peau)[0]
    est_crane = np.zeros(len(faces), bool)
    if len(fp_idx):
        comp_p = _composantes(pos, faces[fp_idx])
        garder_c = np.unique(comp_p[~(pos[faces[fp_idx], 1] > y0).all(1)])
        est_crane[fp_idx[~np.isin(comp_p, garder_c)]] = True
    return est_crane


def couper(pos, faces, mat_face, i_peau, i_visage, cou_tete, axe, fige=None):
    """Body skin faces to remove (mask over `faces`) and summary.

    pos: vertices (n, 3); cou_tete: neck vertices of the donor head, placed (m, 3);
    axe: (x, z) of the neck axis (bone 頭); fige: vertices never cut or moved (the neck and its faces).
    """
    pos = np.asarray(pos, float)
    fv = faces[np.isin(mat_face, i_visage)]
    y_bas = pos[np.unique(fv), 1].min()
    y0 = y_bas - ZONE_MM * MM
    c = np.asarray(axe, float)
    est_crane = _crane_du_corps(pos, faces, mat_face, i_peau, y0)
    cou = np.asarray(cou_tete, float).reshape(-1, 3)
    cou = cou[(cou[:, 1] > y0) & (np.hypot(cou[:, 0] - c[0], cou[:, 2] - c[1]) < RAYON_MM * MM)]
    haut = np.full(SECTEURS, -np.inf)
    if len(cou):
        np.maximum.at(haut, _secteur(cou, c), cou[:, 1])
    vus = np.isfinite(haut)
    if vus.sum() < SECTEURS // 3:
        haut[:] = y_bas + SANS_COU_MM * MM
        source = "default"
    else:
        haut = _lisser_secteurs(haut, vus)
        source = "head neck"
    # where the donor head's neck stopped under its chin, the cut edge showed as a V under the jaw:
    # the body always rises under the face
    qv = pos[np.unique(fv)]
    bas_v = np.full(SECTEURS, np.inf)
    np.minimum.at(bas_v, _secteur(qv, c), qv[:, 1] + SOUS_VISAGE_MM * MM)
    haut = np.where(np.isfinite(bas_v), np.maximum(haut, bas_v), haut)
    fp = (mat_face == i_peau) & ~est_crane
    vp = np.unique(faces[fp])
    q = pos[vp]
    pres = (q[:, 1] > y0) & (np.hypot(q[:, 0] - c[0], q[:, 2] - c[1]) < RAYON_MM * MM)
    lim = np.full(len(pos), np.inf)
    lim[vp[pres]] = _hauteur(haut, q[pres], c)
    if fige is not None:
        lim[fige] = np.inf
    trop = pos[:, 1] > lim
    # a straddling face stays, its upper vertices are brought down to the cut height: clean edge
    retirer = est_crane | (fp & trop[faces].all(1))
    bord = np.unique(faces[fp & ~retirer][trop[faces[fp & ~retirer]]])
    bilan = {"faces_retirees": int(retirer.sum()), "faces_crane": int(est_crane.sum()),
             "sommets_bord": int(len(bord)), "coupe": source,
             "haut_du_cou_mm": [round(float((h - y_bas) / MM), 1) for h in (haut.min(), np.median(haut), haut.max())]}
    return retirer, bord, lim, bilan


def _sur_la_coupe(pos, Fp, bord, lim):
    """Where each vertex of `bord` goes down to: the surface point at the cut height, on the most vertical
    edge linking it to a neighbour below the cut.

    Moved straight down, a nape vertex (which slants) ended up to 3 mm under the surface, next to the
    ring below: steps in the side silhouette. Copies of a vertex (UV seams) go down together; with no
    edge crossing the cut, it goes straight down.

    Without a hairstyle, the side outline stays dashed at the nape: the shell has nothing behind an open
    edge. That edge is under the hair, as on a game model (survey of 10/11 on four short cuts,
    Maëlys, Anby, Corin, Seth: hidden)."""
    pos = np.asarray(pos, float)
    _, rep = np.unique(np.round(pos, 5), axis=0, return_inverse=True)
    rep = rep.ravel()
    trop = pos[:, 1] > lim
    E = np.concatenate([Fp[:, [0, 1]], Fp[:, [1, 2]], Fp[:, [2, 0]]])
    E = np.concatenate([E, E[:, ::-1]])
    E = E[trop[E[:, 0]] & (pos[E[:, 1], 1] <= lim[E[:, 0]])]
    a, b, h = pos[E[:, 0]], pos[E[:, 1]], lim[E[:, 0]]
    chute = np.maximum(a[:, 1] - b[:, 1], 1e-12)
    pt = a + ((a[:, 1] - h) / chute)[:, None] * (b - a)
    pente = np.hypot(b[:, 0] - a[:, 0], b[:, 2] - a[:, 2]) / chute
    ordre = np.lexsort((pente, rep[E[:, 0]]))
    r_e = rep[E[ordre, 0]]
    premier = np.r_[True, r_e[1:] != r_e[:-1]]
    choisi = dict(zip(r_e[premier].tolist(), pt[ordre][premier]))
    return [choisi.get(int(rep[v]), np.array([pos[v, 0], lim[v], pos[v, 2]])) for v in bord.tolist()]


def _hauteur(haut, q, c):
    """Cut height interpolated between sectors, at the azimuth of each point."""
    a = (np.arctan2(q[:, 2] - c[1], q[:, 0] - c[0]) + np.pi) / (2 * np.pi) * SECTEURS - 0.5
    k = np.arange(SECTEURS)
    return np.interp(a % SECTEURS, np.concatenate([k - SECTEURS, k, k + SECTEURS]), np.tile(haut, 3))



def _sous_le_visage(pos, nor, Fv, q):
    """Signed distance (positive outside) from each point of `q` to the skin `Fv`, and the outward normal at the foot;
    nan when no triangle receives it vertically (at the face edge, the sign means nothing)."""
    a, b, c = pos[Fv[:, 0]], pos[Fv[:, 1]], pos[Fv[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    n *= np.sign((nor[Fv].sum(1) * n).sum(1) + 1e-12)[:, None]
    e0, e1 = b - a, c - a
    d00, d01, d11 = (e0 * e0).sum(1), (e0 * e1).sum(1), (e1 * e1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
    dist = np.full(len(q), np.nan)
    nn = np.zeros((len(q), 3))
    for k, p in enumerate(q):
        h = ((p - a) * n).sum(1)
        w = p - h[:, None] * n - a
        d20, d21 = (w * e0).sum(1), (w * e1).sum(1)
        v, u = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        ok = (v >= 0) & (u >= 0) & (u + v <= 1)
        if ok.any():
            j = np.where(ok)[0][np.abs(h[ok]).argmin()]
            dist[k], nn[k] = h[j], n[j]
    return dist, nn


def _rentrer(pos, nor, faces, fm, i_peau, i_vis, fige):
    """The kept body head that lies flush with the face goes under it.

    The cut keeps the body up to `SOUS_VISAGE_MM` under the bottom of the face: otherwise, a gap under the face.
    In that strip, the body skin stayed at the back of the head 0.4-1.3 mm in front of the face, the two
    skins crossing (survey of 10/11 on the sixteen heads: 2 to 16 faces each). Returns (positions, summary);
    the frozen neck does not move. The faces moved under remain: on Nicole, 36 of the 53 are still visible
    from outside between the face edge and the skin (rays, hair on, 10/11)."""
    Fv = faces[np.isin(fm, i_vis)]
    vp = np.unique(faces[fm == i_peau])
    vp = vp[~fige[vp]]
    pv = pos[np.unique(Fv)]
    lo, hi = pv.min(0) - AFFLEURE_MM * MM, pv.max(0) + AFFLEURE_MM * MM
    vp = vp[((pos[vp] > lo) & (pos[vp] < hi)).all(1)]
    d, n = _sous_le_visage(pos, nor, Fv, pos[vp])
    pos = pos.copy()
    bouge = (d > -RENTRE_MM * MM) & (d < AFFLEURE_MM * MM)
    pos[vp[bouge]] -= ((d[bouge] + RENTRE_MM * MM)[:, None]) * n[bouge]
    return pos, {"sommets_rentres": int(bouge.sum())}


def _unifier_normales(s, faces, fm, i_visage):
    """A single normal on both sides of the face's internal seam (颜 / 颜_A)."""
    if len(i_visage) < 2:
        return
    pos = np.asarray(s["pos"], float)
    nor = np.asarray(s["nor"], float)
    v0 = np.unique(faces[fm == i_visage[0]])
    v1 = np.unique(faces[fm == i_visage[1]])
    cle = lambda v: [tuple(x) for x in np.round(pos[v] / 1e-4).astype(np.int64)]
    m1 = {k: j for k, j in zip(cle(v1), v1)}
    for a, k in zip(v0.tolist(), cle(v0)):
        if k in m1:
            b = m1[k]
            nm = nor[a] + nor[b]
            nm /= max(float(np.linalg.norm(nm)), 1e-12)
            s["nor"][a] = s["nor"][b] = tuple(float(x) for x in nm)


def _soudes(pos):
    """Shared label for the copies of a same vertex (UV and material seams)."""
    _, rep = np.unique(np.round(np.asarray(pos, float) / 1e-4).astype(np.int64), axis=0, return_inverse=True)
    return rep.ravel()


def _aretes(rep, F):
    return np.sort(rep[np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])], axis=1)


def _peau_pres(pos, Fb, Fi):
    """Faces of `Fb` that have a vertex near the bounding box of faces `Fi`."""
    vi = np.unique(Fi)
    marge = (PROCHE_MM + NUQUE_MM) * MM
    lo, hi = pos[vi].min(0) - marge, pos[vi].max(0) + marge
    return Fb[((pos[Fb] > lo) & (pos[Fb] < hi)).all(2).any(1)]


def _point_triangle(p, a, b, c):
    """Point of each triangle (a, b, c) closest to p (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac = b - a, c - a
    d1, d2 = (ab * (p - a)).sum(1), (ac * (p - a)).sum(1)
    d3, d4 = (ab * (p - b)).sum(1), (ac * (p - b)).sum(1)
    d5, d6 = (ab * (p - c)).sum(1), (ac * (p - c)).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        r = a + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]
        for ok, x in (((va <= 0) & (d4 >= d3) & (d5 >= d6), b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]),
                      ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * (d2 / (d2 - d6))[:, None]),
                      ((d6 >= 0) & (d5 <= d6), c),
                      ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * (d1 / (d1 - d3))[:, None]),
                      ((d3 >= 0) & (d4 <= d3), b),
                      ((d1 <= 0) & (d2 <= 0), a)):
            r = np.where(ok[:, None], x, r)
    return np.where(np.isfinite(r), r, a)


def _distance_au_corps(pos, nor, Fb, q):
    """Signed distance (positive outside) from each point of `q` to the closest point of the skin `Fb`,
    outward normal of the triangle carrying it, and that point. The vertical projection of `_sous_le_visage`
    sometimes picked a triangle on the other side of the neck (60 mm inside, survey of 10/11 on Grace, Anby, ZhuYuan)."""
    a, b, c = pos[Fb[:, 0]], pos[Fb[:, 1]], pos[Fb[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    n *= np.sign((nor[Fb].sum(1) * n).sum(1) + 1e-12)[:, None]
    d, nn, pp = np.zeros(len(q)), np.zeros((len(q), 3)), np.zeros((len(q), 3))
    for k, p in enumerate(q):
        r = _point_triangle(p, a, b, c)
        j = int(np.argmin(((r - p) ** 2).sum(1)))
        pp[k], nn[k] = r[j], n[j]
        d[k] = np.linalg.norm(p - r[j]) * (1.0 if float((p - r[j]) @ n[j]) >= 0 else -1.0)
    return d, nn, pp


def _bord_de_nuque(pos, nor, faces, fm, peaux, Fb):
    """Free edge of the head skins `peaux` taken together (without the edges of another part): its lower
    welded vertices within `PROCHE_MM` of the body skin `Fb`, the others, the label `rep` and one vertex
    per label."""
    rep = _soudes(pos)
    m = rep.max() + 1
    premier = np.zeros(m, int)
    premier[rep[::-1]] = np.arange(len(rep))[::-1]
    Fi = faces[np.isin(fm, peaux)]
    u, n = np.unique(_aretes(rep, Fi), axis=0, return_counts=True)
    b = u[n == 1]
    Fo = faces[~np.isin(fm, peaux)]
    if len(Fo):
        Eo = _aretes(rep, Fo)
        b = b[~np.isin(b[:, 0] * m + b[:, 1], Eo[:, 0] * m + Eo[:, 1])]
    vb = np.unique(b)
    if not len(vb) or not len(Fb):
        return vb[:0], vb, rep, premier
    # lower edge (under the jaw, sides, nape): the taken skin is not below the edge; the hairline edge,
    # with skin below it (pointing down by at least 0.5 on four heads, 10/11), also passes near the
    # body at the nape (16 mm on Grace) and does not move
    A = rep[np.concatenate([Fi[:, [0, 1]], Fi[:, [1, 2]], Fi[:, [2, 0]]])]
    t3 = rep[np.concatenate([Fi[:, 2], Fi[:, 0], Fi[:, 1]])]
    As = np.sort(A, axis=1)
    o = np.argsort(As[:, 0] * m + As[:, 1])
    k = o[np.searchsorted((As[:, 0] * m + As[:, 1])[o], b[:, 0] * m + b[:, 1])]
    P = pos[premier]
    e = P[b[:, 1]] - P[b[:, 0]]
    e /= np.maximum(np.linalg.norm(e, axis=1), 1e-12)[:, None]
    dedans = P[t3[k]] - 0.5 * (P[b[:, 0]] + P[b[:, 1]])
    dedans -= (dedans * e).sum(1)[:, None] * e
    dedans /= np.maximum(np.linalg.norm(dedans, axis=1), 1e-12)[:, None]
    bas = np.isin(vb, b[dedans[:, 1] > -0.4])
    d, _, _ = _distance_au_corps(pos, nor, Fb, P[vb])
    proche = bas & (np.abs(d) < PROCHE_MM * MM)
    return vb[proche], vb[~proche], rep, premier


def _melanger(s, v, j, w):
    """Weights of vertex v pulled towards those of vertex j, by the share w (four bones at most)."""
    acc = {}
    for src, f in ((v, 1.0 - w), (j, w)):
        for b, x in zip(s["os"][src], s["poids"][src]):
            if b >= 0 and x > 0:
                acc[int(b)] = acc.get(int(b), 0.0) + f * float(x)
    pris = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(x for _, x in pris)
    os_, p = [b for b, _ in pris], [x / tot for _, x in pris]
    s["sdef"][v] = None
    if len(os_) == 1:
        s["type"][v], s["os"][v], s["poids"][v] = 0, (os_[0], -1, -1, -1), (1.0, 0.0, 0.0, 0.0)
    elif len(os_) == 2:
        s["type"][v], s["os"][v], s["poids"][v] = 1, (os_[0], os_[1], -1, -1), (p[0], 1.0 - p[0], 0.0, 0.0)
    else:
        os_ += [os_[0]] * (4 - len(os_))
        p += [0.0] * (4 - len(p))
        s["type"][v], s["os"][v], s["poids"][v] = 2, tuple(os_), tuple(p)


def _poser_nuque(s, faces, fm, i_peau, i_ile, i_visage):
    """The bottom of the head's scalp slides under the body's neck, which does not move.

    The taken scalp (`crane_du_jeu`) stops where the head carries it at `COU_POIDS`; the donor head's
    neck was thicker than the body's: at that height, this edge stayed 3 to 9 mm outside (survey of
    10/11 on six heads), open under the hair. Each vertex of that edge goes `RENTRE_MM` under the body
    skin; scalp vertices within `NUQUE_MM` of the edge (along the skin) smoothly follow the mean
    displacement of the neighbouring edge vertices (within `LISSE_MM`), with the body's normal and
    weights (they turn with the neck) and no outline; at the top of the strip, nothing changes.
    Measured each on its own, a high vertex under the hair, with no body neck facing it, was pulled
    towards the shoulder (up to 45 mm on Miyabi, 10/11). The bottom of the face (`i_visage`) is taken
    along with the scalp: where the head's neck is cut, its edge stayed up to 10 mm outside the body's
    neck and the back side of the face showed underneath (Jane, Caesar, Anby, 10/11). Vertices
    stitched to another part stay put."""
    pos = np.asarray(s["pos"], float)
    nor = np.asarray(s["nor"], float)
    peaux = [i_ile] + list(i_visage)
    if not (fm == i_ile).any():
        return {"nuque_bord": 0}
    Fi = faces[np.isin(fm, peaux)]
    Fb = _peau_pres(pos, faces[fm == i_peau], Fi)
    graines, loin, rep, premier = _bord_de_nuque(pos, nor, faces, fm, peaux, Fb)
    bilan = {"nuque_bord": int(len(graines)), "nuque_bord_loin": int(len(loin))}
    if not len(graines):
        return bilan
    P = pos[premier]
    E = np.unique(_aretes(rep, Fi), axis=0)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    g = np.full(len(P), np.inf)
    g[graines] = 0.0
    for _ in range(len(P)):
        g2 = g.copy()
        np.minimum.at(g2, E[:, 0], g[E[:, 1]] + L)
        np.minimum.at(g2, E[:, 1], g[E[:, 0]] + L)
        if np.array_equal(g2, g):
            break
        g = g2
    cousu = np.zeros(len(P), bool)
    cousu[rep[np.unique(faces[~np.isin(fm, peaux + [i_peau])])]] = True
    vi = np.unique(Fi)
    r = np.unique(rep[vi])
    r = r[(g[r] < NUQUE_MM * MM) & ~cousu[r]]
    x = 1.0 - g[r] / (NUQUE_MM * MM)
    w = x * x * (3.0 - 2.0 * x)
    d, n, pp = _distance_au_corps(pos, nor, Fb, P[graines])
    dep_g = pp - RENTRE_MM * MM * n - P[graines]
    K = np.exp(-(np.linalg.norm(P[r][:, None] - P[graines][None], axis=2) / (LISSE_MM * MM)) ** 2)
    dep = w[:, None] * (K @ dep_g) / np.maximum(K.sum(1), 1e-12)[:, None]
    est_g = np.isin(r, graines)
    dep[est_g] = dep_g[np.searchsorted(graines, r[est_g])]
    vb = np.unique(Fb)
    cible = P[r] + dep
    j = vb[np.argmin(((cible[:, None] - pos[vb][None]) ** 2).sum(-1), axis=1)]
    a_ile = np.zeros(len(pos), bool)
    a_ile[vi] = True
    for v in np.where(a_ile & np.isin(rep, r))[0].tolist():
        k = int(np.searchsorted(r, rep[v]))
        s["pos"][v] = tuple(float(c) for c in pos[v] + dep[k])
        nv = (1.0 - w[k]) * nor[v] + w[k] * nor[j[k]]
        s["nor"][v] = tuple(float(c) for c in nv / max(float(np.linalg.norm(nv)), 1e-12))
        _melanger(s, v, int(j[k]), float(w[k]))
        s["contour"][v] = float(s["contour"][v]) * (1.0 - float(w[k]))
    bilan.update({"nuque_sommets": int((a_ile & np.isin(rep, r)).sum()),
                  "nuque_deplacement_mm": round(float(np.linalg.norm(dep, axis=1).max() / MM), 1),
                  "nuque_ecart_bord_mm": [round(float(x / MM), 1) for x in np.percentile(d, [0, 50, 100])]})
    return bilan


def appliquer(s, faces, mat_face, materiaux, i_peau, i_visage, cou_tete, axe, i_ile=None, os_tete=None):
    """Remove the body's scalp without touching its neck; returns (kept faces, summary) and updates
    `materiaux`.

    i_ile: material of the ears and scalp brought by the head; the body goes under it as under the
    face, and the bottom of the scalp under the body's neck (`_poser_nuque`).
    os_tete: the head bone and its children (`os_de_tete`); without them, nothing counts as neck."""
    pos0 = np.asarray(s["pos"], float)
    fp = faces[mat_face == i_peau]
    fige = np.zeros(len(pos0), bool)
    if os_tete is not None:
        o, w = np.asarray(s["os"]), np.asarray(s["poids"], float)
        cou = np.zeros(len(pos0), bool)
        cou[np.unique(fp)] = True
        cou &= (w * np.isin(o, os_tete)).sum(1) < COU_POIDS * w.sum(1)
        fige[np.unique(fp[cou[fp].any(1)])] = True
        # copies of a frozen vertex (UV seams) are frozen too: `_rentrer` pushed only one of them and the
        # skin split at the neck edge, 0.6 to 3.5 mm slits on nine heads out of ten (survey of 10/11)
        vp = np.unique(fp)
        rep = _soudes(pos0)
        fige[vp] |= np.isin(rep[vp], rep[fige])
    if i_ile is not None:
        # above the edge of the head scalp that will rest there, the body keeps `RECOUVRE_MM` of skin underneath
        y0 = pos0[np.unique(faces[np.isin(mat_face, i_visage)]), 1].min() - ZONE_MM * MM
        Fb0 = faces[(mat_face == i_peau) & ~_crane_du_corps(pos0, faces, mat_face, i_peau, y0)]
        Fb0 = _peau_pres(pos0, Fb0, faces[mat_face == i_ile])
        graines, _, _, premier = _bord_de_nuque(pos0, np.asarray(s["nor"], float), faces, mat_face, [i_ile], Fb0)
        if len(graines):
            cou_tete = np.vstack([np.asarray(cou_tete, float).reshape(-1, 3),
                                  pos0[premier[graines]] + np.array([0.0, RECOUVRE_MM * MM, 0.0])])
    retirer, bord, lim, bilan = couper(pos0, faces, mat_face, i_peau, i_visage, cou_tete, axe, fige)
    bilan["sommets_figes"] = int(fige.sum())
    garde = ~retirer
    for v, p in zip(bord.tolist(), _sur_la_coupe(pos0, faces[garde & (mat_face == i_peau)], bord, lim)):
        s["pos"][v] = tuple(float(x) for x in p)
    faces, fm = faces[garde], mat_face[garde]
    if i_ile is not None:
        bilan.update(_poser_nuque(s, faces, fm, i_peau, i_ile, list(i_visage)))
    i_vis_tout = list(i_visage) + ([i_ile] if i_ile is not None else [])
    pos1, b_r = _rentrer(np.asarray(s["pos"], float), np.asarray(s["nor"], float), faces, fm, i_peau,
                                  i_vis_tout, fige)
    for v in np.where((pos1 != np.asarray(s["pos"], float)).any(1))[0].tolist():
        s["pos"][v] = tuple(float(x) for x in pos1[v])
    bilan.update(b_r)
    for i, m in enumerate(materiaux):
        m["n_indices"] = int(3 * (fm == i).sum())
    _unifier_normales(s, faces, fm, list(i_visage))
    pos = np.asarray(s["pos"], float)
    vv = np.unique(faces[np.isin(fm, i_vis_tout)])
    vp = np.unique(faces[fm == i_peau])
    lo, hi = pos[vv].min(0) - SANS_TRAIT_MM * MM, pos[vv].max(0) + SANS_TRAIT_MM * MM
    vp = vp[((pos[vp] > lo) & (pos[vp] < hi)).all(1)]
    d = np.array([np.sqrt(((pos[vv] - pos[i]) ** 2).sum(1)).min() for i in vp.tolist()]) if len(vp) else np.zeros(0)
    sans = vp[(d < SANS_TRAIT_MM * MM) & ~fige[vp]]
    for i in sans.tolist():
        s["contour"][i] = 0.0
    bilan["sans_trait"] = int(len(sans))
    return faces, bilan

"""Skin tint: the skin of the host head and body (B) takes the colour of the grafted face (A).

    blender -b --factory-startup --python tint.py -- <resultat.pmx>

Face mode only. Skins are measured in Lab in the textures. Each texture of B
(excluding hair and face parts) is matched to the already-matched skin it touches: B's skull
to A's face, then the neck to the skull. Lightness is shifted so that the strip of skin
along the seam (0.1 interocular distance, one point per face) matches the one opposite; the
hue (a, b) takes the one opposite, B keeping only a quarter of its own hue deviations:
the textures carry gradients, and matching the median alone left the neck yellower than the
face. Each pixel's weight decreases with its distance to its texture's skin: clothes and
hair do not move; a texture whose "skin" is colourless next to that of B's face
(light fabric) stays as is (teinte.json: sans_peau). The recoloured textures go
into <sortie>/teinte/, the PMX is rewritten to use them; teinte.json gives the difference (ΔE Lab)
of the medians and that of each seam, before and after. B's skull is then matched to
A's face point by point along the boundary (raccord_visage; teinte.json: raccord_visage,
median and maximum of the differences at the boundary points).

Forehead of the grafted face: A's texture carries, above the eyes, the painted shadow of its own
fringe. Under B's hair this shadow no longer falls in the right place: on the 颜_A faces
higher than the eyes, pixels far from A's skin are pulled back towards it (copy of the
texture in <sortie>/teinte/, face parts untouched).
"""
import json
import os
import sys

import bpy
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
import pmx as PMX  # noqa: E402
import pmx_write as pmx_ecrire  # noqa: E402
import head as TT  # noqa: E402

# floor of the robust deviation (L, a, b) and reach of the skin mask, in deviations
ECART_MIN = np.array([3.0, 2.0, 2.0])
PORTEE = 3.0
PORTEE_FRONT = 1.5  # forehead: even a light shadow goes
ECHANTILLONS = 200000
VARIATION_TEINTE = 0.25  # share of B's hue deviations around its median that is kept
RAYON_COUTURE = 0.1  # strip on either side of a seam, in interocular distances
# a texture whose detected "skin" has less than this share of the saturation (a, b) of the skin of
# B's face has none: it is a light fabric that the wide reach lets through (Miyabi's
# shirt, Lab 82/0/0 against 83/10/15); it is not recoloured. Real skins: 0.69 to 0.99
SATURATION_MIN = 0.4


def srgb_vers_lab(c):
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def lab_vers_srgb(lab):
    fy = (lab[:, 0] + 16) / 116
    fx, fz = fy + lab[:, 1] / 500, fy - lab[:, 2] / 200
    f = np.stack([fx, fy, fz], 1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    c = np.clip(xyz @ M.T, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def lire_image(chemin):
    im = bpy.data.images.load(chemin)
    w, h = im.size
    px = np.empty(w * h * 4, dtype=np.float32)
    im.pixels.foreach_get(px)
    bpy.data.images.remove(im)
    return px.reshape(h, w, 4)


def ecrire_image(chemin, px):
    h, w = px.shape[:2]
    im = bpy.data.images.new("t", w, h, alpha=True)
    im.pixels.foreach_set(px.astype(np.float32).ravel())
    im.filepath_raw = chemin
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)


def echantillons(img, uv, F, rng):
    """Lab colours of the texture under faces F (random points, in proportion to UV area)."""
    t = uv[F]
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    aire = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if aire.sum() <= 0:
        return np.zeros((0, 3))
    k = rng.choice(len(F), ECHANTILLONS, p=aire / aire.sum())
    r1, r2 = rng.random(ECHANTILLONS), rng.random(ECHANTILLONS)
    s = np.sqrt(r1)
    b = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    p = np.einsum("nk,nkd->nd", b, t[k])
    h, w = img.shape[:2]
    x = (np.mod(p[:, 0], 1) * w).astype(int).clip(0, w - 1)
    y = ((1 - np.mod(p[:, 1], 1)) * h).astype(int).clip(0, h - 1)  # PMX V points down
    c = img[y, x]
    c = c[c[:, 3] > 0.5]
    return srgb_vers_lab(c[:, :3].astype(float))


def stats(lab):
    med = np.median(lab, 0)
    ec = np.maximum(1.4826 * np.median(np.abs(lab - med), 0), ECART_MIN)
    return med, ec


def front_propre(P, sortie, iA, tex, medA, ecA, uv, F, mf):
    """Erases the fringe shadow painted on A's forehead (see the header). Returns the report."""
    pos = np.asarray(P["sommets"]["pos"], dtype=float)
    noms = [m["nom"] for m in P["materiaux"]]
    oeil = np.unique(F[mf == noms.index("目")]) if "目" in noms else np.zeros(0, dtype=int)
    if not len(oeil):
        return {"applique": False}
    cg = pos[oeil][pos[oeil][:, 0] > 0].mean(0)
    cd = pos[oeil][pos[oeil][:, 0] < 0].mean(0)
    iod = float(np.linalg.norm(cg - cd))
    y0 = float(pos[oeil, 1].max()) + 0.05 * iod   # level with the top of the eyes: nothing changes
    y1 = y0 + 0.25 * iod                          # above: full cleanup
    f = tex(iA)
    img = lire_image(f)
    h, w = img.shape[:2]
    poids = np.zeros((h, w))
    for t in F[mf == iA]:
        hy = np.clip((pos[t, 1] - y0) / (y1 - y0), 0, 1)
        if hy.max() <= 0:
            continue
        q = np.stack([np.mod(uv[t, 0], 1) * w, (1 - np.mod(uv[t, 1], 1)) * h], 1)
        x0, x1 = int(np.floor(q[:, 0].min())), int(np.ceil(q[:, 0].max())) + 1
        ya, yb = int(np.floor(q[:, 1].min())), int(np.ceil(q[:, 1].max())) + 1
        x0, ya, x1, yb = max(x0, 0), max(ya, 0), min(x1, w), min(yb, h)
        if x1 <= x0 or yb <= ya:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(ya, yb) + 0.5)
        e1, e2 = q[1] - q[0], q[2] - q[0]
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if abs(det) < 1e-12:
            continue
        dx, dy = X - q[0, 0], Y - q[0, 1]
        b1 = (dx * e2[1] - dy * e2[0]) / det
        b2 = (e1[0] * dy - e1[1] * dx) / det
        b0 = 1 - b1 - b2
        dedans = (b0 >= -1e-3) & (b1 >= -1e-3) & (b2 >= -1e-3)
        v = b0 * hy[0] + b1 * hy[1] + b2 * hy[2]
        v = v * v * (3 - 2 * v)
        bloc = poids[ya:yb, x0:x1]
        bloc[dedans] = np.maximum(bloc[dedans], v[dedans])
    lab = srgb_vers_lab(img[..., :3].reshape(-1, 3).astype(float))
    d = (lab - medA) / ecA
    loin = 1 - np.exp(-0.5 * np.sum(d * d, 1) / PORTEE_FRONT ** 2)
    k = (poids.ravel() * loin)[:, None]
    lab2 = lab + k * (medA - lab)
    px = img.copy()
    px[..., :3] = lab_vers_srgb(lab2).reshape(h, w, 3)
    nom = "front_%s.png" % os.path.splitext(os.path.basename(f))[0]
    ecrire_image(os.path.join(sortie, "teinte", nom), px)
    P["textures"].append("teinte\\" + nom)
    P["materiaux"][iA]["texture"] = len(P["textures"]) - 1
    return {"applique": True, "copie": "teinte/" + nom, "pixels_touches": int((k[:, 0] > 0.05).sum())}


def _iod(P, F, mf):
    pos = np.asarray(P["sommets"]["pos"], dtype=float)
    noms = [m["nom"] for m in P["materiaux"]]
    oeil = np.unique(F[mf == noms.index("目")])
    cg, cd = pos[oeil][pos[oeil][:, 0] > 0].mean(0), pos[oeil][pos[oeil][:, 0] < 0].mean(0)
    return float(np.linalg.norm(cg - cd))


def _distance(X, Y):
    """Distance from each point of X to the closest point of Y (in blocks)."""
    out = np.empty(len(X))
    for k in range(0, len(X), 256):
        out[k:k + 256] = np.sqrt(((X[k:k + 256, None] - Y[None]) ** 2).sum(-1)).min(1)
    return out


def _centres(img, uv, faces):
    """Lab colours of the texture at the centre of each face: near a seam, faces are
    thin and numerous, area-weighted sampling would reach for the large faces further away."""
    h, w = img.shape[:2]
    c = uv[faces].mean(1)
    x = (np.mod(c[:, 0], 1) * w).astype(int).clip(0, w - 1)
    y = ((1 - np.mod(c[:, 1], 1)) * h).astype(int).clip(0, h - 1)
    px = img[y, x]
    px = px[px[:, 3] > 0.5]
    return srgb_vers_lab(px[:, :3].astype(float))


def _bande_img(img, uv, faces, rng, med, ec):
    """Lab median of a texture's skin under some faces (seam strip), or None."""
    if not len(faces):
        return None
    L = _centres(img, uv, faces)
    L = L[np.sum(((L - med) / (PORTEE * ec)) ** 2, 1) < 1]
    return np.median(L, 0) if len(L) >= 20 else None


def _bande(P, tex, uv, F, mf, mats, pres, rng, med, ec):
    """Colour of the already-matched skin (materials mats) along the seam."""
    fR = F[np.isin(mf, mats)]
    L = []
    for i in mats:
        sel = (mf[np.isin(mf, mats)] == i) & pres
        if sel.any():
            L.append(_centres(lire_image(tex(i)), uv, fR[sel]))
    if not L:
        return None
    L = np.concatenate(L)
    L = L[np.sum(((L - med) / (PORTEE * ec)) ** 2, 1) < 1]
    return np.median(L, 0) if len(L) >= 20 else None


def _proche(X, Y):
    """Distance and index of the point of Y closest to each point of X (in blocks)."""
    d, k = np.empty(len(X)), np.empty(len(X), dtype=int)
    for i in range(0, len(X), 256):
        D = np.sqrt(((X[i:i + 256, None] - Y[None]) ** 2).sum(-1))
        k[i:i + 256] = D.argmin(1)
        d[i:i + 256] = D[np.arange(len(D)), k[i:i + 256]]
    return d, k


def _aux_points(img, p):
    """Lab colour of the texture at UV points p (average of 3 x 3 pixels)."""
    h, w = img.shape[:2]
    x = (np.mod(p[:, 0], 1) * w).astype(int)
    y = ((1 - np.mod(p[:, 1], 1)) * h).astype(int)
    c = np.zeros((len(p), 3))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c += img[(y + dy).clip(0, h - 1), (x + dx).clip(0, w - 1), :3]
    return srgb_vers_lab(c / 9)


def raccord_visage(px, imgA, pos, uv, F, mf, iA, mats, autres, iod):
    """Point-by-point match of B's skin at the boundary of A's face. The overall matching gives
    B a single seam colour, but A's changes along the boundary (light cheek, chin,
    temple): a colour step remained that outlined the patch, in triangle teeth
    (A's, which are large). The remaining difference is sampled all along the boundary edges, then spread
    pixel by pixel into B's skin: full at the face boundary, zero where B touches its other
    skins (skull, neck), whose seams therefore do not move."""
    fA, fT = F[mf == iA], F[np.isin(mf, mats)]
    vT, vA = np.unique(fT), np.unique(fA)
    d, k = _proche(pos[vA], pos[vT])
    # a vertex of B may be duplicated at the same place (one texture chart per face): all its duplicates are
    # kept, and the edge chosen is one of a real face of B. With the first duplicate found, an edge straddling
    # two charts read the texture far from the skin (white spot under the jaw)
    place = {}
    for v in vT.tolist():
        place.setdefault(tuple(np.round(pos[v] * 1e4).astype(int)), []).append(v)
    vers_B = {a: place[tuple(np.round(pos[t] * 1e4).astype(int))]
              for a, t in zip(vA[d < 1e-5].tolist(), vT[k[d < 1e-5]].tolist())}
    # a face of B that uses a vertex of A (small seam hole filled with a fan) reads B's texture with
    # A's coordinates, anywhere in the image: it does not give B's colour
    aretes_B = {tuple(sorted((int(t[i]), int(t[(i + 1) % 3])))) for t in fT[~np.isin(fT, vA).any(1)] for i in range(3)}
    P, UA, UB = [], [], []
    u = np.linspace(0, 1, 9)[:, None]
    for t in fA:
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            if a not in vers_B or b not in vers_B:
                continue
            for a2, b2 in ((x, y) for x in vers_B[a] for y in vers_B[b] if tuple(sorted((x, y))) in aretes_B):
                P.append(pos[a] + u * (pos[b] - pos[a]))
                UA.append(uv[a] + u * (uv[b] - uv[a]))
                UB.append(uv[a2] + u * (uv[b2] - uv[a2]))
                break
    if len(P) < 8:
        return px, {"applique": False}
    P, UA, UB = np.concatenate(P), np.concatenate(UA), np.concatenate(UB)
    ecart = _aux_points(imgA, UA) - _aux_points(px, UB)
    brut = np.linalg.norm(ecart, axis=1)
    # smoothed along the boundary: a stroke painted under one point must not spread into the cheek
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    G = np.exp(-0.5 * (D / (0.02 * iod)) ** 2)
    ecart = G @ ecart / G.sum(1)[:, None]
    # blend: 1 at the face boundary, 0 at B's other skins
    d1, _ = _proche(pos[vT], P)
    vO = np.unique(F[np.isin(mf, autres)]) if len(autres) else np.zeros(0, dtype=int)
    if len(vO):
        d2, _ = _proche(pos[vT], pos[vO])
        f = d2 / np.maximum(d1 + d2, 1e-9)
    else:
        f = np.clip(1 - d1 / iod, 0, 1)
    fondu = np.zeros(len(pos))
    fondu[vT] = f * f * (3 - 2 * f)
    # pixels of B's skin: position in space and blend
    h, w = px.shape[:2]
    MARGE = 6
    ou, ptsX, ptsF, prof = [], [], [], []
    for t in fT:
        if fondu[t].max() <= 0:
            continue
        q = np.stack([np.mod(uv[t, 0], 1) * w, (1 - np.mod(uv[t, 1], 1)) * h], 1)
        x0, x1 = max(int(np.floor(q[:, 0].min())) - MARGE, 0), min(int(np.ceil(q[:, 0].max())) + MARGE + 1, w)
        ya, yb = max(int(np.floor(q[:, 1].min())) - MARGE, 0), min(int(np.ceil(q[:, 1].max())) + MARGE + 1, h)
        e1, e2 = q[1] - q[0], q[2] - q[0]
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if x1 <= x0 or yb <= ya or abs(det) < 1e-12:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1), np.arange(ya, yb))
        dx, dy = X + 0.5 - q[0, 0], Y + 0.5 - q[0, 1]
        b1 = (dx * e2[1] - dy * e2[0]) / det
        b2 = (e1[0] * dy - e1[1] * dx) / det
        b = np.stack([1 - b1 - b2, b1, b2], -1)
        # a few pixels of margin around each face: texture filtering (seen from afar, it
        # reads wide) does not find the old colour next to it, which drew a line along the boundary
        dedans = (b >= -MARGE / max(np.sqrt(abs(det)), 1)).all(-1)
        b = b[dedans]
        prof.append(b.min(-1))
        # in the margin, the closest point of the face: clipped without being renormalised to sum 1, the
        # weights inflated the blend along the edges, which showed as lines (a web on the cheek)
        b = b.clip(0, 1)
        b /= b.sum(-1, keepdims=True)
        ou.append((Y[dedans] * w + X[dedans]))
        ptsX.append(b @ pos[t])
        ptsF.append(b @ fondu[t])
    # a pixel seen by several faces goes to the one that contains it best, not to a neighbour's margin
    ou, prof = np.concatenate(ou), np.concatenate(prof)
    ordre = np.lexsort((-prof, ou))
    ou, i1 = np.unique(ou[ordre], return_index=True)
    i1 = ordre[i1]
    ptsX, ptsF = np.concatenate(ptsX)[i1].astype(np.float32), np.concatenate(ptsF)[i1]
    ou, ptsX, ptsF = ou[ptsF > 0.004], ptsX[ptsF > 0.004], ptsF[ptsF > 0.004]
    delta = np.empty((len(ou), 3))
    P32, e32 = P.astype(np.float32), ecart.astype(np.float32)
    eps = np.float32((0.01 * iod) ** 2)
    for i in range(0, len(ou), 8192):
        # same values as ((x - P) ** 2).sum(-1) and 1 / (D2 + eps) ** 2, one coordinate at a time and in place:
        # no (pixels, points, 3) array
        X = ptsX[i:i + 8192]
        E = X[:, 0, None] - P32[None, :, 0]
        W = E * E
        for c in (1, 2):
            np.subtract(X[:, c, None], P32[None, :, c], out=E)
            E *= E
            W += E
        W += eps
        W *= W
        np.reciprocal(W, out=W)
        delta[i:i + 8192] = W @ e32 / W.sum(1)[:, None]
    plat = px[..., :3].reshape(-1, 3).copy()
    plat[ou] = lab_vers_srgb(srgb_vers_lab(plat[ou].astype(float)) + ptsF[:, None] * delta)
    px = px.copy()
    px[..., :3] = plat.reshape(h, w, 3)
    reste = np.linalg.norm(_aux_points(imgA, UA) - _aux_points(px, UB), axis=1)
    return px, {"applique": True, "points": int(len(P)), "pixels": int(len(ou)),
                "dE_avant": [round(float(np.median(brut)), 2), round(float(brut.max()), 2)],
                "dE_apres": [round(float(np.median(reste)), 2), round(float(reste.max()), 2)]}


def main():
    chemin = sys.argv[sys.argv.index("--") + 1:][0]
    sortie = os.path.dirname(os.path.abspath(chemin))
    P = PMX.lire(chemin)
    uv = np.asarray(P["sommets"]["uv"], dtype=float)
    F = np.asarray(P["faces"], dtype=np.int64).reshape(-1, 3)
    mf = np.repeat(np.arange(len(P["materiaux"])), [m["n_indices"] // 3 for m in P["materiaux"]])
    noms = [m["nom"] for m in P["materiaux"]]
    iA, iB = noms.index(TT.PEAU + "_A"), noms.index(TT.PEAU)

    def tex(i):
        t = P["materiaux"][i]["texture"]
        if t < 0:
            return None
        return os.path.normpath(os.path.join(sortie, P["textures"][t].replace("\\", "/")))

    rng = np.random.default_rng(0)
    pos = np.asarray(P["sommets"]["pos"], dtype=float)
    medA, ecA = stats(echantillons(lire_image(tex(iA)), uv, F[mf == iA], rng))
    medB, ecB = stats(echantillons(lire_image(tex(iB)), uv, F[mf == iB], rng))
    gain = np.clip(ecA / ecB, 0.7, 1.4)
    rapport = {"peau_A_lab": medA.round(2).tolist(), "peau_B_lab": medB.round(2).tolist(),
               "dE_avant": round(float(np.linalg.norm(medA - medB)), 2), "textures": []}
    os.makedirs(os.path.join(sortie, "teinte"), exist_ok=True)
    rapport["front"] = front_propre(P, sortie, iA, tex, medA, ecA, uv, F, mf)
    iod = _iod(P, F, mf)

    # rest of the donor's skin (颜_A2: the donor was itself a face placed on another head): a few
    # large faces between its face and B's skin, on a separate texture that nothing matched. They kept
    # their flat colour between the cheek blush and the surround matched to it: a light patch at the temple, in
    # polygons (Orpheya). They take the face colour point by point along their shared boundary.
    iA2 = noms.index(TT.PEAU_COUSUE_2) if TT.PEAU_COUSUE_2 in noms else -1
    if iA2 >= 0 and (mf == iA2).any() and tex(iA2) and os.path.exists(tex(iA2)):
        px, r = raccord_visage(lire_image(tex(iA2)), lire_image(tex(iA)), pos, uv, F, mf, iA, [iA2], [], iod)
        rapport["reste_du_donneur"] = r
        if r["applique"]:
            ecrire_image(os.path.join(sortie, "teinte", "reste_donneur.png"), px)
            P["textures"].append("teinte\\reste_donneur.png")
            P["materiaux"][iA2]["texture"] = len(P["textures"]) - 1
    else:
        iA2 = -1

    # textures of B to recolour, with their materials
    exclus = set(TT.PIECES) | {TT.PEAU_COUSUE, TT.PEAU_COUSUE_2}
    groupes = {}
    for i, m in enumerate(P["materiaux"]):
        if m["nom"] in exclus or any(c in m["nom"] for c in ("髮", "发")) or not (mf == i).any():
            continue
        f = tex(i)
        if f is not None and os.path.exists(f):
            groupes.setdefault(f, []).append(i)
    ordre = sorted(groupes, key=lambda f: iB not in groupes[f])  # B's skull first

    # each texture is matched to the already-matched skin it touches (A's face, then B's
    # skull, then neck...): its skin near the seam takes the colour opposite
    regles = [iA]
    for f in ordre:
        mats = groupes[f]
        img = lire_image(f)
        fT = F[np.isin(mf, mats)]
        echT = echantillons(img, uv, fT, rng)
        echT = echT[np.sum(((echT - medB) / (PORTEE * ecB)) ** 2, 1) < 1]
        if len(echT) < 200:
            continue
        medT, ecT = stats(echT)
        if np.linalg.norm(medT[1:]) < SATURATION_MIN * np.linalg.norm(medB[1:]):
            rapport.setdefault("sans_peau", []).append(
                {"source": os.path.relpath(f, sortie), "materiaux": [noms[i] for i in mats],
                 "mediane_lab": medT.round(2).tolist()})
            continue
        vT = np.unique(fT)
        fR = F[np.isin(mf, regles)]
        pres_R = _distance(pos[fR].mean(1), pos[vT]) < RAYON_COUTURE * iod
        pres_T = _distance(pos[fT].mean(1), pos[np.unique(fR)]) < RAYON_COUTURE * iod
        cible = _bande(P, tex, uv, F, mf, regles, pres_R, rng, medA, ecA)
        bande = _bande_img(img, uv, fT[pres_T], rng, medT, ecT)
        if cible is None or bande is None:
            cible, bande = medA, medT
        h, w = img.shape[:2]
        lab = srgb_vers_lab(img[..., :3].reshape(-1, 3).astype(float))
        # weight: full over all the skin (the reach, twice as wide in lightness: skin in shadow
        # is still skin), then it falls off. As a bell curve from the median, ordinary skin received only
        # half the shift: no effect between two light skins, but under a dark face the neck
        # and ears stayed 15 lightness points too light (Orpheya)
        d = (lab - medT) / (ecT * PORTEE * np.array([2.0, 1.0, 1.0]))
        poids = np.exp(-0.5 * (np.maximum(np.sqrt(np.sum(d * d, 1)) - 1, 0) / 0.5) ** 2)
        # lightness: shifted to match the seam; hue (a, b): the one opposite,
        # with B's own variations reduced (a hue gradient shows from one end to the other)
        # the rest of B's face (its surround) keeps only half of its lightness deviations: they belong to a
        # painted face that is no longer there (rosy cheek), and under a dark face they made a patch at the temple
        g = 0.5 * np.minimum(gain, 1) if iB in mats else gain
        dl = (cible - bande) + (lab - bande) * (g - 1)
        dl[:, 1:] = cible[1:] + (lab[:, 1:] - medT[1:]) * VARIATION_TEINTE - lab[:, 1:]
        lab2 = lab + poids[:, None] * dl
        px = img.copy()
        px[..., :3] = lab_vers_srgb(lab2).reshape(h, w, 3)
        if iB in mats:
            ailleurs = [i for g in groupes.values() if g is not mats for i in g]
            px, rapport["raccord_visage"] = raccord_visage(
                px, lire_image(tex(iA)), pos, uv, F, mf, iA, mats, ailleurs + ([iA2] if iA2 >= 0 else []), iod)
            if iA2 >= 0:
                px, rapport["raccord_reste_donneur"] = raccord_visage(
                    px, lire_image(tex(iA2)), pos, uv, F, mf, iA2, mats, ailleurs + [iA], iod)
        else:
            # a skin matched after the skull (ears and neck) touches the rest of B's face and the body: the same
            # point-by-point match along each of these seams, zero towards the other skins (without it, a
            # hue line from the front of the ear to the jaw, visible under some hairstyles). The overall matching left nearly
            # one lightness point of difference across a smooth surface: a line at the neck ring
            for i in [r for r in regles if r != iA]:
                px, r = raccord_visage(px, lire_image(tex(i)), pos, uv, F, mf, i, mats,
                                       [j for j in regles if j != i], iod)
                rapport.setdefault("raccords", []).append({"de": [noms[j] for j in mats], "sur": noms[i]} | r)
        nom = "%d_%s.png" % (len(rapport["textures"]), os.path.splitext(os.path.basename(f))[0])
        ecrire_image(os.path.join(sortie, "teinte", nom), px)
        P["textures"].append("teinte\\" + nom)
        for i in mats:
            P["materiaux"][i]["texture"] = len(P["textures"]) - 1
        apres = _bande_img(px, uv, fT[pres_T], rng, cible, ecT)
        rapport["textures"].append({
            "source": os.path.relpath(f, sortie), "copie": "teinte/" + nom,
            "materiaux": [noms[i] for i in mats], "part_peau": round(float((poids > 0.5).mean()), 4),
            "couture_dE_avant": round(float(np.linalg.norm(cible - bande)), 2),
            "couture_dE_apres": round(float(np.linalg.norm(cible - apres)), 2) if apres is not None else None})
        regles += mats
    medB2, _ = stats(echantillons(lire_image(tex(iB)), uv, F[mf == iB], rng))
    rapport["dE_apres"] = round(float(np.linalg.norm(medA - medB2)), 2)
    rapport["materiaux_recolores"] = [noms[i] for i in regles[1:]]
    tmp = chemin + ".tmp"
    pmx_ecrire.ecrire(P, tmp)
    os.replace(tmp, chemin)
    j = os.path.join(sortie, "teinte.json")
    with open(j + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rapport, fh, ensure_ascii=False, indent=1)
    os.replace(j + ".tmp", j)
    print("teinte : ΔE %.1f -> %.1f, seams %s" % (rapport["dE_avant"], rapport["dE_apres"],
          ", ".join("%s %.1f -> %s" % ("+".join(t["materiaux"]), t["couture_dE_avant"], t["couture_dE_apres"])
                    for t in rapport["textures"])))


if __name__ == "__main__":
    main()

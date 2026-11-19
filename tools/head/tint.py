"""Skin tint: the skin of the host head and body (B) takes the colour of the grafted face (A).

    blender -b --factory-startup --python tint.py -- <result.pmx>

Face mode only. Skins are measured in Lab in the textures. Each texture of B
(excluding hair and face parts) is matched to the already-matched skin it touches: B's skull
to A's face, then the neck to the skull. Lightness is shifted so that the strip of skin
along the seam (0.1 interocular distance, one point per face) matches the one opposite; the
hue (a, b) takes the one opposite, B keeping only a quarter of its own hue deviations:
the textures carry gradients, and matching the median alone left the neck yellower than the
face. Each pixel's weight decreases with its distance to its texture's skin: clothes and
hair do not move; a texture whose "skin" is colourless next to that of B's face
(light fabric) stays as is (tint.json: without_skin). The recoloured textures go
into <output>/tint/, the PMX is rewritten to use them; tint.json gives the difference (ΔE Lab)
of the medians and that of each seam, before and after. B's skull is then matched to
A's face point by point along the boundary (face_join; tint.json: face_join,
median and maximum of the differences at the boundary points).

Forehead of the grafted face: A's texture carries, above the eyes, the painted shadow of its own
fringe. Under B's hair this shadow no longer falls in the right place: on the 颜_A faces
higher than the eyes, pixels far from A's skin are pulled back towards it (copy of the
texture in <output>/tint/, face parts untouched).
"""
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))
import pmx as PMX  # noqa: E402
import pmx_write as pmx_write  # noqa: E402
import head as TT  # noqa: E402

# floor of the robust deviation (L, a, b) and reach of the skin mask, in deviations
GAP_MIN = np.array([3.0, 2.0, 2.0])
REACH = 3.0
REACH_FRONT = 1.5  # forehead: even a light shadow goes
SAMPLES = 200000
VARIATION_TINT = 0.25  # share of B's hue deviations around its median that is kept
RADIUS_SEAM = 0.1  # strip on either side of a seam, in interocular distances
# a texture whose detected "skin" has less than this share of the saturation (a, b) of the skin of
# B's face has none: it is a light fabric that the wide reach lets through (Miyabi's
# shirt, Lab 82/0/0 against 83/10/15); it is not recoloured. Real skins: 0.69 to 0.99
SATURATION_MIN = 0.4


def srgb_to_lab(c):
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def lab_to_srgb(lab):
    fy = (lab[:, 0] + 16) / 116
    fx, fz = fy + lab[:, 1] / 500, fy - lab[:, 2] / 200
    f = np.stack([fx, fy, fz], 1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    c = np.clip(xyz @ M.T, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def read_image(path):
    im = bpy.data.images.load(path)
    w, h = im.size
    px = np.empty(w * h * 4, dtype=np.float32)
    im.pixels.foreach_get(px)
    bpy.data.images.remove(im)
    return px.reshape(h, w, 4)


def write_image(path, px):
    h, w = px.shape[:2]
    im = bpy.data.images.new("t", w, h, alpha=True)
    im.pixels.foreach_set(px.astype(np.float32).ravel())
    im.filepath_raw = path
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)


def samples(img, uv, F, rng):
    """Lab colours of the texture under faces F (random points, in proportion to UV area)."""
    t = uv[F]
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if area.sum() <= 0:
        return np.zeros((0, 3))
    k = rng.choice(len(F), SAMPLES, p=area / area.sum())
    r1, r2 = rng.random(SAMPLES), rng.random(SAMPLES)
    s = np.sqrt(r1)
    b = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    p = np.einsum("nk,nkd->nd", b, t[k])
    h, w = img.shape[:2]
    x = (np.mod(p[:, 0], 1) * w).astype(int).clip(0, w - 1)
    y = ((1 - np.mod(p[:, 1], 1)) * h).astype(int).clip(0, h - 1)  # PMX V points down
    c = img[y, x]
    c = c[c[:, 3] > 0.5]
    return srgb_to_lab(c[:, :3].astype(float))


def stats(lab):
    med = np.median(lab, 0)
    ec = np.maximum(1.4826 * np.median(np.abs(lab - med), 0), GAP_MIN)
    return med, ec


def front_own(P, output, iA, tex, medA, ecA, uv, F, mf):
    """Erases the fringe shadow painted on A's forehead (see the header). Returns the report."""
    pos = np.asarray(P["vertices"]["pos"], dtype=float)
    names = [m["name"] for m in P["materials"]]
    eye = np.unique(F[mf == names.index("目")]) if "目" in names else np.zeros(0, dtype=int)
    if not len(eye):
        return {"applied": False}
    cg = pos[eye][pos[eye][:, 0] > 0].mean(0)
    cd = pos[eye][pos[eye][:, 0] < 0].mean(0)
    iod = float(np.linalg.norm(cg - cd))
    y0 = float(pos[eye, 1].max()) + 0.05 * iod   # level with the top of the eyes: nothing changes
    y1 = y0 + 0.25 * iod                          # above: full cleanup
    f = tex(iA)
    img = read_image(f)
    h, w = img.shape[:2]
    weight = np.zeros((h, w))
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
        inside = (b0 >= -1e-3) & (b1 >= -1e-3) & (b2 >= -1e-3)
        v = b0 * hy[0] + b1 * hy[1] + b2 * hy[2]
        v = v * v * (3 - 2 * v)
        block = weight[ya:yb, x0:x1]
        block[inside] = np.maximum(block[inside], v[inside])
    lab = srgb_to_lab(img[..., :3].reshape(-1, 3).astype(float))
    d = (lab - medA) / ecA
    far = 1 - np.exp(-0.5 * np.sum(d * d, 1) / REACH_FRONT ** 2)
    k = (weight.ravel() * far)[:, None]
    lab2 = lab + k * (medA - lab)
    px = img.copy()
    px[..., :3] = lab_to_srgb(lab2).reshape(h, w, 3)
    name = "front_%s.png" % os.path.splitext(os.path.basename(f))[0]
    write_image(os.path.join(output, "tint", name), px)
    P["textures"].append("tint\\" + name)
    P["materials"][iA]["texture"] = len(P["textures"]) - 1
    return {"applied": True, "copy": "tint/" + name, "touched_pixels": int((k[:, 0] > 0.05).sum())}


def _iod(P, F, mf):
    pos = np.asarray(P["vertices"]["pos"], dtype=float)
    names = [m["name"] for m in P["materials"]]
    eye = np.unique(F[mf == names.index("目")])
    cg, cd = pos[eye][pos[eye][:, 0] > 0].mean(0), pos[eye][pos[eye][:, 0] < 0].mean(0)
    return float(np.linalg.norm(cg - cd))


def _distance(X, Y):
    """Distance from each point of X to the closest point of Y (in blocks)."""
    out = np.empty(len(X))
    for k in range(0, len(X), 256):
        out[k:k + 256] = np.sqrt(((X[k:k + 256, None] - Y[None]) ** 2).sum(-1)).min(1)
    return out


def _centers(img, uv, faces):
    """Lab colours of the texture at the centre of each face: near a seam, faces are
    thin and numerous, area-weighted sampling would reach for the large faces further away."""
    h, w = img.shape[:2]
    c = uv[faces].mean(1)
    x = (np.mod(c[:, 0], 1) * w).astype(int).clip(0, w - 1)
    y = ((1 - np.mod(c[:, 1], 1)) * h).astype(int).clip(0, h - 1)
    px = img[y, x]
    px = px[px[:, 3] > 0.5]
    return srgb_to_lab(px[:, :3].astype(float))


def _band_img(img, uv, faces, rng, med, ec):
    """Lab median of a texture's skin under some faces (seam strip), or None."""
    if not len(faces):
        return None
    L = _centers(img, uv, faces)
    L = L[np.sum(((L - med) / (REACH * ec)) ** 2, 1) < 1]
    return np.median(L, 0) if len(L) >= 20 else None


def _band(P, tex, uv, F, mf, mats, near, rng, med, ec):
    """Colour of the already-matched skin (materials mats) along the seam."""
    fR = F[np.isin(mf, mats)]
    L = []
    for i in mats:
        sel = (mf[np.isin(mf, mats)] == i) & near
        if sel.any():
            L.append(_centers(read_image(tex(i)), uv, fR[sel]))
    if not L:
        return None
    L = np.concatenate(L)
    L = L[np.sum(((L - med) / (REACH * ec)) ** 2, 1) < 1]
    return np.median(L, 0) if len(L) >= 20 else None


def _near(X, Y):
    """Distance and index of the point of Y closest to each point of X (in blocks)."""
    d, k = np.empty(len(X)), np.empty(len(X), dtype=int)
    for i in range(0, len(X), 256):
        D = np.sqrt(((X[i:i + 256, None] - Y[None]) ** 2).sum(-1))
        k[i:i + 256] = D.argmin(1)
        d[i:i + 256] = D[np.arange(len(D)), k[i:i + 256]]
    return d, k


def _to_points(img, p):
    """Lab colour of the texture at UV points p (average of 3 x 3 pixels)."""
    h, w = img.shape[:2]
    x = (np.mod(p[:, 0], 1) * w).astype(int)
    y = ((1 - np.mod(p[:, 1], 1)) * h).astype(int)
    c = np.zeros((len(p), 3))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c += img[(y + dy).clip(0, h - 1), (x + dx).clip(0, w - 1), :3]
    return srgb_to_lab(c / 9)


def face_join(px, imgA, pos, uv, F, mf, iA, mats, others, iod):
    """Point-by-point match of B's skin at the boundary of A's face. The overall matching gives
    B a single seam colour, but A's changes along the boundary (light cheek, chin,
    temple): a colour step remained that outlined the patch, in triangle teeth
    (A's, which are large). The remaining difference is sampled all along the boundary edges, then spread
    pixel by pixel into B's skin: full at the face boundary, zero where B touches its other
    skins (skull, neck), whose seams therefore do not move."""
    fA, fT = F[mf == iA], F[np.isin(mf, mats)]
    vT, vA = np.unique(fT), np.unique(fA)
    d, k = _near(pos[vA], pos[vT])
    # a vertex of B may be duplicated at the same place (one texture chart per face): all its duplicates are
    # kept, and the edge chosen is one of a real face of B. With the first duplicate found, an edge straddling
    # two charts read the texture far from the skin (white spot under the jaw)
    place = {}
    for v in vT.tolist():
        place.setdefault(tuple(np.round(pos[v] * 1e4).astype(int)), []).append(v)
    to_B = {a: place[tuple(np.round(pos[t] * 1e4).astype(int))]
              for a, t in zip(vA[d < 1e-5].tolist(), vT[k[d < 1e-5]].tolist())}
    # a face of B that uses a vertex of A (small seam hole filled with a fan) reads B's texture with
    # A's coordinates, anywhere in the image: it does not give B's colour
    edges_B = {tuple(sorted((int(t[i]), int(t[(i + 1) % 3])))) for t in fT[~np.isin(fT, vA).any(1)] for i in range(3)}
    P, UA, UB = [], [], []
    u = np.linspace(0, 1, 9)[:, None]
    for t in fA:
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            if a not in to_B or b not in to_B:
                continue
            for a2, b2 in ((x, y) for x in to_B[a] for y in to_B[b] if tuple(sorted((x, y))) in edges_B):
                P.append(pos[a] + u * (pos[b] - pos[a]))
                UA.append(uv[a] + u * (uv[b] - uv[a]))
                UB.append(uv[a2] + u * (uv[b2] - uv[a2]))
                break
    if len(P) < 8:
        return px, {"applied": False}
    P, UA, UB = np.concatenate(P), np.concatenate(UA), np.concatenate(UB)
    gap = _to_points(imgA, UA) - _to_points(px, UB)
    raw = np.linalg.norm(gap, axis=1)
    # smoothed along the boundary: a stroke painted under one point must not spread into the cheek
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    G = np.exp(-0.5 * (D / (0.02 * iod)) ** 2)
    gap = G @ gap / G.sum(1)[:, None]
    # blend: 1 at the face boundary, 0 at B's other skins
    d1, _ = _near(pos[vT], P)
    vO = np.unique(F[np.isin(mf, others)]) if len(others) else np.zeros(0, dtype=int)
    if len(vO):
        d2, _ = _near(pos[vT], pos[vO])
        f = d2 / np.maximum(d1 + d2, 1e-9)
    else:
        f = np.clip(1 - d1 / iod, 0, 1)
    blended = np.zeros(len(pos))
    blended[vT] = f * f * (3 - 2 * f)
    # pixels of B's skin: position in space and blend
    h, w = px.shape[:2]
    MARGIN = 6
    idx, ptsX, ptsF, depth = [], [], [], []
    for t in fT:
        if blended[t].max() <= 0:
            continue
        q = np.stack([np.mod(uv[t, 0], 1) * w, (1 - np.mod(uv[t, 1], 1)) * h], 1)
        x0, x1 = max(int(np.floor(q[:, 0].min())) - MARGIN, 0), min(int(np.ceil(q[:, 0].max())) + MARGIN + 1, w)
        ya, yb = max(int(np.floor(q[:, 1].min())) - MARGIN, 0), min(int(np.ceil(q[:, 1].max())) + MARGIN + 1, h)
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
        inside = (b >= -MARGIN / max(np.sqrt(abs(det)), 1)).all(-1)
        b = b[inside]
        depth.append(b.min(-1))
        # in the margin, the closest point of the face: clipped without being renormalised to sum 1, the
        # weights inflated the blend along the edges, which showed as lines (a web on the cheek)
        b = b.clip(0, 1)
        b /= b.sum(-1, keepdims=True)
        idx.append((Y[inside] * w + X[inside]))
        ptsX.append(b @ pos[t])
        ptsF.append(b @ blended[t])
    # a pixel seen by several faces goes to the one that contains it best, not to a neighbour's margin
    idx, depth = np.concatenate(idx), np.concatenate(depth)
    order = np.lexsort((-depth, idx))
    idx, i1 = np.unique(idx[order], return_index=True)
    i1 = order[i1]
    ptsX, ptsF = np.concatenate(ptsX)[i1].astype(np.float32), np.concatenate(ptsF)[i1]
    idx, ptsX, ptsF = idx[ptsF > 0.004], ptsX[ptsF > 0.004], ptsF[ptsF > 0.004]
    delta = np.empty((len(idx), 3))
    P32, e32 = P.astype(np.float32), gap.astype(np.float32)
    eps = np.float32((0.01 * iod) ** 2)
    for i in range(0, len(idx), 8192):
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
    flat = px[..., :3].reshape(-1, 3).copy()
    flat[idx] = lab_to_srgb(srgb_to_lab(flat[idx].astype(float)) + ptsF[:, None] * delta)
    px = px.copy()
    px[..., :3] = flat.reshape(h, w, 3)
    rest = np.linalg.norm(_to_points(imgA, UA) - _to_points(px, UB), axis=1)
    return px, {"applied": True, "points": int(len(P)), "pixels": int(len(idx)),
                "dE_before": [round(float(np.median(raw)), 2), round(float(raw.max()), 2)],
                "dE_after": [round(float(np.median(rest)), 2), round(float(rest.max()), 2)]}


def main():
    path = sys.argv[sys.argv.index("--") + 1:][0]
    output = os.path.dirname(os.path.abspath(path))
    P = PMX.read(path)
    uv = np.asarray(P["vertices"]["uv"], dtype=float)
    F = np.asarray(P["faces"], dtype=np.int64).reshape(-1, 3)
    mf = np.repeat(np.arange(len(P["materials"])), [m["n_indices"] // 3 for m in P["materials"]])
    names = [m["name"] for m in P["materials"]]
    iA, iB = names.index(TT.SKIN + "_A"), names.index(TT.SKIN)

    def tex(i):
        t = P["materials"][i]["texture"]
        if t < 0:
            return None
        return os.path.normpath(os.path.join(output, P["textures"][t].replace("\\", "/")))

    rng = np.random.default_rng(0)
    pos = np.asarray(P["vertices"]["pos"], dtype=float)
    medA, ecA = stats(samples(read_image(tex(iA)), uv, F[mf == iA], rng))
    medB, ecB = stats(samples(read_image(tex(iB)), uv, F[mf == iB], rng))
    gain = np.clip(ecA / ecB, 0.7, 1.4)
    report = {"skin_A_lab": medA.round(2).tolist(), "skin_B_lab": medB.round(2).tolist(),
               "dE_before": round(float(np.linalg.norm(medA - medB)), 2), "textures": []}
    os.makedirs(os.path.join(output, "tint"), exist_ok=True)
    report["forehead"] = front_own(P, output, iA, tex, medA, ecA, uv, F, mf)
    iod = _iod(P, F, mf)

    # rest of the donor's skin (颜_A2: the donor was itself a face placed on another head): a few
    # large faces between its face and B's skin, on a separate texture that nothing matched. They kept
    # their flat colour between the cheek blush and the surround matched to it: a light patch at the temple, in
    # polygons (Orpheya). They take the face colour point by point along their shared boundary.
    iA2 = names.index(TT.SKIN_SEWN_2) if TT.SKIN_SEWN_2 in names else -1
    if iA2 >= 0 and (mf == iA2).any() and tex(iA2) and os.path.exists(tex(iA2)):
        px, r = face_join(read_image(tex(iA2)), read_image(tex(iA)), pos, uv, F, mf, iA, [iA2], [], iod)
        report["rest_of_donor"] = r
        if r["applied"]:
            write_image(os.path.join(output, "tint", "donor_rest.png"), px)
            P["textures"].append("tint\\donor_rest.png")
            P["materials"][iA2]["texture"] = len(P["textures"]) - 1
    else:
        iA2 = -1

    # textures of B to recolour, with their materials
    excluded = set(TT.PARTS) | {TT.SKIN_SEWN, TT.SKIN_SEWN_2}
    groups = {}
    for i, m in enumerate(P["materials"]):
        if m["name"] in excluded or any(c in m["name"] for c in ("髮", "发")) or not (mf == i).any():
            continue
        f = tex(i)
        if f is not None and os.path.exists(f):
            groups.setdefault(f, []).append(i)
    order = sorted(groups, key=lambda f: iB not in groups[f])  # B's skull first

    # each texture is matched to the already-matched skin it touches (A's face, then B's
    # skull, then neck...): its skin near the seam takes the colour opposite
    rules = [iA]
    for f in order:
        mats = groups[f]
        img = read_image(f)
        fT = F[np.isin(mf, mats)]
        sampT = samples(img, uv, fT, rng)
        sampT = sampT[np.sum(((sampT - medB) / (REACH * ecB)) ** 2, 1) < 1]
        if len(sampT) < 200:
            continue
        medT, ecT = stats(sampT)
        if np.linalg.norm(medT[1:]) < SATURATION_MIN * np.linalg.norm(medB[1:]):
            report.setdefault("without_skin", []).append(
                {"source": os.path.relpath(f, output), "materials": [names[i] for i in mats],
                 "median_lab": medT.round(2).tolist()})
            continue
        vT = np.unique(fT)
        fR = F[np.isin(mf, rules)]
        pres_R = _distance(pos[fR].mean(1), pos[vT]) < RADIUS_SEAM * iod
        pres_T = _distance(pos[fT].mean(1), pos[np.unique(fR)]) < RADIUS_SEAM * iod
        target = _band(P, tex, uv, F, mf, rules, pres_R, rng, medA, ecA)
        band = _band_img(img, uv, fT[pres_T], rng, medT, ecT)
        if target is None or band is None:
            target, band = medA, medT
        h, w = img.shape[:2]
        lab = srgb_to_lab(img[..., :3].reshape(-1, 3).astype(float))
        # weight: full over all the skin (the reach, twice as wide in lightness: skin in shadow
        # is still skin), then it falls off. As a bell curve from the median, ordinary skin received only
        # half the shift: no effect between two light skins, but under a dark face the neck
        # and ears stayed 15 lightness points too light (Orpheya)
        d = (lab - medT) / (ecT * REACH * np.array([2.0, 1.0, 1.0]))
        weight = np.exp(-0.5 * (np.maximum(np.sqrt(np.sum(d * d, 1)) - 1, 0) / 0.5) ** 2)
        # lightness: shifted to match the seam; hue (a, b): the one opposite,
        # with B's own variations reduced (a hue gradient shows from one end to the other)
        # the rest of B's face (its surround) keeps only half of its lightness deviations: they belong to a
        # painted face that is no longer there (rosy cheek), and under a dark face they made a patch at the temple
        g = 0.5 * np.minimum(gain, 1) if iB in mats else gain
        dl = (target - band) + (lab - band) * (g - 1)
        dl[:, 1:] = target[1:] + (lab[:, 1:] - medT[1:]) * VARIATION_TINT - lab[:, 1:]
        lab2 = lab + weight[:, None] * dl
        px = img.copy()
        px[..., :3] = lab_to_srgb(lab2).reshape(h, w, 3)
        if iB in mats:
            elsewhere = [i for g in groups.values() if g is not mats for i in g]
            px, report["face_join"] = face_join(
                px, read_image(tex(iA)), pos, uv, F, mf, iA, mats, elsewhere + ([iA2] if iA2 >= 0 else []), iod)
            if iA2 >= 0:
                px, report["join_rest_donor"] = face_join(
                    px, read_image(tex(iA2)), pos, uv, F, mf, iA2, mats, elsewhere + [iA], iod)
        else:
            # a skin matched after the skull (ears and neck) touches the rest of B's face and the body: the same
            # point-by-point match along each of these seams, zero towards the other skins (without it, a
            # hue line from the front of the ear to the jaw, visible under some hairstyles). The overall matching left nearly
            # one lightness point of difference across a smooth surface: a line at the neck ring
            for i in [r for r in rules if r != iA]:
                px, r = face_join(px, read_image(tex(i)), pos, uv, F, mf, i, mats,
                                       [j for j in rules if j != i], iod)
                report.setdefault("joins", []).append({"from": [names[j] for j in mats], "on": names[i]} | r)
        name = "%d_%s.png" % (len(report["textures"]), os.path.splitext(os.path.basename(f))[0])
        write_image(os.path.join(output, "tint", name), px)
        P["textures"].append("tint\\" + name)
        for i in mats:
            P["materials"][i]["texture"] = len(P["textures"]) - 1
        after = _band_img(px, uv, fT[pres_T], rng, target, ecT)
        report["textures"].append({
            "source": os.path.relpath(f, output), "copy": "tint/" + name,
            "materials": [names[i] for i in mats], "fraction_skin": round(float((weight > 0.5).mean()), 4),
            "seam_dE_before": round(float(np.linalg.norm(target - band)), 2),
            "seam_dE_after": round(float(np.linalg.norm(target - after)), 2) if after is not None else None})
        rules += mats
    medB2, _ = stats(samples(read_image(tex(iB)), uv, F[mf == iB], rng))
    report["dE_after"] = round(float(np.linalg.norm(medA - medB2)), 2)
    report["recolored_materials"] = [names[i] for i in rules[1:]]
    tmp = path + ".tmp"
    pmx_write.write(P, tmp)
    os.replace(tmp, path)
    j = os.path.join(output, "tint.json")
    with open(j + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    os.replace(j + ".tmp", j)
    print("tint: ΔE %.1f -> %.1f, seams %s" % (report["dE_before"], report["dE_after"],
          ", ".join("%s %.1f -> %s" % ("+".join(t["materials"]), t["seam_dE_before"], t["seam_dE_after"])
                    for t in report["textures"])))


if __name__ == "__main__":
    main()

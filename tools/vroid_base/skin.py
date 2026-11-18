"""The body skin of a VRoid model, cleared of what VRoid painted on it.

Measured on 17/11/2026 on the VRoid Studio sample models: the body mesh is whole under the
clothes (a single opening, at the neck, where it joins the face), but its texture is not. VRoid paints on it
the clothing layer that sits on the body (tights, socks, tight top, underwear) and, when
the export removes what is hidden, makes transparent what the outfit covers (the whole trunk of one of the models).

The texels that are skin are kept: opaque and close to the reference tint (that of the face, which
nothing covers). The rest is filled from the neighbouring skin on the texture ("pull-push": averages of
the kept skin over an image pyramid, brought back down where there is none), which tends towards the median tint
of the skin far from any kept skin. The result is opaque everywhere.
"""
import numpy as np

ECART_TEINTE = 0.035        # chromaticity distance (normalized r, g) beyond which a texel is not skin
CLARTE = (0.6, 1.3)         # texel lightness / reference lightness: below this, too strong a shadow or a dark fabric
OPAQUE = 0.5
RONGER = 4                  # texels (2048 texture) removed at the edge of the kept skin: the edge mixes skin and fabric
PORTEE = 96                 # texels (2048 texture): reach of the matching of the two edges of a seam
ISOLE = (32, 0.03)          # half-side of the window (2048 texture), share of skin below which an island
                            # of kept skin is a leftover mix (skin and fabric) in the middle of a painted garment


def chromaticite(rgb):
    """(normalized r, g chromaticity; sum r+g+b) of an array (..., 3) between 0 and 1."""
    s = rgb.sum(-1)
    return rgb[..., :2] / np.maximum(s, 1e-6)[..., None], s


def reference(texture_visage):
    """Skin tint: median of the opaque texels of the face texture (h, w, 4)."""
    t = texture_visage
    px = t[t[..., 3] > OPAQUE][:, :3] if t.shape[-1] == 4 else t.reshape(-1, 3)
    if not len(px):
        raise ValueError("face texture entirely transparent")
    return np.median(px, 0)


def masque_peau(tex, ref, ecart=ECART_TEINTE, clarte=CLARTE):
    """Texels (h, w) of `tex` (h, w, 3|4, 0-1) that are skin: opaque, tint and lightness close to `ref`."""
    c, s = chromaticite(tex[..., :3])
    cr, sr = chromaticite(np.asarray(ref, float)[None])
    q = s / max(float(sr[0]), 1e-6)
    r, v, b = tex[..., 0], tex[..., 1], tex[..., 2]
    m = (np.linalg.norm(c - cr[0], axis=-1) < ecart) & (q > clarte[0]) & (q < clarte[1]) & (r >= v) & (v >= b)
    if tex.shape[-1] == 4:
        m &= tex[..., 3] > OPAQUE
    k = tex.shape[1] / 2048
    m = ronger(m, max(1, round(RONGER * k)))
    return m & (densite(m, max(1, round(ISOLE[0] * k))) >= ISOLE[1])


def densite(m, r):
    """Share of true texels in the (2r+1)² window around each texel (integral images)."""
    h, w = m.shape
    I = np.zeros((h + 1, w + 1))
    I[1:, 1:] = m.cumsum(0).cumsum(1)
    y0, y1 = np.clip(np.arange(h) - r, 0, h), np.clip(np.arange(h) + r + 1, 0, h)
    x0, x1 = np.clip(np.arange(w) - r, 0, w), np.clip(np.arange(w) + r + 1, 0, w)
    n = I[y1][:, x1] - I[y0][:, x1] - I[y1][:, x0] + I[y0][:, x0]
    return n / ((y1 - y0)[:, None] * (x1 - x0)[None, :])


def ronger(m, n):
    """The mask (h, w) stripped of its `n` edge texels (cross neighbourhood)."""
    m = m.copy()
    for _ in range(n):
        r = m.copy()
        r[1:] &= m[:-1]
        r[:-1] &= m[1:]
        r[:, 1:] &= m[:, :-1]
        r[:, :-1] &= m[:, 1:]
        m = r
    return m


def remplir(rgb, connu):
    """Fills the unknown texels of `rgb` (h, w, 3) from the known ones (mask (h, w)): pyramid of weighted
    averages (pull), then each level completes the finer one where it knows nothing (push), bilinearly."""
    if not connu.any():
        raise ValueError("no skin texel to fill from")
    niveaux = []
    c, w = rgb * connu[..., None], connu.astype(float)
    while True:
        niveaux.append((c, w))
        if min(w.shape) <= 1:
            break
        h2, l2 = (w.shape[0] + 1) // 2, (w.shape[1] + 1) // 2
        cp = np.zeros((h2 * 2, l2 * 2, 3))
        wp = np.zeros((h2 * 2, l2 * 2))
        cp[:c.shape[0], :c.shape[1]], wp[:w.shape[0], :w.shape[1]] = c, w
        c = cp.reshape(h2, 2, l2, 2, 3).sum((1, 3))
        w = wp.reshape(h2, 2, l2, 2).sum((1, 3))
    # at the coarsest level, the average; then back down
    c, w = niveaux[-1]
    plein = c / np.maximum(w, 1e-12)[..., None]
    for c, w in reversed(niveaux[:-1]):
        haut = _agrandir(plein, w.shape)
        k = np.clip(w, 0, 1)[..., None]
        plein = k * (c / np.maximum(w, 1e-12)[..., None]) + (1 - k) * haut
    out = rgb.copy()
    out[~connu] = plein[~connu]
    return out


def _agrandir(im, forme):
    """Image (h, w, 3) doubled in size bilinearly (texel centres aligned), cropped to `forme`."""
    h, w = im.shape[:2]
    H, W = forme
    y = (np.arange(H) + 0.5) / 2 - 0.5
    x = (np.arange(W) + 0.5) / 2 - 0.5
    y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
    y1, x1 = np.clip(y0 + 1, 0, h - 1), np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(y - y0, 0, 1)[:, None, None]
    fx = np.clip(x - x0, 0, 1)[None, :, None]
    a = im[y0][:, x0] * (1 - fx) + im[y0][:, x1] * fx
    b = im[y1][:, x0] * (1 - fx) + im[y1][:, x1] * fx
    return a * (1 - fy) + b * fy


def nettoyer(tex_corps, tex_visage):
    """Body texture (h, w, 4 or 3, 0-1) -> (opaque RGBA texture, mask of the texels kept as they are)."""
    ref = reference(tex_visage)
    m = masque_peau(tex_corps, ref)
    rgb = remplir(np.asarray(tex_corps[..., :3], float), m)
    return np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], -1), m


def accorder_coutures(tex, connu, uv_a, uv_b):
    """Matches the two edges of the model's seams on the filled texture: the point uv_a[i] and the point uv_b[i]
    (n, 2) are the same point of the skin, stored at two places in the texture (leg and foot: two islands, filled
    from different neighbours, hence a line). Each repainted edge receives the offset to the shared tint (the
    average, or that of the kept edge), spread to the neighbouring fill and blended over PORTEE texels. The kept skin
    does not change. Returns the texture (h, w, 4)."""
    h, w = connu.shape
    rgb = tex[..., :3]
    pa = _texel(uv_a, w, h)
    pb = _texel(uv_b, w, h)
    ca, cb = rgb[pa[:, 1], pa[:, 0]], rgb[pb[:, 1], pb[:, 0]]
    ka, kb = connu[pa[:, 1], pa[:, 0]], connu[pb[:, 1], pb[:, 0]]
    cible = np.where(ka[:, None] & ~kb[:, None], ca, np.where(kb[:, None] & ~ka[:, None], cb, (ca + cb) / 2))
    corr = np.zeros((h, w, 3))
    poids = np.zeros((h, w))
    for p, c, k in ((pa, ca, ka), (pb, cb, kb)):
        p, d = p[~k], (cible - c)[~k]
        np.add.at(corr, (p[:, 1], p[:, 0]), d)
        np.add.at(poids, (p[:, 1], p[:, 0]), 1)
    pts = poids > 0
    if not pts.any():
        return tex
    corr[pts] /= poids[pts][:, None]
    k = w / 2048
    etendue = remplir(corr, pts)
    t = densite(_dilater(pts, max(1, round(PORTEE * k))), max(1, round(PORTEE * k / 2)))[..., None]
    out = tex.copy()
    out[..., :3] = np.where(connu[..., None], rgb, np.clip(rgb + etendue * t, 0, 1))
    return out


def _texel(uv, w, h):
    uv = np.asarray(uv, float) % 1.0
    return np.stack([np.clip((uv[:, 0] * w).astype(int), 0, w - 1), np.clip((uv[:, 1] * h).astype(int), 0, h - 1)], 1)


def _dilater(m, n):
    return ~ronger(~m, n)

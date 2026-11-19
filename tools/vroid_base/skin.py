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

GAP_TINT = 0.035        # chromaticity distance (normalized r, g) beyond which a texel is not skin
CLARITY = (0.6, 1.3)         # texel lightness / reference lightness: below this, too strong a shadow or a dark fabric
OPAQUE = 0.5
ERODE = 4                  # texels (2048 texture) removed at the edge of the kept skin: the edge mixes skin and fabric
REACH = 96                 # texels (2048 texture): reach of the matching of the two edges of a seam
ISOLATED = (32, 0.03)          # half-side of the window (2048 texture), share of skin below which an island
                            # of kept skin is a leftover mix (skin and fabric) in the middle of a painted garment


def chromaticity(rgb):
    """(normalized r, g chromaticity; sum r+g+b) of an array (..., 3) between 0 and 1."""
    s = rgb.sum(-1)
    return rgb[..., :2] / np.maximum(s, 1e-6)[..., None], s


def reference(texture_face):
    """Skin tint: median of the opaque texels of the face texture (h, w, 4)."""
    t = texture_face
    px = t[t[..., 3] > OPAQUE][:, :3] if t.shape[-1] == 4 else t.reshape(-1, 3)
    if not len(px):
        raise ValueError("face texture entirely transparent")
    return np.median(px, 0)


def mask_skin(tex, ref, gap=GAP_TINT, clarity=CLARITY):
    """Texels (h, w) of `tex` (h, w, 3|4, 0-1) that are skin: opaque, tint and lightness close to `ref`."""
    c, s = chromaticity(tex[..., :3])
    cr, sr = chromaticity(np.asarray(ref, float)[None])
    q = s / max(float(sr[0]), 1e-6)
    r, v, b = tex[..., 0], tex[..., 1], tex[..., 2]
    m = (np.linalg.norm(c - cr[0], axis=-1) < gap) & (q > clarity[0]) & (q < clarity[1]) & (r >= v) & (v >= b)
    if tex.shape[-1] == 4:
        m &= tex[..., 3] > OPAQUE
    k = tex.shape[1] / 2048
    m = erode(m, max(1, round(ERODE * k)))
    return m & (density(m, max(1, round(ISOLATED[0] * k))) >= ISOLATED[1])


def density(m, r):
    """Share of true texels in the (2r+1)² window around each texel (integral images)."""
    h, w = m.shape
    I = np.zeros((h + 1, w + 1))
    I[1:, 1:] = m.cumsum(0).cumsum(1)
    y0, y1 = np.clip(np.arange(h) - r, 0, h), np.clip(np.arange(h) + r + 1, 0, h)
    x0, x1 = np.clip(np.arange(w) - r, 0, w), np.clip(np.arange(w) + r + 1, 0, w)
    n = I[y1][:, x1] - I[y0][:, x1] - I[y1][:, x0] + I[y0][:, x0]
    return n / ((y1 - y0)[:, None] * (x1 - x0)[None, :])


def erode(m, n):
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


def fill(rgb, known):
    """Fills the unknown texels of `rgb` (h, w, 3) from the known ones (mask (h, w)): pyramid of weighted
    averages (pull), then each level completes the finer one where it knows nothing (push), bilinearly."""
    if not known.any():
        raise ValueError("no skin texel to fill from")
    levels = []
    c, w = rgb * known[..., None], known.astype(float)
    while True:
        levels.append((c, w))
        if min(w.shape) <= 1:
            break
        h2, l2 = (w.shape[0] + 1) // 2, (w.shape[1] + 1) // 2
        cp = np.zeros((h2 * 2, l2 * 2, 3))
        wp = np.zeros((h2 * 2, l2 * 2))
        cp[:c.shape[0], :c.shape[1]], wp[:w.shape[0], :w.shape[1]] = c, w
        c = cp.reshape(h2, 2, l2, 2, 3).sum((1, 3))
        w = wp.reshape(h2, 2, l2, 2).sum((1, 3))
    # at the coarsest level, the average; then back down
    c, w = levels[-1]
    full = c / np.maximum(w, 1e-12)[..., None]
    for c, w in reversed(levels[:-1]):
        top = _enlarge(full, w.shape)
        k = np.clip(w, 0, 1)[..., None]
        full = k * (c / np.maximum(w, 1e-12)[..., None]) + (1 - k) * top
    out = rgb.copy()
    out[~known] = full[~known]
    return out


def _enlarge(im, shape):
    """Image (h, w, 3) doubled in size bilinearly (texel centres aligned), cropped to `shape`."""
    h, w = im.shape[:2]
    H, W = shape
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


def clean(tex_body, tex_face):
    """Body texture (h, w, 4 or 3, 0-1) -> (opaque RGBA texture, mask of the texels kept as they are)."""
    ref = reference(tex_face)
    m = mask_skin(tex_body, ref)
    rgb = fill(np.asarray(tex_body[..., :3], float), m)
    return np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], -1), m


def match_seams(tex, known, uv_a, uv_b):
    """Matches the two edges of the model's seams on the filled texture: the point uv_a[i] and the point uv_b[i]
    (n, 2) are the same point of the skin, stored at two places in the texture (leg and foot: two islands, filled
    from different neighbours, hence a line). Each repainted edge receives the offset to the shared tint (the
    average, or that of the kept edge), spread to the neighbouring fill and blended over REACH texels. The kept skin
    does not change. Returns the texture (h, w, 4)."""
    h, w = known.shape
    rgb = tex[..., :3]
    pa = _texel(uv_a, w, h)
    pb = _texel(uv_b, w, h)
    ca, cb = rgb[pa[:, 1], pa[:, 0]], rgb[pb[:, 1], pb[:, 0]]
    ka, kb = known[pa[:, 1], pa[:, 0]], known[pb[:, 1], pb[:, 0]]
    target = np.where(ka[:, None] & ~kb[:, None], ca, np.where(kb[:, None] & ~ka[:, None], cb, (ca + cb) / 2))
    corr = np.zeros((h, w, 3))
    weight = np.zeros((h, w))
    for p, c, k in ((pa, ca, ka), (pb, cb, kb)):
        p, d = p[~k], (target - c)[~k]
        np.add.at(corr, (p[:, 1], p[:, 0]), d)
        np.add.at(weight, (p[:, 1], p[:, 0]), 1)
    pts = weight > 0
    if not pts.any():
        return tex
    corr[pts] /= weight[pts][:, None]
    k = w / 2048
    extent = fill(corr, pts)
    t = density(_dilate(pts, max(1, round(REACH * k))), max(1, round(REACH * k / 2)))[..., None]
    out = tex.copy()
    out[..., :3] = np.where(known[..., None], rgb, np.clip(rgb + extent * t, 0, 1))
    return out


def _texel(uv, w, h):
    uv = np.asarray(uv, float) % 1.0
    return np.stack([np.clip((uv[:, 0] * w).astype(int), 0, w - 1), np.clip((uv[:, 1] * h).astype(int), 0, h - 1)], 1)


def _dilate(m, n):
    return ~erode(~m, n)

"""ZZZ hair highlight layers "+" (髮+, 发+, 发饰+…) baked into the texture they cover.

Measured on 09/11 over the 44 hairstyles in the workshop cache: 277,574 highlight faces, of which
266,769 repeat a hair face vertex for vertex, with the same UVs (Phionie even has them twice).
They are alpha overlays (the ring of light, alpha 0.6 at most) drawn over the hair: half the
faces of a hairstyle, for an image that a texture carries just as well.
So the overlay is composited onto the hair texture, under only the faces it covers, and its
duplicate faces are dropped; a highlight face without a duplicate stays with its material.

Blender is needed to read and write the images: without it nothing is baked and the layers stay.
"""
import hashlib
import os

import numpy as np

LAYER = "+"
PART_MIN = 0.99   # share of a layer's faces that must duplicate a hair face


def _key(pos, f):
    return tuple(sorted(tuple(np.round(pos[v], 5)) for v in f))


def layers(D, mask):
    """Duplicate highlight faces under `masque`: {"+" material: {duplicated material: [(+ face, duplicated face)]}}."""
    names, mf, F, pos = D.names_mat, D.mat_face, D.faces, D.pos
    plus = [i for i in np.unique(mf[mask]) if LAYER in names[i]]
    if not plus:
        return {}
    base = {}
    for f in np.where(mask & ~np.isin(mf, plus))[0]:
        base.setdefault(_key(pos, F[f]), f)
    out = {}
    for p in plus:
        fp = np.where(mask & (mf == p))[0]
        pairs = [(f, base.get(_key(pos, F[f]))) for f in fp]
        pairs = [(f, b) for f, b in pairs if b is not None]
        if len(pairs) < PART_MIN * len(fp):
            continue
        per = {}
        for f, b in pairs:
            per.setdefault(int(mf[b]), []).append((int(f), int(b)))
        out[int(p)] = per
    return out


def _texture(D, i):
    t = D.pmx["materials"][i]["texture"]
    if not 0 <= t < len(D.pmx["textures"]):
        return None
    c = os.path.normpath(os.path.join(os.path.dirname(D.path), D.pmx["textures"][t].replace("\\", "/")))
    return c if os.path.isfile(c) else None


def _fill(mask, uv, w, h):
    """Pixels (row 0 at the bottom, as in Blender) covered by the UV triangles."""
    for t in uv:
        x = t[:, 0] * w
        y = (1 - t[:, 1]) * h
        dx, dy = np.floor(x.min() / w) * w, np.floor(y.min() / h) * h
        x, y = x - dx, y - dy
        x0, x1 = int(np.floor(x.min())), int(np.ceil(x.max())) + 1
        y0, y1 = int(np.floor(y.min())), int(np.ceil(y.max())) + 1
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        (ax, bx, cx), (ay, by, cy) = x, y
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-12:
            continue
        l1 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / d
        l2 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / d
        e = 1.0 / max(1.0, np.abs(d)) ** 0.5   # half-pixel margin at the edge
        in_ = (l1 >= -e) & (l2 >= -e) & (1 - l1 - l2 >= -e)
        yy, xx = gy[in_].astype(int) % h, gx[in_].astype(int) % w
        mask[yy, xx] = True


def bake(D, mask, folder):
    """Bakes the highlight layers of the hairstyle (faces `masque` of D); writes the textures into `dossier`.

    Returns (highlight faces to remove, {duplicated material: path of its baked texture}).
    """
    removes = np.zeros(len(D.faces), bool)
    try:
        import tint as TE
    except ImportError:
        return removes, {}
    C = layers(D, mask)
    if not C:
        return removes, {}
    uv = np.asarray(D.t["uv"], float)
    # one baked texture per hair texture: all the overlays that cover it, under their faces
    per_tex = {}
    for p, per in C.items():
        tp = _texture(D, p)
        if tp is None:
            continue
        a_p = float(D.pmx["materials"][p]["diffuse"][3])
        for b, pairs in per.items():
            tb = _texture(D, b)
            if tb is None:
                continue
            e = per_tex.setdefault(tb, {"mats": set(), "texture_layers": []})
            e["mats"].add(b)
            e["texture_layers"].append((tp, a_p, [f for f, _ in pairs]))
            removes[[f for f, _ in pairs]] = True
    os.makedirs(folder, exist_ok=True)
    textures = {}
    for tb, e in per_tex.items():
        h_ = hashlib.sha1(tb.encode())
        for tp, a_p, fs in e["texture_layers"]:
            h_.update(("%s|%.4f|%d|%d" % (tp, a_p, len(fs), sum(fs))).encode())
        path = os.path.join(folder, "highlight-%s.png" % h_.hexdigest()[:16])
        if not os.path.isfile(path):
            img = TE.read_image(tb).copy()
            h, w = img.shape[:2]
            for tp, a_p, fs in e["texture_layers"]:
                cal = TE.read_image(tp)
                ch, cw = cal.shape[:2]
                if (ch, cw) != (h, w):   # nearest neighbour, at the size of the hair texture
                    cal = cal[(np.arange(h) * ch // h)[:, None], (np.arange(w) * cw // w)[None, :]]
                m = np.zeros((h, w), bool)
                _fill(m, uv[D.faces[fs]], w, h)
                a = (cal[..., 3] * a_p * m)[..., None]
                img[..., :3] = img[..., :3] * (1 - a) + cal[..., :3] * a
            tmp = path + ".tmp.png"
            TE.write_image(tmp, img)
            os.replace(tmp, path)
        for b in e["mats"]:
            textures[b] = path
    return removes, textures

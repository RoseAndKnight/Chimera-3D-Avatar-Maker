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

COUCHE = "+"
PART_MIN = 0.99   # share of a layer's faces that must duplicate a hair face


def _cle(pos, f):
    return tuple(sorted(tuple(np.round(pos[v], 5)) for v in f))


def couches(D, masque):
    """Duplicate highlight faces under `masque`: {"+" material: {duplicated material: [(+ face, duplicated face)]}}."""
    noms, mf, F, pos = D.noms_mat, D.mat_face, D.faces, D.pos
    plus = [i for i in np.unique(mf[masque]) if COUCHE in noms[i]]
    if not plus:
        return {}
    base = {}
    for f in np.where(masque & ~np.isin(mf, plus))[0]:
        base.setdefault(_cle(pos, F[f]), f)
    out = {}
    for p in plus:
        fp = np.where(masque & (mf == p))[0]
        paires = [(f, base.get(_cle(pos, F[f]))) for f in fp]
        paires = [(f, b) for f, b in paires if b is not None]
        if len(paires) < PART_MIN * len(fp):
            continue
        par = {}
        for f, b in paires:
            par.setdefault(int(mf[b]), []).append((int(f), int(b)))
        out[int(p)] = par
    return out


def _texture(D, i):
    t = D.pmx["materiaux"][i]["texture"]
    if not 0 <= t < len(D.pmx["textures"]):
        return None
    c = os.path.normpath(os.path.join(os.path.dirname(D.chemin), D.pmx["textures"][t].replace("\\", "/")))
    return c if os.path.isfile(c) else None


def _remplir(masque, uv, w, h):
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
        dans = (l1 >= -e) & (l2 >= -e) & (1 - l1 - l2 >= -e)
        yy, xx = gy[dans].astype(int) % h, gx[dans].astype(int) % w
        masque[yy, xx] = True


def cuire(D, masque, dossier):
    """Bakes the highlight layers of the hairstyle (faces `masque` of D); writes the textures into `dossier`.

    Returns (highlight faces to remove, {duplicated material: path of its baked texture}).
    """
    retire = np.zeros(len(D.faces), bool)
    try:
        import tint as TE
    except ImportError:
        return retire, {}
    C = couches(D, masque)
    if not C:
        return retire, {}
    uv = np.asarray(D.t["uv"], float)
    # one baked texture per hair texture: all the overlays that cover it, under their faces
    par_tex = {}
    for p, par in C.items():
        tp = _texture(D, p)
        if tp is None:
            continue
        a_p = float(D.pmx["materiaux"][p]["diffus"][3])
        for b, paires in par.items():
            tb = _texture(D, b)
            if tb is None:
                continue
            e = par_tex.setdefault(tb, {"mats": set(), "calques": []})
            e["mats"].add(b)
            e["calques"].append((tp, a_p, [f for f, _ in paires]))
            retire[[f for f, _ in paires]] = True
    os.makedirs(dossier, exist_ok=True)
    textures = {}
    for tb, e in par_tex.items():
        h_ = hashlib.sha1(tb.encode())
        for tp, a_p, fs in e["calques"]:
            h_.update(("%s|%.4f|%d|%d" % (tp, a_p, len(fs), sum(fs))).encode())
        chemin = os.path.join(dossier, "reflet-%s.png" % h_.hexdigest()[:16])
        if not os.path.isfile(chemin):
            img = TE.lire_image(tb).copy()
            h, w = img.shape[:2]
            for tp, a_p, fs in e["calques"]:
                cal = TE.lire_image(tp)
                ch, cw = cal.shape[:2]
                if (ch, cw) != (h, w):   # nearest neighbour, at the size of the hair texture
                    cal = cal[(np.arange(h) * ch // h)[:, None], (np.arange(w) * cw // w)[None, :]]
                m = np.zeros((h, w), bool)
                _remplir(m, uv[D.faces[fs]], w, h)
                a = (cal[..., 3] * a_p * m)[..., None]
                img[..., :3] = img[..., :3] * (1 - a) + cal[..., :3] * a
            tmp = chemin + ".tmp.png"
            TE.ecrire_image(tmp, img)
            os.replace(tmp, chemin)
        for b in e["mats"]:
            textures[b] = chemin
    return retire, textures

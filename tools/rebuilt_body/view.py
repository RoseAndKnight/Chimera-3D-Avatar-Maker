"""Grey views of a mesh, without Blender (numpy only): simple shading, edges optional.

    python3 -I view.py <pmx> <sortie.png> [--mat 肌] [--vues face,profil,dos] [--boite x0,y0,z0,x1,y1,z1 (cm)]
                       [--fil] [--taille 900]
Used to look at a body (reference or remade) when the Blender queue is busy.
"""
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
for d in ("atlas", "head", "rebuilt_head"):
    sys.path.insert(0, os.path.join(ICI, "..", d))

VUES = {"face": (0, 0), "trois": (35, 5), "profil": (90, 0), "dos": (180, 0), "arriere3": (145, 5),
        "dessous": (180, -55), "dessus": (0, 60), "dosbas": (180, -25), "profil2": (-90, 0)}


def base(az, el):
    """Camera frame (right, up, towards the eye); the model faces -z, y up."""
    az, el = np.radians(az), np.radians(el)
    oeil = np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)])
    droite = np.cross([0, 1, 0], oeil)
    droite /= np.linalg.norm(droite)
    haut = np.cross(oeil, droite)
    return droite, haut, oeil


def rendre(V, F, vue, T=900, boite=None, fil=False, couleur=None, lisse=True):
    """Image (T, T, 3) uint8. V in cm, F triangles. couleur: (n_faces, 3) between 0 and 1, or None."""
    d, h, o = base(*vue)
    P = np.stack([V @ d, V @ h, V @ o], 1)
    sel = np.ones(len(V), bool)
    if boite is not None:
        b = np.asarray(boite, float).reshape(2, 3)
        sel = np.all((V >= b[0]) & (V <= b[1]), 1)
    Fs = F[sel[F].all(1)]
    idf = np.where(sel[F].all(1))[0]
    if not len(Fs):
        return np.zeros((T, T, 3), np.uint8)
    u = np.unique(Fs)
    lo, hi = P[u, :2].min(0), P[u, :2].max(0)
    c, r = 0.5 * (lo + hi), 0.54 * float((hi - lo).max())
    X = (P[:, 0] - c[0]) / r * 0.5 * T + 0.5 * T
    Y = 0.5 * T - (P[:, 1] - c[1]) / r * 0.5 * T
    Z = P[:, 2]
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-12)[:, None]
    if lisse:
        vn = np.zeros_like(V)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]
    lum = np.array([-0.35, 0.45, 0.0]) @ np.stack([d, h, o]) * 0 + (d * -0.4 + h * 0.5 + o * 0.77)
    lum /= np.linalg.norm(lum)
    img = np.full((T, T, 3), 0.16, np.float32)
    zb = np.full((T, T), -1e9, np.float32)
    for n, (a, b_, c_) in enumerate(Fs):
        xs, ys, zs = X[[a, b_, c_]], Y[[a, b_, c_]], Z[[a, b_, c_]]
        x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())) + 1, T)
        y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())) + 1, T)
        if x0 >= x1 or y0 >= y1:
            continue
        den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(den) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not m.any():
            continue
        z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        zz = zb[y0:y1, x0:x1]
        m &= z > zz
        if not m.any():
            continue
        if lisse:
            nn = w0[..., None] * vn[a] + w1[..., None] * vn[b_] + w2[..., None] * vn[c_]
            nn /= np.maximum(np.linalg.norm(nn, axis=2), 1e-9)[..., None]
            e = nn @ lum
            dos = (nn @ o) < 0
        else:
            e = np.full(z.shape, fn[idf[n]] @ lum)
            dos = np.full(z.shape, fn[idf[n]] @ o < 0)
        e = 0.28 + 0.72 * np.clip(e, 0, 1)
        col = np.array([0.86, 0.80, 0.78]) if couleur is None else couleur[idf[n]]
        px = e[..., None] * col
        px = np.where(dos[..., None], np.array([0.55, 0.15, 0.15]) * 0.8, px)   # back side: dark red
        if fil:
            bord = np.minimum(np.minimum(w0, w1), w2)
            # distance in pixels to the closest edge (approximated by the triangle's height)
            aire2 = abs(den)
            L = np.array([np.hypot(xs[1] - xs[2], ys[1] - ys[2]), np.hypot(xs[2] - xs[0], ys[2] - ys[0]),
                          np.hypot(xs[0] - xs[1], ys[0] - ys[1])])
            dpx = np.minimum(np.minimum(w0 * aire2 / max(L[0], 1e-6), w1 * aire2 / max(L[1], 1e-6)), w2 * aire2 / max(L[2], 1e-6))
            px = np.where((dpx < 0.6)[..., None], px * 0.25, px)
        sub = img[y0:y1, x0:x1]
        sub[m] = px[m]
        zz[m] = z[m]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def charger(chemin, mats=("肌",)):
    import pmx as PMX
    P = PMX.lire(chemin)
    t = PMX.vers_tableaux(P)
    noms = [m["nom"] for m in P["materiaux"]]
    sel = np.isin(t["mat_face"], [noms.index(n) for n in mats if n in noms])
    return t["pos"] * 8.0, t["faces"][sel]


def planche(V, F, vues, sortie, T=900, boite=None, fil=False, couleur=None, lisse=True):
    import png
    ims = [rendre(V, F, VUES[v] if isinstance(v, str) else v, T, boite, fil, couleur, lisse) for v in vues]
    png.ecrire(sortie, np.concatenate(ims, 1))


if __name__ == "__main__":
    a = sys.argv[1:]

    def opt(nom, defaut=None):
        return a[a.index(nom) + 1] if nom in a else defaut
    V, F = charger(a[0], tuple(opt("--mat", "肌").split(",")))
    b = opt("--boite")
    planche(V, F, opt("--vues", "face,profil,dos").split(","), a[1], int(opt("--taille", "900")),
            [float(x) for x in b.split(",")] if b else None, "--fil" in a, lisse="--plat" not in a)

"""Grey views of a mesh, without Blender (numpy only): simple shading, edges optional.

    python3 -I view.py <pmx> <output.png> [--mat 肌] [--views face,profile,back] [--box x0,y0,z0,x1,y1,z1 (cm)]
                       [--fil] [--size 900]
Used to look at a body (reference or remade) when the Blender queue is busy.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("atlas", "head", "rebuilt_head"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

VIEWS = {"face": (0, 0), "three_quarter": (35, 5), "profile": (90, 0), "back": (180, 0), "back_three_quarter": (145, 5),
         "under": (180, -55), "over": (0, 60), "low_back": (180, -25), "profile_other": (-90, 0)}


def base(az, el):
    """Camera frame (right, up, towards the eye); the model faces -z, y up."""
    az, el = np.radians(az), np.radians(el)
    eye = np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)])
    right = np.cross([0, 1, 0], eye)
    right /= np.linalg.norm(right)
    top = np.cross(eye, right)
    return right, top, eye


def render(V, F, view, T=900, box=None, thread=False, color=None, smooth=True):
    """Image (T, T, 3) uint8. V in cm, F triangles. color: (n_faces, 3) between 0 and 1, or None."""
    d, h, o = base(*view)
    P = np.stack([V @ d, V @ h, V @ o], 1)
    sel = np.ones(len(V), bool)
    if box is not None:
        b = np.asarray(box, float).reshape(2, 3)
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
    if smooth:
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
        if smooth:
            nn = w0[..., None] * vn[a] + w1[..., None] * vn[b_] + w2[..., None] * vn[c_]
            nn /= np.maximum(np.linalg.norm(nn, axis=2), 1e-9)[..., None]
            e = nn @ lum
            back = (nn @ o) < 0
        else:
            e = np.full(z.shape, fn[idf[n]] @ lum)
            back = np.full(z.shape, fn[idf[n]] @ o < 0)
        e = 0.28 + 0.72 * np.clip(e, 0, 1)
        collar = np.array([0.86, 0.80, 0.78]) if color is None else color[idf[n]]
        px = e[..., None] * collar
        px = np.where(back[..., None], np.array([0.55, 0.15, 0.15]) * 0.8, px)   # back side: dark red
        if thread:
            border = np.minimum(np.minimum(w0, w1), w2)
            # distance in pixels to the closest edge (approximated by the triangle's height)
            area2 = abs(den)
            L = np.array([np.hypot(xs[1] - xs[2], ys[1] - ys[2]), np.hypot(xs[2] - xs[0], ys[2] - ys[0]),
                          np.hypot(xs[0] - xs[1], ys[0] - ys[1])])
            dpx = np.minimum(np.minimum(w0 * area2 / max(L[0], 1e-6), w1 * area2 / max(L[1], 1e-6)), w2 * area2 / max(L[2], 1e-6))
            px = np.where((dpx < 0.6)[..., None], px * 0.25, px)
        sub = img[y0:y1, x0:x1]
        sub[m] = px[m]
        zz[m] = z[m]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def load(path, mats=("肌",)):
    import pmx as PMX
    P = PMX.read(path)
    t = PMX.to_arrays(P)
    names = [m["name"] for m in P["materials"]]
    sel = np.isin(t["mat_face"], [names.index(n) for n in mats if n in names])
    return t["pos"] * 8.0, t["faces"][sel]


def sheet(V, F, views, output, T=900, box=None, thread=False, color=None, smooth=True):
    import png
    ims = [render(V, F, VIEWS[v] if isinstance(v, str) else v, T, box, thread, color, smooth) for v in views]
    png.write(output, np.concatenate(ims, 1))


if __name__ == "__main__":
    a = sys.argv[1:]

    def opt(name, default=None):
        return a[a.index(name) + 1] if name in a else default
    V, F = load(a[0], tuple(opt("--mat", "肌").split(",")))
    b = opt("--box")
    sheet(V, F, opt("--views", "face,profile,back").split(","), a[1], int(opt("--size", "900")),
            [float(x) for x in b.split(",")] if b else None, "--fil" in a, smooth="--flat" not in a)

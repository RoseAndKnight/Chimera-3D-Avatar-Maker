"""Images for the library of face parts (run by library.py inside Blender).

    blender -b --factory-startup --python library_images.py -- texture <library> <Character> [...]
    blender -b --factory-startup --python library_images.py -- composer <folder>

texture: the character's skin texture copied into the UVs of the pivot head
(<library>/<Character>/texture.png, at the size of the pivot texture), using the UV grid
sampled by library.py.
composer: the texture of the composed face (<dossier>/visage_compose.png): each part takes
its donor's texture, brought to the skin tone of the base (Lab median over the cheeks);
the pivot texture only where the base has none.
"""
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tint as TE  # noqa: E402  (lire_image, write_image, Lab)

REGIONS = ("eyes", "eyebrows", "nose", "mouth", "base")
GAP_UV = 3.0  # texels: beyond this, grid neighbours lie on either side of a seam
DILATION = 4  # texels added around the islands (mipmap filtering)


def enlarge(grid, ok, n):
    """Grid (m, m, k) sampled at texel centres -> (n, n, k), bilinear where the four
    neighbours are valid and close, nearest otherwise."""
    m = grid.shape[0]
    t = (np.arange(n) + 0.5) * m / n - 0.5
    i0 = np.clip(np.floor(t).astype(int), 0, m - 1)
    i1 = np.clip(i0 + 1, 0, m - 1)
    f = np.clip(t - i0, 0, 1).astype(np.float32)
    r0, r1, c0, c1 = i0[:, None], i1[:, None], i0[None, :], i1[None, :]
    fr, fc = f[:, None, None], f[None, :, None]
    a, b, c, d = grid[r0, c0], grid[r0, c1], grid[r1, c0], grid[r1, c1]
    assessment = (1 - fr) * ((1 - fc) * a + fc * b) + fr * ((1 - fc) * c + fc * d)
    near = grid[np.clip(np.round(t).astype(int), 0, m - 1)[:, None], np.clip(np.round(t).astype(int), 0, m - 1)[None, :]]
    okn = ok[np.clip(np.round(t).astype(int), 0, m - 1)[:, None], np.clip(np.round(t).astype(int), 0, m - 1)[None, :]]
    all_ = ok[r0, c0] & ok[r0, c1] & ok[r1, c0] & ok[r1, c1]
    return assessment, near, okn, all_, (a, b, c, d)


def sample(img, uv):
    """Texture (h, w, 4), Blender pixels (row 0 at the bottom), at the PMX UVs: bilinear."""
    h, w = img.shape[:2]
    x = np.mod(uv[..., 0], 1) * w - 0.5
    y = (1 - np.mod(uv[..., 1], 1)) * h - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0, x1 = np.mod(x0, w), np.mod(x0 + 1, w)
    y0, y1 = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    return (1 - fy) * ((1 - fx) * img[y0, x0] + fx * img[y0, x1]) + fy * ((1 - fx) * img[y1, x0] + fx * img[y1, x1])


def dilate(px, ok, n):
    px, ok = px.copy(), ok.copy()
    for _ in range(n):
        accessory = np.zeros_like(px)
        cnt = np.zeros(ok.shape, dtype=np.float32)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            o = np.roll(ok, (dr, dc), (0, 1))
            accessory += np.roll(px, (dr, dc), (0, 1)) * o[..., None]
            cnt += o
        fresh = ~ok & (cnt > 0)
        px[fresh] = accessory[fresh] / cnt[fresh][:, None]
        ok = ok | fresh
    return px


def texture(lib, names):
    for name in names:
        z = np.load(os.path.join(lib, name, "part.npz"))
        with open(os.path.join(lib, name, "record.json"), encoding="utf-8") as fh:
            card = json.load(fh)
        uv, ok = z["uv"], z["uv_ok"]
        # path relative to the character's folder (Outerplane heads), or absolute (ZZZ)
        img = TE.read_image(os.path.join(lib, name, card["texture"]))
        n = 2 * uv.shape[0]
        assessment, near, okn, all_, (a, b, c, d) = enlarge(uv, ok, n)
        h, w = img.shape[:2]
        px = np.array([w, h], dtype=np.float32)
        gap = np.max([np.abs((q - a) * px).max(-1) for q in (b, c, d)], axis=0)
        smooth = all_ & (gap < GAP_UV)
        uvn = np.where(smooth[..., None], assessment, near)
        out = sample(img, uvn).astype(np.float32)
        out[..., 3] = np.where(okn, 1.0, 0.0)
        out[~okn, :3] = 0
        TE.write_image(os.path.join(lib, name, "texture.png"), out)
        print("images : texture of %s" % name, flush=True)


def compose(folder):
    with open(os.path.join(folder, "recipe.json"), encoding="utf-8") as fh:
        r = json.load(fh)
    recipe, lib = r["recipe"], r["library"]
    z = np.load(os.path.join(folder, "texture_weights.npz"))
    piv = TE.read_image(r["texture_pivot"])
    n = piv.shape[0]
    up = lambda g: enlarge(g[..., None].astype(np.float32), np.ones(g.shape, bool), n)[0][..., 0]
    ok = enlarge(z["ok"][..., None].astype(np.float32), np.ones(z["ok"].shape, bool), n)[1][..., 0] > 0.5
    face = up(z["face"])
    weight = {p: up(z["weight_" + p]) for p in REGIONS}
    tex = {}
    for name in set(recipe.values()):
        t = TE.read_image(os.path.join(lib, name, "texture.png"))
        if t.shape[0] != n:
            raise RuntimeError("texture of %s: %d instead of %d" % (name, t.shape[0], n))
        tex[name] = t
    cheeks = ok & (weight["base"] > 0.95) & (face > 0.95)
    for name in tex.values():
        cheeks &= name[..., 3] > 0.5
    lab = {name: TE.srgb_to_lab(t[cheeks][:, :3].astype(float)) for name, t in tex.items()}
    med = {name: np.median(l, 0) for name, l in lab.items()}
    base = recipe["base"]
    comp = np.zeros((int(ok.sum()), 3))
    report = {}
    for p in REGIONS:
        name = recipe[p]
        c = tex[name][ok][:, :3].astype(float)
        if name != base:
            l = TE.srgb_to_lab(c) + (med[base] - med[name])
            c = TE.lab_to_srgb(l)
        report[p] = {"donor": name, "gap_ton": float(np.linalg.norm(med[base] - med[name]))}
        comp += weight[p][ok][:, None] * c
    # the base covers all the skin where its texture exists; the pivot head remains only where
    # the base has nothing (otherwise its pink cheeks show around the face)
    out = piv.copy()
    v = np.clip(tex[base][ok][:, 3:4], 0, 1)
    out[ok, :3] = (piv[ok, :3] + v * (comp - piv[ok, :3])).astype(np.float32)
    # beyond the skin islands, a border of the same colour (mipmap filtering)
    out[..., :3] = dilate(out[..., :3], ok, DILATION)
    TE.write_image(os.path.join(folder, "composed_face.png"), out)
    report["texels_cheeks"] = int(cheeks.sum())
    with open(os.path.join(folder, "texture.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print("images : face composed", flush=True)


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    if a[0] == "texture":
        texture(a[1], a[2:])
    else:
        compose(a[1])
    print("images: done", flush=True)


if __name__ == "__main__":
    main()

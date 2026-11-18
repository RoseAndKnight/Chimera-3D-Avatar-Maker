"""Images for the library of face parts (run by library.py inside Blender).

    blender -b --factory-startup --python library_images.py -- texture <biblio> <Personnage> [...]
    blender -b --factory-startup --python library_images.py -- composer <dossier>

texture: the character's skin texture copied into the UVs of the pivot head
(<biblio>/<Personnage>/texture.png, at the size of the pivot texture), using the UV grid
sampled by library.py.
composer: the texture of the composed face (<dossier>/visage_compose.png): each part takes
its donor's texture, brought to the skin tone of the base (Lab median over the cheeks);
the pivot texture only where the base has none.
"""
import json
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

import tint as TE  # noqa: E402  (lire_image, ecrire_image, Lab)

PARTIES = ("yeux", "sourcils", "nez", "bouche", "base")
ECART_UV = 3.0  # texels: beyond this, grid neighbours lie on either side of a seam
DILATATION = 4  # texels added around the islands (mipmap filtering)


def agrandir(grille, ok, n):
    """Grid (m, m, k) sampled at texel centres -> (n, n, k), bilinear where the four
    neighbours are valid and close, nearest otherwise."""
    m = grille.shape[0]
    t = (np.arange(n) + 0.5) * m / n - 0.5
    i0 = np.clip(np.floor(t).astype(int), 0, m - 1)
    i1 = np.clip(i0 + 1, 0, m - 1)
    f = np.clip(t - i0, 0, 1).astype(np.float32)
    r0, r1, c0, c1 = i0[:, None], i1[:, None], i0[None, :], i1[None, :]
    fr, fc = f[:, None, None], f[None, :, None]
    a, b, c, d = grille[r0, c0], grille[r0, c1], grille[r1, c0], grille[r1, c1]
    bil = (1 - fr) * ((1 - fc) * a + fc * b) + fr * ((1 - fc) * c + fc * d)
    near = grille[np.clip(np.round(t).astype(int), 0, m - 1)[:, None], np.clip(np.round(t).astype(int), 0, m - 1)[None, :]]
    okn = ok[np.clip(np.round(t).astype(int), 0, m - 1)[:, None], np.clip(np.round(t).astype(int), 0, m - 1)[None, :]]
    tous = ok[r0, c0] & ok[r0, c1] & ok[r1, c0] & ok[r1, c1]
    return bil, near, okn, tous, (a, b, c, d)


def echantillon(img, uv):
    """Texture (h, w, 4), Blender pixels (row 0 at the bottom), at the PMX UVs: bilinear."""
    h, w = img.shape[:2]
    x = np.mod(uv[..., 0], 1) * w - 0.5
    y = (1 - np.mod(uv[..., 1], 1)) * h - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0, x1 = np.mod(x0, w), np.mod(x0 + 1, w)
    y0, y1 = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    return (1 - fy) * ((1 - fx) * img[y0, x0] + fx * img[y0, x1]) + fy * ((1 - fx) * img[y1, x0] + fx * img[y1, x1])


def dilater(px, ok, n):
    px, ok = px.copy(), ok.copy()
    for _ in range(n):
        acc = np.zeros_like(px)
        cnt = np.zeros(ok.shape, dtype=np.float32)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            o = np.roll(ok, (dr, dc), (0, 1))
            acc += np.roll(px, (dr, dc), (0, 1)) * o[..., None]
            cnt += o
        neuf = ~ok & (cnt > 0)
        px[neuf] = acc[neuf] / cnt[neuf][:, None]
        ok = ok | neuf
    return px


def texture(biblio, noms):
    for nom in noms:
        z = np.load(os.path.join(biblio, nom, "partie.npz"))
        with open(os.path.join(biblio, nom, "fiche.json"), encoding="utf-8") as fh:
            fiche = json.load(fh)
        uv, ok = z["uv"], z["uv_ok"]
        # path relative to the character's folder (Outerplane heads), or absolute (ZZZ)
        img = TE.lire_image(os.path.join(biblio, nom, fiche["texture"]))
        n = 2 * uv.shape[0]
        bil, near, okn, tous, (a, b, c, d) = agrandir(uv, ok, n)
        h, w = img.shape[:2]
        px = np.array([w, h], dtype=np.float32)
        ecart = np.max([np.abs((q - a) * px).max(-1) for q in (b, c, d)], axis=0)
        lisse = tous & (ecart < ECART_UV)
        uvn = np.where(lisse[..., None], bil, near)
        out = echantillon(img, uvn).astype(np.float32)
        out[..., 3] = np.where(okn, 1.0, 0.0)
        out[~okn, :3] = 0
        TE.ecrire_image(os.path.join(biblio, nom, "texture.png"), out)
        print("images : texture of %s" % nom, flush=True)


def composer(dossier):
    with open(os.path.join(dossier, "recette.json"), encoding="utf-8") as fh:
        r = json.load(fh)
    recette, biblio = r["recette"], r["biblio"]
    z = np.load(os.path.join(dossier, "poids_texture.npz"))
    piv = TE.lire_image(r["texture_pivot"])
    n = piv.shape[0]
    up = lambda g: agrandir(g[..., None].astype(np.float32), np.ones(g.shape, bool), n)[0][..., 0]
    ok = agrandir(z["ok"][..., None].astype(np.float32), np.ones(z["ok"].shape, bool), n)[1][..., 0] > 0.5
    visage = up(z["visage"])
    poids = {p: up(z["poids_" + p]) for p in PARTIES}
    tex = {}
    for nom in set(recette.values()):
        t = TE.lire_image(os.path.join(biblio, nom, "texture.png"))
        if t.shape[0] != n:
            raise RuntimeError("texture of %s: %d instead of %d" % (nom, t.shape[0], n))
        tex[nom] = t
    joues = ok & (poids["base"] > 0.95) & (visage > 0.95)
    for nom in tex.values():
        joues &= nom[..., 3] > 0.5
    lab = {nom: TE.srgb_vers_lab(t[joues][:, :3].astype(float)) for nom, t in tex.items()}
    med = {nom: np.median(l, 0) for nom, l in lab.items()}
    base = recette["base"]
    comp = np.zeros((int(ok.sum()), 3))
    rapport = {}
    for p in PARTIES:
        nom = recette[p]
        c = tex[nom][ok][:, :3].astype(float)
        if nom != base:
            l = TE.srgb_vers_lab(c) + (med[base] - med[nom])
            c = TE.lab_vers_srgb(l)
        rapport[p] = {"donneur": nom, "ecart_ton": float(np.linalg.norm(med[base] - med[nom]))}
        comp += poids[p][ok][:, None] * c
    # the base covers all the skin where its texture exists; the pivot head remains only where
    # the base has nothing (otherwise its pink cheeks show around the face)
    out = piv.copy()
    v = np.clip(tex[base][ok][:, 3:4], 0, 1)
    out[ok, :3] = (piv[ok, :3] + v * (comp - piv[ok, :3])).astype(np.float32)
    # beyond the skin islands, a border of the same colour (mipmap filtering)
    out[..., :3] = dilater(out[..., :3], ok, DILATATION)
    TE.ecrire_image(os.path.join(dossier, "visage_compose.png"), out)
    rapport["texels_joues"] = int(joues.sum())
    with open(os.path.join(dossier, "texture.json"), "w", encoding="utf-8") as fh:
        json.dump(rapport, fh, ensure_ascii=False, indent=1)
    print("images : face composed", flush=True)


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    if a[0] == "texture":
        texture(a[1], a[2:])
    else:
        composer(a[1])
    print("images : fini", flush=True)


if __name__ == "__main__":
    main()

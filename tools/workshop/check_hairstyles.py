"""Check of the cached hairstyles against the skin the workshop shows: does the skin come through the hair?

The hair of the cache is placed once per head (`hairstyles.py`). This check does not trust that
placement: it rebuilds the skin exactly as the page draws it (body of the cache with the head's
neck seam and hidden faces, plus the head's skin) and looks at the head and shoulders from
twelve directions, the way the page does: one depth image of the skin, one of the hair.

A **hole** is a pixel where the skin is in front and the nearest hair lies inside it, less than
EPAISSEUR under its surface: a strand that should cover the skull, the neck or the shoulders passes
under the skin, and the page shows skin through the hair. Hair far behind the skin (the back hair
seen through the face), hair tucked behind an ear (outside the ear, behind it) and roots planted
under the scalp that other hair covers (there, the hair is in front) are no holes.

    python3 check_hairstyles.py                       all the hairstyles of the cache, first head of the catalogue
    python3 check_hairstyles.py --coiffures A,B --tete Miyabi --planche worst.png --max 8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact_sheet as PL  # noqa: E402  first: commun puts tools/clothes (another contact_sheet.py) ahead on the path
import common as C  # noqa: E402
import glb as GLB  # noqa: E402

EPAISSEUR = 0.02     # metres: hair less deep than this under the shown skin should have covered it
PIXEL = 0.0015       # metres per pixel of the depth images
CADRE = 0.56         # metres seen around the head (head, neck and shoulders)
VUE_CM2 = 2.0        # a hairstyle is flagged when one view shows more skin through it than this (a patch of 1.4 cm)
# directions of view: azimuth (0: face) and elevation, in degrees
VUES = [(a, e) for e in (0, 35) for a in (0, 60, 120, 180, 240, 300)]


def _tableaux(chemin):
    doc, b = GLB.lire(chemin)
    acc, vues = doc["accessors"], doc["bufferViews"]

    def tab(i, ty, n):
        a, v = acc[i], vues[acc[i]["bufferView"]]
        return np.frombuffer(b, ty, a["count"] * (1 if a.get("type") == "SCALAR" else n), v.get("byteOffset", 0)).reshape(-1, n)
    pr = doc["meshes"][0]["primitives"]
    V = tab(pr[0]["attributes"]["POSITION"], "<f4", 3).astype(float)
    prims = [(doc["materials"][p["material"]]["name"], tab(p["indices"], "<u4", 3).astype(np.int64)) for p in pr]
    return V, prims


def peau_affichee(tete):
    """(vertices, faces) in metres, glTF frame: the skin as the page draws it under `tete`, and the whole
    head with it (eyes, mouth, brows: opaque, they hide what is behind them)."""
    V, prims = _tableaux(os.path.join(C.CACHE, "corps", "corps.glb"))
    F = np.concatenate([f for _, f in prims])
    d = os.path.join(C.CACHE, "tetes", tete)
    try:
        s = np.fromfile(os.path.join(d, "corps.bin"), "<f4").astype(float)
        if len(s) == 6 * len(V):
            V = s[:3 * len(V)].reshape(-1, 3)
    except OSError:
        pass
    try:
        k = np.fromfile(os.path.join(d, "peau_cachee.bin"), "<u4").astype(np.int64)
        garde = np.ones(len(F), bool)
        garde[k[k < len(F)]] = False
        F = F[garde]
    except OSError:
        pass
    Vt, pt = _tableaux(os.path.join(d, "tete.glb"))
    Ft = np.concatenate([f for _, f in pt]) + len(V)
    return np.concatenate([V, Vt]), np.concatenate([F, Ft])


def coiffure_posee(nom, tete):
    d = os.path.join(C.CACHE, "coiffures", nom)
    V, prims = _tableaux(os.path.join(d, "coiffure.glb"))
    F = np.concatenate([f for _, f in prims])
    pose = os.path.join(d, tete + ".bin")
    if not os.path.isfile(pose):
        return None
    Q = np.fromfile(pose, "<f4").reshape(-1, 3).astype(float)
    return (Q, F) if len(Q) == len(V) else None


def repere(az, el):
    """Right, up and towards-the-camera axes of a view (glTF frame, the character faces +Z)."""
    a, e = np.radians(az), np.radians(el)
    vers = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    droite = np.cross([0.0, 1.0, 0.0], vers)
    droite /= np.linalg.norm(droite)
    return droite, np.cross(vers, droite), vers


def profondeur(V, F, centre, axes, cote, endroit=False):
    """Depth image (distance towards the camera, larger = nearer; -inf where nothing) of the
    triangles, by points spread over each face finer than a pixel. With `endroit`, also the image
    of the nearest surface's side (True where it faces the camera; False: seen from inside) and the
    depth of the nearest surface seen from inside (where the ray leaves the skin again)."""
    droite, haut, vers = axes
    P = V - centre
    X = np.stack([(P @ droite) / PIXEL + cote / 2, cote / 2 - (P @ haut) / PIXEL, P @ vers], 1)
    T = X[F]
    dans = (T[:, :, 0].max(1) >= 0) & (T[:, :, 0].min(1) < cote) & (T[:, :, 1].max(1) >= 0) & (T[:, :, 1].min(1) < cote)
    T = T[dans]
    z = np.full(cote * cote, -np.inf)
    zd, zr = np.full(cote * cote, -np.inf), np.full(cote * cote, -np.inf)
    if not len(T):
        return (z.reshape(cote, cote), np.zeros((cote, cote), bool), zr.reshape(cote, cote)) if endroit else z.reshape(cote, cote)
    # faces of the skin turn their front outwards (glb.vers_gltf): in image space, counter-clockwise
    aire = (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])
    L = np.maximum(np.linalg.norm(T[:, 1, :2] - T[:, 0, :2], axis=1),
                   np.maximum(np.linalg.norm(T[:, 2, :2] - T[:, 1, :2], axis=1), np.linalg.norm(T[:, 0, :2] - T[:, 2, :2], axis=1)))
    n = np.clip(np.ceil(L * 2).astype(int), 1, 400)
    for k in np.unique(n):
        t, dr = T[n == k], (aire[n == k] < 0)
        i, j = np.mgrid[0:k + 1, 0:k + 1]
        g = i + j <= k
        a, b = i[g] / k, j[g] / k
        p = t[:, None, 0] * (1 - a - b)[None, :, None] + t[:, None, 1] * a[None, :, None] + t[:, None, 2] * b[None, :, None]
        p = p.reshape(-1, 3)
        x, y = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
        ok = (x >= 0) & (x < cote) & (y >= 0) & (y < cote)
        np.maximum.at(z, y[ok] * cote + x[ok], p[ok, 2])
        if endroit:
            d = np.repeat(dr, len(a))[ok]
            np.maximum.at(zd, (y[ok] * cote + x[ok])[d], p[ok, 2][d])
            np.maximum.at(zr, (y[ok] * cote + x[ok])[~d], p[ok, 2][~d])
    z = z.reshape(cote, cote)
    return (z, zd.reshape(cote, cote) >= z - 1e-4, zr.reshape(cote, cote)) if endroit else z


class Scene:
    """The skin of a head, seen from every direction once."""

    def __init__(self, tete):
        self.tete = tete
        self.V, self.F = peau_affichee(tete)
        self.cote = int(round(CADRE / PIXEL))
        self._z = {}

    def cadre(self, Q):
        # centred on the top of the hair, a little lower: skull, neck and shoulders
        return np.array([0.0, float(Q[:, 1].max()) - CADRE * 0.42, float(np.median(Q[:, 2]))])

    def peau(self, centre, vue):
        cle = (tuple(np.round(centre, 3)), vue)
        if cle not in self._z:
            self._z[cle] = profondeur(self.V, self.F, centre, repere(*vue), self.cote, endroit=True)
        return self._z[cle]


def mesurer(nom, scene):
    cp = coiffure_posee(nom, scene.tete)
    if cp is None:
        return None
    Q, F = cp
    centre = scene.cadre(Q)
    vues, total = [], 0
    for vue in VUES:
        zp, endroit, sortie = scene.peau(centre, vue)
        zc = profondeur(Q, F, centre, repere(*vue), scene.cote)
        # the nearest hair is inside the skin: behind its outer side, before the ray leaves it again
        # (hair tucked behind an ear is behind the ear's far side: hidden on purpose, not inside).
        # Skin seen from inside (an opening of the skin) is another defect, not the hair's.
        trou = np.isfinite(zp) & endroit & np.isfinite(zc) & (zc < zp) & (zc > zp - EPAISSEUR) & (zc > sortie)
        n = int(trou.sum())
        total += n
        vues.append({"vue": vue, "pixels": n, "_zp": zp, "_zc": zc, "_trou": trou})
    cm2 = total * (PIXEL * 100) ** 2
    pire = max(vues, key=lambda v: v["pixels"])
    return {"coiffure": nom, "tete": scene.tete, "trou_cm2": round(cm2, 2),
            "pire_vue": list(pire["vue"]), "pire_cm2": round(pire["pixels"] * (PIXEL * 100) ** 2, 2),
            "touchee": pire["pixels"] * (PIXEL * 100) ** 2 > VUE_CM2, "_vues": vues}


def image(v):
    """What the page would show (skin, hair), holes in red, light by depth."""
    zp, zc, trou = v["_zp"], v["_zc"], v["_trou"]
    img = np.empty(zp.shape + (3,), float)
    img[:] = PL.FOND
    devant_c = np.isfinite(zc) & (zc >= np.where(np.isfinite(zp), zp, -np.inf))
    devant_p = np.isfinite(zp) & ~devant_c
    for masque, z, coul in ((devant_p, zp, PL.PEAU), (devant_c, zc, PL.CHEVEUX)):
        if masque.any():
            zz = z[masque]
            lum = 0.55 + 0.45 * (zz - zz.min()) / max(float(zz.max() - zz.min()), 1e-9)
            img[masque] = np.array(coul)[None, :] * lum[:, None]
    img[trou] = (230, 30, 30)
    return img.astype(np.uint8)


def planche(resultats, sortie, maxi=8, par_ligne=4):
    """One row per hairstyle, worst first: its four worst views."""
    lignes = []
    for r in resultats[:maxi]:
        vs = sorted(r["_vues"], key=lambda v: -v["pixels"])[:par_ligne]
        im = [image(v)[::2, ::2] for v in vs]
        for x in im:
            x[[0, -1], :] = x[:, [0, -1]] = 60
        lignes.append(np.concatenate(im, 1))
    if lignes:
        PL.ecrire_png(sortie, np.concatenate(lignes, 0))


def tete_par_defaut():
    cat = C.lire_json(os.path.join(C.CACHE, "catalogue.json")) or {}
    t = cat.get("tetes") or []
    return t[0]["id"] if t else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coiffures", help="names, comma-separated (default: the whole cache)")
    ap.add_argument("--tete", help="head the hair is placed on (default: first of the catalogue)")
    ap.add_argument("--planche", help="PNG of the worst hairstyles (four worst views each, holes in red)")
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--json", help="write the measures there")
    a = ap.parse_args()
    scene = Scene(a.tete or tete_par_defaut())
    dos = os.path.join(C.CACHE, "coiffures")
    noms = a.coiffures.split(",") if a.coiffures else sorted(
        n for n in os.listdir(dos) if os.path.isfile(os.path.join(dos, n, "coiffure.glb")))
    res = []
    for n in noms:
        r = mesurer(n, scene)
        if r is None:
            continue
        res.append(r)
        print("%-14s skin through the hair %6.2f cm2 (worst view %s: %5.2f cm2)%s" % (
            n, r["trou_cm2"], tuple(r["pire_vue"]), r["pire_cm2"], "  FLAGGED" if r["touchee"] else ""), flush=True)
    res.sort(key=lambda r: -r["trou_cm2"])
    touchees = [r for r in res if r["touchee"]]
    print("%d / %d hairstyles show skin through the hair on %s (over %.1f cm2 in one view)" % (len(touchees), len(res), scene.tete, VUE_CM2))
    if a.json:
        C.ecrire_json(a.json, {"tete": scene.tete, "epaisseur_m": EPAISSEUR, "seuil_vue_cm2": VUE_CM2,
                               "coiffures": [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]})
    if a.planche:
        planche(res, a.planche, a.max)
        print(a.planche)


if __name__ == "__main__":
    main()

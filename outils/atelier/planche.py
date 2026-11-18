"""Contact sheet of the review: one image of each flagged spot, to judge at a glance without opening Chimera.

A small drawing by program (numpy only, no textures): the skin, each part in one colour, the
hair, seen straight on at the spot (from outside the body), 30 cm wide, the spot circled in red.
The back of a face is darkened: a part turned inside out or fabric seen from inside stands out.

    python3 planche.py --pieces a,b,c [--coiffure name] [--tete name] --sortie planche.png [--max 12]
    python3 planche.py --piece id --sortie planche.png                      a part's own defects, alone on the body
    python3 planche.py --tour --pieces a,b --sortie tour.png                overall view, six directions
    python3 planche.py --json sondage.json --sortie folder/ [--max 12]       one contact sheet per faulty outfit
"""
import argparse
import json
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun as C  # noqa: E402
import combinaison as CB  # noqa: E402
import revue as RV  # noqa: E402

COTE = 300          # pixels of a thumbnail
LARGEUR = 0.30      # metres seen
PEAU = (224, 172, 140)
CHEVEUX = (70, 50, 110)
TEINTES = [(90, 150, 220), (240, 240, 235), (70, 170, 120), (230, 190, 70), (200, 110, 180), (120, 120, 130),
           (230, 130, 80), (60, 190, 200)]
FOND = (245, 240, 230)


def ecrire_png(chemin, img):
    h, w, _ = img.shape
    brut = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def bloc(nom, d):
        return struct.pack(">I", len(d)) + nom + d + struct.pack(">I", zlib.crc32(nom + d) & 0xffffffff)
    with open(chemin, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + bloc(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + bloc(b"IDAT", zlib.compress(brut, 6)) + bloc(b"IEND", b""))


def scene(couches, coif=None, tete=None, cachee=None):
    """[(vertices, faces, colour)] of what Chimera shows: unhidden skin, adjusted parts, hair."""
    objets = []
    pe = CB.peau()
    corps_d, poses = CB.poser_tenue(couches)       # feet posed in a heeled shoe, as on the page
    if pe is not None:
        V, F, _, _ = pe
        V = V + corps_d if corps_d is not None else V
        garde = np.ones(len(F), bool)
        for c in couches:
            try:
                k = cachee[c.id] if cachee and c.id in cachee else np.fromfile(os.path.join(C.CACHE, "pieces", c.id, "peau_cachee.bin"), "<u4")
                garde[k[k < len(F)]] = False
            except OSError:
                pass
        objets.append((V, F[garde], PEAU))
    for i, c in enumerate(couches):
        objets.append((poses.get(c.id, c.Z), c.F[~c.masque], TEINTES[i % len(TEINTES)]))
    if coif:
        V, F = RV.coiffure(coif, tete or coif, couches)
        objets.append((V, F, CHEVEUX))
    return objets


def regard(lieu):
    """Direction from which to look at a spot: from outside the body, horizontally."""
    cp = CB.corps()
    d = np.array([0.0, 0.0, 1.0])
    if cp is not None:
        q, n = CB._pied_corps(np.array([lieu], float), cp)
        v = np.array(lieu, float) - q[0]
        v = v if np.linalg.norm(v) > 0.01 and (v * n[0]).sum() > 0 else n[0]
        v[1] = 0.0
        if np.linalg.norm(v) > 1e-6:
            d = v / np.linalg.norm(v)
    return d


def vignette(objets, lieu, d=None, cote=COTE, largeur=LARGEUR, cercle=True):
    lieu = np.array(lieu, float)
    d = regard(lieu) if d is None else d
    haut = np.array([0.0, 1.0, 0.0])
    droite = np.cross(haut, d)
    droite /= np.linalg.norm(droite)
    img = np.empty((cote, cote, 3), float)
    img[:] = FOND
    zb = np.full((cote, cote), -np.inf)
    ech = cote / largeur
    for V, F, coul in objets:
        if not len(F):
            continue
        P = V - lieu
        X = np.stack([(P * droite).sum(1) * ech + cote / 2, cote / 2 - (P * haut).sum(1) * ech, (P * d).sum(1)], 1)
        T = X[F]
        vu = (T[:, :, 0].max(1) >= 0) & (T[:, :, 0].min(1) < cote) & (T[:, :, 1].max(1) >= 0) & (T[:, :, 1].min(1) < cote)
        T = T[vu]
        nz = np.cross(V[F[vu]][:, 1] - V[F[vu]][:, 0], V[F[vu]][:, 2] - V[F[vu]][:, 0])
        nz /= np.maximum(np.linalg.norm(nz, axis=1), 1e-12)[:, None]
        face = (nz * d).sum(1)
        for t, c in zip(T, face):
            x0, x1 = max(int(np.floor(t[:, 0].min())), 0), min(int(np.ceil(t[:, 0].max())), cote - 1)
            y0, y1 = max(int(np.floor(t[:, 1].min())), 0), min(int(np.ceil(t[:, 1].max())), cote - 1)
            if x1 < x0 or y1 < y0:
                continue
            den = (t[1, 1] - t[2, 1]) * (t[0, 0] - t[2, 0]) + (t[2, 0] - t[1, 0]) * (t[0, 1] - t[2, 1])
            if abs(den) < 1e-9:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            a = ((t[1, 1] - t[2, 1]) * (xs - t[2, 0]) + (t[2, 0] - t[1, 0]) * (ys - t[2, 1])) / den
            b = ((t[2, 1] - t[0, 1]) * (xs - t[2, 0]) + (t[0, 0] - t[2, 0]) * (ys - t[2, 1])) / den
            g = 1 - a - b
            z = a * t[0, 2] + b * t[1, 2] + g * t[2, 2]
            zz = zb[y0:y1 + 1, x0:x1 + 1]
            ok = (a >= 0) & (b >= 0) & (g >= 0) & (z > zz)
            if not ok.any():
                continue
            lum = (0.45 + 0.55 * abs(c)) * (1.0 if c > 0 else 0.55)   # back side darkened
            zz[ok] = z[ok]
            img[y0:y1 + 1, x0:x1 + 1][ok] = np.array(coul) * lum
    yy, xx = np.mgrid[0:cote, 0:cote]
    r = np.hypot(xx - cote / 2, yy - cote / 2)
    if cercle:
        img[np.abs(r - 0.05 * ech) < 1.0] = (230, 30, 30)
    img[[0, -1], :] = img[:, [0, -1]] = 60
    return img.astype(np.uint8)


def tour(ids, coif, tete, sortie, cote=420):
    """Overall view of an outfit: front, three-quarter, profile, back, framed on the parts."""
    pieces = {i: CB.piece(i) for i in ids}
    ret = CB.retirees({i: p.meta for i, p in pieces.items()})
    couches, _ = CB.resoudre([p for i, p in pieces.items() if i not in ret])
    objets = scene(couches, coif, tete)
    P = np.concatenate([c.Z for c in couches])
    centre = (P.min(0) + P.max(0)) / 2
    largeur = float(max((P.max(0) - P.min(0)).max() * 1.15, 0.4))
    vues = [np.array([np.sin(t), 0.0, np.cos(t)]) for t in (0.0, 0.8, 1.57, 3.14, -1.57, -0.8)]
    img = np.concatenate([_sans_cercle(objets, centre, d, cote, largeur) for d in vues], 1)
    ecrire_png(sortie, img)


def _sans_cercle(objets, centre, d, cote, largeur):
    return vignette(objets, centre, d, cote, largeur, cercle=False)


def planche(ids, coif, tete, sortie, defauts=None, maxi=12):
    """Writes the contact sheet of an outfit's spots (worst first); returns the legend, one line per cell."""
    pieces = {i: CB.piece(i) for i in ids}
    ret = CB.retirees({i: p.meta for i, p in pieces.items()})
    couches, _ = CB.resoudre([p for i, p in pieces.items() if i not in ret])
    if defauts is None:
        defauts = RV.revoir_tenue(ids, coif, tete)["defauts"]
    defauts = [d for d in defauts if d.get("lieu")][:maxi]
    if not defauts:
        return []
    objets = scene(couches, coif, tete)
    cases = [vignette(objets, d["lieu"]) for d in defauts]
    col = min(4, len(cases))
    lig = -(-len(cases) // col)
    img = np.full((lig * COTE, col * COTE, 3), 255, np.uint8)
    for k, v in enumerate(cases):
        img[(k // col) * COTE:(k // col + 1) * COTE, (k % col) * COTE:(k % col + 1) * COTE] = v
    ecrire_png(sortie, img)
    noms = {c.id: TEINTES[i % len(TEINTES)] for i, c in enumerate(couches)}
    return ["cell %d (row %d, column %d): %s %s %s" % (k + 1, k // col + 1, k % col + 1, d["genre"], "+".join(d.get("pieces", [])), d["lieu"])
            for k, d in enumerate(defauts)] + ["colours: " + ", ".join("%s rgb%s" % (i, t) for i, t in noms.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces")
    ap.add_argument("--piece")
    ap.add_argument("--tour", action="store_true", help="overall view (six directions) instead of the flagged spots")
    ap.add_argument("--coiffure")
    ap.add_argument("--tete")
    ap.add_argument("--json")
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--max", type=int, default=12)
    a = ap.parse_args()
    if a.tour:
        tour((a.pieces or a.piece).split(","), a.coiffure, a.tete, a.sortie)
        return
    if a.piece:
        r = RV.revoir_piece(a.piece)
        d = sorted(r["defauts"], key=lambda x: -x["cm"] * RV.GRAVE.get(x["genre"], 1))
        print("\n".join(planche([a.piece], a.coiffure, a.tete, a.sortie, d, a.max)))
        return
    if a.pieces:
        print("\n".join(planche(a.pieces.split(","), a.coiffure, a.tete, a.sortie, maxi=a.max)))
        return
    d = json.load(open(a.json, encoding="utf-8"))
    os.makedirs(a.sortie, exist_ok=True)
    for k, t in enumerate(d["tenues"] if isinstance(d, dict) else d):
        if t["defauts"]:
            f = os.path.join(a.sortie, "%02d.png" % k)
            print(f, " + ".join(t["pieces"]))
            print("\n".join("  " + x for x in planche(t["pieces"], d.get("coiffure") if isinstance(d, dict) else None,
                                                     d.get("tete") if isinstance(d, dict) else None, f, t["defauts"], a.max)))


if __name__ == "__main__":
    main()

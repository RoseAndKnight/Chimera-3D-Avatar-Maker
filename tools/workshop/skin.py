"""The skin a part hides, recomputed in the cache, without fitting the part again.

    skin.py mesurer [ids]     what would change, part by part (cm² given back, cm² newly hidden)
    skin.py appliquer [ids]   rewrites peau_cachee.bin (the old one stays next to it: peau_cachee.avant.bin)

A skin face is hidden if, from its three vertices and its centre, going out along its
normal and four directions at 45° around it, each ray meets a face of the part that
really hides: held to the body AND opaque (couvre.bin); a sideways look that the body itself
stops does not count; and this must still hold once the part is adjusted. Fitting looked only at the geometry: an open jacket whose opening
is made by the transparency of its texture hid the chest, and the inside of the back showed
through the body. The ray starts SOUS below the skin: where the fabric goes slightly under the skin,
the skin is what is hidden, not the fabric.
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import outfit as CB  # noqa: E402
import rays as R  # noqa: E402

PORTEE = 0.05   # beyond this, the part floats: the skin is seen from underneath
BIAIS = 1.0     # tangent of the angle of the sideways looks (45°)
DECOLLE = 0.001  # a sideways look starts this far above the skin so as not to hit its own face
CHAUSSANTS = ("chaussures", "bottes")
HAUT_PIED = 0.15   # height of the foot in a shoe: above it is the leg
SOUS_PIED = 0.05   # a shoe does not deform: the foot that pokes through it (up to this) is hidden, not the shoe
ILOT = 20.0        # cm²: a smaller island of skin, isolated at the foot of a shoe, is a crumb
DEHORS = 0.25      # share of an isolated island outside the shoe: beyond this, it is the bare foot poking through it
BORD_PIED = 0.02   # the foot sticks out at most this much on the sides of a shoe (three times as much in front and behind)
GANTS = ("gants",)
FOURREAU = 0.03    # a glove finger does not deform: skin with both of its walls under it within this distance is the finger poking through it
ENFONCE = 0.001    # the fabric is under the skin from this depth on (PEAU_MIN of the review)
SOUS = 0.02     # depth under the skin where the ray starts: the one where the review looks for fabric (PEAU_PORTEE)


def _peau():
    V, F, N, _ = CB.peau()
    T = V[F]
    g = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    aire = np.linalg.norm(g, axis=1) / 2 * 1e4
    nf = N[F].sum(1)
    nf /= np.maximum(np.linalg.norm(nf, axis=1), 1e-12)[:, None]
    return T, nf, aire


_GRILLE = []


def _corps():
    if not _GRILLE:
        V, F, _, _ = CB.peau()
        _GRILLE.append(R.Grille(V, F))
    return _GRILLE[0]


def cachantes(fl, n):
    """Faces of a part that hide the skin: held and opaque. Without couvre.bin, all of them."""
    if fl is None:
        return np.ones(n, bool)
    return ((fl & CB.TENUE) > 0) & ((fl & CB.OPAQUE) > 0)


def couverte(surfaces, parmi=None, portee=PORTEE, sous=SOUS, biais=True):
    """Per skin face: hidden by one of the surfaces [(vertices, hiding faces)]. `parmi`:
    indices of the skin faces to examine (the others return False); `portee`: how far to search;
    `sous`: depth under the skin where the look starts; without `biais`, the straight look only (a
    shoe hugs the foot: nothing shows from under its edge)."""
    T, nf, _ = _peau()
    ou = np.arange(len(T)) if parmi is None else np.asarray(parmi, np.int64)
    res = np.zeros(len(T), bool)
    if not len(ou):
        return res
    pts = np.concatenate([T[ou], T[ou].mean(1)[:, None]], axis=1)          # (n, 4, 3)
    # Straight and sideways: under fabric held away from the body (a jacket at 4 cm), the skin near the
    # edge shows from underneath. Hidden only if none of these looks reaches it.
    n = nf[ou]
    t1 = np.cross(n, np.where(np.abs(n[:, [1]]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]))
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(n, t1)
    dirs = [n] + [(n + BIAIS * (c * t1 + s * t2)) / np.hypot(1.0, BIAIS)
                  for c, s in ((1, 0), (-1, 0), (0, 1), (0, -1)) if biais]
    D = np.stack(dirs, 1)                                                   # (n, 5, 3)
    D = np.repeat(D[:, None], 4, axis=1).reshape(-1, 3)                     # (n*4*5, 3)
    O = np.repeat(pts.reshape(-1, 3), len(dirs), axis=0) - sous * D
    long = (portee + sous) * np.tile([1.0] + [np.hypot(1.0, BIAIS)] * (len(dirs) - 1), len(ou) * 4)
    touche = np.zeros(len(O), bool)   # per ray: two parts can hide a face between them
    for Z, F in surfaces:
        k = np.where(~touche)[0]
        if not len(F) or not len(k):
            continue
        k = k[CB.voisinage(O[k], Z[F].reshape(-1, 3))]
        if len(k):
            touche[k] = G_tirer(Z, F, O[k], D[k], long[k])
    # A sideways look that the body itself stops (in the cleft of the buttocks, between the breasts)
    # is not one: it does not make the skin visible.
    nd = len(dirs)
    face = touche.reshape(-1, 4, nd)[:, :, 0].all(1)        # those the part covers straight on
    k = np.where(~touche & np.repeat(face, 4 * nd) & (np.tile(np.arange(nd), len(ou) * 4) > 0))[0]
    if len(k):
        touche[k] = np.isfinite(_corps().tirer(O[k] + (sous + DECOLLE) * D[k], D[k], long[k] - sous)[0])
    res[ou] = touche.reshape(-1, 4 * len(dirs)).all(1)
    return res


def G_tirer(Z, F, O, D, long):
    """Per ray: hits the surface before its own length."""
    return R.Grille(Z, F).tirer(O, D, float(long.max()))[0] <= long


def _au_pied(Z):
    """Per shod foot: indices of the skin faces in the box of the shoe, below its top
    and within HAUT_PIED of its lowest point."""
    cen = _peau()[0].mean(1)
    for cote in (Z[:, 0] < 0, Z[:, 0] >= 0):
        if not cote.any():
            continue
        a, b = Z[cote].min(0), Z[cote].max(0)
        yield np.where((cen[:, 0] > a[0] - BORD_PIED) & (cen[:, 0] < b[0] + BORD_PIED) & (cen[:, 1] < min(b[1], a[1] + HAUT_PIED))
                       & (cen[:, 2] > a[2] - 3 * BORD_PIED) & (cen[:, 2] < b[2] + 3 * BORD_PIED))[0]


def hors_semelle(Z, F):
    """Per skin face: at the foot of a shoe with nothing of it underneath. A shoe is placed
    as one block: the bare foot of the body has neither its length nor its arch, and what sticks out
    (toes in front of the tip, sole under an arched sole) does not show."""
    T = _peau()[0]
    cen = T.mean(1)
    res = np.zeros(len(T), bool)
    g = R.Grille(Z, F)
    for ou in _au_pied(Z):
        if len(ou):
            t = g.tirer(cen[ou], np.tile([0.0, -1.0, 0.0], (len(ou), 1)), HAUT_PIED + 0.1)[0]
            res[ou[~np.isfinite(t)]] = True
    return res


def miettes(Z, cachee, Fp=None):
    """Per skin face: at the foot of a shoe, still shown, in an island cut off from the rest of
    the leg, and smaller than ILOT (toe tips poking through a closed shoe) or, with the
    faces Fp of the shoe, outside it for more than DEHORS of its area (toes of the flat foot
    poking through the front of a heeled boot)."""
    V, F, _, _ = CB.peau()
    T, nf, aire = _peau()
    g = R.Grille(Z, Fp) if Fp is not None else None
    res = np.zeros(len(F), bool)
    montre = np.ones(len(F), bool)
    montre[cachee] = False
    for ou in _au_pied(Z):
        zone = np.zeros(len(F), bool)
        zone[ou] = True
        f = np.where(zone & montre)[0]
        if not len(f):
            continue
        libre = np.zeros(len(V), bool)            # vertices of a shown face outside the zone: the leg
        libre[F[montre & ~zone].ravel()] = True
        e, tient = _ilots(f, libre)
        a = np.bincount(e, aire[f], minlength=len(V))
        petit = a[e] < ILOT
        if g is not None:
            libre_ = ~np.isfinite(g.tirer(T[f].mean(1) + DECOLLE * nf[f], nf[f], 0.5)[0])
            petit |= np.bincount(e, aire[f] * libre_, minlength=len(V))[e] > DEHORS * a[e]
        res[f[~tient[e] & petit]] = True
    return res


def _ilots(f, libre):
    """Components (by shared vertices) of the skin faces f: (label per face of f, per
    label: the component touches a `libre` vertex)."""
    V, F, _, _ = CB.peau()
    et = np.arange(len(V))
    for _ in range(200):
        m = et[F[f]].min(1)
        avant = et.copy()
        for c in range(3):
            np.minimum.at(et, F[f, c], m)
        if (et == avant).all():
            break
    e = et[F[f, 0]]
    tient = np.zeros(len(V), bool)
    tient[e[libre[F[f]].any(1)]] = True
    return e, tient


def enfoncee(Z, F, cachee):
    """Per skin face: still shown, in an island the part covers entirely straight on, cut off
    from any other shown skin, and under which the part passes (from ENFONCE to SOUS) somewhere. This
    island was kept only because of a sideways look slipping past a nearby edge, and it pokes through the
    fabric (the tip of a breast in front of the narrow bib of an apron). Where it is the edge of the fabric
    that dips under the skin (a skirt waistband at the back), the skin is attached to the rest: it stays."""
    V, Fp, _, _ = CB.peau()
    T, nf, _ = _peau()
    res = np.zeros(len(T), bool)
    montre = np.ones(len(T), bool)
    montre[cachee] = False
    f = np.where(montre & couverte([(Z, F)], np.where(montre)[0], biais=False))[0]
    if not len(f):
        return res
    reste = montre.copy()
    reste[f] = False
    libre = np.zeros(len(V), bool)
    libre[Fp[reste].ravel()] = True
    e, tient = _ilots(f, libre)
    t = R.Grille(Z, F).tirer(T[f].mean(1), -nf[f], SOUS)[0]
    dessous = np.zeros(len(V), bool)
    dessous[e[np.isfinite(t) & (t > ENFONCE)]] = True
    res[f[~tient[e] & dessous[e]]] = True
    return res


def dans_fourreau(Z, F, cachee):
    """Per skin face: still shown, with the two walls of a glove sheath under it within
    FOURREAU (the near one within SOUS), or in an island cut off from the arm with more than
    DEHORS of its area in that state. The body's hand has neither the thickness nor the pose of the glove's:
    the finger poking through its glove finger does not show, and does not half show. On a
    back of the hand left bare, the skin has only the palm of the glove under it (one wall): it stays."""
    V, Fp, _, _ = CB.peau()
    T, nf, aire = _peau()
    res = np.zeros(len(T), bool)
    montre = np.ones(len(T), bool)
    montre[cachee] = False
    cen = T.mean(1)
    f = np.where(montre)[0]
    f = f[CB.voisinage(cen[f], Z[F].reshape(-1, 3))]
    if not len(f):
        return res
    g = R.Grille(Z, F)
    t1 = g.tirer(cen[f], -nf[f], SOUS)[0]
    a = np.where(np.isfinite(t1) & (t1 > ENFONCE))[0]
    dedans = np.zeros(len(f), bool)
    if len(a):
        t2 = g.tirer(cen[f[a]] - (t1[a] + 1e-4)[:, None] * nf[f[a]], -nf[f[a]], FOURREAU)[0]
        dedans[a[np.isfinite(t2) & (t1[a] + t2 <= FOURREAU)]] = True
    zone = np.zeros(len(T), bool)
    zone[f] = True
    libre = np.zeros(len(V), bool)            # vertices of a shown face far from the glove: the arm
    libre[Fp[montre & ~zone].ravel()] = True
    e, tient = _ilots(f, libre)
    part = np.bincount(e, aire[f] * dedans, minlength=len(V)) > DEHORS * np.bincount(e, aire[f], minlength=len(V))
    res[f[dedans | (~tient[e] & part[e])]] = True
    return res


def de(p):
    """Indices of the skin faces the part hides, fitted alone: at rest, and again once
    adjusted. Adjusting pulls the free flaps out of the skin and drags the nearby fabric (up to 2 cm):
    under an edge moved away like this, the skin shows again sideways; skin with fabric under it stays
    hidden. A shoe also hides the foot sticking out of its sole, a glove the finger that
    pokes through its glove finger."""
    F = p.F[cachantes(p.drapeaux, len(p.F))]
    pied = (p.meta or {}).get("type") in CHAUSSANTS
    sous = SOUS_PIED if pied else SOUS
    k = np.where(couverte([(p.pos, F)], sous=sous, biais=not pied))[0]
    Z = CB.resoudre([p])[0][0].Z
    if len(k) and np.abs(Z - p.pos).max() > 1e-4:
        k = k[couverte([(Z, F)], k, PORTEE + CB.SORTIE_MAX + 0.002, sous, not pied)[k]]
    k = np.union1d(k, np.where(enfoncee(Z, F, k))[0])
    if pied:
        k = np.union1d(k, np.where(hors_semelle(Z, p.F))[0])
        k = np.union1d(k, np.where(miettes(Z, k, p.F))[0])
    if (p.meta or {}).get("type") in GANTS:
        k = np.union1d(k, np.where(dans_fourreau(Z, F, k))[0])
    return k.astype(np.uint32)


def de_piece(ident):
    return de(CB.piece(ident))


def lire(ident):
    try:
        return np.fromfile(os.path.join(C.CACHE, "pieces", ident, "peau_cachee.bin"), "<u4")
    except OSError:
        return np.zeros(0, np.uint32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("commande", choices=("mesurer", "appliquer"))
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args()
    import review as RV
    _, _, aire = _peau()
    rendu = plus = 0.0
    for i in a.ids or RV.ids_du_cache(True):
        d = os.path.join(C.CACHE, "pieces", i)
        if (C.lire_json(os.path.join(d, "meta.json")) or {}).get("classe") != "vetement":
            continue   # an accessory hides no skin
        try:
            avant, apres = lire(i), de_piece(i)
        except Exception as e:
            print("%-34s not recomputed: %s" % (i, str(e)[:120]))
            continue
        avant = avant[avant < len(aire)]
        r = float(aire[np.setdiff1d(avant, apres)].sum())
        s = float(aire[np.setdiff1d(apres, avant)].sum())
        rendu, plus = rendu + r, plus + s
        if r >= 1 or s >= 1:
            print("%-34s skin given back %7.1f cm²   newly hidden %7.1f cm²" % (i, r, s))
        if a.commande == "appliquer" and (len(avant) != len(apres) or (avant != apres).any()):
            ch = os.path.join(d, "peau_cachee.bin")
            if os.path.isfile(ch) and not os.path.isfile(os.path.join(d, "peau_cachee.avant.bin")):
                shutil.copyfile(ch, os.path.join(d, "peau_cachee.avant.bin"))
            C.ecrire_octets(ch, apres.tobytes())
            m = C.lire_json(os.path.join(d, "meta.json"))
            m["peau_cachee"] = int(len(apres))
            C.ecrire_json(os.path.join(d, "meta.json"), m)
    print("total: skin given back %.0f cm², newly hidden %.0f cm²" % (rendu, plus))


if __name__ == "__main__":
    main()

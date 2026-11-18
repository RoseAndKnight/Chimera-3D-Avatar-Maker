"""Check of an outfit's layers, independent of the layer settling (`combinaison.py`).

The settling casts rays from the vertices, along their line; this check looks at whole
triangles. It asks two exact questions, on the triangles as they are displayed (masked faces
removed):

- **crossings**: does an edge of one part go through a face of the other? Two surfaces with no
  crossing at all cannot poke through each other;
- **outside**: from points scattered on the faces of the lower part, does a ray cast towards the
  body meet the upper part? If so, that point is in front of it: it shows although it should be
  covered. The depth is the length of the ray. A free flap far in front (a frill, a flared skirt
  hem under a jacket) is counted separately: it flares over on purpose.

    python3 controle_couches.py --pieces Atelier__pull,Lucy__veste     one outfit
    python3 controle_couches.py                                        the reference outfits
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combinaison as CB  # noqa: E402
import rayons as R  # noqa: E402
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jeux  # noqa: E402

LONGUEUR = 0.06     # ray towards the body: beyond it, the upper part is no longer "just behind"

# Reference outfits: what is really put together in the workshop, from tight layers to loose ones. They name
# catalogue parts (perso__id), so from the active game and the house: description outils/jeux/<jeu>/jeu.json,
# "etudes.tenues_couches".
TENUES = jeux.actif().valeur("etudes.tenues_couches", {})


def semis(V, F):
    """Four points per face (the centre and three inner points) and the area each one stands for."""
    T = V[F]
    aire = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    W = np.array([[1, 1, 1], [4, 1, 1], [1, 4, 1], [1, 1, 4]], float)
    W /= W.sum(1)[:, None]
    P = np.einsum("kj,fjc->fkc", W, T).reshape(-1, 3)
    return P, np.repeat(aire / 4, 4)


def paire(L, U, cp):
    """Check of part L under part U. None if they do not overlap."""
    Fl, Fu = L.F[~L.masque], U.F[~U.masque]
    if not len(Fl) or not len(Fu) or not CB._boite(U.Z, L.Z, 0.02):
        return None
    gu = R.Grille(U.Z, Fu)
    X = np.concatenate([R.croisements(L.Z, Fl, U.Z, Fu, gu), R.croisements(U.Z, Fu, L.Z, Fl)])
    P, aire = semis(L.Z, Fl)
    pres = CB.voisinage(P, U.Z)
    P, aire = P[pres], aire[pres]
    if not len(P):
        return None
    # the line of each point: that of its face, going away from the skin; the ray stops at the skin
    nf = L.grille()[1][~L.masque]
    nf = nf / np.maximum(np.linalg.norm(nf, axis=1), 1e-12)[:, None]
    hf = (L.h0 - L.a)[L.Fw[~L.masque]].mean(1)
    n, h = np.repeat(nf, 4, axis=0)[pres], np.repeat(hf, 4)[pres]
    lon = np.minimum(np.where(np.isfinite(h), np.maximum(h, 0.0) + 0.003, LONGUEUR), LONGUEUR)
    dedans = gu.tirer(P, -n, lon)[0]   # U between the point and the body: the point is outside
    dessus = gu.tirer(P, n, LONGUEUR)[0]    # U above the point: it is covered
    dehors = np.isfinite(dedans) & (dedans > 2e-4)
    couvert = np.isfinite(dessus) | dehors
    # a free flap far in front of the upper part (a frill, a flared skirt hem) flares over it: not a defect
    evase = dehors & np.repeat(~L.tenue[~L.masque], 4)[pres] & (dedans > CB.LIBRE_MAX)
    aire_evasee = float(aire[evase].sum())
    dehors &= ~evase
    if aire[couvert].sum() < 1e-4 and not len(X):
        return None
    prof = dedans[dehors] * 1000
    return {"dessous": L.id, "dessus": U.id,
            "croisements": int(len(X)),
            "recouvrement_cm2": round(float(aire[couvert].sum()) * 1e4, 1),
            "dehors_cm2": round(float(aire[dehors].sum()) * 1e4, 1),
            "evase_cm2": round(aire_evasee * 1e4, 1),
            "dehors_mm_median": round(float(np.median(prof)), 1) if len(prof) else 0.0,
            "dehors_mm_max": round(float(prof.max()), 1) if len(prof) else 0.0,
            "points": P[dehors], "croix": X}


def controler(couches):
    """Every pair of wrapping parts of a settled outfit (layers ordered from inside to outside)."""
    cp = CB.corps()
    env = [c for c in couches if c.enveloppe]
    out = []
    for i, L in enumerate(env):
        for U in env[i + 1:]:
            r = paire(L, U, cp)
            if r:
                out.append(r)
    return out


def bilan(res):
    return {"croisements": sum(r["croisements"] for r in res),
            "evase_cm2": round(sum(r["evase_cm2"] for r in res), 1),
            "dehors_cm2": round(sum(r["dehors_cm2"] for r in res), 1),
            "dehors_mm_max": max([r["dehors_mm_max"] for r in res] or [0.0])}


def tenue(ids):
    pieces = {i: CB.piece(i) for i in ids}
    ret = CB.retirees({i: p.meta for i, p in pieces.items()})
    couches, notes = CB.resoudre([p for i, p in pieces.items() if i not in ret])
    return couches, controler(couches)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pieces", help="comma-separated identifiers; otherwise the reference outfits")
    ap.add_argument("--json", help="writes the details to this file")
    a = ap.parse_args()
    tenues = {"tenue": a.pieces.split(",")} if a.pieces else TENUES
    tout = {}
    for nom, ids in tenues.items():
        couches, res = tenue(ids)
        b = bilan(res)
        print("%s: %d crossings, %.1f cm² outside, worst %.1f mm, %.1f cm² of flared flaps   (order: %s)" % (
            nom, b["croisements"], b["dehors_cm2"], b["dehors_mm_max"], b["evase_cm2"], " < ".join(c.id for c in couches)))
        for r in res:
            if r["croisements"] or r["dehors_cm2"] >= 0.5:
                print("    %-22s under %-22s %5d crossings, %7.1f cm² outside of %7.1f, median %.1f mm, worst %.1f mm" % (
                    r["dessous"], r["dessus"], r["croisements"], r["dehors_cm2"], r["recouvrement_cm2"],
                    r["dehors_mm_median"], r["dehors_mm_max"]))
        tout[nom] = {"bilan": b, "paires": [{k: v for k, v in r.items() if k not in ("points", "croix")} for r in res]}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(tout, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

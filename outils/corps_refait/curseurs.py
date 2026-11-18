"""Shape sliders of the remade body: vertex morphs (one per slider) written into corps.pmx, and
the tool that applies them to a PMX — the bare body or a body dressed by porter.py: each vertex of a
garment follows the closest skin point.

  python3 -I curseurs.py --corps <corps.pmx refait> --modele <pmx to deform> --valeurs poitrine=1,hanches=-0.5 \
      --sortie <pmx>

A slider changes a volume, not a proportion: limb length and shoulder width belong to the
skeleton (one template per skeleton). Values from -1 to 1; the displacements add up.
"""
import argparse
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
for d in (ICI, os.path.join(ICI, "..", "atlas"), os.path.join(ICI, "..", "tete")):
    if d not in sys.path:
        sys.path.insert(0, d)

import geom as G              # noqa: E402

CM = 8.0
MIR = np.array([-1.0, 1.0, 1.0])
PREFIXE = "forme_"


def cloche(x, centre, largeur):
    return np.exp(-0.5 * ((x - centre) / largeur) ** 2)


def rampe(x, a, b):
    """0 before a, 1 after b, smooth in between."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def autour_axe(P, a, b):
    """Coordinate along the axis a -> b (cm from a), radial vector, length of the axis."""
    d = b - a
    L = np.linalg.norm(d)
    d = d / L
    s = (P - a) @ d
    return s, (P - a) - np.outer(s, d), L


def morphes(g, R):
    """Displacement of each vertex (cm) for each slider at +1."""
    P, sym = g["P"], g["sym"]
    o = R.os
    n = len(P)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    # what belongs to the trunk and legs: not the arm beyond the shoulder
    sb, rb, Lb = autour_axe(P * [1, 1, 1], o["左腕"], o["左ひじ"])
    sbd, rbd, _ = autour_axe(P * MIR, o["左腕"], o["左ひじ"])
    bras_g = (x > 0) & (sb > 2.0) & (np.linalg.norm(rb, axis=1) < 9.0) & (y > 80)
    bras_d = (x < 0) & (sbd > 2.0) & (np.linalg.norm(rbd, axis=1) < 9.0) & (y > 80)
    main = (np.abs(x) > 30) & (y > 80)
    tronc = ~(bras_g | bras_d | main)
    zc = float(o["上半身"][2]) + 3.0                          # trunk axis, front to back
    rad = np.stack([x, np.zeros(n), z - zc], 1)             # horizontal vector from the trunk axis
    M = {}

    # waist: the waist circumference, around its narrowest height
    y_t = float(o["上半身1"][1]) + 4.0
    M["taille"] = rad * (0.12 * cloche(y, y_t, 7.0) * tronc)[:, None]

    # hips: the width of the pelvis and of the top of the thighs
    y_h = float(o["下半身"][1]) - 6.0
    M["hanches"] = np.stack([x * 0.11, np.zeros(n), (z - zc) * 0.04], 1) * (cloche(y, y_h, 8.0) * tronc)[:, None]

    # buttocks: backwards, around the buttock bones
    c = 0.5 * (o["左尻1"] + o["左尻2"])
    d = np.minimum(np.linalg.norm(P - c, axis=1), np.linalg.norm(P - c * MIR, axis=1))
    w = cloche(d, 0.0, 7.5) * rampe(z, zc, zc + 5.0) * tronc
    M["fesses"] = np.stack([x * 0.03, np.zeros(n), np.full(n, 2.2)], 1) * w[:, None]

    # chest: each breast grows from its base
    b = o["左胸上2"]
    base = np.array([b[0] + 1.0, b[1] - 4.5, b[2] + 3.5])
    D = np.zeros((n, 3))
    for s_ in (1.0, -1.0):
        c = base * [s_, 1, 1]
        v = P - c
        w = cloche(np.linalg.norm(v, axis=1), 0.0, 6.5) * rampe(-(z - zc), 4.0, 9.0) * tronc * rampe(x * s_, 0.3, 3.5)
        D += (v * 0.22 + np.array([0.0, -0.3, -1.2])) * w[:, None]
    M["poitrine"] = D

    # thighs: the thigh circumference, from the crotch to the knee
    D = np.zeros((n, 3))
    for s_ in (1.0, -1.0):
        s, r, L = autour_axe(P * [s_, 1, 1], o["左足"], o["左ひざ"])
        w = rampe(s, 3.0, 12.0) * (1 - rampe(s, L - 8.0, L + 2.0)) * (x * s_ > 0) * tronc
        # on the crotch side, the thigh grows less: the two thighs do not run into each other
        dedans = 1.0 - 0.6 * rampe(-(r[:, 0]), 0.0, 4.0)
        D += r * [s_, 1, 1] * (0.10 * w * dedans)[:, None]
    M["cuisses"] = D

    # arms: the circumference of the upper arm and forearm
    D = np.zeros((n, 3))
    for s_, masque in ((1.0, bras_g), (-1.0, bras_d)):
        Q = P * [s_, 1, 1]
        s1, r1, L1 = autour_axe(Q, o["左腕"], o["左ひじ"])
        s2, r2, L2 = autour_axe(Q, o["左ひじ"], o["左手首"])
        haut = s2 < 0
        r = np.where(haut[:, None], r1, r2)
        w = np.where(haut, rampe(s1, 4.0, 10.0), 1 - rampe(s2, L2 - 6.0, L2 - 1.0)) * masque
        D += r * [s_, 1, 1] * (0.10 * w)[:, None]
    M["bras"] = D

    # shoulders: the volume of the shoulder (shoulder width itself belongs to the skeleton)
    D = np.zeros((n, 3))
    for s_ in (1.0, -1.0):
        c = (o["左腕"] + [1.0, -1.0, 0.0]) * [s_, 1, 1]
        v = P - c
        w = cloche(np.linalg.norm(v, axis=1), 0.0, 5.0) * (x * s_ > 0) * ~main
        D += v * (0.13 * w)[:, None]
    M["épaules"] = D

    out = {}
    for nom, D in M.items():
        D = 0.5 * (D + D[sym] * MIR)                 # exact symmetry
        D[np.linalg.norm(D, axis=1) < 0.005] = 0.0
        out[PREFIXE + nom] = D
    return out


# ---------------------------------------------------------------------------------------- application
def suivre(Q, V, D, k=4, bloc=128):
    """Displacement of each point of Q: that of the k closest skin vertices, weighted by the inverse
    square of the distance (a skin vertex itself gets exactly its own). Also returns the distance."""
    dep = np.zeros_like(Q)
    dist = np.zeros(len(Q))
    v2 = (V * V).sum(1)
    for a in range(0, len(Q), bloc):
        q = Q[a:a + bloc]
        d2 = np.maximum((q * q).sum(1)[:, None] - 2 * q @ V.T + v2[None], 0.0)
        i = np.argpartition(d2, k, axis=1)[:, :k]
        dk = np.take_along_axis(d2, i, 1)
        w = 1.0 / (dk + 1e-6)
        w /= w.sum(1, keepdims=True)
        dep[a:a + bloc] = (D[i] * w[:, :, None]).sum(1)
        dist[a:a + bloc] = np.sqrt(dk.min(1))
    return dep, dist


def appliquer(corps, modele, valeurs, portee_cm=8.0, journal=print):
    """corps: remade PMX (its forme_* morphs); modele: PMX to deform (dict read by pmx.lire), modified in
    place. Skin vertices follow their morph; the others follow the closest skin point, less and
    less beyond `portee_cm`."""
    import pmx as PMX
    t = PMX.vers_tableaux(corps)
    peau = [m["nom"] for m in corps["materiaux"]].index("肌")
    D = np.zeros((len(t["pos"]), 3))
    connus = {m["nom"][len(PREFIXE):]: m for m in corps["morphes"] if m["nom"].startswith(PREFIXE)}
    for nom, v in valeurs.items():
        if nom not in connus:
            raise SystemExit(f"unknown slider: {nom} (known: {', '.join(connus)})")
        for i, d in connus[nom]["decalages"]:
            D[i] += v * np.array(d) * CM
    Fp = t["faces"][t["mat_face"] == peau]
    ids = np.unique(Fp)
    V, Ds = t["pos"][ids] * CM, D[ids]
    s = modele["sommets"]
    Q = np.array(s["pos"]) * CM
    dep, dist = suivre(Q, V, Ds)
    dep *= (1 - rampe(dist, portee_cm, 2 * portee_cm))[:, None]
    Q2 = Q + dep
    s["pos"] = [tuple(float(x) for x in p / CM) for p in Q2]
    # normals: recomputed per material, vertices welded
    tm = PMX.vers_tableaux(modele)
    nor = np.array(s["nor"])
    for i in range(len(modele["materiaux"])):
        Fm = tm["faces"][tm["mat_face"] == i]
        if not len(Fm):
            continue
        ids = np.unique(Fm)
        r2, i2 = G.souder(Q2[ids], 1e-3)
        loc = -np.ones(len(Q2), int)
        loc[ids] = i2
        Fl = loc[Fm]
        Fl = Fl[(Fl[:, 0] != Fl[:, 1]) & (Fl[:, 1] != Fl[:, 2]) & (Fl[:, 0] != Fl[:, 2])]
        nor[ids] = G.normales_sommets(Q2[ids][r2], Fl)[i2]
    s["nor"] = [tuple(float(x) for x in v) for v in nor]
    journal(f"sliders {valeurs}: {int((np.linalg.norm(dep, axis=1) > 0.01).sum())} vertices moved, "
            f"up to {np.linalg.norm(dep, axis=1).max() * 10:.1f} mm")
    return modele


def main():
    import pmx as PMX
    import pmx_ecrire
    ap = argparse.ArgumentParser()
    ap.add_argument("--corps", required=True)
    ap.add_argument("--modele", help="PMX to deform (default: the body itself)")
    ap.add_argument("--valeurs", required=True, help="poitrine=1,hanches=-0.5")
    ap.add_argument("--sortie", required=True)
    a = ap.parse_args()
    valeurs = {k: float(v) for k, v in (kv.split("=") for kv in a.valeurs.split(","))}
    corps = PMX.lire(a.corps)
    modele = PMX.lire(a.modele or a.corps)
    appliquer(corps, modele, valeurs)
    pmx_ecrire.ecrire(modele, a.sortie)


if __name__ == "__main__":
    main()

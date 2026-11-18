"""Check of the shoes of the cache: is the foot inside the shoe?

Asked on what the workshop shows (the adjusted part, the skin it leaves visible), foot by foot:

- **under the ground** (`sous_cm`): how far the shoe goes below the ground the bare body stands on
  (its sole), once the body is raised by `lever_cm`. A heel worn on a flat foot sinks into the
  ground by its height;
- **skin left out** (`libre_cm2`, `libre_dessus_cm2` for the part facing up): area of visible skin,
  at the height of the shoe and within its outline, that no face of the shoe covers: a ray cast from
  the skin along its normal leaves without meeting the shoe. A foot standing above a shoe that sank
  shows here, as does the skin seen through an opening the shoe has by design (compare a shoe with
  itself, before and after);
- **skin through** (`traverse_cm2`): the part of it with the shoe just under the skin: toes or the
  instep that went through the top of the shoe;
- **raised** (`lever_cm`): the height the body is raised by to stand on the shoe's sole (0 when the
  foot is not posed);
- **torn** (`dechire_cm2`, `dechire_faces`, `etire_max`): a shoe is made of rigid pieces (sole, upper, buckle, strap,
  tongue: each connected piece of its source mesh). Each edge's length, as displayed, is compared with
  its length on the source, relative to its piece's own median (the piece may be scaled as a whole):
  the area of faces with an edge stretched or shrunk past ETIRE, and the worst ratio;
- **width** (`largeur`): the shoe's width at the bottom (lowest BAS_LARGEUR) relative to the source
  shoe's width scaled like the body (`agrandie`), and to the bare foot's width (`sur_pied`).

    python3 check_feet.py                         every shoe of the cache
    python3 check_feet.py --pieces a,b --planche pieds.png
    python3 check_feet.py --json mesures.json     measurements written as JSON
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import outfit as CB  # noqa: E402
import coverage as K  # noqa: E402
import geom as G  # noqa: E402
import contact_sheet as PL  # noqa: E402
import rays as R  # noqa: E402
import pmx as PMX  # noqa: E402

PMX_M = 0.08        # metres per PMX unit

ETIRE = 1.5         # an edge this many times longer (or shorter) than its piece's median ratio is torn
BAS_LARGEUR = 0.04  # metres above the bottom where the widths are taken
HAUT_PIED = 0.16    # metres: above this, it is no longer the foot (a boot's shaft is checked elsewhere)
RAYON = 0.05        # ray from the skin: beyond this, nothing covers it
DEDANS = 0.03       # the shoe this close inside the skin: the skin went through it (an opening leaves nothing there)
DEBORD = 0.01       # margin around the shoe's outline
COTE, LARG = 300, 0.34


def chaussures():
    out = []
    for ident in sorted(os.listdir(os.path.join(C.CACHE, "pieces"))):
        d = os.path.join(C.CACHE, "pieces", ident)
        try:
            m = C.lire_json(os.path.join(d, "meta.json"))
        except (OSError, ValueError):
            continue
        if m.get("type") in K.TYPES_PIED and os.path.isfile(os.path.join(d, "piece.glb")):
            out.append(ident)
    return out


def _aire(V, F):
    T = V[F]
    return 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)


def _source(meta):
    """(source part PMX as tables, source character PMX) of a cache part, the PMX porter fits."""
    d = C.absolu(meta["dossier"])
    fi = C.lire_json(os.path.join(d, "fiche.json"))
    g = fi.get("garnie")
    chemin = os.path.join(d, g["pmx"]) if g and os.path.exists(os.path.join(d, g["pmx"])) else os.path.join(d, fi["pmx"])
    return PMX.lire(chemin), PMX.lire(fi["source"]["pmx"])


def _hauteur(pmx):
    """Head to ankle height of a skeleton (its units), the body's scale."""
    pos = {o["nom"]: o["pos"] for o in pmx["os"]}
    return pos["頭"][1] - (pos["左足首"][1] + pos["右足首"][1]) / 2


def _morceaux(V, F):
    """Connected piece of each face (vertices welded by position)."""
    _, inv = G.souder(V, 1e-6)
    lab = np.arange(inv.max() + 1)
    E = inv[F]
    while True:
        m = lab[E].min(1)
        nouv = lab.copy()
        for k in range(3):
            np.minimum.at(nouv, E[:, k], m)
        nouv = nouv[nouv]
        if (nouv == lab).all():
            break
        lab = nouv
    return lab[E[:, 0]]


def dechirure(Z, F, meta):
    """Tear of the displayed shoe (Z, F: cache frame, metres) against its source: {"dechire_cm2",
    "etire_max", "largeur"}; None when the faces do not match the source's."""
    try:
        S, Ssrc = _source(meta)
    except (OSError, ValueError, KeyError):
        return None
    T = PMX.vers_tableaux(S)
    noms = [m["nom"] for m in S["materiaux"]]
    garde = [noms.index(n) for n in meta.get("materiaux") or [] if n in noms]
    Fs = np.concatenate([T["faces"][T["mat_face"] == i] for i in garde]) if garde else T["faces"][:0]
    if len(Fs) != len(F):
        return None
    X = np.asarray(T["pos"], float) * PMX_M
    piece = _morceaux(X, Fs)
    # the cache's faces may be turned over (glTF winding: corners 0, 2, 1); its vertices keep the
    # source's order, so the corners' ranks tell which faces were
    def longueurs(V, Fx):
        return np.stack([np.linalg.norm(V[Fx[:, k]] - V[Fx[:, (k + 1) % 3]], axis=1) for k in range(3)], 1)
    ls, lo = longueurs(X, Fs), longueurs(Z, F)
    tourne = (np.argsort(np.argsort(Fs, 1), 1) != np.argsort(np.argsort(F, 1), 1)).any(1)
    lo[tourne] = lo[tourne][:, [2, 1, 0]]
    ok = ls > 5e-4
    r = np.where(ok, lo / np.maximum(ls, 1e-12), np.nan)
    rel = np.full_like(r, np.nan)
    for k in np.unique(piece):
        m = piece == k
        med = np.nanmedian(r[m]) if np.isfinite(r[m]).any() else np.nan
        rel[m] = r[m] / med
    dev = np.nanmax(np.abs(np.log(np.where(np.isfinite(rel), rel, 1.0))), 1)
    dechire = dev > np.log(ETIRE)
    aire = _aire(Z, F)
    res = {"dechire_cm2": round(1e4 * float(aire[dechire].sum()), 1), "dechire_faces": int(dechire.sum()),
           "etire_max": round(float(np.exp(dev.max())) if len(dev) else 1.0, 2)}
    # widths at the bottom: displayed, source scaled like the body, bare foot
    ech = _hauteur(PMX.lire(C.CORPS)) / _hauteur(Ssrc)
    Vn = CB.peau()[0]
    larg = {}
    for nom, cote in (("gauche", 1), ("droite", -1)):
        u = np.unique(F)
        zo = Z[u][cote * Z[u][:, 0] > 0]
        xs = X[np.unique(Fs)]
        xs = xs[cote * xs[:, 0] > 0]
        nu = Vn[(cote * Vn[:, 0] > 0) & (Vn[:, 1] < HAUT_PIED)]
        if not len(zo) or not len(xs) or not len(nu):
            continue
        bas = lambda P: P[P[:, 1] < P[:, 1].min() + BAS_LARGEUR]
        w_o, w_s = np.ptp(bas(zo)[:, 0]), np.ptp(bas(xs)[:, 0]) * ech
        w_p = np.ptp(nu[nu[:, 1] < nu[:, 1].min() + BAS_LARGEUR][:, 0])
        larg[nom] = {"agrandie": round(float(w_o / w_s), 2), "sur_pied": round(float(w_o / w_p), 2)}
    res["largeur"] = larg
    return res


def mesurer(ident):
    """Measurements of one shoe, per foot (left = +x), on the scene the workshop shows."""
    p = CB.piece(ident)
    couches, _ = CB.resoudre([p])
    objets = PL.scene(couches)
    Vp, Fp = objets[0][0], objets[0][1]          # visible skin
    Vs, Fs = objets[1][0], objets[1][1]          # shoe as displayed
    Vn = CB.peau()[0]                            # whole bare skin (the sole, even hidden)
    grille = R.Grille(Vs, Fs)
    c = couches[0]
    tear = dechirure(CB.poser_tenue(couches)[1].get(c.id, c.Z), c.F, p.meta)
    pied = p.meta.get("pied") or {}
    res = {"id": ident, "retenue": p.meta.get("retenue"), "dessus": p.meta.get("dessus_du_pied"),
           "lever_cm": round(100 * float(pied.get("lever", 0.0)), 1), "cotes": {},
           "dechirure": tear}
    for nom, cote in (("gauche", 1), ("droite", -1)):
        s = Vs[np.unique(Fs)] if len(Fs) else Vs[:0]
        s = s[(cote * s[:, 0] > 0) & (s[:, 1] < HAUT_PIED)]
        nu = Vn[(cote * Vn[:, 0] > 0) & (Vn[:, 1] < HAUT_PIED)]
        if not len(s) or not len(nu):
            continue
        plante = float(nu[:, 1].min())
        lo, hi = s.min(0) - DEBORD, s.max(0) + DEBORD
        g = Vp[Fp].mean(1)
        zone = (cote * g[:, 0] > 0) & (g[:, 1] < min(hi[1], HAUT_PIED)) & (g[:, 1] > plante + 0.002) \
            & (g[:, 0] > lo[0]) & (g[:, 0] < hi[0]) & (g[:, 2] > lo[2]) & (g[:, 2] < hi[2])
        Fz = Fp[zone]
        n = np.cross(Vp[Fz][:, 1] - Vp[Fz][:, 0], Vp[Fz][:, 2] - Vp[Fz][:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
        t, _ = grille.tirer(g[zone] + 1e-4 * n, n, RAYON)
        t_in, _ = grille.tirer(g[zone] - 1e-4 * n, -n, DEDANS)
        libre = ~np.isfinite(t)                           # nothing over it
        traverse = libre & np.isfinite(t_in)              # and the shoe under it: it went through
        a = _aire(Vp, Fz)
        haut = n[:, 1] > K.PIED_NORMALE
        res["cotes"][nom] = {
            "sous_cm": round(100 * (plante - float(s[:, 1].min())) - res["lever_cm"], 1),
            "libre_cm2": round(1e4 * float(a[libre].sum()), 1),
            "libre_dessus_cm2": round(1e4 * float(a[libre & haut].sum()), 1),
            "traverse_cm2": round(1e4 * float(a[traverse].sum()), 1),
            "centre": [float(x) for x in (s.min(0) + s.max(0)) / 2],
            "faces_dehors": np.where(zone)[0][libre].tolist(),
        }
    return res, objets


def ligne(objets, mes):
    """Front, outer side, back, top, three-quarter, cut through the middle of the foot (left foot);
    skin left out in red, the ground (y = 0) as a red line."""
    c = np.array(mes["cotes"].get("gauche", next(iter(mes["cotes"].values())))["centre"])
    c[1] = 0.07
    peau = list(objets[0])
    k = mes["cotes"].get("gauche", {}).get("faces_dehors", [])
    garde = np.ones(len(peau[1]), bool)
    garde[k] = False
    objs = [(peau[0], peau[1][garde], peau[2]), (peau[0], peau[1][~garde], (230, 40, 40))] + list(objets[1:])
    vues = [np.array(v, float) for v in ((0, 0, 1), (1, 0, 0), (0, 0, -1), (0, 1, 1e-3), (0.7, 0.3, 0.7))]
    imgs = [PL.vignette(objs, c, v / np.linalg.norm(v), COTE, LARG, cercle=False) for v in vues]
    coupe = []
    for Vo, Fo, coul in objs:
        g = Vo[Fo].mean(1) if len(Fo) else np.zeros((0, 3))
        coupe.append((Vo, Fo[(g[:, 0] > c[0]) & (g[:, 0] < c[0] + 0.25)], coul))
    imgs.append(PL.vignette(coupe, c, np.array([-1.0, 0, 0]), COTE, LARG, cercle=False))
    img = np.concatenate(imgs, 1)
    sol = int(COTE / 2 + c[1] * COTE / LARG)
    for j in (0, 1, 2, 5):
        if 0 <= sol < COTE:
            img[sol, j * COTE:(j + 1) * COTE] = (200, 40, 40)
    return img


def p_v(ident):
    return (C.lire_json(os.path.join(C.CACHE, "pieces", ident, "meta.json")) or {}).get("v_portage")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pieces", help="comma-separated part ids (default: every shoe of the cache)")
    ap.add_argument("--planche", help="PNG contact sheet, one row per shoe")
    ap.add_argument("--json", help="measurements as JSON")
    a = ap.parse_args()
    ids = a.pieces.split(",") if a.pieces else chaussures()
    tout, lignes = [], []
    for ident in ids:
        mes, objets = mesurer(ident)
        if not mes["cotes"]:
            continue
        g = mes["cotes"]
        print("%-32s %-5s v%s dessus %-5s lever %4.1f cm  sous %s cm  libre %s cm2 (dessus %s)  traverse %s cm2" % (
            ident, mes["retenue"], p_v(ident), mes["dessus"], mes["lever_cm"],
            "/".join("%4.1f" % v["sous_cm"] for v in g.values()),
            "/".join("%5.1f" % v["libre_cm2"] for v in g.values()),
            "/".join("%5.1f" % v["libre_dessus_cm2"] for v in g.values()),
            "/".join("%4.1f" % v["traverse_cm2"] for v in g.values())))
        t = mes["dechirure"]
        if t:
            print("%-32s torn %5.1f cm2, %d faces (worst x%.2f)  width vs source %s  vs foot %s" % (
                "", t["dechire_cm2"], t["dechire_faces"], t["etire_max"], "/".join("%.2f" % v["agrandie"] for v in t["largeur"].values()),
                "/".join("%.2f" % v["sur_pied"] for v in t["largeur"].values())))
        if a.planche:
            lignes.append(ligne(objets, mes))
        for v in g.values():
            v.pop("faces_dehors")
        tout.append(mes)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(tout, f, ensure_ascii=False, indent=1)
    if a.planche and lignes:
        PL.ecrire_png(a.planche, np.concatenate(lignes, 0))
        print(a.planche)


if __name__ == "__main__":
    main()

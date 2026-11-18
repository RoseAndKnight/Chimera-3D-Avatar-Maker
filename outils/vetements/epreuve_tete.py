"""Animation test of a head stitched on by tete_entiere: does the seam break when the model moves?

    python3 -I outils/vetements/epreuve_tete.py FOLDER [--poses-pmx]

FOLDER is the output of tete_entiere (resultat.pmx, raccord.json). Each pose (neck, head, chest,
shoulders, arms) is applied by forward kinematics and linear skinning (outils/atlas/cinematique.py,
SDEF read as BDEF2). Measurements, on the skin (颜, 肌_tete, 肌):

 - opening: spread of the vertices welded at rest (seam copies included), in mm; 0 expected;
 - stretch: |ln(posed length / rest length)| of the edges;
 - fold: change in the dihedral angle of the edges (degrees).

Zero-area faces (glued scalp strip) have no normal and are left out of the folds. Stretch and fold are
reported near the neck seam (less than 15 mm from the cut) and near the scalp edge, and on a control
zone of the same model that the seam does not touch: the body's neck 15 to 40 mm below the cut, and the
body's scalp 25 to 50 mm from the edge. A sound seam folds and stretches no more than its control.

Output: FOLDER/epreuve.json; with --poses-pmx, FOLDER/pose_<name>.pmx for the renders.
"""
import json
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI, os.path.join(ICI, "..", "atlas"), os.path.join(ICI, "..", "tete")]
import pmx as PMX  # noqa: E402
import pmx_ecrire  # noqa: E402
import cinematique as CIN  # noqa: E402
from tete_entiere import soudure, point_polyligne, MM  # noqa: E402

PEAU = ("颜", "肌_tete", "肌")
X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)     # PMX frame: y up, facing -z


def q(*rots):
    out = CIN.IDENTITE.copy()
    for axe, deg in rots:
        out = CIN.q_mul(out, CIN.q_axe_angle(axe, deg))
    return out


POSES = {
    "tete_lacet+60": {"頭": q((Y, 60))},
    "tete_lacet-60": {"頭": q((Y, -60))},
    "tete_tangage+30": {"頭": q((X, 30))},
    "tete_tangage-30": {"頭": q((X, -30))},
    "tete_roulis+25": {"頭": q((Z, 25))},
    "cou_tete_lacet_tangage": {"首": q((Y, 30), (X, 20)), "頭": q((Y, 30), (X, 15))},
    "cou_tete_renverse": {"首": q((X, 35)), "頭": q((X, 30))},
    "cou_tete_penche": {"首": q((X, -30)), "頭": q((X, -25))},
    "cou_tete_roulis": {"首": q((Z, -20)), "頭": q((Z, -20), (Y, 25))},
    "buste_torsion": {"上半身2": q((Y, 30)), "首": q((Y, 20)), "頭": q((Y, 30))},
    "epaules_haussees": {"左肩": q((Z, 20)), "右肩": q((Z, -20))},
    "bras_leves": {"左腕": q((Z, 70)), "右腕": q((Z, -70))},
}
RENDUS = ("tete_lacet+60", "tete_tangage-30", "cou_tete_renverse", "cou_tete_roulis", "buste_torsion", "bras_leves")


def squelette(R):
    os_ = R["os"]
    tetes = np.array([o["pos"] for o in os_], float)
    parents = np.array([o["parent"] for o in os_], int)
    src = np.full(len(os_), -1)
    fac = np.zeros(len(os_))
    arot = np.zeros(len(os_), bool)
    adep = np.zeros(len(os_), bool)
    for i, o in enumerate(os_):
        a = o.get("ajout")
        if a:
            src[i], fac[i] = a["source"], a["facteur"]
            arot[i], adep[i] = a["rotation"], a["deplacement"]
    return tetes, parents, src, fac, arot, adep


def aretes_et_diedres(F):
    """Edges (sorted pairs of welded vertices) and, for those with two faces, those two faces."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    E = np.sort(E, 1)
    cle = E[:, 0].astype(np.int64) * (E.max() + 1) + E[:, 1]
    o = np.argsort(cle, kind="stable")
    cle, E, fid = cle[o], E[o], fid[o]
    debut = np.r_[True, cle[1:] != cle[:-1]]
    uniques = E[debut]
    groupe = np.cumsum(debut) - 1
    n = np.bincount(groupe)
    deux = np.where(n == 2)[0]
    premier = np.where(debut)[0]
    paires = np.stack([fid[premier[deux]], fid[premier[deux] + 1]], 1)
    return uniques, uniques[deux], paires


def normales_faces(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)


def resume(x):
    return {"max": round(float(x.max()), 4) if len(x) else None,
            "p99": round(float(np.percentile(x, 99)), 4) if len(x) else None}


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    dossier = a[0]
    R = PMX.lire(os.path.join(dossier, "resultat.pmx"))
    J = json.load(open(os.path.join(dossier, "raccord.json"), encoding="utf-8"))
    t = PMX.vers_tableaux(R)
    P0 = t["pos"]
    noms = [m["nom"] for m in R["materiaux"]]
    ip = [noms.index(n) for n in PEAU if n in noms]
    F = t["faces"][np.isin(t["mat_face"], ip)]
    # soudure returns a group number: each used vertex is replaced by the first used vertex of its
    # group (an orphan vertex, with no face, never serves as representative)
    utiles = np.unique(F)
    lab = soudure(P0[utiles])
    premier = np.zeros(lab.max() + 1, int)
    premier[lab[::-1]] = utiles[::-1]
    sou = np.arange(len(P0))
    sou[utiles] = premier[lab]
    Fs = sou[F]                                          # faces on the welded vertices
    aire = np.linalg.norm(np.cross(P0[Fs[:, 1]] - P0[Fs[:, 0]], P0[Fs[:, 2]] - P0[Fs[:, 0]]), axis=1) / 2
    Fs = Fs[aire > (0.05 * MM) ** 2]                    # zero-area faces (glued scalp strip): meaningless normal
    E, E2, paires = aretes_et_diedres(Fs)
    anneau = np.array(J["indices"]["anneau"])
    crane = np.array(J["indices"]["crane"])

    # zones at rest
    vs = np.unique(Fs)
    d_an = np.full(len(P0), np.inf)
    d_cr = np.full(len(P0), np.inf)
    d_an[vs] = point_polyligne(P0[vs], P0[anneau])[4] / MM
    d_cr[vs] = point_polyligne(P0[vs], P0[crane])[4] / MM
    y_cou = P0[anneau, 1].mean()
    c_cou = P0[anneau].mean(0)
    r_cou = np.linalg.norm((P0[anneau] - c_cou)[:, [0, 2]], axis=1).max()
    dy = (y_cou - P0[:, 1]) / MM
    pres_axe = np.linalg.norm((P0 - c_cou)[:, [0, 2]], axis=1) < 1.5 * r_cou
    zones = {"raccord_cou": d_an < 15, "temoin_cou": (dy > 15) & (dy < 40) & pres_axe,
             "bord_crane": d_cr < 15, "temoin_crane": (d_cr > 25) & (d_cr < 50) & (P0[:, 1] > y_cou + 0.3)}

    # welded groups: a single point at rest
    grp = {}
    for v in np.unique(F):
        grp.setdefault(int(sou[v]), []).append(int(v))
    multi = [g for g in grp.values() if len(g) > 1]

    sq = squelette(R)
    nom_os = {o["nom"]: i for i, o in enumerate(R["os"])}
    idx, pw = t["os"], t["poids"]
    n0 = normales_faces(P0, Fs)
    ang0 = np.degrees(np.arccos(np.clip((n0[paires[:, 0]] * n0[paires[:, 1]]).sum(1), -1, 1)))
    L0 = np.linalg.norm(P0[E[:, 0]] - P0[E[:, 1]], axis=1)

    bilan = {"modele": os.path.join(dossier, "resultat.pmx"), "skinning": "linear, SDEF read as BDEF2",
             "zones_sommets": {k: int(v.sum()) for k, v in zones.items()},
             "groupes_soudes": len(multi), "poses": {}}
    for nom, rots in POSES.items():
        manque = [b for b in rots if b not in nom_os]
        if manque:
            bilan["poses"][nom] = {"os_absents": manque}
            continue
        Rm, tm = CIN.cinematique(*sq, rotations={nom_os[b]: qq for b, qq in rots.items()})
        P = CIN.skinning(P0, idx, pw, sq[0], Rm, tm, True)
        ouv = max((np.linalg.norm(P[g] - P[g].mean(0), axis=1).max() for g in multi), default=0.0) / MM
        L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
        et = np.where(L0 > 0.01 * MM, np.abs(np.log(np.maximum(L, 1e-12) / np.maximum(L0, 1e-12))), 0.0)
        n1 = normales_faces(P, Fs)
        ang = np.degrees(np.arccos(np.clip((n1[paires[:, 0]] * n1[paires[:, 1]]).sum(1), -1, 1)))
        pli = np.abs(ang - ang0)
        res = {"ouverture_mm": round(float(ouv), 6)}
        for z, m in zones.items():
            ze = m[E[:, 0]] & m[E[:, 1]]
            zd = m[E2[:, 0]] & m[E2[:, 1]]
            res[z] = {"etirement": resume(et[ze]), "pli_deg": resume(pli[zd])}
        bilan["poses"][nom] = res
        if "--poses-pmx" in a and nom in RENDUS:
            Q = dict(R)
            Q["sommets"] = dict(R["sommets"])
            Q["sommets"]["pos"] = [tuple(p) for p in P.tolist()]
            ch = os.path.join(dossier, "pose_%s.pmx" % nom)
            pmx_ecrire.ecrire(Q, ch + ".tmp")
            os.replace(ch + ".tmp", ch)
    ch = os.path.join(dossier, "epreuve.json")
    with open(ch + ".tmp", "w", encoding="utf-8") as f:
        json.dump(bilan, f, ensure_ascii=False, indent=1)
    os.replace(ch + ".tmp", ch)
    for nom, r in bilan["poses"].items():
        if "ouverture_mm" not in r:
            print(nom, r)
            continue
        print("%-24s open %.4f  neck str %.3f/%.3f fold %5.1f/%5.1f   scalp str %.3f/%.3f fold %5.1f/%5.1f" % (
            nom, r["ouverture_mm"],
            r["raccord_cou"]["etirement"]["max"], r["temoin_cou"]["etirement"]["max"],
            r["raccord_cou"]["pli_deg"]["max"], r["temoin_cou"]["pli_deg"]["max"],
            r["bord_crane"]["etirement"]["max"], r["temoin_crane"]["etirement"]["max"],
            r["bord_crane"]["pli_deg"]["max"], r["temoin_crane"]["pli_deg"]["max"]))


if __name__ == "__main__":
    main()

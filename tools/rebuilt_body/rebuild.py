"""Remaking the base body: a mesh of our own, placed on the current body taken as the shape reference.

  python3 -I rebuild.py --reference <corps.pmx> --sortie <folder>

Writes <folder>/corps.pmx (same bones, same materials, original head and skull kept as they are, body
skin remade), <folder>/tex/ (copy) and <folder>/mesures.json. Blender's Python (numpy), without bpy.
"""
import argparse
import json
import os
import pickle
import shutil
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
for d in (ICI, os.path.join(ICI, "..", "atlas"), os.path.join(ICI, "..", "head")):
    if d not in sys.path:
        sys.path.insert(0, d)

import fit as AJ          # noqa: E402
import sliders as CU         # noqa: E402
import geom as G              # noqa: E402
import geometry as GE        # noqa: E402
import pmx_write as pmx_ecrire             # noqa: E402
import grid as QD      # noqa: E402
import reference as RF        # noqa: E402

CM = RF.CM
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import settings as _R  # noqa: E402
DONNEES = _R.chemin("paths.data")   # the user's data folder (tools/settings.py)
REFERENCE = os.path.join(DONNEES, "vetements", "tenue", "corps", "corps.pmx")
LISSAGE_POIDS = 4                       # weight averaging passes between neighbours
ZONE_REPLIS = (70.0, 115.0, 25.0)       # min y, max y, max |x|: where the reference has folds to remove


def charger_reference(chemin, journal=print):
    R = RF.Reference(chemin)
    y0, y1, x1 = ZONE_REPLIS
    R.deplier(zone=lambda p: (p[:, 1] > y0) & (p[:, 1] < y1) & (np.abs(p[:, 0]) < x1), journal=journal)
    return R


# ---------------------------------------------------------------------------------------------- weights
def table_miroir(pmx):
    noms = [o["nom"] for o in pmx["os"]]
    idx = {n: i for i, n in enumerate(noms)}
    out = np.arange(len(noms))
    for i, n in enumerate(noms):
        m = n.replace("左", "\0").replace("右", "左").replace("\0", "右")
        out[i] = idx.get(m, i)
    return out


def poids(g, R):
    """Weights of each vertex: read on the reference at the closest point (left side and middle), the
    right side mirrored; vertices taken from the reference keep their own."""
    P, sym, source, mir = g["P"], g["sym"], g["source"], g["miroir"]
    n = len(P)
    tm = table_miroir(R.pmx)
    F = np.concatenate([R.F[R.f_corps], R.F[R.f_main["G"]], R.F[R.f_main["D"]]])
    gauche = (P[:, 0] >= -1e-6) & (source < 0)
    idx = np.flatnonzero(gauche)
    _, f, bary, _ = G.plus_proche(P[idx], R.V, F)
    W = [None] * n
    for i, fi, b in zip(idx, f, bary):
        acc = {}
        for v, bw in zip(F[fi], b):
            for o, w in zip(R.os_s[v], R.poids_s[v]):
                if w > 0:
                    acc[int(o)] = acc.get(int(o), 0.0) + bw * w
        W[i] = acc
    for i in np.flatnonzero(source >= 0):
        acc = {}
        for o, w in zip(R.os_s[source[i]], R.poids_s[source[i]]):
            if w > 0:
                o = int(tm[o]) if mir[i] else int(o)
                acc[o] = acc.get(o, 0.0) + float(w)
        W[i] = acc
    for i in range(n):
        if W[i] is None:
            W[i] = {int(tm[o]): w for o, w in W[sym[i]].items()}
    for i in np.flatnonzero(sym == np.arange(n)):       # middle: symmetric weights
        acc = dict(W[i])
        for o, w in W[i].items():
            acc[int(tm[o])] = acc.get(int(tm[o]), 0.0) + w
        W[i] = {o: w / 2 for o, w in acc.items()}
    # smoothing: in the hollow of the armpit and of the crotch, the closest point jumps from one side to the other
    # of the reference's fold; a few averaging passes between neighbours give continuous weights. The
    # vertices taken as is (hands, feet) keep their own.
    if LISSAGE_POIDS:
        nb = 1 + max(max(acc) for acc in W)
        M = np.zeros((n, nb))
        for i, acc in enumerate(W):
            for o, w in acc.items():
                M[i, o] = w
        M /= M.sum(1, keepdims=True)
        lib = np.flatnonzero(source < 0)
        vois = g["vois"]
        for _ in range(LISSAGE_POIDS):
            moy = np.stack([M[vois[i]].mean(0) for i in lib])
            M[lib] = 0.5 * M[lib] + 0.5 * moy
        W = [{int(o): float(M[i, o]) for o in np.flatnonzero(M[i] > 1e-4)} for i in range(n)]
    os_, pw, typ = np.zeros((n, 4), int), np.zeros((n, 4)), np.zeros(n, int)
    for i, acc in enumerate(W):
        top = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
        s = sum(w for _, w in top)
        for k, (o, w) in enumerate(top):
            os_[i, k], pw[i, k] = o, w / s
        typ[i] = 0 if len(top) == 1 else (1 if len(top) == 2 else 2)
    return os_, pw, typ


# ---------------------------------------------------------------------------------------------- UV
def cartes_uv(g, R):
    """Texture coordinates: one map per tube of the grid (cut along a column; the
    palm is in the arm's map, the foot in the leg's, each finger and each toe has its own).
    Returns (corners (nt, 3) split-vertex indices, original vertex of each split vertex, uv)."""
    P, quads, tris, cartes = g["P"], g["quads"], g["tris"], g["cartes"]
    iles = {}                # name -> list of (vertex indices, uv in cm)
    NT = QD.NT
    # tubes: size of a cell in cm, average of the map
    par_carte = {}
    for q, c in zip(quads, cartes):
        if c is not None:
            par_carte.setdefault(c[0], []).append((q, np.array(c[1], float)))
    for nom, L in par_carte.items():
        du = dv = 0.0
        nu = nv = 0
        for q, uv in L:
            for a in range(4):
                b = (a + 1) % 4
                d = np.linalg.norm(P[q[a]] - P[q[b]])
                if uv[a][1] == uv[b][1]:
                    du, nu = du + d, nu + 1
                else:
                    dv, nv = dv + d, nv + 1
        su, sv = du / max(nu, 1), dv / max(nv, 1)
        out = []
        for q, uv in L:
            uv = uv.copy()
            if nom == "tronc":               # the seam in the middle of the back
                u0 = uv[:, 0].min()
                uv[:, 0] += ((u0 + NT // 2) % NT) - u0
            uv = uv * [su, sv]
            out.append((q[[0, 1, 2]], uv[[0, 1, 2]]))
            out.append((q[[0, 2, 3]], uv[[0, 2, 3]]))
        iles[nom] = out
    assert all(c is not None for c in cartes) and len(tris) == 0, "every quad has its map"
    # packing: shelves, from the tallest map to the shortest
    boites = {}
    for nom, L in iles.items():
        if not L:
            continue
        uv = np.concatenate([u for _, u in L])
        boites[nom] = (uv.min(0), uv.max(0) - uv.min(0))
    marge = 1.0
    aire = sum((b[1][0] + marge) * (b[1][1] + marge) for b in boites.values())
    cote_ = np.sqrt(aire) * 1.12
    while True:
        x = y = h = 0.0
        place = {}
        for nom in sorted(boites, key=lambda k: -boites[k][1][1]):
            w_, h_ = boites[nom][1] + marge
            if x + w_ > cote_:
                x, y, h = 0.0, y + h, 0.0
            place[nom] = (x, y)
            x, h = x + w_, max(h, h_)
        if y + h <= cote_ and max(b[1][0] + marge for b in boites.values()) <= cote_:
            break
        cote_ *= 1.05
    cle, origine, uvs, coins = {}, [], [], []
    for nom, L in iles.items():
        if not L:
            continue
        dec = np.array(place[nom]) + 0.5 * marge - boites[nom][0]
        for t, uv in L:
            tri = []
            for v, u in zip(t, uv):
                u = (u + dec) / cote_
                k = (int(v), nom, int(round(u[0] * 1e5)), int(round(u[1] * 1e5)))
                if k not in cle:
                    cle[k] = len(origine)
                    origine.append(int(v))
                    uvs.append(u)
                tri.append(cle[k])
            coins.append(tri)
    return np.array(coins), np.array(origine), np.array(uvs), {k: [round(float(v), 1) for v in b[1]] for k, b in boites.items()}


# ---------------------------------------------------------------------------------------------- PMX
def assembler(g, R, morphes, journal=print):
    """The new PMX: everything that is not the skin of the body and hands is kept as is."""
    src = R.pmx
    t = R.t
    P = g["P"]
    T = np.concatenate([AJ.triangles(g["quads"]), g["tris"]])
    nor = G.normales_sommets(P, T)
    os_, pw, typ = poids(g, R)
    coins, origine, uvs, boites = cartes_uv(g, R)
    # faces kept from the reference: all materials except the skin, and the skull within the skin
    crane = np.zeros(len(t["faces"]), bool)
    crane[R.faces_brut[R.f_crane]] = True
    garde = (t["mat_face"] != R.i_peau) | crane
    util = np.unique(t["faces"][garde])
    nouveau = -np.ones(len(t["pos"]), int)
    nouveau[util] = np.arange(len(util))
    s0 = src["sommets"]
    som = {k: [s0[k][i] for i in util] for k in s0}
    n_anc = len(util)
    for i, (v, uv) in enumerate(zip(origine, uvs)):
        som["pos"].append(tuple(float(x) for x in P[v] / CM))
        som["nor"].append(tuple(float(x) for x in nor[v]))
        som["uv"].append((float(uv[0]), float(uv[1])))
        som["uv_add"].append([(0.0, 0.0, 0.0, 0.0)] * src["n_uv_add"])
        som["type"].append(int(typ[v]))
        som["os"].append([int(x) for x in os_[v]])
        som["poids"].append([float(x) for x in pw[v]])
        som["sdef"].append(None)
        som["contour"].append(1.0)
    faces, mats = [], []
    for i, m in enumerate(src["materiaux"]):
        fm = nouveau[t["faces"][garde & (t["mat_face"] == i)]].tolist()
        if i == R.i_peau:
            fm += (coins + n_anc).tolist()
        faces += fm
        mats.append(dict(m, n_indices=3 * len(fm)))
    out = dict(src)
    out["sommets"] = som
    out["faces"] = faces
    out["materiaux"] = mats
    # shape morphs: one vertex morph per slider
    eclate = {}
    for i, v in enumerate(origine):
        eclate.setdefault(int(v), []).append(n_anc + i)
    out["morphes"] = list(src["morphes"])
    for nom, D in morphes.items():
        dec = []
        for v in np.flatnonzero(np.linalg.norm(D, axis=1) > 1e-4):
            for i in eclate.get(int(v), []):
                dec.append((i, tuple(float(x) for x in D[v] / CM)))
        out["morphes"].append({"nom": nom, "nom_en": nom, "panneau": 4, "type": 1, "decalages": dec})
    return out, T, boites, (os_, pw)


def mesurer(g, R, T, boites, W):
    P = g["P"]
    ang, ar = G.diedres(P, T)
    libre = g["libre"][ar].all(1)
    S = R.V[np.unique(R.F[R.f_corps])]
    d = G.plus_proche(S, P, T)[3] * 10
    Sb = R.V_brut[np.unique(R.F[R.f_corps])] if hasattr(R, "V_brut") else S
    angr, arr = G.diedres(R.V_brut, R.F[R.f_corps])
    haut = lambda V, e: V[e].mean(1)[:, 1] > 20        # feet excluded
    return {
        "sommets": int(len(P)), "triangles": int(len(T)), "quads": int(len(g["quads"])),
        "bords": sorted(len(b) for b in G.boucles_bord(T)),
        "aretes_vives_plus_de_40_hors_mains_pieds": int(((ang > 40) & libre).sum()),
        "angle_max_hors_mains_pieds": round(float(ang[libre].max()), 1),
        "reference_aretes_vives_plus_de_40_hors_pieds": int(((angr > 40) & haut(R.V_brut, arr)).sum()),
        "reference_angle_max_hors_pieds": round(float(angr[haut(R.V_brut, arr)].max()), 1),
        "ecart_reference_mm": {"mediane": round(float(np.median(d)), 2), "p95": round(float(np.percentile(d, 95)), 2),
                               "p99": round(float(np.percentile(d, 99)), 2), "max": round(float(d.max()), 2)},
        "symetrie_mm": round(float(np.abs(P - P[g["sym"]] * [-1, 1, 1]).max() * 10), 4),
        "cartes_uv_cm": boites,
        "os_par_sommet_max": int((W[1] > 0).sum(1).max()),
        "construction": g["info"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--forme", help="reread the shape already computed (forme.pkl in the output folder)", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.sortie, exist_ok=True)
    R = charger_reference(a.reference)
    cache = os.path.join(a.sortie, "forme.pkl")
    if a.forme and os.path.exists(cache):
        g = pickle.load(open(cache, "rb"))
    else:
        g = GE.construire(R)
        pickle.dump(g, open(cache, "wb"))
    morphes = CU.morphes(g, R)
    pmx, T, boites, W = assembler(g, R, morphes)
    pmx_ecrire.ecrire(pmx, os.path.join(a.sortie, "corps.pmx"))
    tex = os.path.join(os.path.dirname(a.reference), "tex")
    if os.path.isdir(tex) and os.path.abspath(tex) != os.path.abspath(os.path.join(a.sortie, "tex")):
        shutil.copytree(tex, os.path.join(a.sortie, "tex"), dirs_exist_ok=True)
    m = mesurer(g, R, T, boites, W)
    m["curseurs"] = {k: round(float(np.linalg.norm(D, axis=1).max() * 10), 1) for k, D in morphes.items()}
    json.dump(m, open(os.path.join(a.sortie, "mesures.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Pivot head: the game's human head closest to the average, through which face
transfers go.

    python3 -I tools/head/pivot.py --output <dossier> [--base Belle] [--jobs 4] [--average]

Method (mean template, in the manner of the FLAME or Basel face models):
1. a base head is registered (registration.py: landmarks, eye and mouth rims, non-rigid
   ICP) onto the skin of each human character: the same mesh for all, point to point;
2. generalised Procrustes analysis (translation, rotation, scale) of the registered shapes,
   each vertex weighted by its reliability for each character (close enough to the
   character's skin, not on its boundary); the mean is made left-right symmetric;
3. second pass using as base the character closest to that mean: this is the
   pivot head, taken as is (06/11: deforming it towards the mean brings it only 7.6 %
   closer to the others and damages its morphs, 38 flipped triangles against 8).

With --average, also writes tete-moyenne.pmx: the pivot head in the mean shape (reference
for family morphs); displacement taken where half of the heads are reliable, extended
smoothly elsewhere, eyes and mouth kept in the base's shape, morphs rotated and scaled
locally, neck seam held.

Outputs: <sortie>/tete-pivot.json (character and PMX path), mesures.json (characters,
distances to the mean and to the pivot head), formes.npz; with --average, tete-moyenne.pmx,
its checks and planche_textures.png.
"""
import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
sys.path.append(os.path.join(ICI, ".."))
import settings as R  # noqa: E402
import games as jeux  # noqa: E402

import geom as G  # noqa: E402
import pmx_write as pmx_ecrire  # noqa: E402
import registration as RC  # noqa: E402
import retarget_head as RT  # noqa: E402
import head as TT  # noqa: E402
import checks as VF  # noqa: E402

# The models of the active game (description tools/games/<game>/game.json): one per human character, path relative to the
# mmd_extrait source folder; the description says which ones are excluded and why.
JEU = jeux.actif()
SOURCES = JEU.chemin("mmd_extrait")
PERSONNAGES = JEU.modeles()
# Vertex reliable for a character: within FIABLE_DIST interocular distances of its skin after
# registration, and not on the boundary of its skin.
FIABLE_DIST = 0.03
ITER_PROCUSTE = 8
SEUIL_FIABLES = 0.5
LISSAGE_PIVOT = 1.0
RAYON_OEIL = 0.3
RAYON_BOUCHE = 0.3


def journal(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _base(chemin):
    return TT.Tete(chemin)


def recaler_un(args):
    """Registers the base skin onto a character's; returns X (base, base frame),
    the reliability of each vertex and the final distance."""
    chemin_base, nom, chemin = args
    A = _base(chemin_base)
    B = TT.Tete(chemin)
    s = float(np.sqrt(A.W * A.H_crane / (B.W * B.H_crane)))
    B.placer(s, B.centre_visage(), A.centre_visage())
    XB = B._hote(B.V)
    rec = RC.recaler(A, A.V, B, XB)
    bord = np.zeros(B.n, dtype=bool)
    for b in B.boucles:
        bord[b] = True
    tri = B.F[rec["face"]]
    # projected onto a face with a vertex on the boundary: the character's skin ends there
    sur_bord = bord[tri].any(1)
    fiable = (rec["dist"] < FIABLE_DIST * A.iod) & ~sur_bord
    return nom, rec["X"], fiable, float(np.mean(rec["dist"]) / A.iod)


def similitude(X, Y, w):
    """Similarity (s, R, t) mapping X onto Y in the weighted least-squares sense (Umeyama)."""
    w = w / w.sum()
    mx, my = (w[:, None] * X).sum(0), (w[:, None] * Y).sum(0)
    Xc, Yc = X - mx, Y - my
    C = (w[:, None] * Yc).T @ Xc
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (w * (Xc ** 2).sum(1)).sum())
    return s, R, my - s * R @ mx


def appliquer(X, sRt):
    s, R, t = sRt
    return s * X @ R.T + t


def procuste(formes, poids, ref):
    """Weighted generalised Procrustes analysis. formes: (k, n, 3), poids: (k, n).
    The mean keeps the size and position of ref. Returns the mean and the aligned shapes."""
    M = ref.copy()
    Y = formes.copy()
    for _ in range(ITER_PROCUSTE):
        for i in range(len(formes)):
            Y[i] = appliquer(formes[i], similitude(formes[i], M, poids[i] + 1e-6))
        W = poids + 1e-3
        Mn = (W[..., None] * Y).sum(0) / W.sum(0)[:, None]
        M = appliquer(Mn, similitude(Mn, ref, np.ones(len(ref))))
    return M, Y


def miroir(V):
    """Index of the mirror vertex (closest to the reflection in x) and distance to the reflection."""
    Q = V * np.array([-1.0, 1.0, 1.0])
    d, j = RT._plus_proche_sommet(V, Q)
    return j, d


def symetriser(M, j, ok):
    S = M[j] * np.array([-1.0, 1.0, 1.0])
    out = M.copy()
    out[ok] = 0.5 * (M[ok] + S[ok])
    return out


REFUS = {}      # excluded model -> reason


def recaler_ou_dire(tache):
    try:
        return recaler_un(tache)
    except Exception as e:  # noqa: BLE001
        return tache[1], None, None, "%s : %s" % (type(e).__name__, e)


def passage(chemin_base, travaux):
    taches = [(chemin_base, n, os.path.join(SOURCES, PERSONNAGES[n])) for n in sorted(PERSONNAGES)]
    res = {}
    with ProcessPoolExecutor(travaux) as ex:
        for nom, X, f, d in ex.map(recaler_ou_dire, taches):
            if X is None:      # a model that cannot be registered does not stop the others
                REFUS[nom] = d
                journal("  %-12s excluded: %s" % (nom, d))
                continue
            res[nom] = (X, f, d)
            journal("  %-12s mean distance %.4f interocular, reliable %d %%" % (nom, d, 100 * f.mean()))
    noms = sorted(res)
    if not noms:
        raise SystemExit("pivot: no model could be registered (%s)" % "; ".join("%s: %s" % kv for kv in REFUS.items()))
    A = _base(chemin_base)
    formes = np.stack([res[n][0] for n in noms])
    poids = np.stack([res[n][1] for n in noms]).astype(float)
    M, Y = procuste(formes, poids, A.V)
    j, dm = miroir(A.V)
    ok = dm < 0.02 * A.iod
    M = symetriser(M, j, ok)
    # distance of each character to the mean, in interocular distances of the mean
    iod = float(np.linalg.norm(M[A.boucle_oeil["g"]].mean(0) - M[A.boucle_oeil["d"]].mean(0)))
    dist = {}
    for i, n in enumerate(noms):
        Yi = appliquer(formes[i], similitude(formes[i], M, poids[i] + 1e-6))
        w = poids[i]
        dist[n] = float(np.sqrt((w * ((Yi - M) ** 2).sum(1)).sum() / w.sum()) / iod)
    return {"A": A, "noms": noms, "formes": formes, "poids": poids, "M": M, "Y": Y,
            "distances": dist, "symetrie_sommets": int(ok.sum()), "recalage": {n: res[n][2] for n in noms}}


def echelles_locales(X0, X, adj):
    """Ratio of the mean lengths of neighbouring edges, after over before, at each vertex."""
    k = np.ones(len(X0))
    for i in range(len(X0)):
        nb = list(adj[i])
        if nb:
            k[i] = np.linalg.norm(X[nb] - X[i], axis=1).mean() / max(np.linalg.norm(X0[nb] - X0[i], axis=1).mean(), 1e-12)
    return k


def ecrire_pivot(A, M, poids, sortie, dist):
    """Writes the base with the skin in shape M; neck seam held, the rest carried along.
    The displacement M - base is taken only where at least half of the heads are reliable
    (eye, mouth and neck rims: few heads, folded mean); elsewhere it is
    extended smoothly (min sum c |D - D0|² + mu |L D|², c grows with the reliable share)."""
    XA = A.V.copy()
    dcou = G.dijkstra(XA, A.adj, A.cou) if len(A.cou) else np.full(A.n, np.inf)
    p = 1 - G.lisser(dcou / (RT.RAYON_COU * A.iod))
    part = poids.mean(0)
    c = np.clip((part - SEUIL_FIABLES) / (1 - SEUIL_FIABLES), 0, 1) ** 2 + 1e-4
    c = np.where(p >= 0.999, 1e4, c)
    L = G.laplacien(A.n, A.F)
    D0 = (1 - p)[:, None] * (M - XA)
    X = XA + np.linalg.solve(np.diag(c) + LISSAGE_PIVOT * G.ltl(L), c[:, None] * D0)
    # eyes and mouth keep the base's shape (moved, rotated, scaled): their
    # morphs (blink, あ) are made for that particular slit
    for boucle, rayon in ((A.boucle_oeil["g"], RAYON_OEIL), (A.boucle_oeil["d"], RAYON_OEIL),
                          (A.boucle_bouche, RAYON_BOUCHE)):
        dz = G.dijkstra(XA, A.adj, np.asarray(boucle)) / (rayon * A.iod)
        r = 1 - G.lisser((dz - 0.5) / 0.5)
        dedans = r > 0.5
        X = r[:, None] * appliquer(XA, similitude(XA[dedans], X[dedans], np.ones(int(dedans.sum())))) + \
            (1 - r[:, None]) * X
    D = X - XA
    K = echelles_locales(XA, X, A.adj)
    P = copy.deepcopy(A.pmx)
    P["textures"] = [RT._texture_rel(x, os.path.dirname(A.chemin), sortie) for x in P["textures"]]
    pos = np.asarray(A.pos).copy()
    nor = np.asarray(A.t["nor"]).copy()
    vb = A.peau_bruts
    k = A.b2s[vb]
    pos[vb] = X[k]
    Rv = RT.rotations_locales(XA, X, A.F)
    nor[vb] = np.einsum("nij,nj->ni", Rv[k], nor[vb])
    autres = np.where((A.poids_tete > 0) & (A.b2s < 0))[0]
    Dsh = G.shepard(XA, D, A.pos[autres], k=16)
    pos[autres] += Dsh * A.poids_tete[autres][:, None]
    _, jn = RT._plus_proche_sommet(XA, A.pos[autres])
    nor[autres] = np.einsum("nij,nj->ni", Rv[jn], nor[autres])
    # morphs: local rotation and local scale (as in forme mode; the full gradient
    # degenerates on thin eyelids and flips the triangles)
    Jb = {}
    for i, kk in zip(vb, k):
        Jb[int(i)] = Rv[kk] * K[kk]
    for i, jj in zip(autres, jn):
        Jb[int(i)] = Rv[jj] * K[jj]
    for m in P["morphes"]:
        if m["type"] != 1:
            continue
        m["decalages"] = [(i, tuple(Jb[i] @ np.asarray(d)) if i in Jb else d) for i, d in m["decalages"]]
    P["sommets"]["pos"] = [tuple(x) for x in pos]
    P["sommets"]["nor"] = [tuple(x) for x in nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)]
    P["nom"] = "tete-moyenne"
    P["commentaire"] = "[mean head: mean shape of %d human heads from %s, mesh of %s]" % (
        len(dist), JEU.valeur("abrege", JEU.titre), A.nom)
    chemin = os.path.join(sortie, "tete-moyenne.pmx")
    pmx_ecrire.ecrire(P, chemin)
    return chemin, X


def verifier(A, chemin):
    """Checks of the written PMX: mesh, quality, morphs (as in retarget_tete)."""
    R = TT.Tete(chemin)
    m, echecs = {}, []

    def note(nom, d):
        m[nom] = d
        if not d.get("passe", True):
            echecs.append(nom)

    bR, nvR, orR = VF._bord_et_orientation(R.F)
    bA = VF._bord_et_orientation(A.F)[0]
    note("topologie", VF._ok({"aretes_bord": bR, "attendues": bA, "non_varietes": nvR}, bR == bA and nvR == 0 and orR == 0))
    qR, qA = G.qualite(R.V, R.F), G.qualite(A.V, A.F)
    note("qualite", VF._ok({"resultat": qR, "base": qA},
                           qR["diedre_q99_deg"] <= qA["diedre_q99_deg"] * 1.10 + 1.0 and
                           qR["part_aretes_sup15"] <= qA["part_aretes_sup15"] + 0.01))
    mA, mR = VF._mesures_morphes(A), VF._mesures_morphes(R)
    note("clignement", VF._ok({"base": mA["cligne_aire_ratio"], "pivot": mR["cligne_aire_ratio"]},
                              all(mR["cligne_aire_ratio"][c] <= mA["cligne_aire_ratio"][c] + 0.05 for c in "gd")))
    ra = mR["bouche_a_iod2"] / max(mA["bouche_a_iod2"], 1e-12)
    note("bouche", VF._ok({"base_a_iod2": mA["bouche_a_iod2"], "pivot_a_iod2": mR["bouche_a_iod2"], "rapport": ra},
                          0.7 <= ra <= 1.3))
    tol = max(1, int(0.01 * mR["n_sommets_oeil"]))
    note("oeil_dans_la_peau", VF._ok({"base": [mA["oeil_devant_repos"], mA["oeil_devant_cligne"]],
                                      "pivot": [mR["oeil_devant_repos"], mR["oeil_devant_cligne"]]},
                                     mR["oeil_devant_repos"] <= mA["oeil_devant_repos"] + tol and
                                     mR["oeil_devant_cligne"] <= mA["oeil_devant_cligne"] + tol))
    fA, fR = VF._retournes_total(A, sorted(A.morphes)), VF._retournes_total(R, sorted(R.morphes))
    note("retournements_morphes", VF._ok({"base": fA, "pivot": fR}, fR <= fA))
    m["verdict"] = {"passe": not echecs, "echecs": echecs}
    for k, v in m.items():
        if k != "verdict":
            journal("  %-24s %s" % (k, "ok" if v["passe"] else "FAILED"))
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", dest="sortie", required=True)
    ap.add_argument("--base", default=JEU.valeur("tete_base.premier_passage"), help="base head for the first pass")
    ap.add_argument("--jobs", dest="travaux", type=int, default=4)
    ap.add_argument("--no-sheet", dest="sans_planche", action="store_true")
    ap.add_argument("--average", dest="moyenne", action="store_true", help="also write tete-moyenne.pmx")
    ap.add_argument("--resume", dest="reprendre", action="store_true",
                    help="reuse the pass-2 registrations already done (formes.npz, mesures.json in --output)")
    a = ap.parse_args(argv)
    if not PERSONNAGES:
        raise SystemExit("pivot: no model found in %s (source of the active game)" % SOURCES)
    if a.base not in PERSONNAGES:      # the game's sample head is not on this machine: the configured one, otherwise the first
        nom = JEU.valeur("tete_base.nom")
        autre = nom if nom in PERSONNAGES else sorted(PERSONNAGES)[0]
        journal("base \"%s\" not among the models: first pass on \"%s\"" % (a.base, autre))
        a.base = autre
    sortie = os.path.abspath(os.path.expanduser(a.sortie))
    os.makedirs(sortie, exist_ok=True)
    t0 = time.time()
    if a.reprendre:
        anc = json.load(open(os.path.join(sortie, "mesures.json")))
        z = np.load(os.path.join(sortie, "formes.npz"))
        medoide = anc["base"]
        p2 = {"A": _base(os.path.join(SOURCES, PERSONNAGES[medoide])), "M": z["M"], "poids": z["poids"],
              "formes": z["formes"], "noms": [str(n) for n in z["noms"]],
              "distances": anc["distance_a_la_moyenne_iod"], "recalage": anc["recalage_distance_iod"],
              "symetrie_sommets": anc["sommets_symetrises"]}
        a.base = anc["base_passage_1"]
        journal("resuming registrations: base %s" % medoide)
    else:
        journal("pass 1: base %s, %d characters" % (a.base, len(PERSONNAGES)))
        p1 = passage(os.path.join(SOURCES, PERSONNAGES[a.base]), a.travaux)
        medoide = min(p1["distances"], key=p1["distances"].get)
        journal("character closest to the mean: %s (%.4f)" % (medoide, p1["distances"][medoide]))
        journal("pass 2: base %s" % medoide)
        p2 = passage(os.path.join(SOURCES, PERSONNAGES[medoide]), a.travaux)
    A, M = p2["A"], p2["M"]
    noms, fo, po = p2["noms"], p2["formes"], p2["poids"]
    iod = float(np.linalg.norm(M[A.boucle_oeil["g"]].mean(0) - M[A.boucle_oeil["d"]].mean(0)))

    def ecart(ref, i):
        Y = appliquer(fo[i], similitude(fo[i], ref, po[i] + 1e-6))
        return float(np.sqrt((po[i] * ((Y - ref) ** 2).sum(1)).sum() / po[i].sum()) / iod)

    vers_pivot = {n: round(ecart(A.V, i), 4) for i, n in enumerate(noms) if n != medoide}
    vers_moyenne = {n: round(ecart(M, i), 4) for i, n in enumerate(noms) if n != medoide}
    chemin_pivot = os.path.join(SOURCES, PERSONNAGES[medoide])
    RT.ecrire_json(os.path.join(sortie, "tete-pivot.json"), {"personnage": medoide, "pmx": chemin_pivot})
    journal("pivot head: %s (%s)" % (medoide, chemin_pivot))
    d = p2["distances"]
    mesures = {
        "base_passage_1": a.base, "base": medoide, "tete_pivot": chemin_pivot, "personnages": noms, "ecartes": dict(REFUS),
        "distance_a_la_moyenne_iod": {k: round(v, 4) for k, v in sorted(d.items(), key=lambda x: x[1])},
        # what the mean shape would gain over the pivot head taken as is
        "ecart_aux_autres_iod": {"tete_pivot": round(float(np.mean(list(vers_pivot.values()) or [0.0])), 4),
                                 "forme_moyenne": round(float(np.mean(list(vers_moyenne.values()) or [0.0])), 4),
                                 "par_personnage_tete_pivot": dict(sorted(vers_pivot.items(), key=lambda x: x[1]))},
        "recalage_distance_iod": {k: round(v, 5) for k, v in p2["recalage"].items()},
        "sommets_symetrises": p2["symetrie_sommets"], "sommets_peau": int(A.n),
    }
    X = None
    if a.moyenne:
        chemin, X = ecrire_pivot(A, M, po, sortie, d)
        journal("mean head written: %s" % chemin)
        mesures["verifications_tete_moyenne"] = verifier(A, chemin)
        if not a.sans_planche:
            journal("textured contact sheet: %s" % RT.planche_textures(A.chemin, A.chemin, chemin, sortie))
    np.savez_compressed(os.path.join(sortie, "formes.npz"), M=M, formes=fo, poids=po, noms=np.array(noms), F=A.F,
                        **({"X": X} if X is not None else {}))
    mesures["duree_s"] = round(time.time() - t0, 1)
    RT.ecrire_json(os.path.join(sortie, "mesures.json"), mesures)
    return 0


if __name__ == "__main__":
    sys.exit(main())

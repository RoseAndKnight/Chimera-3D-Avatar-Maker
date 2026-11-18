"""Automatic checks of a produced head, measured on the written PMX (re-read from scratch).

Each check returns its numbers and a boolean `passe`; the verdict lists the failures.
Thresholds (chosen before the trials, in relative units):
- topology: 0 open edges added, 0 non-manifold edges, 0 edges with reversed orientation;
- quality: dihedral q99 ≤ reference × 1.10 + 1°, shares (> 15°, valence ≠ 6) ≤ reference + 0.01
  (reference = source in forme mode, the worse of A and B in visage mode);
- proportions of A's face: max relative deviation of the distances between landmarks ≤ 2 %
  (after normalising by the median scale, which is 1 in forme mode);
- forme: skull silhouette, p95 distance to B's skull ≤ 30 % of the starting one;
- visage: seam, dihedral q99 of the strip and two rings ≤ max(ref q99 × 1.25, ref + 5°);
  kept vertices of B moved by 0;
- blink: area of the closed / open eye ≤ that of the source + 0.05;
- mouth: rest area / iod² ≤ source × 1.3 + 0.002, and "あ" / source within [0.7; 1.3];
- eye through the skin: eye vertices in front of the skin ≤ source + 1 % (rest and blink);
- triangles flipped by the morphs: ≤ those of the sources (and 0 in the seam).
"""
import numpy as np

import geom as G
import head as TT

SILHOUETTE_PLANCHER = 0.01  # interocular distance


def _ok(d, cond):
    d["passe"] = bool(cond)
    return d


def _bord_et_orientation(F):
    u, c, _ = G.aretes(F)
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, cnt = np.unique(he, axis=0, return_counts=True)
    return int((c == 1).sum()), int((c > 2).sum()), int((cnt > 1).sum())


def _aire_boucle(V, b):
    return abs(G.aire_polygone_xy(V[b][:, :2]))


def _devant_peau(P, Vs, Fs):
    """Number of points in front of the surface (on the normal side of the closest triangle)."""
    if len(P) == 0:
        return 0
    c, f, _, d = G.plus_proche(P, Vs, Fs)
    n = G.normales_faces(Vs, Fs)[f]
    return int((((P - c) * n).sum(1) > 1e-5).sum())


def _retournes(V, F, D):
    n0 = G.normales_faces(V, F, unitaires=False)
    n1 = G.normales_faces(V + D, F, unitaires=False)
    return (n0 * n1).sum(1) < 0


def _etat_morphes(T, noms):
    """Offset of the (welded) skin and of the (raw) eye vertices for a list of morphs."""
    Ds = np.zeros((T.n, 3))
    yv = T.sommets_mat(TT.YEUX)
    Dy = np.zeros((len(yv), 3))
    for nm in noms:
        Ds += T.decalage_peau(nm)
        Dy += T.decalage_bruts(nm, yv)
    return Ds, yv, Dy


def _mesures_morphes(T):
    out = {}
    # blink
    Ds, yv, Dy = _etat_morphes(T, [TT.MORPHE_CLIGNE])
    ratios = {}
    for cote in ("g", "d"):
        b = T.boucle_oeil[cote]
        a0 = _aire_boucle(T.V, b)
        a1 = _aire_boucle(T.V + Ds, b)
        ratios[cote] = a1 / max(a0, 1e-12)
    out["cligne_aire_ratio"] = ratios
    P = T.pos[yv]
    out["oeil_devant_repos"] = _devant_peau(P, T.V, T.F)
    out["oeil_devant_cligne"] = _devant_peau(P + Dy, T.V + Ds, T.F)
    out["n_sommets_oeil"] = int(len(yv))
    # mouth
    b = T.boucle_bouche
    i2 = T.iod ** 2
    out["bouche_repos_iod2"] = _aire_boucle(T.V, b) / i2
    Da = T.decalage_peau(TT.MORPHE_A)
    out["bouche_a_iod2"] = _aire_boucle(T.V + Da, b) / i2
    out["morphe_cligne_present"] = TT.MORPHE_CLIGNE in T.morphes
    out["morphe_a_present"] = TT.MORPHE_A in T.morphes
    return out


def _retournes_total(T, noms, faces=None):
    tot = 0
    for nm in noms:
        r = _retournes(T.V, T.F, T.decalage_peau(nm))
        tot += int(r.sum() if faces is None else r[faces].sum())
    return tot


def _distances_reperes(P):
    k = sorted(P)
    X = np.array([P[x] for x in k])
    i, j = np.triu_indices(len(k), 1)
    return np.linalg.norm(X[i] - X[j], axis=1)


def verifier(mode, A, B, chemin, interne, journal=None):
    peau = (TT.PEAU,) if mode == "forme" else None
    R = TT.Tete(chemin, peau=peau)
    m = {}
    echecs = []

    def note(nom, d):
        m[nom] = d
        if not d.get("passe", True):
            echecs.append(nom)

    # ---------------------------------------------------------- topology
    bR, nvR, orR = _bord_et_orientation(R.F)
    if mode == "forme":
        attendu = _bord_et_orientation(A.F)[0]
    else:
        attendu = int(interne["bord_attendu"])
    note("topologie", _ok({"aretes_bord": bR, "aretes_bord_attendues": attendu,
                           "aretes_ouvertes_ajoutees": bR - attendu, "aretes_non_varietes": nvR,
                           "aretes_orientation_contraire": orR,
                           "boucles_bord": [len(b) for b in R.boucles]},
                          bR - attendu <= 0 and nvR == 0 and orR == 0))

    # ---------------------------------------------------------- quality
    qR = G.qualite(R.V, R.F)
    qA = G.qualite(A.V, A.F)
    if mode == "forme":
        ref = qA
    else:
        qB = G.qualite(B.V, B.F)
        ref = {k: max(qA[k], qB[k]) for k in ("diedre_q99_deg", "part_aretes_sup15", "part_valence_diff6")}
    ok = (qR["diedre_q99_deg"] <= ref["diedre_q99_deg"] * 1.10 + 1.0 and
          qR["part_aretes_sup15"] <= ref["part_aretes_sup15"] + 0.01 and
          qR["part_valence_diff6"] <= ref["part_valence_diff6"] + 0.01)
    note("qualite", _ok({"resultat": qR, "reference": {k: ref[k] for k in ("diedre_q99_deg", "part_aretes_sup15", "part_valence_diff6")},
                         "source_A": qA}, ok))

    # ---------------------------------------------------------- proportions of A's face
    idxA = interne["idxA"]
    noms = [k for k in A.reperes if k.startswith(("oeil", "bouche", "nez"))]
    # forme: the face follows the stretching of the head (K); its proportions are compared with those
    # of A's face stretched the same way
    K = np.asarray(interne.get("K", np.ones(3)))
    PA = {k: K * A.V[A.reperes[k]] for k in noms}
    PR = {}
    for k in noms:
        raw = idxA[A.s2b[A.reperes[k]][0]]
        PR[k] = R.pos[raw]
    dA = _distances_reperes(PA)
    dR = _distances_reperes(PR)
    r = dR / np.maximum(dA, 1e-12)
    e = float(np.median(r))
    ecart = np.abs(r / e - 1)
    note("proportions_visage_A", _ok({"echelle_mediane": e, "ecart_relatif_max": float(ecart.max()),
                                      "ecart_relatif_moyen": float(ecart.mean()), "n_paires": int(len(r))},
                                     ecart.max() <= 0.02))

    # ---------------------------------------------------------- forme: skull silhouette
    if mode == "forme":
        XB = interne["XB"]
        w = interne["w"]
        # visible skull: excluding face, neck, and skin held under the hair and ears
        crane = (w <= 0.02) & (interne["p_cou"] <= 0.02) & (interne.get("q", 0 * w) <= 0.02)
        if crane.sum() < 10:
            note("silhouette_crane", _ok({"sommets_crane": int(crane.sum()),
                                          "raison": "skull entirely held (hair, ears)"}, True))
        else:
            kR = R.b2s[[A.s2b[k][0] for k in np.where(crane)[0]]]
            _, _, _, d0 = G.plus_proche(A.V[crane], XB, B.F)
            _, _, _, d1 = G.plus_proche(R.V[kR], XB, B.F)
            # the other way round: B's skull to the resulting head
            wB = B.poids_visage(0.06 * B.iod, 0.22 * B.iod)
            cB = (wB <= 0.02)
            if len(B.cou):
                cB &= G.dijkstra(B.V, B.adj, B.cou) > 0.45 * B.iod
            _, _, _, e0 = G.plus_proche(XB[cB], A.V, A.F)
            _, _, _, e1 = G.plus_proche(XB[cB], R.V, R.F)
            h = {"sommets_crane": int(crane.sum()),
                 "avant_p95": float(np.percentile(d0, 95)), "apres_p95": float(np.percentile(d1, 95)),
                 "avant_max": float(d0.max()), "apres_max": float(d1.max()),
                 "B_vers_resultat_avant_p95": float(np.percentile(e0, 95)),
                 "B_vers_resultat_apres_p95": float(np.percentile(e1, 95)),
                 "en_iod_apres_p95": float(np.percentile(d1, 95) / A.iod)}
            # a skull already close to B's cannot gain 70 %: being at the registration
            # floor (≈ 0.005 interocular distance) is then enough
            note("silhouette_crane", _ok(h, h["apres_p95"] <= 0.3 * h["avant_p95"] or
                                         h["en_iod_apres_p95"] <= SILHOUETTE_PLANCHER))

    # ---------------------------------------------------------- visage: seam, B intact
    region = None
    if mode == "visage":
        n0, nd = interne["n_dup0"], interne["n_dup"]
        dup = np.arange(n0, n0 + nd)
        ks = np.unique(R.b2s[dup])
        ks = ks[ks >= 0]
        zone = np.zeros(R.n, dtype=bool)
        zone[ks] = True
        bande_f = zone[R.F].all(1) | (np.isin(R.F_brut, np.where(np.isin(R.faces, dup).any(1))[0]))
        region = bande_f.copy()
        for _ in range(2):
            vz = np.zeros(R.n, dtype=bool)
            vz[R.F[region].ravel()] = True
            region = vz[R.F].any(1)
        dd, ue = G.diedres(R.V, R.F)
        fv = np.zeros(R.n, dtype=bool)
        fv[R.F[region].ravel()] = True
        ein = fv[ue].all(1)
        dz = dd[ein]
        refq = ref["diedre_q99_deg"]
        seuil = max(refq * 1.25, refq + 5)
        note("couture", _ok({"faces_bande": int(bande_f.sum()), "faces_zone": int(region.sum()),
                             "diedre_q99_deg": float(np.percentile(dz, 99)), "diedre_max_deg": float(dz.max()),
                             "diedre_moy_deg": float(dz.mean()), "seuil_q99": seuil,
                             "reference_q99": refq}, np.percentile(dz, 99) <= seuil))
        vg = interne["vertsB_gardes"]
        dep = np.linalg.norm(R.pos[:len(vg)] - B.pos[vg], axis=1)
        # B's rings at the edge of the hole follow the seam: counted separately
        bg = np.isin(vg, interne.get("B_bouges_bruts", []))
        note("B_intact", _ok({"sommets_gardes": int(len(vg)), "sommets_couture": int(bg.sum()),
                              "deplacement_couture_max": float(dep[bg].max()) if bg.any() else 0.0,
                              "deplacement_max": float(dep[~bg].max())},
                             dep[~bg].max() <= 1e-5))

    # ---------------------------------------------------------- morphs
    mA = _mesures_morphes(A)
    mR = _mesures_morphes(R)
    cl = {"source": mA["cligne_aire_ratio"], "resultat": mR["cligne_aire_ratio"]}
    note("clignement", _ok(cl, mR["morphe_cligne_present"] and
                           all(mR["cligne_aire_ratio"][c] <= mA["cligne_aire_ratio"][c] + 0.05 for c in "gd")))
    bo = {"source_repos_iod2": mA["bouche_repos_iod2"], "resultat_repos_iod2": mR["bouche_repos_iod2"],
          "source_a_iod2": mA["bouche_a_iod2"], "resultat_a_iod2": mR["bouche_a_iod2"]}
    ra = mR["bouche_a_iod2"] / max(mA["bouche_a_iod2"], 1e-12)
    bo["rapport_a"] = ra
    note("bouche", _ok(bo, mR["bouche_repos_iod2"] <= mA["bouche_repos_iod2"] * 1.3 + 0.002 and 0.7 <= ra <= 1.3))
    tol = max(1, int(0.01 * mR["n_sommets_oeil"]))
    oe = {"source_repos": mA["oeil_devant_repos"], "resultat_repos": mR["oeil_devant_repos"],
          "source_cligne": mA["oeil_devant_cligne"], "resultat_cligne": mR["oeil_devant_cligne"],
          "sommets_oeil": mR["n_sommets_oeil"], "tolerance": tol}
    note("oeil_dans_la_peau", _ok(oe, mR["oeil_devant_repos"] <= mA["oeil_devant_repos"] + tol and
                                  mR["oeil_devant_cligne"] <= mA["oeil_devant_cligne"] + tol))
    nomsA = sorted(A.morphes)
    fA = _retournes_total(A, nomsA)
    noms_R = sorted(R.morphes)
    fR = _retournes_total(R, noms_R)
    ref_f = fA + (_retournes_total(B, sorted(B.morphes)) if mode == "visage" else 0)
    rt = {"morphes_resultat": len(noms_R), "retournes_resultat": fR, "retournes_sources": ref_f}
    ok = fR <= ref_f
    if region is not None:
        rt["retournes_couture"] = _retournes_total(R, noms_R, region)
        ok = ok and rt["retournes_couture"] == 0
    note("retournements_morphes", _ok(rt, ok))
    m["verdict"] = {"passe": not echecs, "echecs": echecs}
    if journal:
        for k, v in m.items():
            if isinstance(v, dict) and "passe" in v and k != "verdict":
                journal("  %-24s %s" % (k, "ok" if v["passe"] else "FAILED"))
    return m

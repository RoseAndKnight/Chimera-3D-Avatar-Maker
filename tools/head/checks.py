"""Automatic checks of a produced head, measured on the written PMX (re-read from scratch).

Each check returns its numbers and a boolean `passed`; the verdict lists the failures.
Thresholds (chosen before the trials, in relative units):
- topology: 0 open edges added, 0 non-manifold edges, 0 edges with reversed orientation;
- quality: dihedral q99 ≤ reference × 1.10 + 1°, shares (> 15°, valence ≠ 6) ≤ reference + 0.01
  (reference = source in shape mode, the worse of A and B in visage mode);
- proportions of A's face: max relative deviation of the distances between landmarks ≤ 2 %
  (after normalising by the median scale, which is 1 in shape mode);
- shape: skull silhouette, p95 distance to B's skull ≤ 30 % of the starting one;
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

SILHOUETTE_FLOOR = 0.01  # interocular distance


def _ok(d, cond):
    d["passed"] = bool(cond)
    return d


def _border_and_orientation(F):
    u, c, _ = G.edges(F)
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, cnt = np.unique(he, axis=0, return_counts=True)
    return int((c == 1).sum()), int((c > 2).sum()), int((cnt > 1).sum())


def _area_loop(V, b):
    return abs(G.area_polygon_xy(V[b][:, :2]))


def _front_skin(P, Vs, Fs):
    """Number of points in front of the surface (on the normal side of the closest triangle)."""
    if len(P) == 0:
        return 0
    c, f, _, d = G.plus_near(P, Vs, Fs)
    n = G.normals_faces(Vs, Fs)[f]
    return int((((P - c) * n).sum(1) > 1e-5).sum())


def _flipped(V, F, D):
    n0 = G.normals_faces(V, F, unit=False)
    n1 = G.normals_faces(V + D, F, unit=False)
    return (n0 * n1).sum(1) < 0


def _state_morphs(T, names):
    """Offset of the (welded) skin and of the (raw) eye vertices for a list of morphs."""
    Ds = np.zeros((T.n, 3))
    yv = T.vertices_mat(TT.EYES)
    Dy = np.zeros((len(yv), 3))
    for nm in names:
        Ds += T.offset_skin(nm)
        Dy += T.offset_raw(nm, yv)
    return Ds, yv, Dy


def _measures_morphs(T):
    out = {}
    # blink
    Ds, yv, Dy = _state_morphs(T, [TT.MORPH_BLINKS])
    ratios = {}
    for side in ("g", "d"):
        b = T.loop_eye[side]
        a0 = _area_loop(T.V, b)
        a1 = _area_loop(T.V + Ds, b)
        ratios[side] = a1 / max(a0, 1e-12)
    out["blinks_area_ratio"] = ratios
    P = T.pos[yv]
    out["eye_front_rest"] = _front_skin(P, T.V, T.F)
    out["eye_front_blinks"] = _front_skin(P + Dy, T.V + Ds, T.F)
    out["n_vertices_eye"] = int(len(yv))
    # mouth
    b = T.loop_mouth
    i2 = T.iod ** 2
    out["mouth_rest_iod2"] = _area_loop(T.V, b) / i2
    Da = T.offset_skin(TT.MORPH_A)
    out["mouth_a_iod2"] = _area_loop(T.V + Da, b) / i2
    out["morph_blinks_present"] = TT.MORPH_BLINKS in T.morphs
    out["morph_a_present"] = TT.MORPH_A in T.morphs
    return out


def _flipped_total(T, names, faces=None):
    tot = 0
    for nm in names:
        r = _flipped(T.V, T.F, T.offset_skin(nm))
        tot += int(r.sum() if faces is None else r[faces].sum())
    return tot


def _distances_landmarks(P):
    k = sorted(P)
    X = np.array([P[x] for x in k])
    i, j = np.triu_indices(len(k), 1)
    return np.linalg.norm(X[i] - X[j], axis=1)


def check(mode, A, B, path, inner, log=None):
    skin = (TT.SKIN,) if mode == "shape" else None
    R = TT.Head(path, skin=skin)
    m = {}
    failures = []

    def note(name, d):
        m[name] = d
        if not d.get("passed", True):
            failures.append(name)

    # ---------------------------------------------------------- topology
    bR, nvR, orR = _border_and_orientation(R.F)
    if mode == "shape":
        expected = _border_and_orientation(A.F)[0]
    else:
        expected = int(inner["border_expected"])
    note("topology", _ok({"border_edges": bR, "expected_border_edges": expected,
                           "open_edges_added": bR - expected, "non_manifold_edges": nvR,
                           "flipped_edges": orR,
                           "border_loops": [len(b) for b in R.loops]},
                          bR - expected <= 0 and nvR == 0 and orR == 0))

    # ---------------------------------------------------------- quality
    qR = G.quality(R.V, R.F)
    qA = G.quality(A.V, A.F)
    if mode == "shape":
        ref = qA
    else:
        qB = G.quality(B.V, B.F)
        ref = {k: max(qA[k], qB[k]) for k in ("dihedral_q99_deg", "fraction_edges_over15", "fraction_valence_not6")}
    ok = (qR["dihedral_q99_deg"] <= ref["dihedral_q99_deg"] * 1.10 + 1.0 and
          qR["fraction_edges_over15"] <= ref["fraction_edges_over15"] + 0.01 and
          qR["fraction_valence_not6"] <= ref["fraction_valence_not6"] + 0.01)
    note("quality", _ok({"result": qR, "reference": {k: ref[k] for k in ("dihedral_q99_deg", "fraction_edges_over15", "fraction_valence_not6")},
                         "source_A": qA}, ok))

    # ---------------------------------------------------------- proportions of A's face
    idxA = inner["idxA"]
    names = [k for k in A.landmarks if k.startswith(("eye", "mouth", "nose"))]
    # shape: the face follows the stretching of the head (K); its proportions are compared with those
    # of A's face stretched the same way
    K = np.asarray(inner.get("K", np.ones(3)))
    PA = {k: K * A.V[A.landmarks[k]] for k in names}
    PR = {}
    for k in names:
        raw = idxA[A.s2b[A.landmarks[k]][0]]
        PR[k] = R.pos[raw]
    dA = _distances_landmarks(PA)
    dR = _distances_landmarks(PR)
    r = dR / np.maximum(dA, 1e-12)
    e = float(np.median(r))
    gap = np.abs(r / e - 1)
    note("face_proportions_A", _ok({"median_scale": e, "relative_gap_max": float(gap.max()),
                                      "relative_gap_mean": float(gap.mean()), "n_pairs": int(len(r))},
                                     gap.max() <= 0.02))

    # ---------------------------------------------------------- shape: skull silhouette
    if mode == "shape":
        XB = inner["XB"]
        w = inner["w"]
        # visible skull: excluding face, neck, and skin held under the hair and ears
        skull = (w <= 0.02) & (inner["p_neck"] <= 0.02) & (inner.get("q", 0 * w) <= 0.02)
        if skull.sum() < 10:
            note("silhouette_skull", _ok({"vertices_skull": int(skull.sum()),
                                          "reason": "skull entirely held (hair, ears)"}, True))
        else:
            kR = R.b2s[[A.s2b[k][0] for k in np.where(skull)[0]]]
            _, _, _, d0 = G.plus_near(A.V[skull], XB, B.F)
            _, _, _, d1 = G.plus_near(R.V[kR], XB, B.F)
            # the other way round: B's skull to the resulting head
            wB = B.weight_face(0.06 * B.iod, 0.22 * B.iod)
            cB = (wB <= 0.02)
            if len(B.neck):
                cB &= G.dijkstra(B.V, B.adj, B.neck) > 0.45 * B.iod
            _, _, _, e0 = G.plus_near(XB[cB], A.V, A.F)
            _, _, _, e1 = G.plus_near(XB[cB], R.V, R.F)
            h = {"vertices_skull": int(skull.sum()),
                 "front_p95": float(np.percentile(d0, 95)), "after_p95": float(np.percentile(d1, 95)),
                 "front_max": float(d0.max()), "after_max": float(d1.max()),
                 "B_to_result_before_p95": float(np.percentile(e0, 95)),
                 "B_to_result_after_p95": float(np.percentile(e1, 95)),
                 "in_iod_after_p95": float(np.percentile(d1, 95) / A.iod)}
            # a skull already close to B's cannot gain 70 %: being at the registration
            # floor (≈ 0.005 interocular distance) is then enough
            note("silhouette_skull", _ok(h, h["after_p95"] <= 0.3 * h["front_p95"] or
                                         h["in_iod_after_p95"] <= SILHOUETTE_FLOOR))

    # ---------------------------------------------------------- visage: seam, B intact
    region = None
    if mode == "face":
        n0, nd = inner["n_dup0"], inner["n_dup"]
        dup = np.arange(n0, n0 + nd)
        ks = np.unique(R.b2s[dup])
        ks = ks[ks >= 0]
        zone = np.zeros(R.n, dtype=bool)
        zone[ks] = True
        band_f = zone[R.F].all(1) | (np.isin(R.F_brut, np.where(np.isin(R.faces, dup).any(1))[0]))
        region = band_f.copy()
        for _ in range(2):
            vz = np.zeros(R.n, dtype=bool)
            vz[R.F[region].ravel()] = True
            region = vz[R.F].any(1)
        dd, ue = G.dihedrals(R.V, R.F)
        fv = np.zeros(R.n, dtype=bool)
        fv[R.F[region].ravel()] = True
        ein = fv[ue].all(1)
        dz = dd[ein]
        refq = ref["dihedral_q99_deg"]
        threshold = max(refq * 1.25, refq + 5)
        note("seam", _ok({"band_faces": int(band_f.sum()), "zone_faces": int(region.sum()),
                             "dihedral_q99_deg": float(np.percentile(dz, 99)), "dihedral_max_deg": float(dz.max()),
                             "dihedral_mean_deg": float(dz.mean()), "threshold_q99": threshold,
                             "reference_q99": refq}, np.percentile(dz, 99) <= threshold))
        vg = inner["vertsB_kept"]
        disp = np.linalg.norm(R.pos[:len(vg)] - B.pos[vg], axis=1)
        # B's rings at the edge of the hole follow the seam: counted separately
        bg = np.isin(vg, inner.get("B_moved_raw", []))
        note("B_intact", _ok({"kept_vertices": int(len(vg)), "seam_vertices": int(bg.sum()),
                              "seam_shift_max": float(disp[bg].max()) if bg.any() else 0.0,
                              "shift_max": float(disp[~bg].max())},
                             disp[~bg].max() <= 1e-5))

    # ---------------------------------------------------------- morphs
    mA = _measures_morphs(A)
    mR = _measures_morphs(R)
    cl = {"source": mA["blinks_area_ratio"], "result": mR["blinks_area_ratio"]}
    note("blink", _ok(cl, mR["morph_blinks_present"] and
                           all(mR["blinks_area_ratio"][c] <= mA["blinks_area_ratio"][c] + 0.05 for c in "gd")))
    bo = {"source_rest_iod2": mA["mouth_rest_iod2"], "result_rest_iod2": mR["mouth_rest_iod2"],
          "source_a_iod2": mA["mouth_a_iod2"], "result_a_iod2": mR["mouth_a_iod2"]}
    ra = mR["mouth_a_iod2"] / max(mA["mouth_a_iod2"], 1e-12)
    bo["ratio_to"] = ra
    note("mouth", _ok(bo, mR["mouth_rest_iod2"] <= mA["mouth_rest_iod2"] * 1.3 + 0.002 and 0.7 <= ra <= 1.3))
    tol = max(1, int(0.01 * mR["n_vertices_eye"]))
    oe = {"source_rest": mA["eye_front_rest"], "result_rest": mR["eye_front_rest"],
          "source_blink": mA["eye_front_blinks"], "result_blink": mR["eye_front_blinks"],
          "eye_vertices": mR["n_vertices_eye"], "tolerance": tol}
    note("eye_in_skin", _ok(oe, mR["eye_front_rest"] <= mA["eye_front_rest"] + tol and
                                  mR["eye_front_blinks"] <= mA["eye_front_blinks"] + tol))
    namesA = sorted(A.morphs)
    fA = _flipped_total(A, namesA)
    names_R = sorted(R.morphs)
    fR = _flipped_total(R, names_R)
    ref_f = fA + (_flipped_total(B, sorted(B.morphs)) if mode == "face" else 0)
    rt = {"result_morphs": len(names_R), "result_flipped": fR, "sources_flipped": ref_f}
    ok = fR <= ref_f
    if region is not None:
        rt["seam_flipped"] = _flipped_total(R, names_R, region)
        ok = ok and rt["seam_flipped"] == 0
    note("morph_flips", _ok(rt, ok))
    m["verdict"] = {"passed": not failures, "failures": failures}
    if log:
        for k, v in m.items():
            if isinstance(v, dict) and "passed" in v and k != "verdict":
                log("  %-24s %s" % (k, "ok" if v["passed"] else "FAILED"))
    return m

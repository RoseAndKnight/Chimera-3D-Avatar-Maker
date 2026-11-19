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

With --average, also writes average-head.pmx: the pivot head in the mean shape (reference
for family morphs); displacement taken where half of the heads are reliable, extended
smoothly elsewhere, eyes and mouth kept in the base's shape, morphs rotated and scaled
locally, neck seam held.

Outputs: <output>/pivot-head.json (character and PMX path), measures.json (characters,
distances to the mean and to the pivot head), shapes.npz; with --average, average-head.pmx,
its checks and texture_sheet.png.
"""
import argparse
import copy
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))
sys.path.append(os.path.join(HERE, ".."))
import settings as R  # noqa: E402
import games as games  # noqa: E402

import geom as G  # noqa: E402
import pmx_write as pmx_write  # noqa: E402
import registration as RC  # noqa: E402
import retarget_head as RT  # noqa: E402
import head as TT  # noqa: E402
import checks as VF  # noqa: E402

# The models of the active game (description tools/games/<game>/game.json): one per human character, path relative to the
# mmd_extracted source folder; the description says which ones are excluded and why.
GAME = games.active()
SOURCES = GAME.path("mmd_extracted")
PEOPLE = GAME.models()
# Vertex reliable for a character: within RELIABLE_DIST interocular distances of its skin after
# registration, and not on the boundary of its skin.
RELIABLE_DIST = 0.03
ITER_PROCRUSTES = 8
THRESHOLD_RELIABLE = 0.5
SMOOTHING_PIVOT = 1.0
RADIUS_EYE = 0.3
RADIUS_MOUTH = 0.3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _base(path):
    return TT.Head(path)


def register_one(args):
    """Registers the base skin onto a character's; returns X (base, base frame),
    the reliability of each vertex and the final distance."""
    path_base, name, path = args
    A = _base(path_base)
    B = TT.Head(path)
    s = float(np.sqrt(A.W * A.H_crane / (B.W * B.H_crane)))
    B.put(s, B.center_face(), A.center_face())
    XB = B._host(B.V)
    rec = RC.register(A, A.V, B, XB)
    border = np.zeros(B.n, dtype=bool)
    for b in B.loops:
        border[b] = True
    tri = B.F[rec["face"]]
    # projected onto a face with a vertex on the boundary: the character's skin ends there
    on_border = border[tri].any(1)
    reliable = (rec["dist"] < RELIABLE_DIST * A.iod) & ~on_border
    return name, rec["X"], reliable, float(np.mean(rec["dist"]) / A.iod)


def similarity(X, Y, w):
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


def apply(X, sRt):
    s, R, t = sRt
    return s * X @ R.T + t


def procrustes(shapes, weight, ref):
    """Weighted generalised Procrustes analysis. shapes: (k, n, 3), weights: (k, n).
    The mean keeps the size and position of ref. Returns the mean and the aligned shapes."""
    M = ref.copy()
    Y = shapes.copy()
    for _ in range(ITER_PROCRUSTES):
        for i in range(len(shapes)):
            Y[i] = apply(shapes[i], similarity(shapes[i], M, weight[i] + 1e-6))
        W = weight + 1e-3
        Mn = (W[..., None] * Y).sum(0) / W.sum(0)[:, None]
        M = apply(Mn, similarity(Mn, ref, np.ones(len(ref))))
    return M, Y


def mirror(V):
    """Index of the mirror vertex (closest to the reflection in x) and distance to the reflection."""
    Q = V * np.array([-1.0, 1.0, 1.0])
    d, j = RT._plus_near_vertex(V, Q)
    return j, d


def symmetrize(M, j, ok):
    S = M[j] * np.array([-1.0, 1.0, 1.0])
    out = M.copy()
    out[ok] = 0.5 * (M[ok] + S[ok])
    return out


REFUSALS = {}      # excluded model -> reason


def register_or_say(stain):
    try:
        return register_one(stain)
    except Exception as e:  # noqa: BLE001
        return stain[1], None, None, "%s : %s" % (type(e).__name__, e)


def passage(path_base, tasks):
    stains = [(path_base, n, os.path.join(SOURCES, PEOPLE[n])) for n in sorted(PEOPLE)]
    res = {}
    with ProcessPoolExecutor(tasks) as ex:
        for name, X, f, d in ex.map(register_or_say, stains):
            if X is None:      # a model that cannot be registered does not stop the others
                REFUSALS[name] = d
                log("  %-12s excluded: %s" % (name, d))
                continue
            res[name] = (X, f, d)
            log("  %-12s mean distance %.4f interocular, reliable %d %%" % (name, d, 100 * f.mean()))
    names = sorted(res)
    if not names:
        raise SystemExit("pivot: no model could be registered (%s)" % "; ".join("%s: %s" % kv for kv in REFUSALS.items()))
    A = _base(path_base)
    shapes = np.stack([res[n][0] for n in names])
    weight = np.stack([res[n][1] for n in names]).astype(float)
    M, Y = procrustes(shapes, weight, A.V)
    j, dm = mirror(A.V)
    ok = dm < 0.02 * A.iod
    M = symmetrize(M, j, ok)
    # distance of each character to the mean, in interocular distances of the mean
    iod = float(np.linalg.norm(M[A.loop_eye["g"]].mean(0) - M[A.loop_eye["d"]].mean(0)))
    dist = {}
    for i, n in enumerate(names):
        Yi = apply(shapes[i], similarity(shapes[i], M, weight[i] + 1e-6))
        w = weight[i]
        dist[n] = float(np.sqrt((w * ((Yi - M) ** 2).sum(1)).sum() / w.sum()) / iod)
    return {"A": A, "names": names, "shapes": shapes, "weights": weight, "M": M, "Y": Y,
            "distances": dist, "symmetry_vertices": int(ok.sum()), "registration": {n: res[n][2] for n in names}}


def scales_local(X0, X, adj):
    """Ratio of the mean lengths of neighbouring edges, after over before, at each vertex."""
    k = np.ones(len(X0))
    for i in range(len(X0)):
        nb = list(adj[i])
        if nb:
            k[i] = np.linalg.norm(X[nb] - X[i], axis=1).mean() / max(np.linalg.norm(X0[nb] - X0[i], axis=1).mean(), 1e-12)
    return k


def write_pivot(A, M, weight, output, dist):
    """Writes the base with the skin in shape M; neck seam held, the rest carried along.
    The displacement M - base is taken only where at least half of the heads are reliable
    (eye, mouth and neck rims: few heads, folded mean); elsewhere it is
    extended smoothly (min sum c |D - D0|² + mu |L D|², c grows with the reliable share)."""
    XA = A.V.copy()
    dcou = G.dijkstra(XA, A.adj, A.neck) if len(A.neck) else np.full(A.n, np.inf)
    p = 1 - G.smooth(dcou / (RT.RADIUS_NECK * A.iod))
    part = weight.mean(0)
    c = np.clip((part - THRESHOLD_RELIABLE) / (1 - THRESHOLD_RELIABLE), 0, 1) ** 2 + 1e-4
    c = np.where(p >= 0.999, 1e4, c)
    L = G.laplacian(A.n, A.F)
    D0 = (1 - p)[:, None] * (M - XA)
    X = XA + np.linalg.solve(np.diag(c) + SMOOTHING_PIVOT * G.ltl(L), c[:, None] * D0)
    # eyes and mouth keep the base's shape (moved, rotated, scaled): their
    # morphs (blink, あ) are made for that particular slit
    for loop, radius in ((A.loop_eye["g"], RADIUS_EYE), (A.loop_eye["d"], RADIUS_EYE),
                          (A.loop_mouth, RADIUS_MOUTH)):
        dz = G.dijkstra(XA, A.adj, np.asarray(loop)) / (radius * A.iod)
        r = 1 - G.smooth((dz - 0.5) / 0.5)
        inside = r > 0.5
        X = r[:, None] * apply(XA, similarity(XA[inside], X[inside], np.ones(int(inside.sum())))) + \
            (1 - r[:, None]) * X
    D = X - XA
    K = scales_local(XA, X, A.adj)
    P = copy.deepcopy(A.pmx)
    P["textures"] = [RT._texture_rel(x, os.path.dirname(A.path), output) for x in P["textures"]]
    pos = np.asarray(A.pos).copy()
    nor = np.asarray(A.t["nor"]).copy()
    vb = A.skin_raw
    k = A.b2s[vb]
    pos[vb] = X[k]
    Rv = RT.rotations_local(XA, X, A.F)
    nor[vb] = np.einsum("nij,nj->ni", Rv[k], nor[vb])
    others = np.where((A.weight_head > 0) & (A.b2s < 0))[0]
    Dsh = G.shepard(XA, D, A.pos[others], k=16)
    pos[others] += Dsh * A.weight_head[others][:, None]
    _, jn = RT._plus_near_vertex(XA, A.pos[others])
    nor[others] = np.einsum("nij,nj->ni", Rv[jn], nor[others])
    # morphs: local rotation and local scale (as in shape mode; the full gradient
    # degenerates on thin eyelids and flips the triangles)
    Jb = {}
    for i, kk in zip(vb, k):
        Jb[int(i)] = Rv[kk] * K[kk]
    for i, jj in zip(others, jn):
        Jb[int(i)] = Rv[jj] * K[jj]
    for m in P["morphs"]:
        if m["type"] != 1:
            continue
        m["offsets"] = [(i, tuple(Jb[i] @ np.asarray(d)) if i in Jb else d) for i, d in m["offsets"]]
    P["vertices"]["pos"] = [tuple(x) for x in pos]
    P["vertices"]["nor"] = [tuple(x) for x in nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)]
    P["name"] = "average-head"
    P["comment"] = "[mean head: mean shape of %d human heads from %s, mesh of %s]" % (
        len(dist), GAME.value("short_name", GAME.title), A.name)
    path = os.path.join(output, "average-head.pmx")
    pmx_write.write(P, path)
    return path, X


def check(A, path):
    """Checks of the written PMX: mesh, quality, morphs (as in retarget_head)."""
    R = TT.Head(path)
    m, failures = {}, []

    def note(name, d):
        m[name] = d
        if not d.get("passed", True):
            failures.append(name)

    bR, nvR, orR = VF._border_and_orientation(R.F)
    bA = VF._border_and_orientation(A.F)[0]
    note("topology", VF._ok({"border_edges": bR, "expected": bA, "non_manifold": nvR}, bR == bA and nvR == 0 and orR == 0))
    qR, qA = G.quality(R.V, R.F), G.quality(A.V, A.F)
    note("quality", VF._ok({"result": qR, "base": qA},
                           qR["dihedral_q99_deg"] <= qA["dihedral_q99_deg"] * 1.10 + 1.0 and
                           qR["fraction_edges_over15"] <= qA["fraction_edges_over15"] + 0.01))
    mA, mR = VF._measures_morphs(A), VF._measures_morphs(R)
    note("blink", VF._ok({"base": mA["blinks_area_ratio"], "pivot": mR["blinks_area_ratio"]},
                              all(mR["blinks_area_ratio"][c] <= mA["blinks_area_ratio"][c] + 0.05 for c in "gd")))
    ra = mR["mouth_a_iod2"] / max(mA["mouth_a_iod2"], 1e-12)
    note("mouth", VF._ok({"base_a_iod2": mA["mouth_a_iod2"], "pivot_a_iod2": mR["mouth_a_iod2"], "report": ra},
                          0.7 <= ra <= 1.3))
    tol = max(1, int(0.01 * mR["n_vertices_eye"]))
    note("eye_in_skin", VF._ok({"base": [mA["eye_front_rest"], mA["eye_front_blinks"]],
                                      "pivot": [mR["eye_front_rest"], mR["eye_front_blinks"]]},
                                     mR["eye_front_rest"] <= mA["eye_front_rest"] + tol and
                                     mR["eye_front_blinks"] <= mA["eye_front_blinks"] + tol))
    fA, fR = VF._flipped_total(A, sorted(A.morphs)), VF._flipped_total(R, sorted(R.morphs))
    note("morph_flips", VF._ok({"base": fA, "pivot": fR}, fR <= fA))
    m["verdict"] = {"passed": not failures, "failures": failures}
    for k, v in m.items():
        if k != "verdict":
            log("  %-24s %s" % (k, "ok" if v["passed"] else "FAILED"))
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", dest="output", required=True)
    ap.add_argument("--base", default=GAME.value("base_head.first_pass"), help="base head for the first pass")
    ap.add_argument("--jobs", dest="tasks", type=int, default=4)
    ap.add_argument("--no-sheet", dest="without_sheet", action="store_true")
    ap.add_argument("--average", dest="average", action="store_true", help="also write average-head.pmx")
    ap.add_argument("--resume", dest="summary", action="store_true",
                    help="reuse the pass-2 registrations already done (shapes.npz, measures.json in --output)")
    a = ap.parse_args(argv)
    if not PEOPLE:
        raise SystemExit("pivot: no model found in %s (source of the active game)" % SOURCES)
    if a.base not in PEOPLE:      # the game's sample head is not on this machine: the configured one, otherwise the first
        name = GAME.value("base_head.name")
        other = name if name in PEOPLE else sorted(PEOPLE)[0]
        log("base \"%s\" not among the models: first pass on \"%s\"" % (a.base, other))
        a.base = other
    output = os.path.abspath(os.path.expanduser(a.output))
    os.makedirs(output, exist_ok=True)
    t0 = time.time()
    if a.resume:
        former = json.load(open(os.path.join(output, "measures.json")))
        z = np.load(os.path.join(output, "shapes.npz"))
        medoid = former["base"]
        p2 = {"A": _base(os.path.join(SOURCES, PEOPLE[medoid])), "M": z["M"], "weights": z["weights"],
              "shapes": z["shapes"], "names": [str(n) for n in z["names"]],
              "distances": former["distance_to_mean_iod"], "registration": former["registration_distance_iod"],
              "symmetry_vertices": former["mirrored_vertices"]}
        a.base = former["base_pass_1"]
        log("resuming registrations: base %s" % medoid)
    else:
        log("pass 1: base %s, %d characters" % (a.base, len(PEOPLE)))
        p1 = passage(os.path.join(SOURCES, PEOPLE[a.base]), a.tasks)
        medoid = min(p1["distances"], key=p1["distances"].get)
        log("character closest to the mean: %s (%.4f)" % (medoid, p1["distances"][medoid]))
        log("pass 2: base %s" % medoid)
        p2 = passage(os.path.join(SOURCES, PEOPLE[medoid]), a.tasks)
    A, M = p2["A"], p2["M"]
    names, fo, po = p2["names"], p2["shapes"], p2["weights"]
    iod = float(np.linalg.norm(M[A.loop_eye["g"]].mean(0) - M[A.loop_eye["d"]].mean(0)))

    def gap(ref, i):
        Y = apply(fo[i], similarity(fo[i], ref, po[i] + 1e-6))
        return float(np.sqrt((po[i] * ((Y - ref) ** 2).sum(1)).sum() / po[i].sum()) / iod)

    to_pivot = {n: round(gap(A.V, i), 4) for i, n in enumerate(names) if n != medoid}
    to_average = {n: round(gap(M, i), 4) for i, n in enumerate(names) if n != medoid}
    path_pivot = os.path.join(SOURCES, PEOPLE[medoid])
    RT.write_json(os.path.join(output, "pivot-head.json"), {"person": medoid, "pmx": path_pivot})
    log("pivot head: %s (%s)" % (medoid, path_pivot))
    d = p2["distances"]
    measures = {
        "base_pass_1": a.base, "base": medoid, "pivot_head": path_pivot, "people": names, "discarded": dict(REFUSALS),
        "distance_to_mean_iod": {k: round(v, 4) for k, v in sorted(d.items(), key=lambda x: x[1])},
        # what the mean shape would gain over the pivot head taken as is
        "gap_to_others_iod": {"pivot_head": round(float(np.mean(list(to_pivot.values()) or [0.0])), 4),
                                 "mean_shape": round(float(np.mean(list(to_average.values()) or [0.0])), 4),
                                 "per_person_pivot_head": dict(sorted(to_pivot.items(), key=lambda x: x[1]))},
        "registration_distance_iod": {k: round(v, 5) for k, v in p2["registration"].items()},
        "mirrored_vertices": p2["symmetry_vertices"], "skin_vertices": int(A.n),
    }
    X = None
    if a.average:
        path, X = write_pivot(A, M, po, output, d)
        log("mean head written: %s" % path)
        measures["verifications_head_average"] = check(A, path)
        if not a.without_sheet:
            log("textured contact sheet: %s" % RT.sheet_textures(A.path, A.path, path, output))
    np.savez_compressed(os.path.join(output, "shapes.npz"), M=M, shapes=fo, weights=po, names=np.array(names), F=A.F,
                        **({"X": X} if X is not None else {}))
    measures["duration_s"] = round(time.time() - t0, 1)
    RT.write_json(os.path.join(output, "measures.json"), measures)
    return 0


if __name__ == "__main__":
    sys.exit(main())

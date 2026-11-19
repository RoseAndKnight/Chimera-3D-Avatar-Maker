"""Animation test of a head stitched on by whole_head: does the seam break when the model moves?

    python3 -I tools/clothes/head_motion_test.py FOLDER [--poses-pmx]

FOLDER is the output of whole_head (result.pmx, seam.json). Each pose (neck, head, chest,
shoulders, arms) is applied by forward kinematics and linear skinning (tools/atlas/kinematics.py,
SDEF read as BDEF2). Measurements, on the skin (颜, 肌_tete, 肌):

 - opening: spread of the vertices welded at rest (seam copies included), in mm; 0 expected;
 - stretch: |ln(posed length / rest length)| of the edges;
 - fold: change in the dihedral angle of the edges (degrees).

Zero-area faces (glued scalp strip) have no normal and are left out of the folds. Stretch and fold are
reported near the neck seam (less than 15 mm from the cut) and near the scalp edge, and on a control
zone of the same model that the seam does not touch: the body's neck 15 to 40 mm below the cut, and the
body's scalp 25 to 50 mm from the edge. A sound seam folds and stretches no more than its control.

Output: FOLDER/trial.json; with --poses-pmx, FOLDER/pose_<name>.pmx for the renders.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "atlas"), os.path.join(HERE, "..", "head")]
import pmx as PMX  # noqa: E402
import pmx_write as pmx_write  # noqa: E402
import kinematics as KIN  # noqa: E402
from whole_head import weld, point_polyline, MM  # noqa: E402

SKIN = ("颜", "肌_tete", "肌")
X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)     # PMX frame: y up, facing -z


def q(*rots):
    out = KIN.IDENTITY.copy()
    for axis, deg in rots:
        out = KIN.q_mul(out, KIN.q_axe_angle(axis, deg))
    return out


POSES = {
    "head_yaw+60": {"頭": q((Y, 60))},
    "head_yaw-60": {"頭": q((Y, -60))},
    "head_pitch+30": {"頭": q((X, 30))},
    "head_pitch-30": {"頭": q((X, -30))},
    "head_roll+25": {"頭": q((Z, 25))},
    "neck_head_lace_pitch": {"首": q((Y, 30), (X, 20)), "頭": q((Y, 30), (X, 15))},
    "neck_head_back": {"首": q((X, 35)), "頭": q((X, 30))},
    "neck_head_tilted": {"首": q((X, -30)), "頭": q((X, -25))},
    "neck_head_roll": {"首": q((Z, -20)), "頭": q((Z, -20), (Y, 25))},
    "bust_twist": {"上半身2": q((Y, 30)), "首": q((Y, 20)), "頭": q((Y, 30))},
    "shoulders_haussees": {"左肩": q((Z, 20)), "右肩": q((Z, -20))},
    "arm_raised": {"左腕": q((Z, 70)), "右腕": q((Z, -70))},
}
RENDERS = ("head_yaw+60", "head_pitch-30", "neck_head_back", "neck_head_roll", "bust_twist", "arm_raised")


def skeleton(R):
    os_ = R["bones"]
    heads = np.array([o["pos"] for o in os_], float)
    parents = np.array([o["parent"] for o in os_], int)
    src = np.full(len(os_), -1)
    fac = np.zeros(len(os_))
    arot = np.zeros(len(os_), bool)
    adep = np.zeros(len(os_), bool)
    for i, o in enumerate(os_):
        a = o.get("addition")
        if a:
            src[i], fac[i] = a["source"], a["factor"]
            arot[i], adep[i] = a["rotation"], a["displacement"]
    return heads, parents, src, fac, arot, adep


def edges_and_dihedrals(F):
    """Edges (sorted pairs of welded vertices) and, for those with two faces, those two faces."""
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.tile(np.arange(len(F)), 3)
    E = np.sort(E, 1)
    the_key = E[:, 0].astype(np.int64) * (E.max() + 1) + E[:, 1]
    o = np.argsort(the_key, kind="stable")
    the_key, E, fid = the_key[o], E[o], fid[o]
    start = np.r_[True, the_key[1:] != the_key[:-1]]
    uniques = E[start]
    group = np.cumsum(start) - 1
    n = np.bincount(group)
    two = np.where(n == 2)[0]
    first = np.where(start)[0]
    pairs = np.stack([fid[first[two]], fid[first[two] + 1]], 1)
    return uniques, uniques[two], pairs


def normals_faces(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)


def summary(x):
    return {"max": round(float(x.max()), 4) if len(x) else None,
            "p99": round(float(np.percentile(x, 99)), 4) if len(x) else None}


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    folder = a[0]
    R = PMX.read(os.path.join(folder, "result.pmx"))
    J = json.load(open(os.path.join(folder, "seam.json"), encoding="utf-8"))
    t = PMX.to_arrays(R)
    P0 = t["pos"]
    names = [m["name"] for m in R["materials"]]
    ip = [names.index(n) for n in SKIN if n in names]
    F = t["faces"][np.isin(t["mat_face"], ip)]
    # weld returns a group number: each used vertex is replaced by the first used vertex of its
    # group (an orphan vertex, with no face, never serves as representative)
    useful = np.unique(F)
    lab = weld(P0[useful])
    first = np.zeros(lab.max() + 1, int)
    first[lab[::-1]] = useful[::-1]
    sub = np.arange(len(P0))
    sub[useful] = first[lab]
    Fs = sub[F]                                          # faces on the welded vertices
    area = np.linalg.norm(np.cross(P0[Fs[:, 1]] - P0[Fs[:, 0]], P0[Fs[:, 2]] - P0[Fs[:, 0]]), axis=1) / 2
    Fs = Fs[area > (0.05 * MM) ** 2]                    # zero-area faces (glued scalp strip): meaningless normal
    E, E2, pairs = edges_and_dihedrals(Fs)
    loop = np.array(J["indices"]["loop"])
    skull = np.array(J["indices"]["skull"])

    # zones at rest
    vs = np.unique(Fs)
    d_an = np.full(len(P0), np.inf)
    d_cr = np.full(len(P0), np.inf)
    d_an[vs] = point_polyline(P0[vs], P0[loop])[4] / MM
    d_cr[vs] = point_polyline(P0[vs], P0[skull])[4] / MM
    y_neck = P0[loop, 1].mean()
    c_neck = P0[loop].mean(0)
    r_neck = np.linalg.norm((P0[loop] - c_neck)[:, [0, 2]], axis=1).max()
    dy = (y_neck - P0[:, 1]) / MM
    near_axe = np.linalg.norm((P0 - c_neck)[:, [0, 2]], axis=1) < 1.5 * r_neck
    zones = {"join_neck": d_an < 15, "witness_neck": (dy > 15) & (dy < 40) & near_axe,
             "border_skull": d_cr < 15, "witness_skull": (d_cr > 25) & (d_cr < 50) & (P0[:, 1] > y_neck + 0.3)}

    # welded groups: a single point at rest
    grp = {}
    for v in np.unique(F):
        grp.setdefault(int(sub[v]), []).append(int(v))
    multi = [g for g in grp.values() if len(g) > 1]

    sq = skeleton(R)
    name_os = {o["name"]: i for i, o in enumerate(R["bones"])}
    idx, pw = t["bones"], t["weights"]
    n0 = normals_faces(P0, Fs)
    ang0 = np.degrees(np.arccos(np.clip((n0[pairs[:, 0]] * n0[pairs[:, 1]]).sum(1), -1, 1)))
    L0 = np.linalg.norm(P0[E[:, 0]] - P0[E[:, 1]], axis=1)

    assessment = {"model": os.path.join(folder, "result.pmx"), "skinning": "linear, SDEF read as BDEF2",
             "zones_vertices": {k: int(v.sum()) for k, v in zones.items()},
             "groups_welded": len(multi), "poses": {}}
    for name, rots in POSES.items():
        missing = [b for b in rots if b not in name_os]
        if missing:
            assessment["poses"][name] = {"os_absents": missing}
            continue
        Rm, tm = KIN.kinematics(*sq, rotations={name_os[b]: qq for b, qq in rots.items()})
        P = KIN.skinning(P0, idx, pw, sq[0], Rm, tm, True)
        open_ = max((np.linalg.norm(P[g] - P[g].mean(0), axis=1).max() for g in multi), default=0.0) / MM
        L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
        strain = np.where(L0 > 0.01 * MM, np.abs(np.log(np.maximum(L, 1e-12) / np.maximum(L0, 1e-12))), 0.0)
        n1 = normals_faces(P, Fs)
        ang = np.degrees(np.arccos(np.clip((n1[pairs[:, 0]] * n1[pairs[:, 1]]).sum(1), -1, 1)))
        fold = np.abs(ang - ang0)
        res = {"opening_mm": round(float(open_), 6)}
        for z, m in zones.items():
            ze = m[E[:, 0]] & m[E[:, 1]]
            zd = m[E2[:, 0]] & m[E2[:, 1]]
            res[z] = {"stretch": summary(strain[ze]), "fold_deg": summary(fold[zd])}
        assessment["poses"][name] = res
        if "--poses-pmx" in a and name in RENDERS:
            Q = dict(R)
            Q["vertices"] = dict(R["vertices"])
            Q["vertices"]["pos"] = [tuple(p) for p in P.tolist()]
            ch = os.path.join(folder, "pose_%s.pmx" % name)
            pmx_write.write(Q, ch + ".tmp")
            os.replace(ch + ".tmp", ch)
    ch = os.path.join(folder, "trial.json")
    with open(ch + ".tmp", "w", encoding="utf-8") as f:
        json.dump(assessment, f, ensure_ascii=False, indent=1)
    os.replace(ch + ".tmp", ch)
    for name, r in assessment["poses"].items():
        if "opening_mm" not in r:
            print(name, r)
            continue
        print("%-24s open %.4f  neck str %.3f/%.3f fold %5.1f/%5.1f   scalp str %.3f/%.3f fold %5.1f/%5.1f" % (
            name, r["opening_mm"],
            r["join_neck"]["stretch"]["max"], r["witness_neck"]["stretch"]["max"],
            r["join_neck"]["fold_deg"]["max"], r["witness_neck"]["fold_deg"]["max"],
            r["border_skull"]["stretch"]["max"], r["witness_skull"]["stretch"]["max"],
            r["border_skull"]["fold_deg"]["max"], r["witness_skull"]["fold_deg"]["max"]))


if __name__ == "__main__":
    main()

"""Remaking the base body: a mesh of our own, placed on the current body taken as the shape reference.

  python3 -I rebuild.py --reference <corps.pmx> --output <folder>

Writes <folder>/corps.pmx (same bones, same materials, original head and skull kept as they are, body
skin remade), <folder>/tex/ (copy) and <folder>/measures.json. Blender's Python (numpy), without bpy.
"""
import argparse
import json
import os
import pickle
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "atlas"), os.path.join(HERE, "..", "head")):
    if d not in sys.path:
        sys.path.insert(0, d)

import fit as AJ          # noqa: E402
import sliders as CU         # noqa: E402
import geom as G              # noqa: E402
import geometry as GE        # noqa: E402
import pmx_write as pmx_write             # noqa: E402
import grid as QD      # noqa: E402
import reference as RF        # noqa: E402

CM = RF.CM
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import settings as _R  # noqa: E402
DATA = _R.path("paths.data")   # the user's data folder (tools/settings.py)
REFERENCE = os.path.join(DATA, "clothes", "tenue", "corps", "corps.pmx")   # fr-ok: folders of the owner's data
SMOOTHING_WEIGHT = 4                       # weight averaging passes between neighbours
ZONE_FOLDS = (70.0, 115.0, 25.0)       # min y, max y, max |x|: where the reference has folds to remove


def load_reference(path, log=print):
    R = RF.Reference(path)
    y0, y1, x1 = ZONE_FOLDS
    R.unfold(zone=lambda p: (p[:, 1] > y0) & (p[:, 1] < y1) & (np.abs(p[:, 0]) < x1), journal=log)
    return R


# ---------------------------------------------------------------------------------------------- weights
def table_mirror(pmx):
    names = [o["name"] for o in pmx["bones"]]
    idx = {n: i for i, n in enumerate(names)}
    out = np.arange(len(names))
    for i, n in enumerate(names):
        m = n.replace("左", "\0").replace("右", "左").replace("\0", "右")
        out[i] = idx.get(m, i)
    return out


def weight(g, R):
    """Weights of each vertex: read on the reference at the closest point (left side and middle), the
    right side mirrored; vertices taken from the reference keep their own."""
    P, sym, source, mir = g["P"], g["sym"], g["source"], g["mirror"]
    n = len(P)
    tm = table_mirror(R.pmx)
    F = np.concatenate([R.F[R.f_body], R.F[R.f_main["G"]], R.F[R.f_main["D"]]])
    left = (P[:, 0] >= -1e-6) & (source < 0)
    idx = np.flatnonzero(left)
    _, f, bary, _ = G.plus_near(P[idx], R.V, F)
    W = [None] * n
    for i, fi, b in zip(idx, f, bary):
        accessory = {}
        for v, bw in zip(F[fi], b):
            for o, w in zip(R.os_s[v], R.weight_s[v]):
                if w > 0:
                    accessory[int(o)] = accessory.get(int(o), 0.0) + bw * w
        W[i] = accessory
    for i in np.flatnonzero(source >= 0):
        accessory = {}
        for o, w in zip(R.os_s[source[i]], R.weight_s[source[i]]):
            if w > 0:
                o = int(tm[o]) if mir[i] else int(o)
                accessory[o] = accessory.get(o, 0.0) + float(w)
        W[i] = accessory
    for i in range(n):
        if W[i] is None:
            W[i] = {int(tm[o]): w for o, w in W[sym[i]].items()}
    for i in np.flatnonzero(sym == np.arange(n)):       # middle: symmetric weights
        accessory = dict(W[i])
        for o, w in W[i].items():
            accessory[int(tm[o])] = accessory.get(int(tm[o]), 0.0) + w
        W[i] = {o: w / 2 for o, w in accessory.items()}
    # smoothing: in the hollow of the armpit and of the crotch, the closest point jumps from one side to the other
    # of the reference's fold; a few averaging passes between neighbours give continuous weights. The
    # vertices taken as is (hands, feet) keep their own.
    if SMOOTHING_WEIGHT:
        nb = 1 + max(max(accessory) for accessory in W)
        M = np.zeros((n, nb))
        for i, accessory in enumerate(W):
            for o, w in accessory.items():
                M[i, o] = w
        M /= M.sum(1, keepdims=True)
        lib = np.flatnonzero(source < 0)
        neigh = g["neighbors"]
        for _ in range(SMOOTHING_WEIGHT):
            avg = np.stack([M[neigh[i]].mean(0) for i in lib])
            M[lib] = 0.5 * M[lib] + 0.5 * avg
        W = [{int(o): float(M[i, o]) for o in np.flatnonzero(M[i] > 1e-4)} for i in range(n)]
    os_, pw, typ = np.zeros((n, 4), int), np.zeros((n, 4)), np.zeros(n, int)
    for i, accessory in enumerate(W):
        top = sorted(accessory.items(), key=lambda kv: -kv[1])[:4]
        s = sum(w for _, w in top)
        for k, (o, w) in enumerate(top):
            os_[i, k], pw[i, k] = o, w / s
        typ[i] = 0 if len(top) == 1 else (1 if len(top) == 2 else 2)
    return os_, pw, typ


# ---------------------------------------------------------------------------------------------- UV
def maps_uv(g, R):
    """Texture coordinates: one map per tube of the grid (cut along a column; the
    palm is in the arm's map, the foot in the leg's, each finger and each toe has its own).
    Returns (corners (nt, 3) split-vertex indices, original vertex of each split vertex, uv)."""
    P, quads, tris, maps = g["P"], g["quads"], g["tris"], g["maps"]
    islands = {}                # name -> list of (vertex indices, uv in cm)
    NT = QD.NT
    # tubes: size of a cell in cm, average of the map
    per_map = {}
    for q, c in zip(quads, maps):
        if c is not None:
            per_map.setdefault(c[0], []).append((q, np.array(c[1], float)))
    for name, L in per_map.items():
        d_u = dv = 0.0
        nu = nv = 0
        for q, uv in L:
            for a in range(4):
                b = (a + 1) % 4
                d = np.linalg.norm(P[q[a]] - P[q[b]])
                if uv[a][1] == uv[b][1]:
                    d_u, nu = d_u + d, nu + 1
                else:
                    dv, nv = dv + d, nv + 1
        su, sv = d_u / max(nu, 1), dv / max(nv, 1)
        out = []
        for q, uv in L:
            uv = uv.copy()
            if name == "trunk":               # the seam in the middle of the back
                u0 = uv[:, 0].min()
                uv[:, 0] += ((u0 + NT // 2) % NT) - u0
            uv = uv * [su, sv]
            out.append((q[[0, 1, 2]], uv[[0, 1, 2]]))
            out.append((q[[0, 2, 3]], uv[[0, 2, 3]]))
        islands[name] = out
    assert all(c is not None for c in maps) and len(tris) == 0, "every quad has its map"
    # packing: shelves, from the tallest map to the shortest
    boxes = {}
    for name, L in islands.items():
        if not L:
            continue
        uv = np.concatenate([u for _, u in L])
        boxes[name] = (uv.min(0), uv.max(0) - uv.min(0))
    margin = 1.0
    area = sum((b[1][0] + margin) * (b[1][1] + margin) for b in boxes.values())
    side_ = np.sqrt(area) * 1.12
    while True:
        x = y = h = 0.0
        place = {}
        for name in sorted(boxes, key=lambda k: -boxes[k][1][1]):
            w_, h_ = boxes[name][1] + margin
            if x + w_ > side_:
                x, y, h = 0.0, y + h, 0.0
            place[name] = (x, y)
            x, h = x + w_, max(h, h_)
        if y + h <= side_ and max(b[1][0] + margin for b in boxes.values()) <= side_:
            break
        side_ *= 1.05
    the_key, origin, uvs, corners = {}, [], [], []
    for name, L in islands.items():
        if not L:
            continue
        offset = np.array(place[name]) + 0.5 * margin - boxes[name][0]
        for t, uv in L:
            tri = []
            for v, u in zip(t, uv):
                u = (u + offset) / side_
                k = (int(v), name, int(round(u[0] * 1e5)), int(round(u[1] * 1e5)))
                if k not in the_key:
                    the_key[k] = len(origin)
                    origin.append(int(v))
                    uvs.append(u)
                tri.append(the_key[k])
            corners.append(tri)
    return np.array(corners), np.array(origin), np.array(uvs), {k: [round(float(v), 1) for v in b[1]] for k, b in boxes.items()}


# ---------------------------------------------------------------------------------------------- PMX
def assemble(g, R, morphs, log=print):
    """The new PMX: everything that is not the skin of the body and hands is kept as is."""
    src = R.pmx
    t = R.t
    P = g["P"]
    T = np.concatenate([AJ.triangles(g["quads"]), g["tris"]])
    nor = G.normals_vertices(P, T)
    os_, pw, typ = weight(g, R)
    corners, origin, uvs, boxes = maps_uv(g, R)
    # faces kept from the reference: all materials except the skin, and the skull within the skin
    skull = np.zeros(len(t["faces"]), bool)
    skull[R.faces_raw[R.f_skull]] = True
    guard = (t["mat_face"] != R.i_skin) | skull
    util = np.unique(t["faces"][guard])
    new = -np.ones(len(t["pos"]), int)
    new[util] = np.arange(len(util))
    s0 = src["vertices"]
    vert = {k: [s0[k][i] for i in util] for k in s0}
    n_old = len(util)
    for i, (v, uv) in enumerate(zip(origin, uvs)):
        vert["pos"].append(tuple(float(x) for x in P[v] / CM))
        vert["nor"].append(tuple(float(x) for x in nor[v]))
        vert["uv"].append((float(uv[0]), float(uv[1])))
        vert["uv_add"].append([(0.0, 0.0, 0.0, 0.0)] * src["n_uv_add"])
        vert["type"].append(int(typ[v]))
        vert["bones"].append([int(x) for x in os_[v]])
        vert["weights"].append([float(x) for x in pw[v]])
        vert["sdef"].append(None)
        vert["contour"].append(1.0)
    faces, mats = [], []
    for i, m in enumerate(src["materials"]):
        fm = new[t["faces"][guard & (t["mat_face"] == i)]].tolist()
        if i == R.i_skin:
            fm += (corners + n_old).tolist()
        faces += fm
        mats.append(dict(m, n_indices=3 * len(fm)))
    out = dict(src)
    out["vertices"] = vert
    out["faces"] = faces
    out["materials"] = mats
    # shape morphs: one vertex morph per slider
    exploded = {}
    for i, v in enumerate(origin):
        exploded.setdefault(int(v), []).append(n_old + i)
    out["morphs"] = list(src["morphs"])
    for name, D in morphs.items():
        offset = []
        for v in np.flatnonzero(np.linalg.norm(D, axis=1) > 1e-4):
            for i in exploded.get(int(v), []):
                offset.append((i, tuple(float(x) for x in D[v] / CM)))
        out["morphs"].append({"name": name, "name_en": name, "panel": 4, "type": 1, "offsets": offset})
    return out, T, boxes, (os_, pw)


def measure(g, R, T, boxes, W):
    P = g["P"]
    ang, ar = G.dihedrals(P, T)
    free = g["free"][ar].all(1)
    S = R.V[np.unique(R.F[R.f_body])]
    d = G.plus_near(S, P, T)[3] * 10
    Sb = R.V_brut[np.unique(R.F[R.f_body])] if hasattr(R, "V_brut") else S
    angr, arr = G.dihedrals(R.V_brut, R.F[R.f_body])
    top = lambda V, e: V[e].mean(1)[:, 1] > 20        # feet excluded
    return {
        "vertices": int(len(P)), "triangles": int(len(T)), "quads": int(len(g["quads"])),
        "borders": sorted(len(b) for b in G.border_loops(T)),
        "sharp_edges_over_40_except_hands_feet": int(((ang > 40) & free).sum()),
        "angle_max_except_hands_feet": round(float(ang[free].max()), 1),
        "reference_sharp_edges_over_40_except_feet": int(((angr > 40) & top(R.V_brut, arr)).sum()),
        "reference_angle_max_except_feet": round(float(angr[top(R.V_brut, arr)].max()), 1),
        "reference_gap_mm": {"median": round(float(np.median(d)), 2), "p95": round(float(np.percentile(d, 95)), 2),
                               "p99": round(float(np.percentile(d, 99)), 2), "max": round(float(d.max()), 2)},
        "symetrie_mm": round(float(np.abs(P - P[g["sym"]] * [-1, 1, 1]).max() * 10), 4),
        "maps_uv_cm": boxes,
        "bones_per_vertex_max": int((W[1] > 0).sum(1).max()),
        "construction": g["info"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--output", dest="output", required=True)
    ap.add_argument("--shape", dest="shape", help="reread the shape already computed (shape.pkl in the output folder)", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    R = load_reference(a.reference)
    cache = os.path.join(a.output, "shape.pkl")
    if a.shape and os.path.exists(cache):
        g = pickle.load(open(cache, "rb"))
    else:
        g = GE.build(R)
        pickle.dump(g, open(cache, "wb"))
    morphs = CU.morphs(g, R)
    pmx, T, boxes, W = assemble(g, R, morphs)
    pmx_write.write(pmx, os.path.join(a.output, "corps.pmx"))
    tex = os.path.join(os.path.dirname(a.reference), "tex")
    if os.path.isdir(tex) and os.path.abspath(tex) != os.path.abspath(os.path.join(a.output, "tex")):
        shutil.copytree(tex, os.path.join(a.output, "tex"), dirs_exist_ok=True)
    m = measure(g, R, T, boxes, W)
    m["curseurs"] = {k: round(float(np.linalg.norm(D, axis=1).max() * 10), 1) for k, D in morphs.items()}
    json.dump(m, open(os.path.join(a.output, "measures.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

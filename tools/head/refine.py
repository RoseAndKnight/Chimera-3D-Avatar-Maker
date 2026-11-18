"""Local refinement of a PMX skin (red-green subdivision, without changing the surface).

Marked faces are split into four through the midpoints of their edges; a neighbouring face
with two or three split edges is split into four as well (closure), a face with only one
is split in two. New vertices sit at the edge midpoint: shape, UVs, bone weights, normals
and the offsets of all vertex and UV morphs are the average of the two ends. Existing
vertices keep their index; new ones are appended at the end. Classic method (Bank, Sherman
and Weiser 1983).
"""
import copy

import numpy as np

import geom as G


def _poids(s, v):
    t = s["type"][v]
    o = s["os"][v]
    w = s["poids"][v]
    if t == 0:
        return {int(o[0]): 1.0}
    if t in (1, 3):
        d = {int(o[0]): float(w[0])}
        if o[1] >= 0:
            d[int(o[1])] = d.get(int(o[1]), 0.0) + float(w[1])
        return d
    d = {}
    for b, x in zip(o, w):
        if b >= 0 and x > 0:
            d[int(b)] = d.get(int(b), 0.0) + float(x)
    return d


def _milieu_poids(s, a, b):
    d = {}
    for v in (a, b):
        for k, x in _poids(s, v).items():
            d[k] = d.get(k, 0.0) + 0.5 * x
    it = sorted(d.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(x for _, x in it) or 1.0
    it = [(k, x / tot) for k, x in it]
    if len(it) == 1:
        return 0, (it[0][0], -1, -1, -1), (1.0, 0.0, 0.0, 0.0)
    if len(it) == 2:
        return 1, (it[0][0], it[1][0], -1, -1), (it[0][1], it[1][1], 0.0, 0.0)
    while len(it) < 4:
        it.append((-1, 0.0))
    return 2, tuple(k for k, _ in it), tuple(x for _, x in it)


def raffiner(pmx, mat, faces_marquees):
    """pmx: loaded model; mat: index of the material to refine; faces_marquees: indices of
    raw (global) faces of that material. Returns (new pmx, number of split faces)."""
    P = copy.deepcopy(pmx)
    s = P["sommets"]
    faces = [tuple(f) for f in P["faces"]]
    debut = sum(m["n_indices"] // 3 for m in P["materiaux"][:mat])
    fin = debut + P["materiaux"][mat]["n_indices"] // 3
    pos = np.asarray(s["pos"], dtype=float)
    Fm = np.asarray(faces[debut:fin], dtype=np.int64)
    vb = np.unique(Fm)
    _, inv = G.souder(pos[vb])
    b2s = np.full(len(pos), -1)
    b2s[vb] = inv

    def arete(a, b):
        x, y = int(b2s[a]), int(b2s[b])
        return (x, y) if x < y else (y, x)

    marque = np.zeros(len(Fm), dtype=bool)
    for f in faces_marquees:
        if debut <= f < fin:
            marque[f - debut] = True
    coupees = set()
    for f in np.where(marque)[0]:
        a, b, c = Fm[f]
        coupees.update([arete(a, b), arete(b, c), arete(c, a)])
    while True:
        ajout = False
        for f in np.where(~marque)[0]:
            a, b, c = Fm[f]
            k = sum(e in coupees for e in (arete(a, b), arete(b, c), arete(c, a)))
            if k >= 2:
                marque[f] = True
                coupees.update([arete(a, b), arete(b, c), arete(c, a)])
                ajout = True
        if not ajout:
            break
    nuv = P["n_uv_add"]
    milieux = {}

    def milieu(a, b):
        cle = (min(a, b), max(a, b))
        if cle in milieux:
            return milieux[cle]
        t, o, w = _milieu_poids(s, a, b)
        n = np.asarray(s["nor"][a]) + np.asarray(s["nor"][b])
        n = n / max(np.linalg.norm(n), 1e-12)
        s["pos"].append(tuple(0.5 * (np.asarray(s["pos"][a]) + np.asarray(s["pos"][b]))))
        s["nor"].append(tuple(n))
        s["uv"].append(tuple(0.5 * (np.asarray(s["uv"][a]) + np.asarray(s["uv"][b]))))
        s["uv_add"].append([tuple(0.5 * (np.asarray(s["uv_add"][a][k]) + np.asarray(s["uv_add"][b][k])))
                            for k in range(nuv)])
        s["type"].append(t)
        s["os"].append(o)
        s["poids"].append(w)
        s["sdef"].append(None)
        s["contour"].append(0.5 * (s["contour"][a] + s["contour"][b]))
        milieux[cle] = len(s["pos"]) - 1
        return milieux[cle]

    nouv = []
    n4 = n2 = 0
    for f in range(len(Fm)):
        a, b, c = (int(x) for x in Fm[f])
        if marque[f]:
            ab, bc, ca = milieu(a, b), milieu(b, c), milieu(c, a)
            nouv += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
            n4 += 1
            continue
        cote = [arete(a, b) in coupees, arete(b, c) in coupees, arete(c, a) in coupees]
        if not any(cote):
            nouv.append((a, b, c))
            continue
        k = cote.index(True)
        p, q, r = [(a, b, c), (b, c, a), (c, a, b)][k]
        m = milieu(p, q)
        nouv += [(p, m, r), (m, q, r)]
        n2 += 1
    P["faces"] = faces[:debut] + nouv + faces[fin:]
    P["materiaux"][mat]["n_indices"] = 3 * len(nouv)
    # vertex and UV morphs: average of the two ends
    for m in P["morphes"]:
        if not (m["type"] == 1 or 3 <= m["type"] <= 7) or not m["decalages"]:
            continue
        d = {}
        for v, x in m["decalages"]:
            d[int(v)] = np.asarray(x, dtype=float)
        ajout = []
        for (a, b), k in milieux.items():
            if a in d or b in d:
                x = 0.5 * (d.get(a, 0.0) + d.get(b, 0.0))
                x = np.broadcast_to(x, d[a].shape if a in d else d[b].shape)
                if np.abs(x).max() > 0:
                    ajout.append((k, tuple(float(y) for y in x)))
        m["decalages"] = list(m["decalages"]) + ajout
    return P, {"faces_en_4": n4, "faces_en_2": n2, "sommets_ajoutes": len(milieux)}

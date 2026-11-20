"""Perfect Sync expressions for VRoid faces: the 52 ARKit shapes (eyeBlinkLeft, jawOpen, mouthSmileLeft, cheekPuff,
tongueOut...) and the extra shapes of Hinzka's free models, given to any face made with VRoid Studio.

    python3 -I perfect_sync.py <file.vrm>... [--donor auto|female|male]
        says how the face matches the donor and what would be added, without writing anything

The donors are the face meshes of hinzka/52blendshapes-for-VRoid-face v1.1.3 (VRoid Studio V1.1.0 sample models,
https://github.com/hinzka/52blendshapes-for-VRoid-face; free for personal and commercial use, redistribution and
sale of modified data allowed, no credit required, no warranty), read from the `perfect_sync` source of the "vroid"
game (VroidV110 Female and Male files). Without them, nothing is added and the import goes on as before.

Why not by vertex number (as NEB does): VRoid removes the transparent parts of the face when it exports, so a face
has 3 989 to 4 709 vertices and the numbering shifts. But every vertex keeps its material and its UV: a vertex of
the face is matched to the donor vertex of the same material (face skin, eye white, iris...) and the same UV (within
UV_TOL). The two irises (and a few seams) share their UV: those candidates are told apart by position, after the
donor face is brought onto the face by a scale and a translation fitted on the unique matches. Female or male donor
(`auto`): the one whose face, so fitted, lies closest to the face.

Each donor shape the face lacks is added under its own name, its offsets copied as they are, in the frame and units
of the PMX (the importer's conversion). Unmatched vertices stay still; they are counted.

Calibration: a copied shape moves the donor's eyelid on a face whose eye may be larger or smaller. For each closing
shape (CLOSING), the gap between the upper and lower edge of the face skin's opening (eye, mouth) is measured once
the shape is applied, and the shape is scaled so that the gap is the donor's own (closed just so: Hinzka's lid
stops about 0.005 MMD units under the lower lid), least squares over the edge, the factor kept in SCALE_RANGE.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "atlas"), os.path.join(HERE, "..", "head")]
import importer as I          # noqa: E402

DONOR_FILES = {"female": "VRoid_V110_Female_v1.1.3.vrm", "male": "VRoid_V110_Male_v1.1.3.vrm"}
SKIN = "颜"                    # common name of the face skin (game.json materials)
UV_TOL = 1e-4                  # UV distance (largest coordinate difference) under which two vertices are the same vertex
MIN_MATCHED = 0.95             # below this share of matched face vertices, the face is not the donor's: nothing added
THRESHOLD = 1e-5               # MMD units: an offset below this does not move the vertex
SCALE_RANGE = (0.5, 2.0)       # calibration factor kept within this range (outside: the measure is not trusted)
# closing shape -> (side of the face, base shape applied with it, which lid moves: "upper" or "lower")
CLOSING = {"eyeBlinkLeft": ("left", None), "eyeBlinkRight": ("right", None), "mouthClose": ("mouth", "jawOpen")}

# defects the check reports by construction (the donor has them too): said in the report, not counted as faults
EXPECTED = {"tongueOut": "the tongue is meant to come out in front of the lips",
            "mouthClose": "meant to be used with jawOpen (alone it pushes the closed lips into the teeth)"}

_donors = {}


def donor_folder():
    try:
        import games as games
        return games.load("vroid").path("perfect_sync")
    except Exception:       # no settings or no source: no donor
        return None


# =================================================================== reading a face

def read_face(path, D=None):
    """The face mesh of a VRM (the mesh carrying the face skin), in the importer's frame (MMD units): dict with pos
    (n, 3), uv (n, 2), key (n,) common material name, faces {key: (m, 3)}, targets {short name: (n, 3)}."""
    D = D or I.description()
    M = I.Model(path)
    j, b = M.j, M.b
    for node, mesh, skin in M.meshes():
        keys = [I.translate(j["materials"][p["material"]].get("name", ""), D["materials"])[0] for p in mesh["primitives"]]
        if SKIN not in keys:
            continue
        B = M.bond(node, skin)
        blocks, pos, uv, n, faces, targets = {}, [], [], 0, {}, {}
        for pr, k in zip(mesh["primitives"], keys):
            at = pr["attributes"]
            the_key = at["POSITION"]
            if the_key not in blocks:
                p = I.accessory(j, b, the_key).astype(float)
                L = None
                if B is not None:
                    w = I.accessory(j, b, at["WEIGHTS_0"]).astype(float)
                    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
                    jt = I.accessory(j, b, at["JOINTS_0"]).astype(int)
                    L = np.einsum("nk,nkij->nij", w, B[np.minimum(jt, len(B) - 1)])
                    p = np.einsum("nij,nj->ni", L[:, :3, :3], p) + L[:, :3, 3]
                blocks[the_key] = n
                pos.append(p * M.S)
                uv.append(I.accessory(j, b, at["TEXCOORD_0"]).astype(float) if "TEXCOORD_0" in at else np.zeros((len(p), 2)))
                names = pr.get("extras", {}).get("targetNames") or mesh.get("extras", {}).get("targetNames") or []
                for t, target in enumerate(pr.get("targets", [])):
                    if "POSITION" not in target:
                        continue
                    d = I.accessory(j, b, target["POSITION"]).astype(float)
                    if L is not None:
                        d = np.einsum("nij,nj->ni", L[:, :3, :3], d)
                    targets.setdefault(I.name_court(names[t] if t < len(names) else "shape%d" % t), []).append((n, d * M.S))
                n += len(p)
            f = I.accessory(j, b, pr["indices"]).reshape(-1, 3).astype(np.int64) + blocks[the_key]
            faces.setdefault(k or "?", []).append(f)
        key = np.full(n, "", object)
        faces = {k: np.concatenate(f) for k, f in faces.items()}
        for k, f in faces.items():
            key[f.ravel()] = k
        dense = {}
        for name, parts in targets.items():
            d = np.zeros((n, 3))
            for o, x in parts:
                d[o:o + len(x)] = x
            dense[name] = d
        return {"pos": np.concatenate(pos), "uv": np.concatenate(uv), "key": key, "faces": faces, "targets": dense,
                "path": path}
    raise I.Refusals("%s has no VRoid face mesh" % path)


def donor(sort, folder=None):
    """The donor face ("female" or "male"), read once; None if its file is missing."""
    folder = folder or donor_folder()
    if not folder:
        return None
    path = os.path.join(folder, DONOR_FILES[sort])
    if path not in _donors:
        _donors[path] = read_face(path) if os.path.isfile(path) else None
        if _donors[path] is not None:
            _donors[path]["sort"] = sort
            _donors[path]["edges"] = {}
    return _donors[path]


# =================================================================== correspondence

def match(Dn, pos, uv, key):
    """Donor vertex of each face vertex (-1: none), and the report. A vertex is matched to the donor vertices of the
    same material and UV; several candidates (the two irises, seams) are told apart by position once the donor is fitted
    onto the face (scale and translation, least squares on the unique matches)."""
    n = len(pos)
    candidates = [None] * n
    for k in set(key.tolist()):
        A = np.where(Dn["key"] == k)[0]
        rows = np.where(key == k)[0]
        if not len(A):
            for r in rows:
                candidates[r] = A
            continue
        for s in range(0, len(rows), 512):
            r = rows[s:s + 512]
            d = np.abs(uv[r][:, None, :] - Dn["uv"][A][None]).max(2)
            for i, line in zip(r, d < UV_TOL):
                candidates[i] = A[line]
    nc = np.array([len(c) for c in candidates])
    m = np.full(n, -1, np.int64)
    one = nc == 1
    m[one] = [c[0] for c, o in zip(candidates, one) if o]
    report = {"vertices": n, "unique": int(one.sum()), "ambiguous": int((nc > 1).sum()), "unmatched": int((nc == 0).sum())}
    some = nc > 0
    if not some.any():
        report.update(scale=None, rms=None)
        return m, report
    if one.sum() >= 10:
        A, B = Dn["pos"][m[one]], pos[one]
        am, bm = A.mean(0), B.mean(0)
        s = float(((A - am) * (B - bm)).sum() / max(((A - am) ** 2).sum(), 1e-12))
    else:       # too few unique matches: the two clouds' centres and spreads
        A, B = Dn["pos"][np.unique(np.concatenate([c for c in candidates if len(c)]))], pos[some]
        am, bm = A.mean(0), B.mean(0)
        s = float(np.sqrt(((B - bm) ** 2).sum(1).mean() / max(((A - am) ** 2).sum(1).mean(), 1e-12)))
    t = bm - s * am
    ties = 0
    for i in np.where(nc > 1)[0]:
        c = candidates[i]
        d = np.linalg.norm(s * Dn["pos"][c] + t - pos[i], axis=1)
        o = np.argsort(d)
        m[i] = c[o[0]]
        # two candidates as close (seam duplicates): harmless only if they move alike
        if d[o[1]] - d[o[0]] < 1e-3 and any(np.abs(x[c[o[0]]] - x[c[o[1]]]).max() > THRESHOLD for x in Dn["targets"].values()):
            ties += 1
    r = np.linalg.norm(s * Dn["pos"][m[m >= 0]] + t - pos[m >= 0], axis=1)
    report.update(scale=round(s, 4), translation=[round(float(x), 4) for x in t], rms=round(float(np.sqrt((r ** 2).mean())), 4),
                  p95=round(float(np.percentile(r, 95)), 4), ties=ties)
    return m, report


# =================================================================== calibration

def opening(pos, faces, d, side, open_d=None):
    """Edge of the face skin's opening on a side ("left": +x eye, "right": -x eye, "mouth": centre): the boundary loop
    (welded skin) that the shape d moves most, split into its two lids by the line joining its corners, the opening
    seen open (pos + open_d: the mouth is a slit at rest). Returns (moving lid, other lid, sign) with sign +1 when the
    moving lid is the upper one, or None."""
    from geom import weld
    rep, inv = weld(pos, 1e-4)
    fw = inv[faces]
    E = np.sort(np.concatenate([fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    be = u[c == 1]
    if not len(be):
        return None
    parent = np.arange(len(rep))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in be:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
    loops = {}
    for v in np.unique(be.ravel()):
        loops.setdefault(root(v), []).append(v)
    best, score = None, 0.0
    for vs in loops.values():
        vs = rep[np.array(vs)]
        x = pos[vs, 0].mean()
        if (side == "left" and x <= 0) or (side == "right" and x >= 0) or (side == "mouth" and abs(x) > 0.1 * np.ptp(pos[:, 0])):
            continue
        mv = np.linalg.norm(d[vs], axis=1).max()
        if mv > score and len(vs) >= 6:
            best, score = vs, mv
    if best is None:
        return None
    Q = pos[best] + (open_d[best] if open_d is not None else 0.0)
    a, b = int(np.argmin(Q[:, 0])), int(np.argmax(Q[:, 0]))
    line = Q[a, 1] + (Q[:, 0] - Q[a, 0]) * (Q[b, 1] - Q[a, 1]) / max(Q[b, 0] - Q[a, 0], 1e-9)
    corner = np.zeros(len(best), bool)
    corner[[a, b]] = True
    up, low = (Q[:, 1] > line) & ~corner, (Q[:, 1] <= line) & ~corner
    if up.sum() < 2 or low.sum() < 2:
        return None
    mv = np.linalg.norm(d[best], axis=1)
    if mv[up].mean() >= mv[low].mean():
        return best[up], best[low | corner], 1.0
    return best[low], best[up | corner], -1.0


def gaps(pos, moving, still, sign, d):
    """Signed vertical gap of each moving edge vertex to the other edge, with the offsets d applied: positive while
    the opening stays open, negative when the lids cross."""
    P = pos + d
    o = np.argsort(P[still, 0])
    xs, ys = P[still][o, 0], P[still][o, 1]
    y = np.interp(P[moving, 0], xs, ys)
    return sign * (P[moving, 1] - y)


def donor_gaps(Dn, name, side, base):
    """The donor's own gap with its shape fully applied, per donor vertex of the moving lid."""
    the_key = (name, side)
    if the_key not in Dn["edges"]:
        d = Dn["targets"][name] + (Dn["targets"][base] if base else 0.0)
        select = Dn["targets"][base or name]
        e = opening(Dn["pos"], Dn["faces"][SKIN], select, side, Dn["targets"][base] if base else None)
        Dn["edges"][the_key] = None if e is None else dict(zip(e[0].tolist(), gaps(Dn["pos"], *e, d=d).tolist()))
    return Dn["edges"][the_key]


def calibrate(pos, faces_skin, m, Dn, name, d_shape, d_base, side, base):
    """Factor bringing the face's gap, shape applied, to the donor's (least squares over the moving lid, by a fine
    search within SCALE_RANGE). Returns (factor, report) or (None, reason)."""
    select = d_base if base else d_shape        # the lid that moves: the one the shape closes, or the base opens
    e = opening(pos, faces_skin, select, side, d_base)
    if e is None:
        return None, "opening not found"
    moving, still, sign = e
    target_of = donor_gaps(Dn, name, side, base)
    if not target_of:
        return None, "opening not found on the donor"
    default = float(np.median(list(target_of.values())))
    target = np.array([target_of.get(int(m[v]), default) for v in moving])
    base_d = d_base if d_base is not None else 0.0
    S = np.linspace(SCALE_RANGE[0], SCALE_RANGE[1], 301)
    cost = [((gaps(pos, moving, still, sign, base_d + s * d_shape) - target) ** 2).sum() for s in S]
    k = int(np.argmin(cost))
    g0 = gaps(pos, moving, still, sign, base_d + d_shape)
    g1 = gaps(pos, moving, still, sign, base_d + S[k] * d_shape)
    out = {"factor": round(float(S[k]), 3), "gap_before": [round(float(g0.min()), 4), round(float(g0.max()), 4)],
           "gap_after": [round(float(g1.min()), 4), round(float(g1.max()), 4)],
           "gap_donor": [round(float(target.min()), 4), round(float(target.max()), 4)], "edge_vertices": int(len(moving))}
    if k in (0, len(S) - 1):
        return None, "factor outside %s (%s)" % (SCALE_RANGE, out)
    return float(S[k]), out


# =================================================================== the whole step

def add(pos, uv, rows, key, faces_skin, present, sort="auto", folder=None, log=print):
    """Perfect Sync shapes for the face vertices `rows` of an imported model (pos: all vertices, MMD units; key:
    common material name of each row; faces_skin: face skin faces, indices into pos; present: short names of
    the shapes the model already has). Returns ({name: (indices into pos, offsets)}, report)."""
    sorts = ("female", "male") if sort == "auto" else (sort,)
    found = [(s, donor(s, folder)) for s in sorts]
    found = [(s, x) for s, x in found if x is not None]
    if not found:
        return {}, {"skipped": "no donor file (source perfect_sync of the vroid game)"}
    P, UV = pos[rows], uv[rows]
    trials = []
    for s, Dn in found:
        m, rep = match(Dn, P, UV, key)
        trials.append((rep["rms"] if rep["rms"] is not None else np.inf, s, Dn, m, rep))
    trials.sort(key=lambda x: x[0])
    _, s, Dn, m, rep = trials[0]
    rep = dict(rep, donor=s, donor_file=os.path.basename(Dn["path"]))
    if len(trials) > 1:
        rep["rms_other_donor"] = trials[1][4]["rms"]
    if rep["vertices"] == 0 or (rep["unique"] + rep["ambiguous"]) < MIN_MATCHED * rep["vertices"]:
        rep["skipped"] = "only %d of %d face vertices match the donor" % (rep["unique"] + rep["ambiguous"], rep["vertices"])
        return {}, rep
    ok = m >= 0
    # local faces of the skin (rows): calibration works on the face mesh alone
    local = -np.ones(len(pos), np.int64)
    local[rows] = np.arange(len(rows))
    fs = local[faces_skin]
    fs = fs[(fs >= 0).all(1)]
    dense = {}
    for name, d in Dn["targets"].items():
        if name in present:
            continue
        x = np.zeros((len(rows), 3))
        x[ok] = d[m[ok]]
        dense[name] = x
    calib = {}
    for name, (side, base) in CLOSING.items():
        if name not in dense:
            continue
        d_base = dense.get(base) if base else None
        if base and d_base is None:
            continue
        f, r = calibrate(P, fs, m, Dn, name, dense[name], d_base, side, base)
        calib[name] = r
        if f is not None:
            dense[name] = dense[name] * f
    out = {}
    for name, x in dense.items():
        moved = np.abs(x).max(1) > THRESHOLD
        if moved.any():
            out[name] = (rows[moved], x[moved])
    rep.update(added=len(out), names=sorted(out), calibration=calib)
    log("perfect sync: donor %s, %d/%d face vertices matched (%d by position), fit rms %s; %d shapes added; %s" % (
        s, rep["unique"] + rep["ambiguous"], rep["vertices"], rep["ambiguous"], rep["rms"], len(out),
        ", ".join("%s x%s" % (n, c["factor"]) if isinstance(c, dict) else "%s: %s" % (n, c) for n, c in calib.items())))
    return out, rep


def checked(pmx_path, names):
    """morphs.py check on the added shapes: (lines with a defect, number checked)."""
    import morphs as MO
    Mo = MO.Model(pmx_path)
    names = [n for n in names if n in Mo.index]
    lines = MO.check(Mo, names)
    bad = [{"morph": L["morph"], "frozen": L["frozen"], "pierces": L["percent"]} for L in lines
           if L["frozen"] or L["percent"] or max(L["materials"].values() or [0.0]) < MO.THRESHOLD]
    for b in bad:
        L = next(x for x in lines if x["morph"] == b["morph"])
        b["empty"] = max(L["materials"].values() or [0.0]) < MO.THRESHOLD
        if b["morph"] in EXPECTED and not b["frozen"] and not b["empty"]:
            b["expected"] = EXPECTED[b["morph"]]
    return bad, len(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    sort = "auto"
    if "--donor" in argv:
        k = argv.index("--donor")
        sort = argv[k + 1]
        del argv[k:k + 2]
    for path in argv:
        F = read_face(path)
        rows = np.arange(len(F["pos"]))
        _, rep = add(F["pos"], F["uv"], rows, F["key"], F["faces"][SKIN], set(F["targets"]), sort,
                     log=lambda x: print("%s: %s" % (os.path.basename(path), x)))
        print({k: v for k, v in rep.items() if k != "names"})
    return 0


if __name__ == "__main__":
    sys.exit(main())

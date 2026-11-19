"""Library of face parts: eyes, eyebrows, nose, mouth and face base (forehead, cheeks, jaw,
chin: the background on which the other parts sit) of each character, stored on the pivot
head; a face is then composed part by part.

    python3 -I tools/head/library.py extract --output <biblio> [--jobs 4] [--only Jane,Ellen]
    python3 -I tools/head/library.py extract --outerplane --output <biblio> [--only Idith,...]
    python3 -I tools/head/library.py composed --library <biblio> --output <folder> \
        --base Jane --eyes Ellen --eyebrows Miyabi --nose Miyabi --mouth Anby [--onto Belle \
        [--hair Miyabi [--ears Nekomata|aucune]]]

Extract, for each character:
1. the pivot head's skin is registered onto theirs (same registration as pivot.py): same
   mesh for everyone, point to point; each part is a soft mask on this mesh;
2. their skin morphs are sampled at the registered points (blink, vowels, smile...);
3. their skin texture is copied into the pivot head's UVs (for each pivot texel, the
   registered point, then the character's texture at that point); the images are made in
   Blender (library_images.py);
4. their parts (eyeballs, lashes, eyebrows, teeth, tongue...) are sorted by face part, by
   connected piece and by position: eyes, eyebrows, mouth, otherwise base.
Outputs: <biblio>/<Character>/part.npz, texture.png, fiche.json; <biblio>/index.json
(for each part, the characters that provide it, with the registration reliability; the
parts not provided and why). The index is merged with the one already in <biblio>:
characters that have no folder there keep their entry.

--outerplane: Outerplane heads (outerplane.py) are first converted to a donor PMX in
<biblio>/<Name>/<Name>/head.pmx (textures next to it), then extracted like the ZZZ ones; the
paths in their record are relative to <biblio>/<Name> (movable folder). A head rejected by
the conversion or by head.Head is recorded as not provided, with the reason.

Compose: the shape is the base's, and each part takes its donor's shape in it, moved to the
place it has in the base (without changing size: the donor is already brought to the pivot's
scale); soft masks between parts. Morphs, texture (brought to the base's skin tone) and
parts likewise come from each part's donor. The result is the pivot head (Yuzuha) wearing
the composed face; --onto <character> then places it in one pass of retarget_head visage.
"""
import argparse
import copy
import heapq
import json
import os
import subprocess
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

import geom as G  # noqa: E402
import morphs as MO  # noqa: E402
import pivot as PV  # noqa: E402
import pmx_write as pmx_write  # noqa: E402
import registration as RC  # noqa: E402
import retarget_head as RT  # noqa: E402
import head as TT  # noqa: E402

PIVOT_JSON = os.path.join(R.path("paths.data"), "pivot-head", "pivot-head.json")
REGIONS = ("eyes", "eyebrows", "nose", "mouth")
BASE = "base"
# masks (in interocular distances): full zone up to the margin, then a transition over the band
MARGIN_EYE, BAND_EYE = 0.10, 0.15
MARGIN_EYEBROW, BAND_EYEBROW = 0.06, 0.12
MARGIN_MOUTH, BAND_MOUTH = 0.08, 0.15
NOSE_RADIUS_TOP, NOSE_RADIUS_BOTTOM, BAND_NOSE = 0.10, 0.20, 0.12
DEPTH, BAND_DEPTH = 0.30, 0.20  # beyond this, towards the back, it is no longer the part
# parts: nearest face part (front view) under these distances, otherwise base
PART_EYE, PART_EYEBROW, PART_MOUTH = 0.6, 0.6, 0.6
# unambiguous materials; the others (lashes and eyebrows mixed, 颜2) are sorted by position
PER_MATERIAL = {"eyes": ("目", "白目", "目影", "目光", "目光2", "瞳光", "目色", "二重", "二重影"),
                "eyebrows": ("眉", "眉毛"), "mouth": ("齿", "口舌", "口线")}
SIDE_RADIUS = 0.25  # interocular distances: around a mouth or an eye, upper or lower lip (eyelid)
TEX = 1024  # sampling grid for the donor's UVs (texture rendered at the pivot's size)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_pivot():
    with open(PIVOT_JSON, encoding="utf-8") as fh:
        return TT.Head(json.load(fh)["pmx"])


# =================================================================== part masks

def _slice(sd, margin, band):
    return 1 - G.smooth((sd - margin) / band)


def _depth(V, z0, iod):
    # the face looks towards -Z: what lies far behind the part is not part of it
    return 1 - G.smooth((V[:, 2] - z0 - DEPTH * iod) / (BAND_DEPTH * iod))


def _dist_polyline_xy(Q, P):
    d = np.full(len(Q), np.inf)
    for a, b in zip(P[:-1], P[1:]):
        ab = b - a
        t = np.clip(((Q - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(Q - (a + t[:, None] * ab), axis=1))
    return d


def masks(A):
    """Part masks on the pivot's welded skin: name -> (n,) in [0, 1]. Eyes and eyebrows have
    one mask per side (the name of the part followed by the side). The sum is at most 1; the base takes the rest."""
    V, iod = A.V, A.iod
    xy = V[:, :2]
    raw = {}
    for c in "gd":
        b = A.loop_eye[c]
        sd = G.distance_signed_polygon(xy, G.hull_convex_2d(V[b, :2]))
        raw["eyes" + c] = _slice(sd, MARGIN_EYE * iod, BAND_EYE * iod) * _depth(V, V[b, 2].mean(), iod)
        P = np.array([A.points_eyebrows["eyebrow_%s_%s" % (c, k)] for k in ("int", "top", "ext")])
        d = _dist_polyline_xy(xy, P[:, :2])
        raw["eyebrows" + c] = _slice(d, MARGIN_EYEBROW * iod, BAND_EYEBROW * iod) * _depth(V, P[:, 2].mean(), iod)
    b = A.loop_mouth
    sd = G.distance_signed_polygon(xy, G.hull_convex_2d(V[b, :2]))
    raw["mouth"] = _slice(sd, MARGIN_MOUTH * iod, BAND_MOUTH * iod) * _depth(V, V[b, 2].mean(), iod)
    # nose: capsule from the bridge (between the eyes) to below the tip, wider at the bottom
    yo = 0.5 * (A.center_eye["g"][1] + A.center_eye["d"][1])
    pn = V[A.landmarks["nose"]]
    h, bottom = np.array([0.0, yo - 0.15 * iod]), np.array([0.0, pn[1] - 0.05 * iod])
    ab = bottom - h
    t = np.clip(((xy - h) @ ab) / (ab @ ab), 0, 1)
    d = np.linalg.norm(xy - (h + t[:, None] * ab), axis=1)
    r = (NOSE_RADIUS_TOP + t * (NOSE_RADIUS_BOTTOM - NOSE_RADIUS_TOP)) * iod
    raw["nose"] = (1 - G.smooth((d - r) / (BAND_NOSE * iod))) * _depth(V, pn[2], iod)
    S = sum(raw.values())
    k = np.maximum(S, 1.0)
    out = {n: m / k for n, m in raw.items()}
    out[BASE] = 1 - np.minimum(S, 1.0)
    return out


def mask_region(M, region):
    return sum(m for n, m in M.items() if n == region or n.startswith(region + "_"))


# =================================================================== extraction

def _geodesic(T, sources):
    """Distance along the welded skin from source vertices (Dijkstra)."""
    adj = G.neighbors(T.n, T.F)
    d = np.full(T.n, np.inf)
    pile = [(0.0, int(v)) for v in sources]
    for _, v in pile:
        d[v] = 0.0
    heapq.heapify(pile)
    while pile:
        dv, v = heapq.heappop(pile)
        if dv > d[v]:
            continue
        for w in adj[v]:
            nd = dv + float(np.linalg.norm(T.V[v] - T.V[w]))
            if nd < d[w]:
                d[w] = nd
                heapq.heappush(pile, (nd, w))
    return d


def sides(T, radius):
    """Side of each welded vertex around the openings (mouth, eyes): +1 upper lip or eyelid,
    -1 lower, 0 beyond the radius (along the skin) or at the corners. A closed mouth is an
    almost zero-width slit: without this side, a point of the upper lip projects onto the
    donor's lower lip and takes its motion (the skin folds over the opening)."""
    c = np.zeros(T.n, dtype=np.int8)
    for b in [T.loop_mouth, T.loop_eye["g"], T.loop_eye["d"]]:
        b = list(b)
        x = T.V[b, 0]
        i, j = int(np.argmin(x)), int(np.argmax(x))
        if i > j:
            i, j = j, i
        arcs = [b[i + 1:j], b[j + 1:] + b[:i]]
        if min(len(a) for a in arcs) == 0:
            continue
        d0, d1 = _geodesic(T, arcs[0]), _geodesic(T, arcs[1])
        near = np.minimum(d0, d1) < radius
        r0, r1 = near & (d0 < d1), near & (d1 < d0)
        # closed mouth: both edges of the slit are at the same height; the top is the side
        # whose skin is highest
        sg = 1 if T.V[r0, 1].mean() > T.V[r1, 1].mean() else -1
        c[r0], c[r1] = sg, -sg
    return c


def _components(D, faces):
    """Connected pieces (by vertices welded by position, without crossing a material change)
    among raw faces."""
    pos = D.pos
    vb = np.unique(D.faces[faces])
    _, inv = G.weld(pos[vb])
    s = np.full(len(pos), -1)
    s[vb] = inv
    Fw = s[D.faces[faces]]
    per = np.arange(inv.max() + 1)

    def root(i):
        while per[i] != i:
            per[i] = per[per[i]]
            i = per[i]
        return i
    for a, b, c in Fw:
        ra, rb, rc = root(a), root(b), root(c)
        per[rb] = ra
        per[root(rc)] = ra
    r = np.array([root(int(f[0])) for f in Fw])
    return r * (int(D.mat_face.max()) + 1) + D.mat_face[faces]


def arrange_parts(D):
    """Raw faces of D's parts -> face part (eyes, eyebrows, mouth, base) per connected piece,
    from the position (front view, in the host frame) of its centre: distance to each part's
    zone, the one used by the masks (hull of the eye or mouth opening, inner-top-outer eyebrow
    line), in interocular distances.

    The distance to the eye centre (former rule) sorted into the eyebrows the pieces placed at
    the top of the eye, far from its centre: eyelid folds of Yixuan, Alice, Aria, Zhao, Caesar,
    NangongYu, lash tips at the outer corner of Miyabi, Grace, Manato (0.04 to 0.11 from the
    eye opening, 0.08 to 0.30 from the eyebrow line; real eyebrows are at most 0.05 from their
    line, at least 0.08 from the eye). Composed with someone else's eyes, they stayed with the
    eyebrows: Miyabi's lash tip, carried by her eyes' blink, fell below Maelys's closed eye
    (a spot under the eye)."""
    faces = np.where(np.isin(D.mat_face, D.i_parts))[0]
    out = {p: [] for p in ("eyes", "eyebrows", "mouth", BASE)}
    if not len(faces):
        return {p: np.zeros(0, dtype=np.int64) for p in out}
    lab = _components(D, faces)
    iod = D.iod * D.scale
    eyes = [G.hull_convex_2d(D._host(D.V[D.loop_eye[c]])[:, :2]) for c in "gd"]
    eyebrows = [D._host(np.array([D.points_eyebrows["eyebrow_%s_%s" % (c, k)] for k in ("int", "top", "ext")]))[:, :2]
                for c in "gd"]
    mouth = G.hull_convex_2d(D._host(D.V[D.loop_mouth])[:, :2])
    mat_part = {n: p for p, ns in PER_MATERIAL.items() for n in ns}
    for l in np.unique(lab):
        fs = faces[lab == l]
        p = mat_part.get(D.names_mat[int(D.mat_face[fs[0]])])
        if p is not None:
            out[p].append(fs)
            continue
        c = D._host(D.pos[np.unique(D.faces[fs])].mean(0))[None, :2]
        de = max(min(G.distance_signed_polygon(c, H)[0] for H in eyes), 0.0) / iod
        ds = min(_dist_polyline_xy(c, S)[0] for S in eyebrows) / iod
        dm = max(G.distance_signed_polygon(c, mouth)[0], 0.0) / iod
        choice = [(de / PART_EYE, "eyes"), (ds / PART_EYEBROW, "eyebrows"), (dm / PART_MOUTH, "mouth")]
        r, p = min(choice)
        out[p if r < 1 else BASE].append(fs)
    return {p: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for p, v in out.items()}


def raster_uv(uv_tri, n):
    """Texels (row 0 at the bottom, like Blender pixels; PMX V points down) covered by the UV
    triangles: triangle index (-1 outside) and barycentrics."""
    tri = np.full((n, n), -1, dtype=np.int64)
    bary = np.zeros((n, n, 3), dtype=np.float32)
    # in grid coordinates: x = u n, y = (1 - v) n
    T = np.stack([uv_tri[..., 0] * n, (1 - uv_tri[..., 1]) * n], -1)
    for i, t in enumerate(T):
        lo = np.maximum(np.floor(t.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(t.max(0)).astype(int), n)
        if (hi <= lo).any():
            continue
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
        p = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
        a, b, c = t
        M = np.array([b - a, c - a]).T
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        l = np.linalg.solve(M, (p - a).T).T
        ok = (l >= -1e-6).all(1) & (l.sum(1) <= 1 + 1e-6)
        x, y = xs.ravel()[ok], ys.ravel()[ok]
        tri[y, x] = i
        bary[y, x] = np.stack([1 - l[ok].sum(1), l[ok, 0], l[ok, 1]], 1)
    return tri, bary


def _rel_inside(path, folder):
    """Path relative to dossier if it is inside it (movable library), otherwise unchanged."""
    r = os.path.relpath(path, folder)
    return path if r.startswith("..") else r


def read_card(lib, name):
    """fiche.json with pmx and texture as absolute paths (stored relative to the character folder)."""
    d = os.path.join(lib, name)
    with open(os.path.join(d, "record.json"), encoding="utf-8") as fh:
        f = json.load(fh)
    for k in ("pmx", "texture"):
        if f.get(k):
            f[k] = os.path.normpath(os.path.join(d, f[k]))
    return f


# 6 axes and 8 diagonals: normal sorting directions (oriented texture sampling)
_DIRS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                 + [(x, y, z) for x in (1, -1) for y in (1, -1) for z in (1, -1)], dtype=np.float64)
_DIRS /= np.linalg.norm(_DIRS, axis=1)[:, None]


def _normals(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    return n / (np.linalg.norm(n, axis=1)[:, None] + 1e-12)


def _reorient(P3, nA, XD, DF, fD, bD, dD):
    """Texels whose nearest donor face points the wrong way (inside of a lip or of a folded
    eyelid): they are reprojected onto faces turned the same way. Returns the number of
    texels redone."""
    nD = _normals(XD, DF)
    bad = np.where((nA * nD[fD]).sum(1) < 0)[0]
    if not len(bad):
        return 0
    lab = np.argmax(nA[bad] @ _DIRS.T, 1)
    for j in np.unique(lab):
        sel = bad[lab == j]
        good = np.where(nD @ _DIRS[j] > 0.2)[0]
        if not len(good):
            continue
        _, f2, b2, d2 = G.plus_near(P3[sel], XD, DF[good])
        fD[sel], bD[sel], dD[sel] = good[f2], b2, d2
    return int(len(bad))


def extract_or_say(args):
    try:
        return extract_one(args)
    except Exception as e:  # noqa: BLE001
        return args[0], "%s: %s" % (type(e).__name__, e)


def extract_one(args):
    """args = (name, pmx, sortie[, oriented]); oriented: texture sampling limited to the donor
    faces turned like the pivot's (Outerplane: inside of the lips painted red)."""
    name, path, output = args[:3]
    oriented = len(args) > 3 and args[3]
    A = load_pivot()
    D = TT.Head(path)
    if oriented:  # Outerplane: eye rim seen from the front, not the folded slit of the hole
        D.eye_visible()
    s = float(np.sqrt(A.W * A.H_crane / (D.W * D.H_crane)))
    D.put(s, D.center_face(), A.center_face())
    XD = D._host(D.V)
    rec = RC.register(A, A.V, D, XD)
    X = rec["X"]
    border = np.zeros(D.n, dtype=bool)
    for b in D.loops:
        border[b] = True
    reliable = (rec["dist"] < PV.RELIABLE_DIST * A.iod) & ~border[D.F[rec["face"]]].any(1)
    # skin morphs sampled at the registered points, at the pivot's scale
    names = [n for n in A.morphs]
    delta = np.zeros((len(names), A.n, 3), dtype=np.float32)
    present = np.zeros(len(names), dtype=bool)
    # around the openings, each point projects onto the same side on the donor
    face, bary = rec["face"].copy(), rec["bary"].copy()
    cA = sides(A, SIDE_RADIUS * A.iod)
    cD = sides(D, SIDE_RADIUS * D.iod)
    cF = np.sign(cD[D.F].astype(int).sum(1))
    for sg in (1, -1):
        qa = np.where(cA == sg)[0]
        fd = np.where(cF != -sg)[0]
        if len(qa) and len(fd):
            _, f2, b2, _ = G.plus_near(X[qa], XD, D.F[fd])
            face[qa], bary[qa] = fd[f2], b2
    tri = D.F[face]
    for k, n in enumerate(names):
        if n in D.morphs:
            d = D.offset_skin(n)
            delta[k] = s * (bary[:, :, None] * d[tri]).sum(1)
            present[k] = np.abs(d).max() > 0
    # donor UVs for each texel of the pivot's skin
    uvA = A.t["uv"]
    fa = A.faces[A.skin_faces_raw]
    tri_t, bary_t = raster_uv(uvA[fa], TEX)
    ok = tri_t >= 0
    wv = A.b2s[fa[tri_t[ok]]]  # pivot's welded vertices
    P3 = (bary_t[ok][:, :, None] * X[wv]).sum(1)
    _, fD, bD, dD = G.plus_near(P3, XD, D.F)
    # pivot normals at rest: those of the registered pivot (X) flip where the registration
    # folds the skin (mouth corners, eyelid rims) and scattered red texels there
    resumed = _reorient(P3, _normals(A.V, wv), XD, D.F, fD, bD, dD) if oriented else None
    uvD = D.t["uv"][D.faces[D.F_brut[fD]]]
    uv = np.zeros((TEX, TEX, 2), dtype=np.float32)
    uv[ok] = (bD[:, :, None] * uvD).sum(1)
    m = D.pmx["materials"][D.i_skin[0]]
    tex = os.path.normpath(os.path.join(os.path.dirname(path), D.pmx["textures"][m["texture"]].replace("\\", "/")))
    parts = arrange_parts(D)
    d_ = os.path.join(output, name)
    os.makedirs(d_, exist_ok=True)
    np.savez_compressed(os.path.join(d_, "part.npz"), X=X.astype(np.float32), reliable=reliable,
                        dist=(rec["dist"] / A.iod).astype(np.float32), names=np.array(names), delta=delta,
                        present=present, uv=uv, uv_ok=ok, scale=s, c_src=D.c_src, c_dst=D.c_dst,
                        **{"parts_" + p: v for p, v in parts.items()})
    M = masks(A)
    card = {"person": name, "pmx": _rel_inside(path, d_), "texture": _rel_inside(tex, d_), "scale": s,
             "mean_distance_iod": float(np.mean(rec["dist"]) / A.iod),
             "far_texels": float(np.mean(dD > 0.03 * A.iod)), "regions": {}}
    if resumed is not None:
        card["texels_reorientes"] = float(resumed / len(P3))
    mats = D.names_mat
    for p in REGIONS + (BASE,):
        w = mask_region(M, p)
        fs = parts.get(p, np.zeros(0, dtype=np.int64))
        card["regions"][p] = {
            "reliability": float((w * reliable).sum() / w.sum()),
            "parts": sorted(set(mats[i] for i in D.mat_face[fs])),
            "part_faces": int(len(fs)),
        }
    card["morphs"] = [n for n, ok_ in zip(names, present) if ok_]
    card["missing_morphs"] = [n for n, ok_ in zip(names, present) if not ok_]
    _write_json(os.path.join(d_, "record.json"), card)
    return name, card


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# face parts of an Outerplane head: the piece required for it to be provided
REQUIRED_OP = {"eyes": ("目", "eye plate missing"), "eyebrows": ("眉", "eyebrows missing from the geometry")}
# carried blink: fraction of the pivot's eye hole that stays open when the blink sampled on the
# donor is applied to it; ZZZ: 0 to 0.27 (Jane), median 0.03. Beyond that, the donor's eye closes
# by something other than its skin rim (Outerplane: an eyelid sliding in front of a folded
# socket) and its eyes, placed on another head, stay open when blinking.
BLINKS_MAX = 0.3


def blink_worn(A, X, d):
    """{side: area of the registered pivot's (X) eye hole closed (X + d) / open area}."""
    area = lambda V, b: abs(G.area_polygon_xy(V[b][:, :2]))  # noqa: E731
    return {c: round(float(area(X + d, b) / max(area(X, b), 1e-12)), 3) for c, b in A.loop_eye.items()}


def extract_outerplane_one(args):
    """Conversion (outerplane.py) then extraction; (name, record or None, reason)."""
    import outerplane as OP
    name, folder, output = args
    src = os.path.join(output, name, name, "head.pmx")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    try:
        T = OP.convert(folder, src)
        _, card = extract_one((name, src, output, True))
    except Exception as e:  # noqa: BLE001  (a rejected head does not block the others)
        # nothing of this head stays in the library (trash, no deletion)
        # gio refuses the trash on /tmp: the head is then renamed _refused-<name> in place
        # (the sandbox refuses to move it to another folder: EXDEV)
        d = os.path.join(output, name)
        import shutil
        if os.path.exists(d) and (not shutil.which("gio") or subprocess.run(["gio", "trash", d], capture_output=True).returncode):
            os.replace(d, os.path.join(output, "_refused-" + name))
        return name, None, str(e)
    card["game"] = "Outerplane"
    card["folder_game"] = os.path.basename(folder)
    card["conversion"] = {"mouth": T.mouth, "materials": T.materials}
    card["not_provided"] = {}
    for p, (part_name, reason) in REQUIRED_OP.items():
        if part_name not in card["regions"][p]["parts"]:
            card["not_provided"][p] = reason
    z = np.load(os.path.join(output, name, "part.npz"))
    names = list(z["names"])
    if TT.MORPH_BLINKS in names and z["present"][names.index(TT.MORPH_BLINKS)]:
        cl = blink_worn(load_pivot(), z["X"].astype(float), z["delta"][names.index(TT.MORPH_BLINKS)].astype(float))
        card["blink_worn"] = cl
        if max(cl.values()) > BLINKS_MAX and "eyes" not in card["not_provided"]:
            card["not_provided"]["eyes"] = ("blink not carried: on the pivot, the eye stays %d %% open "
                                             "(ZZZ: 27 %% at most)" % round(100 * max(cl.values())))
    _write_json(os.path.join(output, name, "record.json"), card)
    return name, card, None


def write_index(output, refusals=None):
    """index.json: all characters extracted in sortie, plus those of the existing index that
    have no folder there; refusals: name -> {jeu, dossier, reason} (heads not provided)."""
    fi = os.path.join(output, "index.json")
    old = {}
    if os.path.exists(fi):
        with open(fi, encoding="utf-8") as fh:
            old = json.load(fh)
    here = lambda n: os.path.exists(os.path.join(output, n, "record.json"))  # noqa: E731
    idx = {"pivot": load_pivot().name, "regions": {p: {} for p in REGIONS + (BASE,)}}
    for p, d in old.get("regions", {}).items():
        for n, v in d.items():
            if not here(n):
                idx["regions"].setdefault(p, {})[n] = v
    non = {n: v for n, v in old.get("not_provided", {}).items() if not here(n) and n not in (refusals or {})}
    for n in sorted(os.listdir(output)):
        if not here(n):
            continue
        with open(os.path.join(output, n, "record.json"), encoding="utf-8") as fh:
            f = json.load(fh)
        nf = f.get("not_provided", {})
        for p, v in f["regions"].items():
            if p in nf:
                continue
            e = {"reliability": round(v["reliability"], 3), "parts": v["parts"]}
            if f.get("game"):
                e["game"] = f["game"]
            idx["regions"][p][n] = e
        if nf:
            non[n] = {"game": f.get("game", PV.GAME.value("abrege", PV.GAME.title)), "folder": f.get("folder_game"), "regions": nf}
    for n, r in (refusals or {}).items():
        non[n] = dict(r, regions={p: r["reason"] for p in REGIONS + (BASE,)})
    if non:
        idx["not_provided"] = dict(sorted(non.items()))
    for k in ("hairstyle",):
        if k in old:
            idx[k] = old[k]
    _write_json(fi, idx)
    log("index: %s" % fi)
    return idx


def images(output, names, work="texture"):
    cmd = ["bash", RT.FILE_BLENDER, "-b", "--factory-startup", "--python", os.path.join(HERE, "library_images.py"),
           "--", work, output] + list(names)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or "images: done" not in r.stdout:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("images: Blender failed")


def _map_fork(fn, jobs, tasks):
    """map via os.fork, without semaphores: each child handles one job out of every `tasks` and
    writes its results (pickle) to a temporary file; job order is kept."""
    import pickle
    import tempfile
    rep = tempfile.mkdtemp(prefix="map-")
    pids = []
    for k in range(tasks):
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                out = [(i, fn(jobs[i])) for i in range(k, len(jobs), tasks)]
                with open(os.path.join(rep, "%d.pkl" % k), "wb") as fh:
                    pickle.dump(out, fh)
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                code = 1
            os._exit(code)
        pids.append(pid)
    res = {}
    for k, pid in enumerate(pids):
        os.waitpid(pid, 0)
        f = os.path.join(rep, "%d.pkl" % k)
        if not os.path.exists(f):
            raise RuntimeError("map: worker %d returned nothing" % k)
        with open(f, "rb") as fh:
            res.update(dict(pickle.load(fh)))
        os.replace(f, f + ".lu")
    return [res[i] for i in range(len(jobs))]


def map_(fn, jobs, tasks):
    """Parallel map; serial if tasks <= 1. A memory-capped launcher (compute.launcher) may refuse
    ProcessPoolExecutor's semaphores: os.fork is used instead."""
    jobs = list(jobs)
    if tasks <= 1:
        for j in jobs:
            yield fn(j)
        return
    try:
        ex = ProcessPoolExecutor(tasks)
    except PermissionError:
        yield from _map_fork(fn, jobs, tasks)
        return
    with ex:
        yield from ex.map(fn, jobs)


def extract_outerplane(output, tasks, alone=None):
    import outerplane as OP
    os.makedirs(output, exist_ok=True)
    candidates, discarded = OP.candidates(PV.PEOPLE)
    jobs = [(n, d, output) for n, d in candidates if not alone or n in alone]
    refusals, ok = {}, []
    for (name, f, reason), (_, d, _) in zip(map_(extract_outerplane_one, jobs, tasks), jobs):
        if f is None:
            refusals[name] = {"game": "Outerplane", "folder": os.path.basename(d), "reason": reason}
            log("  %-16s NOT PROVIDED: %s" % (name, reason))
            continue
        ok.append(name)
        log("  %-16s distance %.4f, reliability eyes %.2f nose %.2f mouth %.2f, eye parts %s%s" % (
            name, f["mean_distance_iod"], f["regions"]["eyes"]["reliability"], f["regions"]["nose"]["reliability"],
            f["regions"]["mouth"]["reliability"], ",".join(f["regions"]["eyes"]["parts"]),
            ("; not provided: %s" % f["not_provided"]) if f["not_provided"] else ""))
    if not alone:
        for d, r in discarded.items():
            refusals[OP.name_readable(d) if OP.name_readable(d) != d else d] = {"game": "Outerplane", "folder": d,
                                                                         "reason": "excluded: " + r}
    log("Outerplane: %d heads extracted, %d not provided" % (len(ok), len(refusals)))
    if ok:
        log("textures copied into the pivot's UVs (Blender)")
        images(output, ok)
    write_index(output, refusals)


def extract(output, tasks, alone=None):
    os.makedirs(output, exist_ok=True)
    for n in sorted(set(alone or ()) - set(PV.PEOPLE)):
        log("  %-12s unknown among the active game's models" % n)
    names = [n for n in PV.PEOPLE if not alone or n in alone]
    jobs = [(n, os.path.join(PV.SOURCES, PV.PEOPLE[n]), output) for n in names]
    cards, refusals = {}, {}
    with ProcessPoolExecutor(tasks) as ex:
        for name, f in ex.map(extract_or_say, jobs):
            if isinstance(f, str):      # a head that cannot be sampled does not stop the others
                refusals[name] = {"game": PV.games.name_active(), "folder": os.path.dirname(PV.PEOPLE[name]), "reason": f}
                log("  %-12s excluded: %s" % (name, f))
                continue
            cards[name] = f
            log("  %-12s distance %.4f, reliability eyes %.2f nose %.2f mouth %.2f, eye parts %s" % (
                name, f["mean_distance_iod"], f["regions"]["eyes"]["reliability"], f["regions"]["nose"]["reliability"],
                f["regions"]["mouth"]["reliability"], ",".join(f["regions"]["eyes"]["parts"])))
    log("textures copied into the pivot's UVs (Blender)")
    if cards:
        images(output, [n for n in names if n in cards])
    write_index(output, refusals)


# =================================================================== composition

def _load(lib, name):
    z = np.load(os.path.join(lib, name, "part.npz"))
    return dict(z), read_card(lib, name)


def shape_composed(A, M, npz, recipe):
    """Shape of the composed skin (pivot's frame and weld order), offset of each part
    (zone -> translation) and face weight. npz: donor -> loaded part.npz."""
    Xb = npz[recipe[BASE]]["X"].astype(float)
    # shape: each part (each side for eyes and eyebrows) moved to its place in the base
    X = Xb.copy()
    move = {}
    for zone, m in M.items():
        if zone == BASE:
            continue
        p = zone.split("_")[0]
        Xd = npz[recipe[p]]["X"].astype(float)
        T = (m[:, None] * (Xb - Xd)).sum(0) / m.sum()
        move[zone] = T
        X += m[:, None] * (Xd + T - Xb)
    # the composed face replaces the pivot head only within the face: skull, ears and neck
    # stay the pivot head's (a retarget_head visage pass only takes the face)
    wv = A.weight_face(RT.MARGIN * A.iod, RT.BAND * A.iod)
    return A.V + wv[:, None] * (X - A.V), move, wv


def compose(lib, recipe, output):
    """recipe: part -> character (base required). Writes <sortie>/visage.pmx."""
    os.makedirs(output, exist_ok=True)
    A = load_pivot()
    M = masks(A)
    donors = {n: _load(lib, n) for n in set(recipe.values())}
    # a part the library does not provide (record: not_provided) is not accepted
    for p, n in recipe.items():
        nf = donors[n][1].get("not_provided", {})
        if p in nf:
            raise ValueError("%s does not provide %s: %s (take this part from another character)" % (n, p, nf[p]))
    Xf, move, wv = shape_composed(A, M, {n: z for n, (z, _) in donors.items()}, recipe)
    # skin morphs: from each part's donor, otherwise from the base, otherwise the pivot's;
    # all face morphs of the donors, even those the pivot lacks (morphs.py)
    gifts = MO.Donors(lib, A)
    names = list(dict.fromkeys(list(A.morphs) + MO.names_recipe(gifts, recipe, M, wv)))
    delta = {n: MO.delta_composed(gifts, recipe, M, wv, n) for n in names}
    path = _write_composed(A, Xf, delta, recipe, gifts, move, output)
    # texture: part weights in the pivot's UVs
    uvA = A.t["uv"]
    fa = A.faces[A.skin_faces_raw]
    tri_t, bary_t = raster_uv(uvA[fa], TEX)
    ok = tri_t >= 0
    wvs = A.b2s[fa[tri_t[ok]]]
    weight = {}
    for p in REGIONS + (BASE,):
        g = np.zeros((TEX, TEX), dtype=np.float32)
        g[ok] = (bary_t[ok] * mask_region(M, p)[wvs]).sum(1)
        weight[p] = g
    g = np.zeros((TEX, TEX), dtype=np.float32)
    g[ok] = (bary_t[ok] * wv[wvs]).sum(1)
    m = A.pmx["materials"][A.i_skin[0]]
    np.savez_compressed(os.path.join(output, "texture_weights.npz"), ok=ok, visage=g,
                        **{"weight_" + p: v for p, v in weight.items()})
    _write_json(os.path.join(output, "recipe.json"), {
        "recipe": recipe, "library": lib,
        "texture_pivot": os.path.normpath(os.path.join(os.path.dirname(A.path), A.pmx["textures"][m["texture"]].replace("\\", "/")))})
    images(output, [], work="composed")
    return path


def _write_composed(A, Xf, delta, recipe, gifts, move, output):
    P = copy.deepcopy(A.pmx)
    nuv = P["n_uv_add"]
    P["textures"] = [RT._texture_rel(x, os.path.dirname(A.path), output) for x in P["textures"]]
    S = P["vertices"]
    vb = A.skin_raw
    k = A.b2s[vb]
    nor = G.normals_vertices(Xf, A.F)
    for i, kk in zip(vb, k):
        S["pos"][i] = tuple(Xf[kk])
        S["nor"][i] = tuple(nor[kk])
    # morphs: offsets per raw vertex, written after the parts (MO.write_morph)
    skin = set(int(i) for i in vb)
    accessory = {n: {} for n in delta}
    # parts: the pivot's removed (materials kept, empty), the donors' added right after the
    # skin
    pieces_A = set(A.i_parts)
    new = []  # (material, faces)
    mixed = len({gifts[n][1].get("game", PV.GAME.value("abrege", PV.GAME.title)) for n in set(recipe.values())}) > 1
    for p in ("eyes", "eyebrows", "mouth", BASE):
        name_d = recipe[p]
        z, card, D, _ = gifts[name_d]
        fs = gifts.parts(name_d, p)
        if not len(fs):
            continue
        Pd = D.pmx
        sd = Pd["vertices"]
        map_ = RT._map_os(D, A)
        vs = np.unique(D.faces[fs])
        pos = D._host(D.pos[vs])
        if p in ("eyes", "eyebrows"):
            # each side follows the offset of its eye or eyebrow
            for c, sg in (("g", 1), ("d", -1)):
                side = (pos[:, 0] * sg) > 0
                pos[side] += move["%s_%s" % (p, c)]
        elif p == "mouth":
            pos += move["mouth"]
        base_v = len(S["pos"])
        idx = {int(v): base_v + j for j, v in enumerate(vs)}
        for j, v in enumerate(vs):
            v = int(v)
            t = sd["type"][v]
            o = [int(map_[x]) if x >= 0 else -1 for x in sd["bones"][v]]
            if t == 3:
                t = 1
            if t == 4:
                t = 2
            if t == 0:
                o = [o[0], -1, -1, -1]
            u = list(sd["uv_add"][v])[:nuv]
            S["pos"].append(tuple(pos[j]))
            S["nor"].append(sd["nor"][v])
            S["uv"].append(sd["uv"][v])
            S["uv_add"].append(u + [(0.0, 0.0, 0.0, 0.0)] * (nuv - len(u)))
            S["type"].append(t)
            S["bones"].append(tuple(o))
            S["weights"].append(tuple(sd["weights"][v]))
            S["sdef"].append(None)
            S["contour"].append(sd["contour"][v])
        # part morphs (lash blink...): the donor's at the pivot's scale, plus the difference
        # between the composed skin and the donor's skin under the part (the whole difference
        # if the donor lacks the morph: the part follows the skin); the iris does not follow the skin
        suit = ~np.isin(D.mat_face[fs], [i for i, nm in enumerate(D.names_mat) if nm in MO.FIXES])
        v_suit = np.isin(vs, np.unique(D.faces[fs[suit]]))
        # inside of the mouth and eyes: stays behind the skin (MO.INTERIOR)
        v_int = np.isin(vs, np.unique(D.faces[fs[np.isin(D.mat_face[fs], [i for i, nm in enumerate(D.names_mat) if nm in MO.INTERIOR])]]))
        dv = pos - D.pos[vs]  # translation of the part (move)
        (Fd_, b_d), (Fp_, b_p) = MO.skin_common(
            D.pos, D.faces[D.skin_faces_raw], D.pos[vs], Xf, A.F,
            lambda x, vec: x if vec else x + dv)
        # recipe mixing games: the difference is taken at the same pivot point, the one nearest
        # the placed part (composed skin minus this donor's skin carried onto the pivot). The
        # donor point carried then projected landed, for OP, away from the skin touching the
        # part: lashes frozen during the blink on a ZZZ base.
        # OP donor: its 颜2 makeup plates have no morph (motionless on a skin that moves up to
        # 0.15); a part vertex that the donor's morph leaves in place while the donor's skin
        # moves under it follows the composed skin entirely.
        op = card.get("game") == "Outerplane"
        same_point = mixed or op
        if same_point:
            _, f_m, b_m, _ = G.plus_near(pos, Xf, A.F)
            F_m = A.F[f_m]
        for n in delta:
            dd = gifts.disp(name_d, n)
            d = np.zeros((len(vs), 3)) if dd is None else D.scale * dd[vs]
            if same_point:
                under = (b_m[:, :, None] * delta[n][F_m]).sum(1)
                sk_d = (b_m[:, :, None] * gifts.skin(name_d, n)[F_m]).sum(1) if dd is not None else 0
                e = under - sk_d
                if op and dd is not None:
                    a_d = np.linalg.norm(sk_d, axis=1)
                    frozen = v_suit & (a_d > MO.THRESHOLD) & (np.linalg.norm(d, axis=1) < MO.FROZEN * a_d)
                    d[frozen], e[frozen] = 0.0, under[frozen]
            else:
                # composed skin under the part, minus the donor's skin under the part on the donor
                e = (b_p[:, :, None] * delta[n][Fp_]).sum(1)
                if dd is not None:
                    e -= D.scale * (b_d[:, :, None] * dd[Fd_]).sum(1)
            d[v_suit] += e[v_suit]
            # even when motionless, an inner part stays behind a moving skin
            if v_int.any():
                d[v_int] = MO.keep(Xf, Xf + delta[n], A.F, pos[v_int], d[v_int])[0]
            for j in np.where(np.abs(d).max(1) > 1e-7)[0]:
                accessory[n][idx[int(vs[j])]] = d[j]
        off = len(P["textures"])
        P["textures"] += [RT._texture_rel(x, os.path.dirname(D.path), output) for x in Pd["textures"]]
        for i in np.unique(D.mat_face[fs]):
            mm = copy.deepcopy(Pd["materials"][int(i)])
            for kk in ("texture", "sphere"):
                if mm[kk] >= 0:
                    mm[kk] += off
            if mm["toon_sharing"] == 0 and mm["toon"] >= 0:
                mm["toon"] += off
            # name kept (the head tools recognise parts by their name); the donor is noted in
            # the English name
            mm["name_en"] = "%s (%s)" % (mm["name_en"] or mm["name"], name_d)
            new.append((mm, [tuple(idx[int(v)] for v in D.faces[f]) for f in fs[D.mat_face[fs] == i]]))
    dA_os = {}
    for n, d in delta.items():
        m = next((x for x in P["morphs"] if x["name"] == n), None)
        offset = {}
        if m is not None and m["type"] == 1:
            offset = {int(v): np.asarray(x) for v, x in m["offsets"] if int(v) not in skin}
        elif m is not None:
            # bone (or group) morph of the pivot: becomes a vertex morph, its effect outside the
            # skin kept
            if not dA_os:
                dA_os["Mo"] = MO.Model(pmx=A.pmx)
            dd = dA_os["Mo"].displacement(n)
            offset = {int(v): dd[v] for v in np.where(np.abs(dd).max(1) > 1e-7)[0] if int(v) not in skin}
        for i, kk in zip(vb, k):
            if np.abs(d[kk]).max() > 1e-7:
                offset[int(i)] = d[kk]
        offset.update(accessory[n])
        if m is None and not any(np.linalg.norm(x) >= MO.THRESHOLD for x in offset.values()):
            continue
        panel, name_en = (m["panel"], m["name_en"]) if m is not None else MO.panel_of(gifts, recipe, n)
        MO.write_morph(P, n, offset, panel, name_en)
    faces_A = np.asarray(P["faces"])
    mf = np.repeat(np.arange(len(P["materials"])), [m["n_indices"] // 3 for m in P["materials"]])
    mats, faces, new_index = [], [], {}
    i_skin = A.i_skin[0]
    for i, m in enumerate(P["materials"]):
        m = copy.deepcopy(m)
        fl = [] if i in pieces_A else [tuple(int(x) for x in f) for f in faces_A[mf == i]]
        if i in pieces_A:
            m["name"] += "_pivot"  # emptied: its parts are replaced by the donors'
        if i == i_skin:
            m["texture"] = len(P["textures"])
        m["n_indices"] = 3 * len(fl)
        new_index[i] = len(mats)
        mats.append(m)
        faces += fl
        if i == i_skin:
            for mm, fl2 in new:
                mm["n_indices"] = 3 * len(fl2)
                mats.append(mm)
                faces += fl2
    P["textures"].append("composed_face.png")
    for m in P["morphs"]:
        if m["type"] == 8:
            for d in m["offsets"]:
                if d["material"] >= 0:
                    d["material"] = new_index[d["material"]]
    P["materials"] = mats
    P["faces"] = faces
    P["name"] = "visage-compose"
    P["comment"] = (P.get("comment") or "") + "\r\n[library: %s]" % ", ".join(
        "%s from %s" % (p, n) for p, n in recipe.items())
    path = os.path.join(output, "face.pmx")
    pmx_write.write(P, path)
    return path


# =================================================================== principal

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("extract", "composed"))
    ap.add_argument("--output", dest="output", required=True)
    ap.add_argument("--library", dest="lib")
    ap.add_argument("--jobs", dest="tasks", type=int, default=4)
    ap.add_argument("--only", dest="alone")
    ap.add_argument("--outerplane", action="store_true", help="extract: the Outerplane heads (outerplane.py)")
    ap.add_argument("--base")
    for p in REGIONS:
        ap.add_argument("--" + p)
    ap.add_argument("--onto", dest="on", help="character on which to place the composed face")
    ap.add_argument("--hair", dest="hair", help="with --onto: character that gives its hairstyle (hairstyle.py)")
    ap.add_argument("--ears", dest="ears", help="with --hair: character that gives its animal ears, or 'aucune' (none)")
    a = ap.parse_args(argv)
    if a.action == "extract":
        (extract_outerplane if a.outerplane else extract)(a.output, a.tasks, a.alone.split(",") if a.alone else None)
        return
    recipe = {BASE: a.base}
    for p in REGIONS:
        recipe[p] = getattr(a, p) or a.base
    path = compose(a.lib, recipe, a.output)
    log("composed face: %s" % path)
    if a.on:
        target = os.path.join(PV.SOURCES, PV.PEOPLE[a.on])
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "retarget_head.py"), "face", "--source", path,
                            "--target", target, "--output", os.path.join(a.output, "on-" + a.on)])
        res = os.path.join(a.output, "on-" + a.on, "result.pmx")
        if r.returncode and not os.path.exists(res):
            raise RuntimeError("retarget_head visage failed")
        # a failed measurement (measures.json) leaves a written result: it is finished anyway
        MO.keep_pose(res, log)
        if a.hair:
            cmd = [sys.executable, "-I", os.path.join(HERE, "hairstyle.py"), "place", "--library", a.lib,
                   "--hair", a.hair, "--onto", a.on, "--target", os.path.join(a.output, "on-" + a.on, "result.pmx"),
                   "--output", os.path.join(a.output, "on-%s-hair-%s" % (a.on, a.hair))]
            if a.ears:
                cmd += ["--ears", a.ears]
            if subprocess.run(cmd).returncode:
                raise RuntimeError("coiffure poser failed")
        if r.returncode:
            raise RuntimeError("retarget_head visage: failed measurements (see %s); result written and finished"
                               % os.path.join(a.output, "on-" + a.on, "measures.json"))


if __name__ == "__main__":
    main()

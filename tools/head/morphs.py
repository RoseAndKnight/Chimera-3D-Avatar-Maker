"""Face morphs: evaluation, coverage check, transfer, eyebrow reshaping.

    python3 -I tools/head/morphs.py check --pmx <result.pmx> [--recipe recipe.json] [--json output.json]
    python3 -I tools/head/morphs.py repair --pmx <result.pmx> --recipe recipe.json --output <folder>
    python3 -I tools/head/morphs.py parts --pmx <result.pmx> --recipe recipe.json --output <folder>
    python3 -I tools/head/morphs.py eyebrow --pmx <result.pmx> --output <folder> \
        [--length 1.0] [--thickness 1.3] [--arc 0.04] [--rise 0.0] [--tilt 0.0] [--no-guard]
    python3 -I tools/head/morphs.py guard --pmx <result.pmx> --output <folder> [--min-gap 0.045]
    python3 -I tools/head/morphs.py gaps --pmx <result.pmx> [--json output.json]
    (writing: --textures-for <folder> rewrites texture paths relative to the folder where the
    PMX will be placed, and also looks for textures there)

Evaluate: displacement of each vertex for a morph (vertex, bone, group), at rest elsewhere.
Bone morphs are evaluated by linear skinning (BDEF; SDEF treated as BDEF2), with the bone
hierarchy and the rotation / translation inheritance (付与).

Check (one line per face morph):
- per face material, the maximum displacement;
- "empty": a donor (recipe) has the morph, the composed face does not move;
- "frozen": a part laid on the skin (lashes, eyebrows, shadows, 颜2, 口线) stays in place
  where the skin under it moves;
- "pierces": an inner part (eyeballs, teeth, tongue) comes in front of the skin, while
  at rest it is behind it (teeth poking out at the corner of a smile).

Repair: see repair(); eyebrow reshaping: see deform_eyebrows().
Parts: removes from a finished PMX the faces that an older sorting had put in another region
(remove_parts; the material stays, empty). Guard: minimal eyebrow - eye line gap in
each morph (keep_eyebrows, GAP_EYEBROW_MIN); gaps: the measurement only.
"""
import argparse
import copy
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))

import geom as G  # noqa: E402
import pmx as PMX  # noqa: E402

SKINS = ("颜", "颜_A", "颜_A2")
# parts laid on the skin: they must follow the skin
SURFACE = ("颜2", "口线", "眉", "眉毛", "睫", "眉睫", "睫眉", "右睫眉", "二重", "二重影")
# inner parts: they move as in the donor, but must not pierce the skin
INTERIOR = ("白目", "目", "目影", "目光", "目光2", "瞳光", "目色", "齿", "口舌")
EYEBROWS = ("眉", "眉毛")
THRESHOLD = 0.01       # displacement (MMD units) below which nothing moves
FROZEN = 0.25       # a laid part moves less than this fraction of the skin under it: frozen
PART_FROZEN = 0.3   # fraction of a part's vertices where the skin moves that must be frozen to call it frozen
SKIN_MOVES = 0.4   # the skin under the part moves: at least this fraction of the largest skin displacement
WEAK = 0.5       # the skin moves less than this fraction of the expected composed skin: missing
PIERCED = 0.01       # lead in front of the skin (MMD units) that counts as piercing


# =================================================================== evaluation

def _quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp_id(q, f):
    q = np.asarray(q, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    if q[3] < 0:
        q = -q
    ang = 2 * np.arccos(np.clip(q[3], -1, 1))
    s = np.sin(ang / 2)
    if s < 1e-9:
        return np.array([0, 0, 0, 1.0])
    ax = q[:3] / s
    return np.concatenate([ax * np.sin(f * ang / 2), [np.cos(f * ang / 2)]])


class Model:
    """A loaded PMX, with its arrays and the morph index."""

    def __init__(self, path=None, pmx=None):
        self.path = path
        self.pmx = pmx if pmx is not None else PMX.read(path)
        self.t = PMX.to_arrays(self.pmx)
        self.pos = self.t["pos"]
        self.nv = len(self.pos)
        self.index = {}
        for i, m in enumerate(self.pmx["morphs"]):
            self.index.setdefault(m["name"], i)
        self.names_mat = [m["name"] for m in self.pmx["materials"]]
        self.mat_vert = np.full(self.nv, -1)
        F = self.t["faces"]
        mf = self.t["mat_face"]
        self.mat_vert[F[:, 0]] = mf
        self.mat_vert[F[:, 1]] = mf
        self.mat_vert[F[:, 2]] = mf

    def vertices(self, names):
        ids = [i for i, n in enumerate(self.names_mat) if n in names]
        return np.unique(self.t["faces"][np.isin(self.t["mat_face"], ids)])

    def _accumuler(self, i, w, dv, poses, depth=0):
        m = self.pmx["morphs"][i]
        if depth > 8 or w == 0:
            return
        if m["type"] == 1:
            for v, d in m["offsets"]:
                dv[int(v)] += w * np.asarray(d)
        elif m["type"] == 2:
            for o, t, q in m["offsets"]:
                tt, qq = poses.get(int(o), (np.zeros(3), np.array([0, 0, 0, 1.0])))
                q2 = _slerp_id(q, w)
                # composed rotations (small angles: order matters little)
                x1, y1, z1, w1 = qq
                x2, y2, z2, w2 = q2
                qq = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                               w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
                poses[int(o)] = (tt + w * np.asarray(t), qq)
        elif m["type"] in (0, 9):
            for j, f in m["offsets"]:
                self._accumuler(int(j), w * f, dv, poses, depth + 1)

    def matrices(self, poses):
        """Skinning matrices (nb, 4, 4) relative to rest for local bone poses."""
        os_ = self.pmx["bones"]
        nb = len(os_)
        M = [None] * nb

        def locale(b):
            t, q = poses.get(b, (np.zeros(3), np.array([0, 0, 0, 1.0])))
            a = os_[b]["addition"]
            if a is not None and 0 <= a["source"] < nb and a["source"] in poses:
                ts, qs = poses[a["source"]]
                if a["rotation"]:
                    q2 = _slerp_id(qs, a["factor"])
                    t_, q = t, q
                    x1, y1, z1, w1 = q
                    x2, y2, z2, w2 = q2
                    q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                                  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
                if a["displacement"]:
                    t = t + a["factor"] * np.asarray(ts)
            p = np.asarray(os_[b]["pos"], dtype=float)
            L = np.eye(4)
            R = _quat(q)
            L[:3, :3] = R
            L[:3, 3] = p + np.asarray(t) - R @ p
            return L

        def glob(b, depth=0):
            if M[b] is not None:
                return M[b]
            per = os_[b]["parent"]
            L = locale(b)
            M[b] = (glob(per, depth + 1) @ L) if (0 <= per < nb and depth < 512) else L
            return M[b]
        actives = set(poses)
        for b in range(nb):
            a = os_[b]["addition"]
            if a is not None and a["source"] in poses:
                actives.add(b)
        # only bones under an active bone change: the others stay at identity
        out = np.tile(np.eye(4), (nb, 1, 1))
        if not actives:
            return out
        for b in range(nb):
            out[b] = glob(b)
        return out

    def displacement(self, name, w=1.0):
        """Displacement (nv, 3) of each vertex for morph name at value w."""
        dv = np.zeros((self.nv, 3))
        if name not in self.index:
            return dv
        poses = {}
        self._accumuler(self.index[name], w, dv, poses)
        p = self.pos + dv
        if not poses:
            return dv
        M = self.matrices(poses)
        idx = self.t["bones"]
        wt = self.t["weights"].copy()
        wt[idx < 0] = 0
        ii = np.where(idx < 0, 0, idx)
        ph = np.concatenate([p, np.ones((self.nv, 1))], 1)
        q = np.zeros((self.nv, 3))
        for k in range(4):
            q += wt[:, k:k + 1] * np.einsum("nij,nj->ni", M[ii[:, k]][:, :3], ph)
        s = wt.sum(1, keepdims=True)
        q = np.where(s > 1e-9, q / np.maximum(s, 1e-9), p)
        return q - self.pos

    def morphs_face(self):
        """Face morphs: eyebrow, eye, mouth panels (1 to 3), vertex, bone or group morphs."""
        out = []
        for m in self.pmx["morphs"]:
            if m["panel"] in (1, 2, 3) and m["type"] in (0, 1, 2) and m["name"] not in out:
                out.append(m["name"])
        return out


def keep(V0, V1, F, p0, d, margin=0.002):
    """Inner part (teeth, tongue, eyeballs): vertices behind the skin (V0, F) at rest stay
    behind the deformed skin V1. A vertex that would come in front is pushed back in depth to
    just behind it (by marge; to its rest lead if it already stuck out a little), seen from
    the front like the PIERCED check. Returns (corrected d, count)."""
    a0 = _progress(V0, F, p0)
    a1 = _progress(V1, F, p0 + d)
    target = np.where(a0 <= 0, -margin, a0)
    m = (a0 <= PIERCED) & (a1 > np.maximum(a0, 0))
    d = d.copy()
    d[m, 2] += a1[m] - target[m]
    return d, int(m.sum())


def keep_pose(path, log=print):
    """After placing a composed face on a head (retarget_head visage): each vertex morph keeps
    the inner parts behind the skin (keep), as in the composed face. Placement rotates skin
    and parts each by its own local rotation and smooths the skin near the seam: Jane's tongue
    under a Kanon base stuck out 0.027 in front of the lip in □ on Belle, clean in the composed
    face.
    And a surface part (eyebrow, lashes, blush) that the morph leaves in place above moving
    skin (check criterion: FROZEN, SKIN_MOVES) follows that skin: on a Kanon base, Jane's
    eyebrow, placed lower, covered the eyelid that Jane's 眼角下 lowers.
    Rewrites the PMX; returns {morph: vertices pushed back or re-attached}."""
    import pmx_write as pmx_write
    Mo = Model(path=path)
    ids = [i for i, n in enumerate(Mo.names_mat) if n in SKINS]
    F = Mo.t["faces"][np.isin(Mo.t["mat_face"], ids)]
    vi = Mo.vertices(INTERIOR)
    vs = Mo.vertices(SURFACE)
    vp = Mo.vertices(SKINS)
    rep, inv = G.weld(Mo.pos[vp])
    s_of = np.full(Mo.nv, -1)
    s_of[vp] = inv
    Fs = s_of[F]
    Fs = Fs[(Fs[:, 0] != Fs[:, 1]) & (Fs[:, 1] != Fs[:, 2]) & (Fs[:, 0] != Fs[:, 2])]
    if len(vs):
        _, fs_, bs_, _ = G.plus_near(Mo.pos[vs], Mo.pos[vp[rep]], Fs)
    out, glued = {}, {}
    for m in Mo.pmx["morphs"]:
        if m["type"] != 1:
            continue
        D = Mo.displacement(m["name"])
        N = D.copy()
        k = 0
        if len(vi) and np.abs(D[vi]).max() >= THRESHOLD:
            N[vi], k = keep(Mo.pos, Mo.pos + D, F, Mo.pos[vi], D[vi])
        if len(vs):
            ds = np.zeros((len(rep), 3))
            np.add.at(ds, inv, D[vp])
            ds /= np.maximum(np.bincount(inv, minlength=len(rep)), 1)[:, None]
            under = (bs_[:, :, None] * ds[Fs[fs_]]).sum(1)
            sk = np.linalg.norm(under, axis=1)
            pmax = float(np.linalg.norm(ds, axis=1).max()) if len(ds) else 0.0
            fig = (sk > max(THRESHOLD, SKIN_MOVES * pmax)) & (np.linalg.norm(D[vs], axis=1) < FROZEN * sk)
            if fig.any():
                N[vs[fig]] = under[fig]
                glued[m["name"]] = int(fig.sum())
        ch = np.where(np.abs(N - D).max(1) > 0)[0]
        if not len(ch):
            continue
        offset = {int(v): np.asarray(x, float) for v, x in m["offsets"]}
        for v in ch:
            offset[int(v)] = offset.get(int(v), np.zeros(3)) + (N[v] - D[v])
        m["offsets"] = [(v, tuple(float(x) for x in x3)) for v, x3 in sorted(offset.items())]
        if k:
            out[m["name"]] = k
    if out or glued:
        tmp = path + ".tmp.pmx"
        pmx_write.write(Mo.pmx, tmp)
        os.replace(tmp, path)
        log("  inner parts kept behind the skin after placement: %s" % out)
        log("  surface parts re-attached to the skin after placement: %s" % glued)
    return {**out, **{n: out.get(n, 0) + c for n, c in glued.items()}}


# =================================================================== check

def _progress(V, F, pts):
    """Lead of each point in front of the skin (V, F), seen from the front (the face looks
    towards -Z): z of the frontmost skin at the same (x, y) minus z of the point; -inf where no
    skin covers the point (inside the eye or mouth opening)."""
    T = V[F]
    n = G.normals_faces(V, F)
    T = T[n[:, 2] < 0]
    out = np.full(len(pts), -np.inf)
    a, b, c = T[:, 0, :2], T[:, 1, :2], T[:, 2, :2]
    v0, v1 = b - a, c - a
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(den) > 1e-12
    a, v0, v1, den, T = a[ok], v0[ok], v1[ok], den[ok], T[ok]
    for s in range(0, len(pts), 256):
        q = pts[s:s + 256]
        w = q[:, None, :2] - a[None]
        l1 = (w[..., 0] * v1[None, :, 1] - w[..., 1] * v1[None, :, 0]) / den[None]
        l2 = (v0[None, :, 0] * w[..., 1] - v0[None, :, 1] * w[..., 0]) / den[None]
        inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        z = T[None, :, 0, 2] + l1 * (T[None, :, 1, 2] - T[None, :, 0, 2]) + l2 * (T[None, :, 2, 2] - T[None, :, 0, 2])
        z = np.where(inside, z, np.inf).min(1)
        out[s:s + 256] = np.where(np.isfinite(z), z - q[:, 2], -np.inf)
    return out


def check(Mo, names=None, donors=None):
    """For each morph: max displacement per face material, frozen parts, parts that pierce
    the skin, empty (donors: name -> expected {material: amplitude})."""
    names = names or Mo.morphs_face()
    vp = Mo.vertices(SKINS)
    ids_skin = [i for i, n in enumerate(Mo.names_mat) if n in SKINS]
    Fp = Mo.t["faces"][np.isin(Mo.t["mat_face"], ids_skin)]
    # welded skin (a stitched skin has duplicates)
    rep, inv = G.weld(Mo.pos[vp])
    Vs0 = Mo.pos[vp[rep]]
    s_of = np.full(Mo.nv, -1)
    s_of[vp] = inv
    Fs = s_of[Fp]
    Fs = Fs[(Fs[:, 0] != Fs[:, 1]) & (Fs[:, 1] != Fs[:, 2]) & (Fs[:, 0] != Fs[:, 2])]
    parts = {}
    for i, n in enumerate(Mo.names_mat):
        if n in SURFACE + INTERIOR:
            v = np.unique(Mo.t["faces"][Mo.t["mat_face"] == i])
            if len(v):
                parts[i] = v
    # face only: parts close to the face skin
    allv = np.concatenate(list(parts.values())) if parts else np.zeros(0, int)
    # at rest: signed distance of the inner parts
    rest_pose = {}
    for i, v in parts.items():
        if Mo.names_mat[i] in INTERIOR:
            rest_pose[i] = _progress(Vs0, Fs, Mo.pos[v])
    # skin under each part: closest point on the welded skin (face, barycentrics)
    nearest = {i: G.plus_near(Mo.pos[v], Vs0, Fs)[1:3] for i, v in parts.items()}
    lines = []
    for name in names:
        D = Mo.displacement(name)
        a = np.linalg.norm(D, axis=1)
        ds = np.zeros((len(Vs0), 3))
        np.add.at(ds, inv, D[vp])
        ds /= np.maximum(np.bincount(inv, minlength=len(Vs0)), 1)[:, None]
        L = {"morph": name, "type": Mo.pmx["morphs"][Mo.index[name]]["type"] if name in Mo.index else None,
             "materials": {}, "figees": [], "percent": [], "vides": []}
        L["materials"]["skin"] = round(float(np.linalg.norm(ds, axis=1).max()), 4) if len(ds) else 0.0
        for i, v in parts.items():
            n = Mo.names_mat[i]
            the_key = n + ("#%d" % i if sum(1 for x in parts if Mo.names_mat[x] == n) > 1 else "")
            L["materials"][the_key] = round(float(a[v].max()), 4)
            if n in SURFACE:
                f, b = nearest[i]
                sk = np.linalg.norm((b[:, :, None] * ds[Fs[f]]).sum(1), axis=1)
                moves = sk > max(THRESHOLD, SKIN_MOVES * L["materials"]["skin"])
                if moves.sum() >= max(3, 0.05 * len(v)):
                    fig = moves & (a[v] < FROZEN * sk)
                    if fig.sum() > PART_FROZEN * moves.sum():
                        L["figees"].append({"part_name": the_key, "vertices": int(fig.sum()), "on": int(moves.sum()),
                                            "skin": round(float(sk[moves].mean()), 4),
                                            "part_moves": round(float(a[v][moves].mean()), 4)})
            elif n in INTERIOR and a.max() > THRESHOLD:
                sd = _progress(Vs0 + ds, Fs, Mo.pos[v] + D[v])
                perc = (sd > PIERCED) & (rest_pose[i] <= PIERCED)
                if perc.sum() >= 3:
                    L["percent"].append({"part_name": the_key, "vertices": int(perc.sum()),
                                         "progress_max": round(float(sd[perc].max()), 4)})
        if donors and name in donors:
            for the_key, amp in donors[name].items():
                got = max([v for k, v in L["materials"].items() if k.split("#")[0] == the_key] or [0.0])
                if amp > THRESHOLD and got < (WEAK if the_key == "skin" else 0.2) * amp:
                    L["vides"].append({"material": the_key, "donor": round(amp, 4), "here": got})
        lines.append(L)
    return lines


def amplitudes(Mo, names, mats):
    """Max amplitude per material (among mats) for each morph: name -> {mat: amp}."""
    out = {}
    vs = {n: Mo.vertices((n,)) for n in mats}
    for name in names:
        if name not in Mo.index:
            continue
        a = np.linalg.norm(Mo.displacement(name), axis=1)
        out[name] = {n: float(a[v].max()) for n, v in vs.items() if len(v)}
    return out


# =================================================================== composition

# parts that do not follow the skin when their donor lacks the morph: the iris and its highlights
FIXES = ("目", "目光", "目光2", "瞳光", "目色")


def weld_disp(T, D):
    """Displacement of the raw vertices -> welded skin of head T (mean)."""
    k = T.b2s
    m = k >= 0
    out = np.zeros((T.n, 3))
    np.add.at(out, k[m], D[m])
    cnt = np.bincount(k[m], minlength=T.n)
    return out / np.maximum(cnt, 1)[:, None]


def similarity(P, Q):
    """Least-squares similarity (s, R, t): Q ≈ s R P + t (Umeyama)."""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(Y.T @ X / len(P))
    d = np.sign(np.linalg.det(U @ Vt))
    E = np.diag([1, 1, d])
    R = U @ E @ Vt
    s = float((S * np.diag(E)).sum() / max((X * X).sum() / len(P), 1e-18))
    return s, R, mq - s * R @ mp


def normals_faces(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def project_oriented(q, nq, V, F, cos=0.3):
    """Closest point of the surface (V, F) to each q among the faces oriented like nq
    (faces, barycentrics). With closed lips, the upper and lower lip touch: only the
    orientation tells which one is right."""
    _, f, b, _ = G.plus_near(q, V, F)
    nf = normals_faces(V, F)
    nq = nq / np.maximum(np.linalg.norm(nq, axis=1, keepdims=True), 1e-12)
    for i in np.where((nf[f] * nq).sum(1) < cos)[0]:
        ok = np.where(nf @ nq[i] >= cos)[0]
        if len(ok):
            _, fi, bi, _ = G.plus_near(q[i:i + 1], V, F[ok])
            f[i], b[i] = ok[fi[0]], bi[0]
    return f, b


def skin_common(Vd, Fd, pts, V, F, wear):
    """Skin under part vertices, on the donor (Vd, Fd) and here (V, F): the donor's point,
    carried here (porter(x, est_vecteur)), then projected onto the faces oriented the same way.
    Returns ((faces, bary) on the donor, (faces, bary) here)."""
    _, fd, bd, _ = G.plus_near(pts, Vd, Fd)
    q = wear((bd[:, :, None] * Vd[Fd[fd]]).sum(1), False)
    nq = wear(normals_faces(Vd, Fd)[fd], True)
    f, b = project_oriented(q, nq, V, F)
    return (Fd[fd], bd), (F[f], b)


class Donors:
    """Donors of a recipe: heads placed on the pivot, morphs evaluated on demand."""

    def __init__(self, lib, A):
        import library as BI
        self.BI = BI
        self.lib = os.path.expanduser(lib)
        self.A = A
        self.MoA = Model(pmx=A.pmx)
        self._d, self._disp, self._proj, self._parts = {}, {}, {}, {}

    def __getitem__(self, name):
        if name not in self._d:
            z, card = self.BI._load(self.lib, name)
            import head as TT
            D = TT.Head(card["pmx"])
            D.put(float(z["scale"]), z["c_src"], z["c_dst"])
            self._d[name] = (z, card, D, Model(path=card["pmx"], pmx=D.pmx))
        return self._d[name]

    def parts(self, name, p, stores=False):
        """Raw faces of the parts of region p of donor name: sorted here by
        library.ranger_pieces (current rule), or those sorted at extraction
        (stores=True: part.npz, to re-read a face composed before the current rule)."""
        z, _, D, _ = self[name]
        if stores:
            return z["parts_" + p]
        if name not in self._parts:
            self._parts[name] = self.BI.arrange_parts(D)
        return self._parts[name][p]

    def disp(self, name, n):
        """Displacement (donor's raw vertices, original frame); None if it lacks the morph."""
        if (name, n) not in self._disp:
            Mo = self[name][3] if name is not None else self.MoA
            d = Mo.displacement(n) if n in Mo.index else None
            self._disp[(name, n)] = d if d is not None and np.abs(d).max() > 1e-6 else None
        return self._disp[(name, n)]

    def a(self, name, n):
        return self.disp(name, n) is not None

    def names_face(self, name):
        return self[name][3].morphs_face()

    def _projection(self, name):
        """Face and barycentrics, on the placed donor, of each registered pivot vertex,
        each point keeping its side around the openings (like extract_one)."""
        if name not in self._proj:
            z, _, D, _ = self[name]
            A, BI = self.A, self.BI
            X = z["X"].astype(float)
            XD = D._host(D.V)
            _, face, bary, _ = G.plus_near(X, XD, D.F)
            cA = BI.sides(A, BI.SIDE_RADIUS * A.iod)
            cD = BI.sides(D, BI.SIDE_RADIUS * D.iod)
            cF = np.sign(cD[D.F].astype(int).sum(1))
            for sg in (1, -1):
                qa = np.where(cA == sg)[0]
                fd = np.where(cF != -sg)[0]
                if len(qa) and len(fd):
                    _, f2, b2, _ = G.plus_near(X[qa], XD, D.F[fd])
                    face[qa], bary[qa] = fd[f2], b2
            self._proj[name] = (face, bary)
        return self._proj[name]

    def skin(self, name, n):
        """Offset (A.n, 3) of the pivot skin for the donor's morph n, at pivot scale: the
        library's one if present (vertex morph), otherwise sampled here."""
        z, _, D, Mo = self[name]
        names = list(z["names"])
        m = Mo.pmx["morphs"][Mo.index[n]] if n in Mo.index else None
        if n in names and m is not None and m["type"] == 1:
            return z["delta"][names.index(n)].astype(float)
        d = self.disp(name, n)
        if d is None:
            return np.zeros((self.A.n, 3))
        face, bary = self._projection(name)
        ds = weld_disp(D, d)
        return float(z["scale"]) * (bary[:, :, None] * ds[D.F[face]]).sum(1)

    def skin_pivot(self, n):
        d = self.disp(None, n)
        return np.zeros((self.A.n, 3)) if d is None else weld_disp(self.A, d)


def delta_composed(gifts, recipe, M, wv, n):
    """Offset of the composed skin (pivot frame) for morph n: the donor of each region if it
    has the morph, else the base, else the pivot; outside the face, the pivot.
    The fallback does not cross from one game to another (pivot: the active game, ZZZ): an
    Outerplane region whose donor lacks the morph stays at rest rather than take the eyelid or
    cheek of a ZZZ head, made for another eye (frozen lashes, highlights in front of the skin),
    and vice versa.
    A record without "jeu" is a head of the active game (its short name, description
    tools/games/<game>)."""
    if os.path.join(HERE, "..") not in sys.path:
        sys.path.append(os.path.join(HERE, ".."))
    import games as games
    J = games.active()
    PIVOT = J.value("abrege", J.title)
    BASE = "base"
    game = lambda name: gifts[name][1].get("game", PIVOT)  # noqa: E731
    dA = gifts.skin_pivot(n)
    zero = np.zeros_like(dA)
    nb = recipe[BASE]
    db = gifts.skin(nb, n) if gifts.a(nb, n) else (dA if game(nb) == PIVOT else zero)
    d = M[BASE][:, None] * db
    for zone, m in M.items():
        if zone == BASE:
            continue
        p = zone.split("_")[0]
        nd = recipe[p]
        d = d + m[:, None] * (gifts.skin(nd, n) if gifts.a(nd, n) else db if game(nd) == game(nb) else zero)
    return dA + wv[:, None] * (d - dA)


def names_recipe(gifts, recipe, M=None, wv=None):
    """Face morphs: the pivot's, then the donors' (order of first appearance).
    With M and wv: without those the recipe empties (empty_per_recipe)."""
    out = list(gifts.MoA.morphs_face())
    for nd in dict.fromkeys(recipe.values()):
        for n in gifts.names_face(nd):
            if n not in out and (M is None or not empty_per_recipe(gifts, recipe, M, wv, n)):
                out.append(n)
    return out


def empty_per_recipe(gifts, recipe, M, wv, n, pivot=False):
    """A donor's morph (absent from the pivot) whose composed skin keeps less than WEAK of
    its skin on its donors: it mostly moved a region taken from elsewhere, from another game,
    that lacks it (Kanon's MOUTH_SHY on a Jane mouth: 0.017 composed vs 0.084 on Kanon, less
    than 0.01 once placed). Such a morph no longer shows its expression: it is not written,
    rather than leave a still mouth under moving teeth.
    pivot=True: also applies to a pivot morph (always written, but emptied the same way)."""
    if not pivot and gifts.a(None, n):
        return False
    ref = max([float(np.linalg.norm(wv[:, None] * gifts.skin(nd, n), axis=1).max())
               for nd in dict.fromkeys(recipe.values()) if gifts.a(nd, n)] or [0.0])
    d = delta_composed(gifts, recipe, M, wv, n)
    return ref > THRESHOLD and float(np.linalg.norm(d, axis=1).max()) < WEAK * ref


def panel_of(gifts, recipe, n):
    for Mo in [gifts.MoA] + [gifts[nd][3] for nd in dict.fromkeys(recipe.values())]:
        if n in Mo.index:
            return Mo.pmx["morphs"][Mo.index[n]]["panel"], Mo.pmx["morphs"][Mo.index[n]]["name_en"]
    return 4, n


def parts_order(gifts, recipe, stores=False):
    """Composed parts, in the order _write_composed adds them: (region, donor,
    donor material, donor faces)."""
    out = []
    for p in ("eyes", "eyebrows", "mouth", "base"):
        nd = recipe[p]
        z, _, D, _ = gifts[nd]
        fs = gifts.parts(nd, p, stores)
        for i in np.unique(D.mat_face[fs]) if len(fs) else []:
            out.append((p, nd, int(i), fs[D.mat_face[fs] == i]))
    return out


def write_morph(P, n, offset, panel, name_en):
    """Morph n becomes (or is created as) a vertex morph with offsets dec {vertex: d}.
    A morph of another kind with the same name is replaced in place (no duplicate)."""
    offset = [(int(v), tuple(float(x) for x in d)) for v, d in sorted(offset.items()) if np.abs(d).max() > 1e-7]
    for m in P["morphs"]:
        if m["name"] == n:
            m["type"] = 1
            m["offsets"] = offset
            return False
    P["morphs"].append({"name": n, "name_en": name_en, "panel": panel, "type": 1, "offsets": offset})
    k = len(P["morphs"]) - 1
    for c in P["frames"]:
        if c["name"] == "表情":
            c["elements"] = list(c["elements"]) + [(1, k)]
            break
    return True


# =================================================================== repairing a finished PMX

class Done:
    """A finished PMX (composed face placed by retarget_head visage, then hairstyle, colours)
    read back against its recipe: which vertices come from which part of which donor, and how
    the pivot frame is carried onto it."""

    def __init__(self, path, recipe_json, registration=None):
        import glob
        import library as BI
        import retarget_head as RT
        self.path = path
        with open(recipe_json, encoding="utf-8") as fh:
            R = json.load(fh)
        self.recipe = R["recipe"]
        A = BI.load_pivot()
        self.A = A
        self.M = BI.masks(A)
        self.gifts = Donors(R["library"], A)
        npz = {n: self.gifts[n][0] for n in set(self.recipe.values())}
        Xf, _, self.wv = BI.shape_composed(A, self.M, npz, self.recipe)
        self.P = PMX.read(path)
        self.Mo = Model(path=path, pmx=self.P)
        pos = self.Mo.pos
        # weld order of the composed face (it depends on positions) -> pivot order
        vb = A.skin_raw
        kp = A.b2s[vb]
        _, inv_c = G.weld(Xf[kp])
        self.kp_of_kc = np.zeros(inv_c.max() + 1, dtype=np.int64)
        self.kp_of_kc[inv_c] = kp
        # registration of the retarget_head visage pass: the one whose X lands on the 颜_A skin
        va = self.Mo.vertices(("颜_A",))
        candidates = [registration] if registration else sorted(glob.glob(os.path.join(os.path.dirname(recipe_json), "on-*", "registration.npz"))
                                                                 + glob.glob(os.path.join(os.path.dirname(recipe_json), "sur-*", "registration.npz")))   # fr-ok: older trial folders on disk
        self.rec = None
        for c in candidates:
            z = np.load(c)
            if len(z["X"]) != len(self.kp_of_kc):
                continue
            d, j = _near(pos[va], z["X"])
            if d.max() < 1e-5:
                self.rec, self.path_rec = z, c
                break
        if self.rec is None:
            raise RuntimeError("no registration matches the 颜_A skin of %s" % path)
        X, XA = self.rec["X"], self.rec["XA"]
        # seam band: copies of the patch border, with B's material, at the positions of X
        vB = self.Mo.vertices(("颜",))
        dB, jB = _near(pos[vB], X)
        _, ja = _near(pos[va], X)
        self.skin_ids = np.concatenate([va, vB[dB < 1e-5]])
        self.skin_kc = np.concatenate([ja, jB[dB < 1e-5]])
        self.seeds = vB[dB < 1e-5]
        ids = [i for i, n in enumerate(self.Mo.names_mat) if n in SKINS]
        self.Fpeau = self.Mo.t["faces"][np.isin(self.Mo.t["mat_face"], ids)]
        # scale and local rotations of the face pass (like retarget_head)
        Xc = Xf[self.kp_of_kc]
        self.s = float(np.sqrt(((XA - XA.mean(0)) ** 2).sum() / ((Xc - Xc.mean(0)) ** 2).sum()))
        self.residue_place = float(np.abs(self.s * (Xc - Xc.mean(0)) + XA.mean(0) - XA).max())
        self.Rv = RT.rotations_local(XA, X, self.rec["FA"])
        # parts: materials "name (Donor)" in composition order; current sorting, otherwise
        # the extraction one (face composed earlier); an emptied material is a removed part
        # (remove_parts)
        mats = [i for i, m in enumerate(self.P["materials"]) if (m["name_en"] or "").endswith(")") and "(" in m["name_en"]
                and m["name"] not in SKINS]
        F, mf = self.Mo.t["faces"], self.Mo.t["mat_face"]
        nf = np.bincount(mf, minlength=len(self.P["materials"]))

        def kept(p, nd, fs):
            return fs[np.isin(fs, self.gifts.parts(nd, p))]

        def matched(att):
            return len(att) == len(mats) and all(
                self.P["materials"][j]["name_en"].endswith("(%s)" % nd)
                and nf[j] in (0, len(fs), len(kept(p, nd, fs)))
                for j, (p, nd, i, fs) in zip(mats, att))
        expected = None
        for stores in (False, True):
            att = parts_order(self.gifts, self.recipe, stores)
            if matched(att):
                expected, self.arrangement = att, ("extraction" if stores else "current")
                break
        if expected is None:
            raise RuntimeError("parts: the %d donor materials of %s match no sorting of the recipe"
                               % (len(mats), path))
        self.parts = []  # (final material, region, donor, final vertices, donor vertices, scale, rotation)
        for j, (p, nd, i, fs) in zip(mats, expected):
            D = self.gifts[nd][2]
            Ff = F[mf == j]
            if not len(Ff):
                continue
            if len(Ff) != len(fs):
                fs = kept(p, nd, fs)
            vf, idx = np.unique(Ff.ravel(), return_index=True)
            vd = D.faces[fs].ravel()[idx]
            Pd, Pf = D.pos[vd], pos[vf]
            sR = np.zeros((len(vf), 3, 3))
            tR = np.zeros((len(vf), 3))
            # similarity per side (eyes, eyebrows), in one piece elsewhere
            sides = [Pd[:, 0] > 0, Pd[:, 0] <= 0] if p in ("eyes", "eyebrows") else [np.ones(len(vf), bool)]
            gap = 0.0
            for c in sides:
                if c.sum() < 3:
                    c = np.ones(len(vf), bool)
                sc, Rc, tc = similarity(Pd[c], Pf[c])
                sR[c] = sc * Rc
                tR[c] = tc
                gap = max(gap, float(np.linalg.norm(sc * Pd[c] @ Rc.T + tc - Pf[c], axis=1).max()))
            self.parts.append({"mat": j, "name": self.P["materials"][j]["name"], "region": p, "donor": nd,
                                "fs": fs, "vf": vf, "vd": vd, "sR": sR, "tR": tR, "gap": gap})

    def skin_pivot_to_done(self, dp):
        """Offset of the composed skin (pivot frame, pivot order) -> {finished vertex: d}."""
        kc = self.skin_kc
        d = self.s * np.einsum("nij,nj->ni", self.Rv[kc], dp[self.kp_of_kc[kc]])
        return d

    def _under(self, pc):
        if "under" not in pc:
            Dt = self.gifts[pc["donor"]][2]
            sR, tR = pc["sR"], pc["tR"]
            wear = lambda x, vec: np.einsum("nij,nj->ni", sR, x) + (0 if vec else tR)
            pc["under_d"], pc["under"] = skin_common(Dt.pos, Dt.faces[Dt.skin_faces_raw],
                                                    Dt.pos[pc["vd"]], self.Mo.pos, self.Fpeau, wear)
        return pc

    def skin_under(self, pc, D):
        """Offset of the finished skin (displacement D of all vertices) under each vertex of
        the part, at the point matching the donor's (skin_common)."""
        F, b = self._under(pc)["under"]
        return (b[:, :, None] * D[F]).sum(1)

    def skin_under_donor(self, pc, n):
        """Offset of the donor's skin under each vertex of the part, on the donor, carried here."""
        d = self.gifts.disp(pc["donor"], n)
        if d is None:
            return np.zeros((len(pc["vf"]), 3))
        F, b = self._under(pc)["under_d"]
        return np.einsum("nij,nj->ni", pc["sR"], (b[:, :, None] * d[F]).sum(1))

    def part_donor(self, pc, n):
        d = self.gifts.disp(pc["donor"], n)
        if d is None:
            return None
        return np.einsum("nij,nj->ni", pc["sR"], d[pc["vd"]])


def _near(Q, X):
    """Distance and index of the point of X closest to each point of Q."""
    d = np.zeros(len(Q))
    j = np.zeros(len(Q), dtype=np.int64)
    for s in range(0, len(Q), 1024):
        dd = np.linalg.norm(Q[s:s + 1024][:, None] - X[None], axis=2)
        j[s:s + 1024] = dd.argmin(1)
        d[s:s + 1024] = dd.min(1)
    return d, j


def repair(finished, log=print):
    """Completes the face morphs of a finished PMX: the composed skin and each part, in a morph
    that does not move them, receive their motion (the region's donor, carried by the local
    rotation and scale; otherwise the part follows the skin). What already moves is kept as is;
    the rest of the PMX is not touched. Returns a report per morph."""
    import retarget_head as RT
    Fi, P, Mo, gifts = finished, finished.P, finished.Mo, finished.gifts
    names = names_recipe(gifts, Fi.recipe, Fi.M, Fi.wv)
    for n in Mo.morphs_face():
        if n not in names:
            names.append(n)
    report, new_skin = {}, []
    for n in names:
        D = Mo.displacement(n)
        a = np.linalg.norm(D, axis=1)
        before = D.copy()
        r = {"skin": None, "parts": {}}
        # composed skin
        dp = delta_composed(gifts, Fi.recipe, Fi.M, Fi.wv, n)
        # the face skin receives the composed skin if it does not move, or moves much less
        # (Orpheya's ワ２: 0.025 at the seam edge, a morph of the target head, vs 0.073 of
        # composed skin: without it, Belle's parts moved on a still skin)
        if np.abs(dp).max() > 1e-6:
            T = Fi.skin_pivot_to_done(dp)
            if a[Fi.skin_ids].max() < max(THRESHOLD, WEAK * np.linalg.norm(T, axis=1).max()):
                D[Fi.skin_ids] = T
                r["skin"] = "composed"
                new_skin.append(n)
        # parts: donor, otherwise the skin
        for pc in Fi.parts:
            if a[pc["vf"]].max() >= THRESHOLD:
                continue
            # donor motion, plus the gap between the skin under the part here and the skin under
            # the part on the donor (the whole gap if the donor lacks the morph: the part
            # follows the skin): the part's motion relative to its skin is the donor's
            d = Fi.part_donor(pc, n)
            what = "donor"
            if pc["name"] not in FIXES:
                e = Fi.skin_under(pc, D) - Fi.skin_under_donor(pc, n)
                if np.linalg.norm(e, axis=1).max() >= THRESHOLD:
                    what = "follows the skin" if d is None else "donor + skin gap"
                d = e if d is None else d + e
            if d is not None and pc["name"] in INTERIOR:
                p0 = Fi.Mo.pos[pc["vf"]]
                d, k = keep(Fi.Mo.pos, Fi.Mo.pos + D, Fi.Fpeau, p0, d)
                if k:
                    what += ", %d pushed back" % k
            if d is not None and np.linalg.norm(d, axis=1).max() >= THRESHOLD:
                D[pc["vf"]] = d
                r["parts"][pc["name"] + "/" + pc["donor"]] = what
        if r["skin"] is None and not r["parts"]:
            continue
        ch = np.where(np.abs(D - before).max(1) > 1e-7)[0]
        m0 = P["morphs"][Mo.index[n]] if n in Mo.index else None
        if m0 is not None and m0["type"] == 1:
            offset = {int(v): np.asarray(d) for v, d in m0["offsets"]}
            for v in ch:
                offset[int(v)] = D[v]
        else:
            # other kind (bone, group) or absent: vertex morph, its current effect kept
            offset = {int(v): D[v] for v in np.where(np.abs(D).max(1) > 1e-7)[0]}
        panel, name_en = (m0["panel"], m0["name_en"]) if m0 is not None else panel_of(gifts, Fi.recipe, n)
        r["cree"] = write_morph(P, n, offset, panel, name_en)
        r["type_front"] = None if m0 is None else m0["type"]
        report[n] = r
        log("  %-8s %s%s %s" % (n, "created " if r["cree"] else "", "composed skin" if r["skin"] else "",
                                    ", ".join("%s %s" % kv for kv in r["parts"].items())))
    # seam: offsets harmonised near the band for morphs whose skin is new
    if new_skin and len(Fi.seeds):
        Q = {"vertices": P["vertices"], "faces": P["faces"], "materials": P["materials"],
             "morphs": [m for m in P["morphs"] if m["name"] in new_skin and m["type"] == 1]}
        ids = [i for i, x in enumerate(P["materials"]) if x["name"] in SKINS]
        RT.harmonize_morphs(Q, ids, Fi.seeds)
    return report


def remove_parts(finished, log=print):
    """Parts that a composed face got from an older extraction sorting and that the current
    sorting (library.ranger_pieces) no longer gives to their region: their faces are
    removed (the material stays, empty; vertices, bones and morphs stay: nothing moves at rest
    elsewhere). Those the current sorting would add are not added (recomposing is needed):
    they are only named. Returns the report."""
    Fi, P = finished, finished.P
    F, mf = Fi.Mo.t["faces"], Fi.Mo.t["mat_face"]
    guard = np.ones(len(F), bool)
    report = {"removed": [], "missing": []}
    for pc in Fi.parts:
        fresh = Fi.gifts.parts(pc["donor"], pc["region"])
        out_of = ~np.isin(pc["fs"], fresh)
        if out_of.any():
            ids = np.where(mf == pc["mat"])[0]
            guard[ids[out_of]] = False
            report["removed"].append({"part_name": pc["name"], "donor": pc["donor"], "region": pc["region"],
                                        "faces": int(out_of.sum()), "on": int(len(out_of))})
            log("  removed: %s (%s, %s) %d faces of %d" % (pc["name"], pc["donor"], pc["region"], out_of.sum(), len(out_of)))
    for p, nd, i, fs in parts_order(Fi.gifts, Fi.recipe):
        before = [pc for pc in Fi.parts if pc["region"] == p and pc["donor"] == nd]
        in_plus = fs[~np.isin(fs, np.concatenate([pc["fs"] for pc in before]) if before else [])]
        if len(in_plus):
            D = Fi.gifts[nd][2]
            report["missing"].append({"part_name": D.names_mat[i], "donor": nd, "region": p, "faces": int(len(in_plus))})
            log("  missing (recompose to get it): %s (%s, %s) %d faces" % (D.names_mat[i], nd, p, len(in_plus)))
    if not guard.all():
        n = np.bincount(mf[guard], minlength=len(P["materials"]))
        for k, m in enumerate(P["materials"]):
            m["n_indices"] = int(3 * n[k])
        P["faces"] = [tuple(f) for f in F[guard].tolist()]
    return report


def expected(finished, names):
    """Expected amplitude per material (carried donor parts, composed skin)."""
    out = {}
    for n in names:
        e = {}
        for pc in finished.parts:
            d = finished.part_donor(pc, n)
            if d is not None:
                e[pc["name"]] = max(e.get(pc["name"], 0.0), float(np.linalg.norm(d, axis=1).max()))
        dp = delta_composed(finished.gifts, finished.recipe, finished.M, finished.wv, n)
        # morph emptied by the recipe (ウィンク２ on Kanon eyes, whose smile only moves the
        # eyebrows): only a remnant at the border is left, nothing to expect from the skin
        empty = empty_per_recipe(finished.gifts, finished.recipe, finished.M, finished.wv, n, pivot=True)
        e["skin"] = 0.0 if empty else float(np.linalg.norm(finished.skin_pivot_to_done(dp), axis=1).max())
        out[n] = e
    return out


def display(lines):
    for L in lines:
        mat = " ".join("%s %.3f" % (k, v) for k, v in L["materials"].items() if v > THRESHOLD)
        sheet = []
        if L["vides"]:
            sheet.append("EMPTY " + ",".join(x["material"] for x in L["vides"]))
        if L["figees"]:
            sheet.append("FROZEN " + ",".join("%s(%d/%d)" % (x["part_name"], x["vertices"], x["on"]) for x in L["figees"]))
        if L["percent"]:
            sheet.append("PIERCES " + ",".join("%s(%d, %.3f)" % (x["part_name"], x["vertices"], x["progress_max"])
                                             for x in L["percent"]))
        print("%-8s t%s  %-60s %s" % (L["morph"], L["type"], mat or "(nothing moves)", "  ".join(sheet)))
    n = sum(1 for L in lines if L["vides"] or L["figees"] or L["percent"])
    print("check: %d morphs, %d with defects" % (len(lines), n))
    return n


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------
# eyebrow reshaping


class Eyebrow:
    """One eyebrow (one side) in its own frame: a along the eyebrow (outwards), b upwards,
    c in depth. The deformation happens in the (a, b) plane; depth follows the forehead skin
    (surface c = Q(a, b) fitted to the surrounding skin), so the eyebrow stays laid on it when
    it rises, arches or stretches towards the temple."""

    def __init__(self, pts, skin, side):
        self.c0 = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - self.c0, full_matrices=False)
        u, v = Vt[0], Vt[1]
        if u[0] * side < 0:
            u = -u
        if v[1] < 0:
            v = -v
        v = v - u * (u @ v)
        v /= np.linalg.norm(v)
        self.B = np.stack([u, v, np.cross(u, v)])  # lines : a, b, c
        L = self.local(pts)
        self.L = float(np.abs(L[:, 0]).max())
        # eyebrow centre line b = m(a) (its current arch): thickness is measured around it
        self.m = np.polyfit(L[:, 0], L[:, 1], 2)
        # surrounding skin: c = Q(a, b)
        Lp = self.local(skin)
        # forehead band around the eyebrow (not the eye socket below)
        bm = np.polyval(self.m, np.clip(Lp[:, 0], -self.L, self.L))
        sel = (np.abs(Lp[:, 0]) < 1.6 * self.L) & (np.abs(Lp[:, 1] - bm) < 0.35 * self.L) & (np.abs(Lp[:, 2]) < 0.4 * self.L)
        Lp = Lp[sel]
        self.q = np.linalg.lstsq(self._base(Lp[:, 0], Lp[:, 1]), Lp[:, 2], rcond=None)[0]
        self.n_skin = int(sel.sum())

    def local(self, p):
        return (p - self.c0) @ self.B.T

    def world(self, L):
        return L @ self.B + self.c0

    def _base(self, a, b):
        a, b = a / self.L, b / self.L
        # forehead: curved along the eyebrow (towards the temple), almost flat vertically
        return np.stack([np.ones_like(a), a, b, a * a, a * b, a ** 3], 1)

    def Q(self, a, b):
        return self._base(a, b) @ self.q

    def f(self, p, length, thickness, arc, rise, tilt):
        L = self.local(p)
        a, b, c = L[:, 0], L[:, 1], L[:, 2]
        m = np.polyval(self.m, a)
        a1 = length * a
        # the arch raises the middle, slightly outwards; the ends stay
        s = (a - 0.2 * self.L) / (1.2 * self.L)
        b1 = np.polyval(self.m, a) + thickness * (b - m) + arc * self.L * (1 - s * s) + rise * self.L
        t = np.radians(tilt)
        a2 = np.cos(t) * a1 - np.sin(t) * b1
        b2 = np.sin(t) * a1 + np.cos(t) * b1
        c2 = c + self.Q(a2, b2) - self.Q(a, b)
        return self.world(np.stack([a2, b2, c2], 1))


# ---- gap between eyebrow and eye line (front view)

# eye lines: lashes and eyelid creases (a part of these materials sorted as "eyebrows" is
# an eyebrow)
TRAITS = ("睫", "睫眉", "眉睫", "右睫眉", "二重")
INK_ALPHA = 0.5   # visible texel: texture alpha x diffuse alpha above this
# Minimal eyebrow - eye line gap (in interocular distances, front view), in any morph:
# median, over 48 original ZZZ models, of each one's smallest gap across its morphs (0.0453;
# p25 0.004: 13 models touch, mostly in 下 and 怒り; median at rest 0.191). Measured on
# 07/11 (trait_principal, ink).
GAP_EYEBROW_MIN = 0.045
INK_CONTRAST = 0.25  # ink texel: further than this (RGB 0-1) from the skin colour
INK_STEP = 10      # points per side of each triangle to sample the ink


def find_texture(path_pmx, rel, folders=()):
    """File of a PMX texture: relative to its folder, otherwise to one of the folders."""
    rel = rel.replace("\\", "/")
    for d in [os.path.dirname(os.path.abspath(path_pmx))] + list(folders):
        f = os.path.normpath(os.path.join(d, rel))
        if os.path.exists(f):
            return f
    # relative path written from another folder (an agent's $TMPDIR), then copied: the
    # part after the ".." is often the absolute path
    f = "/" + "/".join(x for x in rel.split("/") if x not in ("..", ".", ""))
    return f if os.path.exists(f) else None


def textures_relative(P, path_pmx, folder, folders=(), log=print):
    """Texture paths of P rewritten relative to dossier (where the PMX will be placed), each
    first located from path_pmx (find_texture). Returns the number of changed paths."""
    n = 0
    for k, t in enumerate(P["textures"]):
        f = find_texture(path_pmx, t, folders)
        if f is None:
            log("  texture not found, path kept: %s" % t)
            continue
        r = os.path.relpath(f, folder).replace("/", "\\")
        if r != t:
            P["textures"][k] = r
            n += 1
    return n


def _texture(Mo, m, folders, cache, log):
    P = Mo.pmx
    t = P["materials"][m]["texture"]
    if not 0 <= t < len(P["textures"]):
        return None
    f = find_texture(Mo.path or "", P["textures"][t], folders)
    if f is None:
        log("  ink: texture %s not found (material %s)" % (P["textures"][t], P["materials"][m]["name"]))
        return None
    if f not in cache:
        import images as IM
        try:
            cache[f] = IM.read(f)
        except ValueError as e:
            log("  ink: %s" % e)
            cache[f] = None
    return cache[f]


def _texels(img, U):
    h, w = img.shape[:2]
    x = np.clip((U[..., 0] % 1.0) * w, 0, w - 1).astype(int)
    y = np.clip((U[..., 1] % 1.0) * h, 0, h - 1).astype(int)
    return img[y, x]


def ink(Mo, faces, folders=(), log=print):
    """Ink points of the faces (indices into Mo.t["faces"]): the points of a barycentric grid
    on each triangle where the texture is visible (alpha) and stands out from the skin (ZZZ
    lash and eyebrow textures are opaque: skin colour around the line). Returns
    (faces (k, 3) of vertices, barycentrics (k, 3)). Without a readable texture, the whole triangle."""
    P, F, mf = Mo.pmx, Mo.t["faces"], Mo.t["mat_face"]
    n = INK_STEP
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    k = (i + j) <= n
    B = np.stack([(n - i - j)[k], i[k], j[k]], 1) / n
    uv = np.asarray(P["vertices"]["uv"], float)
    cache = {}
    # skin colour: median of the skin texture at its vertices
    background = []
    for m, nm in enumerate(Mo.names_mat):
        if nm in SKINS:
            img = _texture(Mo, m, folders, cache, log)
            if img is not None:
                background.append(_texels(img, uv[np.unique(F[mf == m])])[:, :3])
    background = np.median(np.concatenate(background), 0) if background else None
    out_f, out_b = [], []
    for m in np.unique(mf[faces]):
        fm = faces[mf[faces] == m]
        img = _texture(Mo, m, folders, cache, log)
        if img is None:
            vu = np.ones((len(fm), len(B)), bool)
        else:
            c = _texels(img, np.einsum("pk,fkd->fpd", B, uv[F[fm]]))
            vu = c[..., 3] * P["materials"][m]["diffuse"][3] > INK_ALPHA
            if background is not None:
                vu &= np.linalg.norm(c[..., :3] - background, axis=-1) > INK_CONTRAST
        ff, pp = np.nonzero(vu)
        out_f.append(F[fm][ff])
        out_b.append(B[pp])
    if not out_f:
        return np.zeros((0, 3), np.int64), np.zeros((0, 3))
    return np.concatenate(out_f), np.concatenate(out_b)


def _points(V, E):
    Fe, Be = E
    return (Be[:, :, None] * V[Fe]).sum(1)


def gap_points(A, B, block=1024):
    """Smallest (x, y) distance between two point clouds: a bound from subsets, then only the
    points closer than this bound to the other cloud's bounding box."""
    if not len(A) or not len(B):
        return np.inf
    A, B = A[:, :2], B[:, :2]

    def raw(A, B):
        d = np.inf
        for s in range(0, len(A), block):
            q = A[s:s + block]
            d = min(d, float((((q[:, None] - B[None]) ** 2).sum(-1)).min()))
        return np.sqrt(d)
    u = raw(A[::max(1, len(A) // 200)], B[::max(1, len(B) // 2000)])
    A = A[np.all((A >= B.min(0) - u) & (A <= B.max(0) + u), 1)]
    B = B[np.all((B >= A.min(0) - u) & (B <= A.max(0) + u), 1)] if len(A) else B[:0]
    if not len(A) or not len(B):
        return u
    # each point of B: only if it is within u of a point of A (grid of step u)
    k = np.floor(A / max(u, 1e-9)).astype(np.int64)
    keys = set(map(tuple, k))
    kb = np.floor(B / max(u, 1e-9)).astype(np.int64)
    guard = np.array([any((x + i, y + j) in keys for i in (-1, 0, 1) for j in (-1, 0, 1)) for x, y in kb])
    return min(u, raw(A, B[guard])) if guard.any() else u


def _sides(V0, E):
    x = _points(V0, E)[:, 0]
    return {"left": x > 0, "right": x < 0}


def gaps_eyebrows(V0, Es, El, iod, deps):
    """Eyebrow - eye line gap on each side, front view, in interocular distances: smallest
    distance between the eyebrow ink (Es) and the eye line ink (El), for each displacement of
    deps {name: (nv, 3)}."""
    cs, cl = _sides(V0, Es), _sides(V0, El)
    out = {}
    for name, D in deps.items():
        V = V0 + D
        ps, pl = _points(V, Es), _points(V, El)
        out[name] = {c: gap_points(ps[cs[c]], pl[cl[c]]) / iod for c in cs}
    return out


def iod_globes(Mo):
    """Interocular distance: distance between the iris (目) centres on either side."""
    v = Mo.vertices(("目",))
    p = Mo.pos[v]
    return float(np.linalg.norm(p[p[:, 0] > 0].mean(0) - p[p[:, 0] < 0].mean(0)))


def trait_main(Mo, fl):
    """Among eye-line faces, the eye line proper: on each side, the largest connected piece
    (vertices welded by position), by area. The other pieces (eyelid crease 二重, lower lashes,
    corner tips) are discarded: on the originals the eyebrow touches the crease when angry
    (Belle, Ellen, Corin), never or almost never the line (Belle 怒り 0.049, Ellen 下 0.061,
    Miyabi 怒り 0.080 interocular distances)."""
    if not len(fl):
        return fl
    F = Mo.t["faces"][fl]
    vb, Fl = G.under_mesh(Mo.t["faces"], fl)
    _, inv = G.weld(Mo.pos[vb])
    lab = G.components_faces(inv[Fl], np.ones(len(Fl), bool))
    P3 = Mo.pos[F]
    area = 0.5 * np.linalg.norm(np.cross(P3[:, 1] - P3[:, 0], P3[:, 2] - P3[:, 0]), axis=1)
    x = P3[:, :, 0].mean(1)
    guard = np.zeros(len(fl), bool)
    for side in (x > 0, x < 0):
        if side.any():
            a = np.bincount(lab[side], area[side], lab.max() + 1)
            guard |= side & (lab == int(a.argmax()))
    return fl[guard]


def faces_eyebrows_traits(Mo):
    """Face indices of the eyebrows (眉, 眉毛) and of the eye line (TRAITS, trait_principal)
    of a composed face (one part per material)."""
    mf = Mo.t["mat_face"]
    ids_s = [i for i, n in enumerate(Mo.names_mat) if n in EYEBROWS]
    ids_l = [i for i, n in enumerate(Mo.names_mat) if n in TRAITS]
    return np.where(np.isin(mf, ids_s))[0], trait_main(Mo, np.where(np.isin(mf, ids_l))[0])


def measure_gaps(Mo, names=None, folders=(), faces=None):
    """Eyebrow - eye line gap (interocular distances) at rest and for each face morph that
    moves the eyebrows or the lines: {morph: {left, right}}. faces: (eyebrows, lines) as
    face indices, otherwise by material names."""
    if faces is None:
        fs, fl = faces_eyebrows_traits(Mo)
    else:
        fs, fl = faces[0], trait_main(Mo, faces[1])
    Es, El = ink(Mo, fs, folders), ink(Mo, fl, folders)
    vs = np.unique(np.concatenate([Es[0].ravel(), El[0].ravel()]))
    deps = {"rest": np.zeros_like(Mo.pos)}
    for n in names or Mo.morphs_face():
        D = Mo.displacement(n)
        if len(vs) and np.abs(D[vs]).max() > 1e-6:
            deps[n] = D
    return gaps_eyebrows(Mo.pos, Es, El, iod_globes(Mo), deps)


def keep_eyebrows(P, gap_min=None, path=None, folders=(), log=print):
    """Guard: in each face vertex morph, on each side, the eyebrow offset is scaled down
    (common factor t for the side, by bisection) so that their ink stays gap_min
    (interocular distances, front view) away from the eye line's ink, or at the gap it has
    without this offset if it is already closer. The lines and the rest of the face keep
    their offset. Returns {morph: {side: t}}."""
    gap_min = GAP_EYEBROW_MIN if gap_min is None else gap_min
    Mo = Model(path=path, pmx=P)
    fs, fl = faces_eyebrows_traits(Mo)
    if not len(fs) or not len(fl):
        return {}
    Es, El = ink(Mo, fs, folders, log), ink(Mo, fl, folders, log)
    iod = iod_globes(Mo)
    V0 = Mo.pos
    cs, cl = _sides(V0, Es), _sides(V0, El)
    # all eyebrow vertices of one side (not only those of inked triangles):
    # one common factor, so the eyebrow does not tear
    vsrc = np.unique(Mo.t["faces"][fs])
    vcote = {"left": vsrc[V0[vsrc, 0] > 0], "right": vsrc[V0[vsrc, 0] < 0]}
    out = {}
    for m in P["morphs"]:
        if m["type"] != 1 or m["panel"] not in (1, 2, 3):
            continue
        D = np.zeros_like(V0)
        for v, d in m["offsets"]:
            D[int(v)] += d
        pl = _points(V0 + D, El)
        for c in ("left", "right"):
            es = (Es[0][cs[c]], Es[1][cs[c]])
            vs = vcote[c]
            if not len(vs) or np.abs(D[vs]).max() < 1e-6:
                continue
            line = pl[cl[c]]

            Dt = D.copy()
            Dt[vs] = 0.0

            def gap(t):
                Dt[vs] = t * D[vs]
                return gap_points(_points(V0 + Dt, es), line) / iod
            e1 = gap(1.0)
            target = min(gap_min, gap(0.0))
            if e1 >= target:
                continue
            lo, hi = 0.0, 1.0
            for _ in range(25):
                mi = 0.5 * (lo + hi)
                lo, hi = (mi, hi) if gap(mi) >= target else (lo, mi)
            t = lo
            ens = set(int(v) for v in vs)
            m["offsets"] = [(v, tuple(t * x for x in d) if int(v) in ens else d) for v, d in m["offsets"]]
            out.setdefault(m["name"], {})[c] = round(t, 4)
            log("  guard %-6s %-6s: eyebrow offset x %.3f (gap %.4f -> %.4f)" % (m["name"], c, t, e1, gap(t)))
    return out


def deform_eyebrows(P, length=1.0, thickness=1.0, arc=0.0, rise=0.0, tilt=0.0, eps=1e-4):
    """Deforms the eyebrow vertices (眉, 眉毛) of PMX P in place: longueur (length factor, from
    the middle), thickness (thickness factor, around the centre line), arc (rise of the middle,
    as a fraction of the half-length), rise (raise, same unit), tilt (tilt in degrees,
    outer end upwards).
    Each morph offset d on an eyebrow vertex becomes J d, J being the local Jacobian of the
    deformation (central differences); normals go through J^-T. Returns the measurements,
    including the error |f(p+d) - f(p) - J d| over all morphs."""
    Mo = Model(pmx=P)
    vs = Mo.vertices(EYEBROWS)
    skin = Mo.pos[Mo.vertices(SKINS)]
    if not len(vs):
        raise RuntimeError("no eyebrows (%s)" % ", ".join(EYEBROWS))
    per = dict(length=length, thickness=thickness, arc=arc, rise=rise, tilt=tilt)
    S = P["vertices"]
    pos = np.array([S["pos"][v] for v in vs], float)
    nor = np.array([S["nor"][v] for v in vs], float)
    J = np.zeros((len(vs), 3, 3))
    new = np.zeros_like(pos)
    info = {"parameters": per, "sides": {}}
    sc = {}
    for side, name in ((1, "left"), (-1, "right")):
        k = np.where(pos[:, 0] * side > 0)[0]
        if len(k) < 4:
            continue
        so = Eyebrow(pos[k], skin, side)
        sc[name] = (so, k)
        new[k] = so.f(pos[k], **per)
        for j in range(3):
            e = np.zeros(3)
            e[j] = eps
            J[k, :, j] = (so.f(pos[k] + e, **per) - so.f(pos[k] - e, **per)) / (2 * eps)
        info["sides"][name] = {"vertices": int(len(k)), "half_length": so.L, "skin_ajustee": so.n_skin}
    # vertex morphs
    pos_of = {int(v): i for i, v in enumerate(vs)}
    side_of = np.zeros(len(vs), dtype=object)
    for name, (so, k) in sc.items():
        side_of[k] = name
    err, dmax, touched = 0.0, 0.0, []
    for m in P["morphs"]:
        if m["type"] != 1:
            continue
        offset, t = [], False
        for v, d in m["offsets"]:
            i = pos_of.get(int(v))
            if i is None:
                offset.append((v, d))
                continue
            d = np.asarray(d, float)
            d2 = J[i] @ d
            so = sc[side_of[i]][0]
            exact = so.f(pos[i:i + 1] + d, **per)[0] - new[i]
            err = max(err, float(np.linalg.norm(exact - d2)))
            dmax = max(dmax, float(np.linalg.norm(d)))
            offset.append((v, tuple(float(x) for x in d2)))
            t = True
        if t:
            m["offsets"] = offset
            touched.append(m["name"])
    # vertices and normals
    for i, v in enumerate(vs):
        n = np.linalg.solve(J[i].T, nor[i]) if abs(np.linalg.det(J[i])) > 1e-9 else nor[i]
        n /= max(np.linalg.norm(n), 1e-12)
        S["pos"][v] = tuple(float(x) for x in new[i])
        S["nor"][v] = tuple(float(x) for x in n)
    disp = np.linalg.norm(new - pos, axis=1)
    info.update({"vertices": int(len(vs)), "shift_max": float(disp.max()), "morphs": touched,
                 "offset_max": dmax, "gap_jacobienne": err})
    print("eyebrows: %d vertices, moved by %.3f at most; %d morphs carried by the Jacobian; "
          "error |f(p+d)-f(p)-J d| %.4f for an offset of %.3f" % (len(vs), disp.max(), len(touched), err, dmax))
    for name, c in info["sides"].items():
        print("  %s: %d vertices, half-length %.3f, skin fitted on %d vertices" % (name, c["vertices"], c["half_length"], c["skin_ajustee"]))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("check", "repair", "parts", "eyebrow", "guard", "gaps"))
    ap.add_argument("--pmx", required=True)
    ap.add_argument("--recipe", dest="recipe", help="recipe.json of the composed face (empty-morph check, repair, parts)")
    ap.add_argument("--realign", dest="registration", help="registration.npz of the face pass (otherwise searched in on-*/)")
    ap.add_argument("--output", dest="output")
    ap.add_argument("--json")
    ap.add_argument("--textures-for", dest="textures_for", help="folder where the PMX will be placed: texture paths rewritten relative to it "
                                            "(default: --output); textures also searched there (gaps, guard)")
    ap.add_argument("--min-gap", dest="gap_min", type=float, default=None,
                    help="eyebrow guard: minimal gap to the eye line (interocular distances; default GAP_EYEBROW_MIN)")
    ap.add_argument("--no-guard", dest="without_guard", action="store_true", help="eyebrow: no guard after reshaping")
    for k, v in (("length", 1.0), ("thickness", 1.0), ("arc", 0.0), ("rise", 0.0), ("tilt", 0.0)):
        ap.add_argument("--" + k, type=float, default=v)
    a = ap.parse_args(argv)
    import pmx_write as pmx_write
    if a.action == "check":
        if a.recipe:
            Fi = Done(a.pmx, a.recipe, a.registration)
            Mo = Fi.Mo
            names = list(dict.fromkeys(Mo.morphs_face() + names_recipe(Fi.gifts, Fi.recipe, Fi.M, Fi.wv)))
            L = check(Mo, names, expected(Fi, names))
        else:
            Mo = Model(a.pmx)
            L = check(Mo)
        display(L)
        if a.json:
            _write_json(a.json, L)
        return
    folders = [a.textures_for] if a.textures_for else []
    if a.action == "gaps":
        e = measure_gaps(Model(a.pmx), folders=folders)
        for n, v in sorted(e.items(), key=lambda kv: min(kv[1].values())):
            print("%-10s left %.4f  right %.4f%s" % (n, v["left"], v["right"],
                                                       "  < %.3f" % GAP_EYEBROW_MIN if min(v.values()) < GAP_EYEBROW_MIN - 1e-4 else ""))
        if a.json:
            _write_json(a.json, e)
        return
    os.makedirs(a.output, exist_ok=True)
    output = os.path.join(a.output, "result.pmx")
    if a.action in ("repair", "parts"):
        Fi = Done(a.pmx, a.recipe, a.registration)
        print("registration %s; scale %.4f (error %.2g); sorting %s; parts: %s" % (
            Fi.path_rec, Fi.s, Fi.residue_place, Fi.arrangement,
            ", ".join("%s/%s %.3f" % (p["name"], p["donor"], p["gap"]) for p in Fi.parts)))
        rap = repair(Fi) if a.action == "repair" else remove_parts(Fi)
        P = Fi.P
        info = {"source": a.pmx, "recipe": a.recipe, "registration": Fi.path_rec, "arrangement": Fi.arrangement,
                "morphs" if a.action == "repair" else "parts": rap}
    else:
        P = PMX.read(a.pmx)
        info = {}
        if a.action == "eyebrow":
            info = deform_eyebrows(P, length=a.length, thickness=a.thickness, arc=a.arc, rise=a.rise,
                                     tilt=a.tilt)
        if a.action == "guard" or not a.without_guard:
            info["guard"] = {"gap_min": a.gap_min if a.gap_min is not None else GAP_EYEBROW_MIN,
                             "factors": keep_eyebrows(P, a.gap_min, path=a.pmx, folders=folders)}
        info["source"] = a.pmx
    n = textures_relative(P, a.pmx, os.path.abspath(a.textures_for or a.output), folders=folders)
    if n:
        print("  %d texture paths rewritten relative to %s" % (n, a.textures_for or a.output))
    tmp = output + ".tmp.pmx"
    pmx_write.write(P, tmp)
    os.replace(tmp, output)
    _write_json(os.path.join(a.output, a.action + ".json"), info)
    print("written: %s" % output)


if __name__ == "__main__":
    main()

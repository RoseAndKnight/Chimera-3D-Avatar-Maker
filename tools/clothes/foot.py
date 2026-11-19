"""The foot posed in its shoe (numpy only: used by Blender and by the workshop's server).

A high-heeled shoe is modelled around a foot on tiptoe. Worn on a body whose foot stays flat, it
either keeps its shape and sinks its heel into the ground while the toes go through its top, or it
gets crushed onto the flat foot. Here the body takes the shoe's pose instead: the ankle bends (toes
down) and the toes bend back (toes up), then the whole model is raised by the heel's height.

Pose of a pair, as stored in the part's meta.json (`pied`):

    {"gauche": {"ankle": deg, "toes": deg}, "right": {...}, "lever": metres}

"ankle" > 0 points the foot down around the ankle's head, "toes" > 0 bends the toes up around
the toes' head; both turn around the horizontal axis across the foot. The angles do not depend on a
skeleton: the pivots and the axis are read on the skeleton that is posed (the body of the cache, the
assembled model, the house skeleton). Left (左) is +x in PMX and glTF frames.

Linear skinning with the weights of the foot's two groups of bones:

    v' = (1 - wa - wt) v + wa Ta(v) + wt Ta(Tt(v))

wa: weight of the ankle and its descendants except the toes, wt: weight of the toes and their
descendants. The inverse (`deposit`) is solved by fixed point: the fitted shoe is stored in the rest
frame, and posing it again with the same weights gives back what was fitted.
"""
import math

import numpy as np

SIDES = (("left", "左", 1.0), ("right", "右", -1.0))
ANKLE = ("足首", "足首D")
TOES = ("足先EX", "つま先")
TOP = np.array([0.0, 1.0, 0.0])


def _descendants(bones, roots):
    """Bones under `roots` (parents, and bones copying the rotation of one of them), roots included."""
    inside = set(roots)
    change = True
    while change:
        change = False
        for i, o in enumerate(bones):
            if i in inside:
                continue
            a = o.get("addition") or {}
            if o["parent"] in inside or (a.get("rotation") and a.get("source", -1) in inside):
                inside.add(i)
                change = True
    return inside


def chains(bones):
    """Per side present in the skeleton: {"pied": bone set, "toes": bone set, "ha": ankle head,
    "ht": toe head, "axe": unit axis}. Positions in the skeleton's units."""
    idx = {o["name"]: i for i, o in enumerate(bones)}
    out = {}
    for name, c, _ in SIDES:
        ch = [idx[c + n] for n in ANKLE if c + n in idx]
        if not ch:
            continue
        foot = _descendants(bones, ch)
        t = [idx[c + n] for n in TOES if c + n in idx and idx[c + n] in foot]
        if not t:
            continue
        toes = _descendants(bones, t)
        ha = np.asarray(bones[ch[0]]["pos"], float)
        ht = np.asarray(bones[t[0]]["pos"], float)
        av = ht - ha
        av[1] = 0.0
        if np.linalg.norm(av) < 1e-6:
            continue
        axis = np.cross(TOP, av / np.linalg.norm(av))
        out[name] = {"foot": foot, "toes": toes, "ha": ha, "ht": ht, "axis": axis / np.linalg.norm(axis)}
    return out


def rotation(axis, deg):
    a = math.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def transforms(ch, pose):
    """Per side: ((Ra, ta), (Rat, tat)), the affine maps of the ankle group and of the toe group."""
    out = {}
    for name, c in ch.items():
        p = (pose or {}).get(name) or {}
        Ra = rotation(c["axis"], p.get("ankle", 0.0))
        Rt = rotation(c["axis"], -p.get("toes", 0.0))
        ta = c["ha"] - Ra @ c["ha"]
        tt = c["ht"] - Rt @ c["ht"]
        out[name] = ((Ra, ta), (Ra @ Rt, Ra @ tt + ta))
    return out


def weight(ch, I, W):
    """Per vertex: (wa, wt, side name index) from bone indices I (n, k) and weights W (n, k).
    Side index: 0 left, 1 right, -1 none."""
    n = len(I)
    wa, wt, side = np.zeros(n), np.zeros(n), np.full(n, -1)
    for k, (name, _, _) in enumerate(SIDES):
        c = ch.get(name)
        if c is None:
            continue
        p = np.isin(I, list(c["foot"] - c["toes"])) & (I >= 0)
        o = np.isin(I, list(c["toes"])) & (I >= 0)
        a, t = (W * p).sum(1), (W * o).sum(1)
        m = (a + t) > 1e-6
        wa[m], wt[m], side[m] = a[m], t[m], k
    return wa, wt, side


def place(X, wa, wt, side, T, vectors=False):
    """Posed points (or directions, renormalised, with vectors=True)."""
    Y = np.array(X, float, copy=True)
    for k, (name, _, _) in enumerate(SIDES):
        if name not in T:
            continue
        m = side == k
        if not m.any():
            continue
        (Ra, ta), (Rt, tt) = T[name]
        x = X[m]
        a, t = wa[m, None], wt[m, None]
        if vectors:
            Y[m] = (1 - a - t) * x + a * (x @ Ra.T) + t * (x @ Rt.T)
        else:
            Y[m] = (1 - a - t) * x + a * (x @ Ra.T + ta) + t * (x @ Rt.T + tt)
    if vectors:
        Y /= np.maximum(np.linalg.norm(Y, axis=1), 1e-12)[:, None]
    return Y


def deposit(Y, wa, wt, side, T, vectors=False):
    """Inverse of `poser`: the rest points that land on Y."""
    if vectors:       # a blended rotation is inverted by the blend of the inverse rotations, close enough for a normal
        Ti = {n: ((Ra.T, -Ra.T @ ta), (Rt.T, -Rt.T @ tt)) for n, ((Ra, ta), (Rt, tt)) in T.items()}
        return place(Y, wa, wt, side, Ti, vectors=True)
    # poser is affine per vertex (Y = J x + b), so it is inverted exactly: an iteration
    # x += Y - poser(x) diverges past about 60 degrees (|I - R| > 1)
    X = np.array(Y, float, copy=True)
    for k, (name, _, _) in enumerate(SIDES):
        if name not in T:
            continue
        m = side == k
        if not m.any():
            continue
        (Ra, ta), (Rt, tt) = T[name]
        a, t = wa[m, None, None], wt[m, None, None]
        J = (1 - a - t) * np.eye(3) + a * Ra + t * Rt
        b = wa[m, None] * ta + wt[m, None] * tt
        X[m] = np.linalg.solve(J, (Y[m] - b)[:, :, None])[:, :, 0]
    return X


def transform_os(ch, T, n):
    """Affine map of each of n bones (identity outside the feet): [(R, t)]."""
    out = [(np.eye(3), np.zeros(3))] * n
    for name, c in ch.items():
        (Ra, ta), (Rt, tt) = T[name]
        for i in c["foot"]:
            out[i] = (Rt, tt) if i in c["toes"] else (Ra, ta)
    return out


def place_os(bones, ch, T, inverse=False):
    """Moves the foot bones of a PMX skeleton (heads, tail offsets, axes) in place."""
    for i, (R, t) in enumerate(transform_os(ch, T, len(bones))):
        if R is None or (np.allclose(R, np.eye(3)) and not t.any()):
            continue
        if inverse:
            R, t = R.T, -R.T @ t
        o = bones[i]
        o["pos"] = tuple(float(x) for x in R @ np.asarray(o["pos"], float) + t)
        if o.get("tail_bone") is None and o.get("tail_offset") is not None:
            o["tail_offset"] = tuple(float(x) for x in R @ np.asarray(o["tail_offset"], float))
        if o.get("fixed_axis") is not None:
            o["fixed_axis"] = tuple(float(x) for x in R @ np.asarray(o["fixed_axis"], float))
        if o.get("local_axes") is not None:
            o["local_axes"] = tuple(tuple(float(x) for x in R @ np.asarray(a, float)) for a in o["local_axes"])


def weight_pmx(pmx, ch):
    I = np.asarray(pmx["vertices"]["bones"], np.int64)
    W = np.asarray(pmx["vertices"]["weights"], float)
    W = np.where(I >= 0, W, 0)
    W = W / np.maximum(W.sum(1), 1e-12)[:, None]
    return weight(ch, I, W)


def place_pmx(pmx, pose, inverse=False, ch=None):
    """Poses (or un-poses) a whole PMX in place: vertices, normals, bones. `ch`: the chains read on the
    rest skeleton (required for inverse=True, since the posed skeleton has moved its toe pivots).
    Returns the chains used."""
    ch = ch or chains(pmx["bones"])
    if not ch or not pose:
        return ch
    T = transforms(ch, pose)
    wa, wt, side = weight_pmx(pmx, ch)
    X = np.asarray(pmx["vertices"]["pos"], float)
    N = np.asarray(pmx["vertices"]["nor"], float)
    if inverse:
        X, N = deposit(X, wa, wt, side, T), deposit(N, wa, wt, side, T, vectors=True)
    else:
        X, N = place(X, wa, wt, side, T), place(N, wa, wt, side, T, vectors=True)
    pmx["vertices"]["pos"] = [tuple(float(v) for v in x) for x in X]
    pmx["vertices"]["nor"] = [tuple(float(v) for v in x) for x in N]
    place_os(pmx["bones"], ch, T, inverse=inverse)
    return ch


def lift_pmx(pmx, h):
    """Raises a whole PMX by h (its units): vertices, bones, rigid bodies, joints."""
    if not h:
        return
    d = np.array([0.0, h, 0.0])
    pmx["vertices"]["pos"] = [tuple(float(v) for v in np.asarray(x, float) + d) for x in pmx["vertices"]["pos"]]
    for o in pmx["bones"]:
        o["pos"] = tuple(float(v) for v in np.asarray(o["pos"], float) + d)
    for r in pmx.get("rigid_bodies", []):
        r["pos"] = tuple(float(v) for v in np.asarray(r["pos"], float) + d)
    for j in pmx.get("joints", []):
        j["pos"] = tuple(float(v) for v in np.asarray(j["pos"], float) + d)


# ---------------------------------------------------------------------- search of the pose

GRID_ANKLE = tuple(range(-10, 72, 2))
RADIUS = 0.05          # metres: the shoe is found this close along the skin's normal, or the skin is outside
PREFERENCE = 0.002    # per 10 degrees: between two poses that fit as well, the smaller one
CROSSES = 0.015      # metres: skin this little outside the shoe's wall went through it
SOLE = 0.5          # the sole of the foot: its normal points down by more than this


def _inside(grid, nf, X, N):
    """Points X (normals N) inside the shoe: the ray along the normal leaves it (meets the back of
    an outward face) within RAYON."""
    t, f = grid.pull(X + 1e-4 * N, N, RADIUS)
    ok = np.isfinite(t)
    ok[ok] &= (nf[f[ok]] * N[ok]).sum(1) > 0
    return ok


def _outside(grid, nf, X, N):
    """Points X (normals N) of the foot out of the shoe where they should be in it: through its
    wall (the shoe's outer face lies just under the skin, within TRAVERSE) or, on the sole of the
    foot (normal down), not on the shoe's sole. The top of the foot above a low shoe's opening is
    not out of it: counting it, the search raised the heel of a low shoe until the instep went
    under the opening and the heel came out at the back."""
    inside = _inside(grid, nf, X, N)
    t, f = grid.pull(X + 1e-4 * N, -N, CROSSES)
    across = np.isfinite(t)
    across[across] &= (nf[f[across]] * N[across]).sum(1) > 0
    return ~inside & (across | (N[:, 1] < -SOLE))


def search(body_pmx, Y, F, unit, rays):
    """Pose of the body's feet that puts them inside the shoe (Y, F) placed rigidly on it at the
    ankle, in the body's frame and units (`unite` metres per unit; `rays`: the module
    tools/workshop/rays.py). The toes bend back by the angle the ankle points down: the front of
    the foot stays flat on the shoe's front, as in a heel (a free bend of the toes was found to
    point them down through the front of the shoe). Only the ankle's angle is searched: the one
    with the smallest share of the foot out of the shoe (_outside).
    Returns ({side: {"ankle", "toes", "outside_before", "outside"}}, chains)."""
    ch = chains(body_pmx["bones"])
    wa, wt, side = weight_pmx(body_pmx, ch)
    X = np.asarray(body_pmx["vertices"]["pos"], float)
    N = np.asarray(body_pmx["vertices"]["nor"], float)
    T0 = transforms(ch, {})
    pose = {}
    for k, (name, _, sg) in enumerate(SIDES):
        if name not in ch:
            continue
        pts = np.where((side == k) & (wa + wt > 0.5))[0]
        faces = F[(sg * Y[F].mean(1)[:, 0] > 0)]
        if not len(pts) or not len(faces):
            continue
        V = Y * unit
        nf = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
        # outward faces: vote of the faces against the foot's vertical axis
        c = V[np.unique(faces)].mean(0)
        r = V[faces].mean(1) - c
        r[:, 1] = 0
        nf *= 1.0 if float((nf * r).sum()) > 0 else -1.0
        nf /= np.maximum(np.linalg.norm(nf, axis=1), 1e-20)[:, None]
        grid = rays.Grid(V, faces)

        def cost(a):
            T = dict(T0)
            T.update(transforms({name: ch[name]}, {name: {"ankle": a, "toes": max(a, 0.0)}}))
            Xp = place(X[pts], wa[pts], wt[pts], side[pts], T) * unit
            Np = place(N[pts], wa[pts], wt[pts], side[pts], T, vectors=True)
            return float(_outside(grid, nf, Xp, Np).mean()) + PREFERENCE * abs(a) / 10

        a = min(GRID_ANKLE, key=cost)
        pose[name] = {"ankle": float(a), "toes": float(max(a, 0.0)),
                     "outside_before": round(cost(0.0), 3), "outside": round(cost(a), 3)}
    return pose, ch

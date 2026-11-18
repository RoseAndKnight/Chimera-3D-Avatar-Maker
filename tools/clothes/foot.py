"""The foot posed in its shoe (numpy only: used by Blender and by the workshop's server).

A high-heeled shoe is modelled around a foot on tiptoe. Worn on a body whose foot stays flat, it
either keeps its shape and sinks its heel into the ground while the toes go through its top, or it
gets crushed onto the flat foot. Here the body takes the shoe's pose instead: the ankle bends (toes
down) and the toes bend back (toes up), then the whole model is raised by the heel's height.

Pose of a pair, as stored in the part's meta.json (`pied`):

    {"gauche": {"cheville": deg, "orteils": deg}, "droite": {...}, "lever": metres}

"cheville" > 0 points the foot down around the ankle's head, "orteils" > 0 bends the toes up around
the toes' head; both turn around the horizontal axis across the foot. The angles do not depend on a
skeleton: the pivots and the axis are read on the skeleton that is posed (the body of the cache, the
assembled model, the house skeleton). Left (左) is +x in PMX and glTF frames.

Linear skinning with the weights of the foot's two groups of bones:

    v' = (1 - wa - wt) v + wa Ta(v) + wt Ta(Tt(v))

wa: weight of the ankle and its descendants except the toes, wt: weight of the toes and their
descendants. The inverse (`deposer`) is solved by fixed point: the fitted shoe is stored in the rest
frame, and posing it again with the same weights gives back what was fitted.
"""
import math

import numpy as np

COTES = (("gauche", "左", 1.0), ("droite", "右", -1.0))
CHEVILLE = ("足首", "足首D")
ORTEILS = ("足先EX", "つま先")
HAUT = np.array([0.0, 1.0, 0.0])


def _descendants(oss, racines):
    """Bones under `racines` (parents, and bones copying the rotation of one of them), racines included."""
    dedans = set(racines)
    change = True
    while change:
        change = False
        for i, o in enumerate(oss):
            if i in dedans:
                continue
            a = o.get("ajout") or {}
            if o["parent"] in dedans or (a.get("rotation") and a.get("source", -1) in dedans):
                dedans.add(i)
                change = True
    return dedans


def chaines(oss):
    """Per side present in the skeleton: {"pied": bone set, "orteils": bone set, "ha": ankle head,
    "ht": toe head, "axe": unit axis}. Positions in the skeleton's units."""
    idx = {o["nom"]: i for i, o in enumerate(oss)}
    out = {}
    for nom, c, _ in COTES:
        ch = [idx[c + n] for n in CHEVILLE if c + n in idx]
        if not ch:
            continue
        pied = _descendants(oss, ch)
        t = [idx[c + n] for n in ORTEILS if c + n in idx and idx[c + n] in pied]
        if not t:
            continue
        orteils = _descendants(oss, t)
        ha = np.asarray(oss[ch[0]]["pos"], float)
        ht = np.asarray(oss[t[0]]["pos"], float)
        av = ht - ha
        av[1] = 0.0
        if np.linalg.norm(av) < 1e-6:
            continue
        axe = np.cross(HAUT, av / np.linalg.norm(av))
        out[nom] = {"pied": pied, "orteils": orteils, "ha": ha, "ht": ht, "axe": axe / np.linalg.norm(axe)}
    return out


def rotation(axe, deg):
    a = math.radians(deg)
    K = np.array([[0, -axe[2], axe[1]], [axe[2], 0, -axe[0]], [-axe[1], axe[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def transfos(ch, pose):
    """Per side: ((Ra, ta), (Rat, tat)), the affine maps of the ankle group and of the toe group."""
    out = {}
    for nom, c in ch.items():
        p = (pose or {}).get(nom) or {}
        Ra = rotation(c["axe"], p.get("cheville", 0.0))
        Rt = rotation(c["axe"], -p.get("orteils", 0.0))
        ta = c["ha"] - Ra @ c["ha"]
        tt = c["ht"] - Rt @ c["ht"]
        out[nom] = ((Ra, ta), (Ra @ Rt, Ra @ tt + ta))
    return out


def poids(ch, I, W):
    """Per vertex: (wa, wt, side name index) from bone indices I (n, k) and weights W (n, k).
    Side index: 0 left, 1 right, -1 none."""
    n = len(I)
    wa, wt, cote = np.zeros(n), np.zeros(n), np.full(n, -1)
    for k, (nom, _, _) in enumerate(COTES):
        c = ch.get(nom)
        if c is None:
            continue
        p = np.isin(I, list(c["pied"] - c["orteils"])) & (I >= 0)
        o = np.isin(I, list(c["orteils"])) & (I >= 0)
        a, t = (W * p).sum(1), (W * o).sum(1)
        m = (a + t) > 1e-6
        wa[m], wt[m], cote[m] = a[m], t[m], k
    return wa, wt, cote


def poser(X, wa, wt, cote, T, vecteurs=False):
    """Posed points (or directions, renormalised, with vecteurs=True)."""
    Y = np.array(X, float, copy=True)
    for k, (nom, _, _) in enumerate(COTES):
        if nom not in T:
            continue
        m = cote == k
        if not m.any():
            continue
        (Ra, ta), (Rt, tt) = T[nom]
        x = X[m]
        a, t = wa[m, None], wt[m, None]
        if vecteurs:
            Y[m] = (1 - a - t) * x + a * (x @ Ra.T) + t * (x @ Rt.T)
        else:
            Y[m] = (1 - a - t) * x + a * (x @ Ra.T + ta) + t * (x @ Rt.T + tt)
    if vecteurs:
        Y /= np.maximum(np.linalg.norm(Y, axis=1), 1e-12)[:, None]
    return Y


def deposer(Y, wa, wt, cote, T, vecteurs=False):
    """Inverse of `poser`: the rest points that land on Y."""
    if vecteurs:       # a blended rotation is inverted by the blend of the inverse rotations, close enough for a normal
        Ti = {n: ((Ra.T, -Ra.T @ ta), (Rt.T, -Rt.T @ tt)) for n, ((Ra, ta), (Rt, tt)) in T.items()}
        return poser(Y, wa, wt, cote, Ti, vecteurs=True)
    # poser is affine per vertex (Y = J x + b), so it is inverted exactly: an iteration
    # x += Y - poser(x) diverges past about 60 degrees (|I - R| > 1)
    X = np.array(Y, float, copy=True)
    for k, (nom, _, _) in enumerate(COTES):
        if nom not in T:
            continue
        m = cote == k
        if not m.any():
            continue
        (Ra, ta), (Rt, tt) = T[nom]
        a, t = wa[m, None, None], wt[m, None, None]
        J = (1 - a - t) * np.eye(3) + a * Ra + t * Rt
        b = wa[m, None] * ta + wt[m, None] * tt
        X[m] = np.linalg.solve(J, (Y[m] - b)[:, :, None])[:, :, 0]
    return X


def transfo_os(ch, T, n):
    """Affine map of each of n bones (identity outside the feet): [(R, t)]."""
    out = [(np.eye(3), np.zeros(3))] * n
    for nom, c in ch.items():
        (Ra, ta), (Rt, tt) = T[nom]
        for i in c["pied"]:
            out[i] = (Rt, tt) if i in c["orteils"] else (Ra, ta)
    return out


def poser_os(oss, ch, T, inverse=False):
    """Moves the foot bones of a PMX skeleton (heads, tail offsets, axes) in place."""
    for i, (R, t) in enumerate(transfo_os(ch, T, len(oss))):
        if R is None or (np.allclose(R, np.eye(3)) and not t.any()):
            continue
        if inverse:
            R, t = R.T, -R.T @ t
        o = oss[i]
        o["pos"] = tuple(float(x) for x in R @ np.asarray(o["pos"], float) + t)
        if o.get("queue_os") is None and o.get("queue_dec") is not None:
            o["queue_dec"] = tuple(float(x) for x in R @ np.asarray(o["queue_dec"], float))
        if o.get("axe_fixe") is not None:
            o["axe_fixe"] = tuple(float(x) for x in R @ np.asarray(o["axe_fixe"], float))
        if o.get("axes_locaux") is not None:
            o["axes_locaux"] = tuple(tuple(float(x) for x in R @ np.asarray(a, float)) for a in o["axes_locaux"])


def poids_pmx(pmx, ch):
    I = np.asarray(pmx["sommets"]["os"], np.int64)
    W = np.asarray(pmx["sommets"]["poids"], float)
    W = np.where(I >= 0, W, 0)
    W = W / np.maximum(W.sum(1), 1e-12)[:, None]
    return poids(ch, I, W)


def poser_pmx(pmx, pose, inverse=False, ch=None):
    """Poses (or un-poses) a whole PMX in place: vertices, normals, bones. `ch`: the chains read on the
    rest skeleton (required for inverse=True, since the posed skeleton has moved its toe pivots).
    Returns the chains used."""
    ch = ch or chaines(pmx["os"])
    if not ch or not pose:
        return ch
    T = transfos(ch, pose)
    wa, wt, cote = poids_pmx(pmx, ch)
    X = np.asarray(pmx["sommets"]["pos"], float)
    N = np.asarray(pmx["sommets"]["nor"], float)
    if inverse:
        X, N = deposer(X, wa, wt, cote, T), deposer(N, wa, wt, cote, T, vecteurs=True)
    else:
        X, N = poser(X, wa, wt, cote, T), poser(N, wa, wt, cote, T, vecteurs=True)
    pmx["sommets"]["pos"] = [tuple(float(v) for v in x) for x in X]
    pmx["sommets"]["nor"] = [tuple(float(v) for v in x) for x in N]
    poser_os(pmx["os"], ch, T, inverse=inverse)
    return ch


def lever_pmx(pmx, h):
    """Raises a whole PMX by h (its units): vertices, bones, rigid bodies, joints."""
    if not h:
        return
    d = np.array([0.0, h, 0.0])
    pmx["sommets"]["pos"] = [tuple(float(v) for v in np.asarray(x, float) + d) for x in pmx["sommets"]["pos"]]
    for o in pmx["os"]:
        o["pos"] = tuple(float(v) for v in np.asarray(o["pos"], float) + d)
    for r in pmx.get("rigides", []):
        r["pos"] = tuple(float(v) for v in np.asarray(r["pos"], float) + d)
    for j in pmx.get("joints", []):
        j["pos"] = tuple(float(v) for v in np.asarray(j["pos"], float) + d)


# ---------------------------------------------------------------------- search of the pose

GRILLE_CHEVILLE = tuple(range(-10, 72, 2))
RAYON = 0.05          # metres: the shoe is found this close along the skin's normal, or the skin is outside
PREFERENCE = 0.002    # per 10 degrees: between two poses that fit as well, the smaller one
TRAVERSE = 0.015      # metres: skin this little outside the shoe's wall went through it
PLANTE = 0.5          # the sole of the foot: its normal points down by more than this


def _dedans(grille, nf, X, N):
    """Points X (normals N) inside the shoe: the ray along the normal leaves it (meets the back of
    an outward face) within RAYON."""
    t, f = grille.tirer(X + 1e-4 * N, N, RAYON)
    ok = np.isfinite(t)
    ok[ok] &= (nf[f[ok]] * N[ok]).sum(1) > 0
    return ok


def _dehors(grille, nf, X, N):
    """Points X (normals N) of the foot out of the shoe where they should be in it: through its
    wall (the shoe's outer face lies just under the skin, within TRAVERSE) or, on the sole of the
    foot (normal down), not on the shoe's sole. The top of the foot above a low shoe's opening is
    not out of it: counting it, the search raised the heel of a low shoe until the instep went
    under the opening and the heel came out at the back."""
    dedans = _dedans(grille, nf, X, N)
    t, f = grille.tirer(X + 1e-4 * N, -N, TRAVERSE)
    travers = np.isfinite(t)
    travers[travers] &= (nf[f[travers]] * N[travers]).sum(1) > 0
    return ~dedans & (travers | (N[:, 1] < -PLANTE))


def chercher(corps_pmx, Y, F, unite, rayons):
    """Pose of the body's feet that puts them inside the shoe (Y, F) placed rigidly on it at the
    ankle, in the body's frame and units (`unite` metres per unit; `rayons`: the module
    tools/workshop/rays.py). The toes bend back by the angle the ankle points down: the front of
    the foot stays flat on the shoe's front, as in a heel (a free bend of the toes was found to
    point them down through the front of the shoe). Only the ankle's angle is searched: the one
    with the smallest share of the foot out of the shoe (_dehors).
    Returns ({side: {"cheville", "orteils", "dehors_avant", "dehors"}}, chains)."""
    ch = chaines(corps_pmx["os"])
    wa, wt, cote = poids_pmx(corps_pmx, ch)
    X = np.asarray(corps_pmx["sommets"]["pos"], float)
    N = np.asarray(corps_pmx["sommets"]["nor"], float)
    T0 = transfos(ch, {})
    pose = {}
    for k, (nom, _, sg) in enumerate(COTES):
        if nom not in ch:
            continue
        pts = np.where((cote == k) & (wa + wt > 0.5))[0]
        faces = F[(sg * Y[F].mean(1)[:, 0] > 0)]
        if not len(pts) or not len(faces):
            continue
        V = Y * unite
        nf = np.cross(V[faces[:, 1]] - V[faces[:, 0]], V[faces[:, 2]] - V[faces[:, 0]])
        # outward faces: vote of the faces against the foot's vertical axis
        c = V[np.unique(faces)].mean(0)
        r = V[faces].mean(1) - c
        r[:, 1] = 0
        nf *= 1.0 if float((nf * r).sum()) > 0 else -1.0
        nf /= np.maximum(np.linalg.norm(nf, axis=1), 1e-20)[:, None]
        grille = rayons.Grille(V, faces)

        def cout(a):
            T = dict(T0)
            T.update(transfos({nom: ch[nom]}, {nom: {"cheville": a, "orteils": max(a, 0.0)}}))
            Xp = poser(X[pts], wa[pts], wt[pts], cote[pts], T) * unite
            Np = poser(N[pts], wa[pts], wt[pts], cote[pts], T, vecteurs=True)
            return float(_dehors(grille, nf, Xp, Np).mean()) + PREFERENCE * abs(a) / 10

        a = min(GRILLE_CHEVILLE, key=cout)
        pose[nom] = {"cheville": float(a), "orteils": float(max(a, 0.0)),
                     "dehors_avant": round(cout(0.0), 3), "dehors": round(cout(a), 3)}
    return pose, ch

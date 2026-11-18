"""Quaternions, MMD-style forward kinematics and linear skinning, in numpy only.

Convention: quaternions (w, x, y, z), Blender frame (facing -Y, +Z up,
左 side at +X). Each bone has a rest frame aligned with the world (as in
MMD, where bones have no orientation of their own): a local rotation of a bone is
therefore a rotation about the world axes at rest, applied under its parent.

Rotation append (付与): q_eff = q_src_eff ** w  (quaternion power,
w < 0 = inverse) composed before the bone's own pose.
"""
import numpy as np

IDENTITE = np.array([1.0, 0.0, 0.0, 0.0])


def q_axe_angle(axe, degres):
    a = np.asarray(axe, dtype=np.float64)
    a = a / np.linalg.norm(a)
    t = np.radians(degres) / 2.0
    return np.concatenate([[np.cos(t)], np.sin(t) * a])


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def q_puissance(q, t):
    """q ** t (slerp from the identity to q at fraction t; t < 0 gives the inverse)."""
    q = np.asarray(q, dtype=np.float64)
    if q[0] < 0:
        q = -q
    w = min(1.0, max(-1.0, q[0]))
    ang = 2 * np.arccos(w)
    s = np.sqrt(max(0.0, 1 - w * w))
    if s < 1e-12:
        return IDENTITE.copy()
    axe = q[1:] / s
    return np.concatenate([[np.cos(ang * t / 2)], np.sin(ang * t / 2) * axe])


def q_miroir_x(q):
    """Mirror rotation through the plane x = 0 (left <-> right)."""
    return np.array([q[0], q[1], -q[2], -q[3]])


def ordre_topologique(parents, sources_ajout):
    """Order in which each bone comes after its parent and after the source of its append."""
    n = len(parents)
    deps = [set() for _ in range(n)]
    for i in range(n):
        if parents[i] >= 0:
            deps[i].add(int(parents[i]))
        if sources_ajout[i] >= 0 and sources_ajout[i] != i:
            deps[i].add(int(sources_ajout[i]))
    ordre, fait, en_cours = [], [False] * n, [False] * n

    def visite(i):
        if fait[i]:
            return
        if en_cours[i]:
            raise ValueError("cycle in the bones around %d" % i)
        en_cours[i] = True
        for d in deps[i]:
            visite(d)
        en_cours[i] = False
        fait[i] = True
        ordre.append(i)

    for i in range(n):
        visite(i)
    return ordre


def cinematique(tetes, parents, ajout_source, ajout_facteur, ajout_rot, ajout_dep,
                rotations=None, deplacements=None, ajouts_actifs=None):
    """World matrices of each bone (B, 3, 3) and translations (B, 3).

    rotations: dict bone -> local quaternion; deplacements: dict bone -> vector.
    ajouts_actifs: boolean array (B,) to switch an append off (tests E-06, E-07).
    Returns (R, t) such that p' = R[b] @ (p - tetes[b]) + t[b].
    """
    n = len(parents)
    rotations = rotations or {}
    deplacements = deplacements or {}
    if ajouts_actifs is None:
        ajouts_actifs = np.ones(n, dtype=bool)
    q_eff = [IDENTITE.copy() for _ in range(n)]
    d_eff = [np.zeros(3) for _ in range(n)]
    R = np.zeros((n, 3, 3))
    t = np.zeros((n, 3))
    for b in ordre_topologique(parents, ajout_source):
        q = np.asarray(rotations.get(b, IDENTITE), dtype=np.float64)
        d = np.asarray(deplacements.get(b, np.zeros(3)), dtype=np.float64)
        s = ajout_source[b]
        if s >= 0 and ajouts_actifs[b]:
            w = float(ajout_facteur[b])
            if ajout_rot[b]:
                q = q_mul(q_puissance(q_eff[s], w), q)
            if ajout_dep[b]:
                d = d + w * d_eff[s]
        q_eff[b] = q
        d_eff[b] = d
        p = parents[b]
        Rl = q_mat(q)
        if p < 0:
            R[b] = Rl
            t[b] = tetes[b] + d
        else:
            R[b] = R[p] @ Rl
            t[b] = R[p] @ (tetes[b] - tetes[p] + d) + t[p]
    return R, t


def skinning(pos, idx, poids, tetes, R, t, normaliser=True):
    """Linear skinning. idx, poids: (n, K) with idx = -1 for an empty slot.

    normaliser=True reproduces Blender's Armature modifier (division by the
    sum of the weights, vertices with a zero sum left in place); False reproduces MMD.
    """
    pos = np.asarray(pos, dtype=np.float64)
    out = np.zeros_like(pos)
    somme = np.zeros(len(pos))
    for k in range(idx.shape[1]):
        b = idx[:, k]
        w = np.where(b >= 0, poids[:, k], 0.0)
        bb = np.where(b >= 0, b, 0)
        loc = pos - tetes[bb]
        dep = np.einsum("nij,nj->ni", R[bb], loc) + t[bb]
        out += w[:, None] * dep
        somme += w
    if normaliser:
        ok = somme > 1e-4
        out[ok] /= somme[ok, None]
        out[~ok] = pos[~ok]
    return out

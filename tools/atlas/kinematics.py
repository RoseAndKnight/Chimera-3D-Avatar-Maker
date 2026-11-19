"""Quaternions, MMD-style forward kinematics and linear skinning, in numpy only.

Convention: quaternions (w, x, y, z), Blender frame (facing -Y, +Z up,
左 side at +X). Each bone has a rest frame aligned with the world (as in
MMD, where bones have no orientation of their own): a local rotation of a bone is
therefore a rotation about the world axes at rest, applied under its parent.

Rotation append (付与): q_eff = q_src_eff ** w  (quaternion power,
w < 0 = inverse) composed before the bone's own pose.
"""
import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def q_axe_angle(axis, degrees):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    t = np.radians(degrees) / 2.0
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


def q_power(q, t):
    """q ** t (slerp from the identity to q at fraction t; t < 0 gives the inverse)."""
    q = np.asarray(q, dtype=np.float64)
    if q[0] < 0:
        q = -q
    w = min(1.0, max(-1.0, q[0]))
    ang = 2 * np.arccos(w)
    s = np.sqrt(max(0.0, 1 - w * w))
    if s < 1e-12:
        return IDENTITY.copy()
    axis = q[1:] / s
    return np.concatenate([[np.cos(ang * t / 2)], np.sin(ang * t / 2) * axis])


def q_mirror_x(q):
    """Mirror rotation through the plane x = 0 (left <-> right)."""
    return np.array([q[0], q[1], -q[2], -q[3]])


def topological_order(parents, sources_addition):
    """Order in which each bone comes after its parent and after the source of its append."""
    n = len(parents)
    deps = [set() for _ in range(n)]
    for i in range(n):
        if parents[i] >= 0:
            deps[i].add(int(parents[i]))
        if sources_addition[i] >= 0 and sources_addition[i] != i:
            deps[i].add(int(sources_addition[i]))
    order, done, in_progress = [], [False] * n, [False] * n

    def visited(i):
        if done[i]:
            return
        if in_progress[i]:
            raise ValueError("cycle in the bones around %d" % i)
        in_progress[i] = True
        for d in deps[i]:
            visited(d)
        in_progress[i] = False
        done[i] = True
        order.append(i)

    for i in range(n):
        visited(i)
    return order


def kinematics(heads, parents, addition_source, addition_factor, addition_rot, addition_disp,
                rotations=None, displacements=None, additions_actives=None):
    """World matrices of each bone (B, 3, 3) and translations (B, 3).

    rotations: dict bone -> local quaternion; displacements: dict bone -> vector.
    additions_actives: boolean array (B,) to switch an append off (tests E-06, E-07).
    Returns (R, t) such that p' = R[b] @ (p - heads[b]) + t[b].
    """
    n = len(parents)
    rotations = rotations or {}
    displacements = displacements or {}
    if additions_actives is None:
        additions_actives = np.ones(n, dtype=bool)
    q_eff = [IDENTITY.copy() for _ in range(n)]
    d_eff = [np.zeros(3) for _ in range(n)]
    R = np.zeros((n, 3, 3))
    t = np.zeros((n, 3))
    for b in topological_order(parents, addition_source):
        q = np.asarray(rotations.get(b, IDENTITY), dtype=np.float64)
        d = np.asarray(displacements.get(b, np.zeros(3)), dtype=np.float64)
        s = addition_source[b]
        if s >= 0 and additions_actives[b]:
            w = float(addition_factor[b])
            if addition_rot[b]:
                q = q_mul(q_power(q_eff[s], w), q)
            if addition_disp[b]:
                d = d + w * d_eff[s]
        q_eff[b] = q
        d_eff[b] = d
        p = parents[b]
        Rl = q_mat(q)
        if p < 0:
            R[b] = Rl
            t[b] = heads[b] + d
        else:
            R[b] = R[p] @ Rl
            t[b] = R[p] @ (heads[b] - heads[p] + d) + t[p]
    return R, t


def skinning(pos, idx, weight, heads, R, t, normalize=True):
    """Linear skinning. idx, weight: (n, K) with idx = -1 for an empty slot.

    normalize=True reproduces Blender's Armature modifier (division by the
    sum of the weights, vertices with a zero sum left in place); False reproduces MMD.
    """
    pos = np.asarray(pos, dtype=np.float64)
    out = np.zeros_like(pos)
    total = np.zeros(len(pos))
    for k in range(idx.shape[1]):
        b = idx[:, k]
        w = np.where(b >= 0, weight[:, k], 0.0)
        bb = np.where(b >= 0, b, 0)
        loc = pos - heads[bb]
        disp = np.einsum("nij,nj->ni", R[bb], loc) + t[bb]
        out += w[:, None] * disp
        total += w
    if normalize:
        ok = total > 1e-4
        out[ok] /= total[ok, None]
        out[~ok] = pos[~ok]
    return out

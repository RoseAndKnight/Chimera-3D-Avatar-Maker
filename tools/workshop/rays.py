"""Exact intersection of segments and triangles (numpy only).

Used by the layer adjustment (`outfit.py`) and its check (`check_layers.py`): the
question "is this point in front of or behind this surface, and by how much?" is asked there with a
ray and not with the distance to the closest face, which gets it wrong at the edges of parts and between
the vertices of coarse meshes.
"""
import numpy as np

CELL = 0.03
PAIRS = 2_000_000


def _keys(v):
    return v[:, 0] * 1000003 + v[:, 1] * 1009 + v[:, 2]


def _cells(lo, hi):
    """All cells of the boxes [lo, hi] (integers): (box index, cell key)."""
    n = hi - lo + 1
    ids, keys = [], []
    for dx in range(int(n[:, 0].max())):
        mx = n[:, 0] > dx
        for dy in range(int(n[:, 1].max())):
            mxy = mx & (n[:, 1] > dy)
            for dz in range(int(n[:, 2].max())):
                idx = np.where(mxy & (n[:, 2] > dz))[0]
                if len(idx):
                    ids.append(idx)
                    keys.append(_keys(lo[idx] + [dx, dy, dz]))
    return np.concatenate(ids), np.concatenate(keys)


class Grid:
    """The triangles (V, F) sorted into cells, to cast segments through them."""

    def __init__(self, V, F):
        self.T = V[F] if len(F) else np.zeros((0, 3, 3))
        self.empty = not len(F)
        if self.empty:
            return
        lo = np.floor(self.T.min(1) / CELL).astype(np.int64)
        hi = np.floor(self.T.max(1) / CELL).astype(np.int64)
        big = (hi - lo + 1).max(1) > 6
        self.big = np.where(big)[0]
        small = np.where(~big)[0]
        if len(small):
            f_id, f_key = _cells(lo[small], hi[small])
            order = np.argsort(f_key, kind="stable")
            self.f_id, self.f_key = small[f_id][order], f_key[order]
        else:
            self.f_id, self.f_key = np.zeros(0, np.int64), np.zeros(0, np.int64)

    def pull(self, O, D, L, last=False, all_=False):
        """Intersection of each segment O + t·D, t in ]0, L], with the surface: (t, face).
        The first one, or the last one with `last`. Without intersection, t is inf and the face -1.
        With `all_`: (segment, t, face) of every intersection, sorted by segment then by t."""
        n = len(O)
        t_out, f_out = np.full(n, np.inf), np.full(n, -1, np.int64)
        if not n or self.empty:
            return (np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)) if all_ else (t_out, f_out)
        L = np.broadcast_to(np.asarray(L, float), (n,))
        # a long segment is cut into pieces of at most two cells
        k = np.maximum(np.ceil(L / (2 * CELL)).astype(int), 1)
        seg = np.repeat(np.arange(n), k)
        rank = np.arange(len(seg)) - np.repeat(np.cumsum(k) - k, k)
        stage = (L / k)[seg]
        start = rank * stage
        A = O[seg] + D[seg] * start[:, None]
        B = A + D[seg] * stage[:, None]
        s_id, s_key = _cells(np.floor(np.minimum(A, B) / CELL).astype(np.int64),
                                np.floor(np.maximum(A, B) / CELL).astype(np.int64))
        g = np.searchsorted(self.f_key, s_key)
        nb = np.searchsorted(self.f_key, s_key, side="right") - g
        T = self.T
        touched = []

        def tester(si, fi):
            o, d = A[si], D[seg[si]]
            e1, e2 = T[fi, 1] - T[fi, 0], T[fi, 2] - T[fi, 0]
            h = np.cross(d, e2)
            a = (e1 * h).sum(1)
            ok = np.abs(a) > 1e-14
            f = 1.0 / np.where(ok, a, 1.0)
            s = o - T[fi, 0]
            u = f * (s * h).sum(1)
            q = np.cross(s, e1)
            v = f * (d * q).sum(1)
            t = f * (e2 * q).sum(1)
            ok &= (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9) & (t <= stage[si])
            if ok.any():
                touched.append((seg[si[ok]], t[ok] + start[si[ok]], fi[ok]))

        guard = nb > 0
        s_id, g, nb = s_id[guard], g[guard], nb[guard]
        bound = np.concatenate([[0], np.cumsum(nb)])
        a = 0
        while a < len(s_id):
            b = max(int(np.searchsorted(bound, bound[a] + PAIRS, side="right")) - 1, a + 1)
            m = nb[a:b]
            offset = np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)
            tester(np.repeat(s_id[a:b], m), self.f_id[np.repeat(g[a:b], m) + offset])
            a = b
        for fg in self.big:  # very large faces are tested against all segments
            tester(np.arange(len(A)), np.full(len(A), fg))
        if touched:
            which = np.concatenate([x[0] for x in touched])
            t = np.concatenate([x[1] for x in touched])
            f = np.concatenate([x[2] for x in touched])
            order = np.lexsort((-t if last else t, which))
            which, t, f = which[order], t[order], f[order]
            if all_:
                return which, t, f
            first = np.concatenate([[True], which[1:] != which[:-1]])
            t_out[which[first]], f_out[which[first]] = t[first], f[first]
        if all_:
            return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
        return t_out, f_out


def edges(F):
    return np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)


def crossings(Va, Fa, Vb, Fb, grid_b=None):
    """Points where an edge of A crosses a face of B."""
    if not len(Fa) or not len(Fb):
        return np.zeros((0, 3))
    E = edges(Fa)
    P0, P1 = Va[E[:, 0]], Va[E[:, 1]]
    lo, hi = Vb[Fb].reshape(-1, 3).min(0) - 1e-3, Vb[Fb].reshape(-1, 3).max(0) + 1e-3
    near = (np.maximum(P0, P1) >= lo).all(1) & (np.minimum(P0, P1) <= hi).all(1)
    P0, P1 = P0[near], P1[near]
    L = np.linalg.norm(P1 - P0, axis=1)
    ok = L > 1e-9
    P0, P1, L = P0[ok], P1[ok], L[ok]
    D = (P1 - P0) / L[:, None]
    t, _ = (grid_b or Grid(Vb, Fb)).pull(P0, D, L)
    m = np.isfinite(t)
    return P0[m] + D[m] * t[m, None]

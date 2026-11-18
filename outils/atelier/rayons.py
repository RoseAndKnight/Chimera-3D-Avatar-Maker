"""Exact intersection of segments and triangles (numpy only).

Used by the layer adjustment (`combinaison.py`) and its check (`controle_couches.py`): the
question "is this point in front of or behind this surface, and by how much?" is asked there with a
ray and not with the distance to the closest face, which gets it wrong at the edges of parts and between
the vertices of coarse meshes.
"""
import numpy as np

CELLULE = 0.03
PAIRES = 2_000_000


def _cles(v):
    return v[:, 0] * 1000003 + v[:, 1] * 1009 + v[:, 2]


def _cellules(lo, hi):
    """All cells of the boxes [lo, hi] (integers): (box index, cell key)."""
    n = hi - lo + 1
    ids, cles = [], []
    for dx in range(int(n[:, 0].max())):
        mx = n[:, 0] > dx
        for dy in range(int(n[:, 1].max())):
            mxy = mx & (n[:, 1] > dy)
            for dz in range(int(n[:, 2].max())):
                ou = np.where(mxy & (n[:, 2] > dz))[0]
                if len(ou):
                    ids.append(ou)
                    cles.append(_cles(lo[ou] + [dx, dy, dz]))
    return np.concatenate(ids), np.concatenate(cles)


class Grille:
    """The triangles (V, F) sorted into cells, to cast segments through them."""

    def __init__(self, V, F):
        self.T = V[F] if len(F) else np.zeros((0, 3, 3))
        self.vide = not len(F)
        if self.vide:
            return
        lo = np.floor(self.T.min(1) / CELLULE).astype(np.int64)
        hi = np.floor(self.T.max(1) / CELLULE).astype(np.int64)
        gros = (hi - lo + 1).max(1) > 6
        self.gros = np.where(gros)[0]
        petits = np.where(~gros)[0]
        if len(petits):
            f_id, f_cle = _cellules(lo[petits], hi[petits])
            ordre = np.argsort(f_cle, kind="stable")
            self.f_id, self.f_cle = petits[f_id][ordre], f_cle[ordre]
        else:
            self.f_id, self.f_cle = np.zeros(0, np.int64), np.zeros(0, np.int64)

    def tirer(self, O, D, L, dernier=False, toutes=False):
        """Intersection of each segment O + t·D, t in ]0, L], with the surface: (t, face).
        The first one, or the last one with `dernier`. Without intersection, t is inf and the face -1.
        With `toutes`: (segment, t, face) of every intersection, sorted by segment then by t."""
        n = len(O)
        t_out, f_out = np.full(n, np.inf), np.full(n, -1, np.int64)
        if not n or self.vide:
            return (np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)) if toutes else (t_out, f_out)
        L = np.broadcast_to(np.asarray(L, float), (n,))
        # a long segment is cut into pieces of at most two cells
        k = np.maximum(np.ceil(L / (2 * CELLULE)).astype(int), 1)
        seg = np.repeat(np.arange(n), k)
        rang = np.arange(len(seg)) - np.repeat(np.cumsum(k) - k, k)
        pas = (L / k)[seg]
        debut = rang * pas
        A = O[seg] + D[seg] * debut[:, None]
        B = A + D[seg] * pas[:, None]
        s_id, s_cle = _cellules(np.floor(np.minimum(A, B) / CELLULE).astype(np.int64),
                                np.floor(np.maximum(A, B) / CELLULE).astype(np.int64))
        g = np.searchsorted(self.f_cle, s_cle)
        nb = np.searchsorted(self.f_cle, s_cle, side="right") - g
        T = self.T
        touches = []

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
            ok &= (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9) & (t <= pas[si])
            if ok.any():
                touches.append((seg[si[ok]], t[ok] + debut[si[ok]], fi[ok]))

        garde = nb > 0
        s_id, g, nb = s_id[garde], g[garde], nb[garde]
        borne = np.concatenate([[0], np.cumsum(nb)])
        a = 0
        while a < len(s_id):
            b = max(int(np.searchsorted(borne, borne[a] + PAIRES, side="right")) - 1, a + 1)
            m = nb[a:b]
            dec = np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)
            tester(np.repeat(s_id[a:b], m), self.f_id[np.repeat(g[a:b], m) + dec])
            a = b
        for fg in self.gros:  # very large faces are tested against all segments
            tester(np.arange(len(A)), np.full(len(A), fg))
        if touches:
            qui = np.concatenate([x[0] for x in touches])
            t = np.concatenate([x[1] for x in touches])
            f = np.concatenate([x[2] for x in touches])
            ordre = np.lexsort((-t if dernier else t, qui))
            qui, t, f = qui[ordre], t[ordre], f[ordre]
            if toutes:
                return qui, t, f
            premier = np.concatenate([[True], qui[1:] != qui[:-1]])
            t_out[qui[premier]], f_out[qui[premier]] = t[premier], f[premier]
        if toutes:
            return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
        return t_out, f_out


def aretes(F):
    return np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)


def croisements(Va, Fa, Vb, Fb, grille_b=None):
    """Points where an edge of A crosses a face of B."""
    if not len(Fa) or not len(Fb):
        return np.zeros((0, 3))
    E = aretes(Fa)
    P0, P1 = Va[E[:, 0]], Va[E[:, 1]]
    lo, hi = Vb[Fb].reshape(-1, 3).min(0) - 1e-3, Vb[Fb].reshape(-1, 3).max(0) + 1e-3
    pres = (np.maximum(P0, P1) >= lo).all(1) & (np.minimum(P0, P1) <= hi).all(1)
    P0, P1 = P0[pres], P1[pres]
    L = np.linalg.norm(P1 - P0, axis=1)
    ok = L > 1e-9
    P0, P1, L = P0[ok], P1[ok], L[ok]
    D = (P1 - P0) / L[:, None]
    t, _ = (grille_b or Grille(Vb, Fb)).tirer(P0, D, L)
    m = np.isfinite(t)
    return P0[m] + D[m] * t[m, None]

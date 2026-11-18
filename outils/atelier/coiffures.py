"""Hairstyle adapter for the cache: the computation of `coiffure.poser` (passing through the
pivot head, getting out of the skin) and the tests of `coiffure.verifier`, without writing a PMX, to
store a single set of positions per hairstyle x head pair.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun as C  # noqa: E402
import glb as GLB  # noqa: E402

import bibliotheque as BI  # noqa: E402
import coiffure as CO  # noqa: E402
import reflets as RF  # noqa: E402
import geom as G  # noqa: E402
import tete as TT  # noqa: E402

_pivot, _masques = None, None
_tetes = {}
_peau = {}


def peau_affichee():
    """Skin the page draws under the hair, in the frame of the placed hair (MMD frame of the base
    head sewn on the common body, C.BASE_PMX): the body skin it leaves (skull, ears, neck,
    shoulders) and the face. Placing on the face alone left strands inside the body's skull,
    which the model's own head does not have (its skin stops under the hair): skin seen through."""
    if "peau" not in _peau:
        import pmx as PMX
        D = PMX.lire(C.BASE_PMX)
        T = PMX.vers_tableaux(D)
        B = TT.Tete(C.BASE_PMX)
        mats = [i for i, m in enumerate(D["materiaux"]) if m["nom"].startswith("肌")] + list(B.i_peau)
        vs, Fl = G.sous_maillage(T["faces"], np.isin(T["mat_face"], mats))
        _peau["peau"] = (np.asarray(T["pos"], float)[vs], Fl, np.asarray(T["nor"], float)[vs])
    return _peau["peau"]


RAYON_POUSSEE = 0.25   # eye distances: a push moves all the hair within this radius with it


def champ_de_poussee(P, u, rayon):
    """Smooth displacement of the points P from the pushes u (nonzero on few points): each point
    takes the strongest push around it, weighted by (1 - (d/rayon)^2)^2. Hair cards are stacked in
    layers a few millimetres apart; pushing each vertex to the skin margin on its own laid the layers
    onto one shell, where they cut through each other (dark shards). Moved together, they keep their order."""
    j = np.flatnonzero(np.linalg.norm(u, axis=1) > 0)
    out = np.zeros_like(P)
    if not len(j):
        return out
    best = np.zeros(len(P))
    for k in range(0, len(j), 256):
        jj = j[k:k + 256]
        d2 = ((P[:, None, :] - P[jj][None, :, :]) ** 2).sum(2) / rayon ** 2
        w = np.clip(1 - d2, 0, None) ** 2 * np.linalg.norm(u[jj], axis=1)[None, :]
        a = w.argmax(1)
        m = w[np.arange(len(P)), a]
        plus = m > best
        best[plus] = m[plus]
        out[plus] = u[jj[a[plus]]] * (m[plus] / np.linalg.norm(u[jj[a[plus]]], axis=1))[:, None]
    return out


def sortir_de_peau_affichee(pos, iod, repos, tours=4):
    """Hair (final frame) pushed out of the displayed skin. `repos` (distance to the model's own
    skin) keeps planted roots only where it means something: a vertex far "inside" its own open skin
    is merely behind it (the back of the head, where the model has none). Returns the positions and
    the signed distance left, relative to that of planted roots."""
    V, F, N = peau_affichee()
    lo, hi = pos.min(0) - iod, pos.max(0) + iod
    c = V[F].mean(1)
    F = F[((c > lo) & (c < hi)).all(1)]
    if not len(F):
        return pos, np.full(len(pos), np.inf)
    r = np.where(repos > -CO.PROFOND_MAX * iod, repos, np.inf)
    marge = np.minimum(CO.MARGE_PEAU * iod, r - 1e-4 * iod)
    W = np.asarray(pos, float).copy()
    for _ in range(tours):
        s, n = signee(W, V, F, N)
        e = (s < marge) & (s > -CO.PROFOND_MAX * iod)
        if not e.any():
            break
        u = np.zeros_like(W)
        u[e] = (marge[e] - s[e])[:, None] * n[e]
        W += champ_de_poussee(W, u, RAYON_POUSSEE * iod)
    s, _ = signee(W, V, F, N)
    return W, np.where(np.isfinite(r), s - np.minimum(r, 0), s)


def signee(P, V, F, N):
    """`coiffure.signee`, with the closest point looked up by Blender's BVH tree when it is available."""
    if not len(F):
        return np.full(len(P), np.inf), np.zeros((len(P), 3))
    try:
        from mathutils.bvhtree import BVHTree
    except ImportError:
        q, f, bary, _ = G.plus_proche(P, V, F)
    else:
        arbre = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
        q, f = np.zeros((len(P), 3)), np.zeros(len(P), np.int64)
        for k, p in enumerate(P.tolist()):
            loc, _, i, _ = arbre.find_nearest(p)
            q[k], f[k] = loc, i
        a, b, c = (V[F[f, j]] for j in range(3))
        v0, v1, v2 = b - a, c - a, q - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-30, d00 * d11 - d01 * d01, 1e-30)
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        bary = np.clip(np.stack([1 - w1 - w2, w1, w2], 1), 0, 1)
    n = (bary[:, :, None] * N[F[f]]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    return CO._cote(P, q, n) * np.linalg.norm(P - q, axis=1), n


def pivot():
    global _pivot, _masques
    if _pivot is None:
        _pivot = BI.charger_pivot()
        _masques = BI.masques(_pivot)
    return _pivot, _masques


def tete_cible(biblio, nom):
    if nom not in _tetes:
        B = TT.Tete(BI.lire_fiche(biblio, nom)["pmx"])
        _tetes[nom] = (B, CO.reperer(B))
    return _tetes[nom]


class Preparation:
    def __init__(self, biblio, nom):
        self.biblio, self.nom = biblio, nom
        self.A = TT.Tete(BI.lire_fiche(biblio, nom)["pmx"])
        self.rA = CO.reperer(self.A)
        A = self.A
        self.masque = self.rA["coiffure"]
        if not self.masque.any():
            raise ValueError("no hairstyle found")
        # "+" highlights baked into the hair texture (tete/reflets.py): their duplicate faces go
        retire, self.tex_reflets = RF.cuire(A, self.masque, os.path.join(C.CACHE, "coiffures", nom))
        self.masque = self.masque & ~retire
        self.vs, self.Fl = G.sous_maillage(A.faces, self.masque)
        mf = A.mat_face[self.masque]
        self.mats = [int(m) for m in np.unique(mf)]
        self.mf = mf
        # distance of each vertex to the skin of its own head: placing it does not bring it any closer
        self.repos, _ = signee(A.pos[self.vs], *CO._surface(A, np.isin(A.mat_face, A.i_peau)))
        self._poses = {}

    def glb_base(self):
        A = self.A
        pos, nor, Fl = GLB.vers_gltf(A.pos[self.vs], A.t["nor"][self.vs], self.Fl)
        self.par_mat = [Fl[self.mf == m] for m in self.mats]
        prims = [{"faces": f, "materiau": k} for k, f in enumerate(self.par_mat)]
        return pos, nor, A.t["uv"][self.vs], prims

    def poser_sur(self, tete):
        """Positions (glTF frame, on the common body) and verdict for a target head."""
        nomB = tete["biblio"]
        cle = (nomB, tuple(tete["translation"]))   # all heads of the catalogue have the base head
        if cle not in self._poses:
            self._poses[cle] = self._poser(tete)
        return dict(self._poses[cle])

    def _poser(self, tete):
        nomB = tete["biblio"]
        B, rB = tete_cible(self.biblio, nomB)
        A, rA = self.A, self.rA
        p0, M = pivot()
        pAB = CO.Passage(self.biblio, self.nom, nomB, p0, M)
        pos = pAB(A.pos[self.vs])
        Vp, Fp, Np = CO._surface(B, np.isin(B.mat_face, B.i_peau))
        repos = self.repos * pAB.echelle
        pos, _ = CO.sortir_de_peau(pos, self.Fl, Vp, Fp, Np, B.iod, signee=signee, repos=repos)
        s, _ = signee(pos, Vp, Fp, Np)
        # inside the skin: gone in although it was outside on its own head (the roots of a made
        # hairstyle are planted there, up to 77 mm for Orpheya: hidden, they do not count)
        t = np.asarray(tete["translation"], float)
        final, s2 = sortir_de_peau_affichee(pos + t, B.iod, repos)
        pos = final - t
        dedans = float((((s < 0) & (s > -CO.PROFOND_MAX * B.iod) & (repos >= 0)) |
                        ((s2 < 0) & (s2 > -CO.PROFOND_MAX * B.iod))).mean())
        yeux = np.array([B.centre_oeil[c] for c in "gd"])
        T = pos[self.Fl]
        cov = CO.couverture(B.V, B.F, T, B.V.mean(0), yeux[:, 1].mean(), B.iod)
        cov_c = CO.couverture_crane(B.V, B.F, T, B.V.mean(0), yeux[:, 1].mean(), B.iod)
        plancher = min(rA["couverture"], rB["couverture"])
        tests = {"crane_couvert": bool(cov >= plancher - 0.03),
                 "crane_sous_cheveux": bool(cov_c >= rA["couverture_cheveux"] - 0.05),
                 "cheveux_hors_peau": bool(dedans <= 0.005)}
        echecs = [k for k, v in tests.items() if not v]
        q =np.asarray(final, np.float32) * 0.08
        q[:, 2] *= -1
        return {"positions": q, "passe": not echecs, "echecs": echecs, "couverture": round(float(cov), 3),
                "couverture_cheveux": round(float(cov_c), 3), "part_dedans": round(dedans, 4)}

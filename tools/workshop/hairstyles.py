"""Hairstyle adapter for the cache: the computation of `hairstyle.place` (passing through the
pivot head, getting out of the skin) and the tests of `hairstyle.check`, without writing a PMX, to
store a single set of positions per hairstyle x head pair.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import glb as GLB  # noqa: E402

import library as BI  # noqa: E402
import hairstyle as CO  # noqa: E402
import highlights as RF  # noqa: E402
import geom as G  # noqa: E402
import head as TT  # noqa: E402

_pivot, _masks = None, None
_heads = {}
_skin = {}


def skin_displayed():
    """Skin the page draws under the hair, in the frame of the placed hair (MMD frame of the base
    head sewn on the common body, C.BASE_PMX): the body skin it leaves (skull, ears, neck,
    shoulders) and the face. Placing on the face alone left strands inside the body's skull,
    which the model's own head does not have (its skin stops under the hair): skin seen through."""
    if "skin" not in _skin:
        import pmx as PMX
        D = PMX.read(C.BASE_PMX)
        T = PMX.to_arrays(D)
        B = TT.Head(C.BASE_PMX)
        mats = [i for i, m in enumerate(D["materials"]) if m["name"].startswith("肌")] + list(B.i_skin)
        vs, Fl = G.under_mesh(T["faces"], np.isin(T["mat_face"], mats))
        _skin["skin"] = (np.asarray(T["pos"], float)[vs], Fl, np.asarray(T["nor"], float)[vs])
    return _skin["skin"]


RADIUS_PUSH = 0.25   # eye distances: a push moves all the hair within this radius with it


def field_of_push(P, u, radius):
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
        d2 = ((P[:, None, :] - P[jj][None, :, :]) ** 2).sum(2) / radius ** 2
        w = np.clip(1 - d2, 0, None) ** 2 * np.linalg.norm(u[jj], axis=1)[None, :]
        a = w.argmax(1)
        m = w[np.arange(len(P)), a]
        plus = m > best
        best[plus] = m[plus]
        out[plus] = u[jj[a[plus]]] * (m[plus] / np.linalg.norm(u[jj[a[plus]]], axis=1))[:, None]
    return out


def leave_of_skin_displayed(pos, iod, rest_pose, perimeters=4):
    """Hair (final frame) pushed out of the displayed skin. `rest_pose` (distance to the model's own
    skin) keeps planted roots only where it means something: a vertex far "inside" its own open skin
    is merely behind it (the back of the head, where the model has none). Returns the positions and
    the signed distance left, relative to that of planted roots."""
    V, F, N = skin_displayed()
    lo, hi = pos.min(0) - iod, pos.max(0) + iod
    c = V[F].mean(1)
    F = F[((c > lo) & (c < hi)).all(1)]
    if not len(F):
        return pos, np.full(len(pos), np.inf)
    r = np.where(rest_pose > -CO.DEEP_MAX * iod, rest_pose, np.inf)
    margin = np.minimum(CO.MARGIN_SKIN * iod, r - 1e-4 * iod)
    W = np.asarray(pos, float).copy()
    for _ in range(perimeters):
        s, n = signed(W, V, F, N)
        e = (s < margin) & (s > -CO.DEEP_MAX * iod)
        if not e.any():
            break
        u = np.zeros_like(W)
        u[e] = (margin[e] - s[e])[:, None] * n[e]
        W += field_of_push(W, u, RADIUS_PUSH * iod)
    s, _ = signed(W, V, F, N)
    return W, np.where(np.isfinite(r), s - np.minimum(r, 0), s)


def signed(P, V, F, N):
    """`hairstyle.signed`, with the closest point looked up by Blender's BVH tree when it is available."""
    if not len(F):
        return np.full(len(P), np.inf), np.zeros((len(P), 3))
    try:
        from mathutils.bvhtree import BVHTree
    except ImportError:
        q, f, bary, _ = G.plus_near(P, V, F)
    else:
        tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
        q, f = np.zeros((len(P), 3)), np.zeros(len(P), np.int64)
        for k, p in enumerate(P.tolist()):
            loc, _, i, _ = tree.find_nearest(p)
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
    return CO._side(P, q, n) * np.linalg.norm(P - q, axis=1), n


def pivot():
    global _pivot, _masks
    if _pivot is None:
        _pivot = BI.load_pivot()
        _masks = BI.masks(_pivot)
    return _pivot, _masks


def head_target(lib, name):
    if name not in _heads:
        B = TT.Head(BI.read_card(lib, name)["pmx"])
        _heads[name] = (B, CO.locate(B))
    return _heads[name]


class Preparation:
    def __init__(self, lib, name):
        self.lib, self.name = lib, name
        self.A = TT.Head(BI.read_card(lib, name)["pmx"])
        self.rA = CO.locate(self.A)
        A = self.A
        self.mask = self.rA["hairstyle"]
        if not self.mask.any():
            raise ValueError("no hairstyle found")
        # "+" highlights baked into the hair texture (head/highlights.py): their duplicate faces go
        removes, self.tex_reflections = RF.bake(A, self.mask, os.path.join(C.CACHE, "hairstyles", name))
        self.mask = self.mask & ~removes
        self.vs, self.Fl = G.under_mesh(A.faces, self.mask)
        mf = A.mat_face[self.mask]
        self.mats = [int(m) for m in np.unique(mf)]
        self.mf = mf
        # distance of each vertex to the skin of its own head: placing it does not bring it any closer
        self.rest_pose, _ = signed(A.pos[self.vs], *CO._surface(A, np.isin(A.mat_face, A.i_skin)))
        self._poses = {}

    def glb_base(self):
        A = self.A
        pos, nor, Fl = GLB.to_gltf(A.pos[self.vs], A.t["nor"][self.vs], self.Fl)
        self.per_mat = [Fl[self.mf == m] for m in self.mats]
        prims = [{"faces": f, "material": k} for k, f in enumerate(self.per_mat)]
        return pos, nor, A.t["uv"][self.vs], prims

    def place_on(self, head):
        """Positions (glTF frame, on the common body) and verdict for a target head."""
        nameB = head["library"]
        the_key = (nameB, tuple(head["translation"]))   # all heads of the catalogue have the base head
        if the_key not in self._poses:
            self._poses[the_key] = self._place(head)
        return dict(self._poses[the_key])

    def _place(self, head):
        nameB = head["library"]
        B, rB = head_target(self.lib, nameB)
        A, rA = self.A, self.rA
        p0, M = pivot()
        pAb = CO.Passage(self.lib, self.name, nameB, p0, M)
        pos = pAb(A.pos[self.vs])
        Vp, Fp, Np = CO._surface(B, np.isin(B.mat_face, B.i_skin))
        rest_pose = self.rest_pose * pAb.scale
        pos, _ = CO.leave_of_skin(pos, self.Fl, Vp, Fp, Np, B.iod, signed=signed, rest_pose=rest_pose)
        s, _ = signed(pos, Vp, Fp, Np)
        # inside the skin: gone in although it was outside on its own head (the roots of a made
        # hairstyle are planted there, up to 77 mm for Orpheya: hidden, they do not count)
        t = np.asarray(head["translation"], float)
        final, s2 = leave_of_skin_displayed(pos + t, B.iod, rest_pose)
        pos = final - t
        inside = float((((s < 0) & (s > -CO.DEEP_MAX * B.iod) & (rest_pose >= 0)) |
                        ((s2 < 0) & (s2 > -CO.DEEP_MAX * B.iod))).mean())
        eyes = np.array([B.center_eye[c] for c in "gd"])
        T = pos[self.Fl]
        cov = CO.coverage(B.V, B.F, T, B.V.mean(0), eyes[:, 1].mean(), B.iod)
        cov_c = CO.coverage_skull(B.V, B.F, T, B.V.mean(0), eyes[:, 1].mean(), B.iod)
        floor = min(rA["coverage"], rB["coverage"])
        tests = {"skull_covered": bool(cov >= floor - 0.03),
                 "skull_under_hair": bool(cov_c >= rA["hair_coverage"] - 0.05),
                 "hair_off_skin": bool(inside <= 0.005)}
        failures = [k for k, v in tests.items() if not v]
        q =np.asarray(final, np.float32) * 0.08
        q[:, 2] *= -1
        return {"positions": q, "passed": not failures, "failures": failures, "coverage": round(float(cov), 3),
                "hair_coverage": round(float(cov_c), 3), "fraction_inside": round(inside, 4)}

"""Layers of an outfit (numpy only), the same computation for the page preview and for the final PMX.

Each part was fitted alone on the skin. An outfit is adjusted as a whole, in a common frame:
each garment vertex has its **line** (the direction moving away from the skin at that spot) and its
**height** above the skin. It moves only along its line. The rule to keep fits in one
sentence: along a line, parts meet in the order of their ranks, at least
`MARGIN_TUCKED` apart. It is measured by exact rays (`rays.py`),
both ways: from the vertices underneath to the surface on top, and from the vertices on top
to the surface underneath (this catches wide faces, none of whose vertices pokes through).
A vertex's margin grows with the length of its edges (`arrow`): between its vertices, a
wide face drifts away from the curved surface it follows.

1. remove: an undergarment the bottom fully covers is not worn (underwear under shorts);
2. order: from inside to outside, by rank. A top whose hem goes over
   the bottom stays outside; boots and shoes are always over the trousers;
3. tuck in: from top to bottom, each part goes under the ones above. It is pressed towards the skin
   but never goes below `FLOOR`: a garment does not enter the body (except under a
   boot, which replaces the leg: the floor goes down with it);
4. push out: from bottom to top, where the floor left no room (two tight parts one
   on the other), the upper part moves out, by what is missing and by `SORTIE_MAX` at
   most: a jacket does not puff up. A belt never squeezes the fabric: it is always pushed out;
5. hide: faces fully hidden by an upper part held to the body are removed. None is
   left to poke through when moving;
6. remove what pokes through: faces underneath left in front of an opaque face on top (flap, bow,
   ornament nothing could tuck away) are removed, except at the edge of the upper part where the part shows again;
7. placed accessories (necklace, bag…) move out above what they touch.

Two deliberate limits on the movement: a free flap (frill, skirt hem) more than
`FREE_MAX` in front of the upper part is not pushed in, nor is a volume too large to go
underneath (bow on the back of a dress under a jacket). What pokes through the upper part is
removed (6): the upper part hides the rest.
What still breaks the rule after adjustment is counted by the same rays and reported; the independent
check (`check_layers.py`) counts edge and face crossings instead.

Everything is in metres in the cache's glTF frame.
"""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import glb as GLB  # noqa: E402

import geom as G  # noqa: E402
import rays as R  # noqa: E402

# Adjustable thresholds: section layers.thresholds of layers.default.json (repository root), in millimetres, which
# settings.json overrides key by key; their meaning is written next to each one ("_help") and in the README.
_S = C.LAYERS["thresholds"]
MARGIN = _S["accessory_gap_mm"] / 1000             # gap of an accessory above what it touches
MARGIN_TUCKED = _S["layer_gap_mm"] / 1000  # gap between two layers, along a line
FLOOR = _S["floor_mm"] / 1000                  # height above the skin below which a part is not pressed
REACH_RADIUS = _S["ray_reach_mm"] / 1000          # beyond this, two parts are no longer one on the other
FREE_MAX = _S["free_flap_mm"] / 1000                # a free flap (skirt hem, frill) further than this from the upper part flares over it: it stays
OUTPUT_MAX = _S["max_outward_mm"] / 1000              # an upper part never moves out more than this; the rest is reported
RADIUS_LIMB = _S["limb_radius_mm"] / 1000          # curvature of a limb: an edge of length e drifts from it by e² / (8 × radius)
ARROW_MAX = _S["max_sag_mm"] / 1000
REACH_MASK = _S["hide_reach_mm"] / 1000        # beyond this, the upper part floats: one sees underneath through the opening
TOLERANCE_MASK = _S["hide_tolerance_mm"] / 1000
PASSES = int(_S["passes"])
THRESHOLD = _S["flag_depth_mm"] / 1000            # a point further than this in front of the upper part goes through it
PART_CROSSES = _S["flag_share"]                    # share of points going through from which a pair is reported
ORDER_OUTSIDE = _S["top_over_mm"] / 1000          # median outward offset from which a top is worn over the bottom

# Algorithm details, not settings: search ranges, geometric criteria, grids.
REACH_TUCKED = 0.05   # range of the search for a top's hem over the bottom (order)
REACH_SKIN = 0.06      # beyond this, the height above the skin is unknown
MATCH = 0.3            # two lines point the same way: otherwise it is the other side of a limb
END = 0.0002            # gap to the rule taken as zero
FINGERPRINT = 0.8         # |distance along the normal| / distance: the point is "under" the surface, not beside it
CONTACT = 0.005         # closer than this to the surface, the point's direction says nothing: it touches it
REACH_PAN = 0.3        # a free flap underneath that poked through the upper part is searched this far in front of it
REACH_HAIR = 0.06   # a strand deeper than this under a part stays there (it is elsewhere: under a flap, behind a bag)
PASSES_HAIR = 4
MARGIN_HAIR = 0.004
RANK_HEAD = 82          # from this rank on, an accessory is worn on the head or over everything
ON_HAIR = ("headwear", "hood", "headphones", "ears", "glasses", "mask")
UNDER_SKIN = 0.02        # a vertex lower than this under the skin is there on purpose
HOLLOW_MAX = 0.008       # the floor of a part with large faces does not rise higher than this
REACH = 0.025          # boxes further apart than this: the pair is not measured by traverse()
CELL = 0.08
VOXEL = 0.04
VOXEL_BODY = 0.012
NEIGHBORS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])

# covers.bin, one byte per face
OUTFIT, OPAQUE, OUTER = 1, 2, 4
COVERING = OUTFIT | OPAQUE

# Outfit rules: section layers.outfit of the settings.
_T = C.LAYERS["outfit"]
RANKS = {k: v for k, v in _T["ranks_in_outfit"].items()}   # a belt is worn over the bottom and under the open jacket, whatever its accessory rank
BOTTOM_CLOSED = tuple(t for t in _T["closed_bottoms"])      # they cover the underwear: it is not worn underneath
TOPS = tuple(t for t in _T["tops"])
BOTTOM = tuple(t for t in _T["bottoms"])

_parts, _lock = {}, threading.Lock()
_combos = {}


def _read_covers(path, n):
    try:
        with open(path, "rb") as f:
            o = np.frombuffer(f.read(), np.uint8)
    except FileNotFoundError:
        return None
    return o if len(o) == n else None


class Part:
    def __init__(self, ident):
        self.id = ident
        d = os.path.join(C.CACHE, "parts", ident)
        self.meta = C.arrange(C.read_json(os.path.join(d, "meta.json")))   # rank and tab: those of the settings
        doc, b = GLB.read(os.path.join(d, "part.glb"))
        accessory = doc["accessors"]
        views = doc["bufferViews"]

        def tab(i, ty, n):
            a, v = accessory[i], views[accessory[i]["bufferView"]]
            count = a["count"] * (1 if a.get("type") == "SCALAR" else n)  # indices are counted one by one
            return np.frombuffer(b, ty, count, v.get("byteOffset", 0)).reshape(-1, n).copy()
        pr = doc["meshes"][0]["primitives"]
        self.pos = tab(pr[0]["attributes"]["POSITION"], "<f4", 3).astype(float)
        self.nor = tab(pr[0]["attributes"]["NORMAL"], "<f4", 3).astype(float)
        self.F = np.concatenate([tab(p["indices"], "<u4", 3).astype(np.int64) for p in pr])
        self.rep, self.inv = G.weld(self.pos, 1e-5)
        self.name = self.meta.get("type_name") or ident
        self.flags = _read_covers(os.path.join(d, "covers.bin"), len(self.F))


def _date(ident):
    """Date of the part's file: a re-fitted part is no longer the one held in memory."""
    try:
        return os.stat(os.path.join(C.CACHE, "parts", ident, "part.glb")).st_mtime_ns
    except OSError:
        return 0


def part_name(ident):
    d = _date(ident)
    with _lock:
        if ident not in _parts or _parts[ident][0] != d:
            if len(_parts) > 40:
                _parts.clear()
            _parts[ident] = (d, Part(ident))
        return _parts[ident][1]


class Group:
    """A part of the final model, like a cache `Piece` (metres, local faces)."""

    def __init__(self, pos, nor, F, meta=None, ident="", flags=None):
        self.pos, self.nor, self.F = pos, nor, F
        self.rep, self.inv = G.weld(pos, 1e-5)
        self.meta, self.id, self.name = meta or {}, ident, (meta or {}).get("type_name") or ident
        self.flags = flags if flags is not None and len(flags) == len(F) else None


# ------------------------------------------------------------------ neighbourhood search
def _keys(v):
    return v[:, 0] * 1000003 + v[:, 1] * 1009 + v[:, 2]


def neighborhood(Q, P):
    """Mask of the points of Q close (a few cm) to a point of P."""
    v = np.unique(np.floor(P / VOXEL).astype(np.int64), axis=0)
    occ = np.unique(_keys((v[:, None] + NEIGHBORS[None]).reshape(-1, 3)))
    return np.isin(_keys(np.floor(Q / VOXEL).astype(np.int64)), occ)


def nearest(Q, V, F, N, reach, with_face=False):
    """Surface (V, F) seen from each point of Q: signed distance along the normal, normal,
    distance, point. With no surface within `reach`, the distance is inf. Faces are sorted
    into cells: only those of neighbouring cells are examined."""
    n = len(Q)
    s, d, nor, pts = np.zeros(n), np.full(n, np.inf), np.zeros((n, 3)), np.zeros((n, 3))
    face = np.full(n, -1, np.int64)
    if not n or not len(F):
        return (s, nor, d, pts, face) if with_face else (s, nor, d, pts)
    T = V[F]
    cen = T.mean(1)
    big = np.where(np.linalg.norm(T - cen[:, None], axis=2).max(1) > CELL - reach)[0]
    k_f = _keys(np.floor(cen / CELL).astype(np.int64))
    order = np.argsort(k_f)
    k_f = k_f[order]
    cq = np.floor(Q / CELL).astype(np.int64)
    ucq, inv = np.unique(cq, axis=0, return_inverse=True)
    inv = inv.ravel()
    tri = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[tri], np.arange(len(ucq) + 1))
    for i, c in enumerate(ucq):
        keys = np.unique(_keys(c[None] + NEIGHBORS))
        g, h = np.searchsorted(k_f, keys), np.searchsorted(k_f, keys, side="right")
        chunks = [order[a:b] for a, b in zip(g, h) if b > a]
        if len(big):
            chunks.append(big)
        if not chunks:
            continue
        candidate = np.unique(np.concatenate(chunks))
        idx = tri[bounds[i]:bounds[i + 1]]
        q, f, b, dd = G.plus_near(Q[idx], V, F[candidate], k=12, block=512)
        ff = F[candidate][f]
        m = (b[:, :, None] * N[ff]).sum(1)
        m /= np.maximum(np.linalg.norm(m, axis=1), 1e-12)[:, None]
        s[idx], nor[idx], d[idx], pts[idx] = ((Q[idx] - q) * m).sum(1), m, dd, q
        face[idx] = candidate[f]
    d[d > reach] = np.inf
    return (s, nor, d, pts, face) if with_face else (s, nor, d, pts)


_body = None


def body():
    """(points, normals) of the bare body, one point per voxel; None outside a complete cache."""
    global _body
    if _body is None:
        path = os.path.join(C.CACHE, "body", "body.glb")
        if not os.path.isfile(path):
            return None
        doc, b = GLB.read(path)
        accessory, views = doc["accessors"], doc["bufferViews"]
        at = doc["meshes"][0]["primitives"][0]["attributes"]

        def tab(i):
            return np.frombuffer(b, "<f4", accessory[i]["count"] * 3, views[accessory[i]["bufferView"]].get("byteOffset", 0)).reshape(-1, 3).astype(float)
        pos, nor = tab(at["POSITION"]), tab(at["NORMAL"])
        _, i = np.unique(np.floor(pos / VOXEL_BODY).astype(np.int64), axis=0, return_index=True)
        _body = (pos[i], nor[i])
    return _body


_skin = None


def skin():
    """(vertices, faces, normals, faces at the edge of an opening) of the bare body; None outside a complete cache."""
    global _skin
    if _skin is None:
        path = os.path.join(C.CACHE, "body", "body.glb")
        if not os.path.isfile(path):
            return None
        doc, b = GLB.read(path)
        accessory, views = doc["accessors"], doc["bufferViews"]

        def tab(i, ty, n):
            a, v = accessory[i], views[accessory[i]["bufferView"]]
            count = a["count"] * (1 if a.get("type") == "SCALAR" else n)
            return np.frombuffer(b, ty, count, v.get("byteOffset", 0)).reshape(-1, n)
        pr = doc["meshes"][0]["primitives"]
        V = tab(pr[0]["attributes"]["POSITION"], "<f4", 3).astype(float)
        N = tab(pr[0]["attributes"]["NORMAL"], "<f4", 3).astype(float)
        F = np.concatenate([tab(p["indices"], "<u4", 3).astype(np.int64) for p in pr])
        _, inv = G.weld(V, 1e-5)
        Fw = inv[F]
        E = np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1)
        u, nb = np.unique(E, axis=0, return_counts=True)
        to_border = np.zeros(int(Fw.max()) + 1, bool)
        to_border[u[nb == 1].ravel()] = True   # there, "inside" and "outside" no longer mean anything
        _skin = (V, F, N, to_border[Fw].any(1))
    return _skin


def lines(W, fold):
    """Line and height of each point of W above the skin: (unit directions, heights).
    The line is the direction from the closest skin point to the point (the gradient of the
    distance to the skin); the height is inf where the skin is too far or open, and the line
    is then `fold`."""
    n, h = fold.copy(), np.full(len(W), np.inf)
    pe = skin()
    if pe is None or not len(W):
        return n, h
    V, F, N, border = pe
    s, nor, d, q, f = nearest(W, V, F, N, REACH_SKIN, with_face=True)
    ok = np.isfinite(d)
    ok[ok] &= ~border[f[ok]]
    far = ok & (d > 0.002)
    sign = np.where(s[far] >= 0, 1.0, -1.0)
    n[far] = (W[far] - q[far]) / d[far, None] * sign[:, None]
    near = ok & ~far
    n[near] = nor[near]
    h[ok] = np.where(s[ok] >= 0, d[ok], -d[ok])
    return n, h


def _foot_body(Q, cp):
    """Closest body point to each point of Q, and its normal."""
    q, n = np.zeros((len(Q), 3)), np.zeros((len(Q), 3))
    if not len(Q):
        return q, n
    # wide cells (a body point is always close: garments follow it), fallback on the whole body
    grid_cell = 0.1
    cq = np.floor(Q / grid_cell).astype(np.int64)
    ucq, inv = np.unique(cq, axis=0, return_inverse=True)
    inv = inv.ravel()
    tri = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[tri], np.arange(len(ucq) + 1))
    k_c = _keys(np.floor(cp[0] / grid_cell).astype(np.int64))
    for i, c in enumerate(ucq):
        idx = tri[bounds[i]:bounds[i + 1]]
        candidate = np.where(np.isin(k_c, _keys(c[None] + NEIGHBORS)))[0]
        if not len(candidate):
            candidate = np.arange(len(cp[0]))
        for a in range(0, len(idx), 2048):
            b = idx[a:a + 2048]
            k = candidate[((Q[b, None] - cp[0][candidate][None]) ** 2).sum(-1).argmin(1)]
            q[b], n[b] = cp[0][k], cp[1][k]
    return q, n


def _normal_body(Q, cp):
    """Normal of the closest body point to each point of Q."""
    return _foot_body(Q, cp)[1]


def to_over(P, q, s, d, cp):
    """Is point P straight above q, seen from the body? The line q -> P must follow the body's
    normal: otherwise P is not under the part but beside it (a lower leg higher than the hem
    of a boot, an undergarment lower than the hem of a top). A point in contact always
    counts: that is where wide faces poke through the neighbouring surface."""
    if cp is None:
        return (np.abs(s) > FINGERPRINT * d) | (d < CONTACT)
    n = _normal_body(P, cp)
    v = P - q
    return (np.abs((v * n).sum(1)) > FINGERPRINT * d) | (d < CONTACT)


def _box(Z, Q, gap):
    return (Z.max(0) >= Q.min(0) - gap).all() and (Z.min(0) <= Q.max(0) + gap).all()


# ------------------------------------------------------------------ a layer being adjusted
class Layer:
    def __init__(self, p):
        self.p, self.id, self.meta, self.name = p, p.id, p.meta, p.name
        self.F, self.nor, self.rep, self.inv = p.F, p.nor, p.rep, p.inv
        self.Z = p.pos.copy()
        self.mask = np.zeros(len(p.F), bool)
        self.type = p.meta.get("type")
        self.hull = C.enveloping(p.meta)
        self.rank = RANKS.get(self.type, p.meta.get("rank", 50))
        if p.flags is not None:
            self.fl = p.flags
        else:
            T = p.pos[p.F]
            cen = T.mean(1)
            rad = cen - cen.mean(0)
            rad[:, 1] = 0.0
            outside = (np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]) * rad).sum(1) > 0
            self.fl = np.where(outside, COVERING | OUTER, COVERING).astype(np.uint8)
        self.Fext = p.F[(self.fl & OUTER) > 0]
        self.Fcouvre = p.F[((self.fl & COVERING) == COVERING) & ((self.fl & OUTER) > 0)]
        self.outfit = (self.fl & OUTFIT) > 0
        Fw = p.inv[p.F]
        self.E = np.unique(np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1), axis=0)

        # adjustment frame, per welded vertex: line, rest height, push-in along the line
        self.Fw = Fw
        nr = p.nor[p.rep]
        self.n = nr / np.maximum(np.linalg.norm(nr, axis=1), 1e-12)[:, None]
        self.h0 = np.full(len(p.rep), np.inf)
        self.a = np.zeros(len(p.rep))
        self.outfit_s = np.zeros(len(p.rep), bool)
        self.outfit_s[Fw[self.outfit].ravel()] = True
        self.pose = self.type in C.BELTS   # laid over the fabrics, never squeezed underneath
        # sag: between its vertices, a wide face drifts away from the curved surface it follows.
        # Two parts whose vertices all keep the rule still cross by this amount.
        W = p.pos[p.rep]
        e = np.zeros(len(p.rep))
        if len(self.E):
            lon = np.linalg.norm(W[self.E[:, 0]] - W[self.E[:, 1]], axis=1)
            np.maximum.at(e, self.E[:, 0], lon)
            np.maximum.at(e, self.E[:, 1], lon)
        self.arrow = np.minimum(e * e / (8 * RADIUS_LIMB), ARROW_MAX)
        self.hollow = np.minimum(e * e / (8 * RADIUS_LIMB), HOLLOW_MAX)   # how far a face dips below its vertices, on a limb
        self.version, self._grid = 0, None

    def seat(self, cp):
        """Takes the skin as the frame: lines and heights. Without skin, the part's normals.
        The frame depends only on the part at rest: it is kept with it."""
        if cp is None:
            return
        guard = getattr(self.p, "_lines", None)
        if guard is None:
            W = self.p.pos[self.rep]
            guard = lines(W, _normal_body(W, cp))
            try:
                self.p._lines = guard
            except AttributeError:
                pass
        self.n, self.h0 = guard

    def leave_of_skin(self):
        """What does not hide the skin must go in front of it: a transparent fabric, a free
        flap. Their vertices left under the skin (down to UNDER_SKIN) rise to the floor;
        lower, it is a part tucked in on purpose (a boot shaft in the leg), left alone."""
        show = np.zeros(len(self.p.rep), bool)
        show[self.Fw[(self.fl & COVERING) != COVERING].ravel()] = True
        need = FLOOR + self.hollow - self.h0
        where_i = show & np.isfinite(self.h0) & (need > 0) & (self.h0 > -UNDER_SKIN)
        if where_i.any():
            self.sink(-self.spread(np.where(where_i, need, 0.0)))

    @property
    def height(self):
        return self.h0 - self.a

    def grid(self):
        if self._grid is None or self._grid[0] != self.version:
            self._grid = (self.version, R.Grid(self.Z, self.F), self.n[self.Fw].sum(1))
        return self._grid[1], self._grid[2]

    def spread(self, a, rings=8, slope=0.75):
        """Spreads a push-in along the edges: the part moves as a whole, with no step or tear."""
        a = a.copy()
        if len(self.E):
            for _ in range(rings):
                for i, j in ((0, 1), (1, 0)):
                    where_i = np.where(slope * a[self.E[:, i]] > a[self.E[:, j]])[0]
                    a[self.E[where_i, j]] = slope * a[self.E[where_i, i]]
        return a

    def sink(self, a):
        """Moves each vertex by `a` along its line, towards the skin (or outwards if a < 0)."""
        self.a = self.a + a
        self.Z = self.p.pos - (self.a[:, None] * self.n)[self.inv]
        self.version += 1


def _probes(Z, Zu, reach):
    return (Z[:, 1] > Zu[:, 1].min() - reach) & (Z[:, 1] < Zu[:, 1].max() + reach) & neighborhood(Z, Zu)


def gaps(L, U):
    """What is missing for L to be under U: (push-in owed per vertex of L, push-out owed per vertex
    of U, ground of L), in metres, zero where the rule holds. Each gap is measured by a ray
    along a line and applies to both parts: either one can close it. The ground is the
    height below which L may go at that spot: under a part that is itself under the skin
    (a boot replaces the leg), it goes down with it; elsewhere it is inf (the floor)."""
    bL, bU, ground = np.zeros(len(L.rep)), np.zeros(len(U.rep)), np.full(len(L.rep), np.inf)
    if not len(L.F) or not len(U.F) or not _box(U.Z, L.Z, REACH_RADIUS):
        return bL, bU, ground
    WL, WU = L.Z[L.rep], U.Z[U.rep]
    hU = U.height

    def note(b_itself, itself, b_other, faces_other, f, need):
        np.maximum.at(b_itself, itself, need)
        np.maximum.at(b_other, faces_other[f].ravel(), np.repeat(need, 3))

    # 1. vertices of L: in front of U (ray towards the skin), or underneath but too close (ray outwards)
    sel = np.where(_probes(WL, U.Z, REACH_RADIUS))[0]
    if len(sel):
        g, nf = U.grid()
        h = L.height[sel]
        margin = MARGIN_TUCKED + L.arrow[sel]
        lon = np.minimum(np.where(np.isfinite(h), np.maximum(h, 0.0) + 0.003, REACH_RADIUS), REACH_RADIUS)
        for direction, length, last in ((-1.0, lon, True), (1.0, margin, False)):
            t, f = g.pull(WL[sel], direction * L.n[sel], length, last)
            ok = np.isfinite(t)
            ok[ok] &= (L.n[sel][ok] * nf[f[ok]]).sum(1) > MATCH
            need = (t + margin) if direction < 0 else (margin - t)
            ok &= L.outfit_s[sel] | (need < FREE_MAX)
            note(bL, sel[ok], bU, U.Fw, f[ok], need[ok])
            hf = hU[U.Fw[f[ok]]].min(1)
            bottom = np.isfinite(hf) & (hf - margin[ok] < FLOOR)
            np.minimum.at(ground, sel[ok][bottom], (hf - margin[ok])[bottom])
    # 2. vertices of U: behind a face of L (ray outwards), or in front but too close.
    #    Further than U can move out, it is no longer U's job to fix.
    sel = np.where(_probes(WU, L.Z, REACH_RADIUS))[0]
    if len(sel):
        g, nf = L.grid()
        for direction, length in ((1.0, FREE_MAX), (-1.0, MARGIN_TUCKED + ARROW_MAX)):
            t, f = g.pull(WU[sel], direction * U.n[sel], length)
            ok = np.isfinite(t)
            ok[ok] &= (U.n[sel][ok] * nf[f[ok]]).sum(1) > MATCH
            margin = np.zeros(len(sel))
            margin[ok] = MARGIN_TUCKED + L.arrow[L.Fw[f[ok]]].max(1)
            need = (t + margin) if direction > 0 else (margin - t)
            ok &= need > 0
            note(bU, sel[ok], bL, L.Fw, f[ok], need[ok])
            hs = hU[sel[ok]]
            bottom = np.isfinite(hs) & (hs - margin[ok] < FLOOR)
            for k in range(3):
                np.minimum.at(ground, L.Fw[f[ok]][bottom, k], (hs - margin[ok])[bottom])
    return bL, bU, ground


class _Pairs:
    """Gaps of the pairs, recomputed only when one of the two parts has moved."""

    def __init__(self):
        self.seen = {}

    def __call__(self, L, U):
        the_key, state = (id(L), id(U)), (L.version, U.version)
        if the_key not in self.seen or self.seen[the_key][0] != state:
            self.seen[the_key] = (state, gaps(L, U))
        return self.seen[the_key][1]


def go_under(L, over, pairs=None):
    """Pushes L under the parts `over`, without going below the floor. Where its height
    is unknown (flap far from the body), it is not pushed in by more than `FREE_MAX` in all.
    Returns the largest push-in made (metres)."""
    pairs = pairs or _Pairs()
    done = 0.0
    for _ in range(PASSES):
        # the floor rises with edge length: between two vertices lying flush with the skin,
        # the face of a part with large faces cuts under the curved skin (it showed in front)
        b, ground = np.zeros(len(L.rep)), FLOOR + L.hollow
        for U in over:
            e = pairs(L, U)
            b, ground = np.maximum(b, e[0]), np.minimum(ground, e[2])
        h = L.height
        place = np.where(np.isfinite(h), np.maximum(h - ground, 0.0), np.maximum(FREE_MAX - L.a, 0.0))
        a = np.minimum(b, place)
        if a.max(initial=0.0) < END:
            break
        L.sink(np.minimum(L.spread(a), np.maximum(place, 0.0)))
        done = max(done, float(a.max()))
    return done


def go_over(U, under, pairs=None):
    """Pushes U out by the gap left with the parts `under`, by `SORTIE_MAX` at most in all
    (`FREE_MAX` for a belt). Returns the largest push-out made."""
    pairs = pairs or _Pairs()
    done = 0.0
    bound = FREE_MAX if U.pose else OUTPUT_MAX   # a belt always moves out: it does not squeeze the fabric
    for _ in range(PASSES):
        b = np.zeros(len(U.rep))
        for L in under:
            b = np.maximum(b, pairs(L, U)[1])
        b = np.minimum(U.spread(b), np.maximum(bound + np.minimum(U.a, 0.0), 0.0))
        if b.max(initial=0.0) < END:
            break
        U.sink(-b)
        done = max(done, float(b.max()))
    return done


def tuck(L, U, cp=None):
    """Puts L under U. Returns the number of vertices moved."""
    before = L.a.copy()
    go_under(L, [U])
    return int((np.abs(L.a - before) > END).sum())


def hide(L, U, cp=None):
    """Marks the faces of L fully under a covering, body-held area of U."""
    if not len(U.Fcouvre) or not _box(U.Z, L.Z, REACH_MASK):
        return
    a_see = np.where(L.outfit & ~L.mask)[0]
    if not len(a_see):
        return
    T = L.Z[L.F[a_see]]
    pts = np.concatenate([T, T.mean(1)[:, None]], axis=1).reshape(-1, 3)
    near = _probes(pts, U.Z, REACH_MASK)
    ok = np.zeros(len(pts), bool)
    if near.any():
        s, _, d, q = nearest(pts[near], U.Z, U.Fcouvre, U.nor, REACH_MASK)
        v = np.isfinite(d) & (s < TOLERANCE_MASK)
        v[v] &= to_over(pts[near][v], q[v], s[v], d[v], cp)
        ok[near] = v
    k = np.where(ok.reshape(-1, 4).all(1))[0]
    opq = ((U.fl & OPAQUE) > 0) & ~U.mask
    if len(k) and opq.any():
        # at the edge of U the closest point is indeed on top, but from the front nothing is in front:
        # the face goes only if, along its normal, an opaque face of U comes before or after it
        g = R.Grid(U.Z, U.F[opq])
        n = np.cross(T[k, 1] - T[k, 0], T[k, 2] - T[k, 0])
        n = np.repeat(n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None], 4, axis=0)
        P = pts.reshape(-1, 4, 3)[k]
        P = (0.9 * P + 0.1 * P[:, 3:4]).reshape(-1, 3)   # slightly inwards: a vertex right under the edge of U would miss it
        vu = np.isfinite(g.pull(P, n, REACH_MASK)[0]) | np.isfinite(g.pull(P, -n, REACH_MASK)[0])
        k = k[vu.reshape(-1, 4).all(1)]
    L.mask[a_see[k]] = True


def remove_front(L, U):
    """Marks the faces of L left in front of U after adjustment: what neither the push-in (floor, free
    flap) nor U's push-out could tuck away, flaps and ornaments underneath poking through the top.
    A face goes if one of its vertices is in front of an opaque face of U (ray towards the skin) and
    the others are in front or underneath: at the edge of U, where the lower part comes out into the
    open, nothing is removed. What remains of it under U is not seen."""
    opq = ((U.fl & OPAQUE) > 0) & ((U.fl & OUTER) > 0) & ~U.mask
    if not opq.any() or not len(L.F) or not _box(U.Z, L.Z, REACH):
        return
    W = L.Z[L.rep]
    V = np.unique(L.Fw[~L.mask])
    if len(V):   # everything inside U's box: a flap that poked through may be far in front
        V = V[((W[V] >= U.Z.min(0) - REACH_PAN) & (W[V] <= U.Z.max(0) + REACH_PAN)).all(1)]
    if not len(V):
        return
    g, nf = R.Grid(U.Z, U.F[opq]), U.n[U.Fw[opq]].sum(1)
    h = L.height[V]
    lon = np.minimum(np.where(np.isfinite(h), np.maximum(h, 0.0) + 0.003, REACH_RADIUS), REACH_RADIUS)
    ahead = np.zeros(len(V), bool)
    # near the body: the last surface before the skin; far (free flap): the first one met
    for length, last in ((lon, True), (np.where(np.isfinite(h), lon, REACH_PAN), False)):
        t, f = g.pull(W[V], -L.n[V], length, last)
        d = np.isfinite(t) & (t > THRESHOLD)
        d[d] &= (L.n[V][d] * nf[f[d]]).sum(1) > MATCH
        ahead |= d
    if not ahead.any():
        return
    t2, f2 = g.pull(W[V], L.n[V], REACH_RADIUS)
    under = np.isfinite(t2)
    under[under] &= (L.n[V][under] * nf[f2[under]]).sum(1) > MATCH
    dv, ds = np.zeros(len(L.rep), bool), np.zeros(len(L.rep), bool)
    dv[V[ahead]], ds[V[ahead | under]] = True, True
    L.mask |= dv[L.Fw].any(1) & ds[L.Fw].all(1)


def push(p, Z, under):
    """Offset (all vertices of p) that moves p out to MARGE from the surfaces underneath:
    `under` = [(vertices, faces, normals)]."""
    W = Z[p.rep]
    d = np.zeros_like(W)
    for V, F, N in under:
        if not len(F):
            continue
        sel = np.where(neighborhood(W, V[F].reshape(-1, 3)))[0]
        if not len(sel):
            continue
        s, nor, dist, _ = nearest(W[sel] + d[sel], V, F, N, 0.03)
        a = np.isfinite(dist) & (s < MARGIN)
        extra = np.zeros_like(d)
        extra[sel[a]] = (MARGIN - s[a])[:, None] * nor[a]
        d += extra
    Fw = p.inv[p.F]
    E = np.unique(np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1), axis=0)
    if len(E):
        for _ in range(6):
            m = np.linalg.norm(d, axis=1)
            for a, b in ((0, 1), (1, 0)):
                where_i = np.where(0.72 * m[E[:, a]] > m[E[:, b]])[0]
                d[E[where_i, b]] = 0.72 * d[E[where_i, a]]
    return d[p.inv]


def hairstyle(name, head):
    """Cached hair placed on the head `head`: (vertices, normals, faces), in metres."""
    d = os.path.join(C.CACHE, "hairstyles", name)
    doc, b = GLB.read(os.path.join(d, "hairstyle.glb"))
    accessory, views = doc["accessors"], doc["bufferViews"]

    def tab(i, ty, n):
        a, v = accessory[i], views[accessory[i]["bufferView"]]
        return np.frombuffer(b, ty, a["count"] * (1 if a.get("type") == "SCALAR" else n), v.get("byteOffset", 0)).reshape(-1, n)
    pr = doc["meshes"][0]["primitives"]
    V = tab(pr[0]["attributes"]["POSITION"], "<f4", 3).astype(float)
    N = tab(pr[0]["attributes"]["NORMAL"], "<f4", 3).astype(float)
    F = np.concatenate([tab(q["indices"], "<u4", 3).astype(np.int64) for q in pr])
    pose = os.path.join(d, "%s.bin" % head)
    if os.path.exists(pose):
        Z = np.fromfile(pose, "<f4").reshape(-1, 3).astype(float)
        if len(Z) == len(V):
            V = Z
    return V, N, F


def under_hair(c):
    """The hair falls over this part (not a hat, glasses, a hood)."""
    return c.type not in ON_HAIR and (c.hull or c.rank < RANK_HEAD)


def style(layers, V, N, F):
    """Offset of the hairstyle's vertices (V, N, F: placed on the head, metres) that makes it
    fall over the adjusted outfit: a vertex under the outer face of a part (straight under it,
    not beside it) moves out to MARGIN_HAIR along the part's normal, by the shortest way; the
    strand follows as a whole. A strand deeper than REACH_HAIR stays where it is."""
    on = [c for c in layers if under_hair(c) and _box(V, c.Z, 0.0)]
    if not on or not len(F):
        return np.zeros_like(V)
    g = Group(V, N, F)
    W = V[g.rep]
    d = np.zeros_like(W)
    Fw = g.inv[F]
    E = np.unique(np.sort(np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]), axis=1), axis=0)
    for _ in range(PASSES_HAIR):
        moves = False
        for c in on:
            Fc = c.F[((c.fl & OUTER) > 0) & ~c.mask] if len(c.Fext) else c.F[~c.mask]
            if not len(Fc):
                continue
            P = W + d
            sel = np.where(neighborhood(P, c.Z[Fc].reshape(-1, 3)))[0]
            if not len(sel):
                continue
            s, nor, dist, _ = nearest(P[sel], c.Z, Fc, c.nor, REACH_HAIR)
            a = np.isfinite(dist) & (s < MARGIN_HAIR) & ((np.abs(s) > FINGERPRINT * dist) | (dist < CONTACT))
            if a.any():
                d[sel[a]] += (MARGIN_HAIR - s[a])[:, None] * nor[a]
                moves = True
        if not moves:
            break
        for _ in range(10):                      # the strand follows: no step between two neighbouring vertices
            m = np.linalg.norm(d, axis=1)
            for i, j in ((0, 1), (1, 0)):
                where_i = np.where(0.85 * m[E[:, i]] > m[E[:, j]])[0]
                d[E[where_i, j]] = 0.85 * d[E[where_i, i]]
    return d[g.inv]


def crosses(L, U, cp=None):
    """Share of the visible vertices of L that are in front of U, among those U covers, and depth
    of the worst. Measured by rays along the lines. Returns None if the parts do not overlap."""
    V = np.unique(L.Fw[~L.mask])
    if not len(V) or not len(U.F) or not _box(U.Z, L.Z, REACH):
        return None
    W = L.Z[L.rep]
    V = V[_probes(W[V], U.Z, REACH_RADIUS)]
    if not len(V):
        return None
    g, nf = U.grid()
    h = L.height[V]
    lon = np.minimum(np.where(np.isfinite(h), np.maximum(h, 0.0) + 0.003, REACH_RADIUS), REACH_RADIUS)
    t, f = g.pull(W[V], -L.n[V], lon, True)
    outside = np.isfinite(t) & (t > THRESHOLD)
    outside[outside] &= (L.n[V][outside] * nf[f[outside]]).sum(1) > MATCH
    outside &= L.outfit_s[V] | (t < FREE_MAX)
    t2, f2 = g.pull(W[V], L.n[V], REACH_RADIUS)
    covered = np.isfinite(t2)
    covered[covered] &= (L.n[V][covered] * nf[f2[covered]]).sum(1) > MATCH
    near = outside | covered
    if near.sum() < 20:
        return None
    return {"vertices": int(near.sum()), "fraction": round(float(outside.sum() / near.sum()), 4),
            "worst_mm": round(float(t[outside].max() * 1000), 1) if outside.any() else 0.0}


# ------------------------------------------------------------------ the outfit
def removed(metas):
    """{id: why} of the parts not worn in this outfit."""
    types = {m.get("type"): m for m in metas.values()}
    out = {}
    for i, m in metas.items():
        if m.get("type") == "underwear":
            for t in BOTTOM_CLOSED:
                if t in types:
                    out[i] = "%s removed: the %s covers it" % (m.get("type_name") or "underwear", types[t].get("type_name") or t)
    return out


def output_median(H, B, cp=None):
    """Median outward offset of H from B where they overlap (metres), None without overlap."""
    if not len(B.Fext):
        return None
    W = H.Z[H.rep]
    sel = np.where(_probes(W, B.Z, REACH_TUCKED))[0]
    if not len(sel):
        return None
    s, _, d, q = nearest(W[sel], B.Z, B.Fext, B.nor, REACH_TUCKED)
    near = np.isfinite(d)
    near[near] &= to_over(W[sel][near], q[near], s[near], d[near], cp)
    return float(np.median(s[near])) if near.sum() >= 20 else None


def order(layers, cp=None):
    """Sorts from inside to outside; a top whose hem goes over the bottom is worn outside. A
    sock that stops below the underwear's waist (stocking, not tights) goes underneath."""
    notes = []
    for C_ in layers:
        if C_.type == "underwear":
            for B in layers:
                if B.type == "socks" and B.Z[:, 1].max() < C_.Z[:, 1].max():
                    B.rank = min(B.rank, C_.rank - 0.5)
    for H in layers:
        if H.type not in TOPS:
            continue
        for B in layers:
            if B.type in BOTTOM:
                m = output_median(H, B, cp)
                if m is not None and m > ORDER_OUTSIDE:
                    H.rank = max(H.rank, B.rank + 0.5)
                    notes.append("%s worn over %s" % (H.name, B.name))
    return sorted(layers, key=lambda c: c.rank), notes


def adjust(env):
    """Puts the wrapping parts (sorted from inside to outside) in order along each line."""
    pairs = _Pairs()
    for i in range(len(env) - 2, -1, -1):      # top to bottom: each one is pressed under the ones above
        if not env[i].pose:
            go_under(env[i], [U for U in env[i + 1:] if not U.pose], pairs)
    for i in range(1, len(env)):               # bottom to top: what found no room moves out
        go_over(env[i], env[:i], pairs)


def resolve(parts):
    """Adjusts a list of Part/Group objects. Returns (sorted layers, notes)."""
    cp = body()
    layers, notes = order([Layer(p) for p in parts], cp)
    env = [c for c in layers if c.hull]
    for c in env:
        c.seat(cp)
        c.leave_of_skin()
    adjust(env)
    for i, L in enumerate(env):
        for U in env[i + 1:]:
            hide(L, U, cp)
    for i, L in enumerate(env):
        for U in env[i + 1:]:
            remove_front(L, U)
    under = []
    for c in layers:
        if not c.hull and under:
            d = push(c, c.Z, under)
            c.Z = c.Z + d
        under.append((c.Z, c.Fext if len(c.Fext) else c.F, c.nor))
    return layers, notes


def flag(layers):
    out = []
    env = [c for c in layers if c.hull]
    for i, L in enumerate(env):
        for U in env[i + 1:]:
            r = crosses(L, U)
            if r:
                out.append({"under": L.id, "over": U.id, "name_under": L.name, "name_over": U.name, **r})
    return out


def part_tucked(c):
    """Share of a part's vertices moved under another (by more than 1 mm): they are no longer seen."""
    if not c.hull or c.type in C.BELTS:
        return 0.0
    return float((np.linalg.norm(c.Z - c.p.pos, axis=1) > 0.001).mean())


def crossing_visible(value, part_hidden, threshold=0.031):
    """Skin poking through a part where another part covers it is not seen."""
    return value > threshold + part_hidden


def resolve_pmx(P, infos, hair=()):
    """Applies the preview's adjustment to the PMX. `infos`: per part, id, meta and materials (names);
    the flags come from the cache. Moves the vertices and removes the hidden faces; the
    hair materials fall over the adjusted outfit.
    Returns (checked layers, notes, number of faces removed)."""
    pos = np.array(P["vertices"]["pos"], float) * C.METER
    nor = np.array(P["vertices"]["nor"], float)
    faces = np.array(P["faces"], np.int64)
    start, per_name, ranges = 0, {}, {}
    for m in P["materials"]:
        n = m["n_indices"] // 3
        per_name[m["name"]] = faces[start:start + n]
        ranges[m["name"]] = (start, start + n)
        start += n
    parts, vertices, local = [], {}, {}
    for info in infos:
        mats = [n for n in info["mats"] if n in per_name and len(per_name[n])]
        if not mats:
            continue
        F = np.concatenate([per_name[n] for n in mats])
        vs, loc = np.unique(F, return_inverse=True)
        Fl = loc.reshape(-1, 3)
        fl = _read_covers(os.path.join(C.CACHE, "parts", info["id"], "covers.bin"), len(Fl))
        parts.append(Group(pos[vs], nor[vs], Fl, info.get("meta"), info["id"], fl))
        vertices[info["id"]], local[info["id"]] = vs, mats
    layers, notes = resolve(parts)
    guard = np.ones(len(faces), bool)
    for c in layers:
        pos[vertices[c.id]] = c.Z
        k = 0
        for n in local[c.id]:
            a, b = ranges[n]
            guard[a:b] = ~c.mask[k:k + b - a]
            k += b - a
    Fch = [per_name[n] for n in hair if n in per_name and len(per_name[n])]
    if Fch and layers:
        Fch = np.concatenate(Fch)
        vs, loc = np.unique(Fch, return_inverse=True)
        pos[vs] += style(layers, pos[vs], nor[vs], loc.reshape(-1, 3))
    faces_kept, removes = [], 0
    for m in P["materials"]:
        a, b = ranges[m["name"]]
        rest = int(guard[a:b].sum())
        removes += b - a - rest
        m["n_indices"] = rest * 3
        faces_kept.append(faces[a:b][guard[a:b]])
    P["faces"] = [tuple(int(x) for x in f) for f in (np.concatenate(faces_kept) if faces_kept else faces[:0])]
    P["vertices"]["pos"] = [tuple(float(x) for x in v) for v in pos / C.METER]
    return layers, notes, removes


# ---------------------------------------------------------------------- feet posed in a shoe
# A shoe with a heel was fitted on a body whose feet take its pose (clothes/foot.py) and is stored
# in the rest frame. The outfit is adjusted in the rest frame, then the body and every part are posed:
# the shoe with its own weights (pied.bin), the other parts with those of the closest skin.
BOTTOM_POSE = 0.6      # metres: above this, nothing follows the feet
_foot_cache = None


def _foot():
    """(chains, wa, wt, side) of the bare body, per vertex of corps.glb; None without a foot chain."""
    global _foot_cache
    if _foot_cache is None or _foot_cache[0] != C.BODY:
        import foot as PD
        import pmx as PMX
        Pc = PMX.read(C.BODY)
        ch = PD.chains(Pc["bones"])
        wa, wt, side = PD.weight_pmx(Pc, ch)
        T = PMX.to_arrays(Pc)
        vs = np.unique(T["faces"][T["mat_face"] == [m["name"] for m in Pc["materials"]].index("肌")])
        _foot_cache = (C.BODY, (ch, wa[vs], wt[vs], side[vs]) if ch else None)
    return _foot_cache[1]


def place_gltf(V, wa, wt, side, ch, pose):
    """Points of the cache's glTF frame (metres, z flipped) posed by foot.py, which works in the PMX frame."""
    import foot as PD
    X = V / C.METER
    X[:, 2] *= -1
    Y = PD.place(X, wa, wt, side, PD.transforms(ch, pose)) * C.METER
    Y[:, 2] *= -1
    return Y


def _plus_near(Q, P):
    """Index of the closest point of P to each point of Q (by blocks)."""
    out = np.zeros(len(Q), np.int64)
    for a in range(0, len(Q), 2048):
        d = ((Q[a:a + 2048, None, :] - P[None]) ** 2).sum(2)
        out[a:a + 2048] = d.argmin(1)
    return out


def pose_outfit(layers):
    """The feet's pose of an outfit: that of its shoe, None without one."""
    return next((c.meta["foot"] for c in layers if c.meta.get("foot")), None)


def place_outfit(layers):
    """(body offset, {id: posed vertices}) of an outfit worn with a posed shoe; (None, {}) otherwise."""
    pose, pd, pe = pose_outfit(layers), _foot(), skin()
    if not pose or pd is None or pe is None:
        return None, {}
    ch, wa, wt, side = pd
    Vb = pe[0]
    body_d = place_gltf(Vb, wa, wt, side, ch, pose) - Vb
    bottom = np.where(Vb[:, 1] < BOTTOM_POSE)[0]
    poses = {}
    for c in layers:
        Z = c.Z
        q = None
        try:
            q = np.fromfile(os.path.join(C.CACHE, "parts", c.id, "foot.bin"), "<f4").reshape(-1, 3)
        except (OSError, ValueError):
            pass
        if q is not None and len(q) == len(Z):
            a, t, k = q[:, 0].astype(float), q[:, 1].astype(float), q[:, 2].astype(np.int64)
        else:
            m = np.where(Z[:, 1] < BOTTOM_POSE)[0]
            if not len(m) or not len(bottom):
                continue
            a, t, k = np.zeros(len(Z)), np.zeros(len(Z)), np.full(len(Z), -1)
            j = bottom[_plus_near(Z[m], Vb[bottom])]
            a[m], t[m], k[m] = wa[j], wt[j], side[j]
        if not (a + t > 0).any():
            continue
        poses[c.id] = place_gltf(Z, a, t, k, ch, pose)
    return body_d, poses


def combine(ids, hairdo=None, head=None):
    """ids (in any order) -> {"order", "offsets" {id: Float32 bytes},
    "masques" {id: uint8 bytes per face}, "retirees" {id: why}, "notes", "signal",
    "coiffure": Float32 bytes of the offset of the hair `coif` placed on `head`, or None}."""
    the_key = "|".join("%s@%d" % (i, _date(i)) for i in sorted(ids))   # once re-fitted, a part changes key
    the_key += "|cheveux:%s:%s" % (hairdo, head) if hairdo else ""   # fr-ok: part of the cache key, kept so caches stay valid
    with _lock:
        if the_key in _combos:
            return _combos[the_key]
    parts = {i: part_name(i) for i in ids}
    ret = removed({i: p.meta for i, p in parts.items()})
    layers, notes = resolve([p for i, p in parts.items() if i not in ret])
    offset, masks = {}, {}
    body_d, poses = place_outfit(layers)
    for c in layers:
        d = poses.get(c.id, c.Z) - c.p.pos
        if np.abs(d).max() > 1e-6:
            offset[c.id] = d.astype("<f4").tobytes()
        if c.mask.any():
            masks[c.id] = c.mask.astype(np.uint8).tobytes()
    signal = [s for s in flag(layers) if s["fraction"] >= PART_CROSSES]
    hair = None
    if hairdo and os.path.isfile(os.path.join(C.CACHE, "hairstyles", hairdo, "hairstyle.glb")):
        d = style(layers, *hairstyle(hairdo, head or hairdo))
        if np.abs(d).max() > 1e-6:
            hair = d.astype("<f4").tobytes()
    res = {"order": [c.id for c in layers], "offsets": offset, "masks": masks, "removed": ret,
           "notes": list(ret.values()) + notes, "signal": signal, "hairstyle": hair,
           "body": body_d.astype("<f4").tobytes() if body_d is not None else None}
    with _lock:
        if len(_combos) > 200:
            _combos.clear()
        _combos[the_key] = res
    return res

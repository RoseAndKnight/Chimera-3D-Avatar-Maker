"""The grid of the remade body: its topology (our own) and the first placement of each vertex on the
reference.

Everything is quads, in tubes whose rows are perpendicular to the bones:

- trunk: NT columns, horizontal rows from the crotch to the armpits; above, the rows rise
  with the shoulder line and stop at the armhole (a notch of Q columns on the side, like a
  T-shirt); the last row closes on itself along the shoulder (seam of Z edges) and leaves
  the neckline, where the neck starts;
- arms: the first row is the edge of the notch; hand: palm, thumb in a window of the palm,
  four fingers sharing the last row of the palm;
- legs: the last row of the trunk splits in two through a crotch strip; foot: the rows
  turn around the instep, then five toes share the last row.

An "anchored" vertex is placed by a cut of the reference (contour around the bone, equal steps). The others
(junctions) are placed afterwards by `fit.py`: interpolation between their neighbours, then relaxation on the
surface. The right side is the exact mirror of the left.
"""
import numpy as np

import cut as CP
import geom as G

NT = 72            # trunk columns (0 = front middle, NT/4 = left side, NT/2 = middle of the back)
Q = 6              # width of the armhole notch, in columns
P_ARMHOLE = 10         # height of the notch, in rows
Z = 7              # edges of the shoulder seam
NA = 2 * P_ARMHOLE + Q # arm columns (26)
NN = NT - 2 * (Q + 2 * Z)   # neck columns (32)
W = 2              # width of the crotch strip
MB = 8             # length of the crotch strip
NL = NT // 2 - W + MB        # leg columns (42)
FINGERS = (3, 3, 3, 3)       # edges on top of each finger, from index to little finger (rings of 8)
TOES = (6, 4, 4, 3, 3)   # from the big toe to the little one
NP = 10            # rows of the palm, from the wrist to the fingers
FEN_R, FEN_C = 5, 17   # thumb window in the palm: rows FEN_R..FEN_R+2, columns FEN_C..FEN_C+2
NF = 12            # fan rows around the instep (from the ankle plane to the vertical plane)
assert 2 * sum(FINGERS) + 2 == NA and 2 * sum(TOES) + 2 == NL

# Landmark heights, in cm: those of the ZZZ reference body. Another reference gives its own
# (R.landmarks, see landmarks()): nothing else in the construction is an absolute height.
LANDMARKS = {
    "y_crotch": 89.0,          # first row of the trunk (the body is still a single contour there)
    "y_pelvis": 108.0,             # where the side column leaves the middle of the pelvis towards the underside of the arm
    "neck_sweep": (150.0, 138.0),  # heights where to look for the base of the neck, top to bottom
    "y_neck_default": 142.5,
    "y_top": None,                # last row of the neck; None: the top of the body skin
    "toe_tour_max": 8.0,        # circumference below which a vertical cut of the foot is a toe
}


def landmarks(R):
    """The same landmarks for another body, taken from its skeleton and its skin (offsets measured on ZZZ)."""
    o = R.os
    V = R.V[np.unique(R.F[R.f_body])]
    middle = V[np.abs(V[:, 0]) < 0.5]
    bottom = float(middle[middle[:, 1] > 0.5 * o["左足"][1]][:, 1].min())      # the bottom of the crotch
    return {"y_crotch": bottom + 1.8, "y_pelvis": float(o["上半身"][1]) + 5.8,
            "neck_sweep": (float(o["頭"][1]) + 0.6, float(o["首"][1]) - 4.6),
            "y_neck_default": float(o["首"][1]) - 0.1, "y_top": _bottom_of_border(R, float(o["首"][1])) - 0.25}


def _bottom_of_border(R, y_min):
    """The lowest point of the skin boundary above y_min: where the neck stops being closed."""
    F = R.F[R.f_body]
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, c = np.unique(e, axis=0, return_counts=True)
    y = R.V[np.unique(u[c == 1])][:, 1]
    return float(y[y > y_min].min())
J_A = 32              # armpit row
J_TOP = J_A + P_ARMHOLE
S_ARMHOLE = 6.0           # along the arm from the shoulder: where the arm is a contour of its own
STEP_ARM = 1.15
STEP_LEG = 1.5
STEP_NECK = 1.0
STEP_FINGER = 0.7
STEP_TOE = 0.6


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Mesh:
    def __init__(self):
        self.the_key = {}          # key -> index
        self.keys = []
        self.pos = []          # initial position or None
        self.anchor = []
        self.zone = []         # what the vertex lands on: "body", "hand", "toe" (the body), "none" (smoothed only)
        self.quads = []
        self.alias = {}
        self.map_ = []        # (map name, u, v) for the texture coordinates

    def canonical(self, k):
        while k in self.alias:
            k = self.alias[k]
        return k

    def s(self, k, pos=None, anchor=False, zone="body", map_=None):
        k = self.canonical(k)
        i = self.the_key.get(k)
        if i is None:
            i = len(self.keys)
            self.the_key[k] = i
            self.keys.append(k)
            self.pos.append(None)
            self.anchor.append(False)
            self.zone.append(zone)
            self.map_.append(None)
        if pos is not None:
            self.pos[i] = np.asarray(pos, float)
            self.anchor[i] = anchor
            self.zone[i] = zone
        if map_ is not None and self.map_[i] is None:
            self.map_[i] = map_
        return i

    def quad(self, a, b, c, d, map_=None, reverse=False):
        """Four keys. map_: (name, [(u, v) of the four corners]) for the texture coordinates."""
        ids = [self.s(k) for k in (a, b, c, d)]
        if len(set(ids)) == 4:
            if reverse:
                ids = ids[::-1]
                map_ = None if map_ is None else (map_[0], map_[1][::-1])
            self.quads.append((ids, map_))

    def grid(self, key_of, ranks, cols, closed, map_, u0=0, exists=None, reverse=False):
        """Quads of a grid: key_of(r, c). ranks: list of consecutive rows; cols: number of columns;
        closed: the last column joins the first."""
        nc = cols if closed else cols - 1
        for r0, r1 in zip(ranks[:-1], ranks[1:]):
            for c in range(nc):
                c1 = (c + 1) % cols if closed else c + 1
                if exists is not None and not exists(r0, c):
                    continue
                self.quad(key_of(r0, c), key_of(r0, c1), key_of(r1, c1), key_of(r1, c),
                          (map_, [(u0 + c, r0), (u0 + c + 1, r0), (u0 + c + 1, r1), (u0 + c, r1)]), reverse)


def mirror(p):
    return np.asarray(p) * np.array([-1.0, 1.0, 1.0])


def symmetrize(P):
    """Closed row of n points, the first on the symmetry plane: made symmetric (i <-> n - i)."""
    n = len(P)
    out = P.copy()
    for i in range(n):
        j = (-i) % n
        out[i] = 0.5 * (P[i] + mirror(P[j]))
    out[0, 0] = 0.0
    if n % 2 == 0:
        out[n // 2, 0] = 0.0
    return out


class Site_:
    """Builds the mesh on a reference (unfolded reference.Reference)."""

    def __init__(self, R, log=print):
        self.R = R
        self.j = log
        self.M = Mesh()
        self.V = R.V
        self.Fc = R.F[R.f_body]
        self.Fm = R.F[R.f_main["G"]]
        self.os = R.os
        self.info = {}
        self.rep = dict(LANDMARKS, **getattr(R, "landmarks", {}))

    # ------------------------------------------------------------------ trunk
    def key_trunk(self, j, i):
        return ("T", j, i % NT)

    def line_shoulder(self):
        """Height of the shoulder line y(x) between the neck and the armhole, on the reference (left side)."""
        o = self.os
        a, b = o["左腕"], o["左ひじ"]
        n = unit(b - a)
        c = a + S_ARMHOLE * n
        self.armhole_c, self.armhole_n = c, n
        V, F = self.V, self.Fc
        cf = V[F].mean(1)
        # faces on the trunk side: not the arm beyond the armhole plane (nor its mirror); the arm is
        # recognized by its weights
        def of_arm(side, sign):
            names = [k for k in o if k.startswith(side) and any(m in k for m in ("腕", "ひじ", "手"))]
            w = self.R.weight_os(names)[F].mean(1)
            return (w > 0.5) & (((cf * [sign, 1, 1]) - c) @ n > 0)
        arm_g, arm_d = of_arm("左", 1), of_arm("右", -1)
        self.F_trunk = F[~(arm_g | arm_d)]
        return c, n

    def trunk(self):
        M, V = self.M, self.V
        c_armhole, n_armhole = self.line_shoulder()
        Ft = self.F_trunk
        axe_z = 3.0
        # --- armhole ring: cut of the arm at S_EMM, first point at the top, towards the back
        r = CP.loop(V, self.Fc, c_armhole, n_armhole, [0, 1, 0], NA)
        assert r is not None, "armhole"
        armhole, per = r
        self.info["armhole_cm"] = round(per, 1)
        self.armhole = armhole
        # matching of the ring with the notch (columns ci..ci+Q, rows J_A..J_TOP)
        ci = NT // 4 - Q // 2          # 15
        cs = ci + Q                    # 21
        self.ci, self.cs = ci, cs
        keys_armhole = [("T", J_TOP, cs)]                                  # c = 0: the acromion (ci and cs stitched)
        keys_armhole += [("T", j, cs) for j in range(J_TOP - 1, J_A - 1, -1)]   # back, going down
        keys_armhole += [("T", J_A, i) for i in range(cs - 1, ci, -1)]         # bottom of the armpit
        keys_armhole += [("T", j, ci) for j in range(J_A, J_TOP)]              # front, going up
        assert len(keys_armhole) == NA
        # shoulder seam: (J_TOP, ci - k) = (J_TOP, cs + k), and its mirror
        for k in range(Z + 1):
            M.alias[("T", J_TOP, cs + k)] = ("T", J_TOP, ci - k)
            M.alias[("T", J_TOP, (NT - cs - k) % NT)] = ("T", J_TOP, (NT - ci + k) % NT)
        self.keys_armhole = keys_armhole
        for k, p in zip(keys_armhole, armhole):
            M.s(k, p, True)
            kk = M.canonical(k)
            M.s(("T", kk[1], (NT - kk[2]) % NT), mirror(p), True)
        top_armhole = armhole[0]                       # acromion
        z_arm = float(armhole[NA // 2 - 2:NA // 2 + 3, 2].mean())
        y_a = float(armhole[:, 1].min()) - 0.3      # armpit height: just below the arm
        self.info["y_armpit"] = round(y_a, 1)
        # --- horizontal rows, from the crotch to the armpit
        ys = np.linspace(self.rep["y_crotch"], y_a, J_A + 1)
        for j in range(J_A + 1):
            # the side column follows a line that starts at the middle of the pelvis and joins the underside of the arm
            yb = self.rep["y_pelvis"]
            t = np.clip((ys[j] - yb) / (y_a - yb), 0.0, 1.0)
            zm = self.z_middle(ys[j])
            P = self.rank_trunk(ys[j], zm, (1 - t) * zm + t * z_arm)
            for i in range(NT):
                if j == J_A and (ci <= i <= cs or NT - cs <= i <= NT - ci):
                    continue            # bottom of the notch: it is the arm ring
                M.s(("T", j, i), P[i], True)
        # --- neck: horizontal rows from the base
        y_neck0 = self.base_neck()
        y_end = self.rep["y_top"]
        if y_end is None:
            y_end = float(V[np.unique(self.Fc)][:, 1].max()) - 0.25
        n_neck = int(round((y_end - y_neck0) / STEP_NECK))
        self.n_neck = n_neck
        neck = []
        for r_ in range(n_neck + 1):
            y = y_neck0 + (y_end - y_neck0) * r_ / n_neck
            c = np.array([0.0, y, self.z_neck(y)])
            rr = CP.loop(V, Ft, c, [0, -1, 0], [0, 0, -1], NN)
            assert rr is not None, ("neck", r_)
            neck.append(symmetrize(rr[0]))
        self.info["y_base_neck"] = round(y_neck0, 1)
        # the neckline: front i = -(NT/4 - Q/2 - Z) .. +; back i = cs + Z .. NT - cs - Z
        d = ci - Z                      # 8: half-width of the neckline at the front, in columns
        collar_neck = list(range(0, d + 1)) + list(range(cs + Z + 1, NT - cs - Z)) + list(range(NT - d, NT))
        assert len(collar_neck) == NN, len(collar_neck)
        self.collar_neck = collar_neck

        def key_neck(r_, c):
            return ("T", J_TOP, collar_neck[c % NN]) if r_ == 0 else ("N", r_, c % NN)
        self.key_neck = key_neck
        for r_ in range(n_neck + 1):
            for c in range(NN):
                M.s(key_neck(r_, c), neck[r_][c], True)
        # --- shoulder seam: from the acromion to the corner of the neckline, on the ridge
        corner = neck[0][d]
        crest = self.crest(top_armhole, corner, Z + 1)
        for k in range(1, Z):
            M.s(("T", J_TOP, ci - k), crest[k], True)
            M.s(("T", J_TOP, (NT - ci + k) % NT), mirror(crest[k]), True)
        # --- rows above the armpit: left free (placed by the fitting step), between the armhole, the
        # armpit row, the seam and the neckline
        for j in range(J_A + 1, J_TOP):
            for i in range(NT):
                if ci < i < cs or NT - cs < i < NT - ci:
                    continue
                M.s(("T", j, i))
        # --- quads

        def exists(j, i):
            if j >= J_A and (ci <= i < cs or NT - cs <= i < NT - ci):
                return False
            return True
        M.grid(self.key_trunk, list(range(J_TOP + 1)), NT, True, "trunk", exists=exists, reverse=True)
        M.grid(key_neck, list(range(n_neck + 1)), NN, True, "neck", reverse=True)

    def rank_trunk(self, y, zm, z_side):
        """Horizontal row of the trunk: NT points, symmetric; column NT/4 is on the side at z = z_cote,
        equal steps in front of it, equal steps behind."""
        c = np.array([0.0, y, zm])
        r = CP.outline_around(self.V, self.F_trunk, c, [0, -1, 0], [0, 0, -1])
        assert r is not None, ("trunk", y)
        K = 720
        D = symmetrize(CP.resample(r[0], K, c, r[1], r[2], 0.0))
        H = D[:K // 2 + 1]                          # left half, from front to back
        dz = H[:, 2] - z_side
        crossed = np.flatnonzero((dz[:-1] < 0) & (dz[1:] >= 0))
        assert len(crossed), ("side", y)
        s = int(crossed[np.argmax(H[crossed, 0])]) + 1

        def step_equal(A, k):
            L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(A, axis=0), axis=1))])
            t = np.linspace(0, L[-1], k + 1)
            return np.stack([np.interp(t, L, A[:, a]) for a in range(3)], 1)
        G_ = np.concatenate([step_equal(H[:s + 1], NT // 4), step_equal(H[s:], NT // 4)[1:]])   # NT/2 + 1 points
        P = np.zeros((NT, 3))
        P[:NT // 2 + 1] = G_
        P[NT // 2 + 1:] = (G_[1:NT // 2] * [-1, 1, 1])[::-1]
        return P

    def z_middle(self, y):
        """Front-to-back middle of the trunk contour at height y (where it crosses the plane x = 0)."""
        zs = []
        for P, closed in CP.cut_plan(self.V, self.F_trunk, np.array([0.0, y, 0.0]), [0, -1, 0]):
            if closed and P[:, 0].min() < 0 < P[:, 0].max():
                zs += [P[:, 2].min(), P[:, 2].max()]
        return 0.5 * (min(zs) + max(zs))

    def z_neck(self, y):
        o = self.os
        a, b = o["首"], o["頭"]
        t = (y - a[1]) / (b[1] - a[1])
        return float(a[2] + np.clip(t, -0.5, 1.5) * (b[2] - a[2])) - 0.3

    def base_neck(self):
        """Height where the horizontal contour is only the neck: perimeter below 1.25 times that of the neck."""
        V, Ft = self.V, self.F_trunk
        ref = None
        ys = np.arange(*self.rep["neck_sweep"], -0.25)
        pers = []
        for y in ys:
            r = CP.outline_around(V, Ft, np.array([0.0, y, self.z_neck(y)]), [0, -1, 0], [0, 0, -1])
            pers.append(CP.length(r[0]) if r else np.nan)
        pers = np.array(pers)
        ref = np.nanmin(pers[:16])
        for y, p in zip(ys, pers):
            if p > 1.3 * ref:
                return float(y) + 0.5
        return self.rep["y_neck_default"]

    def crest(self, a, b, n):
        """n + 1 points from a to b on the surface, along the ridge of the shoulder: for each x, the highest
        point of the sagittal cut near the straight line a-b."""
        V, F = self.V, self.F_trunk
        out = [np.asarray(a, float)]
        for k in range(1, n):
            t = k / n
            p = a + t * (b - a)
            L = CP.cut_plan(V, F, p, [1, 0, 0])
            better = None
            for P, _ in L:
                m = np.abs(P[:, 2] - p[2]) < 4.0
                if m.any():
                    q = P[m][np.argmax(P[m][:, 1])]
                    if better is None or q[1] > better[1]:
                        better = q
            out.append(better if better is not None else p)
        out.append(np.asarray(b, float))
        return np.array(out)

    # ------------------------------------------------------------------ tube along bones
    def axis(self, names, smooth=3.0):
        """Bone polyline -> functions centre(s), tangent(s); the tangent turns over ±smooth cm at the bends."""
        pts = [np.asarray(self.os[n] if isinstance(n, str) else n, float) for n in names]
        seg = [np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])]
        cum = np.concatenate([[0], np.cumsum(seg)])
        dirs = [unit(b - a) for a, b in zip(pts[:-1], pts[1:])]

        def center(s):
            k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(dirs) - 1))
            return pts[k] + (s - cum[k]) * dirs[k]

        def tangent(s):
            k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(dirs) - 1))
            t = dirs[k].copy()
            if k + 1 < len(dirs) and cum[k + 1] - s < smooth:
                w = 0.5 * (1 - (cum[k + 1] - s) / smooth)
                t = unit((1 - w) * dirs[k] + w * dirs[k + 1])
            if k > 0 and s - cum[k] < smooth:
                w = 0.5 * (1 - (s - cum[k]) / smooth)
                t = unit((1 - w) * dirs[k] + w * dirs[k - 1])
            return t
        return center, tangent, cum

    # ------------------------------------------------------------------ arms
    def arm(self):
        M, V = self.M, self.V
        center, tangent, cum = self.axis(["左腕", "左ひじ", "左手首"])
        s_end = cum[-1] - 0.4
        n = int(round((s_end - S_ARMHOLE) / STEP_ARM))
        self.n_arm = n

        def the_key(r, c):
            return self.keys_armhole[c % NA] if r == 0 else ("A", "G", r, c % NA)
        self.key_arm = the_key
        pers = []
        for r in range(1, n + 1):
            s = S_ARMHOLE + (s_end - S_ARMHOLE) * r / n
            rr = CP.loop(V, self.Fc, center(s), tangent(s), [0, 1, 0], NA)
            assert rr is not None, ("arm", r)
            pers.append(rr[1])
            for c in range(NA):
                M.s(the_key(r, c), rr[0][c], True)
        self.info["arm_perimeters_cm"] = [round(pers[0], 1), round(min(pers), 1), round(pers[-1], 1)]
        M.grid(the_key, list(range(n + 1)), NA, True, "arm")

    # ------------------------------------------------------------------ legs
    def leg(self):
        M, V = self.M, self.V

        def key_band(rb, cb):          # rb = 0 (front) .. M (back), cb = -1, 0, 1
            if rb == 0:
                return ("T", 0, cb % NT)
            if rb == MB:
                return ("T", 0, (NT // 2 - cb) % NT)
            return ("B", rb, cb)
        self.key_band = key_band
        h = W // 2
        for rb in range(MB):
            for cb in range(-h, h):
                M.quad(key_band(rb, cb), key_band(rb, cb + 1), key_band(rb + 1, cb + 1), key_band(rb + 1, cb),
                       ("band", [(cb, rb), (cb + 1, rb), (cb + 1, rb + 1), (cb, rb + 1)]))
        for rb in range(1, MB):
            for cb in range(-h, h + 1):
                M.s(key_band(rb, cb))
        ring0 = [("T", 0, i) for i in range(h, NT // 2 - h + 1)] + [key_band(rb, h) for rb in range(MB - 1, 0, -1)]
        assert len(ring0) == NL, len(ring0)
        center, tangent, cum = self.axis(["左足", "左ひざ", "左足首"])
        # first anchored row: just below the crotch
        s0 = None
        for s in np.arange(6.0, 16.0, 0.25):
            r = CP.outline_around(V, self.Fc, center(s), tangent(s), [0, 0, -1])
            if r is not None and CP.length(r[0]) < 75 and r[0][:, 0].min() > 0.3:
                s0 = s
                break
        assert s0 is not None
        self.info["leg_s0"] = round(float(s0), 2)
        s_end = cum[-1] - 2.5            # above the ankle
        n = int(round((s_end - s0) / STEP_LEG))
        self.n_leg = n + 1             # rows 2..n+1 anchored, row 1 free
        k_int = NL - MB // 2              # middle column of the inner side (38)
        angle0 = np.radians(270.0 + (0 - k_int) * 360.0 / NL)

        def the_key(r, k):
            return ring0[k % NL] if r == 0 else ("L", "G", r, k % NL)
        self.key_leg = the_key
        for k in range(NL):
            M.s(the_key(1, k))
        for r in range(2, n + 2):
            s = s0 + (s_end - s0) * (r - 2) / n
            rr = CP.loop(V, self.Fc, center(s), tangent(s), [0, 0, -1], NL, angle0)
            assert rr is not None, ("leg", r)
            for k in range(NL):
                M.s(the_key(r, k), rr[0][k], True)
        M.grid(the_key, list(range(n + 2)), NL, True, "leg")
        self.angle0_leg = angle0

    # ------------------------------------------------------------------ fingers and toes
    def loop_finger(self, F, c, n, ref, a, first_on_ref=False):
        """Row of a finger with 2a + 2 vertices: a + 1 on top (centred on ref), a + 1 underneath; or, for the
        thumb, the first vertex in the direction ref."""
        r = CP.outline_around(self.V, F, c, n, ref)
        if r is None:
            return None
        m = 2 * a + 2
        if first_on_ref:
            return CP.resample(r[0], m, c, r[1], r[2], 0.0)
        S = CP.resample(r[0], 2 * m, c, r[1], r[2], 0.0)
        return np.array([S[(-a + 2 * i) % (2 * m)] for i in range(m)])

    def finger(self, name, base, rings, tip, zone):
        """A finger: tube that starts from `base` (keys of the shared row, in ring direction), goes through the
        anchored `rings` and closes on `bout` (one vertex, fan quads)."""
        M = self.M
        m = len(base)
        offset = m // 2                 # the map seam runs underneath

        def the_key(r, c):
            c = (c + offset) % m
            return base[c] if r == 0 else ("D", "G", name, r, c)
        for r, A in enumerate(rings, 1):
            for c in range(m):
                M.s(("D", "G", name, r, c), A[c], True, zone)
        n = len(rings)
        map_ = "digit_" + name
        M.grid(the_key, list(range(n + 1)), m, True, map_)
        kb = ("D", "G", name, "tip", 0)
        M.s(kb, tip, True, zone)
        for k in range(m // 2):
            M.quad(kb, the_key(n, 2 * k), the_key(n, 2 * k + 1), the_key(n, 2 * k + 2),
                   (map_, [(2 * k + 1, n + 1), (2 * k, n), (2 * k + 1, n), (2 * k + 2, n)]))

    def on_surface(self, p, F):
        return G.plus_near(np.array([p], float), self.V, F)[0][0]

    def tube_finger(self, F, names, ref, a, stage, first_on_ref=False):
        """Rings of a finger of the hand along its bones: from the fold where it leaves the palm (the
        perimeter of the cut drops to that of a finger) to the tip. Returns (rings, bout)."""
        V, o = self.V, self.os
        pts = [o[n] for n in names]
        d = unit(pts[-1] - pts[-2])
        Vm = V[np.unique(F)]
        rel = Vm - pts[-1]
        t = rel @ d
        r = np.linalg.norm(rel - np.outer(t, d), axis=1)
        pts = pts + [pts[-1] + float(t[r < 1.0].max()) * d]
        center, tangent, cum = self.axis(pts, smooth=0.6)
        ss = np.arange(0.0, cum[-1] - 0.2, 0.1)
        per = []
        for s in ss:
            rr = CP.outline_around(V, F, center(s), tangent(s), ref)
            per.append(CP.length(rr[0]) if rr else np.inf)
        per = np.array(per)
        threshold = 1.4 * np.median(per[len(per) // 2:])
        big = np.flatnonzero(per > threshold)
        s0 = (ss[big.max()] + 0.1 if len(big) else 0.0) + 0.3
        s1 = cum[-1] - 0.3
        n = max(2, int(round((s1 - s0) / stage)))
        rings = []
        for s in np.linspace(s0, s1, n + 1):
            A = self.loop_finger(F, center(s), tangent(s), ref, a, first_on_ref)
            assert A is not None, (names[0], s)
            rings.append(A)
        return rings, self.on_surface(pts[-1], F), round(float(s0), 2)

    # ------------------------------------------------------------------ hand
    def main(self):
        """Palm (the continuation of the arm tube), thumb in a window, four fingers on the last row."""
        M, V, o, Fm = self.M, self.V, self.os, self.Fm
        wrist = o["左手首"]
        ax = unit(o["左中指１"] - wrist)
        back = unit(np.cross(o["左中指１"] - wrist, o["左人指１"] - o["左小指１"]))
        n0 = self.n_arm
        the_key = self.key_arm
        fen = lambda r, c: FEN_R <= r - n0 <= FEN_R + 2 and FEN_C <= c <= FEN_C + 2
        for i in range(1, NP + 1):
            if i in (2, 3):                 # where the reference's hand is a single clean contour
                rr = CP.loop(V, Fm, wrist + (1.6 + (i - 2) * 1.0) * ax, ax, [0, 1, 0], NA)
                assert rr is not None, ("palm", i)
                for c in range(NA):
                    M.s(the_key(n0 + i, c), rr[0][c], True, "hand")
            else:
                for c in range(NA):
                    if not (i == FEN_R + 1 and c == FEN_C + 1):
                        M.s(the_key(n0 + i, c), zone="none" if i == 1 else "hand")
        M.grid(the_key, list(range(n0, n0 + NP + 1)), NA, True, "arm",
                 exists=lambda r, c: not (fen(r, c) and fen(r + 1, c + 1)))
        infos = {}
        # thumb
        names = ["左親指０", "左親指１", "左親指２"]
        d_p = unit(o[names[1]] - o[names[0]])
        e1 = unit(ax - (ax @ d_p) * d_p)
        an, tip, infos["thumb"] = self.tube_finger(Fm, names, e1, 3, STEP_FINGER, True)
        R0, C0 = n0 + FEN_R, FEN_C
        girth = [(2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]     # towards the back first
        if np.cross(d_p, e1) @ back < 0:
            girth = [girth[0]] + girth[:0:-1]
        self.finger("thumb", [the_key(R0 + a, C0 + b) for a, b in girth], an, tip, "hand")
        # fingers
        for f, d in enumerate("人中薬小"):
            base = [the_key(n0 + NP, 20 + 3 * f + i) for i in range(4)] + [the_key(n0 + NP, 16 - 3 * f + i) for i in range(4)]
            an, tip, infos[d] = self.tube_finger(Fm, [f"左{d}指{k}" for k in "１２３"], back, FINGERS[f], STEP_FINGER)
            self.finger("finger%d" % f, base, an, tip, "hand")
        self.info["digits_start_cm"] = infos

    # ------------------------------------------------------------------ foot
    def foot(self):
        """The leg tube turns around the instep (fan rows, heel included), goes forward
        to the toes, and five toes share the last row."""
        M, V = self.M, self.V
        cf = V[self.Fc].mean(1)
        Fg = self.Fc[(cf[:, 0] > 0) & (cf[:, 1] < 30)]
        nj = self.n_leg
        the_key = self.key_leg
        # toes: vertical cuts, narrow contours tracked from one cut to the next
        zmin = float(V[np.unique(Fg)][:, 2].min())
        chains = []                       # each one: list of (z, x0, x1, centre)
        for z in np.arange(zmin + 6.0, zmin, -0.1):
            for Pc, closed in CP.cut_plan(V, Fg, np.array([0.0, 0.0, z]), [0, 0, -1]):
                if not closed or CP.length(Pc) > self.rep["toe_tour_max"]:
                    continue
                x0, x1 = float(Pc[:, 0].min()), float(Pc[:, 0].max())
                for ch in chains:
                    z_, a0, a1, _ = ch[-1]
                    if z_ - z < 0.25 and min(x1, a1) - max(x0, a0) > 0.5 * min(x1 - x0, a1 - a0):
                        ch.append((z, x0, x1, Pc.mean(0)))
                        break
                else:
                    chains.append([(z, x0, x1, Pc.mean(0))])
        chains = sorted([ch for ch in chains if len(ch) >= 4], key=lambda ch: np.mean([c[3][0] for c in ch]))
        assert len(chains) == 5, [len(ch) for ch in chains]
        z_toes = max(ch[0][0] for ch in chains)          # the first toe to separate
        self.info["toes_z"] = [[round(ch[0][0], 1), round(ch[-1][0], 1)] for ch in chains]
        # fan: planes that turn around a line in front of the ankle
        y_p = float(np.mean([M.pos[M.s(the_key(nj, k))][1] for k in range(NL)]))
        z_p = z_toes + 3.5
        pivot = np.array([0.0, y_p, z_p])
        ranks = []
        for i in range(1, NF + 1):
            th = np.radians(90.0 * i / NF)
            ranks.append((pivot, np.array([0.0, -np.cos(th), -np.sin(th)]), np.array([0.0, np.sin(th), -np.cos(th)])))
        for dz in (1.2, 2.4):
            ranks.append((pivot + [0, 0, -dz], np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])))
        for i, (p0, n, e1) in enumerate(ranks, 1):
            L = [Pc for Pc, closed in CP.cut_plan(V, Fg, p0, n) if closed]
            assert L, ("foot", i)
            c = max(L, key=CP.length).mean(0)
            rr = CP.loop(V, Fg, c, n, e1, NL, self.angle0_leg)
            assert rr is not None, ("foot", i)
            for k in range(NL):
                M.s(the_key(nj + i, k), rr[0][k], True)
        nr = len(ranks)
        for i in (nr + 1, nr + 2):
            for k in range(NL):
                M.s(the_key(nj + i, k), zone="toe")
        M.grid(the_key, list(range(nj, nj + nr + 3)), NL, True, "leg")
        last = nj + nr + 2
        # toes: top from the inside (big toe) outwards, underside in the other direction
        k_int = NL - MB // 2                # middle column of the inner side
        top, bottom = k_int + 1, k_int        # top: top, top + 1, ...; underside: ..., bottom - 1, bottom
        over = 0
        for t, (ch, a) in enumerate(zip(chains, TOES)):
            base = [the_key(last, top + over + i) for i in range(a + 1)]
            base += [the_key(last, bottom - over - a + i) for i in range(a + 1)]
            over += a
            z0, z1 = ch[0][0] - 0.2, ch[-1][0] + 0.4          # the last row far enough from the tip for a round tip
            n = max(2, int(round((z0 - z1) / STEP_TOE)) + 1)
            an = []
            for z in list(np.linspace(z0, z1, n)) + [ch[-1][0] + 0.12]:      # a small row near the tip: a dome, not a point
                c = min(ch, key=lambda e: abs(e[0] - z))[3].copy()
                c[2] = z
                A = self.loop_finger(Fg, c, [0, 0, -1], [0, 1, 0], a)
                assert A is not None, ("toe", t, z)
                an.append(A)
            c = ch[-1][3].copy()
            c[2] = ch[-1][0] - 0.1
            self.finger("toe%d" % t, base, an, self.on_surface(c, Fg), "toe")

    # ------------------------------------------------------------------ face orientation
    def orient(self, n_safe):
        """Gives the quads added after the first n_safe the orientation of their neighbours (step by step)."""
        M = self.M
        per_edge = {}
        for q, (ids, _) in enumerate(M.quads):
            for a, b in zip(ids, ids[1:] + ids[:1]):
                per_edge.setdefault((min(a, b), max(a, b)), []).append(q)
        done = [q < n_safe for q in range(len(M.quads))]
        pile = [q for q in range(n_safe)]
        while pile:
            q = pile.pop()
            ids = M.quads[q][0]
            for a, b in zip(ids, ids[1:] + ids[:1]):
                for v in per_edge[(min(a, b), max(a, b))]:
                    if done[v]:
                        continue
                    iv, cv = M.quads[v]
                    i = iv.index(a)
                    if iv[(i + 1) % 4] == b:          # same direction on the shared edge: to be flipped
                        M.quads[v] = (iv[::-1], None if cv is None else (cv[0], cv[1][::-1]))
                    done[v] = True
                    pile.append(v)
        assert all(done)

    # ------------------------------------------------------------------ mirror and output
    def key_mirror(self, k):
        if k[0] == "T":
            return ("T", k[1], (NT - k[2]) % NT)
        if k[0] == "N":
            return ("N", k[1], (NN - k[2]) % NN)
        if k[0] == "B":
            return ("B", k[1], -k[2])
        return (k[0], "D" if k[1] == "G" else "G") + tuple(k[2:])

    def reflect(self):
        """Adds the right side: mirrored vertices and quads of the left limbs."""
        M = self.M
        n0 = len(M.keys)
        for i in range(n0):
            k = M.keys[i]
            if k[0] in ("T", "N", "B"):
                continue
            km = self.key_mirror(k)
            M.s(km, None if M.pos[i] is None else mirror(M.pos[i]), M.anchor[i], M.zone[i])
        new = []
        for ids, map_ in M.quads:
            ks = [self.key_mirror(M.keys[i]) for i in ids]
            if any(M.keys[i][0] not in ("T", "N", "B") for i in ids):
                ids2 = [M.s(k) for k in ks][::-1]
                c2 = None if map_ is None else (map_[0] + "_R" if not map_[0].endswith("_R") else map_[0], map_[1][::-1])
                new.append((ids2, c2))
        M.quads += new
        self.sym = np.array([M.s(self.key_mirror(k)) for k in M.keys])

    def arrays(self):
        M = self.M
        n = len(M.keys)
        P = np.full((n, 3), np.nan)
        for i, p in enumerate(M.pos):
            if p is not None:
                P[i] = p
        Qd = np.array([q for q, _ in M.quads])
        return P, np.array(M.anchor), Qd, np.array(M.zone)

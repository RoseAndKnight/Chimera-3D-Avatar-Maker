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
(junctions) are placed afterwards by `ajuster.py`: interpolation between their neighbours, then relaxation on the
surface. The right side is the exact mirror of the left.
"""
import numpy as np

import coupe as CP
import geom as G

NT = 72            # trunk columns (0 = front middle, NT/4 = left side, NT/2 = middle of the back)
Q = 6              # width of the armhole notch, in columns
P_EMM = 10         # height of the notch, in rows
Z = 7              # edges of the shoulder seam
NA = 2 * P_EMM + Q # arm columns (26)
NN = NT - 2 * (Q + 2 * Z)   # neck columns (32)
W = 2              # width of the crotch strip
MB = 8             # length of the crotch strip
NL = NT // 2 - W + MB        # leg columns (42)
DOIGTS = (3, 3, 3, 3)       # edges on top of each finger, from index to little finger (rings of 8)
ORTEILS = (6, 4, 4, 3, 3)   # from the big toe to the little one
NP = 10            # rows of the palm, from the wrist to the fingers
FEN_R, FEN_C = 5, 17   # thumb window in the palm: rows FEN_R..FEN_R+2, columns FEN_C..FEN_C+2
NF = 12            # fan rows around the instep (from the ankle plane to the vertical plane)
assert 2 * sum(DOIGTS) + 2 == NA and 2 * sum(ORTEILS) + 2 == NL

# Landmark heights, in cm: those of the ZZZ reference body. Another reference gives its own
# (R.reperes, see reperes()): nothing else in the construction is an absolute height.
REPERES = {
    "y_entrejambe": 89.0,          # first row of the trunk (the body is still a single contour there)
    "y_bassin": 108.0,             # where the side column leaves the middle of the pelvis towards the underside of the arm
    "cou_balayage": (150.0, 138.0),  # heights where to look for the base of the neck, top to bottom
    "y_cou_defaut": 142.5,
    "y_haut": None,                # last row of the neck; None: the top of the body skin
    "orteil_tour_max": 8.0,        # circumference below which a vertical cut of the foot is a toe
}


def reperes(R):
    """The same landmarks for another body, taken from its skeleton and its skin (offsets measured on ZZZ)."""
    o = R.os
    V = R.V[np.unique(R.F[R.f_corps])]
    milieu = V[np.abs(V[:, 0]) < 0.5]
    bas = float(milieu[milieu[:, 1] > 0.5 * o["左足"][1]][:, 1].min())      # the bottom of the crotch
    return {"y_entrejambe": bas + 1.8, "y_bassin": float(o["上半身"][1]) + 5.8,
            "cou_balayage": (float(o["頭"][1]) + 0.6, float(o["首"][1]) - 4.6),
            "y_cou_defaut": float(o["首"][1]) - 0.1, "y_haut": _bas_du_bord(R, float(o["首"][1])) - 0.25}


def _bas_du_bord(R, y_min):
    """The lowest point of the skin boundary above y_min: where the neck stops being closed."""
    F = R.F[R.f_corps]
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, c = np.unique(e, axis=0, return_counts=True)
    y = R.V[np.unique(u[c == 1])][:, 1]
    return float(y[y > y_min].min())
J_A = 32              # armpit row
J_TOP = J_A + P_EMM
S_EMM = 6.0           # along the arm from the shoulder: where the arm is a contour of its own
PAS_BRAS = 1.15
PAS_JAMBE = 1.5
PAS_COU = 1.0
PAS_DOIGT = 0.7
PAS_ORTEIL = 0.6


def unitaire(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Maillage:
    def __init__(self):
        self.cle = {}          # key -> index
        self.cles = []
        self.pos = []          # initial position or None
        self.ancre = []
        self.zone = []         # what the vertex lands on: "corps", "main", "orteil" (body), "aucune" (smoothed only)
        self.quads = []
        self.alias = {}
        self.carte = []        # (map name, u, v) for the texture coordinates

    def canon(self, k):
        while k in self.alias:
            k = self.alias[k]
        return k

    def s(self, k, pos=None, ancre=False, zone="corps", carte=None):
        k = self.canon(k)
        i = self.cle.get(k)
        if i is None:
            i = len(self.cles)
            self.cle[k] = i
            self.cles.append(k)
            self.pos.append(None)
            self.ancre.append(False)
            self.zone.append(zone)
            self.carte.append(None)
        if pos is not None:
            self.pos[i] = np.asarray(pos, float)
            self.ancre[i] = ancre
            self.zone[i] = zone
        if carte is not None and self.carte[i] is None:
            self.carte[i] = carte
        return i

    def quad(self, a, b, c, d, carte=None, envers=False):
        """Four keys. carte: (name, [(u, v) of the four corners]) for the texture coordinates."""
        ids = [self.s(k) for k in (a, b, c, d)]
        if len(set(ids)) == 4:
            if envers:
                ids = ids[::-1]
                carte = None if carte is None else (carte[0], carte[1][::-1])
            self.quads.append((ids, carte))

    def grille(self, cle_de, rangs, cols, ferme, carte, u0=0, existe=None, envers=False):
        """Quads of a grid: cle_de(r, c). rangs: list of consecutive rows; cols: number of columns;
        ferme: the last column joins the first."""
        nc = cols if ferme else cols - 1
        for r0, r1 in zip(rangs[:-1], rangs[1:]):
            for c in range(nc):
                c1 = (c + 1) % cols if ferme else c + 1
                if existe is not None and not existe(r0, c):
                    continue
                self.quad(cle_de(r0, c), cle_de(r0, c1), cle_de(r1, c1), cle_de(r1, c),
                          (carte, [(u0 + c, r0), (u0 + c + 1, r0), (u0 + c + 1, r1), (u0 + c, r1)]), envers)


def miroir(p):
    return np.asarray(p) * np.array([-1.0, 1.0, 1.0])


def symetriser(P):
    """Closed row of n points, the first on the symmetry plane: made symmetric (i <-> n - i)."""
    n = len(P)
    out = P.copy()
    for i in range(n):
        j = (-i) % n
        out[i] = 0.5 * (P[i] + miroir(P[j]))
    out[0, 0] = 0.0
    if n % 2 == 0:
        out[n // 2, 0] = 0.0
    return out


class Chantier:
    """Builds the mesh on a reference (unfolded reference.Reference)."""

    def __init__(self, R, journal=print):
        self.R = R
        self.j = journal
        self.M = Maillage()
        self.V = R.V
        self.Fc = R.F[R.f_corps]
        self.Fm = R.F[R.f_main["G"]]
        self.os = R.os
        self.info = {}
        self.rep = dict(REPERES, **getattr(R, "reperes", {}))

    # ------------------------------------------------------------------ trunk
    def cle_tronc(self, j, i):
        return ("T", j, i % NT)

    def ligne_epaule(self):
        """Height of the shoulder line y(x) between the neck and the armhole, on the reference (left side)."""
        o = self.os
        a, b = o["左腕"], o["左ひじ"]
        n = unitaire(b - a)
        c = a + S_EMM * n
        self.emm_c, self.emm_n = c, n
        V, F = self.V, self.Fc
        cf = V[F].mean(1)
        # faces on the trunk side: not the arm beyond the armhole plane (nor its mirror); the arm is
        # recognized by its weights
        def du_bras(cote, signe):
            noms = [k for k in o if k.startswith(cote) and any(m in k for m in ("腕", "ひじ", "手"))]
            w = self.R.poids_os(noms)[F].mean(1)
            return (w > 0.5) & (((cf * [signe, 1, 1]) - c) @ n > 0)
        bras_g, bras_d = du_bras("左", 1), du_bras("右", -1)
        self.F_tronc = F[~(bras_g | bras_d)]
        return c, n

    def tronc(self):
        M, V = self.M, self.V
        c_emm, n_emm = self.ligne_epaule()
        Ft = self.F_tronc
        axe_z = 3.0
        # --- armhole ring: cut of the arm at S_EMM, first point at the top, towards the back
        r = CP.anneau(V, self.Fc, c_emm, n_emm, [0, 1, 0], NA)
        assert r is not None, "emmanchure"
        emm, per = r
        self.info["emmanchure_cm"] = round(per, 1)
        self.emm = emm
        # matching of the ring with the notch (columns ci..ci+Q, rows J_A..J_TOP)
        ci = NT // 4 - Q // 2          # 15
        cs = ci + Q                    # 21
        self.ci, self.cs = ci, cs
        cles_emm = [("T", J_TOP, cs)]                                  # c = 0: the acromion (ci and cs stitched)
        cles_emm += [("T", j, cs) for j in range(J_TOP - 1, J_A - 1, -1)]   # back, going down
        cles_emm += [("T", J_A, i) for i in range(cs - 1, ci, -1)]         # bottom of the armpit
        cles_emm += [("T", j, ci) for j in range(J_A, J_TOP)]              # front, going up
        assert len(cles_emm) == NA
        # shoulder seam: (J_TOP, ci - k) = (J_TOP, cs + k), and its mirror
        for k in range(Z + 1):
            M.alias[("T", J_TOP, cs + k)] = ("T", J_TOP, ci - k)
            M.alias[("T", J_TOP, (NT - cs - k) % NT)] = ("T", J_TOP, (NT - ci + k) % NT)
        self.cles_emm = cles_emm
        for k, p in zip(cles_emm, emm):
            M.s(k, p, True)
            kk = M.canon(k)
            M.s(("T", kk[1], (NT - kk[2]) % NT), miroir(p), True)
        haut_emm = emm[0]                       # acromion
        z_bras = float(emm[NA // 2 - 2:NA // 2 + 3, 2].mean())
        y_a = float(emm[:, 1].min()) - 0.3      # armpit height: just below the arm
        self.info["y_aisselle"] = round(y_a, 1)
        # --- horizontal rows, from the crotch to the armpit
        ys = np.linspace(self.rep["y_entrejambe"], y_a, J_A + 1)
        for j in range(J_A + 1):
            # the side column follows a line that starts at the middle of the pelvis and joins the underside of the arm
            yb = self.rep["y_bassin"]
            t = np.clip((ys[j] - yb) / (y_a - yb), 0.0, 1.0)
            zm = self.z_milieu(ys[j])
            P = self.rang_tronc(ys[j], zm, (1 - t) * zm + t * z_bras)
            for i in range(NT):
                if j == J_A and (ci <= i <= cs or NT - cs <= i <= NT - ci):
                    continue            # bottom of the notch: it is the arm ring
                M.s(("T", j, i), P[i], True)
        # --- neck: horizontal rows from the base
        y_cou0 = self.base_cou()
        y_fin = self.rep["y_haut"]
        if y_fin is None:
            y_fin = float(V[np.unique(self.Fc)][:, 1].max()) - 0.25
        n_cou = int(round((y_fin - y_cou0) / PAS_COU))
        self.n_cou = n_cou
        cou = []
        for r_ in range(n_cou + 1):
            y = y_cou0 + (y_fin - y_cou0) * r_ / n_cou
            c = np.array([0.0, y, self.z_cou(y)])
            rr = CP.anneau(V, Ft, c, [0, -1, 0], [0, 0, -1], NN)
            assert rr is not None, ("cou", r_)
            cou.append(symetriser(rr[0]))
        self.info["y_base_cou"] = round(y_cou0, 1)
        # the neckline: front i = -(NT/4 - Q/2 - Z) .. +; back i = cs + Z .. NT - cs - Z
        d = ci - Z                      # 8: half-width of the neckline at the front, in columns
        col_cou = list(range(0, d + 1)) + list(range(cs + Z + 1, NT - cs - Z)) + list(range(NT - d, NT))
        assert len(col_cou) == NN, len(col_cou)
        self.col_cou = col_cou

        def cle_cou(r_, c):
            return ("T", J_TOP, col_cou[c % NN]) if r_ == 0 else ("N", r_, c % NN)
        self.cle_cou = cle_cou
        for r_ in range(n_cou + 1):
            for c in range(NN):
                M.s(cle_cou(r_, c), cou[r_][c], True)
        # --- shoulder seam: from the acromion to the corner of the neckline, on the ridge
        coin = cou[0][d]
        crete = self.crete(haut_emm, coin, Z + 1)
        for k in range(1, Z):
            M.s(("T", J_TOP, ci - k), crete[k], True)
            M.s(("T", J_TOP, (NT - ci + k) % NT), miroir(crete[k]), True)
        # --- rows above the armpit: left free (placed by ajuster), between the armhole, the
        # armpit row, the seam and the neckline
        for j in range(J_A + 1, J_TOP):
            for i in range(NT):
                if ci < i < cs or NT - cs < i < NT - ci:
                    continue
                M.s(("T", j, i))
        # --- quads

        def existe(j, i):
            if j >= J_A and (ci <= i < cs or NT - cs <= i < NT - ci):
                return False
            return True
        M.grille(self.cle_tronc, list(range(J_TOP + 1)), NT, True, "tronc", existe=existe, envers=True)
        M.grille(cle_cou, list(range(n_cou + 1)), NN, True, "cou", envers=True)

    def rang_tronc(self, y, zm, z_cote):
        """Horizontal row of the trunk: NT points, symmetric; column NT/4 is on the side at z = z_cote,
        equal steps in front of it, equal steps behind."""
        c = np.array([0.0, y, zm])
        r = CP.contour_autour(self.V, self.F_tronc, c, [0, -1, 0], [0, 0, -1])
        assert r is not None, ("tronc", y)
        K = 720
        D = symetriser(CP.reechantillonner(r[0], K, c, r[1], r[2], 0.0))
        H = D[:K // 2 + 1]                          # left half, from front to back
        dz = H[:, 2] - z_cote
        croise = np.flatnonzero((dz[:-1] < 0) & (dz[1:] >= 0))
        assert len(croise), ("side", y)
        s = int(croise[np.argmax(H[croise, 0])]) + 1

        def pas_egal(A, k):
            L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(A, axis=0), axis=1))])
            t = np.linspace(0, L[-1], k + 1)
            return np.stack([np.interp(t, L, A[:, a]) for a in range(3)], 1)
        G_ = np.concatenate([pas_egal(H[:s + 1], NT // 4), pas_egal(H[s:], NT // 4)[1:]])   # NT/2 + 1 points
        P = np.zeros((NT, 3))
        P[:NT // 2 + 1] = G_
        P[NT // 2 + 1:] = (G_[1:NT // 2] * [-1, 1, 1])[::-1]
        return P

    def z_milieu(self, y):
        """Front-to-back middle of the trunk contour at height y (where it crosses the plane x = 0)."""
        zs = []
        for P, ferme in CP.coupe_plan(self.V, self.F_tronc, np.array([0.0, y, 0.0]), [0, -1, 0]):
            if ferme and P[:, 0].min() < 0 < P[:, 0].max():
                zs += [P[:, 2].min(), P[:, 2].max()]
        return 0.5 * (min(zs) + max(zs))

    def z_cou(self, y):
        o = self.os
        a, b = o["首"], o["頭"]
        t = (y - a[1]) / (b[1] - a[1])
        return float(a[2] + np.clip(t, -0.5, 1.5) * (b[2] - a[2])) - 0.3

    def base_cou(self):
        """Height where the horizontal contour is only the neck: perimeter below 1.25 times that of the neck."""
        V, Ft = self.V, self.F_tronc
        ref = None
        ys = np.arange(*self.rep["cou_balayage"], -0.25)
        pers = []
        for y in ys:
            r = CP.contour_autour(V, Ft, np.array([0.0, y, self.z_cou(y)]), [0, -1, 0], [0, 0, -1])
            pers.append(CP.longueur(r[0]) if r else np.nan)
        pers = np.array(pers)
        ref = np.nanmin(pers[:16])
        for y, p in zip(ys, pers):
            if p > 1.3 * ref:
                return float(y) + 0.5
        return self.rep["y_cou_defaut"]

    def crete(self, a, b, n):
        """n + 1 points from a to b on the surface, along the ridge of the shoulder: for each x, the highest
        point of the sagittal cut near the straight line a-b."""
        V, F = self.V, self.F_tronc
        out = [np.asarray(a, float)]
        for k in range(1, n):
            t = k / n
            p = a + t * (b - a)
            L = CP.coupe_plan(V, F, p, [1, 0, 0])
            mieux = None
            for P, _ in L:
                m = np.abs(P[:, 2] - p[2]) < 4.0
                if m.any():
                    q = P[m][np.argmax(P[m][:, 1])]
                    if mieux is None or q[1] > mieux[1]:
                        mieux = q
            out.append(mieux if mieux is not None else p)
        out.append(np.asarray(b, float))
        return np.array(out)

    # ------------------------------------------------------------------ tube along bones
    def axe(self, noms, lisse=3.0):
        """Bone polyline -> functions centre(s), tangent(s); the tangent turns over ±lisse cm at the bends."""
        pts = [np.asarray(self.os[n] if isinstance(n, str) else n, float) for n in noms]
        seg = [np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])]
        cum = np.concatenate([[0], np.cumsum(seg)])
        dirs = [unitaire(b - a) for a, b in zip(pts[:-1], pts[1:])]

        def centre(s):
            k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(dirs) - 1))
            return pts[k] + (s - cum[k]) * dirs[k]

        def tangente(s):
            k = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(dirs) - 1))
            t = dirs[k].copy()
            if k + 1 < len(dirs) and cum[k + 1] - s < lisse:
                w = 0.5 * (1 - (cum[k + 1] - s) / lisse)
                t = unitaire((1 - w) * dirs[k] + w * dirs[k + 1])
            if k > 0 and s - cum[k] < lisse:
                w = 0.5 * (1 - (s - cum[k]) / lisse)
                t = unitaire((1 - w) * dirs[k] + w * dirs[k - 1])
            return t
        return centre, tangente, cum

    # ------------------------------------------------------------------ arms
    def bras(self):
        M, V = self.M, self.V
        centre, tangente, cum = self.axe(["左腕", "左ひじ", "左手首"])
        s_fin = cum[-1] - 0.4
        n = int(round((s_fin - S_EMM) / PAS_BRAS))
        self.n_bras = n

        def cle(r, c):
            return self.cles_emm[c % NA] if r == 0 else ("A", "G", r, c % NA)
        self.cle_bras = cle
        pers = []
        for r in range(1, n + 1):
            s = S_EMM + (s_fin - S_EMM) * r / n
            rr = CP.anneau(V, self.Fc, centre(s), tangente(s), [0, 1, 0], NA)
            assert rr is not None, ("bras", r)
            pers.append(rr[1])
            for c in range(NA):
                M.s(cle(r, c), rr[0][c], True)
        self.info["bras_perimetres_cm"] = [round(pers[0], 1), round(min(pers), 1), round(pers[-1], 1)]
        M.grille(cle, list(range(n + 1)), NA, True, "bras")

    # ------------------------------------------------------------------ legs
    def jambe(self):
        M, V = self.M, self.V

        def cle_bande(rb, cb):          # rb = 0 (front) .. M (back), cb = -1, 0, 1
            if rb == 0:
                return ("T", 0, cb % NT)
            if rb == MB:
                return ("T", 0, (NT // 2 - cb) % NT)
            return ("B", rb, cb)
        self.cle_bande = cle_bande
        h = W // 2
        for rb in range(MB):
            for cb in range(-h, h):
                M.quad(cle_bande(rb, cb), cle_bande(rb, cb + 1), cle_bande(rb + 1, cb + 1), cle_bande(rb + 1, cb),
                       ("bande", [(cb, rb), (cb + 1, rb), (cb + 1, rb + 1), (cb, rb + 1)]))
        for rb in range(1, MB):
            for cb in range(-h, h + 1):
                M.s(cle_bande(rb, cb))
        anneau0 = [("T", 0, i) for i in range(h, NT // 2 - h + 1)] + [cle_bande(rb, h) for rb in range(MB - 1, 0, -1)]
        assert len(anneau0) == NL, len(anneau0)
        centre, tangente, cum = self.axe(["左足", "左ひざ", "左足首"])
        # first anchored row: just below the crotch
        s0 = None
        for s in np.arange(6.0, 16.0, 0.25):
            r = CP.contour_autour(V, self.Fc, centre(s), tangente(s), [0, 0, -1])
            if r is not None and CP.longueur(r[0]) < 75 and r[0][:, 0].min() > 0.3:
                s0 = s
                break
        assert s0 is not None
        self.info["jambe_s0"] = round(float(s0), 2)
        s_fin = cum[-1] - 2.5            # above the ankle
        n = int(round((s_fin - s0) / PAS_JAMBE))
        self.n_jambe = n + 1             # rows 2..n+1 anchored, row 1 free
        k_int = NL - MB // 2              # middle column of the inner side (38)
        angle0 = np.radians(270.0 + (0 - k_int) * 360.0 / NL)

        def cle(r, k):
            return anneau0[k % NL] if r == 0 else ("L", "G", r, k % NL)
        self.cle_jambe = cle
        for k in range(NL):
            M.s(cle(1, k))
        for r in range(2, n + 2):
            s = s0 + (s_fin - s0) * (r - 2) / n
            rr = CP.anneau(V, self.Fc, centre(s), tangente(s), [0, 0, -1], NL, angle0)
            assert rr is not None, ("jambe", r)
            for k in range(NL):
                M.s(cle(r, k), rr[0][k], True)
        M.grille(cle, list(range(n + 2)), NL, True, "jambe")
        self.angle0_jambe = angle0

    # ------------------------------------------------------------------ fingers and toes
    def anneau_doigt(self, F, c, n, ref, a, premier_sur_ref=False):
        """Row of a finger with 2a + 2 vertices: a + 1 on top (centred on ref), a + 1 underneath; or, for the
        thumb, the first vertex in the direction ref."""
        r = CP.contour_autour(self.V, F, c, n, ref)
        if r is None:
            return None
        m = 2 * a + 2
        if premier_sur_ref:
            return CP.reechantillonner(r[0], m, c, r[1], r[2], 0.0)
        S = CP.reechantillonner(r[0], 2 * m, c, r[1], r[2], 0.0)
        return np.array([S[(-a + 2 * i) % (2 * m)] for i in range(m)])

    def doigt(self, nom, base, anneaux, bout, zone):
        """A finger: tube that starts from `base` (keys of the shared row, in ring direction), goes through the
        anchored `anneaux` and closes on `bout` (one vertex, fan quads)."""
        M = self.M
        m = len(base)
        dec = m // 2                 # the map seam runs underneath

        def cle(r, c):
            c = (c + dec) % m
            return base[c] if r == 0 else ("D", "G", nom, r, c)
        for r, A in enumerate(anneaux, 1):
            for c in range(m):
                M.s(("D", "G", nom, r, c), A[c], True, zone)
        n = len(anneaux)
        carte = "doigt_" + nom
        M.grille(cle, list(range(n + 1)), m, True, carte)
        kb = ("D", "G", nom, "bout", 0)
        M.s(kb, bout, True, zone)
        for k in range(m // 2):
            M.quad(kb, cle(n, 2 * k), cle(n, 2 * k + 1), cle(n, 2 * k + 2),
                   (carte, [(2 * k + 1, n + 1), (2 * k, n), (2 * k + 1, n), (2 * k + 2, n)]))

    def sur_surface(self, p, F):
        return G.plus_proche(np.array([p], float), self.V, F)[0][0]

    def tube_doigt(self, F, noms, ref, a, pas, premier_sur_ref=False):
        """Rings of a finger of the hand along its bones: from the fold where it leaves the palm (the
        perimeter of the cut drops to that of a finger) to the tip. Returns (anneaux, bout)."""
        V, o = self.V, self.os
        pts = [o[n] for n in noms]
        d = unitaire(pts[-1] - pts[-2])
        Vm = V[np.unique(F)]
        rel = Vm - pts[-1]
        t = rel @ d
        r = np.linalg.norm(rel - np.outer(t, d), axis=1)
        pts = pts + [pts[-1] + float(t[r < 1.0].max()) * d]
        centre, tangente, cum = self.axe(pts, lisse=0.6)
        ss = np.arange(0.0, cum[-1] - 0.2, 0.1)
        per = []
        for s in ss:
            rr = CP.contour_autour(V, F, centre(s), tangente(s), ref)
            per.append(CP.longueur(rr[0]) if rr else np.inf)
        per = np.array(per)
        seuil = 1.4 * np.median(per[len(per) // 2:])
        gros = np.flatnonzero(per > seuil)
        s0 = (ss[gros.max()] + 0.1 if len(gros) else 0.0) + 0.3
        s1 = cum[-1] - 0.3
        n = max(2, int(round((s1 - s0) / pas)))
        anneaux = []
        for s in np.linspace(s0, s1, n + 1):
            A = self.anneau_doigt(F, centre(s), tangente(s), ref, a, premier_sur_ref)
            assert A is not None, (noms[0], s)
            anneaux.append(A)
        return anneaux, self.sur_surface(pts[-1], F), round(float(s0), 2)

    # ------------------------------------------------------------------ hand
    def main(self):
        """Palm (the continuation of the arm tube), thumb in a window, four fingers on the last row."""
        M, V, o, Fm = self.M, self.V, self.os, self.Fm
        poignet = o["左手首"]
        ax = unitaire(o["左中指１"] - poignet)
        dos = unitaire(np.cross(o["左中指１"] - poignet, o["左人指１"] - o["左小指１"]))
        n0 = self.n_bras
        cle = self.cle_bras
        fen = lambda r, c: FEN_R <= r - n0 <= FEN_R + 2 and FEN_C <= c <= FEN_C + 2
        for i in range(1, NP + 1):
            if i in (2, 3):                 # where the reference's hand is a single clean contour
                rr = CP.anneau(V, Fm, poignet + (1.6 + (i - 2) * 1.0) * ax, ax, [0, 1, 0], NA)
                assert rr is not None, ("paume", i)
                for c in range(NA):
                    M.s(cle(n0 + i, c), rr[0][c], True, "main")
            else:
                for c in range(NA):
                    if not (i == FEN_R + 1 and c == FEN_C + 1):
                        M.s(cle(n0 + i, c), zone="aucune" if i == 1 else "main")
        M.grille(cle, list(range(n0, n0 + NP + 1)), NA, True, "bras",
                 existe=lambda r, c: not (fen(r, c) and fen(r + 1, c + 1)))
        infos = {}
        # thumb
        noms = ["左親指０", "左親指１", "左親指２"]
        d_p = unitaire(o[noms[1]] - o[noms[0]])
        e1 = unitaire(ax - (ax @ d_p) * d_p)
        an, bout, infos["pouce"] = self.tube_doigt(Fm, noms, e1, 3, PAS_DOIGT, True)
        R0, C0 = n0 + FEN_R, FEN_C
        tour = [(2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]     # towards the back first
        if np.cross(d_p, e1) @ dos < 0:
            tour = [tour[0]] + tour[:0:-1]
        self.doigt("pouce", [cle(R0 + a, C0 + b) for a, b in tour], an, bout, "main")
        # fingers
        for f, d in enumerate("人中薬小"):
            base = [cle(n0 + NP, 20 + 3 * f + i) for i in range(4)] + [cle(n0 + NP, 16 - 3 * f + i) for i in range(4)]
            an, bout, infos[d] = self.tube_doigt(Fm, [f"左{d}指{k}" for k in "１２３"], dos, DOIGTS[f], PAS_DOIGT)
            self.doigt("doigt%d" % f, base, an, bout, "main")
        self.info["doigts_debut_cm"] = infos

    # ------------------------------------------------------------------ foot
    def pied(self):
        """The leg tube turns around the instep (fan rows, heel included), goes forward
        to the toes, and five toes share the last row."""
        M, V = self.M, self.V
        cf = V[self.Fc].mean(1)
        Fg = self.Fc[(cf[:, 0] > 0) & (cf[:, 1] < 30)]
        nj = self.n_jambe
        cle = self.cle_jambe
        # toes: vertical cuts, narrow contours tracked from one cut to the next
        zmin = float(V[np.unique(Fg)][:, 2].min())
        chaines = []                       # each one: list of (z, x0, x1, centre)
        for z in np.arange(zmin + 6.0, zmin, -0.1):
            for Pc, ferme in CP.coupe_plan(V, Fg, np.array([0.0, 0.0, z]), [0, 0, -1]):
                if not ferme or CP.longueur(Pc) > self.rep["orteil_tour_max"]:
                    continue
                x0, x1 = float(Pc[:, 0].min()), float(Pc[:, 0].max())
                for ch in chaines:
                    z_, a0, a1, _ = ch[-1]
                    if z_ - z < 0.25 and min(x1, a1) - max(x0, a0) > 0.5 * min(x1 - x0, a1 - a0):
                        ch.append((z, x0, x1, Pc.mean(0)))
                        break
                else:
                    chaines.append([(z, x0, x1, Pc.mean(0))])
        chaines = sorted([ch for ch in chaines if len(ch) >= 4], key=lambda ch: np.mean([c[3][0] for c in ch]))
        assert len(chaines) == 5, [len(ch) for ch in chaines]
        z_orteils = max(ch[0][0] for ch in chaines)          # the first toe to separate
        self.info["orteils_z"] = [[round(ch[0][0], 1), round(ch[-1][0], 1)] for ch in chaines]
        # fan: planes that turn around a line in front of the ankle
        y_p = float(np.mean([M.pos[M.s(cle(nj, k))][1] for k in range(NL)]))
        z_p = z_orteils + 3.5
        pivot = np.array([0.0, y_p, z_p])
        rangs = []
        for i in range(1, NF + 1):
            th = np.radians(90.0 * i / NF)
            rangs.append((pivot, np.array([0.0, -np.cos(th), -np.sin(th)]), np.array([0.0, np.sin(th), -np.cos(th)])))
        for dz in (1.2, 2.4):
            rangs.append((pivot + [0, 0, -dz], np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])))
        for i, (p0, n, e1) in enumerate(rangs, 1):
            L = [Pc for Pc, ferme in CP.coupe_plan(V, Fg, p0, n) if ferme]
            assert L, ("pied", i)
            c = max(L, key=CP.longueur).mean(0)
            rr = CP.anneau(V, Fg, c, n, e1, NL, self.angle0_jambe)
            assert rr is not None, ("pied", i)
            for k in range(NL):
                M.s(cle(nj + i, k), rr[0][k], True)
        nr = len(rangs)
        for i in (nr + 1, nr + 2):
            for k in range(NL):
                M.s(cle(nj + i, k), zone="orteil")
        M.grille(cle, list(range(nj, nj + nr + 3)), NL, True, "jambe")
        dernier = nj + nr + 2
        # toes: top from the inside (big toe) outwards, underside in the other direction
        k_int = NL - MB // 2                # middle column of the inner side
        haut, bas = k_int + 1, k_int        # top: haut, haut + 1, ...; underside: ..., bas - 1, bas
        dessus = 0
        for t, (ch, a) in enumerate(zip(chaines, ORTEILS)):
            base = [cle(dernier, haut + dessus + i) for i in range(a + 1)]
            base += [cle(dernier, bas - dessus - a + i) for i in range(a + 1)]
            dessus += a
            z0, z1 = ch[0][0] - 0.2, ch[-1][0] + 0.4          # the last row far enough from the tip for a round tip
            n = max(2, int(round((z0 - z1) / PAS_ORTEIL)) + 1)
            an = []
            for z in list(np.linspace(z0, z1, n)) + [ch[-1][0] + 0.12]:      # a small row near the tip: a dome, not a point
                c = min(ch, key=lambda e: abs(e[0] - z))[3].copy()
                c[2] = z
                A = self.anneau_doigt(Fg, c, [0, 0, -1], [0, 1, 0], a)
                assert A is not None, ("orteil", t, z)
                an.append(A)
            c = ch[-1][3].copy()
            c[2] = ch[-1][0] - 0.1
            self.doigt("orteil%d" % t, base, an, self.sur_surface(c, Fg), "orteil")

    # ------------------------------------------------------------------ face orientation
    def orienter(self, n_surs):
        """Gives the quads added after the first n_surs the orientation of their neighbours (step by step)."""
        M = self.M
        par_arete = {}
        for q, (ids, _) in enumerate(M.quads):
            for a, b in zip(ids, ids[1:] + ids[:1]):
                par_arete.setdefault((min(a, b), max(a, b)), []).append(q)
        fait = [q < n_surs for q in range(len(M.quads))]
        pile = [q for q in range(n_surs)]
        while pile:
            q = pile.pop()
            ids = M.quads[q][0]
            for a, b in zip(ids, ids[1:] + ids[:1]):
                for v in par_arete[(min(a, b), max(a, b))]:
                    if fait[v]:
                        continue
                    iv, cv = M.quads[v]
                    i = iv.index(a)
                    if iv[(i + 1) % 4] == b:          # same direction on the shared edge: to be flipped
                        M.quads[v] = (iv[::-1], None if cv is None else (cv[0], cv[1][::-1]))
                    fait[v] = True
                    pile.append(v)
        assert all(fait)

    # ------------------------------------------------------------------ mirror and output
    def cle_miroir(self, k):
        if k[0] == "T":
            return ("T", k[1], (NT - k[2]) % NT)
        if k[0] == "N":
            return ("N", k[1], (NN - k[2]) % NN)
        if k[0] == "B":
            return ("B", k[1], -k[2])
        return (k[0], "D" if k[1] == "G" else "G") + tuple(k[2:])

    def refleter(self):
        """Adds the right side: mirrored vertices and quads of the left limbs."""
        M = self.M
        n0 = len(M.cles)
        for i in range(n0):
            k = M.cles[i]
            if k[0] in ("T", "N", "B"):
                continue
            km = self.cle_miroir(k)
            M.s(km, None if M.pos[i] is None else miroir(M.pos[i]), M.ancre[i], M.zone[i])
        nouveaux = []
        for ids, carte in M.quads:
            ks = [self.cle_miroir(M.cles[i]) for i in ids]
            if any(M.cles[i][0] not in ("T", "N", "B") for i in ids):
                ids2 = [M.s(k) for k in ks][::-1]
                c2 = None if carte is None else (carte[0] + "_D" if not carte[0].endswith("_D") else carte[0], carte[1][::-1])
                nouveaux.append((ids2, c2))
        M.quads += nouveaux
        self.sym = np.array([M.s(self.cle_miroir(k)) for k in M.cles])

    def tableaux(self):
        M = self.M
        n = len(M.cles)
        P = np.full((n, 3), np.nan)
        for i, p in enumerate(M.pos):
            if p is not None:
                P[i] = p
        Qd = np.array([q for q, _ in M.quads])
        return P, np.array(M.ancre), Qd, np.array(M.zone)

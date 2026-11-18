"""Fit a part from the garment library onto another ZZZ body.

    bin/blender-file -b --factory-startup --python outils/vetements/porter.py -- \
        --piece <library>/<Character>/<part> --corps BODY.pmx --sortie DIR [--nom NAME]
        [--marge-mm 3] [--peau garder|retirer] [--rendre]

The body may be a PMX that is already dressed (output of a previous fitting): the part is then
placed over the garment (a belt over a skirt). Output: DIR/NAME.pmx (body + part, textures in
DIR/tex, relative paths), DIR/portage.json (materials of the part, type for mesure.py,
fitting figures), and with --rendre DIR/rendu.png (front, back, side).

Steps (MMD units and frame, 1 unit = 8 cm):
1. skeletons: for each body bone present on both sides (same name), transform
   source head -> target head, minimal rotation of the head->tail direction, scale along the
   bone (length ratio, clamped to 0.85-1.2) and across it (size ratio). For a shoe, a tail that is a bone
   found on both sides is taken at that bone's head on the target too (a VRoid knee's own tail
   stops mid-shin), so two bones meet at their joint on the target as on the source; a deformation
   duplicate without a tail (ひざD) turns and scales like the bone it copies. This per-bone fit
   now only places the hands, the feet, the head and the part's bones (see 3). The part's own
   bones (physics chains) and the bones missing from the target follow their first fitted
   ancestor, except a deformation duplicate (足首D copies 足首's rotation from its head), which
   is placed with the bone it copies.
2. linear skinning of the part's vertices with their original weights.
3. body-to-body transport (`transport.py`): around each body segment (thigh, arm,
   pelvis, chest...), each vertex keeps its position along the segment, its angle and its offset
   from the skin; the source body is its skin where it is visible, the garment underneath
   elsewhere, the part itself for a tight-fitting part. No bound. Hands, feet and head stay on
   the per-bone fit of step 2.
4. push out of the surface underneath (body, or body + garments already fitted): each vertex
   inside, or closer than the margin (vertex, or face centre and edge midpoint less than
   15 mm inside), is moved out along the normal; the displacement is spread to neighbours within
   3 cm, until nothing is left inside (at most 12 passes; `reste_dedans` counts what is
   left). `sous_la_peau` then counts the vertices inside the body with no depth limit
   (ray parity): this is the check that rejects a part in the workshop.
5. weights: near the body (< 3 cm), the part keeps its original weights, corrected by the
   difference between the target skin and what it had underneath on its source (a panel hanging
   from the chest next to the arm does not take the arm's weights); with nothing underneath on
   the source, it takes those of the target skin. The part's own bones keep their share.
6. hidden skin (garments only): skin faces whose outgoing ray hits the part within 5 cm,
   on an area held by the body (not by a physics bone), are removed.
7. the part's rigid bodies and joints follow their bone (position and rotation).
"""
import copy
import json
import math
import os
import re
import sys
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(RACINE, "atlas"))
sys.path.insert(0, os.path.join(RACINE, "tete"))
import modele as MD  # noqa: E402
import pied as PD  # noqa: E402
import pmx as PMXL  # noqa: E402

MM = 1 / 80.0          # MMD units per mm
UNITE = 0.08           # metres per MMD unit
LACET_PIED = 20.0      # degrees: a shoe turns at most this much about the vertical to the target foot's heading
TYPES_PIED = ("chaussures", "bottes")
SEMELLE_PASSES = 3      # a posed shoe is lowered until the foot no longer goes through its sole
TYPES_MESURE = {"veste": "veste_ouverte", "gilet": "veste_ouverte", "manteau": "manteau_long",
                "haut": "haut_ample", "combinaison": "haut_ample", "robe": "jupe", "jupe": "jupe",
                "pantalon": "pantalon", "short": "short", "chaussettes": "pantalon"}


def journal(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ====================================================================== rotations

def rot_mmd(r):
    """Rotation of a PMX rigid body or joint (YXZ order: R = Ry · Rx · Rz)."""
    x, y, z = r
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def euler_mmd(R):
    x = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    return (x, math.atan2(R[0, 2], R[2, 2]), math.atan2(R[1, 0], R[1, 1]))


def rotation_minimale(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -0.999:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


# ====================================================================== skeletons

def queue(oss, i):
    o = oss[i]
    if o["queue_os"] is not None:
        q = o["queue_os"]
        return np.asarray(oss[q]["pos"], float) if 0 <= q < len(oss) and q != i else None
    d = np.asarray(o["queue_dec"], float)
    return np.asarray(o["pos"], float) + d if np.linalg.norm(d) > 1e-6 else None


def hauteur(pmx):
    p = np.asarray(pmx["sommets"]["pos"], float)
    return float(p[:, 1].max() - p[:, 1].min())


class Recalage:
    """Per-bone transforms, source -> target: x' = h_t + R · A · (x - h_s)."""

    def __init__(self, src, cible, s_global, pieds_rigides=False, chaussure=False):
        """chaussure: a shoe's fit. Its bones are then measured between the same joints on both
        bodies (_queue_cible) and a tail-less duplicate (ひざD) turns like the bone it copies. Every
        other part keeps the plain per-bone fit: its limbs are placed by the transport, and those
        two rules moved the normals and the off-segment bones of shorts and socks for nothing."""
        self.src, self.cible = src["os"], cible["os"]
        self.chaussure = chaussure
        self._en_cours = set()
        self.nom_c = {o["nom"]: i for i, o in enumerate(self.cible)}
        n = len(self.src)
        self.R = [None] * n
        self.A = [None] * n
        self.hs = [None] * n
        self.ht = [None] * n
        self.direct = np.zeros(n, bool)
        self.angles = {}
        self.doubles = {}
        for i, o in enumerate(self.src):
            j = self.nom_c.get(o["nom"])
            if j is None:
                j = self._double(i)
            if j is None:
                continue
            hs = np.asarray(o["pos"], float)
            ht = np.asarray(self.cible[j]["pos"], float)
            self.hs[i], self.ht[i] = hs, ht
            self.direct[i] = True
            qs, qt = queue(self.src, i), (self._queue_cible(i, j) if chaussure else queue(self.cible, j))
            if qs is None or qt is None:
                continue
            ds, dt = qs - hs, qt - ht
            ls, lt = np.linalg.norm(ds), np.linalg.norm(dt)
            if ls < 0.05 or lt < 0.05:
                continue
            ang = math.degrees(math.acos(max(-1, min(1, ds @ dt / ls / lt))))
            if ang > 30:          # tails defined differently: the direction is not trusted
                continue
            self.angles[o["nom"]] = round(ang, 1)
            u = ds / ls
            sa = min(1.2, max(0.85, lt / ls))
            self.R[i] = rotation_minimale(ds, dt)
            self.A[i] = s_global * np.eye(3) + (sa - s_global) * np.outer(u, u)
        # a deformation duplicate with no tail of its own (ひざD) moves like the bone it copies: inheriting
        # its parent's (足D, the thigh) would map the shin away from the ankle, and a boot shaft weighted
        # between the knee and the ankle would be torn apart by the gap between the two
        for i in range(n if chaussure else 0):
            k = self._copie(i)
            if self.R[i] is None and k is not None and self.R[k] is not None and self.hs[i] is not None:
                self.R[i], self.A[i] = self.R[k], self.A[k]
        if pieds_rigides:
            self._pieds_rigides(s_global)
        # inheritance: rotation/scale of the first ancestor that has one, head of the first fitted ancestor
        self.s_global = s_global
        for i in range(n):
            self._resoudre(i)

    def _pieds_rigides(self, s_global):
        """A shoe keeps the shape it has on its source: every bone of the source foot moves it as one
        block around the ankle's head (scale s_global, turned only about the vertical to the target
        foot's heading). The target foot is posed into it (pied.py), not the other way round."""
        cs, ct = PD.chaines(self.src), PD.chaines(self.cible)
        for nom, c in cs.items():
            if nom not in ct:
                continue
            a, b = c["ht"] - c["ha"], ct[nom]["ht"] - ct[nom]["ha"]
            a[1] = b[1] = 0.0
            lacet = math.degrees(math.atan2(np.cross(a, b)[1], float(a @ b)))
            R = PD.rotation(np.array([0.0, 1.0, 0.0]), max(-LACET_PIED, min(LACET_PIED, lacet)))
            for i in c["pied"]:
                self.hs[i], self.ht[i] = c["ha"], ct[nom]["ha"]
                self.R[i], self.A[i] = R, s_global * np.eye(3)
                self.direct[i] = True

    def _copie(self, i):
        """Source bone whose rotation bone i copies whole (grant factor 1) from the same head, or None."""
        o = self.src[i]
        a = o.get("ajout")
        k = a["source"] if a else -1
        if not (0 <= k < len(self.src)) or k == i or not a.get("rotation") or abs(a.get("facteur", 0) - 1) > 1e-3:
            return None
        if np.linalg.norm(np.asarray(o["pos"], float) - np.asarray(self.src[k]["pos"], float)) > 1e-3:
            return None
        return k

    def _queue_cible(self, i, j):
        """Target end of source bone i (target bone j): when the source's tail is another bone found on
        the target (same name, or the bone a missing duplicate copies), that bone's head, so that both
        sides measure the bone between the same joints; otherwise the target bone's own tail."""
        q = self.src[i]["queue_os"]
        if q is not None and 0 <= q < len(self.src) and q != i:
            jq = self.nom_c.get(self.src[q]["nom"])
            if jq is None:
                jq = self._double(q)
            if jq is not None and jq != j:
                return np.asarray(self.cible[jq]["pos"], float)
        return queue(self.cible, j)

    def _double(self, i):
        """Target bone standing for source bone i when the target lacks it: the bone whose rotation i copies
        whole (grant factor 1) from the same head. That is a deformation duplicate (MMD 足首D of 足首): a
        target without duplicates (a VRoid body) moves the duplicate's vertices with the original. Without
        it, the duplicate would follow its first fitted ancestor (下半身) and a shoe would land at the hips' offset."""
        k = self._copie(i)
        if k is None or i in self._en_cours:       # two bones copying each other: no original
            return None
        j = self.nom_c.get(self.src[k]["nom"])
        if j is None:
            self._en_cours.add(i)
            j = self._double(k)
            self._en_cours.discard(i)
        if j is not None:
            self.doubles[self.src[i]["nom"]] = self.cible[j]["nom"]
        return j

    def _resoudre(self, i):
        if self.R[i] is not None and self.hs[i] is not None:
            return
        p = self.src[i]["parent"]
        if 0 <= p < len(self.src) and p != i:
            self._resoudre(p)
            if self.hs[i] is None:      # bone missing from the target: follows its parent (rigidly)
                self.hs[i], self.ht[i] = self.hs[p], self.ht[p]
            if self.R[i] is None:
                self.R[i], self.A[i] = self.R[p], self.A[p]
        else:
            if self.hs[i] is None:
                self.hs[i] = self.ht[i] = np.zeros(3)
            if self.R[i] is None:
                self.R[i], self.A[i] = np.eye(3), self.s_global * np.eye(3)

    def point(self, i, x):
        return self.ht[i] + self.R[i] @ (self.A[i] @ (np.asarray(x, float) - self.hs[i]))

    def points(self, I, W, X):
        """LBS of X (n, 3) with indices I (n, 4) and weights W (n, 4)."""
        out = np.zeros_like(X)
        for b in np.unique(I[I >= 0]):
            m = (I == b) & (W > 0)
            rows = np.where(m.any(1))[0]
            w = (W * m).sum(1)[rows]
            Y = self.ht[b] + ((X[rows] - self.hs[b]) @ self.A[b].T) @ self.R[b].T
            out[rows] += w[:, None] * Y
        return out

    def vecteurs(self, I, W, N):
        out = np.zeros_like(N)
        for b in np.unique(I[I >= 0]):
            m = (I == b) & (W > 0)
            rows = np.where(m.any(1))[0]
            w = (W * m).sum(1)[rows]
            out[rows] += w[:, None] * (N[rows] @ self.R[b].T)
        return out / np.maximum(np.linalg.norm(out, axis=1), 1e-12)[:, None]


# ====================================================================== surfaces (BVH)

def bvh(P, F):
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons([tuple(p) for p in P], [tuple(int(x) for x in f) for f in F], all_triangles=True)


def normales_faces(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-20)[:, None]


def signe_dehors(P, F, nf):
    """+1 if the face winding gives outward normals (MMD frame), otherwise -1: vote of the faces
    against the body's vertical axis."""
    c = P[F].mean(1)
    axe = np.array([np.median(P[:, 0]), 0, np.median(P[:, 2])])
    r = c - axe
    r[:, 1] = 0
    return 1.0 if float((nf * r).sum(1).mean()) > 0 else -1.0


def sous_la_peau(Z, arbre, mm, sg=1.0):
    """Vertices inside the body, with no depth limit: parity of the surface crossings along three
    rays (majority of the three, since the body has openings). A short ray along the normal does
    not see a part sunk several centimetres deep; parity does.
    Parity is wrong near openings: a vertex is only counted if the nearest face also sees it from
    behind (two independent clues).
    Counting is done by position: the centre of a fan-shaped cap (the waist bottom of a skirt,
    hidden in the body as on its source) is a single point repeated about twenty times,
    and twenty buried "vertices" got the part rejected for a single point."""
    Z = np.asarray(Z, float)
    if len(Z):
        Z = Z[np.unique(np.round(Z / (0.1 * mm)).astype(np.int64), axis=0, return_index=True)[1]]
    dirs = np.array([(0.577, 0.577, 0.578), (-0.8, 0.1, 0.59), (0.3, -0.2, -0.933)])
    votes = np.zeros(len(Z), int)
    for D in dirs:
        for i, z in enumerate(Z):
            n, o = 0, np.asarray(z, float)
            for _ in range(64):
                loc, _n, _k, dist = arbre.ray_cast(tuple(o), tuple(D))
                if loc is None:
                    break
                n += 1
                o = o + D * (dist + 1e-4)
            votes[i] += n % 2
    prof = np.zeros(len(Z))
    for i in np.where(votes >= 2)[0]:
        loc, _n, _k, dist = arbre.find_nearest(tuple(Z[i]))
        if loc is not None and sg * float(np.dot(np.asarray(Z[i]) - np.asarray(loc), _n)) < 0:
            prof[i] = dist
    d = prof > 1 * mm
    return {"part": round(float(d.mean()), 4) if len(Z) else 0.0, "sommets": int(d.sum()),
            "prof_p50_mm": round(float(np.median(prof[d]) / mm), 1) if d.any() else 0.0,
            "prof_max_mm": round(float(prof.max() / mm), 1) if d.any() else 0.0}


def couvercles(X, Fp, oss):
    """Caps of the part: a closed fan of faces around a centre, crossed by a body bone.
    This is the bottom that closes an opening around the body (armhole of a jacket, end of a
    sleeve, top of a sock): the bone continues on each side, so the body passes through it, the
    bottom is hidden inside on its source and stays hidden on the target. A bone that ends inside
    the part (hand in a glove, foot in a shoe) does not count: that end is visible.
    Returns (centre vertices, cap faces) as booleans and, per cap, (centre vertices,
    rim vertices). The centre may be a single vertex or a point repeated on each face."""
    import transport as TR
    cle = np.unique(np.round(X / (0.1 * MM)).astype(np.int64), axis=0, return_inverse=True)[1].ravel()
    Fw = cle[Fp]
    vrai = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 2] != Fw[:, 0])
    nb = np.bincount(Fw[vrai].ravel(), minlength=int(cle.max()) + 1)
    axes = []
    for a, b in COUVERCLE_OS:
        h, q = TR._joint(oss, (a,)), TR._joint(oss, (b,))
        if h is not None and q is not None and np.linalg.norm(q - h) > 0.05:
            axes.append((h, q))
    centres, faces, liens = np.zeros(len(X), bool), np.zeros(len(Fp), bool), []
    for c in np.where(nb >= COUVERCLE_MIN)[0]:
        fs = np.where(vrai & (Fw == c).any(1))[0]
        tour = Fw[fs][Fw[fs] != c]
        if len(tour) != 2 * len(fs) or (np.bincount(tour)[np.unique(tour)] != 2).any():
            continue                                    # open fan: not a bottom
        T = X[Fp[fs]]
        e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
        perce = False
        for a, b in axes:                               # segment against triangles (Möller-Trumbore)
            d = b - a
            pv = np.cross(d, e2)
            det = (e1 * pv).sum(1)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            tv = a - T[:, 0]
            uu = (tv * pv).sum(1) * inv
            qv = np.cross(tv, e1)
            vv = (qv @ d) * inv
            tt = (e2 * qv).sum(1) * inv
            if (ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt >= 0) & (tt <= 1)).any():
                perce = True
                break
        if perce:
            centres[cle == c] = True
            faces[fs] = True
            liens.append((np.where(cle == c)[0], np.where(np.isin(cle, np.unique(tour)))[0]))
    return centres, faces, liens


# ====================================================================== settings

PASSES_MAX = 12
COUVERCLE_MIN = 8         # faces around the centre of a cap
COUVERCLE_OS = tuple(     # body bones that continue on each side of what they cross
    [("下半身", "上半身"), ("上半身", "上半身2"), ("上半身2", "首"), ("首", "頭")]
    + [(c + a, c + b) for c in "左右" for a, b in (("肩", "腕"), ("腕", "ひじ"), ("ひじ", "手首"), ("足", "ひざ"), ("ひざ", "足首"))]
    + [("上半身2", c + "肩") for c in "左右"] + [("下半身", c + "足") for c in "左右"])
MOULANTS = ("chaussettes", "collant")     # the part counts as the body: it is placed on the skin


# ====================================================================== rings attached to their carrier

ANNEAU_SECTEURS = 24
ANNEAU_ENTOURE_MIN = 0.5    # share of sectors where the ring has something between its centre and itself (porter.entoure)
ANNEAU_TOUR_MIN = 0.75       # share of sectors covered by the part around its axis
BORD_TOUR_MIN = 0.6          # share of sectors where a garment has a measurable (top or bottom) edge
BORD_ECART = 40 * MM         # distance (along the axis) between the ring and the edge of the source carrier
ECART_MAX = 20 * MM          # clearance kept between the ring and what it surrounds (source), clamped
APPUI_MIN = 0.5              # share of hits under the ring that land on fabric, below which it slides onto its carrier


def repere_anneau(X):
    """Centre, axis (normal of the mean plane, pointing up or away from the body), two axes
    of the plane, and share of sectors covered around the axis."""
    c = X.mean(0)
    _, _, Vt = np.linalg.svd(X - c, full_matrices=False)
    n = Vt[2]
    if abs(n[1]) >= 0.3:
        n = n if n[1] > 0 else -n
    else:
        o = np.array([c[0], 0, c[2]])
        n = n if n @ o >= 0 else -n
    e1 = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(e1) < 0.3:
        e1 = np.cross(n, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    rel = X - c
    ang = np.arctan2(rel @ e2, rel @ e1)
    sect = ((ang + np.pi) / (2 * np.pi) * ANNEAU_SECTEURS).astype(int) % ANNEAU_SECTEURS
    return c, n, e1, e2, len(np.unique(sect)) / ANNEAU_SECTEURS


def bords(V, c, n, e1, e2, rmax, hmax):
    """Top and bottom edges of a garment (vertices V) around the axis (c, n): per sector, the
    highest vertex ("haut" edge: the garment is below, waistband of a skirt) or the lowest
    ("bas" edge: hem of a top). Returns {direction: (centre, normal, coverage, height)} for
    edges that are complete enough; centre and normal of a plane fitted to the edge points."""
    rel = V - c
    h = rel @ n
    rv = rel - h[:, None] * n
    r = np.linalg.norm(rv, axis=1)
    m = (r < rmax) & (np.abs(h) < hmax) & (r > 1e-6)
    if m.sum() < 8:
        return {}
    idx = np.where(m)[0]
    ang = np.arctan2(rv[idx] @ e2, rv[idx] @ e1)
    sect = ((ang + np.pi) / (2 * np.pi) * ANNEAU_SECTEURS).astype(int) % ANNEAU_SECTEURS
    out = {}
    for sens, choix in (("haut", np.argmax), ("bas", np.argmin)):
        pts = []
        for s in np.unique(sect):
            k = idx[sect == s]
            pts.append(V[k[choix(h[k])]])
        cov = len(pts) / ANNEAU_SECTEURS
        if cov < BORD_TOUR_MIN:
            continue
        pts = np.array(pts)
        cb = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - cb, full_matrices=False)
        # a sector with no edge vertex (sparse mesh, slit) gives a point much lower down
        # (Anby, skirt: 3 sectors out of 24 at -50..-200 mm) that tilts the plane: the worst point is
        # dropped and the plane refitted, as long as it clearly stands apart from the others
        while len(pts) > 6:
            e = np.abs((pts - cb) @ Vt[2])
            if e.max() <= max(10 * MM, 3 * np.median(e)):
                break
            pts = np.delete(pts, np.argmax(e), 0)
            cb = pts.mean(0)
            _, _, Vt = np.linalg.svd(pts - cb, full_matrices=False)
        nb = Vt[2] if Vt[2] @ n >= 0 else -Vt[2]
        if nb @ n < 0.7:            # edge tilted too much: not a garment waistband
            nb = n
        out[sens] = (cb, nb, cov, float((cb - c) @ n))
    return out


def rayons_surface(arbre, c, n, e1, e2, hs, r_ext, tissu=None):
    """Per sector, radius of the outermost surface under the ring (ray cast from r_ext
    towards the axis, at heights hs); nan if nothing is hit. With `tissu` (boolean per tree face),
    also returns the share of hits that land on fabric (not on skin)."""
    out = np.full(ANNEAU_SECTEURS, np.nan)
    nt = nk = 0
    for s in range(ANNEAU_SECTEURS):
        a = (s + 0.5) / ANNEAU_SECTEURS * 2 * np.pi - np.pi
        d = math.cos(a) * e1 + math.sin(a) * e2
        best = np.nan
        for h in hs:
            o = c + h * n + r_ext[s] * d
            loc, _, fi, dist = arbre.ray_cast(tuple(o), tuple(-d), float(r_ext[s]))
            if loc is not None:
                rs = r_ext[s] - dist
                best = rs if np.isnan(best) else max(best, rs)
                if tissu is not None:
                    nk += 1
                    nt += bool(tissu[fi])
        out[s] = best
    if tissu is not None:
        return out, (nt / nk if nk else 0.0)
    return out


def entoure(arbre, c, n, e1, e2, hs, rin):
    """Share of sectors where, going from the centre towards the ring, one meets what it surrounds
    (a limb, the waist, the neck, the head) before reaching its inner edge. A plate or a pendant
    lying next to the body also covers the full turn around its centre, but surrounds nothing."""
    k = t = 0
    for s in range(ANNEAU_SECTEURS):
        if np.isnan(rin[s]):
            continue
        a = (s + 0.5) / ANNEAU_SECTEURS * 2 * np.pi - np.pi
        d = math.cos(a) * e1 + math.sin(a) * e2
        k += 1
        t += any(arbre.ray_cast(tuple(c + h * n), tuple(d), float(rin[s]) + 5 * MM)[0] is not None for h in hs)
    return t / k if k else 0.0


def profil_anneau(X, c, n, e1, e2):
    """Per sector: inner (min) radius of the ring, min/max heights; sector of each vertex."""
    rel = X - c
    h = rel @ n
    rv = rel - h[:, None] * n
    r = np.linalg.norm(rv, axis=1)
    ang = np.arctan2(rv @ e2, rv @ e1)
    sect = ((ang + np.pi) / (2 * np.pi) * ANNEAU_SECTEURS).astype(int) % ANNEAU_SECTEURS
    rin = np.full(ANNEAU_SECTEURS, np.nan)
    for s in np.unique(sect):
        rin[s] = r[sect == s].min()
    return rin, h, r, ang, rv


def combler(v):
    """Empty sectors (nan): interpolated circularly between the known neighbours."""
    v = np.asarray(v, float).copy()
    ok = ~np.isnan(v)
    if not ok.any():
        return np.zeros_like(v)
    k = np.arange(len(v))
    kk = np.concatenate([k[ok] - len(v), k[ok], k[ok] + len(v)])
    vv = np.tile(v[ok], 3)
    return np.interp(k, kk, vv)


DESSOUS_PRES = 5 * MM     # below this, the direction to the surface is not reliable
DESSOUS_COS = 0.7         # beyond 45° from the normal, the nearby point is beside, not underneath
PORTEE_LOIN = 15 * 10     # mm: reach of the target skin when what lies under the source is known


def accrocher_anneau(fiche, P, src, Y, s_glob, C, Fc, mfc, fdes, arbre_dessous, marge, sur_tissu=True, regler=True):
    """A ring-shaped accessory (belt, garter, armband...) is fitted onto the garment worn by the
    target: on the source, its position is measured relative to the edge of the garment it
    surrounds (height along the axis, relative tilt of the plane, clearance between the ring and
    the fabric); on the target, the edge of the same direction (top of a skirt, of shorts) nearest
    to the part fitted by bones is found, the ring is placed there (same relative height, plane
    parallel to the edge), then its circumference is adjusted sector by sector onto the worn
    surface (body + garments), at the source's clearance. With no carrying edge, only the
    circumference is adjusted. Returns None if the part is not a ring, otherwise (Z, rigid
    transform (Rm, t), info)."""
    Xs = np.asarray(P["sommets"]["pos"], float)
    c_s, n_s, e1_s, e2_s, tour = repere_anneau(Xs)
    if tour < ANNEAU_TOUR_MIN:
        return None
    info = {"tour_source": round(tour, 2)}
    rin_s, h_s, _, _, _ = profil_anneau(Xs, c_s, n_s, e1_s, e2_s)
    rmed = float(np.nanmedian(rin_s))
    # ---- source: garments from the cutting, surface under the ring (everything except the part)
    dos = os.path.dirname(os.path.dirname(os.path.join(fiche["_dossier"], "")))
    dec = json.load(open(os.path.join(dos, "decoupe.json"), encoding="utf-8"))
    Ps = np.asarray(src["sommets"]["pos"], float)
    Fs = np.asarray(src["faces"], np.int64).reshape(-1, 3)
    a_moi = set(dec["pieces"][fiche["id"]]["faces"]) | set(dec["pieces"][fiche["id"]]["faces_calque"])
    surf = [f for f in dec["peau"]["faces"]]
    porteur_s = None
    for pid, pc in dec["pieces"].items():
        if pid == fiche["id"]:
            continue
        surf.extend(pc["faces"])
        fch = os.path.join(dos, pid, "fiche.json")
        if not os.path.exists(fch) or json.load(open(fch, encoding="utf-8"))["classe"] != "vetement":
            continue
        V = Ps[np.unique(Fs[pc["faces"]])]
        for sens, (cb, nb, cov, hb) in bords(V, c_s, n_s, e1_s, e2_s, 2.5 * rmed, 40 * 10 * MM).items():
            if h_s.min() - BORD_ECART <= hb <= h_s.max() + BORD_ECART:
                if porteur_s is None or abs(hb) < abs(porteur_s[2][3]):
                    porteur_s = (pid, sens, (cb, nb, cov, hb))
    surf = np.array(sorted(set(surf) - a_moi), np.int64)
    arbre_s = bvh(Ps, Fs[surf])
    tissu_s = ~np.isin(surf, dec["peau"]["faces"])
    r_ext = np.nan_to_num(rin_s, nan=rmed) + 60 * MM
    hs = [float(np.percentile(h_s, q)) for q in (15, 50, 85)]
    rsurf_s, appui_s = rayons_surface(arbre_s, c_s, n_s, e1_s, e2_s, hs, r_ext, tissu_s)
    jeu = np.clip(combler(rin_s - rsurf_s), 0, ECART_MAX)
    info["jeu_source_mm"] = round(float(np.median(jeu)) / MM, 1)
    info["appui_tissu_source"] = round(appui_s, 2)
    info["entoure_source"] = round(entoure(arbre_s, c_s, n_s, e1_s, e2_s, hs, rin_s), 2)
    # a ring surrounds something: a wing, a bag, a harness also cover the full turn around
    # their centre, and adjusting their "circumference" onto the body stretched them around it
    # (Phionie's wings: 7.5 cm; Koleda's harness: 7.3 cm). For those, `porter` also tries placing
    # them as one block without adjusting the circumference (`regler=False`) and the transport.
    info["anneau"] = bool(info["entoure_source"] >= ANNEAU_ENTOURE_MIN)
    # ---- target: frame of the part fitted by bones, edges of the garments worn
    c_y, n_y, e1_y, e2_y, _ = repere_anneau(Y)
    noms_c = [m["nom"] for m in C["materiaux"]]
    Pc = np.asarray(C["sommets"]["pos"], float)
    tissu_c = ~np.array([bool(MD.PEAU.match(nm)) for nm in noms_c])[mfc[fdes]]
    Q = rotation_minimale(n_s, n_y)
    c_t = c_y
    bord_t = None
    if porteur_s is not None and sur_tissu:
        pid, sens, (cb_s, nb_s, _, _) = porteur_s
        info["porteur_source"] = "%s (edge %s)" % (pid, sens)
        best = None
        for i, nm in enumerate(noms_c):
            if MD.NON_VETEMENT.match(nm) or MD.PEAU.match(nm) or C["materiaux"][i]["n_indices"] == 0:
                continue
            V = Pc[np.unique(Fc[mfc == i])]
            b = bords(V, c_y, n_y, e1_y, e2_y, 2.5 * rmed * s_glob, 40 * 10 * MM).get(sens)
            if b is not None and (best is None or abs(b[3]) < abs(best[1][3])):
                best = (nm, b)
        if best is not None and abs(best[1][3]) < 30 * 10 * MM:
            nm, (cb_t, nb_t, cov_t, hb_t) = best
            # centre: same position relative to the edge as on the source; plane: that of the carrying edge.
            # The source's relative tilt is not kept: it was measured against another garment
            # (Lucy: belt tilted 7° on the edge of her shorts), and carried over to another edge
            # it would tilt the ring on it; a ring held by an edge follows that edge.
            c_t = cb_t + s_glob * (rotation_minimale(nb_s, nb_t) @ (c_s - cb_s))
            Q = rotation_minimale(n_s, nb_t)
            info["porteur_cible"] = "%s (edge %s, %d %% of the circumference)" % (nm, sens, round(100 * cov_t))
            info["deplacement_bord_mm"] = round(float(np.linalg.norm(c_t - c_y)) / MM, 1)
            info["inclinaison_bord_deg"] = round(math.degrees(math.acos(max(-1, min(1, nb_s @ nb_t)))), 1)
            bord_t = (cb_t, nb_t, sens)
    n_t, e1_t, e2_t = Q @ n_s, Q @ e1_s, Q @ e2_s
    hs_t = [s_glob * h for h in hs]

    def appui(ct):
        Zt = Xs @ (s_glob * Q).T + (ct - s_glob * Q @ c_s)
        rin = profil_anneau(Zt, ct, n_t, e1_t, e2_t)[0]
        return rayons_surface(arbre_dessous, ct, n_t, e1_t, e2_t, hs_t,
                              np.nan_to_num(rin, nan=rmed * s_glob) + 80 * MM, tissu_c)[1]
    # a fitted accessory rests on its carrier: if the ring lands mostly on skin on the target
    # (bare waist above a low skirt), it slides along the axis towards the carrier until it is
    # entirely on it (its top edge at the garment's top edge, or its bottom edge at the hem).
    # This holds even if the source wore it on the skin (Lucy: belt on the bare belly,
    # support 0.18): once fitted, it holds to the garment.
    if bord_t is not None:
        a0 = appui(c_t)
        info["appui_tissu_cible"] = round(a0, 2)
        cb_t, nb_t, sens = bord_t
        h_rel = s_glob * h_s - s_glob * (h_s.mean())          # heights of the ring around its centre
        haut_anneau = float((c_t - cb_t) @ nb_t) + (np.percentile(h_rel, 97) if sens == "haut" else np.percentile(h_rel, 3))
        if a0 < APPUI_MIN and ((sens == "haut" and haut_anneau > 0) or (sens == "bas" and haut_anneau < 0)):
            dec_h = -(haut_anneau + (2 * MM if sens == "haut" else -2 * MM))
            c_t = c_t + dec_h * nb_t
            info["glisse_sur_porteur_mm"] = round(float(dec_h) / MM, 1)
            info["appui_tissu_cible"] = round(appui(c_t), 2)
    # rigid placement: x' = c_t + s Q (x - c_s)
    Rm = s_glob * Q
    t = c_t - Rm @ c_s
    Z = Xs @ Rm.T + t
    if not regler:
        return Z, (Rm, t, Q), info
    # ---- circumference: per sector, the ring rests on the worn surface, at the source's clearance
    rin_t, h_t, _, ang_t, rv_t = profil_anneau(Z, c_t, n_t, e1_t, e2_t)
    r_ext = np.nan_to_num(rin_t, nan=rmed * s_glob) + 80 * MM
    rsurf_t = rayons_surface(arbre_dessous, c_t, n_t, e1_t, e2_t, hs_t, r_ext)
    voulu = rsurf_t + np.maximum(jeu * s_glob, marge)
    delta = combler(voulu - rin_t)
    delta = (np.roll(delta, 1) + 2 * delta + np.roll(delta, -1)) / 4
    fs = (ang_t + np.pi) / (2 * np.pi) * ANNEAU_SECTEURS - 0.5
    s0 = np.floor(fs).astype(int)
    b = fs - s0
    dv = (1 - b) * delta[s0 % ANNEAU_SECTEURS] + b * delta[(s0 + 1) % ANNEAU_SECTEURS]
    ru = rv_t / np.maximum(np.linalg.norm(rv_t, axis=1), 1e-9)[:, None]
    Z = Z + dv[:, None] * ru
    info["tour_mm"] = {"min": round(float(delta.min()) / MM, 1), "median": round(float(np.median(delta)) / MM, 1),
                       "max": round(float(delta.max()) / MM, 1)}
    return Z, (Rm, t, Q), info


# ====================================================================== fitting

SEMELLE_SOL = 3 * 10 * MM     # a sole touches the ground: the bottom of the part is closer than this to y = 0 on its source
SEMELLE_AIRE = 40.0           # cm² of flat faces (vertical normal) in the bottom 4 cm: a sock has none


def a_semelle(P):
    """The part has a sole: it reaches the ground and its underside is flat. Measured on the
    library: 0 to 4 cm² of flat area for socks, 72 to 580 for shoes and boots (only one boot
    under 72, with no sole edge)."""
    X = np.asarray(P["sommets"]["pos"], float)
    F = np.asarray(P["faces"], np.int64).reshape(-1, 3)
    if not len(F) or X[:, 1].min() > SEMELLE_SOL:
        return False
    T = X[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    a = np.linalg.norm(n, axis=1) / 2
    plat = (T[:, :, 1].max(1) < X[:, 1].min() + 40 * MM) & (np.abs(n[:, 1]) > 0.9 * 2 * a)
    return float(a[plat].sum()) / (10 * MM) ** 2 >= SEMELLE_AIRE


def charger_piece(dossier):
    if dossier.endswith(".pmx"):
        dossier = os.path.dirname(dossier)
    fiche = json.load(open(os.path.join(dossier, "fiche.json"), encoding="utf-8"))
    fiche["_dossier"] = os.path.abspath(dossier)
    # a garment is fitted with its trimmings (buttons, pockets, straps: bibliotheque.garnir_perso)
    g = fiche.get("garnie")
    chemin = os.path.join(dossier, fiche["pmx"])
    if g and os.path.exists(os.path.join(dossier, g["pmx"])):
        fiche["os_propres"] = g["os_propres"]
        chemin = os.path.join(dossier, g["pmx"])
    # the cutting names "chaussettes" (socks) whatever dresses the leg when the character has no
    # separate shoes: with a sole, it is a boot (rigid object, not a second skin)
    if fiche.get("type") == "chaussettes" and a_semelle(PMXL.lire(chemin)):
        fiche["type"] = "bottes"
    # it also sometimes separates the sole and heel of a shoe from its upper, and names them
    # "chevillere" (ankle accessory): it is a shoe, fitted and checked as such
    elif fiche.get("classe") == "accessoire" and fiche.get("sorte") == "chevillere" and a_semelle(PMXL.lire(chemin)):
        fiche["classe"], fiche["type"] = "vetement", "chaussures"
    return fiche, chemin


def couches_internes(X, N, Ny, Z, Fp, ecart_max_mm=30.0, garde_mm=1.5):
    """A single part can have several layers (lining under the outer layer of a skirt). The body
    pushes back the one touching it; without this it pokes out through the outer one. Each vertex
    that had its own part above it in the original model (X, along its normal N) must stay that
    way: it is the outer layer that moves away, so as not to push the lining back into the skin.
    Returns (corrected Z, number of vertices moved)."""
    from mathutils import Vector
    from mathutils.kdtree import KDTree
    a = bvh(X, Fp)
    liens = []
    for v in range(len(X)):
        n0 = np.linalg.norm(N[v])
        n1 = np.linalg.norm(Ny[v])
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        n = N[v] / n0
        hit, nf, f, dist = a.ray_cast(Vector(X[v] + n * 0.3 * MM), Vector(n), ecart_max_mm * MM)
        if hit is None or dist < garde_mm * MM or v in Fp[f]:
            continue
        # a layer lying above is hit on its back side; hit on its front side, it is the facing
        # fabric (the other leg at the crotch, a neighbouring finger), not a layer
        if float(np.dot(nf, n)) <= 0:
            continue
        liens.append((v, Fp[f], barycentre(np.asarray(hit), X[Fp[f]]), Ny[v] / n1, min(dist, garde_mm * MM)))
    # the other way round: a vertex of the outer layer under which the lining passes (coarse mesh:
    # the lining can poke through between the vertices of the outer layer)
    dessous = []
    for v in range(len(X)):
        n0 = np.linalg.norm(N[v])
        n1 = np.linalg.norm(Ny[v])
        if n0 < 1e-9 or n1 < 1e-9:
            continue
        n = -N[v] / n0
        hit, nf, f, dist = a.ray_cast(Vector(X[v] + n * 0.3 * MM), Vector(n), ecart_max_mm * MM)
        if hit is None or dist < garde_mm * MM or v in Fp[f]:
            continue
        if float(np.dot(nf, n)) >= 0:      # the lining underneath is hit on its front side
            continue
        dessous.append((v, Fp[f], barycentre(np.asarray(hit), X[Fp[f]]), Ny[v] / n1, min(dist, garde_mm * MM)))
    bouges = set()
    rr = 2 * 10 * MM
    for _ in range(3):
        d0, prof = np.zeros_like(Z), np.zeros(len(Z))
        for v, tri, bc, d, garde in liens:
            e = float((Z[v] - (bc[:, None] * Z[tri]).sum(0)) @ d) + garde
            if e > 0:
                for i in tri:
                    if e > prof[i]:
                        prof[i], d0[i] = e, e * d
        for v, tri, bc, d, garde in dessous:
            e = garde - float((Z[v] - (bc[:, None] * Z[tri]).sum(0)) @ d)
            if e > prof[v]:
                prof[v], d0[v] = e, e * d
        dedans = np.where(prof > 0)[0]
        if not len(dedans):
            break
        kd = KDTree(len(Z))
        for i, z in enumerate(Z):
            kd.insert(tuple(z), i)
        kd.balance()
        dlt, mag = np.zeros_like(Z), np.zeros(len(Z))
        nyu = Ny / np.maximum(np.linalg.norm(Ny, axis=1), 1e-12)[:, None]
        for i in dedans:
            for _, j, dd in kd.find_range(tuple(Z[i]), rr):
                if j != i and float(d0[i] @ nyu[j]) <= 0:
                    continue
                f = (1 - dd / rr) ** 2
                if prof[i] * f > mag[j]:
                    mag[j], dlt[j] = prof[i] * f, d0[i] * f
        Z = Z + dlt
        bouges.update(int(i) for i in dedans)
    return Z, len(bouges)


ALLONGE = 1.3             # an edge longer than this (times its scaled length) counts as stretched
MEMBRE_GAIN = 0.01     # stretch share a limb must gain to be fitted differently from the rest of the part
RIGIDE_MIN = 8            # vertices: below this, a piece has no shape to keep
RIGIDE_ACCORD = 0.5       # share of transported vertices that must agree with a one-block placement
RIGIDE_ECHELLE = (0.85, 1.2)


def morceaux(X, Fp):
    """Piece number of each vertex: faces that touch, coincident vertices merged."""
    _, w = np.unique(np.round(X, 4), axis=0, return_inverse=True)
    w = w.ravel()
    e = np.arange(int(w.max()) + 1)
    Fw = w[Fp]
    for _ in range(500):
        avant = e.copy()
        m = e[Fw].min(1)
        for k in range(3):
            np.minimum.at(e, Fw[:, k], m)
        e = e[e]
        if (e == avant).all():
            break
    return e[w]


MEMBRE_DES_BOUTS = {"頭": 0, "左手首": 1, "右手首": 2, "左足首": 3, "右足首": 4, "左足首D": 3, "右足首D": 4}


def membres(X, Fp, Is, W, src):
    """Limb of each vertex (0 trunk and head, 1-2 arms, 3-4 legs; -1 none), the same for a whole
    piece: the one that weighs most in the piece's weights, the part's own bones counted for the
    limb they are attached to."""
    import transport as TR
    oss = src["os"]
    seg = TR._segments_os(oss)
    noms_seg = list(TR.SEGMENTS)
    ch_de_seg = np.full(len(noms_seg), -1)
    for c, chaine in enumerate(TR.CHAINES):
        for n in chaine:
            ch_de_seg[noms_seg.index(n)] = c
    ch_os = np.where(seg >= 0, ch_de_seg[np.maximum(seg, 0)], -1)
    for i in np.where(seg == -2)[0]:        # hand, foot, head: the limb that carries them
        k, n = int(i), 0
        while 0 <= k < len(oss) and n < 256 and oss[k]["nom"] not in MEMBRE_DES_BOUTS:
            k, n = oss[k]["parent"], n + 1
        if 0 <= k < len(oss):
            ch_os[i] = MEMBRE_DES_BOUTS.get(oss[k]["nom"], -1)
    ch = np.where(Is >= 0, ch_os[np.clip(Is, 0, len(ch_os) - 1)], -1)
    c = morceaux(X, Fp)
    out = np.full(len(X), -1)
    for z in np.unique(c):
        v = c == z
        poids = [float((W[v] * (ch[v] == m)).sum()) for m in range(len(TR.CHAINES))]
        if max(poids) > 0:
            out[v] = int(np.argmax(poids))
    return out


def morceaux_rigides(X, Fp, I, W):
    """Pieces of the part (faces that touch, coincident vertices merged) whose vertices are all
    held by one and the same bone, unshared: on its source such a piece never deforms
    (backpack, buckle, spike, plate). Returns the list of vertex index arrays."""
    c = morceaux(X, Fp)
    k = W.argmax(1)
    os1 = I[np.arange(len(I)), k]
    seul = W.max(1) > 0.999
    out = []
    for z in np.unique(c):
        v = np.where(c == z)[0]
        if len(v) >= RIGIDE_MIN and seul[v].all() and (os1[v] == os1[v[0]]).all():
            out.append(v)
    return out


def poser_rigide(A, B):
    """Similarity transform (rotation, clamped scale, translation) that brings A onto B, fitted on
    the points that agree (the others set aside): returns (A placed, share of agreeing points)."""
    ok = np.ones(len(A), bool)
    for _ in range(8):
        a, b = A[ok] - A[ok].mean(0), B[ok] - B[ok].mean(0)
        U, sv, Vt = np.linalg.svd(a.T @ b)
        d = np.sign(np.linalg.det(U @ Vt)) or 1.0
        D = np.diag([1.0, 1.0, d])
        ech = min(RIGIDE_ECHELLE[1], max(RIGIDE_ECHELLE[0], (sv * np.diag(D)).sum() / max((a * a).sum(), 1e-12)))
        Pz = (A - A[ok].mean(0)) @ (ech * (U @ D @ Vt)) + B[ok].mean(0)
        r = np.linalg.norm(Pz - B, axis=1)
        ok2 = r <= max(2.5 * np.median(r), 3 * MM)
        if ok2.sum() < 4 or (ok2 == ok).all():
            break
        ok = ok2
    return Pz, float(ok.mean()), U @ D @ Vt


def poids_np(pmx):
    I = np.asarray(pmx["sommets"]["os"], np.int64)
    W = np.asarray(pmx["sommets"]["poids"], float)
    W = np.where(I >= 0, W, 0)
    W = W / np.maximum(W.sum(1), 1e-12)[:, None]
    return I, W


def faces_mats(pmx):
    F = np.asarray(pmx["faces"], np.int64).reshape(-1, 3)
    mf = np.zeros(len(F), np.int64)
    d = 0
    for i, m in enumerate(pmx["materiaux"]):
        k = m["n_indices"] // 3
        mf[d:d + k] = i
        d += k
    return F, mf


POINTURES = tuple(1.0 + 0.03 * i for i in range(11))   # a shoe grows at most 30 % to fit the foot (Promeia on the house body: 1.3, nothing smaller holds the foot)
POINTURE_LIBRE = 1.15     # beyond this, a size is only tried when none up to it holds the foot without lowering
                          # the sole, and only taken if it holds the foot (nothing inside, sole not lowered)
SEMELLE_JEU = 5 * MM      # lowering the shoe by more than this onto its sole means it does not hold the foot
PIED_PLAT = 8.0           # degrees: a foot posed less than this in its shoe (no heel) stays at rest in it
POINTURE_DEDANS = 0.0     # share of the foot part's vertices allowed inside the foot
CHAUSSER_ECART = 0.1      # units of height per unit of shift: the smallest move wins a tie
TIGE_FONDU = 80 * MM      # above the ankle, a boot's shaft goes from sized with the foot to left as it is
                          # (over 4 cm, a shoe grown 15 % sheared its collar apart)


def _semelle(Z, m, poussees, seuil, compte=None):
    """Lowers the vertices m of a shoe as a whole until the foot no longer goes through its sole
    (the vertical drop that brings each vertex pushed downwards out, a few times since the nearest
    skin changes as it goes down). Only the vertices `compte` (default m: the foot part, a boot's
    shaft is fitted around the leg afterwards) are checked.
    Returns (Z[m] lowered, drop, share of `compte` more than `seuil` deep, margin included)."""
    compte = m if compte is None else compte
    Z2 = Z.copy()
    dy = 0.0
    for _ in range(SEMELLE_PASSES):
        d0, prof = poussees(Z2, faces=False)
        n = d0 / np.maximum(np.linalg.norm(d0, axis=1), 1e-12)[:, None]
        sous = (prof > 0) & (n[:, 1] < -0.5) & compte
        if not sous.any():
            break
        d = float((prof[sous] / -n[sous, 1]).max())
        Z2[m, 1] -= d
        dy += d
    _, prof = poussees(Z2, faces=False)
    return Z2[m], dy, float(((prof > seuil) & compte).sum() / max(compte.sum(), 1))


AVANCES = tuple(x * 5 * MM * k for x in range(7) for k in ((1,) if x == 0 else (-1, 1)))  # up to 3 cm


GAINE_PAS = 10 * MM        # height of the bands a boot's shaft is fitted by
GAINE_FONDU = 50 * MM      # a band's fit fades out over this height above and below it
GAINE_JAMBE = 90 * MM      # skin this close to the leg's axis is the leg's (not the other leg's)
GAINE_SECTEURS = 12        # directions around the leg a shaft's band is widened by
GAINE_ANGLE = np.pi / 3    # a direction's widening fades out over this angle on each side
GAINE_PASSES = 4
GAINE_PAROI = 0.5          # vertices nearer the leg's centre than this share of their band's median
                           # are inner parts (lining, sole, straps' backs), not the shaft's wall


def _lisser(t, tb, valeurs, poids_min=1e-6):
    """Per-band values (one row per band centre tb) spread over the heights t with a tent of
    half-width GAINE_FONDU, as a weighted mean (bands without a value carry weight 0)."""
    w = np.clip(1.0 - np.abs(t[:, None] - tb[None, :]) / GAINE_FONDU, 0.0, 1.0)
    return (w @ valeurs) / np.maximum(w.sum(1), poids_min)[:, None]


def gainer(Z, tige, axes, poussees, peau):
    """A boot's shaft fitted around the leg like a sleeve, band by band along the leg's axis:
    first each band is moved so that its centre is the leg's centre at that height (the source's
    legs are further apart or closer), then its wall is moved outwards from that centre by an
    offset that depends only on height and direction (GAINE_SECTEURS directions), the depth by
    which the leg goes through the wall there, spread smoothly over GAINE_FONDU and GAINE_ANGLE and
    faded in from the ankle by `tige` (0 on the foot part, 1 from TIGE_FONDU above the ankle).
    Pushing the shaft vertex by vertex tore it into spikes and flaps; one widening factor per band
    ballooned an oval shaft on its long side. axes: [(vertices of one side's shaft, ankle head,
    unit axis up to the knee)]; peau: the body's skin points.
    Returns (Z, {"recentre" (m), "passes", "elargi": the largest outward offset (m)})."""
    Z = Z.copy()
    centres, deplace, elargi, n = [], 0.0, 0.0, 0
    for m, h, u in axes:
        V = Z[m] - h
        t = V @ u
        R = V - t[:, None] * u
        nb = int(np.floor((t.max() - t.min()) / GAINE_PAS)) + 1
        k = np.minimum(np.floor((t - t.min()) / GAINE_PAS).astype(int), nb - 1)
        tb = t.min() + (np.arange(nb) + 0.5) * GAINE_PAS
        Q = peau - h
        tq = Q @ u
        Rq = Q - tq[:, None] * u
        proche = np.linalg.norm(Rq, axis=1) < GAINE_JAMBE
        cj, cs, a = np.zeros((nb, 3)), np.zeros((nb, 3)), np.zeros((nb, 1))
        for b in range(nb):
            j = proche & (np.abs(tq - tb[b]) <= GAINE_PAS / 2)
            v = k == b
            if j.sum() >= 8 and v.any():
                cj[b] = Rq[j].mean(0)
                cs[b] = (R[v].min(0) + R[v].max(0)) / 2      # middle of the band, straps and all
                a[b] = 1.0
        if a.any():
            dec = _lisser(t, tb, (cj - cs) * a) * tige[m][:, None]
            Z[m] = Z[m] + dec
            deplace = max(deplace, float(np.linalg.norm(dec, axis=1).max()))
        else:
            cj[:] = 0.0
        centre = np.zeros((nb, 3)) + (cj * a).sum(0) / max(a.sum(), 1.0)
        centre[a[:, 0] > 0] = cj[a[:, 0] > 0]
        centres.append((m, h, u, t, k, tb, centre))
    for n in range(1, GAINE_PASSES + 1):
        _, prof = poussees(Z)
        bouge = False
        for m, h, u, t, k, tb, centre in centres:
            c = h + t[:, None] * u + _lisser(t, tb, centre)
            R = Z[m] - c
            R = R - (R @ u)[:, None] * u
            r = np.maximum(np.linalg.norm(R, axis=1), 1e-9)
            e1 = np.cross(u, [0.0, 0.0, 1.0])
            e1 = e1 / np.linalg.norm(e1)
            a = np.arctan2(R @ np.cross(u, e1), R @ e1)
            sa = np.floor((a + np.pi) / (2 * np.pi) * GAINE_SECTEURS).astype(int) % GAINE_SECTEURS
            besoin = np.zeros((len(tb), GAINE_SECTEURS))
            for b in range(len(tb)):
                v = k == b
                if v.any():
                    paroi = v & (r >= GAINE_PAROI * np.median(r[v]))
                    np.maximum.at(besoin[b], sa[paroi], prof[m][paroi])
            if besoin.max() <= 0.2 * MM:
                continue
            ac = (np.arange(GAINE_SECTEURS) + 0.5) / GAINE_SECTEURS * 2 * np.pi - np.pi
            wt = np.clip(1.0 - np.abs(t[:, None] - tb[None, :]) / GAINE_FONDU, 0.0, 1.0)
            da = np.abs((a[:, None] - ac[None, :] + np.pi) % (2 * np.pi) - np.pi)
            wa = np.clip(1.0 - da / GAINE_ANGLE, 0.0, 1.0)
            d = (wt[:, :, None] * wa[:, None, :] * besoin[None]).max((1, 2)) * tige[m]
            if d.max() > 0.2 * MM:
                bouge = True
                elargi = max(elargi, float(d.max()))
                Z[m] = Z[m] + R / r[:, None] * d[:, None]
        if not bouge:
            break
    return Z, {"recentre": round(deplace * UNITE, 4), "passes": n, "elargi": round(elargi * UNITE, 4)}


def chausser(Z, m, centre, avant, cheville_y, poussees, seuil):
    """Size and place of a shoe (vertices m of Z) for a foot of another body: the shoe is the
    source's foot, smaller or larger than this one, its toes further forward or back from the
    ankle. Among the uniform scales about the foot's centre (POINTURES) and the shifts along the
    foot (AVANCES, `avant`: horizontal forward) with which the foot no longer goes through the shoe
    once it is lowered onto its sole (_semelle), the smallest size that holds the foot without
    lowering more than SEMELLE_JEU (a larger shoe than the foot needs looks like a clog around the
    ankle), then the shift whose bottom ends highest (the body is raised least); if none does,
    the one whose bottom ends highest: the foot rests on the sole instead of being held up by its toes against the
    front. Scaling and moving keep the shoe's shape, where pushing vertex by vertex tears it.
    Only the foot part (up to the ankle, `cheville_y`) is sized and checked; a boot's shaft fades
    from it to unscaled over TIGE_FONDU, moves with it, and is fitted around the leg afterwards.
    Returns (Z[m], {"pointure", "avance" (metres), "semelle" (metres)}, share of each vertex sized with the foot)."""
    w = np.clip(1.0 - (Z[:, 1] - cheville_y) / TIGE_FONDU, 0.0, 1.0)
    pied = m & (w >= 1.0)
    essais = []
    for s in POINTURES:
        if s > POINTURE_LIBRE + 1e-9 and not min(essais, key=lambda e: e[:5])[2]:
            break
        k = (1.0 + w[m] * (s - 1.0))[:, None]
        for dz in AVANCES:
            Zs = Z.copy()
            Zs[m] = centre + k * (Z[m] - centre) + dz * avant
            Zm, dy, dedans = _semelle(Zs, m, poussees, seuil, compte=pied)
            cout = -float(Zm[w[m] >= 1.0, 1].min()) + CHAUSSER_ECART * abs(dz)
            trop = dedans > POINTURE_DEDANS
            enfonce = dy > SEMELLE_JEU      # the foot only fits by pushing the sole down: too small
            if s > POINTURE_LIBRE + 1e-9 and (trop or enfonce):
                continue
            essais.append((trop, dedans if trop else 0.0, enfonce, cout if enfonce else s, cout, Zm,
                           {"pointure": round(s, 3), "avance": round(dz * UNITE, 4), "semelle": round(dy * UNITE, 4)}))
    *_, Zm, info = min(essais, key=lambda e: e[:5])
    return Zm, info, w


def deposer_sortie(out, C0, n0, nb_c, nr0, ch, pose):
    """Fitted on the posed body, written in the rest frame: the body's vertices and bones are those of the
    rest body C0, the part's vertices, bones, rigid bodies and joints are un-posed with their weights."""
    S = out["sommets"]
    S["pos"][:n0] = [tuple(p) for p in C0["sommets"]["pos"]]
    S["nor"][:n0] = [tuple(p) for p in C0["sommets"]["nor"]]
    for o, r in zip(out["os"][:nb_c], C0["os"]):
        for k in ("pos", "queue_dec", "axe_fixe", "axes_locaux"):
            o[k] = copy.deepcopy(r[k])
    T = PD.transfos(ch, pose)
    cho = PD.chaines(out["os"])       # rest skeleton plus the part's bones, hung under the feet
    for nom in cho:
        if nom in ch:
            cho[nom].update({k: ch[nom][k] for k in ("ha", "ht", "axe")})
    I = np.asarray(S["os"][n0:], np.int64)
    Wt = np.asarray(S["poids"][n0:], float)
    Wt = np.where(I >= 0, Wt, 0)
    Wt = Wt / np.maximum(Wt.sum(1), 1e-12)[:, None]
    wa, wt, cote = PD.poids(cho, I, Wt)
    Zp, Np = np.asarray(S["pos"][n0:], float), np.asarray(S["nor"][n0:], float)
    S["pos"][n0:] = [tuple(float(v) for v in x) for x in PD.deposer(Zp, wa, wt, cote, T)]
    S["nor"][n0:] = [tuple(float(v) for v in x) for x in PD.deposer(Np, wa, wt, cote, T, vecteurs=True)]
    tos = PD.transfo_os(cho, T, len(out["os"]))
    for i in range(nb_c, len(out["os"])):
        R, t = tos[i]
        o = out["os"][i]
        o["pos"] = tuple(float(x) for x in R.T @ (np.asarray(o["pos"], float) - t))
        for k in ("queue_dec", "axe_fixe"):
            if o.get(k) is not None and (k != "queue_dec" or o.get("queue_os") is None):
                o[k] = tuple(float(x) for x in R.T @ np.asarray(o[k], float))
        if o.get("axes_locaux") is not None:
            o["axes_locaux"] = tuple(tuple(float(x) for x in R.T @ np.asarray(a, float)) for a in o["axes_locaux"])
    for r in out["rigides"][nr0:]:
        R, t = tos[r["os"]] if 0 <= r["os"] < len(tos) else (np.eye(3), np.zeros(3))
        r["pos"] = tuple(float(x) for x in R.T @ (np.asarray(r["pos"], float) - t))
        r["rot"] = euler_mmd(R.T @ rot_mmd(r["rot"]))
    for j in out["joints"]:
        if j["a"] < nr0:
            continue
        b = out["rigides"][j["a"]]["os"]
        R, t = tos[b] if 0 <= b < len(tos) else (np.eye(3), np.zeros(3))
        j["pos"] = tuple(float(x) for x in R.T @ (np.asarray(j["pos"], float) - t))
        j["rot"] = euler_mmd(R.T @ rot_mmd(j["rot"]))


def porter(piece_dir, corps, sortie, nom=None, marge_mm=3.0, peau="retirer", surface="tout", pied=None):
    """pied (shoes with a sole only): "chercher" finds the pose of the body's feet inside the shoe,
    a pose dict (pied.py) reuses one; the part is fitted on the posed body and written back in the
    rest frame (res["pied"]: the pose and the height to raise the model by). None: flat foot.
    "plat": flat foot, but the shoe's foot part stays as the bones carry it (a heel sinks into the
    ground and the foot goes through the sole, the shoe stays whole); only a boot's shaft is
    pushed out of the leg."""
    fiche, chemin_piece = charger_piece(piece_dir)
    nom = nom or "%s-%s" % (fiche["perso"], fiche["id"])
    P = PMXL.lire(chemin_piece)
    C = PMXL.lire(corps)
    src_complet = PMXL.lire(fiche["source"]["pmx"])
    # part bones -> bones of the full source (reliable tails)
    nom_src = {o["nom"]: i for i, o in enumerate(src_complet["os"])}
    s_glob = min(1.15, max(0.87, hauteur(C) / hauteur(src_complet)))
    po = [nom_src.get(o["nom"], -1) for o in P["os"]]          # part bone -> source bone
    propres = set(fiche.get("os_propres") or [])
    X = np.asarray(P["sommets"]["pos"], float)
    N = np.asarray(P["sommets"]["nor"], float)
    I, W = poids_np(P)
    Is = np.where(I >= 0, np.asarray(po)[np.maximum(I, 0)], -1)
    # ---------------------------------------------------------------- 0. foot posed in the shoe
    C0, pose_pied, ch_pied = None, None, None
    plat = (pied == "plat" and fiche.get("type") in TYPES_PIED and a_semelle(P)
            and bool(PD.chaines(C["os"])) and bool(PD.chaines(src_complet["os"])))
    if plat:
        ch_pied = PD.chaines(C["os"])
    elif (pied not in (None, "plat") and fiche.get("type") in TYPES_PIED and a_semelle(P)
            and PD.chaines(C["os"]) and PD.chaines(src_complet["os"])):
        if isinstance(pied, dict):
            pose_pied = {k: {"cheville": v["cheville"], "orteils": v["orteils"]}
                         for k, v in pied.items() if isinstance(v, dict)}
            ch_pied = PD.chaines(C["os"])
        else:
            sys.path.insert(0, os.path.join(RACINE, "atelier"))
            import rayons
            Y0 = Recalage(src_complet, C, s_glob, pieds_rigides=True, chaussure=True).points(Is, W, X)
            pose_pied, ch_pied = PD.chercher(C, Y0, faces_mats(P)[0], UNITE, rayons)
        C0, C = C, copy.deepcopy(C)
        if all(max(abs(v["cheville"]), abs(v["orteils"])) < PIED_PLAT for v in pose_pied.values()):
            # a flat shoe: the foot already sits in it and stays at rest, but the shoe is still placed
            # in one block (pushing it out of the foot vertex by vertex tears it to shards)
            pose_pied = {k: {"cheville": 0.0, "orteils": 0.0} for k in pose_pied}
        else:
            PD.poser_pmx(C, pose_pied, ch=ch_pied)
    rc = Recalage(src_complet, C, s_glob, pieds_rigides=pose_pied is not None,
                  chaussure=fiche.get("type") in TYPES_PIED and a_semelle(P))
    # ---------------------------------------------------------------- 1-2. LBS
    Y = rc.points(Is, W, X)
    Ny = rc.vecteurs(Is, W, N)
    # ---------------------------------------------------------------- surface underneath
    Fc, mfc = faces_mats(C)
    Pc = np.asarray(C["sommets"]["pos"], float)
    noms_c = [m["nom"] for m in C["materiaux"]]
    # surface="peau": the part is placed on the skin alone, as if it were worn alone;
    # the outfit's layers are then adjusted together (outils/atelier/combinaison.py)
    garde_surface = (lambda n: bool(MD.PEAU.match(n))) if surface == "peau" else (lambda n: not MD.NON_VETEMENT.match(n))
    dessous = np.array([garde_surface(n) and C["materiaux"][i]["n_indices"] > 0
                        for i, n in enumerate(noms_c)])[mfc]
    fdes = np.where(dessous)[0]
    nfc = normales_faces(Pc, Fc)
    sg = signe_dehors(Pc, Fc[fdes], nfc[fdes])
    nfc = nfc * sg
    arbre = bvh(Pc, Fc[fdes])
    marge = marge_mm * MM
    Fp, mfp = faces_mats(P)

    fond_s, fond_f, fond_liens = couvercles(X, Fp, src_complet["os"])

    def poussees(Z, faces=True):
        d0 = np.zeros_like(Z)
        prof = np.zeros(len(Z))
        for i, z in enumerate(Z):
            if fond_s[i]:               # centre of a cap: hidden in the body, as on its source
                continue
            loc, _, k, dist = arbre.find_nearest(tuple(z), 4 * marge + 0.4)
            if loc is None:
                continue
            n = nfc[fdes[k]]
            s = float((z - np.asarray(loc)) @ n)
            if s < marge and dist < 0.4:
                d0[i] = (marge - s) * n
                prof[i] = marge - s
        # inside the faces: centre and edge midpoints (a wide fold can pass under the skin
        # between vertices that are outside); a point inside pushes the three vertices of its face
        T = Z[Fp]
        ech = [T.mean(1), (T[:, 0] + T[:, 1]) / 2, (T[:, 1] + T[:, 2]) / 2, (T[:, 2] + T[:, 0]) / 2]
        for E in (ech if faces else []):
            for k, z in enumerate(E):
                if fond_f[k]:
                    continue
                loc, _, kk, dist = arbre.find_nearest(tuple(z), 4 * marge + 0.4)
                if loc is None or dist >= 0.4:
                    continue
                n = nfc[fdes[kk]]
                sv = float((z - np.asarray(loc)) @ n)
                # between vertices that are outside, a face only sinks by its sag (a few mm):
                # any deeper, it is a wrongly oriented body normal, so it is ignored
                if marge - 15 * MM < sv < marge:
                    for i in Fp[k]:
                        if marge - sv > prof[i]:
                            d0[i] = (marge - sv) * n
                            prof[i] = marge - sv
        return d0, prof

    from mathutils.kdtree import KDTree
    stats = {}
    accroche = None
    if fiche["classe"] == "accessoire":
        accroche = accrocher_anneau(fiche, P, src_complet, Y, s_glob, C, Fc, mfc, fdes, arbre, marge, sur_tissu=surface != "peau")
    def finir(accroche, Ny):
        stats = {}
        Ny = Ny.copy()
        if accroche == "os":        # the part follows the target's bones, as it follows those of its source
            Z = Y.copy()
        elif accroche is not None:
            Z, rigide, stats["anneau"] = accroche
            Ny = N @ rigide[2].T
        elif pose_pied is not None:
            # a shoe is rigid: it stays as placed around the posed foot, lowered as a whole until
            # the foot no longer goes through its sole. Pushing the sole vertex by vertex drags the
            # whole sole down pass after pass (a solid platform), and tears it
            Z = Y.copy()
            wa_c, wt_c, cote_c = PD.poids_pmx(C, ch_pied)
            stats["chausse"] = {}
            tige = np.zeros(len(Z))
            for k_c, (nom_c, _, sg_c) in enumerate(PD.COTES):
                # the side is the source's: a foot set close to the other one has its inner edge
                # at x = 0 once worn, and a vertex across it would be moved with the other shoe
                m = sg_c * X[:, 0] > 0
                pied_c = Pc[(cote_c == k_c) & (wa_c + wt_c > 0.5)]
                if not m.any() or not len(pied_c) or nom_c not in ch_pied:
                    continue
                av = ch_pied[nom_c]["ht"] - ch_pied[nom_c]["ha"]
                av[1] = 0.0
                Z[m], stats["chausse"][nom_c], w_c = chausser(Z, m, pied_c.mean(0), av / np.linalg.norm(av),
                                                              ch_pied[nom_c]["ha"][1], poussees, marge + 1 * MM)
                tige[m] = 1.0 - w_c[m]
        else:
            import transport as TR
            moulant = fiche.get("type") in MOULANTS or all(re.match(r"体", m["nom"]) for m in P["materiaux"])
            Z, stats["transport"] = TR.transporter(X, Is, W, src_complet, C, propres, arbre, Y, s_glob, Fp, moulant, sg, faces_cible=Fc[fdes])
            if plat:
                # the foot part stays as carried; a boot's shaft is pushed, fading in above the ankle
                tige = np.zeros(len(Z))
                for nom_c, _, sg_c in PD.COTES:
                    m = sg_c * X[:, 0] > 0
                    if nom_c in ch_pied:
                        tige[m] = np.clip((Y[m, 1] - ch_pied[nom_c]["ha"][1]) / TIGE_FONDU, 0.0, 1.0)
        # until nothing is left inside (at most PASSES_MAX); what remains is counted, not hidden
        stats["reste_dedans"] = 0
        Z_avant = Z.copy()
        # a shoe's foot part was placed as a whole (chausser, or carried flat) and stays so; a boot's
        # shaft is widened around the leg like a sleeve (gainer), never pushed vertex by vertex
        chausse = (pose_pied is not None or plat) and accroche is None
        if chausse:
            axes = []
            idx_c = {o["nom"]: i for i, o in enumerate(C["os"])}
            for nom_c, pre_c, sg_c in PD.COTES:
                m = (sg_c * X[:, 0] > 0) & (tige > 0)
                if nom_c not in ch_pied or not m.any():
                    continue
                h = ch_pied[nom_c]["ha"]
                g = idx_c.get(pre_c + "ひざ")
                u = (np.asarray(C["os"][g]["pos"], float) - h) if g is not None else np.array([0.0, 1.0, 0.0])
                axes.append((m, h, u / np.linalg.norm(u)))
            Z, stats["gaine"] = gainer(Z, tige, axes, poussees, Pc[np.unique(Fc[fdes])])
            stats["reste_dedans"] = int(((poussees(Z)[1] > 1 * MM) & (tige > 0)).sum())
        for passe in range(0 if chausse else PASSES_MAX):
            d0, prof = poussees(Z)
            dedans = np.where(prof > 0)[0]
            stats["reste_dedans"] = int((prof > 1 * MM).sum())
            stats["passe_%d" % passe] = {"sommets": int(len(dedans)),
                                         "prof_max_mm": round(float(prof.max() / MM), 1) if len(dedans) else 0}
            if not len(dedans):
                break
            kd = KDTree(len(Z))
            for i, z in enumerate(Z):
                kd.insert(tuple(z), i)
            kd.balance()
            rr = 3 * 10 * MM if passe == 0 else 1.0 * 10 * MM
            dlt = np.zeros_like(Z)
            mag = np.zeros(len(Z))
            nyu = Ny / np.maximum(np.linalg.norm(Ny, axis=1), 1e-12)[:, None]
            for i in dedans:
                for _, j, d in kd.find_range(tuple(Z[i]), rr):
                    # a vertex's push only drags neighbours facing the same way: between the
                    # thighs or the fingers, the facing fabric is close and moves out the opposite way
                    if j != i and float(d0[i] @ nyu[j]) <= 0:
                        continue
                    f = (1 - d / rr) ** 2
                    m = prof[i] * f
                    if m > mag[j]:
                        mag[j] = m
                        dlt[j] = d0[i] * f
            dlt[dedans] = np.where((np.linalg.norm(d0[dedans], axis=1) > mag[dedans])[:, None], d0[dedans], dlt[dedans])
            Z = Z + dlt
        for centre, tour in fond_liens:         # the bottom follows the rim it closes, without leaving the body
            Z[centre] = Z_avant[centre] + (Z[tour] - Z_avant[tour]).mean(0)
        if accroche is None:
            Z, stats["couches_internes"] = couches_internes(X, N, Ny, Z, Fp)
            # what is rigid on the source stays rigid: the transport places each vertex according to
            # the body and twists a backpack or a buckle like fabric. Each piece held by a single bone
            # takes back its shape, at the position and orientation the transport gave it, then
            # moves out of the skin as one block; if it ends up deeper than before, the transport is kept.
            def enfonce(Zc):
                pire, dv = 0.0, np.zeros(3)
                for z in Zc:
                    loc, _, k, dist = arbre.find_nearest(tuple(z), 4 * marge + 0.4)
                    if loc is None or dist >= 0.4:
                        continue
                    n = nfc[fdes[k]]
                    pr = marge - float((z - np.asarray(loc)) @ n)
                    if pr > pire:
                        pire, dv = pr, pr * n
                return pire, dv

            poses = 0
            # a shoe was not transported: its foot part is already one block, its shaft kept its shape
            for v in ([] if chausse else morceaux_rigides(X, Fp, I, W)):
                Pz, accord, Rz = poser_rigide(X[v], Z[v])
                if accord < RIGIDE_ACCORD:
                    continue
                avant = enfonce(Z[v])[0]
                for _ in range(6):
                    pire, dv = enfonce(Pz)
                    if pire <= 0:
                        break
                    Pz = Pz + dv
                if enfonce(Pz)[0] > avant + 1 * MM:
                    continue
                Z[v] = Pz
                Ny[v] = N[v] @ Rz
                poses += len(v)
            stats["rigides_sommets"] = int(poses)
        return Z, Ny, stats

    # what does not clearly surround something: placed as one block (tail, wings, far from the body),
    # transported like the rest (harness, tie, bag against the body) or adjusted as a ring; none
    # works everywhere, so the one whose edges stretch least once the part is out of the skin
    # is kept
    essais = [("anneau", accroche)]
    if accroche is not None:
        bloc = accrocher_anneau(fiche, P, src_complet, Y, s_glob, C, Fc, mfc, fdes, arbre, marge, sur_tissu=surface != "peau", regler=False)
        essais = [("bloc", bloc), ("transport", None), ("anneau", accroche)]
        if accroche[2]["anneau"]:           # on a tie, what clearly surrounds something stays a ring
            essais = [essais[2], essais[0], essais[1]]
    faits = []
    for nom_e, a in essais:
        Z, Nya, st = finir(a, Ny)
        E_ = np.concatenate([Fp[:, [0, 1]], Fp[:, [1, 2]], Fp[:, [2, 0]]])
        l0 = s_glob * np.linalg.norm(X[E_[:, 0]] - X[E_[:, 1]], axis=1)
        l1 = np.linalg.norm(Z[E_[:, 0]] - Z[E_[:, 1]], axis=1)
        st["allongement"] = round(float(np.maximum(l1 - ALLONGE * l0, 0).sum() / max(l0.sum(), 1e-9)), 4)
        faits.append((st["allongement"], len(faits), Z, Nya, st, a, nom_e))
    _, _, Z, Ny, st, accroche, nom_e = min(faits, key=lambda f: f[:2])
    if len(faits) > 1:
        st["essais"] = {f[6]: f[0] for f in faits}
        # a part attached to several limbs (a tail shipped with its arm ornaments) is not fitted
        # in a single way: the tail is placed as one block, what is attached to the arms follows them.
        # Each limb takes, among the chosen placement, the other family (block or ring / transport)
        # and the per-bone placement (what is attached to an arm follows it as on its source), the one
        # that stretches its edges least.
        mb = membres(X, Fp, Is, W, src_complet)
        if len(np.unique(mb[mb >= 0])) > 1:
            blocs = [f for f in faits if f[5] is not None]
            autres = [min(blocs, key=lambda f: f[:2]) if accroche is None else next(f for f in faits if f[5] is None)]
            Zo, Nyo, _ = finir("os", Ny)
            autres.append((0, 0, Zo, Nyo, None, None, "os"))
            Z, Ny = Z.copy(), Ny.copy()
            em = mb[E_[:, 0]]
            par = {}
            for m in np.unique(mb[mb >= 0]):
                k, v = em == m, mb == m

                def allonge(Zc):
                    l1 = np.linalg.norm(Zc[E_[k, 0]] - Zc[E_[k, 1]], axis=1)
                    return float(np.maximum(l1 - ALLONGE * l0[k], 0).sum() / max(l0[k].sum(), 1e-9))
                ici = allonge(Z)
                par[str(int(m))] = {nom_e: round(ici, 4)}
                for f in autres:
                    la = allonge(f[2])
                    par[str(int(m))][f[6]] = round(la, 4)
                    if la < ici - MEMBRE_GAIN or (f[6] == "os" and la < ici):     # roughly equal: what is attached to a limb follows its bones
                        Z[v], Ny[v], ici = f[2][v], f[3][v], la
                        par[str(int(m))]["pris"] = f[6]
            if any("pris" in x for x in par.values()):
                st["par_membre"] = par
                if accroche is None and any(x.get("pris") == autres[0][6] for x in par.values()):
                    accroche = autres[0][5]        # a limb placed as one block: its own bones follow that block
    stats.update(st)
    stats["etendue_mm"] = round(float(np.linalg.norm(Z.max(0) - Z.min(0)) / MM), 1)
    # what the part has underneath on its source (body or underlying garment, excluding face, hair
    # and the part itself), with its weights mapped to the target's bones
    Ps_ = np.asarray(src_complet["sommets"]["pos"], float)
    Fs_, mfs_ = faces_mats(src_complet)
    mats_p = {m["nom"] for m in P["materiaux"]}
    gs_ = np.array([not MD.NON_VETEMENT.match(m["nom"]) and m["nom"] not in mats_p
                    for m in src_complet["materiaux"]])[mfs_]
    Fs_ = Fs_[gs_]
    arbre_src = bvh(Ps_, Fs_) if len(Fs_) else None
    stats["sous_la_peau"] = sous_la_peau(Z[~fond_s], arbre, MM, sg)
    stats["couvercles"] = int(fond_f.sum())
    stats["pose_rigide"] = accroche is not None
    # a shoe's foot part kept in one block (chausser, or carried flat) holds the foot inside it by nature
    stats["pied_en_bloc"] = (pose_pied is not None or plat) and accroche is None
    stats["deplacement_mm"] = {"p50": round(float(np.median(np.linalg.norm(Z - Y, axis=1)) / MM), 1),
                               "p90": round(float(np.percentile(np.linalg.norm(Z - Y, axis=1), 90) / MM), 1),
                               "max": round(float(np.linalg.norm(Z - Y, axis=1).max() / MM), 1)}
    # ---------------------------------------------------------------- 3. output bones
    out = copy.deepcopy(C)
    nb_c = len(out["os"])
    nom_cible = {o["nom"]: i for i, o in enumerate(out["os"])}
    carte = {}                    # part bone -> output bone
    ajoutes = []
    for i, o in enumerate(P["os"]):
        if o["nom"] in nom_cible and o["nom"] not in propres:
            carte[i] = nom_cible[o["nom"]]
    noms_pris = set(nom_cible)
    for i, o in enumerate(P["os"]):
        if i in carte:
            continue
        carte[i] = nb_c + len(ajoutes)
        ajoutes.append(i)
    for i in ajoutes:
        o = copy.deepcopy(P["os"][i])
        b = po[i]
        if accroche is not None:
            Rm, tr, _ = accroche[1]
            o["pos"] = tuple(float(v) for v in Rm @ np.asarray(o["pos"], float) + tr)
            M3 = Rm
        elif b >= 0:
            o["pos"] = tuple(float(v) for v in rc.point(b, o["pos"]))
            M3 = rc.R[b] @ rc.A[b]
        else:
            M3 = np.eye(3)
        nm = o["nom"]
        while nm in noms_pris:
            nm += "'"
        noms_pris.add(nm)
        o["nom"] = nm
        o["parent"] = carte.get(o["parent"], -1) if o["parent"] >= 0 else -1
        if o["queue_os"] is not None:
            o["queue_os"] = carte.get(o["queue_os"], -1) if o["queue_os"] >= 0 else -1
        elif o["queue_dec"] is not None:
            o["queue_dec"] = tuple(float(v) for v in M3 @ np.asarray(o["queue_dec"], float))
        if o["ajout"]:
            o["ajout"]["source"] = carte.get(o["ajout"]["source"], -1)
            if o["ajout"]["source"] < 0:
                o["ajout"] = None
                o["drapeaux"] &= ~0x0300
        if o["ik"]:
            o["ik"]["cible"] = carte.get(o["ik"]["cible"], -1)
            for l in o["ik"]["liens"]:
                l["os"] = carte.get(l["os"], -1)
        if o.get("axe_fixe") is not None and b >= 0:
            v = rc.R[b] @ np.asarray(o["axe_fixe"], float)
            o["axe_fixe"] = tuple(float(x) for x in v / max(np.linalg.norm(v), 1e-12))
        if o.get("axes_locaux") is not None and b >= 0:
            o["axes_locaux"] = tuple(tuple(float(x) for x in rc.R[b] @ np.asarray(a, float)) for a in o["axes_locaux"])
        out["os"].append(o)
    # ---------------------------------------------------------------- 4. weights
    Io = np.where(I >= 0, np.vectorize(lambda b: carte.get(int(b), -1))(np.maximum(I, 0)), -1)
    Io[I < 0] = -1
    Wo = W.copy()
    corps_out = np.zeros(len(out["os"]), bool)
    corps_out[:nb_c] = True
    for i in ajoutes:      # source body bones missing from the target: counted as body
        if P["os"][i]["nom"] not in propres:
            corps_out[carte[i]] = True
    Ic, Wc = poids_np(C)
    Isrc, Wsrc = poids_np(src_complet)
    vers_c = np.array([nom_cible.get(o["nom"], -1) for o in src_complet["os"]])

    def melange(arb, z, T, Pt, It, Wt, portee, vers=None, dessous=False):
        if arb is None:
            return None
        loc, nrm, k, dist = arb.find_nearest(tuple(z), portee)
        if loc is None:
            return None
        if dessous and dist > DESSOUS_PRES:
            # "underneath" = along the surface's axis. A nearest point reached at a slant is the
            # edge of a hole (skin removed under the garment) or the neighbouring skin: nothing underneath.
            if abs(float(np.dot(np.asarray(z) - np.asarray(loc), nrm))) < DESSOUS_COS * dist:
                return None
        f = T[k]
        cw = {}
        for a_, t in zip(f, barycentre(np.asarray(loc), Pt[f])):
            for b_, w in zip(It[a_], Wt[a_]):
                if b_ >= 0 and w > 0:
                    b_ = int(b_) if vers is None else int(vers[b_])
                    if b_ >= 0:
                        cw[b_] = cw.get(b_, 0) + t * w
        tot = sum(cw.values())
        return {b_: w / tot for b_, w in cw.items()} if tot > 1e-6 else None

    repris = corriges = 0
    for v in range(len(Z)):
        sw = melange(arbre_src, X[v], Fs_, Ps_, Isrc, Wsrc, 6 * 10 * MM, vers_c, dessous=True)
        # with nothing known underneath on the source, the target skin only counts up close; with it,
        # the target − source difference stays meaningful far from the body (a flared panel that followed the chest)
        cw = melange(arbre, Z[v], Fc[fdes], Pc, Ic, Wc, (PORTEE_LOIN if sw is not None else 3 * 10) * MM)
        if cw is None:
            continue
        iv, wv = Io[v], Wo[v]
        own = {int(b): float(w) for b, w in zip(iv, wv) if b >= 0 and w > 0 and not corps_out[b]}
        part_corps = 1 - sum(own.values())
        if part_corps < 0.05:
            continue
        # the garment does not necessarily follow the nearest skin (a panel hanging from the chest
        # next to the arm): its original weights are kept, corrected by the difference between the
        # target skin and what it had underneath on its source. Nothing underneath on the source (skin
        # removed under a closed garment): the garment stood in for the body there, it takes the target skin.
        if sw is not None:
            g = {int(b): float(w) / part_corps for b, w in zip(iv, wv) if b >= 0 and w > 0 and corps_out[b]}
            cw = {b: max(0.0, g.get(b, 0) + cw.get(b, 0) - sw.get(b, 0)) for b in set(g) | set(cw) | set(sw)}
            corriges += 1
        lst = sorted(own.items(), key=lambda x: -x[1])[:4]
        reste = 4 - len(lst)
        cs = [x for x in sorted(cw.items(), key=lambda x: -x[1])[:reste] if x[1] > 0]
        tot = sum(w for _, w in cs)
        if not cs or tot <= 0:
            continue
        lst += [(b, part_corps * w / tot) for b, w in cs]
        lst += [(-1, 0.0)] * (4 - len(lst))
        Io[v] = [b for b, _ in lst]
        Wo[v] = [w for _, w in lst]
        repris += 1
    stats["poids_repris_du_corps"] = repris
    stats["poids_corriges_par_la_source"] = corriges
    # ---------------------------------------------------------------- vertices / faces / materials
    S = out["sommets"]
    n0 = len(S["pos"])
    k_add = out["n_uv_add"]
    for v in range(len(Z)):
        S["pos"].append(tuple(float(x) for x in Z[v]))
        S["nor"].append(tuple(float(x) for x in Ny[v]))
        S["uv"].append(P["sommets"]["uv"][v])
        ua = list(P["sommets"]["uv_add"][v]) if P["n_uv_add"] else []
        S["uv_add"].append(tuple((ua + [(0.0, 0.0, 0.0, 0.0)] * k_add)[:k_add]))
        nz = int((Wo[v] > 0).sum())
        if nz <= 1:
            b = int(Io[v][np.argmax(Wo[v])])
            S["type"].append(0)
            S["os"].append((b, -1, -1, -1))
            S["poids"].append((1.0, 0.0, 0.0, 0.0))
        else:
            S["type"].append(2)
            S["os"].append(tuple(int(b) if b >= 0 else 0 for b in Io[v]))
            S["poids"].append(tuple(float(w) for w in Wo[v]))
        S["sdef"].append(None)
        S["contour"].append(P["sommets"]["contour"][v])
    # textures: everything in sortie/tex (hard links), relative paths
    os.makedirs(os.path.join(sortie, "tex"), exist_ok=True)
    tex_out, tmap = [], {}

    def lier_tex(chemin_rel, base_dir):
        src = os.path.normpath(os.path.join(base_dir, chemin_rel.replace("\\", "/")))
        if src in tmap:
            return tmap[src]
        b = os.path.basename(src)
        noms = {os.path.basename(t.replace("\\", "/")) for t in tex_out}
        racine, ext = os.path.splitext(b)
        k = 1
        while b in noms:
            b = "%s_%d%s" % (racine, k, ext)
            k += 1
        if os.path.exists(src):
            dst = os.path.join(sortie, "tex", b)
            if not os.path.exists(dst):
                try:
                    os.link(src, dst + ".tmp")
                except OSError:
                    import shutil
                    shutil.copyfile(src, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
        tmap[src] = len(tex_out)
        tex_out.append("tex\\" + b)
        return tmap[src]
    dir_c, dir_p = os.path.dirname(os.path.abspath(corps)), os.path.dirname(os.path.abspath(chemin_piece))

    def retex(m, textures, base):
        for k in ("texture", "sphere"):
            if 0 <= m[k] < len(textures):
                m[k] = lier_tex(textures[m[k]], base)
            else:
                m[k] = -1
        if m.get("toon_partage", 1) == 0:
            m["toon"] = lier_tex(textures[m["toon"]], base) if 0 <= m["toon"] < len(textures) else -1
    for m in out["materiaux"]:
        retex(m, C["textures"], dir_c)
    noms_mats = {m["nom"] for m in out["materiaux"]}
    mats_piece = []
    for m in P["materiaux"]:
        m = copy.deepcopy(m)
        retex(m, P["textures"], dir_p)
        nm = m["nom"]
        while nm in noms_mats:
            nm += "'"
        noms_mats.add(nm)
        m["nom"] = nm
        mats_piece.append(m)
    out["textures"] = tex_out
    # ---------------------------------------------------------------- 5. hidden skin
    Fcorps = [tuple(int(x) for x in f) for f in Fc]
    garde_c = np.ones(len(Fc), bool)
    if fiche["classe"] == "vetement" and peau in ("retirer", "noter"):
        # areas held by the body (not the physics panels)
        own_part = np.array([sum(w for b, w in zip(Io[v], Wo[v]) if b >= 0 and not corps_out[b]) for v in range(len(Z))])
        fp_tenue = np.where(own_part[Fp].max(1) < 0.5)[0]
        if len(fp_tenue):
            arbre_p = bvh(Z, Fp[fp_tenue])
            roles_peau = np.array([bool(MD.PEAU.match(n)) for n in noms_c])[mfc]
            for k in np.where(roles_peau)[0]:
                f = Fc[k]
                n = nfc[k]
                pts = list(Pc[f]) + [Pc[f].mean(0)]
                if all(arbre_p.ray_cast(tuple(p + n * 1e-4), tuple(n), 50 * MM)[0] is not None for p in pts):
                    garde_c[k] = False
    stats["faces_peau_retirees"] = int((~garde_c).sum())
    # peau="noter": the skin stays whole (the next part is placed on it), the hidden faces are returned
    peau_cachee = [int(k) for k in np.where(~garde_c)[0]]
    if peau == "noter":
        garde_c[:] = True
    # body faces kept, per material
    Fout = []
    for i, m in enumerate(out["materiaux"]):
        sel = (mfc == i) & garde_c
        m["n_indices"] = int(3 * sel.sum())
        Fout.extend(Fcorps[k] for k in np.where(sel)[0])
    for i, m in enumerate(mats_piece):
        sel = mfp == i
        Fout.extend(tuple(int(x) + n0 for x in f) for f in Fp[sel])
        out["materiaux"].append(m)
    out["faces"] = Fout
    # ---------------------------------------------------------------- 6. rigid bodies and joints
    nr0 = len(out["rigides"])
    nom_rig = {r["nom"] for r in out["rigides"]}
    for r in P["rigides"]:
        r = copy.deepcopy(r)
        b = po[r["os"]] if 0 <= r["os"] < len(po) else -1
        if accroche is not None and r["os"] in ajoutes:
            Rm, tr, Qa = accroche[1]
            r["pos"] = tuple(float(x) for x in Rm @ np.asarray(r["pos"], float) + tr)
            r["rot"] = euler_mmd(Qa @ rot_mmd(r["rot"]))
        elif b >= 0:
            r["pos"] = tuple(float(x) for x in rc.point(b, r["pos"]))
            r["rot"] = euler_mmd(rc.R[b] @ rot_mmd(r["rot"]))
        r["os"] = carte.get(r["os"], -1)
        while r["nom"] in nom_rig:
            r["nom"] += "'"
        nom_rig.add(r["nom"])
        out["rigides"].append(r)
    for j in P["joints"]:
        j = copy.deepcopy(j)
        ra = P["rigides"][j["a"]]
        b = po[ra["os"]] if 0 <= ra["os"] < len(po) else -1
        if accroche is not None and ra["os"] in ajoutes:
            Rm, tr, Qa = accroche[1]
            j["pos"] = tuple(float(x) for x in Rm @ np.asarray(j["pos"], float) + tr)
            j["rot"] = euler_mmd(Qa @ rot_mmd(j["rot"]))
        elif b >= 0:
            j["pos"] = tuple(float(x) for x in rc.point(b, j["pos"]))
            j["rot"] = euler_mmd(rc.R[b] @ rot_mmd(j["rot"]))
        j["a"], j["b"] = j["a"] + nr0, j["b"] + nr0
        out["joints"].append(j)
    # display frame for the added bones
    if ajoutes:
        out["cadres"].append({"nom": fiche["id"][:15], "nom_en": fiche["id"][:15], "special": 0,
                              "elements": [(0, carte[i]) for i in ajoutes]})
    if pose_pied is not None:
        # the shoe's bottom brought to where the bare body stands (its lowest point, not y = 0:
        # a body may stand a little above its origin)
        sol = float(np.asarray(C0["sommets"]["pos"], float)[:, 1].min())
        lever = (sol - float(Z[:, 1].min())) * UNITE
        stats["pied"] = {**pose_pied, "lever": round(lever, 4)}
        deposer_sortie(out, C0, n0, nb_c, nr0, ch_pied, pose_pied)
    out["nom"] = nom
    out["commentaire"] = "%s + %s from %s (porter.py)" % (C.get("nom", ""), fiche["id"], fiche["perso"])
    import pmx_ecrire
    os.makedirs(sortie, exist_ok=True)
    chemin = os.path.join(sortie, nom + ".pmx")
    pmx_ecrire.ecrire(out, chemin + ".tmp.pmx")
    os.replace(chemin + ".tmp.pmx", chemin)
    res = {"piece": fiche["perso"] + "/" + fiche["id"], "classe": fiche["classe"],
           "type": fiche.get("type"), "sorte": fiche.get("sorte"),
           "type_mesure": TYPES_MESURE.get(fiche.get("type"), "haut_ample"),
           "corps": os.path.abspath(corps), "pmx": chemin, "mats": [m["nom"] for m in mats_piece],
           "echelle_globale": round(s_glob, 3), "os_recales": int(rc.direct.sum()),
           "os_ajoutes": len(ajoutes), "angles_deg_max": max(rc.angles.values()) if rc.angles else 0,
           "rigides": len(P["rigides"]), "joints": len(P["joints"]), "marge_mm": marge_mm, **stats}
    with open(os.path.join(sortie, "portage.json.tmp"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    os.replace(os.path.join(sortie, "portage.json.tmp"), os.path.join(sortie, "portage.json"))
    journal("fitted", nom, json.dumps(stats, ensure_ascii=False))
    res["peau_cachee"] = peau_cachee
    return res


def barycentre(p, T):
    a, b, c = T
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-20:
        return np.array([1 / 3, 1 / 3, 1 / 3])
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bc = np.clip(np.array([1 - v - w, v, w]), 0, 1)
    return bc / bc.sum()


# ====================================================================== render

def rendre(res, chemin_png):
    import bpy
    import importlib.util
    import tempfile
    import planche as PL
    # the garment library, loaded by file: outils/tete has one with the same name, earlier in sys.path
    # (the view names below, "face", "dos", "profil", "près ...", also build file names: kept as is)
    sp = importlib.util.spec_from_file_location("bibliotheque_vetements", os.path.join(ICI, "bibliotheque.py"))
    BI = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(BI)
    tmp = tempfile.mkdtemp(prefix="rendus-porter-")
    M = MD.Modele(os.path.basename(res["pmx"]), chemin=res["pmx"])
    PL.vider()
    BI._bpy_scene()
    piece = np.isin(M.mf, [i for i, n in enumerate(M.noms_mat) if n in res["mats"]])
    tout = np.where(~M.faces_role(["vide"]))[0]
    PL.objet(M, tout, "tout", {})
    lo, hi = M.pos.min(0), M.pos.max(0)
    vs = np.unique(M.F[piece])
    plo, phi = M.pos[vs].min(0) - 0.05, M.pos[vs].max(0) + 0.05
    ligne = []
    for ang, nm, (a, b) in ((0, "face", (lo, hi)), (180, "dos", (lo, hi)), (90, "profil", (lo, hi)),
                            (0, "près face", (plo, phi)), (180, "près dos", (plo, phi)), (35, "près 3/4", (plo, phi))):
        f = os.path.join(tmp, "%s_%d_%s.png" % (os.path.basename(res["pmx"]), ang, nm.replace(" ", "_").replace("/", "")))
        BI._rendre(a, b, ang, f, 300, 480, texte=nm if ang or nm != "face" else res["piece"])
        ligne.append(f)
    BI._coller([ligne], chemin_png)


def main(argv):
    val = lambda o, d=None: argv[argv.index(o) + 1] if o in argv else d  # noqa: E731
    res = porter(val("--piece"), val("--corps"), val("--sortie"), val("--nom"),
                 float(val("--marge-mm", 3.0)), val("--peau", "retirer"))
    if "--rendre" in argv:
        rendre(res, os.path.join(val("--sortie"), "rendu.png"))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(a)

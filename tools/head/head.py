"""A head read from a PMX: welded face skin (颜), separate parts, computed landmarks,
face zone / skull zone, neck seam, vertices that follow the head.

Frame: the PMX one (MMD units, Y up, the face looks towards -Z).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))
sys.path.insert(0, HERE)

import geom as G  # noqa: E402
import pmx as PMX  # noqa: E402

SKIN = "颜"
# face sewn by retarget_head visage: a result reads back as a head
SKIN_SEWN = "颜_A"
# rest of the skin of a face that was itself composed (retarget_head visage from a result)
SKIN_SEWN_2 = "颜_A2"
# face parts: follow the face, and are replaced in visage mode
PARTS = ("颜2", "齿", "口线", "眉", "眉毛", "睫", "眉睫", "睫眉", "右睫眉", "二重", "二重影", "白目", "目",
          "口舌", "口", "舌", "瞳", "痣", "眉睫影", "目影", "瞳光", "目光", "目光2", "目色")
EYES = ("白目", "目")
MORPH_BLINKS = "まばたき"
MORPH_A = "あ"
MORPHS_SMILE = ("笑い", "口角上げ")


def _surrounds(P, c):
    """Does polygon P (n, 2) enclose point c? (even-odd rule)"""
    x, y = P[:, 0], P[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cutting = (y > c[1]) != (y2 > c[1])
    xs = x + (x2 - x) * (c[1] - y) / np.where(cutting, y2 - y, 1.0)
    return bool((cutting & (c[0] < xs)).sum() % 2)


class Head:
    def __init__(self, path, skin=None, pmx=None):
        self.path = path
        self.name = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
        self.pmx = pmx if pmx is not None else PMX.read(path)
        if skin is None:
            skin = (SKIN,) + tuple(n for n in (SKIN_SEWN, SKIN_SEWN_2) if any(m["name"] == n for m in self.pmx["materials"]))
        self.mats_skin = tuple(skin)
        t = PMX.to_arrays(self.pmx)
        self.t = t
        self.pos = t["pos"]
        self.faces = t["faces"]
        self.mat_face = t["mat_face"]
        self.names_mat = [m["name"] for m in self.pmx["materials"]]
        self.names_os = [o["name"] for o in self.pmx["bones"]]
        if SKIN not in self.names_mat:
            raise ValueError("%s: no %s material" % (path, SKIN))
        self.i_skin = [i for i, n in enumerate(self.names_mat) if n in self.mats_skin]
        self.scale, self.c_src, self.c_dst = 1.0, np.zeros(3), np.zeros(3)
        self.i_parts = [i for i, n in enumerate(self.names_mat) if n in PARTS]
        self._skin()
        self._morphs()
        self._loops()
        self._landmarks()
        self._seam_neck()
        self._track_head()

    # -------------------------------------------------------------- welded skin
    def _skin(self):
        fb = np.where(np.isin(self.mat_face, self.i_skin))[0]
        self.skin_faces_raw = fb
        Fb = self.faces[fb]
        vb = np.unique(Fb)
        rep, inv = G.weld(self.pos[vb])
        # welded vertex k -> raw vertices; raw vertex -> welded
        self.skin_raw = vb
        self.b2s = np.full(len(self.pos), -1)
        self.b2s[vb] = inv
        self.V = self.pos[vb[rep]].copy()
        self.s2b = [[] for _ in range(len(rep))]
        for i, k in zip(vb, inv):
            self.s2b[k].append(int(i))
        F = self.b2s[Fb]
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        self.F = F[ok]
        self.F_brut = fb[ok]  # raw face of each welded face
        self.adj = G.neighbors(len(self.V), self.F)
        self.n = len(self.V)

    def _morphs(self):
        """Vertex morphs: name -> (raw vertices, offsets)."""
        self.morphs = {}
        for m in self.pmx["morphs"]:
            if m["type"] != 1 or not m["offsets"]:
                continue
            i = np.array([d[0] for d in m["offsets"]], dtype=np.int64)
            d = np.array([d[1] for d in m["offsets"]], dtype=np.float64)
            self.morphs[m["name"]] = (i, d)

    def offset_skin(self, name):
        """Offset (n, 3) of the welded skin for a morph (average of the raw vertices)."""
        out = np.zeros((self.n, 3))
        if name not in self.morphs:
            return out
        i, d = self.morphs[name]
        k = self.b2s[i]
        m = k >= 0
        cnt = np.zeros(self.n)
        np.add.at(out, k[m], d[m])
        np.add.at(cnt, k[m], 1)
        return out / np.maximum(cnt, 1)[:, None]

    def offset_raw(self, name, idx):
        out = np.zeros((len(idx), 3))
        if name not in self.morphs:
            return out
        i, d = self.morphs[name]
        pos = {int(a): j for j, a in enumerate(idx)}
        for a, x in zip(i, d):
            j = pos.get(int(a))
            if j is not None:
                out[j] = x
        return out

    def vertices_mat(self, names):
        ids = [i for i, n in enumerate(self.names_mat) if n in names]
        return np.unique(self.faces[np.isin(self.mat_face, ids)])

    # -------------------------------------------------------------- boundary loops
    def _loops(self):
        bs = [np.array(b) for b in G.border_loops(self.F)]
        bs.sort(key=len, reverse=True)
        # outer boundary: the longest loop (by length, not by vertex count: in
        # Outerplane the neck is sparse and the eye or mouth hole has more vertices)
        lg = [float(np.linalg.norm(self.V[b] - self.V[np.roll(b, -1)], axis=1).sum()) for b in bs]
        bs.insert(0, bs.pop(int(np.argmax(lg))))
        self.loops = bs
        self.border = np.zeros(self.n, dtype=bool)
        for b in bs:
            self.border[b] = True
        # eye centres from the eye whites
        yv = self.vertices_mat(EYES)
        P = self.pos[yv]
        self.center_eye = {}
        for side, sg in (("g", 1), ("d", -1)):
            Q = P[P[:, 0] * sg > 0]
            self.center_eye[side] = Q.mean(0)
        outer = bs[0]
        self.loop_ext = outer
        self.loop_eye = {}
        for side in ("g", "d"):
            c = self.center_eye[side]
            candidate = [b for b in bs[1:] if np.linalg.norm(self.V[b].mean(0)[:2] - c[:2]) < 0.35]
            if not candidate:
                raise ValueError("%s: no eye hole %s" % (self.name, side))
            # the hole that encloses the eye centre (seen from the front): Yixuan also has an eyelid
            # crease and a slit cut near the eye, longer, which threw the whole registration off
            around = [b for b in candidate if _surrounds(self.V[b][:, :2], c[:2])]
            self.loop_eye[side] = max(around or candidate, key=len)
        # mouth: loop on the midline, below the eyes
        ym = min(self.center_eye["g"][1], self.center_eye["d"][1])
        candidate = [b for b in bs[1:] if abs(self.V[b].mean(0)[0]) < 0.05 and self.V[b].mean(0)[1] < ym - 0.2]
        if not candidate:
            raise ValueError("%s: no mouth hole" % self.name)
        self.loop_mouth = max(candidate, key=len)

    # -------------------------------------------------------------- landmarks
    def _landmarks(self):
        V = self.V
        R = {}
        for side, sg in (("g", 1), ("d", -1)):
            b = self.loop_eye[side]
            P = V[b]
            R["eye_%s_int" % side] = int(b[np.argmin(np.abs(P[:, 0]))])
            R["eye_%s_ext" % side] = int(b[np.argmax(np.abs(P[:, 0]))])
            R["eye_%s_top" % side] = int(b[np.argmax(P[:, 1])])
            R["eye_%s_bottom" % side] = int(b[np.argmin(P[:, 1])])
        b = self.loop_mouth
        P = V[b]
        R["mouth_l"] = int(b[np.argmax(P[:, 0])])
        R["mouth_r"] = int(b[np.argmin(P[:, 0])])
        mil = b[np.abs(P[:, 0]) < 0.25 * np.abs(P[:, 0]).max()]
        R["mouth_top"] = int(mil[np.argmax(V[mil, 1])])
        R["mouth_bottom"] = int(mil[np.argmin(V[mil, 1])])
        self.W = 2 * float(np.abs(V[:, 0]).max())
        med = np.where(np.abs(V[:, 0]) < 0.02 * self.W)[0]
        if len(med) < 10:
            med = np.argsort(np.abs(V[:, 0]))[:40]
        yo = 0.5 * (self.center_eye["g"][1] + self.center_eye["d"][1])
        yb = V[b, 1].mean()
        m = med[(V[med, 1] < yo) & (V[med, 1] > yb)]
        R["nose"] = int(m[np.argmin(V[m, 2])])
        cb = V[b].mean(0)
        m = med[V[med, 1] < cb[1]]
        R["chin"] = int(m[np.argmax(-(V[m, 1] - cb[1]) - (V[m, 2] - cb[2]))])
        R["forehead_top"] = int(med[np.argmax(V[med, 1])])
        for side, sg in (("g", 1), ("d", -1)):
            R["ear_%s" % side] = int(np.argmax(sg * V[:, 0]))
        self.landmarks = R
        # eyebrows: points of the 眉 (outside the skin); when eyelashes and eyebrows share a material
        # (眉睫, 睫眉), the eyebrows are the part well above the top of the eye
        sv = self.vertices_mat(("眉", "眉毛"))
        S = self.pos[sv]
        if not len(sv):
            S = self.pos[self.vertices_mat(("眉睫", "睫眉", "右睫眉"))]
            iod = float(np.linalg.norm(self.center_eye["g"] - self.center_eye["d"]))
            top_eye = max(V[self.loop_eye[c], 1].max() for c in "gd")
            S = S[S[:, 1] > top_eye + 0.12 * iod]
        if not len(S) or (S[:, 0] > 0).all() or (S[:, 0] < 0).all():
            raise ValueError("%s: no eyebrows in the geometry" % self.name)
        self.points_eyebrows = {}
        for side, sg in (("g", 1), ("d", -1)):
            Q = S[S[:, 0] * sg > 0]
            self.points_eyebrows["eyebrow_%s_int" % side] = Q[np.argmin(np.abs(Q[:, 0]))]
            self.points_eyebrows["eyebrow_%s_ext" % side] = Q[np.argmax(np.abs(Q[:, 0]))]
            self.points_eyebrows["eyebrow_%s_top" % side] = Q[np.argmax(Q[:, 1])]
        self.iod = float(np.linalg.norm(self.center_eye["g"] - self.center_eye["d"]))
        self.H = float(V[R["forehead_top"], 1] - V[R["chin"], 1])
        # top of the skull: the face skin often stops under the hair, higher or lower
        # depending on the model; the underside of the hair cap, straight above the middle of
        # the head, stands in for it (the higher of the two)
        top = float(V[R["forehead_top"], 1])
        hv = self.pos[self.vertices_mat([n for n in self.names_mat if "髮" in n or "发" in n])]
        zc = 0.5 * (V[:, 2].min() + V[:, 2].max())
        r = 0.1 * self.iod
        cal = hv[(np.abs(hv[:, 0]) < r) & (np.abs(hv[:, 2] - zc) < r) & (hv[:, 1] > top - r)] if len(hv) else hv
        self.y_skull = max(top, float(cal[:, 1].min())) if len(cal) else top
        self.H_crane = self.y_skull - float(V[R["chin"], 1])

    def eye_visible(self, radius=0.45, depth=0.02):
        """Eye rim seen from the front (Outerplane heads): the eyelid skin dives behind
        the eye plate and lines an orbit, and its hole is only a folded slit (area
        0.03-0.04 for Mene, Kanon, Idith, against 0.05-0.09 of visible iris). Registered onto that
        slit, the pivot's hole covered the iris (89 % of Mene's 目光 under the skin). Here the rim
        is that of the visible skin: forward-facing faces hidden by no skin (within
        prof × IOD), closer than rayon × IOD to the eye centre, in one piece; the
        smallest boundary loop of that piece that encloses the centre. Replaces
        the eye loops and recomputes the landmarks. Returns {side: (area before, area after)}."""
        from morphs import _progress
        V, F = self.V, self.F
        c3 = V[F].mean(1)
        vis = (G.normals_faces(V, F)[:, 2] < 0) & (_progress(V, F, c3) > -depth * self.iod)
        area = lambda b: abs(G.area_polygon_xy(V[b][:, :2]))  # noqa: E731
        new, out = {}, {}
        for side in ("g", "d"):
            c = self.center_eye[side][:2]
            sel = vis & (np.linalg.norm(c3[:, :2] - c, axis=1) < radius * self.iod)
            lab = G.components_faces(F, sel)
            if (lab >= 0).sum() == 0:
                break
            large = np.bincount(lab[lab >= 0]).argmax()
            a0 = area(self.loop_eye[side])
            bs = sorted((np.array(b) for b in G.border_loops(F[lab == large]) if len(b) >= 6), key=area)
            bs = [b for b in bs[:-1] if area(b) >= 0.5 * a0]  # without the outer boundary (rayon)
            if not bs:
                break
            # the hole that encloses the centre, otherwise the closest (centre lying on the slit)
            inside = [b for b in bs if _surrounds(V[b][:, :2], c)]
            b = inside[0] if inside else min(bs, key=lambda b: np.linalg.norm(V[b][:, :2].mean(0) - c))
            new[side], out[side] = b, (round(a0, 4), round(area(b), 4))
        if len(new) < 2:  # an eye without a visible rim: keep both holes, no asymmetry
            return {}
        self.loop_eye.update(new)
        self._landmarks()
        return out

    def points_face(self):
        """Face landmarks (those whose proportions are kept): name -> position."""
        names = [k for k in self.landmarks if k.startswith(("eye", "mouth", "nose"))]
        P = {k: self.V[self.landmarks[k]] for k in names}
        P.update(self.points_eyebrows)
        return P

    def center_face(self):
        P = self.points_face()
        return np.mean([P[k] for k in sorted(P)], axis=0)

    # -------------------------------------------------------------- host frame
    def put(self, scale, c_src, c_dst):
        """Similarity without rotation into the working frame: p -> e (p - c_src) + c_dst."""
        self.scale, self.c_src, self.c_dst = float(scale), np.asarray(c_src), np.asarray(c_dst)

    def _host(self, p):
        return self.scale * (np.asarray(p) - self.c_src) + self.c_dst

    # -------------------------------------------------------------- zones
    def weight_face(self, margin, band, V=None):
        """1 in the face (front-view hull of the eyes, eyebrows, nose, mouth, dilated by
        `margin`), 0 on the skull, smooth transition over `band` (head units)."""
        V = self.V if V is None else V
        pts = [self.V[self.loop_eye["g"]], self.V[self.loop_eye["d"]],
               self.V[self.loop_mouth], self.V[[self.landmarks["nose"]]],
               np.array(list(self.points_eyebrows.values()))]
        Hc = G.hull_convex_2d(np.vstack(pts)[:, :2])
        sd = G.distance_signed_polygon(V[:, :2], Hc)
        w = 1 - G.smooth((sd - margin) / band)
        # the back and the sides (normal facing backwards) are never face; this
        # criterion does not apply inside the hull of the features (the underside of the lip and
        # nostrils faces down while still being face) and takes full effect at `marge`
        n = G.normals_vertices(self.V, self.F) if V is self.V else G.normals_vertices(V, self.F)
        face = G.smooth((-n[:, 2] - 0.15) / 0.35)
        w = w * (1 - G.smooth(sd / margin) * (1 - face))
        return w

    # -------------------------------------------------------------- neck seam
    def _seam_neck(self):
        """Skin vertices coincident with a vertex of a material other than the face
        (the body): they hold the neck seam and stay in place in shape mode."""
        others = [i for i, n in enumerate(self.names_mat) if n not in self.mats_skin and n not in PARTS]
        vo = np.unique(self.faces[np.isin(self.mat_face, others)])
        q = lambda P: {tuple(x) for x in np.round(P / 1e-5).astype(np.int64)}
        the_key = q(self.pos[vo])
        Vq = np.round(self.V / 1e-5).astype(np.int64)
        self.neck = np.array([k for k in range(self.n) if self.border[k] and tuple(Vq[k]) in the_key],
                            dtype=np.int64)

    # -------------------------------------------------------------- vertices following the head
    def _track_head(self):
        """Head weight of each raw vertex: sum of the weights on 頭 and its descendants."""
        per = np.array([o["parent"] for o in self.pmx["bones"]])
        head = self.names_os.index("頭") if "頭" in self.names_os else None
        desc = np.zeros(len(per), dtype=bool)
        if head is not None:
            desc[head] = True
            for _ in range(64):
                nd = desc | ((per >= 0) & desc[np.maximum(per, 0)])
                if (nd == desc).all():
                    break
                desc = nd
        o, w = self.t["bones"], self.t["weights"]
        m = (o >= 0) & desc[np.maximum(o, 0)]
        self.weight_head = np.clip((w * m).sum(1), 0, 1)
        self.os_head = desc

    def summary(self):
        return {"path": self.path, "skin_vertices": self.n, "skin_faces": len(self.F),
                "loops": [len(b) for b in self.loops], "width": round(self.W, 4),
                "height": round(self.H, 4), "iod": round(self.iod, 4),
                "neck_vertices": int(len(self.neck)),
                "landmarks": {k: [round(float(x), 4) for x in self.V[v]] for k, v in self.landmarks.items()}}

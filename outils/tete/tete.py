"""A head read from a PMX: welded face skin (颜), separate parts, computed landmarks,
face zone / skull zone, neck seam, vertices that follow the head.

Frame: the PMX one (MMD units, Y up, the face looks towards -Z).
"""
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
sys.path.insert(0, ICI)

import geom as G  # noqa: E402
import pmx as PMX  # noqa: E402

PEAU = "颜"
# face sewn by retarget_tete visage: a result reads back as a head
PEAU_COUSUE = "颜_A"
# rest of the skin of a face that was itself composed (retarget_tete visage from a result)
PEAU_COUSUE_2 = "颜_A2"
# face parts: follow the face, and are replaced in visage mode
PIECES = ("颜2", "齿", "口线", "眉", "眉毛", "睫", "眉睫", "睫眉", "右睫眉", "二重", "二重影", "白目", "目",
          "口舌", "口", "舌", "瞳", "痣", "眉睫影", "目影", "瞳光", "目光", "目光2", "目色")
YEUX = ("白目", "目")
MORPHE_CLIGNE = "まばたき"
MORPHE_A = "あ"
MORPHES_SOURIRE = ("笑い", "口角上げ")


def _entoure(P, c):
    """Does polygon P (n, 2) enclose point c? (even-odd rule)"""
    x, y = P[:, 0], P[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    coupe = (y > c[1]) != (y2 > c[1])
    xs = x + (x2 - x) * (c[1] - y) / np.where(coupe, y2 - y, 1.0)
    return bool((coupe & (c[0] < xs)).sum() % 2)


class Tete:
    def __init__(self, chemin, peau=None, pmx=None):
        self.chemin = chemin
        self.nom = os.path.basename(os.path.dirname(chemin)) or os.path.basename(chemin)
        self.pmx = pmx if pmx is not None else PMX.lire(chemin)
        if peau is None:
            peau = (PEAU,) + tuple(n for n in (PEAU_COUSUE, PEAU_COUSUE_2) if any(m["nom"] == n for m in self.pmx["materiaux"]))
        self.mats_peau = tuple(peau)
        t = PMX.vers_tableaux(self.pmx)
        self.t = t
        self.pos = t["pos"]
        self.faces = t["faces"]
        self.mat_face = t["mat_face"]
        self.noms_mat = [m["nom"] for m in self.pmx["materiaux"]]
        self.noms_os = [o["nom"] for o in self.pmx["os"]]
        if PEAU not in self.noms_mat:
            raise ValueError("%s: no %s material" % (chemin, PEAU))
        self.i_peau = [i for i, n in enumerate(self.noms_mat) if n in self.mats_peau]
        self.echelle, self.c_src, self.c_dst = 1.0, np.zeros(3), np.zeros(3)
        self.i_pieces = [i for i, n in enumerate(self.noms_mat) if n in PIECES]
        self._peau()
        self._morphes()
        self._boucles()
        self._reperes()
        self._couture_cou()
        self._suivi_tete()

    # -------------------------------------------------------------- welded skin
    def _peau(self):
        fb = np.where(np.isin(self.mat_face, self.i_peau))[0]
        self.faces_peau_brutes = fb
        Fb = self.faces[fb]
        vb = np.unique(Fb)
        rep, inv = G.souder(self.pos[vb])
        # welded vertex k -> raw vertices; raw vertex -> welded
        self.peau_bruts = vb
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
        self.adj = G.voisins(len(self.V), self.F)
        self.n = len(self.V)

    def _morphes(self):
        """Vertex morphs: name -> (raw vertices, offsets)."""
        self.morphes = {}
        for m in self.pmx["morphes"]:
            if m["type"] != 1 or not m["decalages"]:
                continue
            i = np.array([d[0] for d in m["decalages"]], dtype=np.int64)
            d = np.array([d[1] for d in m["decalages"]], dtype=np.float64)
            self.morphes[m["nom"]] = (i, d)

    def decalage_peau(self, nom):
        """Offset (n, 3) of the welded skin for a morph (average of the raw vertices)."""
        out = np.zeros((self.n, 3))
        if nom not in self.morphes:
            return out
        i, d = self.morphes[nom]
        k = self.b2s[i]
        m = k >= 0
        cnt = np.zeros(self.n)
        np.add.at(out, k[m], d[m])
        np.add.at(cnt, k[m], 1)
        return out / np.maximum(cnt, 1)[:, None]

    def decalage_bruts(self, nom, idx):
        out = np.zeros((len(idx), 3))
        if nom not in self.morphes:
            return out
        i, d = self.morphes[nom]
        pos = {int(a): j for j, a in enumerate(idx)}
        for a, x in zip(i, d):
            j = pos.get(int(a))
            if j is not None:
                out[j] = x
        return out

    def sommets_mat(self, noms):
        ids = [i for i, n in enumerate(self.noms_mat) if n in noms]
        return np.unique(self.faces[np.isin(self.mat_face, ids)])

    # -------------------------------------------------------------- boundary loops
    def _boucles(self):
        bs = [np.array(b) for b in G.boucles_bord(self.F)]
        bs.sort(key=len, reverse=True)
        # outer boundary: the longest loop (by length, not by vertex count: in
        # Outerplane the neck is sparse and the eye or mouth hole has more vertices)
        lg = [float(np.linalg.norm(self.V[b] - self.V[np.roll(b, -1)], axis=1).sum()) for b in bs]
        bs.insert(0, bs.pop(int(np.argmax(lg))))
        self.boucles = bs
        self.bord = np.zeros(self.n, dtype=bool)
        for b in bs:
            self.bord[b] = True
        # eye centres from the eye whites
        yv = self.sommets_mat(YEUX)
        P = self.pos[yv]
        self.centre_oeil = {}
        for cote, sg in (("g", 1), ("d", -1)):
            Q = P[P[:, 0] * sg > 0]
            self.centre_oeil[cote] = Q.mean(0)
        exterieure = bs[0]
        self.boucle_ext = exterieure
        self.boucle_oeil = {}
        for cote in ("g", "d"):
            c = self.centre_oeil[cote]
            cand = [b for b in bs[1:] if np.linalg.norm(self.V[b].mean(0)[:2] - c[:2]) < 0.35]
            if not cand:
                raise ValueError("%s: no eye hole %s" % (self.nom, cote))
            # the hole that encloses the eye centre (seen from the front): Yixuan also has an eyelid
            # crease and a slit cut near the eye, longer, which threw the whole registration off
            autour = [b for b in cand if _entoure(self.V[b][:, :2], c[:2])]
            self.boucle_oeil[cote] = max(autour or cand, key=len)
        # mouth: loop on the midline, below the eyes
        ym = min(self.centre_oeil["g"][1], self.centre_oeil["d"][1])
        cand = [b for b in bs[1:] if abs(self.V[b].mean(0)[0]) < 0.05 and self.V[b].mean(0)[1] < ym - 0.2]
        if not cand:
            raise ValueError("%s: no mouth hole" % self.nom)
        self.boucle_bouche = max(cand, key=len)

    # -------------------------------------------------------------- landmarks
    def _reperes(self):
        V = self.V
        R = {}
        for cote, sg in (("g", 1), ("d", -1)):
            b = self.boucle_oeil[cote]
            P = V[b]
            R["oeil_%s_int" % cote] = int(b[np.argmin(np.abs(P[:, 0]))])
            R["oeil_%s_ext" % cote] = int(b[np.argmax(np.abs(P[:, 0]))])
            R["oeil_%s_haut" % cote] = int(b[np.argmax(P[:, 1])])
            R["oeil_%s_bas" % cote] = int(b[np.argmin(P[:, 1])])
        b = self.boucle_bouche
        P = V[b]
        R["bouche_g"] = int(b[np.argmax(P[:, 0])])
        R["bouche_d"] = int(b[np.argmin(P[:, 0])])
        mil = b[np.abs(P[:, 0]) < 0.25 * np.abs(P[:, 0]).max()]
        R["bouche_haut"] = int(mil[np.argmax(V[mil, 1])])
        R["bouche_bas"] = int(mil[np.argmin(V[mil, 1])])
        self.W = 2 * float(np.abs(V[:, 0]).max())
        med = np.where(np.abs(V[:, 0]) < 0.02 * self.W)[0]
        if len(med) < 10:
            med = np.argsort(np.abs(V[:, 0]))[:40]
        yo = 0.5 * (self.centre_oeil["g"][1] + self.centre_oeil["d"][1])
        yb = V[b, 1].mean()
        m = med[(V[med, 1] < yo) & (V[med, 1] > yb)]
        R["nez"] = int(m[np.argmin(V[m, 2])])
        cb = V[b].mean(0)
        m = med[V[med, 1] < cb[1]]
        R["menton"] = int(m[np.argmax(-(V[m, 1] - cb[1]) - (V[m, 2] - cb[2]))])
        R["front_haut"] = int(med[np.argmax(V[med, 1])])
        for cote, sg in (("g", 1), ("d", -1)):
            R["oreille_%s" % cote] = int(np.argmax(sg * V[:, 0]))
        self.reperes = R
        # eyebrows: points of the 眉 (outside the skin); when eyelashes and eyebrows share a material
        # (眉睫, 睫眉), the eyebrows are the part well above the top of the eye
        sv = self.sommets_mat(("眉", "眉毛"))
        S = self.pos[sv]
        if not len(sv):
            S = self.pos[self.sommets_mat(("眉睫", "睫眉", "右睫眉"))]
            iod = float(np.linalg.norm(self.centre_oeil["g"] - self.centre_oeil["d"]))
            haut_oeil = max(V[self.boucle_oeil[c], 1].max() for c in "gd")
            S = S[S[:, 1] > haut_oeil + 0.12 * iod]
        if not len(S) or (S[:, 0] > 0).all() or (S[:, 0] < 0).all():
            raise ValueError("%s: no eyebrows in the geometry" % self.nom)
        self.points_sourcils = {}
        for cote, sg in (("g", 1), ("d", -1)):
            Q = S[S[:, 0] * sg > 0]
            self.points_sourcils["sourcil_%s_int" % cote] = Q[np.argmin(np.abs(Q[:, 0]))]
            self.points_sourcils["sourcil_%s_ext" % cote] = Q[np.argmax(np.abs(Q[:, 0]))]
            self.points_sourcils["sourcil_%s_haut" % cote] = Q[np.argmax(Q[:, 1])]
        self.iod = float(np.linalg.norm(self.centre_oeil["g"] - self.centre_oeil["d"]))
        self.H = float(V[R["front_haut"], 1] - V[R["menton"], 1])
        # top of the skull: the face skin often stops under the hair, higher or lower
        # depending on the model; the underside of the hair cap, straight above the middle of
        # the head, stands in for it (the higher of the two)
        haut = float(V[R["front_haut"], 1])
        hv = self.pos[self.sommets_mat([n for n in self.noms_mat if "髮" in n or "发" in n])]
        zc = 0.5 * (V[:, 2].min() + V[:, 2].max())
        r = 0.1 * self.iod
        cal = hv[(np.abs(hv[:, 0]) < r) & (np.abs(hv[:, 2] - zc) < r) & (hv[:, 1] > haut - r)] if len(hv) else hv
        self.y_crane = max(haut, float(cal[:, 1].min())) if len(cal) else haut
        self.H_crane = self.y_crane - float(V[R["menton"], 1])

    def oeil_visible(self, rayon=0.45, prof=0.02):
        """Eye rim seen from the front (Outerplane heads): the eyelid skin dives behind
        the eye plate and lines an orbit, and its hole is only a folded slit (area
        0.03-0.04 for Mene, Kanon, Idith, against 0.05-0.09 of visible iris). Registered onto that
        slit, the pivot's hole covered the iris (89 % of Mene's 目光 under the skin). Here the rim
        is that of the visible skin: forward-facing faces hidden by no skin (within
        prof × IOD), closer than rayon × IOD to the eye centre, in one piece; the
        smallest boundary loop of that piece that encloses the centre. Replaces
        boucle_oeil and recomputes the landmarks. Returns {side: (area before, area after)}."""
        from morphes import _avance
        V, F = self.V, self.F
        c3 = V[F].mean(1)
        vis = (G.normales_faces(V, F)[:, 2] < 0) & (_avance(V, F, c3) > -prof * self.iod)
        aire = lambda b: abs(G.aire_polygone_xy(V[b][:, :2]))  # noqa: E731
        nouv, out = {}, {}
        for cote in ("g", "d"):
            c = self.centre_oeil[cote][:2]
            sel = vis & (np.linalg.norm(c3[:, :2] - c, axis=1) < rayon * self.iod)
            lab = G.composantes_faces(F, sel)
            if (lab >= 0).sum() == 0:
                break
            grand = np.bincount(lab[lab >= 0]).argmax()
            a0 = aire(self.boucle_oeil[cote])
            bs = sorted((np.array(b) for b in G.boucles_bord(F[lab == grand]) if len(b) >= 6), key=aire)
            bs = [b for b in bs[:-1] if aire(b) >= 0.5 * a0]  # without the outer boundary (rayon)
            if not bs:
                break
            # the hole that encloses the centre, otherwise the closest (centre lying on the slit)
            dedans = [b for b in bs if _entoure(V[b][:, :2], c)]
            b = dedans[0] if dedans else min(bs, key=lambda b: np.linalg.norm(V[b][:, :2].mean(0) - c))
            nouv[cote], out[cote] = b, (round(a0, 4), round(aire(b), 4))
        if len(nouv) < 2:  # an eye without a visible rim: keep both holes, no asymmetry
            return {}
        self.boucle_oeil.update(nouv)
        self._reperes()
        return out

    def points_visage(self):
        """Face landmarks (those whose proportions are kept): name -> position."""
        noms = [k for k in self.reperes if k.startswith(("oeil", "bouche", "nez"))]
        P = {k: self.V[self.reperes[k]] for k in noms}
        P.update(self.points_sourcils)
        return P

    def centre_visage(self):
        P = self.points_visage()
        return np.mean([P[k] for k in sorted(P)], axis=0)

    # -------------------------------------------------------------- host frame
    def placer(self, echelle, c_src, c_dst):
        """Similarity without rotation into the working frame: p -> e (p - c_src) + c_dst."""
        self.echelle, self.c_src, self.c_dst = float(echelle), np.asarray(c_src), np.asarray(c_dst)

    def _hote(self, p):
        return self.echelle * (np.asarray(p) - self.c_src) + self.c_dst

    # -------------------------------------------------------------- zones
    def poids_visage(self, marge, bande, V=None):
        """1 in the face (front-view hull of the eyes, eyebrows, nose, mouth, dilated by
        `marge`), 0 on the skull, smooth transition over `bande` (head units)."""
        V = self.V if V is None else V
        pts = [self.V[self.boucle_oeil["g"]], self.V[self.boucle_oeil["d"]],
               self.V[self.boucle_bouche], self.V[[self.reperes["nez"]]],
               np.array(list(self.points_sourcils.values()))]
        Hc = G.enveloppe_convexe_2d(np.vstack(pts)[:, :2])
        sd = G.distance_signee_polygone(V[:, :2], Hc)
        w = 1 - G.lisser((sd - marge) / bande)
        # the back and the sides (normal facing backwards) are never face; this
        # criterion does not apply inside the hull of the features (the underside of the lip and
        # nostrils faces down while still being face) and takes full effect at `marge`
        n = G.normales_sommets(self.V, self.F) if V is self.V else G.normales_sommets(V, self.F)
        face = G.lisser((-n[:, 2] - 0.15) / 0.35)
        w = w * (1 - G.lisser(sd / marge) * (1 - face))
        return w

    # -------------------------------------------------------------- neck seam
    def _couture_cou(self):
        """Skin vertices coincident with a vertex of a material other than the face
        (the body): they hold the neck seam and stay in place in forme mode."""
        autres = [i for i, n in enumerate(self.noms_mat) if n not in self.mats_peau and n not in PIECES]
        vo = np.unique(self.faces[np.isin(self.mat_face, autres)])
        q = lambda P: {tuple(x) for x in np.round(P / 1e-5).astype(np.int64)}
        cle = q(self.pos[vo])
        Vq = np.round(self.V / 1e-5).astype(np.int64)
        self.cou = np.array([k for k in range(self.n) if self.bord[k] and tuple(Vq[k]) in cle],
                            dtype=np.int64)

    # -------------------------------------------------------------- vertices following the head
    def _suivi_tete(self):
        """Head weight of each raw vertex: sum of the weights on 頭 and its descendants."""
        par = np.array([o["parent"] for o in self.pmx["os"]])
        tete = self.noms_os.index("頭") if "頭" in self.noms_os else None
        desc = np.zeros(len(par), dtype=bool)
        if tete is not None:
            desc[tete] = True
            for _ in range(64):
                nd = desc | ((par >= 0) & desc[np.maximum(par, 0)])
                if (nd == desc).all():
                    break
                desc = nd
        o, w = self.t["os"], self.t["poids"]
        m = (o >= 0) & desc[np.maximum(o, 0)]
        self.poids_tete = np.clip((w * m).sum(1), 0, 1)
        self.os_tete = desc

    def resume(self):
        return {"chemin": self.chemin, "sommets_peau": self.n, "faces_peau": len(self.F),
                "boucles": [len(b) for b in self.boucles], "largeur": round(self.W, 4),
                "hauteur": round(self.H, 4), "iod": round(self.iod, 4),
                "sommets_cou": int(len(self.cou)),
                "reperes": {k: [round(float(x), 4) for x in self.V[v]] for k, v in self.reperes.items()}}

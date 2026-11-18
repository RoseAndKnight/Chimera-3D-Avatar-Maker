"""A head read from a GLB (a studio other than the active game): same attributes as tete.Tete for
registration onto the pivot head (pivot.recaler_un, recalage.recaler) and the signature spectrum.

    python3 -I tools/head/head_glb.py inspecter <model.glb>     # materials, chosen skin, landmarks

Reading: binary glTF 2.0 in numpy (JSON + buffer), rest positions of the primitives (bind
pose, mesh frame), without the morphs. The face skin is the material whose
name contains one of the PEAU_MOTIFS patterns (face separate from the skull); the eyes, eyebrows and
hair are found the same way from their material names.

Frame: brought back into the PMX one (Y up, the face looks towards -Z); the gaze direction
is glTF's (+Z). Units: those of the GLB
scaled to the typical interocular distance of the active game's heads (nothing depends on it after the similarity, only the
initial placement and the relative thresholds).

Nothing is written: the geometry stays in memory.
"""
import json
import os
import re
import struct
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
sys.path.append(os.path.join(ICI, ".."))

import geom as G  # noqa: E402
import games as jeux  # noqa: E402

TYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TAILLES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

# (lowercase) material name patterns, per studio
PEAU_MOTIFS = ("face_new", "_face", "face", "visage")  # visage: Keijo (Orpheya)
PEAU_EXCLUS = ("mask", "eye", "brow", "hair", "lash", "jiemao", "mei", "teeth", "tooth", "tongue")
YEUX_MOTIFS = ("_eyes", "eyeball", "_eye", "eye", "blanc_des_yeux")
YEUX_EXCLUS = ("brow", "lash", "bantou", "touming", "gaoguang", "highlight", "shadow")
SOURCILS_MOTIFS = ("brow", "meimao", "_mei", "sourcils")
CHEVEUX_MOTIFS = ("hair", "chevelure")
# typical interocular distance of the active game's heads (MMD units, description: tetes.ecart_yeux_typique): used to
# bring the head to a similar scale
IOD_JEU = jeux.actif().valeur("tetes.ecart_yeux_typique")


def lire_glb(chemin):
    """Returns (json, binary buffer). chemin: a file, or the bytes of a GLB already in memory."""
    if isinstance(chemin, (bytes, bytearray)):
        b = bytes(chemin)
    else:
        with open(chemin, "rb") as f:
            b = f.read()
    if b[:4] != b"glTF":
        raise ValueError("not a GLB")
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    p = 20 + n
    binaire = b""
    if p + 8 <= len(b):
        m = struct.unpack("<I", b[p:p + 4])[0]
        binaire = b[p + 8:p + 8 + m]
    return j, binaire


def accesseur(j, binaire, k):
    a = j["accessors"][k]
    t = TYPES[a["componentType"]]
    c = TAILLES[a["type"]]
    n = a["count"]
    out = np.zeros((n, c), dtype=t)
    if "bufferView" in a:
        bv = j["bufferViews"][a["bufferView"]]
        dep = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        pas = bv.get("byteStride", 0) or c * np.dtype(t).itemsize
        brut = np.frombuffer(binaire, dtype=np.uint8, count=(n - 1) * pas + c * np.dtype(t).itemsize, offset=dep)
        if pas == c * np.dtype(t).itemsize:
            out = brut.view(t).reshape(n, c).copy()
        else:
            lignes = np.lib.stride_tricks.as_strided(brut, shape=(n, c * np.dtype(t).itemsize), strides=(pas, 1))
            out = lignes.copy().view(t).reshape(n, c)
    if "sparse" in a:
        s = a["sparse"]
        bi, bv_ = s["indices"], s["values"]
        vi = j["bufferViews"][bi["bufferView"]]
        idx = np.frombuffer(binaire, dtype=TYPES[bi["componentType"]], count=s["count"],
                            offset=vi.get("byteOffset", 0) + bi.get("byteOffset", 0))
        vv = j["bufferViews"][bv_["bufferView"]]
        val = np.frombuffer(binaire, dtype=t, count=s["count"] * c,
                            offset=vv.get("byteOffset", 0) + bv_.get("byteOffset", 0)).reshape(-1, c)
        out = out.copy()
        out[idx] = val
    if a.get("normalized") and t != np.float32:
        out = out.astype(np.float64) / np.iinfo(t).max
    return out


def _matrice_noeud(nd):
    if "matrix" in nd:
        return np.array(nd["matrix"], dtype=float).reshape(4, 4).T
    T = np.eye(4)
    x, y, z, w = nd.get("rotation", (0, 0, 0, 1))
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T[:3, :3] = R * np.array(nd.get("scale", (1, 1, 1)))[None, :]
    T[:3, 3] = nd.get("translation", (0, 0, 0))
    return T


def _monde(j):
    """World matrix of each node."""
    par = {}
    for i, nd in enumerate(j.get("nodes", [])):
        for c in nd.get("children", []):
            par[c] = i
    cache = {}

    def m(i):
        if i not in cache:
            loc = _matrice_noeud(j["nodes"][i])
            cache[i] = m(par[i]) @ loc if i in par else loc
        return cache[i]
    return [m(i) for i in range(len(j.get("nodes", [])))]


def maillage(chemin):
    """All triangulated primitives, in the scene frame: positions (n, 3),
    faces (f, 3), material of each face, material names."""
    j, binaire = lire_glb(chemin)
    noms_mat = [m.get("name", "mat%d" % i) for i, m in enumerate(j.get("materials", []))]
    mondes = _monde(j)
    pos, faces, mat = [], [], []
    base = 0
    for i, nd in enumerate(j.get("nodes", [])):
        if "mesh" not in nd:
            continue
        # skinned mesh: vertices in the mesh frame, under the skeleton's parent;
        # the node's world matrix (often the identity) is kept to orient the scene
        M = mondes[i]
        for pr in j["meshes"][nd["mesh"]]["primitives"]:
            if pr.get("mode", 4) != 4:
                continue
            P = accesseur(j, binaire, pr["attributes"]["POSITION"]).astype(np.float64)
            P = P @ M[:3, :3].T + M[:3, 3]
            if "indices" in pr:
                I = accesseur(j, binaire, pr["indices"]).astype(np.int64).reshape(-1, 3)
            else:
                I = np.arange(len(P)).reshape(-1, 3)
            pos.append(P)
            faces.append(I + base)
            mat.append(np.full(len(I), pr.get("material", -1)))
            base += len(P)
    return np.vstack(pos), np.vstack(faces), np.concatenate(mat), noms_mat


def glb_depuis_pmx(chemin, metres=0.08):
    """GLB (bytes, in memory only) of a PMX: glTF frame (gaze +Z, the character's
    left at +X: half-turn around Y), in metres, one primitive per material. Used for the
    check: a ZZZ head sent through the GLB path must land back on its own shape."""
    import pmx as PMX
    P = PMX.lire(chemin)
    t = PMX.vers_tableaux(P)
    pos = (t["pos"] * np.array([-1.0, 1.0, -1.0]) * metres).astype(np.float32)
    noms = [m["nom"] for m in P["materiaux"]]
    blocs = [pos.tobytes()]
    vues = [{"buffer": 0, "byteOffset": 0, "byteLength": pos.nbytes}]
    acc = [{"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
            "min": pos.min(0).tolist(), "max": pos.max(0).tolist()}]
    prims = []
    dep = pos.nbytes
    for i in range(len(noms)):
        I = t["faces"][t["mat_face"] == i].astype(np.uint32).ravel()
        if not len(I):
            continue
        blocs.append(I.tobytes())
        vues.append({"buffer": 0, "byteOffset": dep, "byteLength": I.nbytes})
        acc.append({"bufferView": len(vues) - 1, "componentType": 5125, "count": len(I), "type": "SCALAR"})
        prims.append({"attributes": {"POSITION": 0}, "indices": len(acc) - 1, "material": i})
        dep += I.nbytes
    binaire = b"".join(blocs)
    j = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binaire)}], "bufferViews": vues,
         "accessors": acc, "materials": [{"name": n} for n in noms],
         "meshes": [{"primitives": prims}], "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0}
    js = json.dumps(j, ensure_ascii=False).encode()
    js += b" " * (-len(js) % 4)
    binaire += b"\0" * (-len(binaire) % 4)
    tot = 12 + 8 + len(js) + 8 + len(binaire)
    return (b"glTF" + struct.pack("<II", 2, tot) + struct.pack("<I", len(js)) + b"JSON" + js +
            struct.pack("<I", len(binaire)) + b"BIN\0" + binaire)


def indices_zzz(noms):
    """Materials of a ZZZ head (for the check): skin, eyes, eyebrows, hair."""
    import head as TT
    ix = lambda ok: [i for i, n in enumerate(noms) if ok(n)]
    # tete.Tete rule: 眉/眉毛 taken whole, the height filter only applies to
    # materials shared with the eyelashes
    seuls = ix(lambda n: n in ("眉", "眉毛"))
    return dict(peau=ix(lambda n: n == TT.PEAU), yeux=ix(lambda n: n in TT.YEUX),
                sourcils=seuls or ix(lambda n: n in ("眉睫", "睫眉", "右睫眉")), filtrer_sourcils=not seuls,
                garder_ilots=True,
                cheveux=ix(lambda n: "髮" in n or "发" in n))


def _choisir(noms, motifs, exclus=()):
    """Indices of the materials whose name contains a pattern (the first pattern that matches)."""
    bas = [n.lower() for n in noms]
    for mo in motifs:
        r = [i for i, n in enumerate(bas) if mo in n and not any(e in n for e in exclus)]
        if r:
            return r
    return []


def _dans(poly, p):
    """Is the (2D) point p inside polygon poly (n, 2)? (ray casting rule)"""
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    coupe = (y > p[1]) != (y2 > p[1])
    xs = x + (p[1] - y) * (x2 - x) / np.where(y2 == y, 1e-12, y2 - y)
    return bool((coupe & (xs > p[0])).sum() % 2)


class TeteGLB:
    """Face skin of a GLB, same attributes as tete.Tete for registration."""

    def __init__(self, chemin, nom=None, peau=None, yeux=None, sourcils=None, cheveux=None, maillage_=None,
                 filtrer_sourcils=True, garder_ilots=False):
        self.chemin = chemin if isinstance(chemin, str) else "<memory>"
        self.nom = nom or os.path.basename(os.path.dirname(self.chemin)) or os.path.basename(self.chemin)
        pos, faces, mat, noms = maillage_ if maillage_ is not None else maillage(chemin)
        self.noms_mat = noms
        self.i_peau = peau if peau is not None else _choisir(noms, PEAU_MOTIFS, PEAU_EXCLUS)
        self.i_yeux = yeux if yeux is not None else _choisir(noms, YEUX_MOTIFS, YEUX_EXCLUS)
        self.i_sourcils = sourcils if sourcils is not None else _choisir(noms, SOURCILS_MOTIFS)
        self.i_cheveux = cheveux if cheveux is not None else _choisir(noms, CHEVEUX_MOTIFS)
        if not self.i_peau:
            raise ValueError("%s: no face material among %s" % (self.nom, noms))
        if not self.i_yeux:
            raise ValueError("%s: no eye material among %s" % (self.nom, noms))
        self.pos, self.faces, self.mat_face = pos, faces, mat
        self.filtrer_sourcils = filtrer_sourcils
        self.garder_ilots = garder_ilots
        self._orienter()
        self.echelle, self.c_src, self.c_dst = 1.0, np.zeros(3), np.zeros(3)
        self._peau()
        self._boucles()
        self._reperes()

    def sommets_mat(self, ids):
        return np.unique(self.faces[np.isin(self.mat_face, list(ids))])

    # -------------------------------------------------------------- PMX frame
    def _orienter(self):
        """Y up, gaze towards -Z, interocular distance scaled to ZZZ's; centre at the eyes."""
        P = self.pos
        peau = P[self.sommets_mat(self.i_peau)]
        yeux = P[self.sommets_mat(self.i_yeux)]
        # vertical axis: the glTF scene is Y up; some engine exports stay Z
        # up (the face then extends further in z than in y around the eyes)
        e = yeux.mean(0)
        haut = np.array([0.0, 1.0, 0.0])
        if np.ptp(peau[:, 2]) > 1.3 * np.ptp(peau[:, 1]):
            haut = np.array([0.0, 0.0, 1.0])
        # forward: glTF convention, the character looks towards +Z (towards -Y if the scene is Z
        # up); guessing the direction from the eyes fails when the skin goes up behind the ears
        av = np.array([0.0, 0.0, 1.0]) if haut[1] == 1 else np.array([0.0, -1.0, 0.0])
        # unless the sides of the face (temples, ears: the most lateral fifth of the
        # skin) are in front of the eyes: the model then faces backwards (Outerplane 2000002,
        # 4, 6, 7: sides +0.5 to +0.9 interocular distance in front; the other 71 at −0.1 to −1.2)
        lat = np.cross(haut, av)
        u = (peau - e) @ lat
        cotes = peau[np.abs(u) > 0.8 * np.abs(u).max()]
        if ((cotes - e) @ av).mean() > 0:
            av = -av
        # PMX frame: x = left (front view, on the right of the image), y = up, -z = forward
        z = -av
        x = np.cross(haut, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        Q = (P - e) @ R.T
        # interocular distance: centres of the two eyeballs
        Y = Q[self.sommets_mat(self.i_yeux)]
        iod = abs(Y[Y[:, 0] > 0].mean(0)[0] - Y[Y[:, 0] < 0].mean(0)[0])
        self.unite = IOD_JEU / iod
        self.pos = Q * self.unite
        self.rotation = R

    # -------------------------------------------------------------- welded skin
    def _peau(self):
        fb = np.where(np.isin(self.mat_face, self.i_peau))[0]
        Fb = self.faces[fb]
        vb = np.unique(Fb)
        rep, inv = G.souder(self.pos[vb], tol=1e-5 * self.unite)
        self.peau_bruts = vb
        self.b2s = np.full(len(self.pos), -1)
        self.b2s[vb] = inv
        self.V = self.pos[vb[rep]].copy()
        F = self.b2s[Fb]
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F = F[ok]
        # the large connected components (detached islands: tears, blush, duplicates); a ZZZ
        # head (check) keeps everything, like tete.Tete (eyelid islands)
        self.F = F if self.garder_ilots else self._garder_principale(F)
        used = np.unique(self.F)
        remap = np.full(len(self.V), -1)
        remap[used] = np.arange(len(used))
        self.V = self.V[used]
        self.F = remap[self.F]
        self.adj = G.voisins(len(self.V), self.F)
        self.n = len(self.V)

    @staticmethod
    def _garder_principale(F):
        n = int(F.max()) + 1
        par = np.arange(n)

        def trouver(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        for a, b, c in F:
            ra, rb, rc = trouver(a), trouver(b), trouver(c)
            par[rb] = ra
            par[trouver(rc)] = ra
        r = np.array([trouver(i) for i in range(n)])
        cf = r[F[:, 0]]
        vals, cnt = np.unique(cf, return_counts=True)
        # components with more than 5 % of the faces (the skin and, for ZZZ, eyelid islands)
        garder = vals[cnt > 0.05 * len(F)]
        return F[np.isin(cf, garder)]

    # -------------------------------------------------------------- boundary loops
    def _boucles(self):
        bs = [np.array(b) for b in G.boucles_bord(self.F)]
        bs.sort(key=len, reverse=True)
        # the outer boundary is the longest loop by length, not by vertex count: on a
        # sparse mesh (Outerplane, ~1100 vertices), the rim of an eye has more vertices than
        # the outline of the face; taken for the boundary, it let the outline pass for an eye
        # (eye landmarks one interocular distance away from their place, failed registration)
        lg = [np.linalg.norm(self.V[b] - self.V[np.roll(b, -1)], axis=1).sum() for b in bs]
        bs.insert(0, bs.pop(int(np.argmax(lg))))
        self.boucles = bs
        self.bord = np.zeros(self.n, dtype=bool)
        for b in bs:
            self.bord[b] = True
        P = self.pos[self.sommets_mat(self.i_yeux)]
        self.centre_oeil = {}
        for cote, sg in (("g", 1), ("d", -1)):
            self.centre_oeil[cote] = P[P[:, 0] * sg > 0].mean(0)
        self.iod = float(np.linalg.norm(self.centre_oeil["g"] - self.centre_oeil["d"]))
        self.boucle_ext = bs[0]
        self.boucle_oeil = {}
        r = 0.35 * self.iod / IOD_JEU
        for cote in ("g", "d"):
            c = self.centre_oeil[cote]
            cand = [b for b in bs[1:] if np.linalg.norm(self.V[b].mean(0)[:2] - c[:2]) < r]
            if not cand:
                raise ValueError("%s: no eye hole %s" % (self.nom, cote))
            # preferably a hole that encloses the centre of the eyeball (seen from the front): some
            # faces have other holes near the eyes (Outerplane: eyelid slits, accessories)
            autour = [b for b in cand if _dans(self.V[b][:, :2], c[:2])]
            self.boucle_oeil[cote] = max(autour or cand, key=len)
        ym = min(self.centre_oeil["g"][1], self.centre_oeil["d"][1])
        cand = [b for b in bs[1:] if abs(self.V[b].mean(0)[0]) < 0.05 * self.iod / IOD_JEU
                and self.V[b].mean(0)[1] < ym - 0.2 * self.iod / IOD_JEU
                and not any(b is x for x in self.boucle_oeil.values())]
        if not cand:
            raise ValueError("%s: no mouth hole" % self.nom)
        self.boucle_bouche = max(cand, key=len)

    # -------------------------------------------------------------- landmarks (as in tete.Tete)
    def _reperes(self):
        V = self.V
        R = {}
        for cote in ("g", "d"):
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
        self.points_sourcils = self._sourcils()
        self.H = float(V[R["front_haut"], 1] - V[R["menton"], 1])
        haut = float(V[R["front_haut"], 1])
        hv = self.pos[self.sommets_mat(self.i_cheveux)] if self.i_cheveux else np.zeros((0, 3))
        zc = 0.5 * (V[:, 2].min() + V[:, 2].max())
        r = 0.1 * self.iod
        cal = hv[(np.abs(hv[:, 0]) < r) & (np.abs(hv[:, 2] - zc) < r) & (hv[:, 1] > haut - r)] if len(hv) else hv
        self.y_crane = max(haut, float(cal[:, 1].min())) if len(cal) else haut
        self.H_crane = self.y_crane - float(V[R["menton"], 1])

    def _sourcils(self):
        """Eyebrow points in the geometry, otherwise None (eyebrows painted in the texture)."""
        if not self.i_sourcils:
            return None
        S = self.pos[self.sommets_mat(self.i_sourcils)]
        # only the points well above the top of the eye (an eyebrow material sometimes
        # carries other strokes: for Grace, a 眉 vertex below the eyes)
        if self.filtrer_sourcils:
            haut_oeil = max(self.V[self.boucle_oeil[c], 1].max() for c in "gd")
            S = S[S[:, 1] > haut_oeil + 0.12 * self.iod]
        if not len(S) or (S[:, 0] > 0).all() or (S[:, 0] < 0).all():
            return None
        out = {}
        for cote, sg in (("g", 1), ("d", -1)):
            Q = S[S[:, 0] * sg > 0]
            out["sourcil_%s_int" % cote] = Q[np.argmin(np.abs(Q[:, 0]))]
            out["sourcil_%s_ext" % cote] = Q[np.argmax(np.abs(Q[:, 0]))]
            out["sourcil_%s_haut" % cote] = Q[np.argmax(Q[:, 1])]
        return out

    def points_visage(self):
        noms = [k for k in self.reperes if k.startswith(("oeil", "bouche", "nez"))]
        P = {k: self.V[self.reperes[k]] for k in noms}
        P.update(self.points_sourcils or {})
        return P

    def centre_visage(self):
        P = self.points_visage()
        return np.mean([P[k] for k in sorted(P)], axis=0)

    def placer(self, echelle, c_src, c_dst):
        self.echelle, self.c_src, self.c_dst = float(echelle), np.asarray(c_src), np.asarray(c_dst)

    def _hote(self, p):
        return self.echelle * (np.asarray(p) - self.c_src) + self.c_dst

    def resume(self):
        return {"chemin": self.chemin, "peau": [self.noms_mat[i] for i in self.i_peau],
                "yeux": [self.noms_mat[i] for i in self.i_yeux],
                "sourcils": [self.noms_mat[i] for i in self.i_sourcils] if self.points_sourcils else None,
                "sommets_peau": self.n, "faces_peau": len(self.F),
                "boucles": [len(b) for b in self.boucles[:8]], "largeur": round(self.W, 4),
                "hauteur": round(self.H, 4), "hauteur_crane": round(self.H_crane, 4), "iod": round(self.iod, 4),
                "unite_vers_mmd": round(self.unite, 5)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("inspecter")
    p.add_argument("glb", nargs="+")
    a = ap.parse_args(argv)
    for c in a.glb:
        try:
            print(json.dumps(TeteGLB(c).resume(), ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            j, _ = lire_glb(c)
            print(json.dumps({"chemin": c, "erreur": str(e),
                              "materiaux": [m.get("name") for m in j.get("materials", [])]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

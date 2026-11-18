"""Outerplane head (game GLB) -> "ZZZ-style" donor PMX for the face library.

An Outerplane head does not have the materials of a ZZZ head: the whole face is in a single
material M_<id>_FACE (skin of the whole skull, ears included, plus detached parts: eye plates,
highlights, eyelid crease, inside of the mouth, blush hidden under the cheeks, sometimes
earrings), eyelashes and eyebrows are in M_<id>_FACE_EYE (drawn through the hair in game), and
the mouth is closed at rest by a curtain of squashed skin faces between the lips, which the
MOUTH_* morphs stretch. Each connected part is sorted into the ZZZ material of its role
(颜, 目, 目光, 二重, 眉, 睫, 齿, 口舌, 颜2, 髮, 飾り), the mouth curtain is removed (lips that
coincide at rest are moved apart by a hair's breadth so that the welding in tete.Tete leaves
the hole), the morphs are renamed to MMD names and the head is scaled and placed on the eyes of
the pivot head. The whole ZZZ pipeline (tete.Tete, bibliotheque, morphs, hairstyle) then reads
this PMX without knowing anything about Outerplane.

    python3 -I outerplane.py convertir <raw character folder> <output.pmx>
    python3 -I outerplane.py diag <raw folder>...
"""
import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))
sys.path.append(os.path.join(ICI, ".."))
import reglages as R  # noqa: E402
import jeux  # noqa: E402

import geom as G  # noqa: E402
import images as IM  # noqa: E402
import pmx as PMX  # noqa: E402
import pmx_ecrire  # noqa: E402
import tete_glb as TG  # noqa: E402

# what is known about Outerplane is in its description (outils/jeux/outerplane/jeu.json): characters folder,
# heads left out of the spectrum (exclus) and MMD names of its morphs (morphes_mmd)
OP = jeux.charger("outerplane")
BRUT = OP.chemin("personnages")
EXCLUS = OP.valeur("exclus")
# pivot head of the active game (description: tete_base): distance between the eye centres and midpoint of the
# eyes (MMD units). The thresholds of tete.Tete (eye radius 0.35, mouth at |x| < 0.05) are absolute, tuned to
# this scale.
_BASE = jeux.actif().valeur("tete_base")
IOD_PIVOT = _BASE["ecart_yeux"]
CENTRE_PIVOT = np.array(_BASE["centre_yeux"])
MMD = OP.valeur("morphes_mmd")
# single-eye morphs derived from a both-eyes morph: (source, left eye +x, right eye -x);
# ウィンク２ is the smiling wink (arched eyelid) of the ZZZ heads
MOITIES = [("まばたき", "ウィンク", "ウィンク右"), ("笑い", "ウィンク２", "ウィンク２右"),
           ("びっくり", "びっくり左", "びっくり右")]




def normaliser_morphe(nom):
    n = nom.strip().upper().replace("MOUTHE_", "MOUTH_")
    n = re.sub(r"SM[A-Z]*LE|SMAILE|SMIALE", "SMILE", n)
    n = re.sub(r"SU[A-Z]*PR[A-Z]*", "SURPRISE", n)
    return n


def nom_lisible(dossier):
    """2000010_Idith -> Idith; 2000055 -> 2000055."""
    b = os.path.basename(os.path.normpath(dossier))
    m = re.match(r"^(\d+)_(.+)$", b)
    return m.group(2) if m else b


# =================================================================== GLB reading

# role of a primitive from the common name of its material ("materiaux" table of Outerplane's description)
ROLES = {"颜": "visage", "眉睫": "cils", "髮": "cheveux", "体": "corps"}


def _role(nom_mat):
    return ROLES.get(OP.traduire(nom_mat, sinon=None))


def lire(dossier):
    """Useful primitives of the check GLB (glTF scene frame)."""
    j, b = TG.lire_glb(os.path.join(dossier, "controle.glb"))
    mondes = TG._monde(j)
    prims = []
    for i, nd in enumerate(j.get("nodes", [])):
        if "mesh" not in nd:
            continue
        M = mondes[i]
        me = j["meshes"][nd["mesh"]]
        noms_c = me.get("extras", {}).get("targetNames") or []
        for ip, pr in enumerate(me["primitives"]):
            if pr.get("mode", 4) != 4:
                continue
            nm = j["materials"][pr.get("material", 0)].get("name", "")
            role = _role(nm)
            if role is None:
                continue
            I = TG.accesseur(j, b, pr["indices"]).astype(np.int64).reshape(-1, 3)
            used, I = np.unique(I, return_inverse=True)
            I = I.reshape(-1, 3)
            at = pr["attributes"]
            P = TG.accesseur(j, b, at["POSITION"]).astype(np.float64)[used]
            P = P @ M[:3, :3].T + M[:3, 3]
            N = TG.accesseur(j, b, at["NORMAL"]).astype(np.float64)[used] @ M[:3, :3].T if "NORMAL" in at \
                else np.zeros_like(P)
            UV = TG.accesseur(j, b, at["TEXCOORD_0"]).astype(np.float64)[used] if "TEXCOORD_0" in at \
                else np.zeros((len(P), 2))
            cibles = {}
            for k, t in enumerate(pr.get("targets", [])):
                if "POSITION" not in t or k >= len(noms_c):
                    continue
                d = TG.accesseur(j, b, t["POSITION"]).astype(np.float64)[used] @ M[:3, :3].T
                if np.abs(d).max() > 0:
                    cibles[noms_c[k]] = d
            prims.append({"role": role, "materiau": nm, "P": P, "N": N, "UV": UV, "I": I,
                          "cibles": cibles, "maillage": nd["mesh"], "primitive": ip})
    with open(os.path.join(dossier, "rendu.json"), encoding="utf-8") as fh:
        rendu = json.load(fh)["materiaux"]
    for p in prims:
        tx = rendu.get(p["materiau"], {}).get("textures", {})
        f = (tx.get("_MainTex") or tx.get("_Base") or {}).get("fichier")
        p["texture"] = os.path.join(dossier, f) if f else None
    return prims


def composantes(P, I, tol):
    """Connected-component label (vertices welded by position) of each face."""
    _, inv = G.souder(P, tol)
    F = inv[I]
    n = int(inv.max()) + 1
    par = np.arange(n)

    def r(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a, b_, c in F:
        ra = r(a)
        par[r(b_)] = ra
        par[r(c)] = ra
    racines = np.array([r(i) for i in range(n)])
    lab = racines[F[:, 0]]
    _, lab = np.unique(lab, return_inverse=True)
    return lab.ravel()


def orienter(F):
    """Faces of a mesh oriented consistently (the majority direction of each component), by
    propagation across two-face edges. On Beth, a row of flipped faces at the edge of the right
    eye chopped the eye hole into 2-vertex pieces."""
    F = np.array(F)
    u, c, fe = G.aretes(F)
    par_arete = [[] for _ in range(len(u))]
    for f in range(len(F)):
        for e in fe[f]:
            par_arete[e].append(f)
    sens = np.zeros(len(F), dtype=np.int8)  # 0: not seen; 1 keep; -1 flip

    def dir_(f, a, b):
        x = list(F[f])
        i = x.index(a)
        return x[(i + 1) % 3] == b
    for f0 in range(len(F)):
        if sens[f0]:
            continue
        sens[f0] = 1
        comp, pile = [f0], [f0]
        while pile:
            f = pile.pop()
            for e in fe[f]:
                if c[e] != 2:
                    continue
                g = par_arete[e][0] if par_arete[e][1] == f else par_arete[e][1]
                if sens[g]:
                    continue
                a, b = u[e]
                # consistent if the two faces traverse the edge in opposite directions
                memef = dir_(f, a, b) != (sens[f] < 0)
                memeg = dir_(g, a, b)
                sens[g] = -1 if memef == memeg else 1
                comp.append(g)
                pile.append(g)
        if (sens[comp] < 0).sum() * 2 > len(comp):
            sens[comp] *= -1
    out = F.copy()
    out[sens < 0] = out[sens < 0][:, ::-1]
    return out


# =================================================================== PMX frame

class Tete:
    """Outerplane head put in the frame and at the scale of the pivot head, parts sorted."""

    def __init__(self, dossier):
        self.dossier = dossier
        self.nom = nom_lisible(dossier)
        self.prims = lire(dossier)
        vis = [p for p in self.prims if p["role"] == "visage"]
        cils = [p for p in self.prims if p["role"] == "cils"]
        if not vis:
            raise ValueError("%s: no face primitive (FACE)" % self.nom)
        # Kanon, Edelweiss: a second FACE without morphs (face accessory), set aside
        vis.sort(key=lambda p: (len(p["cibles"]) > 0, len(p["I"])), reverse=True)
        for p in vis[1:]:
            p["role"] = "accessoire"
        vis = vis[:1]
        if not cils:
            raise ValueError("%s: no eyelash or eyebrow geometry (no FACE_EYE)" % self.nom)
        self.vis, self.cils = vis[0], cils
        P = self.vis["P"]
        self.tol = 1e-5 * float(np.linalg.norm(np.ptp(P, 0)))
        lab = composantes(P, self.vis["I"], self.tol)
        self.vis["lab"] = lab
        cnt = np.bincount(lab)
        self.lab_peau = int(np.argmax(cnt))
        for c in cils:
            c["lab"] = composantes(c["P"], c["I"], self.tol)
        self._orienter()
        self._ranger()

    # ---------------------------------------------------------------- frame
    def _orienter(self):
        V = self.vis
        fp = V["lab"] == self.lab_peau
        peau = V["P"][np.unique(V["I"][fp])]
        x0 = 0.5 * (peau[:, 0].min() + peau[:, 0].max())
        # eyelashes: the largest FACE_EYE part on each side (the upper lashes, larger than the
        # eyebrow), its centre is above the eye hole
        C = np.vstack([c["P"] for c in self.cils])
        haut = np.array([0.0, 1.0, 0.0])
        rep_cote = {}
        for c in self.cils:
            for l in np.unique(c["lab"]):
                Q = c["P"][np.unique(c["I"][c["lab"] == l])]
                cote = 1 if Q[:, 0].mean() > x0 else -1
                n = int((c["lab"] == l).sum())
                if n > rep_cote.get(cote, (0, None))[0]:
                    rep_cote[cote] = (n, Q.mean(0))
        if len(rep_cote) != 2:
            raise ValueError("%s: eyelashes on one side only" % self.nom)
        # holes of the welded skin: the one nearest the eyelashes on each side is the eye
        Fp = V["I"][fp]
        _, inv = G.souder(V["P"], self.tol)
        Fs = inv[Fp]
        ok = (Fs[:, 0] != Fs[:, 1]) & (Fs[:, 1] != Fs[:, 2]) & (Fs[:, 0] != Fs[:, 2])
        Fs = orienter(Fs[ok])
        rep = np.zeros(int(inv.max()) + 1, dtype=np.int64)
        rep[inv] = np.arange(len(inv))
        Ps = V["P"][rep]
        bs = [np.array(b) for b in G.boucles_bord(Fs)]
        lg = [np.linalg.norm(Ps[b] - Ps[np.roll(b, -1)], axis=1).sum() for b in bs]
        bs.pop(int(np.argmax(lg)))  # the neck
        self.trou = {}
        for cote, (_, c) in rep_cote.items():
            b = min(bs, key=lambda b: np.linalg.norm(Ps[b].mean(0)[:2] - c[:2]))
            self.trou["g" if cote > 0 else "d"] = Ps[b]
        e = 0.5 * (self.trou["g"].mean(0) + self.trou["d"].mean(0))
        # front: +Z (glTF convention), unless the sides of the head are in front of the eyes
        av = np.array([0.0, 0.0, 1.0])
        lat = np.array([1.0, 0.0, 0.0])
        u = (peau - e) @ lat
        cotes = peau[np.abs(u) > 0.8 * np.abs(u).max()]
        if ((cotes - e) @ av).mean() > 0:
            av = -av
        # PMX frame: x = character's left, y = up, -z = front; glTF puts the character's left
        # at +X when it faces +Z: z = -front, x = z ^ up (mirror: det -1)
        z = -av
        x = np.cross(z, haut)
        R = np.stack([x, haut, z])
        iod = float(np.linalg.norm(self.trou["g"].mean(0) - self.trou["d"].mean(0)))
        self.unite = IOD_PIVOT / iod
        self.R, self.e = R, e
        self.avant_glTF = "+Z" if av[2] > 0 else "-Z"
        for k in self.trou:
            self.trou[k] = self.vers_pmx(self.trou[k])
        # side g (character's left) must be at +x
        if self.trou["g"].mean(0)[0] < 0:
            self.trou = {"g": self.trou["d"], "d": self.trou["g"]}
        self.oeil = {k: v.mean(0) for k, v in self.trou.items()}

    def vers_pmx(self, P):
        return (np.asarray(P) - self.e) @ self.R.T * self.unite + CENTRE_PIVOT

    def dir_pmx(self, D):
        return np.asarray(D) @ self.R.T * self.unite

    # ---------------------------------------------------------------- parts
    def _ranger(self):
        """ZZZ material of each face of each primitive (list parallel to the faces)."""
        iod = IOD_PIVOT
        self.pieces = []  # (primitive, faces, material name)
        haut_oeil = max(self.trou[k][:, 1].max() for k in "gd")
        V = self.vis
        Q = self.vers_pmx(V["P"])
        img = IM.lire(V["texture"]) if V["texture"] else None
        self.diag = []
        noms = np.empty(len(V["I"]), dtype=object)
        for l in np.unique(V["lab"]):
            fs = V["lab"] == l
            vs = np.unique(V["I"][fs])
            c = Q[vs].mean(0)
            if l == self.lab_peau:
                noms[fs] = "颜"
                continue
            rel = (c - CENTRE_PIVOT) / iod
            # eye part: centre inside the eye hole (seen from the front, margin 0.05 eye
            # distance) or just above it (eyelid crease), within 0.2 in depth
            k = min("gd", key=lambda k: np.linalg.norm(c[:2] - self.oeil[k][:2]))
            H = G.enveloppe_convexe_2d(self.trou[k][:, :2])
            ds = G.distance_signee_polygone(c[None, :2], H)[0] / iod
            dz = (c[2] - self.oeil[k][2]) / iod  # > 0: behind the hole
            nom = None
            # eyeballs sometimes set back (2000008: 0.4 eye distance behind the hole)
            if ds < 0.05 and -0.2 < dz < 0.6:
                nom = "目光"
            elif ds < 0.15 and abs(dz) < 0.2 and c[1] > haut_oeil:
                nom = "二重"
            elif abs(rel[0]) < 0.45 and -1.6 < rel[1] < -0.4:
                blanc = False
                if img is not None:
                    uv = V["UV"][vs].mean(0)
                    h, w = img.shape[:2]
                    px = img[int(np.clip(uv[1] * h, 0, h - 1)), int(np.clip(uv[0] * w, 0, w - 1))]
                    blanc = bool(px[:3].min() > 0.75)
                nom = "齿" if blanc else "口舌"
            elif abs(rel[0]) < 1.0 and -1.2 < rel[1] < 0.2 and rel[2] * iod > -0.6 * iod:
                nom = "颜2"
            else:
                nom = "飾り"
            noms[fs] = nom
            self.diag.append((nom, int(fs.sum()), np.round(rel, 2)))
        # eye plate: the widest (seen from the front) of the parts of each eye -> 目
        for k in "gd":
            best, aire = None, 0
            for l in np.unique(V["lab"]):
                fs = V["lab"] == l
                if noms[np.argmax(fs)] != "目光":
                    continue
                P2 = Q[np.unique(V["I"][fs])]
                if np.linalg.norm(P2.mean(0)[:2] - self.oeil[k][:2]) > 0.3 * iod:
                    continue
                a = np.ptp(P2[:, 0]) * np.ptp(P2[:, 1])
                if a > aire:
                    best, aire = fs, a
            if best is None:
                # eyeball welded to the skin (Beth, Mero...): the eye hole is closed or fragmented
                raise ValueError("%s: no eye plate %s (eyeball welded to the skin, eye hole closed)"
                                 % (self.nom, k))
            noms[best] = "目"
        V["noms"] = noms
        # eyelashes and eyebrows: above the top of the eye -> eyebrow
        for c in self.cils:
            Qc = self.vers_pmx(c["P"])
            nm = np.empty(len(c["I"]), dtype=object)
            for l in np.unique(c["lab"]):
                fs = c["lab"] == l
                y = Qc[np.unique(c["I"][fs])][:, 1]
                # eyebrows 0.18 to 0.30 eye distance above the eye centre, eyelashes around 0
                nm[fs] = "眉" if np.median(y) > self.oeil["g"][1] + 0.1 * iod else "睫"
                self.diag.append((nm[fs][0], int(fs.sum()), np.round((Qc[np.unique(c["I"][fs])].mean(0) - CENTRE_PIVOT) / iod, 2)))
            c["noms"] = nm
        k = 0
        for p in self.prims:
            if p["role"] == "cheveux":
                p["noms"] = np.full(len(p["I"]), "髮" if k == 0 else "髮%d" % (k + 1), dtype=object)
                k += 1

    # ---------------------------------------------------------------- morphs
    def morphes(self):
        """MMD name -> (panel, PMX offsets (n, 3) per primitive)."""
        out = {}
        for ip, p in enumerate(self.prims):
            for n, D in p["cibles"].items():
                nn = normaliser_morphe(n)
                if np.abs(D).max() <= 0:
                    continue
                nom = MMD.get(nn, nn)
                pan = 2 if nn.startswith("EYE") else 3 if nn.startswith("MOUTH") else 4
                e = out.setdefault(nom, (pan, {}))
                e[1][ip] = self.dir_pmx(D)
        # a single eye (MMD names of the ZZZ heads; no suffix or 左: the left one, +x; 右: the
        # right one): half of a both-eyes morph. Without them, a composite face took the eyelid
        # motion of the ZZZ base, which does not fit the Outerplane eye (frozen eyelashes)
        for src, gauche, droite in MOITIES:
            if src not in out:
                continue
            for nom, sg in ((gauche, 1), (droite, -1)):
                d = {}
                for ip, D in out[src][1].items():
                    x = (self.vers_pmx(self.prims[ip]["P"])[:, 0] - CENTRE_PIVOT[0]) / IOD_PIVOT
                    w = np.clip(0.5 + sg * x / 0.2, 0, 1)
                    d[ip] = D * w[:, None]
                out.setdefault(nom, (2, d))
        return out

    # ---------------------------------------------------------------- mouth
    def ouvrir_bouche(self, morphes):
        """Skin faces to remove and offset of the raw vertices so that the mouth, closed at rest,
        is a hole for tete.Tete (welding by position at 1e-5, degenerate faces removed).

        Two ways of closing the mouth at rest in Outerplane: upper and lower lips coinciding
        (distinct vertices at the same place, which MOUTH_A separates: Idith, Beth), or a
        curtain of tiny faces between them, which MOUTH_A stretches (Francesca). The coinciding
        vertices are moved apart by a hair's breadth (2e-4, in the direction the morph separates
        them) and the degenerate faces and the curtain are removed: small at rest, large in a
        mouth morph; the threshold is widened step by step until a mouth hole appears."""
        V = self.vis
        ip = next(i for i, q in enumerate(self.prims) if q is V)
        Q = self.vers_pmx(V["P"])
        peau = np.where(V["noms"] == "颜")[0]
        F = V["I"][peau]
        bouche = [(n, e[1][ip]) for n, e in morphes.items() if ip in e[1] and e[0] == 3]
        dec = np.zeros_like(Q)
        # 1. coinciding vertices that the mouth morphs separate
        vs = np.unique(F)
        _, inv = G.souder(Q[vs], 1e-5)
        sig = np.hstack([D[vs] for _, D in bouche]) if bouche else np.zeros((len(vs), 0))
        ordre = np.argsort(inv, kind="stable")
        bornes = np.flatnonzero(np.diff(inv[ordre])) + 1
        n_ecartes = 0
        for g in np.split(ordre, bornes):
            if len(g) < 2 or not sig.shape[1] or np.ptp(sig[g], 0).max() < 1e-3:
                continue
            m = sig[g].mean(0)
            for i in g:
                dv = np.zeros(3)
                for k, (_, D) in enumerate(bouche):
                    dv += D[vs[i]] - D[vs[g]].mean(0)
                nv = np.linalg.norm(dv)
                if nv > 1e-4:
                    dec[vs[i]] = 2e-4 * dv / nv
                    n_ecartes += 1
            del m
        Q2 = Q + dec
        # 2. faces to remove
        def aire(X):
            return 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
        a0 = aire(Q)
        med = float(np.median(a0))
        am = np.zeros_like(a0)
        for _, D in bouche:
            am = np.maximum(am, aire(Q + D))
        c = (Q[F].mean(1) - CENTRE_PIVOT) / IOD_PIVOT
        zone = (np.abs(c[:, 0]) < 0.5) & (c[:, 1] < -0.3) & (c[:, 1] > -1.2)
        oter = np.zeros(len(F), dtype=bool)
        for s0, s1 in ((0.0, 9.0), (0.1, 0.3), (0.2, 0.2), (0.3, 0.15), (0.5, 0.1)):
            oter = zone & (((a0 <= 1e-6 * med)) | ((a0 < s0 * med) & (am > s1 * med)))
            if self._trou_bouche(Q2, F[~oter]):
                self.bouche = {"sommets_ecartes": n_ecartes, "faces_otees": int(oter.sum()),
                               "seuils": [s0, s1]}
                return peau[oter], dec
        raise ValueError("%s: mouth closed (no mouth hole even after removing the curtain)" % self.nom)

    @staticmethod
    def _trou_bouche(Q, F):
        """Same test as tete.Tete: a boundary loop on the midline, below the eyes."""
        vs = np.unique(F)
        rep, inv = G.souder(Q[vs], 1e-5)
        b2s = np.full(len(Q), -1)
        b2s[vs] = inv
        Fs = b2s[F]
        ok = (Fs[:, 0] != Fs[:, 1]) & (Fs[:, 1] != Fs[:, 2]) & (Fs[:, 0] != Fs[:, 2])
        V = Q[vs[rep]]
        bs = [np.array(b) for b in G.boucles_bord(Fs[ok])]
        if not bs:
            return False
        lg = [np.linalg.norm(V[b] - V[np.roll(b, -1)], axis=1).sum() for b in bs]
        bs.pop(int(np.argmax(lg)))  # the outer boundary (neck), as in tete.Tete
        for b in bs:
            if len(b) < 4:
                continue
            c = V[b].mean(0)
            # in front (the neck hole, Kanon, is on the midline below the eyes, but behind)
            if abs(c[0] - CENTRE_PIVOT[0]) < 0.05 and c[1] < CENTRE_PIVOT[1] - 0.2 and \
                    c[2] < CENTRE_PIVOT[2] + 0.3 * IOD_PIVOT and np.ptp(V[b][:, 0]) > 0.15 * IOD_PIVOT:
                return True
        return False

    # ---------------------------------------------------------------- PMX
    def pmx(self, dossier_textures):
        """PMX dictionary (pmx_ecrire); textures copied into dossier_textures."""
        morphes = self.morphes()
        oter, dec = self.ouvrir_bouche(morphes)
        prims = [p for p in self.prims if "noms" in p]
        base, pos, nor, uv = {}, [], [], []
        n = 0
        for p in prims:
            base[id(p)] = n
            d = dec if p is self.vis else 0
            pos.append(self.vers_pmx(p["P"]) + d)
            N = self.dir_pmx(p["N"])
            nor.append(N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12))
            uv.append(p["UV"])
            n += len(p["P"])
        pos, nor, uv = np.vstack(pos), np.vstack(nor), np.vstack(uv)
        # faces per material; the vertex order is reversed (glTF -> PMX: mirror)
        mats = {}
        for p in prims:
            garde = np.ones(len(p["I"]), dtype=bool)
            if p is self.vis:
                garde[oter] = False
            I = p["I"][:, ::-1] + base[id(p)]
            noms = p["noms"]
            for nom in dict.fromkeys(noms):
                m = garde & (noms == nom)
                if not m.any():
                    continue
                cle = (nom, p["texture"])
                mats.setdefault(cle, []).append(I[m])
        ordre = ["颜", "颜2", "目", "目光", "二重", "睫", "眉", "齿", "口舌"]
        cles = sorted(mats, key=lambda k: (ordre.index(k[0]) if k[0] in ordre else len(ordre), k[0]))
        os.makedirs(dossier_textures, exist_ok=True)
        textures, faces, materiaux = [], [], []
        for cle in cles:
            nom, tex = cle
            ti = -1
            if tex:
                b = os.path.basename(tex)
                if b not in textures:
                    dst = os.path.join(dossier_textures, b)
                    if not os.path.exists(dst):
                        shutil.copyfile(tex, dst + ".tmp")
                        os.replace(dst + ".tmp", dst)
                    textures.append(b)
                ti = textures.index(b)
            I = np.vstack(mats[cle])
            faces.extend(I.tolist())
            peau_ou_cheveux = nom == "颜" or "髮" in nom
            nm = nom
            k = 2
            while nm in [m["nom"] for m in materiaux]:
                nm = "%s%d" % (nom, k)
                k += 1
            materiaux.append({
                "nom": nm, "nom_en": nm, "diffus": (1.0, 1.0, 1.0, 1.0), "speculaire": (0.0, 0.0, 0.0),
                "brillance": 5.0, "ambiant": (0.5, 0.5, 0.5),
                # double-sided except skin and hair (opaque, outline); cast shadows for all
                "drapeaux": 0x1E if peau_ou_cheveux else 0x0F,
                "contour_couleur": (0.0, 0.0, 0.0, 1.0), "contour_taille": 0.6 if peau_ou_cheveux else 0.0,
                "texture": ti, "sphere": -1, "sphere_mode": 0, "toon_partage": 1, "toon": 0,
                "memo": "Outerplane %s" % os.path.basename(tex or ""), "n_indices": 3 * len(I)})
        # bones: 全ての親 -> 首 -> 頭; everything is weighted to 頭 (no physics: rigid hair)
        vis = pos[base[id(self.vis)]:base[id(self.vis)] + len(self.vis["P"])]
        y_bas = float(vis[:, 1].min())
        y_cou = CENTRE_PIVOT[1] - 1.3 * IOD_PIVOT
        z_c = float(np.median(vis[:, 2]))
        fl = PMX.OS_ROTATION | PMX.OS_VISIBLE | PMX.OS_ACTIF
        def os_(nom, en, p, parent, dep=False):
            return {"nom": nom, "nom_en": en, "pos": tuple(float(x) for x in p), "parent": parent, "couche": 0,
                    "drapeaux": fl | (PMX.OS_DEPLACEMENT if dep else 0), "queue_os": None,
                    "queue_dec": (0.0, 1.0, 0.0), "ajout": None, "axe_fixe": None, "axes_locaux": None,
                    "parent_externe": None, "ik": None, "apres_physique": False}
        oss = [os_("全ての親", "master", (0, 0, 0), -1, True),
               os_("首", "neck", (0, min(y_bas, y_cou), z_c), 0),
               os_("頭", "head", (0, y_cou, z_c), 1)]
        nmor = []
        for nom, (pan, par) in morphes.items():
            dl = []
            for p in prims:
                ip = next(i for i, q in enumerate(self.prims) if q is p)
                if ip not in par:
                    continue
                D = par[ip]
                for i in np.flatnonzero(np.abs(D).max(1) > 1e-6):
                    dl.append((int(i + base[id(p)]), tuple(float(x) for x in D[i])))
            if dl:
                nmor.append({"nom": nom, "nom_en": nom, "panneau": pan, "type": 1, "decalages": dl})
        cadres = [{"nom": "Root", "nom_en": "Root", "special": 1, "elements": [(0, 0)]},
                  {"nom": "表情", "nom_en": "Exp", "special": 1, "elements": [(1, i) for i in range(len(nmor))]},
                  {"nom": "頭", "nom_en": "head", "special": 0, "elements": [(0, 1), (0, 2)]}]
        n = len(pos)
        return {"version": 2.0, "encodage": "UTF-16", "n_uv_add": 0,
                "nom": self.nom, "nom_en": self.nom,
                "commentaire": "Outerplane head %s converted by outerplane.py (face library)" %
                               os.path.basename(os.path.normpath(self.dossier)), "commentaire_en": "",
                "sommets": {"pos": [tuple(x) for x in pos], "nor": [tuple(x) for x in nor],
                            "uv": [tuple(x) for x in uv], "uv_add": [[] for _ in range(n)],
                            "type": [0] * n, "os": [(2, -1, -1, -1)] * n, "poids": [(1.0, 0.0, 0.0, 0.0)] * n,
                            "sdef": [None] * n, "contour": [1.0] * n},
                "faces": faces, "textures": textures, "materiaux": materiaux, "os": oss,
                "morphes": nmor, "cadres": cadres, "rigides": [], "joints": []}

    def resume(self):
        return {"nom": self.nom, "avant_glTF": self.avant_glTF, "unite": round(self.unite, 3),
                "oeil": {k: np.round(v - CENTRE_PIVOT, 3).tolist() for k, v in self.oeil.items()},
                "pieces": [(n, f, r.tolist()) for n, f, r in self.diag]}


def candidats(noms_pris=()):
    """Heads to convert: [(name, folder)], and the excluded ones {folder: reason}.

    Readable name of the folder (2000010_Idith -> Idith), otherwise the identifier; "OP-" in
    front of a name already taken (noms_pris: the ZZZ characters); two folders with the same
    readable name (2000068_Dahlia, 2000090_Dahlia) each get "-<identifier>"."""
    garde, ecartes = [], {}
    for d in sorted(os.listdir(BRUT)):
        if not os.path.exists(os.path.join(BRUT, d, "controle.glb")):
            continue
        if d in EXCLUS:
            ecartes[d] = EXCLUS[d]
            continue
        garde.append(d)
    nl = {d: nom_lisible(d) for d in garde}
    compte = {}
    for n in nl.values():
        compte[n] = compte.get(n, 0) + 1
    pris = set(noms_pris)
    out = []
    for d in garde:
        n = nl[d]
        if compte[n] > 1:
            n = "%s-%s" % (n, d.split("_")[0])
        if n in pris:
            n = "OP-" + n
        out.append((n, os.path.join(BRUT, d)))
    return out, ecartes


def convertir(dossier, chemin):
    """Writes the donor PMX (textures next to it); returns the converted head."""
    T = Tete(dossier)
    P = T.pmx(os.path.dirname(os.path.abspath(chemin)))
    tmp = chemin + ".tmp.pmx"
    pmx_ecrire.ecrire(P, tmp)
    os.replace(tmp, chemin)
    T.materiaux = [m["nom"] for m in P["materiaux"]]
    T.noms_morphes = [m["nom"] for m in P["morphes"]]
    return T


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="action", required=True)
    d = sp.add_parser("diag")
    d.add_argument("dossiers", nargs="+")
    c = sp.add_parser("convertir")
    c.add_argument("dossier")
    c.add_argument("sortie", help="path of the PMX; the textures are copied next to it")
    a = ap.parse_args(argv)
    if a.action == "convertir":
        T = convertir(a.dossier, a.sortie)
        print(T.nom, "->", a.sortie, T.materiaux, "mouth", T.bouche, "morphs", T.noms_morphes)
        return 0
    if a.action == "diag":
        for x in a.dossiers:
            T = Tete(x)
            r = T.resume()
            print(r["nom"], r["avant_glTF"], r["unite"], r["oeil"])
            for p in r["pieces"]:
                print("   ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

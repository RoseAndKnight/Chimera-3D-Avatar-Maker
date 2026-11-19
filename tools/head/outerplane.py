"""Outerplane head (game GLB) -> "ZZZ-style" donor PMX for the face library.

An Outerplane head does not have the materials of a ZZZ head: the whole face is in a single
material M_<id>_FACE (skin of the whole skull, ears included, plus detached parts: eye plates,
highlights, eyelid crease, inside of the mouth, blush hidden under the cheeks, sometimes
earrings), eyelashes and eyebrows are in M_<id>_FACE_EYE (drawn through the hair in game), and
the mouth is closed at rest by a curtain of squashed skin faces between the lips, which the
MOUTH_* morphs stretch. Each connected part is sorted into the ZZZ material of its role
(颜, 目, 目光, 二重, 眉, 睫, 齿, 口舌, 颜2, 髮, 飾り), the mouth curtain is removed (lips that
coincide at rest are moved apart by a hair's breadth so that the welding in head.Head leaves
the hole), the morphs are renamed to MMD names and the head is scaled and placed on the eyes of
the pivot head. The whole ZZZ pipeline (head.Head, library, morphs, hairstyle) then reads
this PMX without knowing anything about Outerplane.

    python3 -I outerplane.py convert <raw character folder> <output.pmx>
    python3 -I outerplane.py diag <raw folder>...
"""
import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))
sys.path.append(os.path.join(HERE, ".."))
import settings as R  # noqa: E402
import games as games  # noqa: E402

import geom as G  # noqa: E402
import images as IM  # noqa: E402
import pmx as PMX  # noqa: E402
import pmx_write as pmx_write  # noqa: E402
import head_glb as TG  # noqa: E402

# what is known about Outerplane is in its description (tools/games/outerplane/game.json): characters folder,
# heads left out of the spectrum (excluded) and MMD names of its morphs (morphs_mmd)
OP = games.load("outerplane")
RAW = OP.path("people")
EXCLUDED = OP.value("excluded")
# pivot head of the active game (description: base_head): distance between the eye centres and midpoint of the
# eyes (MMD units). The thresholds of head.Head (eye radius 0.35, mouth at |x| < 0.05) are absolute, tuned to
# this scale.
_BASE = games.active().value("base_head")
IOD_PIVOT = _BASE["eye_gap"]
CENTER_PIVOT = np.array(_BASE["eyes_center"])
MMD = OP.value("morphs_mmd")
# single-eye morphs derived from a both-eyes morph: (source, left eye +x, right eye -x);
# ウィンク２ is the smiling wink (arched eyelid) of the ZZZ heads
HALVES = [("まばたき", "ウィンク", "ウィンク右"), ("笑い", "ウィンク２", "ウィンク２右"),
           ("びっくり", "びっくり左", "びっくり右")]




def normalize_morph(name):
    n = name.strip().upper().replace("MOUTHE_", "MOUTH_")
    n = re.sub(r"SM[A-Z]*LE|SMAILE|SMIALE", "SMILE", n)   # fr-ok: typos of SMILE in the game files
    n = re.sub(r"SU[A-Z]*PR[A-Z]*", "SURPRISE", n)
    return n


def name_readable(folder):
    """2000010_Idith -> Idith; 2000055 -> 2000055."""
    b = os.path.basename(os.path.normpath(folder))
    m = re.match(r"^(\d+)_(.+)$", b)
    return m.group(2) if m else b


# =================================================================== GLB reading

# role of a primitive from the common name of its material ("materials" table of Outerplane's description)
ROLES = {"颜": "face", "眉睫": "lashes", "髮": "hair", "体": "body"}


def _role(name_mat):
    return ROLES.get(OP.translate(name_mat, otherwise=None))


def read(folder):
    """Useful primitives of the check GLB (glTF scene frame)."""
    j, b = TG.read_glb(os.path.join(folder, "controle.glb"))   # fr-ok: file name set by the third-party extraction
    worlds = TG._world(j)
    prims = []
    for i, nd in enumerate(j.get("nodes", [])):
        if "mesh" not in nd:
            continue
        M = worlds[i]
        me = j["meshes"][nd["mesh"]]
        names_c = me.get("extras", {}).get("targetNames") or []
        for ip, pr in enumerate(me["primitives"]):
            if pr.get("mode", 4) != 4:
                continue
            nm = j["materials"][pr.get("material", 0)].get("name", "")
            role = _role(nm)
            if role is None:
                continue
            I = TG.accessor(j, b, pr["indices"]).astype(np.int64).reshape(-1, 3)
            used, I = np.unique(I, return_inverse=True)
            I = I.reshape(-1, 3)
            at = pr["attributes"]
            P = TG.accessor(j, b, at["POSITION"]).astype(np.float64)[used]
            P = P @ M[:3, :3].T + M[:3, 3]
            N = TG.accessor(j, b, at["NORMAL"]).astype(np.float64)[used] @ M[:3, :3].T if "NORMAL" in at \
                else np.zeros_like(P)
            UV = TG.accessor(j, b, at["TEXCOORD_0"]).astype(np.float64)[used] if "TEXCOORD_0" in at \
                else np.zeros((len(P), 2))
            targets = {}
            for k, t in enumerate(pr.get("targets", [])):
                if "POSITION" not in t or k >= len(names_c):
                    continue
                d = TG.accessor(j, b, t["POSITION"]).astype(np.float64)[used] @ M[:3, :3].T
                if np.abs(d).max() > 0:
                    targets[names_c[k]] = d
            prims.append({"role": role, "material": nm, "P": P, "N": N, "UV": UV, "I": I,
                          "targets": targets, "mesh": nd["mesh"], "primitive": ip})
    with open(os.path.join(folder, "rendu.json"), encoding="utf-8") as fh:   # fr-ok: file name set by the third-party extraction
        rendered = json.load(fh)["materials"]
    for p in prims:
        tx = rendered.get(p["material"], {}).get("textures", {})
        f = (tx.get("_MainTex") or tx.get("_Base") or {}).get("file")
        p["texture"] = os.path.join(folder, f) if f else None
    return prims


def components(P, I, tol):
    """Connected-component label (vertices welded by position) of each face."""
    _, inv = G.weld(P, tol)
    F = inv[I]
    n = int(inv.max()) + 1
    per = np.arange(n)

    def r(a):
        while per[a] != a:
            per[a] = per[per[a]]
            a = per[a]
        return a
    for a, b_, c in F:
        ra = r(a)
        per[r(b_)] = ra
        per[r(c)] = ra
    roots = np.array([r(i) for i in range(n)])
    lab = roots[F[:, 0]]
    _, lab = np.unique(lab, return_inverse=True)
    return lab.ravel()


def orient(F):
    """Faces of a mesh oriented consistently (the majority direction of each component), by
    propagation across two-face edges. On Beth, a row of flipped faces at the edge of the right
    eye chopped the eye hole into 2-vertex pieces."""
    F = np.array(F)
    u, c, fe = G.edges(F)
    per_edge = [[] for _ in range(len(u))]
    for f in range(len(F)):
        for e in fe[f]:
            per_edge[e].append(f)
    direction = np.zeros(len(F), dtype=np.int8)  # 0: not seen; 1 keep; -1 flip

    def dir_(f, a, b):
        x = list(F[f])
        i = x.index(a)
        return x[(i + 1) % 3] == b
    for f0 in range(len(F)):
        if direction[f0]:
            continue
        direction[f0] = 1
        comp, pile = [f0], [f0]
        while pile:
            f = pile.pop()
            for e in fe[f]:
                if c[e] != 2:
                    continue
                g = per_edge[e][0] if per_edge[e][1] == f else per_edge[e][1]
                if direction[g]:
                    continue
                a, b = u[e]
                # consistent if the two faces traverse the edge in opposite directions
                memef = dir_(f, a, b) != (direction[f] < 0)
                memeg = dir_(g, a, b)
                direction[g] = -1 if memef == memeg else 1
                comp.append(g)
                pile.append(g)
        if (direction[comp] < 0).sum() * 2 > len(comp):
            direction[comp] *= -1
    out = F.copy()
    out[direction < 0] = out[direction < 0][:, ::-1]
    return out


# =================================================================== PMX frame

class Head:
    """Outerplane head put in the frame and at the scale of the pivot head, parts sorted."""

    def __init__(self, folder):
        self.folder = folder
        self.name = name_readable(folder)
        self.prims = read(folder)
        vis = [p for p in self.prims if p["role"] == "face"]
        lashes = [p for p in self.prims if p["role"] == "lashes"]
        if not vis:
            raise ValueError("%s: no face primitive (FACE)" % self.name)
        # Kanon, Edelweiss: a second FACE without morphs (face accessory), set aside
        vis.sort(key=lambda p: (len(p["targets"]) > 0, len(p["I"])), reverse=True)
        for p in vis[1:]:
            p["role"] = "accessory"
        vis = vis[:1]
        if not lashes:
            raise ValueError("%s: no eyelash or eyebrow geometry (no FACE_EYE)" % self.name)
        self.vis, self.lashes = vis[0], lashes
        P = self.vis["P"]
        self.tol = 1e-5 * float(np.linalg.norm(np.ptp(P, 0)))
        lab = components(P, self.vis["I"], self.tol)
        self.vis["lab"] = lab
        cnt = np.bincount(lab)
        self.lab_skin = int(np.argmax(cnt))
        for c in lashes:
            c["lab"] = components(c["P"], c["I"], self.tol)
        self._orient()
        self._arrange()

    # ---------------------------------------------------------------- frame
    def _orient(self):
        V = self.vis
        fp = V["lab"] == self.lab_skin
        skin = V["P"][np.unique(V["I"][fp])]
        x0 = 0.5 * (skin[:, 0].min() + skin[:, 0].max())
        # eyelashes: the largest FACE_EYE part on each side (the upper lashes, larger than the
        # eyebrow), its centre is above the eye hole
        C = np.vstack([c["P"] for c in self.lashes])
        top = np.array([0.0, 1.0, 0.0])
        rep_side = {}
        for c in self.lashes:
            for l in np.unique(c["lab"]):
                Q = c["P"][np.unique(c["I"][c["lab"] == l])]
                side = 1 if Q[:, 0].mean() > x0 else -1
                n = int((c["lab"] == l).sum())
                if n > rep_side.get(side, (0, None))[0]:
                    rep_side[side] = (n, Q.mean(0))
        if len(rep_side) != 2:
            raise ValueError("%s: eyelashes on one side only" % self.name)
        # holes of the welded skin: the one nearest the eyelashes on each side is the eye
        Fp = V["I"][fp]
        _, inv = G.weld(V["P"], self.tol)
        Fs = inv[Fp]
        ok = (Fs[:, 0] != Fs[:, 1]) & (Fs[:, 1] != Fs[:, 2]) & (Fs[:, 0] != Fs[:, 2])
        Fs = orient(Fs[ok])
        rep = np.zeros(int(inv.max()) + 1, dtype=np.int64)
        rep[inv] = np.arange(len(inv))
        Ps = V["P"][rep]
        bs = [np.array(b) for b in G.border_loops(Fs)]
        lg = [np.linalg.norm(Ps[b] - Ps[np.roll(b, -1)], axis=1).sum() for b in bs]
        bs.pop(int(np.argmax(lg)))  # the neck
        self.hole = {}
        for side, (_, c) in rep_side.items():
            b = min(bs, key=lambda b: np.linalg.norm(Ps[b].mean(0)[:2] - c[:2]))
            self.hole["g" if side > 0 else "d"] = Ps[b]
        e = 0.5 * (self.hole["g"].mean(0) + self.hole["d"].mean(0))
        # front: +Z (glTF convention), unless the sides of the head are in front of the eyes
        av = np.array([0.0, 0.0, 1.0])
        lat = np.array([1.0, 0.0, 0.0])
        u = (skin - e) @ lat
        sides = skin[np.abs(u) > 0.8 * np.abs(u).max()]
        if ((sides - e) @ av).mean() > 0:
            av = -av
        # PMX frame: x = character's left, y = up, -z = front; glTF puts the character's left
        # at +X when it faces +Z: z = -front, x = z ^ up (mirror: det -1)
        z = -av
        x = np.cross(z, top)
        R = np.stack([x, top, z])
        iod = float(np.linalg.norm(self.hole["g"].mean(0) - self.hole["d"].mean(0)))
        self.unit = IOD_PIVOT / iod
        self.R, self.e = R, e
        self.front_glTF = "+Z" if av[2] > 0 else "-Z"
        for k in self.hole:
            self.hole[k] = self.to_pmx(self.hole[k])
        # side g (character's left) must be at +x
        if self.hole["g"].mean(0)[0] < 0:
            self.hole = {"g": self.hole["d"], "d": self.hole["g"]}
        self.eye = {k: v.mean(0) for k, v in self.hole.items()}

    def to_pmx(self, P):
        return (np.asarray(P) - self.e) @ self.R.T * self.unit + CENTER_PIVOT

    def dir_pmx(self, D):
        return np.asarray(D) @ self.R.T * self.unit

    # ---------------------------------------------------------------- parts
    def _arrange(self):
        """ZZZ material of each face of each primitive (list parallel to the faces)."""
        iod = IOD_PIVOT
        self.parts = []  # (primitive, faces, material name)
        top_eye = max(self.hole[k][:, 1].max() for k in "gd")
        V = self.vis
        Q = self.to_pmx(V["P"])
        img = IM.read(V["texture"]) if V["texture"] else None
        self.diag = []
        names = np.empty(len(V["I"]), dtype=object)
        for l in np.unique(V["lab"]):
            fs = V["lab"] == l
            vs = np.unique(V["I"][fs])
            c = Q[vs].mean(0)
            if l == self.lab_skin:
                names[fs] = "颜"
                continue
            rel = (c - CENTER_PIVOT) / iod
            # eye part: centre inside the eye hole (seen from the front, margin 0.05 eye
            # distance) or just above it (eyelid crease), within 0.2 in depth
            k = min("gd", key=lambda k: np.linalg.norm(c[:2] - self.eye[k][:2]))
            H = G.hull_convex_2d(self.hole[k][:, :2])
            ds = G.distance_signed_polygon(c[None, :2], H)[0] / iod
            dz = (c[2] - self.eye[k][2]) / iod  # > 0: behind the hole
            name = None
            # eyeballs sometimes set back (2000008: 0.4 eye distance behind the hole)
            if ds < 0.05 and -0.2 < dz < 0.6:
                name = "目光"
            elif ds < 0.15 and abs(dz) < 0.2 and c[1] > top_eye:
                name = "二重"
            elif abs(rel[0]) < 0.45 and -1.6 < rel[1] < -0.4:
                white = False
                if img is not None:
                    uv = V["UV"][vs].mean(0)
                    h, w = img.shape[:2]
                    px = img[int(np.clip(uv[1] * h, 0, h - 1)), int(np.clip(uv[0] * w, 0, w - 1))]
                    white = bool(px[:3].min() > 0.75)
                name = "齿" if white else "口舌"
            elif abs(rel[0]) < 1.0 and -1.2 < rel[1] < 0.2 and rel[2] * iod > -0.6 * iod:
                name = "颜2"
            else:
                name = "飾り"
            names[fs] = name
            self.diag.append((name, int(fs.sum()), np.round(rel, 2)))
        # eye plate: the widest (seen from the front) of the parts of each eye -> 目
        for k in "gd":
            best, area = None, 0
            for l in np.unique(V["lab"]):
                fs = V["lab"] == l
                if names[np.argmax(fs)] != "目光":
                    continue
                P2 = Q[np.unique(V["I"][fs])]
                if np.linalg.norm(P2.mean(0)[:2] - self.eye[k][:2]) > 0.3 * iod:
                    continue
                a = np.ptp(P2[:, 0]) * np.ptp(P2[:, 1])
                if a > area:
                    best, area = fs, a
            if best is None:
                # eyeball welded to the skin (Beth, Mero...): the eye hole is closed or fragmented
                raise ValueError("%s: no eye plate %s (eyeball welded to the skin, eye hole closed)"
                                 % (self.name, k))
            names[best] = "目"
        V["names"] = names
        # eyelashes and eyebrows: above the top of the eye -> eyebrow
        for c in self.lashes:
            Qc = self.to_pmx(c["P"])
            nm = np.empty(len(c["I"]), dtype=object)
            for l in np.unique(c["lab"]):
                fs = c["lab"] == l
                y = Qc[np.unique(c["I"][fs])][:, 1]
                # eyebrows 0.18 to 0.30 eye distance above the eye centre, eyelashes around 0
                nm[fs] = "眉" if np.median(y) > self.eye["g"][1] + 0.1 * iod else "睫"
                self.diag.append((nm[fs][0], int(fs.sum()), np.round((Qc[np.unique(c["I"][fs])].mean(0) - CENTER_PIVOT) / iod, 2)))
            c["names"] = nm
        k = 0
        for p in self.prims:
            if p["role"] == "hair":
                p["names"] = np.full(len(p["I"]), "髮" if k == 0 else "髮%d" % (k + 1), dtype=object)
                k += 1

    # ---------------------------------------------------------------- morphs
    def morphs(self):
        """MMD name -> (panel, PMX offsets (n, 3) per primitive)."""
        out = {}
        for ip, p in enumerate(self.prims):
            for n, D in p["targets"].items():
                nn = normalize_morph(n)
                if np.abs(D).max() <= 0:
                    continue
                name = MMD.get(nn, nn)
                pan = 2 if nn.startswith("EYE") else 3 if nn.startswith("MOUTH") else 4
                e = out.setdefault(name, (pan, {}))
                e[1][ip] = self.dir_pmx(D)
        # a single eye (MMD names of the ZZZ heads; no suffix or 左: the left one, +x; 右: the
        # right one): half of a both-eyes morph. Without them, a composite face took the eyelid
        # motion of the ZZZ base, which does not fit the Outerplane eye (frozen eyelashes)
        for src, left, right in HALVES:
            if src not in out:
                continue
            for name, sg in ((left, 1), (right, -1)):
                d = {}
                for ip, D in out[src][1].items():
                    x = (self.to_pmx(self.prims[ip]["P"])[:, 0] - CENTER_PIVOT[0]) / IOD_PIVOT
                    w = np.clip(0.5 + sg * x / 0.2, 0, 1)
                    d[ip] = D * w[:, None]
                out.setdefault(name, (2, d))
        return out

    # ---------------------------------------------------------------- mouth
    def open_mouth(self, morphs):
        """Skin faces to remove and offset of the raw vertices so that the mouth, closed at rest,
        is a hole for head.Head (welding by position at 1e-5, degenerate faces removed).

        Two ways of closing the mouth at rest in Outerplane: upper and lower lips coinciding
        (distinct vertices at the same place, which MOUTH_A separates: Idith, Beth), or a
        curtain of tiny faces between them, which MOUTH_A stretches (Francesca). The coinciding
        vertices are moved apart by a hair's breadth (2e-4, in the direction the morph separates
        them) and the degenerate faces and the curtain are removed: small at rest, large in a
        mouth morph; the threshold is widened step by step until a mouth hole appears."""
        V = self.vis
        ip = next(i for i, q in enumerate(self.prims) if q is V)
        Q = self.to_pmx(V["P"])
        skin = np.where(V["names"] == "颜")[0]
        F = V["I"][skin]
        mouth = [(n, e[1][ip]) for n, e in morphs.items() if ip in e[1] and e[0] == 3]
        offset = np.zeros_like(Q)
        # 1. coinciding vertices that the mouth morphs separate
        vs = np.unique(F)
        _, inv = G.weld(Q[vs], 1e-5)
        sig = np.hstack([D[vs] for _, D in mouth]) if mouth else np.zeros((len(vs), 0))
        order = np.argsort(inv, kind="stable")
        bounds = np.flatnonzero(np.diff(inv[order])) + 1
        n_discarded = 0
        for g in np.split(order, bounds):
            if len(g) < 2 or not sig.shape[1] or np.ptp(sig[g], 0).max() < 1e-3:
                continue
            m = sig[g].mean(0)
            for i in g:
                dv = np.zeros(3)
                for k, (_, D) in enumerate(mouth):
                    dv += D[vs[i]] - D[vs[g]].mean(0)
                nv = np.linalg.norm(dv)
                if nv > 1e-4:
                    offset[vs[i]] = 2e-4 * dv / nv
                    n_discarded += 1
            del m
        Q2 = Q + offset
        # 2. faces to remove
        def area(X):
            return 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
        a0 = area(Q)
        med = float(np.median(a0))
        am = np.zeros_like(a0)
        for _, D in mouth:
            am = np.maximum(am, area(Q + D))
        c = (Q[F].mean(1) - CENTER_PIVOT) / IOD_PIVOT
        zone = (np.abs(c[:, 0]) < 0.5) & (c[:, 1] < -0.3) & (c[:, 1] > -1.2)
        remove = np.zeros(len(F), dtype=bool)
        for s0, s1 in ((0.0, 9.0), (0.1, 0.3), (0.2, 0.2), (0.3, 0.15), (0.5, 0.1)):
            remove = zone & (((a0 <= 1e-6 * med)) | ((a0 < s0 * med) & (am > s1 * med)))
            if self._hole_mouth(Q2, F[~remove]):
                self.mouth = {"vertices_discarded": n_discarded, "faces_otees": int(remove.sum()),
                               "thresholds": [s0, s1]}
                return skin[remove], offset
        raise ValueError("%s: mouth closed (no mouth hole even after removing the curtain)" % self.name)

    @staticmethod
    def _hole_mouth(Q, F):
        """Same test as head.Head: a boundary loop on the midline, below the eyes."""
        vs = np.unique(F)
        rep, inv = G.weld(Q[vs], 1e-5)
        b2s = np.full(len(Q), -1)
        b2s[vs] = inv
        Fs = b2s[F]
        ok = (Fs[:, 0] != Fs[:, 1]) & (Fs[:, 1] != Fs[:, 2]) & (Fs[:, 0] != Fs[:, 2])
        V = Q[vs[rep]]
        bs = [np.array(b) for b in G.border_loops(Fs[ok])]
        if not bs:
            return False
        lg = [np.linalg.norm(V[b] - V[np.roll(b, -1)], axis=1).sum() for b in bs]
        bs.pop(int(np.argmax(lg)))  # the outer boundary (neck), as in head.Head
        for b in bs:
            if len(b) < 4:
                continue
            c = V[b].mean(0)
            # in front (the neck hole, Kanon, is on the midline below the eyes, but behind)
            if abs(c[0] - CENTER_PIVOT[0]) < 0.05 and c[1] < CENTER_PIVOT[1] - 0.2 and \
                    c[2] < CENTER_PIVOT[2] + 0.3 * IOD_PIVOT and np.ptp(V[b][:, 0]) > 0.15 * IOD_PIVOT:
                return True
        return False

    # ---------------------------------------------------------------- PMX
    def pmx(self, folder_textures):
        """PMX dictionary (pmx_write); textures copied into dossier_textures."""
        morphs = self.morphs()
        remove, offset = self.open_mouth(morphs)
        prims = [p for p in self.prims if "names" in p]
        base, pos, nor, uv = {}, [], [], []
        n = 0
        for p in prims:
            base[id(p)] = n
            d = offset if p is self.vis else 0
            pos.append(self.to_pmx(p["P"]) + d)
            N = self.dir_pmx(p["N"])
            nor.append(N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12))
            uv.append(p["UV"])
            n += len(p["P"])
        pos, nor, uv = np.vstack(pos), np.vstack(nor), np.vstack(uv)
        # faces per material; the vertex order is reversed (glTF -> PMX: mirror)
        mats = {}
        for p in prims:
            guard = np.ones(len(p["I"]), dtype=bool)
            if p is self.vis:
                guard[remove] = False
            I = p["I"][:, ::-1] + base[id(p)]
            names = p["names"]
            for name in dict.fromkeys(names):
                m = guard & (names == name)
                if not m.any():
                    continue
                the_key = (name, p["texture"])
                mats.setdefault(the_key, []).append(I[m])
        order = ["颜", "颜2", "目", "目光", "二重", "睫", "眉", "齿", "口舌"]
        keys = sorted(mats, key=lambda k: (order.index(k[0]) if k[0] in order else len(order), k[0]))
        os.makedirs(folder_textures, exist_ok=True)
        textures, faces, materials = [], [], []
        for the_key in keys:
            name, tex = the_key
            ti = -1
            if tex:
                b = os.path.basename(tex)
                if b not in textures:
                    dst = os.path.join(folder_textures, b)
                    if not os.path.exists(dst):
                        shutil.copyfile(tex, dst + ".tmp")
                        os.replace(dst + ".tmp", dst)
                    textures.append(b)
                ti = textures.index(b)
            I = np.vstack(mats[the_key])
            faces.extend(I.tolist())
            skin_or_hair = name == "颜" or "髮" in name
            nm = name
            k = 2
            while nm in [m["name"] for m in materials]:
                nm = "%s%d" % (name, k)
                k += 1
            materials.append({
                "name": nm, "name_en": nm, "diffuse": (1.0, 1.0, 1.0, 1.0), "specular": (0.0, 0.0, 0.0),
                "brightness": 5.0, "ambient": (0.5, 0.5, 0.5),
                # double-sided except skin and hair (opaque, outline); cast shadows for all
                "flags": 0x1E if skin_or_hair else 0x0F,
                "outline_color": (0.0, 0.0, 0.0, 1.0), "outline_size": 0.6 if skin_or_hair else 0.0,
                "texture": ti, "sphere": -1, "sphere_mode": 0, "toon_sharing": 1, "toon": 0,
                "memo": "Outerplane %s" % os.path.basename(tex or ""), "n_indices": 3 * len(I)})
        # bones: 全ての親 -> 首 -> 頭; everything is weighted to 頭 (no physics: rigid hair)
        vis = pos[base[id(self.vis)]:base[id(self.vis)] + len(self.vis["P"])]
        y_bottom = float(vis[:, 1].min())
        y_neck = CENTER_PIVOT[1] - 1.3 * IOD_PIVOT
        z_c = float(np.median(vis[:, 2]))
        fl = PMX.OS_ROTATION | PMX.OS_VISIBLE | PMX.OS_ACTIVE
        def os_(name, en, p, parent, disp=False):
            return {"name": name, "name_en": en, "pos": tuple(float(x) for x in p), "parent": parent, "layer": 0,
                    "flags": fl | (PMX.OS_DISPLACEMENT if disp else 0), "tail_bone": None,
                    "tail_offset": (0.0, 1.0, 0.0), "addition": None, "fixed_axis": None, "local_axes": None,
                    "external_parent": None, "ik": None, "after_physics": False}
        bones = [os_("全ての親", "master", (0, 0, 0), -1, True),
               os_("首", "neck", (0, min(y_bottom, y_neck), z_c), 0),
               os_("頭", "head", (0, y_neck, z_c), 1)]
        nmor = []
        for name, (pan, per) in morphs.items():
            dl = []
            for p in prims:
                ip = next(i for i, q in enumerate(self.prims) if q is p)
                if ip not in per:
                    continue
                D = per[ip]
                for i in np.flatnonzero(np.abs(D).max(1) > 1e-6):
                    dl.append((int(i + base[id(p)]), tuple(float(x) for x in D[i])))
            if dl:
                nmor.append({"name": name, "name_en": name, "panel": pan, "type": 1, "offsets": dl})
        frames = [{"name": "Root", "name_en": "Root", "special": 1, "elements": [(0, 0)]},
                  {"name": "表情", "name_en": "Exp", "special": 1, "elements": [(1, i) for i in range(len(nmor))]},
                  {"name": "頭", "name_en": "head", "special": 0, "elements": [(0, 1), (0, 2)]}]
        n = len(pos)
        return {"version": 2.0, "encoding": "UTF-16", "n_uv_add": 0,
                "name": self.name, "name_en": self.name,
                "comment": "Outerplane head %s converted by outerplane.py (face library)" %
                               os.path.basename(os.path.normpath(self.folder)), "comment_en": "",
                "vertices": {"pos": [tuple(x) for x in pos], "nor": [tuple(x) for x in nor],
                            "uv": [tuple(x) for x in uv], "uv_add": [[] for _ in range(n)],
                            "type": [0] * n, "bones": [(2, -1, -1, -1)] * n, "weights": [(1.0, 0.0, 0.0, 0.0)] * n,
                            "sdef": [None] * n, "contour": [1.0] * n},
                "faces": faces, "textures": textures, "materials": materials, "bones": bones,
                "morphs": nmor, "frames": frames, "rigid_bodies": [], "joints": []}

    def summary(self):
        return {"name": self.name, "front_glTF": self.front_glTF, "unit": round(self.unit, 3),
                "eye": {k: np.round(v - CENTER_PIVOT, 3).tolist() for k, v in self.eye.items()},
                "parts": [(n, f, r.tolist()) for n, f, r in self.diag]}


def candidates(names_taken=()):
    """Heads to convert: [(name, folder)], and the excluded ones {folder: reason}.

    Readable name of the folder (2000010_Idith -> Idith), otherwise the identifier; "OP-" in
    front of a name already taken (names_taken: the ZZZ characters); two folders with the same
    readable name (2000068_Dahlia, 2000090_Dahlia) each get "-<identifier>"."""
    guard, discarded = [], {}
    for d in sorted(os.listdir(RAW)):
        if not os.path.exists(os.path.join(RAW, d, "controle.glb")):   # fr-ok: file name set by the third-party extraction
            continue
        if d in EXCLUDED:
            discarded[d] = EXCLUDED[d]
            continue
        guard.append(d)
    nl = {d: name_readable(d) for d in guard}
    count = {}
    for n in nl.values():
        count[n] = count.get(n, 0) + 1
    taken = set(names_taken)
    out = []
    for d in guard:
        n = nl[d]
        if count[n] > 1:
            n = "%s-%s" % (n, d.split("_")[0])
        if n in taken:
            n = "OP-" + n
        out.append((n, os.path.join(RAW, d)))
    return out, discarded


def convert(folder, path):
    """Writes the donor PMX (textures next to it); returns the converted head."""
    T = Head(folder)
    P = T.pmx(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp.pmx"
    pmx_write.write(P, tmp)
    os.replace(tmp, path)
    T.materials = [m["name"] for m in P["materials"]]
    T.names_morphs = [m["name"] for m in P["morphs"]]
    return T


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="action", required=True)
    d = sp.add_parser("diag")
    d.add_argument("folders", nargs="+")
    c = sp.add_parser("convert")
    c.add_argument("folder")
    c.add_argument("output", help="path of the PMX; the textures are copied next to it")
    a = ap.parse_args(argv)
    if a.action == "convert":
        T = convert(a.folder, a.output)
        print(T.name, "->", a.output, T.materials, "mouth", T.mouth, "morphs", T.names_morphs)
        return 0
    if a.action == "diag":
        for x in a.folders:
            T = Head(x)
            r = T.summary()
            print(r["name"], r["front_glTF"], r["unit"], r["eye"])
            for p in r["parts"]:
                print("   ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

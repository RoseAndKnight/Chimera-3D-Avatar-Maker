"""Face morphs: evaluation, coverage check, transfer, eyebrow reshaping.

    python3 -I outils/tete/morphes.py controle --pmx <resultat.pmx> [--recette recette.json] [--json sortie.json]
    python3 -I outils/tete/morphes.py reparer --pmx <resultat.pmx> --recette recette.json --sortie <dossier>
    python3 -I outils/tete/morphes.py pieces --pmx <resultat.pmx> --recette recette.json --sortie <dossier>
    python3 -I outils/tete/morphes.py sourcil --pmx <resultat.pmx> --sortie <dossier> \
        [--longueur 1.0] [--epaisseur 1.3] [--arc 0.04] [--hausse 0.0] [--inclinaison 0.0] [--sans-garde]
    python3 -I outils/tete/morphes.py garde --pmx <resultat.pmx> --sortie <dossier> [--ecart-min 0.045]
    python3 -I outils/tete/morphes.py ecarts --pmx <resultat.pmx> [--json sortie.json]
    (writing: --textures-pour <folder> rewrites texture paths relative to the folder where the
    PMX will be placed, and also looks for textures there)

Evaluate: displacement of each vertex for a morph (vertex, bone, group), at rest elsewhere.
Bone morphs are evaluated by linear skinning (BDEF; SDEF treated as BDEF2), with the bone
hierarchy and the rotation / translation inheritance (付与).

Check (one line per face morph):
- per face material, the maximum displacement;
- "empty": a donor (recipe) has the morph, the composed face does not move;
- "frozen": a part laid on the skin (lashes, eyebrows, shadows, 颜2, 口线) stays in place
  where the skin under it moves;
- "pierces": an inner part (eyeballs, teeth, tongue) comes in front of the skin, while
  at rest it is behind it (teeth poking out at the corner of a smile).

Repair: see reparer(); eyebrow reshaping: see deformer_sourcils().
Parts: removes from a finished PMX the faces that an older sorting had put in another region
(retirer_pieces; the material stays, empty). Guard: minimal eyebrow - eye line gap in
each morph (garder_sourcils, ECART_SOURCIL_MIN); ecarts: the measurement only.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))

import geom as G  # noqa: E402
import pmx as PMX  # noqa: E402

PEAUX = ("颜", "颜_A", "颜_A2")
# parts laid on the skin: they must follow the skin
SURFACE = ("颜2", "口线", "眉", "眉毛", "睫", "眉睫", "睫眉", "右睫眉", "二重", "二重影")
# inner parts: they move as in the donor, but must not pierce the skin
INTERIEUR = ("白目", "目", "目影", "目光", "目光2", "瞳光", "目色", "齿", "口舌")
SOURCILS = ("眉", "眉毛")
SEUIL = 0.01       # displacement (MMD units) below which nothing moves
FIGEE = 0.25       # a laid part moves less than this fraction of the skin under it: frozen
PART_FIGEE = 0.3   # fraction of a part's vertices where the skin moves that must be frozen to call it frozen
PEAU_BOUGE = 0.4   # the skin under the part moves: at least this fraction of the largest skin displacement
FAIBLE = 0.5       # the skin moves less than this fraction of the expected composed skin: missing
PERCE = 0.01       # lead in front of the skin (MMD units) that counts as piercing


# =================================================================== evaluation

def _quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp_id(q, f):
    q = np.asarray(q, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    if q[3] < 0:
        q = -q
    ang = 2 * np.arccos(np.clip(q[3], -1, 1))
    s = np.sin(ang / 2)
    if s < 1e-9:
        return np.array([0, 0, 0, 1.0])
    ax = q[:3] / s
    return np.concatenate([ax * np.sin(f * ang / 2), [np.cos(f * ang / 2)]])


class Modele:
    """A loaded PMX, with its arrays and the morph index."""

    def __init__(self, chemin=None, pmx=None):
        self.chemin = chemin
        self.pmx = pmx if pmx is not None else PMX.lire(chemin)
        self.t = PMX.vers_tableaux(self.pmx)
        self.pos = self.t["pos"]
        self.nv = len(self.pos)
        self.index = {}
        for i, m in enumerate(self.pmx["morphes"]):
            self.index.setdefault(m["nom"], i)
        self.noms_mat = [m["nom"] for m in self.pmx["materiaux"]]
        self.mat_som = np.full(self.nv, -1)
        F = self.t["faces"]
        mf = self.t["mat_face"]
        self.mat_som[F[:, 0]] = mf
        self.mat_som[F[:, 1]] = mf
        self.mat_som[F[:, 2]] = mf

    def sommets(self, noms):
        ids = [i for i, n in enumerate(self.noms_mat) if n in noms]
        return np.unique(self.t["faces"][np.isin(self.t["mat_face"], ids)])

    def _accumuler(self, i, w, dv, poses, prof=0):
        m = self.pmx["morphes"][i]
        if prof > 8 or w == 0:
            return
        if m["type"] == 1:
            for v, d in m["decalages"]:
                dv[int(v)] += w * np.asarray(d)
        elif m["type"] == 2:
            for o, t, q in m["decalages"]:
                tt, qq = poses.get(int(o), (np.zeros(3), np.array([0, 0, 0, 1.0])))
                q2 = _slerp_id(q, w)
                # composed rotations (small angles: order matters little)
                x1, y1, z1, w1 = qq
                x2, y2, z2, w2 = q2
                qq = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                               w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
                poses[int(o)] = (tt + w * np.asarray(t), qq)
        elif m["type"] in (0, 9):
            for j, f in m["decalages"]:
                self._accumuler(int(j), w * f, dv, poses, prof + 1)

    def matrices(self, poses):
        """Skinning matrices (nb, 4, 4) relative to rest for local bone poses."""
        os_ = self.pmx["os"]
        nb = len(os_)
        M = [None] * nb

        def locale(b):
            t, q = poses.get(b, (np.zeros(3), np.array([0, 0, 0, 1.0])))
            a = os_[b]["ajout"]
            if a is not None and 0 <= a["source"] < nb and a["source"] in poses:
                ts, qs = poses[a["source"]]
                if a["rotation"]:
                    q2 = _slerp_id(qs, a["facteur"])
                    t_, q = t, q
                    x1, y1, z1, w1 = q
                    x2, y2, z2, w2 = q2
                    q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                                  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
                if a["deplacement"]:
                    t = t + a["facteur"] * np.asarray(ts)
            p = np.asarray(os_[b]["pos"], dtype=float)
            L = np.eye(4)
            R = _quat(q)
            L[:3, :3] = R
            L[:3, 3] = p + np.asarray(t) - R @ p
            return L

        def glob(b, prof=0):
            if M[b] is not None:
                return M[b]
            par = os_[b]["parent"]
            L = locale(b)
            M[b] = (glob(par, prof + 1) @ L) if (0 <= par < nb and prof < 512) else L
            return M[b]
        actifs = set(poses)
        for b in range(nb):
            a = os_[b]["ajout"]
            if a is not None and a["source"] in poses:
                actifs.add(b)
        # only bones under an active bone change: the others stay at identity
        out = np.tile(np.eye(4), (nb, 1, 1))
        if not actifs:
            return out
        for b in range(nb):
            out[b] = glob(b)
        return out

    def deplacement(self, nom, w=1.0):
        """Displacement (nv, 3) of each vertex for morph nom at value w."""
        dv = np.zeros((self.nv, 3))
        if nom not in self.index:
            return dv
        poses = {}
        self._accumuler(self.index[nom], w, dv, poses)
        p = self.pos + dv
        if not poses:
            return dv
        M = self.matrices(poses)
        idx = self.t["os"]
        wt = self.t["poids"].copy()
        wt[idx < 0] = 0
        ii = np.where(idx < 0, 0, idx)
        ph = np.concatenate([p, np.ones((self.nv, 1))], 1)
        q = np.zeros((self.nv, 3))
        for k in range(4):
            q += wt[:, k:k + 1] * np.einsum("nij,nj->ni", M[ii[:, k]][:, :3], ph)
        s = wt.sum(1, keepdims=True)
        q = np.where(s > 1e-9, q / np.maximum(s, 1e-9), p)
        return q - self.pos

    def morphes_visage(self):
        """Face morphs: eyebrow, eye, mouth panels (1 to 3), vertex, bone or group morphs."""
        out = []
        for m in self.pmx["morphes"]:
            if m["panneau"] in (1, 2, 3) and m["type"] in (0, 1, 2) and m["nom"] not in out:
                out.append(m["nom"])
        return out


def retenir(V0, V1, F, p0, d, marge=0.002):
    """Inner part (teeth, tongue, eyeballs): vertices behind the skin (V0, F) at rest stay
    behind the deformed skin V1. A vertex that would come in front is pushed back in depth to
    just behind it (by marge; to its rest lead if it already stuck out a little), seen from
    the front like the PERCE check. Returns (corrected d, count)."""
    a0 = _avance(V0, F, p0)
    a1 = _avance(V1, F, p0 + d)
    cible = np.where(a0 <= 0, -marge, a0)
    m = (a0 <= PERCE) & (a1 > np.maximum(a0, 0))
    d = d.copy()
    d[m, 2] += a1[m] - cible[m]
    return d, int(m.sum())


def retenir_pose(chemin, journal=print):
    """After placing a composed face on a head (retarget_tete visage): each vertex morph keeps
    the inner parts behind the skin (retenir), as in the composed face. Placement rotates skin
    and parts each by its own local rotation and smooths the skin near the seam: Jane's tongue
    under a Kanon base stuck out 0.027 in front of the lip in □ on Belle, clean in the composed
    face.
    And a surface part (eyebrow, lashes, blush) that the morph leaves in place above moving
    skin (check criterion: FIGEE, PEAU_BOUGE) follows that skin: on a Kanon base, Jane's
    eyebrow, placed lower, covered the eyelid that Jane's 眼角下 lowers.
    Rewrites the PMX; returns {morph: vertices pushed back or re-attached}."""
    import pmx_ecrire
    Mo = Modele(chemin=chemin)
    ids = [i for i, n in enumerate(Mo.noms_mat) if n in PEAUX]
    F = Mo.t["faces"][np.isin(Mo.t["mat_face"], ids)]
    vi = Mo.sommets(INTERIEUR)
    vs = Mo.sommets(SURFACE)
    vp = Mo.sommets(PEAUX)
    rep, inv = G.souder(Mo.pos[vp])
    s_de = np.full(Mo.nv, -1)
    s_de[vp] = inv
    Fs = s_de[F]
    Fs = Fs[(Fs[:, 0] != Fs[:, 1]) & (Fs[:, 1] != Fs[:, 2]) & (Fs[:, 0] != Fs[:, 2])]
    if len(vs):
        _, fs_, bs_, _ = G.plus_proche(Mo.pos[vs], Mo.pos[vp[rep]], Fs)
    out, colles = {}, {}
    for m in Mo.pmx["morphes"]:
        if m["type"] != 1:
            continue
        D = Mo.deplacement(m["nom"])
        N = D.copy()
        k = 0
        if len(vi) and np.abs(D[vi]).max() >= SEUIL:
            N[vi], k = retenir(Mo.pos, Mo.pos + D, F, Mo.pos[vi], D[vi])
        if len(vs):
            ds = np.zeros((len(rep), 3))
            np.add.at(ds, inv, D[vp])
            ds /= np.maximum(np.bincount(inv, minlength=len(rep)), 1)[:, None]
            sous = (bs_[:, :, None] * ds[Fs[fs_]]).sum(1)
            sk = np.linalg.norm(sous, axis=1)
            pmax = float(np.linalg.norm(ds, axis=1).max()) if len(ds) else 0.0
            fig = (sk > max(SEUIL, PEAU_BOUGE * pmax)) & (np.linalg.norm(D[vs], axis=1) < FIGEE * sk)
            if fig.any():
                N[vs[fig]] = sous[fig]
                colles[m["nom"]] = int(fig.sum())
        ch = np.where(np.abs(N - D).max(1) > 0)[0]
        if not len(ch):
            continue
        dec = {int(v): np.asarray(x, float) for v, x in m["decalages"]}
        for v in ch:
            dec[int(v)] = dec.get(int(v), np.zeros(3)) + (N[v] - D[v])
        m["decalages"] = [(v, tuple(float(x) for x in x3)) for v, x3 in sorted(dec.items())]
        if k:
            out[m["nom"]] = k
    if out or colles:
        tmp = chemin + ".tmp.pmx"
        pmx_ecrire.ecrire(Mo.pmx, tmp)
        os.replace(tmp, chemin)
        journal("  inner parts kept behind the skin after placement: %s" % out)
        journal("  surface parts re-attached to the skin after placement: %s" % colles)
    return {**out, **{n: out.get(n, 0) + c for n, c in colles.items()}}


# =================================================================== check

def _avance(V, F, pts):
    """Lead of each point in front of the skin (V, F), seen from the front (the face looks
    towards -Z): z of the frontmost skin at the same (x, y) minus z of the point; -inf where no
    skin covers the point (inside the eye or mouth opening)."""
    T = V[F]
    n = G.normales_faces(V, F)
    T = T[n[:, 2] < 0]
    out = np.full(len(pts), -np.inf)
    a, b, c = T[:, 0, :2], T[:, 1, :2], T[:, 2, :2]
    v0, v1 = b - a, c - a
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(den) > 1e-12
    a, v0, v1, den, T = a[ok], v0[ok], v1[ok], den[ok], T[ok]
    for s in range(0, len(pts), 256):
        q = pts[s:s + 256]
        w = q[:, None, :2] - a[None]
        l1 = (w[..., 0] * v1[None, :, 1] - w[..., 1] * v1[None, :, 0]) / den[None]
        l2 = (v0[None, :, 0] * w[..., 1] - v0[None, :, 1] * w[..., 0]) / den[None]
        dedans = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        z = T[None, :, 0, 2] + l1 * (T[None, :, 1, 2] - T[None, :, 0, 2]) + l2 * (T[None, :, 2, 2] - T[None, :, 0, 2])
        z = np.where(dedans, z, np.inf).min(1)
        out[s:s + 256] = np.where(np.isfinite(z), z - q[:, 2], -np.inf)
    return out


def controle(Mo, noms=None, donneurs=None):
    """For each morph: max displacement per face material, frozen parts, parts that pierce
    the skin, empty (donneurs: name -> expected {material: amplitude})."""
    noms = noms or Mo.morphes_visage()
    vp = Mo.sommets(PEAUX)
    ids_peau = [i for i, n in enumerate(Mo.noms_mat) if n in PEAUX]
    Fp = Mo.t["faces"][np.isin(Mo.t["mat_face"], ids_peau)]
    # welded skin (a stitched skin has duplicates)
    rep, inv = G.souder(Mo.pos[vp])
    Vs0 = Mo.pos[vp[rep]]
    s_de = np.full(Mo.nv, -1)
    s_de[vp] = inv
    Fs = s_de[Fp]
    Fs = Fs[(Fs[:, 0] != Fs[:, 1]) & (Fs[:, 1] != Fs[:, 2]) & (Fs[:, 0] != Fs[:, 2])]
    pieces = {}
    for i, n in enumerate(Mo.noms_mat):
        if n in SURFACE + INTERIEUR:
            v = np.unique(Mo.t["faces"][Mo.t["mat_face"] == i])
            if len(v):
                pieces[i] = v
    # face only: parts close to the face skin
    allv = np.concatenate(list(pieces.values())) if pieces else np.zeros(0, int)
    # at rest: signed distance of the inner parts
    repos = {}
    for i, v in pieces.items():
        if Mo.noms_mat[i] in INTERIEUR:
            repos[i] = _avance(Vs0, Fs, Mo.pos[v])
    # skin under each part: closest point on the welded skin (face, barycentrics)
    proche = {i: G.plus_proche(Mo.pos[v], Vs0, Fs)[1:3] for i, v in pieces.items()}
    lignes = []
    for nom in noms:
        D = Mo.deplacement(nom)
        a = np.linalg.norm(D, axis=1)
        ds = np.zeros((len(Vs0), 3))
        np.add.at(ds, inv, D[vp])
        ds /= np.maximum(np.bincount(inv, minlength=len(Vs0)), 1)[:, None]
        L = {"morphe": nom, "type": Mo.pmx["morphes"][Mo.index[nom]]["type"] if nom in Mo.index else None,
             "materiaux": {}, "figees": [], "percent": [], "vides": []}
        L["materiaux"]["peau"] = round(float(np.linalg.norm(ds, axis=1).max()), 4) if len(ds) else 0.0
        for i, v in pieces.items():
            n = Mo.noms_mat[i]
            cle = n + ("#%d" % i if sum(1 for x in pieces if Mo.noms_mat[x] == n) > 1 else "")
            L["materiaux"][cle] = round(float(a[v].max()), 4)
            if n in SURFACE:
                f, b = proche[i]
                sk = np.linalg.norm((b[:, :, None] * ds[Fs[f]]).sum(1), axis=1)
                bouge = sk > max(SEUIL, PEAU_BOUGE * L["materiaux"]["peau"])
                if bouge.sum() >= max(3, 0.05 * len(v)):
                    fig = bouge & (a[v] < FIGEE * sk)
                    if fig.sum() > PART_FIGEE * bouge.sum():
                        L["figees"].append({"piece": cle, "sommets": int(fig.sum()), "sur": int(bouge.sum()),
                                            "peau": round(float(sk[bouge].mean()), 4),
                                            "piece_bouge": round(float(a[v][bouge].mean()), 4)})
            elif n in INTERIEUR and a.max() > SEUIL:
                sd = _avance(Vs0 + ds, Fs, Mo.pos[v] + D[v])
                perc = (sd > PERCE) & (repos[i] <= PERCE)
                if perc.sum() >= 3:
                    L["percent"].append({"piece": cle, "sommets": int(perc.sum()),
                                         "avance_max": round(float(sd[perc].max()), 4)})
        if donneurs and nom in donneurs:
            for cle, amp in donneurs[nom].items():
                got = max([v for k, v in L["materiaux"].items() if k.split("#")[0] == cle] or [0.0])
                if amp > SEUIL and got < (FAIBLE if cle == "peau" else 0.2) * amp:
                    L["vides"].append({"materiau": cle, "donneur": round(amp, 4), "ici": got})
        lignes.append(L)
    return lignes


def amplitudes(Mo, noms, mats):
    """Max amplitude per material (among mats) for each morph: name -> {mat: amp}."""
    out = {}
    vs = {n: Mo.sommets((n,)) for n in mats}
    for nom in noms:
        if nom not in Mo.index:
            continue
        a = np.linalg.norm(Mo.deplacement(nom), axis=1)
        out[nom] = {n: float(a[v].max()) for n, v in vs.items() if len(v)}
    return out


# =================================================================== composition

# parts that do not follow the skin when their donor lacks the morph: the iris and its highlights
FIXES = ("目", "目光", "目光2", "瞳光", "目色")


def souder_dep(T, D):
    """Displacement of the raw vertices -> welded skin of head T (mean)."""
    k = T.b2s
    m = k >= 0
    out = np.zeros((T.n, 3))
    np.add.at(out, k[m], D[m])
    cnt = np.bincount(k[m], minlength=T.n)
    return out / np.maximum(cnt, 1)[:, None]


def similitude(P, Q):
    """Least-squares similarity (s, R, t): Q ≈ s R P + t (Umeyama)."""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(Y.T @ X / len(P))
    d = np.sign(np.linalg.det(U @ Vt))
    E = np.diag([1, 1, d])
    R = U @ E @ Vt
    s = float((S * np.diag(E)).sum() / max((X * X).sum() / len(P), 1e-18))
    return s, R, mq - s * R @ mp


def normales_faces(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def projeter_oriente(q, nq, V, F, cos=0.3):
    """Closest point of the surface (V, F) to each q among the faces oriented like nq
    (faces, barycentrics). With closed lips, the upper and lower lip touch: only the
    orientation tells which one is right."""
    _, f, b, _ = G.plus_proche(q, V, F)
    nf = normales_faces(V, F)
    nq = nq / np.maximum(np.linalg.norm(nq, axis=1, keepdims=True), 1e-12)
    for i in np.where((nf[f] * nq).sum(1) < cos)[0]:
        ok = np.where(nf @ nq[i] >= cos)[0]
        if len(ok):
            _, fi, bi, _ = G.plus_proche(q[i:i + 1], V, F[ok])
            f[i], b[i] = ok[fi[0]], bi[0]
    return f, b


def peau_commune(Vd, Fd, pts, V, F, porter):
    """Skin under part vertices, on the donor (Vd, Fd) and here (V, F): the donor's point,
    carried here (porter(x, est_vecteur)), then projected onto the faces oriented the same way.
    Returns ((faces, bary) on the donor, (faces, bary) here)."""
    _, fd, bd, _ = G.plus_proche(pts, Vd, Fd)
    q = porter((bd[:, :, None] * Vd[Fd[fd]]).sum(1), False)
    nq = porter(normales_faces(Vd, Fd)[fd], True)
    f, b = projeter_oriente(q, nq, V, F)
    return (Fd[fd], bd), (F[f], b)


class Donneurs:
    """Donors of a recipe: heads placed on the pivot, morphs evaluated on demand."""

    def __init__(self, biblio, A):
        import bibliotheque as BI
        self.BI = BI
        self.biblio = os.path.expanduser(biblio)
        self.A = A
        self.MoA = Modele(pmx=A.pmx)
        self._d, self._dep, self._proj, self._pieces = {}, {}, {}, {}

    def __getitem__(self, nom):
        if nom not in self._d:
            z, fiche = self.BI._charger(self.biblio, nom)
            import tete as TT
            D = TT.Tete(fiche["pmx"])
            D.placer(float(z["echelle"]), z["c_src"], z["c_dst"])
            self._d[nom] = (z, fiche, D, Modele(chemin=fiche["pmx"], pmx=D.pmx))
        return self._d[nom]

    def pieces(self, nom, p, stocke=False):
        """Raw faces of the parts of region p of donor nom: sorted here by
        bibliotheque.ranger_pieces (current rule), or those sorted at extraction
        (stocke=True: partie.npz, to re-read a face composed before the current rule)."""
        z, _, D, _ = self[nom]
        if stocke:
            return z["pieces_" + p]
        if nom not in self._pieces:
            self._pieces[nom] = self.BI.ranger_pieces(D)
        return self._pieces[nom][p]

    def dep(self, nom, n):
        """Displacement (donor's raw vertices, original frame); None if it lacks the morph."""
        if (nom, n) not in self._dep:
            Mo = self[nom][3] if nom is not None else self.MoA
            d = Mo.deplacement(n) if n in Mo.index else None
            self._dep[(nom, n)] = d if d is not None and np.abs(d).max() > 1e-6 else None
        return self._dep[(nom, n)]

    def a(self, nom, n):
        return self.dep(nom, n) is not None

    def noms_visage(self, nom):
        return self[nom][3].morphes_visage()

    def _projection(self, nom):
        """Face and barycentrics, on the placed donor, of each registered pivot vertex,
        each point keeping its side around the openings (like extraire_un)."""
        if nom not in self._proj:
            z, _, D, _ = self[nom]
            A, BI = self.A, self.BI
            X = z["X"].astype(float)
            XD = D._hote(D.V)
            _, face, bary, _ = G.plus_proche(X, XD, D.F)
            cA = BI.cotes(A, BI.COTE_RAYON * A.iod)
            cD = BI.cotes(D, BI.COTE_RAYON * D.iod)
            cF = np.sign(cD[D.F].astype(int).sum(1))
            for sg in (1, -1):
                qa = np.where(cA == sg)[0]
                fd = np.where(cF != -sg)[0]
                if len(qa) and len(fd):
                    _, f2, b2, _ = G.plus_proche(X[qa], XD, D.F[fd])
                    face[qa], bary[qa] = fd[f2], b2
            self._proj[nom] = (face, bary)
        return self._proj[nom]

    def peau(self, nom, n):
        """Offset (A.n, 3) of the pivot skin for the donor's morph n, at pivot scale: the
        library's one if present (vertex morph), otherwise sampled here."""
        z, _, D, Mo = self[nom]
        noms = list(z["noms"])
        m = Mo.pmx["morphes"][Mo.index[n]] if n in Mo.index else None
        if n in noms and m is not None and m["type"] == 1:
            return z["delta"][noms.index(n)].astype(float)
        d = self.dep(nom, n)
        if d is None:
            return np.zeros((self.A.n, 3))
        face, bary = self._projection(nom)
        ds = souder_dep(D, d)
        return float(z["echelle"]) * (bary[:, :, None] * ds[D.F[face]]).sum(1)

    def peau_pivot(self, n):
        d = self.dep(None, n)
        return np.zeros((self.A.n, 3)) if d is None else souder_dep(self.A, d)


def delta_compose(dons, recette, M, wv, n):
    """Offset of the composed skin (pivot frame) for morph n: the donor of each region if it
    has the morph, else the base, else the pivot; outside the face, the pivot.
    The fallback does not cross from one game to another (pivot: the active game, ZZZ): an
    Outerplane region whose donor lacks the morph stays at rest rather than take the eyelid or
    cheek of a ZZZ head, made for another eye (frozen lashes, highlights in front of the skin),
    and vice versa.
    A record without "jeu" is a head of the active game (its short name, description
    outils/jeux/<jeu>)."""
    if os.path.join(ICI, "..") not in sys.path:
        sys.path.append(os.path.join(ICI, ".."))
    import jeux
    J = jeux.actif()
    PIVOT = J.valeur("abrege", J.titre)
    BASE = "base"
    jeu = lambda nom: dons[nom][1].get("jeu", PIVOT)  # noqa: E731
    dA = dons.peau_pivot(n)
    zero = np.zeros_like(dA)
    nb = recette[BASE]
    db = dons.peau(nb, n) if dons.a(nb, n) else (dA if jeu(nb) == PIVOT else zero)
    d = M[BASE][:, None] * db
    for zone, m in M.items():
        if zone == BASE:
            continue
        p = zone.split("_")[0]
        nd = recette[p]
        d = d + m[:, None] * (dons.peau(nd, n) if dons.a(nd, n) else db if jeu(nd) == jeu(nb) else zero)
    return dA + wv[:, None] * (d - dA)


def noms_recette(dons, recette, M=None, wv=None):
    """Face morphs: the pivot's, then the donors' (order of first appearance).
    With M and wv: without those the recipe empties (vide_par_recette)."""
    out = list(dons.MoA.morphes_visage())
    for nd in dict.fromkeys(recette.values()):
        for n in dons.noms_visage(nd):
            if n not in out and (M is None or not vide_par_recette(dons, recette, M, wv, n)):
                out.append(n)
    return out


def vide_par_recette(dons, recette, M, wv, n, pivot=False):
    """A donor's morph (absent from the pivot) whose composed skin keeps less than FAIBLE of
    its skin on its donors: it mostly moved a region taken from elsewhere, from another game,
    that lacks it (Kanon's MOUTH_SHY on a Jane mouth: 0.017 composed vs 0.084 on Kanon, less
    than 0.01 once placed). Such a morph no longer shows its expression: it is not written,
    rather than leave a still mouth under moving teeth.
    pivot=True: also applies to a pivot morph (always written, but emptied the same way)."""
    if not pivot and dons.a(None, n):
        return False
    ref = max([float(np.linalg.norm(wv[:, None] * dons.peau(nd, n), axis=1).max())
               for nd in dict.fromkeys(recette.values()) if dons.a(nd, n)] or [0.0])
    d = delta_compose(dons, recette, M, wv, n)
    return ref > SEUIL and float(np.linalg.norm(d, axis=1).max()) < FAIBLE * ref


def panneau_de(dons, recette, n):
    for Mo in [dons.MoA] + [dons[nd][3] for nd in dict.fromkeys(recette.values())]:
        if n in Mo.index:
            return Mo.pmx["morphes"][Mo.index[n]]["panneau"], Mo.pmx["morphes"][Mo.index[n]]["nom_en"]
    return 4, n


def pieces_ordre(dons, recette, stocke=False):
    """Composed parts, in the order _ecrire_compose adds them: (region, donor,
    donor material, donor faces)."""
    out = []
    for p in ("yeux", "sourcils", "bouche", "base"):
        nd = recette[p]
        z, _, D, _ = dons[nd]
        fs = dons.pieces(nd, p, stocke)
        for i in np.unique(D.mat_face[fs]) if len(fs) else []:
            out.append((p, nd, int(i), fs[D.mat_face[fs] == i]))
    return out


def ecrire_morphe(P, n, dec, panneau, nom_en):
    """Morph n becomes (or is created as) a vertex morph with offsets dec {vertex: d}.
    A morph of another kind with the same name is replaced in place (no duplicate)."""
    dec = [(int(v), tuple(float(x) for x in d)) for v, d in sorted(dec.items()) if np.abs(d).max() > 1e-7]
    for m in P["morphes"]:
        if m["nom"] == n:
            m["type"] = 1
            m["decalages"] = dec
            return False
    P["morphes"].append({"nom": n, "nom_en": nom_en, "panneau": panneau, "type": 1, "decalages": dec})
    k = len(P["morphes"]) - 1
    for c in P["cadres"]:
        if c["nom"] == "表情":
            c["elements"] = list(c["elements"]) + [(1, k)]
            break
    return True


# =================================================================== repairing a finished PMX

class Fini:
    """A finished PMX (composed face placed by retarget_tete visage, then hairstyle, colours)
    read back against its recipe: which vertices come from which part of which donor, and how
    the pivot frame is carried onto it."""

    def __init__(self, chemin, recette_json, recalage=None):
        import glob
        import bibliotheque as BI
        import retarget_tete as RT
        self.chemin = chemin
        with open(recette_json, encoding="utf-8") as fh:
            R = json.load(fh)
        self.recette = R["recette"]
        A = BI.charger_pivot()
        self.A = A
        self.M = BI.masques(A)
        self.dons = Donneurs(R["biblio"], A)
        npz = {n: self.dons[n][0] for n in set(self.recette.values())}
        Xf, _, self.wv = BI.forme_composee(A, self.M, npz, self.recette)
        self.P = PMX.lire(chemin)
        self.Mo = Modele(chemin=chemin, pmx=self.P)
        pos = self.Mo.pos
        # weld order of the composed face (it depends on positions) -> pivot order
        vb = A.peau_bruts
        kp = A.b2s[vb]
        _, inv_c = G.souder(Xf[kp])
        self.kp_de_kc = np.zeros(inv_c.max() + 1, dtype=np.int64)
        self.kp_de_kc[inv_c] = kp
        # registration of the retarget_tete visage pass: the one whose X lands on the 颜_A skin
        va = self.Mo.sommets(("颜_A",))
        cands = [recalage] if recalage else sorted(glob.glob(os.path.join(os.path.dirname(recette_json), "sur-*", "recalage.npz")))
        self.rec = None
        for c in cands:
            z = np.load(c)
            if len(z["X"]) != len(self.kp_de_kc):
                continue
            d, j = _proche(pos[va], z["X"])
            if d.max() < 1e-5:
                self.rec, self.chemin_rec = z, c
                break
        if self.rec is None:
            raise RuntimeError("no registration matches the 颜_A skin of %s" % chemin)
        X, XA = self.rec["X"], self.rec["XA"]
        # seam band: copies of the patch border, with B's material, at the positions of X
        vB = self.Mo.sommets(("颜",))
        dB, jB = _proche(pos[vB], X)
        _, ja = _proche(pos[va], X)
        self.peau_ids = np.concatenate([va, vB[dB < 1e-5]])
        self.peau_kc = np.concatenate([ja, jB[dB < 1e-5]])
        self.graines = vB[dB < 1e-5]
        ids = [i for i, n in enumerate(self.Mo.noms_mat) if n in PEAUX]
        self.Fpeau = self.Mo.t["faces"][np.isin(self.Mo.t["mat_face"], ids)]
        # scale and local rotations of the face pass (like retarget_tete)
        Xc = Xf[self.kp_de_kc]
        self.s = float(np.sqrt(((XA - XA.mean(0)) ** 2).sum() / ((Xc - Xc.mean(0)) ** 2).sum()))
        self.residu_place = float(np.abs(self.s * (Xc - Xc.mean(0)) + XA.mean(0) - XA).max())
        self.Rv = RT.rotations_locales(XA, X, self.rec["FA"])
        # parts: materials "name (Donor)" in composition order; current sorting, otherwise
        # the extraction one (face composed earlier); an emptied material is a removed part
        # (retirer_pieces)
        mats = [i for i, m in enumerate(self.P["materiaux"]) if (m["nom_en"] or "").endswith(")") and "(" in m["nom_en"]
                and m["nom"] not in PEAUX]
        F, mf = self.Mo.t["faces"], self.Mo.t["mat_face"]
        nf = np.bincount(mf, minlength=len(self.P["materiaux"]))

        def gardees(p, nd, fs):
            return fs[np.isin(fs, self.dons.pieces(nd, p))]

        def accorde(att):
            return len(att) == len(mats) and all(
                self.P["materiaux"][j]["nom_en"].endswith("(%s)" % nd)
                and nf[j] in (0, len(fs), len(gardees(p, nd, fs)))
                for j, (p, nd, i, fs) in zip(mats, att))
        attendu = None
        for stocke in (False, True):
            att = pieces_ordre(self.dons, self.recette, stocke)
            if accorde(att):
                attendu, self.rangement = att, ("extraction" if stocke else "current")
                break
        if attendu is None:
            raise RuntimeError("parts: the %d donor materials of %s match no sorting of the recipe"
                               % (len(mats), chemin))
        self.pieces = []  # (final material, region, donor, final vertices, donor vertices, scale, rotation)
        for j, (p, nd, i, fs) in zip(mats, attendu):
            D = self.dons[nd][2]
            Ff = F[mf == j]
            if not len(Ff):
                continue
            if len(Ff) != len(fs):
                fs = gardees(p, nd, fs)
            vf, idx = np.unique(Ff.ravel(), return_index=True)
            vd = D.faces[fs].ravel()[idx]
            Pd, Pf = D.pos[vd], pos[vf]
            sR = np.zeros((len(vf), 3, 3))
            tR = np.zeros((len(vf), 3))
            # similarity per side (eyes, eyebrows), in one piece elsewhere
            cotes = [Pd[:, 0] > 0, Pd[:, 0] <= 0] if p in ("yeux", "sourcils") else [np.ones(len(vf), bool)]
            ecart = 0.0
            for c in cotes:
                if c.sum() < 3:
                    c = np.ones(len(vf), bool)
                sc, Rc, tc = similitude(Pd[c], Pf[c])
                sR[c] = sc * Rc
                tR[c] = tc
                ecart = max(ecart, float(np.linalg.norm(sc * Pd[c] @ Rc.T + tc - Pf[c], axis=1).max()))
            self.pieces.append({"mat": j, "nom": self.P["materiaux"][j]["nom"], "partie": p, "donneur": nd,
                                "fs": fs, "vf": vf, "vd": vd, "sR": sR, "tR": tR, "ecart": ecart})

    def peau_pivot_vers_fini(self, dp):
        """Offset of the composed skin (pivot frame, pivot order) -> {finished vertex: d}."""
        kc = self.peau_kc
        d = self.s * np.einsum("nij,nj->ni", self.Rv[kc], dp[self.kp_de_kc[kc]])
        return d

    def _sous(self, pc):
        if "sous" not in pc:
            Dt = self.dons[pc["donneur"]][2]
            sR, tR = pc["sR"], pc["tR"]
            porter = lambda x, vec: np.einsum("nij,nj->ni", sR, x) + (0 if vec else tR)
            pc["sous_d"], pc["sous"] = peau_commune(Dt.pos, Dt.faces[Dt.faces_peau_brutes],
                                                    Dt.pos[pc["vd"]], self.Mo.pos, self.Fpeau, porter)
        return pc

    def peau_sous(self, pc, D):
        """Offset of the finished skin (displacement D of all vertices) under each vertex of
        the part, at the point matching the donor's (peau_commune)."""
        F, b = self._sous(pc)["sous"]
        return (b[:, :, None] * D[F]).sum(1)

    def peau_sous_donneur(self, pc, n):
        """Offset of the donor's skin under each vertex of the part, on the donor, carried here."""
        d = self.dons.dep(pc["donneur"], n)
        if d is None:
            return np.zeros((len(pc["vf"]), 3))
        F, b = self._sous(pc)["sous_d"]
        return np.einsum("nij,nj->ni", pc["sR"], (b[:, :, None] * d[F]).sum(1))

    def piece_donneur(self, pc, n):
        d = self.dons.dep(pc["donneur"], n)
        if d is None:
            return None
        return np.einsum("nij,nj->ni", pc["sR"], d[pc["vd"]])


def _proche(Q, X):
    """Distance and index of the point of X closest to each point of Q."""
    d = np.zeros(len(Q))
    j = np.zeros(len(Q), dtype=np.int64)
    for s in range(0, len(Q), 1024):
        dd = np.linalg.norm(Q[s:s + 1024][:, None] - X[None], axis=2)
        j[s:s + 1024] = dd.argmin(1)
        d[s:s + 1024] = dd.min(1)
    return d, j


def reparer(fini, journal=print):
    """Completes the face morphs of a finished PMX: the composed skin and each part, in a morph
    that does not move them, receive their motion (the region's donor, carried by the local
    rotation and scale; otherwise the part follows the skin). What already moves is kept as is;
    the rest of the PMX is not touched. Returns a report per morph."""
    import retarget_tete as RT
    Fi, P, Mo, dons = fini, fini.P, fini.Mo, fini.dons
    noms = noms_recette(dons, Fi.recette, Fi.M, Fi.wv)
    for n in Mo.morphes_visage():
        if n not in noms:
            noms.append(n)
    rapport, nouveaux_peau = {}, []
    for n in noms:
        D = Mo.deplacement(n)
        a = np.linalg.norm(D, axis=1)
        avant = D.copy()
        r = {"peau": None, "pieces": {}}
        # composed skin
        dp = delta_compose(dons, Fi.recette, Fi.M, Fi.wv, n)
        # the face skin receives the composed skin if it does not move, or moves much less
        # (Orpheya's ワ２: 0.025 at the seam edge, a morph of the target head, vs 0.073 of
        # composed skin: without it, Belle's parts moved on a still skin)
        if np.abs(dp).max() > 1e-6:
            T = Fi.peau_pivot_vers_fini(dp)
            if a[Fi.peau_ids].max() < max(SEUIL, FAIBLE * np.linalg.norm(T, axis=1).max()):
                D[Fi.peau_ids] = T
                r["peau"] = "composed"
                nouveaux_peau.append(n)
        # parts: donor, otherwise the skin
        for pc in Fi.pieces:
            if a[pc["vf"]].max() >= SEUIL:
                continue
            # donor motion, plus the gap between the skin under the part here and the skin under
            # the part on the donor (the whole gap if the donor lacks the morph: the part
            # follows the skin): the part's motion relative to its skin is the donor's
            d = Fi.piece_donneur(pc, n)
            quoi = "donor"
            if pc["nom"] not in FIXES:
                e = Fi.peau_sous(pc, D) - Fi.peau_sous_donneur(pc, n)
                if np.linalg.norm(e, axis=1).max() >= SEUIL:
                    quoi = "follows the skin" if d is None else "donor + skin gap"
                d = e if d is None else d + e
            if d is not None and pc["nom"] in INTERIEUR:
                p0 = Fi.Mo.pos[pc["vf"]]
                d, k = retenir(Fi.Mo.pos, Fi.Mo.pos + D, Fi.Fpeau, p0, d)
                if k:
                    quoi += ", %d pushed back" % k
            if d is not None and np.linalg.norm(d, axis=1).max() >= SEUIL:
                D[pc["vf"]] = d
                r["pieces"][pc["nom"] + "/" + pc["donneur"]] = quoi
        if r["peau"] is None and not r["pieces"]:
            continue
        ch = np.where(np.abs(D - avant).max(1) > 1e-7)[0]
        m0 = P["morphes"][Mo.index[n]] if n in Mo.index else None
        if m0 is not None and m0["type"] == 1:
            dec = {int(v): np.asarray(d) for v, d in m0["decalages"]}
            for v in ch:
                dec[int(v)] = D[v]
        else:
            # other kind (bone, group) or absent: vertex morph, its current effect kept
            dec = {int(v): D[v] for v in np.where(np.abs(D).max(1) > 1e-7)[0]}
        panneau, nom_en = (m0["panneau"], m0["nom_en"]) if m0 is not None else panneau_de(dons, Fi.recette, n)
        r["cree"] = ecrire_morphe(P, n, dec, panneau, nom_en)
        r["type_avant"] = None if m0 is None else m0["type"]
        rapport[n] = r
        journal("  %-8s %s%s %s" % (n, "created " if r["cree"] else "", "composed skin" if r["peau"] else "",
                                    ", ".join("%s %s" % kv for kv in r["pieces"].items())))
    # seam: offsets harmonised near the band for morphs whose skin is new
    if nouveaux_peau and len(Fi.graines):
        Q = {"sommets": P["sommets"], "faces": P["faces"], "materiaux": P["materiaux"],
             "morphes": [m for m in P["morphes"] if m["nom"] in nouveaux_peau and m["type"] == 1]}
        ids = [i for i, x in enumerate(P["materiaux"]) if x["nom"] in PEAUX]
        RT.harmoniser_morphes(Q, ids, Fi.graines)
    return rapport


def retirer_pieces(fini, journal=print):
    """Parts that a composed face got from an older extraction sorting and that the current
    sorting (bibliotheque.ranger_pieces) no longer gives to their region: their faces are
    removed (the material stays, empty; vertices, bones and morphs stay: nothing moves at rest
    elsewhere). Those the current sorting would add are not added (recomposing is needed):
    they are only named. Returns the report."""
    Fi, P = fini, fini.P
    F, mf = Fi.Mo.t["faces"], Fi.Mo.t["mat_face"]
    garde = np.ones(len(F), bool)
    rapport = {"retirees": [], "manquantes": []}
    for pc in Fi.pieces:
        neuf = Fi.dons.pieces(pc["donneur"], pc["partie"])
        hors = ~np.isin(pc["fs"], neuf)
        if hors.any():
            ids = np.where(mf == pc["mat"])[0]
            garde[ids[hors]] = False
            rapport["retirees"].append({"piece": pc["nom"], "donneur": pc["donneur"], "partie": pc["partie"],
                                        "faces": int(hors.sum()), "sur": int(len(hors))})
            journal("  removed: %s (%s, %s) %d faces of %d" % (pc["nom"], pc["donneur"], pc["partie"], hors.sum(), len(hors)))
    for p, nd, i, fs in pieces_ordre(Fi.dons, Fi.recette):
        avant = [pc for pc in Fi.pieces if pc["partie"] == p and pc["donneur"] == nd]
        en_plus = fs[~np.isin(fs, np.concatenate([pc["fs"] for pc in avant]) if avant else [])]
        if len(en_plus):
            D = Fi.dons[nd][2]
            rapport["manquantes"].append({"piece": D.noms_mat[i], "donneur": nd, "partie": p, "faces": int(len(en_plus))})
            journal("  missing (recompose to get it): %s (%s, %s) %d faces" % (D.noms_mat[i], nd, p, len(en_plus)))
    if not garde.all():
        n = np.bincount(mf[garde], minlength=len(P["materiaux"]))
        for k, m in enumerate(P["materiaux"]):
            m["n_indices"] = int(3 * n[k])
        P["faces"] = [tuple(f) for f in F[garde].tolist()]
    return rapport


def attendus(fini, noms):
    """Expected amplitude per material (carried donor parts, composed skin)."""
    out = {}
    for n in noms:
        e = {}
        for pc in fini.pieces:
            d = fini.piece_donneur(pc, n)
            if d is not None:
                e[pc["nom"]] = max(e.get(pc["nom"], 0.0), float(np.linalg.norm(d, axis=1).max()))
        dp = delta_compose(fini.dons, fini.recette, fini.M, fini.wv, n)
        # morph emptied by the recipe (ウィンク２ on Kanon eyes, whose smile only moves the
        # eyebrows): only a remnant at the border is left, nothing to expect from the skin
        vide = vide_par_recette(fini.dons, fini.recette, fini.M, fini.wv, n, pivot=True)
        e["peau"] = 0.0 if vide else float(np.linalg.norm(fini.peau_pivot_vers_fini(dp), axis=1).max())
        out[n] = e
    return out


def imprimer(lignes):
    for L in lignes:
        mat = " ".join("%s %.3f" % (k, v) for k, v in L["materiaux"].items() if v > SEUIL)
        drap = []
        if L["vides"]:
            drap.append("EMPTY " + ",".join(x["materiau"] for x in L["vides"]))
        if L["figees"]:
            drap.append("FROZEN " + ",".join("%s(%d/%d)" % (x["piece"], x["sommets"], x["sur"]) for x in L["figees"]))
        if L["percent"]:
            drap.append("PIERCES " + ",".join("%s(%d, %.3f)" % (x["piece"], x["sommets"], x["avance_max"])
                                             for x in L["percent"]))
        print("%-8s t%s  %-60s %s" % (L["morphe"], L["type"], mat or "(nothing moves)", "  ".join(drap)))
    n = sum(1 for L in lignes if L["vides"] or L["figees"] or L["percent"])
    print("check: %d morphs, %d with defects" % (len(lignes), n))
    return n


def _ecrire_json(chemin, obj):
    tmp = chemin + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, chemin)


# ---------------------------------------------------------------------------------------------
# eyebrow reshaping


class Sourcil:
    """One eyebrow (one side) in its own frame: a along the eyebrow (outwards), b upwards,
    c in depth. The deformation happens in the (a, b) plane; depth follows the forehead skin
    (surface c = Q(a, b) fitted to the surrounding skin), so the eyebrow stays laid on it when
    it rises, arches or stretches towards the temple."""

    def __init__(self, pts, peau, cote):
        self.c0 = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - self.c0, full_matrices=False)
        u, v = Vt[0], Vt[1]
        if u[0] * cote < 0:
            u = -u
        if v[1] < 0:
            v = -v
        v = v - u * (u @ v)
        v /= np.linalg.norm(v)
        self.B = np.stack([u, v, np.cross(u, v)])  # lignes : a, b, c
        L = self.local(pts)
        self.L = float(np.abs(L[:, 0]).max())
        # eyebrow centre line b = m(a) (its current arch): thickness is measured around it
        self.m = np.polyfit(L[:, 0], L[:, 1], 2)
        # surrounding skin: c = Q(a, b)
        Lp = self.local(peau)
        # forehead band around the eyebrow (not the eye socket below)
        bm = np.polyval(self.m, np.clip(Lp[:, 0], -self.L, self.L))
        sel = (np.abs(Lp[:, 0]) < 1.6 * self.L) & (np.abs(Lp[:, 1] - bm) < 0.35 * self.L) & (np.abs(Lp[:, 2]) < 0.4 * self.L)
        Lp = Lp[sel]
        self.q = np.linalg.lstsq(self._base(Lp[:, 0], Lp[:, 1]), Lp[:, 2], rcond=None)[0]
        self.n_peau = int(sel.sum())

    def local(self, p):
        return (p - self.c0) @ self.B.T

    def monde(self, L):
        return L @ self.B + self.c0

    def _base(self, a, b):
        a, b = a / self.L, b / self.L
        # forehead: curved along the eyebrow (towards the temple), almost flat vertically
        return np.stack([np.ones_like(a), a, b, a * a, a * b, a ** 3], 1)

    def Q(self, a, b):
        return self._base(a, b) @ self.q

    def f(self, p, longueur, epaisseur, arc, hausse, inclinaison):
        L = self.local(p)
        a, b, c = L[:, 0], L[:, 1], L[:, 2]
        m = np.polyval(self.m, a)
        a1 = longueur * a
        # the arch raises the middle, slightly outwards; the ends stay
        s = (a - 0.2 * self.L) / (1.2 * self.L)
        b1 = np.polyval(self.m, a) + epaisseur * (b - m) + arc * self.L * (1 - s * s) + hausse * self.L
        t = np.radians(inclinaison)
        a2 = np.cos(t) * a1 - np.sin(t) * b1
        b2 = np.sin(t) * a1 + np.cos(t) * b1
        c2 = c + self.Q(a2, b2) - self.Q(a, b)
        return self.monde(np.stack([a2, b2, c2], 1))


# ---- gap between eyebrow and eye line (front view)

# eye lines: lashes and eyelid creases (a part of these materials sorted as "sourcils" is
# an eyebrow)
TRAITS = ("睫", "睫眉", "眉睫", "右睫眉", "二重")
ENCRE_ALPHA = 0.5   # visible texel: texture alpha x diffuse alpha above this
# Minimal eyebrow - eye line gap (in interocular distances, front view), in any morph:
# median, over 48 original ZZZ models, of each one's smallest gap across its morphs (0.0453;
# p25 0.004: 13 models touch, mostly in 下 and 怒り; median at rest 0.191). Measured on
# 07/11 (trait_principal, encre).
ECART_SOURCIL_MIN = 0.045
ENCRE_CONTRASTE = 0.25  # ink texel: further than this (RGB 0-1) from the skin colour
ENCRE_PAS = 10      # points per side of each triangle to sample the ink


def trouver_texture(chemin_pmx, rel, dossiers=()):
    """File of a PMX texture: relative to its folder, otherwise to one of the folders."""
    rel = rel.replace("\\", "/")
    for d in [os.path.dirname(os.path.abspath(chemin_pmx))] + list(dossiers):
        f = os.path.normpath(os.path.join(d, rel))
        if os.path.exists(f):
            return f
    # relative path written from another folder (an agent's $TMPDIR), then copied: the
    # part after the ".." is often the absolute path
    f = "/" + "/".join(x for x in rel.split("/") if x not in ("..", ".", ""))
    return f if os.path.exists(f) else None


def textures_relatives(P, chemin_pmx, dossier, dossiers=(), journal=print):
    """Texture paths of P rewritten relative to dossier (where the PMX will be placed), each
    first located from chemin_pmx (trouver_texture). Returns the number of changed paths."""
    n = 0
    for k, t in enumerate(P["textures"]):
        f = trouver_texture(chemin_pmx, t, dossiers)
        if f is None:
            journal("  texture not found, path kept: %s" % t)
            continue
        r = os.path.relpath(f, dossier).replace("/", "\\")
        if r != t:
            P["textures"][k] = r
            n += 1
    return n


def _texture(Mo, m, dossiers, cache, journal):
    P = Mo.pmx
    t = P["materiaux"][m]["texture"]
    if not 0 <= t < len(P["textures"]):
        return None
    f = trouver_texture(Mo.chemin or "", P["textures"][t], dossiers)
    if f is None:
        journal("  ink: texture %s not found (material %s)" % (P["textures"][t], P["materiaux"][m]["nom"]))
        return None
    if f not in cache:
        import images as IM
        try:
            cache[f] = IM.lire(f)
        except ValueError as e:
            journal("  ink: %s" % e)
            cache[f] = None
    return cache[f]


def _texels(img, U):
    h, w = img.shape[:2]
    x = np.clip((U[..., 0] % 1.0) * w, 0, w - 1).astype(int)
    y = np.clip((U[..., 1] % 1.0) * h, 0, h - 1).astype(int)
    return img[y, x]


def encre(Mo, faces, dossiers=(), journal=print):
    """Ink points of the faces (indices into Mo.t["faces"]): the points of a barycentric grid
    on each triangle where the texture is visible (alpha) and stands out from the skin (ZZZ
    lash and eyebrow textures are opaque: skin colour around the line). Returns
    (faces (k, 3) of vertices, barycentrics (k, 3)). Without a readable texture, the whole triangle."""
    P, F, mf = Mo.pmx, Mo.t["faces"], Mo.t["mat_face"]
    n = ENCRE_PAS
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
    k = (i + j) <= n
    B = np.stack([(n - i - j)[k], i[k], j[k]], 1) / n
    uv = np.asarray(P["sommets"]["uv"], float)
    cache = {}
    # skin colour: median of the skin texture at its vertices
    fond = []
    for m, nm in enumerate(Mo.noms_mat):
        if nm in PEAUX:
            img = _texture(Mo, m, dossiers, cache, journal)
            if img is not None:
                fond.append(_texels(img, uv[np.unique(F[mf == m])])[:, :3])
    fond = np.median(np.concatenate(fond), 0) if fond else None
    out_f, out_b = [], []
    for m in np.unique(mf[faces]):
        fm = faces[mf[faces] == m]
        img = _texture(Mo, m, dossiers, cache, journal)
        if img is None:
            vu = np.ones((len(fm), len(B)), bool)
        else:
            c = _texels(img, np.einsum("pk,fkd->fpd", B, uv[F[fm]]))
            vu = c[..., 3] * P["materiaux"][m]["diffus"][3] > ENCRE_ALPHA
            if fond is not None:
                vu &= np.linalg.norm(c[..., :3] - fond, axis=-1) > ENCRE_CONTRASTE
        ff, pp = np.nonzero(vu)
        out_f.append(F[fm][ff])
        out_b.append(B[pp])
    if not out_f:
        return np.zeros((0, 3), np.int64), np.zeros((0, 3))
    return np.concatenate(out_f), np.concatenate(out_b)


def _points(V, E):
    Fe, Be = E
    return (Be[:, :, None] * V[Fe]).sum(1)


def ecart_points(A, B, bloc=1024):
    """Smallest (x, y) distance between two point clouds: a bound from subsets, then only the
    points closer than this bound to the other cloud's bounding box."""
    if not len(A) or not len(B):
        return np.inf
    A, B = A[:, :2], B[:, :2]

    def brute(A, B):
        d = np.inf
        for s in range(0, len(A), bloc):
            q = A[s:s + bloc]
            d = min(d, float((((q[:, None] - B[None]) ** 2).sum(-1)).min()))
        return np.sqrt(d)
    u = brute(A[::max(1, len(A) // 200)], B[::max(1, len(B) // 2000)])
    A = A[np.all((A >= B.min(0) - u) & (A <= B.max(0) + u), 1)]
    B = B[np.all((B >= A.min(0) - u) & (B <= A.max(0) + u), 1)] if len(A) else B[:0]
    if not len(A) or not len(B):
        return u
    # each point of B: only if it is within u of a point of A (grid of step u)
    k = np.floor(A / max(u, 1e-9)).astype(np.int64)
    cles = set(map(tuple, k))
    kb = np.floor(B / max(u, 1e-9)).astype(np.int64)
    garde = np.array([any((x + i, y + j) in cles for i in (-1, 0, 1) for j in (-1, 0, 1)) for x, y in kb])
    return min(u, brute(A, B[garde])) if garde.any() else u


def _cotes(V0, E):
    x = _points(V0, E)[:, 0]
    return {"gauche": x > 0, "droit": x < 0}


def ecarts_sourcils(V0, Es, El, iod, deps):
    """Eyebrow - eye line gap on each side, front view, in interocular distances: smallest
    distance between the eyebrow ink (Es) and the eye line ink (El), for each displacement of
    deps {name: (nv, 3)}."""
    cs, cl = _cotes(V0, Es), _cotes(V0, El)
    out = {}
    for nom, D in deps.items():
        V = V0 + D
        ps, pl = _points(V, Es), _points(V, El)
        out[nom] = {c: ecart_points(ps[cs[c]], pl[cl[c]]) / iod for c in cs}
    return out


def iod_globes(Mo):
    """Interocular distance: distance between the iris (目) centres on either side."""
    v = Mo.sommets(("目",))
    p = Mo.pos[v]
    return float(np.linalg.norm(p[p[:, 0] > 0].mean(0) - p[p[:, 0] < 0].mean(0)))


def trait_principal(Mo, fl):
    """Among eye-line faces, the eye line proper: on each side, the largest connected piece
    (vertices welded by position), by area. The other pieces (eyelid crease 二重, lower lashes,
    corner tips) are discarded: on the originals the eyebrow touches the crease when angry
    (Belle, Ellen, Corin), never or almost never the line (Belle 怒り 0.049, Ellen 下 0.061,
    Miyabi 怒り 0.080 interocular distances)."""
    if not len(fl):
        return fl
    F = Mo.t["faces"][fl]
    vb, Fl = G.sous_maillage(Mo.t["faces"], fl)
    _, inv = G.souder(Mo.pos[vb])
    lab = G.composantes_faces(inv[Fl], np.ones(len(Fl), bool))
    P3 = Mo.pos[F]
    aire = 0.5 * np.linalg.norm(np.cross(P3[:, 1] - P3[:, 0], P3[:, 2] - P3[:, 0]), axis=1)
    x = P3[:, :, 0].mean(1)
    garde = np.zeros(len(fl), bool)
    for cote in (x > 0, x < 0):
        if cote.any():
            a = np.bincount(lab[cote], aire[cote], lab.max() + 1)
            garde |= cote & (lab == int(a.argmax()))
    return fl[garde]


def faces_sourcils_traits(Mo):
    """Face indices of the eyebrows (眉, 眉毛) and of the eye line (TRAITS, trait_principal)
    of a composed face (one part per material)."""
    mf = Mo.t["mat_face"]
    ids_s = [i for i, n in enumerate(Mo.noms_mat) if n in SOURCILS]
    ids_l = [i for i, n in enumerate(Mo.noms_mat) if n in TRAITS]
    return np.where(np.isin(mf, ids_s))[0], trait_principal(Mo, np.where(np.isin(mf, ids_l))[0])


def mesurer_ecarts(Mo, noms=None, dossiers=(), faces=None):
    """Eyebrow - eye line gap (interocular distances) at rest and for each face morph that
    moves the eyebrows or the lines: {morph: {gauche, droit}}. faces: (eyebrows, lines) as
    face indices, otherwise by material names."""
    if faces is None:
        fs, fl = faces_sourcils_traits(Mo)
    else:
        fs, fl = faces[0], trait_principal(Mo, faces[1])
    Es, El = encre(Mo, fs, dossiers), encre(Mo, fl, dossiers)
    vs = np.unique(np.concatenate([Es[0].ravel(), El[0].ravel()]))
    deps = {"repos": np.zeros_like(Mo.pos)}
    for n in noms or Mo.morphes_visage():
        D = Mo.deplacement(n)
        if len(vs) and np.abs(D[vs]).max() > 1e-6:
            deps[n] = D
    return ecarts_sourcils(Mo.pos, Es, El, iod_globes(Mo), deps)


def garder_sourcils(P, ecart_min=None, chemin=None, dossiers=(), journal=print):
    """Guard: in each face vertex morph, on each side, the eyebrow offset is scaled down
    (common factor t for the side, by bisection) so that their ink stays ecart_min
    (interocular distances, front view) away from the eye line's ink, or at the gap it has
    without this offset if it is already closer. The lines and the rest of the face keep
    their offset. Returns {morph: {side: t}}."""
    ecart_min = ECART_SOURCIL_MIN if ecart_min is None else ecart_min
    Mo = Modele(chemin=chemin, pmx=P)
    fs, fl = faces_sourcils_traits(Mo)
    if not len(fs) or not len(fl):
        return {}
    Es, El = encre(Mo, fs, dossiers, journal), encre(Mo, fl, dossiers, journal)
    iod = iod_globes(Mo)
    V0 = Mo.pos
    cs, cl = _cotes(V0, Es), _cotes(V0, El)
    # all eyebrow vertices of one side (not only those of inked triangles):
    # one common factor, so the eyebrow does not tear
    vsrc = np.unique(Mo.t["faces"][fs])
    vcote = {"gauche": vsrc[V0[vsrc, 0] > 0], "droit": vsrc[V0[vsrc, 0] < 0]}
    out = {}
    for m in P["morphes"]:
        if m["type"] != 1 or m["panneau"] not in (1, 2, 3):
            continue
        D = np.zeros_like(V0)
        for v, d in m["decalages"]:
            D[int(v)] += d
        pl = _points(V0 + D, El)
        for c in ("gauche", "droit"):
            es = (Es[0][cs[c]], Es[1][cs[c]])
            vs = vcote[c]
            if not len(vs) or np.abs(D[vs]).max() < 1e-6:
                continue
            ligne = pl[cl[c]]

            Dt = D.copy()
            Dt[vs] = 0.0

            def ecart(t):
                Dt[vs] = t * D[vs]
                return ecart_points(_points(V0 + Dt, es), ligne) / iod
            e1 = ecart(1.0)
            cible = min(ecart_min, ecart(0.0))
            if e1 >= cible:
                continue
            lo, hi = 0.0, 1.0
            for _ in range(25):
                mi = 0.5 * (lo + hi)
                lo, hi = (mi, hi) if ecart(mi) >= cible else (lo, mi)
            t = lo
            ens = set(int(v) for v in vs)
            m["decalages"] = [(v, tuple(t * x for x in d) if int(v) in ens else d) for v, d in m["decalages"]]
            out.setdefault(m["nom"], {})[c] = round(t, 4)
            journal("  guard %-6s %-6s: eyebrow offset x %.3f (gap %.4f -> %.4f)" % (m["nom"], c, t, e1, ecart(t)))
    return out


def deformer_sourcils(P, longueur=1.0, epaisseur=1.0, arc=0.0, hausse=0.0, inclinaison=0.0, eps=1e-4):
    """Deforms the eyebrow vertices (眉, 眉毛) of PMX P in place: longueur (length factor, from
    the middle), epaisseur (thickness factor, around the centre line), arc (rise of the middle,
    as a fraction of the half-length), hausse (raise, same unit), inclinaison (tilt in degrees,
    outer end upwards).
    Each morph offset d on an eyebrow vertex becomes J d, J being the local Jacobian of the
    deformation (central differences); normals go through J^-T. Returns the measurements,
    including the error |f(p+d) - f(p) - J d| over all morphs."""
    Mo = Modele(pmx=P)
    vs = Mo.sommets(SOURCILS)
    peau = Mo.pos[Mo.sommets(PEAUX)]
    if not len(vs):
        raise RuntimeError("no eyebrows (%s)" % ", ".join(SOURCILS))
    par = dict(longueur=longueur, epaisseur=epaisseur, arc=arc, hausse=hausse, inclinaison=inclinaison)
    S = P["sommets"]
    pos = np.array([S["pos"][v] for v in vs], float)
    nor = np.array([S["nor"][v] for v in vs], float)
    J = np.zeros((len(vs), 3, 3))
    nouv = np.zeros_like(pos)
    info = {"parametres": par, "cotes": {}}
    sc = {}
    for cote, nom in ((1, "gauche"), (-1, "droit")):
        k = np.where(pos[:, 0] * cote > 0)[0]
        if len(k) < 4:
            continue
        so = Sourcil(pos[k], peau, cote)
        sc[nom] = (so, k)
        nouv[k] = so.f(pos[k], **par)
        for j in range(3):
            e = np.zeros(3)
            e[j] = eps
            J[k, :, j] = (so.f(pos[k] + e, **par) - so.f(pos[k] - e, **par)) / (2 * eps)
        info["cotes"][nom] = {"sommets": int(len(k)), "demi_longueur": so.L, "peau_ajustee": so.n_peau}
    # vertex morphs
    ou = {int(v): i for i, v in enumerate(vs)}
    cote_de = np.zeros(len(vs), dtype=object)
    for nom, (so, k) in sc.items():
        cote_de[k] = nom
    err, dmax, touches = 0.0, 0.0, []
    for m in P["morphes"]:
        if m["type"] != 1:
            continue
        dec, t = [], False
        for v, d in m["decalages"]:
            i = ou.get(int(v))
            if i is None:
                dec.append((v, d))
                continue
            d = np.asarray(d, float)
            d2 = J[i] @ d
            so = sc[cote_de[i]][0]
            exact = so.f(pos[i:i + 1] + d, **par)[0] - nouv[i]
            err = max(err, float(np.linalg.norm(exact - d2)))
            dmax = max(dmax, float(np.linalg.norm(d)))
            dec.append((v, tuple(float(x) for x in d2)))
            t = True
        if t:
            m["decalages"] = dec
            touches.append(m["nom"])
    # vertices and normals
    for i, v in enumerate(vs):
        n = np.linalg.solve(J[i].T, nor[i]) if abs(np.linalg.det(J[i])) > 1e-9 else nor[i]
        n /= max(np.linalg.norm(n), 1e-12)
        S["pos"][v] = tuple(float(x) for x in nouv[i])
        S["nor"][v] = tuple(float(x) for x in n)
    dep = np.linalg.norm(nouv - pos, axis=1)
    info.update({"sommets": int(len(vs)), "deplacement_max": float(dep.max()), "morphes": touches,
                 "decalage_max": dmax, "ecart_jacobienne": err})
    print("eyebrows: %d vertices, moved by %.3f at most; %d morphs carried by the Jacobian; "
          "error |f(p+d)-f(p)-J d| %.4f for an offset of %.3f" % (len(vs), dep.max(), len(touches), err, dmax))
    for nom, c in info["cotes"].items():
        print("  %s: %d vertices, half-length %.3f, skin fitted on %d vertices" % (nom, c["sommets"], c["demi_longueur"], c["peau_ajustee"]))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("controle", "reparer", "pieces", "sourcil", "garde", "ecarts"))
    ap.add_argument("--pmx", required=True)
    ap.add_argument("--recette", help="recette.json of the composed face (empty-morph check, repair, parts)")
    ap.add_argument("--recalage", help="recalage.npz of the face pass (otherwise searched in sur-*/)")
    ap.add_argument("--sortie")
    ap.add_argument("--json")
    ap.add_argument("--textures-pour", help="folder where the PMX will be placed: texture paths rewritten relative to it "
                                            "(default: --sortie); textures also searched there (ecarts, garde)")
    ap.add_argument("--ecart-min", type=float, default=None,
                    help="eyebrow guard: minimal gap to the eye line (interocular distances; default ECART_SOURCIL_MIN)")
    ap.add_argument("--sans-garde", action="store_true", help="sourcil: no guard after reshaping")
    for k, v in (("longueur", 1.0), ("epaisseur", 1.0), ("arc", 0.0), ("hausse", 0.0), ("inclinaison", 0.0)):
        ap.add_argument("--" + k, type=float, default=v)
    a = ap.parse_args(argv)
    import pmx_ecrire
    if a.action == "controle":
        if a.recette:
            Fi = Fini(a.pmx, a.recette, a.recalage)
            Mo = Fi.Mo
            noms = list(dict.fromkeys(Mo.morphes_visage() + noms_recette(Fi.dons, Fi.recette, Fi.M, Fi.wv)))
            L = controle(Mo, noms, attendus(Fi, noms))
        else:
            Mo = Modele(a.pmx)
            L = controle(Mo)
        imprimer(L)
        if a.json:
            _ecrire_json(a.json, L)
        return
    dossiers = [a.textures_pour] if a.textures_pour else []
    if a.action == "ecarts":
        e = mesurer_ecarts(Modele(a.pmx), dossiers=dossiers)
        for n, v in sorted(e.items(), key=lambda kv: min(kv[1].values())):
            print("%-10s left %.4f  right %.4f%s" % (n, v["gauche"], v["droit"],
                                                       "  < %.3f" % ECART_SOURCIL_MIN if min(v.values()) < ECART_SOURCIL_MIN - 1e-4 else ""))
        if a.json:
            _ecrire_json(a.json, e)
        return
    os.makedirs(a.sortie, exist_ok=True)
    sortie = os.path.join(a.sortie, "resultat.pmx")
    if a.action in ("reparer", "pieces"):
        Fi = Fini(a.pmx, a.recette, a.recalage)
        print("registration %s; scale %.4f (error %.2g); sorting %s; parts: %s" % (
            Fi.chemin_rec, Fi.s, Fi.residu_place, Fi.rangement,
            ", ".join("%s/%s %.3f" % (p["nom"], p["donneur"], p["ecart"]) for p in Fi.pieces)))
        rap = reparer(Fi) if a.action == "reparer" else retirer_pieces(Fi)
        P = Fi.P
        info = {"source": a.pmx, "recette": a.recette, "recalage": Fi.chemin_rec, "rangement": Fi.rangement,
                "morphes" if a.action == "reparer" else "pieces": rap}
    else:
        P = PMX.lire(a.pmx)
        info = {}
        if a.action == "sourcil":
            info = deformer_sourcils(P, longueur=a.longueur, epaisseur=a.epaisseur, arc=a.arc, hausse=a.hausse,
                                     inclinaison=a.inclinaison)
        if a.action == "garde" or not a.sans_garde:
            info["garde"] = {"ecart_min": a.ecart_min if a.ecart_min is not None else ECART_SOURCIL_MIN,
                             "facteurs": garder_sourcils(P, a.ecart_min, chemin=a.pmx, dossiers=dossiers)}
        info["source"] = a.pmx
    n = textures_relatives(P, a.pmx, os.path.abspath(a.textures_pour or a.sortie), dossiers=dossiers)
    if n:
        print("  %d texture paths rewritten relative to %s" % (n, a.textures_pour or a.sortie))
    tmp = sortie + ".tmp.pmx"
    pmx_ecrire.ecrire(P, tmp)
    os.replace(tmp, sortie)
    _ecrire_json(os.path.join(a.sortie, a.action + ".json"), info)
    print("written: %s" % sortie)


if __name__ == "__main__":
    main()

"""Hairstyles and ears: a character's hair (with what it wears on its head:
hat, ornaments, animal ears), its bones and its physics, placed on another character's head.

    python3 -I outils/tete/coiffure.py extraire --biblio <biblio> [--travaux 4] [--seuls Jane,Ellen]
    python3 -I outils/tete/coiffure.py poser --biblio <biblio> --cheveux Miyabi --sur Evelyn \
        --sortie <dossier> [--oreilles Nekomata|aucune] [--cible <pmx>] [--sans-planche]
    python3 -I outils/tete/coiffure.py planche ...same options...   (only redoes the render)

Measurements on the 49 heads (06/11): the face skin covers only two thirds of the top of the
head, the hair (or a hat) fills the rest; human ears are small skin bumps (4 to 23 vertices)
hidden under the hair on every head: the target's ears are kept. Animal ears are weighted to
bones of their own (耳, Ear, Mimi): these are what --oreilles swaps.

Extract, for each character in the library (<biblio>/<P>/coiffure.npz, coiffure.json):
- hairstyle: the hair materials (髮, 髪, 发) whole and, outside the skin and the face parts, the
  connected pieces weighted to the head (頭 and its descendants), except those worn in front of
  the face (glasses, masks, beauty marks);
- ears: the pieces of the hairstyle weighted mostly to ear bones; separable when they are not
  attached to the rest;
- hairstyle bones: those that weigh only on it, their tips and their weightless roots;
- coverage: share of the rays cast upward from the middle of the head that hit the skin or the
  hairstyle (the floor a placed hairstyle must reach).

Placing A on B: B's hairstyle (and its rigid bodies) is removed; A's goes from A's frame to B's
through the pivot head (similarity from each head to the pivot, a similarity fitted on the skull
and face-outline points, without the features, then their local offset); the vertices that enter
B's skin are pushed out of it, the push spread along the strands. A's bones, rigid bodies and
joints follow the same displacement; the weights stay A's, on B's bones of the same name
(頭, 首...), otherwise on the copied bones. --oreilles C: A's separable ears are removed and C's
are placed, sunk into the hairstyle as they were on C.
Outputs: resultat.pmx, mesures.json, planche_textures.png (front, 3/4, profile, back).
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.join(ICI, "..", "atlas"))

import bibliotheque as BI  # noqa: E402
import geom as G  # noqa: E402
import pivot as PV  # noqa: E402
import pmx_ecrire  # noqa: E402
import reflets as RF  # noqa: E402
import retarget_tete as RT  # noqa: E402
import tete as TT  # noqa: E402

CHEVEUX = ("髮", "髪", "发")
# ear bones: names containing 耳 / Ear / Mimi, except earrings, earphones and ear strands
OREILLE_OUI = ("耳", "ear", "mimi")
OREILLE_NON = ("ring", "phone", "耳坠", "耳机", "耳饰", "earhair", "tea_", "forearm", "figure")
# face parts, neck parts and head skin (肌: the human ears, separate from the face on 33 heads)
# under other names than the library's: they stay with the head, not with the hairstyle
VISAGE = ("口", "舌", "牙", "齿", "歯", "目", "瞳", "睫", "眉", "表情", "痣", "脖", "首", "肌")
POIDS_TETE = 0.8      # a non-hair piece belongs to the hairstyle if it weighs on average on the head
POIDS_OREILLE = 0.5   # a vertex belongs to the ear above this weight on the ear bones
PART_OREILLE = 0.6    # a piece is an ear if this share of its vertices is
DEVANT = 0.25         # eye distances: further in front of the eyes than this, within the
                      # face frame, a piece is worn on the face
POINTS_PASSAGE, VOISINS_PASSAGE = 160, 12
MARGE_PEAU = 0.02     # eye distances: the hair stays at least this far outside the skin
PROFOND_MAX = 0.6     # beyond this, an "inside" vertex is far from the skin: not an entry
BIAIS = 0.5           # cosine with the normal below which a point is beyond the edge of the skin
RAYONS = 600


def journal(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =================================================================== detection

def _descendants(par, graines):
    d = np.zeros(len(par), dtype=bool)
    d[list(graines)] = True
    for _ in range(128):
        nd = d | ((par >= 0) & d[np.maximum(par, 0)])
        if (nd == d).all():
            break
        d = nd
    return d


def _os_oreille(noms, par):
    g = [i for i, n in enumerate(noms) if any(k in n.lower() for k in OREILLE_OUI)
         and not any(k in n.lower() for k in OREILLE_NON)]
    return _descendants(par, g) if g else np.zeros(len(noms), dtype=bool)


def _masse(t, sommets, nb_os):
    """Total weight carried by each bone over these vertices."""
    o, w = t["os"][sommets], t["poids"][sommets]
    m = np.zeros(nb_os)
    ok = o >= 0
    np.add.at(m, o[ok], w[ok])
    return m


def couverture(V_peau, F_peau, T_autre, centre, y_yeux, iod):
    """Share of the rays cast upward from the middle of the head (at eye height) that hit the
    skin or T_autre (triangles)."""
    rng = np.random.default_rng(0)
    D = rng.normal(size=(RAYONS, 3))
    D /= np.linalg.norm(D, axis=1)[:, None]
    D = D[D[:, 1] > 0.2]
    O = np.repeat(np.array([centre[0], y_yeux, centre[2]])[None], len(D), 0)
    t = RT._rayons_touchent(O, D, V_peau[F_peau], 5 * iod) if V_peau is not None else np.zeros(len(D), bool)
    if len(T_autre) and (~t).any():
        t[~t] = RT._rayons_touchent(O[~t], D[~t], T_autre, 5 * iod)
    return float(t.mean())


def couverture_crane(V_peau, F_peau, T_cheveux, centre, y_yeux, iod):
    """Share of the skull directions covered by the hair: among the rays of couverture() that miss
    the skin (V_peau, F_peau: the face, whose skin stops under the hair), those that hit T_cheveux.
    Counting every ray took a higher forehead for a bare skull: on the base head the lost rays all
    went out through the face skin."""
    rng = np.random.default_rng(0)
    D = rng.normal(size=(RAYONS, 3))
    D /= np.linalg.norm(D, axis=1)[:, None]
    D = D[D[:, 1] > 0.2]
    O = np.repeat(np.array([centre[0], y_yeux, centre[2]])[None], len(D), 0)
    crane = ~RT._rayons_touchent(O, D, V_peau[F_peau], 5 * iod)
    if not crane.any() or not len(T_cheveux):
        return 1.0 if not crane.any() else 0.0
    return float(RT._rayons_touchent(O[crane], D[crane], T_cheveux, 5 * iod).mean())


def reperer(D):
    """Hairstyle, ears and hairstyle bones of a head D (TT.Tete)."""
    t, F, mf = D.t, D.faces, D.mat_face
    noms_os = D.noms_os
    par = np.array([o["parent"] for o in D.pmx["os"]])
    poids_tete = D.poids_tete
    cheveux = [i for i, n in enumerate(D.noms_mat) if any(c in n for c in CHEVEUX)]
    visage = [i for i, n in enumerate(D.noms_mat) if any(k in n for k in VISAGE) and i not in cheveux]
    exclus = set(D.i_peau) | set(D.i_pieces) | set(cheveux) | set(visage)
    autres = np.array([mf[f] not in exclus for f in range(len(F))])
    lab = G.composantes_faces(F, autres)
    coif = np.zeros(len(F), dtype=bool)
    lab_ch = G.composantes_faces(F, np.isin(mf, cheveux))
    for k in range(lab_ch.max() + 1 if len(cheveux) else 0):
        fk = np.where(lab_ch == k)[0]
        if poids_tete[np.unique(F[fk])].max() > 0.1:
            coif[fk] = True
    yeux = np.array([D.centre_oeil[c] for c in "gd"])
    z_yeux, y_yeux = yeux[:, 2].mean(), yeux[:, 1].mean()
    menton = D.V[D.reperes["menton"]]
    haut_sourcils = max(p[1] for k, p in D.points_sourcils.items() if k.endswith("_haut"))
    z_milieu = float(D.V[:, 2].mean())
    portes_visage = []
    for k in range(lab.max() + 1):
        fk = np.where(lab == k)[0]
        vk = np.unique(F[fk])
        if poids_tete[vk].mean() < POIDS_TETE:
            continue
        c = D.pos[vk].mean(0)
        # in front of the eyes (glasses, masks) or in the front half of the head (mouth,
        # teeth, tongue, eyeballs), within the face frame: worn by the face
        # if it reaches below the top of the eyebrows: a forehead brooch (Yixuan, Seed) or a
        # hair clip (Dialyn) stays entirely above, and goes with the hairstyle
        devant = (c[2] < max(z_yeux - DEVANT * D.iod, z_milieu) and abs(c[0]) < 1.2 * D.iod
                  and menton[1] - 0.2 * D.iod < c[1] < y_yeux + 0.8 * D.iod
                  and D.pos[vk][:, 1].min() < haut_sourcils)
        if devant:
            portes_visage.append(D.noms_mat[mf[fk[0]]])
            continue
        coif[fk] = True
    # ears: connected pieces of the hairstyle weighted to the ear bones
    oo = _os_oreille(noms_os, par)
    o, w = t["os"], t["poids"]
    w_or = np.clip((w * ((o >= 0) & oo[np.maximum(o, 0)])).sum(1), 0, 1)
    lab_c = G.composantes_faces(F, coif)
    oreille = np.zeros(len(F), dtype=bool)
    liees = 0
    for k in range(lab_c.max() + 1 if coif.any() else 0):
        fk = np.where(lab_c == k)[0]
        vk = np.unique(F[fk])
        part = float((w_or[vk] > POIDS_OREILLE).mean())
        if part >= PART_OREILLE:
            oreille[fk] = True
        elif part > 0.05:
            liees += int((w_or[F[fk]] > POIDS_OREILLE).all(1).sum())
    # hairstyle bones: weigh on it and almost nowhere else
    vc = np.unique(F[coif])
    hors = np.setdiff1d(np.arange(len(D.pos)), vc)
    Wc, Wn = _masse(t, vc, len(noms_os)), _masse(t, hors, len(noms_os))
    graines = (Wc > 0) & (Wn <= 0.02 * Wc)
    sans_poids = (Wc + Wn) <= 0
    os_c = graines.copy()
    # weightless tips below the hairstyle bones, weightless roots above
    for _ in range(128):
        nd = os_c | (sans_poids & (par >= 0) & os_c[np.maximum(par, 0)])
        if (nd == os_c).all():
            break
        os_c = nd
    for i in np.where(graines)[0]:
        p = par[i]
        while p >= 0 and sans_poids[p] and not os_c[p] and noms_os[p] != "頭":
            os_c[p] = True
            p = par[p]
    # own coverage
    Tc = D.pos[F[coif]]
    cov = couverture(D.V, D.F, Tc, D.V.mean(0), y_yeux, D.iod)
    cov_c = couverture_crane(D.V, D.F, Tc, D.V.mean(0), y_yeux, D.iod)
    # sunken ears: share of their vertices inside the hairstyle (without them) or the skin
    dedans_or = None
    if oreille.any():
        reste = coif & ~oreille
        dedans_or = part_dedans(D.pos[np.unique(F[oreille])], D, reste)
    return {"coiffure": coif, "oreilles": oreille, "os": os_c, "couverture": cov,
            "couverture_cheveux": cov_c, "dedans_oreilles": dedans_or, "oreilles_liees": liees, "portes_visage": sorted(set(portes_visage)),
            "materiaux": sorted({D.noms_mat[i] for i in np.unique(mf[coif])})}


def _surface(D, faces):
    """Vertices, faces and normals (from the PMX) of a selection of D's raw faces."""
    vs, Fl = G.sous_maillage(D.faces, faces)
    return D.pos[vs], Fl, D.t["nor"][vs]


def signee(P, V, F, N):
    """Signed distance from P to the surface (V, F): positive on the normals' side."""
    if not len(F):
        return np.full(len(P), np.inf), np.zeros((len(P), 3))
    q, f, b, d = G.plus_proche(P, V, F)
    n = (b[:, :, None] * N[F[f]]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    return _cote(P, q, n) * d, n


def _cote(P, q, n):
    """Side of the surface where P is, q being its nearest point and n the normal: +1 outside.
    A head's skin is open (the face stops under the hair): a point beyond its edge, which sees
    that edge at a slant, is neither inside nor outside. Counting it as inside pushed it "out of
    the skin" in one big jump, into two bumps on top of the skull, like ears."""
    c = ((P - q) * n).sum(1)
    s = np.sign(c)
    s[(s == 0) | (np.abs(c) < BIAIS * np.linalg.norm(P - q, axis=1))] = 1
    return s


def part_dedans(P, D, faces_coif):
    """Share of the points P inside the hairstyle (faces_coif) or D's skin."""
    Vh, Fh, Nh = _surface(D, faces_coif)
    sh, _ = signee(P, Vh, Fh, Nh)
    Vp, Fp, Np = _surface(D, np.isin(D.mat_face, D.i_peau))
    sp, _ = signee(P, Vp, Fp, Np)
    return float(((sh < 0) & (sh > -PROFOND_MAX * D.iod) | (sp < 0) & (sp > -PROFOND_MAX * D.iod)).mean())


# =================================================================== extraction

def extraire_un(args):
    nom, biblio = args
    with open(os.path.join(biblio, nom, "fiche.json"), encoding="utf-8") as fh:
        fiche = json.load(fh)  # pmx as is in coiffure.json (relative for Outerplane)
    D = TT.Tete(BI.lire_fiche(biblio, nom)["pmx"])
    r = reperer(D)
    np.savez_compressed(os.path.join(biblio, nom, "coiffure.npz"), coiffure=r["coiffure"],
                        oreilles=r["oreilles"], os=r["os"])
    info = {"personnage": nom, "pmx": fiche["pmx"], "materiaux": r["materiaux"],
            "faces_coiffure": int(r["coiffure"].sum()), "faces_oreilles": int(r["oreilles"].sum()),
            "oreilles_liees_faces": r["oreilles_liees"], "os_coiffure": int(r["os"].sum()),
            "rigides_coiffure": int(sum(1 for c in D.pmx["rigides"] if c["os"] >= 0 and r["os"][c["os"]])),
            "couverture": round(r["couverture"], 3),
            "dedans_oreilles": None if r["dedans_oreilles"] is None else round(r["dedans_oreilles"], 3),
            "portes_visage": r["portes_visage"]}
    BI._ecrire_json(os.path.join(biblio, nom, "coiffure.json"), info)
    return info


def extraire(biblio, travaux, seuls=None):
    with open(os.path.join(biblio, "index.json"), encoding="utf-8") as fh:
        idx = json.load(fh)
    noms = sorted(idx["parties"]["yeux"])
    if seuls:
        noms = [n for n in noms if n in seuls]
    res = list(BI.carte(extraire_un, [(n, biblio) for n in noms], travaux))
    for r in res:
        journal("%-12s hairstyle %5d faces, ears %4d (attached %d), bones %3d, rigid bodies %3d, coverage %.2f" % (
            r["personnage"], r["faces_coiffure"], r["faces_oreilles"], r["oreilles_liees_faces"],
            r["os_coiffure"], r["rigides_coiffure"], r["couverture"]))
    # merge: the hairstyles already recorded for the other characters are kept
    idx.setdefault("coiffure", {}).update({r["personnage"]: {"couverture": r["couverture"],
                                                             "oreilles": r["faces_oreilles"] > 0} for r in res})
    idx["coiffure"] = dict(sorted(idx["coiffure"].items()))
    BI._ecrire_json(os.path.join(biblio, "index.json"), idx)


# =================================================================== mapping from one head to another

class Passage:
    """Mapping from A's frame to B's through the pivot head: similarity from each head to the
    pivot, a similarity fitted (Procrustes) on the reliable points, then the local offset of
    these points interpolated (weighted neighbours). A thin-plate spline fitted on the face alone
    extrapolated far from it (top and back of the skull) with a shear: between two very
    different faces (Dialyn on Yixuan), the top of the hairstyle ended up below the eyes."""

    def __init__(self, biblio, nomA, nomB, pivot, masques):
        za, zb = (np.load(os.path.join(biblio, n, "partie.npz")) for n in (nomA, nomB))
        self.sA, self.csA, self.cdA = float(za["echelle"]), za["c_src"], za["c_dst"]
        self.sB, self.csB, self.cdB = float(zb["echelle"]), zb["c_src"], zb["c_dst"]
        XA, XB = za["X"].astype(float), zb["X"].astype(float)
        # points reliable on both sides, away from the features (eyes, eyebrows, nose, mouth)
        traits = np.zeros(pivot.n)
        for z, m in masques.items():
            if z != BI.BASE:
                traits = np.maximum(traits, m)
        ok = np.where(za["fiable"] & zb["fiable"] & (traits < 0.05))[0]
        ok = ok[_echantillon(pivot.V[ok], POINTS_PASSAGE)]
        self.src, self.dst = XA[ok], XB[ok]
        self.points = int(len(ok))
        # similarity src -> dst (Umeyama)
        ma, mb = self.src.mean(0), self.dst.mean(0)
        a, b = self.src - ma, self.dst - mb
        U, S, Vt = np.linalg.svd(b.T @ a)
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.diag([1.0, 1.0, d])
        self.R = U @ D @ Vt
        self.s = float((S * np.diag(D)).sum() / (a * a).sum())
        self.t = mb - self.s * self.R @ ma
        self.ecart = self.dst - self._sim(self.src)
        self.echelle = self.sA / self.sB * self.s

    def _sim(self, q):
        return self.s * q @ self.R.T + self.t

    def __call__(self, p):
        q = self.sA * (np.asarray(p, dtype=float) - self.csA) + self.cdA
        if len(q):
            q = self._sim(q) + G.shepard(self.src, self.ecart, q, k=VOISINS_PASSAGE)
        return (q - self.cdB) / self.sB + self.csB


def _echantillon(P, n):
    """n points far from one another (farthest first)."""
    if len(P) <= n:
        return np.arange(len(P))
    choisi = [int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))]
    d = np.linalg.norm(P - P[choisi[0]], axis=1)
    for _ in range(n - 1):
        j = int(np.argmax(d))
        choisi.append(j)
        d = np.minimum(d, np.linalg.norm(P - P[j], axis=1))
    return np.array(choisi)


def sortir_de_peau(P, F_loc, Vp, Fp, Np, iod, tours=4, signee=signee, repos=None):
    """Vertices P (mesh F_loc) that are inside the skin, or closer than MARGE_PEAU, pushed out
    along the skin normal; the push spreads to the neighbours (whole strands).

    `repos`: signed distance from each vertex to the skin of its original head. A vertex is
    then pushed out only if it is closer or deeper than on its own head: roots planted under
    the scalp (hand-made haircuts) and strands lying on the skin stay as they were made, and a
    hairstyle put back on its own head does not move."""
    rep, inv = G.souder(P, 1e-6)
    W = P[rep].copy()
    marge = MARGE_PEAU * iod if repos is None else np.minimum(MARGE_PEAU * iod, np.asarray(repos, float)[rep] - 1e-4 * iod)
    Fw = inv[F_loc]
    Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
    E, _, _ = G.aretes(Fw)
    a, b = np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]])
    nv = np.maximum(np.bincount(a, minlength=len(W)), 1)[:, None]
    entres = 0
    for tour in range(tours):
        s, n = signee(W, Vp, Fp, Np)
        e = (s < marge) & (s > -PROFOND_MAX * iod)
        if tour == 0:
            entres = int(e.sum())
        if not e.any():
            break
        u = np.zeros_like(W)
        u[e] = ((marge[e] if repos is not None else marge) - s[e])[:, None] * n[e]
        # spreading: each vertex takes the strongest of the pushes averaged around it
        for _ in range(12):
            m = np.zeros_like(u)
            np.add.at(m, a, u[b])
            m /= nv
            plus = np.linalg.norm(m, axis=1) > np.linalg.norm(u, axis=1)
            u[plus] = m[plus]
        W += u
    s, _ = signee(W, Vp, Fp, Np)
    reste = int(((s < 0) & (s > -PROFOND_MAX * iod)).sum())
    return W[inv], {"sommets_entres": entres, "sommets_dedans_apres": reste, "sommets": int(len(W))}


# =================================================================== placing

def _rel(chemin_tex, dossier, sortie):
    return RT._texture_rel(chemin_tex, dossier, sortie)


def poser(biblio, nomA, nomB, sortie, oreilles=None, cible=None):
    os.makedirs(sortie, exist_ok=True)
    fiche = lambda n: BI.lire_fiche(biblio, n)  # noqa: E731
    A = TT.Tete(fiche(nomA)["pmx"])
    B = TT.Tete(cible or fiche(nomB)["pmx"])
    pivot = BI.charger_pivot()
    M = BI.masques(pivot)
    rA, rB = reperer(A), reperer(B)
    pAB = Passage(biblio, nomA, nomB, pivot, M)
    journal("hairstyle of %s (%d faces) on %s, replacing its own (%d faces)" % (
        nomA, rA["coiffure"].sum(), nomB, rB["coiffure"].sum()))
    pieces = [(A, rA, rA["coiffure"].copy(), pAB, "coiffure")]
    info = {"cheveux": nomA, "sur": nomB, "points_passage": pAB.points, "echelle": pAB.echelle}
    # "+" highlights baked into the hair texture: their duplicate faces are removed
    retire, tex_reflets = RF.cuire(A, pieces[0][2], sortie)
    pieces[0][2][retire] = False
    info["reflets_cuits"] = int(retire.sum())
    if oreilles:
        pieces[0][2][rA["oreilles"]] = False
        info["oreilles_retirees"] = int(rA["oreilles"].sum())
        if oreilles != "aucune":
            C = TT.Tete(fiche(oreilles)["pmx"])
            rC = reperer(C)
            if not rC["oreilles"].any():
                raise ValueError("%s: no separable ears" % oreilles)
            pieces.append((C, rC, rC["oreilles"].copy(), Passage(biblio, oreilles, nomB, pivot, M), "oreilles"))
            info["oreilles"] = oreilles

    Pb = B.pmx
    P = copy.deepcopy(Pb)
    nuv = Pb["n_uv_add"]
    # --- B's skin (to push the hair out): the face skin and the other skins of the head
    peauB = np.isin(B.mat_face, B.i_peau)
    Vp, Fp, Np = _surface(B, peauB)
    # --- B's hairstyle removed: faces dropped, rigid bodies and joints of its bones removed
    garde_fB = ~rB["coiffure"]
    retire_rig = [i for i, c in enumerate(Pb["rigides"]) if c["os"] >= 0 and rB["os"][c["os"]]]
    idx_rig = np.full(len(Pb["rigides"]), -1)
    k = 0
    for i in range(len(Pb["rigides"])):
        if i not in set(retire_rig):
            idx_rig[i] = k
            k += 1
    P["rigides"] = [copy.deepcopy(c) for i, c in enumerate(Pb["rigides"]) if idx_rig[i] >= 0]
    P["joints"] = []
    for j in Pb["joints"]:
        if idx_rig[j["a"]] >= 0 and idx_rig[j["b"]] >= 0:
            j = copy.deepcopy(j)
            j["a"], j["b"] = int(idx_rig[j["a"]]), int(idx_rig[j["b"]])
            P["joints"].append(j)
    for m in P["morphes"]:
        if m["type"] == 10:
            m["decalages"] = [(int(idx_rig[d[0]]),) + tuple(d[1:]) for d in m["decalages"] if idx_rig[d[0]] >= 0]
    rig_de_os_B = {}
    for i, c in enumerate(P["rigides"]):
        rig_de_os_B.setdefault(P["os"][c["os"]]["nom"] if c["os"] >= 0 else None, i)

    S = {k: list(v) for k, v in Pb["sommets"].items()}
    faces_par_mat = {}
    for f in np.where(garde_fB)[0]:
        faces_par_mat.setdefault(("B", int(B.mat_face[f])), []).append(tuple(int(v) for v in B.faces[f]))
    ordre = [("B", i) for i in range(len(Pb["materiaux"]))]
    P["textures"] = [_rel(x, os.path.dirname(B.chemin), sortie) for x in Pb["textures"]]
    noms_B = {o["nom"]: i for i, o in enumerate(Pb["os"])}
    tete_B = noms_B.get("頭", 0)
    nouveaux_os = []
    mesures_pieces = []
    for n_p, (D, rD, faces, passe, quoi) in enumerate(pieces):
        Pd = D.pmx
        cle = "D%d" % n_p
        # bones: the hairstyle's are copied (renamed if B already has that name), the others
        # mapped by name onto B
        carte = np.full(len(Pd["os"]), tete_B)
        copies = np.where(rD["os"])[0]
        base = len(P["os"])
        for j, i in enumerate(copies):
            carte[i] = base + j
        for i, o in enumerate(Pd["os"]):
            if not rD["os"][i]:
                carte[i] = noms_B.get(o["nom"], tete_B)
        noms_pris = {o["nom"] for o in P["os"]}
        for i in copies:
            o = copy.deepcopy(Pd["os"][i])
            if o["nom"] in noms_pris:
                o["nom"] = o["nom"] + "_" + D.nom[:8]
            noms_pris.add(o["nom"])
            o["pos"] = tuple(passe(np.array([o["pos"]]))[0])
            o["parent"] = int(carte[o["parent"]]) if o["parent"] >= 0 else -1
            if o["queue_os"] is not None and o["queue_os"] >= 0:
                o["queue_os"] = int(carte[o["queue_os"]])
            elif o["queue_dec"] is not None:
                o["queue_dec"] = tuple(passe.echelle * np.asarray(o["queue_dec"]))
            if o["ajout"] is not None:
                o["ajout"]["source"] = int(carte[o["ajout"]["source"]])
            if o["ik"] is not None:
                o["ik"]["cible"] = int(carte[o["ik"]["cible"]])
                for li in o["ik"]["liens"]:
                    li["os"] = int(carte[li["os"]])
            P["os"].append(o)
            nouveaux_os.append(len(P["os"]) - 1)
        # vertices: position mapped onto B, then pushed out of B's skin
        vs, Fl = G.sous_maillage(D.faces, faces)
        pos = passe(D.pos[vs])
        if quoi == "coiffure":
            repos, _ = signee(D.pos[vs], *_surface(D, np.isin(D.mat_face, D.i_peau)))
            pos, sortie_peau = sortir_de_peau(pos, Fl, Vp, Fp, Np, B.iod, repos=repos * passe.echelle)
        else:
            pos, sortie_peau = poser_oreilles(pos, Fl, rD, D, B, P, S, faces_par_mat, ordre, Vp, Fp, Np), {}
        nor = D.t["nor"][vs]
        n0 = len(S["pos"])
        sd = Pd["sommets"]
        for j, v in enumerate(vs):
            t = sd["type"][v]
            S["pos"].append(tuple(pos[j]))
            S["nor"].append(tuple(nor[j]))
            S["uv"].append(sd["uv"][v])
            u = list(sd["uv_add"][v])[:nuv] if Pd["n_uv_add"] else []
            S["uv_add"].append(u + [(0.0, 0.0, 0.0, 0.0)] * (nuv - len(u)))
            S["type"].append(t)
            S["os"].append(tuple(int(carte[x]) if x >= 0 else -1 for x in sd["os"][v]))
            S["poids"].append(sd["poids"][v])
            S["sdef"].append(sd["sdef"][v])
            S["contour"].append(sd["contour"][v])
        for f, (a, b, c) in zip(np.where(faces)[0], Fl):
            faces_par_mat.setdefault((cle, int(D.mat_face[f])), []).append((n0 + int(a), n0 + int(b), n0 + int(c)))
        off_tex = len(P["textures"])
        P["textures"] += [_rel(x, os.path.dirname(D.chemin), sortie) for x in Pd["textures"]]
        mats_D = sorted({int(D.mat_face[f]) for f in np.where(faces)[0]})
        ordre += [(cle, i) for i in mats_D]
        pieces[n_p] = pieces[n_p] + (off_tex,)
        # rigid bodies and joints of the hairstyle
        rig_D = {}
        for i, c in enumerate(Pd["rigides"]):
            if c["os"] >= 0 and rD["os"][c["os"]]:
                c = copy.deepcopy(c)
                c["os"] = int(carte[c["os"]])
                c["pos"] = tuple(passe(np.array([c["pos"]]))[0])
                c["taille"] = tuple(passe.echelle * np.asarray(c["taille"]))
                rig_D[i] = len(P["rigides"])
                P["rigides"].append(c)
        for j in Pd["joints"]:
            a, b = rig_D.get(j["a"]), rig_D.get(j["b"])
            if a is None and b is None:
                continue

            def ailleurs(r):
                c = Pd["rigides"][r]
                return rig_de_os_B.get(Pd["os"][c["os"]]["nom"] if c["os"] >= 0 else None, rig_de_os_B.get("頭"))
            a = a if a is not None else ailleurs(j["a"])
            b = b if b is not None else ailleurs(j["b"])
            if a is None or b is None:
                continue
            j = copy.deepcopy(j)
            j["a"], j["b"] = int(a), int(b)
            j["pos"] = tuple(passe(np.array([j["pos"]]))[0])
            j["dep_min"] = tuple(passe.echelle * np.asarray(j["dep_min"]))
            j["dep_max"] = tuple(passe.echelle * np.asarray(j["dep_max"]))
            P["joints"].append(j)
        mesures_pieces.append({"de": D.nom, "quoi": quoi, "sommets": int(len(vs)), "faces": int(faces.sum()),
                               "os_copies": int(len(copies)), "rigides_copies": len(rig_D), **sortie_peau})
    # --- materials and faces
    mats, Ftot = [], []
    src = {"B": (Pb, 0)}
    for n_p, pc in enumerate(pieces):
        src["D%d" % n_p] = (pc[0].pmx, pc[5])
    for cle in ordre:
        s_, i = cle
        Ps, off = src[s_]
        m = copy.deepcopy(Ps["materiaux"][i])
        if s_ != "B":
            for kk in ("texture", "sphere"):
                if m[kk] >= 0:
                    m[kk] += off
            if m["toon_partage"] == 0 and m["toon"] >= 0:
                m["toon"] += off
            m["nom"] = m["nom"] + "_" + pieces[int(s_[1:])][0].nom[:8]
        if s_ == "D0" and i in tex_reflets:
            P["textures"].append(os.path.relpath(tex_reflets[i], sortie).replace("/", "\\"))
            m["texture"] = len(P["textures"]) - 1
        fl = faces_par_mat.get(cle, [])
        m["n_indices"] = 3 * len(fl)
        mats.append(m)
        Ftot += fl
    P["materiaux"], P["faces"], P["sommets"] = mats, Ftot, S
    # B's material morphs: same indices (its materials stay first, in order)
    if nouveaux_os:
        P["cadres"].append({"nom": "髪_" + nomA[:8], "nom_en": "hair_" + nomA[:8], "special": 0,
                            "elements": [(0, i) for i in nouveaux_os]})
    P["commentaire"] = (P.get("commentaire") or "") + "\r\n[hairstyle: hair of %s%s]" % (
        nomA, (", ears of %s" % oreilles) if oreilles else "")
    chemin = os.path.join(sortie, "resultat.pmx")
    pmx_ecrire.ecrire(P, chemin)
    info["pieces"] = mesures_pieces
    info["os_retires_de_B"] = int(rB["os"].sum())
    info["rigides_retires_de_B"] = len(retire_rig)
    return chemin, info, rA, rB


def poser_oreilles(pos, Fl, rC, C, B, P, S, faces_par_mat, ordre, Vp, Fp, Np):
    """C's ears (already mapped onto B) sunk into the placed hairstyle as they were on C: they
    are slid toward the middle of the head until the same share of vertices is inside."""
    voulu = rC["dedans_oreilles"] or 0.2
    # surface of what is already placed on B (skin and added hairstyle)
    Vall = np.asarray(S["pos"], dtype=float)
    Nall = np.asarray(S["nor"], dtype=float)
    Fn = np.array([f for cle, fl in faces_par_mat.items() if cle[0] != "B" for f in fl], dtype=np.int64)
    centre = B.V.mean(0)
    sortie = []
    lab = G.composantes_faces(Fl, np.ones(len(Fl), dtype=bool))
    for k in range(lab.max() + 1):
        vk = np.unique(Fl[lab == k])
        Q = pos[vk]
        d = centre - Q.mean(0)
        d /= max(np.linalg.norm(d), 1e-12)

        def part(t):
            X = Q + t * d
            sh, _ = signee(X, Vall, Fn, Nall) if len(Fn) else (np.full(len(X), np.inf), None)
            sp, _ = signee(X, Vp, Fp, Np)
            lim = -PROFOND_MAX * B.iod
            return float((((sh < 0) & (sh > lim)) | ((sp < 0) & (sp > lim))).mean())
        lo, hi = -0.5 * B.iod, 1.0 * B.iod
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if part(mid) < voulu:
                lo = mid
            else:
                hi = mid
        pos[vk] = Q + 0.5 * (lo + hi) * d
        sortie.append(round(0.5 * (lo + hi) / B.iod, 3))
    return pos


# =================================================================== checks

def verifier(chemin, nomA, rA, B, rB, info):
    import pmx as PMX
    R = TT.Tete(chemin)
    t = R.t
    nouv = np.array([("_" + nomA[:8]) in R.noms_mat[i] for i in R.mat_face])
    yeux = np.array([R.centre_oeil[c] for c in "gd"])
    cov = couverture(R.V, R.F, R.pos[R.faces[nouv]], R.V.mean(0), yeux[:, 1].mean(), R.iod)
    # the skull skin counts in the coverage: a hairstyle placed too low leaves it bare
    # without the coverage dropping; the hairstyle alone must cover the skull directions as much as on A
    cov_c = couverture_crane(R.V, R.F, R.pos[R.faces[nouv]], R.V.mean(0), yeux[:, 1].mean(), R.iod)
    plancher = min(rA["couverture"], rB["couverture"])
    Vp, Fp, Np = _surface(R, np.isin(R.mat_face, R.i_peau))
    vn = np.unique(R.faces[nouv])
    s, _ = signee(R.pos[vn], Vp, Fp, Np)
    dedans = float(((s < 0) & (s > -PROFOND_MAX * R.iod)).mean())
    # bones of the added vertices: all valid
    o = t["os"][vn]
    os_ok = bool(((o < len(R.noms_os)) & (o >= -1)).all())
    rig = R.pmx["rigides"]
    rig_ok = all(-1 <= c["os"] < len(R.noms_os) for c in rig) and all(
        0 <= j["a"] < len(rig) and 0 <= j["b"] < len(rig) for j in R.pmx["joints"])
    q = G.qualite(R.pos, R.faces[nouv]) if nouv.any() else {}
    tests = {
        "relu": {"passe": True, "materiaux": len(R.noms_mat), "os": len(R.noms_os), "rigides": len(rig)},
        "crane_couvert": {"passe": cov >= plancher - 0.03, "couverture": round(cov, 3),
                          "plancher": round(plancher, 3), "A": rA["couverture"], "B": rB["couverture"]},
        "crane_sous_cheveux": {"passe": cov_c >= rA["couverture_cheveux"] - 0.05,
                               "couverture": round(cov_c, 3), "A": round(rA["couverture_cheveux"], 3)},
        "cheveux_hors_peau": {"passe": dedans <= 0.005, "part_dedans": round(dedans, 4)},
        "os_et_physique": {"passe": os_ok and rig_ok},
        "coiffure_de_B_retiree": {"passe": not any(("_" + nomA[:8]) not in n and n in rB["materiaux"] and
                                                   R.pmx["materiaux"][i]["n_indices"] > 0
                                                   for i, n in enumerate(R.noms_mat)
                                                   if any(c in n for c in CHEVEUX))},
    }
    if isinstance(q, dict) and q:
        tests["qualite_maillage"] = {"passe": True, **{k: v for k, v in q.items() if isinstance(v, (int, float))}}
    echecs = [k for k, v in tests.items() if not v["passe"]]
    return {"tests": tests, "verdict": {"passe": not echecs, "echecs": echecs}, "construction": info}


def planche(source, cible, resultat, sortie):
    cmd = ["bash", RT.FILE_BLENDER, "-b", "--factory-startup", "--python", os.path.join(ICI, "planche_textures.py"), "--",
           "--coiffure", os.path.join(sortie, "planche_textures.png"), source, os.path.dirname(source),
           cible, os.path.dirname(cible), resultat, sortie]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("textured contact sheet: Blender failed")
    return os.path.join(sortie, "planche_textures.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("extraire", "poser", "planche"))
    ap.add_argument("--biblio", required=True)
    ap.add_argument("--travaux", type=int, default=4)
    ap.add_argument("--seuls")
    ap.add_argument("--cheveux")
    ap.add_argument("--oreilles", help="character that gives its animal ears, or 'aucune' (none)")
    ap.add_argument("--sur", help="library character whose head receives the hairstyle")
    ap.add_argument("--cible", help="PMX to put the hairstyle on (default: the --sur one; e.g. a composite face placed on it)")
    ap.add_argument("--sortie")
    ap.add_argument("--sans-planche", action="store_true")
    a = ap.parse_args(argv)
    biblio = os.path.abspath(os.path.expanduser(a.biblio))
    if a.action == "extraire":
        extraire(biblio, a.travaux, a.seuls.split(",") if a.seuls else None)
        return 0
    sortie = os.path.abspath(os.path.expanduser(a.sortie))
    fiche = lambda n: BI.lire_fiche(biblio, n)["pmx"]  # noqa: E731
    if a.action == "planche":  # redo the render of an existing placement
        journal("textured contact sheet: %s" % planche(fiche(a.cheveux), a.cible or fiche(a.sur),
                                                      os.path.join(sortie, "resultat.pmx"), sortie))
        return 0
    t0 = time.time()
    chemin, info, rA, rB = poser(biblio, a.cheveux, a.sur, sortie, a.oreilles, a.cible)
    B = TT.Tete(a.cible or fiche(a.sur))
    mes = verifier(chemin, a.cheveux, rA, B, rB, info)
    mes["duree_s"] = round(time.time() - t0, 1)
    BI._ecrire_json(os.path.join(sortie, "mesures.json"), mes)
    for k, v in mes["tests"].items():
        journal("  %-24s %s" % (k, "ok" if v["passe"] else "FAILED"))
    journal("measurements: %s" % ("ALL PASS" if mes["verdict"]["passe"] else "FAILURES: " + ", ".join(mes["verdict"]["echecs"])))
    if not a.sans_planche:
        src = fiche(a.cheveux)
        journal("textured contact sheet: %s" % planche(src, B.chemin, chemin, sortie))
    return 0 if mes["verdict"]["passe"] else 1


if __name__ == "__main__":
    sys.exit(main())

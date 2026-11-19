"""Hairstyles and ears: a character's hair (with what it wears on its head:
hat, ornaments, animal ears), its bones and its physics, placed on another character's head.

    python3 -I tools/head/hairstyle.py extract --library <biblio> [--jobs 4] [--only Jane,Ellen]
    python3 -I tools/head/hairstyle.py place --library <biblio> --hair Miyabi --onto Evelyn \
        --output <dossier> [--ears Nekomata|aucune] [--target <pmx>] [--no-sheet]
    python3 -I tools/head/hairstyle.py contact_sheet ...same options...   (only redoes the render)

Measurements on the 49 heads (06/11): the face skin covers only two thirds of the top of the
head, the hair (or a hat) fills the rest; human ears are small skin bumps (4 to 23 vertices)
hidden under the hair on every head: the target's ears are kept. Animal ears are weighted to
bones of their own (耳, Ear, Mimi): these are what --ears swaps.

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
(頭, 首...), otherwise on the copied bones. --ears C: A's separable ears are removed and C's
are placed, sunk into the hairstyle as they were on C.
Outputs: result.pmx, measures.json, texture_sheet.png (front, 3/4, profile, back).
"""
import argparse
import copy
import json
import os
import subprocess
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "atlas"))

import library as BI  # noqa: E402
import geom as G  # noqa: E402
import pivot as PV  # noqa: E402
import pmx_write as pmx_write  # noqa: E402
import highlights as RF  # noqa: E402
import retarget_head as RT  # noqa: E402
import head as TT  # noqa: E402

HAIR = ("髮", "髪", "发")
# ear bones: names containing 耳 / Ear / Mimi, except earrings, earphones and ear strands
EAR_YES = ("耳", "ear", "mimi")
EAR_NON = ("ring", "phone", "耳坠", "耳机", "耳饰", "earhair", "tea_", "forearm", "figure")
# face parts, neck parts and head skin (肌: the human ears, separate from the face on 33 heads)
# under other names than the library's: they stay with the head, not with the hairstyle
FACE = ("口", "舌", "牙", "齿", "歯", "目", "瞳", "睫", "眉", "表情", "痣", "脖", "首", "肌")
WEIGHT_HEAD = 0.8      # a non-hair piece belongs to the hairstyle if it weighs on average on the head
WEIGHT_EAR = 0.5   # a vertex belongs to the ear above this weight on the ear bones
PART_EAR = 0.6    # a piece is an ear if this share of its vertices is
FRONT = 0.25         # eye distances: further in front of the eyes than this, within the
                      # face frame, a piece is worn on the face
POINTS_PASSAGE, NEIGHBORS_PASSAGE = 160, 12
MARGIN_SKIN = 0.02     # eye distances: the hair stays at least this far outside the skin
DEEP_MAX = 0.6     # beyond this, an "inside" vertex is far from the skin: not an entry
BIAS = 0.5           # cosine with the normal below which a point is beyond the edge of the skin
RAYS = 600


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =================================================================== detection

def _descendants(per, seeds):
    d = np.zeros(len(per), dtype=bool)
    d[list(seeds)] = True
    for _ in range(128):
        nd = d | ((per >= 0) & d[np.maximum(per, 0)])
        if (nd == d).all():
            break
        d = nd
    return d


def _os_ear(names, per):
    g = [i for i, n in enumerate(names) if any(k in n.lower() for k in EAR_YES)
         and not any(k in n.lower() for k in EAR_NON)]
    return _descendants(per, g) if g else np.zeros(len(names), dtype=bool)


def _mass(t, vertices, nb_os):
    """Total weight carried by each bone over these vertices."""
    o, w = t["bones"][vertices], t["weights"][vertices]
    m = np.zeros(nb_os)
    ok = o >= 0
    np.add.at(m, o[ok], w[ok])
    return m


def coverage(V_skin, F_skin, T_other, center, y_eyes, iod):
    """Share of the rays cast upward from the middle of the head (at eye height) that hit the
    skin or T_other (triangles)."""
    rng = np.random.default_rng(0)
    D = rng.normal(size=(RAYS, 3))
    D /= np.linalg.norm(D, axis=1)[:, None]
    D = D[D[:, 1] > 0.2]
    O = np.repeat(np.array([center[0], y_eyes, center[2]])[None], len(D), 0)
    t = RT._rays_touch(O, D, V_skin[F_skin], 5 * iod) if V_skin is not None else np.zeros(len(D), bool)
    if len(T_other) and (~t).any():
        t[~t] = RT._rays_touch(O[~t], D[~t], T_other, 5 * iod)
    return float(t.mean())


def coverage_skull(V_skin, F_skin, T_hair, center, y_eyes, iod):
    """Share of the skull directions covered by the hair: among the rays of coverage() that miss
    the skin (V_skin, F_skin: the face, whose skin stops under the hair), those that hit T_hair.
    Counting every ray took a higher forehead for a bare skull: on the base head the lost rays all
    went out through the face skin."""
    rng = np.random.default_rng(0)
    D = rng.normal(size=(RAYS, 3))
    D /= np.linalg.norm(D, axis=1)[:, None]
    D = D[D[:, 1] > 0.2]
    O = np.repeat(np.array([center[0], y_eyes, center[2]])[None], len(D), 0)
    skull = ~RT._rays_touch(O, D, V_skin[F_skin], 5 * iod)
    if not skull.any() or not len(T_hair):
        return 1.0 if not skull.any() else 0.0
    return float(RT._rays_touch(O[skull], D[skull], T_hair, 5 * iod).mean())


def locate(D):
    """Hairstyle, ears and hairstyle bones of a head D (TT.Head)."""
    t, F, mf = D.t, D.faces, D.mat_face
    names_os = D.names_os
    per = np.array([o["parent"] for o in D.pmx["bones"]])
    weight_head = D.weight_head
    hair = [i for i, n in enumerate(D.names_mat) if any(c in n for c in HAIR)]
    face = [i for i, n in enumerate(D.names_mat) if any(k in n for k in FACE) and i not in hair]
    excluded = set(D.i_skin) | set(D.i_parts) | set(hair) | set(face)
    others = np.array([mf[f] not in excluded for f in range(len(F))])
    lab = G.components_faces(F, others)
    hairdo = np.zeros(len(F), dtype=bool)
    lab_ch = G.components_faces(F, np.isin(mf, hair))
    for k in range(lab_ch.max() + 1 if len(hair) else 0):
        fk = np.where(lab_ch == k)[0]
        if weight_head[np.unique(F[fk])].max() > 0.1:
            hairdo[fk] = True
    eyes = np.array([D.center_eye[c] for c in "gd"])
    z_eyes, y_eyes = eyes[:, 2].mean(), eyes[:, 1].mean()
    chin = D.V[D.landmarks["chin"]]
    top_eyebrows = max(p[1] for k, p in D.points_eyebrows.items() if k.endswith("_top"))
    z_middle = float(D.V[:, 2].mean())
    worn_on_face = []
    for k in range(lab.max() + 1):
        fk = np.where(lab == k)[0]
        vk = np.unique(F[fk])
        if weight_head[vk].mean() < WEIGHT_HEAD:
            continue
        c = D.pos[vk].mean(0)
        # in front of the eyes (glasses, masks) or in the front half of the head (mouth,
        # teeth, tongue, eyeballs), within the face frame: worn by the face
        # if it reaches below the top of the eyebrows: a forehead brooch (Yixuan, Seed) or a
        # hair clip (Dialyn) stays entirely above, and goes with the hairstyle
        ahead = (c[2] < max(z_eyes - FRONT * D.iod, z_middle) and abs(c[0]) < 1.2 * D.iod
                  and chin[1] - 0.2 * D.iod < c[1] < y_eyes + 0.8 * D.iod
                  and D.pos[vk][:, 1].min() < top_eyebrows)
        if ahead:
            worn_on_face.append(D.names_mat[mf[fk[0]]])
            continue
        hairdo[fk] = True
    # ears: connected pieces of the hairstyle weighted to the ear bones
    oo = _os_ear(names_os, per)
    o, w = t["bones"], t["weights"]
    w_or = np.clip((w * ((o >= 0) & oo[np.maximum(o, 0)])).sum(1), 0, 1)
    lab_c = G.components_faces(F, hairdo)
    ear = np.zeros(len(F), dtype=bool)
    linked = 0
    for k in range(lab_c.max() + 1 if hairdo.any() else 0):
        fk = np.where(lab_c == k)[0]
        vk = np.unique(F[fk])
        part = float((w_or[vk] > WEIGHT_EAR).mean())
        if part >= PART_EAR:
            ear[fk] = True
        elif part > 0.05:
            linked += int((w_or[F[fk]] > WEIGHT_EAR).all(1).sum())
    # hairstyle bones: weigh on it and almost nowhere else
    vc = np.unique(F[hairdo])
    out_of = np.setdiff1d(np.arange(len(D.pos)), vc)
    Wc, Wn = _mass(t, vc, len(names_os)), _mass(t, out_of, len(names_os))
    seeds = (Wc > 0) & (Wn <= 0.02 * Wc)
    without_weight = (Wc + Wn) <= 0
    os_c = seeds.copy()
    # weightless tips below the hairstyle bones, weightless roots above
    for _ in range(128):
        nd = os_c | (without_weight & (per >= 0) & os_c[np.maximum(per, 0)])
        if (nd == os_c).all():
            break
        os_c = nd
    for i in np.where(seeds)[0]:
        p = per[i]
        while p >= 0 and without_weight[p] and not os_c[p] and names_os[p] != "頭":
            os_c[p] = True
            p = per[p]
    # own coverage
    Tc = D.pos[F[hairdo]]
    cov = coverage(D.V, D.F, Tc, D.V.mean(0), y_eyes, D.iod)
    cov_c = coverage_skull(D.V, D.F, Tc, D.V.mean(0), y_eyes, D.iod)
    # sunken ears: share of their vertices inside the hairstyle (without them) or the skin
    inside_or = None
    if ear.any():
        rest = hairdo & ~ear
        inside_or = fraction_inside(D.pos[np.unique(F[ear])], D, rest)
    return {"hairstyle": hairdo, "ears": ear, "bones": os_c, "coverage": cov,
            "hair_coverage": cov_c, "inside_ears": inside_or, "ears_linked": linked, "worn_on_face": sorted(set(worn_on_face)),
            "materials": sorted({D.names_mat[i] for i in np.unique(mf[hairdo])})}


def _surface(D, faces):
    """Vertices, faces and normals (from the PMX) of a selection of D's raw faces."""
    vs, Fl = G.under_mesh(D.faces, faces)
    return D.pos[vs], Fl, D.t["nor"][vs]


def signed(P, V, F, N):
    """Signed distance from P to the surface (V, F): positive on the normals' side."""
    if not len(F):
        return np.full(len(P), np.inf), np.zeros((len(P), 3))
    q, f, b, d = G.plus_near(P, V, F)
    n = (b[:, :, None] * N[F[f]]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    return _side(P, q, n) * d, n


def _side(P, q, n):
    """Side of the surface where P is, q being its nearest point and n the normal: +1 outside.
    A head's skin is open (the face stops under the hair): a point beyond its edge, which sees
    that edge at a slant, is neither inside nor outside. Counting it as inside pushed it "out of
    the skin" in one big jump, into two bumps on top of the skull, like ears."""
    c = ((P - q) * n).sum(1)
    s = np.sign(c)
    s[(s == 0) | (np.abs(c) < BIAS * np.linalg.norm(P - q, axis=1))] = 1
    return s


def fraction_inside(P, D, faces_hair):
    """Share of the points P inside the hairstyle (faces_coif) or D's skin."""
    Vh, Fh, Nh = _surface(D, faces_hair)
    sh, _ = signed(P, Vh, Fh, Nh)
    Vp, Fp, Np = _surface(D, np.isin(D.mat_face, D.i_skin))
    sp, _ = signed(P, Vp, Fp, Np)
    return float(((sh < 0) & (sh > -DEEP_MAX * D.iod) | (sp < 0) & (sp > -DEEP_MAX * D.iod)).mean())


# =================================================================== extraction

def extract_one(args):
    name, lib = args
    with open(os.path.join(lib, name, "record.json"), encoding="utf-8") as fh:
        card = json.load(fh)  # pmx as is in coiffure.json (relative for Outerplane)
    D = TT.Head(BI.read_card(lib, name)["pmx"])
    r = locate(D)
    np.savez_compressed(os.path.join(lib, name, "hairstyle.npz"), hairstyle=r["hairstyle"],
                        ears=r["ears"], bones=r["bones"])
    info = {"person": name, "pmx": card["pmx"], "materials": r["materials"],
            "hairstyle_faces": int(r["hairstyle"].sum()), "ear_faces": int(r["ears"].sum()),
            "ears_linked_faces": r["ears_linked"], "hairstyle_bones": int(r["bones"].sum()),
            "hairstyle_rigid_bodies": int(sum(1 for c in D.pmx["rigid_bodies"] if c["bones"] >= 0 and r["bones"][c["bones"]])),
            "coverage": round(r["coverage"], 3),
            "inside_ears": None if r["inside_ears"] is None else round(r["inside_ears"], 3),
            "worn_on_face": r["worn_on_face"]}
    BI._write_json(os.path.join(lib, name, "hairstyle.json"), info)
    return info


def extract(lib, tasks, alone=None):
    with open(os.path.join(lib, "index.json"), encoding="utf-8") as fh:
        idx = json.load(fh)
    names = sorted(idx["regions"]["eyes"])
    if alone:
        names = [n for n in names if n in alone]
    res = list(BI.map_(extract_one, [(n, lib) for n in names], tasks))
    for r in res:
        log("%-12s hairstyle %5d faces, ears %4d (attached %d), bones %3d, rigid bodies %3d, coverage %.2f" % (
            r["person"], r["hairstyle_faces"], r["ear_faces"], r["ears_linked_faces"],
            r["hairstyle_bones"], r["hairstyle_rigid_bodies"], r["coverage"]))
    # merge: the hairstyles already recorded for the other characters are kept
    idx.setdefault("hairstyle", {}).update({r["person"]: {"coverage": r["coverage"],
                                                             "ears": r["ear_faces"] > 0} for r in res})
    idx["hairstyle"] = dict(sorted(idx["hairstyle"].items()))
    BI._write_json(os.path.join(lib, "index.json"), idx)


# =================================================================== mapping from one head to another

class Passage:
    """Mapping from A's frame to B's through the pivot head: similarity from each head to the
    pivot, a similarity fitted (Procrustes) on the reliable points, then the local offset of
    these points interpolated (weighted neighbours). A thin-plate spline fitted on the face alone
    extrapolated far from it (top and back of the skull) with a shear: between two very
    different faces (Dialyn on Yixuan), the top of the hairstyle ended up below the eyes."""

    def __init__(self, lib, nameA, nameB, pivot, masks):
        za, zb = (np.load(os.path.join(lib, n, "part.npz")) for n in (nameA, nameB))
        self.sA, self.csA, self.cdA = float(za["scale"]), za["c_src"], za["c_dst"]
        self.sB, self.csB, self.cdB = float(zb["scale"]), zb["c_src"], zb["c_dst"]
        XA, XB = za["X"].astype(float), zb["X"].astype(float)
        # points reliable on both sides, away from the features (eyes, eyebrows, nose, mouth)
        traits = np.zeros(pivot.n)
        for z, m in masks.items():
            if z != BI.BASE:
                traits = np.maximum(traits, m)
        ok = np.where(za["reliable"] & zb["reliable"] & (traits < 0.05))[0]
        ok = ok[_sample(pivot.V[ok], POINTS_PASSAGE)]
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
        self.gap = self.dst - self._sim(self.src)
        self.scale = self.sA / self.sB * self.s

    def _sim(self, q):
        return self.s * q @ self.R.T + self.t

    def __call__(self, p):
        q = self.sA * (np.asarray(p, dtype=float) - self.csA) + self.cdA
        if len(q):
            q = self._sim(q) + G.shepard(self.src, self.gap, q, k=NEIGHBORS_PASSAGE)
        return (q - self.cdB) / self.sB + self.csB


def _sample(P, n):
    """n points far from one another (farthest first)."""
    if len(P) <= n:
        return np.arange(len(P))
    chosen = [int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))]
    d = np.linalg.norm(P - P[chosen[0]], axis=1)
    for _ in range(n - 1):
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(P - P[j], axis=1))
    return np.array(chosen)


def leave_of_skin(P, F_loc, Vp, Fp, Np, iod, perimeters=4, signed=signed, rest_pose=None):
    """Vertices P (mesh F_loc) that are inside the skin, or closer than MARGIN_SKIN, pushed out
    along the skin normal; the push spreads to the neighbours (whole strands).

    `rest_pose`: signed distance from each vertex to the skin of its original head. A vertex is
    then pushed out only if it is closer or deeper than on its own head: roots planted under
    the scalp (hand-made haircuts) and strands lying on the skin stay as they were made, and a
    hairstyle put back on its own head does not move."""
    rep, inv = G.weld(P, 1e-6)
    W = P[rep].copy()
    margin = MARGIN_SKIN * iod if rest_pose is None else np.minimum(MARGIN_SKIN * iod, np.asarray(rest_pose, float)[rep] - 1e-4 * iod)
    Fw = inv[F_loc]
    Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
    E, _, _ = G.edges(Fw)
    a, b = np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]])
    nv = np.maximum(np.bincount(a, minlength=len(W)), 1)[:, None]
    entered = 0
    for girth in range(perimeters):
        s, n = signed(W, Vp, Fp, Np)
        e = (s < margin) & (s > -DEEP_MAX * iod)
        if girth == 0:
            entered = int(e.sum())
        if not e.any():
            break
        u = np.zeros_like(W)
        u[e] = ((margin[e] if rest_pose is not None else margin) - s[e])[:, None] * n[e]
        # spreading: each vertex takes the strongest of the pushes averaged around it
        for _ in range(12):
            m = np.zeros_like(u)
            np.add.at(m, a, u[b])
            m /= nv
            plus = np.linalg.norm(m, axis=1) > np.linalg.norm(u, axis=1)
            u[plus] = m[plus]
        W += u
    s, _ = signed(W, Vp, Fp, Np)
    rest = int(((s < 0) & (s > -DEEP_MAX * iod)).sum())
    return W[inv], {"vertices_entered": entered, "vertices_inside_after": rest, "vertices": int(len(W))}


# =================================================================== placing

def _rel(path_tex, folder, output):
    return RT._texture_rel(path_tex, folder, output)


def place(lib, nameA, nameB, output, ears=None, target=None):
    os.makedirs(output, exist_ok=True)
    card = lambda n: BI.read_card(lib, n)  # noqa: E731
    A = TT.Head(card(nameA)["pmx"])
    B = TT.Head(target or card(nameB)["pmx"])
    pivot = BI.load_pivot()
    M = BI.masks(pivot)
    rA, rB = locate(A), locate(B)
    pAb = Passage(lib, nameA, nameB, pivot, M)
    log("hairstyle of %s (%d faces) on %s, replacing its own (%d faces)" % (
        nameA, rA["hairstyle"].sum(), nameB, rB["hairstyle"].sum()))
    parts = [(A, rA, rA["hairstyle"].copy(), pAb, "hairstyle")]
    info = {"hair": nameA, "on": nameB, "points_passage": pAb.points, "scale": pAb.scale}
    # "+" highlights baked into the hair texture: their duplicate faces are removed
    removes, tex_reflections = RF.bake(A, parts[0][2], output)
    parts[0][2][removes] = False
    info["reflections_cuits"] = int(removes.sum())
    if ears:
        parts[0][2][rA["ears"]] = False
        info["ears_removed"] = int(rA["ears"].sum())
        if ears != "aucune":
            C = TT.Head(card(ears)["pmx"])
            rC = locate(C)
            if not rC["ears"].any():
                raise ValueError("%s: no separable ears" % ears)
            parts.append((C, rC, rC["ears"].copy(), Passage(lib, ears, nameB, pivot, M), "ears"))
            info["ears"] = ears

    Pb = B.pmx
    P = copy.deepcopy(Pb)
    nuv = Pb["n_uv_add"]
    # --- B's skin (to push the hair out): the face skin and the other skins of the head
    skinB = np.isin(B.mat_face, B.i_skin)
    Vp, Fp, Np = _surface(B, skinB)
    # --- B's hairstyle removed: faces dropped, rigid bodies and joints of its bones removed
    kept_fB = ~rB["hairstyle"]
    removes_rig = [i for i, c in enumerate(Pb["rigid_bodies"]) if c["bones"] >= 0 and rB["bones"][c["bones"]]]
    idx_rig = np.full(len(Pb["rigid_bodies"]), -1)
    k = 0
    for i in range(len(Pb["rigid_bodies"])):
        if i not in set(removes_rig):
            idx_rig[i] = k
            k += 1
    P["rigid_bodies"] = [copy.deepcopy(c) for i, c in enumerate(Pb["rigid_bodies"]) if idx_rig[i] >= 0]
    P["joints"] = []
    for j in Pb["joints"]:
        if idx_rig[j["a"]] >= 0 and idx_rig[j["b"]] >= 0:
            j = copy.deepcopy(j)
            j["a"], j["b"] = int(idx_rig[j["a"]]), int(idx_rig[j["b"]])
            P["joints"].append(j)
    for m in P["morphs"]:
        if m["type"] == 10:
            m["offsets"] = [(int(idx_rig[d[0]]),) + tuple(d[1:]) for d in m["offsets"] if idx_rig[d[0]] >= 0]
    rig_de_os_B = {}
    for i, c in enumerate(P["rigid_bodies"]):
        rig_de_os_B.setdefault(P["bones"][c["bones"]]["name"] if c["bones"] >= 0 else None, i)

    S = {k: list(v) for k, v in Pb["vertices"].items()}
    faces_per_mat = {}
    for f in np.where(kept_fB)[0]:
        faces_per_mat.setdefault(("B", int(B.mat_face[f])), []).append(tuple(int(v) for v in B.faces[f]))
    order = [("B", i) for i in range(len(Pb["materials"]))]
    P["textures"] = [_rel(x, os.path.dirname(B.path), output) for x in Pb["textures"]]
    names_B = {o["name"]: i for i, o in enumerate(Pb["bones"])}
    head_B = names_B.get("頭", 0)
    new_os = []
    measures_parts = []
    for n_p, (D, rD, faces, passed, what) in enumerate(parts):
        Pd = D.pmx
        the_key = "D%d" % n_p
        # bones: the hairstyle's are copied (renamed if B already has that name), the others
        # mapped by name onto B
        map_ = np.full(len(Pd["bones"]), head_B)
        copies = np.where(rD["bones"])[0]
        base = len(P["bones"])
        for j, i in enumerate(copies):
            map_[i] = base + j
        for i, o in enumerate(Pd["bones"]):
            if not rD["bones"][i]:
                map_[i] = names_B.get(o["name"], head_B)
        names_taken = {o["name"] for o in P["bones"]}
        for i in copies:
            o = copy.deepcopy(Pd["bones"][i])
            if o["name"] in names_taken:
                o["name"] = o["name"] + "_" + D.name[:8]
            names_taken.add(o["name"])
            o["pos"] = tuple(passed(np.array([o["pos"]]))[0])
            o["parent"] = int(map_[o["parent"]]) if o["parent"] >= 0 else -1
            if o["tail_bone"] is not None and o["tail_bone"] >= 0:
                o["tail_bone"] = int(map_[o["tail_bone"]])
            elif o["tail_offset"] is not None:
                o["tail_offset"] = tuple(passed.scale * np.asarray(o["tail_offset"]))
            if o["addition"] is not None:
                o["addition"]["source"] = int(map_[o["addition"]["source"]])
            if o["ik"] is not None:
                o["ik"]["target"] = int(map_[o["ik"]["target"]])
                for li in o["ik"]["links"]:
                    li["bones"] = int(map_[li["bones"]])
            P["bones"].append(o)
            new_os.append(len(P["bones"]) - 1)
        # vertices: position mapped onto B, then pushed out of B's skin
        vs, Fl = G.under_mesh(D.faces, faces)
        pos = passed(D.pos[vs])
        if what == "hairstyle":
            rest_pose, _ = signed(D.pos[vs], *_surface(D, np.isin(D.mat_face, D.i_skin)))
            pos, output_skin = leave_of_skin(pos, Fl, Vp, Fp, Np, B.iod, rest_pose=rest_pose * passed.scale)
        else:
            pos, output_skin = place_ears(pos, Fl, rD, D, B, P, S, faces_per_mat, order, Vp, Fp, Np), {}
        nor = D.t["nor"][vs]
        n0 = len(S["pos"])
        sd = Pd["vertices"]
        for j, v in enumerate(vs):
            t = sd["type"][v]
            S["pos"].append(tuple(pos[j]))
            S["nor"].append(tuple(nor[j]))
            S["uv"].append(sd["uv"][v])
            u = list(sd["uv_add"][v])[:nuv] if Pd["n_uv_add"] else []
            S["uv_add"].append(u + [(0.0, 0.0, 0.0, 0.0)] * (nuv - len(u)))
            S["type"].append(t)
            S["bones"].append(tuple(int(map_[x]) if x >= 0 else -1 for x in sd["bones"][v]))
            S["weights"].append(sd["weights"][v])
            S["sdef"].append(sd["sdef"][v])
            S["contour"].append(sd["contour"][v])
        for f, (a, b, c) in zip(np.where(faces)[0], Fl):
            faces_per_mat.setdefault((the_key, int(D.mat_face[f])), []).append((n0 + int(a), n0 + int(b), n0 + int(c)))
        off_tex = len(P["textures"])
        P["textures"] += [_rel(x, os.path.dirname(D.path), output) for x in Pd["textures"]]
        mats_D = sorted({int(D.mat_face[f]) for f in np.where(faces)[0]})
        order += [(the_key, i) for i in mats_D]
        parts[n_p] = parts[n_p] + (off_tex,)
        # rigid bodies and joints of the hairstyle
        rig_D = {}
        for i, c in enumerate(Pd["rigid_bodies"]):
            if c["bones"] >= 0 and rD["bones"][c["bones"]]:
                c = copy.deepcopy(c)
                c["bones"] = int(map_[c["bones"]])
                c["pos"] = tuple(passed(np.array([c["pos"]]))[0])
                c["size"] = tuple(passed.scale * np.asarray(c["size"]))
                rig_D[i] = len(P["rigid_bodies"])
                P["rigid_bodies"].append(c)
        for j in Pd["joints"]:
            a, b = rig_D.get(j["a"]), rig_D.get(j["b"])
            if a is None and b is None:
                continue

            def elsewhere(r):
                c = Pd["rigid_bodies"][r]
                return rig_de_os_B.get(Pd["bones"][c["bones"]]["name"] if c["bones"] >= 0 else None, rig_de_os_B.get("頭"))
            a = a if a is not None else elsewhere(j["a"])
            b = b if b is not None else elsewhere(j["b"])
            if a is None or b is None:
                continue
            j = copy.deepcopy(j)
            j["a"], j["b"] = int(a), int(b)
            j["pos"] = tuple(passed(np.array([j["pos"]]))[0])
            j["disp_min"] = tuple(passed.scale * np.asarray(j["disp_min"]))
            j["disp_max"] = tuple(passed.scale * np.asarray(j["disp_max"]))
            P["joints"].append(j)
        measures_parts.append({"from": D.name, "what": what, "vertices": int(len(vs)), "faces": int(faces.sum()),
                               "os_copies": int(len(copies)), "rigid_bodies_copies": len(rig_D), **output_skin})
    # --- materials and faces
    mats, Ftot = [], []
    src = {"B": (Pb, 0)}
    for n_p, pc in enumerate(parts):
        src["D%d" % n_p] = (pc[0].pmx, pc[5])
    for the_key in order:
        s_, i = the_key
        Ps, off = src[s_]
        m = copy.deepcopy(Ps["materials"][i])
        if s_ != "B":
            for kk in ("texture", "sphere"):
                if m[kk] >= 0:
                    m[kk] += off
            if m["toon_sharing"] == 0 and m["toon"] >= 0:
                m["toon"] += off
            m["name"] = m["name"] + "_" + parts[int(s_[1:])][0].name[:8]
        if s_ == "D0" and i in tex_reflections:
            P["textures"].append(os.path.relpath(tex_reflections[i], output).replace("/", "\\"))
            m["texture"] = len(P["textures"]) - 1
        fl = faces_per_mat.get(the_key, [])
        m["n_indices"] = 3 * len(fl)
        mats.append(m)
        Ftot += fl
    P["materials"], P["faces"], P["vertices"] = mats, Ftot, S
    # B's material morphs: same indices (its materials stay first, in order)
    if new_os:
        P["frames"].append({"name": "髪_" + nameA[:8], "name_en": "hair_" + nameA[:8], "special": 0,
                            "elements": [(0, i) for i in new_os]})
    P["comment"] = (P.get("comment") or "") + "\r\n[hairstyle: hair of %s%s]" % (
        nameA, (", ears of %s" % ears) if ears else "")
    path = os.path.join(output, "result.pmx")
    pmx_write.write(P, path)
    info["parts"] = measures_parts
    info["os_retires_de_B"] = int(rB["bones"].sum())
    info["rigid_removed_from_B"] = len(removes_rig)
    return path, info, rA, rB


def place_ears(pos, Fl, rC, C, B, P, S, faces_per_mat, order, Vp, Fp, Np):
    """C's ears (already mapped onto B) sunk into the placed hairstyle as they were on C: they
    are slid toward the middle of the head until the same share of vertices is inside."""
    wanted = rC["inside_ears"] or 0.2
    # surface of what is already placed on B (skin and added hairstyle)
    Vall = np.asarray(S["pos"], dtype=float)
    Nall = np.asarray(S["nor"], dtype=float)
    Fn = np.array([f for the_key, fl in faces_per_mat.items() if the_key[0] != "B" for f in fl], dtype=np.int64)
    center = B.V.mean(0)
    output = []
    lab = G.components_faces(Fl, np.ones(len(Fl), dtype=bool))
    for k in range(lab.max() + 1):
        vk = np.unique(Fl[lab == k])
        Q = pos[vk]
        d = center - Q.mean(0)
        d /= max(np.linalg.norm(d), 1e-12)

        def part(t):
            X = Q + t * d
            sh, _ = signed(X, Vall, Fn, Nall) if len(Fn) else (np.full(len(X), np.inf), None)
            sp, _ = signed(X, Vp, Fp, Np)
            lim = -DEEP_MAX * B.iod
            return float((((sh < 0) & (sh > lim)) | ((sp < 0) & (sp > lim))).mean())
        lo, hi = -0.5 * B.iod, 1.0 * B.iod
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if part(mid) < wanted:
                lo = mid
            else:
                hi = mid
        pos[vk] = Q + 0.5 * (lo + hi) * d
        output.append(round(0.5 * (lo + hi) / B.iod, 3))
    return pos


# =================================================================== checks

def check(path, nameA, rA, B, rB, info):
    import pmx as PMX
    R = TT.Head(path)
    t = R.t
    new = np.array([("_" + nameA[:8]) in R.names_mat[i] for i in R.mat_face])
    eyes = np.array([R.center_eye[c] for c in "gd"])
    cov = coverage(R.V, R.F, R.pos[R.faces[new]], R.V.mean(0), eyes[:, 1].mean(), R.iod)
    # the skull skin counts in the coverage: a hairstyle placed too low leaves it bare
    # without the coverage dropping; the hairstyle alone must cover the skull directions as much as on A
    cov_c = coverage_skull(R.V, R.F, R.pos[R.faces[new]], R.V.mean(0), eyes[:, 1].mean(), R.iod)
    floor = min(rA["coverage"], rB["coverage"])
    Vp, Fp, Np = _surface(R, np.isin(R.mat_face, R.i_skin))
    vn = np.unique(R.faces[new])
    s, _ = signed(R.pos[vn], Vp, Fp, Np)
    inside = float(((s < 0) & (s > -DEEP_MAX * R.iod)).mean())
    # bones of the added vertices: all valid
    o = t["bones"][vn]
    os_ok = bool(((o < len(R.names_os)) & (o >= -1)).all())
    rig = R.pmx["rigid_bodies"]
    rig_ok = all(-1 <= c["bones"] < len(R.names_os) for c in rig) and all(
        0 <= j["a"] < len(rig) and 0 <= j["b"] < len(rig) for j in R.pmx["joints"])
    q = G.quality(R.pos, R.faces[new]) if new.any() else {}
    tests = {
        "reread": {"passed": True, "materials": len(R.names_mat), "bones": len(R.names_os), "rigid_bodies": len(rig)},
        "skull_covered": {"passed": cov >= floor - 0.03, "coverage": round(cov, 3),
                          "floor": round(floor, 3), "A": rA["coverage"], "B": rB["coverage"]},
        "skull_under_hair": {"passed": cov_c >= rA["hair_coverage"] - 0.05,
                               "coverage": round(cov_c, 3), "A": round(rA["hair_coverage"], 3)},
        "hair_off_skin": {"passed": inside <= 0.005, "fraction_inside": round(inside, 4)},
        "os_and_physics": {"passed": os_ok and rig_ok},
        "coiffure_de_B_retiree": {"passed": not any(("_" + nameA[:8]) not in n and n in rB["materials"] and
                                                   R.pmx["materials"][i]["n_indices"] > 0
                                                   for i, n in enumerate(R.names_mat)
                                                   if any(c in n for c in HAIR))},
    }
    if isinstance(q, dict) and q:
        tests["quality_mesh"] = {"passed": True, **{k: v for k, v in q.items() if isinstance(v, (int, float))}}
    failures = [k for k, v in tests.items() if not v["passed"]]
    return {"tests": tests, "verdict": {"passed": not failures, "failures": failures}, "construction": info}


def sheet(source, target, result, output):
    cmd = ["bash", RT.FILE_BLENDER, "-b", "--factory-startup", "--python", os.path.join(HERE, "texture_sheet.py"), "--",
           "--hairstyle", os.path.join(output, "texture_sheet.png"), source, os.path.dirname(source),
           target, os.path.dirname(target), result, output]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("textured contact sheet: Blender failed")
    return os.path.join(output, "texture_sheet.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("extract", "place", "sheet"))
    ap.add_argument("--library", dest="lib", required=True)
    ap.add_argument("--jobs", dest="tasks", type=int, default=4)
    ap.add_argument("--only", dest="alone")
    ap.add_argument("--hair", dest="hair")
    ap.add_argument("--ears", dest="ears", help="character that gives its animal ears, or 'aucune' (none)")
    ap.add_argument("--onto", dest="on", help="library character whose head receives the hairstyle")
    ap.add_argument("--target", dest="target", help="PMX to put the hairstyle on (default: the --onto one; e.g. a composite face placed on it)")
    ap.add_argument("--output", dest="output")
    ap.add_argument("--no-sheet", dest="without_sheet", action="store_true")
    a = ap.parse_args(argv)
    lib = os.path.abspath(os.path.expanduser(a.lib))
    if a.action == "extract":
        extract(lib, a.tasks, a.alone.split(",") if a.alone else None)
        return 0
    output = os.path.abspath(os.path.expanduser(a.output))
    card = lambda n: BI.read_card(lib, n)["pmx"]  # noqa: E731
    if a.action == "sheet":  # redo the render of an existing placement
        log("textured contact sheet: %s" % sheet(card(a.hair), a.target or card(a.on),
                                                      os.path.join(output, "result.pmx"), output))
        return 0
    t0 = time.time()
    path, info, rA, rB = place(lib, a.hair, a.on, output, a.ears, a.target)
    B = TT.Head(a.target or card(a.on))
    mes = check(path, a.hair, rA, B, rB, info)
    mes["duration_s"] = round(time.time() - t0, 1)
    BI._write_json(os.path.join(output, "measures.json"), mes)
    for k, v in mes["tests"].items():
        log("  %-24s %s" % (k, "ok" if v["passed"] else "FAILED"))
    log("measurements: %s" % ("ALL PASS" if mes["verdict"]["passed"] else "FAILURES: " + ", ".join(mes["verdict"]["failures"])))
    if not a.without_sheet:
        src = card(a.hair)
        log("textured contact sheet: %s" % sheet(src, B.path, path, output))
    return 0 if mes["verdict"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

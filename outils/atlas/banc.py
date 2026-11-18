"""Stress bench of method v2 (§2.3, §2.5): poses P13-P15 and amplitude variants
added to P-08, left-right floor P-12, analytic lighting P-13,
indicators D1 (share out of bounds), D3 (area), D4 (signed), D6 to D10.

Numpy only, except D4 and D10 (mathutils.bvhtree, hence Blender's Python).
`primitives.py` and `cinematique.py` are read, never modified: the poses of A
(P0-P12) stay those of `primitives.pose`; this module only adds the new ones.

Conventions (those of `primitives`): Blender frame, facing -Y, +Z up,
side 左 at +X; local bone rotations in the world frame at rest.
"""
import numpy as np

import cinematique as cin
import primitives as P

AVANT, HAUT, BAS = P.AVANT, P.HAUT, P.BAS
LATERAL = {"L": np.array([1.0, 0.0, 0.0]), "R": np.array([-1.0, 0.0, 0.0])}

# ======================================================================= P-08 extended

# Amplitude variants of D §4 (method v2 §2.3, P-08): pose of A -> changed angles
VARIANTES = {
    "P1@90": "arms raised by 90° (P1 at 150°)",
    "P4@90": "elbow bent by 90° (P4 at 130°)",
    "P4@140": "elbow bent by 140°",
    "P5@60": "left thigh raised by 60° (P5 at 90°)",
    "P6@90": "left knee bent by 90° (P6 at 120°)",
    "P6@140": "left knee bent by 140°",
    "P7@D": "crouch of D (P9 of D): hips 110°, knees 140°, ankles 30°",
}
NOUVELLES = {
    "P13": "shoulder shrug of 20°, then arms crossed in front",
    "P14": "T-pose",
    "P15": "forearm in pronation of -90° (P3 in the opposite direction)",
}
# Full bench of E-18: P-08 of A (without the rest pose), new poses, variants
BANC = ["P1", "P1@90", "P2", "P3", "P15", "P4", "P4@90", "P4@140", "P5", "P5@60", "P6",
        "P6@90", "P6@140", "P7", "P7@D", "P8", "P9", "P10", "P11", "P12", "P13", "P14"]


def _nom(c, os_):
    return ("左" if c == "L" else "右") + os_


def _rot_vers(sq, nom, vers, cible, degres=None):
    """Index and rotation that brings the direction of `nom` (towards the head of `vers`) towards
    `cible`, by `degres` (None: all the way to the target). None if a bone is missing."""
    if not sq.a(nom) or not sq.a(vers):
        return None
    b = sq.index(nom)
    d, _ = sq.direction(b, vers)
    c = np.asarray(cible, dtype=float)
    c = c / np.linalg.norm(c)
    if degres is None:
        degres = float(np.degrees(np.arccos(np.clip(d @ c, -1.0, 1.0))))
    axe = np.cross(d, c)
    if np.linalg.norm(axe) < 1e-9:
        axe = np.cross(d, AVANT)
    return b, cin.q_axe_angle(axe, degres)


def _ajouter(r, x):
    if x is None:
        return
    b, q = x
    r[b] = cin.q_mul(r.get(b, cin.IDENTITE), q)


def _rot_parent(sq, r, b):
    """World matrix of the parent of b under the rotations r (to give a target in
    the rest frame of b when its ancestors are already rotated)."""
    R, _ = sq.cinematique(r)
    p = sq.parents[b]
    return R[p] if p >= 0 else np.eye(3)


def pose_banc(nom, sq, cotes=("L", "R")):
    """Local rotations (bone index -> quaternion) of a bench pose.

    P0-P12: `primitives.pose` (A), unchanged. P13-P15 and variants `Pk@x`: here.
    Angles counted from the delivered rest pose, as in A."""
    if nom in P.POSES:
        return P.pose(nom, sq, cotes)
    r = {}
    if nom == "P13":
        for c in cotes:
            # shrug: 肩 raised by 20° upwards
            _ajouter(r, _rot_vers(sq, _nom(c, "肩"), _nom(c, "腕"), HAUT, 20))
            # arms in front, a little low, inwards; elbow bent by 110° inwards
            med = -LATERAL[c]
            _ajouter(r, _rot_vers(sq, _nom(c, "腕"), _nom(c, "ひじ"), AVANT + 0.5 * BAS + 0.4 * med))
            if sq.a(_nom(c, "ひじ")):
                b = sq.index(_nom(c, "ひじ"))
                Rp = _rot_parent(sq, r, b)
                _ajouter(r, _rot_vers(sq, _nom(c, "ひじ"), _nom(c, "手首"), Rp.T @ med, 110))
    elif nom == "P14":
        for c in cotes:
            _ajouter(r, _rot_vers(sq, _nom(c, "腕"), _nom(c, "ひじ"), LATERAL[c]))
    elif nom == "P15":
        os_ = "手捩" if sq.a("左手捩") else "手首"
        for c in cotes:
            n, v = _nom(c, "ひじ"), _nom(c, "手首")
            if not sq.a(n) or not sq.a(v) or not sq.a(_nom(c, os_)):
                continue
            d, _ = sq.direction(sq.index(n), v)
            _ajouter(r, (sq.index(_nom(c, os_)), cin.q_axe_angle(d, -90 if c == "L" else 90)))
    elif nom == "P1@90":
        for c in cotes:
            _ajouter(r, _rot_vers(sq, _nom(c, "腕"), _nom(c, "ひじ"), HAUT, 90))
    elif nom in ("P4@90", "P4@140"):
        a = float(nom.split("@")[1])
        for c in cotes:
            _ajouter(r, _rot_vers(sq, _nom(c, "ひじ"), _nom(c, "手首"), AVANT, a))
    elif nom == "P5@60":
        _ajouter(r, _rot_vers(sq, "左足", "左ひざ", AVANT, 60))
    elif nom in ("P6@90", "P6@140"):
        a = float(nom.split("@")[1])
        _ajouter(r, _rot_vers(sq, "左ひざ", "左足首", -AVANT, a))
    elif nom == "P7@D":
        for c in cotes:
            _ajouter(r, _rot_vers(sq, _nom(c, "足"), _nom(c, "ひざ"), AVANT, 110))
            _ajouter(r, _rot_vers(sq, _nom(c, "ひざ"), _nom(c, "足首"), -AVANT, 140))
            _ajouter(r, _rot_vers(sq, _nom(c, "足首"), _nom(c, "つま先"), HAUT, 30))
    elif nom == "ASSIS":      # sitting on a chair (outside the bench): hips and knees at 90°
        for c in cotes:
            _ajouter(r, _rot_vers(sq, _nom(c, "足"), _nom(c, "ひざ"), AVANT, 90))
            _ajouter(r, _rot_vers(sq, _nom(c, "ひざ"), _nom(c, "足首"), -AVANT, 90))
    else:
        raise ValueError("unknown bench pose: %s" % nom)
    return r


def description_pose(nom):
    return VARIANTES.get(nom) or NOUVELLES.get(nom) or "P-08 of A (%s)" % nom


# ======================================================================= P-12

index_miroir_os = P.miroir_os  # by name; `tol` adds the position


def miroir_rotations(rotations, sq, im=None):
    """Mirror pose (P-12): each rotation moves to the symmetric bone, reflected through x = 0."""
    im = index_miroir_os(sq) if im is None else im
    out = {}
    for b, q in rotations.items():
        mb = int(im[b])
        out[mb] = cin.q_mul(out.get(mb, cin.IDENTITE), cin.q_miroir_x(q))
    return out


def cotes_sommets(pos, H, tol=1e-3):
    """Side of each vertex: 'L' (x > tol·H), 'R' (x < -tol·H), 'M' otherwise."""
    x = pos[:, 0]
    return np.where(x > tol * H, "L", np.where(x < -tol * H, "R", "M"))


def plancher(m_gauche_p, m_droite_miroir):
    """P-12: P(z, p) = |m(z_G, p) - m(z_D, miroir(p))|. None if either is missing."""
    if m_gauche_p is None or m_droite_miroir is None:
        return None
    return float(abs(m_gauche_p - m_droite_miroir))


# ======================================================================= P-13

ELEVATIONS = (0.0, 30.0, -20.0)


def lumieres(pas=5.0, elevations=ELEVATIONS):
    """P-13: 72 azimuths × 3 elevations = 216 directions (towards the light), unit length.
    Azimuth 0 = light from the front (-Y), counted towards +X (side 左)."""
    az = np.radians(np.arange(0.0, 360.0, pas))
    out = []
    for e in np.radians(elevations):
        out.append(np.stack([np.sin(az) * np.cos(e), -np.cos(az) * np.cos(e),
                             np.full_like(az, np.sin(e))], axis=1))
    return np.concatenate(out)


def eclairement(N, L):
    """N·L, (n_lumieres, n_sommets)."""
    N = np.asarray(N, dtype=float)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return np.asarray(L) @ N.T


def ombre(N, L, seuil=0.5):
    """Vertex in shadow if N·L < 0.5 (analytic toon, no render)."""
    return eclairement(N, L) < seuil


def normales_sommets(pos, faces):
    """Normals recomputed from the geometry (angle-weighted average)."""
    pos = np.asarray(pos, dtype=float)
    f = np.asarray(faces)
    nf, _ = P.normales_faces(pos, f)
    N = np.zeros_like(pos)
    for k in range(3):
        a, b, c = f[:, k], f[:, (k + 1) % 3], f[:, (k + 2) % 3]
        u = pos[b] - pos[a]
        v = pos[c] - pos[a]
        cu = np.einsum("ij,ij->i", u, v) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-20)
        ang = np.arccos(np.clip(cu, -1, 1))
        np.add.at(N, a, nf * ang[:, None])
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


# ======================================================================= D1

BORNES_D1 = (0.7, 1.4)


def d1_rapports(pos0, pos1, aretes):
    """r = ‖e_p‖ / ‖e_repos‖ per edge (nan for a zero-length edge)."""
    l0 = np.linalg.norm(pos0[aretes[:, 0]] - pos0[aretes[:, 1]], axis=1)
    l1 = np.linalg.norm(pos1[aretes[:, 0]] - pos1[aretes[:, 1]], axis=1)
    r = np.full(len(aretes), np.nan)
    ok = l0 > 1e-9
    r[ok] = l1[ok] / l0[ok]
    return r


def d1_zone(r, masque=None, bornes=BORNES_D1):
    """D1 on a zone: share outside [0.7, 1.4] (D), percentiles 1 and 99 (A), min, max,
    standard deviation (B03). Breaks: share > 2 %."""
    x = r if masque is None else r[masque]
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    hors = float(((x < bornes[0]) | (x > bornes[1])).mean())
    return {"part_hors": hors, "c01": float(np.percentile(x, 1)), "c99": float(np.percentile(x, 99)),
            "min": float(x.min()), "max": float(x.max()), "ecart_type": float(x.std()),
            "n": int(len(x)), "casse": hors > 0.02}


def d1_par_sommet(r, aretes, n):
    """Strongest stretch or squash of the edges of each vertex: max(r, 1/r)."""
    s = np.where(np.isfinite(r), np.maximum(r, 1.0 / np.maximum(r, 1e-12)), 1.0)
    out = np.ones(n)
    np.maximum.at(out, aretes[:, 0], s)
    np.maximum.at(out, aretes[:, 1], s)
    return out


# ======================================================================= D2

def d2_masque(pos0, pos1, faces, R_dominant):
    """Flipped faces (v2: posed normal · rotation of the dominant bone × rest normal < 0)."""
    n0, a0 = P.normales_faces(pos0, faces)
    n1, a1 = P.normales_faces(pos1, faces)
    att = np.einsum("mij,mj->mi", R_dominant, n0)
    return (np.einsum("mi,mi->m", att, n1) < 0) & (a0 > 1e-12) & (a1 > 1e-12)


# ======================================================================= D3

SEUILS_D3 = (0.85, 1.15)


def aire_boucle(points, normale):
    """Area of the polygon (points in loop order) projected onto its plane."""
    p = np.asarray(points, dtype=float)
    c = p.mean(axis=0)
    q = p - c
    s = np.cross(q, np.roll(q, -1, axis=0)).sum(axis=0)
    return float(abs(s @ np.asarray(normale, dtype=float)) / 2.0)


def d3_aire(pos0, pos1, faces, sq, os_, vers, R, t, fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
            masques=None, cache0=None, anneaux_bande=3, jaccard_min=0.5):
    """D3 of D: posed / rest cross-section area along the bone (P-04 on both).

    masques: list of face masks tried in order; the first one that gives a closed
    loop at rest is kept, and the same one for the pose. cache0: dict to keep
    the rest cuts from one call to the next. Returns a list per fraction and the
    minimum / maximum of the ratio. Breaks: < 0.85 or > 1.15.

    The posed cut is searched in a band around the rest loop (faces within
    `anneaux_bande` neighbourhood rings of the edges cut at rest): without it, the
    posed cut catches another loop (sleeve, neighbouring thigh) and the ratio no
    longer measures the same section."""
    b = sq.index(os_)
    d0, L = sq.direction(b, vers)
    d1 = R[b] @ d0
    masques = masques or [None]
    out = []
    for f in fractions:
        cle = (os_, vers, f)
        if cache0 is not None and cle in cache0:
            k, a0 = cache0[cle]
        else:
            k, a0 = None, None
            for i, mk in enumerate(masques):
                a = P.anneau(pos0, faces, sq.tetes[b], d0, L, f, mk)
                if a and a.get("fermee"):
                    k, a0 = i, a
                    break
            if a0 is not None:
                a0 = dict(a0)
                a0["bande"] = bande_faces(faces, np.array(a0["aretes"]).ravel(), anneaux_bande, masques[k])
            if cache0 is not None:
                cache0[cle] = (k, a0)
        if a0 is None:
            out.append({"t": f, "rapport": None, "raison": "no closed loop at rest"})
            continue
        a1 = P.anneau(pos1, faces, t[b], d1, L, f, a0["bande"])
        if not a1 or not a1.get("fermee"):
            out.append({"t": f, "rapport": None, "raison": "no closed loop in the pose"})
            continue
        e0 = {tuple(x) for x in a0["aretes"]}
        e1 = {tuple(x) for x in a1["aretes"]}
        jac = len(e0 & e1) / max(len(e0 | e1), 1)
        s0 = aire_boucle(a0["points"], a0["normale"])
        s1 = aire_boucle(a1["points"], a1["normale"])
        o = {"t": f, "aire0": s0, "aire1": s1, "rapport": s1 / s0 if s0 > 0 else None,
             "n0": a0["n_sommets"], "n1": a1["n_sommets"], "jaccard": jac}
        if jac < jaccard_min:
            o["rapport_brut"], o["rapport"] = o["rapport"], None
            o["raison"] = "another loop in the pose (shared cut edges: %.2f)" % jac
        out.append(o)
    # rest loop that is not a section of the segment (joint: cut through the trunk)
    n0s = [o["n0"] for o in out if o.get("n0")]
    if n0s:
        med = float(np.median(n0s))
        for o in out:
            if o.get("n0") and o["n0"] > 2 * med and o.get("rapport") is not None:
                o["rapport_brut"], o["rapport"] = o["rapport"], None
                o["raison"] = "rest loop outside the segment (%d edges, median %.0f)" % (o["n0"], med)
    rs = [o["rapport"] for o in out if o.get("rapport") is not None]
    res = {"profil": out, "min": min(rs) if rs else None, "max": max(rs) if rs else None}
    res["casse"] = bool(rs) and (res["min"] < SEUILS_D3[0] or res["max"] > SEUILS_D3[1])
    res["ecartees"] = sum(1 for o in out if o.get("rapport") is None)
    return res


def bande_faces(faces, sommets, n, masque=None):
    """Faces within n neighbourhood rings (by faces) of the given vertices, inside `masque`."""
    f = np.asarray(faces)
    v = np.zeros(int(f.max()) + 1, dtype=bool)
    v[np.asarray(sommets, dtype=np.int64)] = True
    sel = v[f].any(axis=1)
    for _ in range(n):
        v[f[sel].ravel()] = True
        sel = v[f].any(axis=1)
    return sel if masque is None else sel & np.asarray(masque, dtype=bool)


# ======================================================================= D4

def d4_signe(pos_a, pos_b, faces_b, limite):
    """D4 of D: signed distance of each point of A to the surface of B (BVH, sign from
    the normal of the closest face of B). nan beyond `limite`."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    out = np.full(len(pos_a), np.nan)
    if len(faces_b) == 0 or len(pos_a) == 0:
        return out
    bvh = BVHTree.FromPolygons(np.asarray(pos_b).tolist(), np.asarray(faces_b).tolist(),
                               all_triangles=True)
    for k, p in enumerate(np.asarray(pos_a)):
        loc, nor, _, d = bvh.find_nearest(Vector(p), limite)
        if loc is None:
            continue
        out[k] = float((Vector(p) - loc).dot(nor))
    return out


def composantes_masque(n, aretes, masque):
    """Size of each connected component of the vertices of `masque` (edge graph)."""
    sel = masque[aretes[:, 0]] & masque[aretes[:, 1]]
    lab = np.arange(n)
    e = aretes[sel]
    if len(e):
        while True:
            m = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
            ancien = lab.copy()
            np.minimum.at(lab, e[:, 0], m)
            np.minimum.at(lab, e[:, 1], m)
            lab = lab[lab]
            if np.array_equal(lab, ancien):
                break
    l = lab[masque]
    if len(l) == 0:
        return lab, np.array([], dtype=np.int64)
    _, tailles = np.unique(l, return_counts=True)
    return lab, tailles


def d4_zone(s_pose, s_repos, H, aretes_a=None, n_a=None, profondeur=0.002):
    """D4 on the points of a zone of A: share beyond -0.002 H in the pose that were not
    there at rest, largest connected island (if aretes_a, local indices).
    Breaks: share > 0.5 % or island > 5 vertices."""
    lim = -profondeur * H
    nouv = (s_pose < lim) & ~(s_repos < lim)
    nouv = np.where(np.isfinite(s_pose), nouv, False)
    n = len(s_pose)
    if n == 0:
        return None
    ilot = 0
    if aretes_a is not None and nouv.any():
        _, t = composantes_masque(n_a or n, aretes_a, nouv)
        ilot = int(t.max()) if len(t) else 0
    part = float(nouv.mean())
    return {"part": part, "n": int(nouv.sum()), "ilot_max": ilot,
            "profondeur_max_H": float(-np.nanmin(np.where(nouv, s_pose, np.nan)) / H) if nouv.any() else 0.0,
            "casse": part > 0.005 or ilot > 5}


# ======================================================================= D6, D7

def d6_monotonie(N, N_ref, aretes, masque, L=None, delta=0.05, seuil=0.5):
    """D6 (D5 of D): monotonicity of the shadow under P-13.

    - Vertex in violation for a light: in shadow (N·L < 0.5) while a
      deeper neighbour according to the reference (N_ref·L smaller by `delta`) is lit.
    - Shadow area A(θ) of the zone along each elevation: its direction of change
      reverses where the reference does not change that way ("no lighting reason").
    N_ref = normals of the geometry (normales_sommets): the "lighting reason".
    Breaks: more than 0.5 % of the zone's vertices in violation under the same light."""
    L = lumieres() if L is None else L
    idx = np.where(masque)[0]
    if len(idx) == 0:
        return None
    e = aretes[masque[aretes[:, 0]] & masque[aretes[:, 1]]]
    s = eclairement(N[idx], L)
    sr = eclairement(N_ref[idx], L)
    loc = -np.ones(len(N), dtype=np.int64)
    loc[idx] = np.arange(len(idx))
    a, b = loc[e[:, 0]], loc[e[:, 1]]
    om = s < seuil
    viol = np.zeros(om.shape, dtype=bool)
    # a in shadow, b lit and deeper according to the reference (and the reverse)
    v1 = om[:, a] & ~om[:, b] & (sr[:, b] < sr[:, a] - delta)
    v2 = om[:, b] & ~om[:, a] & (sr[:, a] < sr[:, b] - delta)
    for k in range(len(L)):
        viol[k, a[v1[k]]] = True
        viol[k, b[v2[k]]] = True
    part_lum = viol.mean(axis=1)
    # shadow area along the azimuth
    n_el = len(L) // 72 if len(L) % 72 == 0 else 1
    A = om.reshape(n_el, -1, len(idx)).mean(axis=2)
    Ar = (sr < seuil).reshape(n_el, -1, len(idx)).mean(axis=2)
    dA = np.diff(np.concatenate([A, A[:, :1]], axis=1), axis=1)
    dAr = np.diff(np.concatenate([Ar, Ar[:, :1]], axis=1), axis=1)
    tol = 1.0 / len(idx)
    inversions = int(((np.sign(dA) * np.sign(dAr) < 0) & (np.abs(dA) > tol)).sum())
    k = int(np.argmax(part_lum))
    return {"part_max": float(part_lum[k]), "lumiere": k, "part_moyenne": float(part_lum.mean()),
            "sommets_touches": float(viol.any(axis=0).mean()), "inversions_aire": inversions,
            "casse": bool(part_lum[k] > 0.005)}


def aire_sommets(pos, faces):
    """Area assigned to each vertex (a third of the neighbouring faces)."""
    _, a = P.normales_faces(pos, faces)
    out = np.zeros(len(pos))
    for k in range(3):
        np.add.at(out, faces[:, k], a / 3.0)
    return out


def d7_taches(N, aretes, masque, aire_v, L=None, part_aire=0.01, part_lum=0.05, seuil=0.5):
    """D7 (D6 of D): shadow spots. Per light, connected components of the vertices of the
    zone in shadow; island = component of less than 1 % of the zone's area.
    Breaks: at least one island in more than 5 % of the lights."""
    L = lumieres() if L is None else L
    idx = np.where(masque)[0]
    if len(idx) == 0:
        return None
    aire_z = float(aire_v[idx].sum())
    om = ombre(N[idx], L, seuil)
    loc = -np.ones(len(N), dtype=np.int64)
    loc[idx] = np.arange(len(idx))
    e = aretes[masque[aretes[:, 0]] & masque[aretes[:, 1]]]
    e = np.stack([loc[e[:, 0]], loc[e[:, 1]]], axis=1)
    av = aire_v[idx]
    n_ilots = np.zeros(len(L), dtype=np.int64)
    for k in range(len(L)):
        if not om[k].any():
            continue
        lab, _ = composantes_masque(len(idx), e, om[k])
        l = lab[om[k]]
        u, inv = np.unique(l, return_inverse=True)
        a = np.bincount(inv, weights=av[om[k]])
        n_ilots[k] = int((a < part_aire * aire_z).sum())
    avec = float((n_ilots > 0).mean())
    return {"part_lumieres_avec_ilot": avec, "ilots_max": int(n_ilots.max()),
            "casse": avec > part_lum}


# ======================================================================= D8

def d8_contour(pos, nor, s_v, e_mat, b2s, dessine=None, echelle=0.08):
    """D8 (D7 of D): break of the inverted-hull outline. Offset position = vertex + e·s_v·N_v
    (e = outline size of the material, in MMD units × scale). At split vertices
    (same welded vertex b2s), largest distance between offset positions, relative to e.
    dessine: mask of the vertices whose material draws the outline.
    Breaks: gap > 0.25 × e."""
    pos = np.asarray(pos, dtype=float)
    nor = np.asarray(nor, dtype=float)
    nor = nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-12)
    e = np.asarray(e_mat, dtype=float) * echelle
    dec = pos + (e * np.asarray(s_v, dtype=float))[:, None] * nor
    sel = np.ones(len(pos), dtype=bool) if dessine is None else np.asarray(dessine, dtype=bool)
    ordre = np.argsort(b2s, kind="stable")
    ordre = ordre[sel[ordre]]
    g = np.asarray(b2s)[ordre]
    debut = np.r_[0, np.where(np.diff(g) != 0)[0] + 1]
    fin = np.r_[debut[1:], len(g)]
    rapports = []
    sommets = []
    for d, f in zip(debut, fin):
        if f - d < 2:
            continue
        v = ordre[d:f]
        D = np.linalg.norm(dec[v][:, None, :] - dec[v][None, :, :], axis=2)
        em = np.maximum(e[v][:, None], e[v][None, :])
        r = np.where(em > 0, D / np.maximum(em, 1e-12), 0.0)
        rapports.append(float(r.max()))
        sommets.append(int(g[d]))
    rapports = np.array(rapports)
    if len(rapports) == 0:
        return {"n_coutures": 0, "max": 0.0, "part_casse": 0.0, "casse": False}
    return {"n_coutures": int(len(rapports)), "max": float(rapports.max()),
            "c99": float(np.percentile(rapports, 99)),
            "n_casse": int((rapports > 0.25).sum()),
            "part_casse": float((rapports > 0.25).mean()),
            "casse": bool((rapports > 0.25).any()),
            "_par_sommet_soude": dict(zip(sommets, rapports.tolist()))}


# ======================================================================= D9

def d9_ouverture(pos0, pos1, haut, bas):
    """D9: eye opening at the middle of the eyelid (pairs of vertices haut[i] / bas[i]),
    posed / rest. Breaks on blink: opening > 10 %."""
    o0 = np.linalg.norm(pos0[haut] - pos0[bas], axis=1).mean()
    o1 = np.linalg.norm(pos1[haut] - pos1[bas], axis=1).mean()
    r = float(o1 / o0) if o0 > 0 else None
    return {"ouverture": r, "casse_clignement": r is not None and r > 0.10}


def d9_angle_coin(pos, coin, a, b):
    """D9: angle at the corner of the eye between the directions towards a and b (degrees).
    Breaks: angle < 170° where the corner must stay open (threshold of D)."""
    u = pos[a] - pos[coin]
    v = pos[b] - pos[coin]
    c = float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-20))
    ang = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return {"angle": ang, "casse": ang < 170.0}


# ======================================================================= D10

def directions_cone(axe, n=8):
    """n directions perpendicular to `axe`, evenly spaced (cone of E-11)."""
    a = np.asarray(axe, dtype=float)
    a = a / np.linalg.norm(a)
    t = np.cross(a, [0.0, 0.0, 1.0])
    if np.linalg.norm(t) < 1e-6:
        t = np.cross(a, [1.0, 0.0, 0.0])
    t /= np.linalg.norm(t)
    u = np.cross(a, t)
    return [np.cos(2 * np.pi * k / n) * t + np.sin(2 * np.pi * k / n) * u for k in range(n)]


def d10_marge(positions_a, touche, amplitudes):
    """D10: first amplitude (in the given order) where `touche(positions)` is true.

    positions_a(amplitude) -> posed positions; touche(positions) -> bool (D4 against the
    body). Returns {"marge": amplitude or None if nothing touches, "derniere_libre": ...}.
    Quantity without a threshold, compared between intact and ablation."""
    libre = None
    for a in amplitudes:
        if touche(positions_a(a)):
            return {"marge": float(a), "derniere_libre": libre}
        libre = float(a)
    return {"marge": None, "derniere_libre": libre}


def rotation_chaine(sq, racine, axe_rotation, degres, rotations=None):
    """Rotations that turn the root of a chain by `degres` around `axe_rotation`
    (world frame at rest), added to an existing pose."""
    r = dict(rotations or {})
    b = sq.index(racine)
    r[b] = cin.q_mul(r.get(b, cin.IDENTITE), cin.q_axe_angle(axe_rotation, degres))
    return r

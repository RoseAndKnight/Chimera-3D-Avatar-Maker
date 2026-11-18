"""Anatomical landmarks of the trunk, measured on a surface (Blender's mathutils, MMD frame: y up, the
character faces -z, 1 unit = 8 cm).

The skeleton does not say where the anatomy is: the spine bones are placed differently by each rig (VRoid's
upper chest bone sits 4 cm under the neck and its spine bone 5 cm above the hips; a ZZZ model's 上半身2 is
under the breasts and its 上半身 at the waist). Only the joints a limb turns about (hips 足, shoulders 腕)
are anatomical everywhere. The landmarks are therefore read on the surface itself:

- C, crotch: lowest point of the trunk between the legs (a ray cast upwards at x = 0);
- W, waist: narrowest horizontal section (perimeter) between the hips and the underbust;
- U, underbust: below the bust apex, where the front profile has receded by 60 % of the breast's projection;
- B, bust apex: most forward point of the front of the chest;
- S, shoulders: height of the shoulder joints (腕), from the skeleton.

Sections are read from the inside out (from the trunk axis, first surface met): on a source whose skin was
removed under its clothes, the innermost garment stands in for the body, as in `transport.py`.
"""
import math

import numpy as np

CM = 8.0                   # cm per MMD unit
PAS_Y = 0.5 / CM           # step between sections (0.5 cm)
SECTEURS = 32              # directions around the trunk axis
AVANT = math.radians(60)   # half-angle of the front sector (bust apex)
RECUL_U = 0.6              # share of the breast's projection lost at the underbust
RAYON_MIN = 0.4            # a waist section reaches at least this share of the hips' mean radius
NOMS = ("C", "W", "U", "B", "S")


def _joint(oss, noms):
    idx = {o["nom"]: i for i, o in enumerate(oss)}
    p = [np.asarray(oss[idx[n]]["pos"], float) for n in noms if n in idx]
    return np.mean(p, 0) if len(p) == len(noms) else None


def axe_z(oss):
    """z of the trunk axis as a function of height: interpolated between the spine joints (any rig has them;
    only their heights differ)."""
    pts = [_joint(oss, (n,)) for n in ("下半身", "上半身", "上半身1", "上半身2", "首")]
    pts = sorted([p for p in pts if p is not None], key=lambda p: p[1])
    ys = np.array([p[1] for p in pts])
    zs = np.array([p[2] for p in pts])
    return lambda y: np.interp(y, ys, zs)


def arbre(P, F):
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons([tuple(p) for p in np.asarray(P, float)],
                                [tuple(int(x) for x in f) for f in np.asarray(F, np.int64)], all_triangles=True)


def sections(bvh, oss, y0, y1, pas=PAS_Y):
    """Per height y (y0..y1): hit points of rays cast from the trunk axis outwards (SECTEURS directions in the
    horizontal plane), first surface met. Returns (ys, angles, points (ny, ns, 3), NaN where nothing was hit).
    The axis is re-centred once between front and back (a dressed source can have a different depth)."""
    ys = np.arange(y0, y1 + 1e-9, pas)
    ang = (np.arange(SECTEURS) + 0.5) / SECTEURS * 2 * np.pi
    D = np.stack([np.sin(ang), np.zeros_like(ang), -np.cos(ang)], 1)     # angle 0 = front (-z), pi/2 = +x
    zax = axe_z(oss)
    out = np.full((len(ys), SECTEURS, 3), np.nan)
    for j, y in enumerate(ys):
        c = np.array([0.0, y, float(zax(y))])
        for _ in range(2):
            for s in range(SECTEURS):
                loc = bvh.ray_cast(tuple(c), tuple(D[s]), 5.0)[0]
                out[j, s] = np.nan if loc is None else np.asarray(loc)
            av, ar = out[j, 0], out[j, SECTEURS // 2]
            if np.isnan(av).any() or np.isnan(ar).any():
                break
            cz = 0.5 * (av[2] + ar[2])
            if abs(cz - c[2]) < 0.05:
                break
            c[2] = cz
    return ys, ang, out


def _perimetre(pts):
    ok = ~np.isnan(pts[:, 0])
    if ok.sum() < 0.75 * len(pts):
        return np.nan
    q = pts[ok]
    return float(np.linalg.norm(np.roll(q, -1, 0) - q, axis=1).sum())


def entrejambe(bvh, oss, epaisseur=3.0 / CM):
    """Height of the crotch: rays cast upwards at x = 0, under the hips, across the depth of the trunk;
    the lowest hit that stays below the hips."""
    h = _joint(oss, ("左足", "右足"))
    zax = axe_z(oss)
    z0 = float(zax(h[1]))
    best = np.nan
    for dz in np.linspace(-epaisseur, epaisseur, 7):
        loc = bvh.ray_cast((0.0, h[1] - 3.0, z0 + dz), (0.0, 1.0, 0.0), 3.0 + 0.5)[0]
        if loc is not None and loc[1] < h[1] + 0.5:
            best = loc[1] if np.isnan(best) else min(best, loc[1])
    return best


def avant_profil(ang, pts, zc):
    """Forward extent (-z, relative to zc) of the front sector, per section."""
    m = (np.abs(((ang + np.pi) % (2 * np.pi)) - np.pi) <= AVANT)
    f = -(pts[:, m, 2] - zc[:, None])
    with np.errstate(all="ignore"):
        return np.nanmax(np.where(np.isnan(f), -np.inf, f), 1)


def pli(ys, front, j_apex, portee):
    """Underbust: going down from the apex, first section where the front has receded by RECUL_U of the
    projection (drop between the apex and the deepest point within `portee` below it). Returns (y, projection)."""
    lo = np.searchsorted(ys, ys[j_apex] - portee)
    seg = front[lo:j_apex + 1]
    if not np.isfinite(seg).any():
        return np.nan, 0.0
    R = front[j_apex] - np.nanmin(np.where(np.isfinite(seg), seg, np.nan))
    for j in range(j_apex, lo - 1, -1):
        if np.isfinite(front[j]) and front[j] <= front[j_apex] - RECUL_U * R:
            return float(ys[j]), float(R)
    return float(ys[lo]), float(R)


def reperes(P, F, oss):
    """Landmarks C, W, U, B, S (heights, MMD units) of the body surface (P, F) with skeleton `oss`;
    plus the bust projection and the section data (for drawings)."""
    hanche = _joint(oss, ("左足", "右足"))
    epaule = _joint(oss, ("左腕", "右腕"))
    L = float(epaule[1] - hanche[1])
    b = arbre(P, F)
    ys, ang, pts = sections(b, oss, hanche[1] - 0.05 * L, epaule[1])
    zc = axe_z(oss)(ys)
    front = avant_profil(ang, pts, zc)
    zone = (ys >= hanche[1] + 0.45 * L) & (ys <= epaule[1] - 0.05 * L) & np.isfinite(front)
    jb = int(np.where(zone)[0][np.argmax(front[zone])])
    yU, proj = pli(ys, front, jb, 0.3 * L)
    per = np.array([_perimetre(pts[j]) for j in range(len(ys))])
    # A section whose rays stop close to the axis met something inside the trunk (a gap between two garments
    # of a dressed source, a belt's inner side): it says nothing about the waist.
    with np.errstate(all="ignore"):
        rad = np.hypot(pts[..., 0], pts[..., 2] - zc[:, None])
    r_hanche = np.nanmean(rad[int(np.argmin(np.abs(ys - hanche[1])))])
    per[np.nanmin(np.where(np.isnan(rad), np.inf, rad), 1) < RAYON_MIN * r_hanche] = np.nan
    zw =(ys >= hanche[1] + 0.1 * L) & (ys <= yU) & np.isfinite(per)
    yW = float(ys[zw][np.argmin(per[zw])]) if zw.any() else np.nan
    yC = entrejambe(b, oss)
    return {"C": float(yC), "W": yW, "U": yU, "B": float(ys[jb]), "S": float(epaule[1]),
            "hanche": float(hanche[1]), "projection": proj,
            "_sections": (ys, ang, pts, front, per)}


def carte(src, cib, noms=NOMS):
    """Height map from the source trunk to the target trunk: piecewise linear through the landmarks
    `noms` (default C, W, U, B, S) of both bodies, extended past the ends with the slope of the nearest
    interval. Landmarks that are missing or out of order on either side are left out. Returns (to_target, to_source)."""
    ks = [(src[n], cib[n]) for n in noms if np.isfinite(src.get(n, np.nan)) and np.isfinite(cib.get(n, np.nan))]
    ks.sort()
    a, b = [], []
    for s, t in ks:                               # strictly increasing on both sides
        if not a or (s > a[-1] + 1e-3 and t > b[-1] + 1e-3):
            a.append(s)
            b.append(t)
    a, b = np.array(a), np.array(b)
    if len(a) < 2:
        return None

    def lin(x, xa, xb):
        x = np.asarray(x, float)
        y = np.interp(x, xa, xb)
        y = np.where(x < xa[0], xb[0] + (x - xa[0]) * (xb[1] - xb[0]) / (xa[1] - xa[0]), y)
        return np.where(x > xa[-1], xb[-1] + (x - xa[-1]) * (xb[-1] - xb[-2]) / (xa[-1] - xa[-2]), y)
    return (lambda x: lin(x, a, b)), (lambda x: lin(x, b, a))

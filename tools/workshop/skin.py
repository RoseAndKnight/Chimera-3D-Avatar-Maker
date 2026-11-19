"""The skin a part hides, recomputed in the cache, without fitting the part again.

    skin.py measure [ids]     what would change, part by part (cm² given back, cm² newly hidden)
    skin.py apply [ids]   rewrites hidden_skin.bin (the old one stays next to it: hidden_skin.before.bin)

A skin face is hidden if, from its three vertices and its centre, going out along its
normal and four directions at 45° around it, each ray meets a face of the part that
really hides: held to the body AND opaque (covers.bin); a sideways look that the body itself
stops does not count; and this must still hold once the part is adjusted. Fitting looked only at the geometry: an open jacket whose opening
is made by the transparency of its texture hid the chest, and the inside of the back showed
through the body. The ray starts UNDER below the skin: where the fabric goes slightly under the skin,
the skin is what is hidden, not the fabric.
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import outfit as CB  # noqa: E402
import rays as R  # noqa: E402

REACH = 0.05   # beyond this, the part floats: the skin is seen from underneath
BIAS = 1.0     # tangent of the angle of the sideways looks (45°)
DETACHED = 0.001  # a sideways look starts this far above the skin so as not to hit its own face
FOOTWEAR = ("shoes", "boots")
TOP_FOOT = 0.15   # height of the foot in a shoe: above it is the leg
UNDER_FOOT = 0.05   # a shoe does not deform: the foot that pokes through it (up to this) is hidden, not the shoe
ISLAND = 20.0        # cm²: a smaller island of skin, isolated at the foot of a shoe, is a crumb
OUTSIDE = 0.25      # share of an isolated island outside the shoe: beyond this, it is the bare foot poking through it
BORDER_FOOT = 0.02   # the foot sticks out at most this much on the sides of a shoe (three times as much in front and behind)
GLOVES = ("gloves",)
SCABBARD = 0.03    # a glove finger does not deform: skin with both of its walls under it within this distance is the finger poking through it
SUNK = 0.001    # the fabric is under the skin from this depth on (SKIN_MIN of the review)
UNDER = 0.02     # depth under the skin where the ray starts: the one where the review looks for fabric (SKIN_REACH)


def _skin():
    V, F, N, _ = CB.skin()
    T = V[F]
    g = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = np.linalg.norm(g, axis=1) / 2 * 1e4
    nf = N[F].sum(1)
    nf /= np.maximum(np.linalg.norm(nf, axis=1), 1e-12)[:, None]
    return T, nf, area


_GRID = []


def _body():
    if not _GRID:
        V, F, _, _ = CB.skin()
        _GRID.append(R.Grid(V, F))
    return _GRID[0]


def hiding(fl, n):
    """Faces of a part that hide the skin: held and opaque. Without covers.bin, all of them."""
    if fl is None:
        return np.ones(n, bool)
    return ((fl & CB.OUTFIT) > 0) & ((fl & CB.OPAQUE) > 0)


def covered(surfaces, among=None, reach=REACH, under=UNDER, bias=True):
    """Per skin face: hidden by one of the surfaces [(vertices, hiding faces)]. `among`:
    indices of the skin faces to examine (the others return False); `reach`: how far to search;
    `under`: depth under the skin where the look starts; without `bias`, the straight look only (a
    shoe hugs the foot: nothing shows from under its edge)."""
    T, nf, _ = _skin()
    idx = np.arange(len(T)) if among is None else np.asarray(among, np.int64)
    res = np.zeros(len(T), bool)
    if not len(idx):
        return res
    pts = np.concatenate([T[idx], T[idx].mean(1)[:, None]], axis=1)          # (n, 4, 3)
    # Straight and sideways: under fabric held away from the body (a jacket at 4 cm), the skin near the
    # edge shows from underneath. Hidden only if none of these looks reaches it.
    n = nf[idx]
    t1 = np.cross(n, np.where(np.abs(n[:, [1]]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]))
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(n, t1)
    dirs = [n] + [(n + BIAS * (c * t1 + s * t2)) / np.hypot(1.0, BIAS)
                  for c, s in ((1, 0), (-1, 0), (0, 1), (0, -1)) if bias]
    D = np.stack(dirs, 1)                                                   # (n, 5, 3)
    D = np.repeat(D[:, None], 4, axis=1).reshape(-1, 3)                     # (n*4*5, 3)
    O = np.repeat(pts.reshape(-1, 3), len(dirs), axis=0) - under * D
    long = (reach + under) * np.tile([1.0] + [np.hypot(1.0, BIAS)] * (len(dirs) - 1), len(idx) * 4)
    touched = np.zeros(len(O), bool)   # per ray: two parts can hide a face between them
    for Z, F in surfaces:
        k = np.where(~touched)[0]
        if not len(F) or not len(k):
            continue
        k = k[CB.neighborhood(O[k], Z[F].reshape(-1, 3))]
        if len(k):
            touched[k] = G_pull(Z, F, O[k], D[k], long[k])
    # A sideways look that the body itself stops (in the cleft of the buttocks, between the breasts)
    # is not one: it does not make the skin visible.
    nd = len(dirs)
    face = touched.reshape(-1, 4, nd)[:, :, 0].all(1)        # those the part covers straight on
    k = np.where(~touched & np.repeat(face, 4 * nd) & (np.tile(np.arange(nd), len(idx) * 4) > 0))[0]
    if len(k):
        touched[k] = np.isfinite(_body().pull(O[k] + (under + DETACHED) * D[k], D[k], long[k] - under)[0])
    res[idx] = touched.reshape(-1, 4 * len(dirs)).all(1)
    return res


def G_pull(Z, F, O, D, long):
    """Per ray: hits the surface before its own length."""
    return R.Grid(Z, F).pull(O, D, float(long.max()))[0] <= long


def _to_foot(Z):
    """Per shod foot: indices of the skin faces in the box of the shoe, below its top
    and within TOP_FOOT of its lowest point."""
    cen = _skin()[0].mean(1)
    for side in (Z[:, 0] < 0, Z[:, 0] >= 0):
        if not side.any():
            continue
        a, b = Z[side].min(0), Z[side].max(0)
        yield np.where((cen[:, 0] > a[0] - BORDER_FOOT) & (cen[:, 0] < b[0] + BORDER_FOOT) & (cen[:, 1] < min(b[1], a[1] + TOP_FOOT))
                       & (cen[:, 2] > a[2] - 3 * BORDER_FOOT) & (cen[:, 2] < b[2] + 3 * BORDER_FOOT))[0]


def outside_sole(Z, F):
    """Per skin face: at the foot of a shoe with nothing of it underneath. A shoe is placed
    as one block: the bare foot of the body has neither its length nor its arch, and what sticks out
    (toes in front of the tip, sole under an arched sole) does not show."""
    T = _skin()[0]
    cen = T.mean(1)
    res = np.zeros(len(T), bool)
    g = R.Grid(Z, F)
    for idx in _to_foot(Z):
        if len(idx):
            t = g.pull(cen[idx], np.tile([0.0, -1.0, 0.0], (len(idx), 1)), TOP_FOOT + 0.1)[0]
            res[idx[~np.isfinite(t)]] = True
    return res


def crumbs(Z, hidden, Fp=None):
    """Per skin face: at the foot of a shoe, still shown, in an island cut off from the rest of
    the leg, and smaller than ISLAND (toe tips poking through a closed shoe) or, with the
    faces Fp of the shoe, outside it for more than OUTSIDE of its area (toes of the flat foot
    poking through the front of a heeled boot)."""
    V, F, _, _ = CB.skin()
    T, nf, area = _skin()
    g = R.Grid(Z, Fp) if Fp is not None else None
    res = np.zeros(len(F), bool)
    show = np.ones(len(F), bool)
    show[hidden] = False
    for idx in _to_foot(Z):
        zone = np.zeros(len(F), bool)
        zone[idx] = True
        f = np.where(zone & show)[0]
        if not len(f):
            continue
        free = np.zeros(len(V), bool)            # vertices of a shown face outside the zone: the leg
        free[F[show & ~zone].ravel()] = True
        e, holds = _islands(f, free)
        a = np.bincount(e, area[f], minlength=len(V))
        small = a[e] < ISLAND
        if g is not None:
            free_ = ~np.isfinite(g.pull(T[f].mean(1) + DETACHED * nf[f], nf[f], 0.5)[0])
            small |= np.bincount(e, area[f] * free_, minlength=len(V))[e] > OUTSIDE * a[e]
        res[f[~holds[e] & small]] = True
    return res


def _islands(f, free):
    """Components (by shared vertices) of the skin faces f: (label per face of f, per
    label: the component touches a `free` vertex)."""
    V, F, _, _ = CB.skin()
    lab = np.arange(len(V))
    for _ in range(200):
        m = lab[F[f]].min(1)
        before = lab.copy()
        for c in range(3):
            np.minimum.at(lab, F[f, c], m)
        if (lab == before).all():
            break
    e = lab[F[f, 0]]
    holds = np.zeros(len(V), bool)
    holds[e[free[F[f]].any(1)]] = True
    return e, holds


def sunk(Z, F, hidden):
    """Per skin face: still shown, in an island the part covers entirely straight on, cut off
    from any other shown skin, and under which the part passes (from SUNK to UNDER) somewhere. This
    island was kept only because of a sideways look slipping past a nearby edge, and it pokes through the
    fabric (the tip of a breast in front of the narrow bib of an apron). Where it is the edge of the fabric
    that dips under the skin (a skirt waistband at the back), the skin is attached to the rest: it stays."""
    V, Fp, _, _ = CB.skin()
    T, nf, _ = _skin()
    res = np.zeros(len(T), bool)
    show = np.ones(len(T), bool)
    show[hidden] = False
    f = np.where(show & covered([(Z, F)], np.where(show)[0], bias=False))[0]
    if not len(f):
        return res
    rest = show.copy()
    rest[f] = False
    free = np.zeros(len(V), bool)
    free[Fp[rest].ravel()] = True
    e, holds = _islands(f, free)
    t = R.Grid(Z, F).pull(T[f].mean(1), -nf[f], UNDER)[0]
    under = np.zeros(len(V), bool)
    under[e[np.isfinite(t) & (t > SUNK)]] = True
    res[f[~holds[e] & under[e]]] = True
    return res


def in_scabbard(Z, F, hidden):
    """Per skin face: still shown, with the two walls of a glove sheath under it within
    SCABBARD (the near one within UNDER), or in an island cut off from the arm with more than
    OUTSIDE of its area in that state. The body's hand has neither the thickness nor the pose of the glove's:
    the finger poking through its glove finger does not show, and does not half show. On a
    back of the hand left bare, the skin has only the palm of the glove under it (one wall): it stays."""
    V, Fp, _, _ = CB.skin()
    T, nf, area = _skin()
    res = np.zeros(len(T), bool)
    show = np.ones(len(T), bool)
    show[hidden] = False
    cen = T.mean(1)
    f = np.where(show)[0]
    f = f[CB.neighborhood(cen[f], Z[F].reshape(-1, 3))]
    if not len(f):
        return res
    g = R.Grid(Z, F)
    t1 = g.pull(cen[f], -nf[f], UNDER)[0]
    a = np.where(np.isfinite(t1) & (t1 > SUNK))[0]
    inside = np.zeros(len(f), bool)
    if len(a):
        t2 = g.pull(cen[f[a]] - (t1[a] + 1e-4)[:, None] * nf[f[a]], -nf[f[a]], SCABBARD)[0]
        inside[a[np.isfinite(t2) & (t1[a] + t2 <= SCABBARD)]] = True
    zone = np.zeros(len(T), bool)
    zone[f] = True
    free = np.zeros(len(V), bool)            # vertices of a shown face far from the glove: the arm
    free[Fp[show & ~zone].ravel()] = True
    e, holds = _islands(f, free)
    part = np.bincount(e, area[f] * inside, minlength=len(V)) > OUTSIDE * np.bincount(e, area[f], minlength=len(V))
    res[f[inside | (~holds[e] & part[e])]] = True
    return res


def de(p):
    """Indices of the skin faces the part hides, fitted alone: at rest, and again once
    adjusted. Adjusting pulls the free flaps out of the skin and drags the nearby fabric (up to 2 cm):
    under an edge moved away like this, the skin shows again sideways; skin with fabric under it stays
    hidden. A shoe also hides the foot sticking out of its sole, a glove the finger that
    pokes through its glove finger."""
    F = p.F[hiding(p.flags, len(p.F))]
    foot = (p.meta or {}).get("type") in FOOTWEAR
    under = UNDER_FOOT if foot else UNDER
    k = np.where(covered([(p.pos, F)], under=under, bias=not foot))[0]
    Z = CB.resolve([p])[0][0].Z
    if len(k) and np.abs(Z - p.pos).max() > 1e-4:
        k = k[covered([(Z, F)], k, REACH + CB.OUTPUT_MAX + 0.002, under, not foot)[k]]
    k = np.union1d(k, np.where(sunk(Z, F, k))[0])
    if foot:
        k = np.union1d(k, np.where(outside_sole(Z, p.F))[0])
        k = np.union1d(k, np.where(crumbs(Z, k, p.F))[0])
    if (p.meta or {}).get("type") in GLOVES:
        k = np.union1d(k, np.where(in_scabbard(Z, F, k))[0])
    return k.astype(np.uint32)


def of_part(ident):
    return de(CB.part_name(ident))


def read(ident):
    try:
        return np.fromfile(os.path.join(C.CACHE, "parts", ident, "hidden_skin.bin"), "<u4")
    except OSError:
        return np.zeros(0, np.uint32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("measure", "apply"))
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args()
    import review as RV
    _, _, area = _skin()
    rendered = plus = 0.0
    for i in a.ids or RV.ids_of_cache(True):
        d = os.path.join(C.CACHE, "parts", i)
        if (C.read_json(os.path.join(d, "meta.json")) or {}).get("class") != "garment":
            continue   # an accessory hides no skin
        try:
            before, after = read(i), of_part(i)
        except Exception as e:
            print("%-34s not recomputed: %s" % (i, str(e)[:120]))
            continue
        before = before[before < len(area)]
        r = float(area[np.setdiff1d(before, after)].sum())
        s = float(area[np.setdiff1d(after, before)].sum())
        rendered, plus = rendered + r, plus + s
        if r >= 1 or s >= 1:
            print("%-34s skin given back %7.1f cm²   newly hidden %7.1f cm²" % (i, r, s))
        if a.command == "apply" and (len(before) != len(after) or (before != after).any()):
            ch = os.path.join(d, "hidden_skin.bin")
            if os.path.isfile(ch) and not os.path.isfile(os.path.join(d, "hidden_skin.before.bin")):
                shutil.copyfile(ch, os.path.join(d, "hidden_skin.before.bin"))
            C.write_bytes(ch, after.tobytes())
            m = C.read_json(os.path.join(d, "meta.json"))
            m["hidden_skin"] = int(len(after))
            C.write_json(os.path.join(d, "meta.json"), m)
    print("total: skin given back %.0f cm², newly hidden %.0f cm²" % (rendered, plus))


if __name__ == "__main__":
    main()

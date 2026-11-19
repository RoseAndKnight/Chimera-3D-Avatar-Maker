"""Check of an outfit's layers, independent of the layer settling (`outfit.py`).

The settling casts rays from the vertices, along their line; this check looks at whole
triangles. It asks two exact questions, on the triangles as they are displayed (masked faces
removed):

- **crossings**: does an edge of one part go through a face of the other? Two surfaces with no
  crossing at all cannot poke through each other;
- **outside**: from points scattered on the faces of the lower part, does a ray cast towards the
  body meet the upper part? If so, that point is in front of it: it shows although it should be
  covered. The depth is the length of the ray. A free flap far in front (a frill, a flared skirt
  hem under a jacket) is counted separately: it flares over on purpose.

    python3 check_layers.py --parts Atelier__pull,Lucy__jacket     one outfit
    python3 check_layers.py                                        the reference outfits
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outfit as CB  # noqa: E402
import rays as R  # noqa: E402
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import games as games  # noqa: E402

LENGTH = 0.06     # ray towards the body: beyond it, the upper part is no longer "just behind"

# Reference outfits: what is really put together in the workshop, from tight layers to loose ones. They name
# catalogue parts (character__id), so from the active game and the house: description tools/games/<game>/game.json,
# "etudes.outfits_layers".
OUTFITS = games.active().value("etudes.outfits_layers", {})


def seeds(V, F):
    """Four points per face (the centre and three inner points) and the area each one stands for."""
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    W = np.array([[1, 1, 1], [4, 1, 1], [1, 4, 1], [1, 1, 4]], float)
    W /= W.sum(1)[:, None]
    P = np.einsum("kj,fjc->fkc", W, T).reshape(-1, 3)
    return P, np.repeat(area / 4, 4)


def pair(L, U, cp):
    """Check of part L under part U. None if they do not overlap."""
    Fl, Fu = L.F[~L.mask], U.F[~U.mask]
    if not len(Fl) or not len(Fu) or not CB._box(U.Z, L.Z, 0.02):
        return None
    gu = R.Grid(U.Z, Fu)
    X = np.concatenate([R.crossings(L.Z, Fl, U.Z, Fu, gu), R.crossings(U.Z, Fu, L.Z, Fl)])
    P, area = seeds(L.Z, Fl)
    near = CB.neighborhood(P, U.Z)
    P, area = P[near], area[near]
    if not len(P):
        return None
    # the line of each point: that of its face, going away from the skin; the ray stops at the skin
    nf = L.grid()[1][~L.mask]
    nf = nf / np.maximum(np.linalg.norm(nf, axis=1), 1e-12)[:, None]
    hf = (L.h0 - L.a)[L.Fw[~L.mask]].mean(1)
    n, h = np.repeat(nf, 4, axis=0)[near], np.repeat(hf, 4)[near]
    lon = np.minimum(np.where(np.isfinite(h), np.maximum(h, 0.0) + 0.003, LENGTH), LENGTH)
    inside = gu.pull(P, -n, lon)[0]   # U between the point and the body: the point is outside
    over = gu.pull(P, n, LENGTH)[0]    # U above the point: it is covered
    outside = np.isfinite(inside) & (inside > 2e-4)
    covered = np.isfinite(over) | outside
    # a free flap far in front of the upper part (a frill, a flared skirt hem) flares over it: not a defect
    flared = outside & np.repeat(~L.outfit[~L.mask], 4)[near] & (inside > CB.FREE_MAX)
    area_flared = float(area[flared].sum())
    outside &= ~flared
    if area[covered].sum() < 1e-4 and not len(X):
        return None
    depth = inside[outside] * 1000
    return {"under": L.id, "over": U.id,
            "crossings": int(len(X)),
            "recouvrement_cm2": round(float(area[covered].sum()) * 1e4, 1),
            "outside_cm2": round(float(area[outside].sum()) * 1e4, 1),
            "flared_cm2": round(area_flared * 1e4, 1),
            "outside_mm_median": round(float(np.median(depth)), 1) if len(depth) else 0.0,
            "outside_mm_max": round(float(depth.max()), 1) if len(depth) else 0.0,
            "points": P[outside], "cross": X}


def check(layers):
    """Every pair of wrapping parts of a settled outfit (layers ordered from inside to outside)."""
    cp = CB.body()
    env = [c for c in layers if c.hull]
    out = []
    for i, L in enumerate(env):
        for U in env[i + 1:]:
            r = pair(L, U, cp)
            if r:
                out.append(r)
    return out


def assessment(res):
    return {"crossings": sum(r["crossings"] for r in res),
            "flared_cm2": round(sum(r["flared_cm2"] for r in res), 1),
            "outside_cm2": round(sum(r["outside_cm2"] for r in res), 1),
            "outside_mm_max": max([r["outside_mm_max"] for r in res] or [0.0])}


def outfit(ids):
    parts = {i: CB.part_name(i) for i in ids}
    ret = CB.removed({i: p.meta for i, p in parts.items()})
    layers, notes = CB.resolve([p for i, p in parts.items() if i not in ret])
    return layers, check(layers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", dest="parts", help="comma-separated identifiers; otherwise the reference outfits")
    ap.add_argument("--json", help="writes the details to this file")
    a = ap.parse_args()
    outfits = {"outfit": a.parts.split(",")} if a.parts else OUTFITS
    all_ = {}
    for name, ids in outfits.items():
        layers, res = outfit(ids)
        b = assessment(res)
        print("%s: %d crossings, %.1f cm² outside, worst %.1f mm, %.1f cm² of flared flaps   (order: %s)" % (
            name, b["crossings"], b["outside_cm2"], b["outside_mm_max"], b["flared_cm2"], " < ".join(c.id for c in layers)))
        for r in res:
            if r["crossings"] or r["outside_cm2"] >= 0.5:
                print("    %-22s under %-22s %5d crossings, %7.1f cm² outside of %7.1f, median %.1f mm, worst %.1f mm" % (
                    r["under"], r["over"], r["crossings"], r["outside_cm2"], r["recouvrement_cm2"],
                    r["outside_mm_median"], r["outside_mm_max"]))
        all_[name] = {"assessment": b, "pairs": [{k: v for k, v in r.items() if k not in ("points", "cross")} for r in res]}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(all_, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

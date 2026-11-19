"""Check of the shoes of the cache: is the foot inside the shoe?

Asked on what the workshop shows (the adjusted part, the skin it leaves visible), foot by foot:

- **under the ground** (`under_cm`): how far the shoe goes below the ground the bare body stands on
  (its sole), once the body is raised by `lever_cm`. A heel worn on a flat foot sinks into the
  ground by its height;
- **skin left out** (`free_cm2`, `free_over_cm2` for the part facing up): area of visible skin,
  at the height of the shoe and within its outline, that no face of the shoe covers: a ray cast from
  the skin along its normal leaves without meeting the shoe. A foot standing above a shoe that sank
  shows here, as does the skin seen through an opening the shoe has by design (compare a shoe with
  itself, before and after);
- **skin through** (`traverse_cm2`): the part of it with the shoe just under the skin: toes or the
  instep that went through the top of the shoe;
- **raised** (`lever_cm`): the height the body is raised by to stand on the shoe's sole (0 when the
  foot is not posed);
- **torn** (`torn_cm2`, `torn_faces`, `stretched_max`): a shoe is made of rigid pieces (sole, upper, buckle, strap,
  tongue: each connected piece of its source mesh). Each edge's length, as displayed, is compared with
  its length on the source, relative to its piece's own median (the piece may be scaled as a whole):
  the area of faces with an edge stretched or shrunk past STRETCHED, and the worst ratio;
- **width** (`width`): the shoe's width at the bottom (lowest BOTTOM_WIDTH) relative to the source
  shoe's width scaled like the body (`agrandie`), and to the bare foot's width (`on_foot`).

    python3 check_feet.py                         every shoe of the cache
    python3 check_feet.py --parts a,b --sheet feet.png
    python3 check_feet.py --json measures.json     measurements written as JSON
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import outfit as CB  # noqa: E402
import coverage as K  # noqa: E402
import geom as G  # noqa: E402
import contact_sheet as PL  # noqa: E402
import rays as R  # noqa: E402
import pmx as PMX  # noqa: E402

PMX_M = 0.08        # metres per PMX unit

STRETCHED = 1.5         # an edge this many times longer (or shorter) than its piece's median ratio is torn
BOTTOM_WIDTH = 0.04  # metres above the bottom where the widths are taken
TOP_FOOT = 0.16    # metres: above this, it is no longer the foot (a boot's shaft is checked elsewhere)
RADIUS = 0.05        # ray from the skin: beyond this, nothing covers it
INSIDE = 0.03       # the shoe this close inside the skin: the skin went through it (an opening leaves nothing there)
OVERFLOW = 0.01       # margin around the shoe's outline
SIDE, WIDTH = 300, 0.34


def shoes():
    out = []
    for ident in sorted(os.listdir(os.path.join(C.CACHE, "parts"))):
        d = os.path.join(C.CACHE, "parts", ident)
        try:
            m = C.read_json(os.path.join(d, "meta.json"))
        except (OSError, ValueError):
            continue
        if m.get("type") in K.TYPES_FOOT and os.path.isfile(os.path.join(d, "part.glb")):
            out.append(ident)
    return out


def _area(V, F):
    T = V[F]
    return 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)


def _source(meta):
    """(source part PMX as tables, source character PMX) of a cache part, the PMX porter fits."""
    d = C.absolute(meta["folder"])
    fi = C.read_json(os.path.join(d, "record.json"))
    g = fi.get("trimmed")
    path = os.path.join(d, g["pmx"]) if g and os.path.exists(os.path.join(d, g["pmx"])) else os.path.join(d, fi["pmx"])
    return PMX.read(path), PMX.read(fi["source"]["pmx"])


def _height(pmx):
    """Head to ankle height of a skeleton (its units), the body's scale."""
    pos = {o["name"]: o["pos"] for o in pmx["bones"]}
    return pos["頭"][1] - (pos["左足首"][1] + pos["右足首"][1]) / 2


def _chunks(V, F):
    """Connected piece of each face (vertices welded by position)."""
    _, inv = G.weld(V, 1e-6)
    lab = np.arange(inv.max() + 1)
    E = inv[F]
    while True:
        m = lab[E].min(1)
        new = lab.copy()
        for k in range(3):
            np.minimum.at(new, E[:, k], m)
        new = new[new]
        if (new == lab).all():
            break
        lab = new
    return lab[E[:, 0]]


def rip(Z, F, meta):
    """Tear of the displayed shoe (Z, F: cache frame, metres) against its source: {"torn_cm2",
    "stretched_max", "width"}; None when the faces do not match the source's."""
    try:
        S, Ssrc = _source(meta)
    except (OSError, ValueError, KeyError):
        return None
    T = PMX.to_arrays(S)
    names = [m["name"] for m in S["materials"]]
    guard = [names.index(n) for n in meta.get("materials") or [] if n in names]
    Fs = np.concatenate([T["faces"][T["mat_face"] == i] for i in guard]) if guard else T["faces"][:0]
    if len(Fs) != len(F):
        return None
    X = np.asarray(T["pos"], float) * PMX_M
    part_name = _chunks(X, Fs)
    # the cache's faces may be turned over (glTF winding: corners 0, 2, 1); its vertices keep the
    # source's order, so the corners' ranks tell which faces were
    def lengths(V, Fx):
        return np.stack([np.linalg.norm(V[Fx[:, k]] - V[Fx[:, (k + 1) % 3]], axis=1) for k in range(3)], 1)
    ls, lo = lengths(X, Fs), lengths(Z, F)
    turns = (np.argsort(np.argsort(Fs, 1), 1) != np.argsort(np.argsort(F, 1), 1)).any(1)
    lo[turns] = lo[turns][:, [2, 1, 0]]
    ok = ls > 5e-4
    r = np.where(ok, lo / np.maximum(ls, 1e-12), np.nan)
    rel = np.full_like(r, np.nan)
    for k in np.unique(part_name):
        m = part_name == k
        med = np.nanmedian(r[m]) if np.isfinite(r[m]).any() else np.nan
        rel[m] = r[m] / med
    dev = np.nanmax(np.abs(np.log(np.where(np.isfinite(rel), rel, 1.0))), 1)
    torn = dev > np.log(STRETCHED)
    area = _area(Z, F)
    res = {"torn_cm2": round(1e4 * float(area[torn].sum()), 1), "torn_faces": int(torn.sum()),
           "stretched_max": round(float(np.exp(dev.max())) if len(dev) else 1.0, 2)}
    # widths at the bottom: displayed, source scaled like the body, bare foot
    samp = _height(PMX.read(C.BODY)) / _height(Ssrc)
    Vn = CB.skin()[0]
    wid = {}
    for name, side in (("left", 1), ("right", -1)):
        u = np.unique(F)
        zo = Z[u][side * Z[u][:, 0] > 0]
        xs = X[np.unique(Fs)]
        xs = xs[side * xs[:, 0] > 0]
        nu = Vn[(side * Vn[:, 0] > 0) & (Vn[:, 1] < TOP_FOOT)]
        if not len(zo) or not len(xs) or not len(nu):
            continue
        bottom = lambda P: P[P[:, 1] < P[:, 1].min() + BOTTOM_WIDTH]
        w_o, w_s = np.ptp(bottom(zo)[:, 0]), np.ptp(bottom(xs)[:, 0]) * samp
        w_p = np.ptp(nu[nu[:, 1] < nu[:, 1].min() + BOTTOM_WIDTH][:, 0])
        wid[name] = {"agrandie": round(float(w_o / w_s), 2), "on_foot": round(float(w_o / w_p), 2)}
    res["width"] = wid
    return res


def measure(ident):
    """Measurements of one shoe, per foot (left = +x), on the scene the workshop shows."""
    p = CB.part_name(ident)
    layers, _ = CB.resolve([p])
    objects = PL.scene(layers)
    Vp, Fp = objects[0][0], objects[0][1]          # visible skin
    Vs, Fs = objects[1][0], objects[1][1]          # shoe as displayed
    Vn = CB.skin()[0]                            # whole bare skin (the sole, even hidden)
    grid = R.Grid(Vs, Fs)
    c = layers[0]
    tear = rip(CB.place_outfit(layers)[1].get(c.id, c.Z), c.F, p.meta)
    foot = p.meta.get("foot") or {}
    res = {"id": ident, "kept": p.meta.get("kept"), "over": p.meta.get("instep"),
           "lift_cm": round(100 * float(foot.get("lift", 0.0)), 1), "sides": {},
           "rip": tear}
    for name, side in (("left", 1), ("right", -1)):
        s = Vs[np.unique(Fs)] if len(Fs) else Vs[:0]
        s = s[(side * s[:, 0] > 0) & (s[:, 1] < TOP_FOOT)]
        nu = Vn[(side * Vn[:, 0] > 0) & (Vn[:, 1] < TOP_FOOT)]
        if not len(s) or not len(nu):
            continue
        sole = float(nu[:, 1].min())
        lo, hi = s.min(0) - OVERFLOW, s.max(0) + OVERFLOW
        g = Vp[Fp].mean(1)
        zone = (side * g[:, 0] > 0) & (g[:, 1] < min(hi[1], TOP_FOOT)) & (g[:, 1] > sole + 0.002) \
            & (g[:, 0] > lo[0]) & (g[:, 0] < hi[0]) & (g[:, 2] > lo[2]) & (g[:, 2] < hi[2])
        Fz = Fp[zone]
        n = np.cross(Vp[Fz][:, 1] - Vp[Fz][:, 0], Vp[Fz][:, 2] - Vp[Fz][:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
        t, _ = grid.pull(g[zone] + 1e-4 * n, n, RADIUS)
        t_in, _ = grid.pull(g[zone] - 1e-4 * n, -n, INSIDE)
        free = ~np.isfinite(t)                           # nothing over it
        crosses = free & np.isfinite(t_in)              # and the shoe under it: it went through
        a = _area(Vp, Fz)
        top = n[:, 1] > K.FOOT_NORMAL
        res["sides"][name] = {
            "under_cm": round(100 * (sole - float(s[:, 1].min())) - res["lift_cm"], 1),
            "free_cm2": round(1e4 * float(a[free].sum()), 1),
            "free_over_cm2": round(1e4 * float(a[free & top].sum()), 1),
            "crosses_cm2": round(1e4 * float(a[crosses].sum()), 1),
            "center": [float(x) for x in (s.min(0) + s.max(0)) / 2],
            "faces_outside": np.where(zone)[0][free].tolist(),
        }
    return res, objects


def line(objects, mes):
    """Front, outer side, back, top, three-quarter, cut through the middle of the foot (left foot);
    skin left out in red, the ground (y = 0) as a red line."""
    c = np.array(mes["sides"].get("left", next(iter(mes["sides"].values())))["center"])
    c[1] = 0.07
    skin = list(objects[0])
    k = mes["sides"].get("left", {}).get("faces_outside", [])
    guard = np.ones(len(skin[1]), bool)
    guard[k] = False
    objs = [(skin[0], skin[1][guard], skin[2]), (skin[0], skin[1][~guard], (230, 40, 40))] + list(objects[1:])
    views = [np.array(v, float) for v in ((0, 0, 1), (1, 0, 0), (0, 0, -1), (0, 1, 1e-3), (0.7, 0.3, 0.7))]
    imgs = [PL.thumbnail(objs, c, v / np.linalg.norm(v), SIDE, WIDTH, circle=False) for v in views]
    cutting = []
    for Vo, Fo, color in objs:
        g = Vo[Fo].mean(1) if len(Fo) else np.zeros((0, 3))
        cutting.append((Vo, Fo[(g[:, 0] > c[0]) & (g[:, 0] < c[0] + 0.25)], color))
    imgs.append(PL.thumbnail(cutting, c, np.array([-1.0, 0, 0]), SIDE, WIDTH, circle=False))
    img = np.concatenate(imgs, 1)
    ground = int(SIDE / 2 + c[1] * SIDE / WIDTH)
    for j in (0, 1, 2, 5):
        if 0 <= ground < SIDE:
            img[ground, j * SIDE:(j + 1) * SIDE] = (200, 40, 40)
    return img


def p_v(ident):
    return (C.read_json(os.path.join(C.CACHE, "parts", ident, "meta.json")) or {}).get("v_fitting")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", dest="parts", help="comma-separated part ids (default: every shoe of the cache)")
    ap.add_argument("--sheet", dest="sheet", help="PNG contact sheet, one row per shoe")
    ap.add_argument("--json", help="measurements as JSON")
    a = ap.parse_args()
    ids = a.parts.split(",") if a.parts else shoes()
    all_, lines = [], []
    for ident in ids:
        mes, objects = measure(ident)
        if not mes["sides"]:
            continue
        g = mes["sides"]
        print("%-32s %-5s v%s over %-5s lift %4.1f cm  under %s cm  free %s cm2 (over %s)  crosses %s cm2" % (
            ident, mes["kept"], p_v(ident), mes["over"], mes["lift_cm"],
            "/".join("%4.1f" % v["under_cm"] for v in g.values()),
            "/".join("%5.1f" % v["free_cm2"] for v in g.values()),
            "/".join("%5.1f" % v["free_over_cm2"] for v in g.values()),
            "/".join("%4.1f" % v["crosses_cm2"] for v in g.values())))
        t = mes["rip"]
        if t:
            print("%-32s torn %5.1f cm2, %d faces (worst x%.2f)  width vs source %s  vs foot %s" % (
                "", t["torn_cm2"], t["torn_faces"], t["stretched_max"], "/".join("%.2f" % v["agrandie"] for v in t["width"].values()),
                "/".join("%.2f" % v["on_foot"] for v in t["width"].values())))
        if a.sheet:
            lines.append(line(objects, mes))
        for v in g.values():
            v.pop("faces_outside")
        all_.append(mes)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(all_, f, ensure_ascii=False, indent=1)
    if a.sheet and lines:
        PL.write_png(a.sheet, np.concatenate(lines, 0))
        print(a.sheet)


if __name__ == "__main__":
    main()

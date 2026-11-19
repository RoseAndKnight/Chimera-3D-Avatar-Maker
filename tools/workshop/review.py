"""Defect review: where the visible errors are, and how many are left.

Each defect is a measured fact, with its place (centre, in metres, in the page's frame) and its size:
the list says where to look, instead of browsing outfits by eye.

**Per part**: the fitted part is compared with its original shape (the library's PMX, same vertices
in the same order). The transfer changes the size of a part, not its shape: whatever departs from the
part's overall scale is a defect.

- `shine`      stretched faces (an edge at least `SHINE` times longer than expected, and longer by at
               least `SHINE_MIN`): spikes, blades, triangles stretched across the body;
- `squash`     faces shrunk to less than `SQUASH` times their size: gathers, flattened ornaments;
- `rip`  two vertices at the same place in the original that have separated;
- `flips`   neighbouring faces whose angle changed by more than `FOLD`: sharp creases, flipped faces;
- `shape`      share of the area whose scale departs by more than `SHAPE` from the part's
               (a skirt that flares, a sleeve that swells): one figure for the part, not a place.

**Per outfit** (stacked parts, after the layers are settled):

- `traverse`   the lower part stays in front of the upper one (`check_layers.pair`);
- `hole`       a face removed as "covered" that nothing covers: you can see through;
- `hair`    the hairstyle crosses a part (once dropped over the outfit, as in the workshop);
- `skin`       the skin shows in front of a part that should cover it;
- `open_`     skin removed as "hidden" while nothing opaque hides it: you can see through the
               body (fixed by `skin.py apply`).

    python3 review.py pieces [--json sortie.json] [id ...]     every part of the cache, worst to best
    python3 review.py outfit --parts a,b,c [--hairstyle name]    one outfit
    python3 review.py sample 40 [--hairstyle name] [--healthy]   outfits drawn at random

An outfit defect at the place of a part defect is only one defect: `--healthy` only draws parts with
no defect of their own, to judge the layers and the hair alone.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import glb as GLB  # noqa: E402
import outfit as CB  # noqa: E402
import rays as R  # noqa: E402
import pmx as PMX  # noqa: E402  (through common's sys.path)
import model as MD  # noqa: E402

SHINE = 2.0          # an edge stretched by this factor (relative to the part's scale)…
SHINE_MIN = 0.01     # …and lengthened by at least 1 cm
SQUASH = 0.4
SQUASH_MIN = 0.01
TEAR = 0.003
FOLD = 100.0          # degrees
FOLD_MIN = 0.01       # shared edge of at least 1 cm: a crease on a tiny face cannot be seen
SHAPE = 0.3
GRAVE = {"shine": 3.0, "rip": 2.0, "flips": 1.0, "squash": 1.0, "hole": 2.0, "crosses": 2.0, "open": 2.0}   # weight of one centimetre of defect


def _weld(pos):
    the_key = np.round(np.asarray(pos) * 1e5).astype(np.int64)
    _, inv = np.unique(the_key, axis=0, return_inverse=True)
    return inv.ravel()


def origin(meta):
    """Vertices of the part in its original shape (metres), or None if the library no longer has it."""
    src = C.absolute(meta.get("folder") or "")
    try:
        with open(os.path.join(src, "record.json"), encoding="utf-8") as fh:
            card = json.load(fh)
    except OSError:
        return None
    trimmed = card.get("trimmed") or {}
    path = os.path.join(src, trimmed.get("pmx") or card["pmx"])
    if not os.path.exists(path):
        path = os.path.join(src, card["pmx"])
    if not os.path.exists(path):
        return None
    P = PMX.read(path)
    T = PMX.to_arrays(P)
    names = [m["name"] for m in P["materials"]]
    guard = [i for i, n in enumerate(names) if n in set(meta.get("materials") or []) and P["materials"][i]["n_indices"] > 0]
    vs = np.unique(T["faces"][np.isin(T["mat_face"], guard)])
    S = np.asarray(T["pos"][vs], float) * C.METER
    S[:, 2] *= -1
    return S


def _groups(F, mark):
    """Marked faces grouped by adjacency: list of arrays of face indices."""
    idx = np.where(mark)[0]
    if not len(idx):
        return []
    Fm = F[idx]
    lab = MD.components(int(F.max()) + 1, np.concatenate([Fm[:, [0, 1]], Fm[:, [1, 2]]]))
    lf = lab[Fm[:, 0]]
    return [idx[lf == k] for k in np.unique(lf)]


def _spot(Z, F, faces):
    T = Z[F[faces]]
    c = T.mean(1)
    a = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    center = (c * a[:, None]).sum(0) / max(a.sum(), 1e-12) if a.sum() > 0 else c.mean(0)
    return [round(float(x), 3) for x in center], float(a.sum())


def shapes(S, Z, F):
    """Shape defects of the fitted part Z relative to the original S (same vertices, faces F)."""
    w = _weld(S)
    Fw = w[F]
    real = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 2] != Fw[:, 0])
    corners = [(0, 1), (1, 2), (2, 0)]
    ls = np.stack([np.linalg.norm(S[F[:, i]] - S[F[:, j]], axis=1) for i, j in corners], 1)
    lz = np.stack([np.linalg.norm(Z[F[:, i]] - Z[F[:, j]], axis=1) for i, j in corners], 1)
    ok = real & (ls > 1e-5).all(1)
    if not ok.any():
        return {"scale": None, "shape": None, "defaults": []}
    r = np.where(ls > 1e-5, lz / np.maximum(ls, 1e-5), 1.0)
    area = 0.5 * np.linalg.norm(np.cross(S[F[:, 1]] - S[F[:, 0]], S[F[:, 2]] - S[F[:, 0]]), axis=1)
    # scale of the part: median of the ratios, weighted by area
    rr, pp = r[ok].ravel(), np.repeat(area[ok], 3)
    o = np.argsort(rr)
    r0 = float(rr[o][np.searchsorted(np.cumsum(pp[o]), pp.sum() / 2)])
    q = r / r0
    excess = lz - r0 * ls
    defaults = []

    def note(kind, mark, measure):
        for faces in _groups(Fw, mark & ok):
            center, a = _spot(Z, F, faces)
            m = float(measure[faces].max())
            defaults.append({"kind": kind, "spot": center, "faces": int(len(faces)), "cm": round(100 * m, 1),
                            "area_cm2": round(1e4 * a, 1)})

    note("shine", ((q >= SHINE) & (excess >= SHINE_MIN)).any(1), excess.max(1))
    note("squash", ((q <= SQUASH) & (-excess >= SQUASH_MIN)).any(1), (-excess).max(1))
    # tear: vertices that coincided in the original, separated once fitted
    n = int(w.max()) + 1
    cen = np.zeros((n, 3))
    np.add.at(cen, w, Z)
    cen /= np.bincount(w, minlength=n)[:, None]
    gap = np.linalg.norm(Z - cen[w], axis=1)
    note("rip", (gap[F] >= TEAR).any(1), gap[F].max(1) * 2)
    # flip: the angle between two neighbouring faces changed by more than FOLD
    E, _, ef = MD.edges_faces(Fw)
    ns, nz = MD.normals_faces(S, F)[0], MD.normals_faces(Z, F)[0]
    order = np.argsort(ef.ravel(), kind="stable")
    ar, fa = ef.ravel()[order], np.repeat(np.arange(len(F)), 3)[order]
    two = np.where(ar[1:] == ar[:-1])[0]
    f1, f2 = fa[two], fa[two + 1]
    cs, cz = (ns[f1] * ns[f2]).sum(1), (nz[f1] * nz[f2]).sum(1)
    d = np.abs(np.degrees(np.arccos(np.clip(cs, -1, 1))) - np.degrees(np.arccos(np.clip(cz, -1, 1))))
    e = E[ar[two]]
    long_ = np.linalg.norm(Z[_rep(w)[e[:, 0]]] - Z[_rep(w)[e[:, 1]]], axis=1)
    fold = (d >= FOLD) & (long_ >= FOLD_MIN) & ok[f1] & ok[f2]
    mf, lf = np.zeros(len(F), bool), np.zeros(len(F))
    mf[f1[fold]] = mf[f2[fold]] = True
    np.maximum.at(lf, f1[fold], long_[fold])
    np.maximum.at(lf, f2[fold], long_[fold])
    note("flips", mf, lf)
    out_of = (np.abs(np.log(np.clip(q, 1e-3, 1e3))) > np.log(1 + SHAPE)).any(1) & ok
    return {"scale": round(r0, 3), "shape": round(float(area[out_of].sum() / max(area[ok].sum(), 1e-12)), 3),
            "defaults": defaults}


def _rep(w):
    """One representative vertex of each welded vertex."""
    rep = np.zeros(int(w.max()) + 1, np.int64)
    rep[w] = np.arange(len(w))
    return rep


def note(defaults):
    """Severity of a list of defects: centimetres of defect, weighted by kind."""
    return round(sum(GRAVE.get(d["kind"], 1.0) * d["cm"] for d in defaults), 1)


def review_part(ident):
    p = CB.part_name(ident)
    S = origin(p.meta)
    r = {"id": ident, "v_fitting": p.meta.get("v_fitting"), "type": p.meta.get("type")}
    if S is None or len(S) != len(p.pos):
        r.update(without_origin=True, defaults=[], note=None)
        return r
    r.update(shapes(S, p.pos, p.F))
    layers = CB.resolve([p])[0]
    r["defaults"] += skin_outside(layers) + skin_open(layers)
    r["note"] = note(r["defaults"])
    return r


def ids_of_cache(all_=False):
    """The catalogue's parts; tout=True: also those its score set aside."""
    cat = C.read_json(os.path.join(C.CACHE, "catalog.json"), {})
    ids = [x["id"] for x in cat.get("parts", []) if x.get("kept", True)] if isinstance(cat, dict) and not all_ else []
    if not ids:
        d = os.path.join(C.CACHE, "parts")
        ids = sorted(i for i in os.listdir(d) if (C.read_json(os.path.join(d, i, "meta.json"), {}) or {}).get("kept"))
    return ids


# ------------------------------------------------------------------ outfit
CELL = 0.08       # the faulty points of a pair are grouped in cells of this size: one place per cell
HOLE_MIN = 2.0       # cm²: under this area, a face wrongly removed cannot be seen
CROSSES_MIN = 0.5   # cm²
CROSS_MIN = 3        # crossings per cell
SKIN_REACH = 0.02   # depth under the skin where the fabric is looked for
SKIN_MIN = 0.001
OPEN_AREA = 4.0    # cm² of removed skin with nothing in front, per cell, for a defect
SKIN_AREA = 1.0      # cm² of skin in front of a part, per cell, for a defect


def _cases(P, weight=None):
    """Points grouped in cells: [(centre, count, weight)]."""
    if not len(P):
        return []
    k = np.floor(P / CELL).astype(np.int64)
    _, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv)
    c = np.stack([np.bincount(inv, P[:, j]) for j in range(3)], 1) / n[:, None]
    w = np.bincount(inv, weight) if weight is not None else n.astype(float)
    return [([round(float(x), 3) for x in c[i]], int(n[i]), float(w[i])) for i in range(len(n))]


def hairstyle(name, head, layers=None):
    """Hair placed on the head `head`: (vertices, faces), in metres; with `couches`, as the workshop
    shows them, dropped over the outfit."""
    V, N, F = CB.hairstyle(name, head)
    if layers:
        V = V + CB.style(layers, V, N, F)
    return V, F


def holes(layers):
    """Removed faces (mask) that nothing opaque covers or precedes: you can see through the outfit."""
    import check_layers as CC
    env = [c for c in layers if c.hull]
    out = []
    for i, L in enumerate(env):
        if not L.mask.any() or not env[i + 1:]:
            continue
        P, area = CC.seeds(L.Z, L.F[L.mask])
        nf = L.grid()[1][L.mask]
        n = np.repeat(nf / np.maximum(np.linalg.norm(nf, axis=1), 1e-12)[:, None], 4, axis=0)
        vu = np.ones(len(P), bool)
        for U in env[i + 1:]:
            opq = ((U.fl & CB.OPAQUE) > 0) & ~U.mask
            if not opq.any():
                continue
            g = R.Grid(U.Z, U.F[opq])
            vu &= ~np.isfinite(g.pull(P, n, CB.REACH_PAN)[0]) & ~np.isfinite(g.pull(P, -n, CB.REACH_PAN)[0])
        for center, nb, a in _cases(P[vu], area[vu]):
            if a * 1e4 >= HOLE_MIN:
                out.append({"kind": "hole", "parts": [L.id], "spot": center, "area_cm2": round(a * 1e4, 1),
                            "cm": round(100 * np.sqrt(a), 1)})
    return out


def crossings(layers):
    import check_layers as CC
    out = []
    for r in CC.check(layers):
        for center, nb, a in _cases(r["points"], None):
            a = a * r["outside_cm2"] / max(len(r["points"]), 1)
            if a >= CROSSES_MIN:
                out.append({"kind": "crosses", "parts": [r["under"], r["over"]], "spot": center,
                            "area_cm2": round(a, 1), "cm": round(np.sqrt(a), 1)})
        for center, nb, _ in _cases(r["cross"]):
            if nb >= CROSS_MIN:
                out.append({"kind": "crossed", "parts": [r["under"], r["over"]], "spot": center,
                            "crossings": nb, "cm": round(nb / 10.0, 1)})
    return out


def hair(layers, V, F):
    """The hairstyle crosses a part: strands that go into a collar, a jacket, a bag."""
    out = []
    g = R.Grid(V, F)
    for c in layers:
        Fc = c.F[~c.mask]
        if not len(Fc) or not CB._box(V, c.Z, 0.0):
            continue
        X = np.concatenate([R.crossings(c.Z, Fc, V, F, g), R.crossings(V, F, c.Z, Fc)])
        for center, nb, _ in _cases(X):
            if nb >= CROSS_MIN:
                out.append({"kind": "hair", "parts": [c.id], "spot": center, "crossings": nb, "cm": round(nb / 10.0, 1)})
    return out


def skin_outside(layers, hidden=None):
    """The skin shows in front of a part that should cover it. Measured in the body's frame: from
    each shown skin face, go down along its normal; if the front of a part is met within
    SKIN_REACH, the fabric is under the skin."""
    pe = CB.skin()
    if pe is None:
        return []
    V, F, N, _ = pe
    cache = np.zeros(len(F), bool)      # `hidden` {id: skin faces} replaces what the cache says (part being tried)
    for c in layers:
        try:
            k = hidden[c.id] if hidden and c.id in hidden else np.fromfile(os.path.join(C.CACHE, "parts", c.id, "hidden_skin.bin"), "<u4")
            cache[k[k < len(F)]] = True
        except OSError:
            pass
    out = []
    T = V[F]
    cen, nf = T.mean(1), N[F].sum(1)
    nf /= np.maximum(np.linalg.norm(nf, axis=1), 1e-12)[:, None]
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2 * 1e4
    for c in layers:
        Fc = c.F[((c.fl & CB.OUTER) > 0) & ~c.mask]
        if not len(Fc):
            continue
        idx = np.where(~cache & CB.neighborhood(cen, c.Z[Fc].reshape(-1, 3)))[0]
        if not len(idx):
            continue
        t, f = R.Grid(c.Z, Fc).pull(cen[idx], -nf[idx], SKIN_REACH)
        nc = c.nor[Fc].sum(1)
        nc /= np.maximum(np.linalg.norm(nc, axis=1), 1e-12)[:, None]
        a = np.isfinite(t) & (t > SKIN_MIN)
        a[a] &= (nf[idx][a] * nc[f[a]]).sum(1) > 0.3
        for center, nb, s in _cases(cen[idx][a], area[idx][a]):
            if s >= SKIN_AREA:
                out.append({"kind": "skin", "parts": [c.id], "spot": center, "area_cm2": round(s, 1), "cm": round(np.sqrt(s), 1)})
    return out


def skin_open(layers):
    """Skin removed as "hidden" that nothing opaque hides: you can see through the body
    (a jacket opened by the transparency of its texture, openwork tights)."""
    import skin as PE
    pe = CB.skin()
    if pe is None:
        return []
    V, F, _, _ = pe
    out = []
    surfaces = [(c.Z, c.F[((c.fl & CB.OPAQUE) > 0) & ~c.mask]) for c in layers]
    T = V[F]
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2 * 1e4
    for c in layers:
        k = PE.read(c.id)
        k = k[k < len(F)]
        if not len(k):
            continue
        nu = k[~PE.covered(surfaces, k, PE.REACH + CB.OUTPUT_MAX + 0.002, PE.UNDER_FOOT if c.type in PE.FOOTWEAR else PE.UNDER, c.type not in PE.FOOTWEAR)[k]]   # the layer settling may have pushed the part away
        if len(nu):
            nu = nu[~PE.sunk(c.Z, c.F[PE.hiding(c.fl, len(c.F))], np.setdiff1d(k, nu))[nu]]   # skin with the fabric under it is hidden on purpose
        if c.type in PE.FOOTWEAR:
            nu = nu[~PE.outside_sole(c.Z, c.F)[nu]]              # the foot sticking out of a shoe is hidden on purpose,
            nu = nu[~PE.crumbs(c.Z, np.setdiff1d(k, nu), c.F)[nu]]   # and so are the crumbs of skin left around it
        for center, nb, s in _cases(T[nu].mean(1), area[nu]):
            if s >= OPEN_AREA:
                out.append({"kind": "open", "parts": [c.id], "spot": center, "area_cm2": round(s, 1), "cm": round(np.sqrt(s), 1)})
    return out


def review_outfit(ids, hairdo=None, head=None):
    parts = {i: CB.part_name(i) for i in ids}
    ret = CB.removed({i: p.meta for i, p in parts.items()})
    layers, notes = CB.resolve([p for i, p in parts.items() if i not in ret])
    defaults = crossings(layers) + holes(layers) + skin_outside(layers) + skin_open(layers)
    if hairdo:
        defaults += hair(layers, *hairstyle(hairdo, head or hairdo, layers))
    defaults.sort(key=lambda d: -d["cm"])
    return {"parts": ids, "order": [c.id for c in layers], "defaults": defaults, "note": note(defaults)}


SOUND = 20.0   # highest score of a "sound" part: its own defects are not mistaken for the outfit's


def draw(n, seed=7, sound=False):
    """n plausible outfits drawn at random from the cache; `sound`: among parts with no defect of their own."""
    import random
    per = {}
    for i in ids_of_cache():
        if sound and not (review_part(i)["note"] or 0) <= SOUND:
            continue
        m = C.read_json(os.path.join(C.CACHE, "parts", i, "meta.json"), {})
        per.setdefault(m.get("type"), []).append(i)
    rnd = random.Random(seed)

    def pulls(*ts):
        items = [i for t in ts for i in per.get(t, [])]
        return [rnd.choice(items)] if items else []
    for _ in range(n):
        ids = pulls("top") + pulls("skirt", "shorts", "trousers")
        for part, ts in ((0.6, ("jacket", "coat", "waistcoat")), (0.7, ("socks",)), (0.8, ("shoes", "boots")), (0.4, ("belt",))):
            if rnd.random() < part:
                ids += pulls(*ts)
        yield ids


def _say_outfit(r):
    print("%s   score %s" % (" + ".join(r["parts"]), r["note"]))
    for d in r["defaults"][:12]:
        print("    %-9s %-50s at %s  %s" % (d["kind"], " / ".join(d["parts"]), d["spot"],
              "%.1f cm²" % d["area_cm2"] if "area_cm2" in d else "%d crossings" % d["crossings"]))


def _line(r):
    per = {}
    for d in r["defaults"]:
        per.setdefault(d["kind"], []).append(d["cm"])
    return "%-34s v%s  score %7s  shape %5s  %s" % (
        r["id"], r.get("v_fitting"), r["note"], r.get("shape"),
        "  ".join("%s ×%d (max %.1f cm)" % (g, len(v), max(v)) for g, v in sorted(per.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="command", required=True)
    a = sp.add_parser("parts")
    a.add_argument("ids", nargs="*")
    a.add_argument("--json")
    a.add_argument("--all", dest="all_", action="store_true", help="also the parts without defects")
    a.add_argument("--note", dest="note", action="store_true", help="writes the score into each part's record and rebuilds the "
                   "catalogue: beyond NOTE_REFUSALS (common.py), the part is removed from the page")
    b = sp.add_parser("outfit")
    b.add_argument("--parts", dest="parts", required=True)
    c = sp.add_parser("sample")
    c.add_argument("n", type=int)
    c.add_argument("--seed", dest="seed", type=int, default=7)
    c.add_argument("--healthy", dest="sound", action="store_true", help="only parts with no defect of their own: what is left comes from the layers")
    for x in (b, c):
        x.add_argument("--hairstyle", dest="hairstyle")
        x.add_argument("--head", dest="head")
        x.add_argument("--json")
    args = ap.parse_args()
    if args.command == "outfit":
        r = review_outfit(args.parts.split(","), args.hairstyle, args.head)
        _say_outfit(r)
        if args.json:
            C.write_json(args.json, r)
    if args.command == "sample":
        res = []
        for ids in draw(args.n, args.seed, args.sound):
            try:
                res.append(review_outfit(ids, args.hairstyle, args.head))
            except Exception as e:
                res.append({"parts": ids, "error": str(e)[:200], "defaults": [], "note": 0})
        res.sort(key=lambda r: -r["note"])
        for r in res:
            _say_outfit(r)
        per = {}
        for r in res:
            for d in r["defaults"]:
                per[d["kind"]] = per.get(d["kind"], 0) + 1
        print("\n%d outfits, %d without defects; %s" % (len(res), sum(1 for r in res if not r["defaults"]),
              ", ".join("%s %d" % kv for kv in sorted(per.items()))))
        if args.json:
            C.write_json(args.json, res)
    if args.command == "parts":
        res = []
        for i in args.ids or ids_of_cache(args.note):
            try:
                res.append(review_part(i))
            except Exception as e:   # an unreadable part is a result, not a stop
                res.append({"id": i, "error": str(e)[:200], "defaults": [], "note": None})
        res.sort(key=lambda r: -(r["note"] or 0))
        for r in res:
            if r.get("error") or r.get("without_origin"):
                print("%-34s not reviewed: %s" % (r["id"], r.get("error") or "original shape missing or different"))
            elif r["note"] or args.all_:
                print(_line(r))
        views = [r for r in res if r["note"] is not None]
        print("\n%d parts reviewed, %d with defects, %d not reviewed; shape > 0.25: %d" % (
            len(views), sum(1 for r in views if r["note"]), len(res) - len(views),
            sum(1 for r in views if (r.get("shape") or 0) > 0.25)))
        if args.json:
            C.write_json(args.json, res)
        if args.note:
            for r in views:
                ch = os.path.join(C.CACHE, "parts", r["id"], "meta.json")
                m = C.read_json(ch)
                kinds = {}
                for d in r["defaults"]:
                    kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
                m["revue"] = {"note": r["note"], "shape": r.get("shape"), "defaults": kinds}
                C.write_json(ch, m)
            import catalog as catalog
            n = catalog.build()["counts"]["parts"]
            print("scores written; catalogue: %d parts out of %d" % tuple(n))


if __name__ == "__main__":
    main()

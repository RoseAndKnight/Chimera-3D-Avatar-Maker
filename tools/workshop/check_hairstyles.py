"""Check of the cached hairstyles against the skin the workshop shows: does the skin come through the hair?

The hair of the cache is placed once per head (`hairstyles.py`). This check does not trust that
placement: it rebuilds the skin exactly as the page draws it (body of the cache with the head's
neck seam and hidden faces, plus the head's skin) and looks at the head and shoulders from
twelve directions, the way the page does: one depth image of the skin, one of the hair.

A **hole** is a pixel where the skin is in front and the nearest hair lies inside it, less than
THICKNESS under its surface: a strand that should cover the skull, the neck or the shoulders passes
under the skin, and the page shows skin through the hair. Hair far behind the skin (the back hair
seen through the face), hair tucked behind an ear (outside the ear, behind it) and roots planted
under the scalp that other hair covers (there, the hair is in front) are no holes.

    python3 check_hairstyles.py                       all the hairstyles of the cache, first head of the catalogue
    python3 check_hairstyles.py --hairstyles A,B --head Miyabi --sheet worst.png --max 8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact_sheet as PL  # noqa: E402  first: common puts tools/clothes (another contact_sheet.py) ahead on the path
import common as C  # noqa: E402
import glb as GLB  # noqa: E402

THICKNESS = 0.02     # metres: hair less deep than this under the shown skin should have covered it
PIXEL = 0.0015       # metres per pixel of the depth images
FRAME = 0.56         # metres seen around the head (head, neck and shoulders)
VIEW_CM2 = 2.0        # a hairstyle is flagged when one view shows more skin through it than this (a patch of 1.4 cm)
# directions of view: azimuth (0: face) and elevation, in degrees
VIEWS = [(a, e) for e in (0, 35) for a in (0, 60, 120, 180, 240, 300)]


def _arrays(path):
    doc, b = GLB.read(path)
    accessory, views = doc["accessors"], doc["bufferViews"]

    def tab(i, ty, n):
        a, v = accessory[i], views[accessory[i]["bufferView"]]
        return np.frombuffer(b, ty, a["count"] * (1 if a.get("type") == "SCALAR" else n), v.get("byteOffset", 0)).reshape(-1, n)
    pr = doc["meshes"][0]["primitives"]
    V = tab(pr[0]["attributes"]["POSITION"], "<f4", 3).astype(float)
    prims = [(doc["materials"][p["material"]]["name"], tab(p["indices"], "<u4", 3).astype(np.int64)) for p in pr]
    return V, prims


def skin_displayed(head):
    """(vertices, faces) in metres, glTF frame: the skin as the page draws it under `head`, and the whole
    head with it (eyes, mouth, brows: opaque, they hide what is behind them)."""
    V, prims = _arrays(os.path.join(C.CACHE, "body", "body.glb"))
    F = np.concatenate([f for _, f in prims])
    d = os.path.join(C.CACHE, "heads", head)
    try:
        s = np.fromfile(os.path.join(d, "body.bin"), "<f4").astype(float)
        if len(s) == 6 * len(V):
            V = s[:3 * len(V)].reshape(-1, 3)
    except OSError:
        pass
    try:
        k = np.fromfile(os.path.join(d, "hidden_skin.bin"), "<u4").astype(np.int64)
        guard = np.ones(len(F), bool)
        guard[k[k < len(F)]] = False
        F = F[guard]
    except OSError:
        pass
    Vt, pt = _arrays(os.path.join(d, "head.glb"))
    Ft = np.concatenate([f for _, f in pt]) + len(V)
    return np.concatenate([V, Vt]), np.concatenate([F, Ft])


def hairstyle_placed(name, head):
    d = os.path.join(C.CACHE, "hairstyles", name)
    V, prims = _arrays(os.path.join(d, "hairstyle.glb"))
    F = np.concatenate([f for _, f in prims])
    pose = os.path.join(d, head + ".bin")
    if not os.path.isfile(pose):
        return None
    Q = np.fromfile(pose, "<f4").reshape(-1, 3).astype(float)
    return (Q, F) if len(Q) == len(V) else None


def landmark(az, el):
    """Right, up and towards-the-camera axes of a view (glTF frame, the character faces +Z)."""
    a, e = np.radians(az), np.radians(el)
    to = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    right = np.cross([0.0, 1.0, 0.0], to)
    right /= np.linalg.norm(right)
    return right, np.cross(to, right), to


def depth(V, F, center, axes, side, upright=False):
    """Depth image (distance towards the camera, larger = nearer; -inf where nothing) of the
    triangles, by points spread over each face finer than a pixel. With `upright`, also the image
    of the nearest surface's side (True where it faces the camera; False: seen from inside) and the
    depth of the nearest surface seen from inside (where the ray leaves the skin again)."""
    right, top, to = axes
    P = V - center
    X = np.stack([(P @ right) / PIXEL + side / 2, side / 2 - (P @ top) / PIXEL, P @ to], 1)
    T = X[F]
    in_ = (T[:, :, 0].max(1) >= 0) & (T[:, :, 0].min(1) < side) & (T[:, :, 1].max(1) >= 0) & (T[:, :, 1].min(1) < side)
    T = T[in_]
    z = np.full(side * side, -np.inf)
    zd, zr = np.full(side * side, -np.inf), np.full(side * side, -np.inf)
    if not len(T):
        return (z.reshape(side, side), np.zeros((side, side), bool), zr.reshape(side, side)) if upright else z.reshape(side, side)
    # faces of the skin turn their front outwards (glb.to_gltf): in image space, counter-clockwise
    area = (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])
    L = np.maximum(np.linalg.norm(T[:, 1, :2] - T[:, 0, :2], axis=1),
                   np.maximum(np.linalg.norm(T[:, 2, :2] - T[:, 1, :2], axis=1), np.linalg.norm(T[:, 0, :2] - T[:, 2, :2], axis=1)))
    n = np.clip(np.ceil(L * 2).astype(int), 1, 400)
    for k in np.unique(n):
        t, dr = T[n == k], (area[n == k] < 0)
        i, j = np.mgrid[0:k + 1, 0:k + 1]
        g = i + j <= k
        a, b = i[g] / k, j[g] / k
        p = t[:, None, 0] * (1 - a - b)[None, :, None] + t[:, None, 1] * a[None, :, None] + t[:, None, 2] * b[None, :, None]
        p = p.reshape(-1, 3)
        x, y = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
        ok = (x >= 0) & (x < side) & (y >= 0) & (y < side)
        np.maximum.at(z, y[ok] * side + x[ok], p[ok, 2])
        if upright:
            d = np.repeat(dr, len(a))[ok]
            np.maximum.at(zd, (y[ok] * side + x[ok])[d], p[ok, 2][d])
            np.maximum.at(zr, (y[ok] * side + x[ok])[~d], p[ok, 2][~d])
    z = z.reshape(side, side)
    return (z, zd.reshape(side, side) >= z - 1e-4, zr.reshape(side, side)) if upright else z


class Scene:
    """The skin of a head, seen from every direction once."""

    def __init__(self, head):
        self.head = head
        self.V, self.F = skin_displayed(head)
        self.side = int(round(FRAME / PIXEL))
        self._z = {}

    def frame(self, Q):
        # centred on the top of the hair, a little lower: skull, neck and shoulders
        return np.array([0.0, float(Q[:, 1].max()) - FRAME * 0.42, float(np.median(Q[:, 2]))])

    def skin(self, center, view):
        the_key = (tuple(np.round(center, 3)), view)
        if the_key not in self._z:
            self._z[the_key] = depth(self.V, self.F, center, landmark(*view), self.side, upright=True)
        return self._z[the_key]


def measure(name, scene):
    cp = hairstyle_placed(name, scene.head)
    if cp is None:
        return None
    Q, F = cp
    center = scene.frame(Q)
    views, total = [], 0
    for view in VIEWS:
        zp, upright, output = scene.skin(center, view)
        zc = depth(Q, F, center, landmark(*view), scene.side)
        # the nearest hair is inside the skin: behind its outer side, before the ray leaves it again
        # (hair tucked behind an ear is behind the ear's far side: hidden on purpose, not inside).
        # Skin seen from inside (an opening of the skin) is another defect, not the hair's.
        hole = np.isfinite(zp) & upright & np.isfinite(zc) & (zc < zp) & (zc > zp - THICKNESS) & (zc > output)
        n = int(hole.sum())
        total += n
        views.append({"view": view, "pixels": n, "_zp": zp, "_zc": zc, "_hole": hole})
    cm2 = total * (PIXEL * 100) ** 2
    worst = max(views, key=lambda v: v["pixels"])
    return {"hairstyle": name, "head": scene.head, "hole_cm2": round(cm2, 2),
            "worst_view": list(worst["view"]), "worst_cm2": round(worst["pixels"] * (PIXEL * 100) ** 2, 2),
            "touchee": worst["pixels"] * (PIXEL * 100) ** 2 > VIEW_CM2, "_views": views}


def image(v):
    """What the page would show (skin, hair), holes in red, light by depth."""
    zp, zc, hole = v["_zp"], v["_zc"], v["_hole"]
    img = np.empty(zp.shape + (3,), float)
    img[:] = PL.BACKGROUND
    front_c = np.isfinite(zc) & (zc >= np.where(np.isfinite(zp), zp, -np.inf))
    front_p = np.isfinite(zp) & ~front_c
    for mask, z, color in ((front_p, zp, PL.SKIN), (front_c, zc, PL.HAIR)):
        if mask.any():
            zz = z[mask]
            lum = 0.55 + 0.45 * (zz - zz.min()) / max(float(zz.max() - zz.min()), 1e-9)
            img[mask] = np.array(color)[None, :] * lum[:, None]
    img[hole] = (230, 30, 30)
    return img.astype(np.uint8)


def sheet(results, output, max_=8, per_line=4):
    """One row per hairstyle, worst first: its four worst views."""
    lines = []
    for r in results[:max_]:
        vs = sorted(r["_views"], key=lambda v: -v["pixels"])[:per_line]
        im = [image(v)[::2, ::2] for v in vs]
        for x in im:
            x[[0, -1], :] = x[:, [0, -1]] = 60
        lines.append(np.concatenate(im, 1))
    if lines:
        PL.write_png(output, np.concatenate(lines, 0))


def head_per_default():
    cat = C.read_json(os.path.join(C.CACHE, "catalog.json")) or {}
    t = cat.get("heads") or []
    return t[0]["id"] if t else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hairstyles", dest="hairstyles", help="names, comma-separated (default: the whole cache)")
    ap.add_argument("--head", dest="head", help="head the hair is placed on (default: first of the catalogue)")
    ap.add_argument("--sheet", dest="sheet", help="PNG of the worst hairstyles (four worst views each, holes in red)")
    ap.add_argument("--max", type=int, default=8)
    ap.add_argument("--json", help="write the measures there")
    a = ap.parse_args()
    scene = Scene(a.head or head_per_default())
    back = os.path.join(C.CACHE, "hairstyles")
    names = a.hairstyles.split(",") if a.hairstyles else sorted(
        n for n in os.listdir(back) if os.path.isfile(os.path.join(back, n, "hairstyle.glb")))
    res = []
    for n in names:
        r = measure(n, scene)
        if r is None:
            continue
        res.append(r)
        print("%-14s skin through the hair %6.2f cm2 (worst view %s: %5.2f cm2)%s" % (
            n, r["hole_cm2"], tuple(r["worst_view"]), r["worst_cm2"], "  FLAGGED" if r["touchee"] else ""), flush=True)
    res.sort(key=lambda r: -r["hole_cm2"])
    touched = [r for r in res if r["touchee"]]
    print("%d / %d hairstyles show skin through the hair on %s (over %.1f cm2 in one view)" % (len(touched), len(res), scene.head, VIEW_CM2))
    if a.json:
        C.write_json(a.json, {"head": scene.head, "thickness_m": THICKNESS, "threshold_view_cm2": VIEW_CM2,
                               "hairstyles": [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]})
    if a.sheet:
        sheet(res, a.sheet, a.max)
        print(a.sheet)


if __name__ == "__main__":
    main()

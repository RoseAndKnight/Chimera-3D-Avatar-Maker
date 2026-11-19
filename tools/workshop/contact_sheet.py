"""Contact sheet of the review: one image of each flagged spot, to judge at a glance without opening Chimera.

A small drawing by program (numpy only, no textures): the skin, each part in one colour, the
hair, seen straight on at the spot (from outside the body), 30 cm wide, the spot circled in red.
The back of a face is darkened: a part turned inside out or fabric seen from inside stands out.

    python3 contact_sheet.py --parts a,b,c [--hairstyle name] [--head name] --output sheet.png [--max 12]
    python3 contact_sheet.py --part id --output sheet.png                       a part's own defects, alone on the body
    python3 contact_sheet.py --turn --parts a,b --output tour.png                overall view, six directions
    python3 contact_sheet.py --json sample.json --output folder/ [--max 12]       one contact sheet per faulty outfit
"""
import argparse
import json
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import outfit as CB  # noqa: E402
import review as RV  # noqa: E402

SIDE = 300          # pixels of a thumbnail
WIDTH = 0.30      # metres seen
SKIN = (224, 172, 140)
HAIR = (70, 50, 110)
TINTS = [(90, 150, 220), (240, 240, 235), (70, 170, 120), (230, 190, 70), (200, 110, 180), (120, 120, 130),
           (230, 130, 80), (60, 190, 200)]
BACKGROUND = (245, 240, 230)


def write_png(path, img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def block(name, d):
        return struct.pack(">I", len(d)) + name + d + struct.pack(">I", zlib.crc32(name + d) & 0xffffffff)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + block(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + block(b"IDAT", zlib.compress(raw, 6)) + block(b"IEND", b""))


def scene(layers, hairdo=None, head=None, hidden=None):
    """[(vertices, faces, colour)] of what Chimera shows: unhidden skin, adjusted parts, hair."""
    objects = []
    pe = CB.skin()
    body_d, poses = CB.place_outfit(layers)       # feet posed in a heeled shoe, as on the page
    if pe is not None:
        V, F, _, _ = pe
        V = V + body_d if body_d is not None else V
        guard = np.ones(len(F), bool)
        for c in layers:
            try:
                k = hidden[c.id] if hidden and c.id in hidden else np.fromfile(os.path.join(C.CACHE, "parts", c.id, "hidden_skin.bin"), "<u4")
                guard[k[k < len(F)]] = False
            except OSError:
                pass
        objects.append((V, F[guard], SKIN))
    for i, c in enumerate(layers):
        objects.append((poses.get(c.id, c.Z), c.F[~c.mask], TINTS[i % len(TINTS)]))
    if hairdo:
        V, F = RV.hairstyle(hairdo, head or hairdo, layers)
        objects.append((V, F, HAIR))
    return objects


def gaze(spot):
    """Direction from which to look at a spot: from outside the body, horizontally."""
    cp = CB.body()
    d = np.array([0.0, 0.0, 1.0])
    if cp is not None:
        q, n = CB._foot_body(np.array([spot], float), cp)
        v = np.array(spot, float) - q[0]
        v = v if np.linalg.norm(v) > 0.01 and (v * n[0]).sum() > 0 else n[0]
        v[1] = 0.0
        if np.linalg.norm(v) > 1e-6:
            d = v / np.linalg.norm(v)
    return d


def thumbnail(objects, spot, d=None, side=SIDE, width=WIDTH, circle=True):
    spot = np.array(spot, float)
    d = gaze(spot) if d is None else d
    top = np.array([0.0, 1.0, 0.0])
    right = np.cross(top, d)
    right /= np.linalg.norm(right)
    img = np.empty((side, side, 3), float)
    img[:] = BACKGROUND
    zb = np.full((side, side), -np.inf)
    samp = side / width
    for V, F, color in objects:
        if not len(F):
            continue
        P = V - spot
        X = np.stack([(P * right).sum(1) * samp + side / 2, side / 2 - (P * top).sum(1) * samp, (P * d).sum(1)], 1)
        T = X[F]
        vu = (T[:, :, 0].max(1) >= 0) & (T[:, :, 0].min(1) < side) & (T[:, :, 1].max(1) >= 0) & (T[:, :, 1].min(1) < side)
        T = T[vu]
        nz = np.cross(V[F[vu]][:, 1] - V[F[vu]][:, 0], V[F[vu]][:, 2] - V[F[vu]][:, 0])
        nz /= np.maximum(np.linalg.norm(nz, axis=1), 1e-12)[:, None]
        face = (nz * d).sum(1)
        for t, c in zip(T, face):
            x0, x1 = max(int(np.floor(t[:, 0].min())), 0), min(int(np.ceil(t[:, 0].max())), side - 1)
            y0, y1 = max(int(np.floor(t[:, 1].min())), 0), min(int(np.ceil(t[:, 1].max())), side - 1)
            if x1 < x0 or y1 < y0:
                continue
            den = (t[1, 1] - t[2, 1]) * (t[0, 0] - t[2, 0]) + (t[2, 0] - t[1, 0]) * (t[0, 1] - t[2, 1])
            if abs(den) < 1e-9:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            a = ((t[1, 1] - t[2, 1]) * (xs - t[2, 0]) + (t[2, 0] - t[1, 0]) * (ys - t[2, 1])) / den
            b = ((t[2, 1] - t[0, 1]) * (xs - t[2, 0]) + (t[0, 0] - t[2, 0]) * (ys - t[2, 1])) / den
            g = 1 - a - b
            z = a * t[0, 2] + b * t[1, 2] + g * t[2, 2]
            zz = zb[y0:y1 + 1, x0:x1 + 1]
            ok = (a >= 0) & (b >= 0) & (g >= 0) & (z > zz)
            if not ok.any():
                continue
            lum = (0.45 + 0.55 * abs(c)) * (1.0 if c > 0 else 0.55)   # back side darkened
            zz[ok] = z[ok]
            img[y0:y1 + 1, x0:x1 + 1][ok] = np.array(color) * lum
    yy, xx = np.mgrid[0:side, 0:side]
    r = np.hypot(xx - side / 2, yy - side / 2)
    if circle:
        img[np.abs(r - 0.05 * samp) < 1.0] = (230, 30, 30)
    img[[0, -1], :] = img[:, [0, -1]] = 60
    return img.astype(np.uint8)


def girth(ids, hairdo, head, output, side=420):
    """Overall view of an outfit: front, three-quarter, profile, back, framed on the parts."""
    parts = {i: CB.part_name(i) for i in ids}
    ret = CB.removed({i: p.meta for i, p in parts.items()})
    layers, _ = CB.resolve([p for i, p in parts.items() if i not in ret])
    objects = scene(layers, hairdo, head)
    P = np.concatenate([c.Z for c in layers])
    center = (P.min(0) + P.max(0)) / 2
    width = float(max((P.max(0) - P.min(0)).max() * 1.15, 0.4))
    views = [np.array([np.sin(t), 0.0, np.cos(t)]) for t in (0.0, 0.8, 1.57, 3.14, -1.57, -0.8)]
    img = np.concatenate([_without_circle(objects, center, d, side, width) for d in views], 1)
    write_png(output, img)


def _without_circle(objects, center, d, side, width):
    return thumbnail(objects, center, d, side, width, circle=False)


def sheet(ids, hairdo, head, output, defaults=None, max_=12):
    """Writes the contact sheet of an outfit's spots (worst first); returns the legend, one line per cell."""
    parts = {i: CB.part_name(i) for i in ids}
    ret = CB.removed({i: p.meta for i, p in parts.items()})
    layers, _ = CB.resolve([p for i, p in parts.items() if i not in ret])
    if defaults is None:
        defaults = RV.review_outfit(ids, hairdo, head)["defaults"]
    defaults = [d for d in defaults if d.get("spot")][:max_]
    if not defaults:
        return []
    objects = scene(layers, hairdo, head)
    cases = [thumbnail(objects, d["spot"]) for d in defaults]
    collar = min(4, len(cases))
    lig = -(-len(cases) // collar)
    img = np.full((lig * SIDE, collar * SIDE, 3), 255, np.uint8)
    for k, v in enumerate(cases):
        img[(k // collar) * SIDE:(k // collar + 1) * SIDE, (k % collar) * SIDE:(k % collar + 1) * SIDE] = v
    write_png(output, img)
    names = {c.id: TINTS[i % len(TINTS)] for i, c in enumerate(layers)}
    return ["cell %d (row %d, column %d): %s %s %s" % (k + 1, k // collar + 1, k % collar + 1, d["kind"], "+".join(d.get("parts", [])), d["spot"])
            for k, d in enumerate(defaults)] + ["colours: " + ", ".join("%s rgb%s" % (i, t) for i, t in names.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", dest="parts")
    ap.add_argument("--part", dest="part_name")
    ap.add_argument("--turn", dest="girth", action="store_true", help="overall view (six directions) instead of the flagged spots")
    ap.add_argument("--hairstyle", dest="hairstyle")
    ap.add_argument("--head", dest="head")
    ap.add_argument("--json")
    ap.add_argument("--output", dest="output", required=True)
    ap.add_argument("--max", type=int, default=12)
    a = ap.parse_args()
    if a.girth:
        girth((a.parts or a.part_name).split(","), a.hairstyle, a.head, a.output)
        return
    if a.part_name:
        r = RV.review_part(a.part_name)
        d = sorted(r["defaults"], key=lambda x: -x["cm"] * RV.GRAVE.get(x["kind"], 1))
        print("\n".join(sheet([a.part_name], a.hairstyle, a.head, a.output, d, a.max)))
        return
    if a.parts:
        print("\n".join(sheet(a.parts.split(","), a.hairstyle, a.head, a.output, max_=a.max)))
        return
    d = json.load(open(a.json, encoding="utf-8"))
    os.makedirs(a.output, exist_ok=True)
    for k, t in enumerate(d["outfits"] if isinstance(d, dict) else d):
        if t["defaults"]:
            f = os.path.join(a.output, "%02d.png" % k)
            print(f, " + ".join(t["parts"]))
            print("\n".join("  " + x for x in sheet(t["parts"], d.get("hairstyle") if isinstance(d, dict) else None,
                                                     d.get("head") if isinstance(d, dict) else None, f, t["defaults"], a.max)))


if __name__ == "__main__":
    main()

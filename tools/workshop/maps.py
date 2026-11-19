"""Render maps of the parts, computed from the geometry: what games paint by hand and the parts lack.

    blender -b --factory-startup --python tools/workshop/maps.py -- Maelys Orpheya   # head, hairstyle, parts, body

(Blender for its ray tree, `mathutils`.) Next to each GLB, `maps.bin`: four bytes per
vertex, read by the view as one more attribute.

  0  openness    1 = exposed, 0 = in a hollow (armpit, under the chin, fold). Rays cast around the normal.
                 Stands in for Outerplane's `_Mask.g` (shadow threshold) and NTE's `LightMap.y` (shadow offset).
  1  shininess   1 on metal or stone materials (material name), 0 elsewhere.
                 Stands in for Outerplane's `_Mask.r` and `_Mask.a`, NTE's `LightMap.z` and `ID_Tex.y`.
  2  face, light coming from its left (+x)    } up to which light angle the point stays lit, like Outerplane's
  3  face, light coming from its right (−x)   } SDF map: 0 = from the front, 0.5 = from the side, 1 = from behind.
                 Computed from smoothed normals and shadow rays (the nose casts its shadow). 1 outside the face.

The face also has a drawn map, per character (`heads/<id>/face-shade.png` and its frame `.json`): the same
measurement, but per image point of the face seen from the front, hence as fine as a game's drawing where vertices are
sparse (cheek, side of the nose). Red: light from +x; green: from −x; blue: where there is face. On its own: `-- visage:Jane`.
It holds for the face at rest.

The body has one map per head (`heads/<id>/body-maps.bin`): its skin changes under each face and the head casts
a shadow on it under the chin.
"""
import glob
import json
import math
import os
import struct
import sys
import zlib

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import glb as GLB  # noqa: E402

SHINY = ("金属", "吊坠", "宝石", "Metal", "metal")
REACH = 0.10        # m: beyond this, a surface no longer counts for openness
RAYS = 32
ANGLES = 24          # steps of the light around the face, from front to back
WIDTH = 512        # image points around the face, from one ear to the other, for its drawn map
ANGLES_IMAGE = 32


def read(path):
    """Chimera GLB -> positions, normals, [(material name, faces)]."""
    doc, b = GLB.read(path)

    def accessory(i, dt, k):
        a = doc["accessors"][i]
        v = doc["bufferViews"][a["bufferView"]]
        n = a["count"] * (1 if a["type"] == "SCALAR" else k)
        return np.frombuffer(b, dt, n, v.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(-1, k)

    prims = doc["meshes"][0]["primitives"]
    assert len({p["attributes"]["POSITION"] for p in prims}) == 1, path
    pos = accessory(prims[0]["attributes"]["POSITION"], np.float32, 3)
    nor = accessory(prims[0]["attributes"]["NORMAL"], np.float32, 3)
    groups = []
    for p in prims:
        dt = {5125: np.uint32, 5123: np.uint16}[doc["accessors"][p["indices"]]["componentType"]]
        groups.append((doc["materials"][p["material"]]["name"], accessory(p["indices"], dt, 3).astype(np.int64)))
    return pos.astype(np.float64), nor.astype(np.float64), groups


def tree(chunks):
    """A single ray tree for several meshes [(positions, faces)]."""
    P, F, n = [], [], 0
    for pos, faces in chunks:
        P.append(pos)
        F.append(faces + n)
        n += len(pos)
    P, F = np.concatenate(P), np.concatenate(F)
    return BVHTree.FromPolygons([tuple(p) for p in P], [tuple(int(i) for i in f) for f in F])


def directions():
    """Rays spread over the hemisphere, denser near the normal (cosine weighting)."""
    d = []
    for i in range(RAYS):
        z = math.sqrt(1 - (i + 0.5) / RAYS)
        r, a = math.sqrt(1 - z * z), i * 2.39996323
        d.append((r * math.cos(a), r * math.sin(a), z))
    return np.array(d)


def opening(pos, nor, bvh):
    D = directions()
    n = nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-9)
    help_ = np.where(np.abs(n[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])
    t = np.cross(help_, n)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    b = np.cross(n, t)
    o = np.ones(len(pos))
    for i in range(len(pos)):
        disp = Vector(pos[i] + n[i] * 0.003)
        closed = 0.0
        for dx, dy, dz in D:
            h = bvh.ray_cast(disp, Vector(t[i] * dx + b[i] * dy + n[i] * dz), REACH)
            if h[0] is not None:
                closed += 1 - h[3] / REACH
        o[i] = 1 - closed / RAYS
    return o


def neighbors(n, faces):
    a = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.concatenate([a, a[:, ::-1]])


def weld(pos, nor):
    """Welded vertex of each vertex: same place (to 1/100 mm) and close normal. UV and material seams
    duplicate vertices; without welding, each duplicate sees another neighbourhood and openness jumps at the seam
    (seen at the fold of the buttocks, the wrist, the neck). The two sides of a strand (opposite normals) stay apart."""
    the_key = np.round(np.asarray(pos) * 1e5).astype(np.int64)
    _, first, inv = np.unique(the_key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    rep = first[inv]
    n = nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-9)
    rep = np.where((n * n[rep]).sum(1) > 0.5, rep, np.arange(len(pos)))
    _, ids = np.unique(rep, return_inverse=True)
    return ids.ravel(), int(ids.max()) + 1


def smooth_welded(v, faces, ids, m, perimeters):
    """smooth on the welded mesh, mean value of the duplicates, given back to each vertex."""
    s = np.zeros(m)
    c = np.zeros(m)
    np.add.at(s, ids, v)
    np.add.at(c, ids, 1)
    return smooth(s / np.maximum(c, 1), neighbors(m, ids[faces]), m, perimeters)[ids]


def smooth(v, edges, n, perimeters):
    """Average with the neighbours: the value of a vertex must not depend on how the mesh is cut."""
    v = np.asarray(v, np.float64)
    for _ in range(perimeters):
        s = np.zeros_like(v)
        c = np.zeros(n)
        np.add.at(s, edges[:, 0], v[edges[:, 1]])
        np.add.at(c, edges[:, 0], 1)
        c = np.maximum(c, 1)
        v = 0.5 * v + 0.5 * s / (c if v.ndim == 1 else c[:, None])
    return v


def face(pos, nor, faces, bvh):
    """Per vertex and per side: the largest light threshold (0 from the front, 1 from behind) where it stays lit."""
    n = len(pos)
    ar = neighbors(n, faces)
    ns = smooth(nor, ar, n, 6)
    ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-9)
    actives = np.unique(faces)
    v = np.ones((n, 2))
    for k, sign in enumerate((1.0, -1.0)):
        for i in actives:
            disp = Vector(pos[i] + ns[i] * 0.002)
            threshold = 0.0
            for j in range(ANGLES + 1):
                a = math.pi * j / ANGLES
                L = (sign * math.sin(a), 0.0, math.cos(a))
                if ns[i, 0] * L[0] + ns[i, 2] * L[2] <= 0.05 or bvh.ray_cast(disp, Vector(L), 0.3)[0] is not None:
                    break
                threshold = 0.5 - 0.5 * math.cos(a)
            v[i, k] = threshold
    v[actives] = smooth(v, ar, n, 8)[actives]
    return v


def blur(a, m, sigma):
    """Gaussian blur of an image with holes: each point weighs only if it is covered (m)."""
    r = int(sigma * 3)
    g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)

    def passed(x):
        for axis in (0, 1):
            x = sum(k * np.roll(x, d, axis=axis) for k, d in zip(g, range(-r, r + 1)))
        return x

    weight = passed(m.astype(np.float64))
    return passed(a * m) / np.maximum(weight, 1e-9), weight


def shade_face(ident):
    """Shadow map of the face as an image: per point, up to which light angle it stays lit.

    The image is wrapped around the head (angle about a vertical axis, height): seen from the front only, it
    was worthless on the side of the cheeks, which the per-vertex map rendered as teeth along the large triangles."""
    d = os.path.join(C.CACHE, "heads", ident)
    pos, nor, groups = read(os.path.join(d, "head.glb"))
    fv = np.concatenate([f for name, f in groups if name.startswith("颜")])
    skin = np.concatenate([f for name, f in groups if name.startswith(("颜", "肌"))])   # eyelashes and eyebrows cast no shadow
    pc, _, gc = read(os.path.join(C.CACHE, "body", "body.glb"))
    vert = os.path.join(d, "body.bin")
    if os.path.exists(vert):
        a = np.fromfile(vert, np.float32).astype(np.float64)
        if len(a) == 6 * len(pc):
            pc = a[:3 * len(pc)].reshape(-1, 3)
    carriers = tree([(pos, skin), (pc, np.concatenate([f for _, f in gc]))])
    face = BVHTree.FromPolygons([tuple(p) for p in pos], [tuple(int(i) for i in f) for f in fv])
    # heavily smoothed normals: the boundary follows the overall shape of the face, the nose enters only by its cast shadow
    ns = smooth(nor, neighbors(len(pos), fv), len(pos), 12)
    ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-9)
    s = pos[np.unique(fv)]
    margin = 0.025                                    # room for the blend into the head skin, at the edge
    axis = s[:, 2].min() - 0.02                       # the axis passes behind the face: no point beyond a quarter turn
    radius = np.hypot(s[:, 0], s[:, 2] - axis)
    x1 = np.abs(np.arctan2(s[:, 0], s[:, 2] - axis)).max() + margin / np.median(radius)
    x0, y0, y1 = -x1, s[:, 1].min() - margin, s[:, 1].max() + margin
    stage = (x1 - x0) / WIDTH                        # in angle; in height, the same step at the median radius
    step_y = stage * np.median(radius)
    H = int(math.ceil((y1 - y0) / step_y))
    y0 = y1 - H * step_y
    v = np.zeros((H, WIDTH, 2))
    m = np.zeros((H, WIDTH), bool)
    far = radius.max() + 0.05
    for j in range(H):
        for i in range(WIDTH):
            t = x0 + (i + 0.5) * stage
            outside = np.array([math.sin(t), 0.0, math.cos(t)])
            h = face.ray_cast(Vector((far * outside[0], y1 - (j + 0.5) * step_y, axis + far * outside[2])), Vector(-outside))
            if h[0] is None:
                continue
            a, b, c = fv[h[2]]
            # smoothed normal at the hit point, from its weights in the triangle
            e1, e2, q = pos[b] - pos[a], pos[c] - pos[a], np.array(h[0]) - pos[a]
            d11, d12, d22, q1, q2 = e1 @ e1, e1 @ e2, e2 @ e2, q @ e1, q @ e2
            det = max(d11 * d22 - d12 * d12, 1e-18)
            wb, wc = (d22 * q1 - d12 * q2) / det, (d11 * q2 - d12 * q1) / det
            n = ns[a] * (1 - wb - wc) + ns[b] * wb + ns[c] * wc
            n /= max(np.linalg.norm(n), 1e-9)
            if n @ outside <= 0:
                continue
            m[j, i] = True
            disp = h[0] + Vector(n) * 0.003

            def lit(ang, sign):
                lx, lz = sign * math.sin(ang), math.cos(ang)
                if n[0] * lx + n[2] * lz <= 0.05:
                    return False
                # the cast shadow falls slightly downwards, as under a high light; the ray starts at 1 cm:
                # nearby relief (cheek fold, facets) does not count, the nose does
                L = Vector((lx, 0.3, lz)).normalized()
                return carriers.ray_cast(disp + L * 0.010, L, 0.3)[0] is None

            for k, sign in enumerate((1.0, -1.0)):
                # a point turned sideways (edge of the cheek, temple) is not lit from the front: its lit range
                # starts further on. That start in shadow is skipped, otherwise the point is "never lit" and the
                # side of the cheek stays a dark patch with a sharp edge, whatever the light.
                bottom, top, vu = 0.0, None, False
                for t in range(ANGLES_IMAGE + 1):
                    ang = math.pi * t / ANGLES_IMAGE
                    if not lit(ang, sign):
                        if not vu and n[2] < 0.3 and sign * n[0] > 0:
                            continue
                        top = ang
                        break
                    bottom, vu = ang, True
                # between the last lit angle and the first in shadow: as close as possible, without steps
                for _ in range(5 if top else 0):
                    mi = 0.5 * (bottom + top)
                    bottom, top = (mi, top) if lit(mi, sign) else (bottom, mi)
                v[j, i, k] = 0.5 - 0.5 * math.cos(bottom)
    img = np.zeros((H, WIDTH, 3))
    for k in (0, 1):
        img[..., k], weight = blur(v[..., k], m, 5.0)
    # blue: 1 at the core of the face, 0 at its outer edge, over about 1 cm: the view blends the map there into the
    # lighting of the head skin, otherwise the edge of the cheek reads as a line. The eyes and mouth are not an edge.
    full = m.copy()
    for j in np.nonzero(m.any(axis=1))[0]:
        c = np.nonzero(m[j])[0]
        full[j, c[0]:c[-1] + 1] = True
    img[..., 2] = np.clip(blur(full * 1.0, np.ones_like(m), 10.0)[0] * 2 - 1, 0, 1) * m
    # outside the face: the value of the edge, so that filtered texture reads bring back no shadow
    img[..., :2][weight < 1e-6] = 0
    png(os.path.join(d, "face-shade.png"), (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    C.write_json(os.path.join(d, "face-shade.json"), {"frame": [x0, y0, x1, y1], "axis": axis})
    print("%s/face-shade.png: %d x %d, %d face points" % (ident, WIDTH, H, int(m.sum())))


def write(path, open_, brightness, vis):
    c = np.stack([open_, brightness, vis[:, 0], vis[:, 1]], axis=1)
    (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8).tofile(path)
    print("%s: %d vertices, mean openness %.2f, shiny %d" % (os.path.relpath(path, C.CACHE), len(c), open_.mean(), int(brightness.sum())))


def make(path, output=None, others=(), pos=None, nor=None, without_faces=None):
    """Maps of a GLB. others: meshes that shadow it; pos, nor: moved vertices; without_faces: removed faces."""
    p0, n0, groups = read(path)
    pos = p0 if pos is None else pos
    nor = n0 if nor is None else nor
    faces = np.concatenate([f for _, f in groups])
    if without_faces is not None:
        faces = np.delete(faces, without_faces, axis=0)
    bvh = tree([(pos, faces)] + list(others))
    open_ = opening(pos, nor, bvh)
    ids, m = weld(pos, nor)
    open_ = smooth_welded(open_, faces, ids, m, 1)
    open_ = open_ / max(np.percentile(open_, 90), 1e-3)  # strands and chains cover themselves: the most open point of the part is worth 1
    bri = np.zeros(len(pos))
    vis = np.ones((len(pos), 2))
    for name, f in groups:
        if any(x in name for x in SHINY):
            bri[np.unique(f)] = 1
    f_face = [f for name, f in groups if name.startswith("颜")]
    if f_face and os.path.basename(path) == "head.glb":
        f_face = np.concatenate(f_face)
        v = face(pos, nor, f_face, bvh)
        vis[np.unique(f_face)] = v[np.unique(f_face)]
    write(output or os.path.join(os.path.dirname(path), "maps.bin"), open_, bri, vis)


def person(ident):
    d = os.path.join(C.CACHE, "heads", ident)
    body = os.path.join(C.CACHE, "body", "body.glb")
    pc, nc, gc = read(body)
    fc = np.concatenate([f for _, f in gc])
    vert = os.path.join(d, "body.bin")
    if os.path.exists(vert):
        a = np.fromfile(vert, np.float32).astype(np.float64)
        if len(a) == 6 * len(pc):
            pc, nc = a[:3 * len(pc)].reshape(-1, 3), a[3 * len(pc):].reshape(-1, 3)
    cache = os.path.join(d, "hidden_skin.bin")
    removes = np.fromfile(cache, np.uint32) if os.path.exists(cache) else np.zeros(0, np.uint32)
    pt, _, gt = read(os.path.join(d, "head.glb"))
    ft = np.concatenate([f for _, f in gt])
    make(os.path.join(d, "head.glb"), others=[(pc, np.delete(fc, removes, axis=0))])
    shade_face(ident)
    make(body, output=os.path.join(d, "body-maps.bin"), others=[(pt, ft)], pos=pc, nor=nc, without_faces=removes)
    hairstyle = os.path.join(C.CACHE, "hairstyles", ident, "hairstyle.glb")
    if os.path.exists(hairstyle):
        make(hairstyle)
    for f in sorted(glob.glob(os.path.join(C.CACHE, "parts", "*", "meta.json"))):
        m = C.read_json(f)
        if m.get("character") == ident and m.get("kept"):
            make(os.path.join(os.path.dirname(f), "part.glb"))


def png(path, a):
    h, w, c = a.shape
    raw = b"".join(b"\0" + a[y].tobytes() for y in range(h))

    def block(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + block(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2 if c == 3 else 6, 0, 0, 0))
                + block(b"IDAT", zlib.compress(raw, 9)) + block(b"IEND", b""))


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not args:
        sys.exit(__doc__)
    for x in args:
        shade_face(x[7:]) if x.startswith("visage:") else person(x)

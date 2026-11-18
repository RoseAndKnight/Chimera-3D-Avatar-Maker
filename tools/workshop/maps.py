"""Render maps of the parts, computed from the geometry: what games paint by hand and the parts lack.

    blender -b --factory-startup --python tools/workshop/maps.py -- Maelys Orpheya   # head, hairstyle, parts, body

(Blender for its ray tree, `mathutils`.) Next to each GLB, `cartes.bin`: four bytes per
vertex, read by the view as one more attribute.

  0  openness    1 = exposed, 0 = in a hollow (armpit, under the chin, fold). Rays cast around the normal.
                 Stands in for Outerplane's `_Mask.g` (shadow threshold) and NTE's `LightMap.y` (shadow offset).
  1  shininess   1 on metal or stone materials (material name), 0 elsewhere.
                 Stands in for Outerplane's `_Mask.r` and `_Mask.a`, NTE's `LightMap.z` and `ID_Tex.y`.
  2  face, light coming from its left (+x)    } up to which light angle the point stays lit, like Outerplane's
  3  face, light coming from its right (−x)   } SDF map: 0 = from the front, 0.5 = from the side, 1 = from behind.
                 Computed from smoothed normals and shadow rays (the nose casts its shadow). 1 outside the face.

The face also has a drawn map, per character (`tetes/<id>/ombre-visage.png` and its frame `.json`): the same
measurement, but per image point of the face seen from the front, hence as fine as a game's drawing where vertices are
sparse (cheek, side of the nose). Red: light from +x; green: from −x; blue: where there is face. On its own: `-- visage:Jane`.
It holds for the face at rest.

The body has one map per head (`tetes/<id>/cartes-corps.bin`): its skin changes under each face and the head casts
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

BRILLANTS = ("金属", "吊坠", "宝石", "Metal", "metal")
PORTEE = 0.10        # m: beyond this, a surface no longer counts for openness
RAYONS = 32
ANGLES = 24          # steps of the light around the face, from front to back
LARGEUR = 512        # image points around the face, from one ear to the other, for its drawn map
ANGLES_IMAGE = 32


def lire(chemin):
    """Chimera GLB -> positions, normals, [(material name, faces)]."""
    doc, b = GLB.lire(chemin)

    def acc(i, dt, k):
        a = doc["accessors"][i]
        v = doc["bufferViews"][a["bufferView"]]
        n = a["count"] * (1 if a["type"] == "SCALAR" else k)
        return np.frombuffer(b, dt, n, v.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(-1, k)

    prims = doc["meshes"][0]["primitives"]
    assert len({p["attributes"]["POSITION"] for p in prims}) == 1, chemin
    pos = acc(prims[0]["attributes"]["POSITION"], np.float32, 3)
    nor = acc(prims[0]["attributes"]["NORMAL"], np.float32, 3)
    groupes = []
    for p in prims:
        dt = {5125: np.uint32, 5123: np.uint16}[doc["accessors"][p["indices"]]["componentType"]]
        groupes.append((doc["materials"][p["material"]]["name"], acc(p["indices"], dt, 3).astype(np.int64)))
    return pos.astype(np.float64), nor.astype(np.float64), groupes


def arbre(morceaux):
    """A single ray tree for several meshes [(positions, faces)]."""
    P, F, n = [], [], 0
    for pos, faces in morceaux:
        P.append(pos)
        F.append(faces + n)
        n += len(pos)
    P, F = np.concatenate(P), np.concatenate(F)
    return BVHTree.FromPolygons([tuple(p) for p in P], [tuple(int(i) for i in f) for f in F])


def directions():
    """Rays spread over the hemisphere, denser near the normal (cosine weighting)."""
    d = []
    for i in range(RAYONS):
        z = math.sqrt(1 - (i + 0.5) / RAYONS)
        r, a = math.sqrt(1 - z * z), i * 2.39996323
        d.append((r * math.cos(a), r * math.sin(a), z))
    return np.array(d)


def ouverture(pos, nor, bvh):
    D = directions()
    n = nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-9)
    aide = np.where(np.abs(n[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])
    t = np.cross(aide, n)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    b = np.cross(n, t)
    o = np.ones(len(pos))
    for i in range(len(pos)):
        dep = Vector(pos[i] + n[i] * 0.003)
        ferme = 0.0
        for dx, dy, dz in D:
            h = bvh.ray_cast(dep, Vector(t[i] * dx + b[i] * dy + n[i] * dz), PORTEE)
            if h[0] is not None:
                ferme += 1 - h[3] / PORTEE
        o[i] = 1 - ferme / RAYONS
    return o


def voisins(n, faces):
    a = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.concatenate([a, a[:, ::-1]])


def souder(pos, nor):
    """Welded vertex of each vertex: same place (to 1/100 mm) and close normal. UV and material seams
    duplicate vertices; without welding, each duplicate sees another neighbourhood and openness jumps at the seam
    (seen at the fold of the buttocks, the wrist, the neck). The two sides of a strand (opposite normals) stay apart."""
    cle = np.round(np.asarray(pos) * 1e5).astype(np.int64)
    _, premier, inv = np.unique(cle, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    rep = premier[inv]
    n = nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-9)
    rep = np.where((n * n[rep]).sum(1) > 0.5, rep, np.arange(len(pos)))
    _, ids = np.unique(rep, return_inverse=True)
    return ids.ravel(), int(ids.max()) + 1


def lisser_soude(v, faces, ids, m, tours):
    """lisser on the welded mesh, mean value of the duplicates, given back to each vertex."""
    s = np.zeros(m)
    c = np.zeros(m)
    np.add.at(s, ids, v)
    np.add.at(c, ids, 1)
    return lisser(s / np.maximum(c, 1), voisins(m, ids[faces]), m, tours)[ids]


def lisser(v, aretes, n, tours):
    """Average with the neighbours: the value of a vertex must not depend on how the mesh is cut."""
    v = np.asarray(v, np.float64)
    for _ in range(tours):
        s = np.zeros_like(v)
        c = np.zeros(n)
        np.add.at(s, aretes[:, 0], v[aretes[:, 1]])
        np.add.at(c, aretes[:, 0], 1)
        c = np.maximum(c, 1)
        v = 0.5 * v + 0.5 * s / (c if v.ndim == 1 else c[:, None])
    return v


def visage(pos, nor, faces, bvh):
    """Per vertex and per side: the largest light threshold (0 from the front, 1 from behind) where it stays lit."""
    n = len(pos)
    ar = voisins(n, faces)
    ns = lisser(nor, ar, n, 6)
    ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-9)
    actifs = np.unique(faces)
    v = np.ones((n, 2))
    for k, signe in enumerate((1.0, -1.0)):
        for i in actifs:
            dep = Vector(pos[i] + ns[i] * 0.002)
            seuil = 0.0
            for j in range(ANGLES + 1):
                a = math.pi * j / ANGLES
                L = (signe * math.sin(a), 0.0, math.cos(a))
                if ns[i, 0] * L[0] + ns[i, 2] * L[2] <= 0.05 or bvh.ray_cast(dep, Vector(L), 0.3)[0] is not None:
                    break
                seuil = 0.5 - 0.5 * math.cos(a)
            v[i, k] = seuil
    v[actifs] = lisser(v, ar, n, 8)[actifs]
    return v


def flou(a, m, sigma):
    """Gaussian blur of an image with holes: each point weighs only if it is covered (m)."""
    r = int(sigma * 3)
    g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)

    def passe(x):
        for axe in (0, 1):
            x = sum(k * np.roll(x, d, axis=axe) for k, d in zip(g, range(-r, r + 1)))
        return x

    poids = passe(m.astype(np.float64))
    return passe(a * m) / np.maximum(poids, 1e-9), poids


def ombre_visage(ident):
    """Shadow map of the face as an image: per point, up to which light angle it stays lit.

    The image is wrapped around the head (angle about a vertical axis, height): seen from the front only, it
    was worthless on the side of the cheeks, which the per-vertex map rendered as teeth along the large triangles."""
    d = os.path.join(C.CACHE, "tetes", ident)
    pos, nor, groupes = lire(os.path.join(d, "tete.glb"))
    fv = np.concatenate([f for nom, f in groupes if nom.startswith("颜")])
    peau = np.concatenate([f for nom, f in groupes if nom.startswith(("颜", "肌"))])   # eyelashes and eyebrows cast no shadow
    pc, _, gc = lire(os.path.join(C.CACHE, "corps", "corps.glb"))
    som = os.path.join(d, "corps.bin")
    if os.path.exists(som):
        a = np.fromfile(som, np.float32).astype(np.float64)
        if len(a) == 6 * len(pc):
            pc = a[:3 * len(pc)].reshape(-1, 3)
    porteurs = arbre([(pos, peau), (pc, np.concatenate([f for _, f in gc]))])
    face = BVHTree.FromPolygons([tuple(p) for p in pos], [tuple(int(i) for i in f) for f in fv])
    # heavily smoothed normals: the boundary follows the overall shape of the face, the nose enters only by its cast shadow
    ns = lisser(nor, voisins(len(pos), fv), len(pos), 12)
    ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-9)
    s = pos[np.unique(fv)]
    marge = 0.025                                    # room for the blend into the head skin, at the edge
    axe = s[:, 2].min() - 0.02                       # the axis passes behind the face: no point beyond a quarter turn
    rayon = np.hypot(s[:, 0], s[:, 2] - axe)
    x1 = np.abs(np.arctan2(s[:, 0], s[:, 2] - axe)).max() + marge / np.median(rayon)
    x0, y0, y1 = -x1, s[:, 1].min() - marge, s[:, 1].max() + marge
    pas = (x1 - x0) / LARGEUR                        # in angle; in height, the same step at the median radius
    pas_y = pas * np.median(rayon)
    H = int(math.ceil((y1 - y0) / pas_y))
    y0 = y1 - H * pas_y
    v = np.zeros((H, LARGEUR, 2))
    m = np.zeros((H, LARGEUR), bool)
    loin = rayon.max() + 0.05
    for j in range(H):
        for i in range(LARGEUR):
            t = x0 + (i + 0.5) * pas
            dehors = np.array([math.sin(t), 0.0, math.cos(t)])
            h = face.ray_cast(Vector((loin * dehors[0], y1 - (j + 0.5) * pas_y, axe + loin * dehors[2])), Vector(-dehors))
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
            if n @ dehors <= 0:
                continue
            m[j, i] = True
            dep = h[0] + Vector(n) * 0.003

            def eclaire(ang, signe):
                lx, lz = signe * math.sin(ang), math.cos(ang)
                if n[0] * lx + n[2] * lz <= 0.05:
                    return False
                # the cast shadow falls slightly downwards, as under a high light; the ray starts at 1 cm:
                # nearby relief (cheek fold, facets) does not count, the nose does
                L = Vector((lx, 0.3, lz)).normalized()
                return porteurs.ray_cast(dep + L * 0.010, L, 0.3)[0] is None

            for k, signe in enumerate((1.0, -1.0)):
                # a point turned sideways (edge of the cheek, temple) is not lit from the front: its lit range
                # starts further on. That start in shadow is skipped, otherwise the point is "never lit" and the
                # side of the cheek stays a dark patch with a sharp edge, whatever the light.
                bas, haut, vu = 0.0, None, False
                for t in range(ANGLES_IMAGE + 1):
                    ang = math.pi * t / ANGLES_IMAGE
                    if not eclaire(ang, signe):
                        if not vu and n[2] < 0.3 and signe * n[0] > 0:
                            continue
                        haut = ang
                        break
                    bas, vu = ang, True
                # between the last lit angle and the first in shadow: as close as possible, without steps
                for _ in range(5 if haut else 0):
                    mi = 0.5 * (bas + haut)
                    bas, haut = (mi, haut) if eclaire(mi, signe) else (bas, mi)
                v[j, i, k] = 0.5 - 0.5 * math.cos(bas)
    img = np.zeros((H, LARGEUR, 3))
    for k in (0, 1):
        img[..., k], poids = flou(v[..., k], m, 5.0)
    # blue: 1 at the core of the face, 0 at its outer edge, over about 1 cm: the view blends the map there into the
    # lighting of the head skin, otherwise the edge of the cheek reads as a line. The eyes and mouth are not an edge.
    plein = m.copy()
    for j in np.nonzero(m.any(axis=1))[0]:
        c = np.nonzero(m[j])[0]
        plein[j, c[0]:c[-1] + 1] = True
    img[..., 2] = np.clip(flou(plein * 1.0, np.ones_like(m), 10.0)[0] * 2 - 1, 0, 1) * m
    # outside the face: the value of the edge, so that filtered texture reads bring back no shadow
    img[..., :2][poids < 1e-6] = 0
    png(os.path.join(d, "ombre-visage.png"), (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    json.dump({"cadre": [x0, y0, x1, y1], "axe": axe}, open(os.path.join(d, "ombre-visage.json"), "w"))
    print("%s/ombre-visage.png: %d x %d, %d face points" % (ident, LARGEUR, H, int(m.sum())))


def ecrire(chemin, ouv, brillance, vis):
    c = np.stack([ouv, brillance, vis[:, 0], vis[:, 1]], axis=1)
    (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8).tofile(chemin)
    print("%s: %d vertices, mean openness %.2f, shiny %d" % (os.path.relpath(chemin, C.CACHE), len(c), ouv.mean(), int(brillance.sum())))


def faire(chemin, sortie=None, autres=(), pos=None, nor=None, sans_faces=None):
    """Maps of a GLB. autres: meshes that shadow it; pos, nor: moved vertices; sans_faces: removed faces."""
    p0, n0, groupes = lire(chemin)
    pos = p0 if pos is None else pos
    nor = n0 if nor is None else nor
    faces = np.concatenate([f for _, f in groupes])
    if sans_faces is not None:
        faces = np.delete(faces, sans_faces, axis=0)
    bvh = arbre([(pos, faces)] + list(autres))
    ouv = ouverture(pos, nor, bvh)
    ids, m = souder(pos, nor)
    ouv = lisser_soude(ouv, faces, ids, m, 1)
    ouv = ouv / max(np.percentile(ouv, 90), 1e-3)  # strands and chains cover themselves: the most open point of the part is worth 1
    bri = np.zeros(len(pos))
    vis = np.ones((len(pos), 2))
    for nom, f in groupes:
        if any(x in nom for x in BRILLANTS):
            bri[np.unique(f)] = 1
    f_visage = [f for nom, f in groupes if nom.startswith("颜")]
    if f_visage and os.path.basename(chemin) == "tete.glb":
        f_visage = np.concatenate(f_visage)
        v = visage(pos, nor, f_visage, bvh)
        vis[np.unique(f_visage)] = v[np.unique(f_visage)]
    ecrire(sortie or os.path.join(os.path.dirname(chemin), "cartes.bin"), ouv, bri, vis)


def personnage(ident):
    d = os.path.join(C.CACHE, "tetes", ident)
    corps = os.path.join(C.CACHE, "corps", "corps.glb")
    pc, nc, gc = lire(corps)
    fc = np.concatenate([f for _, f in gc])
    som = os.path.join(d, "corps.bin")
    if os.path.exists(som):
        a = np.fromfile(som, np.float32).astype(np.float64)
        if len(a) == 6 * len(pc):
            pc, nc = a[:3 * len(pc)].reshape(-1, 3), a[3 * len(pc):].reshape(-1, 3)
    cache = os.path.join(d, "peau_cachee.bin")
    retire = np.fromfile(cache, np.uint32) if os.path.exists(cache) else np.zeros(0, np.uint32)
    pt, _, gt = lire(os.path.join(d, "tete.glb"))
    ft = np.concatenate([f for _, f in gt])
    faire(os.path.join(d, "tete.glb"), autres=[(pc, np.delete(fc, retire, axis=0))])
    ombre_visage(ident)
    faire(corps, sortie=os.path.join(d, "cartes-corps.bin"), autres=[(pt, ft)], pos=pc, nor=nc, sans_faces=retire)
    coiffure = os.path.join(C.CACHE, "coiffures", ident, "coiffure.glb")
    if os.path.exists(coiffure):
        faire(coiffure)
    for f in sorted(glob.glob(os.path.join(C.CACHE, "pieces", "*", "meta.json"))):
        m = json.load(open(f))
        if m.get("perso") == ident and m.get("retenue"):
            faire(os.path.join(os.path.dirname(f), "piece.glb"))


def png(chemin, a):
    h, w, c = a.shape
    brut = b"".join(b"\0" + a[y].tobytes() for y in range(h))

    def bloc(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    with open(chemin, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + bloc(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2 if c == 3 else 6, 0, 0, 0))
                + bloc(b"IDAT", zlib.compress(brut, 9)) + bloc(b"IEND", b""))


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not args:
        sys.exit(__doc__)
    for x in args:
        ombre_visage(x[7:]) if x.startswith("visage:") else personnage(x)

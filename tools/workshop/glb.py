"""Writing and reading back a minimal GLB (binary glTF 2.0): one mesh, several primitives,
external textures (files shared in the cache, relative URIs). Numpy only.

Frame: MMD (x, y, z; one unit = 8 cm, the character looks towards -Z) -> glTF (x, y, -z in
metres, character facing +Z). Flipping the sign of z reverses the winding of the faces: `to_gltf`
checks it against the mesh normals rather than assuming it.
"""
import json
import os
import struct

import numpy as np

COMPONENT = {"f4": 5126, "u4": 5125, "u2": 5123}
TYPE = {2: "VEC2", 3: "VEC3", 4: "VEC4"}


def to_gltf(pos, nor, faces):
    """Moves vertices and normals from the MMD frame to the glTF frame and orients the faces outwards."""
    p = np.asarray(pos, dtype=np.float32).copy() * 0.08
    n = np.asarray(nor, dtype=np.float32).copy()
    p[:, 2] *= -1
    n[:, 2] *= -1
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f):
        a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
        nf = np.cross(b - a, c - a)
        if (nf * n[f[:, 0]]).sum() < 0:
            f = f[:, [0, 2, 1]]
            nf = -nf
        f = a_l_upright(p, n, f)
    return p, n, f


def a_l_upright(pos, nor, F):
    """Faces drawn inside out, flipped. The page lights the back of a face with the opposite
    normal: a face drawn inside out comes out unlit, as a dark patch (under the buttocks of
    Zhu Yuan's jumpsuit, 16/11), and its outline shell passes in front of the surface (line under
    Maelys's lip, 09/11).

    The winding is first read from the neighbours: two faces sharing an edge traverse it in
    opposite directions. Each sheet is thus made consistent with itself, then turned to the side
    pointed to by the majority of its normals (the normals of an inside-out face are often wrong
    too: they do not decide face by face)."""
    F = np.array(F, np.int64).reshape(-1, 3)
    if not len(F):
        return F
    pos = np.asarray(pos, float)
    w = np.unique(np.round(pos / 1e-5).astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)
    W = w[F]
    # oriented edges of each face; two consistent neighbouring faces traverse them in opposite directions
    a = np.concatenate([W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]])
    which = np.tile(np.arange(len(F)), 3)
    direction = a[:, 0] < a[:, 1]
    the_key = np.sort(a, 1)
    order = np.lexsort((the_key[:, 1], the_key[:, 0]))
    the_key, which, direction = the_key[order], which[order], direction[order]
    deb = np.r_[True, (the_key[1:] != the_key[:-1]).any(1)]
    nb = np.diff(np.r_[np.where(deb)[0], len(the_key)])
    pair = np.where(deb)[0][nb == 2]                  # manifold edges: exactly two faces
    f1, f2 = which[pair], which[pair + 1]
    against = direction[pair] == direction[pair + 1]            # same direction: one of the two is inside out
    ok = (f1 != f2) & (the_key[pair, 0] != the_key[pair, 1])
    f1, f2, against = f1[ok], f2[ok], against[ok]
    neigh = [[] for _ in range(len(F))]
    for x, y, c in zip(f1.tolist(), f2.tolist(), against.tolist()):
        neigh[x].append((y, c))
        neigh[y].append((x, c))
    g = np.cross(pos[F[:, 1]] - pos[F[:, 0]], pos[F[:, 2]] - pos[F[:, 0]])
    match_ = (g * np.asarray(nor, float)[F].sum(1)).sum(1)      # > 0: the face looks the same way as its normals
    flips = np.zeros(len(F), bool)
    vu = np.zeros(len(F), bool)
    for start in range(len(F)):
        if vu[start]:
            continue
        sheet, pile = [start], [start]
        vu[start] = True
        while pile:
            x = pile.pop()
            for y, c in neigh[x]:
                if not vu[y]:
                    vu[y] = True
                    flips[y] = flips[x] ^ c
                    sheet.append(y)
                    pile.append(y)
        sheet = np.array(sheet)
        if np.where(flips[sheet], -match_[sheet], match_[sheet]).sum() < 0:
            flips[sheet] = ~flips[sheet]
    F[flips] = F[flips][:, [0, 2, 1]]
    return F


def reset_a_l_upright(path):
    """Applies a_l_upright to the faces of an already written GLB, in place, and recomputes the normals
    of the vertices of the flipped faces. Returns the number of flipped faces."""
    doc, b = read(path)
    b = bytearray(b)

    def view(i):
        a = doc["accessors"][i]
        return doc["bufferViews"][a["bufferView"]].get("byteOffset", 0) + a.get("byteOffset", 0), a["count"]
    pr = doc["meshes"][0]["primitives"]
    if any(doc["accessors"][x["indices"]]["componentType"] != 5125 for x in pr):
        return 0
    o, n = view(pr[0]["attributes"]["POSITION"])
    pos = np.frombuffer(bytes(b[o:o + n * 12]), "<f4").reshape(-1, 3).astype(float)
    on, n = view(pr[0]["attributes"]["NORMAL"])
    nor = np.frombuffer(bytes(b[on:on + n * 12]), "<f4").reshape(-1, 3).astype(float)
    idx = [view(x["indices"]) for x in pr]
    F = np.concatenate([np.frombuffer(bytes(b[o:o + n * 4]), "<u4").reshape(-1, 3) for o, n in idx]).astype(np.int64)
    G = a_l_upright(pos, nor, F)
    change = (G != F).any(1)
    if not change.any():
        return 0
    k = 0
    for o, n in idx:
        b[o:o + n * 4] = G[k:k + n // 3].astype("<u4").tobytes()
        k += n // 3
    # normals of the affected vertices: smooth, over coincident vertices
    w = np.unique(np.round(pos / 1e-5).astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)
    g = np.cross(pos[G[:, 1]] - pos[G[:, 0]], pos[G[:, 2]] - pos[G[:, 0]])
    accessory = np.zeros((int(w.max()) + 1, 3))
    for c in range(3):
        np.add.at(accessory, w[G[:, c]], g)
    touched = np.zeros(int(w.max()) + 1, bool)
    touched[w[G[change]].ravel()] = True
    N = accessory[w]
    l = np.linalg.norm(N, axis=1)
    redone = touched[w] & (l > 1e-12)
    nor = nor.copy()
    nor[redone] = N[redone] / l[redone][:, None]
    b[on:on + len(nor) * 12] = nor.astype("<f4").tobytes()
    j = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    j += b" " * (-len(j) % 4)
    b += b"\0" * (-len(b) % 4)
    with open(path + ".tmp", "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 28 + len(j) + len(b)) + struct.pack("<I4s", len(j), b"JSON") + j
                 + struct.pack("<I4s", len(b), b"BIN\0") + bytes(b))
    os.replace(path + ".tmp", path)
    return int(change.sum())


def write(path, pos, nor, uv, primitives, materials):
    """primitives: [{"faces": (n,3), "material": i}]; materials: dicts with
    name, image (URI or None), diffuse (rgba), contour (rgba+size), double, alpha, role."""
    pos = np.asarray(pos, np.float32)
    nor = np.asarray(nor, np.float32)
    uv = np.asarray(uv, np.float32)
    bytes_, views, accessors, prims = bytearray(), [], [], []

    def added(tab, target=None, nb_comp=None, type_c="f4", normalized=False):
        nonlocal bytes_
        while len(bytes_) % 4:
            bytes_.append(0)
        deb = len(bytes_)
        bytes_ += tab.astype("<" + type_c).tobytes()
        v = {"buffer": 0, "byteOffset": deb, "byteLength": len(bytes_) - deb}
        if target:
            v["target"] = target
        views.append(v)
        accessory = {"bufferView": len(views) - 1, "componentType": COMPONENT[type_c], "count": int(len(tab)),
               "type": TYPE.get(nb_comp, "SCALAR")}
        if type_c == "f4":
            flattened = tab.reshape(len(tab), -1)
            accessory["min"] = [float(x) for x in flattened.min(0)]
            accessory["max"] = [float(x) for x in flattened.max(0)]
        accessors.append(accessory)
        return len(accessors) - 1

    a_pos = added(pos, 34962, 3)
    a_nor = added(nor, 34962, 3)
    a_uv = added(uv, 34962, 2)
    for pr in primitives:
        f = np.asarray(pr["faces"], np.uint32).ravel()
        prims.append({"attributes": {"POSITION": a_pos, "NORMAL": a_nor, "TEXCOORD_0": a_uv},
                      "indices": added(f, 34963, 1, "u4"), "material": int(pr["material"]), "mode": 4})
    while len(bytes_) % 4:
        bytes_.append(0)
    images, textures, mats = [], [], []
    for m in materials:
        d = list(m.get("diffuse") or (1, 1, 1, 1))
        gm = {"name": str(m.get("name", "")), "extras": {k: m[k] for k in m if k != "image"},
              "doubleSided": bool(m.get("double", True)),
              "pbrMetallicRoughness": {"baseColorFactor": [float(x) for x in d], "metallicFactor": 0, "roughnessFactor": 1}}
        if m.get("alpha"):
            gm["alphaMode"] = "BLEND"
        if m.get("image"):
            if m["image"] not in images:
                images.append(m["image"])
                textures.append({"source": len(images) - 1})
            gm["pbrMetallicRoughness"]["baseColorTexture"] = {"index": images.index(m["image"])}
        mats.append(gm)
    doc = {"asset": {"version": "2.0", "generator": "chimera-workshop"},
           "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": prims}], "materials": mats,
           "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(bytes_)}]}
    if images:
        doc["images"] = [{"uri": u} for u in images]
        doc["textures"] = textures
    js = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bytes_)
    output = struct.pack("<III", 0x46546C67, 2, total)
    output += struct.pack("<II", len(js), 0x4E4F534A) + js
    output += struct.pack("<II", len(bytes_), 0x004E4942) + bytes(bytes_)
    import common as common
    common.write_bytes(path, output)
    return len(output)


def read(path):
    """Reads back a GLB written by `write` (used by: tests, server)."""
    with open(path, "rb") as f:
        b = f.read()
    magic, version, total = struct.unpack_from("<III", b, 0)
    if magic != 0x46546C67 or version != 2 or total != len(b):
        raise ValueError("invalid GLB: %s" % path)
    off, js, bin_ = 12, None, None
    while off < len(b):
        n, ty = struct.unpack_from("<II", b, off)
        d = b[off + 8: off + 8 + n]
        if ty == 0x4E4F534A:
            js = json.loads(d)
        else:
            bin_ = d
        off += 8 + n
    return js, bin_

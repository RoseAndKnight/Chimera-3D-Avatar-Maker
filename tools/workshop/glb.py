"""Writing and reading back a minimal GLB (binary glTF 2.0): one mesh, several primitives,
external textures (files shared in the cache, relative URIs). Numpy only.

Frame: MMD (x, y, z; one unit = 8 cm, the character looks towards -Z) -> glTF (x, y, -z in
metres, character facing +Z). Flipping the sign of z reverses the winding of the faces: `vers_gltf`
checks it against the mesh normals rather than assuming it.
"""
import json
import os
import struct

import numpy as np

COMPOSANTE = {"f4": 5126, "u4": 5125, "u2": 5123}
TYPE = {2: "VEC2", 3: "VEC3", 4: "VEC4"}


def vers_gltf(pos, nor, faces):
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
        f = a_l_endroit(p, n, f)
    return p, n, f


def a_l_endroit(pos, nor, F):
    """Faces drawn inside out, flipped. The page lights the back of a face with the opposite
    normal: a face drawn inside out comes out unlit, as a dark patch (under the buttocks of
    Zhu Yuan's jumpsuit, 16/11), and its outline shell passes in front of the surface (line under
    Maëlys's lip, 09/11).

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
    qui = np.tile(np.arange(len(F)), 3)
    sens = a[:, 0] < a[:, 1]
    cle = np.sort(a, 1)
    ordre = np.lexsort((cle[:, 1], cle[:, 0]))
    cle, qui, sens = cle[ordre], qui[ordre], sens[ordre]
    deb = np.r_[True, (cle[1:] != cle[:-1]).any(1)]
    nb = np.diff(np.r_[np.where(deb)[0], len(cle)])
    paire = np.where(deb)[0][nb == 2]                  # manifold edges: exactly two faces
    f1, f2 = qui[paire], qui[paire + 1]
    contre = sens[paire] == sens[paire + 1]            # same direction: one of the two is inside out
    ok = (f1 != f2) & (cle[paire, 0] != cle[paire, 1])
    f1, f2, contre = f1[ok], f2[ok], contre[ok]
    vois = [[] for _ in range(len(F))]
    for x, y, c in zip(f1.tolist(), f2.tolist(), contre.tolist()):
        vois[x].append((y, c))
        vois[y].append((x, c))
    g = np.cross(pos[F[:, 1]] - pos[F[:, 0]], pos[F[:, 2]] - pos[F[:, 0]])
    accord = (g * np.asarray(nor, float)[F].sum(1)).sum(1)      # > 0: the face looks the same way as its normals
    retourne = np.zeros(len(F), bool)
    vu = np.zeros(len(F), bool)
    for depart in range(len(F)):
        if vu[depart]:
            continue
        nappe, pile = [depart], [depart]
        vu[depart] = True
        while pile:
            x = pile.pop()
            for y, c in vois[x]:
                if not vu[y]:
                    vu[y] = True
                    retourne[y] = retourne[x] ^ c
                    nappe.append(y)
                    pile.append(y)
        nappe = np.array(nappe)
        if np.where(retourne[nappe], -accord[nappe], accord[nappe]).sum() < 0:
            retourne[nappe] = ~retourne[nappe]
    F[retourne] = F[retourne][:, [0, 2, 1]]
    return F


def remettre_a_l_endroit(chemin):
    """Applies a_l_endroit to the faces of an already written GLB, in place, and recomputes the normals
    of the vertices of the flipped faces. Returns the number of flipped faces."""
    doc, b = lire(chemin)
    b = bytearray(b)

    def vue(i):
        a = doc["accessors"][i]
        return doc["bufferViews"][a["bufferView"]].get("byteOffset", 0) + a.get("byteOffset", 0), a["count"]
    pr = doc["meshes"][0]["primitives"]
    if any(doc["accessors"][x["indices"]]["componentType"] != 5125 for x in pr):
        return 0
    o, n = vue(pr[0]["attributes"]["POSITION"])
    pos = np.frombuffer(bytes(b[o:o + n * 12]), "<f4").reshape(-1, 3).astype(float)
    on, n = vue(pr[0]["attributes"]["NORMAL"])
    nor = np.frombuffer(bytes(b[on:on + n * 12]), "<f4").reshape(-1, 3).astype(float)
    ou = [vue(x["indices"]) for x in pr]
    F = np.concatenate([np.frombuffer(bytes(b[o:o + n * 4]), "<u4").reshape(-1, 3) for o, n in ou]).astype(np.int64)
    G = a_l_endroit(pos, nor, F)
    change = (G != F).any(1)
    if not change.any():
        return 0
    k = 0
    for o, n in ou:
        b[o:o + n * 4] = G[k:k + n // 3].astype("<u4").tobytes()
        k += n // 3
    # normals of the affected vertices: smooth, over coincident vertices
    w = np.unique(np.round(pos / 1e-5).astype(np.int64), axis=0, return_inverse=True)[1].reshape(-1)
    g = np.cross(pos[G[:, 1]] - pos[G[:, 0]], pos[G[:, 2]] - pos[G[:, 0]])
    acc = np.zeros((int(w.max()) + 1, 3))
    for c in range(3):
        np.add.at(acc, w[G[:, c]], g)
    touche = np.zeros(int(w.max()) + 1, bool)
    touche[w[G[change]].ravel()] = True
    N = acc[w]
    l = np.linalg.norm(N, axis=1)
    refait = touche[w] & (l > 1e-12)
    nor = nor.copy()
    nor[refait] = N[refait] / l[refait][:, None]
    b[on:on + len(nor) * 12] = nor.astype("<f4").tobytes()
    j = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    j += b" " * (-len(j) % 4)
    b += b"\0" * (-len(b) % 4)
    with open(chemin + ".tmp", "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 28 + len(j) + len(b)) + struct.pack("<I4s", len(j), b"JSON") + j
                 + struct.pack("<I4s", len(b), b"BIN\0") + bytes(b))
    os.replace(chemin + ".tmp", chemin)
    return int(change.sum())


def ecrire(chemin, pos, nor, uv, primitives, materiaux):
    """primitives: [{"faces": (n,3), "materiau": i}]; materiaux: dicts with
    nom, image (URI or None), diffus (rgba), contour (rgba+size), double, alpha, role."""
    pos = np.asarray(pos, np.float32)
    nor = np.asarray(nor, np.float32)
    uv = np.asarray(uv, np.float32)
    octets, vues, accesseurs, prims = bytearray(), [], [], []

    def ajoute(tab, cible=None, nb_comp=None, type_c="f4", normalise=False):
        nonlocal octets
        while len(octets) % 4:
            octets.append(0)
        deb = len(octets)
        octets += tab.astype("<" + type_c).tobytes()
        v = {"buffer": 0, "byteOffset": deb, "byteLength": len(octets) - deb}
        if cible:
            v["target"] = cible
        vues.append(v)
        acc = {"bufferView": len(vues) - 1, "componentType": COMPOSANTE[type_c], "count": int(len(tab)),
               "type": TYPE.get(nb_comp, "SCALAR")}
        if type_c == "f4":
            aplati = tab.reshape(len(tab), -1)
            acc["min"] = [float(x) for x in aplati.min(0)]
            acc["max"] = [float(x) for x in aplati.max(0)]
        accesseurs.append(acc)
        return len(accesseurs) - 1

    a_pos = ajoute(pos, 34962, 3)
    a_nor = ajoute(nor, 34962, 3)
    a_uv = ajoute(uv, 34962, 2)
    for pr in primitives:
        f = np.asarray(pr["faces"], np.uint32).ravel()
        prims.append({"attributes": {"POSITION": a_pos, "NORMAL": a_nor, "TEXCOORD_0": a_uv},
                      "indices": ajoute(f, 34963, 1, "u4"), "material": int(pr["materiau"]), "mode": 4})
    while len(octets) % 4:
        octets.append(0)
    images, textures, mats = [], [], []
    for m in materiaux:
        d = list(m.get("diffus") or (1, 1, 1, 1))
        gm = {"name": str(m.get("nom", "")), "extras": {k: m[k] for k in m if k != "image"},
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
    doc = {"asset": {"version": "2.0", "generator": "atelier-personnage"},
           "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": prims}], "materials": mats,
           "accessors": accesseurs, "bufferViews": vues, "buffers": [{"byteLength": len(octets)}]}
    if images:
        doc["images"] = [{"uri": u} for u in images]
        doc["textures"] = textures
    js = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(octets)
    sortie = struct.pack("<III", 0x46546C67, 2, total)
    sortie += struct.pack("<II", len(js), 0x4E4F534A) + js
    sortie += struct.pack("<II", len(octets), 0x004E4942) + bytes(octets)
    import common as commun
    commun.ecrire_octets(chemin, sortie)
    return len(sortie)


def lire(chemin):
    """Reads back a GLB written by `ecrire` (used by: tests, server)."""
    with open(chemin, "rb") as f:
        b = f.read()
    magic, version, total = struct.unpack_from("<III", b, 0)
    if magic != 0x46546C67 or version != 2 or total != len(b):
        raise ValueError("invalid GLB: %s" % chemin)
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

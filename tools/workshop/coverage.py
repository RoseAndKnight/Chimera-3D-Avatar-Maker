"""What is known about a fitted part, face by face, to adjust the layers (covers.bin):
held to the body (it does not float on physics bones), opaque, outer side.

And, for shoes, whether the top of the foot is covered: some pairs in the library are
only a sole and a heel, the top of the foot coming from the original character's stockings.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import outfit as CB  # noqa: E402
import glb as GLB  # noqa: E402

import pmx as PMX  # noqa: E402

VOXEL_BODY = 0.012
OUTER_MIN = 0.25
FOOT_TOP = 0.10        # metres above the ground: the foot area
FOOT_NORMAL = 0.4      # the top of the foot faces upwards
FOOT_REACH = 0.03
FOOT_MIN = 0.65         # share of the top of the foot a shoe must cover
TYPES_FOOT = ("shoes", "boots")


def read_glb(path):
    """(doc, pos, nor, uv, faces, primitive number per face)"""
    doc, b = GLB.read(path)
    accessory, views = doc["accessors"], doc["bufferViews"]

    def tab(i, ty, n):
        a, v = accessory[i], views[accessory[i]["bufferView"]]
        count = a["count"] * (1 if a.get("type") == "SCALAR" else n)
        return np.frombuffer(b, ty, count, v.get("byteOffset", 0)).reshape(-1, n).copy()
    pr = doc["meshes"][0]["primitives"]
    at = pr[0]["attributes"]
    Fs = [tab(p["indices"], "<u4", 3).astype(np.int64) for p in pr]
    return doc, tab(at["POSITION"], "<f4", 3).astype(float), tab(at["NORMAL"], "<f4", 3).astype(float), \
        tab(at["TEXCOORD_0"], "<f4", 2).astype(float), np.concatenate(Fs), np.repeat(np.arange(len(Fs)), [len(f) for f in Fs])


def points_body():
    """Vertices of the bare body, one per voxel (enough to tell which side it is on)."""
    _, pos, nor, _, _, _ = read_glb(os.path.join(C.CACHE, "body", "body.glb"))
    _, i = np.unique(np.floor(pos / VOXEL_BODY).astype(np.int64), axis=0, return_index=True)
    return (pos[i], nor[i]), pos, nor


def outer(pos, nor, F, body):
    """Faces looking the same way as the closest skin. They are compared with the body's normals,
    not with the body -> face direction: a part dipping slightly under the skin keeps the right side."""
    T = pos[F]
    g = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    g /= np.maximum(np.linalg.norm(g, axis=1), 1e-12)[:, None]
    g *= np.where((g * nor[F].mean(1)).sum(1) < 0, -1.0, 1.0)[:, None]   # direction given by the part's normals
    # the threshold leaves out edge-on faces (hems, edges of a belt): they cover nothing
    return (CB._normal_body(T.mean(1), body) * g).sum(1) > OUTER_MIN


def outfit(card, folder, uv=None, F=None):
    """Per face: the face does not follow bones specific to the part, the physics ones. Those bones
    would move it separately: it cannot be used to hide anything. Without `uv` and `F`, the order is
    the PMX's. With them (those of the cached part), each vertex takes the share of the PMX vertex at
    the same place on the texture: fitting may add or remove faces, not change the UVs."""
    own_ones = set(card.get("own_bones") or [])
    if not own_ones and F is not None:
        return np.ones(len(F), bool)
    P = PMX.read(os.path.join(folder, card["pmx"]))
    t = PMX.to_arrays(P)
    is_ = np.array([o["name"] in own_ones for o in P["bones"]] + [False])  # -1 lands on the last one
    I = np.asarray(P["vertices"]["bones"], np.int64)
    W = np.asarray(P["vertices"]["weights"], float)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    part = (W * is_[I]).sum(1)
    if F is None:
        return part[t["faces"]].max(1) < 0.5
    if len(t["faces"]) == len(F) and len(part) == len(uv):
        return part[F].max(1) < 0.5
    if not len(part):
        return np.ones(len(F), bool)
    which = np.empty(len(uv), np.int64)
    for d in range(0, len(uv), 2000):
        e = uv[d:d + 2000, None, :] - t["uv"][None, :, :2]
        which[d:d + 2000] = (e * e).sum(2).argmin(1)
    return part[which][F].max(1) < 0.5


FINENESS = 12   # points per side of the sample grid that reads a face's opacity


def opaque_per_face(a, uv, F):
    """Per face: the texture (alpha `a`, 0..1, first row at the top) is opaque everywhere on the
    face. Read on a dense sample grid, not only at the vertices: an opening drawn in the texture
    (the front of a jacket) falls between the vertices of a large face."""
    h, w = a.shape
    ij = [(i, j) for i in range(FINENESS + 1) for j in range(FINENESS + 1 - i)]
    B = np.array([(i, j, FINENESS - i - j) for i, j in ij], float) / FINENESS      # (n, 3)
    U = uv[F]                                                                   # (f, 3, 2)
    op = np.ones(len(F), bool)
    for d in range(0, len(F), 4000):
        P = np.einsum("nk,fkc->fnc", B, U[d:d + 4000])
        x = (np.mod(P[..., 0], 1.0) * w).astype(int).clip(0, w - 1)
        y = (np.mod(P[..., 1], 1.0) * h).astype(int).clip(0, h - 1)
        op[d:d + 4000] = (a[y, x] > 0.5).all(1)
    return op


def flags(pos, nor, F, body, holds=None, opaque=None):
    d = np.zeros(len(F), np.uint8)
    d[outer(pos, nor, F, body)] |= CB.OUTER
    d[np.ones(len(F), bool) if holds is None else holds] |= CB.OUTFIT
    d[np.ones(len(F), bool) if opaque is None else opaque] |= CB.OPAQUE
    return d


def instep(pos, nor, F, body_pos, body_nor):
    """Share (0 to 1) of the top of the foot the part covers, within 3 cm of the skin: the lowest
    over the feet it is worn on (an odd boot is worn on only one)."""
    over = (body_pos[:, 1] < FOOT_TOP) & (body_nor[:, 1] > FOOT_NORMAL)
    parts = []
    for side in (-1, 1):
        Q = body_pos[over & (side * body_pos[:, 0] > 0)]
        if not len(Q) or not ((pos[:, 1] < FOOT_TOP) & (side * pos[:, 0] > 0)).any():
            continue
        parts.append(float(np.isfinite(CB.nearest(Q, pos, F, nor, FOOT_REACH)[2]).mean()))
    return min(parts) if parts else 0.0


def compute(ident, opaque_of=None):
    """Writes pieces/<ident>/covers.bin. `opaque_de(doc, uv, F, prim)` returns the opacity per face
    (Blender only); without it faces count as opaque. Returns (flags, foot share)."""
    d = os.path.join(C.CACHE, "parts", ident)
    m = C.read_json(os.path.join(d, "meta.json"))
    folder = C.absolute(m["folder"])
    card = C.read_json(os.path.join(folder, "record.json"))
    doc, pos, nor, uv, F, prim = read_glb(os.path.join(d, "part.glb"))
    body, cp, cn = points_body()
    holds = outfit(card, folder, uv, F)
    op = opaque_of(doc, uv, F, prim) if opaque_of else None
    fl = flags(pos, nor, F, body, holds, op)
    C.write_bytes(os.path.join(d, "covers.bin"), fl.tobytes())
    part = instep(pos, nor, F, cp, cn) if m.get("type") in TYPES_FOOT else None
    return fl, part


def keep(ident):
    """Redoes only the "held" (outfit) flag of an existing covers.bin (opacity needs Blender)."""
    d = os.path.join(C.CACHE, "parts", ident)
    m = C.read_json(os.path.join(d, "meta.json"))
    folder = C.absolute(m["folder"])
    _, _, _, uv, F, _ = read_glb(os.path.join(d, "part.glb"))
    fl = CB._read_covers(os.path.join(d, "covers.bin"), len(F))
    if fl is None:
        return None
    holds = outfit(C.read_json(os.path.join(folder, "record.json")), folder, uv, F)
    fresh = (fl & ~np.uint8(CB.OUTFIT)) | np.where(holds, CB.OUTFIT, 0).astype(np.uint8)
    if (fresh != fl).any():
        C.write_bytes(os.path.join(d, "covers.bin"), fresh.tobytes())
    return int(((fl & CB.OUTFIT) > 0).sum()), int(holds.sum()), len(F)


def reopacify(ident):
    """Redoes only the "opaque" flag of an existing covers.bin, outside Blender (PIL is needed)."""
    from PIL import Image
    d = os.path.join(C.CACHE, "parts", ident)
    doc, _, _, uv, F, prim = read_glb(os.path.join(d, "part.glb"))
    fl = CB._read_covers(os.path.join(d, "covers.bin"), len(F))
    if fl is None:
        return None
    op = np.ones(len(F), bool)
    for i, pr in enumerate(doc["meshes"][0]["primitives"]):
        mat = doc["materials"][pr["material"]]["pbrMetallicRoughness"]
        sel = np.where(prim == i)[0]
        if len(mat.get("baseColorFactor", [1, 1, 1, 1])) > 3 and mat["baseColorFactor"][3] < 0.95:
            op[sel] = False
            continue
        bt = mat.get("baseColorTexture")
        if bt is None:
            continue
        url = doc["images"][doc["textures"][bt["index"]]["source"]]["uri"]
        im = Image.open(os.path.join(C.CACHE, url[len("/data/"):]))
        if im.mode not in ("RGBA", "LA"):
            continue
        a = np.asarray(im.convert("RGBA"), np.float32)[..., 3] / 255.0
        op[sel] = opaque_per_face(a, uv, F[sel])
    fresh = (fl & ~np.uint8(CB.OPAQUE)) | np.where(op, CB.OPAQUE, 0).astype(np.uint8)
    if (fresh != fl).any():
        C.write_bytes(os.path.join(d, "covers.bin"), fresh.tobytes())
    return int(((fl & CB.OPAQUE) > 0).sum()), int(op.sum()), len(F)


if __name__ == "__main__":
    # coverage.py outfit|opaque [ids]: resets one flag of the cache's covers.bin files
    # coverage.py upright [ids]: flips the back-facing faces of the cache's parts
    import review as RV
    if sys.argv[1] == "upright":
        for i in sys.argv[2:] or RV.ids_of_cache(True):
            for f in ("part.glb", "part.before-repair.glb"):
                ch = os.path.join(C.CACHE, "parts", i, f)
                n = GLB.reset_a_l_upright(ch) if os.path.isfile(ch) else 0
                if n and f == "part.glb":
                    print("%-34s faces flipped %5d" % (i, n))
        sys.exit(0)
    if sys.argv[1] == "opaque":
        for i in sys.argv[2:] or RV.ids_of_cache(True):
            r = reopacify(i)
            if r and r[0] != r[1]:
                print("%-34s opaque faces %5d -> %5d of %5d" % (i, r[0], r[1], r[2]))
        sys.exit(0)
    for i in sys.argv[2:] or RV.ids_of_cache(True):
        try:
            r = keep(i)
        except Exception as e:
            print("%-34s not redone: %s" % (i, str(e)[:120]))
            continue
        if r and r[0] != r[1]:
            print("%-34s held faces %5d -> %5d of %5d" % (i, r[0], r[1], r[2]))

"""A head read from a GLB (a studio other than the active game): same attributes as head.Head for
registration onto the pivot head (pivot.register_one, registration.register) and the signature spectrum.

    python3 -I tools/head/head_glb.py inspect <model.glb>     # materials, chosen skin, landmarks

Reading: binary glTF 2.0 in numpy (JSON + buffer), rest positions of the primitives (bind
pose, mesh frame), without the morphs. The face skin is the material whose
name contains one of the SKIN_PATTERNS patterns (face separate from the skull); the eyes, eyebrows and
hair are found the same way from their material names.

Frame: brought back into the PMX one (Y up, the face looks towards -Z); the gaze direction
is glTF's (+Z). Units: those of the GLB
scaled to the typical interocular distance of the active game's heads (nothing depends on it after the similarity, only the
initial placement and the relative thresholds).

Nothing is written: the geometry stays in memory.
"""
import json
import os
import re
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

sys.path.insert(0, os.path.join(HERE, "..", "atlas"))
sys.path.append(os.path.join(HERE, ".."))

import geom as G  # noqa: E402
import games as games  # noqa: E402

TYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

# (lowercase) material name patterns, per studio
SKIN_PATTERNS = ("face_new", "_face", "face", "face")  # visage: Keijo (Orpheya)
SKIN_EXCLUDED = ("mask", "eye", "brow", "hair", "lash", "jiemao", "mei", "teeth", "tooth", "tongue")
EYES_PATTERNS = ("_eyes", "eyeball", "_eye", "eye", "white_of_eyes")
EYES_EXCLUDED = ("brow", "lash", "bantou", "touming", "gaoguang", "highlight", "shadow")
EYEBROWS_PATTERNS = ("brow", "meimao", "_mei", "eyebrows")
HAIR_PATTERNS = ("hair", "chevelure")   # fr-ok: material name written in some models
# typical interocular distance of the active game's heads (MMD units, description: heads.typical_eye_gap): used to
# bring the head to a similar scale
IOD_GAME = games.active().value("heads.typical_eye_gap")


def read_glb(path):
    """Returns (json, binary buffer). path: a file, or the bytes of a GLB already in memory."""
    if isinstance(path, (bytes, bytearray)):
        b = bytes(path)
    else:
        with open(path, "rb") as f:
            b = f.read()
    if b[:4] != b"glTF":
        raise ValueError("not a GLB")
    n = struct.unpack("<I", b[12:16])[0]
    j = json.loads(b[20:20 + n])
    p = 20 + n
    binary = b""
    if p + 8 <= len(b):
        m = struct.unpack("<I", b[p:p + 4])[0]
        binary = b[p + 8:p + 8 + m]
    return j, binary


def accessor(j, binary, k):
    a = j["accessors"][k]
    t = TYPES[a["componentType"]]
    c = SIZES[a["type"]]
    n = a["count"]
    out = np.zeros((n, c), dtype=t)
    if "bufferView" in a:
        bv = j["bufferViews"][a["bufferView"]]
        disp = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stage = bv.get("byteStride", 0) or c * np.dtype(t).itemsize
        raw = np.frombuffer(binary, dtype=np.uint8, count=(n - 1) * stage + c * np.dtype(t).itemsize, offset=disp)
        if stage == c * np.dtype(t).itemsize:
            out = raw.view(t).reshape(n, c).copy()
        else:
            lines = np.lib.stride_tricks.as_strided(raw, shape=(n, c * np.dtype(t).itemsize), strides=(stage, 1))
            out = lines.copy().view(t).reshape(n, c)
    if "sparse" in a:
        s = a["sparse"]
        bi, bv_ = s["indices"], s["values"]
        vi = j["bufferViews"][bi["bufferView"]]
        idx = np.frombuffer(binary, dtype=TYPES[bi["componentType"]], count=s["count"],
                            offset=vi.get("byteOffset", 0) + bi.get("byteOffset", 0))
        vv = j["bufferViews"][bv_["bufferView"]]
        val = np.frombuffer(binary, dtype=t, count=s["count"] * c,
                            offset=vv.get("byteOffset", 0) + bv_.get("byteOffset", 0)).reshape(-1, c)
        out = out.copy()
        out[idx] = val
    if a.get("normalized") and t != np.float32:
        out = out.astype(np.float64) / np.iinfo(t).max
    return out


def _matrix_node(nd):
    if "matrix" in nd:
        return np.array(nd["matrix"], dtype=float).reshape(4, 4).T
    T = np.eye(4)
    x, y, z, w = nd.get("rotation", (0, 0, 0, 1))
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T[:3, :3] = R * np.array(nd.get("scale", (1, 1, 1)))[None, :]
    T[:3, 3] = nd.get("translation", (0, 0, 0))
    return T


def _world(j):
    """World matrix of each node."""
    per = {}
    for i, nd in enumerate(j.get("nodes", [])):
        for c in nd.get("children", []):
            per[c] = i
    cache = {}

    def m(i):
        if i not in cache:
            loc = _matrix_node(j["nodes"][i])
            cache[i] = m(per[i]) @ loc if i in per else loc
        return cache[i]
    return [m(i) for i in range(len(j.get("nodes", [])))]


def mesh(path):
    """All triangulated primitives, in the scene frame: positions (n, 3),
    faces (f, 3), material of each face, material names."""
    j, binary = read_glb(path)
    names_mat = [m.get("name", "mat%d" % i) for i, m in enumerate(j.get("materials", []))]
    worlds = _world(j)
    pos, faces, mat = [], [], []
    base = 0
    for i, nd in enumerate(j.get("nodes", [])):
        if "mesh" not in nd:
            continue
        # skinned mesh: vertices in the mesh frame, under the skeleton's parent;
        # the node's world matrix (often the identity) is kept to orient the scene
        M = worlds[i]
        for pr in j["meshes"][nd["mesh"]]["primitives"]:
            if pr.get("mode", 4) != 4:
                continue
            P = accessor(j, binary, pr["attributes"]["POSITION"]).astype(np.float64)
            P = P @ M[:3, :3].T + M[:3, 3]
            if "indices" in pr:
                I = accessor(j, binary, pr["indices"]).astype(np.int64).reshape(-1, 3)
            else:
                I = np.arange(len(P)).reshape(-1, 3)
            pos.append(P)
            faces.append(I + base)
            mat.append(np.full(len(I), pr.get("material", -1)))
            base += len(P)
    return np.vstack(pos), np.vstack(faces), np.concatenate(mat), names_mat


def glb_since_pmx(path, meters=0.08):
    """GLB (bytes, in memory only) of a PMX: glTF frame (gaze +Z, the character's
    left at +X: half-turn around Y), in metres, one primitive per material. Used for the
    check: a ZZZ head sent through the GLB path must land back on its own shape."""
    import pmx as PMX
    P = PMX.read(path)
    t = PMX.to_arrays(P)
    pos = (t["pos"] * np.array([-1.0, 1.0, -1.0]) * meters).astype(np.float32)
    names = [m["name"] for m in P["materials"]]
    blocks = [pos.tobytes()]
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": pos.nbytes}]
    accessory = [{"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
            "min": pos.min(0).tolist(), "max": pos.max(0).tolist()}]
    prims = []
    disp = pos.nbytes
    for i in range(len(names)):
        I = t["faces"][t["mat_face"] == i].astype(np.uint32).ravel()
        if not len(I):
            continue
        blocks.append(I.tobytes())
        views.append({"buffer": 0, "byteOffset": disp, "byteLength": I.nbytes})
        accessory.append({"bufferView": len(views) - 1, "componentType": 5125, "count": len(I), "type": "SCALAR"})
        prims.append({"attributes": {"POSITION": 0}, "indices": len(accessory) - 1, "material": i})
        disp += I.nbytes
    binary = b"".join(blocks)
    j = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
         "accessors": accessory, "materials": [{"name": n} for n in names],
         "meshes": [{"primitives": prims}], "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0}
    js = json.dumps(j, ensure_ascii=False).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    tot = 12 + 8 + len(js) + 8 + len(binary)
    return (b"glTF" + struct.pack("<II", 2, tot) + struct.pack("<I", len(js)) + b"JSON" + js +
            struct.pack("<I", len(binary)) + b"BIN\0" + binary)


def indices_zzz(names):
    """Materials of a ZZZ head (for the check): skin, eyes, eyebrows, hair."""
    import head as TT
    ix = lambda ok: [i for i, n in enumerate(names) if ok(n)]
    # head.Head rule: 眉/眉毛 taken whole, the height filter only applies to
    # materials shared with the eyelashes
    alone = ix(lambda n: n in ("眉", "眉毛"))
    return dict(skin=ix(lambda n: n == TT.SKIN), eyes=ix(lambda n: n in TT.EYES),
                eyebrows=alone or ix(lambda n: n in ("眉睫", "睫眉", "右睫眉")), filter_eyebrows=not alone,
                keep_islands=True,
                hair=ix(lambda n: "髮" in n or "发" in n))


def _choose(names, patterns, excluded=()):
    """Indices of the materials whose name contains a pattern (the first pattern that matches)."""
    bottom = [n.lower() for n in names]
    for mo in patterns:
        r = [i for i, n in enumerate(bottom) if mo in n and not any(e in n for e in excluded)]
        if r:
            return r
    return []


def _in(poly, p):
    """Is the (2D) point p inside polygon poly (n, 2)? (ray casting rule)"""
    x, y = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cutting = (y > p[1]) != (y2 > p[1])
    xs = x + (p[1] - y) * (x2 - x) / np.where(y2 == y, 1e-12, y2 - y)
    return bool((cutting & (xs > p[0])).sum() % 2)


class HeadGlb:
    """Face skin of a GLB, same attributes as head.Head for registration."""

    def __init__(self, path, name=None, skin=None, eyes=None, eyebrows=None, hair=None, mesh_=None,
                 filter_eyebrows=True, keep_islands=False):
        self.path = path if isinstance(path, str) else "<memory>"
        self.name = name or os.path.basename(os.path.dirname(self.path)) or os.path.basename(self.path)
        pos, faces, mat, names = mesh_ if mesh_ is not None else mesh(path)
        self.names_mat = names
        self.i_skin = skin if skin is not None else _choose(names, SKIN_PATTERNS, SKIN_EXCLUDED)
        self.i_eyes = eyes if eyes is not None else _choose(names, EYES_PATTERNS, EYES_EXCLUDED)
        self.i_eyebrows = eyebrows if eyebrows is not None else _choose(names, EYEBROWS_PATTERNS)
        self.i_hair = hair if hair is not None else _choose(names, HAIR_PATTERNS)
        if not self.i_skin:
            raise ValueError("%s: no face material among %s" % (self.name, names))
        if not self.i_eyes:
            raise ValueError("%s: no eye material among %s" % (self.name, names))
        self.pos, self.faces, self.mat_face = pos, faces, mat
        self.filter_eyebrows = filter_eyebrows
        self.keep_islands = keep_islands
        self._orient()
        self.scale, self.c_src, self.c_dst = 1.0, np.zeros(3), np.zeros(3)
        self._skin()
        self._loops()
        self._landmarks()

    def vertices_mat(self, ids):
        return np.unique(self.faces[np.isin(self.mat_face, list(ids))])

    # -------------------------------------------------------------- PMX frame
    def _orient(self):
        """Y up, gaze towards -Z, interocular distance scaled to ZZZ's; centre at the eyes."""
        P = self.pos
        skin = P[self.vertices_mat(self.i_skin)]
        eyes = P[self.vertices_mat(self.i_eyes)]
        # vertical axis: the glTF scene is Y up; some engine exports stay Z
        # up (the face then extends further in z than in y around the eyes)
        e = eyes.mean(0)
        top = np.array([0.0, 1.0, 0.0])
        if np.ptp(skin[:, 2]) > 1.3 * np.ptp(skin[:, 1]):
            top = np.array([0.0, 0.0, 1.0])
        # forward: glTF convention, the character looks towards +Z (towards -Y if the scene is Z
        # up); guessing the direction from the eyes fails when the skin goes up behind the ears
        av = np.array([0.0, 0.0, 1.0]) if top[1] == 1 else np.array([0.0, -1.0, 0.0])
        # unless the sides of the face (temples, ears: the most lateral fifth of the
        # skin) are in front of the eyes: the model then faces backwards (Outerplane 2000002,
        # 4, 6, 7: sides +0.5 to +0.9 interocular distance in front; the other 71 at −0.1 to −1.2)
        lat = np.cross(top, av)
        u = (skin - e) @ lat
        sides = skin[np.abs(u) > 0.8 * np.abs(u).max()]
        if ((sides - e) @ av).mean() > 0:
            av = -av
        # PMX frame: x = left (front view, on the right of the image), y = up, -z = forward
        z = -av
        x = np.cross(top, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        Q = (P - e) @ R.T
        # interocular distance: centres of the two eyeballs
        Y = Q[self.vertices_mat(self.i_eyes)]
        iod = abs(Y[Y[:, 0] > 0].mean(0)[0] - Y[Y[:, 0] < 0].mean(0)[0])
        self.unit = IOD_GAME / iod
        self.pos = Q * self.unit
        self.rotation = R

    # -------------------------------------------------------------- welded skin
    def _skin(self):
        fb = np.where(np.isin(self.mat_face, self.i_skin))[0]
        Fb = self.faces[fb]
        vb = np.unique(Fb)
        rep, inv = G.weld(self.pos[vb], tol=1e-5 * self.unit)
        self.skin_raw = vb
        self.b2s = np.full(len(self.pos), -1)
        self.b2s[vb] = inv
        self.V = self.pos[vb[rep]].copy()
        F = self.b2s[Fb]
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F = F[ok]
        # the large connected components (detached islands: tears, blush, duplicates); a ZZZ
        # head (check) keeps everything, like head.Head (eyelid islands)
        self.F = F if self.keep_islands else self._keep_main(F)
        used = np.unique(self.F)
        remap = np.full(len(self.V), -1)
        remap[used] = np.arange(len(used))
        self.V = self.V[used]
        self.F = remap[self.F]
        self.adj = G.neighbors(len(self.V), self.F)
        self.n = len(self.V)

    @staticmethod
    def _keep_main(F):
        n = int(F.max()) + 1
        per = np.arange(n)

        def find(a):
            while per[a] != a:
                per[a] = per[per[a]]
                a = per[a]
            return a
        for a, b, c in F:
            ra, rb, rc = find(a), find(b), find(c)
            per[rb] = ra
            per[find(rc)] = ra
        r = np.array([find(i) for i in range(n)])
        cf = r[F[:, 0]]
        vals, cnt = np.unique(cf, return_counts=True)
        # components with more than 5 % of the faces (the skin and, for ZZZ, eyelid islands)
        keep = vals[cnt > 0.05 * len(F)]
        return F[np.isin(cf, keep)]

    # -------------------------------------------------------------- boundary loops
    def _loops(self):
        bs = [np.array(b) for b in G.border_loops(self.F)]
        bs.sort(key=len, reverse=True)
        # the outer boundary is the longest loop by length, not by vertex count: on a
        # sparse mesh (Outerplane, ~1100 vertices), the rim of an eye has more vertices than
        # the outline of the face; taken for the boundary, it let the outline pass for an eye
        # (eye landmarks one interocular distance away from their place, failed registration)
        lg = [np.linalg.norm(self.V[b] - self.V[np.roll(b, -1)], axis=1).sum() for b in bs]
        bs.insert(0, bs.pop(int(np.argmax(lg))))
        self.loops = bs
        self.border = np.zeros(self.n, dtype=bool)
        for b in bs:
            self.border[b] = True
        P = self.pos[self.vertices_mat(self.i_eyes)]
        self.center_eye = {}
        for side, sg in (("g", 1), ("d", -1)):
            self.center_eye[side] = P[P[:, 0] * sg > 0].mean(0)
        self.iod = float(np.linalg.norm(self.center_eye["g"] - self.center_eye["d"]))
        self.loop_ext = bs[0]
        self.loop_eye = {}
        r = 0.35 * self.iod / IOD_GAME
        for side in ("g", "d"):
            c = self.center_eye[side]
            candidate = [b for b in bs[1:] if np.linalg.norm(self.V[b].mean(0)[:2] - c[:2]) < r]
            if not candidate:
                raise ValueError("%s: no eye hole %s" % (self.name, side))
            # preferably a hole that encloses the centre of the eyeball (seen from the front): some
            # faces have other holes near the eyes (Outerplane: eyelid slits, accessories)
            around = [b for b in candidate if _in(self.V[b][:, :2], c[:2])]
            self.loop_eye[side] = max(around or candidate, key=len)
        ym = min(self.center_eye["g"][1], self.center_eye["d"][1])
        candidate = [b for b in bs[1:] if abs(self.V[b].mean(0)[0]) < 0.05 * self.iod / IOD_GAME
                and self.V[b].mean(0)[1] < ym - 0.2 * self.iod / IOD_GAME
                and not any(b is x for x in self.loop_eye.values())]
        if not candidate:
            raise ValueError("%s: no mouth hole" % self.name)
        self.loop_mouth = max(candidate, key=len)

    # -------------------------------------------------------------- landmarks (as in head.Head)
    def _landmarks(self):
        V = self.V
        R = {}
        for side in ("g", "d"):
            b = self.loop_eye[side]
            P = V[b]
            R["eye_%s_int" % side] = int(b[np.argmin(np.abs(P[:, 0]))])
            R["eye_%s_ext" % side] = int(b[np.argmax(np.abs(P[:, 0]))])
            R["eye_%s_top" % side] = int(b[np.argmax(P[:, 1])])
            R["eye_%s_bottom" % side] = int(b[np.argmin(P[:, 1])])
        b = self.loop_mouth
        P = V[b]
        R["mouth_l"] = int(b[np.argmax(P[:, 0])])
        R["mouth_r"] = int(b[np.argmin(P[:, 0])])
        mil = b[np.abs(P[:, 0]) < 0.25 * np.abs(P[:, 0]).max()]
        R["mouth_top"] = int(mil[np.argmax(V[mil, 1])])
        R["mouth_bottom"] = int(mil[np.argmin(V[mil, 1])])
        self.W = 2 * float(np.abs(V[:, 0]).max())
        med = np.where(np.abs(V[:, 0]) < 0.02 * self.W)[0]
        if len(med) < 10:
            med = np.argsort(np.abs(V[:, 0]))[:40]
        yo = 0.5 * (self.center_eye["g"][1] + self.center_eye["d"][1])
        yb = V[b, 1].mean()
        m = med[(V[med, 1] < yo) & (V[med, 1] > yb)]
        R["nose"] = int(m[np.argmin(V[m, 2])])
        cb = V[b].mean(0)
        m = med[V[med, 1] < cb[1]]
        R["chin"] = int(m[np.argmax(-(V[m, 1] - cb[1]) - (V[m, 2] - cb[2]))])
        R["forehead_top"] = int(med[np.argmax(V[med, 1])])
        for side, sg in (("g", 1), ("d", -1)):
            R["ear_%s" % side] = int(np.argmax(sg * V[:, 0]))
        self.landmarks = R
        self.points_eyebrows = self._eyebrows()
        self.H = float(V[R["forehead_top"], 1] - V[R["chin"], 1])
        top = float(V[R["forehead_top"], 1])
        hv = self.pos[self.vertices_mat(self.i_hair)] if self.i_hair else np.zeros((0, 3))
        zc = 0.5 * (V[:, 2].min() + V[:, 2].max())
        r = 0.1 * self.iod
        cal = hv[(np.abs(hv[:, 0]) < r) & (np.abs(hv[:, 2] - zc) < r) & (hv[:, 1] > top - r)] if len(hv) else hv
        self.y_skull = max(top, float(cal[:, 1].min())) if len(cal) else top
        self.H_crane = self.y_skull - float(V[R["chin"], 1])

    def _eyebrows(self):
        """Eyebrow points in the geometry, otherwise None (eyebrows painted in the texture)."""
        if not self.i_eyebrows:
            return None
        S = self.pos[self.vertices_mat(self.i_eyebrows)]
        # only the points well above the top of the eye (an eyebrow material sometimes
        # carries other strokes: for Grace, a 眉 vertex below the eyes)
        if self.filter_eyebrows:
            top_eye = max(self.V[self.loop_eye[c], 1].max() for c in "gd")
            S = S[S[:, 1] > top_eye + 0.12 * self.iod]
        if not len(S) or (S[:, 0] > 0).all() or (S[:, 0] < 0).all():
            return None
        out = {}
        for side, sg in (("g", 1), ("d", -1)):
            Q = S[S[:, 0] * sg > 0]
            out["eyebrow_%s_int" % side] = Q[np.argmin(np.abs(Q[:, 0]))]
            out["eyebrow_%s_ext" % side] = Q[np.argmax(np.abs(Q[:, 0]))]
            out["eyebrow_%s_top" % side] = Q[np.argmax(Q[:, 1])]
        return out

    def points_face(self):
        names = [k for k in self.landmarks if k.startswith(("eye", "mouth", "nose"))]
        P = {k: self.V[self.landmarks[k]] for k in names}
        P.update(self.points_eyebrows or {})
        return P

    def center_face(self):
        P = self.points_face()
        return np.mean([P[k] for k in sorted(P)], axis=0)

    def put(self, scale, c_src, c_dst):
        self.scale, self.c_src, self.c_dst = float(scale), np.asarray(c_src), np.asarray(c_dst)

    def _host(self, p):
        return self.scale * (np.asarray(p) - self.c_src) + self.c_dst

    def summary(self):
        return {"path": self.path, "skin": [self.names_mat[i] for i in self.i_skin],
                "eyes": [self.names_mat[i] for i in self.i_eyes],
                "eyebrows": [self.names_mat[i] for i in self.i_eyebrows] if self.points_eyebrows else None,
                "skin_vertices": self.n, "skin_faces": len(self.F),
                "loops": [len(b) for b in self.loops[:8]], "width": round(self.W, 4),
                "height": round(self.H, 4), "height_skull": round(self.H_crane, 4), "iod": round(self.iod, 4),
                "unit_to_mmd": round(self.unit, 5)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("inspect")
    p.add_argument("glb", nargs="+")
    a = ap.parse_args(argv)
    for c in a.glb:
        try:
            print(json.dumps(HeadGlb(c).summary(), ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            j, _ = read_glb(c)
            print(json.dumps({"path": c, "error": str(e),
                              "materials": [m.get("name") for m in j.get("materials", [])]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

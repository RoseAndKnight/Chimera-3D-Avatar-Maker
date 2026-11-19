"""The free face and hairstyle: enough to open Chimera without any game model.

    python3 -I free_face.py <MakeHuman sources> <folder>      reads body.pmx and shape.pkl; writes
        head.pmx       the free body and its eyes (materials 肌 and 目): Chimera's "base head"
        hairstyle.pmx  a bob with a fringe (material 髪), held by the head bone
        eyes.png, hair.png

Everything comes from MakeHuman (CC0): the eyes are its simple eyes (eyes/low-poly, placed on the helper
vertices `helper-*-eye` of base.obj) and their texture; the hair is its helper volume `helper-hair` (locks
falling from the top of the skull), trimmed here: fringe above the eyes, bob at the chin. Under the locks,
a cap (the skull skin lifted by a hair) closes the gaps. The fringe covers the eyebrows.
"""
import os
import pickle
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(HERE, "..", d) for d in ("custom_body", "rebuilt_body", "atlas", "head", "rebuilt_head")]
import pmx as PMX             # noqa: E402
import pmx_write as pmx_write             # noqa: E402
import png                    # noqa: E402
import geom as G              # noqa: E402
import makehuman as MH        # noqa: E402
import body as CO            # noqa: E402
import free_head as TE       # noqa: E402

CM = 8.0
EYES = {"obj": "eyes/low-poly/low-poly.obj", "mhclo": "eyes/low-poly/low-poly.mhclo", "texture": "eyes/materials/brown_eye.png"}
HAIR = (74, 52, 44)        # flat hair colour
FRINGE = 2.3                  # cm above the centre of the eyes where the fringe stops
SQUARE = -11.0                 # cm relative to the centre of the eyes where the bob stops (sides and nape)
FRINGE_ANGLE = (38.0, 62.0)   # degrees around the head, from the front: end of the fringe, start of the bob
CAP = 0.25                # cm between the skull skin and the cap
CAP_FRONT = 2.5           # cm above the eyes where the cap starts, in front (under the fringe)
CAP_NAPE = -6.0          # and behind


def _smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def _obj(path):
    """Vertices, uv, faces [(vertex, uv), ...] of an .obj."""
    V, UV, F = [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for l in f:
            if l.startswith("v "):
                V.append([float(x) for x in l.split()[1:4]])
            elif l.startswith("vt "):
                UV.append([float(x) for x in l.split()[1:3]])
            elif l.startswith("f "):
                F.append([tuple(int(x) - 1 for x in c.split("/")[:2]) for c in l.split()[1:]])
    return np.array(V), np.array(UV), F


def _indices_mhclo(path):
    """The base.obj vertices that each vertex of the item follows (mhclo with one index per line)."""
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f]
    k = next(i for i, l in enumerate(lines) if l.startswith("verts"))
    out = []
    for l in lines[k + 1:]:
        if not l or l.startswith("#"):
            continue
        if len(l.split()) != 1:
            break
        out.append(int(l))
    return np.array(out)


def _triangles(faces):
    """Faces with 3 or 4 sides -> triangles, winding reversed (Chimera's frame flips z)."""
    T = []
    for f in faces:
        f = list(f)[::-1]
        T += [(f[0], f[k], f[k + 1]) for k in range(1, len(f) - 1)]
    return T


def read(sources, d):
    """MakeHuman's vertices (silhouette of silhouette.json) in the frame of the finished body, at the level of
    the head: centimetres, the head moved back by the straightened neck, raised by the pose."""
    V, _, J, Gr = MH.read(os.path.join(sources, "base.obj"))
    with open(os.path.join(d, "shape.pkl"), "rb") as f:
        g = pickle.load(f)
    ground = J["ground"][1]
    dy = g["head"]["dy"]
    to = lambda p: (np.asarray(p, float) - [0, ground, 0]) * [10.0, 10.0, -10.0] + [0.0, dy, CO.NECK_SETBACK]   # noqa: E731
    return to(V), {k: to(p) for k, p in J.items()}, Gr, g


def eyes(sources, P):
    """The two eyeballs: (positions, uv, triangles)."""
    Vo, UV, F = _obj(os.path.join(sources, EYES["obj"]))
    ids = _indices_mhclo(os.path.join(sources, EYES["mhclo"]))
    assert len(ids) == len(Vo), "the eyes file does not follow one base.obj vertex per vertex"
    corners, T = {}, []
    for t in _triangles(F):
        T.append([corners.setdefault(c, len(corners)) for c in t])
    c = np.array(list(corners))
    return P[ids[c[:, 0]]], UV[c[:, 1]] * [1.0, -1.0] + [0.0, 1.0], np.array(T)


def hair(P, J, Gr, g):
    """The haircut: MakeHuman's locks trimmed, and the cap underneath. (positions, triangles)."""
    eye = 0.5 * (J["l-eye"] + J["r-eye"])
    axis = J["head"]

    def angle(X):
        return np.degrees(np.abs(np.arctan2(X[:, 0], -(X[:, 2] - axis[2]))))

    def between(X, ahead, behind):
        return eye[1] + ahead + (behind - ahead) * _smooth((angle(X) - FRINGE_ANGLE[0]) / (FRINGE_ANGLE[1] - FRINGE_ANGLE[0]))
    ids = np.unique(np.concatenate([np.array(f) for f in Gr["helper-hair"]]))
    loc = -np.ones(len(P), int)
    loc[ids] = np.arange(len(ids))
    M = P[ids]
    Tm = loc[np.array(_triangles(Gr["helper-hair"]))]
    M, Tm = TE._cut(M, Tm, M[:, 1] - between(M, FRINGE, SQUARE))
    t = g["head"]
    N = G.normals_vertices(t["P"], t["tris"])
    Cp = t["P"] + CAP * N
    Cp, Tc = TE._cut(Cp, t["tris"], t["P"][:, 1] - between(t["P"], CAP_FRONT, CAP_NAPE))
    u = np.unique(Tc)                                # the cut keeps every touched vertex: tighten
    loc = -np.ones(len(Cp), int)
    loc[u] = np.arange(len(u))
    return np.concatenate([M, Cp[u]]), np.concatenate([Tm, loc[Tc] + len(M)])


def _material(name, name_en, texture, n_faces, outline=True):
    return {"name": name, "name_en": name_en, "diffuse": (1.0, 1.0, 1.0, 1.0), "specular": (0.0, 0.0, 0.0), "brightness": 5.0,
            "ambient": (0.5, 0.5, 0.5), "flags": 0x1F if outline else 0x0F, "outline_color": (0.2, 0.14, 0.12, 1.0),
            "outline_size": 0.6, "texture": texture, "sphere": -1, "sphere_mode": 0, "toon_sharing": 1, "toon": 0,
            "memo": "", "n_indices": 3 * n_faces}


def _vertices(P, T, uv, i_head):
    n = len(P)
    N = G.normals_vertices(P, T)
    return {"pos": [tuple(p) for p in (P / CM).tolist()], "nor": [tuple(p) for p in N.tolist()],
            "uv": [tuple(float(x) for x in u) for u in uv], "uv_add": [[] for _ in range(n)], "type": [2] * n,
            "bones": [(i_head, 0, 0, 0)] * n, "weights": [(1.0, 0.0, 0.0, 0.0)] * n, "sdef": [None] * n, "contour": [1.0] * n}


def main():
    sources, d = sys.argv[1], sys.argv[2]
    P, J, Gr, g = read(sources, d)
    body = PMX.read(os.path.join(d, "body.pmx"))
    i_head = [o["name"] for o in body["bones"]].index("頭")
    # head: the body, then the eyes after it (the body vertices keep their index)
    Py, uvy, Ty = eyes(sources, P)
    sy = _vertices(Py, Ty, uvy, i_head)
    n0 = len(body["vertices"]["pos"])
    head = dict(body, name="free head", name_en="free head", textures=body["textures"] + ["eyes.png"],
                vertices={k: list(v) + list(sy[k]) for k, v in body["vertices"].items()},
                faces=list(body["faces"]) + [tuple(int(x) + n0 for x in t) for t in Ty],
                materials=body["materials"] + [_material("目", "eyes", len(body["textures"]), len(Ty), outline=False)])
    shutil.copyfile(os.path.join(sources, EYES["texture"]), os.path.join(d, "eyes.png"))
    pmx_write.write(head, os.path.join(d, "head.pmx"))
    Pc, Tc = hair(P, J, Gr, g)
    png.write(os.path.join(d, "hair.png"), np.full((8, 8, 3), HAIR, np.uint8))
    hairstyle = dict(body, name="free hairstyle", name_en="free hair", textures=["hair.png"],
                    vertices=_vertices(Pc, Tc, np.full((len(Pc), 2), 0.5), i_head),
                    faces=[tuple(int(x) for x in t) for t in Tc], materials=[_material("髪", "hair", 0, len(Tc))])
    pmx_write.write(hairstyle, os.path.join(d, "hairstyle.pmx"))
    print("wrote head.pmx (%d eye vertices, %d faces) and hairstyle.pmx (%d vertices, %d faces)" % (len(Py), len(Ty), len(Pc), len(Tc)))


if __name__ == "__main__":
    main()

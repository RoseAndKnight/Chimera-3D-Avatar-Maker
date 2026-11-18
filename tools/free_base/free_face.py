"""The free face and hairstyle: enough to open Chimera without any game model.

    python3 -I free_face.py <MakeHuman sources> <folder>      reads corps.pmx and forme.pkl; writes
        tete.pmx       the free body and its eyes (materials 肌 and 目): Chimera's "base head"
        coiffure.pmx   a bob with a fringe (material 髪), held by the head bone
        yeux.png, cheveux.png

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

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI] + [os.path.join(ICI, "..", d) for d in ("custom_body", "rebuilt_body", "atlas", "head", "rebuilt_head")]
import pmx as PMX             # noqa: E402
import pmx_write as pmx_ecrire             # noqa: E402
import png                    # noqa: E402
import geom as G              # noqa: E402
import makehuman as MH        # noqa: E402
import body as CO            # noqa: E402
import free_head as TE       # noqa: E402

CM = 8.0
YEUX = {"obj": "eyes/low-poly/low-poly.obj", "mhclo": "eyes/low-poly/low-poly.mhclo", "texture": "eyes/materials/brown_eye.png"}
CHEVEUX = (74, 52, 44)        # flat hair colour
FRANGE = 2.3                  # cm above the centre of the eyes where the fringe stops
CARRE = -11.0                 # cm relative to the centre of the eyes where the bob stops (sides and nape)
FRANGE_ANGLE = (38.0, 62.0)   # degrees around the head, from the front: end of the fringe, start of the bob
CALOTTE = 0.25                # cm between the skull skin and the cap
CALOTTE_FRONT = 2.5           # cm above the eyes where the cap starts, in front (under the fringe)
CALOTTE_NUQUE = -6.0          # and behind


def _lisse(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def _obj(chemin):
    """Vertices, uv, faces [(vertex, uv), ...] of an .obj."""
    V, UV, F = [], [], []
    with open(chemin, encoding="utf-8", errors="replace") as f:
        for l in f:
            if l.startswith("v "):
                V.append([float(x) for x in l.split()[1:4]])
            elif l.startswith("vt "):
                UV.append([float(x) for x in l.split()[1:3]])
            elif l.startswith("f "):
                F.append([tuple(int(x) - 1 for x in c.split("/")[:2]) for c in l.split()[1:]])
    return np.array(V), np.array(UV), F


def _indices_mhclo(chemin):
    """The base.obj vertices that each vertex of the item follows (mhclo with one index per line)."""
    with open(chemin, encoding="utf-8") as f:
        lignes = [l.strip() for l in f]
    k = next(i for i, l in enumerate(lignes) if l.startswith("verts"))
    out = []
    for l in lignes[k + 1:]:
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


def lire(sources, d):
    """MakeHuman's vertices (silhouette of silhouette.json) in the frame of the finished body, at the level of
    the head: centimetres, the head moved back by the straightened neck, raised by the pose."""
    V, _, J, Gr = MH.lire(os.path.join(sources, "base.obj"))
    with open(os.path.join(d, "forme.pkl"), "rb") as f:
        g = pickle.load(f)
    sol = J["ground"][1]
    dy = g["tete"]["dy"]
    vers = lambda p: (np.asarray(p, float) - [0, sol, 0]) * [10.0, 10.0, -10.0] + [0.0, dy, CO.COU_RECUL]   # noqa: E731
    return vers(V), {k: vers(p) for k, p in J.items()}, Gr, g


def yeux(sources, P):
    """The two eyeballs: (positions, uv, triangles)."""
    Vo, UV, F = _obj(os.path.join(sources, YEUX["obj"]))
    ids = _indices_mhclo(os.path.join(sources, YEUX["mhclo"]))
    assert len(ids) == len(Vo), "the eyes file does not follow one base.obj vertex per vertex"
    coins, T = {}, []
    for t in _triangles(F):
        T.append([coins.setdefault(c, len(coins)) for c in t])
    c = np.array(list(coins))
    return P[ids[c[:, 0]]], UV[c[:, 1]] * [1.0, -1.0] + [0.0, 1.0], np.array(T)


def cheveux(P, J, Gr, g):
    """The haircut: MakeHuman's locks trimmed, and the cap underneath. (positions, triangles)."""
    oeil = 0.5 * (J["l-eye"] + J["r-eye"])
    axe = J["head"]

    def angle(X):
        return np.degrees(np.abs(np.arctan2(X[:, 0], -(X[:, 2] - axe[2]))))

    def entre(X, devant, derriere):
        return oeil[1] + devant + (derriere - devant) * _lisse((angle(X) - FRANGE_ANGLE[0]) / (FRANGE_ANGLE[1] - FRANGE_ANGLE[0]))
    ids = np.unique(np.concatenate([np.array(f) for f in Gr["helper-hair"]]))
    loc = -np.ones(len(P), int)
    loc[ids] = np.arange(len(ids))
    M = P[ids]
    Tm = loc[np.array(_triangles(Gr["helper-hair"]))]
    M, Tm = TE._couper(M, Tm, M[:, 1] - entre(M, FRANGE, CARRE))
    t = g["tete"]
    N = G.normales_sommets(t["P"], t["tris"])
    Cp = t["P"] + CALOTTE * N
    Cp, Tc = TE._couper(Cp, t["tris"], t["P"][:, 1] - entre(t["P"], CALOTTE_FRONT, CALOTTE_NUQUE))
    u = np.unique(Tc)                                # _couper keeps every touched vertex: tighten
    loc = -np.ones(len(Cp), int)
    loc[u] = np.arange(len(u))
    return np.concatenate([M, Cp[u]]), np.concatenate([Tm, loc[Tc] + len(M)])


def _materiau(nom, nom_en, texture, n_faces, contour=True):
    return {"nom": nom, "nom_en": nom_en, "diffus": (1.0, 1.0, 1.0, 1.0), "speculaire": (0.0, 0.0, 0.0), "brillance": 5.0,
            "ambiant": (0.5, 0.5, 0.5), "drapeaux": 0x1F if contour else 0x0F, "contour_couleur": (0.2, 0.14, 0.12, 1.0),
            "contour_taille": 0.6, "texture": texture, "sphere": -1, "sphere_mode": 0, "toon_partage": 1, "toon": 0,
            "memo": "", "n_indices": 3 * n_faces}


def _sommets(P, T, uv, i_tete):
    n = len(P)
    N = G.normales_sommets(P, T)
    return {"pos": [tuple(p) for p in (P / CM).tolist()], "nor": [tuple(p) for p in N.tolist()],
            "uv": [tuple(float(x) for x in u) for u in uv], "uv_add": [[] for _ in range(n)], "type": [2] * n,
            "os": [(i_tete, 0, 0, 0)] * n, "poids": [(1.0, 0.0, 0.0, 0.0)] * n, "sdef": [None] * n, "contour": [1.0] * n}


def main():
    sources, d = sys.argv[1], sys.argv[2]
    P, J, Gr, g = lire(sources, d)
    corps = PMX.lire(os.path.join(d, "corps.pmx"))
    i_tete = [o["nom"] for o in corps["os"]].index("頭")
    # head: the body, then the eyes after it (the body vertices keep their index)
    Py, uvy, Ty = yeux(sources, P)
    sy = _sommets(Py, Ty, uvy, i_tete)
    n0 = len(corps["sommets"]["pos"])
    tete = dict(corps, nom="tête libre", nom_en="free head", textures=corps["textures"] + ["yeux.png"],
                sommets={k: list(v) + list(sy[k]) for k, v in corps["sommets"].items()},
                faces=list(corps["faces"]) + [tuple(int(x) + n0 for x in t) for t in Ty],
                materiaux=corps["materiaux"] + [_materiau("目", "eyes", len(corps["textures"]), len(Ty), contour=False)])
    shutil.copyfile(os.path.join(sources, YEUX["texture"]), os.path.join(d, "yeux.png"))
    pmx_ecrire.ecrire(tete, os.path.join(d, "tete.pmx"))
    Pc, Tc = cheveux(P, J, Gr, g)
    png.ecrire(os.path.join(d, "cheveux.png"), np.full((8, 8, 3), CHEVEUX, np.uint8))
    coiffure = dict(corps, nom="coiffure libre", nom_en="free hair", textures=["cheveux.png"],
                    sommets=_sommets(Pc, Tc, np.full((len(Pc), 2), 0.5), i_tete),
                    faces=[tuple(int(x) for x in t) for t in Tc], materiaux=[_materiau("髪", "hair", 0, len(Tc))])
    pmx_ecrire.ecrire(coiffure, os.path.join(d, "coiffure.pmx"))
    print("wrote tete.pmx (%d eye vertices, %d faces) and coiffure.pmx (%d vertices, %d faces)" % (len(Py), len(Ty), len(Pc), len(Tc)))


if __name__ == "__main__":
    main()

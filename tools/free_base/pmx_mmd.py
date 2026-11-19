"""The free body written as PMX, on a skeleton with the names of the MMD convention ("semi-standard" bones).

Chimera carries parts from one body to another through these names (下半身, 上半身, 腕, 腕捩, ひじ, 足D...): they are
a public convention of the format, not the property of a game. The positions all come from our skeleton
(body.py); the weights are our weights, translated: one of our bones -> one bone of the convention, and the twist of
our steps spread over those of the convention (0, 1/4, 1/2, 3/4, 1 of the roll).

    python3 -I pmx_mmd.py <folder>      reads shape.pkl, own-rig/skeleton.json, own-rig/weights.npz; writes body.pmx
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(HERE, "..", d) for d in ("custom_body", "rebuilt_body", "atlas", "head", "rebuilt_head")]
import pmx as PMX             # noqa: E402
import pmx_write as pmx_write             # noqa: E402
import png                    # noqa: E402
import weights as PO            # noqa: E402
import rebuild as RF          # noqa: E402  (rebuilt body: UV maps)
import skeleton as SQ        # noqa: E402
import geom as G              # noqa: E402
import free_head as TE           # noqa: E402

CM = 8.0                      # cm per MMD unit
SKIN = (236, 204, 188)        # flat skin colour, until there is a painted skin
TIERS = (0.0, 0.25, 0.5, 0.75, 1.0)
BAND = 0.8                   # share of the texture width left to the body maps; the rest goes to the head
FINGERS = (("thumb", "親指", "０１２"), ("index", "人指", "１２３"), ("middle", "中指", "１２３"),
          ("ring", "薬指", "１２３"), ("little", "小指", "１２３"))


def skeleton_mmd(sq):
    """List of bones {name, parent, pos (cm), addition: (source, factor) or None, layer}."""
    J = {n: sq.heads[k] for n, k in sq.i.items()}
    os_ = []

    def os(name, parent, pos, addition=None, layer=0):
        os_.append({"name": name, "parent": parent, "pos": np.asarray(pos, float), "addition": addition, "layer": layer})

    hip = J["thigh_L"] * [0, 1, 1]
    os("全ての親", None, [0, 0, 0])
    os("操作中心", None, [0, 0, 0])
    os("センター", "全ての親", [0, 0.6 * hip[1], 0])
    os("グルーブ", "センター", [0, 0.6 * hip[1] + 1.0, 0])
    os("腰", "グルーブ", 0.5 * (J["pelvis"] + hip))
    os("下半身", "腰", J["pelvis"])
    os("上半身", "腰", J["spine"])
    os("上半身1", "上半身", J["chest"])
    os("上半身2", "上半身1", J["upper_chest"])
    os("首", "上半身2", J["neck"])
    os("頭", "首", J["head"])
    for c, m in (("L", "左"), ("R", "右")):
        def j(n):
            return J[n + "_" + c]
        os("腰キャンセル" + m, "下半身", j("thigh"), ("腰", -1.0))
        os(m + "足", "腰キャンセル" + m, j("thigh"))
        os(m + "ひざ", m + "足", j("leg"))
        os(m + "足首", m + "ひざ", j("foot"))
        os(m + "つま先", m + "足首", j("toes"))
        os(m + "足D", "腰キャンセル" + m, j("thigh"), (m + "足", 1.0), 1)
        os(m + "ひざD", m + "足D", j("leg"), (m + "ひざ", 1.0), 1)
        os(m + "足首D", m + "ひざD", j("foot"), (m + "足首", 1.0), 1)
        os(m + "足先EX", m + "足首D", j("toes"), None, 1)
        os(m + "足IK親", "全ての親", j("foot") * [1, 0, 1])
        os(m + "足ＩＫ", m + "足IK親", j("foot"))
        os(m + "つま先ＩＫ", m + "足ＩＫ", j("toes"))
        os(m + "肩P", "上半身2", j("shoulder"))
        os(m + "肩", m + "肩P", j("shoulder"))
        os(m + "肩C", m + "肩", j("arm"), (m + "肩P", -1.0))
        os(m + "腕", m + "肩C", j("arm"))
        a, b, p = j("arm"), j("forearm"), j("hand")
        os(m + "腕捩", m + "腕", 0.5 * (a + b))
        for k in (1, 2, 3):
            os(m + "腕捩%d" % k, m + "腕", a + 0.25 * k * (b - a), (m + "腕捩", 0.25 * k))
        os(m + "ひじ", m + "腕捩", b)
        os(m + "手捩", m + "ひじ", 0.5 * (b + p))
        for k in (1, 2, 3):
            os(m + "手捩%d" % k, m + "ひじ", b + 0.25 * k * (p - b), (m + "手捩", 0.25 * k))
        os(m + "手首", m + "手捩", p)
        os(m + "ダミー", m + "手首", 0.5 * (p + j("middle1")))
        for d, name, digits in FINGERS:
            for k in range(3):
                os(m + name + digits[k], m + "手首" if k == 0 else m + name + digits[k - 1], j("%s%d" % (d, k + 1)))
    return os_


def translate(w, sq, names):
    """Weights on our skeleton -> weights on the bones of the convention."""
    pos_of = {n: k for k, n in enumerate(names)}
    out = np.zeros((len(w), len(names)))
    done = np.zeros(w.shape[1], bool)

    def twist(os_, parts, targets):
        """A group of stepped bones: the roll share of each vertex, spread over the steps of the convention."""
        k = [sq.i[o] for o in os_]
        W = w[:, k].sum(1)
        tau = np.clip((w[:, k] * parts).sum(1) / np.maximum(W, 1e-12), 0, 1) * (len(TIERS) - 1)
        bottom = np.minimum(tau.astype(int), len(TIERS) - 2)
        f = tau - bottom
        np.add.at(out, (np.arange(len(w)), np.array([pos_of[c] for c in targets])[bottom]), W * (1 - f))
        np.add.at(out, (np.arange(len(w)), np.array([pos_of[c] for c in targets])[bottom + 1]), W * f)
        done[k] = True

    for c, m in (("L", "左"), ("R", "右")):
        twist(["arm_twist1_" + c, "arm_twist2_" + c, "arm_twist3_" + c, "arm_" + c], (0, 1 / 3, 2 / 3, 1),
                [m + "腕", m + "腕捩1", m + "腕捩2", m + "腕捩3", m + "腕捩"])
        twist(["forearm_" + c, "forearm_twist1_" + c, "forearm_twist2_" + c, "forearm_twist3_" + c],
                (0, 1 / 3, 2 / 3, 1), [m + "ひじ", m + "手捩1", m + "手捩2", m + "手捩3", m + "手捩"])
        simple = {"shoulder": "肩", "hand": "手首", "thigh": "足D", "thigh_twist1": "足D", "thigh_twist2": "足D",
                   "thigh_twist3": "足D", "leg": "ひざD", "foot": "足首D", "toes": "足先EX"}
        simple.update({"toe%d" % k: "足先EX" for k in range(1, 6)})
        simple.update({"%s%d" % (d, k + 1): name + ch[k] for d, name, ch in FINGERS for k in range(3)})
        for a, b in simple.items():
            out[:, pos_of[m + b]] += w[:, sq.i[a + "_" + c]]
            done[sq.i[a + "_" + c]] = True
    for a, b in (("pelvis", "下半身"), ("spine", "上半身"), ("chest", "上半身1"), ("upper_chest", "上半身2"),
                 ("neck", "首"), ("head", "頭"), ("root", "全ての親")):
        out[:, pos_of[b]] += w[:, sq.i[a]]
        done[sq.i[a]] = True
    assert done.all(), [n for n, f in zip(sq.names, done) if not f]
    return out / out.sum(1, keepdims=True)


def write(g, sq, w, path):
    bones = skeleton_mmd(sq)
    names = [o["name"] for o in bones]
    pos_of = {n: k for k, n in enumerate(names)}
    # one texture map per tube of the grid: the seam vertices are split
    F, origin, uv, _ = RF.maps_uv(g, None)
    Pc, T = g["P"], PO.triangles(g["quads"])
    if "head" in g:                                  # MakeHuman's head, after the grid vertices
        t = g["head"]
        nc = len(Pc)
        w = np.concatenate([w, TE.weight(g, t, w, sq.i["head"])])
        Pc = np.concatenate([Pc, t["P"]])
        T = np.concatenate([T, t["tris"] + nc, t["bridge"]])
        first = {}
        for k, v in enumerate(origin):
            first.setdefault(int(v), k)
        to = np.concatenate([[first[v] for v in range(nc)], len(origin) + np.arange(len(t["P"]))])
        F = np.concatenate([F, to[np.concatenate([t["tris"] + nc, t["bridge"]])]])
        origin = np.concatenate([origin, nc + np.arange(len(t["P"]))])
        # head map: front view, in the band that the body maps leave on the right
        bo = t["P"].min(0), t["P"].max(0)
        uvt = (t["P"][:, :2] - bo[0][:2]) / (bo[1][:2] - bo[0][:2]).max()
        uv = np.concatenate([np.asarray(uv) * [BAND, 1.0], np.c_[BAND + 0.01 + uvt[:, 0] * (0.98 - BAND), 0.01 + (1 - uvt[:, 1]) * (0.98 - BAND)]])
    idx, pw = PO.four(translate(w, sq, names))
    idx = np.where(idx < 0, 0, idx)
    N = G.normals_vertices(Pc, T)[origin]
    P = Pc[origin]
    idx, pw = idx[origin], pw[origin]
    n = len(P)
    os_pmx = []
    for o in bones:
        f = PMX.OS_ROTATION | PMX.OS_VISIBLE | PMX.OS_ACTIVE | (PMX.OS_ADDITION_ROT if o["addition"] else 0)
        if o["name"] in ("全ての親", "センター", "グルーブ") or "ＩＫ" in o["name"] or "IK" in o["name"]:
            f |= PMX.OS_DISPLACEMENT
        b = {"name": o["name"], "name_en": o["name"], "pos": tuple(float(x) / CM for x in o["pos"]),
             "parent": -1 if o["parent"] is None else pos_of[o["parent"]], "layer": o["layer"], "flags": f,
             "tail_bone": None, "tail_offset": (0.0, 0.0, 0.0)}
        if o["addition"]:
            b["addition"] = {"source": pos_of[o["addition"][0]], "factor": float(o["addition"][1])}
        os_pmx.append(b)
    vert = {"pos": [tuple(p) for p in (P / CM).tolist()], "nor": [tuple(p) for p in N.tolist()],
           "uv": [tuple(float(x) for x in u) for u in uv], "uv_add": [[] for _ in range(n)], "type": [2] * n,
           "bones": [tuple(int(x) for x in r) for r in idx], "weights": [tuple(float(x) for x in r) for r in pw],
           "sdef": [None] * n, "contour": [1.0] * n}
    c = tuple(x / 255 for x in SKIN)
    mat = {"name": "肌", "name_en": "skin", "diffuse": c + (1.0,), "specular": (0.0, 0.0, 0.0), "brightness": 5.0,
           "ambient": tuple(0.5 * x for x in c), "flags": 0x1E, "outline_color": (0.3, 0.2, 0.2, 1.0),
           "outline_size": 0.6, "texture": 0, "sphere": -1, "sphere_mode": 0, "toon_sharing": 1, "toon": 0,
           "memo": "", "n_indices": 3 * len(F)}
    png.write(os.path.join(os.path.dirname(path), "skin.png"), np.full((8, 8, 3), SKIN, np.uint8))
    pmx_write.write({"version": 2.0, "encoding": "UTF-16", "name": "free base body", "name_en": "free base body",
                       "comment": "Free base body: surface drawn from MakeHuman's mesh (CC0), skeleton and weights "
                                      "computed by Chimera.", "comment_en": "", "n_uv_add": 0, "vertices": vert,
                       "faces": [tuple(int(x) for x in f) for f in F], "textures": ["skin.png"], "materials": [mat],
                       "bones": os_pmx, "morphs": [], "frames": [], "rigid_bodies": [], "joints": []}, path)
    print("wrote %s: %d vertices, %d faces, %d bones" % (path, n, len(F), len(bones)))


def main():
    d = sys.argv[1]
    with open(os.path.join(d, "shape.pkl"), "rb") as f:
        g = pickle.load(f)
    sq = SQ.read(os.path.join(d, "own-rig", "skeleton.json"))
    Z = np.load(os.path.join(d, "own-rig", "weights.npz"))
    assert list(Z["bones"]) == sq.names
    write(g, sq, Z["weights"].astype(float), os.path.join(d, "body.pmx"))


if __name__ == "__main__":
    main()

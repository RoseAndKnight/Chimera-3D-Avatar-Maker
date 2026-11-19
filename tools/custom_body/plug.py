"""Plug the house skeleton into a character assembled by Chimera (rebuilt body, head, hairstyle, outfit).

    python3 -I plug.py INPUT.pmx OUTPUT.pmx [--own-rig FOLDER of the house skeleton]

The character arrives weighted on the ZZZ bones. The body bones of ZZZ (trunk, arms, hands, legs, feet, their
control, twist and IK bones) are replaced by ours; everything else is kept as is and
reattached to our bones: eyes, hair, chest, buttocks, skirts, accessories, their physics, the expressions.

The weights, vertex by vertex, for the share it had on the replaced bones:
 - the skin of the body takes our weights (weights.npz), exactly;
 - a garment takes those of the skin it touches, corrected by what its own weights had that differed
   from that skin:  w = M z(v) + f . (w_skin(s) - M z(s)),  z the ZZZ weights, s the chosen skin point,
   M the translation ZZZ bones -> house bones learnt on the skin (where we have both), f = 1 against the skin and 0
   far from it or when the garment is not weighted like it (a loose skirt keeps its weights, translated by M).
   The skin point is chosen close AND similarly weighted: the left crotch of a pair of trousers does not take
   the skin of the right thigh.
The share placed on kept bones (physics of a skirt, chest) does not change.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(HERE, "..", d) for d in ("rebuilt_body", "clothes", "atlas", "head")]
import pmx as PMX                               # noqa: E402
import pmx_write as pmx_write                               # noqa: E402
import skeleton as SQ                          # noqa: E402

CM = 8.0
import measurements as _ME     # noqa: E402
OWN_RIG = _ME.OWN_RIG       # house skeleton of Chimera's body (CHIMERA_BODY), see measurements.py
NEAR, FAR = 2.0, 7.0       # cm: the skin counts in full up to PRES, not at all from LOIN
WEIGHT_GAP = 12.0          # cm: what an entirely different weight is worth in the choice of the skin point
FINGERS = [d + k for d, ks in (("親指", "０１２"), ("人指", "１２３"), ("中指", "１２３"), ("薬指", "１２３"), ("小指", "１２３")) for k in ks]
PER_SIDE = (["足", "ひざ", "足首", "つま先", "足D", "ひざD", "足首D", "足先EX", "足IK親", "足ＩＫ", "つま先ＩＫ", "肩P", "肩", "肩C", "腕", "腕捩",
             "腕捩1", "腕捩2", "腕捩3", "ひじ", "手捩", "手捩1", "手捩2", "手捩3", "手首", "ダミー"] + FINGERS)
REPLACED = (["全ての親", "操作中心", "センター", "グルーブ", "腰", "下半身", "上半身", "上半身1", "上半身2", "上半身3", "首", "頭", "腰キャンセル左", "腰キャンセル右"]
             + [c + n for c in "左右" for n in PER_SIDE])
# ZZZ bones with no equivalent name in ours -> the ZZZ bone that has one (same side)
TO = {"操作中心": "全ての親", "センター": "全ての親", "グルーブ": "全ての親", "腰": "下半身", "腰キャンセル左": "下半身", "腰キャンセル右": "下半身",
        "上半身3": "上半身2"}
TO_SIDE = {"足D": "足", "ひざD": "ひざ", "足首D": "足首", "足先EX": "つま先", "足IK親": "全ての親", "足ＩＫ": "足首", "つま先ＩＫ": "つま先", "肩P": "肩",
             "肩C": "肩", "腕捩": "腕", "腕捩1": "腕", "腕捩2": "腕", "腕捩3": "腕", "手捩": "ひじ", "手捩1": "ひじ", "手捩2": "ひじ", "手捩3": "ひじ",
             "ダミー": "手首"}
# control bone and deforming bone: a single one for us
SAME = {"足": "足D", "ひざ": "ひざD", "足首": "足首D", "つま先": "足先EX"}


def equivalent(name, of_mmd):
    """Name of the house bone that stands for the ZZZ bone `name` (parents, rigid bodies, weight fallback)."""
    if name in of_mmd:
        return of_mmd[name]
    if name in TO:
        return equivalent(TO[name], of_mmd)
    if name[0] in "左右" and name[1:] in TO_SIDE:
        v = TO_SIDE[name[1:]]
        return equivalent(v if v == "全ての親" else name[0] + v, of_mmd)
    raise KeyError(name)


def canonical(name):
    return name[0] + SAME[name[1:]] if name[0] in "左右" and name[1:] in SAME else name


def dense(I, W, columns, n_collar):
    """Weights (n, 4) on bone indices -> array (n, n_col); columns[bone] = column or -1."""
    out = np.zeros((len(I), n_collar))
    for k in range(I.shape[1]):
        c = np.where(I[:, k] >= 0, columns[np.maximum(I[:, k], 0)], -1)
        ok = (c >= 0) & (W[:, k] > 0)
        np.add.at(out, (np.flatnonzero(ok), c[ok]), W[ok, k])
    return out


def translation(A, B, perimeters=300):
    """M >= 0 that brings A M closest to B (least squares, multiplicative updates), rows summing to 1."""
    AtB, AtA = A.T @ B, A.T @ A
    M = AtB / np.maximum(AtB.sum(1, keepdims=True), 1e-12)
    for _ in range(perimeters):
        M *= AtB / np.maximum(AtA @ M, 1e-12)
    s = M.sum(1, keepdims=True)
    return M / np.maximum(s, 1e-12), s[:, 0] > 1e-6


def reference(own_rig, log=print):
    """The skin of the rebuilt body: positions, ZZZ weights (on the replaced bones) and house weights, at the same vertices."""
    import measurements as ME
    g = ME.load()
    D = np.load(os.path.join(own_rig, "weights.npz"))
    sq = SQ.read(os.path.join(own_rig, "skeleton.json"))
    assert list(D["bones"]) == sq.names and len(D["weights"]) == len(g["P"]), "weights.npz does not match shape.pkl: rebuild the house skeleton"
    Z = PMX.read(os.path.join(os.path.dirname(ME.SHAPE), "body.pmx"))
    t = PMX.to_arrays(Z)
    skin = np.unique(t["faces"][t["mat_face"] == [m["name"] for m in Z["materials"]].index("肌")])
    Q = t["pos"][skin] * CM
    to = np.zeros(len(g["P"]), int)
    for s in range(0, len(to), 500):
        to[s:s + 500] = skin[np.linalg.norm(g["P"][s:s + 500, None] - Q[None], axis=2).argmin(1)]
    return g["P"], Z, t["bones"][to], t["weights"][to], D["weights"].astype(float), sq


def convert(P, own_rig=OWN_RIG, log=print):
    """Returns the PMX `P` (dictionary from pmx.lire) on the house skeleton. P is not modified."""
    Q, Z, zI, zW, Bm, sq = reference(own_rig, log)
    of_mmd = dict(sq.of_mmd)
    names = [o["name"] for o in P["bones"]]
    replaced = [n in REPLACED for n in names]
    cols = sorted({canonical(n) for n in REPLACED})
    collar = {n: k for k, n in enumerate(cols)}
    R, nm = len(cols), len(sq.os)

    # the translation M, learnt on the skin
    zcol = np.array([collar.get(canonical(o["name"]), -1) if o["name"] in REPLACED else -1 for o in Z["bones"]])
    A = dense(zI, zW, zcol, R)
    sa = A.sum(1, keepdims=True)
    A = A / np.maximum(sa, 1e-9)
    M, known = translation(A[sa[:, 0] > 0.5], Bm[sa[:, 0] > 0.5])
    for k, n in enumerate(cols):
        if not known[k]:                      # bone the skin does not know (control, IK): its equivalent
            M[k] = 0
            M[k, sq.i[equivalent(n, of_mmd)]] = 1
    log("  translation ZZZ bones -> house bones: %d bones learnt on the skin, %d by equivalence; mean gap on the skin %.3f"
            % (int(known.sum()), int((~known).sum()), float(np.abs(A @ M - Bm).sum(1).mean())))

    # the character's vertices
    t = PMX.to_arrays(P)
    X, I, W = t["pos"] * CM, t["bones"], t["weights"].copy()
    W[I < 0] = 0
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    pcol = np.array([collar[canonical(n)] if r else -1 for n, r in zip(names, replaced)])
    Zb = dense(I, W, pcol, R)
    sb = Zb.sum(1)
    Zn = Zb / np.maximum(sb, 1e-9)[:, None]

    # the skin point: close and similarly weighted
    a2 = WEIGHT_GAP ** 2 / 2.0               # two weights with nothing in common: squared gap 2
    Ys = np.concatenate([Q, np.sqrt(a2) * A], 1)
    yy = (Ys ** 2).sum(1)
    Wm = np.zeros((len(X), nm))
    part = np.zeros(len(X))
    for s in range(0, len(X), 2000):
        e = slice(s, s + 2000)
        Xs = np.concatenate([X[e], np.sqrt(a2) * Zn[e]], 1)
        j = (yy[None] - 2 * Xs @ Ys.T).argmin(1)
        d = np.linalg.norm(X[e] - Q[j], axis=1)
        gap = 0.5 * np.abs(Zn[e] - A[j]).sum(1)                          # 0 = weighted like the skin, 1 = nothing in common
        f = np.clip((FAR - d) / (FAR - NEAR), 0, 1) * np.clip(1 - 2 * gap, 0, 1)
        f[sa[j, 0] < 0.05] = 0
        w = Zn[e] @ M + f[:, None] * (Bm[j] - A[j] @ M)
        w = np.clip(w, 0, None)
        Wm[e] = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
        part[e] = f
    Wm *= sb[:, None]
    log("  %d vertices: %d on the skin itself, %d held by the skin, %d by their translated weights alone, %d without body bones"
            % (len(X), int((part > 0.999).sum()), int(((part > 0) & (part <= 0.999)).sum()), int(((part == 0) & (sb > 0)).sum()), int((sb == 0).sum())))

    # the bones: ours, then the kept ones
    kept = [k for k, r in enumerate(replaced) if not r]
    new = np.zeros(len(names), int)
    for k, n in enumerate(names):
        if replaced[k]:
            new[k] = sq.i[equivalent(n, of_mmd)]
    for j, k in enumerate(kept):
        new[k] = nm + j

    def ref(i):
        return int(new[i]) if i is not None and i >= 0 else i
    model = next(o for o in P["bones"] if o["name"] == "頭")
    bones = []
    for k, o in enumerate(sq.os):
        mobile = 0x04 if o["name"] in ("root", "pelvis") else 0
        bones.append(dict(model, name=o["name"], name_en=o["role"] or o["name"], pos=tuple(float(x) / CM for x in o["pos"]),
                        parent=int(sq.parents[k]), layer=0, flags=0x1A | mobile, tail_bone=None, tail_offset=(0.0, 0.0, 0.0), addition=None,
                        fixed_axis=None, local_axes=None, external_parent=None, ik=None, after_physics=False))
    for k in kept:
        o = dict(P["bones"][k])
        o["parent"] = ref(o["parent"])
        if o.get("tail_bone") is not None:
            o["tail_bone"] = ref(o["tail_bone"])
        if o.get("addition"):
            o["addition"] = dict(o["addition"], source=ref(o["addition"]["source"]))
        if o.get("ik"):
            o["ik"] = dict(o["ik"], target=ref(o["ik"]["target"]), links=[dict(li, bones=ref(li["bones"])) for li in o["ik"]["links"]])
        bones.append(o)

    # four bones per vertex, without breaks (like weights.four): our bones and the kept ones together
    Ig = np.where((I >= 0) & ~np.array(replaced)[np.maximum(I, 0)], new[np.maximum(I, 0)], -1)
    Wg = np.where(Ig >= 0, W, 0.0)
    all_ = np.concatenate([Wm, Wg], 1)
    indices = np.concatenate([np.tile(np.arange(nm), (len(X), 1)), Ig], 1)
    tri = -np.sort(-all_, axis=1)
    all_ = np.clip(all_ - tri[:, 4:5], 0, None)
    o4 = np.argsort(-all_, axis=1)[:, :4]
    pw = np.take_along_axis(all_, o4, 1)
    pi = np.take_along_axis(indices, o4, 1)
    pw /= np.maximum(pw.sum(1, keepdims=True), 1e-12)
    pi[pw == 0] = 0
    log("  four bones per vertex: the removed fifth weight is at worst %.2f" % float(tri[:, 4].max()))

    n = len(X)
    vert = dict(P["vertices"], type=[2] * n, bones=[tuple(int(x) for x in r) for r in pi], weights=[tuple(float(x) for x in r) for r in pw],
               sdef=[None] * n)
    morphs = []
    for m in P["morphs"]:
        if m["type"] == 2:                     # bone morph: only the kept bones are kept
            m = dict(m, offsets=[(ref(d[0]),) + tuple(d[1:]) for d in m["offsets"] if not replaced[d[0]]])
        morphs.append(m)
    frames = []
    for c in P["frames"]:
        seen, el = set(), []
        for typ, i in c["elements"]:
            e = (typ, ref(i) if typ == 0 else i)
            if e not in seen:
                seen.add(e)
                el.append(e)
        frames.append(dict(c, elements=el))
    cited = {i for c in frames for typ, i in c["elements"] if typ == 0}
    model = frames[-1] if frames else {"name_en": "", "special": 0, "elements": []}     # a model without any display frame
    frames.append(dict(model, name="skeleton", name_en="skeleton", special=0, elements=[(0, k) for k in range(nm) if k not in cited]))
    rigid_bodies = [dict(r, bones=ref(r["bones"])) for r in P["rigid_bodies"]]
    log("  bones: %d from ZZZ replaced by our %d, %d kept (hair, eyes, chest, outfit)" % (sum(replaced), nm, len(kept)))
    return dict(P, vertices=vert, bones=bones, morphs=morphs, frames=frames, rigid_bodies=rigid_bodies,
                comment=(P.get("comment") or "") + "\nhouse skeleton and weights; twist in skeleton.json")


def connect(input_, output, own_rig=OWN_RIG, log=print):
    pmx_write.write(convert(PMX.read(input_), own_rig, log), output)
    log("written", output)


if __name__ == "__main__":
    a = sys.argv[1:]
    connect(a[0], a[1], a[a.index("--own-rig") + 1] if "--own-rig" in a else OWN_RIG)

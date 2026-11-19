"""Expressions of the cached heads: a few faces composed from the model's MMD morphs, for Chimera's view.

    python3 -I tools/workshop/expressions.py Maelys Jane      # or without a name: all heads of the cache

Next to `head.glb`, `expressions.json` (name, start and number of affected vertices of each expression) and
`expressions.bin`: per expression, the indices of the affected vertices (uint32) then their offsets (float32 × 3,
glTF frame, metres). The view adds them to the rest positions; the house render reads its maps at the rest
position, so the drawn shadow, the blush of the cheeks and the lips stay attached to the skin when the face moves.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import glb as GLB  # noqa: E402

import pmx as PMX  # noqa: E402  (via the sys.path set by common)

# displayed name (an identifier, also used in the games' special cases) -> MMD morphs and their weights (those the model lacks are left out)
EXPRESSIONS = {
    "Smile": {"口角上げ": 1.0, "下眼上": 0.5, "にこり": 0.6},
    "Laugh": {"笑い": 1.0, "ワ": 0.9, "にこり": 1.0},
    "Talking": {"あ": 0.75},
    "Surprise": {"びっくり": 1.0, "お": 0.6, "上": 0.7},
    "Anger": {"怒り目": 1.0, "怒り": 1.0, "口角下げ": 0.8},
    "Sad": {"悲しむ": 1.0, "困る": 1.0, "口角下げ": 0.6},
    "Eyes closed": {"まばたき": 1.0},
    # eyes open, iris shrunk, unsettling smile (clenched teeth shown, mouth stretched up to raised corners,
    # barely parted); the view lights up the gaze and shades the top of the face (REGARD)
    "Yandere": {"びっくり": 0.8, "@iris": 0.66, "い": 1.0, "あ": 0.3, "口角上げ": 1.0, "口横広げ": 1.0, "にやり": 1.0, "@teeth": 1.25},
}
# strength per head: replaces the named weights of the common expression (the eyes do not have the same shape);
# "expressions" special cases of the characters, in the description of their game (tools/games)
PER_HEAD = C.games.case_("expressions")
# render settings carried by the expression: glow = iris that lights up (even in shadow), shade = dark band over the eyes
GAZE = {"Yandere": {"glow": 1.0, "shade": 0.75}}


def iris_reduced(pos, prims, scale):
    """Offsets that pull each iris (material 目, one per side of x) towards its centre, to scale of its size."""
    offset = np.zeros_like(pos)
    for side in (1, -1):
        idx = np.unique(np.concatenate([p["indices"] for p in prims if p["name"] == "目"] or [np.zeros(0, np.int64)]))
        idx = idx[(pos[idx, 0] * side) >= 0]
        if not len(idx):
            continue
        c = (pos[idx].min(0) + pos[idx].max(0)) / 2
        offset[idx] = (c - pos[idx]) * (1 - scale)
    return offset


def _rows(p, tri):
    """The two rows of teeth: connected components of the triangles (vertices at the same place merged), the
    higher one first."""
    per = list(range(len(p)))

    def f(x):
        while per[x] != x:
            per[x] = per[per[x]]
            x = per[x]
        return x
    for a, b, c in tri:
        per[f(b)] = f(a)
        per[f(c)] = f(a)
    the_key = {}
    for v in np.unique(tri):
        k = tuple(np.round(p[v], 5))
        if k in the_key:
            per[f(v)] = f(the_key[k])
        else:
            the_key[k] = v
    g = {}
    for v in np.unique(tri):
        g.setdefault(f(v), []).append(v)
    return sorted((np.array(v) for v in g.values()), key=lambda v: -p[v, 1].mean())


def teeth_tight(p, prims, width):
    """Offsets that clench the teeth (material 齿) at their position p in the expression: the open mouth lowers
    the jaw; the two rows close in until, at the front, their edges overlap,
    and both widen by width around their middle to fill the stretched mouth from one corner to the other."""
    offset = np.zeros_like(p)
    tri = np.concatenate([q["indices"] for q in prims if q["name"] == "齿"] or [np.zeros(0, np.int64)]).reshape(-1, 3)
    r = _rows(p, tri)
    if len(r) != 2:
        return offset
    top, bottom = r
    before = lambda v: v[p[v, 2] > p[v, 2].max() - 0.004]   # the front (z towards the face), where the edge is seen
    empty = p[before(top), 1].min() - p[before(bottom), 1].max()
    closed = empty + 0.0016                                  # a little overlap: no gap between the rows
    offset[bottom, 1] += closed * 0.6                             # shared, otherwise the bottom of the mouth empties under the teeth
    offset[top, 1] -= closed * 0.4
    joint = p[before(top), 1].min() - closed * 0.4 + 0.0008
    for v in (top, bottom):                                  # each row stretches from the joint: the opening
        offset[v, 1] += (p[v, 1] + offset[v, 1] - joint) * 0.35  # of the mouth is taller than both together
    idx = np.concatenate([top, bottom])
    cx = (p[idx, 0].min() + p[idx, 0].max()) / 2
    offset[idx, 0] += (p[idx, 0] - cx) * (width - 1)
    return offset


def correspondence(pos_glb, pos_pmx):
    """Model vertex behind each GLB vertex. glb_head keeps the model's vertices in their order:
    we step through the model up to the next vertex at the same place."""
    target = (np.asarray(pos_pmx, np.float32) * 0.08 * np.array([1, 1, -1], np.float32)).astype(np.float32)
    corr, j = np.full(len(pos_glb), -1, np.int64), 0
    for i, p in enumerate(pos_glb):
        while j < len(target) and np.abs(target[j] - p).max() > 2e-6:
            j += 1
        if j == len(target):
            raise RuntimeError("GLB vertex %d not found in the model" % i)
        corr[i] = j
        j += 1
    return corr


def primitives(doc, data):
    """Material name and vertex indices of each primitive of the GLB."""
    out = []
    for p in doc["meshes"][0]["primitives"]:
        a = doc["accessors"][p["indices"]]
        v = doc["bufferViews"][a["bufferView"]]
        typ = {5121: "<u1", 5123: "<u2", 5125: "<u4"}[a["componentType"]]
        ind = np.frombuffer(data, typ, a["count"], v.get("byteOffset", 0) + a.get("byteOffset", 0)).astype(np.int64)
        out.append({"name": doc["materials"][p["material"]]["name"], "indices": ind})
    return out


def make(ident):
    d = os.path.join(C.CACHE, "heads", ident)
    meta = C.read_json(os.path.join(d, "meta.json"))
    P = PMX.read(C.absolute(meta["model"]))
    doc, data = GLB.read(os.path.join(d, "head.glb"))
    a = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    pos = np.frombuffer(data, "<f4", a["count"] * 3, doc["bufferViews"][a["bufferView"]].get("byteOffset", 0)).reshape(-1, 3)
    corr = correspondence(pos, P["vertices"]["pos"])
    to = np.full(len(P["vertices"]["pos"]), -1, np.int64)
    to[corr] = np.arange(len(corr))
    morphs = {m["name"]: m for m in P["morphs"] if m["type"] == 1 and m["offsets"]}
    prims = primitives(doc, data)
    items, blocks, start = [], [], 0
    for name, parts in EXPRESSIONS.items():
        parts = {**parts, **PER_HEAD.get(ident, {}).get(name, {})}   # the head changes only what it names
        offset = np.zeros((len(pos), 3), np.float32)
        for mmd, weight in parts.items():
            if mmd == "@teeth":
                continue                    # after the morphs: depends on the mouth they made
            if mmd == "@iris":
                offset += iris_reduced(pos, prims, weight)
                continue
            if mmd not in morphs:
                continue
            idx = np.array([i for i, _ in morphs[mmd]["offsets"]], np.int64)
            val = np.array([v[:3] for _, v in morphs[mmd]["offsets"]], np.float32) * 0.08 * np.array([1, 1, -1], np.float32)
            g = to[idx]
            np.add.at(offset, g[g >= 0], val[g >= 0] * weight)
        if "@teeth" in parts:
            offset += teeth_tight(pos + offset, prims, parts["@teeth"])
        touched = np.nonzero(np.abs(offset).max(1) > 1e-7)[0].astype("<u4")
        if not len(touched):
            continue
        blocks += [touched.tobytes(), offset[touched].astype("<f4").tobytes()]
        items.append({"name": name, "start": start, "n": int(len(touched)), **({"gaze": GAZE[name]} if name in GAZE else {})})
        start += len(touched) * 16
    with open(os.path.join(d, "expressions.bin"), "wb") as f:
        f.write(b"".join(blocks))
    C.write_json(os.path.join(d, "expressions.json"), {"vertices": int(len(pos)), "expressions": items})
    print(ident, ", ".join("%s %d" % (e["name"], e["n"]) for e in items), flush=True)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(n for n in os.listdir(os.path.join(C.CACHE, "heads"))
                                  if os.path.isfile(os.path.join(C.CACHE, "heads", n, "head.glb")))
    for n in names:
        try:
            make(n)
        except Exception as e:   # a head without a model (base head) or a broken one does not stop the others
            print(n, "skipped:", e, flush=True)

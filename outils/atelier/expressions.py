"""Expressions of the cached heads: a few faces composed from the model's MMD morphs, for Chimera's view.

    python3 -I outils/atelier/expressions.py Maelys Jane      # or without a name: all heads of the cache

Next to `tete.glb`, `expressions.json` (name, start and number of affected vertices of each expression) and
`expressions.bin`: per expression, the indices of the affected vertices (uint32) then their offsets (float32 × 3,
glTF frame, metres). The view adds them to the rest positions; the house render reads its maps at the rest
position, so the drawn shadow, the blush of the cheeks and the lips stay attached to the skin when the face moves.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun as C  # noqa: E402
import glb as GLB  # noqa: E402

import pmx as PMX  # noqa: E402  (via the sys.path set by commun)

# displayed name (an identifier, also used in the games' special cases) -> MMD morphs and their weights (those the model lacks are left out)
EXPRESSIONS = {
    "Sourire": {"口角上げ": 1.0, "下眼上": 0.5, "にこり": 0.6},
    "Rire": {"笑い": 1.0, "ワ": 0.9, "にこり": 1.0},
    "Parle": {"あ": 0.75},
    "Surprise": {"びっくり": 1.0, "お": 0.6, "上": 0.7},
    "Colère": {"怒り目": 1.0, "怒り": 1.0, "口角下げ": 0.8},
    "Triste": {"悲しむ": 1.0, "困る": 1.0, "口角下げ": 0.6},
    "Yeux fermés": {"まばたき": 1.0},
    # eyes open, iris shrunk, unsettling smile (clenched teeth shown, mouth stretched up to raised corners,
    # barely parted); the view lights up the gaze and shades the top of the face (REGARD)
    "Yandere": {"びっくり": 0.8, "@iris": 0.66, "い": 1.0, "あ": 0.3, "口角上げ": 1.0, "口横広げ": 1.0, "にやり": 1.0, "@dents": 1.25},
}
# strength per head: replaces the named weights of the common expression (the eyes do not have the same shape);
# "expressions" special cases of the characters, in the description of their game (outils/jeux)
PAR_TETE = C.jeux.cas("expressions")
# render settings carried by the expression: lueur = iris that lights up (even in shadow), ombre = dark band over the eyes
REGARD = {"Yandere": {"lueur": 1.0, "ombre": 0.75}}


def iris_reduit(pos, prims, echelle):
    """Offsets that pull each iris (material 目, one per side of x) towards its centre, to echelle of its size."""
    dec = np.zeros_like(pos)
    for cote in (1, -1):
        idx = np.unique(np.concatenate([p["indices"] for p in prims if p["nom"] == "目"] or [np.zeros(0, np.int64)]))
        idx = idx[(pos[idx, 0] * cote) >= 0]
        if not len(idx):
            continue
        c = (pos[idx].min(0) + pos[idx].max(0)) / 2
        dec[idx] = (c - pos[idx]) * (1 - echelle)
    return dec


def _rangees(p, tri):
    """The two rows of teeth: connected components of the triangles (vertices at the same place merged), the
    higher one first."""
    par = list(range(len(p)))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b, c in tri:
        par[f(b)] = f(a)
        par[f(c)] = f(a)
    cle = {}
    for v in np.unique(tri):
        k = tuple(np.round(p[v], 5))
        if k in cle:
            par[f(v)] = f(cle[k])
        else:
            cle[k] = v
    g = {}
    for v in np.unique(tri):
        g.setdefault(f(v), []).append(v)
    return sorted((np.array(v) for v in g.values()), key=lambda v: -p[v, 1].mean())


def dents_serrees(p, prims, largeur):
    """Offsets that clench the teeth (material 齿) at their position p in the expression: the open mouth lowers
    the jaw; the two rows close in until, at the front, their edges overlap,
    and both widen by largeur around their middle to fill the stretched mouth from one corner to the other."""
    dec = np.zeros_like(p)
    tri = np.concatenate([q["indices"] for q in prims if q["nom"] == "齿"] or [np.zeros(0, np.int64)]).reshape(-1, 3)
    r = _rangees(p, tri)
    if len(r) != 2:
        return dec
    haut, bas = r
    avant = lambda v: v[p[v, 2] > p[v, 2].max() - 0.004]   # the front (z towards the face), where the edge is seen
    vide = p[avant(haut), 1].min() - p[avant(bas), 1].max()
    ferme = vide + 0.0016                                  # a little overlap: no gap between the rows
    dec[bas, 1] += ferme * 0.6                             # shared, otherwise the bottom of the mouth empties under the teeth
    dec[haut, 1] -= ferme * 0.4
    joint = p[avant(haut), 1].min() - ferme * 0.4 + 0.0008
    for v in (haut, bas):                                  # each row stretches from the joint: the opening
        dec[v, 1] += (p[v, 1] + dec[v, 1] - joint) * 0.35  # of the mouth is taller than both together
    idx = np.concatenate([haut, bas])
    cx = (p[idx, 0].min() + p[idx, 0].max()) / 2
    dec[idx, 0] += (p[idx, 0] - cx) * (largeur - 1)
    return dec


def correspondance(pos_glb, pos_pmx):
    """Model vertex behind each GLB vertex. glb_tete keeps the model's vertices in their order:
    we step through the model up to the next vertex at the same place."""
    cible = (np.asarray(pos_pmx, np.float32) * 0.08 * np.array([1, 1, -1], np.float32)).astype(np.float32)
    corr, j = np.full(len(pos_glb), -1, np.int64), 0
    for i, p in enumerate(pos_glb):
        while j < len(cible) and np.abs(cible[j] - p).max() > 2e-6:
            j += 1
        if j == len(cible):
            raise RuntimeError("GLB vertex %d not found in the model" % i)
        corr[i] = j
        j += 1
    return corr


def primitives(doc, donnees):
    """Material name and vertex indices of each primitive of the GLB."""
    out = []
    for p in doc["meshes"][0]["primitives"]:
        a = doc["accessors"][p["indices"]]
        v = doc["bufferViews"][a["bufferView"]]
        typ = {5121: "<u1", 5123: "<u2", 5125: "<u4"}[a["componentType"]]
        ind = np.frombuffer(donnees, typ, a["count"], v.get("byteOffset", 0) + a.get("byteOffset", 0)).astype(np.int64)
        out.append({"nom": doc["materials"][p["material"]]["name"], "indices": ind})
    return out


def faire(ident):
    d = os.path.join(C.CACHE, "tetes", ident)
    meta = C.lire_json(os.path.join(d, "meta.json"))
    P = PMX.lire(C.absolu(meta["modele"]))
    doc, donnees = GLB.lire(os.path.join(d, "tete.glb"))
    a = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    pos = np.frombuffer(donnees, "<f4", a["count"] * 3, doc["bufferViews"][a["bufferView"]].get("byteOffset", 0)).reshape(-1, 3)
    corr = correspondance(pos, P["sommets"]["pos"])
    vers = np.full(len(P["sommets"]["pos"]), -1, np.int64)
    vers[corr] = np.arange(len(corr))
    morphes = {m["nom"]: m for m in P["morphes"] if m["type"] == 1 and m["decalages"]}
    prims = primitives(doc, donnees)
    liste, blocs, debut = [], [], 0
    for nom, parts in EXPRESSIONS.items():
        parts = {**parts, **PAR_TETE.get(ident, {}).get(nom, {})}   # the head changes only what it names
        dec = np.zeros((len(pos), 3), np.float32)
        for mmd, poids in parts.items():
            if mmd == "@dents":
                continue                    # after the morphs: depends on the mouth they made
            if mmd == "@iris":
                dec += iris_reduit(pos, prims, poids)
                continue
            if mmd not in morphes:
                continue
            idx = np.array([i for i, _ in morphes[mmd]["decalages"]], np.int64)
            val = np.array([v[:3] for _, v in morphes[mmd]["decalages"]], np.float32) * 0.08 * np.array([1, 1, -1], np.float32)
            g = vers[idx]
            np.add.at(dec, g[g >= 0], val[g >= 0] * poids)
        if "@dents" in parts:
            dec += dents_serrees(pos + dec, prims, parts["@dents"])
        touches = np.nonzero(np.abs(dec).max(1) > 1e-7)[0].astype("<u4")
        if not len(touches):
            continue
        blocs += [touches.tobytes(), dec[touches].astype("<f4").tobytes()]
        liste.append({"nom": nom, "debut": debut, "n": int(len(touches)), **({"regard": REGARD[nom]} if nom in REGARD else {})})
        debut += len(touches) * 16
    with open(os.path.join(d, "expressions.bin"), "wb") as f:
        f.write(b"".join(blocs))
    C.ecrire_json(os.path.join(d, "expressions.json"), {"sommets": int(len(pos)), "expressions": liste})
    print(ident, ", ".join("%s %d" % (e["nom"], e["n"]) for e in liste), flush=True)


if __name__ == "__main__":
    noms = sys.argv[1:] or sorted(n for n in os.listdir(os.path.join(C.CACHE, "tetes"))
                                  if os.path.isfile(os.path.join(C.CACHE, "tetes", n, "tete.glb")))
    for n in noms:
        try:
            faire(n)
        except Exception as e:   # a head without a model (base head) or a broken one does not stop the others
            print(n, "skipped:", e, flush=True)

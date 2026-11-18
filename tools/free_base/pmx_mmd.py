"""The free body written as PMX, on a skeleton with the names of the MMD convention ("semi-standard" bones).

Chimera carries parts from one body to another through these names (下半身, 上半身, 腕, 腕捩, ひじ, 足D...): they are
a public convention of the format, not the property of a game. The positions all come from our skeleton
(body.py); the weights are our weights, translated: one of our bones -> one bone of the convention, and the twist of
our steps spread over those of the convention (0, 1/4, 1/2, 3/4, 1 of the roll).

    python3 -I pmx_mmd.py <folder>      reads forme.pkl, maison/squelette.json, maison/poids.npz; writes corps.pmx
"""
import os
import pickle
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI] + [os.path.join(ICI, "..", d) for d in ("custom_body", "rebuilt_body", "atlas", "head", "rebuilt_head")]
import pmx as PMX             # noqa: E402
import pmx_write as pmx_ecrire             # noqa: E402
import png                    # noqa: E402
import weights as PO            # noqa: E402
import rebuild as RF          # noqa: E402  (corps_refait: cartes_uv)
import skeleton as SQ        # noqa: E402
import geom as G              # noqa: E402
import free_head as TE           # noqa: E402

CM = 8.0                      # cm per MMD unit
PEAU = (236, 204, 188)        # flat skin colour, until there is a painted skin
PALIERS = (0.0, 0.25, 0.5, 0.75, 1.0)
BANDE = 0.8                   # share of the texture width left to the body maps; the rest goes to the head
DOIGTS = (("pouce", "親指", "０１２"), ("index", "人指", "１２３"), ("majeur", "中指", "１２３"),
          ("annulaire", "薬指", "１２３"), ("petit_doigt", "小指", "１２３"))


def squelette_mmd(sq):
    """List of bones {nom, parent, pos (cm), ajout: (source, factor) or None, couche}."""
    J = {n: sq.tetes[k] for n, k in sq.i.items()}
    os_ = []

    def os(nom, parent, pos, ajout=None, couche=0):
        os_.append({"nom": nom, "parent": parent, "pos": np.asarray(pos, float), "ajout": ajout, "couche": couche})

    hanche = J["cuisse_G"] * [0, 1, 1]
    os("全ての親", None, [0, 0, 0])
    os("操作中心", None, [0, 0, 0])
    os("センター", "全ての親", [0, 0.6 * hanche[1], 0])
    os("グルーブ", "センター", [0, 0.6 * hanche[1] + 1.0, 0])
    os("腰", "グルーブ", 0.5 * (J["bassin"] + hanche))
    os("下半身", "腰", J["bassin"])
    os("上半身", "腰", J["colonne"])
    os("上半身1", "上半身", J["poitrine"])
    os("上半身2", "上半身1", J["haut_poitrine"])
    os("首", "上半身2", J["cou"])
    os("頭", "首", J["tete"])
    for c, m in (("G", "左"), ("D", "右")):
        def j(n):
            return J[n + "_" + c]
        os("腰キャンセル" + m, "下半身", j("cuisse"), ("腰", -1.0))
        os(m + "足", "腰キャンセル" + m, j("cuisse"))
        os(m + "ひざ", m + "足", j("jambe"))
        os(m + "足首", m + "ひざ", j("pied"))
        os(m + "つま先", m + "足首", j("orteils"))
        os(m + "足D", "腰キャンセル" + m, j("cuisse"), (m + "足", 1.0), 1)
        os(m + "ひざD", m + "足D", j("jambe"), (m + "ひざ", 1.0), 1)
        os(m + "足首D", m + "ひざD", j("pied"), (m + "足首", 1.0), 1)
        os(m + "足先EX", m + "足首D", j("orteils"), None, 1)
        os(m + "足IK親", "全ての親", j("pied") * [1, 0, 1])
        os(m + "足ＩＫ", m + "足IK親", j("pied"))
        os(m + "つま先ＩＫ", m + "足ＩＫ", j("orteils"))
        os(m + "肩P", "上半身2", j("epaule"))
        os(m + "肩", m + "肩P", j("epaule"))
        os(m + "肩C", m + "肩", j("bras"), (m + "肩P", -1.0))
        os(m + "腕", m + "肩C", j("bras"))
        a, b, p = j("bras"), j("avant_bras"), j("main")
        os(m + "腕捩", m + "腕", 0.5 * (a + b))
        for k in (1, 2, 3):
            os(m + "腕捩%d" % k, m + "腕", a + 0.25 * k * (b - a), (m + "腕捩", 0.25 * k))
        os(m + "ひじ", m + "腕捩", b)
        os(m + "手捩", m + "ひじ", 0.5 * (b + p))
        for k in (1, 2, 3):
            os(m + "手捩%d" % k, m + "ひじ", b + 0.25 * k * (p - b), (m + "手捩", 0.25 * k))
        os(m + "手首", m + "手捩", p)
        os(m + "ダミー", m + "手首", 0.5 * (p + j("majeur1")))
        for d, nom, chiffres in DOIGTS:
            for k in range(3):
                os(m + nom + chiffres[k], m + "手首" if k == 0 else m + nom + chiffres[k - 1], j("%s%d" % (d, k + 1)))
    return os_


def traduire(w, sq, noms):
    """Weights on our skeleton -> weights on the bones of the convention."""
    ou = {n: k for k, n in enumerate(noms)}
    out = np.zeros((len(w), len(noms)))
    fait = np.zeros(w.shape[1], bool)

    def torsion(os_, parts, cibles):
        """A group of stepped bones: the roll share of each vertex, spread over the steps of the convention."""
        k = [sq.i[o] for o in os_]
        W = w[:, k].sum(1)
        tau = np.clip((w[:, k] * parts).sum(1) / np.maximum(W, 1e-12), 0, 1) * (len(PALIERS) - 1)
        bas = np.minimum(tau.astype(int), len(PALIERS) - 2)
        f = tau - bas
        np.add.at(out, (np.arange(len(w)), np.array([ou[c] for c in cibles])[bas]), W * (1 - f))
        np.add.at(out, (np.arange(len(w)), np.array([ou[c] for c in cibles])[bas + 1]), W * f)
        fait[k] = True

    for c, m in (("G", "左"), ("D", "右")):
        torsion(["bras_torsion1_" + c, "bras_torsion2_" + c, "bras_torsion3_" + c, "bras_" + c], (0, 1 / 3, 2 / 3, 1),
                [m + "腕", m + "腕捩1", m + "腕捩2", m + "腕捩3", m + "腕捩"])
        torsion(["avant_bras_" + c, "avant_bras_torsion1_" + c, "avant_bras_torsion2_" + c, "avant_bras_torsion3_" + c],
                (0, 1 / 3, 2 / 3, 1), [m + "ひじ", m + "手捩1", m + "手捩2", m + "手捩3", m + "手捩"])
        simples = {"epaule": "肩", "main": "手首", "cuisse": "足D", "cuisse_torsion1": "足D", "cuisse_torsion2": "足D",
                   "cuisse_torsion3": "足D", "jambe": "ひざD", "pied": "足首D", "orteils": "足先EX"}
        simples.update({"orteil%d" % k: "足先EX" for k in range(1, 6)})
        simples.update({"%s%d" % (d, k + 1): nom + ch[k] for d, nom, ch in DOIGTS for k in range(3)})
        for a, b in simples.items():
            out[:, ou[m + b]] += w[:, sq.i[a + "_" + c]]
            fait[sq.i[a + "_" + c]] = True
    for a, b in (("bassin", "下半身"), ("colonne", "上半身"), ("poitrine", "上半身1"), ("haut_poitrine", "上半身2"),
                 ("cou", "首"), ("tete", "頭"), ("racine", "全ての親")):
        out[:, ou[b]] += w[:, sq.i[a]]
        fait[sq.i[a]] = True
    assert fait.all(), [n for n, f in zip(sq.noms, fait) if not f]
    return out / out.sum(1, keepdims=True)


def ecrire(g, sq, w, chemin):
    oss = squelette_mmd(sq)
    noms = [o["nom"] for o in oss]
    ou = {n: k for k, n in enumerate(noms)}
    # one texture map per tube of the grid: the seam vertices are split
    F, origine, uv, _ = RF.cartes_uv(g, None)
    Pc, T = g["P"], PO.triangles(g["quads"])
    if "tete" in g:                                  # MakeHuman's head, after the grid vertices
        t = g["tete"]
        nc = len(Pc)
        w = np.concatenate([w, TE.poids(g, t, w, sq.i["tete"])])
        Pc = np.concatenate([Pc, t["P"]])
        T = np.concatenate([T, t["tris"] + nc, t["pont"]])
        premier = {}
        for k, v in enumerate(origine):
            premier.setdefault(int(v), k)
        vers = np.concatenate([[premier[v] for v in range(nc)], len(origine) + np.arange(len(t["P"]))])
        F = np.concatenate([F, vers[np.concatenate([t["tris"] + nc, t["pont"]])]])
        origine = np.concatenate([origine, nc + np.arange(len(t["P"]))])
        # head map: front view, in the band that the body maps leave on the right
        bo = t["P"].min(0), t["P"].max(0)
        uvt = (t["P"][:, :2] - bo[0][:2]) / (bo[1][:2] - bo[0][:2]).max()
        uv = np.concatenate([np.asarray(uv) * [BANDE, 1.0], np.c_[BANDE + 0.01 + uvt[:, 0] * (0.98 - BANDE), 0.01 + (1 - uvt[:, 1]) * (0.98 - BANDE)]])
    idx, pw = PO.quatre(traduire(w, sq, noms))
    idx = np.where(idx < 0, 0, idx)
    N = G.normales_sommets(Pc, T)[origine]
    P = Pc[origine]
    idx, pw = idx[origine], pw[origine]
    n = len(P)
    os_pmx = []
    for o in oss:
        f = PMX.OS_ROTATION | PMX.OS_VISIBLE | PMX.OS_ACTIF | (PMX.OS_AJOUT_ROT if o["ajout"] else 0)
        if o["nom"] in ("全ての親", "センター", "グルーブ") or "ＩＫ" in o["nom"] or "IK" in o["nom"]:
            f |= PMX.OS_DEPLACEMENT
        b = {"nom": o["nom"], "nom_en": o["nom"], "pos": tuple(float(x) / CM for x in o["pos"]),
             "parent": -1 if o["parent"] is None else ou[o["parent"]], "couche": o["couche"], "drapeaux": f,
             "queue_os": None, "queue_dec": (0.0, 0.0, 0.0)}
        if o["ajout"]:
            b["ajout"] = {"source": ou[o["ajout"][0]], "facteur": float(o["ajout"][1])}
        os_pmx.append(b)
    som = {"pos": [tuple(p) for p in (P / CM).tolist()], "nor": [tuple(p) for p in N.tolist()],
           "uv": [tuple(float(x) for x in u) for u in uv], "uv_add": [[] for _ in range(n)], "type": [2] * n,
           "os": [tuple(int(x) for x in r) for r in idx], "poids": [tuple(float(x) for x in r) for r in pw],
           "sdef": [None] * n, "contour": [1.0] * n}
    c = tuple(x / 255 for x in PEAU)
    mat = {"nom": "肌", "nom_en": "skin", "diffus": c + (1.0,), "speculaire": (0.0, 0.0, 0.0), "brillance": 5.0,
           "ambiant": tuple(0.5 * x for x in c), "drapeaux": 0x1E, "contour_couleur": (0.3, 0.2, 0.2, 1.0),
           "contour_taille": 0.6, "texture": 0, "sphere": -1, "sphere_mode": 0, "toon_partage": 1, "toon": 0,
           "memo": "", "n_indices": 3 * len(F)}
    png.ecrire(os.path.join(os.path.dirname(chemin), "peau.png"), np.full((8, 8, 3), PEAU, np.uint8))
    pmx_ecrire.ecrire({"version": 2.0, "encodage": "UTF-16", "nom": "corps libre", "nom_en": "free base body",
                       "commentaire": "Free base body: surface drawn from MakeHuman's mesh (CC0), skeleton and weights "
                                      "computed by Chimera.", "commentaire_en": "", "n_uv_add": 0, "sommets": som,
                       "faces": [tuple(int(x) for x in f) for f in F], "textures": ["peau.png"], "materiaux": [mat],
                       "os": os_pmx, "morphes": [], "cadres": [], "rigides": [], "joints": []}, chemin)
    print("wrote %s: %d vertices, %d faces, %d bones" % (chemin, n, len(F), len(oss)))


def main():
    d = sys.argv[1]
    with open(os.path.join(d, "forme.pkl"), "rb") as f:
        g = pickle.load(f)
    sq = SQ.lire(os.path.join(d, "maison", "squelette.json"))
    Z = np.load(os.path.join(d, "maison", "poids.npz"))
    assert list(Z["os"]) == sq.noms
    ecrire(g, sq, Z["poids"].astype(float), os.path.join(d, "corps.pmx"))


if __name__ == "__main__":
    main()

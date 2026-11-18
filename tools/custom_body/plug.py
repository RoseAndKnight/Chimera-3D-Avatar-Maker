"""Plug the house skeleton into a character assembled by Chimera (rebuilt body, head, hairstyle, outfit).

    python3 -I plug.py INPUT.pmx OUTPUT.pmx [--maison FOLDER of the house body]

The character arrives weighted on the ZZZ bones. The body bones of ZZZ (trunk, arms, hands, legs, feet, their
control, twist and IK bones) are replaced by ours; everything else is kept as is and
reattached to our bones: eyes, hair, chest, buttocks, skirts, accessories, their physics, the expressions.

The weights, vertex by vertex, for the share it had on the replaced bones:
 - the skin of the body takes our weights (poids.npz), exactly;
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

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ICI] + [os.path.join(ICI, "..", d) for d in ("rebuilt_body", "clothes", "atlas", "head")]
import pmx as PMX                               # noqa: E402
import pmx_write as pmx_ecrire                               # noqa: E402
import skeleton as SQ                          # noqa: E402

CM = 8.0
import measurements as _ME     # noqa: E402
MAISON = _ME.MAISON       # house skeleton of Chimera's body (CHIMERA_BODY), see measurements.py
PRES, LOIN = 2.0, 7.0       # cm: the skin counts in full up to PRES, not at all from LOIN
POIDS_ECART = 12.0          # cm: what an entirely different weight is worth in the choice of the skin point
DOIGTS = [d + k for d, ks in (("親指", "０１２"), ("人指", "１２３"), ("中指", "１２３"), ("薬指", "１２３"), ("小指", "１２３")) for k in ks]
PAR_COTE = (["足", "ひざ", "足首", "つま先", "足D", "ひざD", "足首D", "足先EX", "足IK親", "足ＩＫ", "つま先ＩＫ", "肩P", "肩", "肩C", "腕", "腕捩",
             "腕捩1", "腕捩2", "腕捩3", "ひじ", "手捩", "手捩1", "手捩2", "手捩3", "手首", "ダミー"] + DOIGTS)
REMPLACES = (["全ての親", "操作中心", "センター", "グルーブ", "腰", "下半身", "上半身", "上半身1", "上半身2", "上半身3", "首", "頭", "腰キャンセル左", "腰キャンセル右"]
             + [c + n for c in "左右" for n in PAR_COTE])
# ZZZ bones with no equivalent name in ours -> the ZZZ bone that has one (same side)
VERS = {"操作中心": "全ての親", "センター": "全ての親", "グルーブ": "全ての親", "腰": "下半身", "腰キャンセル左": "下半身", "腰キャンセル右": "下半身",
        "上半身3": "上半身2"}
VERS_COTE = {"足D": "足", "ひざD": "ひざ", "足首D": "足首", "足先EX": "つま先", "足IK親": "全ての親", "足ＩＫ": "足首", "つま先ＩＫ": "つま先", "肩P": "肩",
             "肩C": "肩", "腕捩": "腕", "腕捩1": "腕", "腕捩2": "腕", "腕捩3": "腕", "手捩": "ひじ", "手捩1": "ひじ", "手捩2": "ひじ", "手捩3": "ひじ",
             "ダミー": "手首"}
# control bone and deforming bone: a single one for us
MEMES = {"足": "足D", "ひざ": "ひざD", "足首": "足首D", "つま先": "足先EX"}


def equivalent(nom, de_mmd):
    """Name of the house bone that stands for the ZZZ bone `nom` (parents, rigid bodies, weight fallback)."""
    if nom in de_mmd:
        return de_mmd[nom]
    if nom in VERS:
        return equivalent(VERS[nom], de_mmd)
    if nom[0] in "左右" and nom[1:] in VERS_COTE:
        v = VERS_COTE[nom[1:]]
        return equivalent(v if v == "全ての親" else nom[0] + v, de_mmd)
    raise KeyError(nom)


def canon(nom):
    return nom[0] + MEMES[nom[1:]] if nom[0] in "左右" and nom[1:] in MEMES else nom


def dense(I, W, colonnes, n_col):
    """Weights (n, 4) on bone indices -> array (n, n_col); colonnes[bone] = column or -1."""
    out = np.zeros((len(I), n_col))
    for k in range(I.shape[1]):
        c = np.where(I[:, k] >= 0, colonnes[np.maximum(I[:, k], 0)], -1)
        ok = (c >= 0) & (W[:, k] > 0)
        np.add.at(out, (np.flatnonzero(ok), c[ok]), W[ok, k])
    return out


def traduction(A, B, tours=300):
    """M >= 0 that brings A M closest to B (least squares, multiplicative updates), rows summing to 1."""
    AtB, AtA = A.T @ B, A.T @ A
    M = AtB / np.maximum(AtB.sum(1, keepdims=True), 1e-12)
    for _ in range(tours):
        M *= AtB / np.maximum(AtA @ M, 1e-12)
    s = M.sum(1, keepdims=True)
    return M / np.maximum(s, 1e-12), s[:, 0] > 1e-6


def reference(maison, journal=print):
    """The skin of the rebuilt body: positions, ZZZ weights (on the replaced bones) and house weights, at the same vertices."""
    import measurements as ME
    g = ME.charger()
    D = np.load(os.path.join(maison, "poids.npz"))
    sq = SQ.lire(os.path.join(maison, "squelette.json"))
    assert list(D["os"]) == sq.noms and len(D["poids"]) == len(g["P"]), "poids.npz does not match forme.pkl: run faire.py again"
    Z = PMX.lire(os.path.join(os.path.dirname(ME.FORME), "corps.pmx"))
    t = PMX.vers_tableaux(Z)
    peau = np.unique(t["faces"][t["mat_face"] == [m["nom"] for m in Z["materiaux"]].index("肌")])
    Q = t["pos"][peau] * CM
    vers = np.zeros(len(g["P"]), int)
    for s in range(0, len(vers), 500):
        vers[s:s + 500] = peau[np.linalg.norm(g["P"][s:s + 500, None] - Q[None], axis=2).argmin(1)]
    return g["P"], Z, t["os"][vers], t["poids"][vers], D["poids"].astype(float), sq


def convertir(P, maison=MAISON, journal=print):
    """Returns the PMX `P` (dictionary from pmx.lire) on the house skeleton. P is not modified."""
    Q, Z, zI, zW, Bm, sq = reference(maison, journal)
    de_mmd = dict(sq.de_mmd)
    noms = [o["nom"] for o in P["os"]]
    rempl = [n in REMPLACES for n in noms]
    cols = sorted({canon(n) for n in REMPLACES})
    col = {n: k for k, n in enumerate(cols)}
    R, nm = len(cols), len(sq.os)

    # the translation M, learnt on the skin
    zcol = np.array([col.get(canon(o["nom"]), -1) if o["nom"] in REMPLACES else -1 for o in Z["os"]])
    A = dense(zI, zW, zcol, R)
    sa = A.sum(1, keepdims=True)
    A = A / np.maximum(sa, 1e-9)
    M, connu = traduction(A[sa[:, 0] > 0.5], Bm[sa[:, 0] > 0.5])
    for k, n in enumerate(cols):
        if not connu[k]:                      # bone the skin does not know (control, IK): its equivalent
            M[k] = 0
            M[k, sq.i[equivalent(n, de_mmd)]] = 1
    journal("  translation ZZZ bones -> house bones: %d bones learnt on the skin, %d by equivalence; mean gap on the skin %.3f"
            % (int(connu.sum()), int((~connu).sum()), float(np.abs(A @ M - Bm).sum(1).mean())))

    # the character's vertices
    t = PMX.vers_tableaux(P)
    X, I, W = t["pos"] * CM, t["os"], t["poids"].copy()
    W[I < 0] = 0
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    pcol = np.array([col[canon(n)] if r else -1 for n, r in zip(noms, rempl)])
    Zb = dense(I, W, pcol, R)
    sb = Zb.sum(1)
    Zn = Zb / np.maximum(sb, 1e-9)[:, None]

    # the skin point: close and similarly weighted
    a2 = POIDS_ECART ** 2 / 2.0               # two weights with nothing in common: squared gap 2
    Ys = np.concatenate([Q, np.sqrt(a2) * A], 1)
    yy = (Ys ** 2).sum(1)
    Wm = np.zeros((len(X), nm))
    part = np.zeros(len(X))
    for s in range(0, len(X), 2000):
        e = slice(s, s + 2000)
        Xs = np.concatenate([X[e], np.sqrt(a2) * Zn[e]], 1)
        j = (yy[None] - 2 * Xs @ Ys.T).argmin(1)
        d = np.linalg.norm(X[e] - Q[j], axis=1)
        ecart = 0.5 * np.abs(Zn[e] - A[j]).sum(1)                          # 0 = weighted like the skin, 1 = nothing in common
        f = np.clip((LOIN - d) / (LOIN - PRES), 0, 1) * np.clip(1 - 2 * ecart, 0, 1)
        f[sa[j, 0] < 0.05] = 0
        w = Zn[e] @ M + f[:, None] * (Bm[j] - A[j] @ M)
        w = np.clip(w, 0, None)
        Wm[e] = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
        part[e] = f
    Wm *= sb[:, None]
    journal("  %d vertices: %d on the skin itself, %d held by the skin, %d by their translated weights alone, %d without body bones"
            % (len(X), int((part > 0.999).sum()), int(((part > 0) & (part <= 0.999)).sum()), int(((part == 0) & (sb > 0)).sum()), int((sb == 0).sum())))

    # the bones: ours, then the kept ones
    gardes = [k for k, r in enumerate(rempl) if not r]
    nouveau = np.zeros(len(noms), int)
    for k, n in enumerate(noms):
        if rempl[k]:
            nouveau[k] = sq.i[equivalent(n, de_mmd)]
    for j, k in enumerate(gardes):
        nouveau[k] = nm + j

    def ref(i):
        return int(nouveau[i]) if i is not None and i >= 0 else i
    modele = next(o for o in P["os"] if o["nom"] == "頭")
    os_ = []
    for k, o in enumerate(sq.os):
        mobile = 0x04 if o["nom"] in ("racine", "bassin") else 0
        os_.append(dict(modele, nom=o["nom"], nom_en=o["role"] or o["nom"], pos=tuple(float(x) / CM for x in o["pos"]),
                        parent=int(sq.parents[k]), couche=0, drapeaux=0x1A | mobile, queue_os=None, queue_dec=(0.0, 0.0, 0.0), ajout=None,
                        axe_fixe=None, axes_locaux=None, parent_externe=None, ik=None, apres_physique=False))
    for k in gardes:
        o = dict(P["os"][k])
        o["parent"] = ref(o["parent"])
        if o.get("queue_os") is not None:
            o["queue_os"] = ref(o["queue_os"])
        if o.get("ajout"):
            o["ajout"] = dict(o["ajout"], source=ref(o["ajout"]["source"]))
        if o.get("ik"):
            o["ik"] = dict(o["ik"], cible=ref(o["ik"]["cible"]), liens=[dict(li, os=ref(li["os"])) for li in o["ik"]["liens"]])
        os_.append(o)

    # four bones per vertex, without breaks (like poids.quatre): our bones and the kept ones together
    Ig = np.where((I >= 0) & ~np.array(rempl)[np.maximum(I, 0)], nouveau[np.maximum(I, 0)], -1)
    Wg = np.where(Ig >= 0, W, 0.0)
    tout = np.concatenate([Wm, Wg], 1)
    indices = np.concatenate([np.tile(np.arange(nm), (len(X), 1)), Ig], 1)
    tri = -np.sort(-tout, axis=1)
    tout = np.clip(tout - tri[:, 4:5], 0, None)
    o4 = np.argsort(-tout, axis=1)[:, :4]
    pw = np.take_along_axis(tout, o4, 1)
    pi = np.take_along_axis(indices, o4, 1)
    pw /= np.maximum(pw.sum(1, keepdims=True), 1e-12)
    pi[pw == 0] = 0
    journal("  four bones per vertex: the removed fifth weight is at worst %.2f" % float(tri[:, 4].max()))

    n = len(X)
    som = dict(P["sommets"], type=[2] * n, os=[tuple(int(x) for x in r) for r in pi], poids=[tuple(float(x) for x in r) for r in pw],
               sdef=[None] * n)
    morphes = []
    for m in P["morphes"]:
        if m["type"] == 2:                     # bone morph: only the kept bones are kept
            m = dict(m, decalages=[(ref(d[0]),) + tuple(d[1:]) for d in m["decalages"] if not rempl[d[0]]])
        morphes.append(m)
    cadres = []
    for c in P["cadres"]:
        vus, el = set(), []
        for typ, i in c["elements"]:
            e = (typ, ref(i) if typ == 0 else i)
            if e not in vus:
                vus.add(e)
                el.append(e)
        cadres.append(dict(c, elements=el))
    cites = {i for c in cadres for typ, i in c["elements"] if typ == 0}
    modele = cadres[-1] if cadres else {"nom_en": "", "special": 0, "elements": []}     # a model without any display frame
    cadres.append(dict(modele, nom="squelette", nom_en="skeleton", special=0, elements=[(0, k) for k in range(nm) if k not in cites]))
    rigides = [dict(r, os=ref(r["os"])) for r in P["rigides"]]
    journal("  bones: %d from ZZZ replaced by our %d, %d kept (hair, eyes, chest, outfit)" % (sum(rempl), nm, len(gardes)))
    return dict(P, sommets=som, os=os_, morphes=morphes, cadres=cadres, rigides=rigides,
                commentaire=(P.get("commentaire") or "") + "\nhouse skeleton and weights; twist in squelette.json")


def brancher(entree, sortie, maison=MAISON, journal=print):
    pmx_ecrire.ecrire(convertir(PMX.lire(entree), maison, journal), sortie)
    journal("written", sortie)


if __name__ == "__main__":
    a = sys.argv[1:]
    brancher(a[0], a[1], a[a.index("--maison") + 1] if "--maison" in a else MAISON)

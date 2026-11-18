"""Defect review: where the visible errors are, and how many are left.

Each defect is a measured fact, with its place (centre, in metres, in the page's frame) and its size:
the list says where to look, instead of browsing outfits by eye.

**Per part**: the fitted part is compared with its original shape (the library's PMX, same vertices
in the same order). The transfer changes the size of a part, not its shape: whatever departs from the
part's overall scale is a defect.

- `eclat`      stretched faces (an edge at least `ECLAT` times longer than expected, and longer by at
               least `ECLAT_MIN`): spikes, blades, triangles stretched across the body;
- `ecrase`     faces shrunk to less than `ECRASE` times their size: gathers, flattened ornaments;
- `dechirure`  two vertices at the same place in the original that have separated;
- `retourne`   neighbouring faces whose angle changed by more than `PLI`: sharp creases, flipped faces;
- `forme`      share of the area whose scale departs by more than `FORME` from the part's
               (a skirt that flares, a sleeve that swells): one figure for the part, not a place.

**Per outfit** (stacked parts, after the layers are settled):

- `traverse`   the lower part stays in front of the upper one (`controle_couches.paire`);
- `trou`       a face removed as "covered" that nothing covers: you can see through;
- `cheveux`    the hairstyle crosses a part (once dropped over the outfit, as in the workshop);
- `peau`       the skin shows in front of a part that should cover it;
- `ouvert`     skin removed as "hidden" while nothing opaque hides it: you can see through the
               body (fixed by `skin.py appliquer`).

    python3 review.py pieces [--json sortie.json] [id ...]     every part of the cache, worst to best
    python3 review.py tenue --pieces a,b,c [--coiffure nom]    one outfit
    python3 review.py sondage 40 [--coiffure nom] [--saines]   outfits drawn at random

An outfit defect at the place of a part defect is only one defect: `--saines` only draws parts with
no defect of their own, to judge the layers and the hair alone.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import glb as GLB  # noqa: E402
import outfit as CB  # noqa: E402
import rays as R  # noqa: E402
import pmx as PMX  # noqa: E402  (through commun's sys.path)
import model as MD  # noqa: E402

ECLAT = 2.0          # an edge stretched by this factor (relative to the part's scale)…
ECLAT_MIN = 0.01     # …and lengthened by at least 1 cm
ECRASE = 0.4
ECRASE_MIN = 0.01
DECHIRURE = 0.003
PLI = 100.0          # degrees
PLI_MIN = 0.01       # shared edge of at least 1 cm: a crease on a tiny face cannot be seen
FORME = 0.3
GRAVE = {"eclat": 3.0, "dechirure": 2.0, "retourne": 1.0, "ecrase": 1.0, "trou": 2.0, "traverse": 2.0, "ouvert": 2.0}   # weight of one centimetre of defect


def _souder(pos):
    cle = np.round(np.asarray(pos) * 1e5).astype(np.int64)
    _, inv = np.unique(cle, axis=0, return_inverse=True)
    return inv.ravel()


def origine(meta):
    """Vertices of the part in its original shape (metres), or None if the library no longer has it."""
    src = C.absolu(meta.get("dossier") or "")
    try:
        with open(os.path.join(src, "fiche.json"), encoding="utf-8") as fh:
            fiche = json.load(fh)
    except OSError:
        return None
    garnie = fiche.get("garnie") or {}
    chemin = os.path.join(src, garnie.get("pmx") or fiche["pmx"])
    if not os.path.exists(chemin):
        chemin = os.path.join(src, fiche["pmx"])
    if not os.path.exists(chemin):
        return None
    P = PMX.lire(chemin)
    T = PMX.vers_tableaux(P)
    noms = [m["nom"] for m in P["materiaux"]]
    garde = [i for i, n in enumerate(noms) if n in set(meta.get("materiaux") or []) and P["materiaux"][i]["n_indices"] > 0]
    vs = np.unique(T["faces"][np.isin(T["mat_face"], garde)])
    S = np.asarray(T["pos"][vs], float) * C.METRE
    S[:, 2] *= -1
    return S


def _groupes(F, marque):
    """Marked faces grouped by adjacency: list of arrays of face indices."""
    ou = np.where(marque)[0]
    if not len(ou):
        return []
    Fm = F[ou]
    lab = MD.composantes(int(F.max()) + 1, np.concatenate([Fm[:, [0, 1]], Fm[:, [1, 2]]]))
    lf = lab[Fm[:, 0]]
    return [ou[lf == k] for k in np.unique(lf)]


def _lieu(Z, F, faces):
    T = Z[F[faces]]
    c = T.mean(1)
    a = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    centre = (c * a[:, None]).sum(0) / max(a.sum(), 1e-12) if a.sum() > 0 else c.mean(0)
    return [round(float(x), 3) for x in centre], float(a.sum())


def formes(S, Z, F):
    """Shape defects of the fitted part Z relative to the original S (same vertices, faces F)."""
    w = _souder(S)
    Fw = w[F]
    vrai = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 2] != Fw[:, 0])
    coins = [(0, 1), (1, 2), (2, 0)]
    ls = np.stack([np.linalg.norm(S[F[:, i]] - S[F[:, j]], axis=1) for i, j in coins], 1)
    lz = np.stack([np.linalg.norm(Z[F[:, i]] - Z[F[:, j]], axis=1) for i, j in coins], 1)
    ok = vrai & (ls > 1e-5).all(1)
    if not ok.any():
        return {"echelle": None, "forme": None, "defauts": []}
    r = np.where(ls > 1e-5, lz / np.maximum(ls, 1e-5), 1.0)
    aire = 0.5 * np.linalg.norm(np.cross(S[F[:, 1]] - S[F[:, 0]], S[F[:, 2]] - S[F[:, 0]]), axis=1)
    # scale of the part: median of the ratios, weighted by area
    rr, pp = r[ok].ravel(), np.repeat(aire[ok], 3)
    o = np.argsort(rr)
    r0 = float(rr[o][np.searchsorted(np.cumsum(pp[o]), pp.sum() / 2)])
    q = r / r0
    exces = lz - r0 * ls
    defauts = []

    def noter(genre, marque, mesure):
        for faces in _groupes(Fw, marque & ok):
            centre, a = _lieu(Z, F, faces)
            m = float(mesure[faces].max())
            defauts.append({"genre": genre, "lieu": centre, "faces": int(len(faces)), "cm": round(100 * m, 1),
                            "aire_cm2": round(1e4 * a, 1)})

    noter("eclat", ((q >= ECLAT) & (exces >= ECLAT_MIN)).any(1), exces.max(1))
    noter("ecrase", ((q <= ECRASE) & (-exces >= ECRASE_MIN)).any(1), (-exces).max(1))
    # tear: vertices that coincided in the original, separated once fitted
    n = int(w.max()) + 1
    cen = np.zeros((n, 3))
    np.add.at(cen, w, Z)
    cen /= np.bincount(w, minlength=n)[:, None]
    ecart = np.linalg.norm(Z - cen[w], axis=1)
    noter("dechirure", (ecart[F] >= DECHIRURE).any(1), ecart[F].max(1) * 2)
    # flip: the angle between two neighbouring faces changed by more than PLI
    E, _, ef = MD.aretes_faces(Fw)
    ns, nz = MD.normales_faces(S, F)[0], MD.normales_faces(Z, F)[0]
    ordre = np.argsort(ef.ravel(), kind="stable")
    ar, fa = ef.ravel()[ordre], np.repeat(np.arange(len(F)), 3)[ordre]
    deux = np.where(ar[1:] == ar[:-1])[0]
    f1, f2 = fa[deux], fa[deux + 1]
    cs, cz = (ns[f1] * ns[f2]).sum(1), (nz[f1] * nz[f2]).sum(1)
    d = np.abs(np.degrees(np.arccos(np.clip(cs, -1, 1))) - np.degrees(np.arccos(np.clip(cz, -1, 1))))
    e = E[ar[deux]]
    long_ = np.linalg.norm(Z[_rep(w)[e[:, 0]]] - Z[_rep(w)[e[:, 1]]], axis=1)
    pli = (d >= PLI) & (long_ >= PLI_MIN) & ok[f1] & ok[f2]
    mf, lf = np.zeros(len(F), bool), np.zeros(len(F))
    mf[f1[pli]] = mf[f2[pli]] = True
    np.maximum.at(lf, f1[pli], long_[pli])
    np.maximum.at(lf, f2[pli], long_[pli])
    noter("retourne", mf, lf)
    hors = (np.abs(np.log(np.clip(q, 1e-3, 1e3))) > np.log(1 + FORME)).any(1) & ok
    return {"echelle": round(r0, 3), "forme": round(float(aire[hors].sum() / max(aire[ok].sum(), 1e-12)), 3),
            "defauts": defauts}


def _rep(w):
    """One representative vertex of each welded vertex."""
    rep = np.zeros(int(w.max()) + 1, np.int64)
    rep[w] = np.arange(len(w))
    return rep


def note(defauts):
    """Severity of a list of defects: centimetres of defect, weighted by kind."""
    return round(sum(GRAVE.get(d["genre"], 1.0) * d["cm"] for d in defauts), 1)


def revoir_piece(ident):
    p = CB.piece(ident)
    S = origine(p.meta)
    r = {"id": ident, "v_portage": p.meta.get("v_portage"), "type": p.meta.get("type")}
    if S is None or len(S) != len(p.pos):
        r.update(sans_origine=True, defauts=[], note=None)
        return r
    r.update(formes(S, p.pos, p.F))
    couches = CB.resoudre([p])[0]
    r["defauts"] += peau_dehors(couches) + peau_ouverte(couches)
    r["note"] = note(r["defauts"])
    return r


def ids_du_cache(tout=False):
    """The catalogue's parts; tout=True: also those its score set aside."""
    cat = C.lire_json(os.path.join(C.CACHE, "catalogue.json"), {})
    ids = [x["id"] for x in cat.get("pieces", []) if x.get("retenue", True)] if isinstance(cat, dict) and not tout else []
    if not ids:
        d = os.path.join(C.CACHE, "pieces")
        ids = sorted(i for i in os.listdir(d) if (C.lire_json(os.path.join(d, i, "meta.json"), {}) or {}).get("retenue"))
    return ids


# ------------------------------------------------------------------ outfit
CELLULE = 0.08       # the faulty points of a pair are grouped in cells of this size: one place per cell
TROU_MIN = 2.0       # cm²: under this area, a face wrongly removed cannot be seen
TRAVERSE_MIN = 0.5   # cm²
CROIX_MIN = 3        # crossings per cell
PEAU_PORTEE = 0.02   # depth under the skin where the fabric is looked for
PEAU_MIN = 0.001
OUVERT_AIRE = 4.0    # cm² of removed skin with nothing in front, per cell, for a defect
PEAU_AIRE = 1.0      # cm² of skin in front of a part, per cell, for a defect


def _cases(P, poids=None):
    """Points grouped in cells: [(centre, count, weight)]."""
    if not len(P):
        return []
    k = np.floor(P / CELLULE).astype(np.int64)
    _, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv)
    c = np.stack([np.bincount(inv, P[:, j]) for j in range(3)], 1) / n[:, None]
    w = np.bincount(inv, poids) if poids is not None else n.astype(float)
    return [([round(float(x), 3) for x in c[i]], int(n[i]), float(w[i])) for i in range(len(n))]


def coiffure(nom, tete, couches=None):
    """Hair placed on the head `tete`: (vertices, faces), in metres; with `couches`, as the workshop
    shows them, dropped over the outfit."""
    V, N, F = CB.coiffure(nom, tete)
    if couches:
        V = V + CB.coiffer(couches, V, N, F)
    return V, F


def trous(couches):
    """Removed faces (mask) that nothing opaque covers or precedes: you can see through the outfit."""
    import check_layers as CC
    env = [c for c in couches if c.enveloppe]
    out = []
    for i, L in enumerate(env):
        if not L.masque.any() or not env[i + 1:]:
            continue
        P, aire = CC.semis(L.Z, L.F[L.masque])
        nf = L.grille()[1][L.masque]
        n = np.repeat(nf / np.maximum(np.linalg.norm(nf, axis=1), 1e-12)[:, None], 4, axis=0)
        vu = np.ones(len(P), bool)
        for U in env[i + 1:]:
            opq = ((U.fl & CB.OPAQUE) > 0) & ~U.masque
            if not opq.any():
                continue
            g = R.Grille(U.Z, U.F[opq])
            vu &= ~np.isfinite(g.tirer(P, n, CB.PORTEE_PAN)[0]) & ~np.isfinite(g.tirer(P, -n, CB.PORTEE_PAN)[0])
        for centre, nb, a in _cases(P[vu], aire[vu]):
            if a * 1e4 >= TROU_MIN:
                out.append({"genre": "trou", "pieces": [L.id], "lieu": centre, "aire_cm2": round(a * 1e4, 1),
                            "cm": round(100 * np.sqrt(a), 1)})
    return out


def traversees(couches):
    import check_layers as CC
    out = []
    for r in CC.controler(couches):
        for centre, nb, a in _cases(r["points"], None):
            a = a * r["dehors_cm2"] / max(len(r["points"]), 1)
            if a >= TRAVERSE_MIN:
                out.append({"genre": "traverse", "pieces": [r["dessous"], r["dessus"]], "lieu": centre,
                            "aire_cm2": round(a, 1), "cm": round(np.sqrt(a), 1)})
        for centre, nb, _ in _cases(r["croix"]):
            if nb >= CROIX_MIN:
                out.append({"genre": "croise", "pieces": [r["dessous"], r["dessus"]], "lieu": centre,
                            "croisements": nb, "cm": round(nb / 10.0, 1)})
    return out


def cheveux(couches, V, F):
    """The hairstyle crosses a part: strands that go into a collar, a jacket, a bag."""
    out = []
    g = R.Grille(V, F)
    for c in couches:
        Fc = c.F[~c.masque]
        if not len(Fc) or not CB._boite(V, c.Z, 0.0):
            continue
        X = np.concatenate([R.croisements(c.Z, Fc, V, F, g), R.croisements(V, F, c.Z, Fc)])
        for centre, nb, _ in _cases(X):
            if nb >= CROIX_MIN:
                out.append({"genre": "cheveux", "pieces": [c.id], "lieu": centre, "croisements": nb, "cm": round(nb / 10.0, 1)})
    return out


def peau_dehors(couches, cachee=None):
    """The skin shows in front of a part that should cover it. Measured in the body's frame: from
    each shown skin face, go down along its normal; if the front of a part is met within
    PEAU_PORTEE, the fabric is under the skin."""
    pe = CB.peau()
    if pe is None:
        return []
    V, F, N, _ = pe
    cache = np.zeros(len(F), bool)      # `cachee` {id: skin faces} replaces what the cache says (part being tried)
    for c in couches:
        try:
            k = cachee[c.id] if cachee and c.id in cachee else np.fromfile(os.path.join(C.CACHE, "pieces", c.id, "peau_cachee.bin"), "<u4")
            cache[k[k < len(F)]] = True
        except OSError:
            pass
    out = []
    T = V[F]
    cen, nf = T.mean(1), N[F].sum(1)
    nf /= np.maximum(np.linalg.norm(nf, axis=1), 1e-12)[:, None]
    aire = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2 * 1e4
    for c in couches:
        Fc = c.F[((c.fl & CB.EXTERIEUR) > 0) & ~c.masque]
        if not len(Fc):
            continue
        ou = np.where(~cache & CB.voisinage(cen, c.Z[Fc].reshape(-1, 3)))[0]
        if not len(ou):
            continue
        t, f = R.Grille(c.Z, Fc).tirer(cen[ou], -nf[ou], PEAU_PORTEE)
        nc = c.nor[Fc].sum(1)
        nc /= np.maximum(np.linalg.norm(nc, axis=1), 1e-12)[:, None]
        a = np.isfinite(t) & (t > PEAU_MIN)
        a[a] &= (nf[ou][a] * nc[f[a]]).sum(1) > 0.3
        for centre, nb, s in _cases(cen[ou][a], aire[ou][a]):
            if s >= PEAU_AIRE:
                out.append({"genre": "peau", "pieces": [c.id], "lieu": centre, "aire_cm2": round(s, 1), "cm": round(np.sqrt(s), 1)})
    return out


def peau_ouverte(couches):
    """Skin removed as "hidden" that nothing opaque hides: you can see through the body
    (a jacket opened by the transparency of its texture, openwork tights)."""
    import skin as PE
    pe = CB.peau()
    if pe is None:
        return []
    V, F, _, _ = pe
    out = []
    surfaces = [(c.Z, c.F[((c.fl & CB.OPAQUE) > 0) & ~c.masque]) for c in couches]
    T = V[F]
    aire = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2 * 1e4
    for c in couches:
        k = PE.lire(c.id)
        k = k[k < len(F)]
        if not len(k):
            continue
        nu = k[~PE.couverte(surfaces, k, PE.PORTEE + CB.SORTIE_MAX + 0.002, PE.SOUS_PIED if c.type in PE.CHAUSSANTS else PE.SOUS, c.type not in PE.CHAUSSANTS)[k]]   # the layer settling may have pushed the part away
        if len(nu):
            nu = nu[~PE.enfoncee(c.Z, c.F[PE.cachantes(c.fl, len(c.F))], np.setdiff1d(k, nu))[nu]]   # skin with the fabric under it is hidden on purpose
        if c.type in PE.CHAUSSANTS:
            nu = nu[~PE.hors_semelle(c.Z, c.F)[nu]]              # the foot sticking out of a shoe is hidden on purpose,
            nu = nu[~PE.miettes(c.Z, np.setdiff1d(k, nu), c.F)[nu]]   # and so are the crumbs of skin left around it
        for centre, nb, s in _cases(T[nu].mean(1), aire[nu]):
            if s >= OUVERT_AIRE:
                out.append({"genre": "ouvert", "pieces": [c.id], "lieu": centre, "aire_cm2": round(s, 1), "cm": round(np.sqrt(s), 1)})
    return out


def revoir_tenue(ids, coif=None, tete=None):
    pieces = {i: CB.piece(i) for i in ids}
    ret = CB.retirees({i: p.meta for i, p in pieces.items()})
    couches, notes = CB.resoudre([p for i, p in pieces.items() if i not in ret])
    defauts = traversees(couches) + trous(couches) + peau_dehors(couches) + peau_ouverte(couches)
    if coif:
        defauts += cheveux(couches, *coiffure(coif, tete or coif, couches))
    defauts.sort(key=lambda d: -d["cm"])
    return {"pieces": ids, "ordre": [c.id for c in couches], "defauts": defauts, "note": note(defauts)}


SAINE = 20.0   # highest score of a "sound" part: its own defects are not mistaken for the outfit's


def tirage(n, graine=7, saines=False):
    """n plausible outfits drawn at random from the cache; `saines`: among parts with no defect of their own."""
    import random
    par = {}
    for i in ids_du_cache():
        if saines and not (revoir_piece(i)["note"] or 0) <= SAINE:
            continue
        m = C.lire_json(os.path.join(C.CACHE, "pieces", i, "meta.json"), {})
        par.setdefault(m.get("type"), []).append(i)
    rnd = random.Random(graine)

    def tire(*ts):
        liste = [i for t in ts for i in par.get(t, [])]
        return [rnd.choice(liste)] if liste else []
    for _ in range(n):
        ids = tire("haut") + tire("jupe", "short", "pantalon")
        for part, ts in ((0.6, ("veste", "manteau", "gilet")), (0.7, ("chaussettes",)), (0.8, ("chaussures", "bottes")), (0.4, ("ceinture",))):
            if rnd.random() < part:
                ids += tire(*ts)
        yield ids


def _dire_tenue(r):
    print("%s   score %s" % (" + ".join(r["pieces"]), r["note"]))
    for d in r["defauts"][:12]:
        print("    %-9s %-50s at %s  %s" % (d["genre"], " / ".join(d["pieces"]), d["lieu"],
              "%.1f cm²" % d["aire_cm2"] if "aire_cm2" in d else "%d crossings" % d["croisements"]))


def _ligne(r):
    par = {}
    for d in r["defauts"]:
        par.setdefault(d["genre"], []).append(d["cm"])
    return "%-34s v%s  score %7s  shape %5s  %s" % (
        r["id"], r.get("v_portage"), r["note"], r.get("forme"),
        "  ".join("%s ×%d (max %.1f cm)" % (g, len(v), max(v)) for g, v in sorted(par.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="commande", required=True)
    a = sp.add_parser("pieces")
    a.add_argument("ids", nargs="*")
    a.add_argument("--json")
    a.add_argument("--tout", action="store_true", help="also the parts without defects")
    a.add_argument("--noter", action="store_true", help="writes the score into each part's record and rebuilds the "
                   "catalogue: beyond NOTE_REFUS (common.py), the part is removed from the page")
    b = sp.add_parser("tenue")
    b.add_argument("--pieces", required=True)
    c = sp.add_parser("sondage")
    c.add_argument("n", type=int)
    c.add_argument("--graine", type=int, default=7)
    c.add_argument("--saines", action="store_true", help="only parts with no defect of their own: what is left comes from the layers")
    for x in (b, c):
        x.add_argument("--coiffure")
        x.add_argument("--tete")
        x.add_argument("--json")
    args = ap.parse_args()
    if args.commande == "tenue":
        r = revoir_tenue(args.pieces.split(","), args.coiffure, args.tete)
        _dire_tenue(r)
        if args.json:
            C.ecrire_json(args.json, r)
    if args.commande == "sondage":
        res = []
        for ids in tirage(args.n, args.graine, args.saines):
            try:
                res.append(revoir_tenue(ids, args.coiffure, args.tete))
            except Exception as e:
                res.append({"pieces": ids, "erreur": str(e)[:200], "defauts": [], "note": 0})
        res.sort(key=lambda r: -r["note"])
        for r in res:
            _dire_tenue(r)
        par = {}
        for r in res:
            for d in r["defauts"]:
                par[d["genre"]] = par.get(d["genre"], 0) + 1
        print("\n%d outfits, %d without defects; %s" % (len(res), sum(1 for r in res if not r["defauts"]),
              ", ".join("%s %d" % kv for kv in sorted(par.items()))))
        if args.json:
            C.ecrire_json(args.json, res)
    if args.commande == "pieces":
        res = []
        for i in args.ids or ids_du_cache(args.noter):
            try:
                res.append(revoir_piece(i))
            except Exception as e:   # an unreadable part is a result, not a stop
                res.append({"id": i, "erreur": str(e)[:200], "defauts": [], "note": None})
        res.sort(key=lambda r: -(r["note"] or 0))
        for r in res:
            if r.get("erreur") or r.get("sans_origine"):
                print("%-34s not reviewed: %s" % (r["id"], r.get("erreur") or "original shape missing or different"))
            elif r["note"] or args.tout:
                print(_ligne(r))
        vues = [r for r in res if r["note"] is not None]
        print("\n%d parts reviewed, %d with defects, %d not reviewed; shape > 0.25: %d" % (
            len(vues), sum(1 for r in vues if r["note"]), len(res) - len(vues),
            sum(1 for r in vues if (r.get("forme") or 0) > 0.25)))
        if args.json:
            C.ecrire_json(args.json, res)
        if args.noter:
            for r in vues:
                ch = os.path.join(C.CACHE, "pieces", r["id"], "meta.json")
                m = C.lire_json(ch)
                genres = {}
                for d in r["defauts"]:
                    genres[d["genre"]] = genres.get(d["genre"], 0) + 1
                m["revue"] = {"note": r["note"], "forme": r.get("forme"), "defauts": genres}
                C.ecrire_json(ch, m)
            import catalog as catalogue
            n = catalogue.construire()["nombres"]["pieces"]
            print("scores written; catalogue: %d parts out of %d" % tuple(n))


if __name__ == "__main__":
    main()

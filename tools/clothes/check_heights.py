"""Height check of fitted garments: does each garment sit on the target body's anatomy as it sat on its source?

    bin/blender-queue -b --factory-startup --python tools/clothes/check_heights.py -- \
        --corps BODY.pmx --sortie DIR <piece folder> [<piece folder> ...]
    python3 -I tools/clothes/check_heights.py --bilan DIR [DIR_BEFORE]   # table (and before/after)

For each part (library folder), the part is fitted onto BODY with `porter.porter` (as the workshop does,
skin removed under it), then landmarks are measured with the same functions on both sides:
- body landmarks (`landmarks.py`): crotch C, waist W, underbust U, bust apex B, shoulders S; on the target
  its skin, on the source the innermost surface (skin, or the garment that stands in for it);
- garment landmarks, on the part alone: its bust apex (most forward point of its front, near the body's
  chest), its underbust fold (where its front recedes, when it follows the breast), its top edge (bottoms),
  its hem (tops), its crotch (ray cast upwards at x = 0).
A garment landmark at height y on the source is expected at carte(y) on the target, carte being the
piecewise-linear map through the two bodies' landmarks (crotch on crotch, waist on waist...). The error is
the distance (cm) between where it lands and where it is expected; |error| > SEUIL_CM is a defect.
Also: whether the target's bust apex is covered by the part when the source's was. A top open in front on
its source (apex not covered) or without a fold under its most forward point (a puffer) has no bust
landmarks; above the shoulders, the neck joint is the landmark.

Writes DIR/<perso>__<part>.json, DIR/<perso>__<part>.png (row 1: source, row 2: target; front, side, close-ups
of the chest from the front and the side, close-up of the pelvis; thin lines = body landmarks, thick ticks at the edges = garment landmarks, same colours:
C blue, W green, U orange, B red, top edge/hem magenta, white tick = expected height on the target).
"""
import glob
import json
import os
import sys
import tempfile

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
OUTILS = os.path.dirname(ICI)
for d in ("rebuilt_head", "rebuilt_body", "head", "atlas"):
    sys.path.insert(0, os.path.join(OUTILS, d))
sys.path.insert(0, ICI)

CM = 8.0
SEUIL_CM = 2.5
HAUTS = ("haut", "veste", "gilet", "manteau", "combinaison", "robe")
BAS = ("short", "pantalon", "jupe", "culotte", "robe", "combinaison")
ENTREJAMBE = ("short", "pantalon", "culotte", "combinaison")
COUL = {"C": (60, 110, 255), "W": (40, 200, 70), "U": (255, 150, 30), "B": (240, 40, 40),
        "haut": (230, 60, 230), "bas": (230, 60, 230)}


# ------------------------------------------------------------------ garment landmarks
def front_piece(bvh, oss, ys, demi_largeur):
    """Forward extent of the part's front per height (rays cast from in front, first hit), relative to the
    trunk axis; -inf where the part is absent."""
    import landmarks as RP
    zc = RP.axe_z(oss)(ys)
    xs = np.arange(-demi_largeur, demi_largeur + 1e-9, 0.5 / CM)
    out = np.full(len(ys), -np.inf)
    for j, y in enumerate(ys):
        for x in xs:
            loc = bvh.ray_cast((float(x), float(y), -10.0), (0.0, 0.0, 1.0), 20.0)[0]
            if loc is not None:
                out[j] = max(out[j], -(loc[2] - zc[j]))
    return out


def part_tronc(pmx, chaines=(0,)):
    """Share of each vertex's weights held by the given chains (transport.CHAINES; 0 = trunk)."""
    import transport as TR
    seg = TR._segments_os(pmx["os"])
    tronc = [list(TR.SEGMENTS).index(n) for c in chaines for n in TR.CHAINES[c]]
    I = np.asarray(pmx["sommets"]["os"], np.int64).reshape(len(pmx["sommets"]["pos"]), -1)
    W = np.asarray(pmx["sommets"]["poids"], float).reshape(I.shape)
    W = np.where(I >= 0, W, 0.0)
    t = np.isin(seg[np.clip(I, 0, len(seg) - 1)], tronc) & (I >= 0)
    return (W * t).sum(1) / np.maximum(W.sum(1), 1e-9)


def bords(V, oss, haut=True):
    """Median over 24 sectors around the trunk axis of the highest (haut) or lowest vertex."""
    import landmarks as RP
    zc = RP.axe_z(oss)(V[:, 1])
    a = np.arctan2(V[:, 0], -(V[:, 2] - zc))
    s = ((a + np.pi) / (2 * np.pi) * 24).astype(int) % 24
    v = [V[s == k, 1].max() if haut else V[s == k, 1].min() for k in range(24) if (s == k).any()]
    return float(np.median(v)) if len(v) >= 12 else np.nan


def reperes_piece(P, F, pmx_os, corps_r, typ, tronc):
    """Garment landmarks (MMD units) of the part (P, F) on a body with landmarks corps_r."""
    import landmarks as RP
    out = {}
    b = RP.arbre(P, F)
    L = corps_r["S"] - corps_r["hanche"]
    if typ in HAUTS:
        ys = np.arange(corps_r["hanche"] + 0.45 * L, corps_r["S"] - 0.05 * L + 1e-9, RP.PAS_Y)
        fr = front_piece(b, pmx_os, ys, 0.3 * L)
        if np.isfinite(fr).mean() >= 0.5:
            j = int(np.argmax(fr))
            out["B"] = float(ys[j])
            # the fold, read over the whole front below the apex (down to the hips)
            ys2 = np.arange(corps_r["hanche"], ys[j] + 1e-9, RP.PAS_Y)
            fr2 = front_piece(b, pmx_os, ys2, 0.3 * L)
            fr2 = np.where(np.isfinite(fr2), fr2, np.nan)
            yU, R = RP.pli(ys2, fr2, len(ys2) - 1, 0.3 * L)
            if R >= 1.0 / CM:
                out["U"] = yU
            out["projection_cm"] = round(R * CM, 1)
    V = P[np.unique(F)]
    tv = tronc[np.unique(F)] >= 0.5
    if typ in BAS and tv.sum() >= 12:
        out["haut"] = bords(V[tv], pmx_os, True)
    if typ in ("haut", "veste", "gilet", "manteau") and tv.sum() >= 12:
        out["bas"] = bords(V[tv], pmx_os, False)
    if typ in ENTREJAMBE:
        y = RP.entrejambe(b, pmx_os)
        if np.isfinite(y):
            out["C"] = float(y)
    return out


def couvert(bvh_piece, Pc, Fc, oss, yB):
    """Is the body's bust apex (most forward point at height yB, either side) under the part?"""
    import landmarks as RP
    b = RP.arbre(Pc, Fc)
    ys, ang, pts = RP.sections(b, oss, yB, yB)
    zc = RP.axe_z(oss)(ys)
    p = pts[0]
    m = np.abs(((ang + np.pi) % (2 * np.pi)) - np.pi) <= RP.AVANT
    ok = m & np.isfinite(p[:, 0])
    if not ok.any():
        return None
    k = np.where(ok)[0][np.argmin(p[ok, 2])]
    o = p[k] + np.array([0, 0, 2e-3])        # just behind the surface: on a source, it may be the part itself
    return bvh_piece.ray_cast(tuple(o), (0.0, 0.0, -1.0), 1.0)[0] is not None


# ------------------------------------------------------------------ drawing
POLICE = {  # 3x5 glyphs, rows top to bottom
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001010010010",
    "8": "111101111101111", "9": "111101111001111", ".": "000000000000010", "-": "000000111000000",
    "+": "000010111010000", " ": "000000000000000", ":": "000010000010000", "=": "000111000111000",
    "_": "000000000000111", "/": "001001010100100", "%": "101001010100101",
}
for _c, _g in {"A": "010101111101101", "B": "110101110101110", "C": "111100100100111", "D": "110101101101110",
               "E": "111100110100111", "F": "111100110100100", "G": "111100101101111", "H": "101101111101101",
               "I": "111010010010111", "J": "001001001101111", "K": "101101110101101", "L": "100100100100111",
               "M": "101111111101101", "N": "110101101101101", "O": "111101101101111", "P": "111101111100100",
               "Q": "111101101111001", "R": "110101110101101", "S": "111100111001111", "T": "111010010010010",
               "U": "101101101101111", "V": "101101101101010", "W": "101101111111101", "X": "101101010101101",
               "Y": "101101010010010", "Z": "111001010100111"}.items():
    POLICE[_c] = _g


def ecrire_texte(img, x, y, texte, coul=(255, 255, 255), k=3):
    for c in texte.upper():
        g = POLICE.get(c, POLICE[" "])
        for i in range(5):
            for j in range(3):
                if g[i * 3 + j] == "1":
                    img[y + i * k:y + (i + 1) * k, x + j * k:x + (j + 1) * k] = coul
        x += 4 * k
    return img


def vue(V, F, couleur, az, boite, T, lignes, ticks, attendus=(), titre=""):
    """Grey render (rebuilt_body/view.py) with horizontal lines: lignes {name: y (cm)} thin and full width,
    ticks {name: y} thick at both edges, attendus [y] white ticks."""
    import view as VU
    img = VU.rendre(V, F, (az, 0), T, boite, couleur=couleur).copy()
    d, h, o = VU.base(az, 0)
    sel = np.all((V >= np.asarray(boite[:3])) & (V <= np.asarray(boite[3:])), 1)
    u = np.unique(F[sel[F].all(1)])
    P = np.stack([V @ d, V @ h], 1)
    lo, hi = P[u].min(0), P[u].max(0)
    c, r = 0.5 * (lo + hi), 0.54 * float((hi - lo).max())

    def py(y):
        return int(round(0.5 * T - (y - c[1]) / r * 0.5 * T))
    for n, y in lignes.items():
        if np.isfinite(y) and 1 <= py(y) < T - 1:
            img[py(y) - 1:py(y) + 1, :] = COUL[n]
    for n, y in ticks.items():
        if np.isfinite(y) and 2 <= py(y) < T - 2:
            img[py(y) - 2:py(y) + 3, :T // 6] = COUL[n]
            img[py(y) - 2:py(y) + 3, -T // 6:] = COUL[n]
    for y in attendus:
        if np.isfinite(y) and 1 <= py(y) < T - 1:
            img[py(y) - 1:py(y) + 2, T // 6:T // 6 + T // 12] = 255
            img[py(y) - 1:py(y) + 2, -T // 6 - T // 12:-T // 6] = 255
    if titre:
        ecrire_texte(img, 6, 6, titre, k=2)
    return img


def planche(cote, T=330):
    """One row: front, side, close-ups of the chest (front and side) and of the pelvis (close-ups and side
    views without the arms, which hide the chest in an A pose). cote: dict with V (cm), F, coul, bras (faces held by an arm), corps_r, piece_r, attendus (cm)."""
    V, F, coul = cote["V"], cote["F"], cote["coul"]
    Fp, cp = F[~cote["bras"]], coul[~cote["bras"]]
    r = {k: cote["corps_r"][k] * CM for k in ("C", "W", "U", "B", "S")}
    lignes = {k: r[k] for k in ("C", "W", "U", "B")}
    ticks = {k: v * CM for k, v in cote["piece_r"].items() if k in COUL and isinstance(v, float)}
    att = [v * CM for v in cote.get("attendus", {}).values()]
    L = r["S"] - r["C"]
    big = [-60, r["C"] - 0.35 * L, -60, 60, r["S"] + 0.2 * L, 60]
    pres = [-60, r["U"] - 0.35 * L, -60, 60, r["S"] + 0.05 * L, 60]
    bassin = [-60, r["C"] - 0.3 * L, -60, 60, r["W"] + 0.1 * L, 60]
    return np.concatenate([vue(V, F, coul, 0, big, T, lignes, ticks, att, cote["titre"]),
                           vue(V, Fp, cp, 90, big, T, lignes, ticks, att, cote.get("texte", "")),
                           vue(V, Fp, cp, 0, pres, T, lignes, ticks, att, "BUSTE FACE"),
                           vue(V, Fp, cp, 90, pres, T, lignes, ticks, att, "BUSTE PROFIL"),
                           vue(V, Fp, cp, 0, bassin, T, lignes, ticks, att, "BASSIN")], 1)


# ------------------------------------------------------------------ one part
def controler(piece_dir, corps, sortie, cache_src={}):
    import pmx as PMX
    import model as MD
    import wear as PO
    import landmarks as RP
    fiche, chemin_piece = PO.charger_piece(piece_dir)
    nom = "%s__%s" % (fiche["perso"], fiche["id"])
    typ = fiche.get("type")
    # ---- target body
    if corps not in cache_src:
        C = PMX.lire(corps)
        T = PMX.vers_tableaux(C)
        noms = [m["nom"] for m in C["materiaux"]]
        peau = np.array([bool(MD.PEAU.match(n)) for n in noms])[T["mat_face"]]
        cache_src[corps] = (C, T, peau, RP.reperes(T["pos"], T["faces"][peau], C["os"]))
    C, Tc, peau_c, rc = cache_src[corps]
    # ---- source body: innermost surface (skin and garments, no face or hair)
    src = fiche["source"]["pmx"]
    if src not in cache_src:
        S = PMX.lire(src)
        T = PMX.vers_tableaux(S)
        noms = [m["nom"] for m in S["materiaux"]]
        garde = np.array([not MD.NON_VETEMENT.match(n) for n in noms])[T["mat_face"]]
        cache_src[src] = (S, T, garde, RP.reperes(T["pos"], T["faces"][garde], S["os"]))
    S, Ts, garde_s, rs = cache_src[src]
    # ---- the part on its source
    Pp = PMX.lire(chemin_piece)
    Tp = PMX.vers_tableaux(Pp)
    gs = reperes_piece(Tp["pos"], Tp["faces"], S["os"], rs, typ, part_tronc(Pp))
    # ---- the part fitted on the target
    tmp = tempfile.mkdtemp(prefix="hauteurs-", dir=sortie)
    res = PO.porter(piece_dir, corps, tmp, nom="piece", marge_mm=3.0, peau="retirer")
    O = PMX.lire(res["pmx"])
    To = PMX.vers_tableaux(O)
    noms_o = [m["nom"] for m in O["materiaux"]]
    fp = np.isin(To["mat_face"], [i for i, n in enumerate(noms_o) if n in res["mats"]])
    gt = reperes_piece(To["pos"], To["faces"][fp], C["os"], rc, typ, part_tronc(O))
    # above the shoulders (collars), the neck joint is the landmark: the joint the head turns about
    for r_, o_ in ((rs, S["os"]), (rc, C["os"])):
        j = RP._joint(o_, ("首",))
        r_["首"] = float(j[1]) if j is not None else np.nan
    vers_t, _ = RP.carte(rs, rc, RP.NOMS + ("首",))
    cov = {}
    if typ in HAUTS:
        # the target's whole skin (as before the fitting removed what the part hides)
        cov["cible"] = couvert(RP.arbre(To["pos"], To["faces"][fp]), Tc["pos"], Tc["faces"][peau_c], C["os"], rc["B"])
        cov["source"] = couvert(RP.arbre(Tp["pos"], Tp["faces"]), Ts["pos"], Ts["faces"][garde_s], S["os"], rs["B"])
    lignes, erreurs = {}, []
    for k in ("C", "W", "U", "B", "haut", "bas"):
        if k in ("U", "B") and cov.get("source") is False:
            continue             # open in front on its source: its apex and fold are not the bust's
        if k == "B" and "U" not in gs:
            continue             # no fold under its most forward point: not a bust shape (a puffer's belly)
        if k in gs and k in gt and np.isfinite(gs[k]) and np.isfinite(gt[k]):
            att = float(vers_t(gs[k]))
            e = (gt[k] - att) * CM
            lignes[k] = {"source_cm": round(gs[k] * CM, 1), "cible_cm": round(gt[k] * CM, 1),
                         "attendu_cm": round(att * CM, 1), "erreur_cm": round(e, 1),
                         # relative to the body's own landmark of the same name (C, U, B), source and target
                         **({"ecart_source_cm": round((gs[k] - rs[k]) * CM, 1),
                             "ecart_cible_cm": round((gt[k] - rc[k]) * CM, 1)} if k in rs else {})}
            erreurs.append(abs(e))
    bilan = {"piece": nom, "type": typ, "corps": os.path.abspath(corps), "source": src,
             "corps_source_cm": {k: round(rs[k] * CM, 1) for k in ("C", "hanche", "W", "U", "B", "S")},
             "corps_cible_cm": {k: round(rc[k] * CM, 1) for k in ("C", "hanche", "W", "U", "B", "S")},
             "reperes": lignes, "buste_couvert": cov, "pire_cm": round(max(erreurs), 1) if erreurs else None,
             "defaut": bool(erreurs and max(erreurs) > SEUIL_CM) or (cov.get("source") is True and cov.get("cible") is False),
             "portage": {k: res.get(k) for k in ("sous_la_peau", "reste_dedans", "allongement", "transport")}}
    with open(os.path.join(sortie, nom + ".json"), "w", encoding="utf-8") as f:
        json.dump(bilan, f, ensure_ascii=False, indent=1)
    # ---- board
    gris = np.array([0.86, 0.80, 0.78])
    bleu = np.array([0.35, 0.55, 0.95])
    autres = np.array([0.62, 0.62, 0.66])
    Fs = Ts["faces"][garde_s]
    cle = {tuple(v) for v in np.round(Tp["pos"], 4)}
    dans = np.array([tuple(v) in cle for v in np.round(Ts["pos"], 4)])
    pf = dans[Fs].all(1)
    peau_s = np.array([bool(MD.PEAU.match(m["nom"])) for m in S["materiaux"]])[Ts["mat_face"]][garde_s]
    coul_s = np.where(pf[:, None], bleu, np.where(peau_s[:, None], gris, autres))
    garde_o = np.array([not MD.NON_VETEMENT.match(n) for n in noms_o])[To["mat_face"]]
    Fo = To["faces"][garde_o]
    coul_o = np.where(fp[garde_o][:, None], bleu, gris)
    att = {k: vers_t(gs[k]) for k in gs if k in ("C", "U", "B", "haut", "bas") and np.isfinite(gs[k])}
    bras_s = part_tronc(S, (0, 3, 4))[Fs].mean(1) < 0.5       # arms and hands
    bras_o = part_tronc(O, (0, 3, 4))[Fo].mean(1) < 0.5
    texte = " ".join("%s%+.0f" % ({"haut": "T", "bas": "H"}.get(k, k), v["erreur_cm"]) for k, v in lignes.items())
    hauts = planche({"V": Ts["pos"] * CM, "F": Fs, "coul": coul_s, "bras": bras_s, "corps_r": rs, "piece_r": gs,
                     "titre": "SOURCE " + nom.replace("__", " ")})
    bas = planche({"V": To["pos"] * CM, "F": Fo, "coul": coul_o, "bras": bras_o, "texte": texte, "corps_r": rc, "piece_r": gt, "attendus": att,
                   "titre": "CIBLE PIRE %s CM" % (bilan["pire_cm"] if bilan["pire_cm"] is not None else "-")})
    import png
    png.ecrire(os.path.join(sortie, nom + ".png"), np.concatenate([hauts, bas], 0))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return bilan


# ------------------------------------------------------------------ table
def juger(b):
    """(worst error, defect) of a check, with the rules of this version (older JSON files are judged again)."""
    rp = b["reperes"]
    e = [abs(v["erreur_cm"]) for k, v in rp.items() if not (k == "B" and "U" not in rp)
         and not (k in ("U", "B") and b["buste_couvert"].get("source") is False)]
    cv = b["buste_couvert"]
    pire = round(max(e), 1) if e else None
    return pire, bool(e and pire > SEUIL_CM) or (cv.get("source") is True and cv.get("cible") is False)


def bilan(dossier, avant=None):
    def lire(d):
        return {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(d, "*.json")))}
    apres = lire(dossier)
    av = lire(avant) if avant else {}
    lignes = ["piece                 | " + " | ".join("%-15s" % k for k in ("B apex", "U fold", "C crotch", "top edge", "hem"))
              + " | covered | defect" + (" | before (worst)" if av else "")]
    for nom, b in apres.items():
        cel = []
        b["pire_cm"], b["defaut"] = juger(b)
        for k in ("B", "U", "C", "haut", "bas"):
            x = b["reperes"].get(k)
            if k == "B" and "U" not in b["reperes"]:
                x = None
            cel.append("%-15s" % ("" if x is None else "%+5.1f (%5.1f)" % (x["erreur_cm"], x["cible_cm"])))
        cv = b["buste_couvert"]
        cov = "" if not cv else "%s>%s" % ("y" if cv.get("source") else "n", "y" if cv.get("cible") else "n")
        s = "%-21s | %s | %-7s | %s" % (nom, " | ".join(cel), cov, "DEFECT" if b["defaut"] else "ok")
        if nom in av:
            s += " | %s (%s)" % (lambda pd: ("DEFECT" if pd[1] else "ok", pd[0]))(juger(av[nom]))
        lignes.append(s)
    lignes.append("cells: error in cm (target height in cm); error = landed height - height expected from the source's "
                  "anatomy; defect if |error| > %.1f cm or bust apex no longer covered" % SEUIL_CM)
    return "\n".join(lignes)


def main(argv):
    if "--bilan" in argv:
        i = argv.index("--bilan")
        print(bilan(argv[i + 1], argv[i + 2] if len(argv) > i + 2 else None))
        return
    val = lambda o: argv[argv.index(o) + 1]  # noqa: E731
    corps, sortie = val("--corps"), val("--sortie")
    os.makedirs(sortie, exist_ok=True)
    pieces = [a for k, a in enumerate(argv) if not a.startswith("--") and (k == 0 or argv[k - 1] not in ("--corps", "--sortie"))]
    for p in pieces:
        try:
            b = controler(p, corps, sortie)
            print("HAUTEURS", b["piece"], "worst %s cm" % b["pire_cm"], "DEFECT" if b["defaut"] else "ok", flush=True)
        except Exception as e:                    # noqa: BLE001  (one part failing does not stop the others)
            import traceback
            traceback.print_exc()
            print("HAUTEURS", p, "FAILED", e, flush=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])

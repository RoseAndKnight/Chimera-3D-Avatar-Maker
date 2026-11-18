"""Builds Chimera's base (bare body, base head, hairstyle) from a VRoid Studio character.

    python3 -I tools/vroid_base/build.py <file.vrm | name of an imported model> [--output <folder>] [--no-sheets]

 1. converts the .vrm (VRM 0.x or 1.0) into a PMX in the common language (tools/vroid/importer.py), in
    <sortie>/modele/<Name>/, keeping all of the body skin (the ordinary import removes the faces that the
    texture erases: VRoid erases the skin under the outfit); a bare name: <the game's sources.vrm>/<name>.vrm if it
    exists, otherwise the already imported PMX (<mmd_extrait>/<name>/<name>.pmx);
 2. repaints the body skin (skin.py): VRoid paints tights, socks, a tight top and underwear on it, or
    makes it transparent under the outfit; whatever is not skin is filled from the neighbouring skin;
 3. gives a shared normal to the split vertices of the skin seams, then writes <sortie>/corps.pmx (the skin alone, 肌: the base body, setting workshop.body), tete.pmx (the same skin,
    vertices in the same order, plus the face, its eyes and its expressions) and coiffure.pmx (the model's
    hair), with their textures. The model's outfit is not included: it enters the catalogue like that of any
    imported character;
 4. measures the head (tete.Tete: distance between the eyes and their midpoint) and stores in <sortie>/base.json what the "vroid" game
    reads of its base head (tete_base: nom, premier_passage, ecart_yeux, centre_yeux) and the display names;
 5. checks (controles.json): humanoid bones with MMD names, arms-down pose, skin closed below the head, opaque skin
    texture, eyes found; a failed blocking check stops everything (exit code 1);
 6. writes the provenance and the terms of use read from the file (VRM meta): provenance.json, PROVENANCE.md;
 7. renders grey contact sheets (without Blender): corps.png, tete.png.
By default, output in {data}/base-vroid. Without Blender; a few seconds.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
OUTILS = os.path.dirname(ICI)
sys.path[:0] = [ICI, OUTILS] + [os.path.join(OUTILS, d) for d in ("vroid", "atlas", "head", "rebuilt_head", "rebuilt_body")]
import settings as R          # noqa: E402
import importer as IM         # noqa: E402
import pmx as PMX             # noqa: E402
import pmx_write as pmx_ecrire             # noqa: E402
import geom as G              # noqa: E402
import images                 # noqa: E402
import png                    # noqa: E402
import skin as PE             # noqa: E402

PEAU, VISAGE_PEAU = "肌", "颜"
VISAGE = ("颜", "口舌", "齿", "眉", "睫", "二重", "白目", "目", "目光", "目光2")
CHEVEUX = re.compile(r"^髪\d*$")
OS_REQUIS = ("全ての親", "センター", "下半身", "上半身", "首", "頭", "左肩", "左腕", "左ひじ", "左手首", "右肩", "右腕",
             "右ひじ", "右手首", "左足", "左ひざ", "左足首", "右足", "右ひざ", "右足首")
BRAS = (40.0, 56.0)          # degrees below the horizontal of the bone 腕 -> ひじ: Chimera's A-pose (47-50°)
TROU_MAX = 6                 # vertices: a smaller opening in the skin is not a hole (seam of the model)
SOUDURE = 0.01               # MMD units: two vertices closer than this are the same point (seams, neck cut)
PEAU_GARDEE_MIN = 0.10       # share of the body texture still skin: below it, almost everything is repainted
CM = 8.0
META = ("title", "version", "author", "contactInformation", "reference", "allowedUserName", "violentUssageName",
        "sexualUssageName", "commercialUssageName", "otherPermissionUrl", "licenseName", "otherLicenseUrl")
META_1 = ("name", "version", "authors", "copyrightInformation", "contactInformation", "references", "thirdPartyLicenses",
          "licenseUrl", "avatarPermission", "allowExcessivelyViolentUsage", "allowExcessivelySexualUsage", "commercialUsage",
          "allowPoliticalOrReligiousUsage", "allowAntisocialOrHateUsage", "creditNotation", "allowRedistribution",
          "modification", "otherLicenseUrl")


# ------------------------------------------------------------------ source
def source(arg, jeu=None):
    """(vrm or None, already imported pmx or None, name) from the argument: a .vrm, or the name of an imported model."""
    if arg.lower().endswith(".vrm"):
        if not os.path.isfile(arg):
            raise SystemExit("VRoid base: %s does not exist" % arg)
        return os.path.abspath(arg), None, os.path.splitext(os.path.basename(arg))[0]
    if jeu is None:
        import games as jeux
        jeu = jeux.charger("vroid")
    vrm = os.path.join(jeu.chemin("vrm"), arg + ".vrm")
    if os.path.isfile(vrm):
        return vrm, None, arg
    pmx = os.path.join(jeu.chemin("mmd_extrait"), arg, arg + ".pmx")
    if os.path.isfile(pmx):
        return None, pmx, arg
    raise SystemExit("VRoid base: neither %s nor %s. Give the path of a .vrm file exported from VRoid Studio or taken from VRoid Hub (VRM 0.x or 1.0)." % (vrm, pmx))


def meta_vrm(vrm):
    """Terms of use declared in the file, under their own keys: VRM 0.x (extensions.VRM.meta) or 1.0
    (extensions.VRMC_vrm.meta, other names and values). "title" is the character's name in both."""
    j, _ = IM.lire_glb(vrm)
    ext = j.get("extensions", {})
    if "VRMC_vrm" in ext:
        m = ext["VRMC_vrm"].get("meta") or {}
        r = {"vrm": "1.0"}
        r.update({k: m.get(k) for k in META_1})
        r["title"] = m.get("name")
        return r
    m = (ext.get("VRM") or {}).get("meta", {})
    r = {"vrm": "0.x"}
    r.update({k: m.get(k) for k in META})
    return r


def empreinte(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ PMX
def sous_pmx(P, T, mats, premiers=()):
    """PMX reduced to the materials `mats` (names), all bones kept. The vertices `premiers` come first, in
    that order, then the other vertices used, in the original order. Returns (pmx, old index -> new)."""
    noms = [m["nom"] for m in P["materiaux"]]
    garde = [i for i, n in enumerate(noms) if n in mats]
    sel = np.isin(T["mat_face"], garde)
    utiles = np.unique(T["faces"][sel])
    premiers = np.asarray(premiers, np.int64)
    ordre = np.concatenate([premiers, np.setdiff1d(utiles, premiers)])
    nv = -np.ones(len(T["pos"]), np.int64)
    nv[ordre] = np.arange(len(ordre))
    s = P["sommets"]
    som = {k: [s[k][i] for i in ordre] for k in ("pos", "nor", "uv", "uv_add", "type", "os", "poids", "sdef", "contour")}
    faces, materiaux, textures, tex_nv = [], [], [], {}
    for i in garde:
        F = T["faces"][T["mat_face"] == i]
        if not len(F):
            continue
        m = dict(P["materiaux"][i])
        t = m["texture"]
        if 0 <= t < len(P["textures"]):
            if t not in tex_nv:
                tex_nv[t] = len(textures)
                textures.append(P["textures"][t])
            m["texture"] = tex_nv[t]
        m["n_indices"] = 3 * len(F)
        materiaux.append(m)
        faces += [tuple(int(x) for x in f) for f in nv[F]]
    morphes = []
    for mo in P["morphes"]:
        if mo["type"] != 1:
            continue
        d = [(int(nv[a]), v) for a, v in mo["decalages"] if nv[a] >= 0]
        if d:
            morphes.append(dict(mo, decalages=d))
    Q = dict(P, sommets=som, faces=faces, materiaux=materiaux, textures=textures, morphes=morphes, rigides=[], joints=[],
             cadres=[{"nom": "Root", "nom_en": "Root", "special": 1, "elements": [(0, 0)]},
                     {"nom": "表情", "nom_en": "Exp", "special": 1, "elements": [(1, k) for k in range(len(morphes))]}])
    return Q, nv


def ecrire(Q, dossier_src, sortie, nom, remplace=None):
    """Writes `nom` (pmx) into `sortie` with its textures, copied from `dossier_src`; `remplace`: {old: new}."""
    remplace = remplace or {}
    Q = dict(Q, textures=[remplace.get(t, t) for t in Q["textures"]])
    for t in Q["textures"]:
        dest = os.path.join(sortie, t)
        if t not in remplace.values() and not os.path.isfile(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(os.path.join(dossier_src, t), dest)
    pmx_ecrire.ecrire(Q, os.path.join(sortie, nom + ".tmp.pmx"))
    os.replace(os.path.join(sortie, nom + ".tmp.pmx"), os.path.join(sortie, nom))


def bords_de_coutures(T, i_mat, soudure=SOUDURE, pas=6):
    """Points of the skin seams: (uv_a, uv_b), two places in the texture for the same point of the model.
    The vertices split at the same spot, and `pas` points along each seam edge."""
    F = T["faces"][T["mat_face"] == i_mat]
    idx = np.unique(F)
    pos, uv = np.asarray(T["pos"]), np.asarray(T["uv"], float)
    _, inv = np.unique(np.round(pos[idx] / soudure).astype(np.int64), axis=0, return_inverse=True)
    point = -np.ones(len(pos), np.int64)
    point[idx] = inv.ravel()
    # edge by its two welded points -> its real edges (vertices)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    cle = np.sort(point[E], 1)
    ordre = np.lexsort((cle[:, 1], cle[:, 0]))
    cle, E = cle[ordre], E[ordre]
    ua, ub = [], []
    debut = np.concatenate([[True], (cle[1:] != cle[:-1]).any(1)])
    groupes = np.split(np.arange(len(E)), np.where(debut)[0][1:])
    t = (np.arange(pas) + 0.5) / pas
    for g in groupes:
        if len(g) < 2:
            continue
        a = E[g[0]]
        if point[a[0]] != cle[g[0], 0]:
            a = a[::-1]
        for j in g[1:]:
            b = E[j]
            if point[b[0]] != cle[j, 0]:
                b = b[::-1]
            if np.abs(uv[a] - uv[b]).max() < 1e-4:
                continue                              # the same edge seen from both triangles: not a seam
            ua.append(uv[a[0]] + t[:, None] * (uv[a[1]] - uv[a[0]]))
            ub.append(uv[b[0]] + t[:, None] * (uv[b[1]] - uv[b[0]]))
    if not ua:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(ua), np.concatenate(ub)


def souder_normales(P, T, i_mat, soudure=SOUDURE):
    """The skin vertices split at the same point (UV seams: ankles, wrists…) take the average of
    their normals. VRoid leaves them separate: under a sock or a glove the break could not be seen; on
    bare skin, toon shading draws a line there. Returns the number of vertices changed."""
    idx = np.unique(T["faces"][T["mat_face"] == i_mat])
    pos = np.asarray(T["pos"])[idx]
    nor = np.asarray(P["sommets"]["nor"], float)[idx]
    _, inv, cpt = np.unique(np.round(pos / soudure).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    somme = np.zeros((len(cpt), 3))
    np.add.at(somme, inv, nor)
    moy = somme / np.maximum(np.linalg.norm(somme, axis=1, keepdims=True), 1e-9)
    change = (cpt[inv] > 1) & (np.abs(moy[inv] - nor).max(1) > 1e-4)
    for i in np.where(change)[0]:
        P["sommets"]["nor"][idx[i]] = tuple(float(x) for x in moy[inv[i]])
    return int(change.sum())


def basculer_diagonales(P, T, i_mat):
    """The skin's shared edges swapped where they cross a fold instead of following it (geom.basculer_diagonales):
    under the breasts, in the armpits and at the elbows the VRoid mesh has diagonals across the crease, which grazing
    light and toon shadows draw as small steps. Vertices and normals do not move; P and T are changed in place, before
    sous_pmx, so that corps.pmx and tete.pmx keep the same body faces. Returns the number of swaps."""
    sel = np.flatnonzero(T["mat_face"] == i_mat)
    F, n = G.basculer_diagonales(np.asarray(T["pos"], float), T["faces"][sel])
    T["faces"][sel] = F
    for k, f in zip(sel, F):
        P["faces"][k] = [int(x) for x in f]
    return n


# ------------------------------------------------------------------ checks
def ouvertures(pos, F, soudure=SOUDURE):
    """Free edges of the surface (vertices welded within `soudure`): [(number of vertices, centre)] from largest to smallest."""
    _, inv = np.unique(np.round(np.asarray(pos) / soudure).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    W = inv[np.asarray(F)]
    W = W[(W[:, 0] != W[:, 1]) & (W[:, 1] != W[:, 2]) & (W[:, 0] != W[:, 2])]
    E = np.sort(np.concatenate([W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    bord = u[c == 1]
    par = {}

    def racine(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in bord:
        par[racine(a)] = racine(b)
    groupes = {}
    for a, b in bord:
        groupes.setdefault(racine(a), set()).update((int(a), int(b)))
    rep = np.zeros(inv.max() + 1, np.int64)
    rep[inv] = np.arange(len(inv))
    res = [(len(g), np.asarray(pos)[rep[list(g)]].mean(0)) for g in groupes.values()]
    return sorted(res, key=lambda x: -x[0])


def angle_bras(oss):
    p = {o["nom"]: np.asarray(o["pos"], float) for o in oss}
    res = []
    for c in "左右":
        d = p[c + "ひじ"] - p[c + "腕"]
        res.append(float(np.degrees(np.arctan2(-d[1], abs(d[0])))))
    return res


def controler(P_corps, T_corps, part_peau, tete, alpha_min):
    """{name: {passe, valeur, bloquant}} of the base's checks."""
    noms_os = {o["nom"] for o in P_corps["os"]}
    manque = [n for n in OS_REQUIS if n not in noms_os]
    c = {"os_humanoides": {"passe": not manque, "bloquant": True, "valeur": manque or "tous"}}
    if not manque:
        a = angle_bras(P_corps["os"])
        c["pose_bras_deg"] = {"passe": all(BRAS[0] <= x <= BRAS[1] for x in a), "bloquant": True, "valeur": [round(x, 1) for x in a]}
        y_cou = next(o["pos"][1] for o in P_corps["os"] if o["nom"] == "首")
        trous = [(n, ctr) for n, ctr in ouvertures(T_corps["pos"], T_corps["faces"]) if ctr[1] < y_cou and n > TROU_MAX]
        c["peau_fermee"] = {"passe": not trous, "bloquant": True,
                            "valeur": [{"sommets": n, "centre_cm": [round(float(x) * CM, 1) for x in ctr]} for n, ctr in trous] or "no hole below the neck"}
    c["peau_opaque"] = {"passe": alpha_min >= 0.999, "bloquant": True, "valeur": round(float(alpha_min), 3)}
    c["peau_d_origine"] = {"passe": part_peau >= PEAU_GARDEE_MIN, "bloquant": False, "valeur": round(part_peau, 3),
                           "note": "share of the body texture kept as it is; the rest is repainted from the neighbouring skin"}
    c["taille_cm"] = {"passe": True, "bloquant": False, "valeur": round(float(np.ptp(T_corps["pos"][:, 1])) * CM, 1)}
    c["yeux"] = {"passe": tete is not None and not isinstance(tete, str), "bloquant": True,
                 "valeur": tete if isinstance(tete, str) else ("distance %.4f" % tete.iod if tete else "not measured")}
    for v in c.values():
        v["passe"] = bool(v["passe"])
    return c


# ------------------------------------------------------------------ provenance
def provenance(meta, src, nom, sha):
    pj = {"modele": nom, "fichier": os.path.basename(src), "sha256": sha, "meta_vrm": meta,
          "outil": "tools/vroid_base/build.py"}
    alertes = []
    if meta and meta.get("vrm") == "1.0":   # an absent key takes the VRM 1.0 default, the strictest one
        if meta.get("avatarPermission") != "everyone":
            alertes.append("the file restricts the use of the avatar (avatarPermission = %s)" % meta.get("avatarPermission"))
        if meta.get("allowRedistribution") is not True:
            alertes.append("the file prohibits redistribution (allowRedistribution = false)")
        if meta.get("commercialUsage") in (None, "personalNonProfit"):
            alertes.append("the file prohibits commercial use (commercialUsage = personalNonProfit)")
        if meta.get("modification") in (None, "prohibited"):
            alertes.append("the file prohibits modification (modification = prohibited), and the base is one")
    elif meta:
        if meta.get("allowedUserName") not in (None, "Everyone"):
            alertes.append("the file restricts the use of the avatar (allowedUserName = %s)" % meta.get("allowedUserName"))
        if meta.get("licenseName") in ("Redistribution_Prohibited",):
            alertes.append("the file prohibits redistribution (licenseName = Redistribution_Prohibited)")
        if meta.get("commercialUssageName") == "Disallow":
            alertes.append("the file prohibits commercial use")
    pj["alertes"] = alertes
    lignes = ["# Provenance of the VRoid base", "",
              "Built by `tools/vroid_base/build.py` from `%s` (sha256 %s), character \"%s\"."
              % (os.path.basename(src), sha[:16], nom), ""]
    if meta:
        cles = META_1 if meta.get("vrm") == "1.0" else META
        lignes += ["Terms of use declared in the file (VRM %s meta):" % meta.get("vrm", "0.x"), ""]
        lignes += ["- `%s` : %s" % (k, meta[k]) for k in cles if meta.get(k) not in (None, "", [])]
    else:
        lignes += ["No .vrm file: terms of use not read. See the comment of the imported PMX."]
    lignes += ["", "These terms are those of the model's author; they apply to the base and to every model "
               "assembled on it (body, face and hairstyle come from it)."]
    if alertes:
        lignes += ["", "**Warning**: " + "; ".join(alertes) + "."]
    return pj, "\n".join(lignes) + "\n"


# ------------------------------------------------------------------ fabrication
def fabriquer(arg, sortie, planches=True, journal=print):
    vrm, pmx_importe, nom = source(arg)
    os.makedirs(sortie, exist_ok=True)
    prec = _lire(os.path.join(sortie, "provenance.json"))
    if vrm:
        dossier = os.path.join(sortie, "modele", nom)
        shutil.rmtree(dossier, ignore_errors=True)
        journal("— converting %s (body skin kept whole)" % vrm)
        IM.convertir(vrm, dossier, journal=journal, garder={PEAU})
        chemin = os.path.join(dossier, nom + ".pmx")
    else:
        journal("— %s: no .vrm, the imported PMX is taken as it is (the skin that its texture erases may be missing)" % pmx_importe)
        chemin = pmx_importe
    src = vrm or pmx_importe
    sha = empreinte(src)
    if prec and prec.get("sha256") != sha:
        journal("Note: %s held the base of another model (%s). Chimera's cache is only valid for one body: "
                "give it a new one (setting workshop.cache)." % (sortie, prec.get("modele")))
    dossier_src = os.path.dirname(chemin)
    P = PMX.lire(chemin)
    T = PMX.vers_tableaux(P)
    noms = [m["nom"] for m in P["materiaux"]]
    for requis in (PEAU, VISAGE_PEAU):
        if requis not in noms:
            raise SystemExit("VRoid base: %s has no material %s (body or face skin): is it a VRoid Studio export?" % (chemin, requis))
    i_peau = noms.index(PEAU)
    # --- repainted skin
    journal("— body skin")
    tex = lambda nom_mat: os.path.join(dossier_src, P["textures"][P["materiaux"][noms.index(nom_mat)]["texture"]])   # noqa: E731
    propre, connu = PE.nettoyer(images.lire(tex(PEAU)), images.lire(tex(VISAGE_PEAU)))
    part = float(connu.mean())
    uv_a, uv_b = bords_de_coutures(T, i_peau)
    propre = PE.accorder_coutures(propre, connu, uv_a, uv_b)
    journal("texture seams: %d points matched" % len(uv_a))
    png.ecrire(os.path.join(sortie, "peau.png"), (propre * 255).round().clip(0, 255).astype(np.uint8))
    journal("skin kept as it is: %.0f %% of the texture, the rest repainted" % (100 * part))
    ancienne = P["textures"][P["materiaux"][i_peau]["texture"]]
    n_coutures = souder_normales(P, T, i_peau)
    journal("skin seams: %d split vertices get a shared normal" % n_coutures)
    n_diag = basculer_diagonales(P, T, i_peau)
    journal("skin diagonals: %d swapped to follow the folds" % n_diag)
    # --- body, head, hairstyle
    corps, nv = sous_pmx(P, T, {PEAU})
    corps["nom"], corps["nom_en"] = "corps " + nom, "body"
    ecrire(corps, dossier_src, sortie, "corps.pmx", {ancienne: "peau.png"})
    premiers = np.where(nv >= 0)[0][np.argsort(nv[nv >= 0])]     # the body vertices, in the order of corps.pmx
    tete, _ = sous_pmx(P, T, {PEAU} | set(VISAGE), premiers)
    tete["nom"], tete["nom_en"] = "tête " + nom, "head"
    ecrire(tete, dossier_src, sortie, "tete.pmx", {ancienne: "peau.png"})
    mats_cheveux = {n for n in noms if CHEVEUX.match(n)}
    if not mats_cheveux:
        journal("Note: no hair (material 髪): no base hairstyle")
    else:
        coiffure, _ = sous_pmx(P, T, mats_cheveux)
        coiffure["nom"], coiffure["nom_en"] = "coiffure " + nom, "hair"
        ecrire(coiffure, dossier_src, sortie, "coiffure.pmx")
    # --- head measurements
    Pc = PMX.lire(os.path.join(sortie, "corps.pmx"))
    Tc = PMX.vers_tableaux(Pc)
    try:
        import head as TE
        t = TE.Tete(os.path.join(sortie, "tete.pmx"))
        centre = (np.asarray(t.centre_oeil["g"]) + np.asarray(t.centre_oeil["d"])) / 2
        mesure = t
    except Exception as e:                     # noqa: BLE001  (an unreadable head is a failed check, not a crash)
        mesure, centre = "%s: %s" % (type(e).__name__, e), None
    alpha = images.lire(os.path.join(sortie, "peau.png"))[..., 3].min()
    ctl = controler(Pc, Tc, part, mesure, alpha)
    meta = meta_vrm(vrm) if vrm else None
    titre = (meta or {}).get("title") or nom
    base = {"modele": nom, "titre": titre,
            "tete_base": {"nom": nom, "premier_passage": nom}}
    if centre is not None:
        base["tete_base"].update({"ecart_yeux": round(float(mesure.iod), 4), "centre_yeux": [round(float(x), 4) for x in centre]})
    _ecrire_json(os.path.join(sortie, "base.json"), base)
    _ecrire_json(os.path.join(sortie, "controles.json"), ctl)
    pj, md = provenance(meta, src, nom, sha)
    _ecrire_json(os.path.join(sortie, "provenance.json"), pj)
    with open(os.path.join(sortie, "PROVENANCE.md"), "w", encoding="utf-8") as f:
        f.write(md)
    for a in pj["alertes"]:
        journal("WARNING: " + a)
    if planches:
        regarder(sortie, Tc, PMX.lire(os.path.join(sortie, "tete.pmx")))
    rates = [k for k, v in ctl.items() if not v["passe"]]
    for k in rates:
        journal("check %s %s: %s" % ("FAILED" if ctl[k]["bloquant"] else "to look at", k, ctl[k]["valeur"]))
    return ctl, base


def regarder(sortie, Tc, Pt):
    """Grey contact sheets: corps.png (front, three-quarter, profile, back), tete.png (close-up)."""
    import view as VU
    vues = [VU.VUES[k] for k in ("face", "trois", "profil", "dos")]
    VU.planche(Tc["pos"] * CM, Tc["faces"], vues, os.path.join(sortie, "corps.png"), T=700)
    Tt = PMX.vers_tableaux(Pt)
    h = next(np.asarray(o["pos"]) * CM for o in Pt["os"] if o["nom"] == "頭")
    VU.planche(Tt["pos"] * CM, Tt["faces"], vues, os.path.join(sortie, "tete.png"), T=700,
               boite=[-14, h[1] - 12, -14, 14, h[1] + 20, 14], fil=True)


def _lire(chemin):
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _ecrire_json(chemin, d):
    with open(chemin + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(chemin + ".tmp", chemin)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modele", help=".vrm file (VRM 0.x or 1.0, VRoid Studio or VRoid Hub), or name of an already imported VRoid model")
    p.add_argument("--output", dest="sortie", default=os.path.join(R.chemin("paths.data"), "base-vroid"))
    p.add_argument("--no-sheets", dest="sans_planches", action="store_true")
    a = p.parse_args(argv)
    ctl, base = fabriquer(a.modele, a.sortie, planches=not a.sans_planches)
    if any(not v["passe"] and v["bloquant"] for v in ctl.values()):
        print("VRoid base: a blocking check failed (see %s); this base must not be used." % os.path.join(a.sortie, "controles.json"))
        return 1
    corps = os.path.join(a.sortie, "corps.pmx")
    print("base body: %s (with tete.pmx and coiffure.pmx: face and hairstyle of \"%s\")" % (corps, base["titre"]))
    if os.path.realpath(corps) != os.path.realpath(R.chemin("workshop.body")):
        print("Chimera reads another body (%s). For this one: \"atelier\": {\"corps\": \"%s\", \"cache\": <a cache of its own>} "
              "in settings.json" % (R.chemin("workshop.body"), corps))
    else:
        print("Next: python3 -I tools/workshop/catalog.py prepare, then bin/chimera")
    print("Its outfit in the catalogue: python3 -I tools/vroid/importer.py <file.vrm>, then the libraries (COMPATIBILITY.md, VRoid).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

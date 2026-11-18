"""Real assembly of a combination (a single Blender process) with the project's tools:
coiffure.poser on the base head, the hairstyle fitted on the chosen face's model (already sewn onto the
common body, see preparer.faire_tete), porter part by part, recolouring, layers, measurement, render.

    blender -b --factory-startup --python assembler_blender.py -- --dossier D
D/selection.json describes the choice; D/etat.json tracks progress (read by the page).
"""
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun as C  # noqa: E402

import pmx as PMX  # noqa: E402
import pmx_ecrire  # noqa: E402

BLOQUANTS = ("traversee_repos", "traversee_repos_p90_mm", "poses_corps_seul")
ETAPES = ["coiffure", "greffe", "pieces", "couleur", "controles", "squelette", "vrm", "rendu"]
D, ETAT = None, {}


def etat(**kw):
    ETAT.update(kw)
    C.ecrire_json(os.path.join(D, "etat.json"), ETAT)


def recolorer(chemin, sel, cheveux):
    """Hair tint: same computation as the page's shader (outils/atelier/page/visionneuse.js)."""
    import bpy
    P = PMX.lire(chemin)
    cheveux = set(cheveux)
    t = sel["teinte"]
    tr, tg, tb = t["r"], t["g"], t["b"]
    vcible = max(tr, tg, tb)
    scible = 0.0 if vcible <= 0 else 1 - min(tr, tg, tb) / vcible
    # target tint as saturated RGB, recomposed on each pixel
    pur = np.array([tr, tg, tb]) / max(vcible, 1e-6)
    ref = max(float(t.get("ref") or 0.5), 1e-3)
    faites = {}
    for m in P["materiaux"]:
        if m["nom"] not in cheveux or m["texture"] < 0:
            continue
        i = m["texture"]
        if i not in faites:
            rel = P["textures"][i].replace("\\", "/")
            src = os.path.join(os.path.dirname(chemin), rel)
            im = bpy.data.images.load(src, check_existing=False)
            w, h = im.size
            px = np.empty(w * h * 4, np.float32)
            im.pixels.foreach_get(px)
            px = px.reshape(h, w, 4)
            v = px[..., :3].max(2)
            v2 = np.minimum(v * vcible / ref, 1.0)
            rgb = (v2[..., None] * pur) * scible + v2[..., None] * (1 - scible)
            px[..., :3] = rgb
            nom, _ = os.path.splitext(rel)
            nouv = nom + "_teinte.png"
            out = bpy.data.images.new("t", w, h, alpha=True)
            out.pixels.foreach_set(px.ravel())
            tmp = os.path.join(os.path.dirname(chemin), nouv)
            out.filepath_raw = tmp
            out.file_format = "PNG"
            out.save()
            bpy.data.images.remove(out)
            bpy.data.images.remove(im)
            P["textures"].append(nouv.replace("/", "\\"))
            faites[i] = len(P["textures"]) - 1
        m["texture"] = faites[i]
    if faites:
        pmx_ecrire.ecrire(P, chemin + ".tmp.pmx")
        os.replace(chemin + ".tmp.pmx", chemin)
    return len(faites)


def garder_morphes(modele, chemin):
    """greffe_tete keeps only the morphs of what it brings (here, hair): those of the face's model
    (expressions) come back as they are. Its vertices, materials and bones stayed first, in order."""
    M, P = PMX.lire(modele), PMX.lire(chemin)
    n = len(M["sommets"]["pos"])
    if not np.array_equal(np.asarray(P["sommets"]["pos"][:n]), np.asarray(M["sommets"]["pos"])) or \
       [m["nom"] for m in P["materiaux"][:len(M["materiaux"])]] != [m["nom"] for m in M["materiaux"]] or \
       [o["nom"] for o in P["os"][:len(M["os"])]] != [o["nom"] for o in M["os"]]:
        raise RuntimeError("the face's model was renumbered by fitting the hair: expressions lost")
    P["morphes"] = list(M["morphes"]) + list(P["morphes"])
    pmx_ecrire.ecrire(P, chemin + ".tmp.pmx")
    os.replace(chemin + ".tmp.pmx", chemin)
    return len(M["morphes"])


def main():
    global D
    a = sys.argv[sys.argv.index("--") + 1:]
    D = a[a.index("--dossier") + 1]
    sel = C.lire_json(os.path.join(D, "selection.json"))
    etat(etape="coiffure", etapes=ETAPES, debut=time.time(), fini=False)
    fiches = {i: C.ranger(C.lire_json(os.path.join(C.CACHE, "pieces", i, "meta.json"))) for i in sel["pieces"]}
    inconnues = [i for i, m in fiches.items() if not m]
    if inconnues:   # the server only sends catalogue ids; a hand-written selection may not (refused before any work)
        raise ValueError("unknown pieces (not in the cache): %s" % ", ".join(repr(i) for i in inconnues))
    biblio = os.path.join(C.CACHE, "biblio")
    import coiffure as CO
    tm = C.lire_json(os.path.join(C.CACHE, "tetes", sel["tete"], "meta.json"))
    d_co = os.path.join(D, "1-coiffure")
    cm = C.lire_json(os.path.join(C.CACHE, "coiffures", sel["coiffure"], "meta.json")) or {}
    if cm.get("libre"):      # free hairstyle: made for the free head, already in place; nothing to pose
        cheveux = cm["pmx"]
        mats_cheveux = [m["nom"] for m in PMX.lire(cheveux)["materiaux"] if m["n_indices"] > 0]
    else:
        CO.poser(biblio, sel["coiffure"], tm["biblio"], d_co)
        cheveux = os.path.join(d_co, "resultat.pmx")
        Pc = PMX.lire(cheveux)
        # the materials posed by coiffure.poser are those the original head did not have
        # (their suffix comes from the source's name, not from the catalogue id)
        import coiffures as CF
        origine = {m["nom"] for m in PMX.lire(CF.BI.lire_fiche(biblio, tm["biblio"])["pmx"])["materiaux"]}
        mats_cheveux = [m["nom"] for m in Pc["materiaux"] if m["n_indices"] > 0 and m["nom"] not in origine]
    etat(etape="greffe")
    d_gr = os.path.join(D, "2-greffe")
    # only the hair goes onto the face's model: the head, its neck seam and the skin are not touched
    C.greffer(["--corps", C.absolu(tm["modele"]), "--tete", cheveux, "--sortie", d_gr, "--prendre", ",".join(mats_cheveux),
               "--retirer", "", "--garder-crane", "--sans-teinte", "--ecart-cou-max", "10",
               # the hair was posed on the game's base head (its skeleton), it is worn on our body (longer neck,
               # forme_maison MANIERE["cou"]): strands that hang on the neck or shoulders follow the body's bones
               "--suivre-os"])
    courant = os.path.join(d_gr, "resultat.pmx")
    print("face expressions:", garder_morphes(C.absolu(tm["modele"]), courant), flush=True)
    import porter as PO
    infos = []
    ordre = sorted(sel["pieces"], key=lambda i: fiches[i].get("rang", 50))
    etat(etape="pieces", pieces_total=len(ordre), piece_courante=0)
    # each part is fitted on the skin alone, whole; the outfit is then settled in one block, like the preview
    import combinaison as CB
    ecartees = CB.retirees({i: fiches[i] for i in ordre})
    ordre = [i for i in ordre if i not in ecartees]
    etat(pieces_total=len(ordre))
    cachee = set()
    for k, ident in enumerate(ordre):
        m = fiches[ident]
        d_p = os.path.join(D, "3-pieces", "%02d" % k)
        # a shoe with a heel is worn on feet posed as in the preview (its pose found when it was prepared)
        res = PO.porter(C.absolu(m["dossier"]), courant, d_p, nom="e", marge_mm=3.0, peau="noter", surface="peau",
                        pied=m.get("pied"))
        courant = os.path.join(d_p, "e.pmx")
        cachee.update(res["peau_cachee"])
        infos.append({"id": ident, "mats": res["mats"], "type_mesure": res["type_mesure"], "nom": "%s · %s" % (m["perso"], m["type_fr"]),
                      "type": m["type"], "enveloppant": C.enveloppant(m), "meta": m,
                      "pied": res.get("pied")})
        etat(piece_courante=k + 1)
    # the final folder holds the last state (PMX + textures)
    final = os.path.join(D, "modele")
    if os.path.isdir(final):
        shutil.rmtree(final)
    shutil.copytree(os.path.dirname(courant), final)
    chemin = os.path.join(final, "resultat.pmx")
    os.replace(os.path.join(final, os.path.basename(courant)), chemin)
    for n in ("portage.json", "greffe.json"):
        if os.path.exists(os.path.join(final, n)):
            os.remove(os.path.join(final, n))
    P = PMX.lire(chemin)
    retirer_faces(P, cachee)
    couches, notes, retirees = CB.resoudre_pmx(P, infos, mats_cheveux)
    # the outfit was settled in the rest frame; with a posed shoe, the feet now take its pose (body and
    # every part, by their weights) and the whole model is raised onto its sole, as in the preview
    # (the height is the one of this fit: the shoe was fitted again over what is already worn)
    pose = next((i["pied"] for i in infos if i.get("pied")), None)
    if pose:
        import pied as PD
        PD.poser_pmx(P, {k: v for k, v in pose.items() if isinstance(v, dict)})
        PD.lever_pmx(P, float(pose.get("lever", 0.0)) / PO.UNITE)
        print("feet posed in the shoe: %s, raised %.1f cm" % (
            ", ".join("%s %.0f/%.0f deg" % (k, v["cheville"], v["orteils"]) for k, v in pose.items() if isinstance(v, dict)),
            100 * float(pose.get("lever", 0.0))), flush=True)
    for m in P["materiaux"]:   # no black outline (the author's choice): not in MMD, not on the contact sheet, not in the VRM
        m["drapeaux"] &= ~0x10
    pmx_ecrire.ecrire(P, chemin + ".tmp.pmx")
    os.replace(chemin + ".tmp.pmx", chemin)
    notes = list(ecartees.values()) + notes
    print("layers: %d faces hidden; %s" % (retirees, "; ".join(notes)), flush=True)
    etat(etape="couleur")
    if sel.get("teinte"):
        recolorer(chemin, sel, mats_cheveux)
    etat(etape="controles")
    rapport = controles(modele_controle(chemin, os.path.join(d_gr, "resultat.pmx")), infos, CB.signaler(couches),
                        {c.id: CB.part_rentree(c) for c in couches})
    rapport["notes"] = notes
    C.ecrire_json(os.path.join(D, "controles.json"), rapport)
    etat(etape="squelette")
    squelette = squelette_maison(chemin)
    etat(etape="vrm")
    vrm_ok, vrm_note = sortir_vrm(chemin, sel, squelette)
    etat(etape="rendu")
    planche_note = None
    try:                       # the model is already written: an image failure does not make it fail
        import rendu
        sys.argv = ["x", "--", "--pmx", modele_rendu(chemin, mats_cheveux), "--sortie", os.path.join(D, "rendus"), "--vues", "face,trois_quarts,dos",
                    "--echantillons", "16", "--taille", "720", "--planche", "planche.png", "--mat", "肌", "--cadre", "corps"]
        rendu.main()
        C.lier(os.path.join(D, "rendus", "planche.png"), os.path.join(D, "planche.png"))
    except (Exception, SystemExit) as e:
        planche_note = repr(e)[:300]
    etat(etape="fin", planche=planche_note is None, planche_note=planche_note, fini=True, ok=rapport["ok"], duree=round(time.time() - ETAT["debut"], 1),
         resume=rapport["resume"], echecs=rapport["echecs"], couches=rapport["couches"], traverse=rapport["traverse"], notes=notes,
         vrm=vrm_ok, vrm_note=vrm_note, squelette=squelette)


def squelette_maison(chemin):
    """Replaces in resultat.pmx the ZZZ body's bones and weights with the house skeleton (outils/corps_maison);
    the version on the ZZZ bones stays next to it (resultat-zzz.pmx). Runs after the checks, which read the ZZZ bones.
    Only on the remade body, the one this skeleton weights; atelier.squelette = zzz (ATELIER_SQUELETTE) keeps the ZZZ bones.
    Returns "maison" or "zzz". A failure here leaves the model on the ZZZ bones."""
    if C.R.valeur("atelier.squelette", "maison") != "maison":
        return "zzz"
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "corps_maison"))
        import brancher as BR
        import mesures as ME
        if not os.path.exists(os.path.join(BR.MAISON, "squelette.json")):
            print("house skeleton missing (%s): ZZZ bones kept" % BR.MAISON, flush=True)
            return "zzz"
        if not os.path.samefile(C.CORPS, os.path.join(os.path.dirname(ME.FORME), "corps.pmx")):
            print("house skeleton: made for the remade body, not for %s; ZZZ bones kept" % C.CORPS, flush=True)
            return "zzz"
        Pm = BR.convertir(PMX.lire(chemin), journal=lambda *a: print(*a, flush=True))
        zzz = os.path.join(os.path.dirname(chemin), "resultat-zzz.pmx")
        shutil.copyfile(chemin, zzz)
        pmx_ecrire.ecrire(Pm, chemin + ".tmp.pmx")
        os.replace(chemin + ".tmp.pmx", chemin)
        return "maison"
    except Exception:
        traceback.print_exc()
        return "zzz"


def sortir_vrm(chemin, sel, squelette="zzz"):
    """Writes modele/resultat.vrm next to the PMX. A failure here does not prevent delivering the PMX."""
    import traceback
    import vrm as VR
    sortie = os.path.join(os.path.dirname(chemin), "resultat.vrm")
    try:
        P = PMX.lire(chemin)
        torsion = None
        if squelette == "maison":
            import brancher as BR
            with open(os.path.join(BR.MAISON, "squelette.json"), encoding="utf-8") as f:
                torsion = {o["nom"]: o["torsion"] for o in json.load(f)["os"] if o["torsion"]}
        r = VR.ecrire_vrm(P, sortie, VR.images_blender(P, os.path.dirname(chemin)), nom="%s · %s" % (sel["tete"], sel["coiffure"]),
                          torsion=torsion)
        defauts = VR.valider(sortie)
        print("vrm:", r, "defects:", defauts, flush=True)
        return not defauts, "; ".join(defauts)
    except Exception as e:
        traceback.print_exc()
        return False, "%s: %s" % (type(e).__name__, e)


def retirer_faces(P, indices):
    """Removes from the PMX the faces with the given indices (numbering of P["faces"])."""
    if not indices:
        return
    k, faces = 0, []
    for m in P["materiaux"]:
        n = m["n_indices"] // 3
        reste = [f for j, f in enumerate(P["faces"][k:k + n], k) if j not in indices]
        m["n_indices"] = len(reste) * 3
        faces += reste
        k += n
    P["faces"] = faces


def modele_controle(chemin, greffe):
    """Copy of the model with all its skin: porter removes the hidden skin, and mesure.py would
    otherwise read the holes as clipping. Same vertices, only the faces of 肌 come back."""
    P, G = PMX.lire(chemin), PMX.lire(greffe)
    if P["materiaux"][0]["nom"] != "肌" or G["materiaux"][0]["nom"] != "肌":
        return chemin
    na, ng = P["materiaux"][0]["n_indices"] // 3, G["materiaux"][0]["n_indices"] // 3
    P["faces"] = list(G["faces"][:ng]) + list(P["faces"][na:])
    P["materiaux"][0]["n_indices"] = ng * 3
    return copie_voisine(P, chemin, "controle")


def copie_voisine(P, chemin, nom):
    """Writes P to D/<nom>/resultat.pmx, with a link to the model's textures."""
    d = os.path.join(D, nom)
    os.makedirs(d, exist_ok=True)
    lien = os.path.join(d, "tex")
    if not os.path.lexists(lien):
        os.symlink(os.path.join(os.path.dirname(chemin), "tex"), lien)
    sortie = os.path.join(d, "resultat.pmx")
    pmx_ecrire.ecrire(P, sortie + ".tmp.pmx")
    os.replace(sortie + ".tmp.pmx", sortie)
    return sortie


def modele_rendu(chemin, cheveux):
    """Copy for the contact sheet, without the "髮+" highlight layers: they repeat exactly
    the hair's faces, and rendu.py's outline (Solidify) then covers the whole hairstyle
    in black. The delivered PMX keeps these layers; MMD shows them normally."""
    P = PMX.lire(chemin)
    reflets = {m["nom"] for m in P["materiaux"] if m["nom"] in set(cheveux) and "+" in m["nom"] and m["n_indices"] > 0}
    if not reflets:
        return chemin
    faces, k = [], 0
    for m in P["materiaux"]:
        n = m["n_indices"] // 3
        if m["nom"] in reflets:
            m["n_indices"] = 0
        else:
            faces += list(P["faces"][k:k + n])
        k += n
    P["faces"] = faces
    return copie_voisine(P, chemin, "rendu")


def controles(chemin, infos, signal, caches=None):
    """Layers between parts (same computation as the page), mesure.py checks on each part."""
    import combinaison as CB
    import modele as MD
    import mesure
    M = None
    echecs, details = [], []
    cps = [{"dessous": r["nom_dessous"], "dessus": r["nom_dessus"], "sommets": r["sommets"], "sortis": r["part"], "pire_mm": r["pire_mm"]}
           for r in signal]
    for inf in infos:
        spec = {"id": inf["id"], "perso": inf["id"], "type": {"culotte": "short"}.get(inf["type"], inf["type_mesure"]), "mats": inf["mats"],
                "calques": [], "pourquoi": "atelier"}
        M = M or MD.Modele("assemblage", chemin=chemin)
        try:
            r = mesure.mesurer(spec, M, True)
        except Exception as e:
            echecs.append("%s: measurement impossible (%s)" % (inf["nom"], str(e)[:80]))
            continue
        ctr = [{"nom": x["nom"], "valeur": x["valeur"], "seuil": x["seuil"], "ok": x["ok"]} for x in r["controles"]]
        details.append({"piece": inf["nom"], "controles": ctr})
        for x in ctr:
            if x["ok"] is False and x["nom"] == "traversee_repos" and not CB.traversee_visible(x["valeur"], (caches or {}).get(inf["id"], 0.0)):
                x["ok"] = None   # in excess only where another part covers this one
            if x["ok"] is False and x["nom"] in BLOQUANTS:
                echecs.append("%s: %s = %s (threshold %s)" % (inf["nom"], x["nom"], x["valeur"], x["seuil"]))
    relu = PMX.lire(chemin)
    n_s, n_m = len(relu["sommets"]["pos"]), len(relu["materiaux"])
    traverse = [c for c in cps if c["sortis"] > 0.01 and c["sortis"] * c["sommets"] >= 6]
    return {"ok": not echecs, "echecs": echecs, "couches": cps, "traverse": traverse, "detail": details,
            "resume": "%d vertices, %d materials, %d bones; %d parts measured, %d layer pairs, %d clippings" % (
                n_s, n_m, len(relu["os"]), len(details), len(cps), len(traverse))}


if __name__ == "__main__":
    try:
        main()
    except (Exception, SystemExit) as e:   # the project's tools refuse by SystemExit (greffe_tete): the page reads why
        traceback.print_exc()
        etat(etape="erreur", fini=True, ok=False,
             erreur=(str(e) if isinstance(e, SystemExit) else "%s: %s" % (type(e).__name__, e))[:400])
        sys.exit(1)

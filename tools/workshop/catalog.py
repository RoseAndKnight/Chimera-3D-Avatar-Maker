"""The workshop's catalogue: preparing the cache, adding a part, summary.

    python3 -I tools/workshop/catalog.py prepare [--level small|full] [--stale] [--order A,B,…]
    python3 -I tools/workshop/catalog.py add <part folder in library format>
    python3 -I tools/workshop/catalog.py summary

`preparer` only redoes what is missing (meta.json present and source unchanged): running it again
resumes where it stopped. The heavy work (Blender) goes through compute.launcher in the settings (CHIMERA_LAUNCHER) if it is set.
"""
import glob
import json
import itertools
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# Faces and hairstyles offered: the house's, then the active game's ("catalogue" in their descriptions,
# tools/games). Each face is sewn into the oval of the base head (C.TETE_BASE, itself in the catalogue).
MAISON = C.jeux.maison()
PAR_LOT = 25
EN_PARALLELE = C.places()
_NUMERO = itertools.count()
# heads computed separately (no folder in the face library): biblio -> (pmx, displayed name)
# (those whose file is not on this machine are ignored: they are ours, not in the repository)
ENTREES_PROPRES = {n: (MAISON.developper(e["pmx"]), e["nom"]) for n, e in MAISON.valeur("tetes").items()
                   if not n.startswith("_") and os.path.isfile(MAISON.developper(e["pmx"]))}
# parts made in the workshop (house description, "fabriquees"), likewise
FABRIQUEES = [{"genre": "fabriquer", **f, "pmx": MAISON.developper(f["pmx"])} for f in MAISON.valeur("fabriquees")
              if os.path.isfile(MAISON.developper(f["pmx"]))]
# Free head and hairstyle (tools/free_base/free_face.py), next to the free base body: without a base head
# from the game, these are what the workshop offers, and the page opens without any game model.
LIBRE = "Libre"
TETE_LIBRE = os.path.join(os.path.dirname(C.CORPS), "tete.pmx")
COIFFURE_LIBRE = os.path.join(os.path.dirname(C.CORPS), "coiffure.pmx")


# base.json next to the body (tools/vroid_base): the measured base head (tete_base). Without it (free MakeHuman
# base), no measured head.
BASE_JSON = C.lire_json(os.path.join(os.path.dirname(C.CORPS), "base.json")) or {}
# Displayed names of the base's own head and hairstyle: their role, in English, translated by the page (workshop.js,
# SERVEUR). Never the name of the model the base was made from: it is in another language than the page (a VRoid
# title in Japanese) and says nothing of the role. Names written by an older base.json ("noms") are ignored.
NOMS_BASE = ({"tete": "Base face", "coiffure": "Base hairstyle"} if BASE_JSON.get("tete_base")
             else {"tete": "Free head", "coiffure": "Free bob"})


def fiche_tete_base():
    """Record of the game's base head in the face catalogue; a path that does not exist when the game has none."""
    return os.path.join(C.BIBLIO_VIS, C.TETE_BASE or "(no base head)", "fiche.json")


def base_libre():
    """Is (or will) the base of this cache be the free head? A base already sewn keeps its nature."""
    m = C.lire_json(os.path.join(C.BASE, "meta.json"))
    if m is not None:
        return bool(m.get("libre"))
    return not os.path.isfile(fiche_tete_base()) and os.path.isfile(TETE_LIBRE) and os.path.isfile(COIFFURE_LIBRE)


def base_ouverte():
    """The base of this cache wears its own head (base_libre) and it is a character head, measured by
    tools/vroid_base (base.json: tete_base): face transfer and hairstyle posing can read it.
    It then also receives the faces and hairstyles of imported models, next to its own. The free MakeHuman
    head (tools/free_base) has no such measurement: its cache offers only that head."""
    return base_libre() and bool(BASE_JSON.get("tete_base"))


def cle_base():
    """Name, in the cache's biblio/, of the head on which all faces are sewn: the base's (LIBRE, read from
    tete.pmx next to the body), otherwise the active game's base head."""
    return LIBRE if base_libre() else C.TETE_BASE


def catalogue(cle, petit):
    """Names offered ("tetes" or "coiffures"): the house, then the active game ("toutes": see coiffures_du_jeu).
    On a base that wears its own head: that one (LIBRE) first, then, if the base is open, the game's, except the
    character the base was made from (its head and hairstyle are already LIBRE)."""
    if base_libre() and not base_ouverte():
        return [LIBRE]
    res = []
    for J in (MAISON, C.JEU):
        v = J.valeur("catalogue.petit." + cle if petit else "catalogue." + cle, [])
        res += coiffures_du_jeu() if v == "toutes" else [n for n in v if J is not MAISON or n in ENTREES_PROPRES]
    if base_libre():
        res = [LIBRE] + [n for n in res if n not in (LIBRE, BASE_JSON["tete_base"].get("nom"))]
    return res


def fiches_bibliotheque():
    """Every part folder of the library whose type/kind has a place in the workshop."""
    res = []
    for f in sorted(glob.glob(os.path.join(C.BIBLIO_VET, "*", "*", "fiche.json"))):
        with open(f, encoding="utf-8") as fh:
            fi = json.load(fh)
        if fi.get("classe") == "vetement" and fi.get("type") in C.VETEMENTS or \
           fi.get("classe") == "accessoire" and fi.get("sorte") in C.ACCESSOIRES:
            res.append((os.path.dirname(f), fi))
    return res


def preparer_biblio(noms, propres=None):
    """Face library entries in the cache: links to the library, own computed entries
    (`propres`: name -> (pmx, displayed name), ENTREES_PROPRES by default). The base's head (LIBRE) is read
    again if its file has changed."""
    import library as BI
    propres = ENTREES_PROPRES if propres is None else propres
    d = os.path.join(C.CACHE, "biblio")
    os.makedirs(d, exist_ok=True)
    for n in noms:
        cible = os.path.join(d, n)
        if n in propres:
            pmx = propres[n][0]
            emp = C.empreinte_fichier(pmx) if n == LIBRE else None
            if not os.path.exists(os.path.join(cible, "partie.npz")) or \
               emp is not None and (C.lire_json(os.path.join(cible, "fiche.json")) or {}).get("empreinte") != emp:
                os.makedirs(cible, exist_ok=True)
                print("reading head", n, flush=True)
                BI.extraire_un((n, pmx, d))
            C.ecrire_json(os.path.join(cible, "fiche.json"), dict({"personnage": n, "pmx": pmx}, **({"empreinte": emp} if emp else {})))
        elif not os.path.exists(cible):
            os.symlink(os.path.join(C.BIBLIO_VIS, n), cible)
    return d


def a_faire_meta(chemin, empreinte=None):
    m = C.lire_json(chemin)
    # heads that failed on a zero-weight bone must be redone: that defect is fixed (C.greffer)
    return m is None or "KeyError" in str(m.get("raison", "")) or (empreinte is not None and m.get("empreinte") != empreinte)


def perimee(m):
    """Part fitted by an older method or on a body other than today's."""
    return bool(m) and (m.get("v_portage") != C.V_PORTAGE or m.get("corps") != C.empreinte_fichier(C.CORPS))


def coiffures_du_jeu():
    """Every head of the face library whose hairstyle has been extracted and whose outfit has been cut."""
    return sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(C.BIBLIO_VIS, "*", "partie.npz"))
                  if os.path.isfile(os.path.join(os.path.dirname(f), "coiffure.json"))
                  and os.path.isdir(os.path.join(C.BIBLIO_VET, os.path.basename(os.path.dirname(f)))))


def plan(niveau):
    petit = niveau == "petit"
    tetes = catalogue("tetes", petit)
    coiffures = catalogue("coiffures", petit)
    libre, base = base_libre(), cle_base()
    propres = dict(ENTREES_PROPRES, **({LIBRE: (TETE_LIBRE, NOMS_BASE["tete"])} if libre else {}))
    # the base alone needs no entry (nor the pivot head): its hairstyle is made for its head
    seule = libre and set(tetes) | set(coiffures) <= {LIBRE}
    biblio = preparer_biblio([] if seule else sorted((set(tetes) | set(coiffures) | {base}) - {None}), propres)
    travail = []
    if a_faire_meta(os.path.join(C.CACHE, "corps", "meta.json"), C.empreinte_fichier(C.CORPS)):
        travail.append({"genre": "corps", "id": "corps"})
    if not os.path.isfile(os.path.join(C.BASE, "meta.json")):
        travail.append({"genre": "base", "id": "base", "libre": True, "tete": TETE_LIBRE} if libre else
                       {"genre": "base", "id": "base", "tete": C.lire_json(fiche_tete_base())["pmx"]})
    infos = []
    for n in tetes:
        pmx = propres[n][0] if n in propres else json.load(open(os.path.join(C.BIBLIO_VIS, n, "fiche.json"), encoding="utf-8"))["pmx"]
        emp = C.empreinte_fichier(pmx)
        infos.append((n, pmx, emp))
        m_t = C.lire_json(os.path.join(C.CACHE, "tetes", n, "meta.json")) or {}
        ir, bo = C.iris(n), C.bouche(n)
        ir, bo = (C.empreinte_fichier(ir) if ir else None), (C.empreinte_fichier(bo) if bo else None)
        commun = {"genre": "tete", "id": n, "biblio": n, "pmx": pmx, "libre": libre and n == LIBRE, "base_biblio": base,
                  "nom": propres[n][1] if n in propres else n}
        if a_faire_meta(os.path.join(C.CACHE, "tetes", n, "meta.json"), emp) or (m_t.get("retenue") and m_t.get("v_peau") != C.V_PEAU):
            travail.append(commun)
        elif m_t.get("retenue") and (m_t.get("iris") != ir or m_t.get("bouche") != bo or m_t.get("biblio") != base):
            # the chosen iris or mouth, or the base head's name in biblio/, has changed: the head keeps its
            # transferred face, only tete.glb and its record are rewritten
            travail.append(dict(commun, garder_modele=True))
    for f in FABRIQUEES:
        fiche = os.path.join(C.CACHE, "fabriquees", f["perso"], f["id"], "fiche.json")
        m = C.lire_json(fiche)
        if m is None or m.get("empreinte") != C.empreinte_fichier(f["pmx"]):
            travail.append(f)
    return travail, tetes, coiffures, biblio, infos


def plan_suite(niveau, tetes, coiffures, biblio, perimees=False):
    """What depends on the heads and the made parts (their records must exist)."""
    travail = []
    tt = []
    for n in tetes:
        m = C.lire_json(os.path.join(C.CACHE, "tetes", n, "meta.json")) or {}
        if m.get("retenue"):
            tt.append({"id": n, "biblio": m["biblio"], "translation": m["translation"]})
    version = "v%d:" % C.V_COIFFURE
    sig = version + ",".join(sorted(t["id"] for t in tt))
    for a in coiffures:
        m = C.lire_json(os.path.join(C.CACHE, "coiffures", a, "meta.json"))
        if a == LIBRE and base_libre():
            e = sig + ":" + str(C.empreinte_fichier(COIFFURE_LIBRE))
            if m is None or m.get("empreinte") != e:
                travail.append({"genre": "coiffure_libre", "id": a, "nom": NOMS_BASE["coiffure"], "pmx": COIFFURE_LIBRE,
                                "tetes": [t["id"] for t in tt], "empreinte": e})
        elif m is None or m.get("empreinte") != sig:
            propre = a if any(t["id"] == a for t in tt) else None
            nom = ENTREES_PROPRES[a][1] if a in ENTREES_PROPRES else a
            it = {"genre": "coiffure", "id": a, "nom": nom, "biblio": biblio, "tetes": tt, "propre": propre, "empreinte": sig}
            # Heads were added since this hairstyle was posed (an extraction adds models lot by lot): pose it on the
            # new heads only and keep the pairs already measured, instead of posing it again on every head.
            deja = (m or {}).get("paires")
            if isinstance(deja, dict) and deja and str(m.get("empreinte") or "").startswith(version) and \
               os.path.isfile(os.path.join(C.CACHE, "coiffures", a, "coiffure.glb")):
                garder = sorted(t["id"] for t in tt if t["id"] in deja
                                and os.path.isfile(os.path.join(C.CACHE, "coiffures", a, t["id"] + ".bin")))
                if garder:
                    it = dict(it, tetes=[t for t in tt if t["id"] not in garder], garder=garder)
            travail.append(it)
    dossiers = [(os.path.join(C.CACHE, "fabriquees", f["perso"], f["id"]), f["perso"], f["id"]) for f in FABRIQUEES]
    if niveau == "petit":
        pris = {}
        for d, fi in fiches_bibliotheque():
            k = fi.get("type") or fi.get("sorte")
            if pris.get(k, 0) < 2 and k in ("jupe", "haut", "chaussures", "ceinture"):
                pris[k] = pris.get(k, 0) + 1
                dossiers.append((d, fi["perso"], fi["id"]))
    else:
        dossiers += [(d, fi["perso"], fi["id"]) for d, fi in fiches_bibliotheque()]
    for d, perso, ident in dossiers:
        cid = C.cle(perso, ident)
        fiche = C.lire_json(os.path.join(d, "fiche.json")) or {}
        src = os.path.join(d, fiche.get("pmx", ""))
        # the same source file as preparer.preparer_piece, whose fingerprint the record keeps: the part with its
        # trims when it has some (otherwise every trimmed part would be fitted again at each preparation)
        garnie = os.path.join(d, (fiche.get("garnie") or {}).get("pmx") or "")
        if (fiche.get("garnie") or {}).get("pmx") and os.path.exists(garnie):
            src = garnie
        chemin_meta = os.path.join(C.CACHE, "pieces", cid, "meta.json")
        if os.path.isfile(src) and (a_faire_meta(chemin_meta, C.empreinte_fichier(src)) or (perimees and perimee(C.lire_json(chemin_meta)))):
            travail.append({"genre": "piece", "id": cid, "dossier": d})
        elif (C.lire_json(chemin_meta) or {}).get("retenue") and not os.path.isfile(os.path.join(C.CACHE, "pieces", cid, "couvre.bin")):
            travail.append({"genre": "couvre", "id": cid})   # part prepared before couvre.bin existed
    return travail


def lancer_blender(liste, nom, attendre=True):
    if not liste:
        return None
    tmp = os.environ.get("TMPDIR") or C.CACHE
    os.makedirs(tmp, exist_ok=True)
    chemin = os.path.join(tmp, "atelier-%s-%d-%d.json" % (nom, os.getpid(), next(_NUMERO)))
    C.ecrire_json(chemin, liste)
    script = os.path.join(C.ICI, "prepare.py")
    cmd = ["bash", C.BLENDER_SH, "-b", "--factory-startup", "--python", script, "--", "--travail", chemin]
    lanceur = C.lanceur()
    if not attendre:
        return subprocess.Popen(lanceur + cmd)
    r = subprocess.run(lanceur + cmd)
    os.remove(chemin)
    if r.returncode:
        print("Blender returned", r.returncode, file=sys.stderr)


def transferer(it):
    """The face of it["pmx"] sewn into the base head, by the tool as is: tetes/<id>/modele."""
    import shutil
    mod = os.path.join(C.CACHE, "tetes", it["id"], "modele")
    shutil.rmtree(mod, ignore_errors=True)
    r = subprocess.run([C.PYTHON, "-I", os.path.join(C.OUTILS, "head", "retarget_head.py"), "visage", "--source", it["pmx"],
                        "--target", C.BASE_PMX, "--output", mod, "--no-sheet"], capture_output=True, text=True)
    ok = os.path.isfile(os.path.join(mod, "mesures.json"))
    if not ok:
        os.makedirs(mod, exist_ok=True)
        C.ecrire_json(os.path.join(mod, "erreur.json"), ((r.stderr or r.stdout).strip().splitlines() or ["no output"])[-1][:200])
    print("face", it["id"], "transferred" if ok else "FAILED", flush=True)


def executer(travail, nom, de_front=1, fini=None):
    """In batches of at most PAR_LOT per Blender. `de_front` > 1: the items do not depend on
    one another (parts, hairstyles), that many Blenders at once, each on one core.
    `fini(lot)`, if given, is called after each Blender batch (from its worker thread)."""
    if de_front <= 1:
        for k in range(0, len(travail), PAR_LOT):
            print("batch %s: %d to %d of %d" % (nom, k + 1, min(k + PAR_LOT, len(travail)), len(travail)), flush=True)
            lancer_blender(travail[k:k + PAR_LOT], nom)
            if fini:
                fini(travail[k:k + PAR_LOT])
        return
    # short batches, dealt in turn: the big parts do not all land in the same one
    taille = max(1, min(PAR_LOT, -(-len(travail) // (de_front * 4))))
    lots = [travail[k:k + taille] for k in range(0, len(travail), taille)]
    print("%s: %d items, %d batches of %d, %d at a time" % (nom, len(travail), len(lots), taille, de_front), flush=True)
    from concurrent.futures import ThreadPoolExecutor
    fait = itertools.count(1)

    def un(lot):
        lancer_blender(lot, nom)
        print("batch %s: %d of %d done" % (nom, next(fait), len(lots)), flush=True)
        if fini:
            fini(lot)
    with ThreadPoolExecutor(de_front) as pool:
        list(pool.map(un, lots))


class Suivi:
    """Progress lines for the workshop server (tools/workshop/inputs.py): one JSON object per line after "@@ ".
    `ordre`: the models of the current extraction lot, in the order they should become usable. Each item of the
    suite (hairstyle, part) belongs to a model or to nobody (shared work); when all the items of a model are done,
    a line "fini" names it, and its face, hairstyle and parts are in the catalogue."""
    MARQUE = "@@ "

    def __init__(self, ordre):
        import threading
        self.ordre = list(ordre or [])
        self.reste = {}                 # model -> items of the suite not done yet
        self.coiffures = set()          # models whose hairstyle is not posed yet
        self.verrou = threading.Lock()

    def dire(self, **kw):
        print(self.MARQUE + json.dumps(kw, ensure_ascii=False), flush=True)

    def modele(self, it):
        if it["genre"] == "coiffure":
            return it["id"] if it["id"] in self.ordre else None
        if it["genre"] in ("piece", "couvre"):
            perso = it["id"].split("__")[0]
            return next((n for n in self.ordre if C.cle(n) == perso), None)
        return None

    def trier(self, suite):
        """Shared work first, then model by model in the lot's order, each model's hairstyle before its pieces."""
        rang = {n: k for k, n in enumerate(self.ordre)}
        par = {it["id"]: self.modele(it) for it in suite}
        suite.sort(key=lambda it: (-1 if par[it["id"]] is None else rang[par[it["id"]]],
                                   0 if it["genre"] in ("coiffure", "coiffure_libre") else 1))
        for it in suite:
            n = par[it["id"]]
            self.reste[n] = self.reste.get(n, 0) + 1
            if it["genre"] == "coiffure" and n is not None:
                self.coiffures.add(n)
        return suite

    def debut(self):
        """Models of the lot with nothing left in the suite are usable already."""
        for n in self.ordre:
            if not self.reste.get(n):
                self.dire(fini=n)
        self.courant()

    def courant(self):
        n = next((n for n in self.ordre if self.reste.get(n)), None)
        if n is not None:
            self.dire(modele=n, pas="coiffure" if n in self.coiffures else "pieces")

    def lot_fini(self, lot):
        with self.verrou:
            for it in lot:
                n = self.modele(it)
                self.reste[n] = self.reste.get(n, 1) - 1
                if it["genre"] == "coiffure":
                    self.coiffures.discard(n)
                if n is not None and self.reste[n] == 0:
                    self.dire(fini=n)
            self.dire(fait=len(lot))
            self.courant()


def manque_pour_preparer():
    """What is missing to prepare the cache (sentences; empty: ready to run). The repository provides neither a base
    body nor a base head: without them, preparer stops here instead of failing later on a missing file."""
    res = []
    import shutil
    blender = C.R.programme("blender") or "blender"
    if not shutil.which(os.path.expanduser(blender)):
        res.append("Blender not found (\"%s\"): install it, or give its path with the setting programs.blender "
                   "(CHIMERA_BLENDER). See INSTALLATION.md." % blender)
    if not os.path.isfile(C.CORPS):
        res.append("No base body: %s does not exist (setting workshop.body, CHIMERA_BODY). "
                   "Build it from a VRoid Studio character: python3 -I tools/vroid_base/build.py <modele.vrm> "
                   "(or, without a VRM, the free base: python3 -I tools/free_base/build.py; see INSTALLATION.md, "
                   "\"What the repository does not provide\")." % C.CORPS)
    fiche = fiche_tete_base()
    if not os.path.isfile(os.path.join(C.BASE, "meta.json")) and not os.path.isfile(fiche) and not base_libre():   # plan() reads this record
        res.append("No base head: neither the base's own (%s and coiffure.pmx, written next to the body by "
                   "tools/vroid_base/build.py or tools/free_base/build.py), nor the active game's \"%s\" (%s, head \"%s\": add the game's models, "
                   "see COMPATIBILITY.md)." % (TETE_LIBRE, C.JEU.nom, fiche, C.TETE_BASE))
    if base_libre() and not base_ouverte() and os.path.isfile(fiche) and os.path.isfile(os.path.join(C.BASE, "meta.json")):
        print("Note: this cache was prepared with the free head and keeps it. To switch to the faces of the game \"%s\", "
              "give the workshop another cache (setting workshop.cache)." % C.JEU.nom, file=sys.stderr)
    return res


def a_faire(cat, corps=None, cache=None):
    """What is missing for the workshop page to open, as sentences for the user (empty: everything is there).
    The page shows the cache's body (corps/corps.glb), then a face and a hairstyle from the catalogue: all three are needed."""
    corps, cache = corps or C.CORPS, cache or C.CACHE
    res = []
    if not os.path.isfile(os.path.join(cache, "corps", "corps.glb")):
        if not os.path.isfile(corps):
            res.append("No base body: %s does not exist (setting workshop.body). Build it from a VRoid Studio "
                       "character: python3 -I tools/vroid_base/build.py <modele.vrm> (or, without a VRM, the free "
                       "base: python3 -I tools/free_base/build.py; see INSTALLATION.md, \"What the repository "
                       "does not provide\")." % corps)
        else:
            res.append("The base body is not prepared for the page yet: "
                       "python3 -I tools/workshop/catalog.py prepare")
    if not cat.get("tetes") or not cat.get("coiffures"):
        res.append("Empty catalogue (%d face(s), %d hairstyle(s) kept in %s): the page needs at least one "
                   "face and one hairstyle. Add your models (COMPATIBILITY.md), set paths.data, then "
                   "python3 -I tools/workshop/catalog.py prepare"
                   % (len(cat.get("tetes") or []), len(cat.get("coiffures") or []), cache))
    return res


def preparer(niveau, perimees=False, ordre=None):
    """`ordre` (the server's extraction lot): models to make usable one after the other, with progress lines
    (Suivi) on the standard output."""
    manque = manque_pour_preparer()
    if manque:
        sys.exit("catalog.py prepare: nothing started.\n- " + "\n- ".join(manque))
    suivi = Suivi(ordre) if ordre else None
    travail, tetes, coiffures, biblio, _ = plan(niveau)
    executer([it for it in travail if it["genre"] != "tete"], "base")
    if any(it["genre"] == "base" for it in travail) and not os.path.isfile(C.BASE_PMX):
        sys.exit("catalog.py prepare: the base (body and base head) could not be made, the rest is not started. "
                 "See above what Blender said.")
    # one face = a transfer of a few minutes, sharing nothing with the others: EN_PARALLELE at a time,
    # outside the Blender queue (the tool takes a place in it itself for its tint)
    t = [it for it in travail if it["genre"] == "tete"]
    if suivi and t:
        suivi.dire(pas="transfert", tetes=[it["id"] for it in t])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(EN_PARALLELE) as pool:
        list(pool.map(transferer, [it for it in t if it["biblio"] != C.TETE_BASE and not it.get("garder_modele")
                                   and not it.get("libre")]))
    executer(t, "tetes")
    if suivi:
        suivi.dire(tetes_faites=[it["id"] for it in t])
    suite = plan_suite(niveau, tetes, coiffures, biblio, perimees)
    if suivi:
        suivi.trier(suite)
        suivi.dire(suite=len(suite))
        suivi.debut()
    else:
        # hairstyles first: they are what the page waits for to show anything
        suite.sort(key=lambda it: 0 if it["genre"] == "coiffure" else 1)
    executer(suite, "suite", EN_PARALLELE, fini=suivi.lot_fini if suivi else None)
    construire()


def _raison(r):
    if not r:
        return "unknown"
    if "KeyError" in r:
        return "to prepare again: zero-weight bone mishandled by the graft (fixed, run preparer again)"
    return r


def construire(ecrire=True):
    """Writes catalogue.json from the cache's records (ecrire=False: only returns it)."""
    def fiches(sous):
        out = []
        for f in sorted(glob.glob(os.path.join(C.CACHE, sous, "*", "meta.json"))):
            try:      # a damaged record (interrupted write) drops its part, not the whole workshop
                m = C.lire_json(f)
                if not isinstance(m, dict) or "id" not in m:
                    raise ValueError("record without an id")
                out.append(m)
            except (ValueError, OSError) as e:
                print("unreadable record, ignored: %s (%s)" % (f, e), file=sys.stderr, flush=True)
        return out

    tetes, coiffures, pieces = fiches("tetes"), fiches("coiffures"), fiches("pieces")
    for m, cle in [(m, "tete") for m in tetes] + [(m, "coiffure") for m in coiffures]:
        if m["id"] == LIBRE:      # the base's own: today's name, not the one written when it was prepared
            m["nom"] = NOMS_BASE[cle]
    for m in pieces:      # tab, slot, rank: those of today's settings, not of the day it was prepared
        C.ranger(m)
        n = (m.get("revue") or {}).get("note")
        if m.get("retenue") and n is not None and n > C.NOTE_REFUS:   # the record keeps retenue: only the catalogue drops it
            m["retenue"], m["raison"] = False, "shape damaged by fitting (review score %s, limit %s)" % (n, C.NOTE_REFUS)
    for m in coiffures:   # the rule may have changed since posing: it is read again from the pairs
        if "paires" in m:
            m["retenue"], m["raison"] = C.coiffure_retenue(m["paires"], m.get("propre"))
    refus = []
    for genre, liste in (("face", tetes), ("hairstyle", coiffures), ("part", pieces)):
        for m in liste:
            if not m.get("retenue"):
                refus.append({"genre": genre, "id": m["id"], "nom": _nom(m), "raison": _raison(m.get("raison"))})
    def vignette(sous, m):
        v = "/data/%s/%s/vignette.png" % (sous, m["id"])
        return v if os.path.isfile(os.path.join(C.CACHE, sous, m["id"], "vignette.png")) else None
    etq = etiquettes()

    def etiquette(sous, m):
        return etq.get("%s/%s" % (sous, m["id"]))

    cat = {"tetes": [{"id": m["id"], "nom": m["nom"], "peau": m.get("peau"), "vignette": vignette("tetes", m), "faces": m["faces"],
                      "etiquette": etiquette("tetes", m)}
                     for m in tetes if m.get("retenue")],
           "coiffures": [{"id": m["id"], "nom": m["nom"], "propre": m.get("propre"), "faces": m["faces"],
                          "vignette": vignette("coiffures", m), "etiquette": etiquette("coiffures", m),
                          "paires": {t: {"passe": p["passe"], "echecs": p["echecs"]} for t, p in m["paires"].items()}}
                         for m in coiffures if m.get("retenue")],
           "pieces": [{k: m.get(k) for k in ("id", "perso", "piece", "type", "type_fr", "onglet", "emplacement", "rang", "occupe",
                                             "faces", "classe", "peau_cachee", "autres_echecs")}
                      | {"nom": _nom(m), "vignette": m.get("vignette") or vignette("pieces", m), "etiquette": etiquette("pieces", m)}
                      for m in pieces if m.get("retenue")],
           "refus": refus}
    src = sources()      # only the models in use, and only pages over https
    noms = {m["id"] for m in cat["tetes"] + cat["coiffures"]} | {m.get("perso") for m in cat["pieces"]}
    cat["sources"] = {n: v for n, v in src.items() if n in noms and isinstance(v, dict) and str(v.get("url", "")).startswith("https://")}
    cat["nombres"] = {"preparees": len(tetes) + len(coiffures) + len(pieces),
                      "retenues": len(cat["tetes"]) + len(cat["coiffures"]) + len(cat["pieces"]), "refusees": len(refus),
                      "tetes": [len(cat["tetes"]), len(tetes)], "coiffures": [len(cat["coiffures"]), len(coiffures)],
                      "pieces": [len(cat["pieces"]), len(pieces)]}
    if ecrire:
        C.ecrire_json(os.path.join(C.CACHE, "catalogue.json"), cat)
    return cat


_ANNEXES = {}


def _annexe(nom):
    """An optional JSON dictionary of the cache, read again when the file changes ({} without it)."""
    f = os.path.join(C.CACHE, nom)
    try:
        cle = (os.path.getmtime(f), os.path.getsize(f))
    except OSError:
        return {}
    if _ANNEXES.get(nom, (None,))[0] != cle:
        d = C.lire_json(f)
        _ANNEXES[nom] = (cle, d if isinstance(d, dict) else {})
    return _ANNEXES[nom][1]


def etiquettes():
    """Readable labels of the items, from etiquettes.json in the cache (optional):
    {"tetes/<id>" | "coiffures/<id>" | "pieces/<id>": {"nom": [en, fr], "description": [en, fr],
    "tags": [[en…], [fr…]]}}. Without a label an item keeps its code name."""
    return _annexe("etiquettes.json")


def sources():
    """Where each model comes from, from sources.json in the cache (optional): {"<model name>": {"url": page of
    the original model, "nom", "auteur", "site", "releve": day the terms were read, "licence": {VRM licence
    fields as read that day, or "nom" and "licenseUrl" for a named licence such as CC0}}}. The name is a part's `perso`, or the id of a face or hairstyle. The page
    links every item to it, so that the current terms can be read again at the source."""
    return _annexe("sources.json")


def _nom(m):
    if "perso" in m:
        return "%s · %s" % (m["perso"], m.get("type_fr") or m["piece"])
    return m.get("nom", m["id"])


def bilan():
    cat = construire()
    n = cat["nombres"]
    print("prepared %d, kept %d, rejected %d" % (n["preparees"], n["retenues"], n["refusees"]))
    for k in ("tetes", "coiffures", "pieces"):
        print(" ", k, "%d kept of %d" % tuple(n[k]))
    raisons = {}
    for r in cat["refus"]:
        cle = r["raison"].split(" = ")[0].split(":")[0].strip()
        raisons[cle] = raisons.get(cle, 0) + 1
    for k, v in sorted(raisons.items(), key=lambda x: -x[1]):
        print("  %4d  %s" % (v, k))
    metas = [C.lire_json(f) or {} for f in glob.glob(os.path.join(C.CACHE, "pieces", "*", "meta.json"))]
    vieux = sum(perimee(m) for m in metas)
    if vieux:
        print("  %d outdated parts of %d (other fitting method or other body): preparer --stale redoes them" % (vieux, len(metas)))
    nm = sum(bool(m.get("non_mesures")) for m in metas)
    if nm:
        print("  %d parts kept with a blocking check not measured (non_mesures in their record)" % nm)
    for m in a_faire(cat):
        print("  to do:", m)


def ajouter(dossier):
    dossier = os.path.abspath(dossier.rstrip("/"))
    fi = C.lire_json(os.path.join(dossier, "fiche.json"))
    if fi is None:
        sys.exit("no fiche.json in %s" % dossier)
    if not (fi.get("classe") == "vetement" and fi.get("type") in C.VETEMENTS or
            fi.get("classe") == "accessoire" and fi.get("sorte") in C.ACCESSOIRES):
        sys.exit("part type not supported by the workshop: %s" % (fi.get("type") or fi.get("sorte")))
    cid = C.cle(fi["perso"], fi["id"])
    lancer_blender([{"genre": "piece", "id": cid, "dossier": dossier}], "ajout")
    m = C.lire_json(os.path.join(C.CACHE, "pieces", cid, "meta.json"))
    construire()
    print("%s: %s" % (cid, "added to the catalogue" if m and m.get("retenue") else "rejected: %s" % (m or {}).get("raison")))


NIVEAUX = {"small": "petit", "full": "complet"}   # --level -> level name of the game descriptions (catalogue.petit)


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("prepare", "add", "summary"):
        sys.exit(__doc__)
    if a[0] == "prepare":
        niveau = a[a.index("--level") + 1] if "--level" in a else "full"
        if niveau not in NIVEAUX:
            sys.exit("--level: small or full")
        preparer(NIVEAUX[niveau], "--stale" in a,
                 [n for n in a[a.index("--order") + 1].split(",") if n] if "--order" in a else None)
    elif a[0] == "add":
        ajouter(a[1])
    else:
        bilan()


if __name__ == "__main__":
    main()

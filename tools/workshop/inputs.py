"""Inputs and outputs of the workshop, for the page's buttons: what the tool accepts and produces, importing a
model, extracting its parts into the catalogue, and downloading an assembled character.

What is said here comes from the code and from the game descriptions (tools/games/<game>/game.json), not written for
the page:
  - inputs: a VRoid VRM (0.x from VRoid Studio, 1.0 from VRoid Hub) goes through the VRoid importer (tools/vroid/importer.py); a PMX must already speak the common
    language (face skin "颜", bone "頭"…);
  - importing is only possible if the active game takes its models from a folder where everyone brings their own
    ("modeles": "*" in its description): a game with a declared list (ZZZ) only sees the models of its list;
  - extraction follows the chain of COMPATIBILITY.md: pivot head (if missing), faces, hairstyles, clothes, then
    catalog.py prepare (Blender through the bin/blender-queue queue, settings calcul.*).

Received files are untrusted data: the archive is checked before it is unpacked, and model files are only read by an
isolated Python (-I) started from the repository.

    python3 -I tools/workshop/inputs.py formats          # what GET /api/formats would say
    python3 -I tools/workshop/inputs.py lire-pmx <pmx>   # (internal) materials, bones and textures of a PMX, as JSON
"""
import glob
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import unicodedata
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

J = C.JEU
DONNEES = C.ATLAS
PIVOT = os.path.join(DONNEES, "tete-pivot")      # where tools/head/library.py looks for it
EXTRACTIONS = os.path.join(C.CACHE, "extractions")
ASSEMBLAGES = os.path.join(C.CACHE, "assemblages")
ATTENTE = os.path.join(C.CACHE, "imports-en-cours")

IMPORT_MAX = 4 << 30             # most bytes received (a VRM is 10 to 100 MB; an archive of several models, GBs)
ZIP_DEPLIE_MAX = 12 << 30        # most bytes once unpacked
ZIP_FICHIERS_MAX = 20000
ZIP_RAPPORT_MAX = 200            # unpacked / compressed size beyond which a large file is a bomb
NOM_MAX = 60
# the VRM importer is written for one game (its bone, material and morph tables live there): tools/vroid
IMPORTEUR_VRM = ("vroid", os.path.join(C.OUTILS, "vroid", "importer.py"))
QUOI = ("visages", "coiffures", "vetements")


class Refus(Exception):
    """Rejection the user can read; `code`: HTTP status."""

    def __init__(self, message, code=400):
        super().__init__(message)
        self.code = code


# ===================================================================== formats

def _compat(jeu):
    return jeu.d.get("compatibilite", {})


def _va_au_bout(jeu):
    """Can the game's models go all the way to the catalogue? It needs a list of models and a base head
    (pivot.py and catalog.py read them); without them the game cannot be the active game."""
    return "modeles" in jeu.d and "tete_base" in jeu.d


def import_possible():
    """(possible, reason): importing writes into the active game's mmd_extrait source, which only a "*" game reads."""
    if J.valeur("modeles", {}) != "*":
        return False, ("The active game \"%s\" reads a declared list of models (\"modeles\" in its description): an "
                       "imported model would never be picked up. Place its files as COMPATIBILITY.md says, or make "
                       "active a game where everyone brings their own models (setting games.active)." % J.titre)
    try:
        J.chemin("mmd_extrait")
    except KeyError:
        return False, "The active game \"%s\" has no mmd_extrait folder." % J.titre
    return True, None


def formats():
    import games as jeux
    ok, raison = import_possible()
    vroid = jeux.charger(IMPORTEUR_VRM[0])
    cv = _compat(vroid)
    lim = cv.get("limites", [])
    entree = [
        {"id": "vrm", "nom": "VRM 0.x or 1.0 (VRoid Studio, VRoid Hub)", "extensions": [".vrm"], "etat": cv.get("niveau", "partiel"),
         "importable": ok,
         "note": " ".join(lim[:3]) or cv.get("resume", "")},
        {"id": "pmx", "nom": "PMX (MikuMikuDance), in a .zip with its textures (one or more models per archive)", "extensions": [".zip"], "etat": "partiel",
         "importable": ok,
         "note": "Names must follow the common language of ZZZ's official MMD models (face skin \"颜\", "
                 "hair \"髮\", skin \"肌\", MMD bone \"頭\"): a PMX without them is rejected at import. "
                 "An archive may hold several models: each PMX that follows the common language is imported, the others "
                 "(a weapon, a prop) are set aside."},
    ]
    sortie = [
        {"id": "pmx", "nom": "PMX (MikuMikuDance)",
         "note": "The assembled model and its textures, as a .zip. No physics (no rigid bodies for the hair, no springs)."},
        {"id": "vrm", "nom": "VRM 1.0",
         "note": "Humanoid bones, 13 expressions, MToon materials, in T-pose. A VRM is a binary glTF file: renamed "
                 "to .glb, a glTF viewer opens it (without the VRM extensions). Missing if writing it failed."},
    ]
    # Only the skeletons the workshop reads are listed (games that can go all the way to the catalogue).
    squelettes = []
    for nom in jeux.tous():
        if nom == "homemade":
            continue
        jeu = jeux.charger(nom)
        c = _compat(jeu)
        if _va_au_bout(jeu):
            vrm = "os_mmd" in jeu.d
            squelettes.append({
                "id": nom, "nom": "%s: %s" % (jeu.titre, "VRM humanoid" if vrm else "standard MMD bones"),
                "etat": c.get("niveau", "partiel"),
                "note": ("Humanoid bones renamed to MMD bones at import, breast bones merged into their parent, springs "
                         "not kept." if vrm else
                         "Bones of the MMD standard (センター, 上半身, 頭…) and materials of the common language, no conversion.")})
    sq = C.R.valeur("workshop.skeleton", "custom")
    return {"entree": entree, "sortie": sortie, "squelettes": squelettes,
            "squelette_sortie": {"id": sq, "note": "Bones with readable names (house skeleton) when its data exists, "
                                                   "otherwise the bones of the base body." if sq in ("custom", "maison") else
                                                   "The bones of the base body, kept as they are."},
            "jeu_actif": {"id": J.nom, "nom": J.titre, "import": ok, "raison": raison}}


# ===================================================================== noms

def nettoyer_nom(texte, maxi=NOM_MAX):
    """Name given by the user, kept readable (letters of any language, digits, space, - _ . ') and bounded."""
    if not isinstance(texte, str):
        return ""
    t = unicodedata.normalize("NFC", texte)
    t = "".join(c if (c.isalnum() or c in " -_.'") else " " for c in t)
    return re.sub(r"\s+", " ", t).strip(" .")[:maxi].strip()


def identifiant(nom_fichier):
    """Identifier of a model taken from its file name: ASCII, safe for a folder and a URL."""
    base = os.path.basename(str(nom_fichier).replace("\\", "/"))
    tige = os.path.splitext(base)[0]
    a = unicodedata.normalize("NFKD", tige).encode("ascii", "ignore").decode()
    a = re.sub(r"[^A-Za-z0-9_-]+", "_", a).strip("_-")[:48]
    if not a or not a[0].isalnum():
        raise Refus("Unreadable file name: rename the file with Latin letters or digits.")
    return a


def nom_telechargement(nom, defaut):
    """(ASCII name, UTF-8 name) for Content-Disposition."""
    n = nettoyer_nom(nom) or defaut
    a = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    a = re.sub(r"[^A-Za-z0-9_.-]+", "_", a).strip("_.") or defaut
    return a, n


# ===================================================================== import

def recevoir(flux, longueur, maxi=IMPORT_MAX):
    """Writes the received body into a waiting file (never whole in memory). Returns its path."""
    if longueur <= 0:
        raise Refus("Empty file.")
    if longueur > maxi:
        raise Refus("File too large (%d MB at most)." % (maxi >> 20), 413)
    os.makedirs(ATTENTE, exist_ok=True)
    chemin = os.path.join(ATTENTE, "recu-%s" % secrets.token_hex(8))
    reste = longueur
    with open(chemin, "wb") as f:
        while reste:
            b = flux.read(min(reste, 1 << 20))
            if not b:
                break
            f.write(b)
            reste -= len(b)
    if reste:
        os.remove(chemin)
        raise Refus("Upload interrupted.")
    return chemin


def entete_vrm(chemin):
    """Checks, without converting anything, that the file is a VRM 0.x or 1.x. Returns the declared title (or None).
    Whether it is a VRoid character is said by the importer, which reads its materials."""
    import struct
    with open(chemin, "rb") as f:
        tete = f.read(20)
        if len(tete) < 20 or tete[:4] != b"glTF":
            raise Refus("This file is not a VRM (nor a binary glTF).")
        version, _, n, genre = struct.unpack_from("<III4s", tete, 4)
        if version != 2 or genre != b"JSON" or n > 64 << 20:
            raise Refus("This file is not a readable VRM (glTF 2.0 expected).")
        try:
            j = json.loads(f.read(n))
        except ValueError:
            raise Refus("This file is not a readable VRM: damaged header.")
    ext = j.get("extensions") or {}
    if "VRM" in ext:
        return ((ext["VRM"].get("meta") or {}).get("title")) or None
    if "VRMC_vrm" in ext:
        v = ext["VRMC_vrm"] if isinstance(ext["VRMC_vrm"], dict) else {}
        if not str(v.get("specVersion", "")).startswith("1."):
            raise Refus("This file declares a VRM version other than 0.x and 1.x: it is not read.")
        return ((v.get("meta") or {}).get("name")) or None
    raise Refus("This file is a glTF without the VRM extension: only a VRM from VRoid (Studio or Hub) is read.")


def _sous(racine, rel):
    c = os.path.realpath(os.path.join(racine, rel))
    if not c.startswith(os.path.realpath(racine) + os.sep):
        raise Refus("Archive rejected: a file escapes its folder (%s)." % rel[:80])
    return c


def _noms_zip(infos):
    """The real name of each file of an archive. An archive that does not declare UTF-8 is read as cp437 by
    zipfile, which damages the Chinese or Japanese texture names (the PMX no longer finds them): the bytes are taken
    back and the first encoding that reads them all is kept (UTF-8, then Shift-JIS, then GB18030)."""
    bruts = [None if i.flag_bits & 0x800 else i.filename.encode("cp437", "replace") for i in infos]
    if all(b is None or b.isascii() for b in bruts):
        return [i.filename for i in infos]
    for codage in ("utf-8", "cp932", "gb18030"):
        try:
            return [i.filename if b is None else b.decode(codage) for i, b in zip(infos, bruts)]
        except UnicodeDecodeError:
            continue
    return [i.filename for i in infos]


def deplier_zip(chemin, dossier):
    """Unpacks a received archive into `dossier` (new), with its guards: paths that escape, links, encrypted
    archive, too many files, too large, compression ratio of a bomb. Returns the relative paths of its PMX files
    (an archive may carry several models, or a character and its weapon)."""
    try:
        z = zipfile.ZipFile(chemin)
    except (zipfile.BadZipFile, OSError):
        raise Refus("This file is not a readable .zip archive.")
    with z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) > ZIP_FICHIERS_MAX:
            raise Refus("Archive rejected: more than %d files." % ZIP_FICHIERS_MAX)
        total = 0
        noms = _noms_zip(infos)
        for i, n in zip(infos, noms):
            if i.flag_bits & 0x1:
                raise Refus("Encrypted archive: unpack it yourself and make a .zip again without a password.")
            if stat.S_ISLNK(i.external_attr >> 16):
                raise Refus("Archive rejected: it contains a link (%s)." % n[:80])
            if n.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", n) or ".." in re.split(r"[\\/]", n) or "\0" in n:
                raise Refus("Archive rejected: a file escapes its folder (%s)." % n[:80])
            total += i.file_size
            if i.file_size > (1 << 20) and i.file_size > ZIP_RAPPORT_MAX * max(1, i.compress_size):
                raise Refus("Archive rejected: abnormal compression (%s)." % n[:80])
        if total > ZIP_DEPLIE_MAX:
            raise Refus("Archive rejected: more than %d GB once unpacked." % (ZIP_DEPLIE_MAX >> 30))
        pmx = [n for n in noms if n.lower().endswith(".pmx")]
        if not pmx:
            raise Refus("The archive contains no .pmx file.")
        os.makedirs(dossier)
        ecrit = 0
        for i, n in zip(infos, noms):
            cible = _sous(dossier, n)
            os.makedirs(os.path.dirname(cible), exist_ok=True)
            with z.open(i) as src, open(cible, "wb") as dst:
                lu = 0
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    lu += len(b)
                    ecrit += len(b)
                    if lu > i.file_size or ecrit > ZIP_DEPLIE_MAX:      # declared size is a lie
                        raise Refus("Archive rejected: a file is larger than it declares.")
                    dst.write(b)
    return sorted(pmx)


def lire_pmx_isole(chemin):
    """Materials, bones and textures of a PMX, read by an isolated Python: a booby-trapped file does not touch the server."""
    r = subprocess.run([C.PYTHON, "-I", os.path.abspath(__file__), "lire-pmx", chemin], cwd=C.DEPOT,
                       capture_output=True, text=True, timeout=300)
    if r.returncode:
        raise Refus("This .pmx file is not readable: %s" % ((r.stderr.strip().splitlines() or ["?"])[-1][:200]))
    return json.loads(r.stdout)


def verifier_langue_commune(info):
    """What the core requires of a PMX (tools/games/README.md, "common language"): the face skin 颜 (tete.Tete
    measures the head on it) and the head bone 頭."""
    manque = []
    if "颜" not in info["materiaux"]:
        manque.append("no material \"颜\" (face skin)")
    if "頭" not in info["os"]:
        manque.append("no bone \"頭\" (head)")
    if manque:
        raise Refus("This PMX does not follow the workshop's common language: %s. See tools/games/README.md." % ", ".join(manque))


def importer(chemin_recu, nom_fichier):
    """Stores a received file as a model of the active game: <mmd_extrait>/<id>/<id>.pmx and its textures. Returns
    the record of the first model stored. The received file is deleted in every case."""
    return importer_lot(chemin_recu, nom_fichier)[0][0]


def importer_lot(chemin_recu, nom_fichier):
    """Like importer, for an archive that may carry several models: returns (records, set aside). A set-aside file is
    {"fichier", "raison"}: a .pmx of the archive that is not a readable character (a weapon), or already imported."""
    try:
        return _importer(chemin_recu, nom_fichier)
    finally:
        if os.path.exists(chemin_recu):
            os.remove(chemin_recu)


def _fiche(dest, ident, titre, fmt, info, manquantes):
    fiche = {"id": ident, "nom": titre or ident, "format": fmt, "materiaux": info["materiaux"], "os": len(info["os"]),
             "textures": len(info["textures"]), "sommets": info["sommets"]}
    if manquantes:
        fiche["textures_manquantes"] = manquantes[:20]
    C.ecrire_json(os.path.join(dest, "import.json"), {k: fiche[k] for k in ("id", "nom", "format")} | {"date": time.time()})
    return fiche


def _idents_du_lot(ident, bons):
    """One identifier per model of an archive that carries several: the name of its folder if it is alone there,
    otherwise the name of its file, otherwise the archive's followed by a number. No duplicate within the batch."""
    par_dossier = {}
    for rel, _, _ in bons:
        par_dossier[os.path.dirname(rel)] = par_dossier.get(os.path.dirname(rel), 0) + 1
    pris, noms = set(), []
    for k, (rel, _, _) in enumerate(bons):
        d = os.path.dirname(rel)
        candidats = ([os.path.basename(d)] if d and par_dossier[d] == 1 else []) + [os.path.basename(rel)]
        if d and par_dossier[d] > 1:
            candidats.insert(0, "%s-%s" % (os.path.basename(d), os.path.splitext(os.path.basename(rel))[0]))
        nom = None
        for c in candidats:
            try:
                nom = identifiant(c + ".x") if not c.lower().endswith(".pmx") else identifiant(c)
            except Refus:
                continue
            if nom not in pris:
                break
            nom = None
        if nom is None:
            n = k + 1
            while "%s_%d" % (ident, n) in pris:
                n += 1
            nom = "%s_%d" % (ident, n)
        pris.add(nom)
        noms.append(nom)
    return noms


def _importer(chemin_recu, nom_fichier):
    ok, raison = import_possible()
    if not ok:
        raise Refus(raison, 409)
    ident = identifiant(nom_fichier)
    ext = os.path.splitext(str(nom_fichier))[1].lower()
    if ext not in (".vrm", ".zip"):
        raise Refus("Unsupported format (%s): a .vrm (VRoid Studio or VRoid Hub) or a .zip containing a .pmx."
                    % (ext or "no extension"), 415)
    dest_racine = J.chemin("mmd_extrait")
    dest = os.path.join(dest_racine, ident)
    deja = "A model \"%s\" is already imported: rename the file to make another one."
    if ext == ".vrm" and os.path.exists(dest):
        raise Refus(deja % ident, 409)
    os.makedirs(dest_racine, exist_ok=True)
    travail = os.path.join(ATTENTE, "import-%s" % secrets.token_hex(8))
    fiches, ecartes = [], []
    try:
        if ext == ".vrm":
            titre = entete_vrm(chemin_recu)
            os.makedirs(travail)
            vrm = os.path.join(travail, ident + ".vrm")
            shutil.move(chemin_recu, vrm)
            sortie = os.path.join(travail, "modele")
            r = subprocess.run([C.PYTHON, "-I", IMPORTEUR_VRM[1], vrm, "--output", sortie], cwd=C.DEPOT,
                               capture_output=True, text=True, timeout=900)
            if r.returncode or not os.path.isfile(os.path.join(sortie, ident, ident + ".pmx")):
                d = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ["no message"])[-1]
                if r.returncode == 2 and d.startswith("refused: "):     # a VRM the importer does not read, with its reason
                    raise Refus("This VRM is not read: %s." % d[len("refused: "):].rstrip(".")[:300], 422)
                d = re.sub(r"^\w+Error: ", "", d).replace(vrm, "the file")
                raise Refus("The VRM conversion failed: %s" % d[:240], 422)
            info = lire_pmx_isole(os.path.join(sortie, ident, ident + ".pmx"))
            shutil.move(os.path.join(sortie, ident), dest)
            try:            # the original .vrm stored in the game's "vrm" source, if it has one
                rang = J.chemin("vrm")
                os.makedirs(rang, exist_ok=True)
                if not os.path.exists(os.path.join(rang, ident + ".vrm")):
                    shutil.move(vrm, os.path.join(rang, ident + ".vrm"))
            except KeyError:
                pass
            fiches.append(_fiche(dest, ident, titre, "vrm", info, []))
        else:
            deplie = os.path.join(travail, "deplie")
            os.makedirs(travail)
            bons = []
            for rel in deplier_zip(chemin_recu, deplie):
                pmx = os.path.join(deplie, rel)
                try:
                    info = lire_pmx_isole(pmx)
                    verifier_langue_commune(info)
                except Refus as e:
                    ecartes.append({"fichier": rel, "raison": str(e)})
                    continue
                bons.append((rel, pmx, info))
            if not bons:        # nothing readable: the reason of the first file (an archive of a single PMX gives its own)
                raise Refus(ecartes[0]["raison"] if len(ecartes) == 1 else
                            "None of the %d .pmx files of the archive is a readable character. The first one: %s"
                            % (len(ecartes), ecartes[0]["raison"]))
            noms = [ident] if len(bons) == 1 else _idents_du_lot(ident, bons)
            dossiers = {os.path.dirname(p) for _, p, _ in bons}
            for nom, (rel, pmx, info) in zip(noms, bons):
                cible = os.path.join(dest_racine, nom)
                if os.path.exists(cible):
                    if len(bons) == 1:
                        raise Refus(deja % nom, 409)
                    ecartes.append({"fichier": rel, "raison": "A model \"%s\" is already imported." % nom})
                    continue
                # the PMX's folder becomes the model's (textures are named relative to it); the other PMX files
                # and the folders of the archive's other models are not copied into it
                racine = os.path.dirname(pmx)

                def hors(d, noms_, racine=racine, pmx=pmx):
                    return [n for n in noms_ if (n.lower().endswith(".pmx") and os.path.join(d, n) != pmx)
                            or (os.path.join(d, n) in dossiers and os.path.join(d, n) != racine)]
                shutil.copytree(racine, cible, ignore=hors)
                os.replace(os.path.join(cible, os.path.basename(pmx)), os.path.join(cible, nom + ".pmx"))
                manquantes = [t for t in info["textures"] if t and not os.path.isfile(
                    os.path.join(cible, t.replace("\\", "/")))]
                fiches.append(_fiche(cible, nom, None, "pmx", info, manquantes))
            if not fiches:
                raise Refus("Every model of this archive is already imported.", 409)
    finally:
        shutil.rmtree(travail, ignore_errors=True)
    return fiches, ecartes


# ===================================================================== state of the models

def _index_visages():
    return C.lire_json(os.path.join(C.BIBLIO_VIS, "index.json"), {}) or {}


def etat_modeles(cat=None):
    """Each model of the active game: what was extracted from it (libraries), kept in the catalogue, and rejected."""
    import catalog as K
    cat = cat if cat is not None else K.construire(ecrire=False)
    idx = _index_visages()
    non_fournies = idx.get("non_fournies", {})
    controle = C.lire_json(os.path.join(C.BIBLIO_VET, "controle.json"), {}) or {}
    tetes = {t["id"] for t in cat.get("tetes", [])}
    coiffures = {c["id"] for c in cat.get("coiffures", [])}
    retenues = {}
    for p in cat.get("pieces", []):
        retenues[p.get("perso")] = retenues.get(p.get("perso"), 0) + 1
    refus_cat = cat.get("refus", [])
    places = K.fiches_bibliotheque()      # parts whose kind has a place in the workshop
    try:
        racine = J.chemin("mmd_extrait")
    except KeyError:
        racine = None
    res = []
    for n, rel in sorted(J.modeles().items()):
        imp = C.lire_json(os.path.join(racine, n, "import.json"), {}) if racine else {}
        vet = os.path.join(C.BIBLIO_VET, n)
        refus = []
        if n in non_fournies:
            refus.append({"nom": "face", "raison": str(non_fournies[n].get("raison", ""))[:300]})
        if isinstance(controle.get(n), dict) and controle[n].get("erreur"):
            refus.append({"nom": "clothes", "raison": "cutting failed: %s" % controle[n]["erreur"][:250]})
        prises = {os.path.realpath(d) for d, _ in places}
        for f in sorted(glob.glob(os.path.join(vet, "*", "fiche.json"))):
            if os.path.realpath(os.path.dirname(f)) not in prises:
                fi = C.lire_json(f, {}) or {}
                refus.append({"nom": "part %s" % os.path.basename(os.path.dirname(f)),
                              "raison": "the workshop does not offer this kind of part on its own (%s)"
                                        % (fi.get("type") or fi.get("sorte") or fi.get("classe") or "no kind")})
        pref = C.cle(n) + "__"
        for r in refus_cat:
            if r["id"] == n or r["id"].startswith(pref):
                refus.append({"nom": "%s %s" % (r["genre"], r["nom"]), "raison": r["raison"]})
        res.append({
            "id": n, "nom": (imp or {}).get("nom") or n, "format": (imp or {}).get("format") or J.d.get("format", "pmx"),
            "present": bool(racine) and os.path.isfile(os.path.join(racine, rel)),
            "extrait": {"visage": os.path.isfile(os.path.join(C.BIBLIO_VIS, n, "partie.npz")),
                        "coiffure": os.path.isfile(os.path.join(C.BIBLIO_VIS, n, "coiffure.json")),
                        "pieces": len(glob.glob(os.path.join(vet, "*", "fiche.json")))},
            "catalogue": {"visage": n in tetes, "coiffure": n in coiffures, "pieces": retenues.get(n, 0)},
            "refus": refus})
    return res


# ===================================================================== extraction

_extraction = threading.Lock()
EN_COURS = {"id": None, "travail": None}     # the running extraction (id, Extraction object), None when idle


class Annulee(Exception):
    """The extraction was cancelled from the page (POST /api/extraction/annuler)."""


def _nouvel_id(dossier, prefixe=""):
    base, k = prefixe + time.strftime("%Y%m%d-%H%M%S"), 1
    ident = base
    while os.path.exists(os.path.join(dossier, ident)):
        k += 1
        ident = "%s-%d" % (base, k)
    os.makedirs(os.path.join(dossier, ident))
    return ident


def demande_extraction(v):
    """Validated body of POST /api/extraire: (model names, what, redo)."""
    if not isinstance(v, dict):
        raise Refus("Unreadable request.")
    connus = J.modeles()
    if v.get("tous") is True:
        noms = sorted(connus)
    else:
        noms = v.get("modeles")
        if not isinstance(noms, list) or not noms or not all(isinstance(n, str) for n in noms):
            raise Refus("Choose at least one model (\"modeles\") or all of them (\"tous\": true).")
        inconnus = [n for n in noms if n not in connus]
        if inconnus:
            raise Refus("Model unknown to the active game: %s." % ", ".join(n[:40] for n in inconnus[:5]))
        noms = sorted(set(noms))
    if not noms:
        raise Refus("No model in the active game: import one first.")
    quoi = v.get("quoi", list(QUOI))
    if not isinstance(quoi, list) or not quoi or any(q not in QUOI for q in quoi):
        raise Refus("\"quoi\": a list among %s." % ", ".join(QUOI))
    return noms, [q for q in QUOI if q in quoi], v.get("refaire") is True


def lancer_extraction(noms, quoi, refaire=False, executer=True):
    """Starts the chain in the background; one at a time. Returns the job's identifier.
    The extraction runs in the server process, not in the page: closing the tab does not stop it."""
    if not _extraction.acquire(blocking=False):
        raise Refus("An extraction is already running (%s): wait for it to finish." % EN_COURS["id"], 409)
    try:
        os.makedirs(EXTRACTIONS, exist_ok=True)
        ident = _nouvel_id(EXTRACTIONS, "x")
        d = os.path.join(EXTRACTIONS, ident)
        C.ecrire_json(os.path.join(d, "demande.json"), {"modeles": noms, "quoi": quoi, "refaire": refaire})
        C.ecrire_json(os.path.join(d, "etat.json"), {"etape": "attente", "part": 0.0, "message": "Waiting…", "fini": False})
        travail = Extraction(d, noms, quoi, refaire)
        EN_COURS.update(id=ident, travail=travail)
    except BaseException:
        _extraction.release()
        raise
    if not executer:          # tests: nothing is started
        EN_COURS.update(id=None, travail=None)
        _extraction.release()
        return ident

    def fil():
        try:
            travail.mener()
        finally:
            EN_COURS.update(id=None, travail=None)
            _extraction.release()
    threading.Thread(target=fil, daemon=True).start()
    return ident


def extraction_en_cours():
    """GET /api/extraction: the running extraction, if any, with its state ({"id": None} when idle)."""
    ident = EN_COURS["id"]
    if ident is None:
        return {"id": None}
    e = lire_travail(ident) or {"etape": "attente", "fini": False}
    return dict(e, id=ident)


def annuler_extraction():
    """POST /api/extraction/annuler: stop the running extraction and the programs it launched. Returns its
    identifier; the state becomes "annulee" once its programs have stopped (Extraction.mener)."""
    travail, ident = EN_COURS["travail"], EN_COURS["id"]
    if travail is None:
        raise Refus("No extraction is running.", 409)
    travail.annuler()
    return ident


class Extraction:
    # Lots of C.places() models (compute.slots): each lot goes through the whole chain (faces, hairstyles,
    # clothes, catalogue) before the next one starts, and inside a lot the catalogue makes the models usable one
    # after the other. A lot of `places` models keeps every place busy in the long parallel steps (face
    # sampling, face transfer, pieces); fixed costs per lot are small (Blender starts in under a second).
    ARRET = 8            # seconds between SIGTERM and SIGKILL when cancelling

    def __init__(self, dossier, noms, quoi, refaire):
        self.d, self.noms, self.quoi, self.refaire = dossier, noms, quoi, refaire
        self.etat = {"etape": "attente", "part": 0.0, "fini": False, "debut": time.time(), "modeles": noms, "quoi": quoi,
                     "finis": [], "version": 0}
        self.journal = os.path.join(dossier, "journal.txt")
        self.verrou = threading.Lock()
        self.proc = None
        self.arret = threading.Event()

    # ------------------------------------------------------------ state
    def poser(self, **kw):
        with self.verrou:
            self.etat.update(kw)
            C.ecrire_json(os.path.join(self.d, "etat.json"), self.etat)

    def note(self, texte):
        with open(self.journal, "a", encoding="utf-8") as j:
            j.write(time.strftime("%H:%M:%S ") + texte + "\n")

    def part(self, lot, dans):
        """Bar position: the pivot takes the first 5 %, then each lot an equal share; `dans` in [0, 1]."""
        n = max(1, len(self.lots))
        return round(0.05 + 0.95 * (lot + min(1.0, max(0.0, dans))) / n, 3)

    # ------------------------------------------------------------ programs
    def annuler(self):
        """Ask the running chain to stop (from the server's request thread): SIGTERM to the process group of the
        running tool, SIGKILL to what is left of it ARRET seconds later."""
        if self.arret.is_set():
            return
        self.arret.set()
        self.note("cancellation requested")
        self.poser(message="Cancelling…", annulation=True)
        p = self.proc
        if p is not None:
            self._signal(p.pid, signal.SIGTERM)
            threading.Timer(self.ARRET, self._signal, (p.pid, signal.SIGKILL)).start()

    @staticmethod
    def _signal(groupe, sig):
        try:
            os.killpg(groupe, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _attendre_groupe(self, groupe):
        """After the leader exits, its group may still hold programs (Blender under systemd-run, process pools):
        wait for them to go, then kill what is left."""
        fin = time.time() + self.ARRET
        while self._signal(groupe, 0):
            if time.time() > fin:
                self._signal(groupe, signal.SIGKILL)
                time.sleep(0.2)
                if time.time() > fin + 5:
                    break
            time.sleep(0.2)

    def commande(self, args, ligne=None):
        """Run a tool of the repository in its own process group (all the programs it starts, Blender included,
        stay in that group, systemd-run --scope too): cancelling stops the whole group. Lines starting with
        Suivi.MARQUE go to `ligne`, the others to the journal."""
        if self.arret.is_set():
            raise Annulee()
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        with open(self.journal, "ab") as j:
            j.write(("$ " + " ".join(os.path.basename(a) if a.endswith(".py") else a for a in args[2:]) + "\n").encode())
            j.flush()
            p = subprocess.Popen([C.PYTHON, "-I"] + args, cwd=C.DEPOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            self.proc = p
            if self.arret.is_set():          # cancelled between the check above and the start
                self._signal(p.pid, signal.SIGTERM)
            try:
                for brut in p.stdout:
                    if brut.startswith(b"@@ "):
                        if ligne and not self.arret.is_set():
                            try:
                                ligne(json.loads(brut[3:].decode("utf-8")))
                            except ValueError:
                                pass
                        continue
                    j.write(brut)
                    j.flush()
                rc = p.wait()
            finally:
                if self.arret.is_set():
                    self._attendre_groupe(p.pid)
                    self._nettoyer(p.pid)
                self.proc = None
        if self.arret.is_set():
            raise Annulee()
        return rc

    def _nettoyer(self, pid):
        """Job lists that catalogue.lancer_blender left behind when its Blender was stopped."""
        tmp = os.environ.get("TMPDIR") or C.CACHE
        for f in glob.glob(os.path.join(tmp, "atelier-*-%d-*.json" % pid)):
            try:
                os.remove(f)
            except OSError:
                pass

    def manque(self, test, noms=None):
        return [n for n in (self.noms if noms is None else noms) if self.refaire or not test(n)]

    # ------------------------------------------------------------ chain
    def mener(self):
        """Run the chain to its end, its failure or its cancellation; the state file always ends "fini"."""
        try:
            self.chaine()
        except Annulee:
            self.note("cancelled")
            self.poser(etape="annulee", fini=True, ok=False, annulee=True, modele=None,
                       message="Extraction cancelled. What was finished stays in the catalogue; extracting again resumes from there.",
                       duree=round(time.time() - self.etat["debut"], 1))
        except Exception as e:   # noqa: BLE001 — any failure ends the job and is reported
            self.note("error: %r" % e)
            self.poser(etape="erreur", fini=True, ok=False, erreur=str(e) if isinstance(e, Refus) else "%s: %s" % (type(e).__name__, e),
                       message="The extraction failed.")

    def chaine(self):
        """Pivot head (once, it needs every model), then the lots one after the other."""
        import catalog as K
        avant = etat_modeles()
        places = max(1, min(C.places(), 12))     # beyond 12 programs at once a desktop machine slows down
        travaux = str(places)
        outils = C.OUTILS
        self.lots = [self.noms[k:k + places] for k in range(0, len(self.noms), places)]
        self.poser(lots=len(self.lots), total=len(self.noms))
        if not os.path.isfile(os.path.join(PIVOT, "tete-pivot.json")):
            self.poser(etape="pivot", pas="pivot", message="Pivot head: the head closest to the game's average…")
            if self.commande([os.path.join(outils, "head", "pivot.py"), "--output", PIVOT, "--no-sheet",
                              "--travaux", travaux]) or not os.path.isfile(os.path.join(PIVOT, "tete-pivot.json")):
                raise Refus("The pivot head could not be computed (see the log).")
        manque = K.manque_pour_preparer()
        if manque:
            raise Refus("The catalogue cannot be prepared: " + " ".join(manque))
        for k, lot in enumerate(self.lots):
            self.lot(k, lot, travaux)
        apres = etat_modeles()
        self.poser(etape="fin", part=1.0, fini=True, ok=True, message="Extraction finished.", modele=None,
                   bilan=bilan(avant, apres, self.noms), duree=round(time.time() - self.etat["debut"], 1))

    def lot(self, k, lot, travaux):
        """One lot through the whole chain: faces, hairstyles, clothes, then the catalogue, model by model."""
        outils = C.OUTILS
        de = sum(len(x) for x in self.lots[:k])
        rangs = {n: de + i + 1 for i, n in enumerate(lot)}
        portee = {"de": de + 1, "a": de + len(lot), "total": len(self.noms), "noms": lot}

        def etape(pas, dans, message):
            self.poser(etape=pas, pas=pas, lot=portee, modele=None, part=self.part(k, dans), message=message)

        rang = "%d-%d of %d" % (portee["de"], portee["a"], portee["total"])
        # faces: prerequisite of hairstyles and clothes (both read the face library)
        v = self.manque(lambda n: os.path.isfile(os.path.join(C.BIBLIO_VIS, n, "fiche.json")), lot)
        if v:
            etape("visages", 0.0, "Models %s: faces" % rang)
            self.commande([os.path.join(outils, "head", "library.py"), "extraire", "--output", C.BIBLIO_VIS,
                           "--travaux", travaux, "--seuls", ",".join(v)])
        avec_visage = [n for n in lot if os.path.isfile(os.path.join(C.BIBLIO_VIS, n, "partie.npz"))]
        if "coiffures" in self.quoi:
            c = [n for n in avec_visage if self.refaire or not os.path.isfile(os.path.join(C.BIBLIO_VIS, n, "coiffure.json"))]
            if c:
                etape("coiffures", 0.35, "Models %s: hairstyles" % rang)
                self.commande([os.path.join(outils, "head", "hairstyle.py"), "extraire", "--library", C.BIBLIO_VIS,
                               "--travaux", travaux, "--seuls", ",".join(c)])
        if "vetements" in self.quoi:
            w = [n for n in avec_visage if self.refaire or not os.path.isfile(os.path.join(C.BIBLIO_VET, n, "decoupe.json"))]
            if w:
                etape("vetements", 0.4, "Models %s: cutting the clothes" % rang)
                self.commande([os.path.join(outils, "clothes", "library.py"), "extraire", "--output", C.BIBLIO_VET,
                               "--seuls", ",".join(w)])
        etape("catalogue", 0.45, "Models %s: fitting the faces on the base body" % rang)
        suite = {"total": 0, "fait": 0}

        def ligne(m):
            """Progress lines of catalog.py prepare (catalogue.Suivi)."""
            if "tetes_faites" in m:
                self.poser(version=self.etat["version"] + 1, part=self.part(k, 0.7))
            if "suite" in m:
                suite["total"] = m["suite"]
            if "fait" in m:
                suite["fait"] += m["fait"]
                self.poser(part=self.part(k, 0.7 + 0.3 * suite["fait"] / max(1, suite["total"])))
            if "fini" in m and m["fini"] in rangs and m["fini"] not in self.etat["finis"]:
                self.note("model ready in the catalogue: %s" % m["fini"])
                self.poser(finis=self.etat["finis"] + [m["fini"]], version=self.etat["version"] + 1)
            if "modele" in m and m["modele"] in rangs:
                pas = {"coiffure": "hairstyle", "pieces": "clothes"}.get(m.get("pas"), m.get("pas"))
                self.poser(modele={"rang": rangs[m["modele"]], "total": len(self.noms), "nom": m["modele"],
                                   "pas": m.get("pas")},
                           message="Model %d of %d: %s, %s" % (rangs[m["modele"]], len(self.noms), m["modele"], pas))
        if self.commande([os.path.join(C.ICI, "catalog.py"), "preparer", "--order", ",".join(lot)], ligne):
            raise Refus("Preparing the catalogue stopped with an error (see the log).")
        # models of the lot that the catalogue had nothing to do for (already there, or refused before it)
        manquants = [n for n in lot if n not in self.etat["finis"]]
        self.poser(finis=self.etat["finis"] + manquants, version=self.etat["version"] + 1, modele=None,
                   part=self.part(k, 1.0))


def bilan(avant, apres, noms):
    a = {m["id"]: m for m in avant}
    res = {"visages": [], "coiffures": [], "pieces": 0, "modeles": {}, "refus": []}
    for m in apres:
        n = m["id"]
        if n not in noms:
            continue
        av = a.get(n, {"catalogue": {"visage": False, "coiffure": False, "pieces": 0}})
        if m["catalogue"]["visage"] and not av["catalogue"]["visage"]:
            res["visages"].append(n)
        if m["catalogue"]["coiffure"] and not av["catalogue"]["coiffure"]:
            res["coiffures"].append(n)
        res["pieces"] += max(0, m["catalogue"]["pieces"] - av["catalogue"]["pieces"])
        res["modeles"][n] = {"extrait": m["extrait"], "catalogue": m["catalogue"]}
        res["refus"] += [{"nom": "%s · %s" % (n, r["nom"]), "raison": r["raison"]} for r in m["refus"]]
    return res


def lire_travail(ident):
    """State of a job (extraction "x…" or assembly), with the end of the log for an extraction."""
    if not re.fullmatch(r"x?[0-9]{8}-[0-9]{6}(-[0-9]{1,4})?", ident or ""):
        return None
    if ident.startswith("x"):
        d = os.path.join(EXTRACTIONS, ident)
        e = C.lire_json(os.path.join(d, "etat.json"))
        if e is None:
            return None
        try:
            with open(os.path.join(d, "journal.txt"), "rb") as f:
                f.seek(max(0, os.path.getsize(f.name) - 6000))
                lignes = f.read().decode("utf-8", "replace").splitlines()[1:]
            e["journal"] = [l[:300] for l in lignes if l.strip() and not l.startswith(("Blender ", "Blender quit"))][-25:]
        except OSError:
            e["journal"] = []
        return e
    return C.lire_json(os.path.join(ASSEMBLAGES, ident, "etat.json"))


# ===================================================================== assemblages

def assemblages():
    res = []
    if not os.path.isdir(ASSEMBLAGES):
        return res
    for ident in sorted(os.listdir(ASSEMBLAGES), reverse=True):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}(-[0-9]{1,4})?", ident):
            continue
        d = os.path.join(ASSEMBLAGES, ident)
        sel = C.lire_json(os.path.join(d, "selection.json"), {}) or {}
        e = C.lire_json(os.path.join(d, "etat.json"), {}) or {}
        fmts = [f for f, c in (("pmx", "resultat.pmx"), ("vrm", "resultat.vrm")) if os.path.isfile(os.path.join(d, "modele", c))]
        res.append({"id": ident, "date": "%s-%s-%sT%s:%s:%s" % (ident[:4], ident[4:6], ident[6:8], ident[9:11], ident[11:13], ident[13:15]),
                    "nom": sel.get("nom") or None, "tete": sel.get("tete"), "coiffure": sel.get("coiffure"),
                    "pieces": len(sel.get("pieces") or []),
                    "planche": "/data/assemblages/%s/planche.png" % ident if os.path.isfile(os.path.join(d, "planche.png")) else None,
                    "formats": fmts, "fini": bool(e.get("fini")), "ok": e.get("ok"), "etape": e.get("etape"),
                    "erreur": e.get("erreur"), "vrm_note": e.get("vrm_note") or None})
    return res


def export(ident, fmt):
    """(path to send, ASCII file name, UTF-8 name, type, temporary?) of an assembly."""
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}(-[0-9]{1,4})?", ident) or fmt not in ("pmx", "vrm"):
        raise Refus("Unknown assembly or format.", 404)
    d = os.path.join(ASSEMBLAGES, ident)
    m = os.path.join(d, "modele")
    sel = C.lire_json(os.path.join(d, "selection.json"), {}) or {}
    a, u = nom_telechargement(sel.get("nom") or "", "character-" + ident)
    if fmt == "vrm":
        c = os.path.join(m, "resultat.vrm")
        if not os.path.isfile(c):
            raise Refus("This assembly has no VRM.", 404)
        return c, a + ".vrm", u + ".vrm", "model/gltf-binary", False
    pmx = os.path.join(m, "resultat.pmx")
    if not os.path.isfile(pmx):
        raise Refus("This assembly has no PMX.", 404)
    tmp = os.path.join(d, ".export-%s.zip" % secrets.token_hex(6))
    racine = os.path.realpath(m)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        z.write(pmx, "%s/%s.pmx" % (a, a))
        for base, dirs, fichiers in os.walk(m):
            dirs[:] = sorted(x for x in dirs if not os.path.islink(os.path.join(base, x)))
            for f in sorted(fichiers):
                c = os.path.join(base, f)
                rel = os.path.relpath(c, m)
                # the PMX on the original bones (resultat-zzz.pmx) and the work files stay in the cache
                if f.lower().endswith((".pmx", ".vrm", ".json")) or os.path.islink(c) or \
                        not os.path.realpath(c).startswith(racine + os.sep):
                    continue
                z.write(c, "%s/%s" % (a, rel.replace(os.sep, "/")),
                        compress_type=zipfile.ZIP_STORED if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) else None)
    return tmp, a + "-pmx.zip", u + "-pmx.zip", "application/zip", True


# ===================================================================== command line

def _lire_pmx(chemin):
    sys.path.insert(0, os.path.join(C.OUTILS, "atlas"))
    import pmx as PMX
    P = PMX.lire(chemin)
    print(json.dumps({"materiaux": [m["nom"] for m in P["materiaux"]], "os": [o["nom"] for o in P["os"]],
                      "textures": P["textures"], "sommets": len(P["sommets"]["pos"])}, ensure_ascii=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["lire-pmx"] and len(a) == 2:
        _lire_pmx(a[1])
    elif a[:1] == ["formats"]:
        print(json.dumps(formats(), ensure_ascii=False, indent=1))
    else:
        sys.exit(__doc__)

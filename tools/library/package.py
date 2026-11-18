"""Builds the ready-made library package from a prepared installation (owner side).

    python3 -I tools/library/library.py build <installation> <output folder> --statuses <statuses.json>
    python3 -I tools/library/library.py check <output folder> --statuses <statuses.json>

<installation> is a folder holding settings.json (the settings of the prepared data, read through CHIMERA_SETTINGS):
nothing is written there. The output folder can be uploaded as is to a Hugging Face dataset repository.

Which models go in: a VRoid Hub model (`h<id>`) only if its status in statuts.json is "originale" and the
licence read on its page (sources.json of the cache) allows redistribution and the redistribution of modified
versions; any other model only if its own file says so (VRM meta: CC0 or a VRoid Hub licence URL allowing both;
for the base, the meta recorded by tools/vroid_base in provenance.json). Everything else is left out everywhere:
records, thumbnails, textures, catalogue, labels, sources, rejected list.

What goes in (README.md of this folder: the list and why): see LECTURES below.
"""
import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import struct
import sys
import tarfile
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor

ICI = os.path.dirname(os.path.abspath(__file__))
OUTILS = os.path.dirname(ICI)
DEPOT = os.path.dirname(OUTILS)
sys.path.append(ICI)
import installation as I  # noqa: E402

VERSION_CODE = 1          # version of this code the package is made for (index.json "code")
LIMITE = 1800 << 20       # raw bytes per archive before compression (HF: < 200 GB recommended per file; small ones resume better)
NIVEAU = 6                # gzip level
TEX = re.compile(rb"tex/([0-9a-f]{16}\.webp)")
MACHINE = re.compile(rb"/(?:home|Users)/[^/\s\"]+/")   # a path into someone's home folder
# in a refused item's folder only these remain (the page lists refused items by their record)
REFUS_GARDE = ("meta.json", "vignette.png")


def dire(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ licences

def licence_hub(v):
    L = (v or {}).get("licence") or {}
    return L.get("allowRedistribution") is True and L.get("modification") == "allowModificationRedistribution"


def meta_vrm(chemin):
    """VRM meta (0.x or 1.0) of a .vrm file, read from its glTF JSON chunk."""
    with open(chemin, "rb") as f:
        tete = f.read(20)
        if tete[:4] != b"glTF":
            raise ValueError("%s: not a glb" % chemin)
        n = struct.unpack("<I", tete[12:16])[0]
        j = json.loads(f.read(n))
    e = j.get("extensions") or {}
    return (e.get("VRMC_vrm") or {}).get("meta") or (e.get("VRM") or {}).get("meta") or {}


def licence_fichier(m):
    """Why a model's own VRM meta allows redistributing modified parts, or None."""
    if not m:
        return None
    if str(m.get("licenseName", "")).upper().startswith("CC0"):
        return "VRM meta: licenseName CC0"
    url = m.get("otherPermissionUrl") or m.get("otherLicenseUrl") or ""
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if q.get("redistribution") == ["allow"] and q.get("modification") == ["allow"]:
        return "VRM meta: VRoid Hub licence with redistribution=allow, modification=allow"
    if m.get("modification") == "allowModificationRedistribution" and m.get("allowRedistribution") is True:
        return "VRM 1.0 meta: allowRedistribution, allowModificationRedistribution"
    return None


# ------------------------------------------------------------------ reading the installation

class Installation:
    def __init__(self, dossier, statuts):
        os.environ["CHIMERA_SETTINGS"] = os.path.join(os.path.abspath(dossier), "settings.json")
        for v in ("CHIMERA_DATA", "CHIMERA_CACHE", "CHIMERA_BODY", "CHIMERA_GAME"):
            os.environ.pop(v, None)
        sys.path.insert(0, os.path.join(OUTILS, "workshop"))
        sys.path.insert(0, os.path.join(OUTILS, "atlas"))
        import common as C
        import games as jeux
        self.C = C
        self.rac = I.racines()
        self.vrm = jeux.charger("vroid").chemin("vrm")
        self.cache = self.rac["cache"]
        self.sources = C.lire_json(os.path.join(self.cache, "sources.json"), {}) or {}
        with open(statuts, encoding="utf-8") as f:
            self.statuts = json.load(f)

    def modeles_presents(self):
        C = self.C
        n = set(os.listdir(os.path.join(self.cache, "tetes"))) | set(os.listdir(os.path.join(self.cache, "coiffures")))
        for d in os.listdir(os.path.join(self.cache, "pieces")):
            m = C.lire_json(os.path.join(self.cache, "pieces", d, "meta.json")) or {}
            if m.get("perso"):
                n.add(m["perso"])
        # a model folder of the libraries: one holding its source PMX, face record or garment folders (not index/, essais/…)
        for r, signe in (("visage", "fiche.json"), ("vetements", "decoupe.json"), ("mmd_extrait", None)):
            for x in os.listdir(self.rac[r]):
                d = os.path.join(self.rac[r], x)
                if os.path.isfile(os.path.join(d, signe or x + ".pmx")):
                    n.add(x)
        return n

    def trier(self):
        """{name: why kept}, {name: why left out}."""
        garder, ecarter = {}, {}
        for n in sorted(self.modeles_presents()):
            src = self.sources.get(n)
            if re.fullmatch(r"h\d+", n):
                st = (self.statuts.get(n[1:]) or {}).get("statut")
                if st != "originale":
                    ecarter[n] = "status %s" % (st or "unknown")
                elif not src:
                    ecarter[n] = "no source page recorded"
                elif not licence_hub(src):
                    ecarter[n] = "licence does not allow redistributing modified versions"
                else:
                    garder[n] = "VRoid Hub, original, licence allows redistribution of modifications"
                continue
            if n == "Libre":
                prov = self.C.lire_json(os.path.join(self.rac["base"], "provenance.json"), {}) or {}
                pourquoi = licence_fichier(prov.get("meta_vrm"))
                quoi = "base (%s, provenance.json)" % prov.get("modele")
            else:
                f = os.path.join(self.vrm, n + ".vrm")
                pourquoi = licence_fichier(meta_vrm(f)) if os.path.isfile(f) else None
                quoi = os.path.basename(f)
            if pourquoi and src:
                garder[n] = "%s, %s" % (quoi, pourquoi)
            else:
                ecarter[n] = "licence not established by the file" if not pourquoi else "no source recorded"
        return garder, ecarter


# ------------------------------------------------------------------ the plan: what goes where

class Plan:
    """files: [(package path, absolute source or None, bytes or None)]; units, shared, links, fingerprints."""

    def __init__(self, inst, garder):
        self.inst, self.C, self.rac, self.garder = inst, inst.C, inst.rac, set(garder)
        self.fichiers, self.unites, self.partages, self.liens = [], [], [], []
        self.familles = {}
        self.retenus = {"tetes": [], "coiffures": [], "pieces": []}
        self.refuses = {"tetes": [], "coiffures": [], "pieces": []}
        self.textures = set()
        self.empreintes = {}
        self.inodes = {}     # (device, inode) -> package path of the first copy: the others become "doubles"
        self.doubles = {}    # first copy -> [other package paths], linked again by the installer
        # owner's roots in the records, longest first ({cache} before a root it would contain)
        self.jetons = sorted(((v, "{%s}" % k) for k, v in self.rac.items()), key=lambda x: -len(x[0]))
        self.jetons += [(os.path.realpath(v), j) for v, j in self.jetons if os.path.realpath(v) != v]

    def ajouter(self, famille, rel, src=None, octets=None):
        if src is not None and not src.endswith(".json"):
            st = os.stat(src)
            if st.st_nlink > 1:       # hard links of the cache (a hairstyle's pairs: one file under every head's name)
                premier = self.inodes.setdefault((st.st_dev, st.st_ino), rel)
                if premier != rel:
                    self.doubles.setdefault(premier, []).append(rel)
                    return
        if src is not None and src.endswith(".json"):
            with open(src, encoding="utf-8") as f:
                octets = self.neutre(f.read()).encode("utf-8")
            src = None
        self.fichiers.append((rel, src, octets))
        self.familles.setdefault(famille, []).append(len(self.fichiers) - 1)
        if src and src.endswith(".pmx"):
            self.empreintes[self.C.empreinte_fichier(src)] = rel

    def neutre(self, texte):
        for v, j in self.jetons:
            texte = texte.replace(json.dumps(v)[1:-1], j)
        return texte

    def dossier(self, famille, rel, src, garde=None, sauf=()):
        for d, ds, fs in os.walk(src):
            ds.sort()
            for f in sorted(fs):
                p = os.path.join(d, f)
                r = os.path.relpath(p, src).replace(os.sep, "/")
                if garde is not None and r not in garde or any(fnmatch.fnmatch(r, s) for s in sauf):
                    continue
                if os.path.islink(p) or not os.path.isfile(p):
                    raise ValueError("not a regular file: %s" % p)
                self.ajouter(famille, rel + "/" + r, p)

    def json_ajoute(self, famille, rel, valeur):
        self.ajouter(famille, rel, octets=self.neutre(json.dumps(valeur, ensure_ascii=False, indent=1)).encode("utf-8"))

    # ------------------------------------------------------------ cache
    def construire(self):
        C, cache = self.C, self.inst.cache
        lire = C.lire_json
        tetes = {}
        for i in sorted(os.listdir(os.path.join(cache, "tetes"))):
            m = lire(os.path.join(cache, "tetes", i, "meta.json"))
            if m and i in self.garder:
                tetes[i] = m
        ret_tetes = {i for i, m in tetes.items() if m.get("retenue")}
        for i, m in tetes.items():
            u = "cache/tetes/" + i
            self.unites.append(u)
            garde = None if i in ret_tetes else REFUS_GARDE
            self.dossier("tetes", u, os.path.join(cache, "tetes", i), garde=garde)
            (self.retenus if i in ret_tetes else self.refuses)["tetes"].append(i)
        for i in sorted(os.listdir(os.path.join(cache, "coiffures"))):
            d = os.path.join(cache, "coiffures", i)
            m = lire(os.path.join(d, "meta.json"))
            if not m or i not in self.garder:
                continue
            m["paires"] = {t: p for t, p in (m.get("paires") or {}).items() if t in tetes}
            ok, _ = C.coiffure_retenue(m["paires"], m.get("propre")) if "paires" in m else (m.get("retenue"), None)
            u = "cache/coiffures/" + i
            self.unites.append(u)
            self.json_ajoute("coiffures", u + "/meta.json", m)
            if ok:
                self.retenus["coiffures"].append(i)
                pairs = {t + ".bin" for t in ret_tetes}
                garde = {f for f in os.listdir(d) if f != "meta.json" and (not f.endswith(".bin") or f in pairs)}
            else:
                self.refuses["coiffures"].append(i)
                garde = set(REFUS_GARDE) - {"meta.json"}
            self.dossier("coiffures", u, d, garde=garde)
        pieces_gardees = []
        for i in sorted(os.listdir(os.path.join(cache, "pieces"))):
            d = os.path.join(cache, "pieces", i)
            m = lire(os.path.join(d, "meta.json"))
            if not m or m.get("perso") not in self.garder:
                continue
            note = (m.get("revue") or {}).get("note")
            ok = m.get("retenue") and not (note is not None and note > C.NOTE_REFUS)
            u = "cache/pieces/" + i
            self.unites.append(u)
            self.dossier("pieces", u, d, garde=None if ok else REFUS_GARDE)
            (self.retenus if ok else self.refuses)["pieces"].append(i)
            if ok:
                pieces_gardees.append(m)
        # base, body, the base head's library entry, the base and the pivot head
        for r in ("base", "corps", "biblio/Libre"):
            if os.path.isdir(os.path.join(cache, r)):
                self.unites.append("cache/" + r)
                self.dossier("base", "cache/" + r, os.path.join(cache, r))
        self.unites += ["base", "pivot"]
        self.dossier("base", "base", self.rac["base"])
        self.dossier("base", "pivot", self.rac["pivot"])
        # face library and source models of the kept models; the cache's biblio/<name> links to it
        for n in sorted(self.garder):
            if os.path.isdir(os.path.join(self.rac["visage"], n)):
                self.unites.append("visage/" + n)
                self.dossier("visage", "visage/" + n, os.path.join(self.rac["visage"], n))
                if os.path.islink(os.path.join(cache, "biblio", n)):
                    self.liens.append({"lien": "cache/biblio/" + n, "cible": "visage/" + n})
            if os.path.isdir(os.path.join(self.rac["mmd_extrait"], n)):
                self.unites.append("mmd_extrait/" + n)
                self.dossier("sources", "mmd_extrait/" + n, os.path.join(self.rac["mmd_extrait"], n))
        self.vetements(pieces_gardees)
        self.textures_cache()
        self.annexes()
        return self

    def vetements(self, pieces):
        """The garment library folders of the kept parts (porter reads them at assembly) and the textures their PMX
        name; a library part that has no record in the cache is left out (it would be prepared at the next preparer)."""
        sys.path.insert(0, os.path.join(OUTILS, "atlas"))
        import pmx as PMX
        racine = os.path.realpath(self.rac["vetements"])
        tex = set()
        for m in pieces:
            d = os.path.realpath(self.C.absolu(m["dossier"]))
            if not d.startswith(racine + os.sep):
                raise ValueError("part folder outside the garment library: %s" % d)
            rel = os.path.relpath(d, racine).replace(os.sep, "/")
            u = "vetements/" + rel
            self.unites.append(u)
            self.dossier("vetements", u, d)
            fi = self.C.lire_json(os.path.join(d, "fiche.json")) or {}
            for k in (fi.get("pmx"), (fi.get("garnie") or {}).get("pmx")):
                if k and os.path.isfile(os.path.join(d, k)):
                    for t in PMX.lire(os.path.join(d, k))["textures"]:
                        tex.add(os.path.normpath(os.path.join(d, t.replace("\\", "/"))))
        for t in sorted(tex):
            if not os.path.isfile(t) or not t.startswith(racine + os.sep):
                dire("  texture named by a part but absent or outside the library: %s" % t)
                continue
            rel = "vetements/" + os.path.relpath(t, racine).replace(os.sep, "/")
            if rel.split("/")[2] != "tex":   # a part's own texture is already inside its folder
                continue
            self.unites.append(rel)
            self.ajouter("vetements", rel, t)

    def textures_cache(self):
        """cache/tex: only the textures named by a kept record or model (glb, json)."""
        for rel, src, octets in list(self.fichiers):
            if not rel.startswith("cache/") or not (rel.endswith(".glb") or rel.endswith(".json")):
                continue
            if octets is None:
                with open(src, "rb") as f:
                    octets = f.read()
            self.textures.update(x.decode() for x in TEX.findall(octets))
        for t in sorted(self.textures):
            p = os.path.join(self.inst.cache, "tex", t)
            if not os.path.isfile(p):
                raise ValueError("texture named but absent: %s" % p)
            self.unites.append("cache/tex/" + t)
            self.ajouter("pieces", "cache/tex/" + t, p)

    def annexes(self):
        """Shared dictionaries, filtered to the kept models and items."""
        C, cache = self.C, self.inst.cache
        items = {"tetes/" + i for i in self.retenus["tetes"] + self.refuses["tetes"]}
        items |= {"coiffures/" + i for i in self.retenus["coiffures"] + self.refuses["coiffures"]}
        items |= {"pieces/" + i for i in self.retenus["pieces"] + self.refuses["pieces"]}
        etq = {k: v for k, v in (C.lire_json(os.path.join(cache, "etiquettes.json"), {}) or {}).items() if k in items}
        src = {}
        for n, v in sorted(self.inst.sources.items()):
            if n in self.garder:
                v = json.loads(json.dumps(v))
                (v.get("licence") or {}).pop("contactInformation", None)   # personal contact: not a term of use
                src[n] = v
        self.sources = src
        vis = C.lire_json(os.path.join(self.rac["visage"], "index.json"), {}) or {}
        if "parties" in vis:
            vis["parties"] = {p: {n: x for n, x in d.items() if n in self.garder} for p, d in vis["parties"].items()}
        ctl = {n: v for n, v in (C.lire_json(os.path.join(self.rac["vetements"], "controle.json"), {}) or {}).items()
               if n in self.garder}
        for rel, v in (("cache/etiquettes.json", etq), ("cache/sources.json", src), ("visage/index.json", vis),
                       ("vetements/controle.json", ctl)):
            self.partages.append(rel)
            self.json_ajoute("base", rel, v)

    def catalogue(self):
        """catalogue.json of the package, made by the workshop's own code on the kept records (no file written)."""
        import catalog as K
        cat = K.construire(ecrire=False)
        tous = {"tetes": set(self.retenus["tetes"] + self.refuses["tetes"]),
                "coiffures": set(self.retenus["coiffures"] + self.refuses["coiffures"]),
                "pieces": set(self.retenus["pieces"] + self.refuses["pieces"])}
        for k in ("tetes", "coiffures", "pieces"):
            cat[k] = [x for x in cat[k] if x["id"] in self.retenus[k]]
        genre = {"face": "tetes", "hairstyle": "coiffures", "part": "pieces"}
        cat["refus"] = [r for r in cat["refus"] if r["id"] in tous[genre[r["genre"]]]]
        cat["sources"] = {n: v for n, v in self.sources.items() if n in cat["sources"]}
        cat["nombres"] = {"preparees": sum(len(v) for v in tous.values()),
                          "retenues": sum(len(cat[k]) for k in tous), "refusees": len(cat["refus"]),
                          **{k: [len(cat[k]), len(tous[k])] for k in tous}}
        self.partages.append("cache/catalogue.json")
        self.json_ajoute("base", "cache/catalogue.json", cat)
        return cat


# ------------------------------------------------------------------ archives

def _ecrire_archive(args):
    chemin, entrees = args
    taille = 0
    with open(chemin + ".tmp", "wb") as brut:
        with gzip.GzipFile(filename="", mode="wb", fileobj=brut, compresslevel=NIVEAU, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
                dossiers = set()
                for rel, src, octets in entrees:
                    parts = rel.split("/")
                    for k in range(1, len(parts)):
                        d = "/".join(parts[:k])
                        if d not in dossiers:
                            dossiers.add(d)
                            ti = tarfile.TarInfo(d)
                            ti.type, ti.mode, ti.mtime = tarfile.DIRTYPE, 0o755, 0
                            t.addfile(ti)
                    ti = tarfile.TarInfo(rel)
                    ti.mode = 0o644
                    if octets is None:
                        st = os.stat(src)
                        ti.size, ti.mtime = st.st_size, int(st.st_mtime)
                        with open(src, "rb") as f:
                            t.addfile(ti, f)
                    else:
                        ti.size, ti.mtime = len(octets), 0
                        t.addfile(ti, io.BytesIO(octets))
                    taille += ti.size
    os.replace(chemin + ".tmp", chemin)
    return chemin, taille, os.path.getsize(chemin), I.sha256(chemin)


def decouper(plan):
    """[(name, [entries])]: one family per archive, cut at LIMITE raw bytes (an item never split)."""
    res = []
    for fam in ("base", "tetes", "coiffures", "pieces", "vetements", "visage", "sources"):
        lots, lot, n = [], [], 0
        for k in plan.familles.get(fam, []):
            rel, src, octets = plan.fichiers[k]
            t = os.path.getsize(src) if octets is None else len(octets)
            if lot and n + t > LIMITE and _unite(rel) != _unite(lot[-1][0]):
                lots.append(lot)
                lot, n = [], 0
            lot.append(plan.fichiers[k])
            n += t
        if lot:
            lots.append(lot)
        for i, l in enumerate(lots, 1):
            res.append(("%s-%02d.tar.gz" % (fam, i), l))
    return res


def doubles(d, tetes):
    """[{"de": first copy, "noms": names next to it | "*tetes"}]: "*tetes" = <head>.bin for every kept head but the
    first copy's own name (a hairstyle's pairs, the same file under every head's name)."""
    tous = {t + ".bin" for t in tetes}
    res = []
    for de, vers in sorted(d.items()):
        dossier, nom = de.rsplit("/", 1)
        if all(v.rsplit("/", 1)[0] == dossier for v in vers):
            noms = sorted(v.rsplit("/", 1)[1] for v in vers)
            res.append({"de": de, "noms": "*tetes" if set(noms) == tous - {nom} else noms})
        else:
            res.append({"de": de, "vers": sorted(vers)})
    return res


def _unite(rel):
    p = rel.split("/")
    return "/".join(p[:3] if p[0] in ("cache", "vetements") else p[:2])


# ------------------------------------------------------------------ texts of the dataset

def _cellule(s, n=90):
    s = re.sub(r"\s+", " ", str(s or "")).strip().replace("|", "\\|")
    return s if len(s) <= n else s[:n - 1] + "…"


def interdits(L):
    noms = (("allowExcessivelySexualUsage", "excessively sexual"), ("allowExcessivelyViolentUsage", "excessively violent"),
            ("allowPoliticalOrReligiousUsage", "political or religious"), ("allowAntisocialOrHateUsage", "antisocial or hateful"))
    return ", ".join(t for k, t in noms if L.get(k) is False) or "none listed"


def credits(sources, usage):
    lignes = ["# Credits", "",
              "Every model below keeps its author's terms. VRoid Hub models: original characters (no existing work, "
              "real person or uncertain source), whose VRM licence, read on the model's page on the date given in "
              "`sources.json`, allows redistribution and the redistribution of modified versions. Other models: "
              "VRoid Studio sample characters whose own file carries a CC0 licence.",
              "",
              "Before using a character you assembled, read the terms of **each** model whose parts it carries "
              "(columns below; the full fields are in `sources.json`, the current terms on the model's page).",
              "", "Parts: F = face, H = hairstyle, P = clothes and accessories.", "",
              "| Id | Model | Author | Page | Parts | Credit | Commercial use | Not allowed | Other terms |",
              "|---|---|---|---|---|---|---|---|---|"]
    for n, v in sorted(sources.items(), key=lambda x: (_cellule(x[1].get("nom")).lower(), x[0])):
        L = v.get("licence") or {}
        autre = "; ".join(x for x in (
            ("licence: " + L["nom"]) if L.get("nom") else "",
            ("other licence: " + _cellule(L["otherLicenseUrl"], 60)) if L.get("otherLicenseUrl") else "",
            ("copyright: " + _cellule(L["copyrightInformation"], 60)) if L.get("copyrightInformation") else "",
            "additional terms in sources.json" if L.get("thirdPartyLicenses") else "") if x)
        lien = "[%s](%s)" % (_cellule(v.get("site") or "page", 30), v["url"]) if v.get("url") else ""
        lignes.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            n, _cellule(v.get("nom")), _cellule(v.get("auteur")), lien, usage.get(n, ""),
            {"required": "**required**", "unnecessary": "not required"}.get(L.get("creditNotation"), _cellule(L.get("creditNotation")) or "?"),
            _cellule(L.get("commercialUsage") or "?"), interdits(L), _cellule(autre, 160) or "-"))
    return "\n".join(lignes) + "\n"


def carte(index, adresse_exemple):
    n = index["nombres"]
    return """---
pretty_name: Chimera VRoid Library
license: other
license_name: per-model-vrm-terms
license_link: https://huggingface.co/datasets/%(repo)s/blob/main/CREDITS.md
tags:
- 3d
- vroid
- vrm
- avatar
- character-creator
viewer: false
---

# Chimera VRoid Library

Ready-made parts library for [Chimera 3D Avatar Maker](https://github.com/RoseAndKnight/Chimera-3D-Avatar-Maker), the character workshop that mixes
the face, hairstyle and clothes of different 3D models on one body and exports the result as PMX and VRM.

It holds **%(modeles)d free VRoid models**, already prepared: %(tetes)d faces, %(coiffures)d hairstyles and
%(pieces)d clothes and accessories that the workshop can show and assemble at once, with their names, descriptions
and tags (English and French) and a link to each original model. Installing it saves the hours of Blender work the
preparation takes.

## Install with Chimera

From a clone of the Chimera repository:

```
python3 -I tools/library/library.py install
bin/chimera
```

The command downloads `index.json`, then the %(archives)d archives listed in it (%(poids).1f GB, about %(installe).1f GB
once installed), checks each one against its sha256 before unpacking, and installs them into the data folder of
your settings. Running it again resumes an interrupted download. It does not overwrite models you added yourself:
if something of yours is in the way it stops and says so (`--replace` sets those items aside, never deletes them).

This repository is not meant to be read with the `datasets` library: the archives are the workshop's own data
(PMX, glTF, textures, records). Format %(format)d, for Chimera code version %(code)d.

## Where the models come from

- VRoid Hub models (ids `h<number>`): original characters published on [VRoid Hub](https://hub.vroid.com) whose VRM
  licence allows redistribution **and** the redistribution of modified versions. Models based on an existing work
  (fan art), on a real person, or whose source is uncertain were left out, with all their parts, textures and
  thumbnails.
- VRoid Studio sample characters (`Base_Female`, `Vita`…, and the base body, `Libre`, made from `Sendagaya_Shibu`):
  CC0, as written in each model's own file.

## Terms of use

There is no single licence for this library: **each model keeps its author's terms** (commercial use, credit,
forbidden uses). `CREDITS.md` gives one line per model (name, author, link, whether credit is required, conditions);
`sources.json` gives every licence field as read on the model's page, and the date it was read. A character you
assemble carries parts of several models: it must respect the terms of each of them. Read the current terms on the
model's page before any use.

The maintainers of this library are not responsible for a use that breaks the terms of a model. The code of
Chimera itself is under the MIT licence; that licence does not cover the models.

## Removal requests

If you are the author of one of these models and want it removed, open a discussion in the Community tab of this
dataset with the model's id (first column of `CREDITS.md`). The model will be removed from the next version.

## Contents

- `index.json`: format version, Chimera code version, the archives with size and sha256, counts.
- `archives/*.tar.gz`: the prepared data, by family (base, faces, hairstyles, parts, garment library, face library,
  source models).
- `CREDITS.md`, `sources.json`: authors and terms.
- `mettre-en-ligne.sh`: the script that uploaded this folder (for the maintainer).
""" % dict(n, archives=len(index["archives"]), poids=sum(a["taille"] for a in index["archives"]) / 1e9,
           installe=index["taille_installee"] / 1e9, repo=adresse_exemple, format=index["format"], code=index["code"])


MISE_EN_LIGNE = r"""#!/bin/bash
# Uploads this folder to the Hugging Face dataset repository named below. Not run by the tools: by the maintainer,
# once logged in. Needs the `hf` command (https://huggingface.co/docs/huggingface_hub/guides/cli):
#     curl -LsSf https://hf.co/cli/install.sh | bash        (or: pip install -U huggingface_hub)
#     hf auth login
# Then: bash mettre-en-ligne.sh [owner/name]
# The address the installer reads (library.url in settings.default.json) must name the same repository:
#     https://huggingface.co/datasets/<owner/name>/resolve/main
set -euo pipefail
REPO="${1:-%(repo)s}"
ICI="$(cd "$(dirname "$0")" && pwd)"
command -v hf >/dev/null || { echo "hf is not installed: curl -LsSf https://hf.co/cli/install.sh | bash" >&2; exit 1; }
hf auth whoami >/dev/null || { echo "not logged in: hf auth login" >&2; exit 1; }
hf repos create "$REPO" --repo-type dataset --exist-ok
# archives first, index.json last: an installer never reads an index whose archives are not there yet
hf upload "$REPO" "$ICI/archives" archives --repo-type dataset --commit-message "Archives"
hf upload "$REPO" "$ICI" . --repo-type dataset --include "README.md" --include "CREDITS.md" --include "sources.json" \
    --include "mettre-en-ligne.sh" --commit-message "Card, credits and sources"
hf upload "$REPO" "$ICI/index.json" index.json --repo-type dataset --commit-message "Index"
echo "Done: https://huggingface.co/datasets/$REPO"
"""


# ------------------------------------------------------------------ build and check

def construire(installation, sortie, statuts, repo="Rose-and-Knight/chimera-vroid-library", travaux=4):
    t0 = time.time()
    inst = Installation(installation, statuts)
    garder, ecarter = inst.trier()
    dire("models: %d kept, %d left out" % (len(garder), len(ecarter)))
    for n, r in sorted(ecarter.items()):
        dire("  left out %s: %s" % (n, r))
    plan = Plan(inst, garder).construire()
    cat = plan.catalogue()
    # fingerprints the records keep and the package cannot rewrite: those items would be prepared again
    connues = set(plan.empreintes)
    manquent = []
    refuses = {"cache/%s/%s" % (k, i) for k, v in plan.refuses.items() for i in v}
    for rel, src, octets in plan.fichiers:
        if _unite(rel) in refuses:   # a refused part's source is not shipped: nothing would prepare it again
            continue
        if rel.endswith("meta.json") or rel.endswith("fiche.json"):
            m = json.loads(octets) if octets is not None else {}
            for k in ("empreinte", "corps"):
                v = str(m.get(k) or "")
                h = v.rsplit(":", 1)[-1]
                if re.fullmatch(r"[0-9a-f]{16}", h) and h not in connues:
                    manquent.append((rel, k, h))
    if manquent:
        dire("WARNING: %d fingerprints without their file in the package (those items would be prepared again):" % len(manquent))
        for x in manquent[:10]:
            dire("  %s %s %s" % x)
    os.makedirs(os.path.join(sortie, "archives"), exist_ok=True)
    lots = decouper(plan)
    with ProcessPoolExecutor(travaux) as pool:
        faits = list(pool.map(_ecrire_archive, [(os.path.join(sortie, "archives", n), l) for n, l in lots]))
    archives = [{"chemin": "archives/" + os.path.basename(c), "taille": t, "sha256": h, "taille_extraite": brut}
                for c, brut, t, h in faits]
    usage = {}
    for k, lettre in (("tetes", "F"), ("coiffures", "H")):
        for i in plan.retenus[k]:
            usage.setdefault(i, set()).add(lettre)
    for p in cat["pieces"]:
        usage.setdefault(p["perso"], set()).add("P")
    usage = {n: "".join(x for x in "FHP" if x in v) for n, v in usage.items()}
    index = {
        "format": I.FORMAT, "code": VERSION_CODE, "version": time.strftime("%Y-%m-%d"),
        "nombres": {"modeles": len(garder), "tetes": len(cat["tetes"]), "coiffures": len(cat["coiffures"]),
                    "pieces": len(cat["pieces"]), "refus": len(cat["refus"])},
        "archives": archives, "taille_installee": sum(a["taille_extraite"] for a in archives),
        "unites": sorted(set(plan.unites)), "partages": plan.partages, "catalogue": "cache/catalogue.json",
        "liens": plan.liens, "empreintes": plan.empreintes, "doubles": doubles(plan.doubles, plan.retenus["tetes"]),
        "tetes_bin": sorted(plan.retenus["tetes"]),
        "modeles": sorted(garder), "ecartes": ecarter,
    }
    I.ecrire_json(os.path.join(sortie, "index.json"), index)
    I.ecrire_json(os.path.join(sortie, "sources.json"), plan.sources)
    with open(os.path.join(sortie, "CREDITS.md"), "w", encoding="utf-8") as f:
        f.write(credits(plan.sources, usage))
    with open(os.path.join(sortie, "README.md"), "w", encoding="utf-8") as f:
        f.write(carte(index, repo))
    with open(os.path.join(sortie, "mettre-en-ligne.sh"), "w", encoding="utf-8") as f:
        f.write(MISE_EN_LIGNE % {"repo": repo})
    os.chmod(os.path.join(sortie, "mettre-en-ligne.sh"), 0o755)
    dire("package: %d models, %d faces, %d hairstyles, %d parts (%d refused records), %d archives, %.2f GB, %.2f GB installed, %.0f s" % (
        len(garder), len(cat["tetes"]), len(cat["coiffures"]), len(cat["pieces"]), len(cat["refus"]), len(archives),
        sum(a["taille"] for a in archives) / 1e9, index["taille_installee"] / 1e9, time.time() - t0))
    return index


def verifier(sortie, statuts):
    """Reads every archive back: sizes and sha256 of index.json, safe members, no machine path, no id of a model
    that is not kept (statuts.json other than "originale", or left out), no texture of cache/tex that no kept record
    names, every unit present. Returns the list of defects."""
    with open(os.path.join(sortie, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    with open(statuts, encoding="utf-8") as f:
        st = json.load(f)
    gardes = set(index["modeles"])
    hors = {i for i, v in st.items() if v.get("statut") != "originale"} | {
        n[1:] if re.fullmatch(r"h\d+", n) else n for n in index["ecartes"]}
    hors -= {n[1:] for n in gardes if n.startswith("h")}
    motif = re.compile(rb"(?<![0-9])(" + b"|".join(re.escape(x.encode()) for x in sorted(hors)) + rb")(?![0-9])") if hors else None
    defauts, noms, textes_tex, tex = [], set(), set(), set()
    for a in index["archives"]:
        c = os.path.join(sortie, a["chemin"])
        if os.path.getsize(c) != a["taille"] or I.sha256(c) != a["sha256"]:
            defauts.append("%s: size or sha256 differs from index.json" % a["chemin"])
        with tarfile.open(c, "r:gz") as t:
            for m in t:
                try:
                    rel = I.membre_sur(m)
                except I.Erreur as e:
                    defauts.append(str(e))
                    continue
                if m.isdir():
                    continue
                noms.add(rel)
                if motif and motif.search(rel.encode()):
                    defauts.append("%s: id of a model left out in the path" % rel)
                b = t.extractfile(m).read()
                if MACHINE.search(b):
                    defauts.append("%s: machine path inside" % rel)
                if motif and (rel.endswith(".json") or rel.endswith(".md")) and motif.search(b):
                    defauts.append("%s: id of a model left out inside" % rel)
                if rel.startswith("cache/tex/"):
                    tex.add(rel[len("cache/tex/"):])
                elif rel.startswith("cache/") and (rel.endswith(".glb") or rel.endswith(".json")):
                    textes_tex.update(x.decode() for x in TEX.findall(b))
    for f in ("CREDITS.md", "README.md", "sources.json"):
        with open(os.path.join(sortie, f), "rb") as fh:
            b = fh.read()
        if MACHINE.search(b) or (motif and motif.search(b)):
            defauts.append("%s: machine path or id of a model left out" % f)
    orphelines = tex - textes_tex
    if orphelines:
        defauts.append("%d textures of cache/tex named by no kept record, e.g. %s" % (len(orphelines), sorted(orphelines)[:3]))
    lies = I.cibles_doubles(index)
    for u in index["unites"] + index["partages"]:
        if u in lies:
            continue
        if u not in noms and not any(n.startswith(u + "/") for n in noms):
            defauts.append("unit missing from the archives: %s" % u)
    dire("checked: %d archives, %d files, %d textures, %d models left out searched; %d defects" % (
        len(index["archives"]), len(noms), len(tex), len(hors), len(defauts)))
    for d in defauts[:30]:
        dire("  DEFECT " + d)
    return defauts

"""Compatibility declared by each game: the "compatibilite" key of tools/games/<game>/game.json.

    python3 -I tools/games                 # known games, level, source folders present or not on this machine
    python3 -I tools/games table         # the per-game sections of COMPATIBILITY.md, drawn from the descriptions
    python3 -I tools/games write          # rewrites them in COMPATIBILITY.md (between the two markers)
    python3 -I tools/games check        # does the document still say what the descriptions say? (exit code 1 if not)

Nothing here changes what the other tools do: this module only reads the descriptions and looks at the disk.

Shape of the key (see tools/games/README.md, "Declaring compatibility"):
    "compatibilite": {
      "niveau": "eprouve" | "partiel" | "non",
      "resume": "one sentence",
      "formats": ["PMX (MMD)"],
      "provenance": "where these files usually come from (never a link)",
      "sources": {"<source>": {"contenu": "what the folder must contain",
                               "verifier": {"fichiers_de": ["<dotted key of the description>"]}
                                         | {"dossiers_avec": ["controle.glb", ...]}}},
      "prise_en_charge": {"<heading>": {"niveau": ..., "detail": "..."}},   (all the RUBRIQUES)
      "etapes": ["command", ...],
      "limites": ["...", ...]
    }
"""
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
OUTILS = os.path.dirname(ICI)
DEPOT = os.path.dirname(OUTILS)
if OUTILS not in sys.path:
    sys.path.append(OUTILS)
import games as jeux  # noqa: E402
import settings as R  # noqa: E402

NIVEAUX = {"eprouve": "proven", "partiel": "partial", "non": "not supported"}
# headings, in table order: key -> what the user reads
RUBRIQUES = (
    ("tetes", "Faces"),
    ("coiffures", "Hairstyles"),
    ("vetements", "Clothes"),
    ("accessoires", "Accessories"),
    ("expressions", "Expressions"),
)
DOCUMENT = os.path.join(DEPOT, "COMPATIBILITY.md")
DEBUT = "<!-- compatibilite:debut (written by \"python3 -I tools/games write\": do not edit by hand) -->"
FIN = "<!-- compatibilite:fin -->"


def declares():
    """The described games that declare a compatibility, from the most proven to the least, then by name
    (fixed order: the document does not depend on the machine's settings)."""
    noms = [n for n in jeux.tous() if "compatibilite" in jeux.charger(n).d]
    rang = list(NIVEAUX)
    return sorted(noms, key=lambda n: (rang.index(jeux.charger(n).d["compatibilite"].get("niveau"))
                                       if jeux.charger(n).d["compatibilite"].get("niveau") in rang else len(rang), n))


def erreurs(nom):
    """What is missing or malformed in a game's declaration (empty list if all is well)."""
    J = jeux.charger(nom)
    c = J.d.get("compatibilite")
    if not isinstance(c, dict):
        return ["%s: no compatibilite key" % nom]
    err = []
    for k in ("niveau", "resume", "formats", "provenance", "sources", "prise_en_charge", "etapes", "limites"):
        if k not in c:
            err.append("%s: compatibilite.%s is missing" % (nom, k))
    if c.get("niveau") not in NIVEAUX:
        err.append("%s: niveau %r (expected: %s)" % (nom, c.get("niveau"), ", ".join(NIVEAUX)))
    pec = c.get("prise_en_charge", {})
    for r, _ in RUBRIQUES:
        e = pec.get(r)
        if not isinstance(e, dict) or e.get("niveau") not in NIVEAUX or not e.get("detail"):
            err.append("%s: prise_en_charge.%s must have a niveau (%s) and a detail" % (nom, r, ", ".join(NIVEAUX)))
    for r in pec:
        if not r.startswith("_") and r not in dict(RUBRIQUES):
            err.append("%s: unknown heading prise_en_charge.%s" % (nom, r))
    for s in c.get("sources", {}):
        if s.startswith("_"):
            continue
        if s not in J.d.get("sources", {}):
            err.append("%s: compatibilite.sources.%s is not a source of the description" % (nom, s))
        for cle in c["sources"][s].get("verifier", {}).get("fichiers_de", []):
            if J.valeur(cle, None) is None:
                err.append("%s: verifier.fichiers_de names a missing key: %s" % (nom, cle))
    return err


def _texte(v):
    return v.replace("|", "\\|").replace("\n", " ")


def section(nom):
    """The Markdown section of a game, drawn from its description (no information specific to the machine)."""
    J = jeux.charger(nom)
    c = J.d["compatibilite"]
    L = ["### %s (`%s`): %s" % (J.titre, nom, NIVEAUX[c["niveau"]]), "", c["resume"], "",
         "- **Expected format**: %s" % ", ".join(c["formats"]),
         "- **Where these files come from**: %s" % c["provenance"], "",
         "| What the tool does | Level | Detail |", "|---|---|---|"]
    for r, titre in RUBRIQUES:
        e = c["prise_en_charge"][r]
        L.append("| %s | %s | %s |" % (titre, NIVEAUX[e["niveau"]], _texte(e["detail"])))
    L += ["", "**Where to put the files**", "",
          "| Source | Default folder | Setting key that moves it | What it must contain |", "|---|---|---|---|"]
    for s, e in c["sources"].items():
        if s.startswith("_"):
            continue
        L.append("| `%s` | `%s` | `games.%s.%s` | %s |" % (s, J.d["sources"][s], nom, s, _texte(e["contenu"])))
    L += ["", "**From your files to Chimera**", ""]
    L += ["%d. %s" % (i + 1, e) for i, e in enumerate(c["etapes"])]
    L += ["", "**Known limits**", ""]
    L += ["- %s" % e for e in c["limites"]]
    return "\n".join(L) + "\n"


def tableau():
    """Overview table, then one section per game."""
    L = ["| Game | Key | Format | Level | Faces | Hairstyles | Clothes |",
         "|---|---|---|---|---|---|---|"]
    for n in declares():
        J = jeux.charger(n)
        c = J.d["compatibilite"]
        p = c["prise_en_charge"]
        L.append("| %s | `%s` | %s | %s | %s |" % (J.titre, n, ", ".join(c["formats"]), NIVEAUX[c["niveau"]], " | ".join(
            NIVEAUX[p[r]["niveau"]] for r in ("tetes", "coiffures", "vetements"))))
    return "\n".join(L) + "\n\n" + "\n".join(section(n) for n in declares())


def _bloc(texte):
    """(before, block, after) of the document, cut at the markers; None if they are missing."""
    if DEBUT not in texte or FIN not in texte:
        return None
    a, reste = texte.split(DEBUT, 1)
    b, c = reste.split(FIN, 1)
    return a, b, c


def verifier(document=DOCUMENT):
    """List of the differences between the document and the descriptions (empty if the document is up to date)."""
    err = [e for n in declares() for e in erreurs(n)]
    try:
        with open(document, encoding="utf-8") as f:
            t = f.read()
    except FileNotFoundError:
        return err + ["%s not found" % document]
    b = _bloc(t)
    if b is None:
        return err + ["%s: start and end markers not found" % document]
    if b[1].strip() != tableau().strip():
        err.append("%s is out of date: python3 -I tools/games write" % os.path.relpath(document, DEPOT))
    return err


def ecrire(document=DOCUMENT):
    with open(document, encoding="utf-8") as f:
        t = f.read()
    b = _bloc(t)
    if b is None:
        sys.exit("%s: markers not found (%s … %s)" % (document, DEBUT, FIN))
    with open(document, "w", encoding="utf-8") as f:
        f.write(b[0] + DEBUT + "\n\n" + tableau() + "\n" + FIN + b[2])
    print("written:", document)


# ---------------------------------------------------------------- what is on this machine

def etat_source(J, s, e):
    """(path, where the path comes from, present?, what the check says)."""
    chemin = J.chemin(s)
    origine = "setting games.%s.%s" % (J.nom, s) if R.valeur("games.%s.%s" % (J.nom, s), None) is not None else "default"
    present = os.path.exists(chemin)
    v = e.get("verifier", {})
    note = ""
    if present and v.get("fichiers_de"):
        att = []
        for cle in v["fichiers_de"]:
            x = J.valeur(cle)
            vals = [x] if isinstance(x, str) else list(x) if isinstance(x, list) else \
                [w for k, w in x.items() if not k.startswith("_")]
            att += vals
        ok = sum(os.path.isfile(os.path.join(chemin, f)) for f in att)
        note = "%d/%d expected files" % (ok, len(att))
        if ok < len(att):
            manque = [f for f in att if not os.path.isfile(os.path.join(chemin, f))]
            note += " (missing: %s%s)" % (", ".join(manque[:3]), "…" if len(manque) > 3 else "")
    elif present and v.get("dossiers_avec") and os.path.isdir(chemin):
        d = [x for x in sorted(os.listdir(chemin)) if os.path.isdir(os.path.join(chemin, x))]
        ok = [x for x in d if all(os.path.isfile(os.path.join(chemin, x, f)) for f in v["dossiers_avec"])]
        note = "%d folders with %s (out of %d)" % (len(ok), " and ".join(v["dossiers_avec"]), len(d))
    return chemin, origine, present, note


def liste():
    actif = jeux.nom_actif()
    for n in jeux.tous():
        J = jeux.charger(n)
        c = J.d.get("compatibilite")
        marque = "  (active game)" if n == actif else ""
        if not c:
            print("%s: %s, no compatibility declared%s" % (n, J.titre, marque))
            continue
        print("%s: %s, %s, %s%s" % (n, J.titre, ", ".join(c["formats"]), NIVEAUX.get(c["niveau"], "?"), marque))
        print("    " + " · ".join("%s %s" % (t.lower(), NIVEAUX[c["prise_en_charge"][r]["niveau"]]) for r, t in RUBRIQUES))
        for s, e in c["sources"].items():
            if s.startswith("_"):
                continue
            chemin, origine, present, note = etat_source(J, s, e)
            print("    %-16s %-8s %s  [%s]%s" % (s, "present" if present else "ABSENT", chemin, origine,
                                                  "  " + note if note else ""))
        for e in erreurs(n):
            print("    incomplete declaration:", e)

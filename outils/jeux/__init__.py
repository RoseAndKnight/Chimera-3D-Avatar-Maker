"""What the code knows about a game, read from its description: outils/jeux/<jeu>/jeu.json (and the files next to it).

The core (workshop, heads, clothes, atlas) names no game and no character: it asks here.

    import jeux
    J = jeux.actif()                 # the game the models come from (setting jeux.actif, ATELIER_JEU)
    J.chemin("mmd_extrait")          # source folder: jeux.<jeu>.mmd_extrait from the settings, otherwise the description's default
    J.valeur("tete_base.nom")        # a dotted key of the description (tete_base: see tete_base_fabriquee)
    J.fichier("familles.json")       # a file stored next to the description (None if there is none)
    J.cas("expressions")             # {character: value} of the special cases in "personnages"
    jeux.charger("outerplane")       # another game, by name (resources taken from one specific game)
    jeux.maison()                    # our own parts (Maëlys…), described the same way
    jeux.cas("expressions")          # special cases of the house set and of the active game (the latter wins)
    J.traduire("M_2000001_FACE")     # name of a game material in the common language ("materiaux" table)
    jeux.jeu_du_fichier(chemin)      # the game with a "materiaux" table one of whose source folders holds this file

The core applies the translation of material names in one place only: atlas/pmx.lire, for any PMX stored
in a source folder of a game that has a "materiaux" table. Without a table (ZZZ), nothing is translated.

Keys that start with "_" are notes: never read as data.
Adding a game: see README.md in this folder.
"""
import json
import os
import re
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
OUTILS = os.path.dirname(ICI)
if OUTILS not in sys.path:
    sys.path.append(OUTILS)
import reglages as R  # noqa: E402

_RIEN = object()
_charges = {}


def tete_base_fabriquee():
    """tete_base as measured by Chimera's base and stored next to its body (base.json, written by
    outils/base_vroid/fabriquer.py): {} when no base is built. A description that says tete_base.depuis_base reads it;
    the setting jeux.<jeu>.tete_base.* always wins."""
    try:
        with open(os.path.join(os.path.dirname(R.chemin("atelier.corps")), "base.json"), encoding="utf-8") as f:
            v = json.load(f).get("tete_base")
    except (OSError, ValueError, KeyError):
        return {}
    return v if isinstance(v, dict) else {}


class Jeu:
    def __init__(self, nom, dossier=None):
        self.nom = nom
        self.dossier = dossier or os.path.join(ICI, nom)
        chemin = os.path.join(self.dossier, "jeu.json")
        if not os.path.isfile(chemin):
            raise KeyError("unknown game: %s (no %s)" % (nom, chemin))
        with open(chemin, encoding="utf-8") as f:
            self.d = json.load(f)
        self.titre = self.d.get("titre", nom)
        self.materiaux = []  # [(compiled pattern, common name)], in the order of the description
        for e in self.d.get("materiaux", []):
            if not (isinstance(e, list) and len(e) == 2 and all(isinstance(x, str) for x in e)):
                raise ValueError("%s: materiaux: each entry is [pattern, common name], not %r" % (nom, e))
            self.materiaux.append((re.compile(e[0]), e[1]))
        self._racines = None

    def __repr__(self):
        return "<jeu %s>" % self.nom

    def valeur(self, cle, defaut=_RIEN):
        """Value of a dotted key of the description (a copy: the description does not change). The setting
        jeux.<jeu>.<cle> replaces it when given: the base head of a family where everyone has their own models."""
        v = R.valeur("jeux.%s.%s" % (self.nom, cle), None)
        if v is not None:
            return json.loads(json.dumps(v))
        d = self.d
        if cle.split(".")[0] == "tete_base" and isinstance(d.get("tete_base"), dict) and d["tete_base"].get("depuis_base"):
            d = dict(d, tete_base=dict(d["tete_base"], **tete_base_fabriquee()))
        for k in cle.split("."):
            if not isinstance(d, dict) or k not in d:
                if defaut is _RIEN:
                    raise KeyError("%s: key missing from the description: %s" % (self.nom, cle))
                return defaut
            d = d[k]
        return json.loads(json.dumps(d))

    def modeles(self):
        """{character: PMX path relative to mmd_extrait}. The description names them one by one, or says "*":
        any <Name>/ folder of mmd_extrait that holds <Name>.pmx (a family where everyone brings their own models)."""
        m = self.valeur("modeles", {})
        if m != "*":
            return {k: v for k, v in m.items() if not k.startswith("_")}
        d = self.chemin("mmd_extrait")
        if not os.path.isdir(d):
            return {}
        return {n: os.path.join(n, n + ".pmx") for n in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, n, n + ".pmx"))}

    def chemin(self, source):
        """Source folder or file: jeux.<jeu>.<source> from the settings if given, otherwise the public default
        of the description (sources.<source>); {donnees} and the other references replaced, absolute path."""
        v = R.valeur("jeux.%s.%s" % (self.nom, source), None)
        if v is None:
            v = self.d.get("sources", {}).get(source)
        if v is None:
            raise KeyError("%s: unknown source: %s" % (self.nom, source))
        return R.developper(v)

    def developper(self, texte):
        """Path written in the description ({donnees}/…) made absolute."""
        return R.developper(texte)

    def fichier(self, nom):
        c = os.path.join(self.dossier, nom)
        return c if os.path.isfile(c) else None

    def cas(self, cle):
        """Special cases per character: {character: personnages.<character>.<cle>}."""
        return {p: json.loads(json.dumps(v[cle])) for p, v in self.d.get("personnages", {}).items()
                if not p.startswith("_") and isinstance(v, dict) and cle in v}

    def traduire(self, nom, sinon=_RIEN):
        """Name of a game material in the common language: the first pattern of "materiaux" (regular expression
        searched in the name) that matches gives the common name (\\1… reuses its groups). No pattern matches:
        the name as is (or the value of sinon). Without a table (ZZZ), always the name as is."""
        for rx, commun in self.materiaux:
            t = rx.search(nom)
            if t:
                return t.expand(commun)
        return nom if sinon is _RIEN else sinon

    def racines(self):
        """Source folders (and files) of the game, real paths (links resolved)."""
        if self._racines is None:
            self._racines = [os.path.realpath(self.chemin(s)) for s in self.d.get("sources", {}) if not s.startswith("_")]
        return self._racines


def charger(nom, dossier=None):
    """The game described in outils/jeux/<nom> (or in dossier, for a description stored elsewhere)."""
    if nom not in _charges or (dossier and _charges[nom].dossier != dossier):
        _charges[nom] = Jeu(nom, dossier)
    return _charges[nom]


def nom_actif():
    return R.valeur("jeux.actif")


def actif():
    return charger(nom_actif())


def maison():
    return charger("maison")


def tous():
    """Names of the games described in this folder."""
    return sorted(n for n in os.listdir(ICI) if os.path.isfile(os.path.join(ICI, n, "jeu.json")))


def cas(cle):
    """Special cases of the house set, then of the active game (the active game wins for the same name)."""
    return {**maison().cas(cle), **actif().cas(cle)}


def jeu_du_fichier(chemin):
    """The game with a "materiaux" table one of whose source folders holds this file (the longest root wins),
    otherwise None. Only games that have a table are considered: without a table, nothing is even resolved."""
    p = os.path.realpath(chemin)
    meilleur, n = None, -1
    for nom in sorted(set(tous()) | set(_charges)):
        J = charger(nom)
        if not J.materiaux:
            continue
        for r in J.racines():
            if (p == r or p.startswith(r.rstrip(os.sep) + os.sep)) and len(r) > n:
                meilleur, n = J, len(r)
    return meilleur

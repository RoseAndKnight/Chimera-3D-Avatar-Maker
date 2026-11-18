"""Machine settings: the only place where the code learns where the data and the programs are.

Three layers, each one overriding the previous one, key by key:
  1. reglages.defaut.json at the repository root (tracked by git, public values), then the other public-value
     files of DEFAUTS (couches.defaut.json: garment types, ranks and thresholds of Chimera's layers);
  2. reglages.json at the repository root (specific to the machine, never tracked; ATELIER_REGLAGES names another file);
  3. the environment variables of ENVIRONNEMENT (ATELIER_ATLAS, ATELIER_CACHE…).

A key is named by its dotted path: "chemins.donnees", "atelier.cache", "jeux.zzz.mmd_extrait".
The games (outils/jeux) give the public default of their source folders themselves; jeux.<jeu>.<source> only
changes it for your machine, jeux.actif (ATELIER_JEU) names the game the models come from.
In a text value, {cle} is replaced by the value of that key ({donnees} = chemins.donnees, {depot} =
the repository root); a relative path starts from the repository root, ~ is expanded.

From a tool run from its own folder (outils/<sub>/x.py):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import reglages as R
    R.chemin("atelier.cache"), R.programme("blender"), R.valeur("calcul.places"), R.section("jeux")

Adding a section: write it in reglages.defaut.json with its public values (and, if an environment variable
must set it, a line in ENVIRONNEMENT, and in CHEMINS_ENV if it is a path); nothing
else to change here.

On the command line (for the bin/ scripts):
    python3 -I outils/reglages.py chemin chemins.donnees | programme blender | valeur calcul.places | tout
"""
import json
import os
import re
import shutil
import sys

DEPOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAUT = os.path.join(DEPOT, "reglages.defaut.json")
# public values, read in this order (each one overrides the previous one key by key)
DEFAUTS = (DEFAUT, os.path.join(DEPOT, "couches.defaut.json"))
LOCAL = os.path.join(DEPOT, "reglages.json")

# environment variable -> key it forces
ENVIRONNEMENT = {
    "ATELIER_ATLAS": "chemins.donnees",
    "ATELIER_CACHE": "atelier.cache",
    "ATELIER_CORPS": "atelier.corps",
    "ATELIER_SQUELETTE": "atelier.squelette",
    "ATELIER_LANCEUR": "calcul.lanceur",
    "ATELIER_PLACES": "calcul.places",
    "ATELIER_BLENDER": "programmes.blender",
    "ATELIER_PYTHON": "programmes.python",
    "ATELIER_JEU": "jeux.actif",
}
# those whose relative path starts from the current folder (like any shell variable), not from the repository
CHEMINS_ENV = {"ATELIER_ATLAS", "ATELIER_CACHE", "ATELIER_CORPS"}

_RIEN = object()
_fichiers = None


def _fusion(base, dessus):
    for k, v in dessus.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _fusion(base[k], v)
        else:
            base[k] = v
    return base


def _lire(chemin):
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _poser(d, cle, v):
    *haut, fin = cle.split(".")
    for k in haut:
        d = d.setdefault(k, {})
    d[fin] = v


def _brut(d, cle):
    for k in cle.split("."):
        if not isinstance(d, dict) or k not in d:
            return _RIEN
        d = d[k]
    return d


def tout():
    """The merged settings (defaults, local file, environment), without substitution."""
    global _fichiers
    if _fichiers is None:
        local = os.environ.get("ATELIER_REGLAGES") or LOCAL
        base = {}
        for d in DEFAUTS:
            _fusion(base, _lire(d))
        _fichiers = _fusion(base, _lire(os.path.expanduser(local)))
    r = json.loads(json.dumps(_fichiers))
    for var, cle in ENVIRONNEMENT.items():
        v = os.environ.get(var)
        if v:
            _poser(r, cle, os.path.abspath(os.path.expanduser(v)) if var in CHEMINS_ENV else v)
    return r


def _substituer(v, r, profondeur=0):
    if not isinstance(v, str) or "{" not in v:
        return v
    if profondeur > 8:
        raise ValueError("settings: circular reference in %r" % v)

    def un(m):
        nom = m.group(1)
        if nom == "depot":
            return DEPOT
        x = _brut(r, nom)
        if x is _RIEN and "." not in nom:
            x = _brut(r, "chemins." + nom)
        if x is _RIEN or x is None:
            raise KeyError("settings: unknown {%s}" % nom)
        return _chemin(_substituer(x, r, profondeur + 1))
    return re.sub(r"\{([A-Za-z0-9_.]+)\}", un, v)


def _chemin(v):
    v = os.path.expanduser(v)
    return v if os.path.isabs(v) else os.path.join(DEPOT, v)


def valeur(cle, defaut=_RIEN):
    """Value of a dotted key, {references} replaced (a text stays a text: no forced absolute path)."""
    r = tout()
    v = _brut(r, cle)
    if v is _RIEN:
        if defaut is _RIEN:
            raise KeyError("unknown setting: %s" % cle)
        return defaut
    if isinstance(v, dict):
        return {k: valeur(cle + "." + k) for k in v}
    return _substituer(v, r)


def chemin(cle):
    """Absolute path of a key (None if the value is null)."""
    v = valeur(cle)
    return None if v in (None, "") else _chemin(v)


def developper(texte):
    """Absolute path of a text written outside the settings (a game description): {references} replaced as
    in a value, relative path from the repository root, ~ expanded (None if empty)."""
    return None if texte in (None, "") else _chemin(_substituer(texte, tout()))


def section(nom):
    """A whole section, references replaced (dictionary)."""
    return valeur(nom, {})


def programme(nom):
    """Executable of programmes.<nom>: a path if it contains /, otherwise looked up in the PATH.
    python empty or null = the running interpreter (sys.executable)."""
    v = valeur("programmes." + nom, None)
    if not v:
        if nom == "python":
            return sys.executable
        v = nom
    if "/" in v or v.startswith("~"):
        return _chemin(v)
    return shutil.which(v) or v


def oublier():
    """Re-reads the files on the next call (after they have been changed)."""
    global _fichiers
    _fichiers = None


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("chemin", "programme", "valeur", "tout") or (a[0] != "tout" and len(a) != 2):
        sys.exit("usage: reglages.py chemin <key> | programme <name> | valeur <key> | tout")
    if a[0] == "tout":
        print(json.dumps(tout(), ensure_ascii=False, indent=1))
    else:
        v = {"chemin": chemin, "programme": programme, "valeur": valeur}[a[0]](a[1])
        print("" if v is None else (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v))

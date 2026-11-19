"""Machine settings: the only place where the code learns where the data and the programs are.

Three layers, each one overriding the previous one, key by key:
  1. settings.default.json at the repository root (tracked by git, public values), then the other public-value
     files of DEFAULTS (layers.default.json: garment types, ranks and thresholds of Chimera's layers);
  2. settings.json at the repository root (specific to the machine, never tracked; CHIMERA_SETTINGS names another file);
  3. the environment variables of ENVIRONMENT (CHIMERA_DATA, CHIMERA_CACHE…).

A key is named by its dotted path: "paths.data", "workshop.cache", "games.zzz.mmd_extracted".
The games (tools/games) give the public default of their source folders themselves; games.<game>.<source> only
changes it for your machine, games.active (CHIMERA_GAME) names the game the models come from.
In a text value, {key} is replaced by the value of that key ({data} = paths.data, {repo} =
the repository root); a relative path starts from the repository root, ~ is expanded.

From a tool run from its own folder (tools/<sub>/x.py):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import settings as R
    R.path("workshop.cache"), R.program("blender"), R.value("compute.slots"), R.section("games")

Adding a section: write it in settings.default.json with its public values (and, if an environment variable
must set it, a line in ENVIRONMENT, and in PATHS_ENV if it is a path); nothing
else to change here.

On the command line (for the bin/ scripts):
    python3 -I tools/settings.py path paths.data | program blender | value compute.slots | all
"""
import json
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(REPO, "settings.default.json")
# public values, read in this order (each one overrides the previous one key by key)
DEFAULTS = (DEFAULT, os.path.join(REPO, "layers.default.json"))
LOCAL = os.path.join(REPO, "settings.json")

# environment variable -> key it forces
ENVIRONMENT = {
    "CHIMERA_DATA": "paths.data",
    "CHIMERA_CACHE": "workshop.cache",
    "CHIMERA_BODY": "workshop.body",
    "CHIMERA_SKELETON": "workshop.skeleton",
    "CHIMERA_LAUNCHER": "compute.launcher",
    "CHIMERA_SLOTS": "compute.slots",
    "CHIMERA_BLENDER": "programs.blender",
    "CHIMERA_PYTHON": "programs.python",
    "CHIMERA_GAME": "games.active",
}
# those whose relative path starts from the current folder (like any shell variable), not from the repository
PATHS_ENV = {"CHIMERA_DATA", "CHIMERA_CACHE", "CHIMERA_BODY"}

_NOTHING = object()
_files = None


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _place(d, the_key, v):
    *top, end = the_key.split(".")
    for k in top:
        d = d.setdefault(k, {})
    d[end] = v


def _raw(d, the_key):
    for k in the_key.split("."):
        if not isinstance(d, dict) or k not in d:
            return _NOTHING
        d = d[k]
    return d


def all_():
    """The merged settings (defaults, local file, environment), without substitution."""
    global _files
    if _files is None:
        local = os.environ.get("CHIMERA_SETTINGS") or LOCAL
        base = {}
        for d in DEFAULTS:
            _merge(base, _read(d))
        _files = _merge(base, _read(os.path.expanduser(local)))
    r = json.loads(json.dumps(_files))
    for var, the_key in ENVIRONMENT.items():
        v = os.environ.get(var)
        if v:
            _place(r, the_key, os.path.abspath(os.path.expanduser(v)) if var in PATHS_ENV else v)
    return r


def _substitute(v, r, depth=0):
    if not isinstance(v, str) or "{" not in v:
        return v
    if depth > 8:
        raise ValueError("settings: circular reference in %r" % v)

    def un(m):
        name = m.group(1)
        if name == "repo":
            return REPO
        x = _raw(r, name)
        if x is _NOTHING and "." not in name:
            x = _raw(r, "paths." + name)
        if x is _NOTHING or x is None:
            raise KeyError("settings: unknown {%s}" % name)
        return _path(_substitute(x, r, depth + 1))
    return re.sub(r"\{([A-Za-z0-9_.]+)\}", un, v)


def _path(v):
    v = os.path.expanduser(v)
    return v if os.path.isabs(v) else os.path.join(REPO, v)


def value(the_key, default=_NOTHING):
    """Value of a dotted key, {references} replaced (a text stays a text: no forced absolute path)."""
    r = all_()
    v = _raw(r, the_key)
    if v is _NOTHING:
        if default is _NOTHING:
            raise KeyError("unknown setting: %s" % the_key)
        return default
    if isinstance(v, dict):
        return {k: value(the_key + "." + k) for k in v}
    return _substitute(v, r)


def path(the_key):
    """Absolute path of a key (None if the value is null)."""
    v = value(the_key)
    return None if v in (None, "") else _path(v)


def expand(text):
    """Absolute path of a text written outside the settings (a game description): {references} replaced as
    in a value, relative path from the repository root, ~ expanded (None if empty)."""
    return None if text in (None, "") else _path(_substitute(text, all_()))


def section(name):
    """A whole section, references replaced (dictionary)."""
    return value(name, {})


def program(name):
    """Executable of programs.<name>: a path if it contains /, otherwise looked up in the PATH.
    python empty or null = the running interpreter (sys.executable)."""
    v = value("programs." + name, None)
    if not v:
        if name == "python":
            return sys.executable
        v = name
    if "/" in v or v.startswith("~"):
        return _path(v)
    return shutil.which(v) or v


def forget():
    """Re-reads the files on the next call (after they have been changed)."""
    global _files
    _files = None


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("path", "program", "value", "all") or (a[0] != "all" and len(a) != 2):
        sys.exit("usage: settings.py path <key> | program <name> | value <key> | all")
    if a[0] == "all":
        print(json.dumps(all_(), ensure_ascii=False, indent=1))
    else:
        v = {"path": path, "program": program, "value": value}[a[0]](a[1])
        print("" if v is None else (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v))

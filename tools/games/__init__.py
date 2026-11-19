"""What the code knows about a game, read from its description: tools/games/<game>/game.json (and the files next to it).

The core (workshop, heads, clothes, atlas) names no game and no character: it asks here.

    import games
    J = games.active()                 # the game the models come from (setting games.active, CHIMERA_GAME)
    J.path("mmd_extracted")          # source folder: games.<game>.mmd_extracted from the settings, otherwise the description's default
    J.value("base_head.name")        # a dotted key of the description (base_head: see base_head_built)
    J.file("families.json")       # a file stored next to the description (None if there is none)
    J.case_("expressions")           # {character: value} of the special cases in "people"
    games.load("outerplane")         # another game, by name (resources taken from one specific game)
    games.own_rig()                  # our own parts (Maelys…), described the same way
    games.case_("expressions")       # special cases of the house set and of the active game (the latter wins)
    J.translate("M_2000001_FACE")    # name of a game material in the common language ("materials" table)
    games.game_of_file(path)         # the game with a "materials" table one of whose source folders holds this file

The core applies the translation of material names in one place only: atlas/pmx.read, for any PMX stored
in a source folder of a game that has a "materials" table. Without a table (ZZZ), nothing is translated.

Keys that start with "_" are notes: never read as data.
Adding a game: see README.md in this folder.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
if TOOLS not in sys.path:
    sys.path.append(TOOLS)
import settings as R  # noqa: E402

_NOTHING = object()
_loads = {}


def base_head_built():
    """base_head as measured by Chimera's base and stored next to its body (base.json, written by
    tools/vroid_base/build.py): {} when no base is built. A description that says base_head.since_base reads it;
    the setting games.<game>.base_head.* always wins."""
    try:
        with open(os.path.join(os.path.dirname(R.path("workshop.body")), "base.json"), encoding="utf-8") as f:
            v = json.load(f).get("base_head")
    except (OSError, ValueError, KeyError):
        return {}
    return v if isinstance(v, dict) else {}


class Game:
    def __init__(self, name, folder=None):
        self.name = name
        self.folder = folder or os.path.join(HERE, name)
        path = os.path.join(self.folder, "game.json")
        if not os.path.isfile(path):
            raise KeyError("unknown game: %s (no %s)" % (name, path))
        with open(path, encoding="utf-8") as f:
            self.d = json.load(f)
        self.title = self.d.get("title", name)
        self.materials = []  # [(compiled pattern, common name)], in the order of the description
        for e in self.d.get("materials", []):
            if not (isinstance(e, list) and len(e) == 2 and all(isinstance(x, str) for x in e)):
                raise ValueError("%s: materials: each entry is [pattern, common name], not %r" % (name, e))
            self.materials.append((re.compile(e[0]), e[1]))
        self._roots = None

    def __repr__(self):
        return "<game %s>" % self.name

    def value(self, the_key, default=_NOTHING):
        """Value of a dotted key of the description (a copy: the description does not change). The setting
        games.<game>.<key> replaces it when given: the base head of a family where everyone has their own models."""
        v = R.value("games.%s.%s" % (self.name, the_key), None)
        if v is not None:
            return json.loads(json.dumps(v))
        d = self.d
        if the_key.split(".")[0] == "base_head" and isinstance(d.get("base_head"), dict) and d["base_head"].get("since_base"):
            d = dict(d, base_head=dict(d["base_head"], **base_head_built()))
        for k in the_key.split("."):
            if not isinstance(d, dict) or k not in d:
                if default is _NOTHING:
                    raise KeyError("%s: key missing from the description: %s" % (self.name, the_key))
                return default
            d = d[k]
        return json.loads(json.dumps(d))

    def models(self):
        """{character: PMX path relative to mmd_extracted}. The description names them one by one, or says "*":
        any <Name>/ folder of mmd_extracted that holds <Name>.pmx (a family where everyone brings their own models)."""
        m = self.value("models", {})
        if m != "*":
            return {k: v for k, v in m.items() if not k.startswith("_")}
        d = self.path("mmd_extracted")
        if not os.path.isdir(d):
            return {}
        return {n: os.path.join(n, n + ".pmx") for n in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, n, n + ".pmx"))}

    def path(self, source):
        """Source folder or file: games.<game>.<source> from the settings if given, otherwise the public default
        of the description (sources.<source>); {data} and the other references replaced, absolute path."""
        v = R.value("games.%s.%s" % (self.name, source), None)
        if v is None:
            v = self.d.get("sources", {}).get(source)
        if v is None:
            raise KeyError("%s: unknown source: %s" % (self.name, source))
        return R.expand(v)

    def expand(self, text):
        """Path written in the description ({data}/…) made absolute."""
        return R.expand(text)

    def file(self, name):
        c = os.path.join(self.folder, name)
        return c if os.path.isfile(c) else None

    def case_(self, the_key):
        """Special cases per character: {character: people.<character>.<key>}."""
        return {p: json.loads(json.dumps(v[the_key])) for p, v in self.d.get("people", {}).items()
                if not p.startswith("_") and isinstance(v, dict) and the_key in v}

    def translate(self, name, otherwise=_NOTHING):
        """Name of a game material in the common language: the first pattern of "materials" (regular expression
        searched in the name) that matches gives the common name (\\1… reuses its groups). No pattern matches:
        the name as is (or the value of otherwise). Without a table (ZZZ), always the name as is."""
        for rx, common in self.materials:
            t = rx.search(name)
            if t:
                return t.expand(common)
        return name if otherwise is _NOTHING else otherwise

    def roots(self):
        """Source folders (and files) of the game, real paths (links resolved)."""
        if self._roots is None:
            self._roots = [os.path.realpath(self.path(s)) for s in self.d.get("sources", {}) if not s.startswith("_")]
        return self._roots


def load(name, folder=None):
    """The game described in tools/games/<name> (or in folder, for a description stored elsewhere)."""
    if name not in _loads or (folder and _loads[name].folder != folder):
        _loads[name] = Game(name, folder)
    return _loads[name]


def name_active():
    return R.value("games.active")


def active():
    return load(name_active())


def own_rig():
    return load("homemade")


def all_():
    """Names of the games described in this folder."""
    return sorted(n for n in os.listdir(HERE) if os.path.isfile(os.path.join(HERE, n, "game.json")))


def case_(the_key):
    """Special cases of the house set, then of the active game (the active game wins for the same name)."""
    return {**own_rig().case_(the_key), **active().case_(the_key)}


def game_of_file(path):
    """The game with a "materials" table one of whose source folders holds this file (the longest root wins),
    otherwise None. Only games that have a table are considered: without a table, nothing is even resolved."""
    p = os.path.realpath(path)
    best, n = None, -1
    for name in sorted(set(all_()) | set(_loads)):
        J = load(name)
        if not J.materials:
            continue
        for r in J.roots():
            if (p == r or p.startswith(r.rstrip(os.sep) + os.sep)) and len(r) > n:
                best, n = J, len(r)
    return best

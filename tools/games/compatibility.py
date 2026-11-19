"""Compatibility declared by each game: the "compatibility" key of tools/games/<game>/game.json.

    python3 -I tools/games                 # known games, level, source folders present or not on this machine
    python3 -I tools/games table         # the per-game sections of COMPATIBILITY.md, drawn from the descriptions
    python3 -I tools/games write          # rewrites them in COMPATIBILITY.md (between the two markers)
    python3 -I tools/games check        # does the document still say what the descriptions say? (exit code 1 if not)

Nothing here changes what the other tools do: this module only reads the descriptions and looks at the disk.

Shape of the key (see tools/games/README.md, "Declaring compatibility"):
    "compatibility": {
      "level": "tested" | "partial" | "non",
      "summary": "one sentence",
      "formats": ["PMX (MMD)"],
      "origin": "where these files usually come from (never a link)",
      "sources": {"<source>": {"content": "what the folder must contain",
                               "check": {"files_of": ["<dotted key of the description>"]}
                                         | {"folders_with": ["check.glb", ...]}}},
      "support": {"<heading>": {"level": ..., "detail": "..."}},   (all the HEADINGS)
      "steps": ["command", ...],
      "limits": ["...", ...]
    }
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
REPO = os.path.dirname(TOOLS)
if TOOLS not in sys.path:
    sys.path.append(TOOLS)
import games as games  # noqa: E402
import settings as R  # noqa: E402

LEVELS = {"tested": "proven", "partial": "partial", "non": "not supported"}
# headings, in table order: key -> what the user reads
HEADINGS = (
    ("heads", "Faces"),
    ("hairstyles", "Hairstyles"),
    ("clothes", "Clothes"),
    ("accessories", "Accessories"),
    ("expressions", "Expressions"),
)
DOCUMENT = os.path.join(REPO, "COMPATIBILITY.md")
START = "<!-- compatibility:start (written by \"python3 -I tools/games write\": do not edit by hand) -->"
END = "<!-- compatibility:end -->"


def declared():
    """The described games that declare a compatibility, from the most proven to the least, then by name
    (fixed order: the document does not depend on the machine's settings)."""
    names = [n for n in games.all_() if "compatibility" in games.load(n).d]
    rank = list(LEVELS)
    return sorted(names, key=lambda n: (rank.index(games.load(n).d["compatibility"].get("level"))
                                       if games.load(n).d["compatibility"].get("level") in rank else len(rank), n))


def errors(name):
    """What is missing or malformed in a game's declaration (empty list if all is well)."""
    J = games.load(name)
    c = J.d.get("compatibility")
    if not isinstance(c, dict):
        return ["%s: no compatibility key" % name]
    err = []
    for k in ("level", "summary", "formats", "origin", "sources", "support", "steps", "limits"):
        if k not in c:
            err.append("%s: compatibility.%s is missing" % (name, k))
    if c.get("level") not in LEVELS:
        err.append("%s: level %r (expected: %s)" % (name, c.get("level"), ", ".join(LEVELS)))
    pec = c.get("support", {})
    for r, _ in HEADINGS:
        e = pec.get(r)
        if not isinstance(e, dict) or e.get("level") not in LEVELS or not e.get("detail"):
            err.append("%s: support.%s must have a level (%s) and a detail" % (name, r, ", ".join(LEVELS)))
    for r in pec:
        if not r.startswith("_") and r not in dict(HEADINGS):
            err.append("%s: unknown heading support.%s" % (name, r))
    for s in c.get("sources", {}):
        if s.startswith("_"):
            continue
        if s not in J.d.get("sources", {}):
            err.append("%s: compatibility.sources.%s is not a source of the description" % (name, s))
        for the_key in c["sources"][s].get("check", {}).get("files_of", []):
            if J.value(the_key, None) is None:
                err.append("%s: check.files_of names a missing key: %s" % (name, the_key))
    return err


def _text(v):
    return v.replace("|", "\\|").replace("\n", " ")


def _marked(line, J):
    """A line that names a string of the description's "_fr_ok" (e.g. a file name set by a third-party tool) carries
    the same exception for the French check of COMPATIBILITY.md."""
    ok = J.d.get("_fr_ok") or {}
    return line + " <!-- fr-ok: see _fr_ok in the game description -->" if any(k in line for k in ok) else line


def section(name):
    """The Markdown section of a game, drawn from its description (no information specific to the machine)."""
    J = games.load(name)
    c = J.d["compatibility"]
    L = ["### %s (`%s`): %s" % (J.title, name, LEVELS[c["level"]]), "", c["summary"], "",
         "- **Expected format**: %s" % ", ".join(c["formats"]),
         "- **Where these files come from**: %s" % c["origin"], "",
         "| What the tool does | Level | Detail |", "|---|---|---|"]
    for r, title in HEADINGS:
        e = c["support"][r]
        L.append("| %s | %s | %s |" % (title, LEVELS[e["level"]], _text(e["detail"])))
    L += ["", "**Where to put the files**", "",
          "| Source | Default folder | Setting key that moves it | What it must contain |", "|---|---|---|---|"]
    for s, e in c["sources"].items():
        if s.startswith("_"):
            continue
        L.append("| `%s` | `%s` | `games.%s.%s` | %s |" % (s, J.d["sources"][s], name, s, _text(e["content"])))
    L += ["", "**From your files to Chimera**", ""]
    L += ["%d. %s" % (i + 1, e) for i, e in enumerate(c["steps"])]
    L += ["", "**Known limits**", ""]
    L += ["- %s" % e for e in c["limits"]]
    return "\n".join(_marked(x, J) for x in L) + "\n"


def array():
    """Overview table, then one section per game."""
    L = ["| Game | Key | Format | Level | Faces | Hairstyles | Clothes |",
         "|---|---|---|---|---|---|---|"]
    for n in declared():
        J = games.load(n)
        c = J.d["compatibility"]
        p = c["support"]
        L.append(_marked("| %s | `%s` | %s | %s | %s |" % (J.title, n, ", ".join(c["formats"]), LEVELS[c["level"]], " | ".join(
            LEVELS[p[r]["level"]] for r in ("heads", "hairstyles", "clothes"))), J))
    return "\n".join(L) + "\n\n" + "\n".join(section(n) for n in declared())


def _block(text):
    """(before, block, after) of the document, cut at the markers; None if they are missing."""
    if START not in text or END not in text:
        return None
    a, rest = text.split(START, 1)
    b, c = rest.split(END, 1)
    return a, b, c


def check(document=DOCUMENT):
    """List of the differences between the document and the descriptions (empty if the document is up to date)."""
    err = [e for n in declared() for e in errors(n)]
    try:
        with open(document, encoding="utf-8") as f:
            t = f.read()
    except FileNotFoundError:
        return err + ["%s not found" % document]
    b = _block(t)
    if b is None:
        return err + ["%s: start and end markers not found" % document]
    if b[1].strip() != array().strip():
        err.append("%s is out of date: python3 -I tools/games write" % os.path.relpath(document, REPO))
    return err


def write(document=DOCUMENT):
    with open(document, encoding="utf-8") as f:
        t = f.read()
    b = _block(t)
    if b is None:
        sys.exit("%s: markers not found (%s … %s)" % (document, START, END))
    with open(document, "w", encoding="utf-8") as f:
        f.write(b[0] + START + "\n\n" + array() + "\n" + END + b[2])
    print("written:", document)


# ---------------------------------------------------------------- what is on this machine

def state_source(J, s, e):
    """(path, where the path comes from, present?, what the check says)."""
    path = J.path(s)
    origin = "setting games.%s.%s" % (J.name, s) if R.value("games.%s.%s" % (J.name, s), None) is not None else "default"
    present = os.path.exists(path)
    v = e.get("check", {})
    note = ""
    if present and v.get("files_of"):
        att = []
        for the_key in v["files_of"]:
            x = J.value(the_key)
            vals = [x] if isinstance(x, str) else list(x) if isinstance(x, list) else \
                [w for k, w in x.items() if not k.startswith("_")]
            att += vals
        ok = sum(os.path.isfile(os.path.join(path, f)) for f in att)
        note = "%d/%d expected files" % (ok, len(att))
        if ok < len(att):
            missing = [f for f in att if not os.path.isfile(os.path.join(path, f))]
            note += " (missing: %s%s)" % (", ".join(missing[:3]), "…" if len(missing) > 3 else "")
    elif present and v.get("folders_with") and os.path.isdir(path):
        d = [x for x in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, x))]
        ok = [x for x in d if all(os.path.isfile(os.path.join(path, x, f)) for f in v["folders_with"])]
        note = "%d folders with %s (out of %d)" % (len(ok), " and ".join(v["folders_with"]), len(d))
    return path, origin, present, note


def items():
    active = games.name_active()
    for n in games.all_():
        J = games.load(n)
        c = J.d.get("compatibility")
        mark = "  (active game)" if n == active else ""
        if not c:
            print("%s: %s, no compatibility declared%s" % (n, J.title, mark))
            continue
        print("%s: %s, %s, %s%s" % (n, J.title, ", ".join(c["formats"]), LEVELS.get(c["level"], "?"), mark))
        print("    " + " · ".join("%s %s" % (t.lower(), LEVELS[c["support"][r]["level"]]) for r, t in HEADINGS))
        for s, e in c["sources"].items():
            if s.startswith("_"):
                continue
            path, origin, present, note = state_source(J, s, e)
            print("    %-16s %-8s %s  [%s]%s" % (s, "present" if present else "ABSENT", path, origin,
                                                  "  " + note if note else ""))
        for e in errors(n):
            print("    incomplete declaration:", e)

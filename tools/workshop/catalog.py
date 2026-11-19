"""The workshop's catalogue: preparing the cache, adding a part, summary.

    python3 -I tools/workshop/catalog.py prepare [--level small|full] [--stale] [--order A,B,…]
    python3 -I tools/workshop/catalog.py add <part folder in library format>
    python3 -I tools/workshop/catalog.py summary

`prepare` only redoes what is missing (meta.json present and source unchanged): running it again
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
# tools/games). Each face is sewn into the oval of the base head (C.BASE_HEAD, itself in the catalogue).
OWN_RIG = C.games.own_rig()
PER_LOT = 25
IN_PARALLEL = C.places()
_NUMBER = itertools.count()
# heads computed separately (no folder in the face library): biblio -> (pmx, displayed name)
# (those whose file is not on this machine are ignored: they are ours, not in the repository)
ENTRIES_OWN = {n: (OWN_RIG.expand(e["pmx"]), e["name"]) for n, e in OWN_RIG.value("heads").items()
                   if not n.startswith("_") and os.path.isfile(OWN_RIG.expand(e["pmx"]))}
# parts made in the workshop (house description, "built"), likewise
BUILT = [{"kind": "build", **f, "pmx": OWN_RIG.expand(f["pmx"])} for f in OWN_RIG.value("built")
              if os.path.isfile(OWN_RIG.expand(f["pmx"]))]
# Free head and hairstyle (tools/free_base/free_face.py), next to the free base body: without a base head
# from the game, these are what the workshop offers, and the page opens without any game model.
FREE = "Free"   # id of the free base's own head and hairstyle (format 2, revision 2)
HEAD_FREE = os.path.join(os.path.dirname(C.BODY), "head.pmx")
HAIRSTYLE_FREE = os.path.join(os.path.dirname(C.BODY), "hairstyle.pmx")


# base.json next to the body (tools/vroid_base): the measured base head (base_head). Without it (free MakeHuman
# base), no measured head.
BASE_JSON = C.read_json(os.path.join(os.path.dirname(C.BODY), "base.json")) or {}
# Displayed names of the base's own head and hairstyle: their role, in English, translated by the page (workshop.js,
# SERVER). Never the name of the model the base was made from: it is in another language than the page (a VRoid
# title in Japanese) and says nothing of the role. Names written by an older base.json ("names") are ignored.
NAMES_BASE = ({"head": "Base face", "hairstyle": "Base hairstyle"} if BASE_JSON.get("base_head")
             else {"head": "Free head", "hairstyle": "Free bob"})


def card_base_head():
    """Record of the game's base head in the face catalogue; a path that does not exist when the game has none."""
    return os.path.join(C.LIBRARY_VIS, C.BASE_HEAD or "(no base head)", "record.json")


def base_free():
    """Is (or will) the base of this cache be the free head? A base already sewn keeps its nature."""
    m = C.read_json(os.path.join(C.BASE, "meta.json"))
    if m is not None:
        return bool(m.get("free"))
    return not os.path.isfile(card_base_head()) and os.path.isfile(HEAD_FREE) and os.path.isfile(HAIRSTYLE_FREE)


def base_open():
    """The base of this cache wears its own head (base_free) and it is a character head, measured by
    tools/vroid_base (base.json: base_head): face transfer and hairstyle posing can read it.
    It then also receives the faces and hairstyles of imported models, next to its own. The free MakeHuman
    head (tools/free_base) has no such measurement: its cache offers only that head."""
    return base_free() and bool(BASE_JSON.get("base_head"))


def key_base():
    """Name, in the cache's biblio/, of the head on which all faces are sewn: the base's (FREE, read from
    head.pmx next to the body), otherwise the active game's base head."""
    return FREE if base_free() else C.BASE_HEAD


def catalog(the_key, small):
    """Names offered ("heads" or "coiffures"): the house, then the active game ("all_": see hairstyles_of_game).
    On a base that wears its own head: that one (FREE) first, then, if the base is open, the game's, except the
    character the base was made from (its head and hairstyle are already FREE)."""
    if base_free() and not base_open():
        return [FREE]
    res = []
    for J in (OWN_RIG, C.GAME):
        v = J.value("catalog.small." + the_key if small else "catalog." + the_key, [])
        res += hairstyles_of_game() if v == "all" else [n for n in v if J is not OWN_RIG or n in ENTRIES_OWN]
    if base_free():
        res = [FREE] + [n for n in res if n not in (FREE, BASE_JSON["base_head"].get("name"))]
    return res


def cards_library():
    """Every part folder of the library whose type/kind has a place in the workshop."""
    res = []
    for f in sorted(glob.glob(os.path.join(C.LIBRARY_GARMENT, "*", "*", "record.json"))):
        with open(f, encoding="utf-8") as fh:
            fi = json.load(fh)
        if fi.get("class") == "garment" and fi.get("type") in C.CLOTHES or \
           fi.get("class") == "accessory" and fi.get("sort") in C.ACCESSORIES:
            res.append((os.path.dirname(f), fi))
    return res


def prepare_library(names, own_ones=None):
    """Face library entries in the cache: links to the library, own computed entries
    (`own_ones`: name -> (pmx, displayed name), ENTRIES_OWN by default). The base's head (FREE) is read
    again if its file has changed."""
    import library as BI
    own_ones = ENTRIES_OWN if own_ones is None else own_ones
    d = os.path.join(C.CACHE, "library")
    os.makedirs(d, exist_ok=True)
    for n in names:
        target = os.path.join(d, n)
        if n in own_ones:
            pmx = own_ones[n][0]
            emp = C.fingerprint_file(pmx) if n == FREE else None
            if not os.path.exists(os.path.join(target, "part.npz")) or \
               emp is not None and (C.read_json(os.path.join(target, "record.json")) or {}).get("fingerprint") != emp:
                os.makedirs(target, exist_ok=True)
                print("reading head", n, flush=True)
                BI.extract_one((n, pmx, d))
            C.write_json(os.path.join(target, "record.json"), dict({"person": n, "pmx": pmx}, **({"fingerprint": emp} if emp else {})))
        elif not os.path.exists(target):
            os.symlink(os.path.join(C.LIBRARY_VIS, n), target)
    return d


def a_make_meta(path, fingerprint=None):
    m = C.read_json(path)
    # heads that failed on a zero-weight bone must be redone: that defect is fixed (C.graft)
    return m is None or "KeyError" in str(m.get("reason", "")) or (fingerprint is not None and m.get("fingerprint") != fingerprint)


def expired(m):
    """Part fitted by an older method or on a body other than today's."""
    return bool(m) and (m.get("v_fitting") != C.V_FITTING or m.get("body") != C.fingerprint_file(C.BODY))


def hairstyles_of_game():
    """Every head of the face library whose hairstyle has been extracted and whose outfit has been cut."""
    return sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(C.LIBRARY_VIS, "*", "part.npz"))
                  if os.path.isfile(os.path.join(os.path.dirname(f), "hairstyle.json"))
                  and os.path.isdir(os.path.join(C.LIBRARY_GARMENT, os.path.basename(os.path.dirname(f)))))


def plan(level):
    small = level == "small"
    heads = catalog("heads", small)
    hairstyles = catalog("hairstyles", small)
    free, base = base_free(), key_base()
    own_ones = dict(ENTRIES_OWN, **({FREE: (HEAD_FREE, NAMES_BASE["head"])} if free else {}))
    # the base alone needs no entry (nor the pivot head): its hairstyle is made for its head
    alone = free and set(heads) | set(hairstyles) <= {FREE}
    lib = prepare_library([] if alone else sorted((set(heads) | set(hairstyles) | {base}) - {None}), own_ones)
    work = []
    if a_make_meta(os.path.join(C.CACHE, "body", "meta.json"), C.fingerprint_file(C.BODY)):
        work.append({"kind": "body", "id": "body"})
    if not os.path.isfile(os.path.join(C.BASE, "meta.json")):
        work.append({"kind": "base", "id": "base", "free": True, "head": HEAD_FREE} if free else
                       {"kind": "base", "id": "base", "head": C.read_json(card_base_head())["pmx"]})
    infos = []
    for n in heads:
        pmx = own_ones[n][0] if n in own_ones else C.read_json(os.path.join(C.LIBRARY_VIS, n, "record.json"))["pmx"]
        emp = C.fingerprint_file(pmx)
        infos.append((n, pmx, emp))
        m_t = C.read_json(os.path.join(C.CACHE, "heads", n, "meta.json")) or {}
        ir, bo = C.iris(n), C.mouth(n)
        ir, bo = (C.fingerprint_file(ir) if ir else None), (C.fingerprint_file(bo) if bo else None)
        common = {"kind": "head", "id": n, "library": n, "pmx": pmx, "free": free and n == FREE, "base_library": base,
                  "name": own_ones[n][1] if n in own_ones else n}
        if a_make_meta(os.path.join(C.CACHE, "heads", n, "meta.json"), emp) or (m_t.get("kept") and m_t.get("v_skin") != C.V_SKIN):
            work.append(common)
        elif m_t.get("kept") and (m_t.get("iris") != ir or m_t.get("mouth") != bo or m_t.get("library") != base):
            # the chosen iris or mouth, or the base head's name in biblio/, has changed: the head keeps its
            # transferred face, only head.glb and its record are rewritten
            work.append(dict(common, keep_model=True))
    for f in BUILT:
        card = os.path.join(C.CACHE, "built", f["character"], f["id"], "record.json")
        m = C.read_json(card)
        if m is None or m.get("fingerprint") != C.fingerprint_file(f["pmx"]):
            work.append(f)
    return work, heads, hairstyles, lib, infos


def plan_next(level, heads, hairstyles, lib, stale_ones=False):
    """What depends on the heads and the made parts (their records must exist)."""
    work = []
    tt = []
    for n in heads:
        m = C.read_json(os.path.join(C.CACHE, "heads", n, "meta.json")) or {}
        if m.get("kept"):
            tt.append({"id": n, "library": m["library"], "translation": m["translation"]})
    version = "v%d:" % C.V_HAIRSTYLE
    sig = version + ",".join(sorted(t["id"] for t in tt))
    for a in hairstyles:
        m = C.read_json(os.path.join(C.CACHE, "hairstyles", a, "meta.json"))
        if a == FREE and base_free():
            e = sig + ":" + str(C.fingerprint_file(HAIRSTYLE_FREE))
            if m is None or m.get("fingerprint") != e:
                work.append({"kind": "hairstyle_free", "id": a, "name": NAMES_BASE["hairstyle"], "pmx": HAIRSTYLE_FREE,
                                "heads": [t["id"] for t in tt], "fingerprint": e})
        elif m is None or m.get("fingerprint") != sig:
            own = a if any(t["id"] == a for t in tt) else None
            name = ENTRIES_OWN[a][1] if a in ENTRIES_OWN else a
            it = {"kind": "hairstyle", "id": a, "name": name, "library": lib, "heads": tt, "own": own, "fingerprint": sig}
            # Heads were added since this hairstyle was posed (an extraction adds models lot by lot): pose it on the
            # new heads only and keep the pairs already measured, instead of posing it again on every head.
            already = (m or {}).get("pairs")
            if isinstance(already, dict) and already and str(m.get("fingerprint") or "").startswith(version) and \
               os.path.isfile(os.path.join(C.CACHE, "hairstyles", a, "hairstyle.glb")):
                keep = sorted(t["id"] for t in tt if t["id"] in already
                                and os.path.isfile(os.path.join(C.CACHE, "hairstyles", a, t["id"] + ".bin")))
                if keep:
                    it = dict(it, heads=[t for t in tt if t["id"] not in keep], keep=keep)
            work.append(it)
    folders = [(os.path.join(C.CACHE, "built", f["character"], f["id"]), f["character"], f["id"]) for f in BUILT]
    if level == "small":
        taken = {}
        for d, fi in cards_library():
            k = fi.get("type") or fi.get("sort")
            if taken.get(k, 0) < 2 and k in ("skirt", "top", "shoes", "belt"):
                taken[k] = taken.get(k, 0) + 1
                folders.append((d, fi["character"], fi["id"]))
    else:
        folders += [(d, fi["character"], fi["id"]) for d, fi in cards_library()]
    for d, character, ident in folders:
        cid = C.the_key(character, ident)
        card = C.read_json(os.path.join(d, "record.json")) or {}
        src = os.path.join(d, card.get("pmx", ""))
        # the same source file as prepare.prepare_part, whose fingerprint the record keeps: the part with its
        # trims when it has some (otherwise every trimmed part would be fitted again at each preparation)
        trimmed = os.path.join(d, (card.get("trimmed") or {}).get("pmx") or "")
        if (card.get("trimmed") or {}).get("pmx") and os.path.exists(trimmed):
            src = trimmed
        path_meta = os.path.join(C.CACHE, "parts", cid, "meta.json")
        if os.path.isfile(src) and (a_make_meta(path_meta, C.fingerprint_file(src)) or (stale_ones and expired(C.read_json(path_meta)))):
            work.append({"kind": "part", "id": cid, "folder": d})
        elif (C.read_json(path_meta) or {}).get("kept") and not os.path.isfile(os.path.join(C.CACHE, "parts", cid, "covers.bin")):
            work.append({"kind": "covers", "id": cid})   # part prepared before covers.bin existed
    return work


def launch_blender(items, name, wait=True):
    if not items:
        return None
    tmp = os.environ.get("TMPDIR") or C.CACHE
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "workshop-%s-%d-%d.json" % (name, os.getpid(), next(_NUMBER)))
    C.write_json(path, items)
    script = os.path.join(C.HERE, "prepare.py")
    cmd = ["bash", C.BLENDER_SH, "-b", "--factory-startup", "--python", script, "--", "--travail", path]
    launcher = C.launcher()
    if not wait:
        return subprocess.Popen(launcher + cmd)
    r = subprocess.run(launcher + cmd)
    os.remove(path)
    if r.returncode:
        print("Blender returned", r.returncode, file=sys.stderr)


def transfer(it):
    """The face of it["pmx"] sewn into the base head, by the tool as is: heads/<id>/model."""
    import shutil
    mod = os.path.join(C.CACHE, "heads", it["id"], "model")
    shutil.rmtree(mod, ignore_errors=True)
    r = subprocess.run([C.PYTHON, "-I", os.path.join(C.TOOLS, "head", "retarget_head.py"), "face", "--source", it["pmx"],
                        "--target", C.BASE_PMX, "--output", mod, "--no-sheet"], capture_output=True, text=True)
    ok = os.path.isfile(os.path.join(mod, "measures.json"))
    if not ok:
        os.makedirs(mod, exist_ok=True)
        C.write_json(os.path.join(mod, "error.json"), ((r.stderr or r.stdout).strip().splitlines() or ["no output"])[-1][:200])
    print("face", it["id"], "transferred" if ok else "FAILED", flush=True)


def run(work, name, of_front=1, finished=None):
    """In batches of at most PAR_LOT per Blender. `de_front` > 1: the items do not depend on
    one another (parts, hairstyles), that many Blenders at once, each on one core.
    `finished(lot)`, if given, is called after each Blender batch (from its worker thread)."""
    if of_front <= 1:
        for k in range(0, len(work), PER_LOT):
            print("batch %s: %d to %d of %d" % (name, k + 1, min(k + PER_LOT, len(work)), len(work)), flush=True)
            launch_blender(work[k:k + PER_LOT], name)
            if finished:
                finished(work[k:k + PER_LOT])
        return
    # short batches, dealt in turn: the big parts do not all land in the same one
    size = max(1, min(PER_LOT, -(-len(work) // (of_front * 4))))
    batches = [work[k:k + size] for k in range(0, len(work), size)]
    print("%s: %d items, %d batches of %d, %d at a time" % (name, len(work), len(batches), size, of_front), flush=True)
    from concurrent.futures import ThreadPoolExecutor
    done = itertools.count(1)

    def un(batch):
        launch_blender(batch, name)
        print("batch %s: %d of %d done" % (name, next(done), len(batches)), flush=True)
        if finished:
            finished(batch)
    with ThreadPoolExecutor(of_front) as pool:
        list(pool.map(un, batches))


class Track:
    """Progress lines for the workshop server (tools/workshop/inputs.py): one JSON object per line after "@@ ".
    `order`: the models of the current extraction lot, in the order they should become usable. Each item of the
    suite (hairstyle, part) belongs to a model or to nobody (shared work); when all the items of a model are done,
    a line "model_done" names it, and its face, hairstyle and parts are in the catalogue."""
    MARK = "@@ "

    def __init__(self, order):
        import threading
        self.order = list(order or [])
        self.rest = {}                 # model -> items of the suite not done yet
        self.hairstyles = set()          # models whose hairstyle is not posed yet
        self.lock = threading.Lock()

    def say(self, **kw):
        print(self.MARK + json.dumps(kw, ensure_ascii=False), flush=True)

    def model(self, it):
        if it["kind"] == "hairstyle":
            return it["id"] if it["id"] in self.order else None
        if it["kind"] in ("part", "covers"):
            character = it["id"].split("__")[0]
            return next((n for n in self.order if C.the_key(n) == character), None)
        return None

    def sort(self, suite):
        """Shared work first, then model by model in the lot's order, each model's hairstyle before its pieces."""
        rank = {n: k for k, n in enumerate(self.order)}
        per = {it["id"]: self.model(it) for it in suite}
        suite.sort(key=lambda it: (-1 if per[it["id"]] is None else rank[per[it["id"]]],
                                   0 if it["kind"] in ("hairstyle", "hairstyle_free") else 1))
        for it in suite:
            n = per[it["id"]]
            self.rest[n] = self.rest.get(n, 0) + 1
            if it["kind"] == "hairstyle" and n is not None:
                self.hairstyles.add(n)
        return suite

    def start(self):
        """Models of the lot with nothing left in the suite are usable already."""
        for n in self.order:
            if not self.rest.get(n):
                self.say(model_done=n)
        self.current()

    def current(self):
        n = next((n for n in self.order if self.rest.get(n)), None)
        if n is not None:
            self.say(model=n, stage="hairstyle" if n in self.hairstyles else "parts")

    def lot_done(self, batch):
        with self.lock:
            for it in batch:
                n = self.model(it)
                self.rest[n] = self.rest.get(n, 1) - 1
                if it["kind"] == "hairstyle":
                    self.hairstyles.discard(n)
                if n is not None and self.rest[n] == 0:
                    self.say(model_done=n)
            self.say(done=len(batch))
            self.current()


def missing_for_prepare():
    """What is missing to prepare the cache (sentences; empty: ready to run). The repository provides neither a base
    body nor a base head: without them, prepare stops here instead of failing later on a missing file."""
    res = []
    import shutil
    blender = C.R.program("blender") or "blender"
    if not shutil.which(os.path.expanduser(blender)):
        res.append("Blender not found (\"%s\"): install it, or give its path with the setting programs.blender "
                   "(CHIMERA_BLENDER). See INSTALLATION.md." % blender)
    if not os.path.isfile(C.BODY):
        res.append("No base body: %s does not exist (setting workshop.body, CHIMERA_BODY). "
                   "Build it from a VRoid Studio character: python3 -I tools/vroid_base/build.py <model.vrm> "
                   "(or, without a VRM, the free base: python3 -I tools/free_base/build.py; see INSTALLATION.md, "
                   "\"What the repository does not provide\")." % C.BODY)
    card = card_base_head()
    if not os.path.isfile(os.path.join(C.BASE, "meta.json")) and not os.path.isfile(card) and not base_free():   # plan() reads this record
        res.append("No base head: neither the base's own (%s and coiffure.pmx, written next to the body by "
                   "tools/vroid_base/build.py or tools/free_base/build.py), nor the active game's \"%s\" (%s, head \"%s\": add the game's models, "
                   "see COMPATIBILITY.md)." % (HEAD_FREE, C.GAME.name, card, C.BASE_HEAD))
    if base_free() and not base_open() and os.path.isfile(card) and os.path.isfile(os.path.join(C.BASE, "meta.json")):
        print("Note: this cache was prepared with the free head and keeps it. To switch to the faces of the game \"%s\", "
              "give the workshop another cache (setting workshop.cache)." % C.GAME.name, file=sys.stderr)
    return res


def a_make(cat, body=None, cache=None):
    """What is missing for the workshop page to open, as sentences for the user (empty: everything is there).
    The page shows the cache's body (corps/corps.glb), then a face and a hairstyle from the catalogue: all three are needed."""
    body, cache = body or C.BODY, cache or C.CACHE
    res = []
    if not os.path.isfile(os.path.join(cache, "body", "body.glb")):
        if not os.path.isfile(body):
            res.append("No base body: %s does not exist (setting workshop.body). Build it from a VRoid Studio "
                       "character: python3 -I tools/vroid_base/build.py <model.vrm> (or, without a VRM, the free "
                       "base: python3 -I tools/free_base/build.py; see INSTALLATION.md, \"What the repository "
                       "does not provide\")." % body)
        else:
            res.append("The base body is not prepared for the page yet: "
                       "python3 -I tools/workshop/catalog.py prepare")
    if not cat.get("heads") or not cat.get("hairstyles"):
        res.append("Empty catalogue (%d face(s), %d hairstyle(s) kept in %s): the page needs at least one "
                   "face and one hairstyle. Add your models (COMPATIBILITY.md), set paths.data, then "
                   "python3 -I tools/workshop/catalog.py prepare"
                   % (len(cat.get("heads") or []), len(cat.get("hairstyles") or []), cache))
    return res


def prepare(level, stale_ones=False, order=None):
    """`order` (the server's extraction lot): models to make usable one after the other, with progress lines
    (Track) on the standard output."""
    missing = missing_for_prepare()
    if missing:
        sys.exit("catalog.py prepare: nothing started.\n- " + "\n- ".join(missing))
    track = Track(order) if order else None
    work, heads, hairstyles, lib, _ = plan(level)
    run([it for it in work if it["kind"] != "head"], "base")
    if any(it["kind"] == "base" for it in work) and not os.path.isfile(C.BASE_PMX):
        sys.exit("catalog.py prepare: the base (body and base head) could not be made, the rest is not started. "
                 "See above what Blender said.")
    # one face = a transfer of a few minutes, sharing nothing with the others: IN_PARALLEL at a time,
    # outside the Blender queue (the tool takes a place in it itself for its tint)
    t = [it for it in work if it["kind"] == "head"]
    if track and t:
        track.say(stage="transfer", heads=[it["id"] for it in t])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(IN_PARALLEL) as pool:
        list(pool.map(transfer, [it for it in t if it["library"] != C.BASE_HEAD and not it.get("keep_model")
                                   and not it.get("free")]))
    run(t, "heads")
    if track:
        track.say(heads_made=[it["id"] for it in t])
    suite = plan_next(level, heads, hairstyles, lib, stale_ones)
    if track:
        track.sort(suite)
        track.say(suite=len(suite))
        track.start()
    else:
        # hairstyles first: they are what the page waits for to show anything
        suite.sort(key=lambda it: 0 if it["kind"] == "hairstyle" else 1)
    run(suite, "suite", IN_PARALLEL, finished=track.lot_done if track else None)
    build()


def _reason(r):
    if not r:
        return "unknown"
    if "KeyError" in r:
        return "to prepare again: zero-weight bone mishandled by the graft (fixed, run prepare again)"
    return r


# folder of each kind of item in the cache (format 2)
FOLDERS = {"heads": "heads", "hairstyles": "hairstyles", "parts": "parts"}


def build(write=True):
    """Writes catalogue.json from the cache's records (write=False: only returns it)."""
    def cards(under):
        out = []
        for f in sorted(glob.glob(os.path.join(C.CACHE, FOLDERS[under], "*", "meta.json"))):
            try:      # a damaged record (interrupted write) drops its part, not the whole workshop
                m = C.read_json(f)
                if not isinstance(m, dict) or "id" not in m:
                    raise ValueError("record without an id")
                out.append(m)
            except (ValueError, OSError) as e:
                print("unreadable record, ignored: %s (%s)" % (f, e), file=sys.stderr, flush=True)
        return out

    heads, hairstyles, parts = cards("heads"), cards("hairstyles"), cards("parts")
    for m, the_key in [(m, "head") for m in heads] + [(m, "hairstyle") for m in hairstyles]:
        if m["id"] == FREE:      # the base's own: today's name, not the one written when it was prepared
            m["name"] = NAMES_BASE[the_key]
    for m in parts:      # tab, slot, rank: those of today's settings, not of the day it was prepared
        C.arrange(m)
        n = (m.get("revue") or {}).get("note")
        if m.get("kept") and n is not None and n > C.NOTE_REFUSALS:   # the record keeps kept: only the catalogue drops it
            m["kept"], m["reason"] = False, "shape damaged by fitting (review score %s, limit %s)" % (n, C.NOTE_REFUSALS)
    for m in hairstyles:   # the rule may have changed since posing: it is read again from the pairs
        if "pairs" in m:
            m["kept"], m["reason"] = C.hairstyle_kept(m["pairs"], m.get("own"))
    refusals = []
    for kind, items in (("face", heads), ("hairstyle", hairstyles), ("fraction", parts)):
        for m in items:
            if not m.get("kept"):
                refusals.append({"kind": kind, "id": m["id"], "name": _name(m), "reason": _reason(m.get("reason"))})
    def thumbnail(under, m):
        v = "/data/%s/%s/thumbnail.png" % (FOLDERS[under], m["id"])
        return v if os.path.isfile(os.path.join(C.CACHE, FOLDERS[under], m["id"], "thumbnail.png")) else None
    lbl = labels()

    def label(under, m):
        return lbl.get("%s/%s" % (FOLDERS[under], m["id"]))

    cat = {"heads": [{"id": m["id"], "name": m["name"], "skin": m.get("skin"), "thumbnail": thumbnail("heads", m), "faces": m["faces"],
                      "label": label("heads", m)}
                     for m in heads if m.get("kept")],
           "hairstyles": [{"id": m["id"], "name": m["name"], "own": m.get("own"), "faces": m["faces"],
                          "thumbnail": thumbnail("hairstyles", m), "label": label("hairstyles", m),
                          "pairs": {t: {"passed": p["passed"], "failures": p["failures"]} for t, p in m["pairs"].items()}}
                         for m in hairstyles if m.get("kept")],
           "parts": [{k: m.get(k) for k in ("id", "character", "part_name", "type", "type_name", "tab", "slot", "rank", "occupies",
                                             "faces", "class", "hidden_skin", "other_failures")}
                      | {"name": _name(m), "thumbnail": m.get("thumbnail") or thumbnail("parts", m), "label": label("parts", m)}
                      for m in parts if m.get("kept")],
           "refusals": refusals}
    src = sources()      # only the models in use, and only pages over https
    names = {m["id"] for m in cat["heads"] + cat["hairstyles"]} | {m.get("character") for m in cat["parts"]}
    cat["sources"] = {n: v for n, v in src.items() if n in names and isinstance(v, dict) and str(v.get("url", "")).startswith("https://")}
    cat["counts"] = {"prepared": len(heads) + len(hairstyles) + len(parts),
                      "kept_count": len(cat["heads"]) + len(cat["hairstyles"]) + len(cat["parts"]), "refused": len(refusals),
                      "heads": [len(cat["heads"]), len(heads)], "hairstyles": [len(cat["hairstyles"]), len(hairstyles)],
                      "parts": [len(cat["parts"]), len(parts)]}
    if write:
        C.write_json(os.path.join(C.CACHE, "catalog.json"), cat)
    return cat


_ANNEXES = {}


def _annex(name):
    """An optional JSON dictionary of the cache, read again when the file changes ({} without it)."""
    f = os.path.join(C.CACHE, name)
    try:
        the_key = (os.path.getmtime(f), os.path.getsize(f))
    except OSError:
        return {}
    if _ANNEXES.get(name, (None,))[0] != the_key:
        d = C.read_json(f)
        _ANNEXES[name] = (the_key, d if isinstance(d, dict) else {})
    return _ANNEXES[name][1]


def labels():
    """Readable labels of the items, from labels.json in the cache (optional):
    {"heads/<id>" | "hairstyles/<id>" | "parts/<id>": {"name": [en, fr], "description": [en, fr],
    "tags": [[en…], [fr…]]}}. Without a label an item keeps its code name."""
    return _annex("labels.json")


def sources():
    """Where each model comes from, from sources.json in the cache (optional): {"<model name>": {"url": page of
    the original model, "name", "author", "site", "reading": day the terms were read, "license": {VRM licence
    fields as read that day, or "name" and "licenseUrl" for a named licence such as CC0}}}. The name is a part's `character`, or the id of a face or hairstyle. The page
    links every item to it, so that the current terms can be read again at the source."""
    return _annex("sources.json")


def _name(m):
    if "character" in m:
        return "%s · %s" % (m["character"], m.get("type_name") or m["part_name"])
    return m.get("name", m["id"])


def assessment():
    cat = build()
    n = cat["counts"]
    print("prepared %d, kept %d, rejected %d" % (n["prepared"], n["kept_count"], n["refused"]))
    for k in ("heads", "hairstyles", "parts"):
        print(" ", k, "%d kept of %d" % tuple(n[k]))
    reasons = {}
    for r in cat["refusals"]:
        the_key = r["reason"].split(" = ")[0].split(":")[0].strip()
        reasons[the_key] = reasons.get(the_key, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print("  %4d  %s" % (v, k))
    metas = [C.read_json(f) or {} for f in glob.glob(os.path.join(C.CACHE, "parts", "*", "meta.json"))]
    stale = sum(expired(m) for m in metas)
    if stale:
        print("  %d outdated parts of %d (other fitting method or other body): prepare --stale redoes them" % (stale, len(metas)))
    nm = sum(bool(m.get("not_measured")) for m in metas)
    if nm:
        print("  %d parts kept with a blocking check not measured (not_measured in their record)" % nm)
    for m in a_make(cat):
        print("  to do:", m)


def add(folder):
    folder = os.path.abspath(folder.rstrip("/"))
    fi = C.read_json(os.path.join(folder, "record.json"))
    if fi is None:
        sys.exit("no fiche.json in %s" % folder)
    if not (fi.get("class") == "garment" and fi.get("type") in C.CLOTHES or
            fi.get("class") == "accessory" and fi.get("sort") in C.ACCESSORIES):
        sys.exit("part type not supported by the workshop: %s" % (fi.get("type") or fi.get("sort")))
    cid = C.the_key(fi["character"], fi["id"])
    launch_blender([{"kind": "part", "id": cid, "folder": folder}], "addition")
    m = C.read_json(os.path.join(C.CACHE, "parts", cid, "meta.json"))
    build()
    print("%s: %s" % (cid, "added to the catalogue" if m and m.get("kept") else "rejected: %s" % (m or {}).get("reason")))


LEVELS = {"small": "small", "full": "complete"}   # --level -> level name of the game descriptions (catalog.small)


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("prepare", "add", "summary"):
        sys.exit(__doc__)
    if a[0] == "prepare":
        level = a[a.index("--level") + 1] if "--level" in a else "full"
        if level not in LEVELS:
            sys.exit("--level: small or full")
        prepare(LEVELS[level], "--stale" in a,
                 [n for n in a[a.index("--order") + 1].split(",") if n] if "--order" in a else None)
    elif a[0] == "add":
        add(a[1])
    else:
        assessment()


if __name__ == "__main__":
    main()

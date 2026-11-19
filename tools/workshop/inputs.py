"""Inputs and outputs of the workshop, for the page's buttons: what the tool accepts and produces, importing a
model, extracting its parts into the catalogue, and downloading an assembled character.

What is said here comes from the code and from the game descriptions (tools/games/<game>/game.json), not written for
the page:
  - inputs: a VRoid VRM (0.x from VRoid Studio, 1.0 from VRoid Hub) goes through the VRoid importer (tools/vroid/importer.py); a PMX must already speak the common
    language (face skin "颜", bone "頭"…);
  - importing is only possible if the active game takes its models from a folder where everyone brings their own
    ("models": "*" in its description): a game with a declared list (ZZZ) only sees the models of its list;
  - extraction follows the chain of COMPATIBILITY.md: pivot head (if missing), faces, hairstyles, clothes, then
    catalog.py prepare (Blender through the bin/blender-queue queue, settings compute.*).

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

J = C.GAME
DATA = C.ATLAS
PIVOT = os.path.join(DATA, "pivot-head")      # where tools/head/library.py looks for it
EXTRACTIONS = os.path.join(C.CACHE, "extractions")
ASSEMBLIES = os.path.join(C.CACHE, "assemblies")
WAITING = os.path.join(C.CACHE, "imports-in-progress")

IMPORT_MAX = 4 << 30             # most bytes received (a VRM is 10 to 100 MB; an archive of several models, GBs)
ZIP_UNFOLDED_MAX = 12 << 30        # most bytes once unpacked
ZIP_FILES_MAX = 20000
ZIP_REPORT_MAX = 200            # unpacked / compressed size beyond which a large file is a bomb
NAME_MAX = 60
# the VRM importer is written for one game (its bone, material and morph tables live there): tools/vroid
INGESTER_VRM = ("vroid", os.path.join(C.TOOLS, "vroid", "importer.py"))
WHAT = ("faces", "hairstyles", "clothes")


class Refusals(Exception):
    """Rejection the user can read; `code`: HTTP status."""

    def __init__(self, message, code=400):
        super().__init__(message)
        self.code = code


# ===================================================================== formats

def _compat(game):
    return game.d.get("compatibility", {})


def _va_to_tip(game):
    """Can the game's models go all the way to the catalogue? It needs a list of models and a base head
    (pivot.py and catalog.py read them); without them the game cannot be the active game."""
    return "models" in game.d and "base_head" in game.d


def import_possible():
    """(possible, reason): importing writes into the active game's mmd_extracted source, which only a "*" game reads."""
    if J.value("models", {}) != "*":
        return False, ("The active game \"%s\" reads a declared list of models (\"models\" in its description): an "
                       "imported model would never be picked up. Place its files as COMPATIBILITY.md says, or make "
                       "active a game where everyone brings their own models (setting games.active)." % J.title)
    try:
        J.path("mmd_extracted")
    except KeyError:
        return False, "The active game \"%s\" has no mmd_extracted folder." % J.title
    return True, None


def formats():
    import games as games
    ok, reason = import_possible()
    vroid = games.load(INGESTER_VRM[0])
    cv = _compat(vroid)
    lim = cv.get("limits", [])
    input_ = [
        {"id": "vrm", "name": "VRM 0.x or 1.0 (VRoid Studio, VRoid Hub)", "extensions": [".vrm"], "state": cv.get("level", "partial"),
         "importable": ok,
         "note": " ".join(lim[:3]) or cv.get("summary", "")},
        {"id": "pmx", "name": "PMX (MikuMikuDance), in a .zip with its textures (one or more models per archive)", "extensions": [".zip"], "state": "partial",
         "importable": ok,
         "note": "Names must follow the common language of ZZZ's official MMD models (face skin \"颜\", "
                 "hair \"髮\", skin \"肌\", MMD bone \"頭\"): a PMX without them is rejected at import. "
                 "An archive may hold several models: each PMX that follows the common language is imported, the others "
                 "(a weapon, a prop) are set aside."},
    ]
    output = [
        {"id": "pmx", "name": "PMX (MikuMikuDance)",
         "note": "The assembled model and its textures, as a .zip. No physics (no rigid bodies for the hair, no springs)."},
        {"id": "vrm", "name": "VRM 1.0",
         "note": "Humanoid bones, 13 expressions, MToon materials, in T-pose. A VRM is a binary glTF file: renamed "
                 "to .glb, a glTF viewer opens it (without the VRM extensions). Missing if writing it failed."},
    ]
    # Only the skeletons the workshop reads are listed (games that can go all the way to the catalogue).
    skeletons = []
    for name in games.all_():
        if name == "homemade":
            continue
        game = games.load(name)
        c = _compat(game)
        if _va_to_tip(game):
            vrm = "os_mmd" in game.d
            skeletons.append({
                "id": name, "name": "%s: %s" % (game.title, "VRM humanoid" if vrm else "standard MMD bones"),
                "state": c.get("level", "partial"),
                "note": ("Humanoid bones renamed to MMD bones at import, breast bones merged into their parent, springs "
                         "not kept." if vrm else
                         "Bones of the MMD standard (センター, 上半身, 頭…) and materials of the common language, no conversion.")})
    sq = C.R.value("workshop.skeleton", "custom")
    return {"input": input_, "output": output, "skeletons": skeletons,
            "output_skeleton": {"id": sq, "note": "Bones with readable names (house skeleton) when its data exists, "
                                                   "otherwise the bones of the base body." if sq in ("custom", "own_rig") else
                                                   "The bones of the base body, kept as they are."},
            "active_game": {"id": J.name, "name": J.title, "import": ok, "reason": reason}}


# ===================================================================== names

def clean_name(text, max_=NAME_MAX):
    """Name given by the user, kept readable (letters of any language, digits, space, - _ . ') and bounded."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = "".join(c if (c.isalnum() or c in " -_.'") else " " for c in t)
    return re.sub(r"\s+", " ", t).strip(" .")[:max_].strip()


def identifier(name_file):
    """Identifier of a model taken from its file name: ASCII, safe for a folder and a URL."""
    base = os.path.basename(str(name_file).replace("\\", "/"))
    stem = os.path.splitext(base)[0]
    a = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    a = re.sub(r"[^A-Za-z0-9_-]+", "_", a).strip("_-")[:48]
    if not a or not a[0].isalnum():
        raise Refusals("Unreadable file name: rename the file with Latin letters or digits.")
    return a


def name_download(name, default):
    """(ASCII name, UTF-8 name) for Content-Disposition."""
    n = clean_name(name) or default
    a = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    a = re.sub(r"[^A-Za-z0-9_.-]+", "_", a).strip("_.") or default
    return a, n


# ===================================================================== import

def receive(flux, length, max_=IMPORT_MAX):
    """Writes the received body into a waiting file (never whole in memory). Returns its path."""
    if length <= 0:
        raise Refusals("Empty file.")
    if length > max_:
        raise Refusals("File too large (%d MB at most)." % (max_ >> 20), 413)
    os.makedirs(WAITING, exist_ok=True)
    path = os.path.join(WAITING, "received-%s" % secrets.token_hex(8))
    rest = length
    with open(path, "wb") as f:
        while rest:
            b = flux.read(min(rest, 1 << 20))
            if not b:
                break
            f.write(b)
            rest -= len(b)
    if rest:
        os.remove(path)
        raise Refusals("Upload interrupted.")
    return path


def header_vrm(path):
    """Checks, without converting anything, that the file is a VRM 0.x or 1.x. Returns the declared title (or None).
    Whether it is a VRoid character is said by the importer, which reads its materials."""
    import struct
    with open(path, "rb") as f:
        head = f.read(20)
        if len(head) < 20 or head[:4] != b"glTF":
            raise Refusals("This file is not a VRM (nor a binary glTF).")
        version, _, n, kind = struct.unpack_from("<III4s", head, 4)
        if version != 2 or kind != b"JSON" or n > 64 << 20:
            raise Refusals("This file is not a readable VRM (glTF 2.0 expected).")
        try:
            j = json.loads(f.read(n))
        except ValueError:
            raise Refusals("This file is not a readable VRM: damaged header.")
    ext = j.get("extensions") or {}
    if "VRM" in ext:
        return ((ext["VRM"].get("meta") or {}).get("title")) or None
    if "VRMC_vrm" in ext:
        v = ext["VRMC_vrm"] if isinstance(ext["VRMC_vrm"], dict) else {}
        if not str(v.get("specVersion", "")).startswith("1."):
            raise Refusals("This file declares a VRM version other than 0.x and 1.x: it is not read.")
        return ((v.get("meta") or {}).get("name")) or None
    raise Refusals("This file is a glTF without the VRM extension: only a VRM from VRoid (Studio or Hub) is read.")


def _under(root, rel):
    c = os.path.realpath(os.path.join(root, rel))
    if not c.startswith(os.path.realpath(root) + os.sep):
        raise Refusals("Archive rejected: a file escapes its folder (%s)." % rel[:80])
    return c


def _names_zip(infos):
    """The real name of each file of an archive. An archive that does not declare UTF-8 is read as cp437 by
    zipfile, which damages the Chinese or Japanese texture names (the PMX no longer finds them): the bytes are taken
    back and the first encoding that reads them all is kept (UTF-8, then Shift-JIS, then GB18030)."""
    raw = [None if i.flag_bits & 0x800 else i.filename.encode("cp437", "replace") for i in infos]
    if all(b is None or b.isascii() for b in raw):
        return [i.filename for i in infos]
    for encoding in ("utf-8", "cp932", "gb18030"):
        try:
            return [i.filename if b is None else b.decode(encoding) for i, b in zip(infos, raw)]
        except UnicodeDecodeError:
            continue
    return [i.filename for i in infos]


def unfold_zip(path, folder):
    """Unpacks a received archive into `dossier` (new), with its guards: paths that escape, links, encrypted
    archive, too many files, too large, compression ratio of a bomb. Returns the relative paths of its PMX files
    (an archive may carry several models, or a character and its weapon)."""
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise Refusals("This file is not a readable .zip archive.")
    with z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) > ZIP_FILES_MAX:
            raise Refusals("Archive rejected: more than %d files." % ZIP_FILES_MAX)
        total = 0
        names = _names_zip(infos)
        for i, n in zip(infos, names):
            if i.flag_bits & 0x1:
                raise Refusals("Encrypted archive: unpack it yourself and make a .zip again without a password.")
            if stat.S_ISLNK(i.external_attr >> 16):
                raise Refusals("Archive rejected: it contains a link (%s)." % n[:80])
            if n.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", n) or ".." in re.split(r"[\\/]", n) or "\0" in n:
                raise Refusals("Archive rejected: a file escapes its folder (%s)." % n[:80])
            total += i.file_size
            if i.file_size > (1 << 20) and i.file_size > ZIP_REPORT_MAX * max(1, i.compress_size):
                raise Refusals("Archive rejected: abnormal compression (%s)." % n[:80])
        if total > ZIP_UNFOLDED_MAX:
            raise Refusals("Archive rejected: more than %d GB once unpacked." % (ZIP_UNFOLDED_MAX >> 30))
        pmx = [n for n in names if n.lower().endswith(".pmx")]
        if not pmx:
            raise Refusals("The archive contains no .pmx file.")
        os.makedirs(folder)
        written = 0
        for i, n in zip(infos, names):
            target = _under(folder, n)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(i) as src, open(target, "wb") as dst:
                lu = 0
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    lu += len(b)
                    written += len(b)
                    if lu > i.file_size or written > ZIP_UNFOLDED_MAX:      # declared size is a lie
                        raise Refusals("Archive rejected: a file is larger than it declares.")
                    dst.write(b)
    return sorted(pmx)


def read_pmx_isolated(path):
    """Materials, bones and textures of a PMX, read by an isolated Python: a booby-trapped file does not touch the server."""
    r = subprocess.run([C.PYTHON, "-I", os.path.abspath(__file__), "lire-pmx", path], cwd=C.REPO,
                       capture_output=True, text=True, timeout=300)
    if r.returncode:
        raise Refusals("This .pmx file is not readable: %s" % ((r.stderr.strip().splitlines() or ["?"])[-1][:200]))
    return json.loads(r.stdout)


def check_language_common(info):
    """What the core requires of a PMX (tools/games/README.md, "common language"): the face skin 颜 (head.Head
    measures the head on it) and the head bone 頭."""
    missing = []
    if "颜" not in info["materials"]:
        missing.append("no material \"颜\" (face skin)")
    if "頭" not in info["bones"]:
        missing.append("no bone \"頭\" (head)")
    if missing:
        raise Refusals("This PMX does not follow the workshop's common language: %s. See tools/games/README.md." % ", ".join(missing))


def importer(path_received, name_file):
    """Stores a received file as a model of the active game: <mmd_extracted>/<id>/<id>.pmx and its textures. Returns
    the record of the first model stored. The received file is deleted in every case."""
    return importer_lot(path_received, name_file)[0][0]


def importer_lot(path_received, name_file):
    """Like importer, for an archive that may carry several models: returns (records, set aside). A set-aside file is
    {"file", "reason"}: a .pmx of the archive that is not a readable character (a weapon), or already imported."""
    try:
        return _importer(path_received, name_file)
    finally:
        if os.path.exists(path_received):
            os.remove(path_received)


def _card(dest, ident, title, fmt, info, missing):
    card = {"id": ident, "name": title or ident, "format": fmt, "materials": info["materials"], "bones": len(info["bones"]),
             "textures": len(info["textures"]), "vertices": info["vertices"]}
    if missing:
        card["textures_missing"] = missing[:20]
    C.write_json(os.path.join(dest, "import.json"), {k: card[k] for k in ("id", "name", "format")} | {"date": time.time()})
    return card


def _ids_of_lot(ident, good):
    """One identifier per model of an archive that carries several: the name of its folder if it is alone there,
    otherwise the name of its file, otherwise the archive's followed by a number. No duplicate within the batch."""
    per_folder = {}
    for rel, _, _ in good:
        per_folder[os.path.dirname(rel)] = per_folder.get(os.path.dirname(rel), 0) + 1
    taken, names = set(), []
    for k, (rel, _, _) in enumerate(good):
        d = os.path.dirname(rel)
        candidates = ([os.path.basename(d)] if d and per_folder[d] == 1 else []) + [os.path.basename(rel)]
        if d and per_folder[d] > 1:
            candidates.insert(0, "%s-%s" % (os.path.basename(d), os.path.splitext(os.path.basename(rel))[0]))
        name = None
        for c in candidates:
            try:
                name = identifier(c + ".x") if not c.lower().endswith(".pmx") else identifier(c)
            except Refusals:
                continue
            if name not in taken:
                break
            name = None
        if name is None:
            n = k + 1
            while "%s_%d" % (ident, n) in taken:
                n += 1
            name = "%s_%d" % (ident, n)
        taken.add(name)
        names.append(name)
    return names


def _importer(path_received, name_file):
    ok, reason = import_possible()
    if not ok:
        raise Refusals(reason, 409)
    ident = identifier(name_file)
    ext = os.path.splitext(str(name_file))[1].lower()
    if ext not in (".vrm", ".zip"):
        raise Refusals("Unsupported format (%s): a .vrm (VRoid Studio or VRoid Hub) or a .zip containing a .pmx."
                    % (ext or "no extension"), 415)
    dest_root = J.path("mmd_extracted")
    dest = os.path.join(dest_root, ident)
    already = "A model \"%s\" is already imported: rename the file to make another one."
    if ext == ".vrm" and os.path.exists(dest):
        raise Refusals(already % ident, 409)
    os.makedirs(dest_root, exist_ok=True)
    work = os.path.join(WAITING, "import-%s" % secrets.token_hex(8))
    cards, discarded = [], []
    try:
        if ext == ".vrm":
            title = header_vrm(path_received)
            os.makedirs(work)
            vrm = os.path.join(work, ident + ".vrm")
            shutil.move(path_received, vrm)
            output = os.path.join(work, "model")
            r = subprocess.run([C.PYTHON, "-I", INGESTER_VRM[1], vrm, "--output", output], cwd=C.REPO,
                               capture_output=True, text=True, timeout=900)
            if r.returncode or not os.path.isfile(os.path.join(output, ident, ident + ".pmx")):
                d = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ["no message"])[-1]
                if r.returncode == 2 and d.startswith("refused: "):     # a VRM the importer does not read, with its reason
                    raise Refusals("This VRM is not read: %s." % d[len("refused: "):].rstrip(".")[:300], 422)
                d = re.sub(r"^\w+Error: ", "", d).replace(vrm, "the file")
                raise Refusals("The VRM conversion failed: %s" % d[:240], 422)
            info = read_pmx_isolated(os.path.join(output, ident, ident + ".pmx"))
            shutil.move(os.path.join(output, ident), dest)
            try:            # the original .vrm stored in the game's "vrm" source, if it has one
                rank = J.path("vrm")
                os.makedirs(rank, exist_ok=True)
                if not os.path.exists(os.path.join(rank, ident + ".vrm")):
                    shutil.move(vrm, os.path.join(rank, ident + ".vrm"))
            except KeyError:
                pass
            cards.append(_card(dest, ident, title, "vrm", info, []))
        else:
            unfolded = os.path.join(work, "unfolded")
            os.makedirs(work)
            good = []
            for rel in unfold_zip(path_received, unfolded):
                pmx = os.path.join(unfolded, rel)
                try:
                    info = read_pmx_isolated(pmx)
                    check_language_common(info)
                except Refusals as e:
                    discarded.append({"file": rel, "reason": str(e)})
                    continue
                good.append((rel, pmx, info))
            if not good:        # nothing readable: the reason of the first file (an archive of a single PMX gives its own)
                raise Refusals(discarded[0]["reason"] if len(discarded) == 1 else
                            "None of the %d .pmx files of the archive is a readable character. The first one: %s"
                            % (len(discarded), discarded[0]["reason"]))
            names = [ident] if len(good) == 1 else _ids_of_lot(ident, good)
            folders = {os.path.dirname(p) for _, p, _ in good}
            for name, (rel, pmx, info) in zip(names, good):
                target = os.path.join(dest_root, name)
                if os.path.exists(target):
                    if len(good) == 1:
                        raise Refusals(already % name, 409)
                    discarded.append({"file": rel, "reason": "A model \"%s\" is already imported." % name})
                    continue
                # the PMX's folder becomes the model's (textures are named relative to it); the other PMX files
                # and the folders of the archive's other models are not copied into it
                root = os.path.dirname(pmx)

                def out_of(d, names_, root=root, pmx=pmx):
                    return [n for n in names_ if (n.lower().endswith(".pmx") and os.path.join(d, n) != pmx)
                            or (os.path.join(d, n) in folders and os.path.join(d, n) != root)]
                shutil.copytree(root, target, ignore=out_of)
                os.replace(os.path.join(target, os.path.basename(pmx)), os.path.join(target, name + ".pmx"))
                missing = [t for t in info["textures"] if t and not os.path.isfile(
                    os.path.join(target, t.replace("\\", "/")))]
                cards.append(_card(target, name, None, "pmx", info, missing))
            if not cards:
                raise Refusals("Every model of this archive is already imported.", 409)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return cards, discarded


# ===================================================================== state of the models

def _index_faces():
    return C.read_json(os.path.join(C.LIBRARY_VIS, "index.json"), {}) or {}


def state_models(cat=None):
    """Each model of the active game: what was extracted from it (libraries), kept in the catalogue, and rejected."""
    import catalog as K
    cat = cat if cat is not None else K.build(write=False)
    idx = _index_faces()
    not_provided = idx.get("not_provided", {})
    check = C.read_json(os.path.join(C.LIBRARY_GARMENT, "check.json"), {}) or {}
    heads = {t["id"] for t in cat.get("heads", [])}
    hairstyles = {c["id"] for c in cat.get("hairstyles", [])}
    kept_count = {}
    for p in cat.get("parts", []):
        kept_count[p.get("character")] = kept_count.get(p.get("character"), 0) + 1
    refusals_cat = cat.get("refusals", [])
    places = K.cards_library()      # parts whose kind has a place in the workshop
    try:
        root = J.path("mmd_extracted")
    except KeyError:
        root = None
    res = []
    for n, rel in sorted(J.models().items()):
        imp = C.read_json(os.path.join(root, n, "import.json"), {}) if root else {}
        garment = os.path.join(C.LIBRARY_GARMENT, n)
        refusals = []
        if n in not_provided:
            refusals.append({"name": "face", "reason": str(not_provided[n].get("reason", ""))[:300]})
        if isinstance(check.get(n), dict) and check[n].get("error"):
            refusals.append({"name": "clothes", "reason": "cutting failed: %s" % check[n]["error"][:250]})
        taken = {os.path.realpath(d) for d, _ in places}
        for f in sorted(glob.glob(os.path.join(garment, "*", "record.json"))):
            if os.path.realpath(os.path.dirname(f)) not in taken:
                fi = C.read_json(f, {}) or {}
                refusals.append({"name": "part %s" % os.path.basename(os.path.dirname(f)),
                              "reason": "the workshop does not offer this kind of part on its own (%s)"
                                        % (fi.get("type") or fi.get("sort") or fi.get("class") or "no kind")})
        pref = C.the_key(n) + "__"
        for r in refusals_cat:
            if r["id"] == n or r["id"].startswith(pref):
                refusals.append({"name": "%s %s" % (r["kind"], r["name"]), "reason": r["reason"]})
        res.append({
            "id": n, "name": (imp or {}).get("name") or n, "format": (imp or {}).get("format") or J.d.get("format", "pmx"),
            "present": bool(root) and os.path.isfile(os.path.join(root, rel)),
            "extracted": {"face": os.path.isfile(os.path.join(C.LIBRARY_VIS, n, "part.npz")),
                        "hairstyle": os.path.isfile(os.path.join(C.LIBRARY_VIS, n, "hairstyle.json")),
                        "parts": len(glob.glob(os.path.join(garment, "*", "record.json")))},
            "catalog": {"face": n in heads, "hairstyle": n in hairstyles, "parts": kept_count.get(n, 0)},
            "refusals": refusals})
    return res


# ===================================================================== extraction

_extraction = threading.Lock()
IN_PROGRESS = {"id": None, "work": None}     # the running extraction (id, Extraction object), None when idle


class Cancelled(Exception):
    """The extraction was cancelled from the page (POST /api/extraction/cancel)."""


def _new_id(folder, prefix=""):
    base, k = prefix + time.strftime("%Y%m%d-%H%M%S"), 1
    ident = base
    while os.path.exists(os.path.join(folder, ident)):
        k += 1
        ident = "%s-%d" % (base, k)
    os.makedirs(os.path.join(folder, ident))
    return ident


def request_extraction(v):
    """Validated body of POST /api/extract: (model names, what, redo)."""
    if not isinstance(v, dict):
        raise Refusals("Unreadable request.")
    known = J.models()
    if v.get("all") is True:
        names = sorted(known)
    else:
        names = v.get("models")
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise Refusals("Choose at least one model (\"models\") or all of them (\"all\": true).")
        unknown = [n for n in names if n not in known]
        if unknown:
            raise Refusals("Model unknown to the active game: %s." % ", ".join(n[:40] for n in unknown[:5]))
        names = sorted(set(names))
    if not names:
        raise Refusals("No model in the active game: import one first.")
    what = v.get("what", list(WHAT))
    if not isinstance(what, list) or not what or any(q not in WHAT for q in what):
        raise Refusals("\"what\": a list among %s." % ", ".join(WHAT))
    return names, [q for q in WHAT if q in what], v.get("redo") is True


def launch_extraction(names, what, redo=False, run=True):
    """Starts the chain in the background; one at a time. Returns the job's identifier.
    The extraction runs in the server process, not in the page: closing the tab does not stop it."""
    if not _extraction.acquire(blocking=False):
        raise Refusals("An extraction is already running (%s): wait for it to finish." % IN_PROGRESS["id"], 409)
    try:
        os.makedirs(EXTRACTIONS, exist_ok=True)
        ident = _new_id(EXTRACTIONS, "x")
        d = os.path.join(EXTRACTIONS, ident)
        C.write_json(os.path.join(d, "request.json"), {"models": names, "what": what, "redo": redo})
        C.write_json(os.path.join(d, "state.json"), {"step": "waiting", "fraction": 0.0, "message": "Waiting…", "done": False})
        work = Extraction(d, names, what, redo)
        IN_PROGRESS.update(id=ident, work=work)
    except BaseException:
        _extraction.release()
        raise
    if not run:          # tests: nothing is started
        IN_PROGRESS.update(id=None, work=None)
        _extraction.release()
        return ident

    def thread():
        try:
            work.lead()
        finally:
            IN_PROGRESS.update(id=None, work=None)
            _extraction.release()
    threading.Thread(target=thread, daemon=True).start()
    return ident


def extraction_in_progress():
    """GET /api/extraction: the running extraction, if any, with its state ({"id": None} when idle)."""
    ident = IN_PROGRESS["id"]
    if ident is None:
        return {"id": None}
    e = read_work(ident) or {"step": "waiting", "done": False}
    return dict(e, id=ident)


def cancel_extraction():
    """POST /api/extraction/cancel: stop the running extraction and the programs it launched. Returns its
    identifier; the state becomes "cancelled" once its programs have stopped."""
    work, ident = IN_PROGRESS["work"], IN_PROGRESS["id"]
    if work is None:
        raise Refusals("No extraction is running.", 409)
    work.cancel()
    return ident


class Extraction:
    # Lots of C.places() models (compute.slots): each lot goes through the whole chain (faces, hairstyles,
    # clothes, catalogue) before the next one starts, and inside a lot the catalogue makes the models usable one
    # after the other. A lot of `places` models keeps every place busy in the long parallel steps (face
    # sampling, face transfer, pieces); fixed costs per lot are small (Blender starts in under a second).
    STOP = 8            # seconds between SIGTERM and SIGKILL when cancelling

    def __init__(self, folder, names, what, redo):
        self.d, self.names, self.what, self.redo = folder, names, what, redo
        self.state = {"step": "waiting", "fraction": 0.0, "done": False, "start": time.time(), "models": names, "what": what,
                     "finished": [], "version": 0}
        self.log = os.path.join(folder, "log.txt")
        self.lock = threading.Lock()
        self.proc = None
        self.stop = threading.Event()

    # ------------------------------------------------------------ state
    def place(self, **kw):
        with self.lock:
            self.state.update(kw)
            C.write_json(os.path.join(self.d, "state.json"), self.state)

    def note(self, text):
        with open(self.log, "a", encoding="utf-8") as j:
            j.write(time.strftime("%H:%M:%S ") + text + "\n")

    def part(self, batch, in_):
        """Bar position: the pivot takes the first 5 %, then each lot an equal share; `in_` in [0, 1]."""
        n = max(1, len(self.batches))
        return round(0.05 + 0.95 * (batch + min(1.0, max(0.0, in_))) / n, 3)

    # ------------------------------------------------------------ programs
    def cancel(self):
        """Ask the running chain to stop (from the server's request thread): SIGTERM to the process group of the
        running tool, SIGKILL to what is left of it STOP seconds later."""
        if self.stop.is_set():
            return
        self.stop.set()
        self.note("cancellation requested")
        self.place(message="Cancelling…", cancellation=True)
        p = self.proc
        if p is not None:
            self._signal(p.pid, signal.SIGTERM)
            threading.Timer(self.STOP, self._signal, (p.pid, signal.SIGKILL)).start()

    @staticmethod
    def _signal(group, sig):
        try:
            os.killpg(group, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _wait_group(self, group):
        """After the leader exits, its group may still hold programs (Blender under systemd-run, process pools):
        wait for them to go, then kill what is left."""
        end = time.time() + self.STOP
        while self._signal(group, 0):
            if time.time() > end:
                self._signal(group, signal.SIGKILL)
                time.sleep(0.2)
                if time.time() > end + 5:
                    break
            time.sleep(0.2)

    def command(self, args, line=None):
        """Run a tool of the repository in its own process group (all the programs it starts, Blender included,
        stay in that group, systemd-run --scope too): cancelling stops the whole group. Lines starting with
        Track.MARK go to `line`, the others to the journal."""
        if self.stop.is_set():
            raise Cancelled()
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        with open(self.log, "ab") as j:
            j.write(("$ " + " ".join(os.path.basename(a) if a.endswith(".py") else a for a in args[2:]) + "\n").encode())
            j.flush()
            p = subprocess.Popen([C.PYTHON, "-I"] + args, cwd=C.REPO, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            self.proc = p
            if self.stop.is_set():          # cancelled between the check above and the start
                self._signal(p.pid, signal.SIGTERM)
            try:
                for raw in p.stdout:
                    if raw.startswith(b"@@ "):
                        if line and not self.stop.is_set():
                            try:
                                line(json.loads(raw[3:].decode("utf-8")))
                            except ValueError:
                                pass
                        continue
                    j.write(raw)
                    j.flush()
                rc = p.wait()
            finally:
                if self.stop.is_set():
                    self._wait_group(p.pid)
                    self._clean(p.pid)
                self.proc = None
        if self.stop.is_set():
            raise Cancelled()
        return rc

    def _clean(self, pid):
        """Job lists that catalogue.lancer_blender left behind when its Blender was stopped."""
        tmp = os.environ.get("TMPDIR") or C.CACHE
        for f in glob.glob(os.path.join(tmp, "atelier-*-%d-*.json" % pid)):
            try:
                os.remove(f)
            except OSError:
                pass

    def missing(self, test, names=None):
        return [n for n in (self.names if names is None else names) if self.redo or not test(n)]

    # ------------------------------------------------------------ chain
    def lead(self):
        """Run the chain to its end, its failure or its cancellation; the state file always ends "finished"."""
        try:
            self.chain()
        except Cancelled:
            self.note("cancelled")
            self.place(step="cancelled", done=True, ok=False, cancelled=True, model=None,
                       message="Extraction cancelled. What was finished stays in the catalogue; extracting again resumes from there.",
                       duration=round(time.time() - self.state["start"], 1))
        except Exception as e:   # noqa: BLE001 — any failure ends the job and is reported
            self.note("error: %r" % e)
            self.place(step="error", done=True, ok=False, error=str(e) if isinstance(e, Refusals) else "%s: %s" % (type(e).__name__, e),
                       message="The extraction failed.")

    def chain(self):
        """Pivot head (once, it needs every model), then the lots one after the other."""
        import catalog as K
        before = state_models()
        places = max(1, min(C.places(), 12))     # beyond 12 programs at once a desktop machine slows down
        tasks = str(places)
        tools = C.TOOLS
        self.batches = [self.names[k:k + places] for k in range(0, len(self.names), places)]
        self.place(batches=len(self.batches), total=len(self.names))
        if not os.path.isfile(os.path.join(PIVOT, "pivot-head.json")):
            self.place(step="pivot", stage="pivot", message="Pivot head: the head closest to the game's average…")
            if self.command([os.path.join(tools, "head", "pivot.py"), "--output", PIVOT, "--no-sheet",
                              "--jobs", tasks]) or not os.path.isfile(os.path.join(PIVOT, "pivot-head.json")):
                raise Refusals("The pivot head could not be computed (see the log).")
        missing = K.missing_for_prepare()
        if missing:
            raise Refusals("The catalogue cannot be prepared: " + " ".join(missing))
        for k, batch in enumerate(self.batches):
            self.batch(k, batch, tasks)
        after = state_models()
        self.place(step="done", fraction=1.0, done=True, ok=True, message="Extraction finished.", model=None,
                   assessment=assessment(before, after, self.names), duration=round(time.time() - self.state["start"], 1))

    def batch(self, k, batch, tasks):
        """One lot through the whole chain: faces, hairstyles, clothes, then the catalogue, model by model."""
        tools = C.TOOLS
        de = sum(len(x) for x in self.batches[:k])
        ranks = {n: de + i + 1 for i, n in enumerate(batch)}
        reach = {"from": de + 1, "to": de + len(batch), "total": len(self.names), "names": batch}

        def step(stage, in_, message):
            self.place(step=stage, stage=stage, batch=reach, model=None, fraction=self.part(k, in_), message=message)

        rank = "%d-%d of %d" % (reach["from"], reach["to"], reach["total"])
        # faces: prerequisite of hairstyles and clothes (both read the face library)
        v = self.missing(lambda n: os.path.isfile(os.path.join(C.LIBRARY_VIS, n, "record.json")), batch)
        if v:
            step("faces", 0.0, "Models %s: faces" % rank)
            self.command([os.path.join(tools, "head", "library.py"), "extract", "--output", C.LIBRARY_VIS,
                           "--jobs", tasks, "--only", ",".join(v)])
        with_face = [n for n in batch if os.path.isfile(os.path.join(C.LIBRARY_VIS, n, "part.npz"))]
        if "hairstyles" in self.what:
            c = [n for n in with_face if self.redo or not os.path.isfile(os.path.join(C.LIBRARY_VIS, n, "hairstyle.json"))]
            if c:
                step("hairstyles", 0.35, "Models %s: hairstyles" % rank)
                self.command([os.path.join(tools, "head", "hairstyle.py"), "extract", "--library", C.LIBRARY_VIS,
                               "--jobs", tasks, "--only", ",".join(c)])
        if "clothes" in self.what:
            w = [n for n in with_face if self.redo or not os.path.isfile(os.path.join(C.LIBRARY_GARMENT, n, "cutout.json"))]
            if w:
                step("clothes", 0.4, "Models %s: cutting the clothes" % rank)
                self.command([os.path.join(tools, "clothes", "library.py"), "extract", "--output", C.LIBRARY_GARMENT,
                               "--only", ",".join(w)])
        step("catalog", 0.45, "Models %s: fitting the faces on the base body" % rank)
        suite = {"total": 0, "done": 0}

        def line(m):
            """Progress lines of catalog.py prepare (catalog.Track)."""
            if "heads_made" in m:
                self.place(version=self.state["version"] + 1, fraction=self.part(k, 0.7))
            if "suite" in m:
                suite["total"] = m["suite"]
            if "done" in m:
                suite["done"] += m["done"]
                self.place(fraction=self.part(k, 0.7 + 0.3 * suite["done"] / max(1, suite["total"])))
            if "model_done" in m and m["model_done"] in ranks and m["model_done"] not in self.state["finished"]:
                self.note("model ready in the catalogue: %s" % m["model_done"])
                self.place(finished=self.state["finished"] + [m["model_done"]], version=self.state["version"] + 1)
            if "model" in m and m["model"] in ranks:
                stage = {"hairstyle": "hairstyle", "parts": "clothes"}.get(m.get("stage"), m.get("stage"))
                self.place(model={"rank": ranks[m["model"]], "total": len(self.names), "name": m["model"],
                                   "stage": m.get("stage")},
                           message="Model %d of %d: %s, %s" % (ranks[m["model"]], len(self.names), m["model"], stage))
        if self.command([os.path.join(C.HERE, "catalog.py"), "prepare", "--order", ",".join(batch)], line):
            raise Refusals("Preparing the catalogue stopped with an error (see the log).")
        # models of the lot that the catalogue had nothing to do for (already there, or refused before it)
        missing = [n for n in batch if n not in self.state["finished"]]
        self.place(finished=self.state["finished"] + missing, version=self.state["version"] + 1, model=None,
                   fraction=self.part(k, 1.0))


def assessment(before, after, names):
    a = {m["id"]: m for m in before}
    res = {"face_count": [], "hairstyles": [], "parts": 0, "models": {}, "refusals": []}
    for m in after:
        n = m["id"]
        if n not in names:
            continue
        av = a.get(n, {"catalog": {"face": False, "hairstyle": False, "parts": 0}})
        if m["catalog"]["face"] and not av["catalog"]["face"]:
            res["face_count"].append(n)
        if m["catalog"]["hairstyle"] and not av["catalog"]["hairstyle"]:
            res["hairstyles"].append(n)
        res["parts"] += max(0, m["catalog"]["parts"] - av["catalog"]["parts"])
        res["models"][n] = {"extracted": m["extracted"], "catalog": m["catalog"]}
        res["refusals"] += [{"name": "%s · %s" % (n, r["name"]), "reason": r["reason"]} for r in m["refusals"]]
    return res


def read_work(ident):
    """State of a job (extraction "x…" or assembly), with the end of the log for an extraction."""
    if not re.fullmatch(r"x?[0-9]{8}-[0-9]{6}(-[0-9]{1,4})?", ident or ""):
        return None
    if ident.startswith("x"):
        d = os.path.join(EXTRACTIONS, ident)
        e = C.read_json(os.path.join(d, "state.json"))
        if e is None:
            return None
        try:
            with open(os.path.join(d, "log.txt"), "rb") as f:
                f.seek(max(0, os.path.getsize(f.name) - 6000))
                lines = f.read().decode("utf-8", "replace").splitlines()[1:]
            e["log"] = [l[:300] for l in lines if l.strip() and not l.startswith(("Blender ", "Blender quit"))][-25:]
        except OSError:
            e["log"] = []
        return e
    return C.read_json(os.path.join(ASSEMBLIES, ident, "state.json"))


# ===================================================================== assemblages

def assemblies():
    res = []
    if not os.path.isdir(ASSEMBLIES):
        return res
    for ident in sorted(os.listdir(ASSEMBLIES), reverse=True):
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}(-[0-9]{1,4})?", ident):
            continue
        d = os.path.join(ASSEMBLIES, ident)
        sel = C.read_json(os.path.join(d, "selection.json"), {}) or {}
        e = C.read_json(os.path.join(d, "state.json"), {}) or {}
        fmts = [f for f, c in (("pmx", "result.pmx"), ("vrm", "result.vrm")) if os.path.isfile(os.path.join(d, "model", c))]
        res.append({"id": ident, "date": "%s-%s-%sT%s:%s:%s" % (ident[:4], ident[4:6], ident[6:8], ident[9:11], ident[11:13], ident[13:15]),
                    "name": sel.get("name") or None, "head": sel.get("head"), "hairstyle": sel.get("hairstyle"),
                    "parts": len(sel.get("parts") or []),
                    "sheet": "/data/assemblies/%s/sheet.png" % ident if os.path.isfile(os.path.join(d, "sheet.png")) else None,
                    "formats": fmts, "done": bool(e.get("done")), "ok": e.get("ok"), "step": e.get("step"),
                    "error": e.get("error"), "vrm_note": e.get("vrm_note") or None})
    return res


def export(ident, fmt):
    """(path to send, ASCII file name, UTF-8 name, type, temporary?) of an assembly."""
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}(-[0-9]{1,4})?", ident) or fmt not in ("pmx", "vrm"):
        raise Refusals("Unknown assembly or format.", 404)
    d = os.path.join(ASSEMBLIES, ident)
    m = os.path.join(d, "model")
    sel = C.read_json(os.path.join(d, "selection.json"), {}) or {}
    a, u = name_download(sel.get("name") or "", "character-" + ident)
    if fmt == "vrm":
        c = os.path.join(m, "result.vrm")
        if not os.path.isfile(c):
            raise Refusals("This assembly has no VRM.", 404)
        return c, a + ".vrm", u + ".vrm", "model/gltf-binary", False
    pmx = os.path.join(m, "result.pmx")
    if not os.path.isfile(pmx):
        raise Refusals("This assembly has no PMX.", 404)
    tmp = os.path.join(d, ".export-%s.zip" % secrets.token_hex(6))
    root = os.path.realpath(m)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        z.write(pmx, "%s/%s.pmx" % (a, a))
        for base, dirs, files in os.walk(m):
            dirs[:] = sorted(x for x in dirs if not os.path.islink(os.path.join(base, x)))
            for f in sorted(files):
                c = os.path.join(base, f)
                rel = os.path.relpath(c, m)
                # the PMX on the original bones (result-zzz.pmx) and the work files stay in the cache
                if f.lower().endswith((".pmx", ".vrm", ".json")) or os.path.islink(c) or \
                        not os.path.realpath(c).startswith(root + os.sep):
                    continue
                z.write(c, "%s/%s" % (a, rel.replace(os.sep, "/")),
                        compress_type=zipfile.ZIP_STORED if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) else None)
    return tmp, a + "-pmx.zip", u + "-pmx.zip", "application/zip", True


# ===================================================================== command line

def _read_pmx(path):
    sys.path.insert(0, os.path.join(C.TOOLS, "atlas"))
    import pmx as PMX
    P = PMX.read(path)
    print(json.dumps({"materials": [m["name"] for m in P["materials"]], "bones": [o["name"] for o in P["bones"]],
                      "textures": P["textures"], "vertices": len(P["vertices"]["pos"])}, ensure_ascii=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["lire-pmx"] and len(a) == 2:
        _read_pmx(a[1])
    elif a[:1] == ["formats"]:
        print(json.dumps(formats(), ensure_ascii=False, indent=1))
    else:
        sys.exit(__doc__)

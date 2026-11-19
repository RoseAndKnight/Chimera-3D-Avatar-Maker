"""Paths, constants and small helpers shared by Chimera's workshop.

Nothing here imports Blender. Folders and programs come from the settings (tools/settings.py):
data in paths.data (CHIMERA_DATA), cache (light parts, assemblies) in workshop.cache (CHIMERA_CACHE).
"""
import hashlib
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
REPO = os.path.dirname(TOOLS)
# head first: tools/head/library.py and tools/clothes/library.py have the same name
for under in ("clothes", "atlas", "head"):
    path = os.path.join(TOOLS, under)
    if path not in sys.path:
        sys.path.insert(0, path)
if TOOLS not in sys.path:
    sys.path.append(TOOLS)
import settings as R  # noqa: E402
import dataformat as FMT  # noqa: E402  refusal of a format-1 data folder (French names)
import games as games  # noqa: E402  what is specific to the models' game: tools/games/<game>/game.json

ATLAS = R.path("paths.data")
# Since 12/11 the workshop starts from the remade body (tools/rebuilt_body) carrying the remade head (tools/rebuilt_head). The previous
# workshop (ZZZ body, cache <data>/atelier) remains whole, with its own launcher (owner's machine).
# Since 13/11: body and head in our own shape, painted skin (research tool, not published); the remade body keeps its
# cache and its own launcher (owner's machine).
CACHE = R.path("workshop.cache")
# A data folder of format 1 (French names, before 19/11/2026) is refused with the command that converts it.
FMT.refused_format1(ATLAS, CACHE)
LIBRARY_GARMENT = os.path.join(ATLAS, "clothes-library")
LIBRARY_VIS = os.path.join(ATLAS, "face-library")
# CHIMERA_BODY: another base body, always with its own cache (CHIMERA_CACHE)
BODY = R.path("workshop.body")
# Fallback: as long as the VRoid base of the public default is not built (tools/vroid_base, a .vrm is needed), the free
# MakeHuman base (tools/free_base) is used if it is built, with its cache. A body set by hand has no fallback.
_DATA = R.path("paths.data")
if (not os.path.isfile(BODY) and BODY == os.path.join(_DATA, "vroid-base", "body.pmx")
        and CACHE == os.path.join(_DATA, "workshop-vroid-base")
        and os.path.isfile(os.path.join(_DATA, "free-base", "body.pmx"))):
    BODY, CACHE = os.path.join(_DATA, "free-base", "body.pmx"), os.path.join(_DATA, "workshop-free-base")
# Base head: the whole head of the game's pivot head (base_head in its description: face, ears and neck
# in one piece) stitched once onto the common body by clothes/whole_head.py. A catalogue face is that
# of another head, stitched into this one's oval by head/retarget_head.py visage: nothing else touches the head.
BASE = os.path.join(CACHE, "base")
BASE_PMX = os.path.join(BASE, "result.pmx")
GAME = games.active()
# None: no base head from the game (VRoid: it comes from the built base, tools/vroid_base, not made yet).
BASE_HEAD = GAME.value("base_head.name", None)
# Irises chosen by hand: references/eyes/<id>.png|webp replaces the painted iris of face <id> (material 目).
EYES = os.path.join(REPO, "references", "eyes")


def iris(id_):
    return _chosen(EYES, id_)


# Mouth repainted by hand: references/mouth/<id>.png, same size as the mouth area (teeth 齿 and
# inside 口舌, margin MARGIN_MOUTH px) cut out of the face texture, pasted back in its place.
MOUTH = os.path.join(REPO, "references", "mouth")
MARGIN_MOUTH = 12


def mouth(id_):
    return _chosen(MOUTH, id_)


def _chosen(folder, id_):
    for ext in (".png", ".webp", ".jpg"):
        c = os.path.join(folder, id_ + ext)
        if os.path.isfile(c):
            return c
    return None
BLENDER_SH = os.path.join(REPO, "bin", "blender-queue")   # queue of Blender launches


def places():
    """Number of heavy computations to run at once on this machine: one per physical core (half
    the threads), without exceeding 2.5 GB of the memory that is free right now each; at least 2.
    compute.slots (CHIMERA_SLOTS) forces it. Measured on 15/11 on a 24-thread processor (242 parts):
    12 at once 168 s, 16: 157 s, 24: 155 s; beyond 12 the processor reaches 100 °C and throttles.
    The memory counted is the available one, not the installed one: 12 at once on a 30 GB machine
    already busy with other programs filled the swap and the kernel killed those programs."""
    if R.value("compute.slots", None):
        return max(1, int(R.value("compute.slots")))
    go = 16.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    go = int(line.split()[1]) / 1048576
    except (OSError, ValueError, IndexError):
        pass
    return max(2, min((os.cpu_count() or 8) // 2, int(go / 2.5)))


PYTHON = R.program("python")


def launcher():
    """Prefix of heavy launches (compute.launcher, CHIMERA_LAUNCHER), as a list of arguments."""
    import shlex
    v = R.value("compute.launcher", "") or ""
    return list(v) if isinstance(v, list) else shlex.split(v)
# One meter = 12.5 MMD units (1 unit = 8 cm); 1 mm = 1/80 of a unit.
METER = 0.08
MM = 1 / 80.0

# Layers: garment types, accessory kinds, ranks, outfit rules and stacking thresholds.
# Public values in layers.default.json (repository root), overridden key by key by the "layers" section
# of settings.json: README, "Adjusting the layers". The tables keep here the shape they always had.
LAYERS = R.section("layers")
if not LAYERS.get("thresholds"):
    raise RuntimeError("settings: no layers section (is layers.default.json missing at the repository root?)")


def _entries(layers, name):
    """Entries of a layer table, without the help keys ("_…"); an entry set to null is removed."""
    return {k: v for k, v in (layers.get(name) or {}).items() if not k.startswith("_") and v is not None}


def _vocab(table):
    """The layer tables of the settings, as they are (types, tabs and slots are English words, as in the code)."""
    return dict(table)


def tables(layers):
    """(CLOTHES, OCCUPIES, ACCESSORIES, BELTS, FR_TYPE) of a "couches" section of the settings."""
    garment, accessory = _vocab(_entries(layers, "garments")), _vocab(_entries(layers, "accessories"))
    for name, table, keys in (("garments", garment, ("tab", "slot", "rank")), ("accessories", accessory, ("rank",))):
        for k, v in table.items():
            missing = [c for c in keys if not isinstance(v, dict) or c not in v]
            if missing:
                raise ValueError("settings, layers.%s.%s: missing %s" % (name, k, ", ".join(missing)))
    return (
        {t: (v["tab"], v["slot"], v["rank"]) for t, v in garment.items()},
        {t: list(v["also_takes"]) for t, v in garment.items() if v.get("also_takes")},
        {s: (v["rank"], bool(v.get("exclusive", False))) for s, v in accessory.items()},
        {s for s, v in accessory.items() if v.get("belt_like")},
        {k: v.get("name") or k for k, v in list(garment.items()) + list(accessory.items())},
    )


# CLOTHES: garment type -> (page tab, exclusive slot, layer rank)
# OCCUPIES: slots taken besides its own (dress and jumpsuit take the place of a top and a bottom)
# ACCESSORIES: accessory kind -> (layer rank, exclusive per kind?)
# BELTS: kinds that go around the body (adjusted with the garments, laid over the fabric, never squeezed)
# FR_TYPE: display name of a type or a kind
CLOTHES, OCCUPIES, ACCESSORIES, BELTS, FR_TYPE = tables(LAYERS)


def enveloping(meta):
    """Only a part that goes around the body has an outer face comparable with the layer underneath
    (a necklace or a bag lying on the chest has none)."""
    return meta.get("tab") != "accessories" or meta.get("type") in BELTS


def arrange(meta):
    """Tab, slot, rank and occupied slots of a part's record, read again from today's
    settings: the record (meta.json) keeps those of the day it was prepared, and changing a rank does
    not require re-fitting the part. A type the settings no longer know keeps those of its
    record; the display name (type_fr) stays the record's. Modifies and returns `meta`."""
    if not meta:
        return meta
    t = meta.get("type")
    if meta.get("class") != "accessory" and t in CLOTHES:
        tab, emp, rank = CLOTHES[t]
    elif t in ACCESSORIES:
        rank, excl = ACCESSORIES[t]
        tab, emp = "accessories", (t if excl else None)
    else:
        return meta
    meta.update(tab=tab, slot=emp, rank=rank, occupies=list(OCCUPIES.get(t, [])))
    return meta


def relative(path):
    """Path stored in a record: relative if it is inside the cache, so the cache stays movable."""
    r = os.path.realpath(path)
    base = os.path.realpath(CACHE) + os.sep
    return r[len(base):] if r.startswith(base) else path


def absolute(path):
    return path if os.path.isabs(path) else os.path.join(CACHE, path)


def the_key(*chunks):
    """Identifier safe for a folder name and a URL."""
    s = "__".join(str(m) for m in chunks)
    return re.sub(r"[^0-9A-Za-z_.-]", lambda c: "%" + c.group(0).encode("utf-8").hex(), s)


def write_bytes(path, bytes_):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d, n = os.path.split(path)
    fd, tmp = tempfile.mkstemp(prefix=n + ".", suffix=".tmp", dir=d)
    os.fchmod(fd, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(bytes_)
    os.replace(tmp, path)


def write_json(path, value):
    """Writes a record of the data (JSON, format 2)."""
    write_bytes(path, json.dumps(value, ensure_ascii=False, indent=1, default=_json_other).encode("utf-8"))


def _json_other(v):
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    raise TypeError(type(v))


def read_json(path, default=None):
    """Reads a record of the data (JSON, format 2); `default` if the file is missing."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def fingerprint_file(path):
    st = os.stat(path)
    return hashlib.sha1(("%s|%d|%d" % (os.path.realpath(path), st.st_size, int(st.st_mtime))).encode()).hexdigest()[:16]


def link(src, dst):
    """Light copy (hard link if possible) from src to dst, without touching src."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        os.link(src, tmp)
    except OSError:
        import shutil
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def without_os_zero(P):
    """Replaces the zero-weight bones of a read PMX with the main bone of the same vertex.

    A vertex often keeps in its empty slots the index of a bone with no influence. graft
    takes over only the bones that carry weight, then translates every slot: it would hit an unknown
    bone (KeyError). The weight stays zero, the deformation does not change. Returns the number of
    vertices corrected."""
    S, n = P["vertices"], 0
    for i, (o, p) in enumerate(zip(S["bones"], S["weights"])):
        ref = next((int(b) for b, w in zip(o, p) if b >= 0 and w > 0), None)
        if ref is None:
            continue
        new = tuple(int(b) if b < 0 or w > 0 else ref for b, w in zip(o, p))
        if new != tuple(int(b) for b in o):
            S["bones"][i] = new
            n += 1
    return n


V_HAIRSTYLE = 9   # 9 = the hair-only coverage counts the skull directions only (coverage_skull); version of hairstyle placement (on the base head, then out of the skin the page draws: skull, ears, neck of the common body)
NOTE_REFUSALS = 100   # review score (review.py pieces --note) above which a fitted part is kept off the page
V_SKIN = 29   # version of the heads (face stitched onto the base head; older heads are remade)
V_FITTING = 16         # 16 = a posed shoe grows up to 30 % to hold the foot (was 15 %), sizes over 15 % only when none smaller holds it; 15 = the per-bone fit of every part but a shoe is that of 13 again (14's two rules are a shoe's), a shoe's shaft is fitted around the leg like a sleeve instead of pushed out vertex by vertex from the ankle up; 14 = for a shoe, a bone whose tail is another bone found on both bodies is measured to that bone's head on the target too (a VRoid knee's tail stops mid-shin), a deformation duplicate without a tail (ひざD) turns like the bone it copies, not like its parent (the thigh); a shoe's foot part is always kept in one block (a flat shoe too, and the last resort carries it flat with the bones, heel sunk) and never pushed out of the foot vertex by vertex; 13 = a shoe whose foot is posed less than 8° (no heel) keeps the flat fit, a posed shoe takes the smallest size that holds the foot, each shoe moves with its own side; 12 = a shoe with a heel is worn on feet posed into it (clothes/foot.py), sized and lowered as a whole onto its sole, the model raised by its height; 11 = a deformation duplicate missing from the target (足首D on a VRoid body) is placed with the bone it copies, not at the hips; 10 = on the trunk a part keeps its anatomical height (landmarks measured on both bodies); 9 = a lid (closed fan a limb goes through) stays hidden in the body instead of being pushed out; 8 = a ring accessory also tries the block and the transport, a piece held by a single bone keeps its shape; 7 = a fabric seen edge-on (backing that closes a hem) does not bound the body; 6 = a limb is measured only on its skin (no more fins at the armpits); 5 = under the crotch and far from the body, the part finds its place from what it surrounds


def hairstyle_kept(pairs, own):
    """(kept, reason) of a hairstyle placed on the catalogue's heads. A hairstyle that fits its
    own head is kept: the page reports, face by face, those where it does not sit cleanly.
    With no original head in the catalogue, it must fit 70 % of the heads."""
    n_ok = sum(1 for r in pairs.values() if r["passed"])
    if own is not None:
        if own not in pairs:
            return False, "original head missing"
        if not pairs[own]["passed"]:
            return False, "fails on its own head : %s" % ", ".join(pairs[own]["failures"])
        return True, None
    if n_ok >= 0.7 * max(1, len(pairs)):
        return True, None
    return False, "sits cleanly on only %d heads out of %d" % (n_ok, len(pairs))


def graft(args):
    """graft.main on PMX files cleared of zero-weight bones (see without_os_zero)."""
    import head_graft as graft_head
    PMX = graft_head.PMX
    read = PMX.read

    def read_own(*a, **k):
        P = read(*a, **k)
        without_os_zero(P)
        return P
    PMX.read = read_own
    try:
        return graft_head.main(args)
    finally:
        PMX.read = read

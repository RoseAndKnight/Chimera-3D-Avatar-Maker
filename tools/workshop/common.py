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

ICI = os.path.dirname(os.path.abspath(__file__))
OUTILS = os.path.dirname(ICI)
DEPOT = os.path.dirname(OUTILS)
# tete first: tools/head/library.py and tools/clothes/library.py have the same name
for sous in ("clothes", "atlas", "head"):
    chemin = os.path.join(OUTILS, sous)
    if chemin not in sys.path:
        sys.path.insert(0, chemin)
if OUTILS not in sys.path:
    sys.path.append(OUTILS)
import settings as R  # noqa: E402
import games as jeux  # noqa: E402  what is specific to the models' game: tools/games/<game>/game.json

ATLAS = R.chemin("paths.data")
# Since 12/11 the workshop starts from the remade body (tools/rebuilt_body) carrying the remade head (tools/rebuilt_head). The previous
# workshop (ZZZ body, cache <data>/atelier) remains whole: bin/atelier-ancien-corps.
# Since 13/11: body and head in our own shape, painted skin (tools/forme_maison); the remade body keeps its
# cache (atelier-corps-tete-refaits): bin/atelier-corps-tete-refaits.
CACHE = R.chemin("workshop.cache")
BIBLIO_VET = os.path.join(ATLAS, "bibliotheque-vetements")
BIBLIO_VIS = os.path.join(ATLAS, "bibliotheque-visage")
# CHIMERA_BODY: another base body, always with its own cache (CHIMERA_CACHE)
CORPS = R.chemin("workshop.body")
# Fallback: as long as the VRoid base of the public default is not built (tools/vroid_base, a .vrm is needed), the free
# MakeHuman base (tools/free_base) is used if it is built, with its cache. A body set by hand has no fallback.
_DONNEES = R.chemin("paths.data")
if (not os.path.isfile(CORPS) and CORPS == os.path.join(_DONNEES, "base-vroid", "corps.pmx")
        and CACHE == os.path.join(_DONNEES, "atelier-base-vroid")
        and os.path.isfile(os.path.join(_DONNEES, "base-libre", "corps.pmx"))):
    CORPS, CACHE = os.path.join(_DONNEES, "base-libre", "corps.pmx"), os.path.join(_DONNEES, "atelier-base-libre")
# Base head: the whole head of the game's pivot head (tete_base in its description: face, ears and neck
# in one piece) stitched once onto the common body by clothes/whole_head.py. A catalogue face is that
# of another head, stitched into this one's oval by head/retarget_head.py visage: nothing else touches the head.
BASE = os.path.join(CACHE, "base")
BASE_PMX = os.path.join(BASE, "resultat.pmx")
JEU = jeux.actif()
# None: no base head from the game (VRoid: it comes from the built base, tools/vroid_base, not made yet).
TETE_BASE = JEU.valeur("tete_base.nom", None)
# Irises chosen by hand: references/yeux/<id>.png|webp replaces the painted iris of face <id> (material 目).
YEUX = os.path.join(DEPOT, "references", "yeux")


def iris(id_):
    return _choisi(YEUX, id_)


# Mouth repainted by hand: references/bouche/<id>.png, same size as the mouth area (teeth 齿 and
# inside 口舌, margin MARGE_BOUCHE px) cut out of the face texture, pasted back in its place.
BOUCHE = os.path.join(DEPOT, "references", "bouche")
MARGE_BOUCHE = 12


def bouche(id_):
    return _choisi(BOUCHE, id_)


def _choisi(dossier, id_):
    for ext in (".png", ".webp", ".jpg"):
        c = os.path.join(dossier, id_ + ext)
        if os.path.isfile(c):
            return c
    return None
BLENDER_SH = os.path.join(DEPOT, "bin", "blender-queue")   # queue of Blender launches


def places():
    """Number of heavy computations to run at once on this machine: one per physical core (half
    the threads), without exceeding 2.5 GB of the memory that is free right now each; at least 2.
    compute.slots (CHIMERA_SLOTS) forces it. Measured on 15/11 on a 24-thread processor (242 parts):
    12 at once 168 s, 16: 157 s, 24: 155 s; beyond 12 the processor reaches 100 °C and throttles.
    The memory counted is the available one, not the installed one: 12 at once on a 30 GB machine
    already busy with other programs filled the swap and the kernel killed those programs."""
    if R.valeur("compute.slots", None):
        return max(1, int(R.valeur("compute.slots")))
    go = 16.0
    try:
        with open("/proc/meminfo") as f:
            for ligne in f:
                if ligne.startswith("MemAvailable:"):
                    go = int(ligne.split()[1]) / 1048576
    except (OSError, ValueError, IndexError):
        pass
    return max(2, min((os.cpu_count() or 8) // 2, int(go / 2.5)))


PYTHON = R.programme("python")


def lanceur():
    """Prefix of heavy launches (compute.launcher, CHIMERA_LAUNCHER), as a list of arguments."""
    import shlex
    v = R.valeur("compute.launcher", "") or ""
    return list(v) if isinstance(v, list) else shlex.split(v)
# One metre = 12.5 MMD units (1 unit = 8 cm); 1 mm = 1/80 of a unit.
METRE = 0.08
MM = 1 / 80.0

# Layers: garment types, accessory kinds, ranks, outfit rules and stacking thresholds.
# Public values in layers.default.json (repository root), overridden key by key by the "layers" section
# of settings.json: README, "Adjusting the layers". The tables keep here the shape they always had.
COUCHES = R.section("layers")
if not COUCHES.get("thresholds"):
    raise RuntimeError("settings: no layers section (is layers.default.json missing at the repository root?)")


def _entrees(couches, nom):
    """Entries of a layer table, without the help keys ("_…"); an entry set to null is removed."""
    return {k: v for k, v in (couches.get(nom) or {}).items() if not k.startswith("_") and v is not None}


def tables(couches):
    """(VETEMENTS, OCCUPE, ACCESSOIRES, CEINTURES, FR_TYPE) of a "couches" section of the settings."""
    vet, acc = _entrees(couches, "garments"), _entrees(couches, "accessories")
    for nom, table, cles in (("garments", vet, ("tab", "slot", "rank")), ("accessories", acc, ("rank",))):
        for k, v in table.items():
            manque = [c for c in cles if not isinstance(v, dict) or c not in v]
            if manque:
                raise ValueError("settings, layers.%s.%s: missing %s" % (nom, k, ", ".join(manque)))
    return (
        {t: (v["tab"], v["slot"], v["rank"]) for t, v in vet.items()},
        {t: list(v["also_takes"]) for t, v in vet.items() if v.get("also_takes")},
        {s: (v["rank"], bool(v.get("exclusive", False))) for s, v in acc.items()},
        {s for s, v in acc.items() if v.get("belt_like")},
        {k: v.get("name") or k for k, v in list(vet.items()) + list(acc.items())},
    )


# VETEMENTS: garment type -> (page tab, exclusive slot, layer rank)
# OCCUPE: slots taken besides its own (dress and jumpsuit take the place of a top and a bottom)
# ACCESSOIRES: accessory kind -> (layer rank, exclusive per kind?)
# CEINTURES: kinds that go around the body (adjusted with the garments, laid over the fabric, never squeezed)
# FR_TYPE: display name of a type or a kind
VETEMENTS, OCCUPE, ACCESSOIRES, CEINTURES, FR_TYPE = tables(COUCHES)


def enveloppant(meta):
    """Only a part that goes around the body has an outer face comparable with the layer underneath
    (a necklace or a bag lying on the chest has none)."""
    return meta.get("onglet") != "accessoires" or meta.get("type") in CEINTURES


def ranger(meta):
    """Tab, slot, rank and occupied slots of a part's record, read again from today's
    settings: the record (meta.json) keeps those of the day it was prepared, and changing a rank does
    not require re-fitting the part. A type the settings no longer know keeps those of its
    record; the display name (type_fr) stays the record's. Modifies and returns `meta`."""
    if not meta:
        return meta
    t = meta.get("type")
    if meta.get("classe") != "accessoire" and t in VETEMENTS:
        onglet, emp, rang = VETEMENTS[t]
    elif t in ACCESSOIRES:
        rang, excl = ACCESSOIRES[t]
        onglet, emp = "accessoires", (t if excl else None)
    else:
        return meta
    meta.update(onglet=onglet, emplacement=emp, rang=rang, occupe=list(OCCUPE.get(t, [])))
    return meta


def relatif(chemin):
    """Path stored in a record: relative if it is inside the cache, so the cache stays movable."""
    r = os.path.realpath(chemin)
    base = os.path.realpath(CACHE) + os.sep
    return r[len(base):] if r.startswith(base) else chemin


def absolu(chemin):
    return chemin if os.path.isabs(chemin) else os.path.join(CACHE, chemin)


def cle(*morceaux):
    """Identifier safe for a folder name and a URL."""
    s = "__".join(str(m) for m in morceaux)
    return re.sub(r"[^0-9A-Za-z_.-]", lambda c: "%" + c.group(0).encode("utf-8").hex(), s)


def ecrire_octets(chemin, octets):
    os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True)
    d, n = os.path.split(chemin)
    fd, tmp = tempfile.mkstemp(prefix=n + ".", suffix=".tmp", dir=d)
    os.fchmod(fd, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(octets)
    os.replace(tmp, chemin)


def ecrire_json(chemin, valeur):
    ecrire_octets(chemin, json.dumps(valeur, ensure_ascii=False, indent=1, default=_json_autre).encode("utf-8"))


def _json_autre(v):
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


def lire_json(chemin, defaut=None):
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return defaut


def empreinte_fichier(chemin):
    st = os.stat(chemin)
    return hashlib.sha1(("%s|%d|%d" % (os.path.realpath(chemin), st.st_size, int(st.st_mtime))).encode()).hexdigest()[:16]


def lier(src, dst):
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


def sans_os_nuls(P):
    """Replaces the zero-weight bones of a read PMX with the main bone of the same vertex.

    A vertex often keeps in its empty slots the index of a bone with no influence. greffe_tete
    takes over only the bones that carry weight, then translates every slot: it would hit an unknown
    bone (KeyError). The weight stays zero, the deformation does not change. Returns the number of
    vertices corrected."""
    S, n = P["sommets"], 0
    for i, (o, p) in enumerate(zip(S["os"], S["poids"])):
        ref = next((int(b) for b, w in zip(o, p) if b >= 0 and w > 0), None)
        if ref is None:
            continue
        nouv = tuple(int(b) if b < 0 or w > 0 else ref for b, w in zip(o, p))
        if nouv != tuple(int(b) for b in o):
            S["os"][i] = nouv
            n += 1
    return n


V_COIFFURE = 9   # 9 = the hair-only coverage counts the skull directions only (couverture_crane); version of hairstyle placement (on the base head, then out of the skin the page draws: skull, ears, neck of the common body)
NOTE_REFUS = 100   # review score (review.py pieces --noter) above which a fitted part is kept off the page
V_PEAU = 29   # version of the heads (face stitched onto the base head; older heads are remade)
V_PORTAGE = 16         # 16 = a posed shoe grows up to 30 % to hold the foot (was 15 %), sizes over 15 % only when none smaller holds it; 15 = the per-bone fit of every part but a shoe is that of 13 again (14's two rules are a shoe's), a shoe's shaft is fitted around the leg like a sleeve instead of pushed out vertex by vertex from the ankle up; 14 = for a shoe, a bone whose tail is another bone found on both bodies is measured to that bone's head on the target too (a VRoid knee's tail stops mid-shin), a deformation duplicate without a tail (ひざD) turns like the bone it copies, not like its parent (the thigh); a shoe's foot part is always kept in one block (a flat shoe too, and the last resort carries it flat with the bones, heel sunk) and never pushed out of the foot vertex by vertex; 13 = a shoe whose foot is posed less than 8° (no heel) keeps the flat fit, a posed shoe takes the smallest size that holds the foot, each shoe moves with its own side; 12 = a shoe with a heel is worn on feet posed into it (clothes/foot.py), sized and lowered as a whole onto its sole, the model raised by its height; 11 = a deformation duplicate missing from the target (足首D on a VRoid body) is placed with the bone it copies, not at the hips; 10 = on the trunk a part keeps its anatomical height (landmarks measured on both bodies); 9 = a lid (closed fan a limb goes through) stays hidden in the body instead of being pushed out; 8 = a ring accessory also tries the block and the transport, a piece held by a single bone keeps its shape; 7 = a fabric seen edge-on (backing that closes a hem) does not bound the body; 6 = a limb is measured only on its skin (no more fins at the armpits); 5 = under the crotch and far from the body, the part finds its place from what it surrounds


def coiffure_retenue(paires, propre):
    """(kept, reason) of a hairstyle placed on the catalogue's heads. A hairstyle that fits its
    own head is kept: the page reports, face by face, those where it does not sit cleanly.
    With no original head in the catalogue, it must fit 70 % of the heads."""
    n_ok = sum(1 for r in paires.values() if r["passe"])
    if propre is not None:
        if propre not in paires:
            return False, "original head missing"
        if not paires[propre]["passe"]:
            return False, "fails on its own head : %s" % ", ".join(paires[propre]["echecs"])
        return True, None
    if n_ok >= 0.7 * max(1, len(paires)):
        return True, None
    return False, "sits cleanly on only %d heads out of %d" % (n_ok, len(paires))


def greffer(args):
    """greffe_tete.main on PMX files cleared of zero-weight bones (see sans_os_nuls)."""
    import head_graft as greffe_tete
    PMX = greffe_tete.PMX
    lire = PMX.lire

    def lire_propre(*a, **k):
        P = lire(*a, **k)
        sans_os_nuls(P)
        return P
    PMX.lire = lire_propre
    try:
        return greffe_tete.main(args)
    finally:
        PMX.lire = lire

"""Converts a data folder from format 1 (French names) to format 2 (English names), in place.

    python3 -I tools/library/library.py migrate <data folder> [--settings <settings.json>]
    python3 -I tools/library/library.py migrate <data folder> --check

What it does, in this order (each step is written in <data>/.chimera-migrate.json once done, so a
conversion cut short starts again where it stopped, and a second run changes nothing):

1. fingerprints: notes the fingerprint (path, size, date) of every file a record points to;
2. names: renames folders and files (tools/dataformat.py: DIRS, FILES, garment types), deepest first;
3. records: rewrites every JSON file (keys, paths, garment types, tabs, slots, codes of checks);
4. models: rewrites the texture table of the .pmx files that name a renamed file (nothing else moves);
5. fingerprints again: the fingerprints of the moved files, put in place of the old ones;
6. links: the symbolic links that pointed inside the data folder point to the new names;
7. settings: the settings.json of this installation (next to the data folder, or --settings);
8. revision 1: the names inside files (arrays of the .npz files, the house skeleton: skeleton.json, weights.npz,
   poses.npz, shape.pkl; the check code waist_outline). A folder converted before this revision gets only it;
9. revision 2: the names format 2 forgot (tools/dataformat.py REVISED_*): the free base's id Libre -> Free (folders,
   <head>.bin files, keys, values, the head lists of the hairstyle signatures), fabriquees/ -> built/, the owner's
   outfit and trial folders (tenue -> outfit, essais -> trials, sur-<head> -> on-<head>...), the built part
   debardeur -> tank_top, the shape morphs forme_ -> shape_ of the rebuilt bodies (patched in place, same size);
   the fingerprints of the moved files are replaced in the records, those of this folder's files and those of
   another folder's files named by a path (a cache kept outside its data folder), so nothing is computed again;
10. marks the folder: chimera-data.json {"format": 2, "revision": 2}.

Every file rewritten is a new file put in place (os.replace): a file with several names (hard links,
cp -al backups) keeps its other names untouched. Dates of files are kept. --check lists what still has
a format-1 name, key or word (proper names and the French texts of labels.json are not names).
"""
import hashlib
import itertools
import json
import shutil
import os
import re
import struct
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import dataformat as F  # noqa: E402

LOG = ".chimera-migrate.json"
PHASES = ("fingerprints", "outside-links", "names", "records", "models", "fingerprints-replace", "links", "settings",
          "revision-1", "revision-2", "marker")
HEX16 = re.compile(r"^[0-9a-f]{16}$")
FIN16 = re.compile(r"^v\d+:.*:[0-9a-f]{16}$", re.S)     # a fingerprint at the end of a signature
# the library's roots (manifest library.json: "units")
ROOTS = {"visage": "face", "vetements": "clothes", "mmd_extrait": "mmd_extracted"}
# keys whose values are garment types, part names or other vocabulary
VOCAB = ("type", "piece", "porteur", "recouvre", "accessoire_reste", "couches_internes_admises", "deux_vetements",
         "garnitures", "sorte", "id", "classe", "pieces", "dossier_type")


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- files
def fingerprint(path):
    """Same as tools/workshop/common.py empreinte_fichier."""
    st = os.stat(path)
    return hashlib.sha1(("%s|%d|%d" % (os.path.realpath(path), st.st_size, int(st.st_mtime))).encode()).hexdigest()[:16]


def replace(path, bytes_):
    """New file put in place of the old one, same date and rights: the old one's other names stay as they are."""
    st = os.stat(path)
    d, n = os.path.split(path)
    fd, tmp = tempfile.mkstemp(prefix=n + ".", suffix=".tmp", dir=d)
    with os.fdopen(fd, "wb") as f:
        f.write(bytes_)
    os.chmod(tmp, st.st_mode & 0o7777)
    os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(tmp, path)


def walk(root):
    """(folder, sub-folders, files) without following links, sources included."""
    for d, ds, fs in os.walk(root):
        ds[:] = [x for x in ds if not os.path.islink(os.path.join(d, x))]
        yield d, ds, fs


def under_folder(path, folder):
    """The end of chemin ("/a/b") below dossier, or None; the same folder seen through two mounts counts."""
    head, rest = os.path.normpath(path), ""
    while head and head != os.path.dirname(head):
        if os.path.isdir(head) and os.path.samefile(head, folder):
            return rest or None
        head, end = os.path.split(head)
        rest = "/" + end + rest
    return None


def link_outside(link, absolute):
    """New target of a broken link that points into another data folder already in format 2, or None."""
    if os.path.exists(link):
        return None
    head = absolute
    while head != os.path.dirname(head):
        head = os.path.dirname(head)
        if os.path.isfile(os.path.join(head, F.MARKER)):
            nv = os.path.join(head, F.path(os.path.relpath(absolute, head)))
            return nv if os.path.exists(nv) else None
    return None


def in_sources(rel):
    return rel == "sources" or rel.startswith("sources" + os.sep)


def new_name(rel, name, folder):
    """Format-2 name of the entry `nom` in the folder `rel` (relative to the data folder)."""
    if in_sources(rel) or (rel == "." and name == "sources"):
        return F.DIRS.get(name, name) if name == "mmd-extrait" else name
    if rel == "." and name in (LOG, F.MARKER):
        return name
    return F.dir_name(name) if folder else F.file_name(name)


# ---------------------------------------------------------------- records
def value_vocab(k, v):
    if isinstance(v, list):
        return [value_vocab(k, x) if isinstance(x, (str, list)) else conv(x) for x in v]
    if isinstance(v, dict):
        return conv(v, k)
    if not isinstance(v, str):
        return v
    if k == "classe":
        return F.TYPES.get(v, v)
    if "/" in v or "\\" in v:
        return chain(v)
    if "__" in v:
        return F.part_id(v)
    if k in ("type", "piece", "id", "pieces") and v.endswith((".pmx", ".json")):
        return F.file_name(v)
    return F.type_name(v)


def chain(v):
    if not isinstance(v, str) or v.startswith(("http://", "https://")) or len(v) > 400:
        return v
    if ("/" in v or "\\" in v) and (not re.search(r"\s", v) or v.startswith("/")):
        if re.match(r"^[a-z_]+/", v) and v.split("/", 1)[0] in ROOTS:      # library units: visage/X
            r, rest = v.split("/", 1)
            return ROOTS[r] + "/" + F.path(rest)
        return F.path(v)
    if " " not in v and re.search(r"\.(pmx|json|png|glb|bin|npz|vrm|txt)$", v):
        return F.file_name(v)
    return v


def conv(x, parent=None):
    """A format-1 record -> format 2 (keys, words, paths, garment types, codes of checks)."""
    if isinstance(x, dict):
        out = {}
        verif = "ok" in x or "seuil" in x
        for k, v in x.items():
            if parent in F.TYPE_KEYED:
                out[F.part_id(k) if "__" in k else F.type_name(k)] = conv(v, None)
                continue
            nk = F._key(k, F._KEY_TO, F.LIMBS)
            k = k if k in F.KEYS else F.KEYS_BACK.get(k, k)     # the French name says what the value holds
            if "/" in nk:                       # labels.json: "tetes/<id>"
                nk = F.path(nk)
            if k in F.OPAQUE:
                out[nk] = v
            elif k == "nom" and verif and isinstance(v, str):
                out[nk] = F.CHECKS.get(v, v)
            elif k in F.CHECK_LISTS:
                out[nk] = ([F.CHECKS.get(c, c) if isinstance(c, str) else conv(c) for c in v] if isinstance(v, list)
                           else F.CHECKS.get(v, v) if isinstance(v, str) else conv(v, k))
            elif k in F.WORD_VALUES or k == "genre":
                out[nk] = ([F.WORDS.get(c, c) if isinstance(c, str) else c for c in v] if isinstance(v, list)
                           else F.WORDS.get(v, v) if isinstance(v, str) else conv(v, k))
            elif k == "onglet" and isinstance(v, str):
                out[nk] = F.TABS.get(v, v)
            elif k in ("emplacement", "occupe"):
                out[nk] = ([F.SLOT_WORDS.get(c, c) for c in v] if isinstance(v, list)
                           else F.SLOT_WORDS.get(v, v) if isinstance(v, str) else v)
            elif k in VOCAB:
                out[nk] = value_vocab(k, v)
            elif k in ("raison", "message") and isinstance(v, str):
                out[nk] = F.codes_in_text(v)
            else:
                out[nk] = conv(v, k)
        return out
    if isinstance(x, list):
        return [conv(v, None if parent in F.TYPE_KEYED else parent) for v in x]
    if isinstance(x, str):
        return chain(x)
    return x


def write_json_like(path, value, text):
    """Same layout as the file had (indent 1 or compact)."""
    indent = 1 if "\n" in text[:200] else None
    return json.dumps(value, ensure_ascii=False, indent=indent).encode("utf-8")


# ---------------------------------------------------------------- .pmx texture tables
def _text(d, p, enc):
    n = struct.unpack_from("<i", d, p)[0]
    return d[p + 4:p + 4 + n].decode("utf-16-le" if enc == 0 else "utf-8"), p + 4 + n


def pmx_textures(d):
    """(offset of the texture table, its end, encoding, the texture paths)."""
    if d[:4] != b"PMX ":
        raise ValueError("not a PMX file")
    n_glob = d[8]
    g = d[9:9 + n_glob]
    enc, n_uv, t_vert, t_tex, t_mat, t_os = g[0], g[1], g[2], g[3], g[4], g[5]
    p = 9 + n_glob
    for _ in range(4):
        _, p = _text(d, p, enc)
    n = struct.unpack_from("<i", d, p)[0]
    p += 4
    weight = {0: t_os, 1: 2 * t_os + 4, 2: 4 * t_os + 16, 3: 2 * t_os + 4 + 36, 4: 4 * t_os + 16}
    base = 32 + 16 * n_uv
    for _ in range(n):
        typ = d[p + base]
        p += base + 1 + weight[typ] + 4
    n = struct.unpack_from("<i", d, p)[0]
    p += 4 + n * t_vert
    start = p
    n = struct.unpack_from("<i", d, p)[0]
    p += 4
    tex = []
    for _ in range(n):
        t, p = _text(d, p, enc)
        tex.append(t)
    return start, p, enc, tex


def pmx_fix(path):
    """Rewrites the texture paths of a .pmx with the format-2 names; True if it changed."""
    with open(path, "rb") as f:
        d = f.read()
    start, end, enc, tex = pmx_textures(d)
    new = [F.path(t) for t in tex]
    if new == tex:
        return False
    code = "utf-16-le" if enc == 0 else "utf-8"
    table = struct.pack("<i", len(new))
    for t in new:
        b = t.encode(code)
        table += struct.pack("<i", len(b)) + b
    replace(path, d[:start] + table + d[end:])
    return True


# ---------------------------------------------------------------- the conversion
# ---------------------------------------------------------------- revision 1: names inside files
WAIST = re.compile(r"\b(%s)\b" % "|".join(F.REVISED_CHECKS))


def npz_members(name, members):
    """{old member: new member} of an .npz file of this name (members without the .npy ending)."""
    table, pre = F.NPZ.get(name, {}), F.NPZ_PREFIXES.get(name, {})
    out = {}
    for m in members:
        n = table.get(m, m)
        for a, b in pre.items():
            if n.startswith(a):
                n = b + n[len(a):]
        if n != m:
            out[m] = n
    return out


def npz_fix(path):
    """Renames the arrays of an .npz file (bytes copied as they are); the bone names of a house weights.npz too."""
    name = os.path.basename(path)
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        ren = npz_members(name, [i.filename[:-4] for i in infos if i.filename.endswith(".npy")])
        house = name == "weights.npz" and os.path.isfile(os.path.join(os.path.dirname(path), "skeleton.json"))
        if not ren and not house:
            return False
        import io
        buf = io.BytesIO()
        changed = bool(ren)
        with zipfile.ZipFile(buf, "w") as out:
            for i in infos:
                data = z.read(i)
                m = i.filename[:-4]
                if house and ren.get(m, m) == "bones":
                    import numpy as np
                    a = np.load(io.BytesIO(data))
                    b = np.array([F.house_bone(str(x)) for x in a])
                    if list(b) != list(a):
                        changed = True
                        o = io.BytesIO()
                        np.save(o, b)
                        data = o.getvalue()
                j = zipfile.ZipInfo(ren.get(m, m) + ".npy", i.date_time)
                j.compress_type, j.external_attr = i.compress_type, i.external_attr
                out.writestr(j, data)
    if changed:
        replace(path, buf.getvalue())
    return changed


def skeleton_fix(x):
    """A house skeleton.json in revision-1 names (keys, bone names, sorts of twist; the texts stay)."""
    if not isinstance(x, dict) or "bones" not in x and "os" not in x:
        return x

    def bone(o):
        o = {F.SKELETON_KEYS.get(k, F.KEYS.get(k, k) if k in ("nom", "os") else k): v for k, v in o.items()}
        o["name"] = F.house_bone(o.get("name"))
        o["parent"] = F.house_bone(o.get("parent"))
        t = o.get("twist")
        if isinstance(t, dict):
            t = {F.SKELETON_KEYS.get(k, k): v for k, v in t.items()}
            t["sort"] = F.TWIST_SORTS.get(t.get("sort"), t.get("sort"))
            t["source"] = F.house_bone(t.get("source"))
            o["twist"] = t
        return o
    out = {}
    for k, v in x.items():
        nk = F.SKELETON_KEYS.get(k, "bones" if k == "os" else k)
        if k in F.SKELETON_TEXTS:
            v = F.SKELETON_TEXTS[k]
        out[nk] = [bone(o) for o in v] if nk == "bones" else v
    return out


def shape_fix(g):
    """A shape.pkl (grid of a rebuilt body) in revision-1 names."""
    if not isinstance(g, dict) or "quads" not in g:
        return g
    out = {F.SHAPE_KEYS.get(k, k): v for k, v in g.items()}
    if isinstance(out.get("head"), dict):
        out["head"] = {F.SHAPE_HEAD_KEYS.get(k, k): v for k, v in out["head"].items()}
    if "maps" in out:
        out["maps"] = [c if c is None else (F.map_name(c[0]),) + tuple(c[1:]) for c in out["maps"]]
    z = out.get("zone")
    if z is not None:
        import numpy as np
        out["zone"] = np.array([F.SHAPE_ZONES.get(str(v), str(v)) for v in z])

    def info(v):
        if isinstance(v, dict):
            return {F.SHAPE_INFO.get(k, k): info(w) for k, w in v.items()}
        return v
    if "info" in out:
        out["info"] = info(out["info"])
    return out


def _same(a, b):
    """Equality of two values holding numpy arrays."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if hasattr(a, "shape"):
        return a.shape == b.shape and a.dtype == b.dtype and bool((a == b).all())
    return a == b


def revision_left(path):
    """What revision 1 would still change in this file: None, or a short reason."""
    name = os.path.basename(path)
    if name.endswith(".npz"):
        try:
            with zipfile.ZipFile(path) as z:
                members = [i[:-4] for i in z.namelist() if i.endswith(".npy")]
        except (zipfile.BadZipFile, OSError):
            return None
        ren = npz_members(name, members)
        if ren:
            return "arrays " + ", ".join(sorted(ren))
        if name == "weights.npz" and "bones" in members and os.path.isfile(os.path.join(os.path.dirname(path), "skeleton.json")):
            import numpy as np
            b = [str(x) for x in np.load(path)["bones"]]
            if [F.house_bone(x) for x in b] != b:
                return "bone names"
    elif name == "skeleton.json":
        with open(path, encoding="utf-8") as f:
            x = json.load(f)
        if skeleton_fix(x) != x:
            return "house skeleton"
    elif name.endswith(".pkl") and name.startswith("shape"):
        import pickle
        with open(path, "rb") as f:
            g = pickle.load(f)
        if not _same(shape_fix(g), g):
            return "grid"
    elif name.endswith(".json"):
        with open(path, "rb") as f:
            if WAIST.search(f.read().decode("utf-8", "replace")):
                return "check code waist_outline"
    return None


# ---------------------------------------------------------------- revision 2: names format 2 forgot
def fingerprint_at(path, real):
    """The fingerprint `path` had (tools/workshop/common.py fingerprint_file), for the file now at `real`: the path
    may no longer exist (its folder was renamed), the size and date are the same."""
    st = os.stat(real)
    return hashlib.sha1(("%s|%d|%d" % (os.path.realpath(path), st.st_size, int(st.st_mtime))).encode()).hexdigest()[:16]


def views(folder):
    """Every path of this folder through the mounts of its file system (a bind mount gives it a second path)."""
    real = os.path.realpath(folder)
    out = [real]
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as f:
            mounts = [ln.split() for ln in f]
    except OSError:
        return out
    mounts = [(m[2], m[3], m[4]) for m in mounts if len(m) > 4]
    own = max((m for m in mounts if real == m[2] or real.startswith(m[2].rstrip("/") + "/")), key=lambda m: len(m[2]), default=None)
    if own is None:
        return out
    inside = (own[1].rstrip("/") + real[len(own[2].rstrip("/")):]) or "/"
    for dev, root, point in mounts:
        if dev == own[0] and (inside == root or inside.startswith(root.rstrip("/") + "/")):
            v = "/" + os.path.normpath(point.rstrip("/") + "/" + inside[len(root.rstrip("/")):]).lstrip("/")
            if v not in out and os.path.isdir(v) and os.path.samefile(v, real):
                out.append(v)
    return out


def cache_roots(D):
    """Workshop caches inside a data folder (or the folder itself): folders holding catalog.json or heads/ and parts/."""
    out = []
    for d, ds, fs in walk(D):
        if in_sources(os.path.relpath(d, D)):
            ds[:] = []
            continue
        if "catalog.json" in fs or ("heads" in ds and ("parts" in ds or "hairstyles" in ds)):
            out.append(d)
    return out


def in_cache(path, caches):
    return any(path == c or path.startswith(c + os.sep) for c in caches)


def revised(x, cache=False, built=False, fingerprints=None):
    """A record with the revision-2 names (keys, ids, signatures, paths) and the fingerprints of moved files."""
    fp = fingerprints or {}
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            nk = F.revised_word(k, cache) if isinstance(k, str) else k
            if built and k in F.REVISED_BUILT_KEYS:
                nk = F.REVISED_BUILT_KEYS[k]
            out[nk] = revised(v, cache, built, fp)
        return out
    if isinstance(x, list):
        return [revised(v, cache, built, fp) for v in x]
    if isinstance(x, str):
        if HEX16.match(x):
            return fp.get(x, x)
        v = F.revised_word(x, cache)
        if FIN16.search(v) and v[-16:] in fp:
            v = v[:-16] + fp[v[-16:]]
        return v
    return x


def morph_prefix_fix(path):
    """Shape morphs of a rebuilt body: forme_<name> -> shape_<name>, bytes patched in place (same length, so the
    size and the fingerprint stay); only when every occurrence of the prefix is in a morph name. 1 if done."""
    old, new = F.MORPH_PREFIX
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] != b"PMX ":
        return 0
    code = "utf-16-le" if d[9] == 0 else "utf-8"
    a, b = old.encode(code), new.encode(code)
    if a not in d:
        return 0
    sys.path.insert(0, os.path.join(TOOLS, "atlas"))
    import pmx as PMX
    names = [t for m in PMX.read(path)["morphs"] for t in (m["name"], m["name_en"])]
    if d.count(a) != sum(t.count(old) for t in names):
        say("  %s: the prefix %s is also outside the morph names, left as it is" % (path, old))
        return 0
    replace(path, d.replace(a, b))
    after = [t for m in PMX.read(path)["morphs"] for t in (m["name"], m["name_en"])]
    if after != [t.replace(old, new) for t in names]:
        raise SystemExit("%s: morph names not as expected after the patch" % path)
    return 1


def revision_2_left(path, rel, cache):
    """What revision 2 would still change in this file: None, or a short reason."""
    name = os.path.basename(path)
    if name.endswith(".json"):
        try:
            with open(path, encoding="utf-8") as f:
                x = json.load(f)
        except ValueError:
            return None
        built = cache and os.sep + "built" + os.sep in path and name == "record.json"
        if revised(x, cache, built) != x:
            return "revision-2 word"
    elif name.endswith(".pmx"):
        with open(path, "rb") as f:
            d = f.read()
        if d[:4] == b"PMX " and F.MORPH_PREFIX[0].encode("utf-16-le" if d[9] == 0 else "utf-8") in d:
            return "morph prefix " + F.MORPH_PREFIX[0]
    return None


class Migration:
    def __init__(self, data, settings=None):
        self.D = os.path.realpath(data)
        self.settings = settings
        self.log_path = os.path.join(self.D, LOG)
        try:
            with open(self.log_path, encoding="utf-8") as f:
                self.j = json.load(f)
        except FileNotFoundError:
            self.j = {"done": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def note(self, phase, **info):
        self.j["done"].append(phase)
        self.j.setdefault("report", {})[phase] = info
        tmp = self.log_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.j, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.log_path)

    def json_files(self):
        for d, _, fs in walk(self.D):
            for n in fs:
                if n.endswith(".json") and n not in (LOG, F.MARKER) and not n.endswith(".tmp"):
                    yield os.path.join(d, n)

    def links_outside(self):
        """Data folders that are links to a folder outside the data (base-vroid -> another installation): the
        conversion must not write there, so it gives this installation its own copy (hard links) first."""
        out = []
        for name in sorted(os.listdir(self.D)):
            p = os.path.join(self.D, name)
            if name in F.DIRS and os.path.islink(p) and os.path.isdir(p):
                target = os.path.realpath(p)
                if not (target + os.sep).startswith(self.D + os.sep):
                    out.append((name, target))
        return out

    def links_a_redo(self):
        """Links into another data folder converted after this one (its old names): the links phase redoes them."""
        for d, ds, fs in walk(self.D):
            for name in ds + fs:
                p = os.path.join(d, name)
                if os.path.islink(p):
                    target = os.readlink(p)
                    if link_outside(p, target if os.path.isabs(target) else os.path.normpath(os.path.join(d, target))):
                        return True
        return False

    def run(self):
        if self.links_outside() and "outside-links" in self.j["done"]:
            self.j["done"].remove("outside-links")
        if "links" in self.j["done"] and self.links_a_redo():
            self.j["done"].remove("links")
        if F.format_of(self.D) == 2 and set(PHASES) <= set(self.j["done"]) and not self.links_outside():
            info = self.phase_settings()        # settings written before the last renaming still follow
            say("%s is already in format 2; settings: %s" % (self.D, info["settings"]))
            return 0
        if F.format_of(self.D) is None and not self.j["done"]:
            say("%s holds no Chimera data of format 1: nothing to do." % self.D)
            return 1
        t0 = time.time()
        for phase in PHASES:
            if phase in self.j["done"]:
                say("· %s: done before" % phase)
                continue
            info = getattr(self, "phase_" + phase.replace("-", "_"))()
            self.note(phase, **info)
            say("· %s: %s" % (phase, ", ".join("%s %s" % (k, v) for k, v in info.items() if k != "map")))
        say("%s converted to format 2 in %.0f s. Check: %s %s --check" % (self.D, time.time() - t0, F.MIGRATE, self.D))
        return 0

    # 1
    def phase_fingerprints(self):
        cited = set()
        for p in self.json_files():
            with open(p, "rb") as f:
                cited.update(m.decode() for m in re.findall(rb'[":]([0-9a-f]{16})"', f.read()))
        before = {}
        outside = [os.walk(os.path.join(self.D, name)) for name, _ in self.links_outside()]
        for d, _, fs in itertools.chain(walk(self.D), *outside):
            for n in fs:
                p = os.path.join(d, n)
                if os.path.islink(p):
                    continue
                e = fingerprint(p)
                if e in cited:
                    before[os.path.relpath(p, self.D)] = e
        self.j["fingerprints"] = before
        return {"files": len(before), "fingerprints in records": len(cited)}

    # 1b
    def phase_outside_links(self):
        links = self.links_outside()
        if not links:
            return {"materialised": 0}
        if any(x != "fingerprints" for x in self.j["done"]):
            # a folder converted earlier without its linked part: fingerprints again (through the link), then every
            # later phase again (each is harmless on what is already format 2)
            self.j["done"] = []
            info = self.phase_fingerprints()
            self.note("fingerprints", **info)
        rank = os.path.join(self.D, ".chimera-migrate-links")
        os.makedirs(rank, exist_ok=True)
        n = 0
        for name, target in links:
            tmp = os.path.join(self.D, name + ".copying")
            for d, ds, fs in os.walk(target):
                q = os.path.join(tmp, os.path.relpath(d, target))
                os.makedirs(q, exist_ok=True)
                for x in fs:
                    src, dst = os.path.join(d, x), os.path.join(q, x)
                    if os.path.lexists(dst):
                        continue
                    if os.path.islink(src):
                        os.symlink(os.readlink(src), dst)
                        continue
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                    n += 1
            os.rename(os.path.join(self.D, name), os.path.join(rank, name))   # the link is kept, set aside
            os.rename(tmp, os.path.join(self.D, name))
        return {"materialised": len(links), "files linked": n, "targets (untouched)": [c for _, c in links]}

    # 2
    def phase_names(self):
        n = 0
        for d, ds, fs in os.walk(self.D, topdown=False):
            rel = os.path.relpath(d, self.D)
            for name, folder in [(x, False) for x in fs] + [(x, True) for x in ds]:
                p = os.path.join(d, name)
                folder = folder and not os.path.islink(p)
                nv = new_name(rel, name, folder)
                if nv == name:
                    continue
                q = os.path.join(d, nv)
                if os.path.lexists(q):
                    raise SystemExit("cannot rename %s: %s already exists" % (p, q))
                os.rename(p, q)
                n += 1
        return {"renamed": n}

    # 3
    def phase_records(self):
        n = 0
        for p in self.json_files():
            with open(p, encoding="utf-8") as f:
                text = f.read()
            try:
                x = json.loads(text)
            except ValueError:
                continue
            nx = conv(x)
            if os.path.basename(p) == "library.json" and isinstance(nx, dict) and nx.get("format") == 1:
                nx["format"] = F.FORMAT
            if nx != x:
                replace(p, write_json_like(p, nx, text))
                n += 1
        return {"rewritten": n}

    # 4
    def phase_models(self):
        words = set(F.DIRS) | set(F.FILES) | set(F.TYPES) | {"peau", "reflet-"}
        needles = [m.encode("utf-16-le") for m in words] + [m.encode("utf-8") for m in words]
        n = seen = 0
        for d, _, fs in walk(self.D):
            if in_sources(os.path.relpath(d, self.D)):
                continue
            for name in fs:
                if not name.endswith(".pmx"):
                    continue
                p = os.path.join(d, name)
                if os.path.islink(p):
                    continue
                seen += 1
                with open(p, "rb") as f:
                    head = f.read()
                if not any(a in head for a in needles):
                    continue
                if pmx_fix(p):
                    n += 1
        return {"models": seen, "rewritten": n}

    # 5
    def phase_fingerprints_replace(self):
        map_ = {}
        for rel, e in self.j.get("fingerprints", {}).items():
            q = os.path.join(self.D, F.path(rel))
            if os.path.isfile(q):
                ne = fingerprint(q)
                if ne != e:
                    map_[e] = ne
        if not map_:
            return {"replaced": 0}

        def rempl(x):
            if isinstance(x, dict):
                return {k: rempl(v) for k, v in x.items()}
            if isinstance(x, list):
                return [rempl(v) for v in x]
            if isinstance(x, str) and HEX16.match(x):
                return map_.get(x, x)
            if isinstance(x, str) and FIN16.search(x):      # "v9:<heads>:<fingerprint>" (the base hairstyle)
                e = x[-16:]
                return x[:-16] + map_.get(e, e)
            return x
        n = 0
        for p in self.json_files():
            with open(p, encoding="utf-8") as f:
                text = f.read()
            if not any(e in text for e in map_):
                continue
            x = json.loads(text)
            nx = rempl(x)
            if isinstance(nx.get("games"), dict):        # games.<game>.<description key>: the description's names
                nx["games"] = {g: game_keys(v) if isinstance(v, dict) else v for g, v in nx["games"].items()}
            if nx != x:
                replace(p, write_json_like(p, nx, text))
                n += 1
        return {"fingerprints": len(map_), "records": n}

    # 6
    def phase_links(self):
        n = 0
        for d, ds, fs in os.walk(self.D):
            for name in ds + fs:
                p = os.path.join(d, name)
                if not os.path.islink(p):
                    continue
                target = os.readlink(p)
                absolute = target if os.path.isabs(target) else os.path.normpath(os.path.join(d, target))
                if not (absolute == self.D or absolute.startswith(self.D + os.sep)):
                    nv = link_outside(p, absolute)       # into another data folder already converted
                    if nv:
                        tmp = p + ".lien.tmp"
                        os.symlink(nv if os.path.isabs(target) else os.path.relpath(nv, d), tmp)
                        os.replace(tmp, p)
                        n += 1
                    continue
                rel = os.path.relpath(absolute, self.D)
                nv = os.path.join(self.D, F.path(rel))
                if os.path.isabs(target):
                    nc = nv
                else:
                    nc = os.path.relpath(nv, d)
                if nc != target:
                    tmp = p + ".lien.tmp"
                    os.symlink(nc, tmp)
                    os.replace(tmp, p)
                    n += 1
            ds[:] = [x for x in ds if not os.path.islink(os.path.join(d, x))]
        return {"links": n}

    # 7
    def phase_settings(self):
        candidates = [self.settings] if self.settings else [os.path.join(os.path.dirname(self.D), "settings.json"),
                                                       os.path.join(self.D, "settings.json")]
        done = []
        for s in candidates:
            if not s or not os.path.isfile(s):
                continue
            with open(s, encoding="utf-8") as f:
                text = f.read()
            x = json.loads(text)

            def rempl(v, the_key=""):
                if isinstance(v, dict):
                    return {k: rempl(w, k) for k, w in v.items()}
                if isinstance(v, str) and ("/" in v) and not the_key.startswith("_") and not v.startswith("http"):
                    if "{data}" in v:
                        return F.path(v)
                    # only the part inside the data folder is renamed; "~/..." keeps its form
                    rest = under_folder(os.path.expanduser(v), self.D)
                    if rest is not None and v.endswith(rest):
                        return v[:len(v) - len(rest)] + F.path(rest)
                return v
            nx = rempl(x)
            if isinstance(nx.get("games"), dict):        # games.<game>.<description key>: the description's names
                nx["games"] = {g: game_keys(v) if isinstance(v, dict) else v for g, v in nx["games"].items()}
            if nx != x:
                replace(s, json.dumps(nx, ensure_ascii=False, indent=1).encode("utf-8"))
                done.append(s)
        return {"settings": done or "none to change"}

    # 8
    def phase_revision_1(self):
        n = {"npz": 0, "skeleton": 0, "grid": 0, "json": 0}
        for d, _, fs in walk(self.D):
            if in_sources(os.path.relpath(d, self.D)):
                continue
            for name in fs:
                p = os.path.join(d, name)
                if os.path.islink(p) or name in (LOG, F.MARKER) or name.endswith(".tmp"):
                    continue
                if name.endswith(".npz"):
                    try:
                        n["npz"] += npz_fix(p)
                    except zipfile.BadZipFile:
                        say("  not an npz archive, left as it is: %s" % p)
                elif name == "skeleton.json":
                    with open(p, encoding="utf-8") as f:
                        text = f.read()
                    x = json.loads(text)
                    nx = skeleton_fix(x)
                    if nx != x:
                        replace(p, write_json_like(p, nx, text))
                        n["skeleton"] += 1
                elif name.endswith(".pkl") and name.startswith("shape"):
                    import pickle
                    with open(p, "rb") as f:
                        g = pickle.load(f)
                    ng = shape_fix(g)
                    if not _same(ng, g):
                        replace(p, pickle.dumps(ng, protocol=pickle.HIGHEST_PROTOCOL))
                        n["grid"] += 1
                elif name.endswith(".json"):
                    with open(p, "rb") as f:
                        raw = f.read()
                    text = raw.decode("utf-8", "replace")
                    if WAIST.search(text):
                        replace(p, WAIST.sub(lambda m: F.REVISED_CHECKS[m.group(1)], raw.decode("utf-8")).encode("utf-8"))
                        n["json"] += 1
        self.mark(1)
        return n

    # 9
    def phase_revision_2(self):
        """Names format 2 forgot (see the docstring). Stateless: run again, it redoes only what is left."""
        caches = cache_roots(self.D)
        inside = [v + os.sep for v in views(self.D)]
        for q in self.json_files():                  # a folder this one names must be converted first
            with open(q, encoding="utf-8") as f:
                for path in set(re.findall(r'"((?:/|~/)[^"\\]+)"', f.read())):
                    full = os.path.expanduser(path)
                    if any(os.path.realpath(full).startswith(v) or full.startswith(v) for v in inside):
                        continue
                    for nv in {F.revised_path(full, True), F.revised_path(full)} - {full}:
                        if os.path.exists(full) and not os.path.exists(nv):
                            raise SystemExit("%s names %s, in a folder not converted to revision 2 yet: convert "
                                             "that data folder first, then this one" % (q, full))
        n = {"renamed": 0, "records": 0, "fingerprints": 0, "morphs": 0, "links": 0}
        # names, deepest first
        for d, ds, fs in os.walk(self.D, topdown=False):
            rel = os.path.relpath(d, self.D)
            if in_sources(rel) or rel.split(os.sep)[0] in (".chimera-migrate-links",):
                continue
            cache = in_cache(d, caches)
            for name, folder in [(x, False) for x in fs] + [(x, True) for x in ds]:
                q = os.path.join(d, name)
                if rel == "." and name in (LOG, F.MARKER, "sources"):
                    continue
                nv = F.revised_name(name, folder and not os.path.islink(q), cache)
                if nv == name:
                    continue
                if os.path.lexists(os.path.join(d, nv)):
                    raise SystemExit("cannot rename %s: %s already exists" % (q, os.path.join(d, nv)))
                os.rename(q, os.path.join(d, nv))
                n["renamed"] += 1
        # fingerprints: this folder's moved files, then the files of other folders named by a path in a record
        # fingerprints (path, size, date) of this folder's moved files, through every view of the folder (a bind
        # mount gives it a second path, hence a second fingerprint): the files keep their size and date, so the
        # fingerprint they had under a former path is computed again; stateless, a second run finds the same
        cited = set()
        for q in self.json_files():
            with open(q, "rb") as f:
                cited.update(m.decode() for m in re.findall(rb'[":]([0-9a-f]{16})"', f.read()))
        map_ = {}
        all_views = views(self.D)
        for d, _, fs in walk(self.D):
            rel_d = os.path.relpath(d, self.D)
            if in_sources(rel_d):
                continue
            for name in fs:
                f = os.path.join(d, name)
                rel = os.path.normpath(os.path.join(rel_d, name))
                formers = [x for x in F.former_paths(rel) if x != rel]
                if not formers or not os.path.isfile(f):
                    continue
                for view in all_views:
                    for x in formers:
                        e = fingerprint_at(os.path.join(view, x), f)
                        if e in cited:
                            map_[e] = fingerprint_at(os.path.join(view, rel), f)
        texts = {}
        for q in self.json_files():
            with open(q, encoding="utf-8") as f:
                texts[q] = f.read()
        for t in texts.values():
            # a path of another folder: the file is at its revision-2 name (that folder converted first); the
            # fingerprints were computed on the path it had (in this record, or already rewritten by an earlier run)
            for path in set(re.findall(r'"((?:/|~/)[^"\\]+)"', t)):
                full = os.path.expanduser(path)
                for cand in (F.revised_path(full, True), F.revised_path(full)):
                    if os.path.isfile(cand):
                        for old in F.former_paths(cand):
                            if old != cand and not os.path.exists(old):
                                map_.setdefault(fingerprint_at(old, cand), fingerprint(cand))
                        break
        map_ = {a: b for a, b in map_.items() if a != b}
        # records
        for q, text in texts.items():
            try:
                x = json.loads(text)
            except ValueError:
                continue
            cache = in_cache(q, caches)
            built = cache and os.sep + "built" + os.sep in q and os.path.basename(q) == "record.json"
            nx = revised(x, cache, built, map_)
            if nx != x:
                replace(q, write_json_like(q, nx, text))
                n["records"] += 1
                n["fingerprints"] += sum(text.count(e) for e in map_)
        # shape morphs of rebuilt bodies
        for d, _, fs in walk(self.D):
            if in_sources(os.path.relpath(d, self.D)):
                continue
            for name in fs:
                if name.endswith(".pmx") and not os.path.islink(os.path.join(d, name)):
                    n["morphs"] += morph_prefix_fix(os.path.join(d, name))
        # links that point to a renamed path (inside this folder, or into another folder of revision 2)
        for d, ds, fs in walk(self.D):
            for name in ds + fs:
                q = os.path.join(d, name)
                if not os.path.islink(q):
                    continue
                target = os.readlink(q)
                nv = F.revised_path(target, in_cache(os.path.join(d, target) if not os.path.isabs(target) else target, caches))
                if nv != target and not os.path.exists(q) and os.path.exists(os.path.join(d, nv)):
                    tmp = q + ".link.tmp"
                    os.symlink(nv, tmp)
                    os.replace(tmp, q)
                    n["links"] += 1
        n["settings"] = self.revised_settings()
        self.mark(2)
        return n

    def revised_settings(self):
        candidates = [self.settings] if self.settings else [os.path.join(os.path.dirname(self.D), "settings.json"),
                                                         os.path.join(self.D, "settings.json")]
        done = []
        for s in candidates:
            if not s or not os.path.isfile(s):
                continue
            with open(s, encoding="utf-8") as f:
                text = f.read()
            x = json.loads(text)

            def rempl(v, the_key=""):
                if isinstance(v, dict):
                    return {k: rempl(w, k) for k, w in v.items()}
                if isinstance(v, str) and "/" in v and not the_key.startswith("_") and not v.startswith("http"):
                    rest = under_folder(os.path.expanduser(v), self.D) if "{data}" not in v else v
                    if rest is not None and v.endswith(rest):
                        return v[:len(v) - len(rest)] + F.revised_path(rest)
                return v
            nx = rempl(x)
            if nx != x:
                replace(s, json.dumps(nx, ensure_ascii=False, indent=1).encode("utf-8"))
                done.append(s)
        return done or "none to change"

    def mark(self, revision):
        p = os.path.join(self.D, F.MARKER)
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
        except (FileNotFoundError, ValueError):
            j = {"format": F.FORMAT, "converted": time.strftime("%Y-%m-%d %H:%M:%S")}
        if j.get("format") != F.FORMAT:
            return
        j["revision"] = max(revision, int(j.get("revision", 0)))
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(j, f, indent=1)
        os.replace(p + ".tmp", p)

    # 9
    def phase_marker(self):
        p = os.path.join(self.D, F.MARKER)
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"format": F.FORMAT, "revision": F.REVISION, "converted": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, indent=1)
        os.replace(p + ".tmp", p)
        return {"format": F.FORMAT}


def game_keys(v):
    """The keys of a games.<game> setting with the names of the game descriptions (mmd_extrait -> mmd_extracted)."""
    if isinstance(v, dict):
        return {F.DESCRIPTION_KEYS.get(k, F.KEYS.get(k, k)): game_keys(w) for k, w in v.items()}
    return v


# ---------------------------------------------------------------- check
def check(data, limit=40):
    """Every format-1 name, key or word left in a data folder; [] when there is none."""
    D = os.path.realpath(data)
    remainders = []
    if F.format_of(D) == F.FORMAT and F.revision_of(D) < F.REVISION:
        remainders.append(("marker", "%s: format 2 before revision %d" % (F.MARKER, F.REVISION)))
    caches = cache_roots(D)
    for d, ds, fs in walk(D):
        rel = os.path.relpath(d, D)
        cache = in_cache(d, caches)
        for name, folder in [(x, True) for x in ds] + [(x, False) for x in fs]:
            if name.endswith(".tmp"):
                remainders.append(("leftover", os.path.join(rel, name)))
            elif new_name(rel, name, folder) != name:
                remainders.append(("name", os.path.join(rel, name)))
            elif not in_sources(rel) and not (rel == "." and name in ("sources", LOG, ".chimera-migrate-links")) and \
                    rel.split(os.sep)[0] != ".chimera-migrate-links" and \
                    F.revised_name(name, folder and not os.path.islink(os.path.join(d, name)), cache) != name:
                remainders.append(("name", os.path.join(rel, name) + " (revision 2)"))
        if in_sources(rel):
            continue
        for name in fs:
            p = os.path.join(d, name)
            if name.endswith(".json") and name != LOG:
                try:
                    with open(p, encoding="utf-8") as f:
                        x = json.load(f)
                except ValueError:
                    continue
                if conv(x) != x:
                    remainders.append(("record", _first_diff(x, conv(x), os.path.join(rel, name))))
            elif name.endswith(".pmx") and not os.path.islink(p):
                with open(p, "rb") as f:
                    _, _, _, tex = pmx_textures(f.read())
                if [F.path(t) for t in tex] != tex:
                    remainders.append(("model", os.path.join(rel, name)))
            if not os.path.islink(p) and name not in (LOG, F.MARKER):
                why = revision_left(p) or revision_2_left(p, rel, cache)
                if why:
                    remainders.append(("inside", "%s: %s" % (os.path.join(rel, name), why)))
        for name in ds + fs:
            p = os.path.join(d, name)
            if os.path.islink(p):
                rel_c = os.path.relpath(os.path.realpath(p), D)
                if not rel_c.startswith("..") and F.path(rel_c) != rel_c:
                    remainders.append(("link", os.path.join(rel, name)))
                elif rel_c.startswith("..") and link_outside(p, os.path.realpath(p)):
                    remainders.append(("link", os.path.join(rel, name)))
    return remainders


def _first_diff(a, b, ou):
    if isinstance(a, dict) and isinstance(b, dict):
        for (k, v), (k2, v2) in zip(a.items(), b.items()):
            if k != k2:
                return "%s: key %s" % (ou, k)
            if v != v2:
                return _first_diff(v, v2, ou + "/" + k)
    if isinstance(a, list) and isinstance(b, list):
        for i, (v, v2) in enumerate(zip(a, b)):
            if v != v2:
                return _first_diff(v, v2, "%s[%d]" % (ou, i))
    return "%s: %r" % (ou, a if not isinstance(a, (dict, list)) else str(a)[:80])


def main(a):
    if not a:
        print(__doc__)
        return 2
    settings = None
    if "--settings" in a:
        i = a.index("--settings")
        settings = a[i + 1]
        del a[i:i + 2]
    data = a[0]
    if not os.path.isdir(data):
        raise SystemExit("no such folder: %s" % data)
    if "--check" in a:
        remainders = check(data)
        for sort, what in remainders[:60]:
            print("  %-8s %s" % (sort, what))
        print("%d format-1 name(s), key(s) or word(s) left in %s" % (len(remainders), data))
        return 1 if remainders else 0
    return Migration(data, settings).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

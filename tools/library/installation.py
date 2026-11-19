"""Installs the ready-made library (a package built by package.py) into this machine's data folder.

Standard library only: the package is fetched over plain HTTPS (or read from a local folder), each archive is
checked against the sha256 of index.json before anything is unpacked, and unpacking accepts regular files and
folders only, inside their declared root (the archives are untrusted data: no absolute path, no "..", no link,
no device; nothing is executed).

Where the files go comes from the settings (tools/settings.py), so that the workshop finds them without any
other change: workshop.cache (cache/), the folder of workshop.body (base/), paths.data/face-library
(face/), paths.data/clothes-library (clothes/), paths.data/pivot-head (pivot/), and the
mmd_extracted source of the "vroid" game (mmd_extracted/).

Package format 2 (English names of folders, files and record keys: tools/dataformat.py). A package of format 1
is refused: update the code, or install a package of the same format.

The package carries no machine path: its records write {cache}, {base}… where the owner's machine had absolute
paths, and the setup program writes this machine's paths there. The records also keep "fingerprints" of their source
files (common.fingerprint_file: path, size and date); the package lists the old ones, the setup program computes them
again on the installed files and rewrites them, so the workshop does not prepare again what is already prepared.
"""
import hashlib
import json
import os
import shutil
import ssl
import sys
import tarfile
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
if TOOLS not in sys.path:
    sys.path.append(TOOLS)
import settings as R  # noqa: E402
import dataformat as FMT  # noqa: E402  format-2 names; standard library only

FORMAT = FMT.FORMAT             # version of the package format this code reads
ROOTS = ("cache", "base", "face", "clothes", "pivot", "mmd_extracted")
MANIFEST = "library.json"  # in the cache: what the last installation put where
WORK = ".chimera-library"  # in the data folder: downloads and unpacking in progress
BLOCK = 1 << 20


class Error(Exception):
    pass


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ fingerprints (same formula as workshop/common.py)

def fingerprint_file(path):
    """common.fingerprint_file, without importing the workshop (it needs its settings and numpy)."""
    st = os.stat(path)
    return hashlib.sha1(("%s|%d|%d" % (os.path.realpath(path), st.st_size, int(st.st_mtime))).encode()).hexdigest()[:16]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(BLOCK), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ where things go

def roots():
    """{package root: absolute folder on this machine}, from the settings."""
    data = R.path("paths.data")
    sys.path.insert(0, TOOLS)
    import games as games
    return {
        "cache": R.path("workshop.cache"),
        "base": os.path.dirname(R.path("workshop.body")),
        "face": os.path.join(data, "face-library"),
        "clothes": os.path.join(data, "clothes-library"),
        "pivot": os.path.join(data, "pivot-head"),
        "mmd_extracted": games.load("vroid").path("mmd_extracted"),
    }


def put(rel, rt):
    """Absolute path of a package path "<root>/<rest>"."""
    r, _, rest = rel.partition("/")
    if r not in rt:
        raise Error("unknown root in the package: %r" % rel)
    return os.path.join(rt[r], *rest.split("/")) if rest else rt[r]


def replace_roots(text, rt):
    for r in ROOTS:
        text = text.replace("{%s}" % r, rt[r])
    return text


# ------------------------------------------------------------------ fetching

# Certificate bundles of the usual systems: a Python built elsewhere (the one inside Blender) can ship without any.
BUNDLES = ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/cert.pem",
           "/usr/local/etc/openssl/cert.pem", "/opt/homebrew/etc/openssl@3/cert.pem")
_context = None


def context_tls():
    """The default TLS context, given the system's certificates when this Python finds none of its own."""
    global _context
    if _context is None:
        _context = ssl.create_default_context()
        if not _context.cert_store_stats().get("x509_ca"):
            for b in BUNDLES:
                if os.path.isfile(b):
                    _context.load_verify_locations(b)
                    break
    return _context


def _open(req):
    return urllib.request.urlopen(req, timeout=60, context=context_tls() if str(getattr(req, "full_url", req)).startswith("https:") else None)


def open_index(source):
    if _remote(source):
        with _open(_url(source, "index.json")) as r:
            return json.loads(r.read().decode("utf-8"))
    with open(os.path.join(source, "index.json"), encoding="utf-8") as f:
        return json.load(f)


def _remote(source):
    return urllib.parse.urlparse(source).scheme in ("http", "https")


def _url(source, rel):
    return source.rstrip("/") + "/" + urllib.parse.quote(rel)


def download(source, rel, dest, size, expected):
    """dest holds the archive, checked. A finished and correct file is kept (running again does not fetch it
    again); a partial one (dest + ".part") is resumed where it stopped when the server allows it."""
    if os.path.isfile(dest) and os.path.getsize(dest) == size and sha256(dest) == expected:
        say("  %s: already here, checked" % rel)
        return dest
    if not _remote(source):
        src = os.path.join(source, *rel.split("/"))
        if not os.path.isfile(src):
            raise Error("missing from the package folder: %s" % src)
        say("  %s: read from %s" % (rel, source))
        if os.path.getsize(src) != size or sha256(src) != expected:
            raise Error("%s: size or sha256 differs from index.json (damaged copy)" % src)
        return src
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    already = os.path.getsize(part) if os.path.isfile(part) else 0
    if already > size:
        os.remove(part)
        already = 0
    for attempt in range(5):
        req = urllib.request.Request(_url(source, rel), headers={"User-Agent": "chimera-library-installer"})
        if already:
            req.add_header("Range", "bytes=%d-" % already)
        try:
            with _open(req) as r:
                if already and r.status != 206:   # the server ignores Range: start again from zero
                    already = 0
                with open(part, "ab" if already else "wb") as f:
                    t0, n0 = time.time(), already
                    while True:
                        b = r.read(BLOCK)
                        if not b:
                            break
                        f.write(b)
                        already += len(b)
                        if time.time() - t0 > 5:
                            say("  %s: %d / %d MB (%.1f MB/s)" % (rel, already >> 20, size >> 20, (already - n0) / (time.time() - t0) / 1e6))
                            t0, n0 = time.time(), already
            break
        except (OSError, urllib.error.URLError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:   # range past the end: the part is whole or wrong
                break
            already = os.path.getsize(part) if os.path.isfile(part) else 0
            say("  %s: interrupted (%s), resuming at %d MB" % (rel, e, already >> 20))
            time.sleep(2 + 3 * attempt)
    else:
        raise Error("%s: download failed five times; run the same command again to resume" % rel)
    if os.path.getsize(part) != size or sha256(part) != expected:
        os.remove(part)
        raise Error("%s: size or sha256 differs from index.json; the file was removed, run the command again" % rel)
    os.replace(part, dest)
    say("  %s: downloaded, sha256 checked" % rel)
    return dest


# ------------------------------------------------------------------ safe unpacking

def limb_on(m):
    """Package path of a tar member, or Error: regular files and folders only, relative, inside a known root."""
    name = m.name
    if "\\" in name or "\0" in name or name.startswith("/"):
        raise Error("unsafe name in the archive: %r" % name)
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or parts[0] not in ROOTS:
        raise Error("path outside the library in the archive: %r" % name)
    if not (m.isreg() or m.isdir()):
        raise Error("the archive holds something other than a file or a folder: %r" % name)
    return "/".join(parts)


def unpack(archive, folder, cap):
    """Unpacks into `dossier` (a new, empty folder). Writes each file itself (never tarfile.extract): nothing but
    the bytes and the date. Returns the number of bytes written; stops past `cap`."""
    base = os.path.realpath(folder)
    total = 0
    with tarfile.open(archive, "r:*") as t:
        for m in t:
            rel = limb_on(m)
            target = os.path.join(base, *rel.split("/"))
            if os.path.commonpath([base, os.path.realpath(os.path.dirname(target))]) != base:
                raise Error("path outside the library in the archive: %r" % m.name)
            if m.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            total += m.size
            if total > cap:
                raise Error("%s unpacks to more than index.json declares: stopped" % archive)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = t.extractfile(m)
            with open(target, "xb") as f:
                shutil.copyfileobj(src, f, BLOCK)
            os.chmod(target, 0o644)
            os.utime(target, (m.mtime, m.mtime))
    return total


# ------------------------------------------------------------------ merging the shared files

def merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
    return base


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


# ------------------------------------------------------------------ installation

def state_destination(index, rt, replace):
    """(to replace silently: our previous installation, conflicts: someone else's files at our place)."""
    old = read_json(os.path.join(rt["cache"], MANIFEST), {}) or {}
    a_us = set(old.get("units", []))
    our, conflicts = [], []
    for u in index["units"]:
        c = put(u, rt)
        if not os.path.lexists(c):
            continue
        (our if u in a_us else conflicts).append(u)
    return our, conflicts


def already_installed(index, rt):
    """The manifest of a finished installation of this same version whose items are all in place, otherwise None."""
    m = read_json(os.path.join(rt["cache"], MANIFEST), {}) or {}
    if m.get("in_progress") or m.get("format") != index.get("format") or not m.get("version") \
            or m.get("version") != index.get("version") or set(index["units"]) - set(m.get("units", [])):
        return None
    if any(not os.path.lexists(put(u, rt)) for u in index["units"]):
        return None
    return m


def install(source, replace=False, keep=False, yes=False):
    rt = roots()
    data = R.path("paths.data")
    say("Chimera ready-made library")
    say("  from: %s" % source)
    index = open_index(source)
    if index.get("format") != FORMAT:
        raise Error("package format %s, this code reads format %s: update the code (git pull) or the library "
                     "(a package of format %s)" % (index.get("format"), FORMAT, FORMAT))
    FMT.refused_format1(data)            # a data folder of format 1 is converted first, never mixed
    check_index(index)
    n = index["counts"]
    say("  %d models, %d faces, %d hairstyles, %d parts; %.1f GB to download, %.1f GB once installed" % (
        n["models"], n["heads"], n["hairstyles"], n["parts"], sum(a["size"] for a in index["archives"]) / 1e9,
        index["installed_size"] / 1e9))
    for r in ROOTS:
        say("  %-11s -> %s" % (r, rt[r]))
    done = already_installed(index, rt)
    if done:
        say("Already installed: version %s (installed %s), all %d items in place. Nothing to download. "
             "Open the workshop: bin/chimera" % (done["version"], done.get("date", "?"), len(index["units"])))
        return
    our, conflicts = state_destination(index, rt, replace)
    if conflicts and not replace:
        examples = "\n".join("    " + put(u, rt) for u in conflicts[:8])
        raise Error("%d item(s) of the library already exist here and were not put there by a previous installation "
                     "(your own models or another base body), for example:\n%s\nNothing was changed. To set them aside "
                     "(moved, not deleted) and install the library in their place, run again with --replace; "
                     "or install into another data folder (paths.data in settings.json)."
                     % (len(conflicts), examples))
    work = os.path.join(data, WORK)
    need = index["installed_size"] + (0 if not _remote(source) else sum(a["size"] for a in index["archives"]))
    os.makedirs(work, exist_ok=True)
    free = shutil.disk_usage(work).free
    if free < need * 1.05:
        raise Error("not enough disk space in %s: %.1f GB free, %.1f GB needed" % (work, free / 1e9, need * 1.05 / 1e9))

    say("Download and check (%d archives)" % len(index["archives"]))
    files = [download(source, a["path"], os.path.join(work, "archives", os.path.basename(a["path"])),
                            a["size"], a["sha256"]) for a in index["archives"]]

    say("Unpack")
    for old_folder in os.listdir(work):    # left by an installation that stopped half-way
        if old_folder.startswith("extracted-"):
            shutil.rmtree(os.path.join(work, old_folder), ignore_errors=True)
    extracted = os.path.join(work, "extracted-%d" % os.getpid())
    os.makedirs(extracted)
    for a, f in zip(index["archives"], files):
        unpack(f, extracted, a["extracted_size"])
        say("  %s" % a["path"])

    say("Install")
    date = time.strftime("%Y%m%d-%H%M%S")
    put_of_side = os.path.join(work, "replaced-" + date)
    for u in conflicts:
        dest = os.path.join(put_of_side, *u.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(put(u, rt), dest)
    if conflicts:
        say("  %d existing item(s) set aside in %s" % (len(conflicts), put_of_side))
    # from here on the items are ours, even if this run stops half-way: a new run replaces them without a word
    manifest = os.path.join(rt["cache"], MANIFEST)
    already = read_json(manifest, {}) or {}
    write_json(manifest, dict(already, in_progress=True, units=sorted(set(already.get("units", [])) | set(index["units"]))))
    for u in our:   # our previous version of the same item
        c = put(u, rt)
        if os.path.isdir(c) and not os.path.islink(c):
            shutil.rmtree(c)
        else:
            os.remove(c)
    linked = targets_duplicates(index)
    for u in index["units"]:
        src = os.path.join(extracted, *u.split("/"))
        if not os.path.lexists(src):
            if u in linked:          # made below as a link to its identical copy
                continue
            raise Error("package incomplete: %s announced by index.json but absent from the archives" % u)
        dest = put(u, rt)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(src, dest)
    for rel in index["shared"]:          # dictionaries shared with the user's own models: merged
        src = os.path.join(extracted, *rel.split("/"))
        dest = put(rel, rt)
        fresh = read_json(src, {})
        if rel == index.get("catalog") and os.path.exists(dest):
            continue                       # rebuilt by the workshop from the records at start
        write_json(dest, merge(read_json(dest, {}) or {}, fresh))
    for l in index["links"]:               # face library entries of the cache (catalog.prepare_library makes the same)
        link, target = put(l["link"], rt), put(l["target"], rt)
        if os.path.lexists(link):
            if os.path.islink(link):
                os.remove(link)
            else:
                continue
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(target, link)

    n_links = 0
    for d in index.get("duplicates", []):     # the same file under several names: hard links (as in the owner's cache)
        de = put(d["from"], rt)
        folder = os.path.dirname(d["from"])
        names = [t + ".bin" for t in index["heads_bin"] if t + ".bin" != os.path.basename(de)] if d.get("names") == "*heads" \
            else d.get("names") or []
        targets = [folder + "/" + _name_simple(n) for n in names] + [_path_package(v) for v in d.get("to", [])]
        for v in targets:
            c = put(v, rt)
            if os.path.lexists(c):
                continue
            os.makedirs(os.path.dirname(c), exist_ok=True)
            try:
                os.link(de, c)
            except OSError:
                shutil.copyfile(de, c)
            n_links += 1
    say("  %d files linked to their identical copy" % n_links)
    say("Adapt the records to this machine")
    stale = {}
    for old, rel in index["fingerprints"].items():
        stale[old] = fingerprint_file(put(rel, rt))
    n_json = 0
    for u in index["units"] + index["shared"]:
        c = put(u, rt)
        pile = [c] if os.path.isfile(c) else [os.path.join(d, f) for d, _, fs in os.walk(c) for f in fs]
        for f in pile:
            if f.endswith(".json"):
                n_json += _adapt(f, rt, stale)
    say("  %d records adapted" % n_json)
    if not os.path.exists(os.path.join(data, FMT.MARKER)):
        write_json(os.path.join(data, FMT.MARKER), {"format": FORMAT, "revision": index.get("revision", 1)})   # an older package: the code asks for migrate
    write_json(manifest, {
        "format": FORMAT, "version": index.get("version"), "source": source, "date": date, "units": index["units"]})
    shutil.rmtree(extracted)
    if not keep and _remote(source):
        shutil.rmtree(os.path.join(work, "archives"), ignore_errors=True)
    say("Done: %d models installed. Open the workshop: bin/chimera" % n["models"])
    if conflicts:
        say("Your items set aside are in %s." % put_of_side)


def targets_duplicates(index):
    """Package paths that the archives do not hold because they are a link to another file."""
    res = set()
    for d in index.get("duplicates", []):
        folder, name = d["from"].rsplit("/", 1)
        names = [t + ".bin" for t in index["heads_bin"] if t + ".bin" != name] if d.get("names") == "*heads" else d.get("names") or []
        res.update(folder + "/" + n for n in names)
        res.update(d.get("to", []))
    return res


def check_index(index):
    """Every path index.json names stays inside a known root (the index is data from the network too)."""
    for rel in index["units"] + index["shared"] + list(index["fingerprints"].values()):
        _path_package(rel)
    for l in index["links"]:
        _path_package(l["link"]), _path_package(l["target"])
    for d in index.get("duplicates", []):
        _path_package(d["from"])
    for t in index.get("heads_bin", []):
        _name_simple(t + ".bin")
    for a in index["archives"]:
        if not a["path"].startswith("archives/") or "/" in a["path"][9:]:
            raise Error("unexpected archive name in index.json: %r" % a["path"])
        _name_simple(a["path"][9:])


def _name_simple(n):
    if not n or "/" in n or "\\" in n or n in (".", "..") or "\0" in n:
        raise Error("unsafe name in index.json: %r" % n)
    return n


def _path_package(rel):
    parts = rel.split("/")
    if not parts or parts[0] not in ROOTS or any(_name_simple(p) != p for p in parts[1:]):
        raise Error("unsafe path in index.json: %r" % rel)
    return rel


def _adapt(path, rt, fingerprints):
    with open(path, encoding="utf-8") as f:
        t = f.read()
    # the records are JSON: a data folder whose name holds a quote or a backslash must stay valid JSON
    n = replace_roots(t, {r: json.dumps(v, ensure_ascii=False)[1:-1] for r, v in rt.items()})
    if fingerprints:
        for a, b in fingerprints.items():
            if a in n:
                n = n.replace(a, b)
    if n == t:
        return 0
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(n)
    os.replace(path + ".tmp", path)
    return 1

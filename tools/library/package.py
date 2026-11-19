"""Builds the ready-made library package from a prepared installation (owner side).

    python3 -I tools/library/library.py build <installation> <output folder> --statuses <statuses.json>
    python3 -I tools/library/library.py check <output folder> --statuses <statuses.json>

<installation> is a folder holding settings.json (the settings of the prepared data, read through CHIMERA_SETTINGS):
nothing is written there. The output folder can be uploaded as is to a Hugging Face dataset repository.

Which models go in: a VRoid Hub model (`h<id>`) only if its status in statuses.json is "originale" and the
licence read on its page (sources.json of the cache) allows redistribution and the redistribution of modified
versions; any other model only if its own file says so (VRM meta: CC0 or a VRoid Hub licence URL allowing both;
for the base, the meta recorded by tools/vroid_base in provenance.json). Everything else is left out everywhere:
records, thumbnails, textures, catalogue, labels, sources, rejected list.

What goes in (README.md of this folder: the list and why): see LECTURES below.
"""
import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import struct
import sys
import tarfile
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
REPO = os.path.dirname(TOOLS)
sys.path.append(HERE)
import installation as I  # noqa: E402

VERSION_CODE = 1          # version of this code the package is made for (index.json "code")
LIMIT = 1800 << 20       # raw bytes per archive before compression (HF: < 200 GB recommended per file; small ones resume better)
LEVEL = 6                # gzip level
TEX = re.compile(rb"tex/([0-9a-f]{16}\.webp)")
MACHINE = re.compile(rb"/(?:home|Users)/[^/\s\"]+/")   # a path into someone's home folder
# in a refused item's folder only these remain (the page lists refused items by their record)
REFUSALS_KEEP = ("meta.json", "thumbnail.png")


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ licences

def license_hub(v):
    L = (v or {}).get("license") or {}
    return L.get("allowRedistribution") is True and L.get("modification") == "allowModificationRedistribution"


def vrm_meta(path):
    """VRM meta (0.x or 1.0) of a .vrm file, read from its glTF JSON chunk."""
    with open(path, "rb") as f:
        head = f.read(20)
        if head[:4] != b"glTF":
            raise ValueError("%s: not a glb" % path)
        n = struct.unpack("<I", head[12:16])[0]
        j = json.loads(f.read(n))
    e = j.get("extensions") or {}
    return (e.get("VRMC_vrm") or {}).get("meta") or (e.get("VRM") or {}).get("meta") or {}


def license_file(m):
    """Why a model's own VRM meta allows redistributing modified parts, or None."""
    if not m:
        return None
    if str(m.get("licenseName", "")).upper().startswith("CC0"):
        return "VRM meta: licenseName CC0"
    url = m.get("otherPermissionUrl") or m.get("otherLicenseUrl") or ""
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if q.get("redistribution") == ["allow"] and q.get("modification") == ["allow"]:
        return "VRM meta: VRoid Hub licence with redistribution=allow, modification=allow"
    if m.get("modification") == "allowModificationRedistribution" and m.get("allowRedistribution") is True:
        return "VRM 1.0 meta: allowRedistribution, allowModificationRedistribution"
    return None


# ------------------------------------------------------------------ reading the installation

def status_of(entry):
    """Status of a model in statuses.json; the maintainer's file writes the key in French."""
    e = entry or {}
    return e.get("status") or e.get("statut")   # fr-ok: key of the maintainer's existing statuses file


class Installation:
    def __init__(self, folder, statuses):
        os.environ["CHIMERA_SETTINGS"] = os.path.join(os.path.abspath(folder), "settings.json")
        for v in ("CHIMERA_DATA", "CHIMERA_CACHE", "CHIMERA_BODY", "CHIMERA_GAME"):
            os.environ.pop(v, None)
        sys.path.insert(0, os.path.join(TOOLS, "workshop"))
        sys.path.insert(0, os.path.join(TOOLS, "atlas"))
        import common as C
        import games as games
        self.C = C
        self.rt = I.roots()
        self.vrm = games.load("vroid").path("vrm")
        self.cache = self.rt["cache"]
        self.sources = C.read_json(os.path.join(self.cache, "sources.json"), {}) or {}
        with open(statuses, encoding="utf-8") as f:
            self.statuses = json.load(f)

    def models_present(self):
        C = self.C
        n = set(os.listdir(os.path.join(self.cache, "heads"))) | set(os.listdir(os.path.join(self.cache, "hairstyles")))
        for d in os.listdir(os.path.join(self.cache, "parts")):
            m = C.read_json(os.path.join(self.cache, "parts", d, "meta.json")) or {}
            if m.get("character"):
                n.add(m["character"])
        # a model folder of the libraries: one holding its source PMX, face record or garment folders (not index/, attempts/…)
        for r, sign in (("face", "record.json"), ("clothes", "cutout.json"), ("mmd_extracted", None)):
            for x in os.listdir(self.rt[r]):
                d = os.path.join(self.rt[r], x)
                if os.path.isfile(os.path.join(d, sign or x + ".pmx")):
                    n.add(x)
        return n

    def sort(self):
        """{name: why kept}, {name: why left out}."""
        keep, discard = {}, {}
        for n in sorted(self.models_present()):
            src = self.sources.get(n)
            if re.fullmatch(r"h\d+", n):
                st = status_of(self.statuses.get(n[1:]))
                if st != "originale":
                    discard[n] = "status %s" % (st or "unknown")
                elif not src:
                    discard[n] = "no source page recorded"
                elif not license_hub(src):
                    discard[n] = "licence does not allow redistributing modified versions"
                else:
                    keep[n] = "VRoid Hub, original, licence allows redistribution of modifications"
                continue
            if n == "Libre":   # fr-ok: "Libre" is the id of the free head and hairstyle, a folder name in installed caches
                prov = self.C.read_json(os.path.join(self.rt["base"], "provenance.json"), {}) or {}
                why = license_file(prov.get("vrm_meta"))
                what = "base (%s, provenance.json)" % prov.get("model")
            else:
                f = os.path.join(self.vrm, n + ".vrm")
                why = license_file(vrm_meta(f)) if os.path.isfile(f) else None
                what = os.path.basename(f)
            if why and src:
                keep[n] = "%s, %s" % (what, why)
            else:
                discard[n] = "licence not established by the file" if not why else "no source recorded"
        return keep, discard


# ------------------------------------------------------------------ the plan: what goes where

class Plan:
    """files: [(package path, absolute source or None, bytes or None)]; units, shared, links, fingerprints."""

    def __init__(self, inst, keep):
        self.inst, self.C, self.rt, self.keep = inst, inst.C, inst.rt, set(keep)
        self.files, self.units, self.shared, self.links = [], [], [], []
        self.families = {}
        self.kept_points = {"heads": [], "hairstyles": [], "parts": []}
        self.refused = {"heads": [], "hairstyles": [], "parts": []}
        self.textures = set()
        self.fingerprints = {}
        self.inodes = {}     # (device, inode) -> package path of the first copy: the others become "doubles"
        self.duplicates = {}    # first copy -> [other package paths], linked again by the setup program
        # owner's roots in the records, longest first ({cache} before a root it would contain)
        self.tokens = sorted(((v, "{%s}" % k) for k, v in self.rt.items()), key=lambda x: -len(x[0]))
        self.tokens += [(os.path.realpath(v), j) for v, j in self.tokens if os.path.realpath(v) != v]

    def add(self, family, rel, src=None, bytes_=None):
        if src is not None and not src.endswith(".json"):
            st = os.stat(src)
            if st.st_nlink > 1:       # hard links of the cache (a hairstyle's pairs: one file under every head's name)
                first = self.inodes.setdefault((st.st_dev, st.st_ino), rel)
                if first != rel:
                    self.duplicates.setdefault(first, []).append(rel)
                    return
        if src is not None and src.endswith(".json"):
            with open(src, encoding="utf-8") as f:
                bytes_ = self.neutral(f.read()).encode("utf-8")
            src = None
        self.files.append((rel, src, bytes_))
        self.families.setdefault(family, []).append(len(self.files) - 1)
        if src and src.endswith(".pmx"):
            self.fingerprints[self.C.fingerprint_file(src)] = rel

    def neutral(self, text):
        for v, j in self.tokens:
            text = text.replace(json.dumps(v)[1:-1], j)
        return text

    def folder(self, family, rel, src, guard=None, except_=()):
        for d, ds, fs in os.walk(src):
            ds.sort()
            for f in sorted(fs):
                p = os.path.join(d, f)
                r = os.path.relpath(p, src).replace(os.sep, "/")
                if guard is not None and r not in guard or any(fnmatch.fnmatch(r, s) for s in except_):
                    continue
                if os.path.islink(p) or not os.path.isfile(p):
                    raise ValueError("not a regular file: %s" % p)
                self.add(family, rel + "/" + r, p)

    def json_added(self, family, rel, value):
        value = value        # records of format 2, as the workshop writes them
        self.add(family, rel, bytes_=self.neutral(json.dumps(value, ensure_ascii=False, indent=1)).encode("utf-8"))

    # ------------------------------------------------------------ cache
    def build(self):
        C, cache = self.C, self.inst.cache
        read = C.read_json
        heads = {}
        for i in sorted(os.listdir(os.path.join(cache, "heads"))):
            m = read(os.path.join(cache, "heads", i, "meta.json"))
            if m and i in self.keep:
                heads[i] = m
        ret_heads = {i for i, m in heads.items() if m.get("kept")}
        for i, m in heads.items():
            u = "cache/heads/" + i
            self.units.append(u)
            guard = None if i in ret_heads else REFUSALS_KEEP
            self.folder("heads", u, os.path.join(cache, "heads", i), guard=guard)
            (self.kept_points if i in ret_heads else self.refused)["heads"].append(i)
        for i in sorted(os.listdir(os.path.join(cache, "hairstyles"))):
            d = os.path.join(cache, "hairstyles", i)
            m = read(os.path.join(d, "meta.json"))
            if not m or i not in self.keep:
                continue
            m["pairs"] = {t: p for t, p in (m.get("pairs") or {}).items() if t in heads}
            ok, _ = C.hairstyle_kept(m["pairs"], m.get("own")) if "pairs" in m else (m.get("kept"), None)
            u = "cache/hairstyles/" + i
            self.units.append(u)
            self.json_added("hairstyles", u + "/meta.json", m)
            if ok:
                self.kept_points["hairstyles"].append(i)
                pairs = {t + ".bin" for t in ret_heads}
                guard = {f for f in os.listdir(d) if f != "meta.json" and (not f.endswith(".bin") or f in pairs)}
            else:
                self.refused["hairstyles"].append(i)
                guard = set(REFUSALS_KEEP) - {"meta.json"}
            self.folder("hairstyles", u, d, guard=guard)
        parts_kept = []
        for i in sorted(os.listdir(os.path.join(cache, "parts"))):
            d = os.path.join(cache, "parts", i)
            m = read(os.path.join(d, "meta.json"))
            if not m or m.get("character") not in self.keep:
                continue
            note = (m.get("revue") or {}).get("note")
            ok = m.get("kept") and not (note is not None and note > C.NOTE_REFUSALS)
            u = "cache/parts/" + i
            self.units.append(u)
            self.folder("parts", u, d, guard=None if ok else REFUSALS_KEEP)
            (self.kept_points if ok else self.refused)["parts"].append(i)
            if ok:
                parts_kept.append(m)
        # base, body, the base head's library entry, the base and the pivot head
        for r in ("base", "body", "library/Libre"):   # fr-ok: "Libre" is the id of the free head and hairstyle, a folder name in installed caches
            if os.path.isdir(os.path.join(cache, r)):
                self.units.append("cache/" + r)
                self.folder("base", "cache/" + r, os.path.join(cache, r))
        self.units += ["base", "pivot"]
        self.folder("base", "base", self.rt["base"])
        self.folder("base", "pivot", self.rt["pivot"])
        # face library and source models of the kept models; the cache's biblio/<name> links to it
        for n in sorted(self.keep):
            if os.path.isdir(os.path.join(self.rt["face"], n)):
                self.units.append("face/" + n)
                self.folder("face", "face/" + n, os.path.join(self.rt["face"], n))
                if os.path.islink(os.path.join(cache, "library", n)):
                    self.links.append({"link": "cache/library/" + n, "target": "face/" + n})
            if os.path.isdir(os.path.join(self.rt["mmd_extracted"], n)):
                self.units.append("mmd_extracted/" + n)
                self.folder("sources", "mmd_extracted/" + n, os.path.join(self.rt["mmd_extracted"], n))
        self.clothes(parts_kept)
        self.textures_cache()
        self.annexes()
        return self

    def clothes(self, parts):
        """The garment library folders of the kept parts (porter reads them at assembly) and the textures their PMX
        name; a library part that has no record in the cache is left out (it would be prepared at the next prepare)."""
        sys.path.insert(0, os.path.join(TOOLS, "atlas"))
        import pmx as PMX
        root = os.path.realpath(self.rt["clothes"])
        tex = set()
        for m in parts:
            d = os.path.realpath(self.C.absolute(m["folder"]))
            if not d.startswith(root + os.sep):
                raise ValueError("part folder outside the garment library: %s" % d)
            rel = os.path.relpath(d, root).replace(os.sep, "/")
            u = "clothes/" + rel
            self.units.append(u)
            self.folder("clothes", u, d)
            fi = self.C.read_json(os.path.join(d, "record.json")) or {}
            for k in (fi.get("pmx"), (fi.get("trimmed") or {}).get("pmx")):
                if k and os.path.isfile(os.path.join(d, k)):
                    for t in PMX.read(os.path.join(d, k))["textures"]:
                        tex.add(os.path.normpath(os.path.join(d, t.replace("\\", "/"))))
        for t in sorted(tex):
            if not os.path.isfile(t) or not t.startswith(root + os.sep):
                say("  texture named by a part but absent or outside the library: %s" % t)
                continue
            rel = "clothes/" + os.path.relpath(t, root).replace(os.sep, "/")
            if rel.split("/")[2] != "tex":   # a part's own texture is already inside its folder
                continue
            self.units.append(rel)
            self.add("clothes", rel, t)

    def textures_cache(self):
        """cache/tex: only the textures named by a kept record or model (glb, json)."""
        for rel, src, bytes_ in list(self.files):
            if not rel.startswith("cache/") or not (rel.endswith(".glb") or rel.endswith(".json")):
                continue
            if bytes_ is None:
                with open(src, "rb") as f:
                    bytes_ = f.read()
            self.textures.update(x.decode() for x in TEX.findall(bytes_))
        for t in sorted(self.textures):
            p = os.path.join(self.inst.cache, "tex", t)
            if not os.path.isfile(p):
                raise ValueError("texture named but absent: %s" % p)
            self.units.append("cache/tex/" + t)
            self.add("parts", "cache/tex/" + t, p)

    def annexes(self):
        """Shared dictionaries, filtered to the kept models and items."""
        C, cache = self.C, self.inst.cache
        items = {"heads/" + i for i in self.kept_points["heads"] + self.refused["heads"]}
        items |= {"hairstyles/" + i for i in self.kept_points["hairstyles"] + self.refused["hairstyles"]}
        items |= {"parts/" + i for i in self.kept_points["parts"] + self.refused["parts"]}
        lbl = {k: v for k, v in (C.read_json(os.path.join(cache, "labels.json"), {}) or {}).items() if k in items}
        src = {}
        for n, v in sorted(self.inst.sources.items()):
            if n in self.keep:
                v = json.loads(json.dumps(v))
                (v.get("license") or {}).pop("contactInformation", None)   # personal contact: not a term of use
                src[n] = v
        self.sources = src
        vis = C.read_json(os.path.join(self.rt["face"], "index.json"), {}) or {}
        if "regions" in vis:
            vis["regions"] = {p: {n: x for n, x in d.items() if n in self.keep} for p, d in vis["regions"].items()}
        ctl = {n: v for n, v in (C.read_json(os.path.join(self.rt["clothes"], "check.json"), {}) or {}).items()
               if n in self.keep}
        for rel, v in (("cache/labels.json", lbl), ("cache/sources.json", src), ("face/index.json", vis),
                       ("clothes/check.json", ctl)):
            self.shared.append(rel)
            self.json_added("base", rel, v)

    def catalog(self):
        """catalogue.json of the package, made by the workshop's own code on the kept records (no file written)."""
        import catalog as K
        cat = K.build(write=False)
        all_ = {"heads": set(self.kept_points["heads"] + self.refused["heads"]),
                "hairstyles": set(self.kept_points["hairstyles"] + self.refused["hairstyles"]),
                "parts": set(self.kept_points["parts"] + self.refused["parts"])}
        for k in ("heads", "hairstyles", "parts"):
            cat[k] = [x for x in cat[k] if x["id"] in self.kept_points[k]]
        kind = {"face": "heads", "hairstyle": "hairstyles", "fraction": "parts"}
        cat["refusals"] = [r for r in cat["refusals"] if r["id"] in all_[kind[r["kind"]]]]
        cat["sources"] = {n: v for n, v in self.sources.items() if n in cat["sources"]}
        cat["counts"] = {"prepared": sum(len(v) for v in all_.values()),
                          "kept_count": sum(len(cat[k]) for k in all_), "refused": len(cat["refusals"]),
                          **{k: [len(cat[k]), len(all_[k])] for k in all_}}
        self.shared.append("cache/catalog.json")
        self.json_added("base", "cache/catalog.json", cat)
        return cat


# ------------------------------------------------------------------ archives

def _write_archive(args):
    path, entries = args
    size = 0
    with open(path + ".tmp", "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=LEVEL, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
                folders = set()
                for rel, src, bytes_ in entries:
                    parts = rel.split("/")
                    for k in range(1, len(parts)):
                        d = "/".join(parts[:k])
                        if d not in folders:
                            folders.add(d)
                            ti = tarfile.TarInfo(d)
                            ti.type, ti.mode, ti.mtime = tarfile.DIRTYPE, 0o755, 0
                            t.addfile(ti)
                    ti = tarfile.TarInfo(rel)
                    ti.mode = 0o644
                    if bytes_ is None:
                        st = os.stat(src)
                        ti.size, ti.mtime = st.st_size, int(st.st_mtime)
                        with open(src, "rb") as f:
                            t.addfile(ti, f)
                    else:
                        ti.size, ti.mtime = len(bytes_), 0
                        t.addfile(ti, io.BytesIO(bytes_))
                    size += ti.size
    os.replace(path + ".tmp", path)
    return path, size, os.path.getsize(path), I.sha256(path)


def slice_(plan):
    """[(name, [entries])]: one family per archive, cut at LIMIT raw bytes (an item never split)."""
    res = []
    names = {"heads": "heads", "hairstyles": "hairstyles", "parts": "parts"}      # format-2 names of the archives
    for family in ("base", "heads", "hairstyles", "parts", "clothes", "face", "sources"):
        batches, batch, n = [], [], 0
        for k in plan.families.get(family, []):
            rel, src, bytes_ = plan.files[k]
            t = os.path.getsize(src) if bytes_ is None else len(bytes_)
            if batch and n + t > LIMIT and _unit(rel) != _unit(batch[-1][0]):
                batches.append(batch)
                batch, n = [], 0
            batch.append(plan.files[k])
            n += t
        if batch:
            batches.append(batch)
        for i, l in enumerate(batches, 1):
            res.append(("%s-%02d.tar.gz" % (names.get(family, family), i), l))
    return res


def duplicates(d, heads):
    """[{"de": first copy, "names": names next to it | "*heads"}]: "*heads" = <head>.bin for every kept head but the
    first copy's own name (a hairstyle's pairs, the same file under every head's name)."""
    all_ = {t + ".bin" for t in heads}
    res = []
    for de, to in sorted(d.items()):
        folder, name = de.rsplit("/", 1)
        if all(v.rsplit("/", 1)[0] == folder for v in to):
            names = sorted(v.rsplit("/", 1)[1] for v in to)
            res.append({"from": de, "names": "*heads" if set(names) == all_ - {name} else names})
        else:
            res.append({"from": de, "to": sorted(to)})
    return res


def _unit(rel):
    p = rel.split("/")
    return "/".join(p[:3] if p[0] in ("cache", "clothes") else p[:2])


# ------------------------------------------------------------------ texts of the dataset

def _cell(s, n=90):
    s = re.sub(r"\s+", " ", str(s or "")).strip().replace("|", "\\|")
    return s if len(s) <= n else s[:n - 1] + "…"


def forbidden(L):
    names = (("allowExcessivelySexualUsage", "excessively sexual"), ("allowExcessivelyViolentUsage", "excessively violent"),
            ("allowPoliticalOrReligiousUsage", "political or religious"), ("allowAntisocialOrHateUsage", "antisocial or hateful"))
    return ", ".join(t for k, t in names if L.get(k) is False) or "none listed"


def credits(sources, usage):
    lines = ["# Credits", "",
              "Every model below keeps its author's terms. VRoid Hub models: original characters (no existing work, "
              "real person or uncertain source), whose VRM licence, read on the model's page on the date given in "
              "`sources.json`, allows redistribution and the redistribution of modified versions. Other models: "
              "VRoid Studio sample characters whose own file carries a CC0 licence.",
              "",
              "Before using a character you assembled, read the terms of **each** model whose parts it carries "
              "(columns below; the full fields are in `sources.json`, the current terms on the model's page).",
              "", "Parts: F = face, H = hairstyle, P = clothes and accessories.", "",
              "| Id | Model | Author | Page | Parts | Credit | Commercial use | Not allowed | Other terms |",
              "|---|---|---|---|---|---|---|---|---|"]
    for n, v in sorted(sources.items(), key=lambda x: (_cell(x[1].get("name")).lower(), x[0])):
        L = v.get("license") or {}
        other = "; ".join(x for x in (
            ("licence: " + L["name"]) if L.get("name") else "",
            ("other licence: " + _cell(L["otherLicenseUrl"], 60)) if L.get("otherLicenseUrl") else "",
            ("copyright: " + _cell(L["copyrightInformation"], 60)) if L.get("copyrightInformation") else "",
            "additional terms in sources.json" if L.get("thirdPartyLicenses") else "") if x)
        link = "[%s](%s)" % (_cell(v.get("site") or "page", 30), v["url"]) if v.get("url") else ""
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            n, _cell(v.get("name")), _cell(v.get("author")), link, usage.get(n, ""),
            {"required": "**required**", "unnecessary": "not required"}.get(L.get("creditNotation"), _cell(L.get("creditNotation")) or "?"),
            _cell(L.get("commercialUsage") or "?"), forbidden(L), _cell(other, 160) or "-"))
    return "\n".join(lines) + "\n"


def map_(index, address_example):
    n = index["counts"]
    return """---
pretty_name: Chimera VRoid Library
license: other
license_name: per-model-vrm-terms
license_link: https://huggingface.co/datasets/%(repo)s/blob/main/CREDITS.md
tags:
- 3d
- vroid
- vrm
- avatar
- character-creator
viewer: false
---

# Chimera VRoid Library

Ready-made parts library for [Chimera 3D Avatar Maker](https://github.com/RoseAndKnight/Chimera-3D-Avatar-Maker), the character workshop that mixes
the face, hairstyle and clothes of different 3D models on one body and exports the result as PMX and VRM.

It holds **%(models)d free VRoid models**, already prepared: %(heads)d faces, %(hairstyles)d hairstyles and
%(parts)d clothes and accessories that the workshop can show and assemble at once, with their names, descriptions
and tags (English and French) and a link to each original model. Installing it saves the hours of Blender work the
preparation takes.

## Install with Chimera

From a clone of the Chimera repository:

```
python3 -I tools/library/library.py install
bin/chimera
```

The command downloads `index.json`, then the %(archives)d archives listed in it (%(size_gb).1f GB, about %(installed).1f GB
once installed), checks each one against its sha256 before unpacking, and installs them into the data folder of
your settings. Running it again resumes an interrupted download. It does not overwrite models you added yourself:
if something of yours is in the way it stops and says so (`--replace` sets those items aside, never deletes them).

This repository is not meant to be read with the `datasets` library: the archives are the workshop's own data
(PMX, glTF, textures, records). Format %(format)d, for Chimera code version %(code)d.

Format 2 names every folder, file and record field in English. A copy installed from the earlier package (format 1,
French names) is converted in place, without downloading again: `python3 -I tools/library/library.py migrate
<data folder>`; or remove it and install again.

## Where the models come from

- VRoid Hub models (ids `h<number>`): original characters published on [VRoid Hub](https://hub.vroid.com) whose VRM
  licence allows redistribution **and** the redistribution of modified versions. Models based on an existing work
  (fan art), on a real person, or whose source is uncertain were left out, with all their parts, textures and
  thumbnails.
- VRoid Studio sample characters (`Base_Female`, `Vita`…, and the base body, `Libre`, made from `Sendagaya_Shibu`): <!-- fr-ok: id of the free base -->
  CC0, as written in each model's own file.

## Terms of use

There is no single licence for this library: **each model keeps its author's terms** (commercial use, credit,
forbidden uses). `CREDITS.md` gives one line per model (name, author, link, whether credit is required, conditions);
`sources.json` gives every licence field as read on the model's page, and the date it was read. A character you
assemble carries parts of several models: it must respect the terms of each of them. Read the current terms on the
model's page before any use.

The maintainers of this library are not responsible for a use that breaks the terms of a model. The code of
Chimera itself is under the MIT licence; that licence does not cover the models.

## Removal requests

If you are the author of one of these models and want it removed, open a discussion in the Community tab of this
dataset with the model's id (first column of `CREDITS.md`). The model will be removed from the next version.

## Contents

- `index.json`: format version, Chimera code version, the archives with size and sha256, counts.
- `archives/*.tar.gz`: the prepared data, by family (base, faces, hairstyles, parts, garment library, face library,
  source models).
- `CREDITS.md`, `sources.json`: authors and terms.
- `upload.sh`: the script that uploaded this folder (for the maintainer).
""" % dict(n, archives=len(index["archives"]), size_gb=sum(a["size"] for a in index["archives"]) / 1e9,
           installed=index["installed_size"] / 1e9, repo=address_example, format=index["format"], code=index["code"])


SETTING_IN_LINE = r"""#!/bin/bash
# Uploads this folder to the Hugging Face dataset repository named below. Not run by the tools: by the maintainer,
# once logged in. Needs the `hf` command (https://huggingface.co/docs/huggingface_hub/guides/cli):
#     curl -LsSf https://hf.co/cli/install.sh | bash        (or: pip install -U huggingface_hub)
#     hf auth login
# Then: bash upload.sh [owner/name]
# The address the setup program reads (library.url in settings.default.json) must name the same repository:
#     https://huggingface.co/datasets/<owner/name>/resolve/main
set -euo pipefail
REPO="${1:-%(repo)s}"
ICI="$(cd "$(dirname "$0")" && pwd)"
command -v hf >/dev/null || { echo "hf is not installed: curl -LsSf https://hf.co/cli/install.sh | bash" >&2; exit 1; }
hf auth whoami >/dev/null || { echo "not logged in: hf auth login" >&2; exit 1; }
hf repos create "$REPO" --repo-type dataset --exist-ok
# archives first, index.json next: a setup program never reads an index whose archives are not there yet
hf upload "$REPO" "$ICI/archives" archives --repo-type dataset --commit-message "Archives (format %(format)s)"
hf upload "$REPO" "$ICI" . --repo-type dataset --include "README.md" --include "CREDITS.md" --include "sources.json" \
    --include "upload.sh" --delete "mettre-en-ligne.sh" --commit-message "Card, credits and sources"   # fr-ok: script name of the format-1 package, removed from the repository
hf upload "$REPO" "$ICI/index.json" index.json --repo-type dataset --commit-message "Index (format %(format)s)"
# last, once the new index is in place: the archives of an earlier package (format 1, with French
# names) that this one does not hold are deleted from the repository
hf upload "$REPO" "$ICI/archives" archives --repo-type dataset --delete "*" --commit-message "Remove the archives of the earlier package"
echo "Done: https://huggingface.co/datasets/$REPO"
"""


# ------------------------------------------------------------------ build and check

def build(installation, output, statuses, repo="Rose-and-Knight/chimera-vroid-library", tasks=4):
    t0 = time.time()
    inst = Installation(installation, statuses)
    keep, discard = inst.sort()
    say("models: %d kept, %d left out" % (len(keep), len(discard)))
    for n, r in sorted(discard.items()):
        say("  left out %s: %s" % (n, r))
    plan = Plan(inst, keep).build()
    cat = plan.catalog()
    # fingerprints the records keep and the package cannot rewrite: those items would be prepared again
    known = set(plan.fingerprints)
    missing = []
    refused = {"cache/%s/%s" % (k, i) for k, v in plan.refused.items() for i in v}
    for rel, src, bytes_ in plan.files:
        if _unit(rel) in refused:   # a refused part's source is not shipped: nothing would prepare it again
            continue
        if rel.endswith("meta.json") or rel.endswith("record.json"):
            m = json.loads(bytes_) if bytes_ is not None else {}
            for k in ("fingerprint", "body"):
                v = str(m.get(k) or "")
                h = v.rsplit(":", 1)[-1]
                if re.fullmatch(r"[0-9a-f]{16}", h) and h not in known:
                    missing.append((rel, k, h))
    if missing:
        say("WARNING: %d fingerprints without their file in the package (those items would be prepared again):" % len(missing))
        for x in missing[:10]:
            say("  %s %s %s" % x)
    os.makedirs(os.path.join(output, "archives"), exist_ok=True)
    batches = slice_(plan)
    with ProcessPoolExecutor(tasks) as pool:
        done = list(pool.map(_write_archive, [(os.path.join(output, "archives", n), l) for n, l in batches]))
    archives = [{"path": "archives/" + os.path.basename(c), "size": t, "sha256": h, "extracted_size": raw}
                for c, raw, t, h in done]
    usage = {}
    for k, letter in (("heads", "F"), ("hairstyles", "H")):
        for i in plan.kept_points[k]:
            usage.setdefault(i, set()).add(letter)
    for p in cat["parts"]:
        usage.setdefault(p["character"], set()).add("P")
    usage = {n: "".join(x for x in "FHP" if x in v) for n, v in usage.items()}
    index = {
        "format": I.FORMAT, "code": VERSION_CODE, "version": time.strftime("%Y-%m-%d"),
        "counts": {"models": len(keep), "heads": len(cat["heads"]), "hairstyles": len(cat["hairstyles"]),
                    "parts": len(cat["parts"]), "refusals": len(cat["refusals"])},
        "archives": archives, "installed_size": sum(a["extracted_size"] for a in archives),
        "units": sorted(set(plan.units)), "shared": plan.shared, "catalog": "cache/catalog.json",
        "links": plan.links, "fingerprints": plan.fingerprints, "duplicates": duplicates(plan.duplicates, plan.kept_points["heads"]),
        "heads_bin": sorted(plan.kept_points["heads"]),
        "models": sorted(keep), "discarded": discard,
    }
    I.write_json(os.path.join(output, "index.json"), index)
    I.write_json(os.path.join(output, "sources.json"), plan.sources)
    with open(os.path.join(output, "CREDITS.md"), "w", encoding="utf-8") as f:
        f.write(credits(plan.sources, usage))
    with open(os.path.join(output, "README.md"), "w", encoding="utf-8") as f:
        f.write(map_(index, repo))
    with open(os.path.join(output, "upload.sh"), "w", encoding="utf-8") as f:
        f.write(SETTING_IN_LINE % {"repo": repo, "format": I.FORMAT})
    os.chmod(os.path.join(output, "upload.sh"), 0o755)
    say("package: %d models, %d faces, %d hairstyles, %d parts (%d refused records), %d archives, %.2f GB, %.2f GB installed, %.0f s" % (
        len(keep), len(cat["heads"]), len(cat["hairstyles"]), len(cat["parts"]), len(cat["refusals"]), len(archives),
        sum(a["size"] for a in archives) / 1e9, index["installed_size"] / 1e9, time.time() - t0))
    return index


def check(output, statuses):
    """Reads every archive back: sizes and sha256 of index.json, safe members, no machine path, no id of a model
    that is not kept (statuses.json other than "originale", or left out), no texture of cache/tex that no kept record
    names, every unit present. Returns the list of defects."""
    index = I.read_json(os.path.join(output, "index.json"))
    with open(statuses, encoding="utf-8") as f:
        st = json.load(f)
    kept = set(index["models"])
    out_of = {i for i, v in st.items() if status_of(v) != "originale"} | {
        n[1:] if re.fullmatch(r"h\d+", n) else n for n in index["discarded"]}
    out_of -= {n[1:] for n in kept if n.startswith("h")}
    pattern = re.compile(rb"(?<![0-9])(" + b"|".join(re.escape(x.encode()) for x in sorted(out_of)) + rb")(?![0-9])") if out_of else None
    defaults, names, texts_tex, tex = [], set(), set(), set()
    for a in index["archives"]:
        c = os.path.join(output, a["path"])
        if os.path.getsize(c) != a["size"] or I.sha256(c) != a["sha256"]:
            defaults.append("%s: size or sha256 differs from index.json" % a["path"])
        with tarfile.open(c, "r:gz") as t:
            for m in t:
                try:
                    rel = I.limb_on(m)
                except I.Error as e:
                    defaults.append(str(e))
                    continue
                if m.isdir():
                    continue
                names.add(rel)
                if pattern and pattern.search(rel.encode()):
                    defaults.append("%s: id of a model left out in the path" % rel)
                b = t.extractfile(m).read()
                if MACHINE.search(b):
                    defaults.append("%s: machine path inside" % rel)
                if pattern and (rel.endswith(".json") or rel.endswith(".md")) and pattern.search(b):
                    defaults.append("%s: id of a model left out inside" % rel)
                if rel.startswith("cache/tex/"):
                    tex.add(rel[len("cache/tex/"):])
                elif rel.startswith("cache/") and (rel.endswith(".glb") or rel.endswith(".json")):
                    texts_tex.update(x.decode() for x in TEX.findall(b))
    for f in ("CREDITS.md", "README.md", "sources.json"):
        with open(os.path.join(output, f), "rb") as fh:
            b = fh.read()
        if MACHINE.search(b) or (pattern and pattern.search(b)):
            defaults.append("%s: machine path or id of a model left out" % f)
    orphan = tex - texts_tex
    if orphan:
        defaults.append("%d textures of cache/tex named by no kept record, e.g. %s" % (len(orphan), sorted(orphan)[:3]))
    linked = I.targets_duplicates(index)
    for u in index["units"] + index["shared"]:
        if u in linked:
            continue
        if u not in names and not any(n.startswith(u + "/") for n in names):
            defaults.append("unit missing from the archives: %s" % u)
    say("checked: %d archives, %d files, %d textures, %d models left out searched; %d defects" % (
        len(index["archives"]), len(names), len(tex), len(out_of), len(defaults)))
    for d in defaults[:30]:
        say("  DEFECT " + d)
    return defaults

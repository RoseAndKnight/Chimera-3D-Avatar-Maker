"""Installs the ready-made library (a package built by package.py) into this machine's data folder.

Standard library only: the package is fetched over plain HTTPS (or read from a local folder), each archive is
checked against the sha256 of index.json before anything is unpacked, and unpacking accepts regular files and
folders only, inside their declared root (the archives are untrusted data: no absolute path, no "..", no link,
no device; nothing is executed).

Where the files go comes from the settings (tools/settings.py), so that the workshop finds them without any
other change: workshop.cache (cache/), the folder of workshop.body (base/), paths.data/bibliotheque-visage
(visage/), paths.data/bibliotheque-vetements (vetements/), paths.data/tete-pivot (pivot/), and the
mmd_extrait source of the "vroid" game (mmd_extrait/).

The package carries no machine path: its records write {cache}, {base}… where the owner's machine had absolute
paths, and the installer writes this machine's paths there. The records also keep "fingerprints" of their source
files (commun.empreinte_fichier: path, size and date); the package lists the old ones, the installer computes them
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

ICI = os.path.dirname(os.path.abspath(__file__))
OUTILS = os.path.dirname(ICI)
if OUTILS not in sys.path:
    sys.path.append(OUTILS)
import settings as R  # noqa: E402

FORMAT = 1                       # version of the package format this code reads
RACINES = ("cache", "base", "visage", "vetements", "pivot", "mmd_extrait")
MANIFESTE = "bibliotheque.json"  # in the cache: what the last installation put where
TRAVAIL = ".chimera-bibliotheque"  # in the data folder: downloads and unpacking in progress
BLOC = 1 << 20


class Erreur(Exception):
    pass


def dire(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ fingerprints (same formula as workshop/common.py)

def empreinte_fichier(chemin):
    """commun.empreinte_fichier, without importing the workshop (it needs its settings and numpy)."""
    st = os.stat(chemin)
    return hashlib.sha1(("%s|%d|%d" % (os.path.realpath(chemin), st.st_size, int(st.st_mtime))).encode()).hexdigest()[:16]


def sha256(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for b in iter(lambda: f.read(BLOC), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ where things go

def racines():
    """{package root: absolute folder on this machine}, from the settings."""
    donnees = R.chemin("paths.data")
    sys.path.insert(0, OUTILS)
    import games as jeux
    return {
        "cache": R.chemin("workshop.cache"),
        "base": os.path.dirname(R.chemin("workshop.body")),
        "visage": os.path.join(donnees, "bibliotheque-visage"),
        "vetements": os.path.join(donnees, "bibliotheque-vetements"),
        "pivot": os.path.join(donnees, "tete-pivot"),
        "mmd_extrait": jeux.charger("vroid").chemin("mmd_extrait"),
    }


def placer(rel, rac):
    """Absolute path of a package path "<root>/<rest>"."""
    r, _, reste = rel.partition("/")
    if r not in rac:
        raise Erreur("unknown root in the package: %r" % rel)
    return os.path.join(rac[r], *reste.split("/")) if reste else rac[r]


def remplacer_racines(texte, rac):
    for r in RACINES:
        texte = texte.replace("{%s}" % r, rac[r])
    return texte


# ------------------------------------------------------------------ fetching

# Certificate bundles of the usual systems: a Python built elsewhere (the one inside Blender) can ship without any.
BUNDLES = ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/cert.pem",
           "/usr/local/etc/openssl/cert.pem", "/opt/homebrew/etc/openssl@3/cert.pem")
_contexte = None


def contexte_tls():
    """The default TLS context, given the system's certificates when this Python finds none of its own."""
    global _contexte
    if _contexte is None:
        _contexte = ssl.create_default_context()
        if not _contexte.cert_store_stats().get("x509_ca"):
            for b in BUNDLES:
                if os.path.isfile(b):
                    _contexte.load_verify_locations(b)
                    break
    return _contexte


def _ouvrir(req):
    return urllib.request.urlopen(req, timeout=60, context=contexte_tls() if str(getattr(req, "full_url", req)).startswith("https:") else None)


def ouvrir_index(source):
    if _distante(source):
        with _ouvrir(_url(source, "index.json")) as r:
            return json.loads(r.read().decode("utf-8"))
    with open(os.path.join(source, "index.json"), encoding="utf-8") as f:
        return json.load(f)


def _distante(source):
    return urllib.parse.urlparse(source).scheme in ("http", "https")


def _url(source, rel):
    return source.rstrip("/") + "/" + urllib.parse.quote(rel)


def telecharger(source, rel, dest, taille, attendu):
    """dest holds the archive, checked. A finished and correct file is kept (running again does not fetch it
    again); a partial one (dest + ".part") is resumed where it stopped when the server allows it."""
    if os.path.isfile(dest) and os.path.getsize(dest) == taille and sha256(dest) == attendu:
        dire("  %s: already here, checked" % rel)
        return dest
    if not _distante(source):
        src = os.path.join(source, *rel.split("/"))
        if not os.path.isfile(src):
            raise Erreur("missing from the package folder: %s" % src)
        dire("  %s: read from %s" % (rel, source))
        if os.path.getsize(src) != taille or sha256(src) != attendu:
            raise Erreur("%s: size or sha256 differs from index.json (damaged copy)" % src)
        return src
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    deja = os.path.getsize(part) if os.path.isfile(part) else 0
    if deja > taille:
        os.remove(part)
        deja = 0
    for essai in range(5):
        req = urllib.request.Request(_url(source, rel), headers={"User-Agent": "chimera-library-installer"})
        if deja:
            req.add_header("Range", "bytes=%d-" % deja)
        try:
            with _ouvrir(req) as r:
                if deja and r.status != 206:   # the server ignores Range: start again from zero
                    deja = 0
                with open(part, "ab" if deja else "wb") as f:
                    t0, n0 = time.time(), deja
                    while True:
                        b = r.read(BLOC)
                        if not b:
                            break
                        f.write(b)
                        deja += len(b)
                        if time.time() - t0 > 5:
                            dire("  %s: %d / %d MB (%.1f MB/s)" % (rel, deja >> 20, taille >> 20, (deja - n0) / (time.time() - t0) / 1e6))
                            t0, n0 = time.time(), deja
            break
        except (OSError, urllib.error.URLError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 416:   # range past the end: the part is whole or wrong
                break
            deja = os.path.getsize(part) if os.path.isfile(part) else 0
            dire("  %s: interrupted (%s), resuming at %d MB" % (rel, e, deja >> 20))
            time.sleep(2 + 3 * essai)
    else:
        raise Erreur("%s: download failed five times; run the same command again to resume" % rel)
    if os.path.getsize(part) != taille or sha256(part) != attendu:
        os.remove(part)
        raise Erreur("%s: size or sha256 differs from index.json; the file was removed, run the command again" % rel)
    os.replace(part, dest)
    dire("  %s: downloaded, sha256 checked" % rel)
    return dest


# ------------------------------------------------------------------ safe unpacking

def membre_sur(m):
    """Package path of a tar member, or Erreur: regular files and folders only, relative, inside a known root."""
    nom = m.name
    if "\\" in nom or "\0" in nom or nom.startswith("/"):
        raise Erreur("unsafe name in the archive: %r" % nom)
    parts = [p for p in nom.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or parts[0] not in RACINES:
        raise Erreur("path outside the library in the archive: %r" % nom)
    if not (m.isreg() or m.isdir()):
        raise Erreur("the archive holds something other than a file or a folder: %r" % nom)
    return "/".join(parts)


def deballer(archive, dossier, plafond):
    """Unpacks into `dossier` (a new, empty folder). Writes each file itself (never tarfile.extract): nothing but
    the bytes and the date. Returns the number of bytes written; stops past `plafond`."""
    base = os.path.realpath(dossier)
    total = 0
    with tarfile.open(archive, "r:*") as t:
        for m in t:
            rel = membre_sur(m)
            cible = os.path.join(base, *rel.split("/"))
            if os.path.commonpath([base, os.path.realpath(os.path.dirname(cible))]) != base:
                raise Erreur("path outside the library in the archive: %r" % m.name)
            if m.isdir():
                os.makedirs(cible, exist_ok=True)
                continue
            total += m.size
            if total > plafond:
                raise Erreur("%s unpacks to more than index.json declares: stopped" % archive)
            os.makedirs(os.path.dirname(cible), exist_ok=True)
            src = t.extractfile(m)
            with open(cible, "xb") as f:
                shutil.copyfileobj(src, f, BLOC)
            os.chmod(cible, 0o644)
            os.utime(cible, (m.mtime, m.mtime))
    return total


# ------------------------------------------------------------------ merging the shared files

def fusion(base, dessus):
    for k, v in dessus.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            fusion(base[k], v)
        else:
            base[k] = v
    return base


def ecrire_json(chemin, valeur):
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin + ".tmp", "w", encoding="utf-8") as f:
        json.dump(valeur, f, ensure_ascii=False, indent=1)
    os.replace(chemin + ".tmp", chemin)


def lire_json(chemin, defaut=None):
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return defaut


# ------------------------------------------------------------------ installation

def etat_destination(index, rac, remplacer):
    """(to replace silently: our previous installation, conflicts: someone else's files at our place)."""
    ancien = lire_json(os.path.join(rac["cache"], MANIFESTE), {}) or {}
    a_nous = set(ancien.get("unites", []))
    nos, conflits = [], []
    for u in index["unites"]:
        c = placer(u, rac)
        if not os.path.lexists(c):
            continue
        (nos if u in a_nous else conflits).append(u)
    return nos, conflits


def deja_installee(index, rac):
    """The manifest of a finished installation of this same version whose items are all in place, otherwise None."""
    m = lire_json(os.path.join(rac["cache"], MANIFESTE), {}) or {}
    if m.get("en_cours") or m.get("format") != index.get("format") or not m.get("version") \
            or m.get("version") != index.get("version") or set(index["unites"]) - set(m.get("unites", [])):
        return None
    if any(not os.path.lexists(placer(u, rac)) for u in index["unites"]):
        return None
    return m


def installer(source, remplacer=False, garder=False, oui=False):
    rac = racines()
    donnees = R.chemin("paths.data")
    dire("Chimera ready-made library")
    dire("  from: %s" % source)
    index = ouvrir_index(source)
    if index.get("format") != FORMAT:
        raise Erreur("package format %s, this code reads format %s: update the code (git pull) or use a matching "
                     "package" % (index.get("format"), FORMAT))
    controler_index(index)
    n = index["nombres"]
    dire("  %d models, %d faces, %d hairstyles, %d parts; %.1f GB to download, %.1f GB once installed" % (
        n["modeles"], n["tetes"], n["coiffures"], n["pieces"], sum(a["taille"] for a in index["archives"]) / 1e9,
        index["taille_installee"] / 1e9))
    for r in RACINES:
        dire("  %-11s -> %s" % (r, rac[r]))
    fait = deja_installee(index, rac)
    if fait:
        dire("Already installed: version %s (installed %s), all %d items in place. Nothing to download. "
             "Open the workshop: bin/chimera" % (fait["version"], fait.get("date", "?"), len(index["unites"])))
        return
    nos, conflits = etat_destination(index, rac, remplacer)
    if conflits and not remplacer:
        exemples = "\n".join("    " + placer(u, rac) for u in conflits[:8])
        raise Erreur("%d item(s) of the library already exist here and were not put there by a previous installation "
                     "(your own models or another base body), for example:\n%s\nNothing was changed. To set them aside "
                     "(moved, not deleted) and install the library in their place, run again with --replace; "
                     "or install into another data folder (paths.data in settings.json)."
                     % (len(conflits), exemples))
    travail = os.path.join(donnees, TRAVAIL)
    besoin = index["taille_installee"] + (0 if not _distante(source) else sum(a["taille"] for a in index["archives"]))
    os.makedirs(travail, exist_ok=True)
    libre = shutil.disk_usage(travail).free
    if libre < besoin * 1.05:
        raise Erreur("not enough disk space in %s: %.1f GB free, %.1f GB needed" % (travail, libre / 1e9, besoin * 1.05 / 1e9))

    dire("Download and check (%d archives)" % len(index["archives"]))
    fichiers = [telecharger(source, a["chemin"], os.path.join(travail, "archives", os.path.basename(a["chemin"])),
                            a["taille"], a["sha256"]) for a in index["archives"]]

    dire("Unpack")
    for vieux_dossier in os.listdir(travail):    # left by an installation that stopped half-way
        if vieux_dossier.startswith("extrait-"):
            shutil.rmtree(os.path.join(travail, vieux_dossier), ignore_errors=True)
    extrait = os.path.join(travail, "extrait-%d" % os.getpid())
    os.makedirs(extrait)
    for a, f in zip(index["archives"], fichiers):
        deballer(f, extrait, a["taille_extraite"])
        dire("  %s" % a["chemin"])

    dire("Install")
    date = time.strftime("%Y%m%d-%H%M%S")
    mis_de_cote = os.path.join(travail, "remplaces-" + date)
    for u in conflits:
        dest = os.path.join(mis_de_cote, *u.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(placer(u, rac), dest)
    if conflits:
        dire("  %d existing item(s) set aside in %s" % (len(conflits), mis_de_cote))
    # from here on the items are ours, even if this run stops half-way: a new run replaces them without a word
    manifeste = os.path.join(rac["cache"], MANIFESTE)
    deja = lire_json(manifeste, {}) or {}
    ecrire_json(manifeste, dict(deja, en_cours=True, unites=sorted(set(deja.get("unites", [])) | set(index["unites"]))))
    for u in nos:   # our previous version of the same item
        c = placer(u, rac)
        if os.path.isdir(c) and not os.path.islink(c):
            shutil.rmtree(c)
        else:
            os.remove(c)
    lies = cibles_doubles(index)
    for u in index["unites"]:
        src = os.path.join(extrait, *u.split("/"))
        if not os.path.lexists(src):
            if u in lies:          # made below as a link to its identical copy
                continue
            raise Erreur("package incomplete: %s announced by index.json but absent from the archives" % u)
        dest = placer(u, rac)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(src, dest)
    for rel in index["partages"]:          # dictionaries shared with the user's own models: merged
        src = os.path.join(extrait, *rel.split("/"))
        dest = placer(rel, rac)
        neuf = lire_json(src, {})
        if rel == index.get("catalogue") and os.path.exists(dest):
            continue                       # rebuilt by the workshop from the records at start
        ecrire_json(dest, fusion(lire_json(dest, {}) or {}, neuf))
    for l in index["liens"]:               # face library entries of the cache (catalogue.preparer_biblio makes the same)
        lien, cible = placer(l["lien"], rac), placer(l["cible"], rac)
        if os.path.lexists(lien):
            if os.path.islink(lien):
                os.remove(lien)
            else:
                continue
        os.makedirs(os.path.dirname(lien), exist_ok=True)
        os.symlink(cible, lien)

    n_liens = 0
    for d in index.get("doubles", []):     # the same file under several names: hard links (as in the owner's cache)
        de = placer(d["de"], rac)
        dossier = os.path.dirname(d["de"])
        noms = [t + ".bin" for t in index["tetes_bin"] if t + ".bin" != os.path.basename(de)] if d.get("noms") == "*tetes" \
            else d.get("noms") or []
        cibles = [dossier + "/" + _nom_simple(n) for n in noms] + [_chemin_paquet(v) for v in d.get("vers", [])]
        for v in cibles:
            c = placer(v, rac)
            if os.path.lexists(c):
                continue
            os.makedirs(os.path.dirname(c), exist_ok=True)
            try:
                os.link(de, c)
            except OSError:
                shutil.copyfile(de, c)
            n_liens += 1
    dire("  %d files linked to their identical copy" % n_liens)
    dire("Adapt the records to this machine")
    vieux = {}
    for ancien, rel in index["empreintes"].items():
        vieux[ancien] = empreinte_fichier(placer(rel, rac))
    n_json = 0
    for u in index["unites"] + index["partages"]:
        c = placer(u, rac)
        pile = [c] if os.path.isfile(c) else [os.path.join(d, f) for d, _, fs in os.walk(c) for f in fs]
        for f in pile:
            if f.endswith(".json"):
                n_json += _adapter(f, rac, vieux)
    dire("  %d records adapted" % n_json)
    ecrire_json(manifeste, {
        "format": FORMAT, "version": index.get("version"), "source": source, "date": date, "unites": index["unites"]})
    shutil.rmtree(extrait)
    if not garder and _distante(source):
        shutil.rmtree(os.path.join(travail, "archives"), ignore_errors=True)
    dire("Done: %d models installed. Open the workshop: bin/chimera" % n["modeles"])
    if conflits:
        dire("Your items set aside are in %s." % mis_de_cote)


def cibles_doubles(index):
    """Package paths that the archives do not hold because they are a link to another file."""
    res = set()
    for d in index.get("doubles", []):
        dossier, nom = d["de"].rsplit("/", 1)
        noms = [t + ".bin" for t in index["tetes_bin"] if t + ".bin" != nom] if d.get("noms") == "*tetes" else d.get("noms") or []
        res.update(dossier + "/" + n for n in noms)
        res.update(d.get("vers", []))
    return res


def controler_index(index):
    """Every path index.json names stays inside a known root (the index is data from the network too)."""
    for rel in index["unites"] + index["partages"] + list(index["empreintes"].values()):
        _chemin_paquet(rel)
    for l in index["liens"]:
        _chemin_paquet(l["lien"]), _chemin_paquet(l["cible"])
    for d in index.get("doubles", []):
        _chemin_paquet(d["de"])
    for t in index.get("tetes_bin", []):
        _nom_simple(t + ".bin")
    for a in index["archives"]:
        if not a["chemin"].startswith("archives/") or "/" in a["chemin"][9:]:
            raise Erreur("unexpected archive name in index.json: %r" % a["chemin"])
        _nom_simple(a["chemin"][9:])


def _nom_simple(n):
    if not n or "/" in n or "\\" in n or n in (".", "..") or "\0" in n:
        raise Erreur("unsafe name in index.json: %r" % n)
    return n


def _chemin_paquet(rel):
    parts = rel.split("/")
    if not parts or parts[0] not in RACINES or any(_nom_simple(p) != p for p in parts[1:]):
        raise Erreur("unsafe path in index.json: %r" % rel)
    return rel


def _adapter(chemin, rac, empreintes):
    with open(chemin, encoding="utf-8") as f:
        t = f.read()
    # the records are JSON: a data folder whose name holds a quote or a backslash must stay valid JSON
    n = remplacer_racines(t, {r: json.dumps(v, ensure_ascii=False)[1:-1] for r, v in rac.items()})
    if empreintes:
        for a, b in empreintes.items():
            if a in n:
                n = n.replace(a, b)
    if n == t:
        return 0
    with open(chemin + ".tmp", "w", encoding="utf-8") as f:
        f.write(n)
    os.replace(chemin + ".tmp", chemin)
    return 1

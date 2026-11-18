"""Chimera's local server: the page, the cache files, assembly jobs, import, extraction and export
(the last three in entrees.py)."""
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commun as C  # noqa: E402
import catalogue as K  # noqa: E402
import combinaison as CB  # noqa: E402
import entrees as E  # noqa: E402

PAGE = os.path.join(C.ICI, "page")
ASSEMBLAGES = os.path.join(C.CACHE, "assemblages")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".glb": "model/gltf-binary", ".webp": "image/webp", ".png": "image/png", ".bin": "application/octet-stream"}
_file = threading.Lock()   # one assembly at a time
_ident = threading.Lock()
_actifs = set()            # assembly jobs queued or running in this server process
CATALOGUE = {}
CORPS_MAX = 1 << 20        # bytes: a selection fits in a few hundred
HOTES = ("127.0.0.1", "localhost", "[::1]")


def du_poste(entetes):
    """The request comes from Chimera's page, not from another site open in the browser: the server
    is named by its local address (not by a domain name leading to it) and, if the browser says where the
    request comes from, it is that same address."""
    h = entetes.get("Host") or ""
    if h.rsplit(":", 1)[0] not in HOTES:
        return False
    o = entetes.get("Origin")
    return o is None or o == "http://" + h


def connus(v):
    """The page's selection reduced to what the catalogue knows: an identifier becomes a path in the cache."""
    if not isinstance(v, dict) or not isinstance(v.get("pieces"), list):
        raise ValueError("unreadable selection")
    pieces = {p["id"] for p in CATALOGUE.get("pieces", [])}
    if any(i not in pieces for i in v["pieces"]):
        raise ValueError("unknown part")
    coif, tete = v.get("coiffure"), v.get("tete")
    if coif is not None and coif not in {c["id"] for c in CATALOGUE.get("coiffures", [])}:
        raise ValueError("unknown hairstyle")
    if tete is not None and tete not in {t["id"] for t in CATALOGUE.get("tetes", [])}:
        raise ValueError("unknown face")
    return v["pieces"], coif, tete


def accueil(manque):
    """Page shown instead of Chimera when it lacks what it needs to open (body, face, hairstyle): what is
    missing and what to do, without going through Chimera's page (it assumes a body and a catalogue)."""
    from html import escape
    items = "".join("<li>%s</li>" % escape(m) for m in manque)
    return ("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Chimera 3D Avatar Maker</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><link rel="icon" href="/page/chimera.svg">
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:46em;margin:48px auto;padding:0 16px;color:#efe9df;background:#17141f}
h1{font-size:1.5em}li{margin:.6em 0}code{background:#272233;padding:.1em .4em;border-radius:4px;color:#f4c95d}
</style></head>
<body><h1>Chimera does not have what it needs to open yet</h1>
<p>The server is running, but this is missing:</p><ul>%s</ul>
<p>The repository ships no model. The base body is built from a VRoid Studio character that you export
yourself (VRM 0.x), then the catalogue is prepared:</p>
<p><code>python3 -I outils/base_vroid/fabriquer.py MyCharacter.vrm</code><br>
<code>python3 -I outils/atelier/catalogue.py preparer</code></p>
<p>Then reload this page (no need to restart the server). The <code>INSTALLATION.md</code> guide gives the details;
<code>COMPATIBILITE.md</code> says which models are read. Settings in use: <code>python3 -I outils/reglages.py tout</code>.</p></body></html>""" % items).encode("utf-8")


def valider(sel):
    if not isinstance(sel, dict) or not isinstance(sel.get("pieces", []), list):
        raise ValueError("unreadable selection")
    tetes = {t["id"] for t in CATALOGUE["tetes"]}
    coif = {c["id"] for c in CATALOGUE["coiffures"]}
    pieces = {p["id"]: p for p in CATALOGUE["pieces"]}
    if sel.get("tete") not in tetes or sel.get("coiffure") not in coif:
        raise ValueError("unknown face or hairstyle")
    ids = [i for i in sel.get("pieces", []) if i in pieces]
    if len(ids) != len(sel.get("pieces", [])):
        raise ValueError("unknown part")
    ids.sort(key=lambda i: pieces[i]["rang"])
    t = sel.get("teinte")
    if t is not None:
        try:
            t = {k: float(t[k]) for k in ("r", "g", "b", "ref")}
        except TypeError:
            raise ValueError("unreadable tint")
    nom = sel.get("nom")
    if nom is not None and not isinstance(nom, str):
        raise ValueError("unreadable name")
    return {"tete": sel["tete"], "coiffure": sel["coiffure"], "pieces": ids, "teinte": t, "nom": E.nettoyer_nom(nom or "")}


def assembler(sel):
    os.makedirs(ASSEMBLAGES, exist_ok=True)
    with _ident:               # two requests in the same second: each its own folder
        base, n = time.strftime("%Y%m%d-%H%M%S"), 1
        ident = base
        while os.path.exists(os.path.join(ASSEMBLAGES, ident)):
            n += 1
            ident = "%s-%d" % (base, n)
        d = os.path.join(ASSEMBLAGES, ident)
        os.makedirs(d)
    C.ecrire_json(os.path.join(d, "selection.json"), sel)
    C.ecrire_json(os.path.join(d, "etat.json"), {"etape": "attente", "fini": False})

    def travail():
        # whatever happens (Blender killed, launcher missing), the job ends in a readable error state
        raison = None
        try:
            with _file:
                cmd = C.lanceur() + [
                    "bash", C.BLENDER_SH, "-b", "--factory-startup", "--python", os.path.join(C.ICI, "assembler_blender.py"), "--", "--dossier", d]
                with open(os.path.join(d, "journal.txt"), "wb") as j:
                    r = subprocess.run(cmd, stdout=j, stderr=subprocess.STDOUT)
                raison = "Blender stopped (code %d), see journal.txt" % r.returncode
        except Exception as x:
            raison = "could not start Blender: %s: %s" % (type(x).__name__, x)
        finally:
            try:
                e = C.lire_json(os.path.join(d, "etat.json"), {})
                if not e.get("fini"):
                    e.update(fini=True, ok=False, etape="erreur", erreur=raison or "assembly interrupted, see journal.txt")
                    C.ecrire_json(os.path.join(d, "etat.json"), e)
            finally:
                _actifs.discard(ident)

    _actifs.add(ident)
    threading.Thread(target=travail, daemon=True).start()
    return ident


class Gestion(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def envoyer(self, code, corps, ctype="application/json", cache=None, marque=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(corps)))
        self.send_header("Cache-Control", cache or "no-store")
        if marque:
            self.send_header("ETag", marque)
        self.end_headers()
        self.wfile.write(corps)

    def json(self, v, code=200):
        self.envoyer(code, json.dumps(v, ensure_ascii=False).encode("utf-8"))

    def fichier(self, chemin, cache=None, revalider=False):
        """`revalider`: the browser keeps its copy but asks every time whether it changed
        (a part repaired in the cache must show at once, without waiting an hour)."""
        try:
            if revalider:
                st = os.stat(chemin)
                marque = '"%x-%x"' % (st.st_mtime_ns, st.st_size)
                if self.headers.get("If-None-Match") == marque:
                    self.send_response(304)
                    self.send_header("ETag", marque)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", "0")
                    return self.end_headers()
            with open(chemin, "rb") as f:
                b = f.read()
        except OSError:
            return self.envoyer(404, b"not found", "text/plain")
        self.envoyer(200, b, TYPES.get(os.path.splitext(chemin)[1], "application/octet-stream"),
                     "no-cache" if revalider else cache, marque if revalider else None)

    def refuse(self):
        """True (and answered) if the request does not come from Chimera's page."""
        if du_poste(self.headers):
            return False
        self.envoyer(403, b"rejected: request from somewhere other than Chimera's page", "text/plain")
        return True

    def do_GET(self):
        if self.refuse():
            return
        u = urllib.parse.urlparse(self.path)
        p = urllib.parse.unquote(u.path)
        if p == "/":   # without a body, a face or a hairstyle Chimera's page does not open: say what to do
            manque = K.a_faire(K.construire(ecrire=False))
            if manque:
                return self.envoyer(200, accueil(manque), TYPES[".html"])
            return self.fichier(os.path.join(PAGE, "index.html"))
        if p.startswith("/page/"):
            return self.fichier(self.sous(PAGE, p[6:]))
        if p.startswith("/data/"):
            return self.fichier(self.sous(C.CACHE, p[6:]), revalider=True)
        if p == "/api/catalogue":   # read again from the records at each load: a preparation in progress shows in it
            global CATALOGUE
            CATALOGUE = K.construire()
            return self.json(CATALOGUE)
        if p == "/api/formats":
            return self.json(E.formats())
        if p == "/api/modeles":
            return self.json(E.etat_modeles())
        if p == "/api/assemblages":
            return self.json(E.assemblages())
        if p == "/api/extraction":   # the running extraction, from the server: the page asks on load, in any tab
            return self.json(E.extraction_en_cours())
        m = re.fullmatch(r"/api/assemblages/([0-9-]{15,20})/(pmx|vrm|glb)", p)
        if m:
            return self.telecharger(m.group(1), m.group(2))
        m = re.fullmatch(r"/api/travaux/(x?[0-9-]{15,20})", p)
        if m:
            inconnu = {"etape": "inconnu", "fini": True, "ok": False, "erreur": "Unknown job."}
            if m.group(1).startswith("x"):
                return self.json(E.lire_travail(m.group(1)) or inconnu)
            e = C.lire_json(os.path.join(ASSEMBLAGES, m.group(1), "etat.json"), inconnu)
            if not e.get("fini") and m.group(1) not in _actifs:
                # left unfinished by an earlier server (stopped mid-job): nothing will finish it any more
                e = dict(e, fini=True, ok=False, etape="erreur", erreur="assembly interrupted (the server stopped during the job)")
            return self.json(e)
        self.envoyer(404, b"not found", "text/plain")

    def telecharger(self, ident, fmt):
        if fmt == "glb":
            return self.json({"erreur": "No separate GLB export: the VRM is a binary glTF file, just rename it "
                                        "to .glb."}, 404)
        try:
            chemin, ascii_, utf8, ctype, temporaire = E.export(ident, fmt)
        except E.Refus as e:
            return self.json({"erreur": str(e)}, e.code)
        try:
            taille = os.path.getsize(chemin)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(taille))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Disposition", "attachment; filename=\"%s\"; filename*=UTF-8''%s"
                             % (ascii_, urllib.parse.quote(utf8, safe="")))
            self.end_headers()
            with open(chemin, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    self.wfile.write(b)
        finally:
            if temporaire:
                try:
                    os.remove(chemin)
                except OSError:
                    pass

    def sous(self, racine, rel):
        """Path under racine, with no way out through ".." (the page stays in the cache)."""
        c = os.path.realpath(os.path.join(racine, rel))
        return c if c.startswith(os.path.realpath(racine) + os.sep) else os.path.join(racine, "__interdit__")

    def lire_corps(self, maxi=CORPS_MAX):
        n = int(self.headers.get("Content-Length") or 0)
        if n < 0 or n > maxi:
            raise ValueError("request body too large")
        return self.rfile.read(n)

    def do_POST(self):
        if self.refuse():
            return
        p = urllib.parse.urlparse(self.path).path
        # Another site cannot send declared JSON without the browser asking for permission first.
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            return self.json({"erreur": "Content-Type application/json expected"}, 415)
        try:
            v = json.loads(self.lire_corps() or b"{}")
            if p == "/api/combinaison":
                r = CB.combiner(*connus(v))
                b64 = lambda t: {k: base64.b64encode(b).decode() for k, b in t.items()}  # noqa: E731
                return self.json({"ordre": r["ordre"], "signal": r["signal"], "notes": r["notes"], "retirees": r["retirees"],
                                  "decalages": b64(r["decalages"]), "masques": b64(r["masques"]),
                                  "coiffure": base64.b64encode(r["coiffure"]).decode() if r["coiffure"] else None,
                                  "corps": base64.b64encode(r["corps"]).decode() if r.get("corps") else None})
            if p == "/api/assembler":
                return self.json({"id": assembler(valider(v))})
            if p == "/api/extraire":
                return self.json({"id": E.lancer_extraction(*E.demande_extraction(v))})
            if p == "/api/extraction/annuler":
                return self.json({"id": E.annuler_extraction(), "annulation": True})
            if p == "/api/ouvrir":
                d = os.path.join(ASSEMBLAGES, re.sub(r"[^0-9-]", "", str(v["id"])))
                if not os.path.isdir(d):
                    raise FileNotFoundError("unknown assembly")
                subprocess.Popen(["xdg-open", d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return self.json({"ok": True, "dossier": d})
        except E.Refus as e:
            return self.json({"erreur": str(e)}, e.code)
        except (KeyError, ValueError, FileNotFoundError) as e:
            return self.json({"erreur": str(e)}, 400)
        except Exception as e:   # unexpected data: tell the page rather than cut the connection
            return self.json({"erreur": "%s: %s" % (type(e).__name__, e)}, 500)
        self.envoyer(404, b"not found", "text/plain")

    def do_PUT(self):
        if self.refuse():
            return
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/importer":
            return self.importer(urllib.parse.parse_qs(u.query).get("nom", [""])[0])
        m = re.fullmatch(r"/api/vignette/(tetes|coiffures|pieces)/([^/]+)", urllib.parse.unquote(self.path))
        if not m:
            return self.envoyer(404, b"not found", "text/plain")
        d = self.sous(C.CACHE, os.path.join(m.group(1), m.group(2)))
        try:
            b = self.lire_corps(2_000_000)
        except ValueError:
            return self.json({"erreur": "rejected"}, 400)
        if os.path.isdir(d) and b[:4] == b"\x89PNG" and len(b) < 2_000_000:
            C.ecrire_octets(os.path.join(d, "vignette.png"), b)
            return self.json({"ok": True})
        self.json({"erreur": "rejected"}, 400)


    def importer(self, nom):
        """PUT /api/importer?nom=<file>: the raw file as body, declared application/octet-stream (another
        site cannot send this type without the browser asking for permission first)."""
        try:
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/octet-stream":
                raise E.Refus("Content-Type application/octet-stream expected.", 415)
            ok, raison = E.import_possible()
            if not ok:
                raise E.Refus(raison, 409)
            if os.path.splitext(nom)[1].lower() not in (".vrm", ".zip"):
                raise E.Refus("Unsupported format: a .vrm (VRoid Studio or VRoid Hub) or a .zip containing a .pmx.", 415)
            E.identifiant(nom)
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise E.Refus("Unreadable file size.")
            recu = E.recevoir(self.rfile, n)
        except E.Refus as e:
            self.close_connection = True       # the body was not read: do not reuse the connection
            return self.json({"erreur": str(e)}, e.code)
        try:
            fiches, ecartes = E.importer_lot(recu, nom)
            return self.json({"modele": fiches[0], "modeles": fiches, "ecartes": ecartes})
        except E.Refus as e:
            return self.json({"erreur": str(e)}, e.code)
        except subprocess.TimeoutExpired:
            return self.json({"erreur": "The conversion took too long: given up."}, 422)
        except Exception as e:   # noqa: BLE001
            return self.json({"erreur": "Import failed: %s: %s" % (type(e).__name__, e)}, 500)
        finally:
            if os.path.exists(recu):
                os.remove(recu)


def main():
    global CATALOGUE
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8767
    CATALOGUE = K.construire()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Gestion)
    n = CATALOGUE["nombres"]
    print("Chimera: http://127.0.0.1:%d  (%d kept, %d rejected)" % (port, n["retenues"], n["refusees"]), flush=True)
    for m in K.a_faire(CATALOGUE):
        print("To do: " + m, flush=True)
    if "--ouvrir" in sys.argv:
        subprocess.Popen(["xdg-open", "http://127.0.0.1:%d" % port], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    srv.serve_forever()


if __name__ == "__main__":
    main()

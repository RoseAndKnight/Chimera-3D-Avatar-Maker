"""Chimera's local server: the page, the cache files, assembly jobs, import, extraction and export
(the last three in inputs.py)."""
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
import common as C  # noqa: E402
import catalog as K  # noqa: E402
import outfit as CB  # noqa: E402
import inputs as E  # noqa: E402

PAGE = os.path.join(C.HERE, "page")
ASSEMBLIES = os.path.join(C.CACHE, "assemblies")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".glb": "model/gltf-binary", ".webp": "image/webp", ".png": "image/png", ".bin": "application/octet-stream"}
_file = threading.Lock()   # one assembly at a time
_id = threading.Lock()
_actives = set()            # assembly jobs queued or running in this server process
CATALOGUE = {}
BODY_MAX = 1 << 20        # bytes: a selection fits in a few hundred
HOSTS = ("127.0.0.1", "localhost", "[::1]")


def of_post(headers):
    """The request comes from Chimera's page, not from another site open in the browser: the server
    is named by its local address (not by a domain name leading to it) and, if the browser says where the
    request comes from, it is that same address."""
    h = headers.get("Host") or ""
    if h.rsplit(":", 1)[0] not in HOSTS:
        return False
    o = headers.get("Origin")
    return o is None or o == "http://" + h


def known(v):
    """The page's selection reduced to what the catalogue knows: an identifier becomes a path in the cache."""
    if not isinstance(v, dict) or not isinstance(v.get("parts"), list):
        raise ValueError("unreadable selection")
    parts = {p["id"] for p in CATALOGUE.get("parts", [])}
    if any(i not in parts for i in v["parts"]):
        raise ValueError("unknown part")
    hairdo, head = v.get("hairstyle"), v.get("head")
    if hairdo is not None and hairdo not in {c["id"] for c in CATALOGUE.get("hairstyles", [])}:
        raise ValueError("unknown hairstyle")
    if head is not None and head not in {t["id"] for t in CATALOGUE.get("heads", [])}:
        raise ValueError("unknown face")
    return v["parts"], hairdo, head


def home(missing):
    """Page shown instead of Chimera when it lacks what it needs to open (body, face, hairstyle): what is
    missing and what to do, without going through Chimera's page (it assumes a body and a catalogue)."""
    from html import escape
    items = "".join("<li>%s</li>" % escape(m) for m in missing)
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
<p><code>python3 -I tools/vroid_base/build.py MyCharacter.vrm</code><br>
<code>python3 -I tools/workshop/catalog.py prepare</code></p>
<p>Then reload this page (no need to restart the server). The <code>INSTALLATION.md</code> guide gives the details;
<code>COMPATIBILITY.md</code> says which models are read. Settings in use: <code>python3 -I tools/settings.py all</code>.</p></body></html>""" % items).encode("utf-8")


def validate(sel):
    if not isinstance(sel, dict) or not isinstance(sel.get("parts", []), list):
        raise ValueError("unreadable selection")
    heads = {t["id"] for t in CATALOGUE["heads"]}
    hairdo = {c["id"] for c in CATALOGUE["hairstyles"]}
    parts = {p["id"]: p for p in CATALOGUE["parts"]}
    if sel.get("head") not in heads or sel.get("hairstyle") not in hairdo:
        raise ValueError("unknown face or hairstyle")
    ids = [i for i in sel.get("parts", []) if i in parts]
    if len(ids) != len(sel.get("parts", [])):
        raise ValueError("unknown part")
    ids.sort(key=lambda i: parts[i]["rank"])
    t = sel.get("tint")
    if t is not None:
        try:
            t = {k: float(t[k]) for k in ("r", "g", "b", "ref")}
        except TypeError:
            raise ValueError("unreadable tint")
    name = sel.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("unreadable name")
    return {"head": sel["head"], "hairstyle": sel["hairstyle"], "parts": ids, "tint": t, "name": E.clean_name(name or "")}


def assemble(sel):
    os.makedirs(ASSEMBLIES, exist_ok=True)
    with _id:               # two requests in the same second: each its own folder
        base, n = time.strftime("%Y%m%d-%H%M%S"), 1
        ident = base
        while os.path.exists(os.path.join(ASSEMBLIES, ident)):
            n += 1
            ident = "%s-%d" % (base, n)
        d = os.path.join(ASSEMBLIES, ident)
        os.makedirs(d)
    C.write_json(os.path.join(d, "selection.json"), sel)
    C.write_json(os.path.join(d, "state.json"), {"step": "waiting", "done": False})

    def work():
        # whatever happens (Blender killed, launcher missing), the job ends in a readable error state
        reason = None
        try:
            with _file:
                cmd = C.launcher() + [
                    "bash", C.BLENDER_SH, "-b", "--factory-startup", "--python", os.path.join(C.HERE, "assemble_blender.py"), "--", "--dossier", d]
                with open(os.path.join(d, "log.txt"), "wb") as j:
                    r = subprocess.run(cmd, stdout=j, stderr=subprocess.STDOUT)
                reason = "Blender stopped (code %d), see log.txt" % r.returncode
        except Exception as x:
            reason = "could not start Blender: %s: %s" % (type(x).__name__, x)
        finally:
            try:
                e = C.read_json(os.path.join(d, "state.json"), {})
                if not e.get("done"):
                    e.update(done=True, ok=False, step="error", error=reason or "assembly interrupted, see log.txt")
                    C.write_json(os.path.join(d, "state.json"), e)
            finally:
                _actives.discard(ident)

    _actives.add(ident)
    threading.Thread(target=work, daemon=True).start()
    return ident


class Management(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json", cache=None, mark=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache or "no-store")
        if mark:
            self.send_header("ETag", mark)
        self.end_headers()
        self.wfile.write(body)

    def json(self, v, code=200):
        """Answers with a JSON value."""
        self.send(code, json.dumps(v, ensure_ascii=False).encode("utf-8"))

    def file(self, path, cache=None, revalidate=False):
        """`revalidate`: the browser keeps its copy but asks every time whether it changed
        (a part repaired in the cache must show at once, without waiting an hour)."""
        try:
            if revalidate:
                st = os.stat(path)
                mark = '"%x-%x"' % (st.st_mtime_ns, st.st_size)
                if self.headers.get("If-None-Match") == mark:
                    self.send_response(304)
                    self.send_header("ETag", mark)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", "0")
                    return self.end_headers()
            with open(path, "rb") as f:
                b = f.read()
        except OSError:
            return self.send(404, b"not found", "text/plain")
        self.send(200, b, TYPES.get(os.path.splitext(path)[1], "application/octet-stream"),
                     "no-cache" if revalidate else cache, mark if revalidate else None)

    def refused(self):
        """True (and answered) if the request does not come from Chimera's page."""
        if of_post(self.headers):
            return False
        self.send(403, b"rejected: request from somewhere other than Chimera's page", "text/plain")
        return True

    def do_GET(self):
        if self.refused():
            return
        u = urllib.parse.urlparse(self.path)
        p = urllib.parse.unquote(u.path)
        if p == "/":   # without a body, a face or a hairstyle Chimera's page does not open: say what to do
            missing = K.a_make(K.build(write=False))
            if missing:
                return self.send(200, home(missing), TYPES[".html"])
            return self.file(os.path.join(PAGE, "index.html"))
        if p.startswith("/page/"):
            return self.file(self.under(PAGE, p[6:]))
        if p.startswith("/data/"):
            return self.file(self.under(C.CACHE, p[6:]), revalidate=True)
        if p == "/api/catalog":   # read again from the records at each load: a preparation in progress shows in it
            global CATALOGUE
            CATALOGUE = K.build()
            return self.json(CATALOGUE)
        if p == "/api/formats":
            return self.json(E.formats())
        if p == "/api/models":
            return self.json(E.state_models())
        if p == "/api/assemblies":
            return self.json(E.assemblies())
        if p == "/api/extraction":   # the running extraction, from the server: the page asks on load, in any tab
            return self.json(E.extraction_in_progress())
        m = re.fullmatch(r"/api/assemblies/([0-9-]{15,20})/(pmx|vrm|glb)", p)
        if m:
            return self.download(m.group(1), m.group(2))
        m = re.fullmatch(r"/api/jobs/(x?[0-9-]{15,20})", p)
        if m:
            unknown = {"step": "unknown", "done": True, "ok": False, "error": "Unknown job."}
            if m.group(1).startswith("x"):
                return self.json(E.read_work(m.group(1)) or unknown)
            e = C.read_json(os.path.join(ASSEMBLIES, m.group(1), "state.json"), unknown)
            if not e.get("done") and m.group(1) not in _actives:
                # left unfinished by an earlier server (stopped mid-job): nothing will finish it any more
                e = dict(e, done=True, ok=False, step="error", error="assembly interrupted (the server stopped during the job)")
            return self.json(e)
        self.send(404, b"not found", "text/plain")

    def download(self, ident, fmt):
        if fmt == "glb":
            return self.json({"error": "No separate GLB export: the VRM is a binary glTF file, just rename it "
                                        "to .glb."}, 404)
        try:
            path, ascii_, utf8, ctype, temporary = E.export(ident, fmt)
        except E.Refusals as e:
            return self.json({"error": str(e)}, e.code)
        try:
            size = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Disposition", "attachment; filename=\"%s\"; filename*=UTF-8''%s"
                             % (ascii_, urllib.parse.quote(utf8, safe="")))
            self.end_headers()
            with open(path, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    self.wfile.write(b)
        finally:
            if temporary:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def under(self, root, rel):
        """Path under racine, with no way out through ".." (the page stays in the cache)."""
        c = os.path.realpath(os.path.join(root, rel))
        return c if c.startswith(os.path.realpath(root) + os.sep) else os.path.join(root, "__interdit__")

    def read_body(self, max_=BODY_MAX):
        n = int(self.headers.get("Content-Length") or 0)
        if n < 0 or n > max_:
            raise ValueError("request body too large")
        return self.rfile.read(n)

    def do_POST(self):
        if self.refused():
            return
        p = urllib.parse.urlparse(self.path).path
        # Another site cannot send declared JSON without the browser asking for permission first.
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            return self.json({"error": "Content-Type application/json expected"}, 415)
        try:
            v = json.loads(self.read_body() or b"{}")
            if p == "/api/combination":
                r = CB.combine(*known(v))
                b64 = lambda t: {k: base64.b64encode(b).decode() for k, b in t.items()}  # noqa: E731
                return self.json({"order": r["order"], "signal": r["signal"], "notes": r["notes"], "removed": r["removed"],
                                  "offsets": b64(r["offsets"]), "masks": b64(r["masks"]),
                                  "hairstyle": base64.b64encode(r["hairstyle"]).decode() if r["hairstyle"] else None,
                                  "body": base64.b64encode(r["body"]).decode() if r.get("body") else None})
            if p == "/api/assemble":
                return self.json({"id": assemble(validate(v))})
            if p == "/api/extract":
                return self.json({"id": E.launch_extraction(*E.request_extraction(v))})
            if p == "/api/extraction/cancel":
                return self.json({"id": E.cancel_extraction(), "cancellation": True})
            if p == "/api/open":
                d = os.path.join(ASSEMBLIES, re.sub(r"[^0-9-]", "", str(v["id"])))
                if not os.path.isdir(d):
                    raise FileNotFoundError("unknown assembly")
                subprocess.Popen(["xdg-open", d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return self.json({"ok": True, "folder": d})
        except E.Refusals as e:
            return self.json({"error": str(e)}, e.code)
        except (KeyError, ValueError, FileNotFoundError) as e:
            return self.json({"error": str(e)}, 400)
        except Exception as e:   # unexpected data: tell the page rather than cut the connection
            return self.json({"error": "%s: %s" % (type(e).__name__, e)}, 500)
        self.send(404, b"not found", "text/plain")

    def do_PUT(self):
        if self.refused():
            return
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/import":
            return self.importer(urllib.parse.parse_qs(u.query).get("name", [""])[0])
        m = re.fullmatch(r"/api/thumbnail/(heads|hairstyles|parts)/([^/]+)", urllib.parse.unquote(self.path))
        if not m:
            return self.send(404, b"not found", "text/plain")
        d = self.under(C.CACHE, os.path.join(m.group(1), m.group(2)))
        try:
            b = self.read_body(2_000_000)
        except ValueError:
            return self.json({"error": "rejected"}, 400)
        if os.path.isdir(d) and b[:4] == b"\x89PNG" and len(b) < 2_000_000:
            C.write_bytes(os.path.join(d, "thumbnail.png"), b)
            return self.json({"ok": True})
        self.json({"error": "rejected"}, 400)


    def importer(self, name):
        """PUT /api/import?name=<file>: the raw file as body, declared application/octet-stream (another
        site cannot send this type without the browser asking for permission first)."""
        try:
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/octet-stream":
                raise E.Refusals("Content-Type application/octet-stream expected.", 415)
            ok, reason = E.import_possible()
            if not ok:
                raise E.Refusals(reason, 409)
            if os.path.splitext(name)[1].lower() not in (".vrm", ".zip"):
                raise E.Refusals("Unsupported format: a .vrm (VRoid Studio or VRoid Hub) or a .zip containing a .pmx.", 415)
            E.identifier(name)
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise E.Refusals("Unreadable file size.")
            received = E.receive(self.rfile, n)
        except E.Refusals as e:
            self.close_connection = True       # the body was not read: do not reuse the connection
            return self.json({"error": str(e)}, e.code)
        try:
            cards, discarded = E.importer_lot(received, name)
            return self.json({"model": cards[0], "models": cards, "discarded": discarded})
        except E.Refusals as e:
            return self.json({"error": str(e)}, e.code)
        except subprocess.TimeoutExpired:
            return self.json({"error": "The conversion took too long: given up."}, 422)
        except Exception as e:   # noqa: BLE001
            return self.json({"error": "Import failed: %s: %s" % (type(e).__name__, e)}, 500)
        finally:
            if os.path.exists(received):
                os.remove(received)


def main():
    global CATALOGUE
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8767
    CATALOGUE = K.build()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Management)
    n = CATALOGUE["counts"]
    print("Chimera: http://127.0.0.1:%d  (%d kept, %d rejected)" % (port, n["kept_count"], n["refused"]), flush=True)
    for m in K.a_make(CATALOGUE):
        print("To do: " + m, flush=True)
    if "--open" in sys.argv:
        subprocess.Popen(["xdg-open", "http://127.0.0.1:%d" % port], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    srv.serve_forever()


if __name__ == "__main__":
    main()

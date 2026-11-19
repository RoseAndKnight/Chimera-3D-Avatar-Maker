"""Import, extraction and export of the workshop (inputs.py) and their routes: guards, booby-trapped archives,
rejected formats, list of formats true to the game descriptions. Nothing is converted or started (no Blender); the
cancellation test runs a small stand-in tool."""
import http.client
import io
import json
import os
import shutil
import stat
import struct
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import inputs as E  # noqa: E402
import server as S  # noqa: E402
import games as games  # noqa: E402  (tools/ is on the path through common)
import common as C  # noqa: E402


def lu(b):
    """A server answer (format 2 on the wire) in the code's names."""
    return json.loads(b)


class GameFalse:
    """Test active game: `models` "*" (importing works) or a declared list (no importing)."""

    def __init__(self, root, star=True):
        self.name, self.title, self.root = "attempt", "Trial", root
        self.d = {"format": "vrm", "models": "*" if star else {"A": "A/A.pmx"}}

    def value(self, the_key, default=None):
        return self.d.get(the_key, default)

    def path(self, source):
        if source not in ("mmd_extracted", "vrm"):
            raise KeyError(source)
        return os.path.join(self.root, source)

    def models(self):
        r = os.path.join(self.root, "mmd_extracted")
        if self.d["models"] != "*":
            return dict(self.d["models"])
        return {n: "%s/%s.pmx" % (n, n) for n in sorted(os.listdir(r))} if os.path.isdir(r) else {}


def one_zip(entries, symlink=None):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for name, bytes_ in entries.items():
            z.writestr(name, bytes_)
        if symlink:
            i = zipfile.ZipInfo(symlink)
            i.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(i, "/etc/passwd")
    return b.getvalue()


def one_glb(extensions):
    j = json.dumps({"asset": {"version": "2.0"}, "extensions": extensions}).encode()
    j += b" " * (-len(j) % 4)
    return b"glTF" + struct.pack("<II", 2, 12 + 8 + len(j)) + struct.pack("<I4s", len(j), b"JSON") + j


def wait(condition, delay=15):
    import time
    end = time.time() + delay
    while not condition():
        if time.time() > end:
            raise AssertionError("condition not met within %d s" % delay)
        time.sleep(0.1)


class Tray(unittest.TestCase):
    """Test folders in place of the cache and of the active game's sources."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.guard = {k: getattr(E, k) for k in ("WAITING", "ASSEMBLIES", "EXTRACTIONS", "J")}
        E.WAITING = os.path.join(self.tmp, "waiting")
        E.ASSEMBLIES = os.path.join(self.tmp, "assemblies")
        E.EXTRACTIONS = os.path.join(self.tmp, "extractions")
        E.J = GameFalse(self.tmp)

    def tearDown(self):
        for k, v in self.guard.items():
            setattr(E, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def deposit(self, bytes_):
        c = os.path.join(self.tmp, "received")
        with open(c, "wb") as f:
            f.write(bytes_)
        return c


class TestArchivesTrapped(Tray):
    def refusals(self, bytes_, expected):
        with self.assertRaises(E.Refusals) as r:
            E.unfold_zip(self.deposit(bytes_), os.path.join(self.tmp, "unfolded"))
        self.assertIn(expected, str(r.exception))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil")))

    def test_path_which_sort(self):
        self.refusals(one_zip({"m.pmx": b"x", "../evil/x.png": b"x"}), "escapes its folder")
        self.refusals(one_zip({"m.pmx": b"x", "/tmp/evil.png": b"x"}), "escapes its folder")
        self.refusals(one_zip({"m.pmx": b"x", "C:/evil.png": b"x"}), "escapes its folder")

    def test_link(self):
        self.refusals(one_zip({"m.pmx": b"x"}, symlink="tex/lien.png"), "link")

    def test_bulge(self):
        self.refusals(one_zip({"m.pmx": b"x", "tex/zero.png": bytes(20 << 20)}), "abnormal compression")

    def test_pmx_alone_and_unique(self):
        self.refusals(one_zip({"a.png": b"x"}), "no .pmx file")
        self.refusals(b"PK not an archive", "not a readable .zip")

    def test_archive_sound(self):
        rel = E.unfold_zip(self.deposit(one_zip({"Character/Character.pmx": b"x", "Character/tex/a.png": b"y"})),
                            os.path.join(self.tmp, "unfolded"))
        self.assertEqual(rel, ["Character/Character.pmx"])
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "unfolded", "Character", "tex", "a.png")))


class TestLot(Tray):
    def test_several_pmx_renders(self):
        rel = E.unfold_zip(self.deposit(one_zip({"A/a.pmx": b"x", "B/b.pmx": b"x", "B/weapon.pmx": b"x"})),
                            os.path.join(self.tmp, "unfolded"))
        self.assertEqual(rel, ["A/a.pmx", "B/b.pmx", "B/weapon.pmx"])

    def test_names_without_flag_utf8(self):
        # a UTF-8 name written without declaring it (what `zip` does on Linux): same bytes, flag missing
        bytes_ = one_zip({"A/a.pmx": b"x", "A/tex/XYZ.tga": b"x"}).replace(b"XYZ.tga", "颜.tga".encode("utf-8"))
        E.unfold_zip(self.deposit(bytes_), os.path.join(self.tmp, "unfolded"))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "unfolded", "A", "tex", "颜.tga")))

    def test_names_of_lot(self):
        good = [("Anby/安比.pmx", None, None), ("Batch/one.pmx", None, None), ("Batch/two.pmx", None, None),
                ("安/安.pmx", None, None), ("Anby2/x.pmx", None, None)]
        names = E._ids_of_lot("archive", good)
        self.assertEqual(names, ["Anby", "Batch-one", "Batch-two", "archive_4", "Anby2"])
        self.assertEqual(len(set(names)), len(names))


class TestImportRefused(Tray):
    def test_format_refused(self):
        with self.assertRaises(E.Refusals) as r:
            E.importer(self.deposit(b"x"), "character.fbx")
        self.assertEqual(r.exception.code, 415)

    def test_game_a_items_declared(self):
        E.J = GameFalse(self.tmp, star=False)
        with self.assertRaises(E.Refusals) as r:
            E.importer(self.deposit(b"x"), "character.vrm")
        self.assertEqual(r.exception.code, 409)

    def test_vrm_1_and_false_vrm(self):
        for bytes_, expected in ((one_glb({"VRMC_vrm": {"specVersion": "2.0"}}), "other than 0.x and 1.x"),
                                (one_glb({}), "without the VRM extension"), (b"pas un glb" * 4, "not a VRM")):
            with self.assertRaises(E.Refusals) as r:
                E.header_vrm(self.deposit(bytes_))
            self.assertIn(expected, str(r.exception))
        self.assertEqual(E.header_vrm(self.deposit(one_glb({"VRM": {"meta": {"title": "Elle"}}}))), "Elle")
        self.assertEqual(E.header_vrm(self.deposit(one_glb({"VRMC_vrm": {"specVersion": "1.0", "meta": {"name": "Lui"}}}))), "Lui")

    def test_already_ingested_and_received_efface(self):
        os.makedirs(os.path.join(self.tmp, "mmd_extracted", "Character"))
        c = self.deposit(b"x")
        with self.assertRaises(E.Refusals) as r:
            E.importer(c, "Character.vrm")
        self.assertEqual(r.exception.code, 409)
        self.assertFalse(os.path.exists(c))

    def test_names(self):
        self.assertEqual(E.identifier("C:\\x\\Élise été.vrm"), "Elise_ete")   # fr-ok: accented names are the point of the test
        self.assertEqual(E.identifier("../../etc/passwd.zip"), "passwd")
        with self.assertRaises(E.Refusals):
            E.identifier("日本.vrm")
        self.assertEqual(E.clean_name("  Rose <script>/../  "), "Rose script")
        self.assertEqual(len(E.clean_name("a" * 200)), 60)
        self.assertEqual(E.name_download("Élise \"x\"", "p"), ("Elise_x", "Élise x"))   # fr-ok: accented names are the point of the test


class TestFormatsFaithful(unittest.TestCase):
    """What /api/formats says comes from the game.json files: an announced skeleton goes all the way to the catalogue."""

    def test_skeletons_and_games(self):
        f = E.formats()
        announcements = {s["id"]: s for s in f["skeletons"]}
        self.assertNotIn("non_taken", f)        # only the skeletons the workshop reads are listed
        for name in games.all_():
            if name == "homemade":
                continue
            d = games.load(name).d
            if "models" in d and "base_head" in d:
                self.assertIn(name, announcements)
                self.assertEqual(announcements[name]["state"], d["compatibility"]["level"])
            else:
                self.assertNotIn(name, announcements)
        self.assertEqual({s["id"] for s in f["output"]}, {"pmx", "vrm"})
        self.assertEqual({e["id"] for e in f["input"]}, {"vrm", "pmx"})
        self.assertEqual(f["active_game"]["id"], E.J.name)


class TestRoutes(Tray):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), S.Management)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def request(self, method, path, body=None, **headers):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(method, path, body=body, headers=headers)
        r = c.getresponse()
        b = r.read()
        c.close()
        return r.status, b, r

    def test_import_kept(self):
        o = {"Content-Type": "application/octet-stream"}
        self.assertEqual(self.request("PUT", "/api/import?name=a.vrm", b"x", Origin="https://evil.example", **o)[0], 403)
        self.assertEqual(self.request("PUT", "/api/import?name=a.vrm", b"x", **{"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("PUT", "/api/import?name=a.exe", b"x", **o)[0], 415)
        s, b, _ = self.request("PUT", "/api/import?name=a.vrm", b"pas un vrm", **o)
        self.assertEqual(s, 400)
        self.assertIn("not a VRM", lu(b)["error"])
        E.J = GameFalse(self.tmp, star=False)
        self.assertEqual(self.request("PUT", "/api/import?name=a.vrm", b"x", **o)[0], 409)

    def test_extract_kept(self):
        j = {"Content-Type": "application/json"}
        self.assertEqual(self.request("POST", "/api/extract", b'{"all": true}', **{"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/extract", b'{"models": ["../x"]}', **j)[0], 400)
        self.assertEqual(self.request("POST", "/api/extract", b'{"all": true}', **j)[0], 400)   # no model
        os.makedirs(os.path.join(self.tmp, "mmd_extracted", "A"))
        self.assertEqual(self.request("POST", "/api/extract", b'{"all": true, "what": ["mouvement"]}', **j)[0], 400)
        E._extraction.acquire()          # an extraction is already running
        try:
            s, b, _ = self.request("POST", "/api/extract", b'{"models": ["A"]}', **j)
            self.assertEqual(s, 409)
            self.assertIn("already running", lu(b)["error"])
        finally:
            E._extraction.release()
        ident = E.launch_extraction(["A"], ["faces"], run=False)
        s, b, _ = self.request("GET", "/api/jobs/" + ident)
        self.assertEqual(lu(b)["step"], "waiting")
        self.assertIn("log", lu(b))
        self.assertTrue(lu(self.request("GET", "/api/jobs/x20990101-000000")[1])["done"])

    def test_extraction_in_progress_and_cancelled(self):
        """The running extraction comes from the server (GET /api/extraction), and cancelling it stops every program
        it started, a child that ignores SIGTERM included, then frees the lock."""
        j = {"Content-Type": "application/json"}
        os.makedirs(os.path.join(self.tmp, "mmd_extracted", "A"))
        tool, pids = os.path.join(self.tmp, "tool.py"), os.path.join(self.tmp, "pids")
        with open(tool, "w") as f:
            f.write("import subprocess, sys, time\n"
                    "a = subprocess.Popen(['sleep', '60'])\n"
                    "b = subprocess.Popen(['bash', '-c', 'trap \"\" TERM; sleep 60 & wait'])\n"
                    "print('@@ {\"model\": \"A\"}', flush=True)\n"
                    "open(sys.argv[1], 'w').write('%d %d' % (a.pid, b.pid))\n"
                    "time.sleep(60)\n")

        def chain(x):
            x.place(step="faces", stage="faces", message="faces")
            x.command([tool, pids], line=lambda m: x.place(model=m["model"]))
            x.place(step="done", done=True, ok=True)
        guard, stop = E.Extraction.chain, E.Extraction.STOP
        E.Extraction.chain, E.Extraction.STOP = chain, 2
        try:
            self.assertEqual(lu(self.request("GET", "/api/extraction")[1]), {"id": None})
            self.assertEqual(self.request("POST", "/api/extraction/cancel", b"{}", **j)[0], 409)   # nothing to cancel
            ident = E.launch_extraction(["A"], ["faces"])
            wait(lambda: os.path.isfile(pids) and os.path.getsize(pids))
            group = E.IN_PROGRESS["work"].proc.pid
            e = lu(self.request("GET", "/api/extraction")[1])
            self.assertEqual((e["id"], e["step"], e["done"], e["model"]), (ident, "faces", False, "A"))
            s, b, _ = self.request("POST", "/api/extract", b'{"models": ["A"]}', **j)
            self.assertEqual(s, 409)
            s, b, _ = self.request("POST", "/api/extraction/cancel", b"{}", **j)
            self.assertEqual((s, lu(b)["id"]), (200, ident))
            wait(lambda: lu(self.request("GET", "/api/extraction")[1])["id"] is None, 30)
            e = lu(self.request("GET", "/api/jobs/" + ident)[1])
            self.assertEqual((e["step"], e["done"], e["cancelled"]), ("cancelled", True, True))
            with self.assertRaises(ProcessLookupError):        # the whole group is gone, the TERM-proof child too
                os.killpg(group, 0)
            self.assertTrue(E._extraction.acquire(blocking=False))   # the lock is free: a new extraction can start
            E._extraction.release()
        finally:
            E.Extraction.chain, E.Extraction.STOP = guard, stop

    def test_download(self):
        d = os.path.join(E.ASSEMBLIES, "20261117-101010")
        os.makedirs(os.path.join(d, "model", "tex"))
        for f, b in (("model/result.pmx", b"P"), ("model/result-zzz.pmx", b"Z"), ("model/tex/a.png", b"T"),
                     ("model/result.vrm", b"V")):
            with open(os.path.join(d, f), "wb") as o:
                o.write(b)
        with open(os.path.join(d, "selection.json"), "w") as o:
            json.dump({"head": "t", "hairstyle": "c", "parts": [], "name": "Élise"}, o)   # fr-ok: accented display name kept as is
        items = lu(self.request("GET", "/api/assemblies")[1])
        self.assertEqual((items[0]["name"], items[0]["formats"]), ("Élise", ["pmx", "vrm"]))   # fr-ok: accented display name kept as is
        s, b, r = self.request("GET", "/api/assemblies/20261117-101010/vrm")
        self.assertEqual((s, b), (200, b"V"))
        self.assertIn('attachment; filename="Elise.vrm"', r.getheader("Content-Disposition"))
        self.assertIn("UTF-8''%C3%89lise.vrm", r.getheader("Content-Disposition"))
        s, b, r = self.request("GET", "/api/assemblies/20261117-101010/pmx")
        self.assertEqual(s, 200)
        names = zipfile.ZipFile(io.BytesIO(b)).namelist()
        self.assertEqual(sorted(names), ["Elise/Elise.pmx", "Elise/tex/a.png"])
        # the server removes its temporary zip right after sending it, possibly after the client has read it all
        wait(lambda: not [f for f in os.listdir(d) if f.startswith(".export")], 5)
        self.assertEqual(self.request("GET", "/api/assemblies/20261117-101010/glb")[0], 404)
        self.assertEqual(self.request("GET", "/api/assemblies/20261117-999999/vrm")[0], 404)
        self.assertEqual(self.request("GET", "/api/assemblies/..%2F..%2Fetc/vrm")[0], 404)

    def test_name_d_assembly(self):
        before = S.CATALOGUE
        S.CATALOGUE = {"parts": [], "hairstyles": [{"id": "c"}], "heads": [{"id": "t"}]}
        try:
            self.assertEqual(S.validate({"head": "t", "hairstyle": "c", "name": " Rose/../x "})["name"], "Rose .. x")
            with self.assertRaises(ValueError):
                S.validate({"head": "t", "hairstyle": "c", "name": 3})
        finally:
            S.CATALOGUE = before


if __name__ == "__main__":
    unittest.main()

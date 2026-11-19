"""The library install on a small handmade package: safe unpacking (booby-trapped archives refused), sha256
checked, records adapted to this machine (roots and fingerprints), identical copies linked, someone else's files
never overwritten without --replace, a download over HTTP resumed or started again. No workshop, no Blender."""
import hashlib
import http.server
import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import installation as I  # noqa: E402


def tar(path, limbs):
    """limbs: [(name, bytes | None for a folder | ("lien", target))]."""
    with tarfile.open(path, "w:gz") as t:
        for name, v in limbs:
            ti = tarfile.TarInfo(name)
            if v is None:
                ti.type = tarfile.DIRTYPE
                t.addfile(ti)
            elif isinstance(v, tuple):
                ti.type, ti.linkname = tarfile.SYMTYPE, v[1]
                t.addfile(ti)
            else:
                ti.size, ti.mtime = len(v), 1700000000
                t.addfile(ti, io.BytesIO(v))


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Package(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="biblio-")
        self.data = os.path.join(self.tmp, "data")
        settings = os.path.join(self.tmp, "settings.json")
        with open(settings, "w") as f:
            json.dump({"paths": {"data": self.data}, "games": {"active": "vroid"}}, f)
        os.environ["CHIMERA_SETTINGS"] = settings
        for v in ("CHIMERA_DATA", "CHIMERA_CACHE", "CHIMERA_BODY", "CHIMERA_GAME"):
            os.environ.pop(v, None)
        I.R.forget()
        self.package = os.path.join(self.tmp, "package")
        os.makedirs(os.path.join(self.package, "archives"))
        self.old = "0123456789abcdef"   # fingerprint of A.pmx on the owner's machine
        self.limbs = [
            ("cache/heads/A/meta.json", json.dumps({"id": "A", "pmx": "{mmd_extracted}/A/A.pmx", "fingerprint": self.old}).encode()),
            ("cache/hairstyles/A/A.bin", b"pair"),
            ("cache/labels.json", json.dumps({"heads/A": {"name": ["A", "A"]}}).encode()),
            ("mmd_extracted/A/A.pmx", b"pmx data"),
            ("face/A/record.json", json.dumps({"pmx": "{mmd_extracted}/A/A.pmx"}).encode()),
            ("base/body.pmx", b"body"),
            ("clothes/A/top/top.pmx", b"part"),
        ]
        self.index = {"format": I.FORMAT, "revision": I.FMT.REVISION, "code": 1, "version": "test",
                      "counts": {"models": 1, "heads": 1, "hairstyles": 1, "parts": 0},
                      "units": ["base", "cache/hairstyles/A", "cache/heads/A", "mmd_extracted/A", "clothes/A", "face/A"],
                      "shared": ["cache/labels.json"], "catalog": "cache/catalog.json",
                      "links": [{"link": "cache/library/A", "target": "face/A"}],
                      "fingerprints": {self.old: "mmd_extracted/A/A.pmx"},
                      "duplicates": [{"from": "cache/hairstyles/A/A.bin", "names": "*heads"},
                                  {"from": "mmd_extracted/A/A.pmx", "to": ["clothes/A/tex/A.pmx"]}], "heads_bin": ["A", "B"]}
        self.write()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CHIMERA_SETTINGS", None)
        I.R.forget()

    def write(self, limbs=None):
        a = os.path.join(self.package, "archives", "tout-01.tar.gz")
        tar(a, limbs or self.limbs)
        self.index["archives"] = [{"path": "archives/tout-01.tar.gz", "size": os.path.getsize(a), "sha256": sha(a),
                                   "extracted_size": 10000}]
        self.index["installed_size"] = 10000
        with open(os.path.join(self.package, "index.json"), "w") as f:
            json.dump(self.index, f)

    def test_installed_and_adapt(self):
        I.install(self.package)
        rt = I.roots()
        self.assertEqual(rt["mmd_extracted"], os.path.join(self.data, "sources", "vroid", "mmd-extracted"))
        pmx = os.path.join(rt["mmd_extracted"], "A", "A.pmx")
        m = I.read_json(os.path.join(rt["cache"], "heads", "A", "meta.json"))
        self.assertEqual(m["pmx"], pmx)                                  # root written for this machine
        self.assertEqual(m["fingerprint"], I.fingerprint_file(pmx))       # fingerprint computed again
        self.assertNotEqual(m["fingerprint"], self.old)
        self.assertEqual(os.readlink(os.path.join(rt["cache"], "library", "A")), os.path.join(rt["face"], "A"))
        b = os.path.join(rt["cache"], "hairstyles", "A", "B.bin")        # identical copy made as a hard link
        self.assertTrue(os.path.samefile(b, os.path.join(rt["cache"], "hairstyles", "A", "A.bin")))
        self.assertTrue(os.path.isfile(os.path.join(rt["base"], "body.pmx")))
        self.assertTrue(os.path.samefile(os.path.join(rt["clothes"], "A", "tex", "A.pmx"), pmx))  # folder of links only
        man = I.read_json(os.path.join(rt["cache"], I.MANIFEST))
        self.assertIn("cache/heads/A", man["units"])
        # running again: the same version, complete, is recognised and nothing is downloaded again
        I.install(self.package)

    def test_keep_models_of_l_user(self):
        rt = I.roots()
        mine = os.path.join(rt["cache"], "heads", "A")
        os.makedirs(mine)
        with open(os.path.join(mine, "meta.json"), "w") as f:
            f.write('{"id": "A", "a moi": true}')
        with open(os.path.join(rt["cache"], "labels.json"), "w") as f:
            json.dump({"heads/Z": {"name": ["Z", "Z"]}}, f)
        with self.assertRaises(I.Error):
            I.install(self.package)
        self.assertTrue(I.read_json(os.path.join(mine, "meta.json"))["a moi"])          # untouched
        I.install(self.package, replace=True)
        put = [d for d in os.listdir(os.path.join(self.data, I.WORK)) if d.startswith("replaced-")]
        self.assertEqual(len(put), 1)                                                    # set aside, not deleted
        self.assertTrue(os.path.isfile(os.path.join(self.data, I.WORK, put[0], "cache", "heads", "A", "meta.json")))
        lbl = I.read_json(os.path.join(rt["cache"], "labels.json"))
        self.assertEqual(set(lbl), {"heads/Z", "heads/A"})                               # merged, theirs kept

    def test_resumes_one_installation_interrupted(self):
        real = I._adapt
        def failure(*a):
            raise OSError("disk full")
        I._adapt = failure
        try:
            with self.assertRaises(OSError):
                I.install(self.package)
        finally:
            I._adapt = real
        I.install(self.package)            # the half-installed items are ours: no --replace needed
        man = I.read_json(os.path.join(I.roots()["cache"], I.MANIFEST))
        self.assertNotIn("in_progress", man)
        remainders = [d for d in os.listdir(os.path.join(self.data, I.WORK)) if d.startswith("extracted-")]
        self.assertEqual(remainders, [])

    def test_archive_trapped(self):
        for bad in (("../outside.txt", b"x"), ("/etc/x", b"x"), ("cache/x/../../../y", b"x"),
                        ("cache/lien", ("link", "/etc/passwd")), ("elsewhere/x", b"x")):
            self.write(self.limbs + [bad])
            with self.assertRaises(I.Error, msg=repr(bad[0])):
                I.install(self.package)
            self.assertFalse(os.path.exists(os.path.join(self.tmp, "outside.txt")))
            self.assertFalse(os.path.exists(I.roots()["cache"] + "/heads"))           # nothing installed

    def test_index_trap(self):
        self.index["units"].append("cache/../../x")
        self.write()
        with self.assertRaises(I.Error):
            I.install(self.package)

    def test_sha256_false(self):
        self.index["archives"][0]["sha256"] = "0" * 64
        with open(os.path.join(self.package, "index.json"), "w") as f:
            json.dump(self.index, f)
        with self.assertRaises(I.Error):
            I.install(self.package)

    def test_http_and_resumption(self):
        rep = self.package

        class Management(http.server.SimpleHTTPRequestHandler):
            def __init__(s, *a, **k):
                super().__init__(*a, directory=rep, **k)

            def log_message(s, *a):
                pass
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Management)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = "http://127.0.0.1:%d" % srv.server_address[1]
            part = os.path.join(self.data, I.WORK, "archives", "tout-01.tar.gz.part")
            os.makedirs(os.path.dirname(part))
            with open(part, "wb") as f:      # an interrupted download; this server ignores Range: started again
                f.write(b"debut")
            I.install(url, keep=True)
            arch = os.path.join(self.data, I.WORK, "archives", "tout-01.tar.gz")
            self.assertEqual(sha(arch), self.index["archives"][0]["sha256"])
            self.assertFalse(os.path.exists(part))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()

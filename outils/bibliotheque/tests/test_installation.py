"""The library installer on a small handmade package: safe unpacking (booby-trapped archives refused), sha256
checked, records adapted to this machine (roots and fingerprints), identical copies linked, someone else's files
never overwritten without --remplacer, a download over HTTP resumed or started again. No workshop, no Blender."""
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

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
import installation as I  # noqa: E402


def tar(chemin, membres):
    """membres: [(name, bytes | None for a folder | ("lien", target))]."""
    with tarfile.open(chemin, "w:gz") as t:
        for nom, v in membres:
            ti = tarfile.TarInfo(nom)
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


class Paquet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="biblio-")
        self.donnees = os.path.join(self.tmp, "donnees")
        reglages = os.path.join(self.tmp, "reglages.json")
        with open(reglages, "w") as f:
            json.dump({"chemins": {"donnees": self.donnees}, "jeux": {"actif": "vroid"}}, f)
        os.environ["ATELIER_REGLAGES"] = reglages
        for v in ("ATELIER_ATLAS", "ATELIER_CACHE", "ATELIER_CORPS", "ATELIER_JEU"):
            os.environ.pop(v, None)
        I.R.oublier()
        self.paquet = os.path.join(self.tmp, "paquet")
        os.makedirs(os.path.join(self.paquet, "archives"))
        self.vieille = "0123456789abcdef"   # fingerprint of A.pmx on the owner's machine
        self.membres = [
            ("cache/tetes/A/meta.json", json.dumps({"id": "A", "pmx": "{mmd_extrait}/A/A.pmx", "empreinte": self.vieille}).encode()),
            ("cache/coiffures/A/A.bin", b"pair"),
            ("cache/etiquettes.json", json.dumps({"tetes/A": {"nom": ["A", "A"]}}).encode()),
            ("mmd_extrait/A/A.pmx", b"pmx data"),
            ("visage/A/fiche.json", json.dumps({"pmx": "{mmd_extrait}/A/A.pmx"}).encode()),
            ("base/corps.pmx", b"body"),
            ("vetements/A/haut/haut.pmx", b"part"),
        ]
        self.index = {"format": I.FORMAT, "code": 1, "version": "test",
                      "nombres": {"modeles": 1, "tetes": 1, "coiffures": 1, "pieces": 0},
                      "unites": ["base", "cache/coiffures/A", "cache/tetes/A", "mmd_extrait/A", "vetements/A", "visage/A"],
                      "partages": ["cache/etiquettes.json"], "catalogue": "cache/catalogue.json",
                      "liens": [{"lien": "cache/biblio/A", "cible": "visage/A"}],
                      "empreintes": {self.vieille: "mmd_extrait/A/A.pmx"},
                      "doubles": [{"de": "cache/coiffures/A/A.bin", "noms": "*tetes"},
                                  {"de": "mmd_extrait/A/A.pmx", "vers": ["vetements/A/tex/A.pmx"]}], "tetes_bin": ["A", "B"]}
        self.ecrire()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("ATELIER_REGLAGES", None)
        I.R.oublier()

    def ecrire(self, membres=None):
        a = os.path.join(self.paquet, "archives", "tout-01.tar.gz")
        tar(a, membres or self.membres)
        self.index["archives"] = [{"chemin": "archives/tout-01.tar.gz", "taille": os.path.getsize(a), "sha256": sha(a),
                                   "taille_extraite": 10000}]
        self.index["taille_installee"] = 10000
        with open(os.path.join(self.paquet, "index.json"), "w") as f:
            json.dump(self.index, f)

    def test_installe_et_adapte(self):
        I.installer(self.paquet)
        rac = I.racines()
        self.assertEqual(rac["mmd_extrait"], os.path.join(self.donnees, "sources", "vroid", "mmd-extrait"))
        pmx = os.path.join(rac["mmd_extrait"], "A", "A.pmx")
        m = I.lire_json(os.path.join(rac["cache"], "tetes", "A", "meta.json"))
        self.assertEqual(m["pmx"], pmx)                                  # root written for this machine
        self.assertEqual(m["empreinte"], I.empreinte_fichier(pmx))       # fingerprint computed again
        self.assertNotEqual(m["empreinte"], self.vieille)
        self.assertEqual(os.readlink(os.path.join(rac["cache"], "biblio", "A")), os.path.join(rac["visage"], "A"))
        b = os.path.join(rac["cache"], "coiffures", "A", "B.bin")        # identical copy made as a hard link
        self.assertTrue(os.path.samefile(b, os.path.join(rac["cache"], "coiffures", "A", "A.bin")))
        self.assertTrue(os.path.isfile(os.path.join(rac["base"], "corps.pmx")))
        self.assertTrue(os.path.samefile(os.path.join(rac["vetements"], "A", "tex", "A.pmx"), pmx))  # folder of links only
        man = I.lire_json(os.path.join(rac["cache"], I.MANIFESTE))
        self.assertIn("cache/tetes/A", man["unites"])
        # running again: our own previous installation is replaced without a word
        I.installer(self.paquet)

    def test_garde_les_modeles_de_l_utilisateur(self):
        rac = I.racines()
        mien = os.path.join(rac["cache"], "tetes", "A")
        os.makedirs(mien)
        with open(os.path.join(mien, "meta.json"), "w") as f:
            f.write('{"id": "A", "a moi": true}')
        with open(os.path.join(rac["cache"], "etiquettes.json"), "w") as f:
            json.dump({"tetes/Z": {"nom": ["Z", "Z"]}}, f)
        with self.assertRaises(I.Erreur):
            I.installer(self.paquet)
        self.assertTrue(I.lire_json(os.path.join(mien, "meta.json"))["a moi"])          # untouched
        I.installer(self.paquet, remplacer=True)
        mis = [d for d in os.listdir(os.path.join(self.donnees, I.TRAVAIL)) if d.startswith("remplaces-")]
        self.assertEqual(len(mis), 1)                                                    # set aside, not deleted
        self.assertTrue(os.path.isfile(os.path.join(self.donnees, I.TRAVAIL, mis[0], "cache", "tetes", "A", "meta.json")))
        etq = I.lire_json(os.path.join(rac["cache"], "etiquettes.json"))
        self.assertEqual(set(etq), {"tetes/Z", "tetes/A"})                               # merged, theirs kept

    def test_reprend_une_installation_interrompue(self):
        vrai = I._adapter
        def panne(*a):
            raise OSError("disk full")
        I._adapter = panne
        try:
            with self.assertRaises(OSError):
                I.installer(self.paquet)
        finally:
            I._adapter = vrai
        I.installer(self.paquet)            # the half-installed items are ours: no --remplacer needed
        man = I.lire_json(os.path.join(I.racines()["cache"], I.MANIFESTE))
        self.assertNotIn("en_cours", man)
        restes = [d for d in os.listdir(os.path.join(self.donnees, I.TRAVAIL)) if d.startswith("extrait-")]
        self.assertEqual(restes, [])

    def test_archive_piegee(self):
        for mauvais in (("../dehors.txt", b"x"), ("/etc/x", b"x"), ("cache/x/../../../y", b"x"),
                        ("cache/lien", ("lien", "/etc/passwd")), ("ailleurs/x", b"x")):
            self.ecrire(self.membres + [mauvais])
            with self.assertRaises(I.Erreur, msg=repr(mauvais[0])):
                I.installer(self.paquet)
            self.assertFalse(os.path.exists(os.path.join(self.tmp, "dehors.txt")))
            self.assertFalse(os.path.exists(I.racines()["cache"] + "/tetes"))           # nothing installed

    def test_index_piege(self):
        self.index["unites"].append("cache/../../x")
        self.ecrire()
        with self.assertRaises(I.Erreur):
            I.installer(self.paquet)

    def test_sha256_faux(self):
        self.index["archives"][0]["sha256"] = "0" * 64
        with open(os.path.join(self.paquet, "index.json"), "w") as f:
            json.dump(self.index, f)
        with self.assertRaises(I.Erreur):
            I.installer(self.paquet)

    def test_http_et_reprise(self):
        rep = self.paquet

        class Gestion(http.server.SimpleHTTPRequestHandler):
            def __init__(s, *a, **k):
                super().__init__(*a, directory=rep, **k)

            def log_message(s, *a):
                pass
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Gestion)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = "http://127.0.0.1:%d" % srv.server_address[1]
            part = os.path.join(self.donnees, I.TRAVAIL, "archives", "tout-01.tar.gz.part")
            os.makedirs(os.path.dirname(part))
            with open(part, "wb") as f:      # an interrupted download; this server ignores Range: started again
                f.write(b"debut")
            I.installer(url, garder=True)
            arch = os.path.join(self.donnees, I.TRAVAIL, "archives", "tout-01.tar.gz")
            self.assertEqual(sha(arch), self.index["archives"][0]["sha256"])
            self.assertFalse(os.path.exists(part))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()

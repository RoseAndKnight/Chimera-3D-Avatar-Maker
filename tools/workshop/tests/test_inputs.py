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
import games as jeux  # noqa: E402  (tools/ is on the path through commun)


class JeuFaux:
    """Test active game: `modeles` "*" (importing works) or a declared list (no importing)."""

    def __init__(self, racine, etoile=True):
        self.nom, self.titre, self.racine = "essai", "Essai", racine
        self.d = {"format": "vrm", "modeles": "*" if etoile else {"A": "A/A.pmx"}}

    def valeur(self, cle, defaut=None):
        return self.d.get(cle, defaut)

    def chemin(self, source):
        if source not in ("mmd_extrait", "vrm"):
            raise KeyError(source)
        return os.path.join(self.racine, source)

    def modeles(self):
        r = os.path.join(self.racine, "mmd_extrait")
        if self.d["modeles"] != "*":
            return dict(self.d["modeles"])
        return {n: "%s/%s.pmx" % (n, n) for n in sorted(os.listdir(r))} if os.path.isdir(r) else {}


def un_zip(entrees, symlien=None):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for nom, octets in entrees.items():
            z.writestr(nom, octets)
        if symlien:
            i = zipfile.ZipInfo(symlien)
            i.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(i, "/etc/passwd")
    return b.getvalue()


def un_glb(extensions):
    j = json.dumps({"asset": {"version": "2.0"}, "extensions": extensions}).encode()
    j += b" " * (-len(j) % 4)
    return b"glTF" + struct.pack("<II", 2, 12 + 8 + len(j)) + struct.pack("<I4s", len(j), b"JSON") + j


def attendre(condition, delai=15):
    import time
    fin = time.time() + delai
    while not condition():
        if time.time() > fin:
            raise AssertionError("condition not met within %d s" % delai)
        time.sleep(0.1)


class Bac(unittest.TestCase):
    """Test folders in place of the cache and of the active game's sources."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.garde = {k: getattr(E, k) for k in ("ATTENTE", "ASSEMBLAGES", "EXTRACTIONS", "J")}
        E.ATTENTE = os.path.join(self.tmp, "attente")
        E.ASSEMBLAGES = os.path.join(self.tmp, "assemblages")
        E.EXTRACTIONS = os.path.join(self.tmp, "extractions")
        E.J = JeuFaux(self.tmp)

    def tearDown(self):
        for k, v in self.garde.items():
            setattr(E, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def deposer(self, octets):
        c = os.path.join(self.tmp, "recu")
        with open(c, "wb") as f:
            f.write(octets)
        return c


class TestArchivesPiegees(Bac):
    def refus(self, octets, attendu):
        with self.assertRaises(E.Refus) as r:
            E.deplier_zip(self.deposer(octets), os.path.join(self.tmp, "deplie"))
        self.assertIn(attendu, str(r.exception))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil")))

    def test_chemin_qui_sort(self):
        self.refus(un_zip({"m.pmx": b"x", "../evil/x.png": b"x"}), "escapes its folder")
        self.refus(un_zip({"m.pmx": b"x", "/tmp/evil.png": b"x"}), "escapes its folder")
        self.refus(un_zip({"m.pmx": b"x", "C:/evil.png": b"x"}), "escapes its folder")

    def test_lien(self):
        self.refus(un_zip({"m.pmx": b"x"}, symlien="tex/lien.png"), "link")

    def test_bombe(self):
        self.refus(un_zip({"m.pmx": b"x", "tex/zero.png": bytes(20 << 20)}), "abnormal compression")

    def test_pmx_seul_et_unique(self):
        self.refus(un_zip({"a.png": b"x"}), "no .pmx file")
        self.refus(b"PK pas une archive", "not a readable .zip")

    def test_archive_saine(self):
        rel = E.deplier_zip(self.deposer(un_zip({"Perso/Perso.pmx": b"x", "Perso/tex/a.png": b"y"})),
                            os.path.join(self.tmp, "deplie"))
        self.assertEqual(rel, ["Perso/Perso.pmx"])
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "deplie", "Perso", "tex", "a.png")))


class TestLot(Bac):
    def test_plusieurs_pmx_rendus(self):
        rel = E.deplier_zip(self.deposer(un_zip({"A/a.pmx": b"x", "B/b.pmx": b"x", "B/arme.pmx": b"x"})),
                            os.path.join(self.tmp, "deplie"))
        self.assertEqual(rel, ["A/a.pmx", "B/arme.pmx", "B/b.pmx"])

    def test_noms_sans_drapeau_utf8(self):
        # a UTF-8 name written without declaring it (what `zip` does on Linux): same bytes, flag missing
        octets = un_zip({"A/a.pmx": b"x", "A/tex/XYZ.tga": b"x"}).replace(b"XYZ.tga", "颜.tga".encode("utf-8"))
        E.deplier_zip(self.deposer(octets), os.path.join(self.tmp, "deplie"))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "deplie", "A", "tex", "颜.tga")))

    def test_noms_du_lot(self):
        bons = [("Anby/安比.pmx", None, None), ("Lot/un.pmx", None, None), ("Lot/deux.pmx", None, None),
                ("安/安.pmx", None, None), ("Anby2/x.pmx", None, None)]
        noms = E._idents_du_lot("archive", bons)
        self.assertEqual(noms, ["Anby", "Lot-un", "Lot-deux", "archive_4", "Anby2"])
        self.assertEqual(len(set(noms)), len(noms))


class TestImportRefuse(Bac):
    def test_format_refuse(self):
        with self.assertRaises(E.Refus) as r:
            E.importer(self.deposer(b"x"), "perso.fbx")
        self.assertEqual(r.exception.code, 415)

    def test_jeu_a_liste_declaree(self):
        E.J = JeuFaux(self.tmp, etoile=False)
        with self.assertRaises(E.Refus) as r:
            E.importer(self.deposer(b"x"), "perso.vrm")
        self.assertEqual(r.exception.code, 409)

    def test_vrm_1_et_faux_vrm(self):
        for octets, attendu in ((un_glb({"VRMC_vrm": {"specVersion": "2.0"}}), "other than 0.x and 1.x"),
                                (un_glb({}), "without the VRM extension"), (b"pas un glb" * 4, "not a VRM")):
            with self.assertRaises(E.Refus) as r:
                E.entete_vrm(self.deposer(octets))
            self.assertIn(attendu, str(r.exception))
        self.assertEqual(E.entete_vrm(self.deposer(un_glb({"VRM": {"meta": {"title": "Elle"}}}))), "Elle")
        self.assertEqual(E.entete_vrm(self.deposer(un_glb({"VRMC_vrm": {"specVersion": "1.0", "meta": {"name": "Lui"}}}))), "Lui")

    def test_deja_importe_et_recu_efface(self):
        os.makedirs(os.path.join(self.tmp, "mmd_extrait", "Perso"))
        c = self.deposer(b"x")
        with self.assertRaises(E.Refus) as r:
            E.importer(c, "Perso.vrm")
        self.assertEqual(r.exception.code, 409)
        self.assertFalse(os.path.exists(c))

    def test_noms(self):
        self.assertEqual(E.identifiant("C:\\x\\Élise été.vrm"), "Elise_ete")
        self.assertEqual(E.identifiant("../../etc/passwd.zip"), "passwd")
        with self.assertRaises(E.Refus):
            E.identifiant("日本.vrm")
        self.assertEqual(E.nettoyer_nom("  Rose <script>/../  "), "Rose script")
        self.assertEqual(len(E.nettoyer_nom("a" * 200)), 60)
        self.assertEqual(E.nom_telechargement("Élise \"x\"", "p"), ("Elise_x", "Élise x"))


class TestFormatsFideles(unittest.TestCase):
    """What /api/formats says comes from the game.json files: an announced skeleton goes all the way to the catalogue."""

    def test_squelettes_et_jeux(self):
        f = E.formats()
        annonces = {s["id"]: s for s in f["squelettes"]}
        self.assertNotIn("non_pris", f)        # only the skeletons the workshop reads are listed
        for nom in jeux.tous():
            if nom == "homemade":
                continue
            d = jeux.charger(nom).d
            if "modeles" in d and "tete_base" in d:
                self.assertIn(nom, annonces)
                self.assertEqual(annonces[nom]["etat"], d["compatibilite"]["niveau"])
            else:
                self.assertNotIn(nom, annonces)
        self.assertEqual({s["id"] for s in f["sortie"]}, {"pmx", "vrm"})
        self.assertEqual({e["id"] for e in f["entree"]}, {"vrm", "pmx"})
        self.assertEqual(f["jeu_actif"]["id"], E.J.nom)


class TestRoutes(Bac):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), S.Gestion)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def demande(self, methode, chemin, corps=None, **entetes):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request(methode, chemin, body=corps, headers=entetes)
        r = c.getresponse()
        b = r.read()
        c.close()
        return r.status, b, r

    def test_import_gardes(self):
        o = {"Content-Type": "application/octet-stream"}
        self.assertEqual(self.demande("PUT", "/api/importer?nom=a.vrm", b"x", Origin="https://evil.example", **o)[0], 403)
        self.assertEqual(self.demande("PUT", "/api/importer?nom=a.vrm", b"x", **{"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.demande("PUT", "/api/importer?nom=a.exe", b"x", **o)[0], 415)
        s, b, _ = self.demande("PUT", "/api/importer?nom=a.vrm", b"pas un vrm", **o)
        self.assertEqual(s, 400)
        self.assertIn("not a VRM", json.loads(b)["erreur"])
        E.J = JeuFaux(self.tmp, etoile=False)
        self.assertEqual(self.demande("PUT", "/api/importer?nom=a.vrm", b"x", **o)[0], 409)

    def test_extraire_gardes(self):
        j = {"Content-Type": "application/json"}
        self.assertEqual(self.demande("POST", "/api/extraire", b'{"tous": true}', **{"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.demande("POST", "/api/extraire", b'{"modeles": ["../x"]}', **j)[0], 400)
        self.assertEqual(self.demande("POST", "/api/extraire", b'{"tous": true}', **j)[0], 400)   # no model
        os.makedirs(os.path.join(self.tmp, "mmd_extrait", "A"))
        self.assertEqual(self.demande("POST", "/api/extraire", b'{"tous": true, "quoi": ["mouvement"]}', **j)[0], 400)
        E._extraction.acquire()          # an extraction is already running
        try:
            s, b, _ = self.demande("POST", "/api/extraire", b'{"modeles": ["A"]}', **j)
            self.assertEqual(s, 409)
            self.assertIn("already running", json.loads(b)["erreur"])
        finally:
            E._extraction.release()
        ident = E.lancer_extraction(["A"], ["visages"], executer=False)
        s, b, _ = self.demande("GET", "/api/travaux/" + ident)
        self.assertEqual(json.loads(b)["etape"], "attente")
        self.assertIn("journal", json.loads(b))
        self.assertTrue(json.loads(self.demande("GET", "/api/travaux/x20990101-000000")[1])["fini"])

    def test_extraction_en_cours_et_annulee(self):
        """The running extraction comes from the server (GET /api/extraction), and cancelling it stops every program
        it started, a child that ignores SIGTERM included, then frees the lock."""
        j = {"Content-Type": "application/json"}
        os.makedirs(os.path.join(self.tmp, "mmd_extrait", "A"))
        outil, pids = os.path.join(self.tmp, "outil.py"), os.path.join(self.tmp, "pids")
        with open(outil, "w") as f:
            f.write("import subprocess, sys, time\n"
                    "a = subprocess.Popen(['sleep', '60'])\n"
                    "b = subprocess.Popen(['bash', '-c', 'trap \"\" TERM; sleep 60 & wait'])\n"
                    "print('@@ {\"modele\": \"A\"}', flush=True)\n"
                    "open(sys.argv[1], 'w').write('%d %d' % (a.pid, b.pid))\n"
                    "time.sleep(60)\n")

        def chaine(x):
            x.poser(etape="visages", pas="visages", message="faces")
            x.commande([outil, pids], ligne=lambda m: x.poser(modele=m["modele"]))
            x.poser(etape="fin", fini=True, ok=True)
        garde, arret = E.Extraction.chaine, E.Extraction.ARRET
        E.Extraction.chaine, E.Extraction.ARRET = chaine, 2
        try:
            self.assertEqual(json.loads(self.demande("GET", "/api/extraction")[1]), {"id": None})
            self.assertEqual(self.demande("POST", "/api/extraction/annuler", b"{}", **j)[0], 409)   # nothing to cancel
            ident = E.lancer_extraction(["A"], ["visages"])
            attendre(lambda: os.path.isfile(pids) and os.path.getsize(pids))
            groupe = E.EN_COURS["travail"].proc.pid
            e = json.loads(self.demande("GET", "/api/extraction")[1])
            self.assertEqual((e["id"], e["etape"], e["fini"], e["modele"]), (ident, "visages", False, "A"))
            s, b, _ = self.demande("POST", "/api/extraire", b'{"modeles": ["A"]}', **j)
            self.assertEqual(s, 409)
            s, b, _ = self.demande("POST", "/api/extraction/annuler", b"{}", **j)
            self.assertEqual((s, json.loads(b)["id"]), (200, ident))
            attendre(lambda: json.loads(self.demande("GET", "/api/extraction")[1])["id"] is None, 30)
            e = json.loads(self.demande("GET", "/api/travaux/" + ident)[1])
            self.assertEqual((e["etape"], e["fini"], e["annulee"]), ("annulee", True, True))
            with self.assertRaises(ProcessLookupError):        # the whole group is gone, the TERM-proof child too
                os.killpg(groupe, 0)
            self.assertTrue(E._extraction.acquire(blocking=False))   # the lock is free: a new extraction can start
            E._extraction.release()
        finally:
            E.Extraction.chaine, E.Extraction.ARRET = garde, arret

    def test_telechargement(self):
        d = os.path.join(E.ASSEMBLAGES, "20261117-101010")
        os.makedirs(os.path.join(d, "modele", "tex"))
        for f, b in (("modele/resultat.pmx", b"P"), ("modele/resultat-zzz.pmx", b"Z"), ("modele/tex/a.png", b"T"),
                     ("modele/resultat.vrm", b"V")):
            with open(os.path.join(d, f), "wb") as o:
                o.write(b)
        with open(os.path.join(d, "selection.json"), "w") as o:
            json.dump({"tete": "t", "coiffure": "c", "pieces": [], "nom": "Élise"}, o)
        liste = json.loads(self.demande("GET", "/api/assemblages")[1])
        self.assertEqual((liste[0]["nom"], liste[0]["formats"]), ("Élise", ["pmx", "vrm"]))
        s, b, r = self.demande("GET", "/api/assemblages/20261117-101010/vrm")
        self.assertEqual((s, b), (200, b"V"))
        self.assertIn('attachment; filename="Elise.vrm"', r.getheader("Content-Disposition"))
        self.assertIn("UTF-8''%C3%89lise.vrm", r.getheader("Content-Disposition"))
        s, b, r = self.demande("GET", "/api/assemblages/20261117-101010/pmx")
        self.assertEqual(s, 200)
        noms = zipfile.ZipFile(io.BytesIO(b)).namelist()
        self.assertEqual(sorted(noms), ["Elise/Elise.pmx", "Elise/tex/a.png"])
        # the server removes its temporary zip right after sending it, possibly after the client has read it all
        attendre(lambda: not [f for f in os.listdir(d) if f.startswith(".export")], 5)
        self.assertEqual(self.demande("GET", "/api/assemblages/20261117-101010/glb")[0], 404)
        self.assertEqual(self.demande("GET", "/api/assemblages/20261117-999999/vrm")[0], 404)
        self.assertEqual(self.demande("GET", "/api/assemblages/..%2F..%2Fetc/vrm")[0], 404)

    def test_nom_d_assemblage(self):
        avant = S.CATALOGUE
        S.CATALOGUE = {"pieces": [], "coiffures": [{"id": "c"}], "tetes": [{"id": "t"}]}
        try:
            self.assertEqual(S.valider({"tete": "t", "coiffure": "c", "nom": " Rose/../x "})["nom"], "Rose .. x")
            with self.assertRaises(ValueError):
                S.valider({"tete": "t", "coiffure": "c", "nom": 3})
        finally:
            S.CATALOGUE = avant


if __name__ == "__main__":
    unittest.main()

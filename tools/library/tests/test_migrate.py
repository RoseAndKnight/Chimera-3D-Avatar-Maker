"""Conversion of a data folder from format 1 (French names) to format 2, on a small handmade folder: names,
records, texture table of a .pmx, fingerprints, links and settings follow; hard-linked copies keep their
old content; a second run changes nothing; a conversion cut short ends cleanly; --check finds nothing left."""
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import dataformat as F  # noqa: E402
import migrate as M  # noqa: E402


def pmx(textures, rest=b"MATERIALS AND THE REST"):
    """A tiny PMX 2.0 (UTF-16): one BDEF1 vertex, one face, the texture table, then anything."""
    d = b"PMX " + struct.pack("<f", 2.0) + bytes([8, 0, 0, 1, 1, 1, 1, 1, 1])
    for _ in range(4):
        d += struct.pack("<i", 0)
    d += struct.pack("<i", 1) + b"\0" * 32 + bytes([0, 0]) + struct.pack("<f", 1.0)
    d += struct.pack("<i", 3) + bytes([0, 0, 0])
    d += struct.pack("<i", len(textures))
    for t in textures:
        b = t.encode("utf-16-le")
        d += struct.pack("<i", len(b)) + b
    return d + rest


def pmx_morphs(names, textures=("skin.png",)):
    """pmx() with a readable tail: no material nor bone, one vertex morph per name, no frame, body or joint."""
    tail = struct.pack("<ii", 0, 0) + struct.pack("<i", len(names))
    for n in names:
        b = n.encode("utf-16-le")
        tail += struct.pack("<i", len(b)) + b + struct.pack("<i", len(b)) + b + bytes([4, 1])
        tail += struct.pack("<i", 1) + bytes([0]) + struct.pack("<3f", 0.0, 0.1, 0.0)
    return pmx(list(textures), tail + struct.pack("<iii", 0, 0, 0))


def write(p, content):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb" if isinstance(content, bytes) else "w") as f:
        f.write(content if isinstance(content, (bytes, str)) else json.dumps(content, indent=1))


def read(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class Migration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inst = self.tmp.name
        D = self.D = os.path.join(self.inst, "donnees")
        write(os.path.join(D, "base-vroid", "corps.pmx"), pmx(["peau.png", "F00_000_Body_00.png"]))
        write(os.path.join(D, "base-vroid", "peau.png"), b"png")
        write(os.path.join(D, "sources", "vroid", "mmd-extrait", "A", "A.pmx"), pmx(["peau.png"]))
        body = os.path.join(D, "base-vroid", "corps.pmx")
        self.e_body = M.fingerprint(body)
        cache = os.path.join(D, "atelier-base-vroid")
        write(os.path.join(D, "base-vroid", "coiffure.pmx"), pmx(["peau.png"]))
        write(os.path.join(cache, "coiffures", "Libre", "meta.json"),
               {"id": "Libre", "empreinte": "v9:A:" + M.fingerprint(os.path.join(D, "base-vroid", "coiffure.pmx"))})
        write(os.path.join(cache, "pieces", "A__jupe-2", "meta.json"), {
            "id": "A__jupe-2", "perso": "A", "piece": "jupe-2", "classe": "vetement", "type": "jupe",
            "onglet": "bas", "emplacement": "jupe", "occupe": ["jupe"], "corps": self.e_body,
            "dossier": os.path.join(D, "bibliotheque-vetements", "A", "jupe-2"),
            "controles": [{"nom": "contour_taille", "valeur": 1, "seuil": 2, "ok": True}],
            "autres_echecs": ["double_face"], "raison": "contour_taille = 3 (threshold 2)",
            "portage": {"sous_la_peau": {"part": 0.1}}, "garnitures": ["ceinture"]})
        write(os.path.join(cache, "pieces", "A__jupe-2", "vignette.png"), b"png")
        write(os.path.join(cache, "catalogue.json"), {"tetes": [{"id": "A", "vignette": "/data/tetes/A/vignette.png"}],
                                                       "nombres": {"pieces": 1}})
        write(os.path.join(cache, "etiquettes.json"), {"tetes/A": {"nom": ["Soft face", "Visage doux"]}})
        write(os.path.join(cache, "tetes", "A", "modele", "resultat.pmx"),
               pmx(["..\\..\\..\\..\\sources\\vroid\\mmd-extrait\\A\\x.png", "teinte\\1_peau.png"]))
        write(os.path.join(cache, "tetes", "A", "modele", "teinte", "1_peau.png"), b"png")
        write(os.path.join(cache, "assemblages", "20261119-120000", "etat.json"), {"etape": "fin", "fini": True, "part": 1})
        write(os.path.join(D, "bibliotheque-vetements", "A", "jupe-2", "jupe-2.pmx"), pmx(["..\\tex\\a.png"]))
        write(os.path.join(D, "bibliotheque-vetements", "A", "decoupe.json"),
               {"pieces": {"jupe-2": {"faces": [1]}, "ceinture": {"faces": [2]}}})
        write(os.path.join(D, "bibliotheque-visage", "A", "fiche.json"), {"personnage": "A", "empreinte": self.e_body})
        os.makedirs(os.path.join(cache, "biblio"))
        os.symlink(os.path.join(D, "bibliotheque-visage", "A"), os.path.join(cache, "biblio", "A"))
        write(os.path.join(self.inst, "settings.json"), {"paths": {"data": D}, "workshop": {
            "cache": "{data}/atelier-base-vroid", "body": "{data}/base-vroid/corps.pmx"}})
        # a backup made with cp -al: same files, other names
        self.saved = os.path.join(self.inst, "donnees.format-1")
        os.system("cp -al '%s' '%s'" % (D, self.saved))

    def tearDown(self):
        self.tmp.cleanup()

    def test_converts(self):
        self.assertEqual(F.format_of(self.D), 1)
        self.assertEqual(M.main([self.D]), 0)
        D, c = self.D, os.path.join(self.D, "workshop-vroid-base")
        self.assertEqual(F.format_of(D), 2)
        m = read(os.path.join(c, "parts", "A__skirt-2", "meta.json"))
        self.assertEqual(m["id"], "A__skirt-2")
        self.assertEqual((m["part_name"], m["class"], m["type"], m["tab"], m["slot"], m["occupies"]),
                         ("skirt-2", "garment", "skirt", "bottom", "skirt", ["skirt"]))
        self.assertEqual(m["checks"][0]["name"], "outline_size")
        self.assertEqual(m["reason"], "outline_size = 3 (threshold 2)")
        self.assertEqual(m["fitting"]["under_skin"]["fraction"], 0.1)
        self.assertEqual(m["folder"], os.path.join(D, "clothes-library", "A", "skirt-2"))
        self.assertEqual(m["trims"], ["belt"])
        body = os.path.join(D, "vroid-base", "body.pmx")
        self.assertEqual(m["body"], M.fingerprint(body))                       # fingerprint of the moved file
        self.assertEqual(read(os.path.join(D, "face-library", "A", "record.json"))["fingerprint"], M.fingerprint(body))
        with open(body, "rb") as f:
            self.assertEqual(M.pmx_textures(f.read())[3], ["skin.png", "F00_000_Body_00.png"])
        self.assertTrue(os.path.isfile(os.path.join(D, "vroid-base", "skin.png")))
        with open(os.path.join(c, "heads", "A", "model", "result.pmx"), "rb") as f:
            self.assertEqual(M.pmx_textures(f.read())[3],
                             ["..\\..\\..\\..\\sources\\vroid\\mmd-extracted\\A\\x.png", "tint\\1_skin.png"])
        with open(os.path.join(D, "sources", "vroid", "mmd-extracted", "A", "A.pmx"), "rb") as f:
            self.assertEqual(M.pmx_textures(f.read())[3], ["peau.png"])          # a model's own files: untouched
        self.assertEqual(read(os.path.join(c, "catalog.json"))["heads"][0]["thumbnail"], "/data/heads/A/thumbnail.png")
        self.assertEqual(read(os.path.join(c, "labels.json")), {"heads/A": {"name": ["Soft face", "Visage doux"]}})
        self.assertEqual(read(os.path.join(c, "assemblies", "20261119-120000", "state.json"))["step"], "done")
        self.assertEqual(set(read(os.path.join(D, "clothes-library", "A", "cutout.json"))["parts"]), {"skirt-2", "belt"})
        self.assertEqual(os.readlink(os.path.join(c, "library", "A")), os.path.join(D, "face-library", "A"))
        s = read(os.path.join(self.inst, "settings.json"))["workshop"]
        self.assertEqual(s, {"cache": "{data}/workshop-vroid-base", "body": "{data}/vroid-base/body.pmx"})
        # the backup keeps its names and contents
        with open(os.path.join(self.saved, "base-vroid", "corps.pmx"), "rb") as f:
            self.assertEqual(M.pmx_textures(f.read())[3], ["peau.png", "F00_000_Body_00.png"])
        self.assertEqual(read(os.path.join(self.saved, "atelier-base-vroid", "pieces", "A__jupe-2", "meta.json"))["type"],
                         "jupe")
        self.assertEqual(M.check(D), [])
        self.assertTrue(M.check(self.saved))
        self.assertEqual(read(os.path.join(c, "hairstyles", "Free", "meta.json"))["fingerprint"],   # inside a signature
                         "v9:A:" + M.fingerprint(os.path.join(D, "vroid-base", "hairstyle.pmx")))
        # again: nothing to do, nothing changes
        before = read(os.path.join(c, "parts", "A__skirt-2", "meta.json"))
        self.assertEqual(M.main([self.D]), 0)
        self.assertEqual(read(os.path.join(c, "parts", "A__skirt-2", "meta.json")), before)

    def test_resumes_after_cutoff(self):
        mig = M.Migration(self.D)
        for phase in M.PHASES[:3]:                       # cut after the records
            mig.note(phase, **getattr(mig, "phase_" + phase.replace("-", "_"))())
        self.assertEqual(M.Migration(self.D).run(), 0)
        self.assertEqual(M.check(self.D), [])
        m = read(os.path.join(self.D, "workshop-vroid-base", "parts", "A__skirt-2", "meta.json"))
        self.assertEqual(m["body"], M.fingerprint(os.path.join(self.D, "vroid-base", "body.pmx")))

    def test_folder_linked_elsewhere(self):
        """base-vroid is a link to another installation: that one is never written; this one gets its own copy
        (hard links), also when it was converted earlier without it."""
        elsewhere = os.path.join(self.inst, "autre", "base-vroid")
        os.makedirs(os.path.dirname(elsewhere))
        os.rename(os.path.join(self.D, "base-vroid"), elsewhere)
        os.symlink(elsewhere, os.path.join(self.D, "base-vroid"))
        e = M.fingerprint(os.path.join(self.D, "base-vroid", "corps.pmx"))      # as prepare saw it: through the link
        for d, _, fs in os.walk(self.D):
            for x in fs:
                if x.endswith(".json"):
                    q = os.path.join(d, x)
                    with open(q) as f:
                        t = f.read()
                    os.remove(q)                                                # not through the cp -al backup
                    with open(q, "w") as f:
                        f.write(t.replace(self.e_body, e))
        real = M.Migration.links_outside
        M.Migration.links_outside = lambda self: []      # the converter of before: it left the link
        try:
            self.assertEqual(M.main([self.D]), 0)
        finally:
            M.Migration.links_outside = real
        self.assertEqual(M.main([self.D]), 0)
        body = os.path.join(self.D, "vroid-base", "body.pmx")
        self.assertFalse(os.path.islink(os.path.join(self.D, "vroid-base")))
        self.assertTrue(os.path.isfile(os.path.join(elsewhere, "corps.pmx")))           # the other one untouched
        self.assertEqual(sorted(os.listdir(elsewhere)), ["coiffure.pmx", "corps.pmx", "peau.png"])
        self.assertTrue(os.path.islink(os.path.join(self.D, ".chimera-migrate-links", "base-vroid")))   # kept
        m = read(os.path.join(self.D, "workshop-vroid-base", "parts", "A__skirt-2", "meta.json"))
        self.assertEqual(m["body"], M.fingerprint(body))
        self.assertEqual(M.check(self.D), [])

    def test_cache_outside_converted_after(self):
        """A cache kept outside the data folder, converted after it: its links into the data folder follow the
        new names; a settings path written with ~ is renamed too (only its part inside the data folder)."""
        outside = os.path.join(self.inst, "cache-dehors")
        os.makedirs(os.path.join(outside, "biblio"))
        os.symlink(os.path.join(self.D, "bibliotheque-visage", "A"), os.path.join(outside, "biblio", "A"))
        write(os.path.join(outside, "tetes", "A", "meta.json"), {"id": "A"})
        settings = os.path.join(self.inst, "autres-settings.json")
        write(settings, {"workshop": {"body": "~/donnees/base-vroid/corps.pmx", "cache": "~/cache-dehors"}})
        home = os.environ.get("HOME")
        os.environ["HOME"] = self.inst
        try:
            self.assertEqual(M.main([self.D, "--settings", settings]), 0)
        finally:
            os.environ["HOME"] = home
        self.assertEqual(read(settings)["workshop"], {"body": "~/donnees/vroid-base/body.pmx", "cache": "~/cache-dehors"})
        self.assertTrue(M.check(outside))                       # the link is broken, its new target exists
        self.assertEqual(M.main([outside]), 0)
        self.assertEqual(os.readlink(os.path.join(outside, "library", "A")), os.path.join(self.D, "face-library", "A"))
        self.assertEqual(M.check(outside), [])

    def test_settings_of_games_follow(self):
        """A games.<game> setting written with the description's old keys gets its new names, even once the
        data folder is in format 2."""
        settings = os.path.join(self.inst, "jeux-settings.json")
        write(settings, {"games": {"active": "zzz", "zzz": {"mmd_extrait": "/x"},
                                   "outerplane": {"personnages": "/y", "textures_luna": "/z"}}})
        os.makedirs(self.D, exist_ok=True)
        write(os.path.join(self.D, F.MARKER), {"format": 2})
        mig = M.Migration(self.D, settings)
        mig.j["done"] = list(M.PHASES)
        self.assertEqual(mig.run(), 0)
        self.assertEqual(read(settings)["games"], {"active": "zzz", "zzz": {"mmd_extracted": "/x"},
                                                   "outerplane": {"people": "/y", "textures_luna": "/z"}})

    def test_keys_of_a_record(self):
        x = {"nom": "A", "controles": [{"nom": "sous_la_peau", "ok": False, "seuil": 1}], "etape": "fin",
             "pieces": {"jupe": {"faces": 1}}, "meta_vrm": {"nom": "kept"}, "par_membre": {"右bras": 1}, "passe_2": 1}
        e = M.conv(x)
        self.assertEqual(e["checks"][0]["name"], "under_skin")
        self.assertEqual(e["step"], "done")
        self.assertEqual(e["vrm_meta"], {"nom": "kept"})
        self.assertEqual(e["per_limb"], {"右arm": 1})
        self.assertEqual(len(set(F.KEYS.values())), len(F.KEYS))           # one English name per French name
        self.assertFalse(set(F.KEYS) & {v for k, v in F.KEYS.items() if k != v})


class Revision1(unittest.TestCase):
    """A format-2 folder converted before revisions 1 and 2: only the names inside files change, the backup keeps its own."""
    def test_inside_files(self):
        import pickle
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            D = os.path.join(tmp, "data")
            write(os.path.join(D, F.MARKER), {"format": 2})
            write(os.path.join(D, LOG_NAME), {"done": [p for p in M.PHASES if p not in ("revision-1", "revision-2")]})
            rig = os.path.join(D, "body", "own-rig")
            write(os.path.join(rig, "skeleton.json"), {"unite": "cm", "repere": "y", "torsion": "t", "bones": [
                {"name": "racine", "parent": None, "torsion": None, "mesure": None},
                {"name": "bras_torsion1_G", "parent": "bras_G", "mesure": None,
                 "torsion": {"sort": "retient", "source": "bras_G", "axis": [1, 0, 0], "fraction": 0.5}}]})
            np.savez_compressed(os.path.join(rig, "weights.npz"), poids=np.eye(2), os=np.array(["racine", "bras_torsion1_G"]))
            np.savez_compressed(os.path.join(D, "hairstyle.npz"), coiffure=np.ones(3, bool), oreilles=np.zeros(3, bool), os=np.ones(2, bool))
            with open(os.path.join(D, "body", "shape.pkl"), "wb") as f:
                pickle.dump({"P": np.zeros((2, 3)), "P_avant": np.zeros((2, 3)), "quads": [], "cartes": [("bras_D", [(0, 1)]), None],
                             "zone": np.array(["corps", "aucune"]), "info": {"doigts_debut_cm": {"pouce": 3.9}}}, f)
            write(os.path.join(D, "meta.json"), {"checks": [{"name": "waist_outline"}], "reason": "waist_outline = 3"})
            mtime = os.stat(os.path.join(D, "hairstyle.npz")).st_mtime_ns
            os.system("cp -al '%s' '%s'" % (D, D + ".format-1"))
            self.assertEqual(F.revision_of(D), 0)
            self.assertTrue(M.check(D))
            self.assertEqual(M.main([D]), 0)
            self.assertEqual(M.check(D), [])
            self.assertEqual(F.revision_of(D), F.REVISION)
            sk = read(os.path.join(rig, "skeleton.json"))
            self.assertEqual(list(sk)[:3], ["unit", "frame", "twist"])
            b = sk["bones"][1]
            self.assertEqual((b["name"], b["parent"], b["twist"]["sort"], b["twist"]["source"]),
                             ("arm_twist1_L", "arm_L", "holds", "arm_L"))
            self.assertIn("measure", b)
            w = np.load(os.path.join(rig, "weights.npz"))
            self.assertEqual(sorted(w.files), ["bones", "weights"])
            self.assertEqual(list(w["bones"]), ["root", "arm_twist1_L"])
            self.assertEqual(sorted(np.load(os.path.join(D, "hairstyle.npz")).files), ["bones", "ears", "hairstyle"])
            self.assertEqual(os.stat(os.path.join(D, "hairstyle.npz")).st_mtime_ns, mtime)
            with open(os.path.join(D, "body", "shape.pkl"), "rb") as f:
                g = pickle.load(f)
            self.assertEqual(g["maps"][0][0], "arm_R")
            self.assertEqual(list(g["zone"]), ["body", "none"])
            self.assertEqual(g["info"], {"digits_start_cm": {"thumb": 3.9}})
            self.assertIn("P_before", g)
            self.assertEqual(read(os.path.join(D, "meta.json"))["checks"][0]["name"], "outline_size")
            # the backup (other names of the same files) is untouched
            self.assertEqual(sorted(np.load(os.path.join(D + ".format-1", "hairstyle.npz")).files), ["coiffure", "oreilles", "os"])
            self.assertEqual(M.main([D]), 0)                 # a second run changes nothing


class Revision2(unittest.TestCase):
    """A folder at revision 1: the ids, folders, built parts and morph names format 2 forgot get their English
    names; the fingerprints of moved files follow; a cache in another folder follows its data folder."""
    def test_names_forgotten(self):
        with tempfile.TemporaryDirectory() as tmp:
            D = os.path.join(tmp, "data")
            write(os.path.join(D, F.MARKER), {"format": 2, "revision": 1})
            write(os.path.join(D, LOG_NAME), {"done": [p for p in M.PHASES if p not in ("revision-2", "marker")]})
            c = os.path.join(D, "workshop-vroid-base")
            write(os.path.join(c, "catalog.json"), {"heads": [{"id": "Libre"}, {"id": "A"}]})
            write(os.path.join(c, "labels.json"), {"heads/Libre": {"name": ["Free", "Libre"]}})
            write(os.path.join(c, "heads", "Libre", "meta.json"), {"id": "Libre"})
            hair = os.path.join(D, "vroid-base", "hairstyle.pmx")
            write(hair, pmx(["skin.png"]))
            write(os.path.join(c, "hairstyles", "Libre", "meta.json"),
                  {"id": "Libre", "fingerprint": "v9:A,Libre,Zed:" + M.fingerprint(hair)})
            write(os.path.join(c, "hairstyles", "Libre", "Libre.bin"), b"bin")
            body = os.path.join(D, "clothes", "tenue", "corps_refait", "body.pmx")
            write(body, pmx_morphs(["forme_waist", "smile"]))
            write(os.path.join(D, "clothes", "tenue-maelys-2", "result.pmx"), pmx(["skin.png"]))
            write(os.path.join(D, "trials", "maelys", "essais", "sur-A-cheveux-B", "x.json"), {"a": 1})
            built = os.path.join(c, "fabriquees", "Maelys", "debardeur")
            write(os.path.join(built, "debardeur.pmx"), pmx(["skin.png"]))
            src = os.path.join(D, "clothes", "tenue-maelys-2", "result.pmx")
            write(os.path.join(built, "record.json"), {"atelier": {"source": src, "fingerprint": M.fingerprint(src)},
                                                       "built": M.fingerprint(os.path.join(built, "debardeur.pmx"))})
            write(os.path.join(c, "parts", "Maelys__debardeur", "meta.json"), {"id": "Maelys__debardeur"})
            # a cache of its own, outside, made from this folder's outfit
            out = os.path.join(tmp, "outside")
            write(os.path.join(out, "catalog.json"), {"parts": [{"id": "Maelys__debardeur", "source": src,
                                                                  "fingerprint": M.fingerprint(src)}]})
            write(os.path.join(out, "parts", "Maelys__debardeur", "meta.json"), {"id": "Maelys__debardeur"})
            write(os.path.join(out, F.MARKER), {"format": 2, "revision": 1})
            write(os.path.join(out, LOG_NAME), {"done": [p for p in M.PHASES if p not in ("revision-2", "marker")]})
            size, mtime = os.path.getsize(body), os.stat(body).st_mtime_ns
            os.system("cp -al '%s' '%s'" % (D, D + ".format-1"))
            self.assertTrue(M.check(D))
            with self.assertRaises(SystemExit):              # the cache before its data folder: refused, untouched
                M.main([out])
            self.assertEqual(read(os.path.join(out, "catalog.json"))["parts"][0]["source"], src)
            self.assertEqual(M.main([D]), 0)
            self.assertEqual(M.main([out]), 0)
            self.assertEqual(M.check(D), [])
            self.assertEqual(M.check(out), [])
            self.assertEqual(F.revision_of(D), 2)
            self.assertEqual(read(os.path.join(c, "catalog.json"))["heads"][0]["id"], "Free")
            self.assertEqual(list(read(os.path.join(c, "labels.json"))), ["heads/Free"])
            self.assertEqual(read(os.path.join(c, "heads", "Free", "meta.json")), {"id": "Free"})
            nhair = os.path.join(D, "vroid-base", "hairstyle.pmx")
            self.assertEqual(read(os.path.join(c, "hairstyles", "Free", "meta.json"))["fingerprint"],
                             "v9:A,Free,Zed:" + M.fingerprint(nhair))
            self.assertTrue(os.path.isfile(os.path.join(c, "hairstyles", "Free", "Free.bin")))
            nbody = os.path.join(D, "clothes", "outfit", "rebuilt-body", "body.pmx")
            self.assertEqual((os.path.getsize(nbody), os.stat(nbody).st_mtime_ns), (size, mtime))
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "atlas"))
            import pmx as PMX
            self.assertEqual([m["name"] for m in PMX.read(nbody)["morphs"]], ["shape_waist", "smile"])
            self.assertTrue(os.path.isdir(os.path.join(D, "trials", "maelys", "trials", "on-A-hair-B")))
            nsrc = os.path.join(D, "clothes", "outfit-maelys-2", "result.pmx")
            nbuilt = os.path.join(c, "built", "Maelys", "tank_top")
            r = read(os.path.join(nbuilt, "record.json"))
            self.assertEqual(r["workshop"], {"source": nsrc, "fingerprint": M.fingerprint(nsrc)})
            self.assertEqual(r["built"], M.fingerprint(os.path.join(nbuilt, "tank_top.pmx")))
            self.assertTrue(os.path.isdir(os.path.join(c, "parts", "Maelys__tank_top")))
            p = read(os.path.join(out, "catalog.json"))["parts"][0]
            self.assertEqual(p, {"id": "Maelys__tank_top", "source": nsrc, "fingerprint": M.fingerprint(nsrc)})
            # the backup keeps its names, and the old morph names
            old = os.path.join(D + ".format-1", "clothes", "tenue", "corps_refait", "body.pmx")
            self.assertEqual([m["name"] for m in PMX.read(old)["morphs"]], ["forme_waist", "smile"])
            self.assertEqual(read(os.path.join(D + ".format-1", "workshop-vroid-base", "catalog.json"))["heads"][0]["id"],
                             "Libre")
            before = read(os.path.join(nbuilt, "record.json"))
            self.assertEqual(M.main([D]), 0)                 # a second run changes nothing
            self.assertEqual(read(os.path.join(nbuilt, "record.json")), before)


LOG_NAME = M.LOG


if __name__ == "__main__":
    unittest.main()

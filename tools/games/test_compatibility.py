"""The compatibility declarations are complete and COMPATIBILITY.md still says what they say.

    <python> -I tools/games/test_compatibility.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compatibility as K  # noqa: E402


class Compatibilite(unittest.TestCase):
    def test_declarations_completes(self):
        self.assertTrue(K.declares())
        for n in K.declares():
            self.assertEqual(K.erreurs(n), [], n)

    def test_document_a_jour(self):
        self.assertEqual(K.verifier(), [])

    def test_section_sans_etat_de_la_machine(self):
        # the document must depend only on the descriptions, not on the settings or the disk of the machine
        t = K.tableau()
        self.assertNotIn(os.path.expanduser("~"), t)
        self.assertNotIn("present", t)

    def test_aucun_lien_de_telechargement(self):
        self.assertNotIn("http", K.tableau())


if __name__ == "__main__":
    unittest.main()

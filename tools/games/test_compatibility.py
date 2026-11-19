"""The compatibility declarations are complete and COMPATIBILITY.md still says what they say.

    <python> -I tools/games/test_compatibility.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compatibility as K  # noqa: E402


class Compatibility(unittest.TestCase):
    def test_declarations_complete(self):
        self.assertTrue(K.declared())
        for n in K.declared():
            self.assertEqual(K.errors(n), [], n)

    def test_document_a_day(self):
        self.assertEqual(K.check(), [])

    def test_section_without_state_of_machine(self):
        # the document must depend only on the descriptions, not on the settings or the disk of the machine
        t = K.array()
        self.assertNotIn(os.path.expanduser("~"), t)
        self.assertNotIn("present", t)

    def test_none_link_of_download(self):
        self.assertNotIn("http", K.array())


if __name__ == "__main__":
    unittest.main()

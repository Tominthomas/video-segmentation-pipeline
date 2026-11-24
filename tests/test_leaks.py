import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaks import pattern, secret_terms  # noqa: E402


class Leaks(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        content = Path(tmp.name)
        entry = {"id": "doe-2026-den-drive", "player": "Jane Doe", "title": "Drive", "url": "https://x.com/a/status/1"}
        junior = {**entry, "id": "jackson-2026-mem-block", "player": "Jaren Jackson Jr."}
        (content / "clips-doe.json").write_text(json.dumps([entry, junior]))
        (content / "judgments.json").write_text(json.dumps({"clips": {}}))
        (content / "approved").mkdir()
        (content / "approved" / "quill-2017-okc-drive.mp4").touch()
        self.found = pattern(secret_terms(content))

    def hits(self, text: str) -> list[str]:
        return [m[0] for m in self.found.finditer(text)]

    def test_names_ids_and_id_prefixes_are_found(self):
        self.assertEqual(self.hits("Jane Doe's drive"), ["Jane Doe"])
        self.assertEqual(self.hits("Doe's close clip"), ["Doe"])
        self.assertEqual(self.hits("`doe-2026-den-drive`"), ["doe-2026-den-drive"])
        self.assertEqual(self.hits("`doe-2025-gsw-iso`"), ["doe-"])
        self.assertEqual(self.hits("quill-2017-okc-drive.mp4"), ["quill-2017-okc-drive"])
        self.assertEqual(self.hits("Jackson's block"), ["Jackson"])

    def test_the_public_fixture_and_ordinary_words_are_not(self):
        self.assertEqual(self.hits("Ray Allen's corner three, ray-allen-2013-finals-g6-corner-three"), [])
        self.assertEqual(self.hits("Doesn't matter, a doe in the den-drive"), [])

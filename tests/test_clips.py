import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clips import SAM_PINS, TRACK_VERSION, sam_missing  # noqa: E402

# sam-requirements.txt's sha256 at each TRACK_VERSION. New pins can change the masks track.py writes, and old masks
# stay cached under the same TRACK_VERSION, so changing the pins means bumping TRACK_VERSION and adding a row here.
PINS_AT = {1: "a52d461a7021fea93dbde7e22d31c662f6453fffa79f6475b957b3966478861a"}


def fake_venv(root: Path, pins: list[str]) -> Path:
    """A venv whose site-packages holds one dist-info per pip freeze line, with direct_url.json for a git pin."""
    venv = root / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()
    for line in pins:
        if " @ git+" in line:
            name, url = line.split(" @ git+")
            url, commit = url.rsplit("@", 1)
            version = "1.0"
        else:
            name, version = line.split("==")
        info = venv / "lib" / "python3.14" / "site-packages" / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        if " @ git+" in line:
            (info / "direct_url.json").write_text(
                json.dumps({"url": url, "vcs_info": {"vcs": "git", "commit_id": commit}})
            )
    return venv


class SamPins(unittest.TestCase):
    def test_pin_change_bumps_track_version(self):
        self.assertEqual(
            PINS_AT.get(TRACK_VERSION),
            hashlib.sha256(SAM_PINS.read_bytes()).hexdigest(),
            "sam-requirements.txt changed: bump TRACK_VERSION in clips.py and add the new sha256 to PINS_AT",
        )

    def test_install_must_match_pins(self):
        pins = SAM_PINS.read_text().splitlines()
        drifted = [line.replace("torch==2.15.0", "torch==2.16.0") for line in pins]
        moved = [line.replace("@2b90b9f5", "@deadbeef") for line in pins]
        with tempfile.TemporaryDirectory() as tmp:

            def missing(name: str, lines: list[str]) -> str | None:
                return sam_missing(fake_venv(Path(tmp) / name, lines))

            self.assertIsNone(missing("pinned", pins))
            self.assertIn(
                "has torch==2.16.0, but sam-requirements.txt pins torch==2.15.0", str(missing("drifted", drifted))
            )
            self.assertIn("@deadbeef", str(missing("moved", moved)))
            self.assertIn(
                "has no torch,", str(missing("short", [line for line in pins if not line.startswith("torch==")]))
            )
            self.assertIn("no SAM venv", str(sam_missing(Path(tmp) / "absent")))


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clips import SAM_PINS, TRACK_VERSION, sam_missing  # noqa: E402

# New pins can change the masks track.py writes, and old masks stay cached under the same TRACK_VERSION, so changing
# the pins means bumping TRACK_VERSION and adding a row here.
PIN_SHA256_BY_TRACK_VERSION = {1: "a52d461a7021fea93dbde7e22d31c662f6453fffa79f6475b957b3966478861a"}


def fake_venv(root: Path, pins: list[str], python: str = "3.14") -> Path:
    venv = root / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python").touch()
    (venv / "pyvenv.cfg").write_text("home = /opt/homebrew/bin\nversion = 3.14.3\n")
    for line in pins:
        if " @ git+" in line:
            name, url = line.split(" @ git+")
            url, commit = url.rsplit("@", 1)
            version = "1.0"
        else:
            name, version = line.split("==")
        info = venv / "lib" / f"python{python}" / "site-packages" / f"{name}-{version}.dist-info"
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
            PIN_SHA256_BY_TRACK_VERSION.get(TRACK_VERSION),
            hashlib.sha256(SAM_PINS.read_bytes()).hexdigest(),
            "sam-requirements.txt changed: bump TRACK_VERSION in clips.py and add a row for its sha256",
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

            stale = fake_venv(Path(tmp) / "stale", drifted)
            fake_venv(Path(tmp) / "stale", pins, python="3.13")
            self.assertIn("has torch==2.16.0", str(sam_missing(stale)))

            editable = fake_venv(Path(tmp) / "editable", pins)
            torch = next((editable / "lib" / "python3.14" / "site-packages").glob("torch-*.dist-info"))
            (torch / "direct_url.json").write_text(
                json.dumps({"url": "file:///src/torch", "dir_info": {"editable": True}})
            )
            self.assertIn("has torch @ file:///src/torch", str(sam_missing(editable)))


if __name__ == "__main__":
    unittest.main()

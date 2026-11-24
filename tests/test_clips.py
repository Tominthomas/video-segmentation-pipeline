import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import clips  # noqa: E402
from clips import SAM_PINS, TRACK_VERSION, sam_missing  # noqa: E402

# New pins can change the masks track.py writes, and old masks stay cached under the same TRACK_VERSION, so changing
# the pins means bumping TRACK_VERSION and adding a row here.
# Version 2 changed the SAM model from small to large under the same pins.
PIN_SHA256_BY_TRACK_VERSION = {
    1: "a52d461a7021fea93dbde7e22d31c662f6453fffa79f6475b957b3966478861a",
    2: "a52d461a7021fea93dbde7e22d31c662f6453fffa79f6475b957b3966478861a",
}


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


class ShapeRecipes(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        self.code = tmp / "code"
        self.code.mkdir()
        for script in ROOT.glob("*.py"):
            shutil.copy(script, self.code)
        folder = tmp / "clip"
        folder.mkdir()
        self.art = clips.Artifacts(folder)
        self.art.ledger["source.mp4"] = "source"
        self.clip = clips.Clip(id="c", player="p", title="t", url="u", track={"objects": [{}, {}]})
        self.objects = [{"kind": "player", "cache": str(folder / "objects" / f"{n}.npz")} for n in "ab"]
        self.runs = []
        for step in ("detail_step", "matte_step"):
            patch = mock.patch.object(clips, step, lambda *args, step=step: self.ran(step, args[-1]))
            patch.start()
            self.addCleanup(patch.stop)
        for name, path in (("DETAIL", self.code / "detail.py"), ("MATTE", self.code / "matte.py")):
            patch = mock.patch.object(clips, name, path)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(clips, "matte_missing", lambda: None)
        patch.start()
        self.addCleanup(patch.stop)

    def ran(self, step: str, out: Path) -> None:
        self.runs.append(step)
        out.write_bytes(b"model output")

    def build(self) -> list:
        """Make both shape artifacts, and return what the qc.json and silhouette.mp4 recipes take from each."""
        render = {"mystery": [0]}
        with contextlib.redirect_stdout(io.StringIO()):
            detail = clips.with_detail(self.clip, self.art, Path("source.mp4"), render, self.objects)["detail"]
            matte = clips.with_matte(self.clip, self.art, Path("source.mp4"), render, self.objects)["matte"]
        return [clips.shape_inputs(self.art, detail), clips.shape_inputs(self.art, matte)]

    def edit(self, script: str, old: str, new: str) -> None:
        path = self.code / script
        text = path.read_text()
        self.assertIn(old, text)
        path.write_text(text.replace(old, new))

    def test_tuning_a_render_rule_re_renders_without_re_running_a_model(self):
        before = self.build()
        self.edit("detail.py", "SURE = 0.65", "SURE = 0.7")
        self.edit("matte.py", "LEVELS = (0.12, 0.5)", "LEVELS = (0.1, 0.5)")
        self.edit("matte.py", "GATE = 0.04", "GATE = 0.05")
        after = self.build()
        self.assertEqual(self.runs, ["detail_step", "matte_step"])
        self.assertNotEqual(before[0], after[0])
        self.assertNotEqual(before[1], after[1])

    def test_a_comment_docstring_or_unreached_function_re_runs_nothing(self):
        self.build()
        self.edit(
            "detail.py",
            '    kinds, masks = load(Path(spec["masks"]))\n',
            '    # A note.\n    kinds, masks = load(Path(spec["masks"]))\n',
        )
        self.edit("matte.py", '"""The model\'s alpha on every crop as uint8.', '"""The alpha on every crop as uint8.')
        self.edit("silhouette.py", "def dilate(", "def unused_by_models():\n    return 0\n\n\ndef dilate(")
        self.build()
        self.assertEqual(self.runs, ["detail_step", "matte_step"])

    def test_changing_what_a_model_step_reaches_re_runs_that_model(self):
        self.build()
        self.edit("detail.py", "BOX_PAD = 0.04", "BOX_PAD = 0.05")
        self.build()
        self.assertEqual(self.runs, ["detail_step", "matte_step", "detail_step"])
        self.edit("matte.py", "MODEL_SIDE = 960", "MODEL_SIDE = 1024")
        self.build()
        self.assertEqual(self.runs[3:], ["matte_step"])

    def test_changing_a_helper_both_model_steps_import_re_runs_both(self):
        self.build()
        self.edit("silhouette.py", "    if radius <= 0 or not len(ys):", "    if radius < 0 or not len(ys):")
        self.build()
        self.assertEqual(self.runs, ["detail_step", "matte_step", "detail_step", "matte_step"])

    def test_code_a_model_step_reaches_indirectly_re_runs_it(self):
        self.build()
        self.edit("matte.py", "MODEL_SIDE = 960\n", "MODEL_SIDE = 960\nMODEL_SIDE *= 2\n")
        self.build()
        self.assertEqual(self.runs[2:], ["detail_step", "matte_step"])
        make = "def make(source: Path, spec: dict, out: Path) -> None:\n"
        self.edit("detail.py", make, make + "    import matte as geometry  # noqa: PLC0415\n\n    geometry.crop_size\n")
        self.build()
        self.assertEqual(self.runs[4:], ["detail_step"])
        self.edit("matte.py", "LEVELS = (0.12, 0.5)", "LEVELS = (0.1, 0.5)")
        self.build()
        self.assertEqual(self.runs[5:], ["detail_step"])
        self.edit("detail.py", "class Detail(NamedTuple):", "@staticmethod\nclass Detail(NamedTuple):")
        self.build()
        self.assertEqual(self.runs[6:], ["detail_step"])
        self.edit("matte.py", "def frame_map(", "READY = print()\n\n\ndef frame_map(")
        self.build()
        self.assertEqual(self.runs[7:], ["detail_step", "matte_step"])

    def test_a_method_the_model_step_calls_re_runs_it_after_any_module_statement(self):
        guard = 'if __name__ == "__main__":'
        self.edit("detail.py", guard, 'np.seterr(invalid="ignore")\n\n\n' + guard)
        self.build()
        self.edit("detail.py", "**self._asdict())", "**self._asdict(), first=999)")
        self.build()
        self.assertEqual(self.runs, ["detail_step", "matte_step", "detail_step"])


if __name__ == "__main__":
    unittest.main()


class Manifest(unittest.TestCase):
    def load(self, entries: object) -> str:
        """The error load_manifest exits with on entries, written as JSON or as raw text."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clips.json"
            path.write_text(entries if isinstance(entries, str) else json.dumps(entries))
            with self.assertRaises(SystemExit) as stop:
                clips.load_manifest(path)
            return str(stop.exception.code)

    def entry(self, **fields: object) -> dict:
        return {"id": "doe-drive", "player": "Jane Doe", "title": "Drive", "url": "https://x.com/a/status/1", **fields}

    def test_bad_json_names_the_file(self):
        self.assertIn("clips.json", self.load("[{"))

    def test_an_id_that_leaves_its_folder_is_refused(self):
        self.assertIn("may hold only", self.load([self.entry(id="../doe")]))

    def test_a_track_with_no_objects_is_refused(self):
        self.assertIn("at least one object", self.load([self.entry(track={"frame": 0, "objects": []})]))


class Approve(unittest.TestCase):
    def test_an_unreadable_render_is_refused_and_the_rest_still_run(self):
        def ffprobe_fails(cmd, check=False, **_):
            """subprocess.run as it behaves when ffprobe can't read the file, so CI needs no ffprobe."""
            if check:
                raise subprocess.CalledProcessError(1, cmd, stderr="Invalid data found")
            return subprocess.CompletedProcess(cmd, 1, "", "Invalid data found")

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(clips, "MEDIA", Path(tmp)),
            mock.patch.object(clips.subprocess, "run", ffprobe_fails),
        ):
            for name in ("broken", "missing-next"):
                (Path(tmp) / "clips" / name).mkdir(parents=True)
            (Path(tmp) / "clips" / "broken" / "silhouette.mp4").write_bytes(b"not a video")
            pair = [clips.Clip(id=i, player="p", title="t", url="u") for i in ("broken", "missing-next")]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ok = clips.approve(pair, Path(tmp) / "approved")
        self.assertFalse(ok)
        self.assertIn("broken: FAIL ffprobe cannot read it", out.getvalue())
        self.assertIn("missing-next: FAIL no", out.getvalue())

#!/usr/bin/env python3
"""Fetch NBA highlight clips and render them as silhouette clues for Bagwork.

  python3 clips.py scout <url>                         frames every second, to pick start/end
  python3 clips.py add --manifest FILE <url> --player NAME [--id ID]   scout, then add a manifest stub
  python3 clips.py build [--manifest FILE] [id ...]    fetch, track, render the silhouette, verify
  python3 clips.py approve [--manifest FILE] id ...    check each silhouette.mp4 against Bagwork's render contract,
                                                       then copy it to $BAGWORK_CONTENT/approved/<id>.mp4
  python3 clips.py preview [--manifest FILE] id        each prompt's mask, before a full track
  python3 clips.py window [--manifest FILE] id --frames 140-170:3 --objects 11
                                                                objects tracked over a few frames, to check a refine
  python3 clips.py propose [--manifest FILE] id --frame N   numbered player and ball boxes to prompt by
  python3 clips.py review [--manifest FILE] id         build, then a local review.html with overlay and QC
  python3 clips.py crops [--manifest FILE] id --frames 140-160:4,171 --box x0,y0,x1,y1
                                                                gridded zooms of a region, to read ball_keys off
  python3 clips.py crops [--manifest FILE] id --frames 40-70:3 --detail
                                                                the detail pass's crops, to read detail refines off

The manifest defaults to fixtures/clips.json, the public sample clips. Real puzzles keep theirs outside the repo, so
add, which writes to the manifest, needs --manifest.

Needs yt-dlp and ffmpeg on PATH. Output goes to media/, which is git-ignored.
"""

import argparse
import ast
import copy
import fcntl
import hashlib
import html
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SAMPLE_MANIFEST = ROOT / "fixtures" / "clips.json"
MEDIA = ROOT / "media"

HEIGHT = 720
FPS = 30
SCALE = f"scale=-2:{HEIGHT},fps={FPS}"
ENCODE = [
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
    "-g",
    str(FPS // 2),
    "-movflags",
    "+faststart",
    "-map_metadata",
    "-1",
    "-map_chapters",
    "-1",
]

# SAM 2 and every pipeline script run in one venv outside the checkout, so every worktree shares one install.
SAM_VENV = Path.home() / ".local" / "share" / "bagwork" / "sam" / "venv"
SAM_PYTHON = SAM_VENV / "bin" / "python"
SAM_PINS = ROOT / "sam-requirements.txt"
TRACK = ROOT / "track.py"
SILHOUETTE = ROOT / "silhouette.py"
QC = ROOT / "qc.py"
PROPOSE = ROOT / "propose.py"
REVIEW = ROOT / "review.py"
CROPS = ROOT / "crops.py"
MATTE = ROOT / "matte.py"
DETAIL = ROOT / "detail.py"
# MatAnyone 2 lives outside the checkout, in its own venv, so every worktree shares one install (docs/SPEC.md#matting).
MATTE_HOME = Path.home() / ".local" / "share" / "bagwork" / "matte"
MATTE_PYTHON = MATTE_HOME / "venv" / "bin" / "python"
MATANYONE = MATTE_HOME / "MatAnyone2"
MATANYONE_COMMIT = "0079197acd6d16a741f71558809c06c586c579e0"
MATTE_WEIGHTS = MATTE_HOME / "matanyone2.pth"
MATTE_WEIGHTS_SHA256 = "5e9821e4087231427376b437c85bb6e072b41e582314f06fd524f75bc4af5914"
TRACK_KEYS = ("frame", "objects")
# Part of every tracked object's cache key. Bump it when a track.py or sam-requirements.txt change alters the masks it
# writes, a change of SAM model included. tests/test_clips.py fails when the pins change and it does not.
TRACK_VERSION = 2
# How the mystery player's shape is drawn (docs/SPEC.md#python-clip-pipeline). "auto" picks "detail" or "matte" by
# his size.
SHAPES = ("auto", "detail", "matte", "sam")
# SAM shares one GPU and most of the memory, so tracking runs one job at a time and only with this much free.
# The lock lives outside the checkout so builds from every worktree on the machine share it.
SAM_LOCK = Path.home() / ".local" / "state" / "bagwork" / "sam.lock"
# Waiters open the lock with "w", which truncates it, so the holder says who it is in a file of its own.
SAM_HOLDER = SAM_LOCK.with_name("sam-holder.json")
WAIT_REPORT = 30
MIN_FREE = int(os.environ.get("BAGWORK_MIN_FREE", "30"))

# The best rendition fetch downloads. add reports its size, so a source is judged by what the build will use. A direct
# link to a video file reports no height, so it falls through to the file itself.
FETCH_FORMAT = "bv*[height<=1080]+ba/b[height<=1080]/b"
SCENE_SCORE_CUT = 0.4
TOOLS = ("ffmpeg", "ffprobe", "yt-dlp")

# Each clue variant hides faces and jersey text with one whole-frame look.
# The blur runs before recoloring so small digits smear instead of turning into crisp edges.
FILTERS = {
    "thermal": "format=gray,gblur=sigma=4,format=yuv420p,pseudocolor=p=inferno",
}


@dataclass(frozen=True)
class Crop:
    """A fixed-size window that slides linearly between (t, x, y) points, in 720p frame coordinates."""

    width: int
    height: int
    path: tuple[tuple[float, int, int], ...]

    def filter(self) -> str:
        def along(axis: int) -> str:
            points = [(t, pos[axis]) for t, *pos in self.path]
            expr = str(points[-1][1])
            for (t0, v0), (t1, v1) in reversed(list(pairwise(points))):
                expr = f"if(lt(t\\,{t1})\\,{v0}+({v1}-{v0})*(t-{t0})/{t1 - t0}\\,{expr})"
            return f"if(lt(t\\,{points[0][0]})\\,{points[0][1]}\\,{expr})"

        return f"crop={self.width}:{self.height}:{along(0)}:{along(1)},scale=-2:{HEIGHT}"


@dataclass(frozen=True)
class Clip:
    id: str
    player: str
    title: str
    url: str
    start: float | None = None
    end: float | None = None
    hide: tuple[tuple[int, int, int, int], ...] = ()
    crop: Crop | None = None
    track: dict | None = None


# An id names the clip's folder and its approved file, so it must be one plain path component.
CLIP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def parse_crop(raw: dict) -> Crop:
    path = tuple(sorted((float(t), int(x), int(y)) for t, x, y in raw["path"]))
    if not path:
        raise TypeError("crop.path needs at least one [t, x, y] point")
    return Crop(width=int(raw["size"][0]), height=int(raw["size"][1]), path=path)


def load_manifest(path: Path) -> list[Clip]:
    if not path.is_file():
        sys.exit(f"no manifest at {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"{path}: {e}")
    clips = []
    for i, entry in enumerate(raw):
        try:
            fields: dict[str, Any] = {
                **entry,
                "hide": tuple(tuple(box) for box in entry.get("hide", [])),
                "crop": parse_crop(entry["crop"]) if "crop" in entry else None,
                # Sorted keys, so a render recipe built from the spec doesn't change with the manifest's key order.
                "track": json.loads(json.dumps(entry["track"], sort_keys=True)) if "track" in entry else None,
            }
            clip = Clip(**fields)
        except TypeError as e:
            sys.exit(f"{path.name} entry {i}: {e}")
        if not CLIP_ID.fullmatch(clip.id):
            sys.exit(
                f"{path.name} entry {i}: id {clip.id!r} may hold only letters, digits, dots, hyphens and underscores"
            )
        if clip.track is not None and not clip.track.get("objects"):
            sys.exit(f"{clip.id}: track needs at least one object")
        if (clip.start is None) != (clip.end is None):
            sys.exit(f"{clip.id}: give both start and end, or neither for the whole video")
        if clip.start is not None and clip.end is not None and not 0 <= clip.start < clip.end:
            sys.exit(f"{clip.id}: start must be >= 0 and before end")
        clips.append(clip)
    ids = [c.id for c in clips]
    if len(ids) != len(set(ids)):
        sys.exit(f"{path.name} has duplicate ids")
    return clips


def run(cmd: list[str]) -> str:
    """Run a command and return its stdout, exiting with its stderr when it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")
    return result.stdout


def probe(path: Path) -> dict:
    """ffprobe's JSON for the file. It raises CalledProcessError on a file ffprobe can't read, so approve can refuse
    that one render and go on to the next."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json", "-show_format", "-show_streams", "-show_chapters", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


class Artifacts:
    """Rebuilds an output only when its recipe changed.

    It writes through a temp file so a crash never leaves a half-written artifact.
    """

    def __init__(self, folder: Path):
        self.folder = folder
        self.ledger_path = folder / "build.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}

    def make(self, name: str, recipe: list, build) -> Path:
        target = self.folder / name
        recipe_hash = hashlib.sha256(json.dumps(recipe).encode()).hexdigest()
        if target.exists() and self.ledger.get(name) == recipe_hash:
            return target
        tmp = target.with_name(f".tmp-{target.name}")
        tmp.unlink(missing_ok=True)
        build(tmp)
        tmp.replace(target)
        self.ledger[name] = recipe_hash
        self.ledger_path.write_text(json.dumps(self.ledger, indent=2))
        print(f"  built {name}")
        return target


def fetch(clip: Clip, out: Path) -> None:
    window = (
        [] if clip.start is None else ["--download-sections", f"*{clip.start}-{clip.end}", "--force-keyframes-at-cuts"]
    )
    run(
        [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "-f",
            FETCH_FORMAT,
            *window,
            "--merge-output-format",
            "mp4",
            "-o",
            str(out),
            clip.url,
        ]
    )


def encode(src: Path, out: Path, vf: str, audio: bool) -> None:
    audio_args = ["-c:a", "aac", "-b:a", "128k"] if audio else ["-an"]
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf, *ENCODE, *audio_args, "-f", "mp4", str(out)])


def venv_step(
    clip: Clip,
    script: Path,
    src: Path,
    spec: dict,
    out: Path,
    log: Path | None = None,
    flags: tuple[str, ...] = (),
    slot: int | None = None,
    python: Path | None = None,
) -> None:
    """Run a pipeline script in a venv, the SAM venv by default, exiting on failure. A held SAM slot's fd passes to
    the script."""
    python = python or sam_python(clip)
    with tempfile.TemporaryDirectory() as tmp, Path(log or os.devnull).open("a") as log_file:
        spec_path = Path(tmp) / "spec.json"
        spec_path.write_text(json.dumps(spec))
        cmd = [str(python), str(script), *flags, str(src), str(spec_path), str(out)]
        stderr = log_file if log else None
        if subprocess.run(cmd, stderr=stderr, pass_fds=() if slot is None else (slot,), check=False).returncode != 0:
            sys.exit(f"{clip.id}: {script.name} failed" + (f", see {log}" if log else ""))


def free_memory() -> int:
    """The percentage of memory free, from memory_pressure on macOS and /proc/meminfo on Linux."""
    if sys.platform == "darwin":
        last = run(["memory_pressure"]).splitlines()[-1]
        return int(last.rsplit(" ", 1)[1].rstrip("%"))
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(100 * int(info["MemAvailable"].split()[0]) / int(info["MemTotal"].split()[0]))


def track_progress(log: Path) -> str:
    """The last group a track log says it started and how far that group's latest SAM pass has run, or the last line
    of a log with no groups, such as a matte's."""
    lines = re.split(r"[\r\n]+", log.read_text(errors="replace"))
    marks = [i for i, line in enumerate(lines) if line.startswith("tracking group")]
    if not marks:
        return next((line for line in reversed(lines) if line.strip()), "loading")
    bars = [
        m
        for line in lines[marks[-1] :]
        if (m := re.search(r"propagate in video:.*?(\d+/\d+) \[[\d:]+<([\d:?]+)", line))
    ]
    return lines[marks[-1]] + (f", pass at {bars[-1][1]} frames, {bars[-1][2]} left" if bars else "")


def sam_holder() -> str:
    """Who holds the SAM slot, from the record it left in SAM_HOLDER, with its track log's progress."""
    if not SAM_HOLDER.exists():
        return "a job that left no record"
    holder = json.loads(SAM_HOLDER.read_text())
    try:
        os.kill(holder["pid"], 0)
        pid = f"pid {holder['pid']}"
    except ProcessLookupError:
        pid = f"pid {holder['pid']} gone, its tracker still holds the slot"
    minutes, seconds = divmod(int(time.time() - holder["started"]), 60)
    log = Path(holder["log"]) if holder["log"] else None
    progress = f"; {track_progress(log)}" if log and log.exists() else ""
    return f"{holder['clip']} {holder['command']} ({pid}) for {minutes}m{seconds:02d}s{progress}"


@contextmanager
def sam_slot(clip: Clip, command: str, log: Path | None = None):
    """Hold the machine's one SAM slot and yield its fd: wait for any other SAM job, then for MIN_FREE percent free.

    A flock lasts until every process sharing its fd exits, so a killed build never leaves a stale lock. Pass the fd
    to the SAM worker, so a killed build whose worker lives on still holds the slot. While it waits, it reports
    who holds the slot every WAIT_REPORT seconds. Once it holds the slot, it records itself for the next waiter.
    """
    SAM_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with SAM_LOCK.open("w") as lock:
        reported = float("-inf")
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - reported >= WAIT_REPORT:
                    print(f"  waiting for SAM, held by {sam_holder()}", flush=True)
                    reported = time.monotonic()
                time.sleep(1)
        record = {"clip": clip.id, "command": command, "pid": os.getpid(), "started": time.time(), "log": log}
        tmp = SAM_HOLDER.with_name(f".tmp-{SAM_HOLDER.name}")
        tmp.write_text(json.dumps(record, default=str))
        tmp.replace(SAM_HOLDER)
        while (free := free_memory()) < MIN_FREE:
            print(f"  waiting for memory: {free}% free, need {MIN_FREE}%", flush=True)
            time.sleep(30)
        yield lock.fileno()


def sam_python(clip: Clip) -> Path:
    """The SAM venv's python, exiting when the venv differs from sam-requirements.txt."""
    if reason := sam_missing(SAM_VENV):
        sys.exit(f"{clip.id}: {reason}. README.md#install-sam installs it")
    return SAM_PYTHON


def pin(line: str) -> tuple[str, str]:
    """A pip freeze line as its package's normalized name and the rest, such as ("torch", "==2.15.0")."""
    name = re.split(r"[=@ ]", line, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower(), line[len(name) :]


def sam_missing(venv: Path) -> str | None:
    """Why the SAM venv can't run, or None when it holds every package at the version sam-requirements.txt pins."""
    cfg = venv / "pyvenv.cfg"
    if not (venv / "bin" / "python").exists() or not cfg.exists():
        return f"no SAM venv at {venv}"
    # Read the site-packages of the python the venv was made with, not every lib/python* left in it.
    version = re.search(r"^version\s*=\s*(\d+\.\d+)", cfg.read_text(), re.MULTILINE)
    site = venv / "lib" / f"python{version[1] if version else ''}" / "site-packages"
    installed = {}
    for dist in importlib.metadata.distributions(path=[str(site)]):
        url = json.loads(dist.read_text("direct_url.json") or "{}")
        vcs = url.get("vcs_info")
        # An install from a URL or a local dir, editable ones included, never matches a == pin.
        source = f" @ {vcs['vcs']}+{url['url']}@{vcs['commit_id']}" if vcs else f" @ {url['url']}" if url else None
        installed[pin(dist.metadata["Name"])[0]] = source or f"=={dist.version}"
    for line in SAM_PINS.read_text().splitlines():
        name, wanted = pin(line)
        if (have := installed.get(name)) != wanted:
            found = f"has {name}{have}" if have else f"has no {name}"
            return f"{venv} {found}, but sam-requirements.txt pins {name}{wanted}"
    return None


def track_objects(spec: dict, folder: Path, clip_recipe_hash: str) -> list[dict]:
    """Each object's kind, its prompts in order, and the cache file named by everything that shapes its masks."""
    objects = []
    for i, obj in enumerate(spec["objects"]):
        # A box joins the prompt only when given, so objects prompted by points alone keep their cache keys.
        first = {
            "frame": obj.get("frame", spec["frame"]),
            "points": obj.get("points", []),
            "avoid": obj.get("avoid", []),
        }
        if "box" in obj:
            first["box"] = obj["box"]
        prompts = [first, *({"points": [], "avoid": [], **refine} for refine in obj.get("refine", []))]
        if not all(p["points"] or "box" in p for p in prompts):
            sys.exit(f"object {i}: every prompt needs points or a box")
        key = hashlib.sha256(
            json.dumps([TRACK_VERSION, clip_recipe_hash, prompts], sort_keys=True).encode()
        ).hexdigest()
        objects.append({"kind": obj["kind"], "prompts": prompts, "cache": str(folder / "objects" / f"{key[:16]}.npz")})
    return objects


def track_masks(clip: Clip, art: Artifacts, video: Path, objects: list[dict], out: Path) -> None:
    """Run track.py under the SAM slot when an object needs tracking, and record what the run cost in build.json."""
    # Fail before queueing for the slot; venv_step checks again once the slot is held.
    sam_python(clip)
    missing = sum(not Path(obj["cache"]).exists() for obj in objects)
    log = art.folder / "track.log"
    # A fresh log per masks build, so the last run's progress never reads as this one's.
    if not missing:
        log.write_text(f"{time.strftime('%X')} all {len(objects)} objects cached, nothing to track\n")
        venv_step(clip, TRACK, video, {"objects": objects}, out)
        return
    log.write_text(f"{time.strftime('%X')} queued, waiting for the SAM slot\n")
    print(f"  tracking {missing} of {len(objects)} objects, progress in {log}", flush=True)
    with sam_slot(clip, "build", log) as slot:
        with log.open("a") as f:
            f.write(f"{time.strftime('%X')} slot acquired, loading SAM for {missing} of {len(objects)} objects\n")
        started = time.monotonic()
        venv_step(clip, TRACK, video, {"objects": objects}, out, log, slot=slot)
    seconds = round(time.monotonic() - started, 1)
    with log.open("a") as f:
        f.write(f"{time.strftime('%X')} finished tracking in {seconds}s\n")
    frames = int(next(s for s in probe(video)["streams"] if s["codec_type"] == "video")["nb_frames"])
    art.ledger["track"] = {"seconds": seconds, "frames": frames, "objects": missing}


# Keys in definitions() for a module's statements other than its definitions, and for all of its statements.
INIT = "<init>"


EVERYTHING = "*"


def definitions(path: Path) -> dict[str, list[ast.AST]]:
    """Each top-level name a module binds, by every statement that binds it, with docstrings dropped. INIT holds its
    other top-level statements, which run whenever it loads, less its __main__ block, and EVERYTHING all of them."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node, clean=False):
            node.body = node.body[1:] or [ast.Pass()]
    found: dict[str, list[ast.AST]] = {INIT: [], EVERYTHING: list(tree.body)}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            # A call runs on import whether or not its name is reached, so it counts like any other statement.
            if node.value and any(isinstance(n, ast.Call) for n in ast.walk(node.value)):
                names.append(INIT)
        elif isinstance(node, ast.ImportFrom) and all(alias.name != "*" for alias in node.names):
            names = [alias.asname or alias.name for alias in node.names]
        elif ast.unparse(node).startswith("if __name__ == '__main__':"):
            names = []
        else:
            names = [INIT]
        for name in names:
            found.setdefault(name, []).append(node)
    return found


def references(node: ast.AST, module: str, folder: Path) -> tuple[list[tuple[str, str]], set[str]]:
    """The (module, name) pairs the node's code in module refers to, a plain or star import of a module in folder as
    all of it, and the attribute names it reads."""
    found, attributes = [], set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.append((module, sub.id))
        elif isinstance(sub, ast.Attribute):
            attributes.add(sub.attr)
        elif isinstance(sub, ast.Import):
            found += [(alias.name, EVERYTHING) for alias in sub.names if (folder / f"{alias.name}.py").exists()]
        elif isinstance(sub, ast.ImportFrom) and (folder / f"{sub.module}.py").exists():
            found += [(str(sub.module), EVERYTHING if a.name == "*" else a.name) for a in sub.names]
    return found, attributes


def model_code(script: Path, entry: str) -> str:
    """A hash of the code a model step runs: entry in script, and every definition it reaches in the modules beside
    it, as ast.unparse writes them without docstrings.

    The render rules in matte.py and detail.py sit beside their model steps but are never reached from them, so
    tuning one leaves the step's artifact current, and a change the step does reach needs no version bump. It errs
    toward a re-run: a reached module's other top-level statements count, a plain or star import of a module beside
    it counts all of that module, and a reached class counts its header and dunder methods, and any method whose
    name reached code reads as an attribute. Third-party code is out, so its pins belong in the recipe.
    """
    modules: dict[str, dict[str, list[ast.AST]]] = {}
    seen: set[tuple[str, str]] = set()
    reached: dict[tuple[str, str], str] = {}
    attributes: set[str] = set()
    methods: dict[tuple[str, str], ast.AST] = {}

    def local(module: str | None) -> bool:
        return bool(module) and (script.parent / f"{module}.py").exists()

    def unparse_and_queue(node: ast.AST, module: str, name: str) -> str:
        """The node's text as reached by name. Queues what it refers to on todo as a side effect."""
        if isinstance(node, ast.ImportFrom) and name not in (INIT, EVERYTHING):
            original = next(a.name for a in node.names if (a.asname or a.name) == name)
            todo.extend([(str(node.module), original)] if local(node.module) else [])
            return f"from {node.module} import {original}"
        if isinstance(node, ast.ClassDef) and name != EVERYTHING:
            calls = [m for m in node.body if isinstance(m, ast.FunctionDef)]
            methods.update({(module, f"{node.name}.{m.name}"): m for m in calls})
            todo.extend((module, f"{node.name}.{m.name}") for m in calls if m.name.startswith("__"))
            node = copy.copy(node)
            node.body = [b for b in node.body if b not in calls] or [ast.Pass()]
        refs, attrs = references(node, module, script.parent)
        todo.extend(refs)
        attributes.update(attrs)
        return ast.unparse(node)

    def methods_now_reached() -> list[tuple[str, str]]:
        """Unvisited methods whose name the reached code reads as an attribute. Only known once todo runs dry."""
        return [key for key in methods if key[1].split(".")[1] in attributes and key not in seen]

    todo = [(script.stem, entry)]
    while todo or (todo := methods_now_reached()):
        module, name = todo.pop()
        if (module, name) in seen:
            continue
        seen.add((module, name))
        if module not in modules:
            modules[module] = definitions(script.parent / f"{module}.py")
            todo.append((module, INIT))
        found = [methods[module, name]] if (module, name) in methods else modules[module].get(name, [])
        if found:
            reached[module, name] = "\n".join(unparse_and_queue(node, module, name) for node in found)
    return hashlib.sha256(json.dumps(sorted(reached.items())).encode()).hexdigest()


def matte_missing() -> str | None:
    """Why this machine can't matte, or None when MatAnyone 2 is installed at its pinned commit and weights."""
    if not MATTE_PYTHON.exists():
        return f"no MatAnyone 2 venv at {MATTE_PYTHON.parent.parent}"
    head = subprocess.run(
        ["git", "-C", str(MATANYONE), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if head.stdout.strip() != MATANYONE_COMMIT:
        return f"{MATANYONE} is not at commit {MATANYONE_COMMIT[:7]}"
    if not MATTE_WEIGHTS.exists():
        return f"no weights at {MATTE_WEIGHTS}"
    with MATTE_WEIGHTS.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != MATTE_WEIGHTS_SHA256:
            return f"{MATTE_WEIGHTS} does not match sha256 {MATTE_WEIGHTS_SHA256[:8]}"
    return None


def matte_step(clip: Clip, art: Artifacts, source: Path, spec: dict, out: Path) -> None:
    """Run matte.py under the SAM slot, which it shares the GPU with, logging to matte.log."""
    log = art.folder / "matte.log"
    log.write_text(f"{time.strftime('%X')} queued, waiting for the SAM slot\n")
    print(f"  matting the mystery player, progress in {log}", flush=True)
    with sam_slot(clip, "matte", log) as slot:
        started = time.monotonic()
        venv_step(clip, MATTE, source, spec, out, log, slot=slot, python=MATTE_PYTHON)
    with log.open("a") as f:
        f.write(f"{time.strftime('%X')} finished matting in {time.monotonic() - started:.1f}s\n")


def detail_step(clip: Clip, art: Artifacts, source: Path, spec: dict, out: Path) -> None:
    """Run detail.py under the SAM slot, logging to detail.log."""
    log = art.folder / "detail.log"
    log.write_text(f"{time.strftime('%X')} queued, waiting for the SAM slot\n")
    print(f"  tracking the mystery player in detail, progress in {log}", flush=True)
    with sam_slot(clip, "detail", log) as slot:
        started = time.monotonic()
        venv_step(clip, DETAIL, source, spec, out, log, slot=slot)
    with log.open("a") as f:
        f.write(f"{time.strftime('%X')} finished the detail pass in {time.monotonic() - started:.1f}s\n")


def shape_of(clip: Clip, art: Artifacts, render: dict) -> str:
    """How the mystery player is drawn: the track spec's "shape", "auto" resolved by his size, or "sam" without one
    mystery player."""
    shape = render.get("shape", "auto")
    if shape not in SHAPES:
        sys.exit(f'{clip.id}: "shape" must be one of {", ".join(SHAPES)}, not {shape!r}')
    if len(render["mystery"]) != 1:
        return "sam"
    if shape != "auto":
        return shape
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "shape.json"
        spec = {"mystery": render["mystery"][0], "frames": render.get("frames")}
        venv_step(clip, DETAIL, art.folder / "masks.npz", spec, out, flags=("--choose",))
        chosen = json.loads(out.read_text())
    print(f"  shape: {chosen['shape']}, by his median height, {chosen['share']:.0%} of the clip's")
    return chosen["shape"]


def prepare_shape(clip: Clip, art: Artifacts, source: Path, render: dict, objects: list[dict]) -> dict:
    """The render spec without "shape", and with "matte" or "detail" set to the artifact that draws the mystery
    player, made first if its recipe changed. Without either, his mask in masks.npz draws him."""
    shape = shape_of(clip, art, render)
    render = {k: v for k, v in render.items() if k != "shape"}
    if shape == "matte":
        return with_matte(clip, art, source, render, objects)
    if shape == "detail":
        return with_detail(clip, art, source, render, objects)
    return render


def with_detail(clip: Clip, art: Artifacts, source: Path, render: dict, objects: list[dict]) -> dict:
    """The render spec with "detail" set to the clip's detail.npz, made first if its recipe changed."""
    assert clip.track is not None, "only a tracked clip has a render spec"
    folder = art.folder
    mystery = render["mystery"][0]
    spec = {
        "clip": str(folder / "clip.mp4"),
        "masks": str(folder / "masks.npz"),
        "mystery": mystery,
        "frames": render.get("frames"),
        "refines": clip.track["objects"][mystery].get("detail", []),
    }
    # Avoid points and seeds come from the other players' masks, so they are inputs too.
    players = [Path(obj["cache"]).name for obj in objects if obj["kind"] == "player"]
    recipe = [
        model_code(DETAIL, "make"),
        TRACK_VERSION,
        art.ledger["source.mp4"],
        Path(objects[mystery]["cache"]).name,
        players,
        spec["frames"],
        spec["refines"],
    ]
    detail = art.make("detail.npz", recipe, lambda out: detail_step(clip, art, source, spec, out))
    return {**render, "detail": str(detail)}


def with_matte(clip: Clip, art: Artifacts, source: Path, render: dict, objects: list[dict]) -> dict:
    """The render spec with "matte" set to the clip's matte.npz, made first if its recipe changed, or without it when
    this machine can't matte."""
    if reason := matte_missing():
        print(f"  matte: off, {reason}; rendering SAM's silhouette. docs/SPEC.md#matting installs it")
        return render
    folder = art.folder
    spec = {
        "clip": str(folder / "clip.mp4"),
        "masks": str(folder / "masks.npz"),
        "mystery": render["mystery"][0],
        "frames": render.get("frames"),
        "weights": str(MATTE_WEIGHTS),
    }
    # Seeds avoid the other players, so their masks are inputs too. The ball's and hoop's are not.
    players = [Path(obj["cache"]).name for obj in objects if obj["kind"] == "player"]
    recipe = [
        model_code(MATTE, "main"),
        art.ledger["source.mp4"],
        Path(objects[spec["mystery"]]["cache"]).name,
        players,
        spec["frames"],
        MATANYONE_COMMIT,
        MATTE_WEIGHTS_SHA256,
    ]
    matte = art.make("matte.npz", recipe, lambda out: matte_step(clip, art, source, spec, out))
    return {**render, "matte": str(matte)}


def shape_inputs(art: Artifacts, shape_path: str | None) -> list:
    """What qc.json and silhouette.mp4 take from the matte.npz or detail.npz that draws the mystery player, if one
    does: its recipe, and the rules in matte.py and detail.py that qc.py and silhouette.py apply to it. The artifact
    holds only model output, so a change to those rules re-renders without re-running the model."""
    return [art.ledger[Path(shape_path).name], MATTE.read_text(), DETAIL.read_text()] if shape_path else []


def window_seconds(clip: Clip, source: Path) -> float:
    """Length of the fetched window, which every variant except a trimmed silhouette matches."""
    if clip.start is not None and clip.end is not None:
        return clip.end - clip.start
    return float(probe(source)["format"]["duration"])


def render_spec(clip: Clip, spec: dict, seconds: float) -> dict:
    """What silhouette.py renders: the track spec without its prompts, the mystery players, and any trim as frames."""
    render = {k: v for k, v in spec.items() if k not in TRACK_KEYS}
    render["mystery"] = [i for i, obj in enumerate(spec["objects"]) if obj.get("mystery")]
    # Trim cuts only the render, so tightening it never re-fetches, re-tracks or shifts prompt frames.
    if "trim" in render:
        t0, t1 = render.pop("trim")
        if not 0 <= t0 < t1 <= seconds:
            sys.exit(f"{clip.id}: trim must be [t0, t1] within the {seconds:.2f}s fetch window")
        render["frames"] = [round(t0 * FPS), round(t1 * FPS)]
    return render


def silhouette(clip: Clip, masks: Path, render: dict, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(tmp) / "frames"
        venv_step(clip, SILHOUETTE, masks, render, frames)
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-framerate",
                str(FPS),
                "-i",
                str(frames / "%05d.png"),
                "-vf",
                f"scale=-2:{HEIGHT}",
                *ENCODE,
                "-an",
                "-f",
                "mp4",
                str(out),
            ]
        )


def contact_sheet(variants: dict[Path, float], out: Path) -> None:
    """Six evenly spaced frames from each variant, one row per variant, given each variant's length in seconds."""
    inputs, chains = [], []
    for i, (path, seconds) in enumerate(variants.items()):
        inputs += ["-i", str(path)]
        chains.append(f"[{i}:v]fps={6 / seconds:.4f},scale=240:-2,tile=6x1[r{i}]")
    stack = "".join(f"[r{i}]" for i in range(len(variants)))
    graph = ";".join(chains) + f";{stack}vstack=inputs={len(variants)}"
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", graph, "-frames:v", "1", "-f", "image2", str(out)])


# Bagwork's render contract, copied from MAX_CLIP_BYTES, RENDER_TAGS and renderProblem in tools/bagwork/library.ts in
# the private bagwork repo, which wins when the two disagree.
MAX_RENDER_BYTES = 2_000_000


RENDER_TAGS = {
    "major_brand": re.compile(r"[a-z0-9]{4}"),
    "minor_version": re.compile(r"\d+", re.ASCII),
    "compatible_brands": re.compile(r"[a-z0-9]+"),
    "encoder": re.compile(r"Lav[fc][\d.]*( libx264)?", re.ASCII),
    "handler_name": re.compile(r"VideoHandler"),
    "language": re.compile(r"[a-z]{3}"),
    "vendor_id": re.compile(r"\[0\]\[0\]\[0\]\[0\]"),
}


def render_problem(info: dict, size: int) -> str | None:
    """Why Bagwork would refuse a render, given its ffprobe JSON with chapters and its size in bytes, or None.

    It checks in Bagwork's order and words each refusal the way Bagwork does, so the two can be compared directly.
    """
    if size >= MAX_RENDER_BYTES:
        return f"{size} bytes, over the {MAX_RENDER_BYTES} limit"
    kinds = [s["codec_type"] for s in info["streams"]]
    if "audio" in kinds:
        return "it has an audio stream"
    if kinds != ["video"]:
        return f"it has {len(kinds)} streams, not one video stream"
    if info["chapters"]:
        return f"it has {len(info['chapters'])} chapters"
    for part in [info["format"], *info["streams"]]:
        for tag, value in part.get("tags", {}).items():
            allowed = RENDER_TAGS.get(tag)
            if allowed is None or not allowed.fullmatch(value):
                return f"it carries a {tag} tag"
    return None


def verify(outputs: dict[str, Path], lengths: dict[str, float]) -> list[str]:
    problems = []
    for name, path in outputs.items():
        info = probe(path)
        video = [s for s in info["streams"] if s["codec_type"] == "video"]
        audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
        duration = float(info["format"]["duration"])
        tags = info["format"].get("tags", {})
        if len(video) != 1 or video[0]["height"] != HEIGHT:
            problems.append(f"{name}: expected one {HEIGHT}p video stream")
        if abs(duration - lengths[name]) > 0.25:
            problems.append(f"{name}: duration {duration:.2f}s, expected {lengths[name]:.2f}s")
        if name != "clip.mp4" and audio:
            problems.append(f"{name}: clue variants must have no audio")
        if any(k.lower() in ("title", "comment", "description", "artist") for k in tags):
            problems.append(f"{name}: identifying metadata tags {sorted(tags)}")
        if name == "silhouette.mp4" and (problem := render_problem(info, path.stat().st_size)):
            problems.append(f"{name}: {problem}")
    return problems


def normalized(clip: Clip) -> tuple[Artifacts, Path, Path]:
    """The clip's artifacts, its fetched source, and the 720p 30 fps clip.mp4 that tracking and the variants read."""
    folder = MEDIA / "clips" / clip.id
    folder.mkdir(parents=True, exist_ok=True)
    art = Artifacts(folder)
    source = art.make("source.mp4", [clip.url, clip.start, clip.end], lambda out: fetch(clip, out))
    video = art.make(
        "clip.mp4",
        ["clip", SCALE, ENCODE, art.ledger["source.mp4"]],
        lambda out: encode(source, out, SCALE, audio=True),
    )
    return art, source, video


def build(clips: list[Clip]) -> bool:
    ok = True
    for clip in clips:
        print(clip.id)
        art, source, video = normalized(clip)
        folder = art.folder
        expected = window_seconds(clip, source)
        outputs = {"clip.mp4": video}
        trimmed = {}
        # Score bugs and on-screen text name the teams, so clue variants black them out first.
        # Crop zooms the clue onto the shooter, which also drops court logos; the reveal clip stays full frame.
        boxes = "".join(f",drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill" for x, y, w, h in clip.hide)
        zoom = f",{clip.crop.filter()}" if clip.crop else ""
        for name, vf in FILTERS.items():
            chain = f"{SCALE}{boxes}{zoom},{vf}"
            outputs[f"{name}.mp4"] = art.make(
                f"{name}.mp4",
                [chain, ENCODE, art.ledger["source.mp4"]],
                lambda out, chain=chain: encode(source, out, chain, audio=False),
            )
        if clip.track:
            spec = clip.track
            render = render_spec(clip, spec, expected)
            if "frames" in render:
                trimmed["silhouette.mp4"] = (render["frames"][1] - render["frames"][0]) / FPS
            objects = track_objects(spec, folder, art.ledger["clip.mp4"])
            (folder / "objects").mkdir(exist_ok=True)
            masks = art.make(
                "masks.npz",
                [[obj["kind"], Path(obj["cache"]).name] for obj in objects],
                lambda out: track_masks(clip, art, outputs["clip.mp4"], objects, out),
            )
            render = prepare_shape(clip, art, source, render, objects)
            shape_path = next((render[k] for k in ("matte", "detail") if k in render), None)
            shaped = shape_inputs(art, shape_path)
            qc = art.make(
                "qc.json",
                [QC.read_text(), SILHOUETTE.read_text(), render["mystery"], art.ledger["masks.npz"], *shaped],
                lambda out: run(
                    [str(sam_python(clip)), str(QC), str(masks), json.dumps(render["mystery"]), str(out)]
                    + ([shape_path] if shape_path else [])
                ),
            )
            for f in json.loads(qc.read_text())["flags"]:
                first, last = f["frames"]
                objects_text = " and ".join(map(str, f["objects"])) or "none"
                print(f"  qc: {f['check']}, object {objects_text}, frames {first}-{last}, worst {f['worst']}")
            outputs["silhouette.mp4"] = art.make(
                "silhouette.mp4",
                [render, SILHOUETTE.read_text(), ENCODE, art.ledger["masks.npz"], *shaped],
                lambda out: silhouette(clip, masks, render, out),
            )
        lengths = {name: trimmed.get(name, expected) for name in outputs}
        art.make(
            "contact.png",
            [list(FILTERS), [art.ledger[n] for n in outputs]],
            lambda out: contact_sheet({outputs[n]: lengths[n] for n in outputs}, out),
        )
        problems = verify(outputs, lengths)
        for p in problems:
            print(f"  FAIL {p}")
        if not problems:
            print(f"  ok: {', '.join(outputs)} verified, sheet at {folder / 'contact.png'}")
        ok = ok and not problems
    return ok


def approve(clips: list[Clip], approved: Path) -> bool:
    """Copy each clip's silhouette.mp4 to approved/<id>.mp4 if Bagwork's render contract accepts it.

    A render that fails is reported and not copied. An approved/<id>.mp4 from an earlier run stays as it was.
    """
    approved.mkdir(exist_ok=True)
    ok = True
    for clip in clips:
        render = MEDIA / "clips" / clip.id / "silhouette.mp4"
        if not render.is_file():
            print(f"{clip.id}: FAIL no {render}; build it first")
            ok = False
            continue
        dest = approved / f"{clip.id}.mp4"
        # The check reads the copy, so a build that rewrites silhouette.mp4 mid-approve can't slip past it.
        tmp = approved / f".tmp-{clip.id}.mp4"
        shutil.copyfile(render, tmp)
        try:
            problem = render_problem(probe(tmp), tmp.stat().st_size)
        except subprocess.CalledProcessError as e:
            problem = f"ffprobe cannot read it: {e.stderr.strip()}"
        if problem:
            tmp.unlink()
            print(f"{clip.id}: FAIL {problem}")
            ok = False
            continue
        tmp.replace(dest)
        print(f"{clip.id}: ok, {dest.stat().st_size} bytes, copied to {dest}")
    return ok


def preview(clip: Clip) -> None:
    """Write each prompted frame with its masks to the clip's preview/ folder, without tracking."""
    if not clip.track:
        sys.exit(f"{clip.id} has no track prompts to preview")
    art, _, video = normalized(clip)
    objects = track_objects(clip.track, art.folder, art.ledger["clip.mp4"])
    out = art.folder / "preview"
    shutil.rmtree(out, ignore_errors=True)
    with sam_slot(clip, "preview") as slot:
        venv_step(clip, TRACK, video, {"objects": objects}, out, flags=("--preview",), slot=slot)


def propose(clip: Clip, frame: int) -> None:
    """Write a numbered sheet of the players and ball a detector finds on one frame, to the proposals/ folder."""
    art, _, video = normalized(clip)
    venv_step(clip, PROPOSE, video, {"frame": frame}, art.folder / "proposals")


def window(clip: Clip, frames: list[int], chosen: list[int]) -> None:
    """Track the chosen objects over the span of the frames and write window.png in the clip's folder, a sheet of the
    frames, so a refine is checked in seconds instead of a full track. It writes no masks, so masks.npz and the object
    caches stay as they were."""
    if not clip.track:
        sys.exit(f"{clip.id} has no track prompts to window")
    art, _, video = normalized(clip)
    objects = track_objects(clip.track, art.folder, art.ledger["clip.mp4"])
    if unknown := [i for i in chosen if not 0 <= i < len(objects)]:
        sys.exit(f"{clip.id} has objects 0 to {len(objects) - 1}, not {', '.join(map(str, unknown))}")
    spec = {"objects": objects, "window": {"objects": sorted(set(chosen)), "frames": frames}}
    with sam_slot(clip, "window") as slot:
        venv_step(clip, TRACK, video, spec, art.folder / "window.png", flags=("--window",), slot=slot)


def crops(clip: Clip, frames: list[int], box: list[int] | None, detail: bool) -> None:
    """Write crops.png in the clip's folder: the box at each frame, gridded in clip.mp4 pixels, for ball_keys.

    The drawn ball from the last build's masks, after the manifest's ball_keys, is ringed where it would render.
    Masks older than clip.mp4 were tracked on other footage, so they are left out. With detail, it writes
    detail.png instead: each frame's crop from the detail pass, for detail refines (docs/SPEC.md#detail).
    """
    art, source, video = normalized(clip)
    masks = art.folder / "masks.npz"
    current = masks.exists() and masks.stat().st_mtime >= video.stat().st_mtime
    if masks.exists() and not current:
        print(f"  masks.npz predates clip.mp4, so no drawn ball; build {clip.id} to place it")
    spec = clip.track or {}
    keys = spec.get("ball_keys", [])
    sheet = {"frames": frames, "box": box, "ball_keys": keys, "masks": str(masks) if current else None, "detail": None}
    if detail:
        found = art.folder / "detail.npz"
        # The detail pass reads the players' masks only, so a re-tracked ball or hoop leaves it current.
        players = [
            Path(obj["cache"])
            for obj in track_objects(spec, art.folder, art.ledger["clip.mp4"])
            if obj["kind"] == "player"
        ]
        made = found.stat().st_mtime if found.exists() else 0
        if not (current and made and all(p.exists() and made >= p.stat().st_mtime for p in players)):
            sys.exit(f'{clip.id} has no detail pass over its current masks; build it with "shape": "detail"')
        mystery = next(i for i, obj in enumerate(spec["objects"]) if obj.get("mystery"))
        refines = spec["objects"][mystery].get("detail", [])
        sheet["detail"] = {"path": str(found), "source": str(source), "mystery": mystery, "refines": refines}
    elif box is None:
        sys.exit("crops needs --box, or --detail for the detail pass's own crops")
    venv_step(clip, CROPS, video, sheet, art.folder / ("detail.png" if detail else "crops.png"))


def review(clip: Clip) -> None:
    """Build the clip, then write review.html in its folder. The page shows source video, so it stays local."""
    if not clip.track:
        sys.exit(f"{clip.id} has no track to review")
    if not build([clip]):
        sys.exit(1)
    art, source, video = normalized(clip)
    spec = clip.track
    render = render_spec(clip, spec, window_seconds(clip, source))
    objects = track_objects(spec, art.folder, art.ledger["clip.mp4"])
    render = prepare_shape(clip, art, source, render, objects)
    flags = json.loads((art.folder / "qc.json").read_text())["flags"]
    venv_step(clip, REVIEW, video, {"render": render, "flags": flags}, art.folder)


def scout(url: str) -> Path:
    """Download a small copy of the post and write a sheet of one frame per second, returning the scout folder."""
    video_id = run(["yt-dlp", "--no-warnings", "--print", "id", url]).strip()
    folder = MEDIA / "scout" / video_id
    frames = folder / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    video = folder / "video.mp4"
    if not video.exists():
        tmp = folder / ".tmp-video.mp4"
        run(
            [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                "--no-playlist",
                "-f",
                "bv*[height<=480]+ba/b[height<=480]/b",
                "--merge-output-format",
                "mp4",
                "-o",
                str(tmp),
                url,
            ]
        )
        tmp.replace(video)
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            "fps=1,scale=320:-2",
            "-start_number",
            "0",
            str(frames / "%04d.jpg"),
        ]
    )
    cells = "".join(
        f'<figure><img src="{f}"><figcaption>{int(f.stem)}s</figcaption></figure>' for f in sorted(frames.glob("*.jpg"))
    )
    sheet = folder / "sheet.html"
    sheet.write_text(
        f"<!doctype html><meta charset=utf-8><title>{html.escape(video_id)}</title>"
        "<style>body{font:13px system-ui;display:grid;"
        "grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px}"
        "figure{margin:0}img{width:100%;display:block}</style>" + cells
    )
    print(f"{len(list(frames.glob('*.jpg')))} frames, one per second: {sheet}")
    return folder


def scene_cuts(video: Path) -> list[float]:
    """Seconds into the video where ffmpeg's scene score says the shot changes."""
    log = subprocess.run(
        ["ffmpeg", "-i", str(video), "-vf", f"select='gt(scene,{SCENE_SCORE_CUT})',showinfo", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    return [float(t) for t in re.findall(r"pts_time:([\d.]+)", log)]


def add(manifest: Path, url: str, player: str, clip_id: str | None) -> None:
    """Scout a post, print what decides whether it can make a clip, and append a stub for it to the manifest.

    The stub holds only id, player, title and url. Prompts, the window and the trim come after scouting.
    """
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    info = json.loads(run(["yt-dlp", "--no-warnings", "--no-playlist", "-f", FETCH_FORMAT, "-j", url]))
    clip_id = clip_id or f"{player.rsplit(maxsplit=1)[-1].lower()}-{info['id']}"
    if not CLIP_ID.fullmatch(clip_id):
        sys.exit(f"id {clip_id!r} may hold only letters, digits, dots, hyphens and underscores; pass --id")
    if any(entry["id"] == clip_id for entry in entries):
        sys.exit(f"{manifest} already has {clip_id}; pass a different --id")
    video = scout(url) / "video.mp4"
    # A direct link to a video file reports no size, and its scout copy is that file.
    scouted = next(s for s in probe(video)["streams"] if s["codec_type"] == "video")
    width, height = info.get("width") or scouted["width"], info.get("height") or scouted["height"]
    shape = "landscape" if width > height else "portrait" if height > width else "square"
    cuts = ", ".join(f"{t:.1f}s" for t in scene_cuts(video)) or "none"
    print(f"{float(probe(video)['format']['duration']):.1f}s, {width}x{height} {shape}, scene cuts: {cuts}")
    entries.append({"id": clip_id, "player": player, "title": info["title"], "url": url})
    tmp = manifest.with_name(f".tmp-{manifest.name}")
    tmp.write_text(json.dumps(entries, indent=2) + "\n")
    tmp.replace(manifest)
    print(f"added {clip_id} to {manifest}")


def frame_list(text: str) -> list[int]:
    """Frames from a comma list of numbers and first-last[:step] ranges, such as "140-160:4,171"."""
    frames = []
    for part in text.split(","):
        span, _, step = part.partition(":")
        first, _, last = span.partition("-")
        frames += range(int(first), int(last or first) + 1, int(step or 1))
    return frames


def command_line() -> argparse.ArgumentParser:
    """The argument parser, kept out of main so main reads as the dispatch it is."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scout").add_argument("url")
    add_cmd = sub.add_parser("add")
    add_cmd.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="the manifest to append to; keep manifests for real puzzles outside the repo",
    )
    add_cmd.add_argument("url")
    add_cmd.add_argument("--player", required=True)
    add_cmd.add_argument("--id")
    build_cmd = sub.add_parser("build")
    build_cmd.add_argument("--manifest", type=Path, default=SAMPLE_MANIFEST)
    build_cmd.add_argument("ids", nargs="*")
    approve_cmd = sub.add_parser("approve")
    approve_cmd.add_argument("--manifest", type=Path, default=SAMPLE_MANIFEST)
    approve_cmd.add_argument("ids", nargs="+")
    for name in ("preview", "propose", "review"):
        one = sub.add_parser(name)
        one.add_argument("--manifest", type=Path, default=SAMPLE_MANIFEST)
        one.add_argument("id")
        if name == "propose":
            one.add_argument("--frame", type=int, required=True)
    crops_cmd = sub.add_parser("crops")
    crops_cmd.add_argument("--manifest", type=Path, default=SAMPLE_MANIFEST)
    crops_cmd.add_argument("id")
    crops_cmd.add_argument("--frames", type=frame_list, required=True)
    crops_cmd.add_argument("--box", type=lambda text: [int(v) for v in text.split(",")])
    crops_cmd.add_argument("--detail", action="store_true", help="each frame's detail crop with his owned mask")
    window_cmd = sub.add_parser("window")
    window_cmd.add_argument("--manifest", type=Path, default=SAMPLE_MANIFEST)
    window_cmd.add_argument("id")
    window_cmd.add_argument("--frames", type=frame_list, required=True, help="the frames to show; tracks their span")
    window_cmd.add_argument("--objects", type=frame_list, required=True, help="object indices, as preview labels them")
    return parser


def main() -> None:
    args = command_line().parse_args()
    # The venv steps print straight to the terminal, so ours must flush per line to stay in order around them.
    sys.stdout.reconfigure(line_buffering=True)  # ty: ignore[missing-union-attribute] a TextIOWrapper at run time
    if missing := [tool for tool in TOOLS if not shutil.which(tool)]:
        sys.exit(f"{', '.join(missing)} not on PATH; brew install ffmpeg yt-dlp installs them")

    if args.command == "scout":
        scout(args.url)
        return
    if args.command == "add":
        add(args.manifest, args.url, args.player, args.id)
        return
    content = os.environ.get("BAGWORK_CONTENT", "")
    if args.command == "approve" and not (content and Path(content).is_dir()):
        sys.exit("approve copies into $BAGWORK_CONTENT/approved; set BAGWORK_CONTENT to an existing folder")
    clips = load_manifest(args.manifest)
    one_clip = {
        "preview": preview,
        "propose": lambda clip: propose(clip, args.frame),
        "review": review,
        "crops": lambda clip: crops(clip, args.frames, args.box, args.detail),
        "window": lambda clip: window(clip, args.frames, args.objects),
    }
    if args.command in one_clip:
        clip = next((c for c in clips if c.id == args.id), None)
        if clip is None:
            sys.exit(f"unknown clip id: {args.id}")
        one_clip[args.command](clip)
        return
    if args.ids:
        unknown = set(args.ids) - {c.id for c in clips}
        if unknown:
            sys.exit(f"unknown clip ids: {', '.join(sorted(unknown))}")
        clips = [c for c in clips if c.id in args.ids]
    if args.command == "approve":
        sys.exit(0 if approve(clips, Path(content) / "approved") else 1)
    sys.exit(0 if build(clips) else 1)


if __name__ == "__main__":
    main()

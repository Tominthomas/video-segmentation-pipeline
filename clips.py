#!/usr/bin/env python3
"""Fetch NBA highlight clips and render Bagwork's filtered variants.

  python3 pipeline/clips.py scout <url>                         frames every second, to pick start/end
  python3 pipeline/clips.py add [--manifest FILE] <url> --player NAME [--id ID]   scout, then add a manifest stub
  python3 pipeline/clips.py build [--manifest FILE] [id ...]    fetch, normalize, filter, verify
  python3 pipeline/clips.py preview [--manifest FILE] id        each prompt's mask, before a full track
  python3 pipeline/clips.py propose [--manifest FILE] id --frame N   numbered player and ball boxes to prompt by
  python3 pipeline/clips.py review [--manifest FILE] id         build, then a local review.html with overlay and QC
  python3 pipeline/clips.py crops [--manifest FILE] id --frames 140-160:4,171 --box x0,y0,x1,y1
                                                                gridded zooms of a region, to read ball_keys off

The manifest defaults to fixtures/clips.json, the public sample clips. Real puzzles keep theirs outside the repo.

Needs yt-dlp and ffmpeg on PATH. Output goes to media/, which is git-ignored.
"""

import argparse
import fcntl
import hashlib
import html
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

ROOT = Path(__file__).resolve().parent.parent
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

# Each clue variant hides faces and jersey text with one whole-frame look.
# The blur runs before recoloring so small digits smear instead of turning into crisp edges.
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
TRACK = ROOT / "pipeline" / "track.py"
SILHOUETTE = ROOT / "pipeline" / "silhouette.py"
QC = ROOT / "pipeline" / "qc.py"
PROPOSE = ROOT / "pipeline" / "propose.py"
REVIEW = ROOT / "pipeline" / "review.py"
CROPS = ROOT / "pipeline" / "crops.py"
TRACK_KEYS = ("frame", "objects")
# Part of every tracked object's cache key. Bump it when a track.py change alters the masks it writes.
TRACK_VERSION = 1
# SAM shares one GPU and most of the memory, so tracking runs one job at a time and only with this much free.
# The lock lives outside the checkout so builds from every worktree on the machine share it.
SAM_LOCK = Path.home() / ".local" / "state" / "bagwork" / "sam.lock"
# Waiters open the lock with "w", which truncates it, so the holder says who it is in a file of its own.
SAM_HOLDER = SAM_LOCK.with_name("sam-holder.json")
WAIT_REPORT = 30
MIN_FREE = int(os.environ.get("BAGWORK_MIN_FREE", "30"))

# The best rendition fetch downloads. add reports its size, so a source is judged by what the build will use.
FETCH_FORMAT = "bv*[height<=1080]+ba/b[height<=1080]"
SCENE_SCORE_CUT = 0.4

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
    track: str | None = None


def parse_crop(raw: dict) -> Crop:
    path = tuple(sorted((float(t), int(x), int(y)) for t, x, y in raw["path"]))
    if not path:
        raise TypeError("crop.path needs at least one [t, x, y] point")
    return Crop(width=int(raw["size"][0]), height=int(raw["size"][1]), path=path)


def load_manifest(path: Path) -> list[Clip]:
    if not path.is_file():
        sys.exit(f"no manifest at {path}")
    raw = json.loads(path.read_text())
    clips = []
    for i, entry in enumerate(raw):
        try:
            fields: dict[str, Any] = {
                **entry,
                "hide": tuple(tuple(box) for box in entry.get("hide", [])),
                "crop": parse_crop(entry["crop"]) if "crop" in entry else None,
                "track": json.dumps(entry["track"], sort_keys=True) if "track" in entry else None,
            }
            clip = Clip(**fields)
        except TypeError as e:
            sys.exit(f"{path.name} entry {i}: {e}")
        if (clip.start is None) != (clip.end is None):
            sys.exit(f"{clip.id}: give both start and end, or neither for the whole video")
        if clip.start is not None and clip.end is not None and not 0 <= clip.start < clip.end:
            sys.exit(f"{clip.id}: start must be >= 0 and before end")
        clips.append(clip)
    ids = [c.id for c in clips]
    if len(ids) != len(set(ids)):
        sys.exit(f"{path.name} has duplicate ids")
    return clips


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
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
        spec = hashlib.sha256(json.dumps(recipe).encode()).hexdigest()
        if target.exists() and self.ledger.get(name) == spec:
            return target
        tmp = target.with_name(f".tmp-{target.name}")
        tmp.unlink(missing_ok=True)
        build(tmp)
        tmp.replace(target)
        self.ledger[name] = spec
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
) -> None:
    """Run a pipeline script in the project venv, exiting on failure. A held SAM slot's fd passes to the script."""
    with tempfile.TemporaryDirectory() as tmp, Path(log or os.devnull).open("a") as log_file:
        spec_path = Path(tmp) / "spec.json"
        spec_path.write_text(json.dumps(spec))
        cmd = [str(VENV_PYTHON), str(script), *flags, str(src), str(spec_path), str(out)]
        stderr = log_file if log else None
        if subprocess.run(cmd, stderr=stderr, pass_fds=() if slot is None else (slot,), check=False).returncode != 0:
            sys.exit(f"{clip.id}: {script.name} failed" + (f", see {log}" if log else ""))


def free_memory() -> int:
    last = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=True).stdout.splitlines()[-1]
    return int(last.rsplit(" ", 1)[1].rstrip("%"))


def track_progress(log: Path) -> str:
    """The last group a track log says it started and how far that group's latest SAM pass has run."""
    lines = re.split(r"[\r\n]+", log.read_text(errors="replace"))
    marks = [i for i, line in enumerate(lines) if line.startswith("tracking group")]
    if not marks:
        return "loading SAM"
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


def track_masks(clip: Clip, art: Artifacts, video: Path, objects: list[dict], out: Path) -> None:
    """Run track.py under the SAM slot when an object needs tracking, and record what the run cost in build.json."""
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


def track_objects(spec: dict, folder: Path, clip_hash: str) -> list[dict]:
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
        key = hashlib.sha256(json.dumps([TRACK_VERSION, clip_hash, prompts], sort_keys=True).encode()).hexdigest()
        objects.append({"kind": obj["kind"], "prompts": prompts, "cache": str(folder / "objects" / f"{key[:16]}.npz")})
    return objects


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


def preview(clip: Clip) -> None:
    """Write each prompted frame with its masks to the clip's preview/ folder, without tracking."""
    if not clip.track:
        sys.exit(f"{clip.id} has no track prompts to preview")
    art, _, video = normalized(clip)
    objects = track_objects(json.loads(clip.track), art.folder, art.ledger["clip.mp4"])
    out = art.folder / "preview"
    shutil.rmtree(out, ignore_errors=True)
    with sam_slot(clip, "preview") as slot:
        venv_step(clip, TRACK, video, {"objects": objects}, out, flags=("--preview",), slot=slot)


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


def propose(clip: Clip, frame: int) -> None:
    """Write a numbered sheet of the players and ball a detector finds on one frame, to the proposals/ folder."""
    art, _, video = normalized(clip)
    venv_step(clip, PROPOSE, video, {"frame": frame}, art.folder / "proposals")


def frame_list(text: str) -> list[int]:
    """Frames from a comma list of numbers and first-last[:step] ranges, such as "140-160:4,171"."""
    frames = []
    for part in text.split(","):
        span, _, step = part.partition(":")
        first, _, last = span.partition("-")
        frames += range(int(first), int(last or first) + 1, int(step or 1))
    return frames


def crops(clip: Clip, frames: list[int], box: list[int]) -> None:
    """Write crops.png in the clip's folder: the box at each frame, gridded in clip.mp4 pixels, for ball_keys.

    The drawn ball from the last build's masks, after the manifest's ball_keys, is ringed where it would render.
    Masks older than clip.mp4 were tracked on other footage, so they are left out.
    """
    art, _, video = normalized(clip)
    masks = art.folder / "masks.npz"
    current = masks.exists() and masks.stat().st_mtime >= video.stat().st_mtime
    if masks.exists() and not current:
        print(f"  masks.npz predates clip.mp4, so no drawn ball; build {clip.id} to place it")
    keys = json.loads(clip.track).get("ball_keys", []) if clip.track else []
    spec = {"frames": frames, "box": box, "ball_keys": keys, "masks": str(masks) if current else None}
    venv_step(clip, CROPS, video, spec, art.folder / "crops.png")


def review(clip: Clip) -> None:
    """Build the clip, then write review.html in its folder. The page shows source video, so it stays local."""
    if not clip.track:
        sys.exit(f"{clip.id} has no track to review")
    if not build([clip]):
        sys.exit(1)
    art, source, video = normalized(clip)
    render = render_spec(clip, json.loads(clip.track), window_seconds(clip, source))
    flags = json.loads((art.folder / "qc.json").read_text())["flags"]
    venv_step(clip, REVIEW, video, {"render": render, "flags": flags}, art.folder)


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
            spec = json.loads(clip.track)
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
            qc = art.make(
                "qc.json",
                [QC.read_text(), SILHOUETTE.read_text(), render["mystery"], art.ledger["masks.npz"]],
                lambda out: run([str(VENV_PYTHON), str(QC), str(masks), json.dumps(render["mystery"]), str(out)]),
            )
            for f in json.loads(qc.read_text())["flags"]:
                first, last = f["frames"]
                objects_text = " and ".join(map(str, f["objects"])) or "none"
                print(f"  qc: {f['check']}, object {objects_text}, frames {first}-{last}, worst {f['worst']}")
            outputs["silhouette.mp4"] = art.make(
                "silhouette.mp4",
                [render, SILHOUETTE.read_text(), ENCODE, art.ledger["masks.npz"]],
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


def scout(url: str) -> Path:
    """Download a small copy of the post and write a sheet of one frame per second, returning the scout folder."""
    video_id = subprocess.run(
        ["yt-dlp", "--no-warnings", "--print", "id", url], capture_output=True, text=True, check=True
    ).stdout.strip()
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
                "bv*[height<=480]+ba/b[height<=480]",
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
    info = json.loads(
        subprocess.run(
            ["yt-dlp", "--no-warnings", "--no-playlist", "-f", FETCH_FORMAT, "-j", url],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    clip_id = clip_id or f"{player.rsplit(maxsplit=1)[-1].lower()}-{info['id']}"
    if any(entry["id"] == clip_id for entry in entries):
        sys.exit(f"{manifest} already has {clip_id}; pass a different --id")
    video = scout(url) / "video.mp4"
    width, height = info["width"], info["height"]
    shape = "landscape" if width > height else "portrait" if height > width else "square"
    cuts = ", ".join(f"{t:.1f}s" for t in scene_cuts(video)) or "none"
    print(f"{float(probe(video)['format']['duration']):.1f}s, {width}x{height} {shape}, scene cuts: {cuts}")
    entries.append({"id": clip_id, "player": player, "title": info["title"], "url": url})
    tmp = manifest.with_name(f".tmp-{manifest.name}")
    tmp.write_text(json.dumps(entries, indent=2) + "\n")
    tmp.replace(manifest)
    print(f"added {clip_id} to {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scout").add_argument("url")
    add_cmd = sub.add_parser("add")
    add_cmd.add_argument("--manifest", type=Path, default=SAMPLE_MANIFEST)
    add_cmd.add_argument("url")
    add_cmd.add_argument("--player", required=True)
    add_cmd.add_argument("--id")
    build_cmd = sub.add_parser("build")
    build_cmd.add_argument("--manifest", type=Path, default=SAMPLE_MANIFEST)
    build_cmd.add_argument("ids", nargs="*")
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
    crops_cmd.add_argument("--box", type=lambda text: [int(v) for v in text.split(",")], required=True)
    args = parser.parse_args()
    # The venv steps print straight to the terminal, so ours must flush per line to stay in order around them.
    sys.stdout.reconfigure(line_buffering=True)  # ty: ignore[missing-union-attribute] a TextIOWrapper at run time

    if args.command == "scout":
        scout(args.url)
        return
    if args.command == "add":
        add(args.manifest, args.url, args.player, args.id)
        return
    clips = load_manifest(args.manifest)
    one_clip = {
        "preview": preview,
        "propose": lambda clip: propose(clip, args.frame),
        "review": review,
        "crops": lambda clip: crops(clip, args.frames, args.box),
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
    sys.exit(0 if build(clips) else 1)


if __name__ == "__main__":
    main()

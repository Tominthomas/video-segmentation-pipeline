#!/usr/bin/env python3
"""Fetch NBA highlight clips and render Moment's filtered variants.

  python3 pipeline/clips.py scout <url>     frames every second, to pick start/end
  python3 pipeline/clips.py build [id ...]  fetch, normalize, filter, verify

Needs yt-dlp and ffmpeg on PATH. Output goes to media/, which is git-ignored.
"""

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "pipeline" / "clips.json"
MEDIA = ROOT / "media"

HEIGHT = 720
FPS = 30
ENCODE = [
    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
    "-g", str(FPS // 2), "-movflags", "+faststart", "-map_metadata", "-1", "-map_chapters", "-1",
]

# Each clue variant hides faces and jersey text with one whole-frame look.
# The blur runs before recoloring so small digits smear instead of turning into crisp edges.
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
SEGMENT = ROOT / "pipeline" / "segment.py"

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
            for (t0, v0), (t1, v1) in reversed(list(zip(points, points[1:]))):
                expr = f"if(lt(t\\,{t1})\\,{v0}+({v1}-{v0})*(t-{t0})/{t1 - t0}\\,{expr})"
            return f"if(lt(t\\,{points[0][0]})\\,{points[0][1]}\\,{expr})"
        return f"crop={self.width}:{self.height}:{along(0)}:{along(1)},scale=-2:{HEIGHT}"


@dataclass(frozen=True)
class Clip:
    id: str
    player: str
    title: str
    url: str
    start: float
    end: float
    hide: tuple[tuple[int, int, int, int], ...] = ()
    crop: Crop | None = None
    track: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_crop(raw: dict) -> Crop:
    path = tuple(sorted((float(t), int(x), int(y)) for t, x, y in raw["path"]))
    if not path:
        raise TypeError("crop.path needs at least one [t, x, y] point")
    return Crop(width=int(raw["size"][0]), height=int(raw["size"][1]), path=path)


def load_manifest() -> list[Clip]:
    raw = json.loads(MANIFEST.read_text())
    clips = []
    for i, entry in enumerate(raw):
        try:
            clip = Clip(**{
                **entry,
                "hide": tuple(tuple(box) for box in entry.get("hide", [])),
                "crop": parse_crop(entry["crop"]) if "crop" in entry else None,
                "track": json.dumps(entry["track"], sort_keys=True) if "track" in entry else None,
            })
        except TypeError as e:
            sys.exit(f"clips.json entry {i}: {e}")
        if not 0 <= clip.start < clip.end:
            sys.exit(f"{clip.id}: start must be >= 0 and before end")
        clips.append(clip)
    ids = [c.id for c in clips]
    if len(ids) != len(set(ids)):
        sys.exit("clips.json has duplicate ids")
    return clips


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{result.stderr[-2000:]}")


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


class Artifacts:
    """Rebuilds an output only when its recipe changed, writing through a temp file so a crash never leaves a half-written artifact."""

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
        os.replace(tmp, target)
        self.ledger[name] = spec
        self.ledger_path.write_text(json.dumps(self.ledger, indent=2))
        print(f"  built {name}")
        return target


def fetch(clip: Clip, out: Path) -> None:
    run([
        "yt-dlp", "--quiet", "--no-warnings", "--no-playlist",
        "-f", "bv*[height<=1080]+ba/b[height<=1080]",
        "--download-sections", f"*{clip.start}-{clip.end}", "--force-keyframes-at-cuts",
        "--merge-output-format", "mp4", "-o", str(out), clip.url,
    ])


def encode(src: Path, out: Path, vf: str, audio: bool) -> None:
    audio_args = ["-c:a", "aac", "-b:a", "128k"] if audio else ["-an"]
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf, *ENCODE, *audio_args, "-f", "mp4", str(out)])


def silhouette(clip: Clip, src: Path, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spec, frames = Path(tmp) / "track.json", Path(tmp) / "frames"
        spec.write_text(clip.track)
        result = subprocess.run([str(VENV_PYTHON), str(SEGMENT), str(src), str(spec), str(frames)], text=True)
        if result.returncode != 0:
            sys.exit(f"{clip.id}: segmentation failed")
        run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.png"),
             "-vf", f"scale=-2:{HEIGHT}", *ENCODE, "-an", "-f", "mp4", str(out)])


def contact_sheet(variants: list[Path], out: Path, duration: float) -> None:
    rate = 6 / duration
    inputs, chains = [], []
    for i, path in enumerate(variants):
        inputs += ["-i", str(path)]
        chains.append(f"[{i}:v]fps={rate:.4f},scale=240:-2,tile=6x1[r{i}]")
    stack = "".join(f"[r{i}]" for i in range(len(variants)))
    graph = ";".join(chains) + f";{stack}vstack=inputs={len(variants)}"
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", graph, "-frames:v", "1", "-f", "image2", str(out)])


def verify(clip: Clip, outputs: dict[str, Path]) -> list[str]:
    problems = []
    for name, path in outputs.items():
        info = probe(path)
        video = [s for s in info["streams"] if s["codec_type"] == "video"]
        audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
        duration = float(info["format"]["duration"])
        tags = info["format"].get("tags", {})
        if len(video) != 1 or video[0]["height"] != HEIGHT:
            problems.append(f"{name}: expected one {HEIGHT}p video stream")
        if abs(duration - clip.duration) > 0.25:
            problems.append(f"{name}: duration {duration:.2f}s, manifest says {clip.duration:.2f}s")
        if name != "clip.mp4" and audio:
            problems.append(f"{name}: clue variants must have no audio")
        if any(k.lower() in ("title", "comment", "description", "artist") for k in tags):
            problems.append(f"{name}: identifying metadata tags {sorted(tags)}")
    return problems


def build(clips: list[Clip]) -> bool:
    ok = True
    for clip in clips:
        print(clip.id)
        folder = MEDIA / "clips" / clip.id
        folder.mkdir(parents=True, exist_ok=True)
        art = Artifacts(folder)
        source = art.make("source.mp4", [clip.url, clip.start, clip.end], lambda out: fetch(clip, out))
        scale = f"scale=-2:{HEIGHT},fps={FPS}"
        outputs = {"clip.mp4": art.make("clip.mp4", ["clip", scale, ENCODE, art.ledger["source.mp4"]],
                                        lambda out: encode(source, out, scale, audio=True))}
        # Score bugs and on-screen text name the teams, so clue variants black them out first.
        # Crop zooms the clue onto the shooter, which also drops court logos; the reveal clip stays full frame.
        boxes = "".join(f",drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill" for x, y, w, h in clip.hide)
        zoom = f",{clip.crop.filter()}" if clip.crop else ""
        for name, vf in FILTERS.items():
            chain = f"{scale}{boxes}{zoom},{vf}"
            outputs[f"{name}.mp4"] = art.make(f"{name}.mp4", [chain, ENCODE, art.ledger["source.mp4"]],
                                              lambda out, chain=chain: encode(source, out, chain, audio=False))
        if clip.track:
            outputs["silhouette.mp4"] = art.make(
                "silhouette.mp4", [clip.track, SEGMENT.read_text(), ENCODE, art.ledger["clip.mp4"]],
                lambda out: silhouette(clip, outputs["clip.mp4"], out))
        art.make("contact.png", [list(FILTERS), [art.ledger[n] for n in outputs]],
                 lambda out: contact_sheet(list(outputs.values()), out, clip.duration))
        problems = verify(clip, outputs)
        for p in problems:
            print(f"  FAIL {p}")
        if not problems:
            print(f"  ok: {', '.join(outputs)} verified, sheet at {folder / 'contact.png'}")
        ok = ok and not problems
    return ok


def scout(url: str) -> None:
    video_id = subprocess.run(["yt-dlp", "--no-warnings", "--print", "id", url],
                              capture_output=True, text=True, check=True).stdout.strip()
    folder = MEDIA / "scout" / video_id
    frames = folder / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    video = folder / "video.mp4"
    if not video.exists():
        tmp = folder / ".tmp-video.mp4"
        run(["yt-dlp", "--quiet", "--no-warnings", "--no-playlist", "-f", "bv*[height<=480]+ba/b[height<=480]",
             "--merge-output-format", "mp4", "-o", str(tmp), url])
        os.replace(tmp, video)
    run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "fps=1,scale=320:-2", "-start_number", "0",
         str(frames / "%04d.jpg")])
    cells = "".join(
        f'<figure><img src="{f}"><figcaption>{int(f.stem)}s</figcaption></figure>'
        for f in sorted(frames.glob("*.jpg"))
    )
    sheet = folder / "sheet.html"
    sheet.write_text(
        f"<!doctype html><meta charset=utf-8><title>{html.escape(video_id)}</title>"
        "<style>body{font:13px system-ui;display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px}"
        "figure{margin:0}img{width:100%;display:block}</style>" + cells
    )
    print(f"{len(list(frames.glob('*.jpg')))} frames, one per second: {sheet}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scout").add_argument("url")
    sub.add_parser("build").add_argument("ids", nargs="*")
    args = parser.parse_args()

    if args.command == "scout":
        scout(args.url)
        return
    clips = load_manifest()
    if args.ids:
        unknown = set(args.ids) - {c.id for c in clips}
        if unknown:
            sys.exit(f"unknown clip ids: {', '.join(sorted(unknown))}")
        clips = [c for c in clips if c.id in args.ids]
    sys.exit(0 if build(clips) else 1)


if __name__ == "__main__":
    main()

"""Write a clip's review page: the silhouette beside a numbered mask overlay, the QC flags, ball and holder on a
frame timeline, and stills at the moments an approval turns on.

Run by clips.py with the project venv:  .venv/bin/python pipeline/review.py <clip.mp4> <review.json> <clip folder>
review.json is {"render", "flags"}: silhouette.py's render spec and qc.json's flags. It writes review.html in the
clip folder, and the overlay video and stills beside it in review/. The overlay shows the source video, so the page
stays on this machine and is never published. A matted or detail render adds a row to the timeline: the frames drawn
from the matte or the detail pass, the frames that fell back to SAM, the seeds, and the detail refines.
"""

import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from detail import Detail
from matte import Matte
from PIL import Image, ImageDraw, ImageFont
from silhouette import TINTS, draw_ball, holders, load

FPS = 30
SOURCE_COLORS = {"mask": "#3c3", "key": "#4cf", "fit": "#fc3", "hold": "#f55"}
# A frame whose ball the silhouette leaves off, whatever its source.
HIDDEN_COLOR = "#777"
SHAPE_COLORS = {"matte": "#a7f", "detail": "#a7f", "SAM fallback": "#f55", "seed": "#fff", "refine": "#fd4"}
ROW = 16


def video_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    width, height = map(int, out.strip().split(","))
    return width, height


def frames_of(path: Path):
    """Yield each frame of the video as an RGB array, decoded one at a time."""
    width, height = video_size(path)
    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE
    )
    while chunk := decoder.stdout.read(width * height * 3):  # ty: ignore[missing-union-attribute] stdout=PIPE sets it
        yield np.frombuffer(chunk, np.uint8).reshape(height, width, 3)
    decoder.wait()


def overlay(image: np.ndarray, frame: int, kinds: list[str], masks: np.ndarray, ball, font) -> Image.Image:
    """The dimmed source frame with each object's raw mask tinted and numbered, and the tracked ball as a ring.

    The ring's color says where that frame's ball came from, per SOURCE_COLORS, or HIDDEN_COLOR where the
    silhouette leaves it off.
    """
    canvas = image.astype(np.float32) * 0.5
    for i in range(len(kinds)):
        canvas[masks[i]] = canvas[masks[i]] * 0.3 + np.asarray(TINTS[i % len(TINTS)]) * 0.7
    picture = Image.fromarray(canvas.astype(np.uint8))
    pen = ImageDraw.Draw(picture)
    for i in range(len(kinds)):
        ys, xs = np.nonzero(masks[i])
        if len(xs):
            at = (int(xs.mean()), int(ys.min()) - 14)
            tint = TINTS[i % len(TINTS)]
            pen.text(at, str(i), font=font, fill=tint, stroke_width=3, stroke_fill="black", anchor="mm")
    if ball is not None:
        track, show = ball
        x, y = track.xy[frame]
        if not np.isnan(x):
            r = track.radius + 4
            color = SOURCE_COLORS[track.source[frame]] if show[frame] else HIDDEN_COLOR
            pen.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
    pen.text((12, 8), f"frame {frame}", font=font, fill="white", stroke_width=3, stroke_fill="black")
    return picture


def write_overlay(clip: Path, first: int, end: int, draw, stills: dict[int, Path], out: Path) -> None:
    """Encode the overlay of frames [first, end) to out, saving the frames named in stills as PNGs too."""
    width, height = video_size(clip)
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-movflags",
            "+faststart",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    for frame, image in enumerate(frames_of(clip)):
        if first <= frame < end:
            picture = draw(image, frame)
            encoder.stdin.write(picture.tobytes())  # ty: ignore[missing-union-attribute] stdin=PIPE sets it
            if frame in stills:
                picture.save(stills[frame])
    encoder.stdin.close()  # ty: ignore[missing-union-attribute] stdin=PIPE sets it
    if encoder.wait() != 0:
        sys.exit("encoding the review overlay failed")


def silhouette_still(video: Path, index: int, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", str(out)],
        check=True,
    )


def runs_of[T](values: list[T], first: int) -> list[tuple[int, int, T]]:
    """Group a per-frame list into (first frame, last frame, value) runs, numbering frames from first."""
    found = []
    for offset, value in enumerate(values):
        frame = first + offset
        if found and found[-1][2] == value and found[-1][1] == frame - 1:
            found[-1] = (found[-1][0], frame, value)
        else:
            found.append((frame, frame, value))
    return found


def timeline(
    first: int, end: int, flags: list[dict], ball, held: list, shape: Matte | Detail | None, picks: dict[str, int]
) -> str:
    """A strip with one column per frame: ball source, holder, the source of his shape, then one row per QC flag,
    with the picks marked.

    The labels sit in their own HTML column, so the frames fill the strip's width however short the clip.
    """
    width = end - first
    rows, labels = [], []

    def row(label: str, spans: list[tuple[int, int, str, str]]) -> None:
        y = len(rows) * ROW
        cells = "".join(
            f'<rect x="{a - first}" y="{y + 2}" width="{b - a + 1}" height="{ROW - 4}" fill="{color}">'
            f"<title>{html.escape(tip)}</title></rect>"
            for a, b, color, tip in spans
        )
        rows.append(cells)
        labels.append(f"<div>{html.escape(label)}</div>")

    if ball is not None:
        track, show = ball
        sources = runs_of(list(zip(track.source[first:end], show[first:end], strict=True)), first)
        row(
            "ball",
            [
                (a, b, SOURCE_COLORS[s], f"{s} {a}-{b}") if on else (a, b, HIDDEN_COLOR, f"{s} hidden {a}-{b}")
                for a, b, (s, on) in sources
                if s
            ],
        )
    holder_spans = [span for span in runs_of(held[first:end], first) if span[2] is not None]
    row("holder", [(a, b, f"rgb{TINTS[h % len(TINTS)]}", f"{h} {a}-{b}") for a, b, h in holder_spans])
    if shape is not None:
        name = "matte" if isinstance(shape, Matte) else "detail"
        sources = ["SAM fallback" if fell else name for fell in shape.fallback]
        marks = [(f, f, SHAPE_COLORS["seed"], f"seed {f}") for f in shape.seeds.tolist()]
        if isinstance(shape, Detail):
            marks += [(f, f, SHAPE_COLORS["refine"], f"refine {f}") for f in shape.refines.tolist()]
        row(name, [(a, b, SHAPE_COLORS[s], f"{s} {a}-{b}") for a, b, s in runs_of(sources, shape.first)] + marks)
    for f in flags:
        a, b = max(f["frames"][0], first), min(f["frames"][1], end - 1)
        objects = " and ".join(map(str, f["objects"])) or "none"
        if a <= b:
            row(f"{f['check']} {objects}", [(a, b, "#e44", f"{f['check']} worst {f['worst']}, {a}-{b}")])
    height = len(rows) * ROW
    marks = "".join(
        f'<line x1="{frame - first + 0.5}" x2="{frame - first + 0.5}" y1="0" y2="{height}" class="pick">'
        f"<title>{name} {frame}</title></line>"
        for name, frame in picks.items()
    )
    return (
        f"<div class=timeline><div class=labels>{''.join(labels)}</div>"
        f'<svg id="timeline" viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'data-first="{first}" data-frames="{width}" style="height:{height * 2}px">'
        f'{"".join(rows)}{marks}<line id="head" x1="0" x2="0" y1="0" y2="{height}"/></svg></div>'
    )


PAGE = """<!doctype html><meta charset=utf-8><title>{title}</title>
<style>
body{{margin:0;padding:16px;background:#111;color:#ddd;font:13px system-ui}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
video,img{{width:100%;display:block;background:#000}}
.timeline{{display:grid;grid-template-columns:max-content 1fr;gap:8px;margin:12px 0}}
.labels div{{height:{row}px;line-height:{row}px;color:#aaa;font-size:11px;text-align:right;white-space:nowrap}}
svg{{width:100%;cursor:pointer}}
.pick{{stroke:#fff;stroke-width:1;vector-effect:non-scaling-stroke}}
#head{{stroke:#fff;stroke-width:2;vector-effect:non-scaling-stroke}}
.stills{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}} figure{{margin:0}}
figcaption{{padding:4px 0}} .local{{color:#f88}}
</style>
<p class=local>Local only. This page shows the source video.</p>
<div class=pair><video id=sil src="silhouette.mp4" controls muted playsinline></video>
<video id=ovl src="review/overlay.mp4" muted playsinline></video></div>
{timeline}
<div class=stills>{stills}</div>
<script>
const sil = document.getElementById("sil"), ovl = document.getElementById("ovl");
const line = document.getElementById("timeline"), head = document.getElementById("head");
const frames = +line.dataset.frames, fps = {fps};
sil.onplay = () => ovl.play(); sil.onpause = () => {{ ovl.pause(); ovl.currentTime = sil.currentTime; }};
sil.onseeked = () => ovl.currentTime = sil.currentTime;
line.onclick = e => {{
  const box = line.getBoundingClientRect();
  const x = (e.clientX - box.left) / box.width * frames;
  sil.currentTime = Math.max(0, Math.min(frames - 1, Math.floor(x))) / fps + 0.001;
}};
(function tick() {{
  const t = Math.min(frames, sil.currentTime * fps);
  head.setAttribute("x1", t); head.setAttribute("x2", t);
  if (Math.abs(ovl.currentTime - sil.currentTime) > 0.1) ovl.currentTime = sil.currentTime;
  requestAnimationFrame(tick);
}})();
</script>
"""


def main() -> None:
    clip, spec_path, folder = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    spec = json.loads(spec_path.read_text())
    render, flags = spec["render"], spec["flags"]
    kinds, masks = load(folder / "masks.npz")
    first, end = render.get("frames", (0, len(masks)))
    # draw_ball rewrites only the ball's masks, so only they are set aside and put back for the overlay.
    raw_ball = masks[:, kinds.index("ball")].copy() if "ball" in kinds else None
    with np.errstate(divide="ignore", invalid="ignore"):
        track = draw_ball(kinds, masks, render.get("ball_keys", []))
    held, release, rim = holders(kinds, masks, track)
    ball = None
    if track is not None:
        masks[:, kinds.index("ball")] = raw_ball
        ball = track, track.shown()
    shape = (
        Matte.load(Path(render["matte"]))
        if "matte" in render
        else Detail.load(Path(render["detail"]))
        if "detail" in render
        else None
    )
    moments = {"first": first, "release": release, "ball at rim": rim, "last": end - 1}
    picks = {name: frame for name, frame in moments.items() if frame is not None and first <= frame < end}
    out = folder / "review"
    out.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=24)
    stills = {frame: out / f"{frame:05d}-overlay.png" for frame in picks.values()}
    write_overlay(
        clip,
        first,
        end,
        lambda image, frame: overlay(image, frame, kinds, masks[frame], ball, font),
        stills,
        out / "overlay.mp4",
    )
    cells = []
    for name, frame in picks.items():
        silhouette_still(folder / "silhouette.mp4", frame - first, out / f"{frame:05d}-silhouette.png")
        cells.append(
            f"<figure><img src='review/{frame:05d}-silhouette.png'><img src='review/{frame:05d}-overlay.png'>"
            f"<figcaption>{name}, frame {frame}</figcaption></figure>"
        )
    page = PAGE.format(
        title=html.escape(folder.name),
        timeline=timeline(first, end, flags, ball, held, shape, picks),
        stills="".join(cells),
        fps=FPS,
        row=ROW * 2,
    )
    (folder / "review.html").write_text(page)
    print(f"  {folder / 'review.html'}")


if __name__ == "__main__":
    main()

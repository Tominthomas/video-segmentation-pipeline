"""Write one sheet of zoomed crops of a region at chosen frames, gridded in source pixels, to place ball keys by.

Run by clips.py with the project venv:  .venv/bin/python pipeline/crops.py <clip.mp4> <crops.json> <sheet.png>
crops.json is {"frames", "box": [x0, y0, x1, y1], "ball_keys", "masks"}. Each tile is the box at one frame,
scaled up, with grid lines labeled in clip.mp4 pixels, so a ball read off the grid is a "ball_keys" entry as is.
When "masks" names the clip's masks.npz, the tracked ball there, after ball_keys, is ringed in the review page's
source colors, gray where the silhouette hides it, and printed per frame. The sheet shows the source video, so it
stays on this machine.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from review import HIDDEN_COLOR, SOURCE_COLORS, video_size
from silhouette import ball_track, load

TILE = 480
MARGIN = 44
COLUMNS = 4
# The grid step is the smallest of these that keeps a tile to about this many lines across.
STEPS = (5, 10, 20, 25, 50, 100, 200)
LINES = 10


def frames_at(clip: Path, frames: list[int]) -> dict[int, np.ndarray]:
    """The listed frames of the video as RGB arrays, decoded in one ffmpeg pass."""
    width, height = video_size(clip)
    pick = "+".join(f"eq(n\\,{f})" for f in frames)
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(clip),
            "-vf",
            f"select={pick}",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    images = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    if len(images) != len(frames):
        sys.exit(f"{clip.name} has {len(images)} of the frames asked for; the last is {frames[-1]}")
    return dict(zip(frames, images, strict=True))


def tile(image: np.ndarray, frame: int, box: list[int], ball, font) -> Image.Image:
    """One frame's crop, scaled to TILE wide, with a grid in source pixels and the drawn ball ringed."""
    x0, y0, x1, y1 = box
    scale = TILE / (x1 - x0)
    crop = Image.fromarray(image[y0:y1, x0:x1]).resize((TILE, round((y1 - y0) * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (crop.width + MARGIN, crop.height + MARGIN), (17, 17, 17))
    canvas.paste(crop, (MARGIN, MARGIN))
    lines = Image.new("RGBA", canvas.size)
    pen, ink = ImageDraw.Draw(canvas), ImageDraw.Draw(lines)
    step = next((s for s in STEPS if (x1 - x0) / s <= LINES), STEPS[-1])
    for x in range(-(-x0 // step) * step, x1, step):
        at = MARGIN + (x - x0) * scale
        ink.line((at, MARGIN, at, canvas.height), fill=(255, 255, 255, 90))
        pen.text((at, MARGIN - 4), str(x), font=font, fill="white", anchor="mb")
    for y in range(-(-y0 // step) * step, y1, step):
        at = MARGIN + (y - y0) * scale
        ink.line((MARGIN, at, canvas.width, at), fill=(255, 255, 255, 90))
        pen.text((MARGIN - 4, at), str(y), font=font, fill="white", anchor="rm")
    canvas.paste(lines, mask=lines)
    pen.text((4, 4), f"frame {frame}", font=font, fill="white")
    if ball is not None:
        track, show = ball
        x, y = track.xy[frame]
        if not np.isnan(x):
            cx, cy, r = MARGIN + (x - x0) * scale, MARGIN + (y - y0) * scale, (track.radius + 3) * scale
            color = SOURCE_COLORS[track.source[frame]] if show[frame] else HIDDEN_COLOR
            pen.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)
    return canvas


def main() -> None:
    clip, spec_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    spec = json.loads(spec_path.read_text())
    width, height = video_size(clip)
    x0, y0, x1, y1 = spec["box"]
    box = [max(0, x0), max(0, y0), min(width, x1), min(height, y1)]
    if box[0] >= box[2] or box[1] >= box[3]:
        sys.exit(f"box {spec['box']} misses the {width}x{height} frame")
    ball = None
    if spec["masks"]:
        kinds, masks = load(Path(spec["masks"]))
        if "ball" in kinds:
            with np.errstate(divide="ignore", invalid="ignore"):
                track = ball_track(kinds, masks, spec["ball_keys"])
            ball = track, track.shown()
    frames = sorted(set(spec["frames"]))
    font = ImageFont.load_default(size=14)
    tiles = [tile(image, frame, box, ball, font) for frame, image in frames_at(clip, frames).items()]
    columns = min(COLUMNS, len(tiles))
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (columns * tiles[0].width, rows * tiles[0].height), (17, 17, 17))
    for n, picture in enumerate(tiles):
        sheet.paste(picture, ((n % columns) * picture.width, (n // columns) * picture.height))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    if ball is not None:
        track, show = ball
        for frame in frames:
            x, y = track.xy[frame]
            placed = f"({x:.0f}, {y:.0f}) from {track.source[frame]}" if not np.isnan(x) else "not placed"
            print(f"  frame {frame}: ball {placed}" + ("" if show[frame] or np.isnan(x) else ", hidden"))
    print(f"  {out}")


if __name__ == "__main__":
    main()

"""Write one sheet of zoomed crops of a region at chosen frames, gridded in clip pixels, to place ball keys or
detail refines by.

Run by clips.py with the SAM venv's python:  python crops.py <clip.mp4> <crops.json> <sheet.png>
crops.json is {"frames", "box": [x0, y0, x1, y1], "ball_keys", "masks", "detail"}. Each tile is the box at one
frame, scaled up, with grid lines labeled in clip.mp4 pixels, so a ball read off the grid is a "ball_keys" entry as
is. When "masks" names the clip's masks.npz, the tracked ball there, after ball_keys, is ringed in the review page's
source colors, gray where the silhouette hides it, and printed per frame.
"detail" is {"path", "source", "mystery", "refines"}: the clip's detail.npz, its source.mp4, the mystery player's
object index and his detail refines. Each tile is then that frame's detail crop from the source, still gridded in
clip pixels, so a point read off it is a refine as is. It marks his owned mask (DETAIL_COLORS) and his masks.npz
mask, and draws each refine's keep and avoid points. The sheet shows the source video, so it stays on this machine.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detail import Detail, scale, to_crop
from review import HIDDEN_COLOR, SOURCE_COLORS, video_size
from silhouette import ball_track, dilate, erode, load, others_of

TILE = 480
MARGIN = 44
COLUMNS = 4
# The grid step is the smallest of these that keeps a tile to about this many lines across.
STEPS = (5, 10, 20, 25, 50, 100, 200)
LINES = 10
# His owned mask is tinted and outlined, ceded pixels and those the speckle rule changed are filled, and his mask in
# masks.npz is outlined.
DETAIL_COLORS = {"owned": (70, 220, 255), "ceded": (255, 60, 60), "speckle": (255, 230, 40), "sam": (255, 80, 255)}
REFINE_COLORS = {"points": "lime", "avoid": "red"}


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


def tile(
    image: np.ndarray,
    frame: int,
    box: list[int],
    ball,
    font,
    ratio: tuple[float, float] = (1.0, 1.0),
    marks: tuple[tuple[float, float, str], ...] | list[tuple[float, float, str]] = (),
) -> Image.Image:
    """One frame's crop of the box in clip pixels, scaled to TILE wide, with a grid in clip pixels, the drawn ball
    ringed and each mark (x, y, color) in clip pixels dotted. The image is ratio times the clip's size per axis."""
    x0, y0, x1, y1 = box
    scale = TILE / (x1 - x0)
    region = (x0 * ratio[0], y0 * ratio[1], x1 * ratio[0], y1 * ratio[1])
    size = (TILE, round((y1 - y0) * scale))
    crop = Image.fromarray(image).resize(size, Image.Resampling.LANCZOS, box=region)
    canvas = Image.new("RGB", (crop.width + MARGIN, crop.height + MARGIN), (17, 17, 17))
    canvas.paste(crop, (MARGIN, MARGIN))
    lines = Image.new("RGBA", canvas.size)
    pen, ink = ImageDraw.Draw(canvas), ImageDraw.Draw(lines)
    step = next((s for s in STEPS if (x1 - x0) / s <= LINES), STEPS[-1])
    for x in range(int(-(-x0 // step) * step), int(x1), step):
        at = MARGIN + (x - x0) * scale
        ink.line((at, MARGIN, at, canvas.height), fill=(255, 255, 255, 90))
        pen.text((at, MARGIN - 4), str(x), font=font, fill="white", anchor="mb")
    for y in range(int(-(-y0 // step) * step), int(y1), step):
        at = MARGIN + (y - y0) * scale
        ink.line((MARGIN, at, canvas.width, at), fill=(255, 255, 255, 90))
        pen.text((MARGIN - 4, at), str(y), font=font, fill="white", anchor="rm")
    canvas.paste(lines, mask=lines)
    pen.text((4, 4), f"frame {frame}", font=font, fill="white")
    for x, y, color in marks:
        cx, cy = MARGIN + (x - x0) * scale, MARGIN + (y - y0) * scale
        pen.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=color, outline="black", width=2)
    if ball is not None:
        track, show = ball
        x, y = track.xy[frame]
        if not np.isnan(x):
            cx, cy, r = MARGIN + (x - x0) * scale, MARGIN + (y - y0) * scale, (track.radius + 3) * scale
            color = SOURCE_COLORS[track.source[frame]] if show[frame] else HIDDEN_COLOR
            pen.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)
    return canvas


def save_sheet(tiles: list[Image.Image], out: Path) -> None:
    columns = min(COLUMNS, len(tiles))
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (columns * tiles[0].width, rows * tiles[0].height), (17, 17, 17))
    for n, picture in enumerate(tiles):
        sheet.paste(picture, ((n % columns) * picture.width, (n // columns) * picture.height))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def detail_tiles(clip: Path, spec: dict, frames: list[int], font) -> list[Image.Image]:
    """Each frame's detail crop from the source with his owned mask marked, gridded in clip pixels."""
    detail = Detail.load(Path(spec["path"]))
    kinds, masks = load(Path(spec["masks"]))
    mystery = spec["mystery"]
    outside = [f for f in frames if f not in detail.frames()]
    if outside:
        sys.exit(f"frames {outside} are outside the detail pass's frames {detail.first}-{detail.frames()[-1]}")
    ratio = scale(video_size(clip), detail.source)
    images = frames_at(Path(spec["source"]), sorted({int(detail.source_frames[f - detail.first]) for f in frames}))
    tiles = []
    for f in frames:
        j = f - detail.first
        his = masks[f, mystery]
        owned = detail.owned(f, his, others_of(kinds, masks[f], mystery))
        (cw, ch), (ox, oy) = detail.crop, detail.origin[j]
        image = images[int(detail.source_frames[j])].astype(np.float32)
        view = image[oy : oy + ch, ox : ox + cw]
        mine = owned.alpha >= 0.5
        sam = to_crop(his, detail.origin[j], detail.crop, detail.source)
        for mask, color, share in (
            (mine, "owned", 0.3),
            (mine & ~erode(mine, 2), "owned", 1),
            (dilate(sam, 1) & ~sam, "sam", 1),
            (owned.ceded, "ceded", 0.7),
            (owned.speckle, "speckle", 1),
        ):
            view[mask] += (np.asarray(DETAIL_COLORS[color]) - view[mask]) * share
        refine = next((r for r in spec["refines"] if r["frame"] == f), {})
        marks = [(x, y, REFINE_COLORS[k]) for k in REFINE_COLORS for x, y in refine.get(k, [])]
        box = [ox / ratio[0], oy / ratio[1], (ox + cw) / ratio[0], (oy + ch) / ratio[1]]
        tiles.append(tile(image.round().astype(np.uint8), f, box, None, font, ratio, marks))
        fallback = ", falls back to masks.npz" if detail.fallback[j] else ""
        area = max(int(mine.sum()), 1)
        agree = detail.clip_iou(f, owned, his)
        ceded, speckle = owned.ceded.sum() / area, owned.speckle.sum() / area
        print(f"  frame {f}: IoU with masks.npz {agree:.2f}, ceded {ceded:.0%}, speckle {speckle:.0%}{fallback}")
    return tiles


def main() -> None:
    clip, spec_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    spec = json.loads(spec_path.read_text())
    font = ImageFont.load_default(size=14)
    if spec["detail"]:
        save_sheet(
            detail_tiles(clip, {**spec["detail"], "masks": spec["masks"]}, sorted(set(spec["frames"])), font), out
        )
        print(f"  {out}")
        return
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
    save_sheet([tile(image, frame, box, ball, font) for frame, image in frames_at(clip, frames).items()], out)
    if ball is not None:
        track, show = ball
        for frame in frames:
            x, y = track.xy[frame]
            placed = f"({x:.0f}, {y:.0f}) from {track.source[frame]}" if not np.isnan(x) else "not placed"
            print(f"  frame {frame}: ball {placed}" + ("" if show[frame] or np.isnan(x) else ", hidden"))
    print(f"  {out}")


if __name__ == "__main__":
    main()

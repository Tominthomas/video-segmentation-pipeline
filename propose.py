"""Detect people and the ball on one frame and number them, so prompts can be picked by box number.

Run by clips.py with the project venv:  .venv/bin/python propose.py <clip.mp4> <spec.json> <out_dir>
spec.json is {"frame"}. It writes <frame>.png, the frame with every box drawn and numbered, and <frame>.json,
[{"n", "kind", "box": [x0, y0, x1, y1], "score", "crowd"}]. Players come first, left to right, then balls, then
the people the crowd filter dropped, drawn dim. Each box drops straight into a manifest object as "box".
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

KINDS = {1: "player", 37: "ball"}
MIN_SCORE = {"player": 0.5, "ball": 0.3}
# Fans and the bench are people too. Seated or far up the stands, a fan stands shorter than this share of the
# tallest people in frame (their 90th percentile).
CROWD_HEIGHT = 0.6
# A player's feet stand on smooth floor, while a fan's box ends in seats or other fans, which this mean edge
# strength under the feet tells apart at 720p. The strip under the nearest players, the tallest in frame, runs
# into floor logos and the score bar, so anyone this share of the tallest height is spared the test.
FLOOR_CLUTTER = 20
NEAR_HEIGHT = 0.9
COLORS = {"player": (80, 220, 255), "ball": (255, 122, 26), "crowd": (110, 110, 110)}


def frame_at(clip: Path, frame: int) -> Image.Image:
    size = subprocess.run(
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
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    width, height = map(int, size.strip().split(","))
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(clip),
            "-vf",
            f"select=eq(n\\,{frame})",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    if not raw:
        sys.exit(f"{clip.name} has no frame {frame}")
    return Image.fromarray(np.frombuffer(raw, np.uint8).reshape(height, width, 3))


def detect(image: Image.Image) -> list[dict]:
    """Every person and sports ball COCO's Faster R-CNN finds above MIN_SCORE, as {"kind", "box", "score"}."""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=min(MIN_SCORE.values())).eval()
    with torch.inference_mode():
        out = model([weights.transforms()(image)])[0]
    found = []
    for label, box, score in zip(out["labels"].tolist(), out["boxes"].tolist(), out["scores"].tolist(), strict=True):
        kind = KINDS.get(label)
        if kind and score >= MIN_SCORE[kind]:
            found.append({"kind": kind, "box": [round(v) for v in box], "score": round(score, 2)})
    return found


def underfoot(edges: np.ndarray, box: list[int]) -> float:
    """Mean edge strength in a strip under the box's feet, a box wide either side, or inf at the frame's bottom."""
    x0, y0, x1, y1 = box
    height, width = y1 - y0, x1 - x0
    strip = edges[y1 : y1 + height // 4, max(0, x0 - width // 2) : x1 + width // 2]
    return float(strip.mean()) if len(strip) >= 3 else float("inf")


def mark_crowd(image: Image.Image, found: list[dict]) -> None:
    """Set "crowd" on each detection: true for a person too short, or not standing on the floor, to be a player."""
    edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    heights = [d["box"][3] - d["box"][1] for d in found if d["kind"] == "player"]
    tall = np.percentile(heights, 90) if heights else 0
    for d in found:
        height = d["box"][3] - d["box"][1]
        off_floor = height < NEAR_HEIGHT * tall and underfoot(edges, d["box"]) > FLOOR_CLUTTER
        d["crowd"] = bool(d["kind"] == "player" and (height < CROWD_HEIGHT * tall or off_floor))


def draw(image: Image.Image, found: list[dict]) -> Image.Image:
    canvas = Image.fromarray((np.asarray(image, dtype=np.float32) * 0.6).astype(np.uint8))
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=22)
    for d in reversed(found):
        color = COLORS["crowd" if d["crowd"] else d["kind"]]
        pen.rectangle(d["box"], outline=color, width=1 if d["crowd"] else 2)
        x0, y0 = d["box"][:2]
        pen.text((x0, y0 - 2), str(d["n"]), font=font, fill=color, stroke_width=3, stroke_fill="black", anchor="lb")
    return canvas


def main() -> None:
    clip, spec_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    frame = json.loads(spec_path.read_text())["frame"]
    image = frame_at(clip, frame)
    found = detect(image)
    mark_crowd(image, found)
    found.sort(key=lambda d: (d["crowd"], d["kind"] != "player", d["box"][0]))
    for n, d in enumerate(found):
        d["n"] = n
    out_dir.mkdir(parents=True, exist_ok=True)
    draw(image, found).save(out_dir / f"{frame:05d}.png")
    (out_dir / f"{frame:05d}.json").write_text(json.dumps(found, indent=2))
    for d in found:
        print(f"  {d['n']:>2} {'crowd' if d['crowd'] else d['kind']:<6} box {d['box']} score {d['score']}")
    print(f"  {out_dir / f'{frame:05d}.png'}")


if __name__ == "__main__":
    main()

"""Render cached masks as silhouette frames: players white, ball orange, hoop gray, on black.

Run by clips.py with the project venv:  .venv/bin/python pipeline/silhouette.py <masks.npz> <render.json> <out_dir>
render.json picks the crop. "follow" is "action" (every player and the hoop), "ball", or "holder"
(the player who has the ball; from the release on it widens to keep the shooter and the hoop in frame).
"zoom" sets the window to 1/zoom of the frame width for "ball" and "holder" (default 2).
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

COLORS = {"player": (255, 255, 255), "ball": (255, 122, 26), "hoop": (150, 150, 150)}
ASPECT = 16 / 9
PAD = 1.25
SMOOTH_FRAMES = 15
# Zooming out reads better slower than panning, so the window size eases over a longer stretch.
ZOOM_SMOOTH_FRAMES = 25
# SAM's masks are blocky once scaled up, so each one is resized smoothly, softened by this blur radius at
# output size, then cut with a narrow ramp so edges come out curved and anti-aliased instead of stepped.
EDGE_BLUR = 2.0
EDGE_RAMP = 0.12
# How close, in 720p pixels, the ball must be to a player's box to count as theirs.
HOLD_DISTANCE = 30


def load(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path)
    shape = tuple(data["shape"])
    masks = np.unpackbits(data["bits"], axis=-1, count=shape[-1]).astype(bool).reshape(shape)
    return [str(k) for k in data["kinds"]], masks


def boxes_of(masks: np.ndarray) -> list[tuple[int, int, int, int] | None]:
    """Per frame, the box around every pixel of the given objects, or None when none are visible."""
    boxes = []
    for frame in masks.any(axis=1):
        ys, xs = np.nonzero(frame)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None)
    return boxes


def smooth(values: np.ndarray, frames: int) -> np.ndarray:
    padded = np.pad(values, (frames // 2, frames // 2), mode="edge")
    return np.convolve(padded, np.ones(frames) / frames, mode="valid")


def glide(targets: list[tuple[float, float, float] | None], frame_w: int, frame_h: int) -> np.ndarray:
    """Per frame (x, y, width, height) of a 16:9 window that eases toward each (center x, center y, width) target.

    Gaps hold the last known target, and the window always stays inside the frame.
    """
    known = [t for t in targets if t]
    if not known:
        sys.exit("the followed objects are never visible; check the track prompts")
    last, filled = known[0], []
    for t in targets:
        last = t or last
        filled.append(last)
    cx, cy, width = (np.asarray(col, dtype=float) for col in zip(*filled))
    width = np.minimum(smooth(width, ZOOM_SMOOTH_FRAMES), min(frame_w, frame_h * ASPECT))
    height = width / ASPECT
    x = np.clip(smooth(cx, SMOOTH_FRAMES) - width / 2, 0, frame_w - width)
    y = np.clip(smooth(cy, SMOOTH_FRAMES) - height / 2, 0, frame_h - height)
    return np.stack([x, y, width, height], axis=1).astype(int)


def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def fit_width(box: tuple[int, int, int, int]) -> float:
    """Width of the padded 16:9 window that holds the box."""
    return max(box[2] - box[0], (box[3] - box[1]) * ASPECT) * PAD


def union(a, b):
    if not a or not b:
        return a or b
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def holders(kinds: list[str], masks: np.ndarray) -> tuple[list[int | None], int | None]:
    """Per frame, which player holds the ball (None while it's loose or in the air), and the release frame.

    The holder is sticky: a defender reaching in doesn't take the ball until it leaves the holder's reach,
    so a contested shot stays on the shooter. Once the ball reaches the rim, the player who released it is
    held for the rest of the clip, so the crop stays on the scorer instead of chasing the inbound pass.
    """
    players = [i for i, k in enumerate(kinds) if k == "player"]
    ball = boxes_of(masks[:, [i for i, k in enumerate(kinds) if k == "ball"]])
    hoop = boxes_of(masks[:, [i for i, k in enumerate(kinds) if k == "hoop"]])
    result, current, last_holder = [], None, None
    shooter = release = None
    for frame, b in enumerate(ball):
        if b and hoop[frame] and shooter is None:
            h, (bx, by) = hoop[frame], ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if h[0] - HOLD_DISTANCE <= bx <= h[2] + HOLD_DISTANCE and h[1] - HOLD_DISTANCE <= by <= h[3] + HOLD_DISTANCE:
                shooter, release = last_holder, frame
                # Re-point the flight frames before the rim at the shooter too, so the hold starts at release.
                while release > 0 and result[release - 1] is None:
                    release -= 1
                    result[release] = shooter
        if shooter is not None:
            result.append(shooter)
            continue
        gaps = {}
        if b:
            bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            for i in players:
                box = boxes_of(masks[frame:frame + 1, [i]])[0]
                if box:
                    gaps[i] = max(box[0] - bx, 0, bx - box[2]) + max(box[1] - by, 0, by - box[3])
        if current not in gaps or gaps[current] > HOLD_DISTANCE:
            near = [i for i, gap in gaps.items() if gap <= HOLD_DISTANCE]
            current = min(near, key=gaps.get) if near else None
        last_holder = current if current is not None else last_holder
        result.append(current)
    return result, release


def windows(kinds: list[str], masks: np.ndarray, render: dict) -> tuple[tuple[int, int], np.ndarray]:
    """The output size and each frame's (x, y, width, height) crop, per the render's follow mode."""
    _, _, h, w = masks.shape
    of_kind = lambda wanted: [i for i, k in enumerate(kinds) if k in wanted]
    follow = render.get("follow", "action")
    if follow == "action":
        boxes = boxes_of(masks[:, of_kind({"player", "hoop"})])
        width = max((fit_width(b) for b in boxes if b), default=w)
        targets = [(*center(b), width) if b else None for b in boxes]
        out_w = int(min(width, w))
    else:
        out_w = int(w / render.get("zoom", 2))
        if follow == "ball":
            targets = [(*center(b), out_w) if b else None for b in boxes_of(masks[:, of_kind({"ball"})])]
        else:
            held, release = holders(kinds, masks)
            hoop = boxes_of(masks[:, of_kind({"hoop"})])
            targets = []
            for frame, i in enumerate(held):
                box = boxes_of(masks[frame:frame + 1, [i]])[0] if i is not None else None
                if release is not None and frame >= release:
                    box = union(box, hoop[frame])
                    targets.append((*center(box), max(out_w, fit_width(box))) if box else None)
                else:
                    targets.append((*center(box), out_w) if box else None)
    return (out_w, int(out_w / ASPECT)), glide(targets, w, h)


def smooth_edges(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A 0..1 coverage map of the mask at output size, with smooth anti-aliased edges."""
    if not mask.any():
        return np.zeros((size[1], size[0]), dtype=np.float32)
    soft = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.BILINEAR)
    soft = np.asarray(soft.filter(ImageFilter.GaussianBlur(EDGE_BLUR)), dtype=np.float32) / 255
    return np.clip((soft - 0.5) / EDGE_RAMP + 0.5, 0, 1)


def main() -> None:
    masks_path, render_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    kinds, masks = load(masks_path)
    size, crops = windows(kinds, masks, json.loads(render_path.read_text()))
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame, (x, y, cw, ch) in enumerate(crops):
        canvas = np.zeros((size[1], size[0], 3), dtype=np.float32)
        for i, kind in enumerate(kinds):
            alpha = smooth_edges(masks[frame, i, y:y + ch, x:x + cw], size)
            canvas += (np.asarray(COLORS[kind], dtype=np.float32) - canvas) * alpha[..., None]
        Image.fromarray(canvas.round().astype(np.uint8)).save(out_dir / f"{frame:05d}.png")


if __name__ == "__main__":
    main()

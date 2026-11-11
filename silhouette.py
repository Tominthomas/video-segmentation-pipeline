"""Render cached masks as silhouette frames: players white, ball orange, hoop gray, on black.

Run by clips.py with the project venv:  .venv/bin/python pipeline/silhouette.py <masks.npz> <render.json> <out_dir>
render.json picks the crop. "follow" is "action" (every player and the hoop), "ball", or "holder"
(the player who has the ball, held on the shooter while it's in the air). "zoom" sets the window to
1/zoom of the frame width for "ball" and "holder" (default 2).
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

COLORS = {"player": (255, 255, 255), "ball": (255, 122, 26), "hoop": (150, 150, 150)}
ASPECT = 16 / 9
PAD = 1.25
SMOOTH_FRAMES = 15
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


def glide(boxes: list, win_w: int, win_h: int, frame_w: int, frame_h: int) -> np.ndarray:
    """Top-left corner per frame for a window centered on the boxes, held through gaps and smoothed."""
    known = [b for b in boxes if b]
    if not known:
        sys.exit("the followed objects are never visible; check the track prompts")
    last = known[0]
    centers = []
    for b in boxes:
        last = b or last
        centers.append(((last[0] + last[2]) / 2, (last[1] + last[3]) / 2))
    padded = np.pad(np.asarray(centers), ((SMOOTH_FRAMES // 2, SMOOTH_FRAMES // 2), (0, 0)), mode="edge")
    kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
    smooth = np.stack([np.convolve(padded[:, a], kernel, mode="valid") for a in (0, 1)], axis=1)
    return np.stack([np.clip(smooth[:, 0] - win_w / 2, 0, frame_w - win_w),
                     np.clip(smooth[:, 1] - win_h / 2, 0, frame_h - win_h)], axis=1).astype(int)


def holders(kinds: list[str], masks: np.ndarray) -> list[int | None]:
    """Per frame, which player holds the ball, or None while it's loose or in the air.

    The holder is sticky: a defender reaching in doesn't take the ball until it leaves the holder's reach,
    so a contested shot stays on the shooter. Once the ball reaches the rim, the player who released it is
    held for the rest of the clip, so the crop stays on the scorer instead of chasing the inbound pass.
    """
    players = [i for i, k in enumerate(kinds) if k == "player"]
    ball = boxes_of(masks[:, [i for i, k in enumerate(kinds) if k == "ball"]])
    hoop = boxes_of(masks[:, [i for i, k in enumerate(kinds) if k == "hoop"]])
    result, current, last_holder = [], None, None
    shooter = None
    for frame, b in enumerate(ball):
        if b and hoop[frame] and shooter is None:
            h, (bx, by) = hoop[frame], ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if h[0] - HOLD_DISTANCE <= bx <= h[2] + HOLD_DISTANCE and h[1] - HOLD_DISTANCE <= by <= h[3] + HOLD_DISTANCE:
                shooter = last_holder
                # Re-point the flight frames before the rim at the shooter too, so the hold starts at release.
                for back in range(frame - 1, -1, -1):
                    if result[back] is not None:
                        break
                    result[back] = shooter
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
    return result


def holder_boxes(kinds: list[str], masks: np.ndarray) -> list[tuple[int, int, int, int] | None]:
    return [boxes_of(masks[f:f + 1, [i]])[0] if i is not None else None for f, i in enumerate(holders(kinds, masks))]


def window(kinds: list[str], masks: np.ndarray, render: dict) -> tuple[int, int, np.ndarray]:
    _, _, h, w = masks.shape
    pick = lambda wanted: masks[:, [i for i, k in enumerate(kinds) if k in wanted]]
    follow = render.get("follow", "action")
    if follow in ("ball", "holder"):
        win_w = int(w / render.get("zoom", 2))
        win_h = int(win_w / ASPECT)
        boxes = holder_boxes(kinds, masks) if follow == "holder" else boxes_of(pick({"ball"}))
        return win_w, win_h, glide(boxes, win_w, win_h, w, h)
    boxes = boxes_of(pick({"player", "hoop"}))
    known = [b for b in boxes if b]
    span_w = max(b[2] - b[0] for b in known) * PAD if known else w
    span_h = max(b[3] - b[1] for b in known) * PAD if known else h
    win_w = int(min(w, max(span_h, span_w / ASPECT) * ASPECT))
    win_h = int(win_w / ASPECT)
    return win_w, win_h, glide(boxes, win_w, win_h, w, h)


def main() -> None:
    masks_path, render_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    kinds, masks = load(masks_path)
    win_w, win_h, origins = window(kinds, masks, json.loads(render_path.read_text()))
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(masks.shape[0]):
        x, y = origins[frame]
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        for i, kind in enumerate(kinds):
            canvas[masks[frame, i, y:y + win_h, x:x + win_w]] = COLORS[kind]
        Image.fromarray(canvas).save(out_dir / f"{frame:05d}.png")


if __name__ == "__main__":
    main()

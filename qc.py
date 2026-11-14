"""Flag frame ranges where tracked masks look wrong, so a review starts at the suspect frames.

Run by clips.py with the project venv:  .venv/bin/python qc.py <masks.npz> <mystery> <qc.json> [shape]
[shape] is the clip's matte.npz or detail.npz, whichever draws the mystery player.
<mystery> is a JSON list of the objects the manifest marks as the mystery player.
qc.json holds {"flags": [{"check", "objects", "frames": [first, last], "worst"}]}, one entry per run of
consecutive flagged frames. Sizes are measured against the median player height, so the checks hold at any zoom.
A clip without exactly one mystery player gets one "mystery count" flag over every frame. So does an object
that is never visible ("never visible"), and a clip whose players give no height to measure by ("no player height"),
which skips the size and jump checks.
With a matte, two flags span its frames when it struggled: "matte fallback", the share of frames that fell back to
SAM, and "matte ghosting", the share of frames whose raw matte left more than GHOST_FRAME of his SAM core under 0.5.
Either means the review should compare the clip with "shape": "detail" and "sam".
With a detail pass, three flags mark runs of frames to look at with `clips.py crops --detail` (docs/SPEC.md#detail):
"detail disagrees", where his owned mask overlaps his SAM mask with an IoU under DISAGREE, "detail speckle", where
the speckle rule changed more than SPECKLED of his area, and "detail ceded", where other players took more than CEDED.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from detail import Detail
from matte import Matte, resize
from silhouette import ball_checks, erode, load, others_of, player_height, runs, shapes

PLAYER_OVERLAP = 0.5
# Players and the hoop never move this far in one frame. The ball can, so it is judged by its shape instead.
MAX_JUMP = 0.35
HOOP_AREA_RATIO = 1.6
# Shape checks flag only runs this long. Shorter dips are a hand or blur crossing the ball and barely show in playback.
MIN_RUN = 6
# A player or the hoop missing this long has lost its track, unless it was at the frame's edge on each side of the
# gap, so it left or entered through it. A gap before the first sighting or after the last has one side. SAM
# recovers from shorter occlusions on its own.
MISSING_RUN = 10
EDGE = 8
# Set from five McDaniels clips and Nembhard's six (docs/SPEC.md#matting). Of the McDaniels mattes, the Houston one
# won blind judging with 10% of frames ghosting, and the ones that lost had 26 to 66%. Nembhard's close clip had 2%.
# judge.py checks all three against every verdict in judgments.json (docs/SPEC.md#judgments).
GHOST_FRAME = 0.05
MATTE_GHOSTING = 0.2
MATTE_FALLBACK = 0.05
DISAGREE = 0.45
SPECKLED = 0.03
CEDED = 0.08


def missing(area: np.ndarray, box: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """Frames of a gap in one object's track, except a gap it leaves or enters through the frame's edge."""
    bad = np.zeros(len(area), bool)
    at_edge = (box[:, :2].min(axis=1) <= EDGE) | (box[:, 2] >= frame_w - EDGE) | (box[:, 3] >= frame_h - EDGE)
    for first, last in runs(np.flatnonzero(area == 0)):
        sides = [f for f in (first - 1, last + 1) if 0 <= f < len(area)]
        if not all(at_edge[sides]):
            bad[first : last + 1] = True
    return bad


def overlap(masks: np.ndarray, stats: list, a: int, b: int) -> np.ndarray:
    """Per frame, the share of the smaller of objects a and b that the other covers, from each object's shapes()."""
    shared = np.zeros(len(masks))
    for f in range(len(masks)):
        x0, y0 = np.fmax(stats[a][1][f, :2], stats[b][1][f, :2])
        x1, y1 = np.fmin(stats[a][1][f, 2:], stats[b][1][f, 2:])
        if x0 < x1 and y0 < y1:
            y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
            both = (masks[f, a, y0:y1, x0:x1] & masks[f, b, y0:y1, x0:x1]).sum()
            shared[f] = both / min(stats[a][0][f], stats[b][0][f])
    return shared


def flagged(check: str, objects: list[int], bad: np.ndarray, values: np.ndarray, worst, min_run=MIN_RUN) -> list[dict]:
    """One flag per run of at least min_run bad frames, with the worst of its values."""
    return [
        {
            "check": check,
            "objects": objects,
            "frames": [first, last],
            "worst": round(float(worst(values[first : last + 1])), 2),
        }
        for first, last in runs(np.flatnonzero(bad))
        if last - first + 1 >= min_run
    ]


def flags(kinds: list[str], masks: np.ndarray, mystery: list[int]) -> list[dict]:
    """Every flagged run of frames in the [frame, object, h, w] masks, by check and object."""
    stats = [shapes(masks[:, i]) for i in range(len(kinds))]
    players = [i for i, kind in enumerate(kinds) if kind == "player"]
    scale = player_height(kinds, stats, masks.shape[2])
    measurable = np.isfinite(scale) and scale > 0
    every = np.ones(len(masks), bool)
    found = []

    def flag(check: str, objects: list[int], bad: np.ndarray, values: np.ndarray, worst, min_run=MIN_RUN) -> None:
        found.extend(flagged(check, objects, bad, values, worst, min_run))

    if len(mystery) != 1:
        flag("mystery count", mystery, every, np.full(len(masks), len(mystery)), np.max, 1)
    if not measurable:
        flag("no player height", players, every, np.zeros(len(masks)), np.max, 1)

    for i, kind in enumerate(kinds):
        area, box, centroid = stats[i]
        if not (area > 0).any():
            flag("never visible", [i], every, area, np.max, 1)
            continue
        if kind == "ball":
            for check, (bad, values) in ball_checks(area, box, scale).items():
                if check != "size" or measurable:
                    flag(f"ball {check}", [i], bad, values, np.nanmin if check == "fill" else np.nanmax)
        if kind == "hoop":
            ratio = area / np.median(area[area > 0])
            off = np.fmax(ratio, 1 / ratio)
            flag("hoop area", [i], (area > 0) & (off > HOOP_AREA_RATIO), off, np.nanmax)
        if kind != "ball":
            flag("missing", [i], missing(area, box, masks.shape[3], masks.shape[2]), every, np.sum, MISSING_RUN)
            if measurable:
                jump = np.r_[0, np.linalg.norm(np.diff(centroid, axis=0), axis=1)] / scale
                flag("jump", [i], jump > MAX_JUMP, jump, np.nanmax, min_run=1)

    for a, b in combinations(players, 2):
        shared = overlap(masks, stats, a, b)
        flag("player overlap", [a, b], shared > PLAYER_OVERLAP, shared, np.max)
    return found


def ghosting(matte: Matte, sam: np.ndarray) -> np.ndarray:
    """Per matted frame, the share of his SAM mask shrunk by the gate that the raw matte left under 0.5."""
    shares = np.zeros(len(matte.alpha))
    (cw, ch), clip_size = matte.crop, sam.shape[:0:-1]
    for j, frame in enumerate(matte.frames()):
        full = np.zeros(matte.source[::-1], np.float32)
        x0, y0 = matte.origin[j]
        full[y0 : y0 + ch, x0 : x0 + cw] = resize(matte.alpha[j].astype(np.float32) / 255, (cw, ch))
        core = erode(sam[frame], matte.gate)
        shares[j] = (resize(full, clip_size)[core] < 0.5).mean() if core.any() else 0
    return shares


def matte_shares(ghost: np.ndarray, fallback: np.ndarray, ghost_frame: float = GHOST_FRAME) -> dict[str, float]:
    """What the clip-wide matte flags measure, from ghosting()'s per-frame shares and the matte's fallback frames.
    judge.py passes other values of ghost_frame to find which ones still agree with every judged clip."""
    return {"matte fallback": float(fallback.mean()), "matte ghosting": float((ghost > ghost_frame).mean())}


def matte_flags(matte: Matte, sam: np.ndarray) -> list[dict]:
    """The clip-wide matte flags, given his [frame, h, w] SAM masks."""
    span = [matte.first, matte.first + len(matte.alpha) - 1]
    limits = {"matte fallback": MATTE_FALLBACK, "matte ghosting": MATTE_GHOSTING}
    return [
        {"check": check, "objects": [], "frames": span, "worst": round(value, 2)}
        for check, value in matte_shares(ghosting(matte, sam), matte.fallback).items()
        if value > limits[check]
    ]


def detail_flags(detail: Detail, kinds: list[str], masks: np.ndarray, mystery: int) -> list[dict]:
    """The detail pass's per-frame flags, given every object's [frame, object, h, w] masks."""
    iou, speckle, ceded = np.ones(len(masks)), np.zeros(len(masks)), np.zeros(len(masks))
    for f in detail.frames():
        his = masks[f, mystery]
        if not his.any():
            continue
        owned = detail.owned(f, his, others_of(kinds, masks[f], mystery))
        area = max(int((owned.alpha >= 0.5).sum()), 1)
        iou[f] = detail.clip_iou(f, owned, his)
        speckle[f], ceded[f] = owned.speckle.sum() / area, owned.ceded.sum() / area
    return [
        *flagged("detail disagrees", [mystery], iou < DISAGREE, iou, np.min, 1),
        *flagged("detail speckle", [mystery], speckle > SPECKLED, speckle, np.max, 1),
        *flagged("detail ceded", [mystery], ceded > CEDED, ceded, np.max, 1),
    ]


def main() -> None:
    masks_path, mystery, out = Path(sys.argv[1]), json.loads(sys.argv[2]), Path(sys.argv[3])
    kinds, masks = load(masks_path)
    # Absent objects make NaN boxes and zero areas, which the comparisons already treat as not flagged.
    with np.errstate(divide="ignore", invalid="ignore"):
        found = flags(kinds, masks, mystery)
    shape = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    if shape and shape.name == "matte.npz":
        found += [{**f, "objects": mystery} for f in matte_flags(Matte.load(shape), masks[:, mystery[0]])]
    if shape and shape.name == "detail.npz":
        found += detail_flags(Detail.load(shape), kinds, masks, mystery[0])
    out.write_text(json.dumps({"flags": found}, indent=2))


if __name__ == "__main__":
    main()

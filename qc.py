"""Flag frame ranges where tracked masks look wrong, so a review starts at the suspect frames.

Run by clips.py with the project venv:  .venv/bin/python pipeline/qc.py <masks.npz> <mystery> <qc.json>
<mystery> is a JSON list of the objects the manifest marks as the mystery player.
qc.json holds {"flags": [{"check", "objects", "frames": [first, last], "worst"}]}, one entry per run of
consecutive flagged frames. Sizes are measured against the median player height, so the checks hold at any zoom.
A clip without exactly one mystery player gets one "mystery count" flag over every frame. So does an object
that is never visible ("never visible"), and a clip whose players give no height to measure by ("no player height"),
which skips the size and jump checks.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from silhouette import ball_checks, load, player_height, runs, shapes

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


def flags(kinds: list[str], masks: np.ndarray, mystery: list[int]) -> list[dict]:
    """Every flagged run of frames in the [frame, object, h, w] masks, by check and object."""
    stats = [shapes(masks[:, i]) for i in range(len(kinds))]
    players = [i for i, kind in enumerate(kinds) if kind == "player"]
    scale = player_height(kinds, stats, masks.shape[2])
    measurable = np.isfinite(scale) and scale > 0
    every = np.ones(len(masks), bool)
    found = []

    def flag(check: str, objects: list[int], bad: np.ndarray, values: np.ndarray, worst, min_run=MIN_RUN) -> None:
        for first, last in (r for r in runs(np.flatnonzero(bad)) if r[1] - r[0] + 1 >= min_run):
            found.append(
                {
                    "check": check,
                    "objects": objects,
                    "frames": [first, last],
                    "worst": round(float(worst(values[first : last + 1])), 2),
                }
            )

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


def main() -> None:
    masks_path, mystery, out = Path(sys.argv[1]), json.loads(sys.argv[2]), Path(sys.argv[3])
    # Absent objects make NaN boxes and zero areas, which the comparisons already treat as not flagged.
    with np.errstate(divide="ignore", invalid="ignore"):
        found = flags(*load(masks_path), mystery)
    out.write_text(json.dumps({"flags": found}, indent=2))


if __name__ == "__main__":
    main()

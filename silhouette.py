"""Render cached masks as silhouette frames: players white, ball orange, hoop gray, on black.

Run by clips.py with the project venv:  .venv/bin/python silhouette.py <masks.npz> <render.json> <out_dir>
render.json picks the crop. "follow" is "action" (every player and the hoop), "ball", "holder"
(the player who has the ball; from the release on it widens to keep the shooter, ball and hoop in frame),
or "full" (the whole frame, pillarboxed in black to 16:9, for portrait sources that no 16:9 crop can hold).
"zoom" sets the window to 1/zoom of the frame width for "ball" and "holder" (default 2).
"frames" [first, end) renders only that slice of the clip; the crop is still planned over the whole clip.
"mystery" lists the mystery player's object index. When it is set, every other player turns dark gray.
"matte" is the path of his matte.npz from matte.py, and "detail" of his detail.npz from detail.py. With either, his
layer is sampled from the source and every frame renders 720 tall; a frame either falls back on draws his SAM mask.
With "detail", the gap between him and a ball he holds is closed (docs/SPEC.md#detail).
With a mystery player, the ball draws in his white, so he and the ball read as one shape; "ball": "orange" keeps
it orange. "ball_edge" draws a dark edge that many pixels wide at 720 tall where the ball overlaps him (default 0).
The ball keeps SAM's own mask on frames where it passes ball_track's checks, so it has the real ball's shape and
hands cover it. A short gap draws a disc on the cleaned track so the ball doesn't flicker, and a longer one leaves
it hidden: the ball isn't the focus, and the player is. The full track still steers the crop. "ball_keys"
[[frame, x, y], ...] place it by hand where SAM's mask can't, such as a ball hidden in a crowd, and always draw.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image, ImageFilter

COLORS = {"player": (255, 255, 255), "ball": (255, 122, 26), "hoop": (150, 150, 150)}
# With a mystery player, each role draws in this order, so he covers a hoop mask that bleeds onto his body
# and only the ball draws over him.
FOCUS = {"player": (90, 90, 90), "hoop": (150, 150, 150), "mystery": (255, 255, 255), "ball": (255, 122, 26)}
TINTS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (170, 110, 40),
]
ASPECT = 16 / 9
PAD = 1.25
SMOOTH_FRAMES = 15
# Zooming out reads better slower than panning, so the window size eases over a longer stretch.
ZOOM_SMOOTH_FRAMES = 25
# SAM's masks are blocky once scaled up, so each one is resized smoothly, softened by this blur radius at
# output size, then cut with a narrow ramp so edges come out curved and anti-aliased instead of stepped.
EDGE_BLUR = 2.0
EDGE_RAMP = 0.12
# A round ball fills pi/4 of its box; a ball mask far from that has merged with a hand, head or nothing.
BALL_FILL = np.pi / 4
BALL_FILL_TOLERANCE = 0.25
# Motion blur stretches a fast lob to about 0.35 of a player's height, so only a larger ball means a merged mask.
BALL_MAX_SIDE = 0.4
# Blurred flights in the first ten clips stay under 2:1. Every accepted mask over 2.5:1 was an arm, the rim or a
# ball cut by the frame's edge.
BALL_MAX_ASPECT = 2.5
# A ball mask this far from the median of its neighbors, in player heights, has jumped onto something else.
BALL_MAX_JUMP = 0.5
BALL_NEIGHBORS = 3
# After a make SAM's ball mask can latch onto the orange rim. A candidate this share inside the hoop's mask whose
# place relative to the hoop moves less than RIM_STILL player heights over RIM_FRAMES frames is the rim. A ball
# rolling on the rim sits mostly outside the hoop's mask, and one dropping through the net is gone well within
# the 0.6 s window, so a ball briefly hidden in the rim keeps its mask.
RIM_INSIDE = 0.6
RIM_STILL = 0.06
RIM_FRAMES = 18
# A gap in the ball's track is fitted through the known centers up to this many frames either side of it.
FIT_FRAMES = 8
# A gap longer than this is more than one flight, such as a dunk dropping through the net and bouncing, so it is
# drawn straight between its two nearest centers instead of extending one arc across it.
ARC_FRAMES = 15
# How close, in player heights, the ball must be to a player's box to count as theirs, and to the hoop's box to
# count as at the rim. About 30 px at the 150 to 170 px players of a broadcast angle.
HOLD_REACH = 0.19
# A new holder must keep the ball in reach this many seen frames (see possessions), so a ball in flight passing a
# contesting defender's hand doesn't take it from the player who has it.
HOLD_FRAMES = 5
# In the first ten clips a contested shot brushes a defender's reach for at most 2 seen frames, and every shooter
# kept the ball for at least 5, so a quick catch and shoot of 3 still names its shooter.
SHOT_FRAMES = 3
# A run of inferred ball frames up to this long is drawn as a disc so the ball doesn't blink. A longer one, such
# as a ball lost against the crowd in flight, is left hidden.
SHOW_GAP = 3
# A matted or detail render's height, or the source's when that is less.
MATTED_HEIGHT = 720
# The ball's colours with a mystery player, by the render's "ball". White is his colour.
BALLS = {"white": FOCUS["mystery"], "orange": FOCUS["ball"]}


def load(path: Path, frames: list[int] | None = None) -> tuple[list[str], np.ndarray]:
    """A packed mask file's object kinds and its masks, only the listed frames when given.

    A clip's masks.npz is [frame, object, h, w] with its kinds; one object's cache is [frame, h, w] with none.
    """
    data = np.load(path)
    bits = data["bits"] if frames is None else data["bits"][frames]
    kinds = [str(k) for k in data["kinds"]] if "kinds" in data else []
    return kinds, np.unpackbits(bits, axis=-1, count=int(data["shape"][-1])).view(bool)


def shapes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame of one object's [frame, h, w] masks: its area, box (x0, y0, x1, y1) and centroid (x, y).

    Box and centroid are NaN on frames where the object isn't visible.
    """
    rows, cols = mask.sum(axis=2), mask.sum(axis=1)
    area = rows.sum(axis=1).astype(float)
    seen = area > 0
    box = np.full((len(mask), 4), np.nan)
    centroid = np.full((len(mask), 2), np.nan)
    for f in np.flatnonzero(seen):
        ys, xs = np.flatnonzero(rows[f]), np.flatnonzero(cols[f])
        box[f] = xs[0], ys[0], xs[-1] + 1, ys[-1] + 1
        centroid[f] = cols[f] @ np.arange(len(cols[f])) / area[f], rows[f] @ np.arange(len(rows[f])) / area[f]
    return area, box, centroid


def runs(frames: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted frame numbers into (first, last) runs of consecutive frames."""
    if not len(frames):
        return []
    breaks = np.flatnonzero(np.diff(frames) > 1)
    firsts, lasts = frames[np.r_[0, breaks + 1]], frames[np.r_[breaks, len(frames) - 1]]
    return list(zip(firsts.tolist(), lasts.tolist(), strict=True))


def player_height(kinds: list[str], stats: list, frame_h: int) -> float:
    """Median player box height over the clip, from each object's shapes(), or a quarter frame without players."""
    heights = [stats[i][1][:, 3] - stats[i][1][:, 1] for i, kind in enumerate(kinds) if kind == "player"]
    return float(np.nanmedian(np.concatenate(heights))) if heights else frame_h / 4


def on_rim(ball: np.ndarray, hoop: np.ndarray, good: np.ndarray, offset: np.ndarray, height: float) -> np.ndarray:
    """Frames whose ball mask is the rim: mostly inside the hoop's mask and still relative to the hoop, by offset
    (ball center minus hoop center), for RIM_FRAMES in a row. Only frames in good are considered."""
    inside = np.zeros(len(ball), bool)
    for f in np.flatnonzero(good):
        inside[f] = (ball[f] & hoop[f]).sum() >= RIM_INSIDE * ball[f].sum()
    rim = np.zeros(len(ball), bool)
    for first, last in runs(np.flatnonzero(inside & ~np.isnan(offset).any(axis=1))):
        for start in range(first, last - RIM_FRAMES + 2):
            window = offset[start : start + RIM_FRAMES]
            if np.ptp(window, axis=0).max() <= RIM_STILL * height:
                rim[start : start + RIM_FRAMES] = True
    return rim


def ball_checks(area: np.ndarray, box: np.ndarray, height: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per shape check a ball mask must pass, the frames that fail it and the value it measures, from shapes():
    "fill" of its box near a disc's, "aspect" no longer than motion blur stretches it, and "size" no bigger than
    BALL_MAX_SIDE of a player. Frames without a mask fail none."""
    width, tall = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    fill = area / (width * tall)
    aspect = np.fmax(width, tall) / np.fmin(width, tall)
    side = np.fmax(width, tall) / height
    return {
        "fill": (np.abs(fill - BALL_FILL) > BALL_FILL_TOLERANCE, fill),
        "aspect": (aspect > BALL_MAX_ASPECT, aspect),
        "size": (side > BALL_MAX_SIDE, side),
    }


class Track(NamedTuple):
    """The ball per frame: its center (x, y), its radius, and where each center came from, per ball_track()."""

    xy: np.ndarray
    radius: float
    source: np.ndarray

    def seen(self) -> np.ndarray:
        """Per frame, whether SAM's mask or a key placed the ball, rather than a fit or hold inferring it."""
        return np.isin(self.source, ("mask", "key"))

    def shown(self) -> np.ndarray:
        """Per frame, whether the silhouette draws the ball: every seen frame, and each run of inferred ("fit" or
        "hold") frames up to SHOW_GAP long."""
        show = self.seen()
        for first, last in runs(np.flatnonzero(np.isin(self.source, ("fit", "hold")))):
            show[first : last + 1] = last - first + 1 <= SHOW_GAP
        return show


def ball_track(kinds: list[str], masks: np.ndarray, keys: list) -> Track:
    """Per frame the ball's center (x, y), its radius, and where each center came from.

    A frame keeps SAM's ball mask ("mask") only when it looks like a ball (see ball_checks), is not latched onto the
    rim (see on_rim), and is near where its neighbors put it. Keys [frame, x, y] override the mask ("key"). Each gap
    up to ARC_FRAMES long is fitted through the known centers on both sides, x linear and y quadratic in time, and a
    longer one runs straight between its nearest two ("fit"). A gap at either end holds the nearest known center
    ("hold"). Both work relative to the hoop when it is in view throughout, so a camera pan cancels out. With no mask
    or key to go on, every center is NaN.
    """
    stats = [shapes(masks[:, i]) for i in range(len(kinds))]
    height = player_height(kinds, stats, masks.shape[2])
    area, box, center = stats[kinds.index("ball")]
    hoop = stats[kinds.index("hoop")][2] if "hoop" in kinds else np.full_like(center, np.nan)
    good = (area > 0) & ~np.any([bad for bad, _ in ball_checks(area, box, height).values()], axis=0)
    if "hoop" in kinds:
        good &= ~on_rim(masks[:, kinds.index("ball")], masks[:, kinds.index("hoop")], good, center - hoop, height)
    candidates = np.flatnonzero(good)
    for f in candidates:
        near = candidates[(np.abs(candidates - f) <= BALL_NEIGHBORS) & (candidates != f)]
        if len(near) >= 2 and np.linalg.norm(center[f] - np.median(center[near], axis=0)) > BALL_MAX_JUMP * height:
            good[f] = False
    radius = float(np.median(np.sqrt(area[good] / np.pi))) if good.any() else 0.06 * height
    xy = np.where(good[:, None], center, np.nan)
    source = np.where(good, "mask", "")
    for f, x, y in keys:
        if not 0 <= f < len(masks):
            sys.exit(f"ball key frame {f} is outside the clip's {len(masks)} frames")
        xy[f], source[f] = (x, y), "key"
    known = np.flatnonzero(source != "")
    for first, last in runs(np.flatnonzero(source == "")):
        before = known[(known < first) & (known >= first - FIT_FRAMES)]
        after = known[(known > last) & (known <= last + FIT_FRAMES)]
        before = before if len(before) and last - first < ARC_FRAMES else known[known < first][-1:]
        after = after if len(after) and last - first < ARC_FRAMES else known[known > last][:1]
        fit = np.r_[before, after]
        if not len(fit):
            continue
        gap = np.arange(first, last + 1)
        span = np.r_[fit, gap]
        origin = hoop if not np.isnan(hoop[span]).any() else np.zeros_like(hoop)
        rel = xy[fit] - origin[fit]
        if len(before) and len(after):
            fx = np.polyfit(fit, rel[:, 0], 1)
            fy = np.polyfit(fit, rel[:, 1], min(2, len(fit) - 1))
            xy[gap] = np.stack([np.polyval(fx, gap), np.polyval(fy, gap)], axis=1) + origin[gap]
            source[gap] = "fit"
        else:
            xy[gap] = rel[0 if len(before) == 0 else -1] + origin[gap]
            source[gap] = "hold"
    return Track(xy, radius, source)


def draw_ball(kinds: list[str], masks: np.ndarray, keys: list) -> Track | None:
    """Keep the ball's masks on ball_track()'s "mask" frames and draw a disc on every other frame, print which
    frames were filled in, and return the track."""
    if "ball" not in kinds:
        return None
    track = ball_track(kinds, masks, keys)
    xy, radius, source = track
    ys, xs = np.ogrid[: masks.shape[2], : masks.shape[3]]
    ball = kinds.index("ball")
    for f, (x, y) in enumerate(xy):
        if source[f] == "mask":
            continue
        masks[f, ball] = (xs - x) ** 2 + (ys - y) ** 2 <= radius**2 if not np.isnan(x) else False
    counts = ", ".join(
        f"{kind} {len(frames)}" + (f" {runs(frames)}" if kind in ("fit", "hold") else "")
        for kind in ("mask", "key", "fit", "hold")
        if len(frames := np.flatnonzero(source == kind))
    )
    print(f"  ball radius {radius:.1f}: {counts or 'never placed'}", flush=True)
    return track


def hide_ball(kinds: list[str], masks: np.ndarray, track: Track) -> None:
    """Clear the ball on every frame the track doesn't show."""
    masks[~track.shown(), kinds.index("ball")] = False


def dilate(image: np.ndarray, radius: int) -> np.ndarray:
    """The image's max over a disk of the radius around each pixel. Only the box around its nonzero pixels is worked."""
    out = image.copy()
    ys, xs = np.nonzero(image)
    if radius <= 0 or not len(ys):
        return out
    h, w = image.shape
    y0, y1 = max(ys.min() - radius, 0), min(ys.max() + radius + 1, h)
    x0, x1 = max(xs.min() - radius, 0), min(xs.max() + radius + 1, w)
    region = out[y0:y1, x0:x1]
    src = np.pad(image[y0:y1, x0:x1], radius)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                shifted = src[radius + dy : radius + dy + y1 - y0, radius + dx : radius + dx + x1 - x0]
                np.maximum(region, shifted, out=region)
    return out


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """The mask shrunk by a disk of the radius. Past the frame's edge counts as inside, so a player the edge cuts
    keeps his core there."""
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    if radius <= 0 or not len(ys):
        return mask.copy()
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    edges = ((y0 == 0, y1 == h), (x0 == 0, x1 == w))
    sub = np.pad(mask[y0:y1, x0:x1], radius, constant_values=edges)
    out[y0:y1, x0:x1] = ~dilate(~sub, radius)[radius:-radius, radius:-radius]
    return out


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
    cx, cy, width = (np.asarray(col, dtype=float) for col in zip(*filled, strict=False))
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


class Possession(NamedTuple):
    """A player keeping the ball in reach from his first seen frame to his last, how many frames the ball was seen in
    his reach, and its side-to-side place on him (ball x minus his box's center x) at the first."""

    player: int
    first: int
    last: int
    seen: int
    side: float


def possessions(near: dict[int, dict[int, tuple[float, float]]], reach: float) -> list[Possession]:
    """Possessions in time order, from each seen frame's players in reach as {player: (gap, side)}.

    One seen HOLD_FRAMES times is held: it lasts while its player keeps the ball in reach, across unseen frames too,
    so a defender reaching in doesn't take it. A shorter one lasts only while he stays nearest with the ball on the
    same side of him, across unseen gaps up to SHOW_GAP, so a ball in flight crossing his box on screen or one lost
    from view for longer ends it.
    """
    found: list[Possession] = []
    previous = None
    for frame, players in near.items():
        nearest = min(players, key=lambda i: players[i][0]) if players else None
        last = found[-1] if found else None
        if (
            last is not None
            and last.last == previous
            and last.player in players
            and (
                last.seen >= HOLD_FRAMES
                or (
                    frame - last.last <= SHOW_GAP + 1
                    and nearest == last.player
                    and abs(players[nearest][1] - last.side) <= reach
                )
            )
        ):
            found[-1] = last._replace(last=frame, seen=last.seen + 1)
        elif nearest is not None:
            found.append(Possession(nearest, frame, frame, 1, players[nearest][1]))
        previous = frame
    return found


def holders(
    kinds: list[str], masks: np.ndarray, track: Track | None
) -> tuple[list[int | None], int | None, int | None]:
    """Per frame, which player holds the ball (None while it's loose or in the air), the release frame, and the
    frame the ball reaches the rim.

    Only frames where the track saw the ball (see Track.seen) count toward possessions or reach the rim. A fitted
    or held center is a guess, and one that crosses a player or the hoop would invent a holder or a finish. A held
    possession (see possessions) is its player's from its first frame through the unseen frames after its last.
    The shooter is the player of the last possession before the rim at least SHOT_FRAMES long, so a quick catch and
    shoot keeps its shooter and a ball in flight brushing a player doesn't. He holds the ball from that
    possession's first frame to the end of the clip, so the crop stays on the scorer instead of chasing the inbound
    pass, and the release is the frame after his last in reach. Without a shooter the release is the rim frame.
    """
    players = [i for i, k in enumerate(kinds) if k == "player"]
    stats = [shapes(masks[:, i]) if i in players else None for i in range(len(kinds))]
    reach = HOLD_REACH * player_height(kinds, stats, masks.shape[2])
    seen = track.seen() if track else np.zeros(len(masks), bool)
    ball = boxes_of(masks[:, [i for i, k in enumerate(kinds) if k == "ball"]])
    ball = {f: center(b) for f, (b, s) in enumerate(zip(ball, seen, strict=True)) if b and s}
    hoop = boxes_of(masks[:, [i for i, k in enumerate(kinds) if k == "hoop"]])
    rim = next(
        (
            f
            for f, (bx, by) in ball.items()
            if (h := hoop[f]) and h[0] - reach <= bx <= h[2] + reach and h[1] - reach <= by <= h[3] + reach
        ),
        None,
    )
    near = {}
    for frame, (bx, by) in ball.items():
        if rim is not None and frame >= rim:
            break
        near[frame] = {}
        for i in players:
            if box := boxes_of(masks[frame : frame + 1, [i]])[0]:
                gap = max(box[0] - bx, 0, bx - box[2]) + max(box[1] - by, 0, by - box[3])
                if gap <= reach:
                    near[frame][i] = gap, bx - center(box)[0]
    found = possessions(near, reach)
    result: list[int | None] = [None] * len(masks)
    for p in found:
        if p.seen >= HOLD_FRAMES:
            result[p.first : p.last + 1] = [p.player] * (p.last - p.first + 1)
    for frame in range(1, len(masks)):
        if frame not in ball and result[frame] is None:
            result[frame] = result[frame - 1]
    shot = next((p for p in reversed(found) if p.seen >= SHOT_FRAMES), None)
    if rim is None or shot is None:
        return result, rim, rim
    result[shot.first :] = [shot.player] * (len(masks) - shot.first)
    return result, shot.last + 1, rim


def windows(
    kinds: list[str], masks: np.ndarray, render: dict, track: Track | None
) -> tuple[tuple[int, int], np.ndarray]:
    """The output size and each frame's (x, y, width, height) crop, per the render's follow mode."""
    _, _, h, w = masks.shape

    def of_kind(wanted):
        return [i for i, k in enumerate(kinds) if k in wanted]

    follow = render.get("follow", "action")
    targets: list[tuple[float, float, float] | None]
    if follow == "full":
        return (max(w, int(h * ASPECT)) // 2 * 2, h), np.tile([0, 0, w, h], (masks.shape[0], 1))
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
            held, release, _ = holders(kinds, masks, track)
            hoop = boxes_of(masks[:, of_kind({"hoop"})])
            ball = boxes_of(masks[:, of_kind({"ball"})])
            targets = []
            for frame, i in enumerate(held):
                box = boxes_of(masks[frame : frame + 1, [i]])[0] if i is not None else None
                if release is not None and frame >= release:
                    # The ball joins the shooter and hoop, so a camera that pans off the shooter still shows the finish.
                    box = union(union(box, hoop[frame]), ball[frame])
                    targets.append((*center(box), max(out_w, fit_width(box))) if box else None)
                else:
                    targets.append((*center(box), out_w) if box else None)
    return (out_w, int(out_w / ASPECT)), glide(targets, w, h)


def layers(
    kinds: list[str], mystery: list[int], ball: tuple[int, int, int] = FOCUS["ball"]
) -> list[tuple[int, tuple[int, int, int]]]:
    """Each object's index and color, in draw order, the ball in the given color. Without a mystery player, manifest
    order in COLORS."""
    if not mystery:
        return [(i, COLORS[kind]) for i, kind in enumerate(kinds)]
    roles = ["mystery" if i in mystery else kind for i, kind in enumerate(kinds)]
    colors = {**FOCUS, "ball": ball}
    return [(i, colors[role]) for role in FOCUS for i, r in enumerate(roles) if r == role]


def smooth_edges(mask: np.ndarray, size: tuple[int, int], blur: float = EDGE_BLUR) -> np.ndarray:
    """A 0..1 coverage map of the mask at output size, with smooth anti-aliased edges."""
    if not mask.any():
        return np.zeros((size[1], size[0]), dtype=np.float32)
    soft = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.BILINEAR)
    soft = np.asarray(soft.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) / 255
    return np.clip((soft - 0.5) / EDGE_RAMP + 0.5, 0, 1)


def ball_edge(ball: np.ndarray, him: np.ndarray, width: float) -> np.ndarray:
    """The dark edge drawn under the ball where it overlaps him: the ball grown by width pixels, a fractional width
    anti-aliased, over his coverage."""
    solid = ball >= 0.5
    ys, xs = np.nonzero(solid)
    if width <= 0 or not len(ys):
        return np.zeros_like(ball)
    reach = int(np.ceil(width + 0.5))
    h, w = ball.shape
    y0, y1 = max(ys.min() - reach, 0), min(ys.max() + reach + 1, h)
    x0, x1 = max(xs.min() - reach, 0), min(xs.max() + reach + 1, w)
    src = np.pad(solid[y0:y1, x0:x1], reach).astype(np.float32)
    grown = np.zeros((y1 - y0, x1 - x0), np.float32)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            weight = min(max(width + 0.5 - float(np.hypot(dy, dx)), 0.0), 1.0)
            if weight:
                shifted = src[reach + dy : reach + dy + y1 - y0, reach + dx : reach + dx + x1 - x0]
                np.maximum(grown, weight * shifted, out=grown)
    edge = np.zeros_like(ball)
    edge[y0:y1, x0:x1] = np.maximum(grown, ball[y0:y1, x0:x1])
    return np.minimum(edge, him)


def close_gap(him: np.ndarray, ball: np.ndarray, radius: int) -> np.ndarray:
    """His coverage at output size with the gap between him and the ball filled where the ball is within radius of
    him, so the two read as one shape."""
    ys, xs = np.nonzero(ball >= 0.5)
    if radius <= 0 or not len(ys):
        return him
    h, w = him.shape
    y0, y1 = max(ys.min() - 3 * radius, 0), min(ys.max() + 3 * radius + 1, h)
    x0, x1 = max(xs.min() - 3 * radius, 0), min(xs.max() + 3 * radius + 1, w)
    near = dilate(ball[y0:y1, x0:x1] >= 0.5, radius)
    if not (him[y0:y1, x0:x1][near] >= 0.5).any():
        return him
    both = np.maximum(him[y0:y1, x0:x1], ball[y0:y1, x0:x1])
    closed = 1 - dilate(1 - dilate(both, radius), radius)
    out = him.copy()
    out[y0:y1, x0:x1] = np.where(near, np.maximum(him[y0:y1, x0:x1], closed), him[y0:y1, x0:x1])
    return out


def paint(region: np.ndarray, color: tuple[int, int, int], alpha: np.ndarray) -> None:
    region += (np.asarray(color, dtype=np.float32) - region) * alpha[..., None]


def others_of(kinds: list[str], masks: np.ndarray, mystery: int) -> np.ndarray:
    """The union of every other player's mask in one frame's [object, h, w] masks."""
    players = [i for i, kind in enumerate(kinds) if kind == "player" and i != mystery]
    return masks[players].any(axis=0) if players else np.zeros(masks.shape[1:], bool)


class Shape(NamedTuple):
    """His matte or detail pass as the render draws it. layer(frame, his, others, box, size) is his alpha over the
    window box in clip pixels at size, given his mask and the other players' union. ball_gap is how far apart, in
    pixels at 720 tall, he and the ball may be for the gap between them to close."""

    data: Any
    layer: Callable[[int, np.ndarray, np.ndarray, tuple[int, int, int, int], tuple[int, int]], np.ndarray]
    ball_gap: int


def load_shape(render: dict) -> Shape | None:
    """His matte or detail pass named in the render spec, or None when his SAM mask draws him. matte.py and detail.py
    import this module, so each loads only for a render that uses it."""
    if "matte" in render:
        from matte import Matte  # noqa: PLC0415

        matte = Matte.load(Path(render["matte"]))
        return Shape(matte, lambda frame, his, _, box, size: matte.window_alpha(frame, his, box, size), 0)
    if "detail" in render:
        from detail import BALL_GAP, Detail  # noqa: PLC0415

        detail = Detail.load(Path(render["detail"]))
        return Shape(detail, detail.layer, BALL_GAP)
    return None


def main() -> None:
    masks_path, render_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    kinds, masks = load(masks_path)
    render = json.loads(render_path.read_text())
    with np.errstate(divide="ignore", invalid="ignore"):
        track = draw_ball(kinds, masks, render.get("ball_keys", []))
    size, crops = windows(kinds, masks, render, track)
    if track:
        hide_ball(kinds, masks, track)
    shaped = load_shape(render)
    # A matted or detail render draws at MATTED_HEIGHT, so his edge keeps the source's detail instead of being drawn
    # at the crop's size and scaled up. Otherwise frames render at the crop's size.
    scale = min(MATTED_HEIGHT, shaped.data.source[1]) / size[1] if shaped else 1.0
    canvas = (round(size[0] * scale) // 2 * 2, round(size[1] * scale)) if shaped else size
    # Only "full" crops narrower than the output; it renders at native size and centers in the black.
    native = min(masks.shape[3], size[0])
    inner = (canvas[0] if native == size[0] else round(native * scale), canvas[1])
    left = (canvas[0] - inner[0]) // 2
    out_dir.mkdir(parents=True, exist_ok=True)
    first, end = render.get("frames", (0, len(crops)))
    mystery = render.get("mystery", [])
    ball = render.get("ball", "white")
    if ball not in BALLS:
        sys.exit(f'"ball" must be one of {", ".join(BALLS)}, not {ball!r}')
    order = layers(kinds, mystery, BALLS[ball])
    edge = render.get("ball_edge", 0) * canvas[1] / 720 if mystery else 0
    gap = round(shaped.ball_gap * canvas[1] / 720) if shaped else 0
    for frame, (x, y, cw, ch) in enumerate(crops[first:end], start=first):
        image = np.zeros((canvas[1], canvas[0], 3), dtype=np.float32)
        region = image[:, left : left + inner[0]]
        him = None
        for i, color in order:
            if shaped and i in mystery and not shaped.data.fallback[frame - shaped.data.first]:
                others = others_of(kinds, masks[frame], i)
                alpha = shaped.layer(frame, masks[frame, i], others, (x, y, x + cw, y + ch), inner)
            else:
                alpha = smooth_edges(masks[frame, i, y : y + ch, x : x + cw], inner, EDGE_BLUR * scale)
            if i in mystery:
                him = alpha
            elif kinds[i] == "ball" and him is not None:
                if gap:
                    closed = close_gap(him, alpha, gap)
                    # He is painted already, so only the coverage the closing adds goes on top.
                    paint(region, FOCUS["mystery"], np.clip((closed - him) / np.maximum(1 - him, 1e-6), 0, 1))
                    him = closed
                if edge:
                    paint(region, (0, 0, 0), ball_edge(alpha, him, edge))
            paint(region, color, alpha)
        Image.fromarray(image.round().astype(np.uint8)).save(out_dir / f"{frame - first:05d}.png")


if __name__ == "__main__":
    main()

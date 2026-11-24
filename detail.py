"""Segment the mystery player with his own SAM 2.1 pass on a crop that follows him in the full-resolution source.

Run by clips.py with the SAM venv's python:  python detail.py <source.mp4> <detail.json> <detail.npz>
detail.json is {"clip", "masks", "mystery", "frames", "refines"}: clip.mp4, masks.npz, the mystery player's object
index, the render's [first, end) frames or null for all, and his detail refines [{"frame", "points", "avoid",
"box"?}] in clip pixels. With --choose first it reads masks.npz in place of the source, takes {"mystery", "frames"},
and writes {"shape", "share"}: the pass "auto" picks and his median height as a share of the clip's.
docs/SPEC.md#detail has the detail.npz layout, the ownership and speckle rules and why each step is here.
silhouette.py and qc.py import Detail and its helpers, so everything outside run_model needs only numpy and Pillow.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from matte import crop_origins, crop_size, frame_map, iou, read_crops, resize, seeds, thumbnails, video_info
from silhouette import dilate, erode, load, others_of, shapes

FPS = 30
# "auto" takes the detail pass when his median box is under this share of the clip's height, the matte otherwise.
# Five wide clips of one player measure 18 to 20% and his close clip 40%. A clip whose matte won is 26%.
DETAIL_BELOW = 0.24
# Widths as a share of his median height in clip pixels: his box's padding in a prompt, how far past his 720p mask
# he may reach, and how far inside it he is solid.
BOX_PAD = 0.04
REACH = 0.08
CORE = 0.06
# SAM is sure of a pixel at or beyond these probabilities, and only then does its answer beat another player's mask.
SURE = 0.65
# Pieces of him under SPECK of his area that don't join the rest are removed, and holes under PINHOLE are filled.
SPECK = 0.02
PINHOLE = 0.01
# Where the ball is this many pixels from him at 720 tall, silhouette.py closes the gap between them.
BALL_GAP = 3
# Owned alpha from RAMP below 0.5 to RAMP above it fades from none to full coverage.
RAMP = 0.25
# Alpha this many crop pixels past what the speckle rule kept is his soft edge. Further out it is a faint island.
FRINGE = 4
# Another player's piece of the crop smaller than this many clip pixels gets no avoid point.
AVOID_AREA = 20
# A frame whose owned mask overlaps his 720p mask less than this draws the 720p mask.
MIN_IOU = 0.3


def choose(share: float, below: float = DETAIL_BELOW) -> str:
    """The pass "auto" picks for a mystery player whose median height is this share of the clip's. judge.py passes
    other values of below to find which ones still agree with every judged clip."""
    return "detail" if share < below else "matte"


def height_share(masks: np.ndarray, mystery: int, frames: list[int] | None) -> float:
    """His median box height over the render's [first, end) frames, or every frame without them, as a share of the
    clip's height. 0 when he is never visible there."""
    first, end = frames or (0, len(masks))
    _, box, _ = shapes(masks[first:end, mystery])
    return float(np.nanmedian(box[:, 3] - box[:, 1])) / masks.shape[2] if not np.isnan(box).all() else 0.0


def scale(clip: tuple[int, int], source: tuple[int, int]) -> tuple[float, float]:
    """Source pixels per clip pixel on each axis, given both (width, height). The clip has an even width, so the two
    can differ."""
    return source[0] / clip[0], source[1] / clip[1]


def to_crop(mask: np.ndarray, origin: np.ndarray, crop: tuple[int, int], source: tuple[int, int]) -> np.ndarray:
    """A clip-size mask cut to the crop at origin, at the crop's size in source pixels."""
    rx, ry = scale(mask.shape[::-1], source)
    x0, y0 = origin
    box = (x0 / rx, y0 / ry, (x0 + crop[0]) / rx, (y0 + crop[1]) / ry)
    image = Image.fromarray(mask.astype(np.uint8) * 255).resize(crop, Image.Resampling.BILINEAR, box=box)
    return np.asarray(image) > 127


def points_to_crop(points: list, origin: np.ndarray, ratio: tuple[float, float]) -> list[list[float]]:
    """Points (x, y) in clip pixels as crop pixels, given the crop's origin and source pixels per clip pixel."""
    return [[float(x * ratio[0] - origin[0]), float(y * ratio[1] - origin[1])] for x, y in points]


def prompt_to_crop(prompt: dict, origin: np.ndarray, ratio: tuple[float, float], crop: tuple[int, int]) -> dict:
    """A prompt from prompts() in crop pixels, without the avoid points that fall outside the crop."""
    box = prompt["box"]
    corners = points_to_crop([box[:2], box[2:]], origin, ratio) if box else None
    avoid = points_to_crop(prompt["avoid"], origin, ratio)
    return {
        "box": [*corners[0], *corners[1]] if corners else None,
        "points": points_to_crop(prompt["points"], origin, ratio),
        "avoid": [[x, y] for x, y in avoid if 0 <= x < crop[0] and 0 <= y < crop[1]],
    }


def components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Each 8-connected piece of the mask labeled by one of its pixels' flat index plus one, 0 off the mask, and each
    label's pixel count."""
    h, w = mask.shape
    off = h * w + 1
    label = np.where(mask, np.arange(1, off).reshape(h, w), off)
    lookup = np.r_[0, label.ravel(), off]
    while True:
        padded = np.pad(label, 1, constant_values=off)
        low = label
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                low = np.minimum(low, padded[dy : dy + h, dx : dx + w])
        low = np.where(mask, low, off)
        # Each label names a pixel whose own label may be lower still, so following it joins a piece in few steps.
        lookup[1:-1] = low.ravel()
        for _ in range(4):
            low = lookup[low]
            lookup[1:-1] = low.ravel()
        if np.array_equal(low, label):
            break
        label = low
    label = np.where(mask, label, 0)
    found, counts = np.unique(label[mask], return_counts=True)
    return label, dict(zip(found.tolist(), counts.tolist(), strict=True))


def despeckle(mask: np.ndarray) -> np.ndarray:
    """The mask without pieces under SPECK of its area that don't join its largest piece, and with its holes under
    PINHOLE of its area filled."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return mask.copy()
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    region = mask[y0:y1, x0:x1]
    labels, sizes = components(region)
    largest = max(sizes, key=lambda k: sizes[k])
    small = [k for k, n in sizes.items() if n < SPECK * len(ys) and k != largest]
    kept = region & ~np.isin(labels, small)
    # Padding joins every gap that reaches the edge into one piece of background, so only enclosed holes remain.
    gaps, sizes = components(np.pad(~kept, 1, constant_values=True))
    outside = gaps[0, 0]
    holes = [k for k, n in sizes.items() if n < PINHOLE * len(ys) and k != outside]
    out = mask.copy()
    out[y0:y1, x0:x1] = kept | np.isin(gaps, holes)[1:-1, 1:-1]
    return out


class Owned(NamedTuple):
    """His alpha on one frame's crop after the ownership and speckle rules, and the crop pixels each rule changed."""

    alpha: np.ndarray
    ceded: np.ndarray
    speckle: np.ndarray


def own(prob: np.ndarray, reach: np.ndarray, core: np.ndarray, claimed: np.ndarray) -> Owned:
    """His 0..1 alpha from SAM's probability, all at crop size, per docs/SPEC.md#detail.

    reach is his 720p mask grown by REACH, past which he is absent. core is that mask shrunk by CORE and off every
    other player's, inside which he is solid. claimed is the other players' 720p masks shrunk by a clip pixel and off
    his, which take a pixel from him only where SAM is unsure.
    """
    unsure = (prob > 1 - SURE) & (prob < SURE)
    ceded = reach & unsure & claimed & (prob >= 0.5)
    alpha = np.where(reach & ~(unsure & claimed), prob, 0).astype(np.float32)
    alpha = np.maximum(alpha, core)
    solid = alpha >= 0.5
    kept = despeckle(solid)
    # coverage() draws alpha from 0.5 - RAMP up, so an island fainter than solid would still show.
    faint = (alpha > 0.5 - RAMP) & ~dilate(kept, FRINGE)
    alpha[(solid & ~kept) | faint] = 0
    alpha[kept & ~solid] = 1
    return Owned(alpha, ceded, (solid ^ kept) | faint)


def coverage(alpha: np.ndarray) -> np.ndarray:
    """Owned alpha as draw coverage: a narrow ramp around 0.5, so SAM's soft edge comes out crisp but anti-aliased."""
    return np.clip((alpha - 0.5) / (2 * RAMP) + 0.5, 0, 1)


def deepest(mask: np.ndarray) -> tuple[float, float] | None:
    """The (x, y) of a pixel as far inside the mask as any, nearest the middle of those, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    # No disk wider than the mask's box fits inside it, and erosion costs the radius squared.
    low, high = 0, (min(np.ptp(ys), np.ptp(xs)) + 1) // 2
    while low < high:
        mid = (low + high + 1) // 2
        low, high = (mid, high) if erode(mask, mid).any() else (low, mid - 1)
    ys, xs = np.nonzero(erode(mask, low))
    k = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
    return float(xs[k]), float(ys[k])


def prompts(kinds: list[str], masks: np.ndarray, mystery: int, height: float, refine: dict | None) -> dict:
    """One frame's prompt in clip pixels from its [object, h, w] masks: his box padded by BOX_PAD, a keep point deep
    in each third of him from top to bottom, and an avoid point deep in each other player away from him. A refine
    adds its points and replaces the box with its own when it has one."""
    his = masks[mystery]
    h, w = his.shape
    _, box, _ = shapes(his[None])
    pad = BOX_PAD * height
    x0, y0, x1, y1 = box[0]
    keep, avoid = [], []
    if his.any():
        for top, bottom in zip(np.linspace(y0, y1, 4)[:-1], np.linspace(y0, y1, 4)[1:], strict=True):
            band = his.copy()
            band[: int(top)], band[int(bottom) :] = False, False
            if point := deepest(band):
                keep.append(point)
        near = dilate(his, max(1, round(pad)))
        for i, kind in enumerate(kinds):
            theirs = masks[i] & ~near
            if kind == "player" and i != mystery and theirs.sum() >= AVOID_AREA and (point := deepest(theirs)):
                avoid.append(point)
    prompt = {
        "box": [max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h)] if his.any() else None,
        "points": keep,
        "avoid": avoid,
    }
    if refine:
        prompt["points"] += [tuple(p) for p in refine.get("points", [])]
        prompt["avoid"] += [tuple(p) for p in refine.get("avoid", [])]
        prompt["box"] = refine.get("box", prompt["box"])
    return prompt


class Detail(NamedTuple):
    """One clip's detail.npz: SAM's probability that each crop pixel is him per frame, and where that crop sits. It
    holds only what the model and his 720p masks give, so the ownership, speckle and fallback rules apply at render
    time and tuning them never re-runs the model."""

    prob: np.ndarray
    first: int
    origin: np.ndarray
    crop: tuple[int, int]
    source: tuple[int, int]
    source_frames: np.ndarray
    height: float
    seeds: np.ndarray
    refines: np.ndarray

    def frames(self) -> range:
        return range(self.first, self.first + len(self.prob))

    def owned(self, frame: int, his: np.ndarray, others: np.ndarray) -> Owned:
        """His owned alpha on the frame's crop, given his clip-size 720p mask and the other players' union."""
        j = frame - self.first

        def region(mask: np.ndarray) -> np.ndarray:
            return to_crop(mask, self.origin[j], self.crop, self.source)

        def width(share: float) -> int:
            return max(1, round(share * self.height))

        return own(
            self.prob[j].astype(np.float32) / 255,
            region(dilate(his, width(REACH))),
            region(erode(his, width(CORE)) & ~others),
            region(erode(others, 1) & ~his),
        )

    def source_coverage(self, frame: int, owned: Owned) -> np.ndarray:
        """The owned alpha's draw coverage pasted into a source-size frame."""
        (cw, ch), (x0, y0) = self.crop, self.origin[frame - self.first]
        full = np.zeros(self.source[::-1], np.float32)
        full[y0 : y0 + ch, x0 : x0 + cw] = coverage(owned.alpha)
        return full

    def layer(
        self, frame: int, his: np.ndarray, others: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]
    ) -> np.ndarray | None:
        """His coverage over the window box (x0, y0, x1, y1) in clip pixels, sampled from the source to size, or None
        on a frame that falls back to his 720p mask."""
        owned = self.owned(frame, his, others)
        if self.falls_back(frame, owned, his):
            return None
        rx, ry = scale(his.shape[::-1], self.source)
        w, h = self.source
        window = (box[0] * rx, box[1] * ry, min(box[2] * rx, w), min(box[3] * ry, h))
        return np.clip(resize(self.source_coverage(frame, owned), size, window), 0, 1)

    def clip_iou(self, frame: int, owned: Owned, his: np.ndarray) -> float:
        """How well the owned mask, scaled to the clip, overlaps his 720p mask."""
        small = resize(self.source_coverage(frame, owned), his.shape[::-1]) >= 0.5
        return iou(small, his)

    def falls_back(self, frame: int, owned: Owned, his: np.ndarray) -> bool:
        """Whether the frame draws his 720p mask: he is absent from it, or his owned mask overlaps it under MIN_IOU."""
        return not his.any() or self.clip_iou(frame, owned, his) < MIN_IOU

    def fallback(self, kinds: list[str], masks: np.ndarray, mystery: int) -> np.ndarray:
        """Per frame of the pass, whether it falls back to his 720p mask, given every object's [frame, object, h, w]
        masks."""
        fell = []
        for f in self.frames():
            his = masks[f, mystery]
            fell.append(self.falls_back(f, self.owned(f, his, others_of(kinds, masks[f], mystery)), his))
        return np.asarray(fell, bool)

    def save(self, path: Path) -> None:
        tmp = path.with_name(f".tmp-{path.name}")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **self._asdict())
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Detail:
        data = np.load(path)
        width, height = (int(v) for v in data["crop"])
        source_w, source_h = (int(v) for v in data["source"])
        return cls(
            prob=data["prob"],
            first=int(data["first"]),
            origin=data["origin"],
            crop=(width, height),
            source=(source_w, source_h),
            source_frames=data["source_frames"],
            height=float(data["height"]),
            seeds=data["seeds"],
            refines=data["refines"],
        )


def run_model(crops: np.ndarray, prompted: dict[int, dict]) -> np.ndarray:
    """SAM's probability that each crop pixel is him as uint8, from the prompts {crop index: {"box", "points",
    "avoid"}} in crop pixels, tracked forward from the first prompted crop and back from it."""
    import torch  # noqa: PLC0415 torch and SAM load only for the model run, so tests and renders skip them
    from sam2.sam2_video_predictor import SAM2VideoPredictor  # noqa: PLC0415

    from track import MODEL  # noqa: PLC0415

    device = os.environ.get("SAM_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    prob = np.zeros(crops.shape[:3], np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        for j, image in enumerate(crops):
            Image.fromarray(image).save(Path(tmp) / f"{j:05d}.jpg", quality=95)
        predictor = SAM2VideoPredictor.from_pretrained(MODEL, device=device)
        # The predictor stores its memory as bfloat16, so the whole run autocasts to match it, as in track.py. An
        # offloaded state copies each mask to the CPU without waiting, and the masks propagation yields then read as
        # empty. One player's crop is small enough to keep on the device.
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            state = predictor.init_state(tmp, offload_video_to_cpu=True, offload_state_to_cpu=False)
            for j, prompt in sorted(prompted.items()):
                clicks = prompt["points"] + prompt["avoid"]
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=j,
                    obj_id=0,
                    points=np.asarray(clicks, np.float32) if clicks else None,
                    labels=np.asarray([1] * len(prompt["points"]) + [0] * len(prompt["avoid"]), np.int32)
                    if clicks
                    else None,
                    box=prompt["box"],
                )
            for reverse in (False, True):
                for j, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                    prob[j] = np.round(torch.sigmoid(logits[0, 0].float()).cpu().numpy() * 255)
    return prob


def choose_main(masks_path: Path, spec: dict, out: Path) -> None:
    _, masks = load(masks_path)
    share = height_share(masks, spec["mystery"], spec["frames"])
    out.write_text(json.dumps({"shape": choose(share), "share": share}))


def make(source: Path, spec: dict, out: Path) -> None:
    """Write the detail pass's detail.npz. clips.py keys the file on the code this reaches, so keep render rules out
    of it."""
    kinds, masks = load(Path(spec["masks"]))
    mystery = spec["mystery"]
    first, end = spec["frames"] or (0, len(masks))
    width, height, rate = video_info(source)
    size = (width, height)
    ratio = scale(masks.shape[:1:-1], size)
    source_frames = frame_map(thumbnails(Path(spec["clip"])), thumbnails(source), float(rate / FPS))[first:end]
    _, box, _ = shapes(masks[first:end, mystery])
    if np.isnan(box).all():
        print(f"  no detail: the mystery player is never visible in frames {first}-{end - 1}; drawing SAM's silhouette")
        Detail(
            prob=np.zeros((end - first, 16, 16), np.uint8),
            first=first,
            origin=np.zeros((end - first, 2), int),
            crop=(16, 16),
            source=size,
            source_frames=source_frames,
            height=0.0,
            seeds=np.zeros(0, int),
            refines=np.zeros(0, int),
        ).save(out)
        return
    tall = float(np.nanmedian(box[:, 3] - box[:, 1]))
    source_box = box * np.asarray([*ratio, *ratio])
    crop = crop_size(source_box, size)
    origin = crop_origins(source_box, crop, size)
    crops = read_crops(source, source_frames, origin, crop, crop)
    refines = {r["frame"]: r for r in spec["refines"]}
    for f in refines:
        if not first <= f < end:
            sys.exit(f"detail refine frame {f} is outside the render's frames {first}-{end - 1}")
    picked = seeds(kinds, masks, mystery, first, end)
    prompted = {}
    for f in sorted({*picked, *refines}):
        prompt = prompts(kinds, masks[f], mystery, tall, refines.get(f))
        if prompt["box"] is None and not prompt["points"]:
            sys.exit(f"frame {f} has no box or keep point for the mystery player; give its refine a box")
        prompted[f - first] = prompt_to_crop(prompt, origin[f - first], ratio, crop)
    print(
        f"detail of frames {first}-{end - 1} from {width}x{height} at {float(rate):.2f} fps: crop {crop[0]}x{crop[1]}, "
        f"his height {tall:.0f}px, seeds {picked}, refines {sorted(refines)}",
        file=sys.stderr,
        flush=True,
    )
    started = time.monotonic()
    prob = run_model(crops, prompted)
    seconds = time.monotonic() - started
    Detail(
        prob=prob,
        first=first,
        origin=origin,
        crop=crop,
        source=size,
        source_frames=source_frames,
        height=tall,
        seeds=np.asarray(picked, int),
        refines=np.asarray(sorted(refines), int),
    ).save(out)
    print(f"  detail of {len(prob)} frames in {seconds:.0f}s at crop {crop[0]}x{crop[1]}", flush=True)


def main() -> None:
    if sys.argv[1] == "--choose":
        choose_main(Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4]))
        return
    make(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]))


if __name__ == "__main__":
    main()

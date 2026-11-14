"""Matte the mystery player with MatAnyone 2 on a crop that follows his SAM box in the full-resolution source.

Run by clips.py with the matte venv:  <venv>/bin/python matte.py <source.mp4> <matte.json> <matte.npz>
matte.json is {"clip", "masks", "mystery", "frames", "weights"}: clip.mp4, masks.npz, the mystery player's object
index, the render's [first, end) frames or null for all, and the MatAnyone 2 weights. docs/SPEC.md#matting has the
matte.npz layout and why each step is here. silhouette.py and qc.py import Matte and its helpers, so everything
outside run_model needs only numpy and Pillow.
"""

import json
import subprocess
import sys
import time
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageFilter

from silhouette import dilate, erode, load, shapes, smooth

FPS = 30
# Raw mattes ghost on motion blur, so alpha is cut at LEVELS[0] and full by LEVELS[0] + LEVELS[1].
LEVELS = (0.12, 0.5)
# The matte is limited to his SAM mask grown by GATE of his median height, and solid inside it shrunk by as much.
GATE = 0.04
MIN_GATE = 3
# A frame whose refined matte overlaps his SAM mask less than this falls back to SAM.
MIN_IOU = 0.5
# The crop is this many times the 95th percentile of his box's longer side, and the model sees it MODEL_SIDE tall.
CROP_PAD = 1.5
MODEL_SIDE = 960
CROP_SMOOTH_FRAMES = 9
# A frame this much brighter than the median of FLASH_NEIGHBORS frames either side is a camera flash.
FLASH = 1.15
FLASH_NEIGHBORS = 3
# A seed's mask grown by SEED_REACH clip pixels touches no other player, and its area is within SEED_AREA of his
# median. One seed per SEED_EVERY frames keeps a lost matte from running far.
SEED_REACH = 12
SEED_AREA = 0.2
SEED_EVERY = 60
# The seed mask is closed by this many model pixels, so SAM's gaps between his fingers don't seed holes.
SEED_CLOSE = 5
# Steps the model repeats on a seed frame before it moves on, so its memory starts from a settled alpha.
WARMUP = 10
MAP_SIZE = (320, 180)
MAP_REACH = 3
EDGE_SOFTEN = 2


def levels(alpha: np.ndarray) -> np.ndarray:
    """The raw uint8 alpha as 0..1 after the levels curve."""
    return np.clip((alpha.astype(np.float32) / 255 - LEVELS[0]) / LEVELS[1], 0, 1)


def resize(
    image: np.ndarray, size: tuple[int, int], box: tuple[float, float, float, float] | None = None
) -> np.ndarray:
    """A float image resized to (width, height), from the box (x0, y0, x1, y1) of it when given. Box-filtered when
    it shrinks, bicubic when it grows."""
    region = box or (0.0, 0.0, float(image.shape[1]), float(image.shape[0]))
    shrinks = region[2] - region[0] > size[0]
    resample = Image.Resampling.BOX if shrinks else Image.Resampling.BICUBIC
    return np.asarray(Image.fromarray(image.astype(np.float32), "F").resize(size, resample, box=region))


def soften(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A bool mask scaled to (width, height) as 0..1, with its edge softened by EDGE_SOFTEN pixels."""
    image = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image.filter(ImageFilter.GaussianBlur(EDGE_SOFTEN)), dtype=np.float32) / 255


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


class Matte(NamedTuple):
    """One clip's matte.npz: the raw alpha on a crop per matted frame, where that crop sits in the source, and his
    median height in clip pixels. It holds only what the model and his SAM masks give, so the rules that refine it
    and pick its fallback frames apply at render time and tuning them never re-runs the model."""

    alpha: np.ndarray
    first: int
    origin: np.ndarray
    crop: tuple[int, int]
    source: tuple[int, int]
    height: float
    source_frames: np.ndarray
    seeds: np.ndarray
    flashes: np.ndarray

    def frames(self) -> range:
        return range(self.first, self.first + len(self.alpha))

    @property
    def gate(self) -> int:
        """How far, in clip pixels, the matte may reach past his SAM mask and is solid inside it."""
        return max(MIN_GATE, round(GATE * self.height))

    def source_alpha(self, frame: int, sam: np.ndarray) -> np.ndarray:
        """His refined alpha on clip frame frame at source size: the levels curve, limited to his SAM mask sam grown
        by the gate and solid inside it shrunk by the gate."""
        j = frame - self.first
        (cw, ch), (x0, y0) = self.crop, self.origin[j]
        full = np.zeros(self.source[::-1], np.float32)
        full[y0 : y0 + ch, x0 : x0 + cw] = resize(levels(self.alpha[j]), (cw, ch))
        full *= soften(dilate(sam, self.gate), self.source)
        return np.maximum(full, soften(erode(sam, self.gate), self.source))

    def window_alpha(self, full: np.ndarray, sam: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]):
        """The source-size alpha full, from source_alpha, over the window box (x0, y0, x1, y1) in clip pixels,
        sampled to size."""
        # The clip is the source scaled to 720 tall with an even width, so each axis has its own ratio.
        rx, ry = self.source[0] / sam.shape[1], self.source[1] / sam.shape[0]
        w, h = self.source
        window = (box[0] * rx, box[1] * ry, min(box[2] * rx, w), min(box[3] * ry, h))
        return np.clip(resize(full, size, window), 0, 1)

    def layer(
        self, frame: int, his: np.ndarray, _others: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]
    ) -> np.ndarray | None:
        """His refined alpha over the window box in clip pixels at size, or None on a frame that falls back to his
        SAM mask."""
        full = self.source_alpha(frame, his)
        return None if disagrees(full, his) else self.window_alpha(full, his, box, size)

    def fallback(self, sam: np.ndarray) -> np.ndarray:
        """Per matted frame, whether it falls back to his SAM mask, given his [frame, h, w] SAM masks."""
        return np.asarray([disagrees(self.source_alpha(f, sam[f]), sam[f]) for f in self.frames()], bool)

    def save(self, path: Path) -> None:
        tmp = path.with_name(f".tmp-{path.name}")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **self._asdict())
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Matte:
        data = np.load(path)
        width, height = (int(v) for v in data["crop"])
        source_w, source_h = (int(v) for v in data["source"])
        return cls(
            alpha=data["alpha"],
            first=int(data["first"]),
            origin=data["origin"],
            crop=(width, height),
            source=(source_w, source_h),
            height=float(data["height"]),
            source_frames=data["source_frames"],
            seeds=data["seeds"],
            flashes=data["flashes"],
        )


def disagrees(full: np.ndarray, sam: np.ndarray) -> bool:
    """Whether a source-size refined alpha overlaps his clip-size SAM mask under MIN_IOU, so the frame falls back to
    SAM."""
    small = resize(full, sam.shape[::-1]) > 0.5
    return iou(small, sam) < MIN_IOU


def frame_map(clip: np.ndarray, source: np.ndarray, ratio: float) -> np.ndarray:
    """Per clip frame, the source frame within MAP_REACH of its timestamp whose thumbnail matches it best."""
    found = []
    for f, thumb in enumerate(clip):
        near = min(round(f * ratio), len(source) - 1)
        candidates = np.arange(max(near - MAP_REACH, 0), min(near + MAP_REACH + 1, len(source)))
        errors = ((source[candidates] - thumb) ** 2).mean(axis=(1, 2))
        found.append(candidates[np.argmin(errors)])
    return np.asarray(found)


def crop_size(box: np.ndarray, source: tuple[int, int]) -> tuple[int, int]:
    """The crop's (width, height) in source pixels from his per-frame box there: a square CROP_PAD times the 95th
    percentile of the box's longer side, rounded up to 16, or the whole frame when that doesn't fit."""
    longer = np.fmax(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    side = int(np.ceil(CROP_PAD * np.nanpercentile(longer, 95) / 16) * 16)
    return (side, side) if side < min(source) else source


def crop_origins(box: np.ndarray, crop: tuple[int, int], source: tuple[int, int]) -> np.ndarray:
    """Per frame, the (x, y) corner of a crop that follows the smoothed center of his box, kept inside the source.
    Frames without a box take the center interpolated between their neighbours."""
    centers = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], axis=1)
    known = np.flatnonzero(~np.isnan(centers[:, 0]))
    origins = []
    for axis in (0, 1):
        center = smooth(np.interp(np.arange(len(box)), known, centers[known, axis]), CROP_SMOOTH_FRAMES)
        origins.append(np.clip(np.round(center - crop[axis] / 2), 0, source[axis] - crop[axis]))
    return np.stack(origins, axis=1).astype(int)


def model_size(crop: tuple[int, int]) -> tuple[int, int]:
    """The (width, height) the model sees the crop at: MODEL_SIDE tall and the crop's aspect, both multiples of 16."""
    return round(crop[0] * MODEL_SIDE / crop[1] / 16) * 16, MODEL_SIDE


def flashes(means: np.ndarray) -> dict[int, float]:
    """Each camera flash frame by index, with the brightness to scale it to: its neighbours' median."""
    found = {}
    for j, mean in enumerate(means):
        near = np.r_[means[max(j - FLASH_NEIGHBORS, 0) : j], means[j + 1 : j + 1 + FLASH_NEIGHBORS]]
        if len(near) and mean > FLASH * np.median(near):
            found[j] = float(np.median(near))
    return found


def seeds(kinds: list[str], masks: np.ndarray, mystery: int, first: int, end: int) -> list[int]:
    """The clip frames that seed the model, one per stretch of up to SEED_EVERY frames.

    A stretch's seed is its frame whose mask, grown by SEED_REACH, touches no other player and whose area is
    nearest his median, within SEED_AREA. A stretch with no such frame has no seed and is matted from its neighbours'.
    With none in the whole clip, the frame touching other players least seeds it all.
    """
    area = masks[first:end, mystery].sum(axis=(1, 2)).astype(float)
    median = np.median(area[area > 0])
    others = [i for i, kind in enumerate(kinds) if kind == "player" and i != mystery]
    with np.errstate(divide="ignore"):
        near = np.fmin(area / median, median / area)
    touch = np.full(len(area), np.inf)
    for j in np.flatnonzero(area > 0):
        reach = dilate(masks[first + j, mystery], SEED_REACH)
        touch[j] = sum(int((reach & masks[first + j, i]).sum()) for i in others)
    clean = (touch == 0) & (near >= 1 - SEED_AREA)
    stretches = np.array_split(np.arange(len(area)), max(1, round(len(area) / SEED_EVERY)))
    picked = [int(ok[np.argmax(near[ok])]) for stretch in stretches if len(ok := stretch[clean[stretch]])]
    if not picked:
        picked = [int(np.lexsort((-near, touch))[0])]
    return [first + j for j in picked]


def video_info(path: Path) -> tuple[int, int, Fraction]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return stream["width"], stream["height"], Fraction(stream["r_frame_rate"])


def decode(path: Path, size: tuple[int, int], pix_fmt: str, scale: bool):
    """Yield each frame of the video as an array, at size when scale, else at the size the video already has."""
    channels = {"gray": 1, "rgb24": 3}[pix_fmt]
    vf = ["-vf", f"scale={size[0]}:{size[1]}"] if scale else []
    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), *vf, "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        stdout=subprocess.PIPE,
    )
    frame_bytes = size[0] * size[1] * channels
    while len(chunk := decoder.stdout.read(frame_bytes)) == frame_bytes:  # ty: ignore[missing-union-attribute]
        yield np.frombuffer(chunk, np.uint8).reshape(size[1], size[0], channels).squeeze()
    decoder.wait()


def thumbnails(path: Path) -> np.ndarray:
    return np.stack(list(decode(path, MAP_SIZE, "gray", scale=True))).astype(np.float32)


def read_crops(
    source: Path, wanted: np.ndarray, origin: np.ndarray, crop: tuple[int, int], size: tuple[int, int]
) -> np.ndarray:
    """The crop of each wanted source frame, at the model's size, as uint8 RGB."""
    width, height, _ = video_info(source)
    crops = np.zeros((len(wanted), size[1], size[0], 3), np.uint8)
    resample = Image.Resampling.BOX if crop[0] > size[0] else Image.Resampling.BICUBIC
    decoded = 0
    for s, image in enumerate(decode(source, (width, height), "rgb24", scale=False)):
        decoded = s + 1
        for j in np.flatnonzero(wanted == s):
            x0, y0 = origin[j]
            crops[j] = np.asarray(Image.fromarray(image[y0 : y0 + crop[1], x0 : x0 + crop[0]]).resize(size, resample))
    if wanted.max() >= decoded:
        sys.exit(f"source has {decoded} frames, the frame map wants frame {wanted.max()}")
    return crops


def seed_mask(
    sam: np.ndarray, origin: np.ndarray, crop: tuple[int, int], source: tuple[int, int], size: tuple[int, int]
) -> np.ndarray:
    """His SAM mask on a seed frame, cut to the crop at the model's size and closed by SEED_CLOSE."""
    full = Image.fromarray(sam.astype(np.uint8) * 255).resize(source, Image.Resampling.BILINEAR)
    x0, y0 = origin
    region = full.crop((x0, y0, x0 + crop[0], y0 + crop[1])).resize(size, Image.Resampling.BILINEAR)
    mask = np.asarray(region) > 127
    return erode(dilate(mask, SEED_CLOSE), SEED_CLOSE)


def run_model(crops: np.ndarray, seed_masks: dict[int, np.ndarray], weights: Path) -> np.ndarray:
    """The model's alpha on every crop as uint8. Each seed mattes the frames nearer to it than to any other seed,
    forward to the next cut and backward to the last."""
    import torch  # noqa: PLC0415 torch and MatAnyone 2 exist only in the matte venv
    from matanyone2.inference.inference_core import InferenceCore  # noqa: PLC0415
    from matanyone2.utils.get_default_model import get_matanyone2_model  # noqa: PLC0415

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = get_matanyone2_model(str(weights), device)
    alpha = np.zeros(crops.shape[:3], np.uint8)

    def image(j: int):
        return torch.from_numpy(crops[j]).permute(2, 0, 1).float().div(255).to(device)

    @torch.inference_mode()
    def matte(order: list[int], mask: np.ndarray) -> None:
        core = InferenceCore(model, cfg=model.cfg)
        start = image(order[0])
        core.step(start, torch.from_numpy(mask.astype(np.float32) * 255).to(device), objects=[1])
        for _ in range(WARMUP):
            prob = core.step(start, first_frame_pred=True)
        for i, j in enumerate(order):
            if i:
                prob = core.step(image(j))
            alpha[j] = np.round(np.clip(core.output_prob_to_mask(prob).cpu().numpy(), 0, 1) * 255)

    at = sorted(seed_masks)
    cuts = [0, *((a + b + 1) // 2 for a, b in pairwise(at)), len(crops)]
    for n, (j, (low, high)) in enumerate(zip(at, pairwise(cuts), strict=True), start=1):
        print(f"seed {n} of {len(at)} at crop {j}: crops {low}-{high - 1}", file=sys.stderr, flush=True)
        matte(list(range(j, high)), seed_masks[j])
        matte(list(range(j, low - 1, -1)), seed_masks[j])
    return alpha


def main() -> None:
    source, spec_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    spec = json.loads(spec_path.read_text())
    kinds, masks = load(Path(spec["masks"]))
    mystery = spec["mystery"]
    first, end = spec["frames"] or (0, len(masks))
    width, height, rate = video_info(source)
    size = (width, height)
    source_frames = frame_map(thumbnails(Path(spec["clip"])), thumbnails(source), float(rate / FPS))[first:end]
    _, box, _ = shapes(masks[first:end, mystery])
    if np.isnan(box).all():
        print(f"  no matte: the mystery player is never visible in frames {first}-{end - 1}; drawing SAM's silhouette")
        Matte(
            alpha=np.zeros((end - first, 16, 16), np.uint8),
            first=first,
            origin=np.zeros((end - first, 2), int),
            crop=size,
            source=size,
            height=0.0,
            source_frames=source_frames,
            seeds=np.zeros(0, int),
            flashes=np.zeros(0, int),
        ).save(out)
        return
    source_box = box * np.asarray([width / masks.shape[3], height / masks.shape[2]] * 2)
    crop = crop_size(source_box, size)
    origin = crop_origins(source_box, crop, size)
    shape = model_size(crop)
    crops = read_crops(source, source_frames, origin, crop, shape)
    flash = flashes(crops.reshape(len(crops), -1).mean(axis=1))
    for j, target in flash.items():
        crops[j] = np.clip(crops[j] * (target / crops[j].mean()), 0, 255).astype(np.uint8)
    picked = seeds(kinds, masks, mystery, first, end)
    print(
        f"matting frames {first}-{end - 1} from {width}x{height} at {float(rate):.2f} fps: crop {crop[0]}x{crop[1]} "
        f"at {shape[0]}x{shape[1]}, seeds {picked}, flashes {[first + j for j in flash]}",
        file=sys.stderr,
        flush=True,
    )
    started = time.monotonic()
    masks_at = {f - first: seed_mask(masks[f, mystery], origin[f - first], crop, size, shape) for f in picked}
    alpha = run_model(crops, masks_at, Path(spec["weights"]))
    seconds = time.monotonic() - started
    matte = Matte(
        alpha=alpha,
        first=first,
        origin=origin,
        crop=crop,
        source=size,
        height=float(np.nanmedian(box[:, 3] - box[:, 1])),
        source_frames=source_frames,
        seeds=np.asarray(picked),
        flashes=np.asarray([first + j for j in flash], dtype=int),
    )
    matte.save(out)
    print(f"  matted {len(alpha)} frames in {seconds:.0f}s from seeds {picked}", flush=True)


if __name__ == "__main__":
    main()

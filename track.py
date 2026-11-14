"""Track prompted players, the ball and the hoop through a clip with SAM 2.1 and save their masks.

Run by clips.py with the project venv:  .venv/bin/python track.py <clip.mp4> <track.json> <masks.npz>
With --preview first, it only adds the prompts and writes one overlay PNG per prompted frame into the
directory given in place of masks.npz, which takes seconds instead of a full track.
track.json lists objects as {"kind", "prompts", "cache"}. Each prompt is {"frame", "points", "avoid"} and may
add "box" [x0, y0, x1, y1]. The first places the object and later ones pull its mask back after it drifts.
"cache" is the object's own mask file, named by clips.py from a hash of its prompts, so an unchanged object is
never tracked again.
Tracking is the slow step (about 0.3 s per frame for the image encoder plus 0.3 s per object per frame on an
M4), so its masks are cached for silhouette.py.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sam2.sam2_video_predictor import SAM2VideoPredictor

from silhouette import TINTS, load

MODEL = "facebook/sam2.1-hiera-small"
# A refine with more than this share of its keep points, or of the mask it makes, on another object's mask is
# probably filed under the wrong object.
MISFILED = 0.5


def save(path: Path, masks: np.ndarray, **extra) -> None:
    """Write packed masks through a temp file, so a killed run never leaves a half-written one."""
    tmp = path.with_name(f".tmp-{path.name}")
    np.savez_compressed(tmp, bits=np.packbits(masks, axis=-1), shape=np.asarray(masks.shape), **extra)
    tmp.replace(path)


def first_frame(obj: dict) -> int:
    return min(p["frame"] for p in obj["prompts"])


@contextmanager
def session(clip: Path, offload_state: bool):
    """Decode the clip and load SAM over it, yielding (predictor, state, frames dir) inside SAM's precision mode.

    An offloaded state saves memory on long tracks, but SAM then copies each prompted mask to the CPU without
    waiting, so the mask add_new_points_or_box returns can be read before it lands. Preview keeps it on device.
    """
    device = os.environ.get("SAM_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(tmp) / "%05d.jpg"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(clip), "-q:v", "2", "-start_number", "0", str(frames)], check=True
        )
        predictor = SAM2VideoPredictor.from_pretrained(MODEL, device=device)
        # The predictor always stores its memory as bfloat16, so the whole run must autocast to match it.
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            state = predictor.init_state(tmp, offload_video_to_cpu=True, offload_state_to_cpu=offload_state)
            yield predictor, state, Path(tmp)


def add_prompts(predictor: SAM2VideoPredictor, state: dict, obj_id: int, obj: dict) -> list[tuple[int, np.ndarray]]:
    """Prompt one object and return its mask on each prompted frame, the masks propagation conditions on."""
    shown = []
    for prompt in obj["prompts"]:
        keep, avoid = prompt["points"], prompt["avoid"]
        clicks = keep + avoid
        frame, obj_ids, logits = predictor.add_new_points_or_box(
            state,
            frame_idx=prompt["frame"],
            obj_id=obj_id,
            points=np.asarray(clicks, dtype=np.float32) if clicks else None,
            labels=np.asarray([1] * len(keep) + [0] * len(avoid), dtype=np.int32) if clicks else None,
            box=prompt.get("box"),
        )
        shown.append((frame, (logits[obj_ids.index(obj_id), 0] > 0).cpu().numpy()))
    return shown


def track(predictor: SAM2VideoPredictor, state: dict, objects: list[dict]) -> None:
    """Track each object and write its [frame, h, w] masks to its cache file as soon as its group finishes.

    SAM starts both propagation passes at the earliest prompt of any object in the state, so an object
    prompted later would never be tracked backward from its own prompt. Objects are therefore tracked in
    groups that share an earliest prompt frame. Objects never influence each other in SAM 2.1, so grouping
    only saves running the image encoder once per group instead of once per object.
    """
    groups = [list(group) for _, group in groupby(sorted(objects, key=first_frame), key=first_frame)]
    for n, group in enumerate(groups, start=1):
        kinds = ", ".join(obj["kind"] for obj in group)
        stamp = time.strftime("%X")
        print(
            f"tracking group {n} of {len(groups)} from frame {first_frame(group[0])}: {kinds}, started {stamp}",
            file=sys.stderr,
            flush=True,
        )
        predictor.reset_state(state)
        for obj_id, obj in enumerate(group):
            add_prompts(predictor, state, obj_id, obj)
        masks = np.zeros((len(group), state["num_frames"], state["video_height"], state["video_width"]), bool)
        for reverse in (False, True):
            for frame, obj_ids, logits in predictor.propagate_in_video(state, reverse=reverse):
                for i, obj_id in enumerate(obj_ids):
                    masks[obj_id, frame] = (logits[i, 0] > 0).cpu().numpy()
        for obj, obj_masks in zip(group, masks, strict=True):
            save(Path(obj["cache"]), obj_masks)


def misfiled(objects: list[dict], i: int, prompt: dict, mask: np.ndarray, others: dict[int, np.ndarray]) -> list[str]:
    """Why a refine of object i looks filed under the wrong object: its keep points, or the mask it gave, mostly
    on another object's mask. A ball held or going through the net sits inside a player's or the hoop's mask, so a
    ball refine is checked only against other balls.
    """
    found = []
    for j, other in others.items():
        if j == i or (objects[i]["kind"] == "ball" and objects[j]["kind"] != "ball"):
            continue
        h, w = other.shape
        keep = [other[min(int(y), h - 1), min(int(x), w - 1)] for x, y in prompt["points"]]
        shares = {"keep points": np.mean(keep) if keep else 0, "mask": (mask & other).sum() / max(mask.sum(), 1)}
        for what, share in shares.items():
            if share > MISFILED:
                found.append(f"{share:.0%} of its {what} on object {j} ({objects[j]['kind']})")
    return found


def label_above(draw: ImageDraw.ImageDraw, mask: np.ndarray, text: str, font, color, lift: int, stroke: int) -> bool:
    """Write text centered just above the mask, or return False when the mask is empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return False
    at = (int(xs.mean()), int(ys.min()) - lift)
    draw.text(at, text, font=font, fill=color, stroke_width=stroke, stroke_fill="black", anchor="mm")
    return True


def draw_prompt(draw: ImageDraw.ImageDraw, prompt: dict, name: str, color, font) -> None:
    """Draw a prompt's box, its keep points as green dots and avoid points as red ones, and its name."""
    if "box" in prompt:
        draw.rectangle(prompt["box"], outline=color, width=2)
    for fill, points in (("lime", prompt["points"]), ("red", prompt["avoid"])):
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=fill, outline=color, width=2)
    at = prompt["box"][:2] if "box" in prompt else prompt["points"][0]
    draw.text((at[0], at[1] - 4), name, font=font, fill=color, stroke_width=2, stroke_fill="black", anchor="lb")


def preview(predictor: SAM2VideoPredictor, state: dict, objects: list[dict], frames: Path, out_dir: Path) -> None:
    """Write one PNG per prompted frame: the dimmed source frame with each object's mask tinted and labeled.

    Each mask and each prompt is labeled with the object's index in the manifest and its kind, refines with their
    number, all in the object's tint. Keep points are green dots and avoid points red ones, so a bad click shows
    beside its mask. On a refine's frame every other object's mask from its last track, where cached, is drawn
    faint, and a refine that lands mostly on another object is reported as likely misfiled.
    """
    prompted: dict[int, list[tuple[int, int, dict, np.ndarray]]] = {}
    for i, obj in enumerate(objects):
        for k, (prompt, (frame, mask)) in enumerate(
            zip(obj["prompts"], add_prompts(predictor, state, i, obj), strict=True)
        ):
            prompted.setdefault(frame, []).append((i, k, prompt, mask))
    refined = sorted(f for f, entries in prompted.items() if any(k for _, k, _, _ in entries))
    tracked = {
        i: dict(zip(refined, load(Path(obj["cache"]), refined)[1], strict=True))
        for i, obj in enumerate(objects)
        if Path(obj["cache"]).exists()
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=18)
    for frame, entries in sorted(prompted.items()):
        masks = {i: mask for i, _, _, mask in entries}
        faint = {j: m[frame] for j, m in tracked.items() if j not in masks and frame in m}
        image = np.asarray(Image.open(frames / f"{frame:05d}.jpg").convert("RGB"), dtype=np.float32) * 0.5
        for j, mask in faint.items():
            image[mask] = image[mask] * 0.7 + np.asarray(TINTS[j % len(TINTS)]) * 0.3
        for i, mask in masks.items():
            image[mask] = image[mask] * 0.3 + np.asarray(TINTS[i % len(TINTS)]) * 0.7
        canvas = Image.fromarray(image.astype(np.uint8))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"frame {frame}", font=font, fill="white", stroke_width=3, stroke_fill="black")
        for j, mask in faint.items():
            label_above(draw, mask, str(j), small, TINTS[j % len(TINTS)], lift=10, stroke=2)
        for i, k, prompt, mask in entries:
            color = TINTS[i % len(TINTS)]
            name = f"{i} {objects[i]['kind']}" + (f" refine {k}" if k else "")
            if k:
                others = {**faint, **{j: m for j, m in masks.items() if j != i}}
                reasons = misfiled(objects, i, prompt, mask, others)
                for why in reasons:
                    print(f"  frame {frame}: object {name} has {why}; is it filed under the right object?")
                name += " ?" if reasons else ""
            draw_prompt(draw, prompt, name, color, small)
        for i, mask in masks.items():
            # Above the mask rather than on it, so a ball held against a body keeps its own label.
            if not label_above(draw, mask, f"{i} {objects[i]['kind']}", font, TINTS[i % len(TINTS)], lift=14, stroke=3):
                print(f"  frame {frame}: object {i} ({objects[i]['kind']}) has an empty mask")
        canvas.save(out_dir / f"{frame:05d}.png")
        print(f"  {out_dir / f'{frame:05d}.png'}")


def main() -> None:
    preview_only = sys.argv[1] == "--preview"
    if preview_only:
        sys.argv.pop(1)
    clip, spec_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    objects = json.loads(spec_path.read_text())["objects"]
    if preview_only:
        with session(clip, offload_state=False) as (predictor, state, frames):
            preview(predictor, state, objects, frames, out)
        return
    missing = [obj for obj in objects if not Path(obj["cache"]).exists()]
    if missing:
        with session(clip, offload_state=True) as (predictor, state, _):
            track(predictor, state, missing)
    print(f"  tracked {len(missing)} of {len(objects)} objects, the rest were cached")
    _, first = load(Path(objects[0]["cache"]))
    masks = np.empty((first.shape[0], len(objects), *first.shape[1:]), dtype=bool)
    for i, obj in enumerate(objects):
        masks[:, i] = first if i == 0 else load(Path(obj["cache"]))[1]
    kinds = [obj["kind"] for obj in objects]
    for kind, share in zip(kinds, masks.reshape(*masks.shape[:2], -1).any(axis=2).mean(axis=0), strict=True):
        print(f"  {kind}: visible in {share:.0%} of frames")
    save(out, masks, kinds=np.asarray(kinds))


if __name__ == "__main__":
    main()

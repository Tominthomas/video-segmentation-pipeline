"""Track prompted players and the ball with SAM 2.1 and render white-on-black silhouette frames.

Run by clips.py with the project venv:  .venv/bin/python pipeline/segment.py <clip.mp4> <track.json> <out_dir>
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor

MODEL = "facebook/sam2.1-hiera-small"
COLORS = {"player": (255, 255, 255), "ball": (255, 122, 26), "hoop": (150, 150, 150)}
# The crop follows these kinds; the ball is left out because its arc would blow the window up.
FRAMED = {"player", "hoop"}
ASPECT = 16 / 9
PAD = 1.25
SMOOTH_FRAMES = 15


def track(frames_dir: Path, spec: dict) -> tuple[list[str], np.ndarray]:
    """Return each object's kind and a [frame, object, h, w] boolean mask array."""
    device = os.environ.get("SAM_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    predictor = SAM2VideoPredictor.from_pretrained(MODEL, device=device)
    kinds = [obj["kind"] for obj in spec["objects"]]
    # The predictor always stores its memory as bfloat16, so the whole run must autocast to match it.
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for obj_id, obj in enumerate(spec["objects"]):
            points = np.asarray(obj["points"], dtype=np.float32)
            predictor.add_new_points_or_box(state, frame_idx=obj.get("frame", spec["frame"]), obj_id=obj_id,
                                            points=points, labels=np.ones(len(points), dtype=np.int32))
        frame_count = state["num_frames"]
        h, w = state["video_height"], state["video_width"]
        masks = np.zeros((frame_count, len(kinds), h, w), dtype=bool)
        for reverse in (False, True):
            for frame, obj_ids, logits in predictor.propagate_in_video(state, reverse=reverse):
                for i, obj_id in enumerate(obj_ids):
                    masks[frame, obj_id] = (logits[i, 0] > 0).cpu().numpy()
    return kinds, masks


def crop_windows(framed_masks: np.ndarray, frame_w: int, frame_h: int) -> tuple[int, int, np.ndarray]:
    """One fixed-size 16:9 window per frame that follows the framed objects' combined box, smoothed so it glides."""
    boxes = []
    for frame in framed_masks.any(axis=1):
        ys, xs = np.nonzero(frame)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None)
    known = [b for b in boxes if b]
    if not known:
        sys.exit("no player or hoop pixels in any frame; check the track prompts")
    span_w = max(b[2] - b[0] for b in known) * PAD
    span_h = max(b[3] - b[1] for b in known) * PAD
    win_h = int(min(frame_h, max(span_h, span_w / ASPECT)))
    win_w = int(min(frame_w, win_h * ASPECT))
    win_h = int(win_w / ASPECT)

    last = known[0]
    centers = []
    for b in boxes:
        last = b or last
        centers.append(((last[0] + last[2]) / 2, (last[1] + last[3]) / 2))
    centers = np.asarray(centers)
    kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
    padded = np.pad(centers, ((SMOOTH_FRAMES // 2, SMOOTH_FRAMES // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, a], kernel, mode="valid") for a in (0, 1)], axis=1)
    origins = np.stack([
        np.clip(smooth[:, 0] - win_w / 2, 0, frame_w - win_w),
        np.clip(smooth[:, 1] - win_h / 2, 0, frame_h - win_h),
    ], axis=1).astype(int)
    return win_w, win_h, origins


def render(kinds: list[str], masks: np.ndarray, out_dir: Path) -> None:
    frame_count, _, h, w = masks.shape
    framed = masks[:, [i for i, k in enumerate(kinds) if k in FRAMED]]
    win_w, win_h, origins = crop_windows(framed, w, h)
    for frame in range(frame_count):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for i, kind in enumerate(kinds):
            canvas[masks[frame, i]] = COLORS[kind]
        x, y = origins[frame]
        Image.fromarray(canvas[y:y + win_h, x:x + win_w]).save(out_dir / f"{frame:05d}.png")


def main() -> None:
    clip, spec_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    spec = json.loads(spec_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(tmp)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-q:v", "2", "-start_number", "0",
                        str(frames / "%05d.jpg")], check=True)
        kinds, masks = track(frames, spec)
    coverage = masks.reshape(masks.shape[0], masks.shape[1], -1).any(axis=2).mean(axis=0)
    for kind, share in zip(kinds, coverage):
        print(f"  {kind}: visible in {share:.0%} of frames")
    render(kinds, masks, out_dir)


if __name__ == "__main__":
    main()

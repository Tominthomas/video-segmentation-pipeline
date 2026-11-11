"""Track prompted players, the ball and the hoop through a clip with SAM 2.1 and save their masks.

Run by clips.py with the project venv:  .venv/bin/python pipeline/track.py <clip.mp4> <track.json> <masks.npz>
Tracking is the slow step (about 0.3 s per object per frame on an M4), so its masks are cached for silhouette.py.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

MODEL = "facebook/sam2.1-hiera-small"


def track(frames_dir: Path, spec: dict) -> np.ndarray:
    """Return a [frame, object, h, w] boolean mask array, one object per entry in spec["objects"]."""
    device = os.environ.get("SAM_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    predictor = SAM2VideoPredictor.from_pretrained(MODEL, device=device)
    # The predictor always stores its memory as bfloat16, so the whole run must autocast to match it.
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for obj_id, obj in enumerate(spec["objects"]):
            keep, avoid = obj["points"], obj.get("avoid", [])
            predictor.add_new_points_or_box(
                state,
                frame_idx=obj.get("frame", spec["frame"]),
                obj_id=obj_id,
                points=np.asarray(keep + avoid, dtype=np.float32),
                labels=np.asarray([1] * len(keep) + [0] * len(avoid), dtype=np.int32),
            )
        masks = np.zeros(
            (state["num_frames"], len(spec["objects"]), state["video_height"], state["video_width"]), dtype=bool
        )
        for reverse in (False, True):
            for frame, obj_ids, logits in predictor.propagate_in_video(state, reverse=reverse):
                for i, obj_id in enumerate(obj_ids):
                    masks[frame, obj_id] = (logits[i, 0] > 0).cpu().numpy()
    return masks


def main() -> None:
    clip, spec_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    spec = json.loads(spec_path.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(clip), "-q:v", "2", "-start_number", "0", str(Path(tmp) / "%05d.jpg")],
            check=True,
        )
        masks = track(Path(tmp), spec)
    kinds = [obj["kind"] for obj in spec["objects"]]
    for kind, share in zip(kinds, masks.reshape(*masks.shape[:2], -1).any(axis=2).mean(axis=0), strict=False):
        print(f"  {kind}: visible in {share:.0%} of frames")
    np.savez_compressed(out, bits=np.packbits(masks, axis=-1), shape=np.asarray(masks.shape), kinds=np.asarray(kinds))


if __name__ == "__main__":
    main()

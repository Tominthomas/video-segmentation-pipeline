# video-segmentation-pipeline

`clips.py` fetches NBA highlight clips and renders the silhouette clues for [Bagwork](https://github.com/Tominthomas/bagwork), a daily guess-the-player game. [docs/SPEC.md](docs/SPEC.md) explains how each artifact is made.

## Checks

The scripts need Python 3.14, ruff and ty (`brew install python ruff ty`). The checks and tests need only numpy and Pillow:

```sh
ruff check && ruff format --check
ty check --python .venv/bin/python
python3 -m unittest discover tests
python3 clips.py --help
```

CI runs the same four on pull requests and on pushes to `main`.

## Build clips

`clips.py` needs `yt-dlp` and `ffmpeg` (`brew install yt-dlp ffmpeg`). Everything it writes goes to `media/`, which git ignores.

1. Find the moment (skip this for clips that are already cut, like most X posts): `python3 clips.py scout <youtube-url>` saves one frame per second and prints the path of a `sheet.html` labelled by second.
2. Add an entry to a clip manifest (`fixtures/clips.json` holds the public samples; real puzzles keep theirs outside the repo and pass `--manifest`) with the `start` and `end` seconds, or leave both out to use the whole video. Optional `hide` boxes (`[x, y, w, h]` in the 720p frame) black out score bugs in clue clips, and an optional `crop` (`size` plus a `path` of `[t, x, y]` points) zooms the clue onto the play.
3. Build: `python3 clips.py build [--manifest FILE] [id ...]` writes `source.mp4`, `clip.mp4` (the full reveal), `thermal.mp4` (the clue) and `contact.png` per clip, then checks size, duration, missing audio on clues and stripped metadata. Reruns skip anything whose inputs did not change.

Replay angles make better clues than the wide broadcast shot: the player is bigger and there is no score bug.

### Silhouettes

A clip with a `track` entry also gets `silhouette.mp4`: players in white, the ball in orange and the hoop in gray on black, cropped to follow the play. Each tracked object lists one or more `[x, y]` click points in the 720p frame on its prompt `frame`, and SAM 2.1 follows it through the clip. This step runs in a project venv:

```sh
python3 -m venv .venv
.venv/bin/pip install torch torchvision pillow numpy huggingface_hub
SAM2_BUILD_CUDA=0 .venv/bin/pip install "git+https://github.com/facebookresearch/sam2"
```

`track.py` runs SAM 2.1 and caches the masks in `masks.npz`, about 0.3 seconds per object per frame on an M4 (`SAM_DEVICE=cpu` overrides the Apple GPU). `silhouette.py` draws them in seconds, so changing the look never re-runs tracking. Objects can also list `avoid` points SAM should leave out, such as a backboard behind the rim.

The `track` entry also picks the crop: `"follow": "holder"` (the player with the ball; at the release it eases out to keep the shooter and the hoop in frame), `"ball"`, or `"action"` (every player and the hoop, the default), with `"zoom"` (default 2) for the first two.

## The render contract

Bagwork takes a render through `bagwork clips add <manifest> --renders <folder>` in the bagwork repo, which reads each manifest entry's `id`, `player`, `title`, `url`, `start` and `end` and uploads `<folder>/<id>.mp4`. `tools/bagwork/library.ts` there refuses a render unless all of these hold:

- It is under 2,000,000 bytes, because the Worker streams every clip whole to a phone.
- ffprobe reads exactly one stream, and it is video. No audio, subtitle or data track.
- It has no chapters.
- Every tag on the container and the stream is one ffmpeg writes by default: `major_brand`, `minor_version`, `compatible_brands`, `encoder` (`Lavf` or `Lavc` with an optional ` libx264`), `handler_name` (`VideoHandler`), `language` and `vendor_id`. Any other tag or value is free text that could name the player.

The `url` must be a YouTube or X link, because the game shows it as the reveal. `silhouette.mp4` from `clips.py build` meets the stream, chapter and tag rules as it is: H.264 from libx264, no audio and stripped metadata. `build` does not check the byte limit, and a long clip can pass it, so check `ls -l` before you approve one. Copy each approved one to `$BAGWORK_CONTENT/approved/<id>.mp4`. The check in the bagwork repo is the source of truth; if this list and that code disagree, the code wins.

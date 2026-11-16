# video-segmentation-pipeline

`clips.py` fetches NBA highlight clips and renders the silhouette clues for [Bagwork](https://github.com/Tominthomas/bagwork), a daily guess-the-player game. [docs/SPEC.md](docs/SPEC.md) explains how each artifact is made.

## Checks

The scripts need Python 3.14, ruff and ty (`brew install python ruff ty`). The checks use the shared SAM venv, which [Silhouettes](#silhouettes) installs once per machine:

```sh
SAM_PY=~/.local/share/bagwork/sam/venv/bin/python
ruff check && ruff format --check
ty check --python $SAM_PY
$SAM_PY -m unittest discover tests
python3 clips.py --help
```

CI runs the same four on pull requests and on pushes to `main`.

## Build clips

`clips.py` needs `yt-dlp` and `ffmpeg` (`brew install yt-dlp ffmpeg`). Everything it writes goes to `media/`, which git ignores.

1. Find the moment (skip this for clips that are already cut, like most X posts): `python3 clips.py scout <youtube-url>` saves one frame per second and prints the path of a `sheet.html` labelled by second.
2. Add an entry to a clip manifest (`fixtures/clips.json` holds the public samples; real puzzles keep theirs outside the repo and pass `--manifest`). `python3 clips.py add --manifest FILE <url> --player NAME` scouts a post and appends a stub. `add` has no default manifest, so an entry that names a player never lands in the repo by accident. Give the entry the `start` and `end` seconds, or leave both out to use the whole video. Optional `hide` boxes (`[x, y, w, h]` in the 720p frame) black out score bugs in clue clips, and an optional `crop` (`size` plus a `path` of `[t, x, y]` points) zooms the clue onto the play.
3. Build: `python3 clips.py build [--manifest FILE] [id ...]` writes `source.mp4`, `clip.mp4` (the full reveal), `thermal.mp4` (the clue) and `contact.png` per clip, then checks size, duration, missing audio on clues and stripped metadata. It also checks `silhouette.mp4` against [the render contract](#the-render-contract). Reruns skip anything whose inputs did not change.

Replay angles make better clues than the wide broadcast shot: the player is bigger and there is no score bug.

### Silhouettes

A clip with a `track` entry also gets `silhouette.mp4`: players in white, the ball in orange and the hoop in gray on black, cropped to follow the play. Each tracked object lists one or more `[x, y]` click points in the 720p frame on its prompt `frame`, and SAM 2.1 follows it through the clip. SAM and every pipeline script run in one venv outside the checkout, so every worktree on a machine shares it:

```sh
S=~/.local/share/bagwork/sam
python3.14 -m venv $S/venv
SAM2_BUILD_CUDA=0 $S/venv/bin/pip install --no-deps -r sam-requirements.txt
```

`--no-deps` holds every package to the version `sam-requirements.txt` pins, SAM 2 included. Before each step that uses the venv, `clips.py` checks it against the pins and stops with the first package that differs. Masks are cached by `TRACK_VERSION`, not by these versions, so a pin change also bumps `TRACK_VERSION` in `clips.py`, and `tests/test_clips.py` fails until it does.

`track.py` runs SAM 2.1 large and caches the masks in `masks.npz`, about 0.5 seconds per object per frame on an M4 (`SAM_DEVICE=cpu` overrides the Apple GPU). `silhouette.py` draws them in seconds, so changing the look never re-runs tracking. Objects can also list `avoid` points SAM should leave out, such as a backboard behind the rim.

A `refine` on an object pulls its mask back after it drifts, and changes its cache key, so `build` tracks it again over the whole clip. To check a refine first, `python3 clips.py window [--manifest FILE] id --frames 150-180:3 --objects 11` tracks only the listed objects over the frames from the first to the last given and writes `window.png` in the clip's folder: each listed frame with the masks tinted and the prompts drawn. It writes no masks, so `masks.npz` and the object caches stay as they were. See [docs/SPEC.md](docs/SPEC.md#window) for how close its masks come to a full track's.

The `track` entry also picks the crop: `"follow": "holder"` (the player with the ball; at the release it eases out to keep the shooter and the hoop in frame), `"ball"`, or `"action"` (every player and the hoop, the default), with `"zoom"` (default 2) for the first two.

Before changing a threshold in `detail.py` or `qc.py`, score it against the recorded verdicts with `~/.local/share/bagwork/sam/venv/bin/python judge.py $BAGWORK_CONTENT/judgments.json <manifest> ...`. It says which verdicts the change breaks. [docs/SPEC.md](docs/SPEC.md#judgments) has the file's format.

## The render contract

Bagwork takes a render through `bagwork clips add <manifest> --renders <folder>` in the bagwork repo, which reads each manifest entry's `id`, `player`, `title`, `url`, `start` and `end` and uploads `<folder>/<id>.mp4`. `tools/bagwork/library.ts` there refuses a render unless all of these hold:

- It is under 2,000,000 bytes, because the Worker streams every clip whole to a phone.
- ffprobe reads exactly one stream, and it is video. No audio, subtitle or data track.
- It has no chapters.
- Every tag on the container and the stream is one ffmpeg writes by default: `major_brand`, `minor_version`, `compatible_brands`, `encoder` (`Lavf` or `Lavc` with an optional ` libx264`), `handler_name` (`VideoHandler`), `language` and `vendor_id`. Any other tag or value is free text that could name the player.

The `url` must be a YouTube or X link, because the game shows it as the reveal. `silhouette.mp4` from `clips.py build` meets the stream, chapter and tag rules as it is: H.264 from libx264, no audio and stripped metadata. A long clip can still go over the byte limit, and `build` reports that as a failure.

To hand renders to Bagwork, run `python3 clips.py approve [--manifest FILE] id ...` with `BAGWORK_CONTENT` set. It checks each clip's `silhouette.mp4` against the contract and copies only the passing ones to `$BAGWORK_CONTENT/approved/<id>.mp4`. It prints why each failing one was refused and exits 1 if any failed. `render_problem` in `clips.py` copies the check from `tools/bagwork/library.ts` in the bagwork repo, with the same order and the same refusal text. That code is the source of truth. If this list or `render_problem` disagrees with it, the code in the bagwork repo wins.

# video-segmentation-pipeline

This repo turns NBA highlight clips into the silhouette clue videos that [Bagwork](https://github.com/Tominthomas/bagwork) serves. [README.md](README.md) says how to run each step, and [docs/SPEC.md](docs/SPEC.md) says how each artifact is made. Read the spec section for the code you touch before you change it.

## Commands

```sh
ruff check && ruff format --check
ty check --python .venv/bin/python
python3 -m unittest discover tests
python3 clips.py --help
python3 clips.py build [--manifest FILE] [id ...]
```

Run the four checks before every commit. `build` with a `track` entry needs the SAM venv the README's Silhouettes section sets up. Matting and SAM runs are heavy, so run one at a time and check `memory_pressure` first.

## Handoff to Bagwork

The only thing Bagwork takes from here is a render that meets [the render contract](README.md#the-render-contract). Check each approved `silhouette.mp4` against it, the 2,000,000-byte limit included, then copy it to `$BAGWORK_CONTENT/approved/<id>.mp4`. Uploading, puzzles and scheduling happen in the bagwork repo with `bagwork clips add`, `bagwork puzzle create` and `bagwork schedule set`.

## Secrecy

Recipe ids, manifests and renders name the player. Real manifests live under `$BAGWORK_CONTENT`, never in this repo. `media/` is git-ignored. Never commit a render, a frame or a manifest for a real puzzle.

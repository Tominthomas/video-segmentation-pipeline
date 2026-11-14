# video-segmentation-pipeline

This repo turns NBA highlight clips into the silhouette clue videos that [Bagwork](https://github.com/Tominthomas/bagwork) serves. [README.md](README.md) says how to run each step, and [docs/SPEC.md](docs/SPEC.md) says how each artifact is made. Read the spec section for the code you touch before you change it.

## Commands

```sh
ruff check && ruff format --check
ty check --python ~/.local/share/bagwork/sam/venv/bin/python
~/.local/share/bagwork/sam/venv/bin/python -m unittest discover tests
python3 clips.py --help
python3 clips.py build [--manifest FILE] [id ...]
python3 clips.py approve [--manifest FILE] id ...
.venv/bin/python judge.py $BAGWORK_CONTENT/judgments.json MANIFEST ...
```

Run the four checks before every commit. The checks and every `build` step use the shared SAM venv that the README's Silhouettes section installs once per machine. Matting and SAM runs are heavy, so run one at a time and check `memory_pressure` first.

## Handoff to Bagwork

The only thing Bagwork takes from here is a render that meets [the render contract](README.md#the-render-contract). `python3 clips.py approve --manifest FILE id ...` checks each `silhouette.mp4` against it and copies only the passing ones to `$BAGWORK_CONTENT/approved/<id>.mp4`. Never copy a render there by hand. Uploading, puzzles and scheduling happen in the bagwork repo with `bagwork clips add`, `bagwork puzzle create` and `bagwork schedule set`.

## Secrecy

Recipe ids, manifests and renders name the player. Real manifests live under `$BAGWORK_CONTENT`, never in this repo. `media/` is git-ignored. Never commit a render, a frame or a manifest for a real puzzle.

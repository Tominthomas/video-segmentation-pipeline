# video-segmentation-pipeline

This repo turns NBA highlight clips into the silhouette clue videos that Bagwork, a private guess-the-player game, serves. [README.md](README.md) says how to run each step, and [docs/SPEC.md](docs/SPEC.md) says how each artifact is made. Read the spec section for the code you touch before you change it.

## Commands

```sh
ruff check && ruff format --check
ty check --python ~/.local/share/bagwork/sam/venv/bin/python
~/.local/share/bagwork/sam/venv/bin/python -m unittest discover tests
python3 clips.py --help
python3 leaks.py $BAGWORK_CONTENT
python3 clips.py build [--manifest FILE] [id ...]
python3 clips.py approve [--manifest FILE] id ...
~/.local/share/bagwork/sam/venv/bin/python judge.py $BAGWORK_CONTENT/judgments.json MANIFEST ...
```

Run the first four checks before every commit, and `leaks.py` before every push. The checks and every `build` step use the shared SAM venv that the README's [Install SAM](README.md#install-sam) section installs once per machine. Matting and SAM runs are heavy, so run one at a time and check `memory_pressure` first.

## Handoff to Bagwork

The only thing Bagwork takes from here is a render that meets [the render contract](README.md#the-render-contract). `python3 clips.py approve --manifest FILE id ...` checks each `silhouette.mp4` against it and copies only the passing ones to `$BAGWORK_CONTENT/approved/<id>.mp4`. Never copy a render there by hand. Uploading, puzzles and scheduling happen in the bagwork repo with `bagwork clips add`, `bagwork puzzle create` and `bagwork schedule set`.

## Secrecy

This repo is public. Clip ids, manifests and renders name the player, and a player named before his day goes live spoils the puzzle. Real manifests live under `$BAGWORK_CONTENT`, never in this repo. `media/` is git-ignored. Never commit a render, a frame or a manifest for a real puzzle.

The same rule covers every word that leaves this machine: code, comments, docs, commit messages, PR titles and PR bodies. When a measurement needs a clip, name it by its slug without the player, such as `den-handles`, and the player by letter, as [docs/SPEC.md](docs/SPEC.md) does. `leaks.py` checks tracked files and unpushed commit messages against every real manifest. It cannot see PR text, so write that by the same rule.

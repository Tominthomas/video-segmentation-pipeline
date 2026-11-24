#!/usr/bin/env python3
"""Fail when a tracked file or an unpushed commit message names a real puzzle's player or clip.

  python3 leaks.py [DIR]

DIR is the folder of real manifests, $BAGWORK_CONTENT by default. Every top-level *.json in it that is a list of clip
entries gives its players' full and last names, its clip ids and each id's first word. The clips in DIR/approved give
their ids too. The players and ids in fixtures/clips.json are public, so they never count. Commit messages are the
ones on HEAD that origin/main lacks.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUFFIXES = {"Jr", "Sr", "II", "III", "IV"}


def terms_of(entries: list[dict]) -> set[str]:
    """The names, ids and id prefixes such as "smith-" that give away each entry's player."""
    terms = set()
    for entry in entries:
        if player := entry.get("player"):
            names = [n for n in player.split() if n.rstrip(".") not in SUFFIXES]
            terms |= {player, names[-1]}
        if clip_id := entry.get("id"):
            terms |= {clip_id, clip_id.split("-")[0] + "-"}
    return {t for t in terms if len(t.rstrip("-")) >= 3}


def secret_terms(content: Path) -> set[str]:
    entries = []
    for path in content.glob("*.json"):
        raw = json.loads(path.read_text())
        if isinstance(raw, list):
            entries += [e for e in raw if isinstance(e, dict)]
    entries += [{"id": mp4.stem} for mp4 in (content / "approved").glob("*.mp4")]
    public = json.loads((ROOT / "fixtures" / "clips.json").read_text())
    return terms_of(entries) - terms_of(public)


def pattern(terms: set[str]) -> re.Pattern:
    """One regex for every term: a name or id as a whole word, and a prefix such as "smith-" at a word's start."""
    alternatives = [
        rf"\b{re.escape(t)}" + ("" if t.endswith("-") else r"\b") for t in sorted(terms, key=len, reverse=True)
    ]
    return re.compile("|".join(alternatives))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def unpushed() -> list[str]:
    """The commits on HEAD that origin/main lacks, or every commit when there is no origin/main."""
    verify = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", "origin/main"], cwd=ROOT, capture_output=True, check=False
    )
    return git("rev-list", "origin/main..HEAD" if verify.returncode == 0 else "HEAD").split()


def leaks(found: re.Pattern) -> list[str]:
    """Each line of a tracked file, and each unpushed commit message, that matches."""
    hits = []
    for name in git("ls-files", "-z").split("\0"):
        path = ROOT / name
        if not name or not path.is_file():
            continue
        for n, line in enumerate(path.read_bytes().decode(errors="ignore").splitlines(), 1):
            hits += [f"{name}:{n}: {m[0]}" for m in found.finditer(line)]
    for commit in unpushed():
        message = git("log", "-1", "--format=%B", commit)
        hits += [f"commit {commit[:8]}: {m[0]}" for m in found.finditer(message)]
    return hits


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BAGWORK_CONTENT", "")
    if not folder or not Path(folder).is_dir():
        sys.exit("usage: python3 leaks.py DIR, or set BAGWORK_CONTENT to the folder of real manifests")
    if not (terms := secret_terms(Path(folder))):
        sys.exit(f"no manifests with players in {folder}, so there is nothing to check against")
    if hits := leaks(pattern(terms)):
        sys.exit("names a real puzzle's player or clip:\n" + "\n".join(hits))
    print(f"no leaks: {len(terms)} names and ids checked")


if __name__ == "__main__":
    main()

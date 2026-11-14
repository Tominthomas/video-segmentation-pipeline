"""Score the thresholds that pick and check the mystery player's shape against every recorded human verdict.

Run with the shared SAM venv:  $SAM_PY judge.py <judgments.json> <manifest.json> [...] [--clips DIR]
$SAM_PY is ~/.local/share/bagwork/sam/venv/bin/python, which the README's Silhouettes section installs.
Real judgments live at $BAGWORK_CONTENT/judgments.json, beside the manifests, because clip ids name the player.
--clips is the folder that holds each clip's artifacts, media/clips by default. Nothing is written.

judgments.json is {"clips": {<id>: {"verdict"?, "flags"?}}}. A verdict is {"winner", "how", "rests_on",
"evidence"}: the shape that won, "matte", "detail" or "sam", how that was found (such as blind judging on a date),
which rules in RULES the result bears on, and the evidence in words. "flags" copies qc.json's flags, each
with "real": true, false or null, for whether a review found the flagged frames really wrong.

For each verdict it measures, over the render's frames in matte.npz, what the thresholds read: his median height
share, the share of frames ghosting and the share falling back. It prints whether today's thresholds give each
verdict, the range of each threshold that gives every verdict with the others held at today's values, and each QC
check's precision over the labelled flags still in qc.json. Each printed range is rounded inward, so every
value in it gives every verdict. It exits 1 when a verdict comes out wrong.
docs/SPEC.md#judgments has the reasoning.
"""

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import numpy as np

from clips import MEDIA, load_manifest
from detail import DETAIL_BELOW, choose, height_share
from matte import Matte
from qc import GHOST_FRAME, MATTE_FALLBACK, MATTE_GHOSTING, ghosting, matte_shares
from silhouette import load

THRESHOLDS = {
    "DETAIL_BELOW": DETAIL_BELOW,
    "GHOST_FRAME": GHOST_FRAME,
    "MATTE_GHOSTING": MATTE_GHOSTING,
    "MATTE_FALLBACK": MATTE_FALLBACK,
}


SHAPES = ("matte", "detail", "sam")


class Verdict(NamedTuple):
    winner: str
    how: str
    rests_on: tuple[str, ...]
    evidence: str


class Measured(NamedTuple):
    """What the thresholds read on one clip: his median height share and the matte's per-frame ghost shares and
    fallback frames."""

    height: float
    ghost: np.ndarray
    fallback: np.ndarray


def picks_matte(m: Measured, t: dict[str, float]) -> bool:
    return choose(m.height, t["DETAIL_BELOW"]) == "matte"


def matte_passes(m: Measured, t: dict[str, float]) -> bool:
    shares = matte_shares(m.ghost, m.fallback, t["GHOST_FRAME"])
    return shares["matte ghosting"] <= t["MATTE_GHOSTING"] and shares["matte fallback"] <= t["MATTE_FALLBACK"]


# Each rule says whether the thresholds t favour his matte on a clip. A verdict names the rules its evidence bears
# on, and each of those must favour the matte exactly when it won. "auto" picks only the matte or the detail pass,
# so a verdict that SAM won is right under it when it avoids the matte.
RULES: dict[str, Callable[[Measured, dict[str, float]], bool]] = {"auto": picks_matte, "matte qc": matte_passes}

BREAKPOINTS: dict[str, Callable[[Measured, dict[str, float]], np.ndarray]] = {
    "DETAIL_BELOW": lambda m, _: np.array([m.height]),
    "GHOST_FRAME": lambda m, _: m.ghost,
    "MATTE_GHOSTING": lambda m, t: np.array([matte_shares(m.ghost, m.fallback, t["GHOST_FRAME"])["matte ghosting"]]),
    "MATTE_FALLBACK": lambda m, _: np.array([m.fallback.mean()]),
}


def parse_verdict(clip_id: str, raw: dict) -> Verdict:
    if raw.get("winner") not in SHAPES:
        sys.exit(f'{clip_id}: verdict "winner" must be one of {", ".join(SHAPES)}')
    unknown = set(raw.get("rests_on", [])) - set(RULES)
    if unknown:
        sys.exit(f'{clip_id}: verdict "rests_on" names unknown rules {sorted(unknown)}; rules are {sorted(RULES)}')
    return Verdict(raw["winner"], raw.get("how", ""), tuple(raw.get("rests_on", [])), raw.get("evidence", ""))


def measure(folder: Path, mystery: int) -> Measured:
    """His measures from a clip folder's masks.npz and matte.npz, over the frames the matte covers."""
    for name in ("masks.npz", "matte.npz"):
        if not (folder / name).exists():
            sys.exit(f"{folder.name} has a verdict but no {name}; build it first")
    _, masks = load(folder / "masks.npz")
    matte = Matte.load(folder / "matte.npz")
    frames = [matte.first, matte.first + len(matte.alpha)]
    his = masks[:, mystery]
    return Measured(height_share(masks, mystery, frames), ghosting(matte, his), matte.fallback(his))


def agrees(verdict: Verdict, m: Measured, t: dict[str, float]) -> dict[str, bool]:
    """Per rule the verdict rests on, whether the thresholds t give it."""
    return {rule: RULES[rule](m, t) == (verdict.winner == "matte") for rule in verdict.rests_on}


def passing_ranges(name: str, scored: list[tuple[Verdict, Measured]]) -> list[tuple[str, float, float, str]]:
    """The intervals of threshold name within [0, 1] that give every verdict, the others at today's values, as
    (open bracket, low, high, close bracket)."""
    points = sorted({0.0, 1.0, *(float(v) for _, m in scored for v in BREAKPOINTS[name](m, THRESHOLDS) if 0 <= v <= 1)})
    points_and_gaps = sorted([(p, p, True) for p in points] + [(a, b, False) for a, b in pairwise(points)])

    def passes(value: float) -> bool:
        t = {**THRESHOLDS, name: value}
        return all(all(agrees(v, m, t).values()) for v, m in scored)

    passing: list[list[tuple[float, float, bool]]] = []
    previous = False
    for low, high, point in points_and_gaps:
        ok = passes((low + high) / 2)
        if ok and previous:
            passing[-1].append((low, high, point))
        elif ok:
            passing.append([(low, high, point)])
        previous = ok
    return [("[" if run[0][2] else "(", run[0][0], run[-1][1], "]" if run[-1][2] else ")") for run in passing]


def shown(interval: tuple[str, float, float, str], places: int = 3) -> str:
    """An interval from passing_ranges rounded inward to places decimals, so every value it shows passes."""
    opening, low, high, closing = interval
    scale = 10**places
    # The tolerance keeps a bound that is already on the grid, such as 0.26, from moving off it by float error.
    up, down = math.ceil(low * scale - 1e-9) / scale, math.floor(high * scale + 1e-9) / scale
    opening = opening if math.isclose(up, low) else "["
    closing = closing if math.isclose(down, high) else "]"
    return f"{opening}{up:.{places}f}, {down:.{places}f}{closing}"


def flag_key(flag: dict) -> tuple:
    return flag["check"], tuple(flag["objects"]), tuple(flag["frames"])


def precision(clips: dict, folder: Path) -> tuple[dict[str, Counter], int]:
    """Per QC check, the count of its current flags labelled real, false and unlabelled, and the count of labels
    whose flag is no longer in qc.json."""
    counts: dict[str, Counter] = {}
    stale = 0
    for clip_id, entry in clips.items():
        qc_path = folder / clip_id / "qc.json"
        current = {flag_key(f): f for f in json.loads(qc_path.read_text())["flags"]} if qc_path.exists() else {}
        labels = {flag_key(f): f.get("real") for f in entry.get("flags", [])}
        stale += len(labels.keys() - current.keys())
        for key in current:
            label = labels.get(key)
            counts.setdefault(key[0], Counter())["unlabelled" if label is None else "real" if label else "false"] += 1
    return counts, stale


def mysteries(manifests: list[Path]) -> dict[str, int]:
    """Each clip's mystery player object index, for the clips in the manifests that mark exactly one."""
    found = {}
    for manifest in manifests:
        for clip in load_manifest(manifest):
            objects = json.loads(clip.track)["objects"] if clip.track else []
            marked = [i for i, obj in enumerate(objects) if obj.get("mystery")]
            if len(marked) == 1:
                found[clip.id] = marked[0]
    return found


def percent(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("judgments", type=Path)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--clips", type=Path, default=MEDIA / "clips")
    args = parser.parse_args()

    clips = json.loads(args.judgments.read_text())["clips"]
    mystery = mysteries(args.manifests)

    scored: list[tuple[Verdict, Measured]] = []
    wrong = 0
    print(f"{'clip':<34} {'winner':<6} {'height':>7} {'ghosting':>8} {'fallback':>8}  verdicts")
    for clip_id, entry in clips.items():
        if "verdict" not in entry:
            continue
        verdict = parse_verdict(clip_id, entry["verdict"])
        if clip_id not in mystery:
            sys.exit(f"{clip_id}: no manifest given marks exactly one mystery player in it")
        m = measure(args.clips / clip_id, mystery[clip_id])
        shares = matte_shares(m.ghost, m.fallback)
        results = agrees(verdict, m, THRESHOLDS)
        wrong += sum(not ok for ok in results.values())
        marks = ", ".join(f"{rule} {'ok' if ok else 'WRONG'}" for rule, ok in results.items())
        print(
            f"{clip_id:<34} {verdict.winner:<6} {percent(m.height):>7} "
            f"{percent(shares['matte ghosting']):>8} {percent(shares['matte fallback']):>8}  "
            f"{marks or 'no rule sees it: ' + verdict.evidence}"
        )
        if results:
            scored.append((verdict, m))

    print(f"\n{'threshold':<15} {'today':>6}  values that give every verdict")
    for name, today in THRESHOLDS.items():
        ranges = passing_ranges(name, scored)
        inside = any(low <= today <= high for _, low, high, _ in ranges)
        values = ", ".join(map(shown, ranges)) or "none"
        print(f"{name:<15} {today:>6}  {values}{'' if inside else '   today is outside'}")

    counts, stale = precision(clips, args.clips)
    print(f"\n{'check':<18} {'real':>5} {'false':>6} {'unlabelled':>11}  precision")
    for check, c in sorted(counts.items()):
        labelled = c["real"] + c["false"]
        share = f"{c['real'] / labelled:.0%}" if labelled else "-"
        print(f"{check:<18} {c['real']:>5} {c['false']:>6} {c['unlabelled']:>11}  {share}")
    if stale:
        print(f"{stale} labels name flags no longer in qc.json; relabel those clips")
    print(f"\n{wrong} verdicts wrong under today's thresholds")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()

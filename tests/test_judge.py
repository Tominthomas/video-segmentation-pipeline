import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judge import THRESHOLDS, Measured, Verdict, agrees, passing_ranges, precision  # noqa: E402


def measured(height=0.3, ghosting=0.0, fallback=0.0, frames=100) -> Measured:
    """A clip whose matte ghosts badly on the first ghosting share of its frames and falls back on the last
    fallback share."""
    ghost = np.zeros(frames)
    ghost[: round(ghosting * frames)] = 0.5
    fell = np.zeros(frames, bool)
    fell[frames - round(fallback * frames) :] = True
    return Measured(height, ghost, fell)


def verdict(won: bool, *rules: str) -> Verdict:
    return Verdict(won, "blind", rules, "")


# Like the spec's Houston and cle-self-lob clips: a matte that won at 26% tall and 10% ghosting, and one that lost
# at 23% tall and 30% ghosting.
WON = (verdict(True, "auto", "matte qc"), measured(0.26, ghosting=0.10))
LOST = (verdict(False, "auto", "matte qc"), measured(0.23, ghosting=0.30))


class Verdicts(unittest.TestCase):
    def test_todays_thresholds_give_both_verdicts(self):
        for v, m in (WON, LOST):
            self.assertEqual(agrees(v, m, THRESHOLDS), {"auto": True, "matte qc": True})

    def test_a_detail_below_over_the_winners_height_gets_its_verdict_wrong(self):
        moved = {**THRESHOLDS, "DETAIL_BELOW": 0.27}
        self.assertEqual(agrees(*WON, moved)["auto"], False)
        self.assertEqual(agrees(*LOST, moved)["auto"], True)

    def test_a_verdict_is_checked_only_by_the_rules_it_rests_on(self):
        unseen = verdict(False)
        self.assertEqual(agrees(unseen, measured(0.8), THRESHOLDS), {})

    def test_fallback_alone_can_fail_a_matte(self):
        self.assertEqual(agrees(verdict(False, "matte qc"), measured(fallback=0.06), THRESHOLDS), {"matte qc": True})


class Ranges(unittest.TestCase):
    def test_detail_below_passes_above_the_loser_up_to_and_including_the_winner(self):
        self.assertEqual(passing_ranges("DETAIL_BELOW", [WON, LOST]), [("(", 0.23, 0.26, "]")])

    def test_matte_ghosting_passes_from_the_winners_share_up_to_the_losers(self):
        self.assertEqual(passing_ranges("MATTE_GHOSTING", [WON, LOST]), [("[", 0.1, 0.3, ")")])

    def test_ghost_frame_passes_below_the_ghost_share_of_a_bad_frame(self):
        self.assertEqual(passing_ranges("GHOST_FRAME", [WON, LOST]), [("[", 0.0, 0.5, ")")])

    def test_verdicts_that_contradict_give_no_range(self):
        backwards = (verdict(False, "auto"), measured(0.40))
        self.assertEqual(passing_ranges("DETAIL_BELOW", [WON, backwards]), [])


class Precision(unittest.TestCase):
    def test_labels_count_only_against_the_flags_still_in_qc_json(self):
        overlap = {"check": "player overlap", "objects": [1, 2], "frames": [10, 20]}
        fill = {"check": "ball fill", "objects": [5], "frames": [30, 40]}
        gone = {"check": "ball fill", "objects": [5], "frames": [90, 99]}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a").mkdir()
            (Path(tmp) / "a" / "qc.json").write_text(json.dumps({"flags": [overlap, fill]}))
            labels = {"a": {"flags": [{**overlap, "real": False}, {**gone, "real": True}]}}
            counts, stale = precision(labels, Path(tmp))
        self.assertEqual(dict(counts["player overlap"]), {"false": 1})
        self.assertEqual(dict(counts["ball fill"]), {"unlabelled": 1})
        self.assertEqual(stale, 1)


if __name__ == "__main__":
    unittest.main()

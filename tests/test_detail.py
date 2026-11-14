import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detail import (  # noqa: E402
    DETAIL_BELOW,
    PINHOLE,
    SPECK,
    Detail,
    choose,
    components,
    despeckle,
    own,
    points_to_crop,
    prompt_to_crop,
    prompts,
    to_crop,
)
from silhouette import close_gap  # noqa: E402


def box(shape, x0, y0, x1, y1, value=True):
    mask = np.zeros(shape, type(value))
    mask[y0:y1, x0:x1] = value
    return mask


class Choose(unittest.TestCase):
    def test_a_small_player_takes_the_detail_pass_and_a_large_one_the_matte(self):
        self.assertEqual(choose(0.18), "detail")
        self.assertEqual(choose(0.40), "matte")
        self.assertEqual(choose(DETAIL_BELOW), "matte")


class Coordinates(unittest.TestCase):
    def test_a_clip_mask_maps_to_the_crop_at_source_scale(self):
        clip = box((72, 128), 40, 20, 50, 40)
        # The source is 3x the clip, and the crop's corner sits on clip pixel (30, 10).
        crop = to_crop(clip, np.array([90, 30]), (90, 120), (384, 216))
        self.assertEqual(crop.shape, (120, 90))
        ys, xs = np.nonzero(crop)
        self.assertEqual((xs.min(), xs.max() + 1, ys.min(), ys.max() + 1), (30, 60, 30, 90))

    def test_clip_points_land_on_the_same_place_in_the_crop(self):
        self.assertEqual(points_to_crop([(45, 30)], np.array([90, 30]), (3.0, 3.0)), [[45.0, 60.0]])

    def test_a_prompt_maps_corner_by_corner_and_keeps_avoid_points_in_the_crop(self):
        prompt = {"box": [40, 20, 50, 40], "points": [(45, 30)], "avoid": [(35, 15), (5, 5)]}
        mapped = prompt_to_crop(prompt, np.array([90, 30]), (3.0, 3.0), (90, 120))
        self.assertEqual(mapped["box"], [30.0, 30.0, 60.0, 90.0])
        self.assertEqual(mapped["points"], [[45.0, 60.0]])
        self.assertEqual(mapped["avoid"], [[15.0, 15.0]], "an avoid point outside the crop is dropped")


class Prompts(unittest.TestCase):
    def test_keep_points_sit_inside_him_and_avoid_points_inside_others(self):
        him = box((80, 80), 10, 10, 20, 70)
        other = box((80, 80), 50, 10, 70, 70)
        prompt = prompts(["player", "player", "ball"], np.stack([him, other, him & False]), 0, 60, None)
        self.assertEqual(len(prompt["points"]), 3)
        self.assertTrue(all(him[int(y), int(x)] for x, y in prompt["points"]))
        self.assertTrue(all(other[int(y), int(x)] for x, y in prompt["avoid"]))
        self.assertEqual(len(prompt["avoid"]), 1)

    def test_a_refine_adds_its_points_and_replaces_the_box(self):
        him = box((80, 80), 10, 10, 20, 70)
        refine = {"frame": 3, "points": [[15, 72]], "avoid": [[30, 30]], "box": [5, 5, 25, 79]}
        prompt = prompts(["player"], him[None], 0, 60, refine)
        self.assertIn((15, 72), prompt["points"])
        self.assertIn((30, 30), prompt["avoid"])
        self.assertEqual(prompt["box"], [5, 5, 25, 79])


class Components(unittest.TestCase):
    def test_pieces_touching_at_a_corner_are_one_and_apart_are_two(self):
        mask = np.zeros((9, 9), bool)
        mask[1:3, 1:3] = True
        mask[3, 3] = True
        mask[6:8, 6:8] = True
        labels, sizes = components(mask)
        self.assertEqual(sorted(sizes.values()), [4, 5])
        self.assertEqual(labels[1, 1], labels[3, 3])
        self.assertNotEqual(labels[1, 1], labels[6, 6])
        self.assertFalse(labels[~mask].any())

    def test_a_long_winding_piece_is_one(self):
        mask = np.zeros((41, 41), bool)
        mask[::4] = True
        for row in range(0, 37, 4):
            mask[row : row + 4, 40 if row % 8 == 0 else 0] = True
        _, sizes = components(mask)
        self.assertEqual(list(sizes.values()), [int(mask.sum())])


class Speckle(unittest.TestCase):
    def test_small_detached_pieces_go_and_large_ones_stay(self):
        body = box((100, 100), 30, 10, 60, 90)
        area = body.sum()
        speck = box((100, 100), 80, 50, 82, 52)
        limb = box((100, 100), 70, 20, 90, 40)
        self.assertLess(speck.sum(), SPECK * area)
        self.assertGreater(limb.sum(), SPECK * area)
        kept = despeckle(body | speck | limb)
        self.assertFalse(kept[speck].any())
        self.assertTrue(kept[limb].all() and kept[body].all())

    def test_pinholes_fill_and_large_holes_and_open_gaps_stay(self):
        body = box((100, 100), 20, 10, 80, 90)
        area = body.sum()
        pinhole = box((100, 100), 30, 20, 32, 22)
        hole = box((100, 100), 40, 40, 70, 60)
        notch = box((100, 100), 20, 80, 25, 90)
        self.assertLess(pinhole.sum(), PINHOLE * area)
        self.assertGreater(hole.sum(), PINHOLE * area)
        kept = despeckle(body & ~pinhole & ~hole & ~notch)
        self.assertTrue(kept[pinhole].all())
        self.assertFalse(kept[hole].any() or kept[notch].any())

    def test_an_empty_mask_stays_empty(self):
        self.assertFalse(despeckle(np.zeros((5, 5), bool)).any())


class Ownership(unittest.TestCase):
    """A 60x40 crop: SAM sees his leg crossing a defender whose 720p mask covers it."""

    SHAPE = (60, 40)

    def setUp(self):
        self.reach = box(self.SHAPE, 0, 0, 40, 60)
        self.core = box(self.SHAPE, 15, 10, 20, 30, 1.0).astype(np.float32)
        self.claimed = box(self.SHAPE, 20, 30, 40, 60)

    def test_his_sure_leg_wins_over_another_players_mask(self):
        prob = box(self.SHAPE, 10, 5, 25, 55, 0.9).astype(np.float32)
        owned = own(prob, self.reach, self.core, self.claimed)
        self.assertTrue((owned.alpha[30:55, 20:25] >= 0.5).all())
        self.assertFalse(owned.ceded.any())

    def test_another_players_mask_takes_only_what_sam_is_unsure_of(self):
        prob = box(self.SHAPE, 10, 5, 25, 55, 0.9).astype(np.float32)
        prob[30:55, 22:25] = 0.55
        owned = own(prob, self.reach, self.core, self.claimed)
        self.assertTrue((owned.alpha[30:55, 22:25] == 0).all())
        self.assertTrue((owned.alpha[30:55, 20:22] >= 0.5).all())
        self.assertEqual(int(owned.ceded.sum()), 25 * 3)

    def test_unsure_pixels_no_one_else_claims_keep_sams_answer(self):
        prob = box(self.SHAPE, 5, 5, 15, 55, 0.55).astype(np.float32)
        owned = own(prob, self.reach, self.core * 0, self.claimed)
        self.assertTrue(np.allclose(owned.alpha[5:55, 5:15], 0.55))

    def test_he_is_absent_past_his_reach_and_solid_in_his_core(self):
        prob = box(self.SHAPE, 0, 0, 40, 60, 0.9).astype(np.float32)
        prob[10:30, 15:20] = 0.1
        reach = box(self.SHAPE, 0, 0, 30, 60)
        owned = own(prob, reach, self.core, np.zeros(self.SHAPE, bool))
        self.assertFalse(owned.alpha[:, 30:].any())
        self.assertTrue((owned.alpha[10:30, 15:20] == 1).all())

    def test_the_speckle_rule_runs_after_ownership_and_reports_what_it_changed(self):
        prob = box(self.SHAPE, 5, 5, 25, 55, 0.9).astype(np.float32)
        prob[2, 35] = 0.9
        prob[20, 10] = 0.1
        owned = own(prob, self.reach, self.core * 0, np.zeros(self.SHAPE, bool))
        self.assertEqual(owned.alpha[2, 35], 0)
        self.assertEqual(owned.alpha[20, 10], 1)
        self.assertEqual(int(owned.speckle.sum()), 2)

    def test_an_island_too_faint_to_be_solid_is_removed_and_his_soft_edge_is_kept(self):
        prob = box(self.SHAPE, 5, 5, 25, 55, 0.9).astype(np.float32)
        prob[2:4, 34:36] = 0.49
        prob[30, 25] = 0.4
        owned = own(prob, self.reach, self.core * 0, np.zeros(self.SHAPE, bool))
        self.assertFalse(owned.alpha[2:4, 34:36].any())
        self.assertAlmostEqual(float(owned.alpha[30, 25]), 0.4, places=5)
        self.assertEqual(int(owned.speckle.sum()), 4)


class BallGap(unittest.TestCase):
    def test_the_ring_between_him_and_a_held_ball_fills_as_him(self):
        him = box((60, 60), 10, 10, 30, 50, 1.0).astype(np.float32)
        ball = box((60, 60), 32, 20, 42, 30, 1.0).astype(np.float32)
        closed = close_gap(him, ball, 3)
        self.assertTrue((closed[21:29, 30:32] > 0.9).all())
        self.assertTrue(np.array_equal(closed[:, :30], him[:, :30]))
        self.assertFalse(closed[40:, 32:].any())

    def test_a_ball_away_from_him_changes_nothing(self):
        him = box((60, 60), 10, 10, 20, 50, 1.0).astype(np.float32)
        ball = box((60, 60), 40, 20, 50, 30, 1.0).astype(np.float32)
        self.assertTrue(np.array_equal(close_gap(him, ball, 3), him))


class Render(unittest.TestCase):
    def test_layer_samples_his_owned_crop_into_the_window(self):
        clip = box((36, 64), 20, 10, 30, 30)
        # The source is 2x the clip and the crop is the 40x40 source square at (30, 10).
        prob = np.zeros((1, 40, 40), np.uint8)
        prob[0, 10 : 50 - 10, 10:30] = 255
        detail = Detail(
            prob=prob,
            first=0,
            origin=np.array([[30, 10]]),
            crop=(40, 40),
            source=(128, 72),
            source_frames=np.zeros(1, int),
            height=20.0,
            seeds=np.zeros(1, int),
            refines=np.zeros(0, int),
        )
        alpha = detail.layer(0, clip, np.zeros_like(clip), (0, 0, 64, 36), (64, 36))
        assert alpha is not None, "the frame draws his shape, not his mask"
        self.assertTrue((alpha[12:18, 21:29] > 0.9).all())
        self.assertFalse(alpha[:, :18].any() or alpha[:, 32:].any())
        owned = detail.owned(0, clip, np.zeros_like(clip))
        self.assertGreater(detail.clip_iou(0, owned, clip), 0.4)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))

from matte import Matte, crop_size, flashes, frame_map, levels, seeds  # noqa: E402
from qc import ghosting, matte_flags  # noqa: E402
from silhouette import BALLS, COLORS, FOCUS, ball_edge, dilate, erode, layers  # noqa: E402

SIZE = (40, 30)


def body(x0=12, y0=6, x1=28, y1=26) -> np.ndarray:
    mask = np.zeros(SIZE[::-1], bool)
    mask[y0:y1, x0:x1] = True
    return mask


def matte_of(alpha: np.ndarray, gate=2, fallback=False) -> Matte:
    """A one-frame matte whose crop is the whole 40x30 frame, at the same size as the clip."""
    return Matte(
        alpha=alpha[None],
        first=0,
        origin=np.zeros((1, 2), int),
        crop=SIZE,
        source=SIZE,
        gate=gate,
        source_frames=np.zeros(1, int),
        seeds=np.zeros(1, int),
        flashes=np.zeros(0, int),
        fallback=np.asarray([fallback]),
    )


class Morphology(unittest.TestCase):
    def test_dilate_grows_a_pixel_into_a_disk_of_the_radius(self):
        point = np.zeros((11, 11), bool)
        point[5, 5] = True
        ys, xs = np.ogrid[:11, :11]
        self.assertTrue(np.array_equal(dilate(point, 3), (ys - 5) ** 2 + (xs - 5) ** 2 <= 9))

    def test_dilate_takes_the_max_of_a_soft_image(self):
        image = np.zeros((9, 9), np.float32)
        image[4, 4] = 0.5
        self.assertEqual(dilate(image, 2)[4, 6], 0.5)
        self.assertEqual(dilate(image, 2)[4, 7], 0)

    def test_erode_shrinks_a_mask_but_keeps_it_where_the_frame_edge_cuts_it(self):
        mask = body(x0=0)
        shrunk = erode(mask, 3)
        self.assertTrue(shrunk[9:23, 0:25].all())
        self.assertFalse(shrunk[:9].any() or shrunk[:, 25:].any() or shrunk[23:].any())


class Refine(unittest.TestCase):
    def test_levels_cut_faint_alpha_and_fill_strong_alpha(self):
        self.assertEqual(levels(np.asarray([30, 159, 200], np.uint8)).tolist(), [0, 1, 1])
        self.assertAlmostEqual(float(levels(np.asarray([94], np.uint8))[0]), 0.5, places=2)

    def test_the_gate_cuts_matte_beyond_his_grown_mask(self):
        alpha = np.full(SIZE[::-1], 255, np.uint8)
        refined = matte_of(alpha).source_alpha(0, body())
        self.assertLess(refined[15, 36], 0.01)
        self.assertLess(refined[0, 20], 0.05)
        self.assertGreater(refined[15, 28], 0.4)

    def test_the_core_is_solid_where_the_matte_left_a_hole(self):
        alpha = np.zeros(SIZE[::-1], np.uint8)
        refined = matte_of(alpha).source_alpha(0, body())
        self.assertGreater(refined[16, 20], 0.99)
        self.assertLess(refined[6, 12], 0.5)

    def test_a_window_maps_each_axis_by_its_own_ratio(self):
        matte = matte_of(np.full((30, 42), 255, np.uint8))._replace(crop=(42, 30), source=(42, 30))
        alpha = matte.window_alpha(0, body(), (0, 0, 40, 30), (80, 60))
        self.assertEqual(alpha.shape, (60, 80))

    def test_a_matte_that_agrees_with_sam_is_kept(self):
        alpha = body().astype(np.uint8) * 255
        self.assertFalse(matte_of(alpha).falls_back(0, body()))

    def test_a_frame_whose_matte_lost_him_falls_back_to_sam(self):
        thin = body(x0=19, x1=21)
        self.assertTrue(matte_of(np.zeros(SIZE[::-1], np.uint8)).falls_back(0, thin))


class Geometry(unittest.TestCase):
    def test_each_clip_frame_takes_the_nearby_source_frame_it_matches(self):
        source = np.stack([np.full((4, 4), v, np.float32) for v in range(12)])
        clip = source[[0, 3, 4, 6, 8]]
        self.assertEqual(frame_map(clip, source, 2.0).tolist(), [0, 3, 4, 6, 8])

    def test_the_crop_is_a_padded_square_or_the_whole_frame(self):
        box = np.tile([0.0, 0.0, 50.0, 100.0], (20, 1))
        self.assertEqual(crop_size(box, (1920, 1080)), (160, 160))
        self.assertEqual(crop_size(box * 8, (1920, 1080)), (1920, 1080))

    def test_a_flash_frame_is_scaled_to_its_neighbours(self):
        found = flashes(np.asarray([100.0, 101, 99, 140, 100, 102, 98]))
        self.assertEqual(list(found), [3])
        self.assertEqual(found[3], 100.0)


class Seeds(unittest.TestCase):
    def test_a_seed_avoids_frames_where_he_touches_another_player(self):
        frames = 120
        masks = np.zeros((frames, 2, 30, 80), bool)
        masks[:, 0, 5:25, 10:20] = True
        masks[:, 1, 5:25, 22:32] = True
        masks[70:, 1] = False
        masks[70:, 1, 5:25, 60:70] = True
        self.assertEqual(seeds(["player", "player"], masks, 0, 0, frames), [70])


class Flags(unittest.TestCase):
    def test_a_matte_that_ghosts_inside_his_core_is_flagged(self):
        sam = body()[None]
        ghost = matte_of(np.zeros(SIZE[::-1], np.uint8))
        self.assertEqual(ghosting(ghost, sam).tolist(), [1.0])
        self.assertEqual([f["check"] for f in matte_flags(ghost, sam)], ["matte ghosting"])

    def test_a_clean_matte_has_no_flags(self):
        sam = body()[None]
        clean = matte_of(body().astype(np.uint8) * 255)
        self.assertEqual(matte_flags(clean, sam), [])

    def test_fallback_frames_are_flagged(self):
        sam = body()[None]
        fell = matte_of(body().astype(np.uint8) * 255, fallback=True)
        self.assertEqual([f["check"] for f in matte_flags(fell, sam)], ["matte fallback"])


class BallEdge(unittest.TestCase):
    def test_the_edge_reaches_its_width_past_the_ball_only_over_him(self):
        ball = np.zeros((21, 21), np.float32)
        ball[8:13, 8:13] = 1
        him = np.zeros((21, 21), np.float32)
        him[:, 10:] = 1
        edge = ball_edge(ball, him, 3)
        self.assertEqual(edge[10, 14], 1)
        self.assertEqual(edge[10, 15], 0.5)
        self.assertEqual(edge[10, 16], 0)
        self.assertEqual(edge[10, 5], 0)

    def test_a_fractional_edge_is_anti_aliased(self):
        ball = np.zeros((21, 21), np.float32)
        ball[8:13, 8:13] = 1
        edge = ball_edge(ball, np.ones_like(ball), 1.5)
        self.assertEqual(edge[10, 13], 1)
        self.assertAlmostEqual(float(edge[10, 14]), 0.0)
        self.assertEqual(ball_edge(ball, np.ones_like(ball), 0).max(), 0)


class BallColor(unittest.TestCase):
    def test_the_ball_takes_his_white_by_default_and_orange_on_request(self):
        kinds = ["player", "player", "ball", "hoop"]
        self.assertEqual(layers(kinds, [1], BALLS["white"])[-1], (2, FOCUS["mystery"]))
        self.assertEqual(layers(kinds, [1], BALLS["orange"])[-1], (2, FOCUS["ball"]))

    def test_without_a_mystery_player_the_ball_stays_orange(self):
        self.assertEqual(layers(["player", "ball"], [], BALLS["white"])[1], (1, COLORS["ball"]))


if __name__ == "__main__":
    unittest.main()

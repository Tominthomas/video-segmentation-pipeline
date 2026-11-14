import copy
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clips import render_problem  # noqa: E402

# What ffprobe reports for a silhouette.mp4 that clips.py build writes: libx264 in an mp4 with stripped metadata.
SILHOUETTE: dict[str, Any] = {
    "streams": [
        {
            "codec_type": "video",
            "tags": {
                "language": "und",
                "handler_name": "VideoHandler",
                "vendor_id": "[0][0][0][0]",
                "encoder": "Lavc62.11.100 libx264",
            },
        }
    ],
    "chapters": [],
    "format": {
        "tags": {
            "major_brand": "isom",
            "minor_version": "512",
            "compatible_brands": "isomiso2avc1mp41",
            "encoder": "Lavf62.3.100",
        }
    },
}
SIZE = 900_000


def probe_with(**changes):
    info = copy.deepcopy(SILHOUETTE)
    info.update(changes)
    return info


def tagged(part: str, tag: str, value: str):
    info = copy.deepcopy(SILHOUETTE)
    target: dict[str, Any] = info["format"] if part == "format" else info["streams"][0]
    target["tags"][tag] = value
    return info


class RenderProblem(unittest.TestCase):
    def test_build_output_passes(self):
        self.assertIsNone(render_problem(SILHOUETTE, SIZE))

    def test_untagged_render_passes(self):
        bare = {"streams": [{"codec_type": "video"}], "chapters": [], "format": {}}
        self.assertIsNone(render_problem(bare, SIZE))

    def test_size_must_stay_under_two_million_bytes(self):
        self.assertIsNone(render_problem(SILHOUETTE, 1_999_999))
        self.assertEqual(render_problem(SILHOUETTE, 2_000_000), "2000000 bytes, over the 2000000 limit")

    def test_audio_is_refused(self):
        audio = probe_with(streams=[*SILHOUETTE["streams"], {"codec_type": "audio"}])
        self.assertEqual(render_problem(audio, SIZE), "it has an audio stream")

    def test_only_one_video_stream(self):
        two = probe_with(streams=[*SILHOUETTE["streams"], {"codec_type": "subtitle"}])
        self.assertEqual(render_problem(two, SIZE), "it has 2 streams, not one video stream")
        self.assertEqual(render_problem(probe_with(streams=[]), SIZE), "it has 0 streams, not one video stream")
        data = probe_with(streams=[{"codec_type": "data"}])
        self.assertEqual(render_problem(data, SIZE), "it has 1 streams, not one video stream")

    def test_chapters_are_refused(self):
        chaptered = probe_with(chapters=[{"tags": {"title": "Chapter 1"}}])
        self.assertEqual(render_problem(chaptered, SIZE), "it has 1 chapters")

    def test_tags_outside_the_contract_are_refused(self):
        for part, tag in [("format", "title"), ("format", "comment"), ("stream", "artist"), ("stream", "rotate")]:
            with self.subTest(part=part, tag=tag):
                self.assertEqual(render_problem(tagged(part, tag, "x"), SIZE), f"it carries a {tag} tag")

    def test_allowed_tags_with_free_text_values_are_refused(self):
        cases = [
            ("stream", "handler_name", "Core Media Video"),
            ("format", "encoder", "Lavf62.3.100 Player Name"),
            ("stream", "encoder", "HandBrake 1.9"),
            ("stream", "language", "eng\n"),
            ("format", "minor_version", "٥١٢"),
            ("format", "major_brand", "isom2"),
        ]
        for part, tag, value in cases:
            with self.subTest(tag=tag, value=value):
                self.assertEqual(render_problem(tagged(part, tag, value), SIZE), f"it carries a {tag} tag")

    def test_encoder_values_ffmpeg_writes_pass(self):
        for value in ("Lavf", "Lavc61.19.100", "Lavc62.11.100 libx264"):
            with self.subTest(value=value):
                self.assertIsNone(render_problem(tagged("stream", "encoder", value), SIZE))


if __name__ == "__main__":
    unittest.main()

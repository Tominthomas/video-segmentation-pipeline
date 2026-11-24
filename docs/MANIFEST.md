# Clip manifest

A manifest is a JSON array of clip entries. `clips.py` reads it with `--manifest FILE`, and defaults to [`fixtures/clips.json`](../fixtures/clips.json). Coordinates are pixels in the 1280×720 clip, and frames count from 0 at the clip's start at 30 fps.

## Entry

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | The clip's folder name under `media/clips/` and its file name when approved. Unique in the manifest. |
| `player` | string | yes | The player the clip is about. |
| `title` | string | yes | A short description of the play. |
| `url` | string | yes | A YouTube or X post, or a direct link to a video file. `yt-dlp` fetches it. |
| `start`, `end` | number | no | The window in seconds. Give both, or neither for the whole video. |
| `hide` | array of `[x, y, w, h]` | no | Boxes blacked out in `thermal.mp4`, such as a score bug. |
| `crop` | object | no | A window that zooms `thermal.mp4` onto the play. See [Crop](#crop). |
| `track` | object | no | Prompts for SAM. An entry with `track` also gets `silhouette.mp4`. See [Track](#track). |

## Crop

| Key | Type | Meaning |
|---|---|---|
| `size` | `[w, h]` | The window's size. |
| `path` | array of `[t, x, y]` | The window's top-left corner at `t` seconds. The window moves in a straight line between points and holds still before the first and after the last. |

## Track

| Key | Type | Default | Meaning |
|---|---|---|---|
| `frame` | int | required | The frame every object is prompted on, unless the object gives its own. |
| `objects` | array | required | The things to track. See [Object](#object). |
| `follow` | string | `"action"` | What the crop follows. `"action"` keeps every player and the hoop in frame. `"ball"` follows the ball. `"holder"` follows the player with the ball and widens at the release to keep the shooter, the ball and the hoop. `"full"` shows the whole frame, pillarboxed to 16:9. |
| `zoom` | number | 2 | For `"ball"` and `"holder"`, the window is 1/`zoom` of the frame's width. |
| `shape` | string | `"auto"` | How the mystery player is drawn: `"matte"`, `"detail"`, `"sam"`, or `"auto"` to pick by his size. [docs/SPEC.md](SPEC.md#python-clip-pipeline) explains each. |
| `ball` | string | `"white"` | With a mystery player, the ball draws in his white. `"orange"` keeps it orange. |
| `ball_edge` | number | 0 | A dark edge this many pixels wide, at 720 tall, where the ball overlaps the mystery player. |
| `ball_keys` | array of `[frame, x, y]` | `[]` | Ball positions placed by hand where SAM loses it. `clips.py crops` reads them off a gridded zoom. |

## Object

| Key | Type | Meaning |
|---|---|---|
| `kind` | string | `"player"`, `"ball"` or `"hoop"`. |
| `points` | array of `[x, y]` | Points on the object, on its prompt frame. An object needs `points` or `box`. |
| `avoid` | array of `[x, y]` | Points SAM must leave out, such as the backboard behind a rim. |
| `box` | `[x0, y0, x1, y1]` | A box around the object. For the hoop, cover the rim and the net. |
| `frame` | int | This object's prompt frame, in place of the track's `frame`. |
| `refine` | array | Extra prompts `{"frame", "points", "avoid", "box"}` that pull the mask back after it drifts. Check one first with `clips.py window`. |
| `mystery` | bool | Marks the player to guess. He draws white and every other player dark gray. |
| `detail` | array | Refines `{"frame", "points", "avoid", "box"}` for the detail pass. Read them off `clips.py crops --detail`. |

## Example

The replay entry in `fixtures/clips.json` tracks three players, the ball and the hoop, and prompts the hoop on frame 30 instead of frame 0.

```json
{
	"id": "ray-allen-2013-finals-g6-corner-three-replay",
	"player": "Ray Allen",
	"title": "Game-tying corner three, 2013 Finals Game 6 (baseline replay)",
	"url": "https://www.youtube.com/watch?v=tr6XsZVb-ZE",
	"start": 51.5,
	"end": 55,
	"track": {
		"frame": 0,
		"objects": [
			{"kind": "player", "points": [[965, 430], [975, 560]]},
			{"kind": "player", "points": [[180, 520]]},
			{"kind": "player", "points": [[600, 500]]},
			{"kind": "ball", "points": [[750, 25]]},
			{"kind": "hoop", "frame": 30, "points": [[185, 305], [215, 255]]}
		]
	}
}
```

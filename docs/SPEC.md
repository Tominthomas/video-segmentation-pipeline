# video-segmentation-pipeline spec

This spec says how the repo turns a clip manifest into the silhouette renders that Bagwork serves. [README.md](../README.md) says how to run it. Bagwork itself lives in `Tominthomas/bagwork`, and the only thing it takes from here is a render that meets [the render contract](../README.md#the-render-contract).

## Python: clip pipeline

`clips.py build` turns a manifest entry into its renders in `media/clips/<id>/`. Each artifact is rebuilt only when its recipe changes, and `build.json` holds each recipe's hash.

| Artifact | Made by | From |
|---|---|---|
| `source.mp4` | yt-dlp | the url and the fetch window |
| `clip.mp4` | ffmpeg | `source.mp4` at 720p and 30 fps |
| `masks.npz` | `track.py`, SAM 2.1 small, under the SAM slot | `clip.mp4` and each object's prompts |
| `matte.npz` | `matte.py`, MatAnyone 2, under the SAM slot | `source.mp4` and the mystery player's masks |
| `detail.npz` | `detail.py`, SAM 2.1 small, under the SAM slot | `source.mp4`, the players' masks and his detail refines |
| `qc.json` | `qc.py` | `masks.npz` and `matte.npz` or `detail.npz` |
| `silhouette.mp4` | `silhouette.py`, then ffmpeg | `masks.npz`, `matte.npz` or `detail.npz`, and the render spec |

A clip with one mystery player draws his shape one of three ways, set by `"shape"` in its track spec.

| `"shape"` | His layer | When it wins |
|---|---|---|
| `"matte"` | MatAnyone 2's matte, see [Matting](#matting) | he is large and clear of others |
| `"detail"` | his own SAM pass on a zoomed crop, see [Detail](#detail) | he is small, as in a broadcast's wide angle |
| `"sam"` | his mask in `masks.npz` | either struggles, or neither is installed |

`"auto"`, the default, picks `"detail"` when his median box in `masks.npz` over the render's frames is under 24% of the clip's height and `"matte"` otherwise. Nembhard's five wide clips measure 18 to 20% and his close clip 40%. McDaniels' Houston clip, whose matte won blind judging, is 26%, and `cle-self-lob`, whose matte lost, is 23%. A clip draws one of them, never two. A machine that can't run the chosen pass prints why and draws `"sam"`.

### Prompting

The hoop object's box covers the rim and the net, not the rim alone, so the net shows in the silhouette. Tomin wants the net to show.

### Tracking speed

SAM 2.1 small tracks one object at about 0.45 s a frame on an M4, and the GPU is the limit: device use sits at 99% while Python uses about 3% of a core. A benchmark on 2026-11-14 tracked one player through all 210 frames of `nembhard-2025-den-handles`, alternating five runs of each small variant.

| Variant | Total time, median and range | Masks against today's |
|---|---|---|
| Today, `offload_state_to_cpu=True` | 98.4 s, 97.3 to 100.6 | |
| State kept on the GPU | 98.3 s, 95.2 to 100.1 | identical |
| SAM 2.1 tiny, 3 runs | 90.7 s, 90.2 to 92.8 | median IoU 0.93, 10th percentile 0.85 |

Keeping SAM's state on the GPU changes nothing, because Apple's memory is shared by the CPU and GPU. Tiny saves 8% but moves his edge enough that one frame in ten agrees with small's below 0.85, so tracking stays on small with the offload as it is. Two SAM jobs at once would only split the GPU between them, which is why the SAM slot runs one at a time.

### Matting

SAM's masks are 720p and blocky at a player's edge, so hands and the ball handler's shape blur together. A matte of the mystery player alone, made from the full-resolution source, gives his edge. Only he is matted. Matting every player costs a model run each and draws the eye away from him.

**When.** A clip whose `"shape"` resolves to `"matte"`. A machine without MatAnyone 2 prints `matte: off` with the reason and renders SAM's silhouette, so the build still works there.

**`matte.npz`.** It covers the render's frames only, `[first, end)` of `clip.mp4`.

| Key | Shape | Holds |
|---|---|---|
| `alpha` | uint8 `[n, h, w]` | the model's raw alpha on the crop, at model size, before any curve |
| `first` | int | the clip frame of `alpha[0]` |
| `origin` | int `[n, 2]` | each frame's crop corner `(x, y)` in source pixels |
| `crop` | int `[2]` | the crop's `(width, height)` in source pixels. The scale is `alpha`'s size over it. |
| `source` | int `[2]` | the source's `(width, height)` |
| `source_frames` | int `[n]` | the source frame each clip frame was matted from |
| `seeds` | int `[k]` | the clip frames that seeded the model |
| `flashes` | int `[m]` | the clip frames dimmed to their neighbours before the model saw them |
| `gate` | int | the gate and core width in clip pixels |
| `fallback` | bool `[n]` | the frames that draw SAM's silhouette instead |

**How `matte.py` makes it.**

- **Frame map.** A 59.94 fps source does not land on the 30 fps clip by a fixed step. Each clip frame takes the source frame within three of its timestamp whose pixels match it best, on 320×180 gray thumbnails.
- **Crop.** A square that follows his smoothed SAM box, 1.5 times the 95th percentile of his box's longer side, in source pixels. The model sees it at 960 pixels. When the square would not fit the source, the crop is the whole frame.
- **Flashes.** A frame more than 15% brighter than the median of the three frames either side of it is a camera flash. It is scaled to that median before the model sees it.
- **Seeds.** A seed is a frame where his SAM mask, grown by 12 pixels, touches no other player and his area is within 20% of his median. One seed per stretch of 60 frames keeps a lost matte from running far. Each seed mattes the frames nearer to it than to any other seed, forward and backward.
- **Refine.** A levels curve `clip((a - 0.12) / 0.5, 0, 1)` stops motion blur ghosting. The result is limited to his SAM mask grown by `gate`, which is 4% of his median height and at least 3 pixels, so it can't drift onto a defender. Inside his SAM mask shrunk by `gate` it is solid, which fills holes.
- **Fallback.** A frame whose refined matte overlaps his SAM mask with an IoU under 0.5 falls back to SAM.

The model step runs under the SAM slot with the same `BAGWORK_MIN_FREE` gate, because it shares the GPU. It runs about 0.65 s a frame on an M4 and peaks near 8 GB.

**Cache recipe.** `MATTE_VERSION` in `clips.py`, bumped when a `silhouette.py` helper that `matte.py` imports changes, the `source.mp4` hash, the mystery player's mask cache key and the other players' keys, because seeds avoid them, the render's frames, the crop and seed settings in `matte.py`, the MatAnyone 2 commit, the weights' sha256, and the text of `matte.py`. Seeds come from those masks by code in `matte.py`, so the key covers them.

**How `silhouette.py` reads it.** `clips.py` passes the matte's path in the render spec. With a matte, the mystery player's layer is the refined alpha, sampled from source pixels straight to the output, and every layer renders at 720 tall, or the source's height when that is less. A fallback frame draws his SAM mask. Without a matte, the render is the same as before matting existed, at the crop's own size, and ffmpeg scales it to 720.

**Ball.** With a mystery player, the ball draws in his white, so he and the ball read as one shape, as on guessjumpshot.com. A ball in front of his body disappears into him, and Tomin accepts that. It draws in the same layer order as before, over him. `"ball": "orange"` in the track spec keeps the orange ball. `"ball_edge": <px>` adds a dark edge that many pixels wide at 720 tall where the ball overlaps him, for either colour, and defaults to 0. The tested widths were 3 for orange and 1.5 for white. A clip with no mystery player keeps the orange ball and no edge. The ball's tracking, its shape checks and the rule that hides it on long gaps are unchanged.

Two looks were tried and not built. His matte drawn over the ball claims 60 to 75% of the ball. A ball that is white only while he holds it depends on `holders`, which gave the ball to the wrong player through a whole handles sequence, and the colour crossfade read as skin.

**QC.** `qc.py` adds two clip-wide flags from `matte.npz`. `matte fallback` flags when more than 5% of matted frames fell back to SAM. `matte ghosting` flags when more than 20% of matted frames have the raw matte leaving over 5% of his SAM core under 0.5. Either flag means the matte struggled, and the review should compare it with `"shape": "detail"` and `"shape": "sam"`.

The thresholds come from blind judging on 2026-11-12. The McDaniels matte that won, Houston, ghosts on 10% of frames. Three that lost ghost on 26 to 66%: `lac-pnr-dunk`, `den-iso` and `cle-self-lob`. `den-iso` also falls back on 6%. Nembhard's close clip, where the matte won 14 of 14 frames, ghosts on 2%. The fourth loser, `den-block`, passes both checks. He fills most of its frame, so the crop is the whole source and the matte adds little over SAM, and no measure here sees that.

**Install.** MatAnyone 2 lives outside the checkout in `~/.local/share/bagwork/matte/`, so every worktree on a machine shares one copy. Its own venv keeps OpenCV, SciPy and a non-commercial package out of `.venv`. Its `pyproject.toml` fails to build a wheel at the pinned commit, so it installs editable from a checkout.

```sh
M=~/.local/share/bagwork/matte
git clone https://github.com/pq-yang/MatAnyone2.git $M/MatAnyone2
git -C $M/MatAnyone2 checkout 0079197acd6d16a741f71558809c06c586c579e0
python3.14 -m venv $M/venv
$M/venv/bin/pip install --no-deps -r matte-requirements.txt
$M/venv/bin/pip install --no-deps -e $M/MatAnyone2
curl -L -o $M/matanyone2.pth https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth
```

`--no-deps` holds every version to `matte-requirements.txt`. MatAnyone 2's own list adds its demo's GUI and training packages, which inference never imports. Before each matte, `clips.py` checks the checkout's commit and the weights' sha256 against the pins and turns matting off with the reason when either differs.

**Licence.** MatAnyone 2 is under the S-Lab License 1.0. Its terms, from `LICENSE.txt` at the pinned commit:

> Redistribution and use for non-commercial purpose in source and binary forms, with or without modification, are permitted provided that the following conditions are met: [...] In the event that redistribution and/or use for commercial purpose in source or binary forms, with or without modification is required, please contact the contributor(s) of the work.

Use is non-commercial only. Renders made with it must not go into a commercial Bagwork without the authors' permission. Turning matting off for every clip removes the dependency.

### Detail

SAM tracks everyone on the 1280×720 clip and predicts each mask on a 256-cell grid over the whole frame, so a player 130 pixels tall is about 26 cells tall and his hands, the ball and his legs blur together. The detail pass gives the mystery player his own SAM 2.1 run on a crop that follows him in the full-resolution source. SAM still predicts on a 256-cell grid, but over the crop, so he is about 180 cells tall. The matte does the same job when he is large and clear of others, and ghosts when he is small or crowded, which is when the detail pass gains most. On a close clip it adds little and can speckle, so it is not the default there.

Prototyping on Nembhard's clips settled four things. SAM's video mode over the crop sequence beats image mode per frame, which grabbed the defender more and flickered about 25% more. The zoomed mask must be his shape itself: used only to gate the matte, it changed almost nothing. A 1080 render adds nothing, so the render stays 720 tall. And the 720p track decides who is who, while the zoomed mask decides his edge.

**`detail.npz`.** It covers the render's frames only, `[first, end)` of `clip.mp4`.

| Key | Shape | Holds |
|---|---|---|
| `prob` | uint8 `[n, ch, cw]` | SAM's probability that each crop pixel is him, the sigmoid of its logits times 255, at the crop's source size |
| `first` | int | the clip frame of `prob[0]` |
| `origin` | int `[n, 2]` | each frame's crop corner `(x, y)` in source pixels |
| `crop` | int `[2]` | the crop's `(width, height)` in source pixels |
| `source` | int `[2]` | the source's `(width, height)` |
| `source_frames` | int `[n]` | the source frame each clip frame was cut from |
| `height` | float | his median box height in clip pixels, which every width below is a share of |
| `seeds` | int `[k]` | the clip frames prompted from his mask in `masks.npz` |
| `refines` | int `[m]` | the clip frames prompted by the manifest's detail refines |
| `fallback` | bool `[n]` | the frames that draw his mask from `masks.npz` instead |

`prob` keeps SAM's raw output, before the ownership rule, so that rule, the speckle rule and their QC run on it at render time and a change to them never re-runs the model.

**How `detail.py` makes it.**

- **Frame map and crop.** The matte's: the source frame that best matches each clip frame, and a square that follows his smoothed box, 1.5 times the 95th percentile of its longer side, or the whole frame when that doesn't fit. SAM sees each crop at its source size and scales it to 1024 itself.
- **Prompts.** Each seed frame, picked by the matte's seed rule, gets his box from `masks.npz` padded by 4% of his height, one keep point deep inside each third of his mask from top to bottom, and one avoid point deep inside each other player's mask in the crop. Every prompt maps clip pixels to crop pixels through the crop's origin and the source's scale.
- **Model.** SAM 2.1 small in video mode over the crop sequence, forward from the first prompted frame and back from it, under the SAM slot with the same `BAGWORK_MIN_FREE` gate. Its state stays on the GPU: offloaded, SAM copies each mask to the CPU without waiting, and the masks propagation yields read as empty. On Nembhard's wide clips, with crops of 432 to 528 pixels, it runs about 0.5 s a frame and the build peaks at 5.6 to 6.9 GB resident.
- **Fallback.** A frame where his mask in `masks.npz` is empty, or where his owned mask scaled to the clip overlaps it with an IoU under 0.3, draws his mask from `masks.npz`. The bar is low because the 720p mask is the coarse one. On `den-handles` the two agree at a median IoU of 0.9 and a 10th percentile of 0.8 when the detail pass is right.

**Ownership.** His mask in `masks.npz` decides who is who, and `prob` decides his edge. Per crop pixel, with `p` its probability:

1. Beyond his mask in `masks.npz` grown by 8% of his height, he is absent. That bounds the pass to him, so a defender it grabs past that reach never draws.
2. Where `p` is at least 0.65 or at most 0.35, SAM is sure, and its answer stands, even over another player's mask. He draws on top of the gray players anyway.
3. Where `p` is between, another player's mask, shrunk by a clip pixel, takes the pixel from him, and the pixel is "ceded". Elsewhere `p` stands.
4. Inside his mask in `masks.npz` shrunk by 6% of his height, away from every other player's mask, he is solid. Dark shorts on a dark defender leave SAM unsure inside his body, and this fills them.

The prototype cut his leg on `den-handles` frame 52 and `okc-pullup-close` frame 63 because it removed every pixel another player's 720p mask held, sure or not. Rule 3 cedes only the unsure ones.

**Speckle.** After ownership, a piece of him not joined to the rest smaller than 2% of his area is removed, and a hole in him smaller than 1% of his area is filled. Joined means 8-connected at the crop's resolution. A real limb past a defender is a bigger piece than that, and SAM's specks on a light floor are smaller.

**Ball.** His edge comes from the crop and the ball's from the 720p clip, so they do not meet where he holds it, and a thin dark ring shows between them. At render, where the ball is within 3 pixels of him at 720 tall, the gap between them is closed by a closing of the two together, and the closed pixels draw as him. With the white ball, the two read as one shape.

**Refines.** An agent fixes a wrong spot on the mystery player object with `"detail"`, a list of `{"frame", "points", "avoid", "box"?}` in clip pixels, like a track refine. Each refine's frame is prompted with the seed's automatic prompts for that frame plus its points, and its box in place of the automatic box when given. A refine re-runs only the detail pass. `clips.py crops <id> --frames 40-70:3 --detail` writes `detail.png`: each frame's crop from the source, gridded in clip pixels so a point read off it is a refine as is, with his owned mask outlined, ceded pixels in red, removed speckle in yellow and the refine's points drawn.

**QC.** `qc.py` adds three per-frame flags from `detail.npz`, each a run of frames.

| Check | Flags a frame where | Worst |
|---|---|---|
| `detail disagrees` | his owned mask at clip size overlaps his `masks.npz` mask with an IoU under 0.45 | lowest IoU |
| `detail speckle` | the speckle rule removed or filled more than 3% of his area | largest share |
| `detail ceded` | other players took more than 8% of his area under rule 3 | largest share |

Each points the review at a frame to look at with `crops --detail` and maybe fix with a refine.

**Cache recipe.** `DETAIL_VERSION` in `clips.py`, bumped when a `silhouette.py` or `matte.py` helper that `detail.py` imports changes, the `source.mp4` hash, the mystery player's mask cache key and the other players' keys, because the avoid points come from them, the render's frames, the detail refines, `TRACK_VERSION`, which a change of SAM model bumps, and the text of `detail.py`. `silhouette.mp4` and `qc.json` add the text of `detail.py` too, because they apply its rules.

**How `silhouette.py` reads it.** `clips.py` passes its path as `"detail"` in the render spec. His layer is the owned alpha, sampled from source pixels to the output like the matte's, at 720 tall. A fallback frame draws his mask from `masks.npz`. The ball's gap is closed as above.


## Tooling

`pyproject.toml` holds the ruff and ty config. Line length is 120. Rule sets are `E`, `F`, `I`, `B`, `UP`, `SIM`, `ERA`, `ARG`, `BLE`, `C4`, `C90`, `PIE`, `RET`, `RUF`, `PTH`, `PL`, `FURB` and `PERF`, less `PLR2004` (magic values), `PLR0913` and `PLR0917` (argument counts). `max-complexity` is 14, so a check table or a state machine can stay one function. `subprocess.run` always names `check`. `ruff format` owns layout.

| Tool | Version | Used for |
|---|---|---|
| `ruff` (brew) | 0.17.4 | lint and format |
| `ty` (brew) | 0.0.100 | type check |
| numpy, Pillow (`.venv`) | 2.5.4, 13.0.0 | every script |
| torch, torchvision, SAM 2.1 (`.venv`) | see [README.md](../README.md#silhouettes) | `track.py`, `detail.py`, `propose.py` |
| MatAnyone 2 (own venv, see [Matting](#matting)) | commit `0079197`, weights sha256 `5e9821e4…5914` | the mystery player's matte. S-Lab License 1.0, non-commercial use only. |

`python.yml` runs on pull requests, on pushes to `main` and on manual dispatch. It runs `ruff check`, `ruff format --check`, `ty check` and `python3 -m unittest discover tests`, and sets `timeout-minutes: 5`. `ty check` runs against numpy and Pillow at the project venv's versions. torch, torchvision, SAM and MatAnyone 2 are too big to install there, so `pyproject.toml` lets their imports go unresolved and they type as `Any`.

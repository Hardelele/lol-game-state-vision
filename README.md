# LoL Game State Vision

Reading game state from League of Legends gameplay videos, using only the
main game view. The current task: tell where the camera is on the map from
a single frame, with the HUD and the minimap masked out.

This is a research project, not a product. Code comments, docstrings and the
dataset notes are in Russian; the README is in English.

## Current state

- **Model.** The patch model (`tools/patch_model.py`, trained by
  `tools/train_patches.py`) is the current best. The input is warped through
  the screen → ground homography to a top-down view at a fixed world scale
  (6 game units per pixel), so the pixel count, resolution and aspect ratio
  of the source frame do not matter. A fully convolutional encoder (3.95 M
  parameters, no positional head) gives every 16 px patch an L2-normalised
  128-d embedding, a distribution over a 64×64 map grid and a "this is game
  scene" score. The camera position is the consensus of patch votes. On four
  held-out matches the median camera error is 30 game units (the map is
  14,800 across), against 44 for `CoordNet` trained on the same split.
- **Robust to the HUD.** On a held-out match the patch model stays at
  30–40 units with no miss over 1,000 units when temporary masks are
  removed, foreign HUD panels or black boxes are added, the crop changes by
  ±8%, the frame is 960×540, 1280×720 or 4:3, or only 30–50% of the frame is
  given. `CoordNet` degrades 2–20× under the same changes. Frozen DINOv2-S
  features with the same head reach 73 units and are 2–4× slower. Details,
  tables and speed: [docs/patch-embeddings.md](docs/patch-embeddings.md).
- **"Not a game" screens.** A small frame head on top of the frozen patch
  model (`tools/train_not_game.py`) tells game frames from streamer
  webcams, the LoL client, loading and intro screens and other games. On
  held-out streams it finds 96% of non-game frames and drops 0.24% of game
  frames (none of 5,877 spectator frames); coordinates are unchanged, the
  patch weights are the same. Details: [docs/not-game.md](docs/not-game.md).
- **Baseline.** `CoordNet` (`tools/coord_model.py`) maps the masked and
  cropped main view to a 32×32 heatmap over the map. It is kept as the
  baseline and still drives the live mode.
- **Labels without people.** The target is read from the camera box on the
  minimap of the same frame (`tools/minimap_camera.py`). The minimap is
  masked out of the model input, so it is the source of the target, not a
  feature. Each match yields thousands of labelled frames.
- **Data.** Spectator replays from the "Challenger Replays" YouTube network
  (`dataset/sources/`). `tools/build_coords.py` reads a video straight from
  the stream and writes the cropped scene, the minimap crop and the camera
  position for every sampled frame. The dense set is local (`data/coords/`,
  not in Git).
- **Inspection.** A local web app (`tools/serve_inspect.py` +
  `tools/webapp/`) shows frames, predictions and heatmaps of evaluated runs,
  steps through the network layer by layer, explains single misses, browses
  channels, collects new videos and runs the model live on a stream.

## Pipeline

```
YouTube channel catalog ─► build_coords.py ─► data/coords/<video>/{scene,mini}/, <video>.csv
                                                   │
                     calibrate_projection.py ◄─────┤ (screen → ground homography)
                     dataset/layouts/projection.json
                                                   ▼
                       train_patches.py / train_coords.py ─► runs/patches|coords/<run>/model.pt
                                                   ▼
                        probe_hud.py / eval_coords.py ─► robustness, predictions, heatmaps
                                                   ▼
                                  serve_inspect.py (web app, live mode on a stream)
```

Run the tools from the repository root with Python 3.12. Install the
dependencies from [requirements.txt](requirements.txt) (it explains how to
get a CUDA build of PyTorch) and put `ffmpeg`/`ffprobe` on `PATH`. A CUDA GPU
is used when present (training, NVDEC video decoding).

```bash
python -m pip install -r requirements.txt
```

```bash
# 1. Dense dataset from a stream (nothing is downloaded), 1 frame per second
python tools/build_coords.py --url ibUVbSX7ARU \
    --layout dataset/layouts/spectator-volibear-challenger.json --fps 1

# 2. Train; the split is by match, never by frame
python tools/train_coords.py --train 58w57eJ5Qks ibUVbSX7ARU \
    --test olmTXkkUv58 zJvTSjEnKNE --epochs 12 --out runs/coords/heatmap

# 3. Evaluate held-out matches, with a black-input and a shuffled-pairs control
python tools/eval_coords.py runs/coords/heatmap/model.pt olmTXkkUv58 zJvTSjEnKNE

# 4. Inspect: open http://127.0.0.1:8732
python tools/serve_inspect.py

# Patch model: train, check robustness to HUD changes, measure speed
python tools/train_patches.py --encoder cnn --train 58w57eJ5Qks ibUVbSX7ARU \
    --test olmTXkkUv58 zJvTSjEnKNE --epochs 40 --eval-every 4 --out runs/patches/cnn
python tools/probe_hud.py 58w57eJ5Qks runs/patches/cnn/model.pt --tag cnn
python tools/bench_models.py --coordnet runs/coords/heatmap/model.pt \
    --patch runs/patches/cnn/model.pt --out runs/checks/bench_models/bench.json
```

The full commands and the 16/4 split behind the reported numbers are in
[docs/patch-embeddings.md](docs/patch-embeddings.md).

Calibration of the screen → map projection, used by the viewer to draw the
visible area and the grid of cells:

```bash
python tools/calibrate_view.py zJvTSjEnKNE olmTXkkUv58          # how far from linear
python tools/calibrate_projection.py --fit 58w57eJ5Qks olmTXkkUv58 --check zJvTSjEnKNE
```

## Tools

| Tool | Purpose |
| --- | --- |
| `patch_model.py`, `train_patches.py` | Patch model: world-scale input, per-patch embeddings and map votes, camera by consensus; training with random crops, masks and synthetic UI |
| `probe_hud.py` | Robustness check of any coordinate model against mask, HUD, crop, resolution and partial-frame changes; share of frames the "not a game" head rejects |
| `build_not_game.py`, `train_not_game.py` | Frames of streams and other games from a YouTube stream; the "game / not a game" frame head on top of the frozen patch model |
| `bench_models.py` | Speed and memory of the models, GPU/CPU and the full live path |
| `coord_model.py` | `CoordNet` baseline: encoder + heatmap head, soft targets, expected/peak point, spread |
| `build_coords.py` | Dense dataset "frame → camera position" from a local file or a YouTube stream (`--url`) |
| `minimap_camera.py` | Camera box on the minimap → map coordinates; used by `build_coords.py` and `live.py`, and as a CLI over `dataset/videos/` |
| `mask_frames.py` | HUD/minimap mask and the crop of the unmasked area by layout |
| `train_coords.py` | Training; split by match or, with `--region`, by map area |
| `eval_coords.py` | Per-video errors, predictions and heatmaps for the viewer, sanity controls |
| `calibrate_view.py`, `calibrate_projection.py` | Measure the screen → map scale; fit the homography in `dataset/layouts/projection.json` |
| `serve_inspect.py`, `webapp/` | Local web viewer (standard library HTTP server, plain HTML/JS) |
| `activations.py`, `explain_miss.py` | Layer-by-layer activations and the report on a single miss, served by the viewer |
| `live.py` | Live mode: frames from a stream → model → viewer, nothing written to disk |
| `channels.py`, `video_side.py` | Channel catalog, video lists, ingest jobs; player role and side from the video description |
| `inspect_coords.py`, `inspect_patches.py`, `inspect_pages.py` | Self-contained HTML pages: predictions of one video; which patch size identifies a place |
| `extract_frames.py` | Full reference frames every N seconds into `dataset/videos/` |
| `paths.py`, `runlog.py` | Directory layout; `train.log` next to the checkpoint |

## Repository layout

- `dataset/` (committed): HUD layouts and the projection (`layouts/`), the
  channel catalog (`sources/`), manifests of reference frames of six
  matches (`videos/<id>/source.json`, `frames.csv`), time-interval labels
  "game / not a game" of streams (`not_game/`). The frames themselves
  are not distributed; they are rebuilt locally from the public videos. See
  [dataset/README.md](dataset/README.md).
- `data/` (ignored): local inputs. `data/videos/<id>.info.json` (video
  metadata, plus the video itself when processed from a file),
  `data/coords/` (the dense dataset), `data/channels/` (custom channels and
  the cached video lists).
- `runs/` (ignored): everything the tools produce.
- Directory layout is defined once in `tools/paths.py`; tool defaults
  resolve from the repository root.

### `runs/` layout

| Path | Contents | Written by |
| --- | --- | --- |
| `runs/coords/<run>/` | `model.pt`, `report.json`, `train.log`; after evaluation `predictions_<video>.csv`, `heatmaps_<video>.npz`, `eval_coords.json` | `train_coords.py`, `eval_coords.py` |
| `runs/patches/<run>/` | Patch models and the `CoordNet` baseline on the 16/4 split: `model.pt`, `report.json`, `train.log` | `train_patches.py` |
| `runs/checks/probe_hud/` | Robustness tables per video (JSON and log) and example inputs | `probe_hud.py` |
| `runs/checks/bench_models/` | Speed measurements | `bench_models.py` |
| `runs/inspect/` | Self-contained HTML pages with a shared `index.html`, miss reports | `inspect_coords.py`, `inspect_patches.py`, `explain_miss.py` |
| `runs/minimap/` | Camera position read from the minimap for `dataset/videos/` frames | `minimap_camera.py` |
| `runs/ingest/` | Logs of dataset builds started from the viewer | `channels.py` |
| `runs/checks/<topic>/` | One-off visual checks: masks, minimap box | `mask_frames.py`, ad hoc |

A run directory is self-contained: training writes its own `train.log` next
to the checkpoint, and evaluation writes into the checkpoint's directory by
default. The viewer lists every directory under `runs/` that holds
`predictions_*.csv`. Name runs by what distinguishes them and pass `--out`
for a new variant instead of overwriting an existing run.

## Limitations and open problems

- **Only one real HUD tested.** The patch model is robust to simulated HUD
  changes, but all videos come from one channel family with one layout and
  one homography (spectator, 1920×1080, default zoom). A different channel
  or zoom level has not been tried. `CoordNet` is tied to the layout
  (removing temporary masks ×3 median error, black zones ×10, a ±8% crop
  ×4–9).
- **"Not a game" is trained on few streams.** The frame head saw one
  streamer's player view (Nemesis) and two other games; another streamer's
  player view loses 4% of game frames, ARAM (a different map) is rejected
  as not a game, and a client screen that shows the map preview passes as
  game.
- **One verified layout.** Only `spectator-volibear-challenger` (16:9) has
  been checked by eye; the mask scales with the frame but rejects other
  aspect ratios. Every new channel needs its own visual check.
- **Map symmetry.** Top and bottom of Summoner's Rift look alike; the
  heatmap may show two peaks, and the expected point then falls between
  them.
- **Single frames only.** There is no temporal model yet; the patch
  embeddings are the intended input for one.
- **Labels from the minimap.** Frames with a weak camera-box detection are
  dropped by `--min-quality`; the box gives the visible area only
  approximately, which is why the projection is calibrated separately.
- **Live mode cannot seek.** YouTube throttles anything but sequential
  reading, so starting later in a video means decoding up to that point.

## Later scope

Patch model in the live mode, a second channel layout. Then locating the player's champion, identifying champions and objects, reading
HP/mana, and tracking state across frames. An SNN comparison may be tried
once a model transfers across layouts.

## References

A survey of GitHub projects and papers on ML for League of Legends (plus
TFT and Wild Rift): [docs/github-lol-ml-index.md](docs/github-lol-ml-index.md).

[DeeperLeague](https://github.com/davidweatherall/DeeperLeague) remains a
reference for a possible minimap module. Its notice offers GPL-3.0 or a
separately obtained closed-source license
([upstream notice](https://github.com/davidweatherall/DeeperLeague/blob/main/LICENSE.md),
inspected 2026-10-06 UTC). No upstream source code, assets or model weights
have been imported here.

## Conventions

- Keep source videos, frames and other bulky local data out of Git:
  `data/` and `dataset/videos/*/frames/` are ignored. Layouts and manifests
  live in `dataset/` and are committed.
- Keep generated artifacts, model weights and experiment outputs outside
  Git; common paths and formats are covered by `.gitignore`.
- Commit source code, configuration and documentation as the
  implementation is added.

## License

The code and the project's own annotations are licensed under the
[Apache License 2.0](LICENSE). Third-party content and dependencies are
described in [NOTICE.md](NOTICE.md); no video frames or footage are
distributed with this repository.

## Legal

LoL Game State Vision was created under Riot Games' "Legal Jibber Jabber"
policy using assets owned by Riot Games. Riot Games does not endorse or
sponsor this project. League of Legends is a trademark of Riot Games, Inc.;
this project is not affiliated with Riot Games.

Questions, problems and removal requests from the authors of referenced
videos: open an issue.

# LoL Game State Vision

Reading game state from League of Legends gameplay videos, using only the
main game view. The current task: tell where the camera is on the map from
a single frame, with the HUD and the minimap masked out.

## Current state

- **Model.** `CoordNet` (`tools/coord_model.py`) is a small CNN that maps
  the masked and cropped main view to a 32×32 heatmap over the map. The
  answer is the expected point of the heatmap; its spread shows how sure the
  model is. A heatmap rather than two numbers, because the map is nearly
  symmetric and a single frame can fit two places.
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
- **In progress (#740).** CoordNet turned out to depend on the HUD mask, the
  crop and the position on screen (see [Limitations](#limitations-and-open-problems)).
  The next model is built from patch embeddings that do not depend on the
  HUD, the resolution or the position in the frame. It is developed in the
  `feat/patch-embeddings` branch; its results will go to
  `docs/patch-embeddings.md` (planned, not yet in this branch).

## Pipeline

```
YouTube channel catalog ─► build_coords.py ─► data/coords/<video>/{scene,mini}/, <video>.csv
                                                   │
                     calibrate_projection.py ◄─────┤ (screen → ground homography)
                     dataset/layouts/projection.json
                                                   ▼
                                          train_coords.py ─► runs/coords/<run>/model.pt
                                                   ▼
                                          eval_coords.py ─► predictions, heatmaps, controls
                                                   ▼
                                  serve_inspect.py (web app, live mode on a stream)
```

Run the tools from the repository root. They need Python 3.12 with
`torch`, `numpy`, `Pillow`, `opencv-python`, `scipy` and `yt-dlp`, and
`ffmpeg`/`ffprobe` on `PATH`. A CUDA GPU is used when present (training,
video decoding). There is no pinned requirements file yet.

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
```

Calibration of the screen → map projection, used by the viewer to draw the
visible area and the grid of cells:

```bash
python tools/calibrate_view.py zJvTSjEnKNE olmTXkkUv58          # how far from linear
python tools/calibrate_projection.py --fit 58w57eJ5Qks olmTXkkUv58 --check zJvTSjEnKNE
```

## Tools

| Tool | Purpose |
| --- | --- |
| `coord_model.py` | `CoordNet`: encoder + heatmap head, soft targets, expected/peak point, spread |
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
  channel catalog (`sources/`), full reference frames of six matches
  (`videos/`). See [dataset/README.md](dataset/README.md).
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

- **CoordNet relies on the HUD layout.** An experiment on 2026-10-09 showed
  that the model is tied to the mask, the crop and the position on screen:
  removing temporary masks raised the median error ×3, black zones of
  another layout ×10, shifting the crop by ±8% ×4–9. A model trained on
  one layout cannot be expected to transfer to another channel, HUD scale
  or resolution. This is the reason for the patch-embedding model (#740).
- **One verified layout.** Only `spectator-volibear-challenger` (16:9) has
  been checked by eye; the mask scales with the frame but rejects other
  aspect ratios. Every new channel needs its own visual check.
- **Map symmetry.** Top and bottom of Summoner's Rift look alike; the
  heatmap may show two peaks, and the expected point then falls between
  them.
- **Single frames only.** There is no temporal model yet; `CoordNet.embed`
  exposes features for one.
- **Labels from the minimap.** Frames with a weak camera-box detection are
  dropped by `--min-quality`; the box gives the visible area only
  approximately, which is why the projection is calibrated separately.
- **Live mode cannot seek.** YouTube throttles anything but sequential
  reading, so starting later in a video means decoding up to that point.

## Later scope

Locating the player's champion, identifying champions and objects, reading
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

- Keep source videos and other bulky local data in `data/` (ignored).
  Reference frames and layouts live in `dataset/` and are committed.
- Keep generated artifacts, model weights and experiment outputs outside
  Git; common paths and formats are covered by `.gitignore`.
- Commit source code, configuration and documentation as the
  implementation is added.

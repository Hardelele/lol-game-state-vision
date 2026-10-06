# LoL Game State Vision

Recognize the player's position and, eventually, other visible game state in League of Legends from screenshots.

## Starting approach

Build this repository as an independent project. Use [DeeperLeague](https://github.com/davidweatherall/DeeperLeague) as the starting reference for synthetic minimap data and as a possible detection baseline or teacher.

The first experiment is:

1. Generate labelled minimap images using the DeeperLeague approach.
2. Adapt the labels into a dataset for locating a target player.
3. Train a small, task-specific CNN from scratch.
4. Measure localization quality, end-to-end latency, parameter count, FLOPs, and memory/VRAM usage.
5. Compare a spiking neural network against that baseline under the same evaluation conditions.

The first model consumes a **cropped minimap**, with the intended output being the target player's **position on that minimap**. Converting that position to game-world coordinates is a separate step. Full-screen capture and minimap cropping can be added around the model later.

## First implementation milestone: the dataset

Inspect and adapt the generator before selecting the final model architecture.

Useful upstream files:

- [`generateTestingData.py`](https://github.com/davidweatherall/DeeperLeague/blob/main/generateTestingData.py): synthetic images and annotations.
- [`champMap.json`](https://github.com/davidweatherall/DeeperLeague/blob/main/champMap.json): mapping between annotation classes and champions.
- [`assets/`](https://github.com/davidweatherall/DeeperLeague/tree/main/assets) and [`champions/`](https://github.com/davidweatherall/DeeperLeague/tree/main/champions): minimap backgrounds, effects, and champion icons.

The existing generator produces multiple champion annotations per minimap in YOLO format: `class_id center_x center_y width height`, with normalized coordinates. It does not directly provide a single target-player label. The dataset adapter must make target selection explicit and define how missing or occluded targets are represented.

Start with a small, inspectable sample. The upstream script defaults to 300,000 images and uses all available CPU cores; expose sample count and worker count before running it as part of this project.

Before training:

- Define the target-player selection rule and the dataset input/output contract.
- Inspect generated images and their labels together.
- Keep a separate set of real minimap screenshots to assess transfer from synthetic data.
- Make generation and dataset splits reproducible.

## Model direction and evaluation

Use a compact CNN as the first custom model. DeeperLeague's YOLOv8 pipeline can serve as a reference, baseline, or teacher; the project's own architecture is still to be implemented.

Evaluate SNNs after the CNN baseline is measurable. Compare both localization quality and latency on the same target hardware; faster inference is a hypothesis to test.

## Later scope

Expand incrementally toward champion identity, HP/mana, bushes, other objects, and state across frames. Each capability may use its own small model and appropriate screenshot region.

pyLoL is a reference for the eventual structured game-state output, including position, champion identity, and timestamp. LeagueAI and lol-vision are secondary references for individual ideas.

## Upstream status

The DeeperLeague README, generator, dependency list, and license notice were inspected on 2026-10-06 (UTC). Its license notice offers GPL-3.0 or a separately obtained closed-source license; see [the upstream notice](https://github.com/davidweatherall/DeeperLeague/blob/main/LICENSE.md). No upstream source code, assets, or model weights have been imported into this repository yet.

## Repository status and conventions

This repository currently contains the project plan and basic repository configuration. The dataset adapter, model, and inference pipeline are not implemented yet.

- Keep local screenshots, recordings, and datasets in `data/` or `datasets/`.
- Keep generated artifacts, model weights, and experiment outputs outside Git; common paths and formats are covered by `.gitignore`.
- Commit source code, configuration, and documentation as the implementation is added.

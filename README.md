# LoL Game State Vision

Recognize the player's position and, eventually, other visible game state in League of Legends from screenshots.

## Initial focus

Start with player localization from a screenshot, then expand in small, measurable steps toward extracting additional game state and statistics.

The research direction is fast, compact, task-specific models. Spiking neural networks are an option to investigate; the model architecture, framework, and any upstream project to reuse have not been selected yet.

## Status

Repository initialized. There is no trained model, dataset, or runnable inference pipeline yet.

## Next decisions

- Define the first prediction target: position in the game viewport or position on the map.
- Select a small set of representative screenshots and define the labels.
- Establish an initial baseline and measure accuracy and end-to-end latency on the target hardware.

## Repository conventions

- Keep local screenshots, recordings, and datasets in `data/` or `datasets/`.
- Keep generated artifacts, model weights, and experiment outputs outside Git; common paths and formats are covered by `.gitignore`.
- Commit source code, configuration, and documentation as the implementation is added.

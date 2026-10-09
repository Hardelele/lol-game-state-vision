# Notices and third-party content

## What the license covers

Copyright 2026 Vladislav Lapshin.

[LICENSE](LICENSE) (Apache License 2.0) covers the source code in `tools/` and
the project's own annotations and metadata in `dataset/`: notes in
`frames.csv`, `source.json`, HUD layouts in `dataset/layouts/` and the channel
catalogue in `dataset/sources/`.

## Video frames are not distributed

The repository contains no frames or other footage from third-party videos.
`dataset/videos/<id>/` keeps only the manifest — the source video ID,
sampling parameters and timestamps — and the frames are rebuilt locally from
the public video (see `dataset/README.md`). Rights to that footage belong to
Riot Games and to the authors of the videos listed in each `source.json`.
If you hold rights to any material referenced here and want it removed, open
an issue.

## Riot Games

LoL Game State Vision was created under Riot Games' "Legal Jibber Jabber"
policy using assets owned by Riot Games. Riot Games does not endorse or
sponsor this project.

League of Legends and Riot Games are trademarks or registered trademarks of
Riot Games, Inc. This project is not affiliated with Riot Games.
See <https://www.riotgames.com/en/legal>.

## YouTube

The tools can read public YouTube videos through yt-dlp. Whoever runs them is
responsible for complying with YouTube's Terms of Service and with the rights
of the video authors. No videos are distributed with this repository.

## Dependencies

Installed separately (see [requirements.txt](requirements.txt)), not
vendored:

| Package | License |
| --- | --- |
| PyTorch | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| Pillow | MIT-CMU |
| opencv-python-headless | Apache-2.0 |
| SciPy | BSD-3-Clause |
| yt-dlp | Unlicense |
| FFmpeg (external program, called as a subprocess) | LGPL-2.1+ / GPL depending on build |

## References

[DeeperLeague](https://github.com/davidweatherall/DeeperLeague) (GPL-3.0 or
commercial) was studied as a reference. No code, assets or model weights from
it are included.

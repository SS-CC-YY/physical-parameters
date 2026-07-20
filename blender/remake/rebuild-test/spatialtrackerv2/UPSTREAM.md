# Upstream provenance

- Repository: `https://github.com/henry123-boy/SpaTrackerV2`
- Commit: `7e12274c52077860cebfe007a6290777db43b63c`
- License: see `upstream/SpaTrackerV2/LICENSE.txt`
- Snapshot date: 2026-07-20

The `examples` Git submodule is intentionally not included because the V1A Seedance videos are the test inputs.

Local changes to `upstream/SpaTrackerV2/inference.py` are limited to benchmark integration:

- accept explicit MP4, query NPZ and output paths;
- convert query pixels through the upstream 518-pixel preprocessing;
- preserve full-resolution 2D tracks, confidence, frame indices and query metadata;
- allow disabling the upstream visualization for faster batch execution.

The benchmark's default runner uses `scripts/spatialtracker_session.py`, an RGB-only extraction of the same official inference path, so the two networks are loaded once for all 27 videos. The patched upstream CLI remains available through `--isolated-process` for failure isolation.

Model weights remain unmodified and are downloaded from:

- `Yuxihenry/SpatialTrackerV2_Front`
- `Yuxihenry/SpatialTrackerV2-Offline`

They are cached under `models/huggingface` and excluded from Git.

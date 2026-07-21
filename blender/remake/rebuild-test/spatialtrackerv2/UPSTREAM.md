# Upstream provenance

- Repository: `https://github.com/henry123-boy/SpaTrackerV2`
- Commit: `7e12274c52077860cebfe007a6290777db43b63c`
- License: see `upstream/SpaTrackerV2/LICENSE.txt`
- Snapshot date: 2026-07-20

The `examples` Git submodule is intentionally not included because the V1A Seedance videos are the test inputs. All tracked files under `upstream/SpaTrackerV2` otherwise remain byte-for-byte identical to the official commit.

Benchmark-specific integration lives outside the upstream tree:

- `scripts/spatialtracker_session.py` follows the official RGB inference path but accepts explicit query pixels, preserves full-resolution 2D tracks and keeps both networks resident across the 27 videos;
- `scripts/run_single_inference.py` provides the same adapter in a fresh process for failure isolation;
- query construction, Blender-metric alignment and visualisation remain separate post-processing stages.

Model weights remain unmodified and are downloaded from:

- `Yuxihenry/SpatialTrackerV2_Front`
- `Yuxihenry/SpatialTrackerV2-Offline`

They are cached under `models/huggingface` and excluded from Git.

The inference environment does not use the upstream Gradio, Segment Anything, or Ray interfaces. `EasternJournalist/utils3d` is vendored at commit `d3a577acf0a9ad7e513a1416449a07b6f47d967f` under `third_party/utils3d` because it is a required core geometry dependency and a Git URL dependency is unreliable on the target server.

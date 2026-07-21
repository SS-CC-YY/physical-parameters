from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REMAKE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REMAKE_ROOT / "rebuild-test" / "spatialtrackerv2" / "scripts"

try:
    import cv2  # noqa: F401
    import numpy as np

    def _load_script(name: str):
        spec = importlib.util.spec_from_file_location(name, SCRIPT_ROOT / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    prepare = _load_script("prepare_queries")
    postprocess = _load_script("postprocess_result")
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/opencv dependencies unavailable")
class SpatialTrackerGeneralizedTests(unittest.TestCase):
    def test_normalizes_all_experiment_sidecar_schema(self) -> None:
        calibration = {
            "source_image": {"width_px": 1280, "height_px": 720},
            "camera": {"K": np.eye(3).tolist()},
            "standard_ball": {
                "object_id": "standard_ball",
                "center_world_m": [1.0, 2.0, 3.0],
                "radius_m": 0.24,
            },
            "frame1_projection": {
                "center_uv_px": [640.0, 200.0],
                "mesh_vertex_bbox_xyxy_px": [620.0, 180.0, 660.0, 220.0],
            },
            # The all-experiment exporter also has this diagnostic under the
            # old key, but with intentionally different field names.
            "projection_validation": {"center_uv_from_opencv_P": [640.0, 200.0]},
        }
        normalized = prepare.normalize_calibration(calibration)
        self.assertEqual(normalized["image"]["width_px"], 1280)
        self.assertEqual(normalized["object"]["initial_center_world_m"], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(normalized["object"]["known_radius_m"], 0.24)
        self.assertEqual(
            normalized["projection_validation"]["initial_center_uv_from_opencv_P"],
            [640.0, 200.0],
        )

    def test_self_frame0_alignment_recovers_metric_translation(self) -> None:
        frames = 8
        object_points = 20
        anchor_points = 24
        radius = 0.24
        center = np.asarray([0.0, 0.0, 1.5])
        rng = np.random.default_rng(7)
        directions = rng.normal(size=(object_points, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        object_reference = center + radius * directions
        anchors = rng.uniform([-3.0, -2.0, 0.0], [3.0, 2.0, 4.0], size=(anchor_points, 3))
        query_kind = np.asarray(["object"] * object_points + ["anchor"] * anchor_points)
        metric_frames = []
        raw_frames = []
        for frame in range(frames):
            translation = np.asarray([0.10 * frame, 0.02 * frame, -0.04 * frame])
            metric = np.concatenate([object_reference + translation, anchors], axis=0)
            # A changing global similarity stands in for arbitrary predicted
            # coordinate gauge/camera drift.  The physical scene itself stays rigid.
            scale = 1.8 + 0.04 * frame
            gauge_translation = np.asarray([1.2 + 0.08 * frame, -0.7, 0.4 - 0.03 * frame])
            raw_frames.append((metric - gauge_translation) / scale)
            metric_frames.append(metric)
        coords = np.asarray(raw_frames)
        points = object_points + anchor_points
        anchor_target = np.full((points, 3), np.nan)
        object_target = np.full((points, 3), np.nan)
        object_target[:object_points] = object_reference
        query_uv = np.column_stack(
            [np.linspace(30.0, 610.0, points), np.mod(np.arange(points) * 53.0, 330.0) + 15.0]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.npz"
            np.savez(
                raw_path,
                coords=coords,
                query_meta_query_kind=query_kind,
                query_meta_anchor_xyz_world_m=anchor_target,
                query_meta_object_xyz_world_initial_m=object_target,
                query_meta_object_initial_center_world_m=center,
                query_meta_known_radius_m=np.asarray(radius),
                query_meta_anchor_reference_mode=np.asarray("self_frame0"),
                query_meta_query_xy_video=query_uv,
                query_meta_video_frame_size_hw=np.asarray([360, 640]),
                visibs=np.ones((frames, points), dtype=bool),
                track_confidence=np.ones((frames, points), dtype=float),
                source_frame_indices=np.arange(frames),
                source_fps=np.asarray(24.0),
            )
            output = root / "result"
            with mock.patch.object(postprocess, "make_plot"), mock.patch.object(
                postprocess, "make_overlay"
            ):
                summary = postprocess.postprocess(raw_path, root / "unused.mp4", output, 0.12)

            self.assertEqual(summary["anchor_reference_mode"], "self_frame0")
            self.assertGreater(summary["trajectory_valid_fraction"], 0.99)
            self.assertEqual(summary["generation_validity"]["status"], "pass")
            self.assertFalse(summary["generation_validity"]["skip_physics_fit"])
            with (output / "trajectory_world.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[-1]["x_raw_m"]), 0.70, places=4)
            self.assertAlmostEqual(float(rows[-1]["z_raw_m"]), 1.22, places=4)


if __name__ == "__main__":
    unittest.main()

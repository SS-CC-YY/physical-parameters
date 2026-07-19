from __future__ import annotations

import sys
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

try:
    import numpy as np

    from remake_benchmark.reconstruction.mvp import (
        DEFAULT_CONFIG,
        _job_worker,
        _mark_identity_and_lift,
        _relative_artifacts,
        _valid_components,
        fit_freefall_physics,
        parse_video_job,
        run_reconstruction_batch,
    )

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@unittest.skipUnless(HAS_NUMPY, "reconstruction dependencies are not installed")
class ReconstructionMvpTests(unittest.TestCase):
    def test_parse_seedance_filename(self) -> None:
        path = Path("v1_A__gravity_g-9p81__indoor3__standard_ball__CAM_Side__seed-341867882.mp4")
        job = parse_video_job(path)
        self.assertEqual(job["experiment_id"], "v1_A")
        self.assertEqual(job["factors"]["scene_id"], "indoor3")
        self.assertEqual(job["factors"]["camera"], "CAM_Side")
        self.assertEqual(job["factors"]["seed"], 341867882)
        self.assertAlmostEqual(job["targets"]["gravity_g"], 9.81)

    def _synthetic_trajectory(self, fps: float, gravity: float) -> tuple[list[dict], dict]:
        pixels_per_meter = 80.0
        duration = 1.0
        rows = []
        for index in range(int(round(duration * fps)) + 1):
            time_s = index / fps
            rows.append(
                {
                    "frame_index": index,
                    "time_s": time_s,
                    "measurement_valid": True,
                    "primary_trajectory": True,
                    "center_y_stabilized_px": 80.0 + 0.5 * gravity * pixels_per_meter * time_s**2,
                    "physics_fit_used": False,
                }
            )
        calibration = {
            "observed_object_diameter_px": 32.0,
            "pixels_per_meter": pixels_per_meter,
            "drop_distance_m": 3.76,
        }
        return rows, calibration

    def test_native_fps_does_not_change_recovered_gravity(self) -> None:
        config = deepcopy(DEFAULT_CONFIG)
        config["physics"]["contact_drop_fraction"] = 2.0
        for fps in (16.0, 24.0):
            trajectory, calibration = self._synthetic_trajectory(fps, 9.81)
            result = fit_freefall_physics(trajectory, 9.81, calibration, config)
            self.assertTrue(result["fit_valid"])
            self.assertAlmostEqual(result["estimated_gravity_m_s2"], 9.81, places=6)
            self.assertGreater(result["parameter_similarity"], 0.999)

    def test_delayed_release_and_contact_are_stable_across_fps(self) -> None:
        config = deepcopy(DEFAULT_CONFIG)
        estimates = []
        gravity = 9.81
        pixels_per_meter = 80.0
        release_s = 0.30
        drop_m = 3.76
        contact_s = release_s + np.sqrt(2.0 * drop_m / gravity)
        for fps in (16.0, 24.0):
            rows = []
            for index in range(int(round(1.6 * fps)) + 1):
                time_s = index / fps
                falling_s = max(0.0, min(time_s, contact_s) - release_s)
                displacement_m = min(drop_m, 0.5 * gravity * falling_s**2)
                rows.append(
                    {
                        "frame_index": index,
                        "time_s": time_s,
                        "measurement_valid": True,
                        "primary_trajectory": True,
                        "center_y_stabilized_px": 80.0 + pixels_per_meter * displacement_m,
                        "physics_fit_used": False,
                    }
                )
            calibration = {
                "observed_object_diameter_px": 32.0,
                "pixels_per_meter": pixels_per_meter,
                "drop_distance_m": drop_m,
            }
            result = fit_freefall_physics(rows, gravity, calibration, config)
            self.assertTrue(result["fit_valid"])
            estimates.append(result["estimated_gravity_m_s2"])
        self.assertAlmostEqual(estimates[0], estimates[1], delta=0.35)
        for estimate in estimates:
            self.assertAlmostEqual(estimate, gravity, delta=0.75)

    def test_physics_failure_is_not_a_reconstruction_failure_field(self) -> None:
        config = deepcopy(DEFAULT_CONFIG)
        config["physics"]["contact_drop_fraction"] = 2.0
        trajectory, calibration = self._synthetic_trajectory(24.0, 3.0)
        result = fit_freefall_physics(trajectory, 9.81, calibration, config)
        self.assertTrue(result["fit_valid"])
        self.assertEqual(result["status"], "fail")
        self.assertLess(result["parameter_similarity"], 0.5)

    def test_full_inverse_affine_stabilizes_static_center(self) -> None:
        angle = np.deg2rad(8.0)
        scale = 1.08
        a, b = scale * np.cos(angle), -scale * np.sin(angle)
        c, d = scale * np.sin(angle), scale * np.cos(angle)
        tx, ty = 7.0, -4.0
        reference = np.asarray([100.0, 60.0])
        current = np.asarray([a * reference[0] + b * reference[1] + tx, c * reference[0] + d * reference[1] + ty])
        tracks = []
        centers = [reference] * 5 + [current]
        for index, center in enumerate(centers):
            tracks.append(
                {
                    "frame_index": index,
                    "time_s": index / 24.0,
                    "found": True,
                    "tracking_source": "template+mask",
                    "center_x_px": float(center[0]),
                    "center_y_px": float(center[1]),
                    "bbox_width_px": 20.0,
                    "bbox_height_px": 20.0,
                }
            )
        camera_rows = [
            {"frame_index": index, "success": True, "tx_px": 0.0, "ty_px": 0.0, "affine_a": 1.0, "affine_b": 0.0, "affine_c": 0.0, "affine_d": 1.0}
            for index in range(5)
        ]
        camera_rows.append(
            {"frame_index": 5, "success": True, "tx_px": tx, "ty_px": ty, "affine_a": a, "affine_b": b, "affine_c": c, "affine_d": d}
        )
        lifted, _, _ = _mark_identity_and_lift(tracks, camera_rows, deepcopy(DEFAULT_CONFIG))
        self.assertAlmostEqual(lifted[5]["center_x_stabilized_px"], reference[0], places=6)
        self.assertAlmostEqual(lifted[5]["center_y_stabilized_px"], reference[1], places=6)

    def test_primary_component_does_not_join_large_track_fragments(self) -> None:
        values = [True] * 12 + [False] * 4 + [True] * 30
        components = _valid_components(values, max_bridge_gap=2)
        self.assertEqual([len(component) for component in components], [12, 30])
        self.assertEqual(components[0][-1], 11)
        self.assertEqual(components[1][0], 16)

    def test_failed_retry_overwrites_stale_success_result(self) -> None:
        with tempfile.TemporaryDirectory(dir=CODE_ROOT / "tests") as temporary:
            root = Path(temporary)
            job_id = "v1_A__gravity_g-2__baseline__standard_ball__CAM_Side__seed-1"
            job_dir = root / "output" / job_id
            job_dir.mkdir(parents=True)
            (job_dir / "result.json").write_text('{"status":"old_success"}', encoding="utf-8")
            job = {
                "job_id": job_id,
                "video_path": str(root / "missing.mp4"),
                "experiment_id": "v1_A",
                "factors": {"scene_id": "baseline", "object_id": "standard_ball", "camera": "CAM_Side"},
                "targets": {"gravity_g": 2.0},
            }
            result = _job_worker(
                {
                    "job": job,
                    "workspace_root": str(root),
                    "output_root": str(root / "output"),
                    "config": deepcopy(DEFAULT_CONFIG),
                    "make_overlay": False,
                }
            )
            stored = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
            self.assertIn("error", result)
            self.assertEqual(stored["error"], result["error"])
            self.assertNotIn("old_success", stored.values())

    def test_batch_lock_rejects_concurrent_output_writer(self) -> None:
        with tempfile.TemporaryDirectory(dir=CODE_ROOT / "tests") as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            (output / ".reconstruction_mvp.lock").write_text("occupied\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                run_reconstruction_batch(root, root / "videos", output)

    def test_artifact_contract_is_portable_relative_to_result(self) -> None:
        artifacts = _relative_artifacts(Path("ignored"), "/machine/specific/path/track_overlay.mp4")
        self.assertEqual(artifacts["path_base"], "result_json_directory")
        self.assertEqual(artifacts["trajectory_world_csv"], "trajectory_world.csv")
        self.assertEqual(artifacts["overlay_video"], "track_overlay.mp4")
        self.assertFalse(Path(artifacts["trajectory_world_csv"]).is_absolute())


if __name__ == "__main__":
    unittest.main()

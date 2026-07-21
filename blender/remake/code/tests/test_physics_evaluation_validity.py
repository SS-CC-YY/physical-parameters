from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from remake_benchmark.reconstruction.physics_evaluation import (
    _is_trusted_measurement,
    aggregate_results,
    benchmark_split,
    run_physics_job,
)


def _registry() -> dict:
    experiment = {
        "id": "v1_A",
        "hidden_parameters": [
            {"name": "gravity_g", "unit": "m/s^2", "valid_range": [2.0, 14.7]}
        ],
        "parameter_tuples": [{"id": "g9p81", "values": {"gravity_g": 9.81}}],
    }
    return {"experiments": [experiment], "experiments_by_id": {"v1_A": experiment}}


def _tracks(deformed: bool) -> list[dict]:
    rows = []
    for frame in range(12):
        bad = deformed and 4 <= frame <= 7
        rows.append(
            {
                "frame_index": frame,
                "found": True,
                "center_u_px": 20.0,
                "center_v_px": 20.0 + frame,
                "measurement_radius_px": 8.0,
                "track_confidence": 0.95,
                "candidate_count": 1,
                "circularity": 0.40 if bad else 0.93,
                "ellipse_axis_ratio": 1.95 if bad else 1.03,
                "touches_frame_boundary": False,
            }
        )
    return rows


class PhysicsEvaluationValidityTests(unittest.TestCase):
    def test_overlay_and_fit_history_reject_unverified_or_predicted_points(self) -> None:
        base = {
            "found": True,
            "observation_status": "measured",
            "measurement_valid": True,
            "identity_verified": True,
        }
        self.assertTrue(_is_trusted_measurement(base))
        self.assertFalse(
            _is_trusted_measurement({**base, "observation_status": "predicted"})
        )
        self.assertFalse(_is_trusted_measurement({**base, "identity_verified": False}))
        self.assertFalse(_is_trusted_measurement({**base, "measurement_valid": False}))

    def test_split_labels_keep_extra_seeds_out_of_headline(self) -> None:
        base = {
            "camera_name": "CAM_Side",
            "seed": 341867882,
        }
        self.assertEqual(benchmark_split(base), "side_primary")
        self.assertEqual(
            benchmark_split({**base, "seed": 1750912582}),
            "side_seed_stability_extra",
        )
        self.assertEqual(
            benchmark_split({**base, "camera_name": "CAM_Main"}),
            "main_robustness",
        )

    def test_obvious_deformation_never_calls_physics_fitter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            video = output / "v1_A__g9p81__baseline__standard_ball__CAM_Side__seed-341867882.mp4"
            with (
                mock.patch(
                    "remake_benchmark.reconstruction.physics_evaluation.load_calibration",
                    return_value={},
                ),
                mock.patch(
                    "remake_benchmark.reconstruction.physics_evaluation.run_ball_tracking",
                    return_value=(
                        [],
                        _tracks(True),
                        {"video": {"fps": 24.0, "width": 64, "height": 64}, "tracking": {"tracked_fraction": 1.0}},
                    ),
                ),
                mock.patch(
                    "remake_benchmark.reconstruction.physics_evaluation.fit_physics_parameters"
                ) as fitter,
            ):
                result = run_physics_job(
                    video,
                    calibration_root=output / "calibration",
                    registry=_registry(),
                    output_root=output / "results",
                    camera_motion_evidence={"final_category": "no_significant_camera_change"},
                )
            self.assertEqual(result["status"], "generation_validity_failed")
            self.assertFalse(result["fit_attempted"])
            self.assertFalse(result["target_lookup_performed"])
            self.assertEqual(result["fit"]["status"], "skipped_video_generation_validity")
            fitter.assert_not_called()

    def test_aggregate_separates_validity_coverage_from_conditional_accuracy(self) -> None:
        def item(status, nmae):
            return {
                "job": {"experiment_id": "v1_A"},
                "video_generation_validity": {"status": status},
                "fit_attempted": status == "pass",
                "metrics": {
                    "experiment_nmae": nmae,
                    "experiment_score_0_100": None if nmae is None else 100 * (1 - nmae),
                    "fit_complete": nmae is not None,
                },
                "pipeline": {"tracking": {"tracked_fraction": 1.0}},
            }

        aggregate = aggregate_results([item("pass", 0.2), item("fail", None)])
        self.assertEqual(aggregate["generation_valid_rate"], 0.5)
        self.assertAlmostEqual(aggregate["conditional_macro_nmae_equal_experiment_weight"], 0.2)
        self.assertEqual(aggregate["generation_validity_counts"], {"pass": 1, "fail": 1})


if __name__ == "__main__":
    unittest.main()

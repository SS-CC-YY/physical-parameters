from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from export_generation_task_pack import (  # noqa: E402
    BUNDLE_VALIDATOR,
    EXPECTED_BUILD_ID,
    MODEL_RUNNER,
    PackExportError,
    validate_frozen_jobs,
)


EXPERIMENT_TUPLE_COUNTS = {
    "v1_A": 3,
    "v1_B": 3,
    "v1_C": 3,
    "v1_D": 3,
    "v2_A": 3,
    "v2_B": 3,
    "v2_C": 4,
    "v2_D": 5,
    "v2_E": 5,
    "v3_A": 4,
    "v3_B": 4,
    "v3_C": 4,
    "v3_D": 4,
}
SCENES = [
    "baseline",
    "indoor1",
    "indoor2",
    "indoor3",
    "indoor4",
    "outdoor1",
    "outdoor2",
    "outdoor3",
    "outdoor4",
]


def make_job(
    experiment: str,
    tuple_id: str,
    scene: str,
    camera: str,
    seed: int,
) -> dict:
    job_id = f"{experiment}__{tuple_id}__{scene}__standard_ball__{camera}__seed-{seed}"
    return {
        "job_id": job_id,
        "build_id": EXPECTED_BUILD_ID,
        "experiment_id": experiment,
        "task_type": "i2v",
        "seed": seed,
        "inputs": {
            "image": f"first_frames_v1_0/images/{experiment}/{scene}/standard_ball/{camera}.png"
        },
        "factors": {
            "parameter_tuple_id": tuple_id,
            "scene_id": scene,
            "object_id": "standard_ball",
            "camera": camera,
        },
        "generation": {"width": 832, "height": 480, "num_frames": 81, "fps": 16.0},
    }


def frozen_fixture() -> list[dict]:
    jobs: list[dict] = []
    for experiment, tuple_count in EXPERIMENT_TUPLE_COUNTS.items():
        tuple_ids = [f"p{index}" for index in range(tuple_count)]
        for tuple_id in tuple_ids:
            for scene in SCENES:
                jobs.append(make_job(experiment, tuple_id, scene, "CAM_Side", 341867882))
        for tuple_id in tuple_ids[:2]:
            for scene in SCENES:
                for camera in ("CAM_Main", "CAM_Top"):
                    jobs.append(make_job(experiment, tuple_id, scene, camera, 341867882))
        for tuple_id in tuple_ids[:2]:
            for seed in (1750912582, 265635392, 135883006):
                jobs.append(make_job(experiment, tuple_id, "baseline", "CAM_Side", seed))
    return jobs


class ExportGenerationTaskPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.jobs = frozen_fixture()

    def test_exact_frozen_contract_is_accepted(self) -> None:
        summary = validate_frozen_jobs(self.jobs)
        self.assertEqual(summary["jobs"], 978)
        self.assertEqual(summary["unique_input_images"], 351)
        self.assertEqual(summary["parameter_tuples"], 48)
        self.assertEqual(
            summary["track_counts"],
            {
                "physics_identification_side": 432,
                "random_seed_stability_audit": 78,
                "viewpoint_robustness_main_top": 468,
            },
        )

    def test_manifest_image_cannot_escape_bundle(self) -> None:
        jobs = list(self.jobs)
        bad = dict(jobs[0])
        bad["inputs"] = {"image": "../outside.png"}
        jobs[0] = bad
        with self.assertRaisesRegex(PackExportError, "unsafe inputs.image"):
            validate_frozen_jobs(jobs)

    def test_generated_standard_library_helpers_compile(self) -> None:
        compile(MODEL_RUNNER, "model_runner_template/run_model.py", "exec")
        compile(BUNDLE_VALIDATOR, "tools/validate_bundle.py", "exec")


if __name__ == "__main__":
    unittest.main()

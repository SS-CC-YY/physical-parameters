from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.core.build import resolve_build  # noqa: E402
from remake_benchmark.core.errors import JobValidationError  # noqa: E402
from remake_benchmark.core.manifest import build_jobs  # noqa: E402
from remake_benchmark.core.schema import validate_job  # noqa: E402
from remake_benchmark.models import get_adapter  # noqa: E402
from remake_benchmark.evaluators import evaluate_video  # noqa: E402
from remake_benchmark.orchestration.evaluate import _overlay_job_ids  # noqa: E402


class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = resolve_build(CODE_ROOT / "builds" / "v1a_wan22_demo.yaml", WORKSPACE_ROOT)
        cls.full_resolved = resolve_build(
            CODE_ROOT / "builds" / "all_experiments_wan22_generation.yaml", WORKSPACE_ROOT
        )
        cls.full_jobs = build_jobs(cls.full_resolved)
        cls.standard_ball_resolved = resolve_build(
            CODE_ROOT / "builds" / "standard_ball_all_experiments_wan22_generation.yaml",
            WORKSPACE_ROOT,
        )
        cls.standard_ball_jobs = build_jobs(cls.standard_ball_resolved)

    def test_demo_build_creates_sixty_three_view_jobs_from_five_sampled_scenes(self) -> None:
        jobs = build_jobs(self.resolved)
        self.assertEqual(len(jobs), 60)
        self.assertEqual({job["targets"]["gravity_g"] for job in jobs}, {2.0, 4.9, 9.81, 14.7})
        self.assertTrue(all(job["task_type"] == "i2v" for job in jobs))
        scenes = {job["factors"]["scene_id"] for job in jobs}
        self.assertIn("baseline", scenes)
        self.assertEqual(len([scene for scene in scenes if scene.startswith("indoor")]), 2)
        self.assertEqual(len([scene for scene in scenes if scene.startswith("outdoor")]), 2)
        self.assertEqual({job["factors"]["camera"] for job in jobs}, {"CAM_Main", "CAM_Side", "CAM_Top"})

    def test_object_values_assignment_avoids_full_cross_product(self) -> None:
        jobs = build_jobs(self.resolved)
        pairs = {
            (job["factors"]["object_id"], job["targets"]["gravity_g"])
            for job in jobs
        }
        self.assertEqual(
            pairs,
            {
                ("standard_ball", 2.0),
                ("standard_cube", 4.9),
                ("volleyball", 9.81),
                ("cardboard_box", 14.7),
            },
        )

    def test_all_prompts_use_common_rule_framework(self) -> None:
        jobs = build_jobs(self.resolved)
        for job in jobs:
            spec = job["prompt_spec"]
            self.assertEqual(spec["prompt_framework_id"], "ppb_common_physics_video_v1")
            self.assertEqual(
                spec["section_order"],
                ["task", "scene", "camera", "dynamics", "parameter", "terminal", "quality"],
            )
            self.assertEqual(set(spec["sections"]), set(spec["section_order"]))
            self.assertIn("static_camera", spec["constraints"])
            self.assertIn("vertical_free_fall", spec["constraints"])

    def test_overlay_selection_covers_all_three_cameras(self) -> None:
        jobs = build_jobs(self.resolved)
        selected_ids = _overlay_job_ids(jobs, 9, 36)
        selected = [job for job in jobs if job["job_id"] in selected_ids]
        counts = {
            camera: sum(job["factors"]["camera"] == camera for job in selected)
            for camera in ("CAM_Main", "CAM_Side", "CAM_Top")
        }
        self.assertEqual(counts, {"CAM_Main": 3, "CAM_Side": 3, "CAM_Top": 3})

    def test_job_schema_requires_i2v_image(self) -> None:
        job = build_jobs(self.resolved, max_jobs=1)[0]
        del job["inputs"]["image"]
        with self.assertRaises(JobValidationError):
            validate_job(job)

    def test_wan_adapter_builds_official_command(self) -> None:
        job = build_jobs(self.resolved, max_jobs=1)[0]
        adapter = get_adapter(self.resolved["model"], WORKSPACE_ROOT)
        adapter.validate(job, dry_run=True)
        invocation = adapter.build_invocation(job, WORKSPACE_ROOT / "outputs" / "demo.mp4")
        self.assertIn("--task", invocation.command)
        self.assertIn("i2v-A14B", invocation.command)
        self.assertIn("--ckpt_dir", invocation.command)
        self.assertIn("--save_file", invocation.command)
        self.assertEqual(invocation.environment["CUDA_VISIBLE_DEVICES"], "0")
        offload_index = invocation.command.index("--offload_model")
        self.assertEqual(invocation.command[offload_index + 1], "False")

    def test_basic_evaluator_marks_missing_video_invalid(self) -> None:
        job = build_jobs(self.resolved, max_jobs=1)[0]
        result = evaluate_video(job, WORKSPACE_ROOT / "does-not-exist.mp4", {"min_bytes": 1024})
        self.assertEqual(result["status"], "invalid")
        self.assertIn("missing_video", result["quality_flags"])

    def test_full_generation_build_is_exhaustive(self) -> None:
        jobs = self.full_jobs
        self.assertEqual(len(jobs), 7452)
        self.assertEqual(len({job["inputs"]["image"] for job in jobs}), 1404)
        parameter_tuples = {
            (job["experiment_id"], job["factors"]["parameter_tuple_id"])
            for job in jobs
        }
        self.assertEqual(len(parameter_tuples), 69)
        tuple_counts = Counter(
            (job["experiment_id"], job["factors"]["parameter_tuple_id"])
            for job in jobs
        )
        self.assertEqual(set(tuple_counts.values()), {108})
        self.assertEqual(
            {job["factors"]["scene_id"] for job in jobs},
            {"baseline", "indoor1", "indoor2", "indoor3", "indoor4", "outdoor1", "outdoor2", "outdoor3", "outdoor4"},
        )
        self.assertEqual({job["factors"]["object_id"] for job in jobs}, {"standard_ball", "standard_cube", "cardboard_box", "volleyball"})
        self.assertEqual({job["factors"]["camera"] for job in jobs}, {"CAM_Main", "CAM_Side", "CAM_Top"})

    def test_full_generation_uses_current_v3b_physics(self) -> None:
        job = next(job for job in self.full_jobs if job["experiment_id"] == "v3_B")
        self.assertEqual(
            set(job["targets"]),
            {"kinetic_friction_mu_k", "left_restitution_e_L", "right_restitution_e_R"},
        )
        self.assertIn("two fixed walls", job["prompt"])
        self.assertIn("There is no spring", job["prompt"])

    def test_full_generation_uses_one_common_prompt_framework(self) -> None:
        self.assertEqual(
            {job["prompt_spec"]["prompt_framework_id"] for job in self.full_jobs},
            {"ppb_common_physics_video_v1"},
        )
        self.assertEqual(
            {job["prompt_template_id"] for job in self.full_jobs},
            {"ppb_all_13_explicit_physics"},
        )

    def test_standard_ball_speed_build_has_1863_jobs(self) -> None:
        jobs = self.standard_ball_jobs
        self.assertEqual(len(jobs), 1863)
        self.assertEqual({job["factors"]["object_id"] for job in jobs}, {"standard_ball"})
        self.assertEqual(len({job["inputs"]["image"] for job in jobs}), 351)
        parameter_tuples = Counter(
            (job["experiment_id"], job["factors"]["parameter_tuple_id"])
            for job in jobs
        )
        self.assertEqual(len(parameter_tuples), 69)
        self.assertEqual(set(parameter_tuples.values()), {27})
        self.assertEqual({job["generation"]["num_frames"] for job in jobs}, {81})
        self.assertEqual(
            int(self.standard_ball_resolved["model"]["generation"]["frame_num"]),
            81,
        )
        self.assertEqual(
            self.standard_ball_resolved["applied_overrides"]["experiment"]["selection"]["objects"],
            ["standard_ball"],
        )


if __name__ == "__main__":
    unittest.main()

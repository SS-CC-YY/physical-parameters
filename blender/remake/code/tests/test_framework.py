from __future__ import annotations

import sys
import unittest
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


class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = resolve_build(CODE_ROOT / "builds" / "v1a_wan22_demo.yaml", WORKSPACE_ROOT)

    def test_demo_build_creates_twenty_jobs_from_five_sampled_scenes(self) -> None:
        jobs = build_jobs(self.resolved)
        self.assertEqual(len(jobs), 20)
        self.assertEqual({job["targets"]["gravity_g"] for job in jobs}, {2.0, 4.9, 9.81, 14.7})
        self.assertTrue(all(job["task_type"] == "i2v" for job in jobs))
        scenes = {job["factors"]["scene_id"] for job in jobs}
        self.assertIn("baseline", scenes)
        self.assertEqual(len([scene for scene in scenes if scene.startswith("indoor")]), 2)
        self.assertEqual(len([scene for scene in scenes if scene.startswith("outdoor")]), 2)

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
        self.assertEqual(invocation.environment["CUDA_VISIBLE_DEVICES"], "7")

    def test_basic_evaluator_marks_missing_video_invalid(self) -> None:
        job = build_jobs(self.resolved, max_jobs=1)[0]
        result = evaluate_video(job, WORKSPACE_ROOT / "does-not-exist.mp4", {"min_bytes": 1024})
        self.assertEqual(result["status"], "invalid")
        self.assertIn("missing_video", result["quality_flags"])


if __name__ == "__main__":
    unittest.main()

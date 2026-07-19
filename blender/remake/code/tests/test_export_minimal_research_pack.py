from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_minimal_research_pack.py"
SPEC = importlib.util.spec_from_file_location("export_minimal_research_pack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MinimalResearchPackTests(unittest.TestCase):
    def test_simplified_task_keeps_generation_inputs_and_physics(self) -> None:
        job = {
            "job_id": "v1_A__g9p81__baseline__standard_ball__CAM_Side__seed-1",
            "experiment_id": "v1_A",
            "seed": 1,
            "inputs": {"image": "first_frames_v1_0/images/v1_A/baseline/standard_ball/CAM_Side.png"},
            "factors": {
                "parameter_tuple_id": "g9p81",
                "scene_id": "baseline",
                "object_id": "standard_ball",
                "camera": "CAM_Side",
            },
            "targets": {"gravity_g": 9.81},
            "units": {"gravity_g": "m/s^2"},
            "prompt": "prompt",
            "negative_prompt": "negative",
            "generation": {"width": 832, "height": 480, "fps": 16.0, "num_frames": 81},
        }
        task = MODULE.simplify_job(job)
        self.assertEqual(task["targets"], {"gravity_g": 9.81})
        self.assertEqual(task["video"]["duration_seconds"], 5.0)
        self.assertEqual(task["output_video"], f"videos/{job['job_id']}.mp4")

    def test_unsafe_first_frame_path_is_rejected(self) -> None:
        with self.assertRaises(MODULE.ExportError):
            MODULE._safe_relative("../outside.png")

    def test_standalone_prompt_builder_compiles(self) -> None:
        compile(MODULE.PROMPT_BUILDER, "prompt_setup/build_prompts.py", "exec")


if __name__ == "__main__":
    unittest.main()

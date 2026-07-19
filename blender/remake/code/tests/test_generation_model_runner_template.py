from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "templates"
    / "generation_model_runner"
    / "run_model.py"
)
SPEC = importlib.util.spec_from_file_location("generation_model_runner_template", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_fixture(root: Path, job_ids: list[str]) -> tuple[Path, Path]:
    bundle = root / "bundle"
    images = bundle / "first_frames"
    images.mkdir(parents=True)
    image = images / "input.png"
    image.write_bytes(b"png fixture")
    jobs = [
        {
            "job_id": job_id,
            "inputs": {"image": "first_frames/input.png"},
            "prompt": "frozen prompt",
            "negative_prompt": "frozen negative prompt",
            "seed": index + 1,
            "generation": {"width": 832, "height": 480, "num_frames": 81, "fps": 16},
        }
        for index, job_id in enumerate(job_ids)
    ]
    (bundle / "manifest.jsonl").write_text(
        "".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8"
    )
    config = root / "model.json"
    config.write_text(
        json.dumps({"model_id": "fake-local-model", "checkpoint_path": "/tmp/fake"}),
        encoding="utf-8",
    )
    return bundle, config


class GenerationModelRunnerTemplateTests(unittest.TestCase):
    def test_model_is_loaded_once_and_outputs_are_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, config = _write_fixture(root, ["job-a", "job-b"])
            output = root / "output"
            loaded_model = object()

            def fake_generate(model, job, input_image, output_video, model_config):
                self.assertIs(model, loaded_model)
                self.assertTrue(input_image.is_file())
                self.assertEqual(model_config["model_id"], "fake-local-model")
                output_video.write_bytes(b"mp4 fixture")
                return {"seed_applied": job["seed"]}

            with mock.patch.object(MODULE, "load_model_once", return_value=loaded_model) as load:
                with mock.patch.object(MODULE, "generate_video", side_effect=fake_generate):
                    summary = MODULE.run_generation(bundle, output, config)

            load.assert_called_once()
            self.assertEqual(summary["ok"], 2)
            self.assertEqual(summary["error"], 0)
            for job_id in ("job-a", "job-b"):
                self.assertTrue((output / "videos" / f"{job_id}.mp4").is_file())
                metadata = json.loads(
                    (output / "metadata" / f"{job_id}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["status"], "succeeded")
                self.assertEqual(metadata["job_id"], job_id)

    def test_failure_continues_and_resume_skips_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, config = _write_fixture(root, ["good", "bad"])
            output = root / "output"

            def fake_generate(_model, job, _input_image, output_video, _config):
                if job["job_id"] == "bad":
                    raise RuntimeError("provider failed")
                output_video.write_bytes(b"mp4 fixture")
                return {}

            with mock.patch.object(MODULE, "load_model_once", return_value=object()):
                with mock.patch.object(MODULE, "generate_video", side_effect=fake_generate):
                    first = MODULE.run_generation(bundle, output, config)
                    second = MODULE.run_generation(bundle, output, config, resume=True)

            self.assertEqual((first["ok"], first["error"]), (1, 1))
            self.assertEqual(second["skip"], 1)
            self.assertEqual(second["error"], 1)
            self.assertTrue((output / "run_summary.json").is_file())
            self.assertTrue((output / "logs" / "bad.log").is_file())
            failed_metadata = json.loads(
                (output / "metadata" / "bad.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed_metadata["status"], "failed")

    def test_input_path_escape_is_rejected_per_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, config = _write_fixture(root, ["unsafe"])
            manifest = bundle / "manifest.jsonl"
            job = json.loads(manifest.read_text(encoding="utf-8"))
            job["inputs"]["image"] = "../outside.png"
            manifest.write_text(json.dumps(job) + "\n", encoding="utf-8")

            with mock.patch.object(MODULE, "load_model_once", return_value=object()):
                summary = MODULE.run_generation(bundle, root / "output", config)

            self.assertEqual(summary["ok"], 0)
            self.assertEqual(summary["error"], 1)


if __name__ == "__main__":
    unittest.main()

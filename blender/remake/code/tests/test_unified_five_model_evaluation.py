from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_unified_five_model_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location("unified_five_model_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UnifiedFiveModelEvaluationTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, dict[str, Path]]:
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "\n".join(
                json.dumps({"job_id": value})
                for value in ("job_a", "job_b")
            )
            + "\n",
            encoding="utf-8",
        )
        registry = root / "registry.json"
        registry.write_text('{"schema_version":"test"}\n', encoding="utf-8")
        signature = {
            "registry_sha256": _sha256(registry),
            "evaluator_source_sha256": "evaluator",
            "physics_fitter_source_sha256": "fitter",
            "dynamic_3d_gate_source_sha256": "gate",
            "evaluator_version": "1.3.0",
        }
        evaluations: dict[str, Path] = {}
        for model in ("wan", "seedance", "cosmos", "helios", "longlive"):
            evaluation = root / model / "evaluation"
            evaluations[model] = evaluation
            for job_id in ("job_a", "job_b"):
                job_dir = evaluation / "jobs" / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                (job_dir / "trajectory_frames.csv").write_text(
                    "frame_index,x_m\n0,0\n",
                    encoding="utf-8",
                )
                result = {
                    "evaluation_lineage": {
                        **signature,
                        "trajectory_sha256": f"trajectory-{model}-{job_id}",
                        "track_result_sha256": f"track-{model}-{job_id}",
                    },
                    "evaluation_reads_video": False,
                    "evaluation_invokes_tracker": False,
                }
                (job_dir / "result.json").write_text(
                    json.dumps(result),
                    encoding="utf-8",
                )
            (evaluation / "evaluation_metadata_all.json").write_text(
                json.dumps(
                    {
                        "phase": "all",
                        "selected_jobs": 2,
                        "evaluation_reads_video": False,
                        "evaluation_invokes_tracker": False,
                    }
                ),
                encoding="utf-8",
            )
        return manifest, registry, evaluations

    def test_accepts_one_common_lineage_with_per_video_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, registry, evaluations = self._fixture(Path(directory))
            audit = MODULE.validate_unified_lineage(
                evaluations,
                manifest_path=manifest,
                registry_path=registry,
            )
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["model_count"], 5)
            self.assertEqual(audit["manifest_job_count"], 2)
            self.assertEqual(audit["common_lineage"]["evaluator_version"], "1.3.0")

    def test_rejects_one_model_with_different_fitter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, registry, evaluations = self._fixture(Path(directory))
            path = evaluations["helios"] / "jobs" / "job_b" / "result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["evaluation_lineage"]["physics_fitter_source_sha256"] = "other"
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "mixed evaluator lineage"):
                MODULE.validate_unified_lineage(
                    evaluations,
                    manifest_path=manifest,
                    registry_path=registry,
                )

    def test_rejects_incomplete_manifest_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, registry, evaluations = self._fixture(Path(directory))
            (
                evaluations["cosmos"] / "jobs" / "job_b" / "result.json"
            ).unlink()
            with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
                MODULE.validate_unified_lineage(
                    evaluations,
                    manifest_path=manifest,
                    registry_path=registry,
                )


if __name__ == "__main__":
    unittest.main()

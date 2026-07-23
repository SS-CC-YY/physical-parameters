from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "export_v131_advisor_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("export_v131_advisor_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExportAdvisorEvidenceTests(unittest.TestCase):
    def test_core_selection_is_fixed_at_eighteen(self) -> None:
        cases = MODULE.core_cases()
        self.assertEqual(len(cases), 18)
        self.assertEqual(
            sum(case.group == "01_five_model_e080" for case in cases), 5
        )
        self.assertEqual(
            sum(case.group == "02_five_model_e058" for case in cases), 5
        )
        self.assertEqual(
            sum(case.group == "06_dynamic_3d_accepted" for case in cases), 1
        )

    def test_resolver_uses_paths_recorded_by_evaluation(self) -> None:
        case = MODULE.core_cases()[0]
        with tempfile.TemporaryDirectory() as temp_text:
            root = Path(temp_text)
            eval_job = (
                root
                / "models"
                / MODULE.MODEL_DIRS[case.model]
                / "evaluation"
                / "jobs"
                / case.job_id
            )
            track_job = root / "tracks" / "jobs" / case.job_id
            videos = root / "videos"
            eval_job.mkdir(parents=True)
            track_job.mkdir(parents=True)
            videos.mkdir()

            video = videos / f"{case.job_id}.mp4"
            overlay = track_job / "object_track_overlay.mp4"
            trajectory = track_job / "trajectory_frames.csv"
            track_result = track_job / "track_result.json"
            plot = eval_job / "trajectory_plot.png"
            for path in (video, overlay, trajectory, plot):
                path.write_bytes(b"test")

            track_result.write_text(
                json.dumps(
                    {
                        "source_video": str(video),
                        "object_track_overlay": str(overlay),
                        "trajectory_frames_csv": str(trajectory),
                    }
                ),
                encoding="utf-8",
            )
            (eval_job / "result.json").write_text(
                json.dumps(
                    {
                        "job": {"video_path": str(video)},
                        "source_extraction": str(track_result),
                        "source_trajectory": str(trajectory),
                        "visual_evidence": {"trajectory_plot": str(plot)},
                        "fit": {"status": "ok"},
                        "metrics": {"parameters": {}},
                    }
                ),
                encoding="utf-8",
            )

            resolved = MODULE.resolve_case(root, case)
            self.assertEqual(resolved.missing, [])
            self.assertEqual(resolved.artifacts["original.mp4"], video)
            self.assertEqual(
                resolved.artifacts["object_track_overlay.mp4"], overlay
            )
            self.assertEqual(resolved.artifacts["trajectory_plot.png"], plot)
            self.assertEqual(
                resolved.artifacts["trajectory_frames.csv"], trajectory
            )

            archive_path = root / "evidence.tar.gz"
            MODULE.create_archive(
                root,
                archive_path,
                [resolved],
                overwrite=False,
            )
            with tarfile.open(archive_path, "r:gz") as archive:
                names = set(archive.getnames())
            case_root = (
                "physparambench_v131_advisor_evidence/cases/"
                f"{case.group}/{MODULE.MODEL_DIRS[case.model]}/{case.job_id}"
            )
            self.assertIn(f"{case_root}/original.mp4", names)
            self.assertIn(f"{case_root}/object_track_overlay.mp4", names)
            self.assertIn(f"{case_root}/trajectory_plot.png", names)


if __name__ == "__main__":
    unittest.main()

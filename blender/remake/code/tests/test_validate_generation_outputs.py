from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_generation_outputs.py"
SPEC = importlib.util.spec_from_file_location("validate_generation_outputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_manifest(path: Path, job_ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"job_id": job_id}) + "\n" for job_id in job_ids),
        encoding="utf-8",
    )


class ValidateGenerationOutputsTests(unittest.TestCase):
    def test_complete_canonical_video_set_is_valid_without_ffprobe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, ["job-a", "job-b"])
            (videos / "job-a.mp4").write_bytes(b"not decoded in this mode")
            (videos / "job-b.mp4").write_bytes(b"not decoded in this mode")

            report = MODULE.validate_generation_outputs(
                manifest, videos, expected_jobs=2, ffprobe_mode="off"
            )

            self.assertTrue(report["valid"])
            self.assertEqual(report["nonempty_expected_videos"], 2)
            self.assertEqual(report["issues"], [])

    def test_missing_empty_and_unexpected_videos_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            videos.mkdir()
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, ["job-a", "job-b"])
            (videos / "job-a.mp4").write_bytes(b"")
            (videos / "extra.mp4").write_bytes(b"data")

            report = MODULE.validate_generation_outputs(manifest, videos, ffprobe_mode="off")

            self.assertFalse(report["valid"])
            self.assertEqual(report["missing_count"], 1)
            self.assertEqual(report["empty_count"], 1)
            self.assertEqual(report["extra_count"], 1)
            self.assertEqual(
                {issue["kind"] for issue in report["issues"]},
                {"missing_videos", "empty_videos", "unexpected_videos"},
            )

    def test_duplicate_manifest_job_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, ["same", "same"])
            with self.assertRaisesRegex(ValueError, "duplicate job_id"):
                MODULE.validate_generation_outputs(manifest, root, ffprobe_mode="off")

    def test_metadata_contract_is_checked_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = root / "videos"
            metadata = root / "metadata"
            videos.mkdir()
            metadata.mkdir()
            manifest = root / "manifest.jsonl"
            write_manifest(manifest, ["job-a", "job-b"])
            for job_id in ("job-a", "job-b"):
                (videos / f"{job_id}.mp4").write_bytes(b"mp4")
            (metadata / "job-a.json").write_text(
                json.dumps({"job_id": "job-a", "status": "succeeded"}), encoding="utf-8"
            )
            (metadata / "job-b.json").write_text(
                json.dumps({"job_id": "wrong", "status": "succeeded"}), encoding="utf-8"
            )

            report = MODULE.validate_generation_outputs(
                manifest,
                videos,
                metadata_dir=metadata,
                ffprobe_mode="off",
            )

            self.assertFalse(report["valid"])
            self.assertEqual(report["metadata"]["invalid_count"], 1)
            self.assertIn("invalid_metadata", {issue["kind"] for issue in report["issues"]})


if __name__ == "__main__":
    unittest.main()

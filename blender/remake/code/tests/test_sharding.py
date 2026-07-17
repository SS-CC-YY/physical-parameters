from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.core.io import write_json, write_jsonl  # noqa: E402
from remake_benchmark.orchestration.sharding import collect_sharded_run, split_job_ranges  # noqa: E402


class ShardingTests(unittest.TestCase):
    def test_standard_ball_ranges_cover_all_jobs_without_overlap(self) -> None:
        ranges = split_job_ranges(1863, [4, 5, 6, 7])
        self.assertEqual(
            ranges,
            [
                {"gpu_id": "4", "start_index": 0, "max_jobs": 466},
                {"gpu_id": "5", "start_index": 466, "max_jobs": 466},
                {"gpu_id": "6", "start_index": 932, "max_jobs": 466},
                {"gpu_id": "7", "start_index": 1398, "max_jobs": 465},
            ],
        )

    def test_reduced_standard_ball_ranges_are_balanced(self) -> None:
        ranges = split_job_ranges(1296, [4, 5, 6, 7])
        self.assertEqual([item["start_index"] for item in ranges], [0, 324, 648, 972])
        self.assertEqual([item["max_jobs"] for item in ranges], [324, 324, 324, 324])

    def test_four_seed_ranges_assign_one_job_per_gpu(self) -> None:
        ranges = split_job_ranges(4, [4, 5, 6, 7])
        self.assertEqual([item["start_index"] for item in ranges], [0, 1, 2, 3])
        self.assertEqual([item["max_jobs"] for item in ranges], [1, 1, 1, 1])

    def test_collect_sharded_run_validates_and_hardlinks_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory)
            jobs = [{"job_id": f"job-{index}"} for index in range(4)]
            write_jsonl(run_root / "control" / "manifest.jsonl", jobs)
            (run_root / "control" / "resolved_build.yaml").write_text("build_id: test\n", encoding="utf-8")
            write_json(run_root / "control" / "manifest.selection.json", {"jobs": 4})

            sources: dict[str, Path] = {}
            for index, gpu_id in enumerate((4, 5, 6, 7)):
                job_id = f"job-{index}"
                shard = run_root / "shards" / f"gpu-{gpu_id}"
                video = shard / "videos" / f"{job_id}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(f"video-{index}".encode())
                write_json(shard / "metadata" / f"{job_id}.json", {"job_id": job_id})
                sources[job_id] = video

            summary = collect_sharded_run(run_root)

            self.assertEqual(summary["total"], 4)
            self.assertEqual(summary["gpu_ids"], ["4", "5", "6", "7"])
            for job_id, source in sources.items():
                collected = run_root / "videos" / f"{job_id}.mp4"
                self.assertTrue(os.path.samefile(source, collected))
            saved_summary = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(saved_summary["sharded"])


if __name__ == "__main__":
    unittest.main()

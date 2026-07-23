from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.reconstruction.deadline_report import (  # noqa: E402
    build_deadline_report,
    build_scan_rows,
    load_parameter_rows,
    parse_model_arguments,
    trajectory_metrics,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _registry() -> dict:
    return {
        "experiments": [
            {
                "id": "v1_A",
                "hidden_parameters": [
                    {"name": "p", "unit": "1", "valid_range": [0.0, 1.0]}
                ],
                "anchor_tuples": [
                    {"id": "p0", "p": 0.0},
                    {"id": "p1", "p": 0.5},
                    {"id": "p2", "p": 1.0},
                ],
            }
        ]
    }


def _result(
    *,
    job_id: str,
    tuple_id: str,
    scene: str,
    camera: str,
    seed: int,
    target: float,
    estimate: float,
    accepted: bool = True,
) -> dict:
    observed = [0.0, 0.5, 1.0]
    predicted = [0.0, 0.45, 0.95]
    return {
        "schema_version": "2.1.0",
        "job": {
            "experiment_id": "v1_A",
            "parameter_tuple_id": tuple_id,
            "scene_id": scene,
            "camera_name": camera,
            "seed": seed,
            "video_name": f"{job_id}.mp4",
        },
        "target_lookup_performed": True,
        "evaluation_lineage": {
            "registry_sha256": "registry",
            "evaluator_source_sha256": "evaluator",
            "physics_fitter_source_sha256": "fitter",
        },
        "fit": {
            "status": "ok" if accepted else "model_mismatch",
            "raw_parameter_estimates": {"p": estimate},
            "parameter_estimates": {"p": estimate if accepted else None},
            "parameter_observed": {"p": accepted},
            "parameter_attribution": {
                "p": {"status": "pass" if accepted else "fail"}
            },
            "segmentation": {
                "segments": [
                    {
                        "start_index": 0,
                        "end_index": 2,
                        "start_source_frame": 4,
                        "end_source_frame": 8,
                    }
                ]
            },
            "fit_input": {
                "metric_point_count": 4,
                "source_frame_indices": [4, 6, 8, 10],
            },
            "diagnostics": {
                "primary": {
                    "fit_series": {
                        "series_name": "q",
                        "time_s": [0.0, 0.1, 0.2],
                        "observed": observed,
                        "predicted": predicted,
                    }
                }
            },
        },
        "metrics": {
            "parameters": {"p": {"gt": target, "estimate_raw": estimate}}
        },
        "video_generation_validity": {"status": "pass"},
        "trajectory_csv": f"/tmp/{job_id}/trajectory_frames.csv",
        "visual_evidence": {
            "trajectory_plot": f"/tmp/{job_id}/trajectory_plot.png"
        },
    }


class DeadlineReportTests(unittest.TestCase):
    def test_fit_series_metrics_are_deduplicated_and_normalized(self) -> None:
        series = {
            "series_name": "q",
            "time_s": [0.0, 1.0, 2.0],
            "observed": [0.0, 1.0, 2.0],
            "predicted": [0.0, 0.8, 1.8],
        }
        metrics = trajectory_metrics(
            {
                "diagnostics": {
                    "a": {"fit_series": series},
                    "duplicate_collection": [series],
                },
                "fit_input": {"metric_point_count": 4},
            }
        )
        self.assertEqual(metrics["fit_series_count"], 1)
        self.assertEqual(metrics["fit_point_count"], 3)
        self.assertAlmostEqual(
            metrics["trajectory_rmse"],
            math.sqrt((0.0**2 + 0.2**2 + 0.2**2) / 3.0),
        )
        self.assertAlmostEqual(metrics["trajectory_mae"], 0.4 / 3.0)
        self.assertEqual(metrics["fit_coverage"], 0.75)
        self.assertGreater(metrics["trajectory_r2"], 0.9)

    def test_seed_medians_form_one_canonical_parameter_channel(self) -> None:
        rows: list[dict[str, object]] = []
        for seed, offset in ((1, -0.02), (2, 0.02)):
            for target in (0.0, 0.5, 1.0):
                rows.append(
                    {
                        "model": "m",
                        "experiment_id": "v1_A",
                        "parameter_name": "p",
                        "parameter_family": "p",
                        "scene_id": "baseline",
                        "camera_name": "CAM_Side",
                        "seed": str(seed),
                        "nuisance_signature": "{}",
                        "parameter_count": 1,
                        "target": target,
                        "estimate": target + offset,
                        "accepted_estimate": target + offset,
                        "valid_min": 0.0,
                        "valid_max": 1.0,
                        "trajectory_nrmse": 0.05,
                        "trajectory_r2": 0.98,
                        "fit_coverage": 0.9,
                    }
                )
        scans = build_scan_rows(rows)
        self.assertEqual(len(scans), 1)
        self.assertTrue(scans[0]["canonical_scan"])
        self.assertEqual(scans[0]["seed_count"], 2)
        self.assertEqual(scans[0]["available_level_count"], 3)
        self.assertEqual(scans[0]["response_grade"], "ACCURATE")
        self.assertAlmostEqual(scans[0]["theil_sen_response_slope"], 1.0)

    def test_full_human_report_handles_matches_and_manifest_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            output = root / "report"
            registry_path = root / "registry.json"
            manifest_path = root / "manifest.jsonl"
            _write_json(registry_path, _registry())
            source_rows: list[dict[str, object]] = []
            manifest_lines: list[str] = []
            jobs: list[tuple[str, str, str, str, int, float, float, bool]] = []
            for index, (tuple_id, target) in enumerate(
                (("p0", 0.0), ("p1", 0.5), ("p2", 1.0))
            ):
                jobs.append(
                    (
                        f"v1_A__{tuple_id}__baseline__ball__CAM_Side__seed-1",
                        tuple_id,
                        "baseline",
                        "CAM_Side",
                        1,
                        target,
                        target + 0.02,
                        True,
                    )
                )
            jobs.extend(
                [
                    (
                        "v1_A__p1__indoor1__ball__CAM_Side__seed-1",
                        "p1",
                        "indoor1",
                        "CAM_Side",
                        1,
                        0.5,
                        0.55,
                        True,
                    ),
                    (
                        "v1_A__p1__baseline__ball__CAM_Main__seed-1",
                        "p1",
                        "baseline",
                        "CAM_Main",
                        1,
                        0.5,
                        0.53,
                        True,
                    ),
                ]
            )
            for job in jobs:
                job_id, tuple_id, scene, camera, seed, target, estimate, accepted = job
                source_rows.append(
                    {
                        "job_id": job_id,
                        "video_name": f"{job_id}.mp4",
                        "experiment_id": "v1_A",
                        "parameter_tuple_id": tuple_id,
                        "scene_id": scene,
                        "camera_name": camera,
                        "seed": seed,
                    }
                )
                manifest_lines.append(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "experiment_id": "v1_A",
                            "seed": seed,
                            "factors": {
                                "parameter_tuple_id": tuple_id,
                                "scene_id": scene,
                                "object_id": "ball",
                                "camera": camera,
                            },
                        }
                    )
                )
                _write_json(
                    evaluation / "jobs" / job_id / "result.json",
                    _result(
                        job_id=job_id,
                        tuple_id=tuple_id,
                        scene=scene,
                        camera=camera,
                        seed=seed,
                        target=target,
                        estimate=estimate,
                        accepted=accepted,
                    ),
                )
            _write_csv(evaluation / "all_jobs.csv", source_rows)
            manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

            summary = build_deadline_report(
                {"synthetic": evaluation},
                output=output,
                registry_path=registry_path,
                manifest_path=manifest_path,
            )
            self.assertEqual(summary["parameter_row_count"], len(jobs))
            self.assertEqual(summary["background_pair_count"], 1)
            self.assertEqual(summary["view_pair_count"], 1)
            self.assertEqual(summary["primary_parameter_channel_count"], 1)
            self.assertEqual(summary["experiment_summary_row_count"], 1)
            self.assertEqual(
                summary["input_completeness"][0]["job_universe_coverage"], 1.0
            )
            for relative in (
                "index.html",
                "REPORT_ZH.md",
                "model_summary.csv",
                "side_parameter_results.csv",
                "side_scan_summary.csv",
                "primary_24_channel_summary.csv",
                "experiment_summary.csv",
                "background_comparison.csv",
                "view_consistency.csv",
                "composition_effect.csv",
                "figures/figure1-parameter-fidelity.svg",
            ):
                self.assertTrue((output / relative).is_file(), relative)
            with (output / "model_summary.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                model_rows = list(csv.DictReader(handle))
            self.assertEqual(model_rows[0]["primary_scan_count"], "1")
            self.assertEqual(model_rows[0]["accurate_scan_count"], "1")

    def test_raw_rejected_candidate_is_not_treated_as_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            _write_json(registry_path, _registry())
            job_id = "v1_A__p1__baseline__ball__CAM_Side__seed-1"
            _write_csv(
                root / "eval" / "all_jobs.csv",
                [
                    {
                        "job_id": job_id,
                        "video_name": f"{job_id}.mp4",
                        "experiment_id": "v1_A",
                        "parameter_tuple_id": "p1",
                        "scene_id": "baseline",
                        "camera_name": "CAM_Side",
                        "seed": 1,
                    }
                ],
            )
            _write_json(
                root / "eval" / "jobs" / job_id / "result.json",
                _result(
                    job_id=job_id,
                    tuple_id="p1",
                    scene="baseline",
                    camera="CAM_Side",
                    seed=1,
                    target=0.5,
                    estimate=0.51,
                    accepted=False,
                ),
            )
            rows = load_parameter_rows(
                "m", root / "eval", _registry()
            )
            self.assertIsNone(rows[0]["estimate"])
            self.assertEqual(rows[0]["candidate_estimate"], 0.51)
            self.assertIsNone(rows[0]["accepted_estimate"])
            self.assertEqual(rows[0]["simple_status"], "CANDIDATE_REJECTED")

    def test_frozen_channel_remains_insufficient_when_levels_are_missing(self) -> None:
        rows = [
            {
                "model": "m",
                "experiment_id": "v1_A",
                "parameter_name": "p",
                "parameter_family": "p",
                "scene_id": "baseline",
                "camera_name": "CAM_Side",
                "seed": "1",
                "nuisance_signature": "{}",
                "parameter_count": 1,
                "target": 0.0,
                "estimate": 0.0,
                "accepted_estimate": 0.0,
                "valid_min": 0.0,
                "valid_max": 1.0,
            }
        ]
        scans = build_scan_rows(rows, _registry(), model_names=["m"])
        canonical = [row for row in scans if row["canonical_scan"]]
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0]["expected_level_count"], 3)
        self.assertEqual(canonical[0]["available_level_count"], 1)
        self.assertEqual(canonical[0]["response_grade"], "INSUFFICIENT")

    def test_canonical_nuisance_branch_is_registry_frozen_not_outcome_selected(self) -> None:
        registry = {
            "experiments": [
                {
                    "id": "v2_X",
                    "hidden_parameters": [
                        {"name": "p", "valid_range": [0.0, 1.0]},
                        {"name": "q", "valid_range": [0.0, 1.0]},
                    ],
                    "anchor_tuples": [
                        {"id": "a", "p": 0.0, "q": 0.0},
                        {"id": "b", "p": 1.0, "q": 0.0},
                        {"id": "c", "p": 0.0, "q": 1.0},
                        {"id": "d", "p": 1.0, "q": 1.0},
                    ],
                }
            ]
        }
        # Only the second, outcome-rich q=1 branch is observed. Registry order
        # still freezes q=0 as canonical on the exact design tie.
        rows = [
            {
                "model": "m",
                "experiment_id": "v2_X",
                "parameter_name": "p",
                "parameter_family": "p",
                "scene_id": "baseline",
                "camera_name": "CAM_Side",
                "seed": "1",
                "nuisance_signature": '{"q":1.0}',
                "parameter_count": 2,
                "target": target,
                "estimate": target,
                "accepted_estimate": target,
                "valid_min": 0.0,
                "valid_max": 1.0,
            }
            for target in (0.0, 1.0)
        ]
        scans = build_scan_rows(rows, registry, model_names=["m"])
        canonical = next(
            row
            for row in scans
            if row["canonical_scan"] and row["parameter_name"] == "p"
        )
        rich = next(row for row in scans if row["nuisance_signature"] == '{"q":1.0}')
        self.assertEqual(canonical["nuisance_signature"], '{"q":0.0}')
        self.assertEqual(canonical["response_grade"], "INSUFFICIENT")
        self.assertFalse(rich["canonical_scan"])

    def test_response_requires_common_seed_across_parameter_levels(self) -> None:
        rows = []
        for target, seed in ((0.0, "1"), (0.5, "2"), (1.0, "3")):
            rows.append(
                {
                    "model": "m",
                    "experiment_id": "v1_A",
                    "parameter_name": "p",
                    "parameter_family": "p",
                    "scene_id": "baseline",
                    "camera_name": "CAM_Side",
                    "seed": seed,
                    "nuisance_signature": "{}",
                    "parameter_count": 1,
                    "target": target,
                    "estimate": target,
                    "accepted_estimate": target,
                    "valid_min": 0.0,
                    "valid_max": 1.0,
                }
            )
        scan = build_scan_rows(rows, _registry(), model_names=["m"])[0]
        self.assertEqual(scan["common_seed_count"], 0)
        self.assertEqual(scan["available_level_count"], 0)
        self.assertEqual(scan["response_grade"], "INSUFFICIENT")

    def test_empty_model_and_mixed_lineage_fail_before_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            _write_json(registry_path, _registry())
            empty = root / "empty"
            empty.mkdir()
            (empty / "all_jobs.csv").write_text(
                "job_id,experiment_id,parameter_tuple_id\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "no rows matching"):
                build_deadline_report(
                    {"empty": empty},
                    output=root / "empty_report",
                    registry_path=registry_path,
                )

            evaluation = root / "mixed"
            rows = []
            for tuple_id, target in (("p0", 0.0), ("p1", 0.5)):
                job_id = f"v1_A__{tuple_id}__baseline__ball__CAM_Side__seed-1"
                rows.append(
                    {
                        "job_id": job_id,
                        "video_name": f"{job_id}.mp4",
                        "experiment_id": "v1_A",
                        "parameter_tuple_id": tuple_id,
                        "scene_id": "baseline",
                        "camera_name": "CAM_Side",
                        "seed": 1,
                    }
                )
                result = _result(
                    job_id=job_id,
                    tuple_id=tuple_id,
                    scene="baseline",
                    camera="CAM_Side",
                    seed=1,
                    target=target,
                    estimate=target,
                )
                if tuple_id == "p1":
                    result.pop("evaluation_lineage")
                _write_json(
                    evaluation / "jobs" / job_id / "result.json",
                    result,
                )
            _write_csv(evaluation / "all_jobs.csv", rows)
            with self.assertRaisesRegex(ValueError, "mixes legacy"):
                build_deadline_report(
                    {"mixed": evaluation},
                    output=root / "mixed_report",
                    registry_path=registry_path,
                )

    def test_duplicate_manifest_job_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            registry_path = root / "registry.json"
            manifest_path = root / "manifest.jsonl"
            _write_json(registry_path, _registry())
            job_id = "v1_A__p0__baseline__ball__CAM_Side__seed-1"
            source = {
                "job_id": job_id,
                "video_name": f"{job_id}.mp4",
                "experiment_id": "v1_A",
                "parameter_tuple_id": "p0",
                "scene_id": "baseline",
                "camera_name": "CAM_Side",
                "seed": 1,
            }
            _write_csv(evaluation / "all_jobs.csv", [source])
            _write_json(
                evaluation / "jobs" / job_id / "result.json",
                _result(
                    job_id=job_id,
                    tuple_id="p0",
                    scene="baseline",
                    camera="CAM_Side",
                    seed=1,
                    target=0.0,
                    estimate=0.0,
                ),
            )
            manifest_row = {
                "job_id": job_id,
                "experiment_id": "v1_A",
                "seed": 1,
                "factors": {
                    "parameter_tuple_id": "p0",
                    "scene_id": "baseline",
                    "object_id": "ball",
                    "camera": "CAM_Side",
                },
            }
            manifest_path.write_text(
                json.dumps(manifest_row) + "\n" + json.dumps(manifest_row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate job_id"):
                build_deadline_report(
                    {"m": evaluation},
                    output=root / "report",
                    registry_path=registry_path,
                    manifest_path=manifest_path,
                )

    def test_model_argument_parser_rejects_ambiguous_values(self) -> None:
        parsed = parse_model_arguments(["wan=/tmp/wan", "seed=/tmp/seed"])
        self.assertEqual(set(parsed), {"wan", "seed"})
        with self.assertRaises(ValueError):
            parse_model_arguments(["missing_separator"])
        with self.assertRaises(ValueError):
            parse_model_arguments(["wan=/a", "wan=/b"])


if __name__ == "__main__":
    unittest.main()

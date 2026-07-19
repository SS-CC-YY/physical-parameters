from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.core.build import resolve_build  # noqa: E402
from remake_benchmark.core.errors import ConfigError  # noqa: E402
from remake_benchmark.core.io import read_json, write_json  # noqa: E402
from remake_benchmark.core.manifest import build_jobs  # noqa: E402
from remake_benchmark.models import get_adapter  # noqa: E402
from remake_benchmark.orchestration.api_summary import summarize_api_run  # noqa: E402
from remake_benchmark.orchestration.runner import _enforce_max_billable_jobs_per_run  # noqa: E402


SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "closed_api_video_job", CODE_ROOT / "scripts" / "closed_api_video_job.py"
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
CLOSED_API_JOB = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(CLOSED_API_JOB)


class _FakeProviderHandler(BaseHTTPRequestHandler):
    provider = "seedance"
    base_url = ""
    last_post_path = ""
    last_post_payload: dict[str, object] = {}
    post_count = 0

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, value: object) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        type(self).post_count += 1
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        type(self).last_post_path = self.path
        type(self).last_post_payload = json.loads(raw.decode("utf-8"))
        if self.provider == "seedance":
            self._json({"id": "seed-task"})
        else:
            self._json(
                {
                    "code": 0,
                    "request_id": "req-1",
                    "data": {"task_id": "kling-task", "task_status": "submitted"},
                }
            )

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/video.mp4":
            body = b"fake-mp4-bytes"
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.provider == "seedance":
            self._json(
                {
                    "id": "seed-task",
                    "model": "doubao-seedance-2-0-260128",
                    "status": "succeeded",
                    "content": {"video_url": f"{self.base_url}/video.mp4"},
                    "usage": {"completion_tokens": 85714},
                    "duration": 5,
                    "resolution": "720p",
                }
            )
            return
        self._json(
            {
                "code": 0,
                "data": {
                    "task_id": "kling-task",
                    "task_status": "succeed",
                    "task_result": {
                        "videos": [{"url": f"{self.base_url}/video.mp4", "duration": "5"}]
                    },
                    "final_unit_deduction": 30,
                },
            }
        )


class ClosedApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seedance = resolve_build(
            CODE_ROOT / "builds" / "v1a_seedance20_cn_api_cost20.yaml", WORKSPACE_ROOT
        )
        cls.kling = resolve_build(
            CODE_ROOT / "builds" / "v1a_kling3_std_cn_api_cost20.yaml", WORKSPACE_ROOT
        )
        cls.seedance_jobs = build_jobs(cls.seedance)
        cls.kling_jobs = build_jobs(cls.kling)

    def test_cost_builds_are_exactly_twenty_semantically_matched_jobs(self) -> None:
        self.assertEqual(len(self.seedance_jobs), 20)
        self.assertEqual(len(self.kling_jobs), 20)
        matched_fields = (
            "job_id", "experiment_id", "task_type", "case_id", "repeat_id", "seed",
            "factors", "targets", "units", "inputs", "prompt", "negative_prompt", "generation",
        )
        self.assertEqual(
            [[job[field] for field in matched_fields] for job in self.seedance_jobs],
            [[job[field] for field in matched_fields] for job in self.kling_jobs],
        )
        self.assertEqual(
            {job["factors"]["camera"] for job in self.seedance_jobs},
            {"CAM_Side"},
        )
        self.assertEqual(
            {job["factors"]["scene_id"] for job in self.seedance_jobs},
            {"baseline", "indoor3", "indoor4", "outdoor1", "outdoor3"},
        )
        self.assertEqual(
            {job["factors"]["object_id"] for job in self.seedance_jobs},
            {"standard_ball"},
        )
        self.assertEqual({job["seed"] for job in self.seedance_jobs}, {341867882})
        self.assertEqual(
            {job["factors"]["scene_selection_seed"] for job in self.seedance_jobs},
            {20260719},
        )
        self.assertEqual(
            {job["targets"]["gravity_g"] for job in self.seedance_jobs},
            {2.0, 4.9, 9.81, 14.7},
        )
        self.assertEqual(
            Counter(
                (job["factors"]["scene_id"], job["targets"]["gravity_g"])
                for job in self.seedance_jobs
            ),
            Counter(
                (scene, gravity)
                for scene in ("baseline", "indoor3", "indoor4", "outdoor1", "outdoor3")
                for gravity in (2.0, 4.9, 9.81, 14.7)
            ),
        )

    def test_billable_job_guard_rejects_twenty_one(self) -> None:
        model = {"safety": {"max_billable_jobs_per_run": 20}}
        _enforce_max_billable_jobs_per_run(model, 20)
        with self.assertRaises(ConfigError):
            _enforce_max_billable_jobs_per_run(model, 21)

    def test_json_metadata_serializes_yaml_dates_as_iso_strings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "metadata.json"
            write_json(
                output,
                {
                    "checked_on": date(2026, 7, 19),
                    "completed_at": datetime(2026, 7, 19, 8, 30, tzinfo=timezone.utc),
                },
            )
            value = read_json(output)
            self.assertEqual(value["checked_on"], "2026-07-19")
            self.assertEqual(value["completed_at"], "2026-07-19T08:30:00+00:00")

    def test_seedance_dry_run_command_never_contains_api_key(self) -> None:
        adapter = get_adapter(self.seedance["model"], WORKSPACE_ROOT)
        job = self.seedance_jobs[0]
        adapter.validate(job, dry_run=True)
        invocation = adapter.build_invocation(job, WORKSPACE_ROOT / "outputs" / "seedance" / "videos" / "demo.mp4")
        command = " ".join(invocation.command)
        self.assertIn("--provider seedance", command)
        self.assertIn("--api-key-env ARK_API_KEY", command)
        self.assertIn("https://ark.cn-beijing.volces.com/api/v3", command)
        self.assertIn("--resolution 480p", command)
        self.assertIn("--ratio 16:9", command)
        self.assertIn("--camera-fixed", invocation.command)
        self.assertIn("--no-send-seed", command)
        self.assertIn("--no-send-camera-fixed", command)
        self.assertNotIn("Bearer", command)
        self.assertEqual(invocation.environment, {})
        self.assertEqual(invocation.result_metadata.name, "demo.api.json")

    def test_kling_dry_run_uses_official_v1_contract_and_external_id(self) -> None:
        adapter = get_adapter(self.kling["model"], WORKSPACE_ROOT)
        job = self.kling_jobs[0]
        adapter.validate(job, dry_run=True)
        invocation = adapter.build_invocation(job, WORKSPACE_ROOT / "outputs" / "kling" / "videos" / "demo.mp4")
        command = " ".join(invocation.command)
        self.assertIn("--provider kling", command)
        self.assertIn("--api-key-env KLING_API_KEY", command)
        self.assertIn("https://api-beijing.klingai.com", command)
        self.assertIn("--model kling-v3", command)
        self.assertIn("--mode std", command)
        self.assertIn("--sound off", command)
        self.assertIn("--image-encoding raw_base64", command)
        self.assertIn("--external-task-id", invocation.command)
        external_index = invocation.command.index("--external-task-id") + 1
        self.assertRegex(invocation.command[external_index], r"^ppb-[0-9a-f]{32}$")
        second_invocation = adapter.build_invocation(
            job, WORKSPACE_ROOT / "outputs" / "kling-new-run" / "videos" / "demo.mp4"
        )
        second_external_index = second_invocation.command.index("--external-task-id") + 1
        self.assertNotEqual(
            invocation.command[external_index], second_invocation.command[second_external_index]
        )
        self.assertNotIn("secret", command.lower())

    def test_api_summary_prefers_provider_cash_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            metadata_dir = run_dir / "metadata"
            metadata_dir.mkdir()
            metrics = {
                "provider": "kling",
                "model_id": "kling-v3",
                "task_id": "task-1",
                "status": "succeeded",
                "request": {"duration_seconds": 5, "resolution": "720p"},
                "timing": {"wall_seconds": 80.0},
                "cost": {
                    "currency": "USD",
                    "cash_amount": 0.56,
                    "configured_list_cost": 0.56,
                },
            }
            (metadata_dir / "job.api.json").write_text(json.dumps(metrics), encoding="utf-8")
            summary = summarize_api_run(run_dir)
            self.assertEqual(summary["succeeded"], 1)
            self.assertEqual(summary["total_preferred_cost"], 0.56)
            self.assertEqual(summary["total_preferred_cost_usd"], 0.56)
            self.assertEqual(summary["currency"], "USD")
            self.assertEqual(summary["total_cash_amount"], 0.56)
            self.assertEqual(summary["metrics_found"], 1)
            self.assertTrue((run_dir / "api_cost_time.csv").is_file())

    def test_api_summary_does_not_label_cny_cost_as_usd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            metadata_dir = run_dir / "metadata"
            metadata_dir.mkdir()
            metrics = {
                "provider": "seedance",
                "model_id": "doubao-seedance-2-0-260128",
                "status": "succeeded",
                "request": {"duration_seconds": 5, "resolution": "720p"},
                "timing": {"wall_seconds": 90.0},
                "cost": {"currency": "CNY", "completion_tokens": 100000, "list_cost": 4.6},
            }
            (metadata_dir / "job.api.json").write_text(json.dumps(metrics), encoding="utf-8")
            summary = summarize_api_run(run_dir)
            self.assertEqual(summary["total_preferred_cost"], 4.6)
            self.assertEqual(summary["currency"], "CNY")
            self.assertIsNone(summary["total_preferred_cost_usd"])
            self.assertEqual(summary["total_completion_tokens"], 100000)

    def test_api_summary_uses_manifest_to_report_missing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            metadata_dir = run_dir / "metadata"
            metadata_dir.mkdir()
            jobs = [{"job_id": "job-a"}, {"job_id": "job-b"}]
            (run_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8"
            )
            metrics = {
                "provider": "kling",
                "model_id": "kling-v3",
                "task_id": "task-a",
                "status": "succeeded",
                "provider_status": "succeed",
                "request": {"duration_seconds": 5, "resolution": "720p"},
                "timing": {"wall_seconds": 80.0},
                "cost": {"currency": "CREDITS", "credits": 30},
            }
            (metadata_dir / "job-a.api.json").write_text(json.dumps(metrics), encoding="utf-8")
            summary = summarize_api_run(run_dir)
            self.assertEqual(summary["jobs"], 2)
            self.assertEqual(summary["expected_manifest_jobs"], 2)
            self.assertEqual(summary["metrics_found"], 1)
            self.assertEqual(summary["missing_metrics"], 1)
            self.assertEqual(summary["total_credits"], 30)

    def _fake_args(self, temporary: str, provider: str) -> SimpleNamespace:
        root = Path(temporary)
        image = root / "input.png"
        image.write_bytes(b"fake-png")
        return SimpleNamespace(
            provider=provider,
            api_base="",
            model=("doubao-seedance-2-0-260128" if provider == "seedance" else "kling-v3"),
            image=image,
            image_encoding=("data_uri" if provider == "seedance" else "raw_base64"),
            prompt="A ball falls vertically.",
            negative_prompt="camera motion",
            negative_prompt_strategy=("append" if provider == "seedance" else "separate"),
            output=root / "output.mp4",
            metrics_out=root / "metrics.json",
            resolution="720p",
            duration=5,
            ratio="16:9",
            mode="std",
            sound="off",
            seed=36,
            external_task_id="job-1",
            poll_interval_seconds=0.01,
            timeout_seconds=5,
            currency=("CNY" if provider == "seedance" else "provider_account"),
            price_per_million_tokens=46.0,
            price_per_second=None,
            camera_fixed=True,
            send_seed=False,
            send_camera_fixed=False,
            generate_audio=False,
            watermark=False,
        )

    def _run_fake_provider(self, provider: str) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fake_args(temporary, provider)
            _FakeProviderHandler.provider = provider
            _FakeProviderHandler.post_count = 0
            server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeProviderHandler)
            base_url = f"http://127.0.0.1:{server.server_port}"
            _FakeProviderHandler.base_url = base_url
            args.api_base = base_url
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            started = time.monotonic()
            metrics: dict[str, object] = {
                "provider": provider,
                "model_id": args.model,
                "status": "starting",
                "timing": {},
                "poll_history": [],
            }
            try:
                if provider == "seedance":
                    CLOSED_API_JOB._run_seedance(args, "test-key", metrics, started)
                else:
                    CLOSED_API_JOB._run_kling(args, "test-key", metrics, started)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            self.assertTrue(args.output.is_file())
            self.assertEqual(_FakeProviderHandler.post_count, 1)
            return metrics

    def test_submission_marker_blocks_automatic_post_without_saved_task_id(self) -> None:
        for provider in ("seedance", "kling"):
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as temporary:
                args = self._fake_args(temporary, provider)
                marker = CLOSED_API_JOB._submission_marker_path(args.metrics_out)
                marker.write_text("{}\n", encoding="utf-8")
                args.metrics_out.write_text(
                    json.dumps(
                        {
                            "provider": provider,
                            "model_id": args.model,
                            "status": "client_error",
                            "submission_may_have_succeeded": True,
                        }
                    ),
                    encoding="utf-8",
                )
                metrics: dict[str, object] = {
                    "provider": provider,
                    "model_id": args.model,
                    "status": "starting",
                    "timing": {},
                    "poll_history": [],
                }
                runner = (
                    CLOSED_API_JOB._run_seedance
                    if provider == "seedance"
                    else CLOSED_API_JOB._run_kling
                )
                with self.assertRaisesRegex(RuntimeError, "submission marker exists"):
                    runner(args, "test-key", metrics, time.monotonic())

    def test_seedance_job_records_token_cost_and_downloads_video(self) -> None:
        metrics = self._run_fake_provider("seedance")
        self.assertEqual(metrics["task_id"], "seed-task")
        self.assertEqual(metrics["cost"]["completion_tokens"], 85714)
        self.assertAlmostEqual(metrics["cost"]["list_cost"], 3.942844, places=4)
        self.assertEqual(metrics["cost"]["currency"], "CNY")
        payload = _FakeProviderHandler.last_post_payload
        self.assertEqual(_FakeProviderHandler.last_post_path, "/contents/generations/tasks")
        self.assertEqual(payload["content"][1]["role"], "first_frame")
        self.assertNotIn("seed", payload)
        self.assertNotIn("camera_fixed", payload)

    def test_kling_job_records_provider_billing_and_downloads_video(self) -> None:
        metrics = self._run_fake_provider("kling")
        self.assertEqual(metrics["task_id"], "kling-task")
        self.assertEqual(metrics["cost"]["credits"], 30)
        self.assertEqual(metrics["output_duration_seconds"], "5")
        payload = _FakeProviderHandler.last_post_payload
        self.assertEqual(_FakeProviderHandler.last_post_path, "/v1/videos/image2video")
        self.assertEqual(payload["model_name"], "kling-v3")
        self.assertEqual(payload["duration"], "5")
        self.assertEqual(payload["mode"], "std")
        self.assertEqual(payload["sound"], "off")
        self.assertNotIn("seed", payload)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from remake_benchmark.core.build import resolve_build  # noqa: E402
from remake_benchmark.core.errors import ConfigError  # noqa: E402
from remake_benchmark.core.manifest import build_jobs  # noqa: E402
from remake_benchmark.orchestration.runner import (  # noqa: E402
    _enforce_concurrency,
    _enforce_max_billable_jobs_per_run,
)


class Seedance978BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with patch.dict(os.environ, {"SEEDANCE_RESOLUTION": "480p"}):
            cls.seedance = resolve_build(
                CODE_ROOT / "builds" / "standard_ball_factorized978_seedance20_generation.yaml",
                WORKSPACE_ROOT,
            )
        cls.wan = resolve_build(
            CODE_ROOT / "builds" / "standard_ball_factorized978_wan22_generation.yaml",
            WORKSPACE_ROOT,
        )
        cls.seedance_jobs = build_jobs(cls.seedance)
        cls.wan_jobs = build_jobs(cls.wan)

    def test_seedance_build_has_exactly_978_jobs(self) -> None:
        self.assertEqual(len(self.seedance_jobs), 978)
        self.assertEqual(len({job["job_id"] for job in self.seedance_jobs}), 978)

    def test_seedance_jobs_match_wan_jobs_field_for_field_except_build_id(self) -> None:
        def without_model_build_identity(job: dict[str, object]) -> dict[str, object]:
            return {key: value for key, value in job.items() if key != "build_id"}

        self.assertEqual(
            [without_model_build_identity(job) for job in self.seedance_jobs],
            [without_model_build_identity(job) for job in self.wan_jobs],
        )

    def test_full_profile_has_expected_generation_and_safety_limits(self) -> None:
        model = self.seedance["model"]
        self.assertEqual(model["adapter"], "seedance_ark_api")
        self.assertEqual(model["generation"]["resolution"], "480p")
        self.assertEqual(model["generation"]["duration"], 5)
        self.assertEqual(model["safety"]["max_billable_jobs_per_run"], 978)
        self.assertEqual(model["safety"]["max_concurrent_jobs"], 3)
        _enforce_max_billable_jobs_per_run(model, 978)
        _enforce_concurrency(model, 3, "subprocess")
        with self.assertRaises(ConfigError):
            _enforce_max_billable_jobs_per_run(model, 979)
        with self.assertRaises(ConfigError):
            _enforce_concurrency(model, 4, "subprocess")

    def test_resolution_can_be_overridden_from_environment(self) -> None:
        with patch.dict(os.environ, {"SEEDANCE_RESOLUTION": "720p"}):
            resolved = resolve_build(
                CODE_ROOT / "builds" / "standard_ball_factorized978_seedance20_generation.yaml",
                WORKSPACE_ROOT,
            )
        self.assertEqual(resolved["model"]["generation"]["resolution"], "720p")

    def test_background_entrypoint_keeps_full_run_behind_canary_guard(self) -> None:
        script = (CODE_ROOT / "scripts" / "run_seedance978_background.sh").read_text(encoding="utf-8")
        self.assertIn('CONFIRM_BILLABLE_978="${CONFIRM_BILLABLE_978:-NO}"', script)
        self.assertIn('RUN_ID_WAS_EXPLICIT=0', script)
        self.assertIn('CANARY_MARKER="${RUN_DIR}/.canary_success"', script)
        self.assertIn('verify_canary_success', script)
        self.assertIn('verify_full_success', script)
        self.assertIn('--concurrency "${CONCURRENCY}"', script)
        self.assertIn('--fail-fast', script)
        self.assertNotIn('--overwrite', script)


if __name__ == "__main__":
    unittest.main()

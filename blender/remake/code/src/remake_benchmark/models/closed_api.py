from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from remake_benchmark.core.errors import ConfigError

from .base import Invocation, ModelAdapter
from .wan22 import as_bool


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ClosedVideoApiAdapter(ModelAdapter):
    provider: str

    @property
    def runtime(self) -> dict[str, Any]:
        return self.model_config["runtime"]

    @property
    def generation(self) -> dict[str, Any]:
        return self.model_config["generation"]

    @property
    def pricing(self) -> dict[str, Any]:
        value = self.model_config.get("pricing", {})
        return value if isinstance(value, dict) else {}

    @property
    def safety(self) -> dict[str, Any]:
        value = self.model_config.get("safety", {})
        return value if isinstance(value, dict) else {}

    def _input_image(self, job: dict[str, Any]) -> Path:
        path = Path(job["inputs"]["image"])
        return path if path.is_absolute() else self.workspace_root / path

    def _api_key_env(self) -> str:
        name = str(self.runtime.get("api_key_env", ""))
        if not _ENV_NAME.fullmatch(name):
            raise ConfigError(f"invalid API key environment variable name: {name!r}")
        return name

    def validate(self, job: dict[str, Any], *, dry_run: bool) -> None:
        if job["task_type"] != "i2v":
            raise ConfigError(f"{self.provider} API adapter only accepts i2v jobs, got {job['task_type']}")
        image = self._input_image(job)
        if not image.is_file():
            raise ConfigError(f"conditioning image not found: {image}")
        api_base = str(self.runtime.get("api_base", ""))
        if not api_base.startswith("https://"):
            raise ConfigError(f"{self.provider} api_base must use https://, got {api_base!r}")
        duration = int(self.generation.get("duration", 0))
        if not 3 <= duration <= 15:
            raise ConfigError(f"{self.provider} smoke profile duration must be in [3, 15], got {duration}")
        if str(self.generation.get("resolution", "")) not in {"480p", "720p", "1080p", "4k"}:
            raise ConfigError(f"unsupported {self.provider} resolution: {self.generation.get('resolution')!r}")
        key_env = self._api_key_env()
        currency = str(self.pricing.get("currency", "")).strip()
        if not currency:
            raise ConfigError(f"{self.provider} pricing.currency must be non-empty")
        if not dry_run and not os.environ.get(key_env, "").strip():
            raise ConfigError(f"required API key environment variable is not set: {key_env}")
        image_encoding = str(self.generation.get("image_encoding", "data_uri"))
        if image_encoding not in {"data_uri", "raw_base64"}:
            raise ConfigError(f"unsupported {self.provider} image_encoding: {image_encoding!r}")
        self._model_prompt(job)
        if self.provider == "seedance":
            if not 4 <= duration <= 15:
                raise ConfigError(f"Seedance 2.0 duration must be in [4, 15], got {duration}")
            ratio = str(self.generation.get("ratio", "adaptive"))
            if ratio not in {"adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"}:
                raise ConfigError(f"unsupported Seedance ratio: {ratio!r}")
            if as_bool(self.generation.get("send_seed"), default=False):
                raise ConfigError("Seedance 2.0 does not accept an API seed field")
            if as_bool(self.generation.get("send_camera_fixed"), default=False):
                raise ConfigError("Seedance 2.0 does not accept an API camera_fixed field")
        elif self.provider == "kling":
            if duration not in {5, 10}:
                raise ConfigError(f"Kling VIDEO 3.0 duration must be 5 or 10 seconds, got {duration}")
            if str(self.generation.get("mode", "std")) not in {"std", "pro"}:
                raise ConfigError(f"unsupported Kling mode: {self.generation.get('mode')!r}")
            if str(self.generation.get("sound", "off")) not in {"on", "off"}:
                raise ConfigError(f"unsupported Kling sound setting: {self.generation.get('sound')!r}")

    def _result_metadata(self, output_video: Path) -> Path:
        run_dir = output_video.parent.parent
        return run_dir / "metadata" / f"{output_video.stem}.api.json"

    def _model_prompt(self, job: dict[str, Any]) -> tuple[str, str, str]:
        strategy = str(self.generation.get("negative_prompt_strategy", "append"))
        if strategy not in {"append", "ignore", "separate"}:
            raise ConfigError(f"unsupported {self.provider} negative_prompt_strategy: {strategy}")
        return str(job["prompt"]), str(job.get("negative_prompt", "")), strategy

    def build_invocation(self, job: dict[str, Any], output_video: Path) -> Invocation:
        code_root = Path(__file__).resolve().parents[3]
        metrics_path = self._result_metadata(output_video)
        prompt, negative_prompt, strategy = self._model_prompt(job)
        run_identity = str(output_video.parent.parent.resolve())
        external_source = f"{run_identity}\0{job['job_id']}"
        external_task_id = "ppb-" + hashlib.sha256(external_source.encode("utf-8")).hexdigest()[:32]
        command = [
            str(self.runtime.get("python", "python")),
            str(code_root / "scripts" / "closed_api_video_job.py"),
            "--provider",
            self.provider,
            "--api-base",
            str(self.runtime["api_base"]),
            "--api-key-env",
            self._api_key_env(),
            "--model",
            str(self.model_config["model_id"]),
            "--image",
            str(self._input_image(job)),
            "--image-encoding",
            str(self.generation.get("image_encoding", "data_uri")),
            "--prompt",
            prompt,
            "--negative-prompt",
            negative_prompt,
            "--negative-prompt-strategy",
            strategy,
            "--output",
            str(output_video),
            "--metrics-out",
            str(metrics_path),
            "--resolution",
            str(self.generation["resolution"]),
            "--duration",
            str(int(self.generation["duration"])),
            "--ratio",
            str(self.generation.get("ratio", "adaptive")),
            "--mode",
            str(self.generation.get("mode", "std")),
            "--sound",
            str(self.generation.get("sound", "off")),
            "--seed",
            str(int(job["seed"])),
            "--external-task-id",
            external_task_id,
            "--poll-interval-seconds",
            str(float(self.runtime.get("poll_interval_seconds", 5))),
            "--timeout-seconds",
            str(float(self.runtime.get("timeout_seconds", 1800))),
            "--currency",
            str(self.pricing.get("currency", "USD")),
        ]
        optional_rates = {
            "price_per_million_tokens": ("usd_per_million_tokens", "--price-per-million-tokens"),
            "price_per_second": ("usd_per_second", "--price-per-second"),
        }
        for key, (legacy_key, flag) in optional_rates.items():
            value = self.pricing.get(key)
            if value is None:
                value = self.pricing.get(legacy_key)
            if value is not None:
                command.extend([flag, str(float(value))])
        boolean_options = {
            "camera_fixed": "--camera-fixed",
            "send_seed": "--send-seed",
            "send_camera_fixed": "--send-camera-fixed",
            "generate_audio": "--generate-audio",
            "watermark": "--watermark",
        }
        for key, flag in boolean_options.items():
            value = as_bool(self.generation.get(key), default=False)
            command.append(flag if value else f"--no-{flag[2:]}")
        return Invocation(
            command=command,
            cwd=code_root,
            output_video=output_video,
            result_metadata=metrics_path,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_id,
            "provider": self.provider,
            "model_id": self.model_config["model_id"],
            "model_revision": self.model_config.get("model_revision", "provider-managed"),
            "api_base": str(self.runtime["api_base"]),
            "api_key_env": self._api_key_env(),
            "generation": dict(self.generation),
            "pricing": dict(self.pricing),
        }


class SeedanceArkApiAdapter(ClosedVideoApiAdapter):
    adapter_id = "seedance_ark_api"
    provider = "seedance"


class KlingApiAdapter(ClosedVideoApiAdapter):
    adapter_id = "kling_api_v1"
    provider = "kling"

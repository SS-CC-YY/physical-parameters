from __future__ import annotations

from pathlib import Path
from typing import Any

from remake_benchmark.core.errors import ConfigError

from .base import Invocation, ModelAdapter


class Wan22Adapter(ModelAdapter):
    """Reference adapter for the official Wan2.2 generate.py I2V interface."""

    adapter_id = "wan22_official"

    @property
    def runtime(self) -> dict[str, Any]:
        return self.model_config["runtime"]

    @property
    def generation(self) -> dict[str, Any]:
        return self.model_config["generation"]

    def _repo(self) -> Path:
        return Path(str(self.runtime["repo"])).expanduser()

    def _checkpoint(self) -> Path:
        return Path(str(self.runtime["checkpoint"])).expanduser()

    def _input_image(self, job: dict[str, Any]) -> Path:
        path = Path(job["inputs"]["image"])
        return path if path.is_absolute() else self.workspace_root / path

    def validate(self, job: dict[str, Any], *, dry_run: bool) -> None:
        if job["task_type"] != "i2v":
            raise ConfigError(f"Wan2.2 reference adapter only accepts i2v jobs, got {job['task_type']}")
        image = self._input_image(job)
        if not image.is_file():
            raise ConfigError(f"conditioning image not found: {image}")
        frame_num = int(self.generation.get("frame_num", job["generation"]["num_frames"]))
        if (frame_num - 1) % 4 != 0:
            raise ConfigError(f"Wan2.2 frame_num must be 4n+1, got {frame_num}")
        expected_size = f"{job['generation']['width']}*{job['generation']['height']}"
        if str(self.generation.get("size", expected_size)) != expected_size:
            raise ConfigError(
                f"canonical job size {expected_size} differs from Wan model profile size {self.generation.get('size')}"
            )
        if dry_run:
            return
        generate_py = self._repo() / "generate.py"
        if not generate_py.is_file():
            raise ConfigError(f"Wan2.2 generate.py not found: {generate_py}")
        if not self._checkpoint().exists():
            raise ConfigError(f"Wan2.2 checkpoint not found: {self._checkpoint()}")

    def _model_prompt(self, job: dict[str, Any]) -> str:
        prompt = str(job["prompt"])
        negative = str(job.get("negative_prompt", "")).strip()
        strategy = str(self.generation.get("negative_prompt_strategy", "append"))
        if not negative or strategy == "ignore":
            return prompt
        if strategy == "append":
            return f"{prompt} Avoid: {negative}"
        raise ConfigError(f"unsupported Wan negative_prompt_strategy: {strategy}")

    def build_invocation(self, job: dict[str, Any], output_video: Path) -> Invocation:
        frame_num = int(self.generation.get("frame_num", job["generation"]["num_frames"]))
        command = [
            str(self.runtime.get("python", "python")),
            str(self._repo() / "generate.py"),
            "--task",
            str(self.generation.get("task", "i2v-A14B")),
            "--size",
            str(self.generation.get("size", "832*480")),
            "--ckpt_dir",
            str(self._checkpoint()),
            "--image",
            str(self._input_image(job)),
            "--prompt",
            self._model_prompt(job),
            "--save_file",
            str(output_video),
            "--base_seed",
            str(job["seed"]),
            "--frame_num",
            str(frame_num),
        ]
        options = {
            "sample_steps": "--sample_steps",
            "sample_shift": "--sample_shift",
            "sample_guide_scale": "--sample_guide_scale",
        }
        for key, flag in options.items():
            value = self.generation.get(key)
            if value is not None:
                command.extend([flag, str(value)])
        if bool(self.generation.get("offload_model", True)):
            command.extend(["--offload_model", "True"])
        if bool(self.generation.get("convert_model_dtype", True)):
            command.append("--convert_model_dtype")
        if bool(self.generation.get("t5_cpu", False)):
            command.append("--t5_cpu")
        environment = {
            "CUDA_VISIBLE_DEVICES": str(self.runtime.get("gpu_ids", "0")),
            "PYTORCH_CUDA_ALLOC_CONF": str(
                self.runtime.get("pytorch_cuda_alloc_conf", "expandable_segments:True,max_split_size_mb:128")
            ),
            "TOKENIZERS_PARALLELISM": "false",
        }
        return Invocation(command=command, cwd=self._repo(), environment=environment, output_video=output_video)

    def provenance(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_id,
            "model_id": self.model_config["model_id"],
            "model_revision": self.model_config.get("model_revision", "unspecified"),
            "model_repo": str(self._repo()),
            "checkpoint": str(self._checkpoint()),
            "generation": dict(self.generation),
        }

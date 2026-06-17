"""Optional Wan2.2 activation capture hook loaded via PYTHONPATH.

Python imports `sitecustomize` automatically at process startup when it is on
PYTHONPATH. This file stays inert unless WAN22_PROBE_CAPTURE=1 is set.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
from pathlib import Path
from typing import Any


if os.environ.get("WAN22_PROBE_CAPTURE") == "1":
    try:
        import numpy as np
        import torch
    except Exception as exc:  # pragma: no cover - only runs inside Wan2.2 env
        print(f"[wan22_probe] hook disabled: import failed: {exc}", flush=True)
    else:
        _lock = threading.Lock()
        _module_to_label: dict[int, str] = {}
        _module_classes: dict[int, str] = {}
        _features: list[np.ndarray] = []
        _records: list[dict[str, Any]] = []
        _call_counts: dict[str, int] = {}

        _feature_file = Path(os.environ.get("WAN22_PROBE_FEATURE_FILE", "wan22_probe_features.npz"))
        _job_id = os.environ.get("WAN22_PROBE_JOB_ID", "unknown_job")
        _class_regex = re.compile(os.environ.get("WAN22_PROBE_CLASS_REGEX", r"(Block|Layer)"), re.IGNORECASE)
        _module_regex = re.compile(os.environ.get("WAN22_PROBE_MODULE_REGEX", r"^wan\.modules\.model"), re.IGNORECASE)
        _max_records = int(os.environ.get("WAN22_PROBE_MAX_RECORDS", "4096"))
        _max_feature_dim = int(os.environ.get("WAN22_PROBE_MAX_FEATURE_DIM", "8192"))

        def _pick_tensor(output: Any) -> torch.Tensor | None:
            if torch.is_tensor(output):
                return output
            if isinstance(output, (tuple, list)):
                for item in output:
                    picked = _pick_tensor(item)
                    if picked is not None:
                        return picked
            if isinstance(output, dict):
                for item in output.values():
                    picked = _pick_tensor(item)
                    if picked is not None:
                        return picked
            return None

        def _should_capture(module: torch.nn.Module, tensor: torch.Tensor) -> bool:
            class_name = module.__class__.__name__
            module_name = module.__class__.__module__
            if not _module_regex.search(module_name):
                return False
            if not _class_regex.search(class_name):
                return False
            if tensor.ndim < 2:
                return False
            if tensor.shape[-1] < 64:
                return False
            return True

        def _summarize_tensor(tensor: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                x = tensor.detach()
                if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    x = x.float()
                else:
                    x = x.float()
                if x.ndim == 1:
                    mean = x
                    std = torch.zeros_like(x)
                else:
                    reduce_dims = tuple(range(x.ndim - 1))
                    mean = x.mean(dim=reduce_dims)
                    std = x.std(dim=reduce_dims, unbiased=False)
                feat = torch.cat([mean.flatten(), std.flatten()], dim=0)
                if feat.numel() > _max_feature_dim:
                    feat = feat[:_max_feature_dim]
                return feat.cpu().numpy().astype(np.float32, copy=False)

        def _forward_hook(module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            tensor = _pick_tensor(output)
            if tensor is None:
                return
            if not _should_capture(module, tensor):
                return
            with _lock:
                if len(_records) >= _max_records:
                    return
                module_id = id(module)
                if module_id not in _module_to_label:
                    label = f"block_{len(_module_to_label):03d}"
                    _module_to_label[module_id] = label
                    _module_classes[module_id] = f"{module.__class__.__module__}.{module.__class__.__name__}"
                label = _module_to_label[module_id]
                call_index = _call_counts.get(label, 0)
                _call_counts[label] = call_index + 1
                feature = _summarize_tensor(tensor)
                _features.append(feature)
                _records.append(
                    {
                        "job_id": _job_id,
                        "block_label": label,
                        "call_index": call_index,
                        "module_class": _module_classes[module_id],
                        "shape": list(tensor.shape),
                        "feature_dim": int(feature.shape[0]),
                    }
                )

        def _save() -> None:
            if not _records:
                print("[wan22_probe] no activation records captured", flush=True)
                return
            _feature_file.parent.mkdir(parents=True, exist_ok=True)
            max_dim = max(int(feat.shape[0]) for feat in _features)
            matrix = np.zeros((len(_features), max_dim), dtype=np.float32)
            feature_dims = np.zeros((len(_features),), dtype=np.int32)
            for idx, feat in enumerate(_features):
                matrix[idx, : feat.shape[0]] = feat
                feature_dims[idx] = feat.shape[0]
            np.savez_compressed(
                _feature_file,
                features=matrix,
                feature_dims=feature_dims,
                records_json=json.dumps(_records),
            )
            print(f"[wan22_probe] saved {len(_records)} records to {_feature_file}", flush=True)

        torch.nn.modules.module.register_module_forward_hook(_forward_hook)
        atexit.register(_save)
        print(
            "[wan22_probe] activation hook enabled; "
            f"class_regex={_class_regex.pattern}; module_regex={_module_regex.pattern}",
            flush=True,
        )

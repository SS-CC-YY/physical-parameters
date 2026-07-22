"""CUDA bootstrap helpers shared by preflight and inference entry points."""

from __future__ import annotations

from typing import Any


def initialize_cuda_before_xformers(torch_module: Any) -> dict[str, object]:
    """Create the CUDA context before xformers is imported.

    xformers queries the active device capability while its module graph is
    still being imported.  On some H20 worker states, letting that nested call
    perform PyTorch's first CUDA lazy-init can deadlock or segfault inside the
    dynamic loader.  Initialising at this explicit boundary avoids that import
    re-entrancy without changing model kernels or numerical settings.
    """

    cuda = torch_module.cuda
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available; SpatialTrackerV2 requires an NVIDIA GPU")
    cuda.init()
    logical_index = int(cuda.current_device())
    properties = cuda.get_device_properties(logical_index)
    return {
        "logical_device_index": logical_index,
        "device_name": str(properties.name),
    }

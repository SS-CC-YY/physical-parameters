#!/usr/bin/env python3
"""Persistent RGB-only SpatialTrackerV2 inference session.

The official quick-start process loads both networks for every invocation. This
adapter keeps them resident while the 27 benchmark videos are processed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import decord
import numpy as np
import torch
import torchvision.transforms as T

from cuda_bootstrap import initialize_cuda_before_xformers


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = Path(
    os.environ.get(
        "SPATIALTRACKERV2_ROOT",
        str(PACKAGE_ROOT / "upstream" / "SpaTrackerV2"),
    )
).expanduser().resolve()
sys.path.insert(0, str(UPSTREAM))

# Do not let xformers perform the process's first CUDA lazy-init from inside
# its own import graph.  That import-order path has produced a native-loader
# SIGSEGV on an otherwise healthy H20 CUDA context.
CUDA_BOOTSTRAP = initialize_cuda_before_xformers(torch)
print(
    "CUDA ready before xformers import: "
    f"logical_device={CUDA_BOOTSTRAP['logical_device_index']} "
    f"name={CUDA_BOOTSTRAP['device_name']}",
    flush=True,
)

from models.SpaTrackV2.models.predictor import Predictor  # noqa: E402
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track  # noqa: E402
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image  # noqa: E402


class SpatialTrackerSession:
    def __init__(self, track_mode: str = "offline", vo_points: int = 256) -> None:
        print("Loading SpatialTrackerV2 front-end once...", flush=True)
        self.front = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
        self.front.eval().to("cuda")
        model_id = (
            "Yuxihenry/SpatialTrackerV2-Offline"
            if track_mode == "offline"
            else "Yuxihenry/SpatialTrackerV2-Online"
        )
        print(f"Loading {model_id} once...", flush=True)
        self.tracker = Predictor.from_pretrained(model_id)
        self.tracker.spatrack.track_num = vo_points
        self.tracker.eval().to("cuda")
        print("SpatialTrackerV2 persistent session ready.", flush=True)

    def run(
        self,
        video_path: Path,
        queries_path: Path,
        output_path: Path,
        source_fps: float,
        frame_stride: int,
    ) -> None:
        reader = decord.VideoReader(str(video_path))
        source_frame_indices = np.arange(0, len(reader), frame_stride, dtype=np.int32)
        video = torch.from_numpy(reader.get_batch(source_frame_indices).asnumpy()).permute(0, 3, 1, 2).float()
        source_h, source_w = int(video.shape[2]), int(video.shape[3])
        video = preprocess_image(video)[None]
        processed_h, processed_w = int(video.shape[-2]), int(video.shape[-1])
        resize_scale_x = processed_w / float(source_w)
        resized_h = round(source_h * (518.0 / source_w) / 14) * 14
        resize_scale_y = resized_h / float(source_h)
        crop_top = max(0.0, (resized_h - processed_h) / 2.0)

        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = self.front(video.cuda() / 255.0)
            extrinsics = predictions["poses_pred"]
            intrinsics = predictions["intrs"]
            depth_map = predictions["points_map"][..., 2]
            depth_confidence = predictions["unc_metric"]

        depth = depth_map.squeeze().cpu().numpy()
        extrinsics_np = extrinsics.squeeze().cpu().numpy()
        intrinsics_np = intrinsics.squeeze().cpu().numpy()
        uncertainty_mask = depth_confidence.squeeze().cpu().numpy() > 0.5
        video = video.squeeze(0)

        query_data = dict(np.load(queries_path, allow_pickle=True))
        query_xy = np.asarray(query_data["query_xy_video"], dtype=np.float32).copy()
        query_xy[:, 0] *= resize_scale_x
        query_xy[:, 1] = query_xy[:, 1] * resize_scale_y - crop_top
        inside = (
            (query_xy[:, 0] >= 0)
            & (query_xy[:, 0] < processed_w)
            & (query_xy[:, 1] >= 0)
            & (query_xy[:, 1] < processed_h)
        )
        if not np.all(inside):
            raise ValueError(f"{int((~inside).sum())} queries are outside the preprocessed frame")
        query_xyt = np.concatenate([np.zeros((len(query_xy), 1), dtype=np.float32), query_xy], axis=1)

        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            (
                camera_to_world,
                intrinsics_out,
                point_map,
                confidence_depth,
                track3d,
                track2d,
                visibility,
                track_confidence,
                _video_out,
            ) = self.tracker.forward(
                video,
                depth=depth,
                intrs=intrinsics_np,
                extrs=extrinsics_np,
                queries=query_xyt,
                fps=1,
                full_point=False,
                iters_track=4,
                query_no_BA=True,
                fixed_cam=False,
                stage=1,
                unc_metric=uncertainty_mask,
                support_frame=len(video) - 1,
                replace_ratio=0.2,
            )

        camera_to_world_cpu = camera_to_world.detach().cpu()
        track3d_cpu = track3d[..., :3].detach().cpu()
        coords = (
            torch.einsum("tij,tnj->tni", camera_to_world_cpu[:, :3, :3], track3d_cpu)
            + camera_to_world_cpu[:, :3, 3][:, None, :]
        ).numpy()
        track2d_input = track2d.detach().cpu().numpy().copy()
        confidence_input = track_confidence.detach().cpu().numpy().copy()

        # Keep raw tracks at model input resolution; only dense maps are reduced for disk I/O.
        max_size = 336
        height, width = video.shape[2:]
        scale = min(max_size / height, max_size / width)
        video_save = video
        point_map_save = point_map
        confidence_depth_save = confidence_depth
        intrinsics_save = intrinsics_out
        if scale < 1:
            new_h, new_w = int(height * scale), int(width * scale)
            video_save = T.Resize((new_h, new_w))(video_save)
            point_map_save = T.Resize((new_h, new_w))(point_map_save)
            confidence_depth_save = T.Resize((new_h, new_w))(confidence_depth_save)
            intrinsics_save = intrinsics_save.clone()
            intrinsics_save[:, :2, :] *= scale

        depth_save = point_map_save[:, 2, ...].clone()
        depth_save[confidence_depth_save < 0.5] = 0
        output = {
            "coords": coords,
            "extrinsics": torch.inverse(camera_to_world_cpu).numpy(),
            "intrinsics": intrinsics_save.detach().cpu().numpy(),
            "depths": depth_save.detach().cpu().numpy(),
            "video": video_save.detach().cpu().numpy() / 255.0,
            "visibs": visibility.detach().cpu().numpy(),
            "track2d_input": track2d_input,
            "track_confidence": confidence_input,
            "query_xyt_processed": query_xyt,
            "source_frame_indices": source_frame_indices,
            "source_fps": np.asarray(source_fps, dtype=np.float32),
            "source_frame_size_hw": np.asarray([source_h, source_w], dtype=np.int32),
            "processed_frame_size_hw": np.asarray([processed_h, processed_w], dtype=np.int32),
            "preprocess_scale_xy": np.asarray([resize_scale_x, resize_scale_y], dtype=np.float32),
            "preprocess_crop_top": np.asarray(crop_top, dtype=np.float32),
            "unc_metric": confidence_depth_save.detach().cpu().numpy(),
        }
        for key, value in query_data.items():
            output[f"query_meta_{key}"] = value
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **output)
        del predictions, depth_map, depth_confidence, point_map, track3d, track2d
        torch.cuda.empty_cache()

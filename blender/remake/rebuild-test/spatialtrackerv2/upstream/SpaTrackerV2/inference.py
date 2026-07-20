import pycolmap
from models.SpaTrackV2.models.predictor import Predictor
import yaml
import easydict
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import io
import moviepy.editor as mp
from models.SpaTrackV2.utils.visualizer import Visualizer
import tqdm
from models.SpaTrackV2.models.utils import get_points_on_a_grid
import glob
from rich import print
import argparse
import decord
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_type", type=str, default="RGBD")
    parser.add_argument("--data_dir", type=str, default="assets/example0")
    parser.add_argument("--video_name", type=str, default="snowboard")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--video_path", type=str, default=None,
                        help="Explicit MP4 path; overrides data_dir/video_name.mp4")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Optional binary mask in original video coordinates")
    parser.add_argument("--queries_path", type=str, default=None,
                        help="NPZ containing query_xy_video in original video coordinates")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--source_fps", type=float, default=24.0)
    parser.add_argument("--no_viz", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_dir = args.output_dir or (args.data_dir + "/results")
    # fps
    fps = int(args.fps)
    mask_dir = args.mask_path or (args.data_dir + f"/{args.video_name}.png")
    
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")

    if args.data_type == "RGBD":
        npz_dir = args.data_dir + f"/{args.video_name}.npz"
        data_npz_load = dict(np.load(npz_dir, allow_pickle=True))
        #TODO: tapip format
        video_tensor = data_npz_load["video"] * 255
        video_tensor = torch.from_numpy(video_tensor)
        video_tensor = video_tensor[::fps]
        depth_tensor = data_npz_load["depths"]
        depth_tensor = depth_tensor[::fps]
        intrs = data_npz_load["intrinsics"]
        intrs = intrs[::fps]
        extrs = np.linalg.inv(data_npz_load["extrinsics"])
        extrs = extrs[::fps]
        unc_metric = None
    elif args.data_type == "RGB":
        vid_dir = args.video_path or os.path.join(args.data_dir, f"{args.video_name}.mp4")
        video_reader = decord.VideoReader(vid_dir)
        source_frame_indices = np.arange(0, len(video_reader), fps, dtype=np.int32)
        video_tensor = torch.from_numpy(video_reader.get_batch(source_frame_indices).asnumpy()).permute(0, 3, 1, 2)  # Convert to tensor and permute to (N, C, H, W)
        source_h, source_w = int(video_tensor.shape[2]), int(video_tensor.shape[3])
        video_tensor = video_tensor.float()

        # process the image tensor
        video_tensor = preprocess_image(video_tensor)[None]
        processed_h, processed_w = int(video_tensor.shape[-2]), int(video_tensor.shape[-1])
        resize_scale_x = processed_w / float(source_w)
        resized_h = round(source_h * (518.0 / source_w) / 14) * 14
        resize_scale_y = resized_h / float(source_h)
        crop_top = max(0.0, (resized_h - processed_h) / 2.0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = vggt4track_model(video_tensor.cuda()/255)
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
        
        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
        extrs = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_tensor.squeeze()
        #NOTE: 20% of the depth is not reliable
        # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

        data_npz_load = {}
    
    if os.path.exists(mask_dir):
        mask_files = mask_dir
        mask = cv2.imread(mask_files)
        mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
        mask = mask.sum(axis=-1)>0
    else:
        mask = np.ones_like(video_tensor[0,0].numpy())>0
        
    # get all data pieces
    viz = not args.no_viz
    os.makedirs(out_dir, exist_ok=True)
        
    # with open(cfg_dir, "r") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = easydict.EasyDict(cfg)
    # cfg.out_dir = out_dir
    # cfg.model.track_num = args.vo_points
    # print(f"Downloading model from HuggingFace: {cfg.ckpts}")
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    # config the model; the track_num is the number of points in the grid
    model.spatrack.track_num = args.vo_points
    
    model.eval()
    model.to("cuda")
    viser = Visualizer(save_dir=out_dir, grayscale=True, 
                     fps=10, pad_value=0, tracks_leave_trace=5)
    
    grid_size = args.grid_size

    # get frame H W
    if video_tensor is  None:
        cap = cv2.VideoCapture(video_path)
        frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        frame_H, frame_W = video_tensor.shape[2:]
    if args.queries_path:
        query_data = dict(np.load(args.queries_path, allow_pickle=True))
        query_xy_video = np.asarray(query_data["query_xy_video"], dtype=np.float32)
        query_xy_processed = query_xy_video.copy()
        query_xy_processed[:, 0] *= resize_scale_x
        query_xy_processed[:, 1] = query_xy_processed[:, 1] * resize_scale_y - crop_top
        inside = (
            (query_xy_processed[:, 0] >= 0)
            & (query_xy_processed[:, 0] < frame_W)
            & (query_xy_processed[:, 1] >= 0)
            & (query_xy_processed[:, 1] < frame_H)
        )
        if not np.all(inside):
            raise ValueError(f"{int((~inside).sum())} explicit queries fall outside the preprocessed frame")
        grid_pts = torch.from_numpy(query_xy_processed[None])
    else:
        query_data = {}
        grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    
    # Sample mask values at grid points and filter out points where mask=0
    if (not args.queries_path) and os.path.exists(mask_dir):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]
    
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 
        
        # resize the results to avoid too large I/O Burden
        # depth and image, the maximum side is 336
        max_size = 336
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs[:,:2,:] = intrs[:,:2,:] * scale
            if depth_tensor is not None:
                if isinstance(depth_tensor, torch.Tensor):
                    depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
                else:
                    depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))

        track2d_input = track2d_pred.detach().cpu().numpy().copy()
        conf_input = conf_pred.detach().cpu().numpy().copy()

        if viz:
            viser.visualize(video=video[None],
                                tracks=track2d_pred[None][...,:2],
                                visibility=vis_pred[None],filename="test")

        # save as the tapip3d format   
        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        depth_save = point_map[:,2,...]
        depth_save[conf_depth<0.5] = 0
        data_npz_load["depths"] = depth_save.cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["track2d_input"] = track2d_input
        data_npz_load["track_confidence"] = conf_input
        data_npz_load["query_xyt_processed"] = query_xyt
        data_npz_load["source_frame_indices"] = source_frame_indices
        data_npz_load["source_fps"] = np.asarray(args.source_fps, dtype=np.float32)
        data_npz_load["source_frame_size_hw"] = np.asarray([source_h, source_w], dtype=np.int32)
        data_npz_load["processed_frame_size_hw"] = np.asarray([processed_h, processed_w], dtype=np.int32)
        data_npz_load["preprocess_scale_xy"] = np.asarray([resize_scale_x, resize_scale_y], dtype=np.float32)
        data_npz_load["preprocess_crop_top"] = np.asarray(crop_top, dtype=np.float32)
        for key, value in query_data.items():
            data_npz_load[f"query_meta_{key}"] = value
        data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
        np.savez(os.path.join(out_dir, f'result.npz'), **data_npz_load)

        print(f"Results saved to {out_dir}.\nTo visualize them with tapip3d, run: [bold yellow]python tapip3d_viz.py {out_dir}/result.npz[/bold yellow]")

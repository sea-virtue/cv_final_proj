"""
pipeline/vggt_inf_improvements.py -- VGGT inference with lightweight post-processing.

Implemented improvement:
    Depthmap + pointmap confidence fusion.

Usage:
    python pipeline/vggt_inf_improvements.py --dataset 数据1-人体
    python pipeline/vggt_inf_improvements.py --dataset 数据3-场景.mp4

Output:
    output/<dataset>/predictions.npz

The output keeps the same interface as vggt_inference.py. For downstream 3DGS,
world_points_from_depth is replaced by the fused point map by default, while the
raw depth-unprojected points are kept as world_points_from_depth_raw.
"""

import argparse
import gc
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.utils import get_dataset_list, get_output_dir, load_dataset_images_and_masks
from pipeline.vggt_inference import (
    _is_video_path,
    apply_mask_filter,
    extract_video_frames,
    load_model,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vggt"))

from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def _load_mask_stack(mask_paths, shape):
    """Load masks as (S, H, W), matching VGGT output resolution."""
    if not mask_paths:
        return None
    s, h, w = shape
    masks = []
    for mask_path in mask_paths[:s]:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append((mask > 128).astype(np.float32))
    if len(masks) != s:
        return None
    return np.stack(masks, axis=0)


def _normalise_conf(conf):
    """Map arbitrary positive confidence values to [0, 1] robustly."""
    conf = np.asarray(conf, dtype=np.float32)
    finite = np.isfinite(conf)
    out = np.zeros_like(conf, dtype=np.float32)
    if not finite.any():
        return out

    valid = conf[finite]
    lo = np.percentile(valid, 1.0)
    hi = np.percentile(valid, 99.0)
    if hi <= lo + 1e-8:
        out[finite] = 1.0
        return out

    out[finite] = np.clip((conf[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def fuse_depth_and_pointmap(predictions, mask_paths=None, eps=1e-6):
    """
    Fuse VGGT depth-unprojected world points and direct pointmap predictions.

    The fused result is stored in world_points_from_depth so existing 3DGS code
    consumes it without changes. Original depth points are kept separately.
    """
    required = {"world_points_from_depth", "world_points", "depth_conf", "world_points_conf"}
    missing = sorted(required - set(predictions.keys()))
    if missing:
        print(f"[VGGT Improvements] Missing {missing}; skip point/depth fusion.")
        predictions["improvement_method"] = np.array("none_missing_pointmap")
        return predictions

    depth_points = predictions["world_points_from_depth"].astype(np.float32)
    point_points = predictions["world_points"].astype(np.float32)
    depth_conf = predictions["depth_conf"].astype(np.float32)
    point_conf = predictions["world_points_conf"].astype(np.float32)

    if depth_points.shape != point_points.shape or depth_conf.shape != point_conf.shape:
        print("[VGGT Improvements] Point/depth shapes differ; skip point/depth fusion.")
        predictions["improvement_method"] = np.array("none_shape_mismatch")
        return predictions

    mask_stack = _load_mask_stack(mask_paths, depth_conf.shape)
    if mask_stack is not None:
        depth_conf = depth_conf * mask_stack
        point_conf = point_conf * mask_stack

    depth_weight = _normalise_conf(depth_conf)
    point_weight = _normalise_conf(point_conf)
    alpha = depth_weight / (depth_weight + point_weight + eps)
    alpha = alpha[..., None].astype(np.float32)

    valid_depth = np.isfinite(depth_points).all(axis=-1, keepdims=True)
    valid_point = np.isfinite(point_points).all(axis=-1, keepdims=True)
    fused = alpha * depth_points + (1.0 - alpha) * point_points
    fused = np.where(valid_depth & valid_point, fused, depth_points)
    fused = np.where(valid_depth, fused, point_points)

    predictions["world_points_from_depth_raw"] = depth_points
    predictions["depth_conf_raw"] = predictions["depth_conf"].astype(np.float32)
    predictions["world_points_conf_raw"] = predictions["world_points_conf"].astype(np.float32)
    predictions["world_points_from_depth"] = fused.astype(np.float32)
    predictions["depth_conf"] = depth_conf.astype(np.float32)
    predictions["world_points_conf"] = point_conf.astype(np.float32)
    predictions["fusion_alpha"] = alpha[..., 0].astype(np.float32)
    predictions["improvement_method"] = np.array("depth_point_confidence_fusion")

    print(
        "[VGGT Improvements] Fused depth/point branches: "
        f"alpha mean={float(alpha.mean()):.3f}, "
        f"depth_conf mean={float(depth_conf.mean()):.3f}, "
        f"point_conf mean={float(point_conf.mean()):.3f}"
    )
    return predictions


def run_inference_improved(
    dataset_path,
    model,
    device="cuda",
    output_dir=None,
    video_max_frames=16,
    video_frame_stride=None,
    video_all_frames=False,
    max_model_frames=128,
    allow_large_sequence=False,
    frames_chunk_size=8,
):
    if _is_video_path(dataset_path):
        if output_dir is None:
            raise ValueError("output_dir is required for video datasets.")
        rgb_paths = extract_video_frames(
            dataset_path,
            output_dir,
            max_frames=video_max_frames,
            frame_stride=video_frame_stride,
            all_frames=video_all_frames,
        )
        mask_paths = []
    else:
        rgb_paths, mask_paths, _ = load_dataset_images_and_masks(dataset_path)

    if not rgb_paths:
        raise ValueError(f"No input images found in {dataset_path}")
    if len(rgb_paths) > max_model_frames and not allow_large_sequence:
        raise ValueError(
            f"VGGT would receive {len(rgb_paths)} frames at once, which is likely to OOM.\n"
            f"Use fewer frames, e.g. --video-max-frames 32 or --video-frame-stride 10.\n"
            f"To override this guard, pass --allow-large-sequence and optionally increase "
            f"--max-model-frames."
        )

    print(f"[VGGT Improvements] Found {len(rgb_paths)} images in {dataset_path}")
    images = load_and_preprocess_images(rgb_paths).to(device)
    print(f"[VGGT Improvements] Image shape: {tuple(images.shape)}")

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    else:
        autocast_ctx = torch.no_grad()

    print(f"[VGGT Improvements] Running VGGT inference on {device}...")
    with torch.no_grad():
        if device == "cuda":
            with autocast_ctx:
                predictions = model(images, frames_chunk_size=frames_chunk_size)
        else:
            predictions = model(images, frames_chunk_size=frames_chunk_size)

    print("[VGGT Improvements] Converting pose encoding to extrinsics/intrinsics...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions.pop("pose_enc_list", None)

    print("[VGGT Improvements] Computing depth-unprojected world points...")
    depth_map = predictions["depth"]
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )

    predictions = apply_mask_filter(predictions, mask_paths)
    predictions = fuse_depth_and_pointmap(predictions, mask_paths)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return predictions


def main():
    parser = argparse.ArgumentParser(description="VGGT inference with lightweight improvements")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video-max-frames", type=int, default=16)
    parser.add_argument("--video-frame-stride", type=int, default=None)
    parser.add_argument("--video-all-frames", action="store_true")
    parser.add_argument("--max-model-frames", type=int, default=128)
    parser.add_argument("--allow-large-sequence", action="store_true")
    parser.add_argument("--frames-chunk-size", type=int, default=8,
                        help="Frames per chunk for VGGT depth/point heads.")
    parser.add_argument("--output-name", default="predictions.npz",
                        help="Output filename under output/<dataset>/; default matches the original pipeline.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    datasets = get_dataset_list()
    if args.dataset not in datasets:
        print(f"Dataset '{args.dataset}' not found. Available: {list(datasets.keys())}")
        sys.exit(1)

    dataset_path = datasets[args.dataset]
    output_dir = get_output_dir(args.dataset)

    model = load_model(args.device)
    predictions = run_inference_improved(
        dataset_path,
        model,
        args.device,
        output_dir,
        video_max_frames=args.video_max_frames,
        video_frame_stride=args.video_frame_stride,
        video_all_frames=args.video_all_frames,
        max_model_frames=args.max_model_frames,
        allow_large_sequence=args.allow_large_sequence,
        frames_chunk_size=args.frames_chunk_size,
    )

    save_path = os.path.join(output_dir, args.output_name)
    print(f"[VGGT Improvements] Saving predictions to {save_path}")
    np.savez(save_path, **predictions)
    print(f"[VGGT Improvements] Done. File size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")

    print("\n--- Improved Prediction Summary ---")
    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    print(f"  Frames: {predictions['extrinsic'].shape[0]}")
    if "improvement_method" in predictions:
        print(f"  Method: {str(predictions['improvement_method'])}")


if __name__ == "__main__":
    main()

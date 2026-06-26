"""
pipeline/vggt_inference.py — VGGT 模型推理封装
用法:
    python pipeline/vggt_inference.py --dataset 数据1-人体
    python pipeline/vggt_inference.py --dataset 数据2-人体
    python pipeline/vggt_inference.py --dataset 数据3-场景.mp4

输出: output/<dataset>/predictions.npz
"""

import os
import sys
import argparse
import glob
import time
import gc
import shutil
import json

# 确保能 import pipeline 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

# 确保能 import vggt 模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from pipeline.utils import load_dataset_images_and_masks, get_output_dir, get_dataset_list, VIDEO_EXTENSIONS


def _is_video_path(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def extract_video_frames(video_path: str, output_dir: str, max_frames: int = 16,
                         frame_stride: int | None = None,
                         all_frames: bool = False) -> list:
    """Extract frames from a video into output_dir/frames."""
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read frame count from video: {video_path}")

    if all_frames:
        frame_indices = np.arange(total, dtype=np.int32)
        mode = "all"
    elif frame_stride is not None:
        if frame_stride <= 0:
            cap.release()
            raise ValueError("--video-frame-stride must be a positive integer.")
        frame_indices = np.arange(0, total, frame_stride, dtype=np.int32)
        mode = f"stride_{frame_stride}"
    else:
        if max_frames <= 0:
            cap.release()
            raise ValueError("--video-max-frames must be a positive integer.")
        frame_indices = np.linspace(0, total - 1, num=min(max_frames, total), dtype=np.int32)
        mode = f"even_{max_frames}"

    manifest_path = os.path.join(frames_dir, "manifest.json")
    expected_manifest = {
        "video_path": os.path.abspath(video_path),
        "mode": mode,
        "total_frames": total,
        "frame_indices": frame_indices.astype(int).tolist(),
    }

    existing = sorted(glob.glob(os.path.join(frames_dir, "rgb_*.png")))
    if existing and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            cached_manifest = json.load(f)
        if cached_manifest == expected_manifest and len(existing) == len(frame_indices):
            cap.release()
            print(f"[VGGT Inference] Reusing {len(existing)} extracted frames from {frames_dir}")
            return existing

    if existing:
        print(f"[VGGT Inference] Re-extracting video frames with mode '{mode}'.")
        shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)

    frame_paths = []
    print(f"[VGGT Inference] Extracting {len(frame_indices)} frames from {video_path} ({mode})")

    for out_idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"  Warning: failed to read video frame {frame_idx}, skip.")
            continue
        frame_path = os.path.join(frames_dir, f"rgb_{out_idx:04d}.png")
        cv2.imwrite(frame_path, frame_bgr)
        frame_paths.append(frame_path)

    cap.release()
    if not frame_paths:
        raise ValueError(f"No frames extracted from video: {video_path}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(expected_manifest, f, ensure_ascii=False, indent=2)
    return frame_paths


def load_model(device: str = "cuda") -> VGGT:
    """加载 VGGT 模型并移动到指定设备。"""
    print(f"[VGGT Inference] Loading VGGT-1B model on {device}...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(_URL, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    print("[VGGT Inference] Model loaded.")
    return model


def apply_mask_filter(predictions: dict, mask_paths: list) -> dict:
    """
    利用 mask 图像过滤背景区域的深度和点云。
    mask 中白色(255)区域为前景，黑色(0)为背景。
    将深度置信度在背景区域置零。
    """
    if not mask_paths:
        print("[VGGT Inference] No mask images found, skipping mask filtering.")
        return predictions

    print(f"[VGGT Inference] Applying mask filtering with {len(mask_paths)} masks...")
    depth_conf = predictions["depth_conf"]  # (S, H, W)
    S, H, W = depth_conf.shape

    for i, msk_path in enumerate(mask_paths):
        if i >= S:
            break
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  Warning: cannot read {msk_path}, skip.")
            continue
        # resize to match model output resolution
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = cv2.resize(mask, (W, H))
        # 白色=前景(255), 背景(0)区域的置信度置零
        mask_binary = (mask > 128).astype(np.float32)
        depth_conf[i] = depth_conf[i] * mask_binary

    predictions["depth_conf"] = depth_conf
    print("[VGGT Inference] Mask filtering done.")
    return predictions


def run_inference(dataset_path: str, model: VGGT, device: str = "cuda",
                  output_dir: str | None = None,
                  video_max_frames: int = 16,
                  video_frame_stride: int | None = None,
                  video_all_frames: bool = False,
                  max_model_frames: int = 128,
                  allow_large_sequence: bool = False) -> dict:
    """
    对单个数据集运行 VGGT 推理，返回 predictions 字典。
    """
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
        msk_paths = []
    else:
        rgb_paths, msk_paths, _ = load_dataset_images_and_masks(dataset_path)

    if not rgb_paths:
        raise ValueError(f"No input images found in {dataset_path}")
    if len(rgb_paths) > max_model_frames and not allow_large_sequence:
        raise ValueError(
            f"VGGT would receive {len(rgb_paths)} frames at once, which is likely to OOM.\n"
            f"Use fewer frames, e.g. --video-max-frames 32 or --video-frame-stride 10.\n"
            f"To override this guard, pass --allow-large-sequence and optionally increase "
            f"--max-model-frames."
        )

    print(f"[VGGT Inference] Found {len(rgb_paths)} images in {dataset_path}")

    # 加载并预处理
    start = time.time()
    images = load_and_preprocess_images(rgb_paths).to(device)
    print(f"[VGGT Inference] Image shape: {images.shape}, load time: {time.time()-start:.1f}s")

    # 推理
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    else:
        dtype = torch.float32
        autocast_ctx = torch.no_grad()  # CPU: no autocast needed

    print(f"[VGGT Inference] Running inference on {device}...")
    start = time.time()
    with torch.no_grad():
        if device == "cuda":
            with autocast_ctx:
                predictions = model(images)
        else:
            predictions = model(images)
    print(f"[VGGT Inference] Inference time: {time.time()-start:.1f}s")

    # 位姿编码 → 外参/内参矩阵
    print("[VGGT Inference] Converting pose encoding to extrinsics/intrinsics...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # 所有 tensor → numpy
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions.pop("pose_enc_list", None)

    # 从深度图计算世界坐标点云
    print("[VGGT Inference] Computing world points from depth...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )
    predictions["world_points_from_depth"] = world_points

    # Mask 过滤
    predictions = apply_mask_filter(predictions, msk_paths)

    # 清理 GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return predictions


def main():
    parser = argparse.ArgumentParser(description="VGGT Inference Pipeline")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name or video file (e.g. 数据1-人体, 数据2-人体, 数据3-场景.mp4)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--video-max-frames", type=int, default=16,
                        help="For video datasets, extract this many evenly spaced frames by default.")
    parser.add_argument("--video-frame-stride", type=int, default=None,
                        help="For video datasets, extract one frame every N frames.")
    parser.add_argument("--video-all-frames", action="store_true",
                        help="For video datasets, extract every frame. This can be very memory intensive.")
    parser.add_argument("--max-model-frames", type=int, default=128,
                        help="Abort before VGGT inference if more than this many frames would be used.")
    parser.add_argument("--allow-large-sequence", action="store_true",
                        help="Disable the frame-count safety guard. This can easily cause CUDA OOM.")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    # 定位数据集
    datasets = get_dataset_list()
    if args.dataset not in datasets:
        print(f"Dataset '{args.dataset}' not found. Available: {list(datasets.keys())}")
        sys.exit(1)

    dataset_path = datasets[args.dataset]
    output_dir = get_output_dir(args.dataset)

    # 加载模型并推理
    model = load_model(args.device)
    predictions = run_inference(
        dataset_path,
        model,
        args.device,
        output_dir,
        video_max_frames=args.video_max_frames,
        video_frame_stride=args.video_frame_stride,
        video_all_frames=args.video_all_frames,
        max_model_frames=args.max_model_frames,
        allow_large_sequence=args.allow_large_sequence,
    )

    # 保存
    save_path = os.path.join(output_dir, "predictions.npz")
    print(f"[VGGT Inference] Saving predictions to {save_path}")
    np.savez(save_path, **predictions)
    print(f"[VGGT Inference] Done. File size: {os.path.getsize(save_path)/1024/1024:.1f} MB")

    # 打印关键信息
    print(f"\n--- Prediction Summary ---")
    for k, v in predictions.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    print(f"  Frames: {predictions['extrinsic'].shape[0]}")


if __name__ == "__main__":
    main()

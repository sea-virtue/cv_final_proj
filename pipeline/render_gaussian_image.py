"""
Render a trained Gaussian point set to a PNG image from a selected camera view.

Example:
    python pipeline/render_gaussian_image.py --dataset 数据1-人体 --frame 0
    python pipeline/render_gaussian_image.py --dataset 数据1-人体 --frame 0 --yaw-deg 20
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.gaussian_splatting import render_gaussians
from pipeline.gradio_viewer import _read_gaussian_ply
from pipeline.utils import get_dataset_list, get_output_dir


def _load_camera(output_dir, frame_idx, use_ba=True):
    pred_path = os.path.join(output_dir, "predictions.npz")
    ba_path = os.path.join(output_dir, "ba_result.npz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing predictions.npz: {pred_path}")

    pred = np.load(pred_path, allow_pickle=True)
    images = pred["images"]
    intrinsics = pred["intrinsic"]
    extrinsics = pred["extrinsic"]

    if use_ba and os.path.exists(ba_path):
        ba = np.load(ba_path, allow_pickle=True)
        extrinsics = ba["extrinsic_opt"]
        ba.close()

    if frame_idx < 0 or frame_idx >= extrinsics.shape[0]:
        raise ValueError(f"--frame must be in [0, {extrinsics.shape[0] - 1}]")

    if images.ndim == 4 and images.shape[1] == 3:
        height, width = images.shape[2], images.shape[3]
    else:
        height, width = images.shape[1], images.shape[2]

    extrinsic = extrinsics[frame_idx]
    intrinsic = intrinsics[frame_idx]
    pred.close()
    return extrinsic, intrinsic, height, width


def _orbit_extrinsic(extrinsic, points, yaw_deg=0.0):
    if abs(yaw_deg) < 1e-8:
        return extrinsic

    ext4 = np.eye(4, dtype=np.float32)
    ext4[:3, :4] = extrinsic
    c2w = np.linalg.inv(ext4)

    center = np.median(points, axis=0)
    yaw = np.deg2rad(yaw_deg)
    rot = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    to_center = np.eye(4, dtype=np.float32)
    to_center[:3, 3] = -center
    from_center = np.eye(4, dtype=np.float32)
    from_center[:3, 3] = center

    c2w = from_center @ rot @ to_center @ c2w
    return np.linalg.inv(c2w)[:3, :4].astype(np.float32)


def render_dataset_view(dataset, frame_idx, yaw_deg=0.0, use_ba=True,
                        max_gaussians=None, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat rendering, but torch.cuda.is_available() is False.")

    output_dir = get_output_dir(dataset)
    ply_path = os.path.join(output_dir, "gaussians.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Missing gaussians.ply: {ply_path}")

    means, colors, opacities, scales, quats = _read_gaussian_ply(ply_path)
    if max_gaussians is not None and len(means) > max_gaussians:
        idx = np.random.choice(len(means), max_gaussians, replace=False)
        means = means[idx]
        colors = colors[idx]
        opacities = opacities[idx]
        scales = scales[idx]
        quats = quats[idx]

    extrinsic, intrinsic, height, width = _load_camera(output_dir, frame_idx, use_ba=use_ba)
    extrinsic = _orbit_extrinsic(extrinsic, means, yaw_deg=yaw_deg)

    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :4] = torch.from_numpy(extrinsic).to(device=device, dtype=torch.float32)
    K = torch.from_numpy(intrinsic).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        image = render_gaussians(
            torch.from_numpy(means).to(device=device, dtype=torch.float32),
            torch.from_numpy(quats).to(device=device, dtype=torch.float32),
            torch.from_numpy(scales).to(device=device, dtype=torch.float32),
            torch.from_numpy(opacities).to(device=device, dtype=torch.float32),
            torch.from_numpy(colors).to(device=device, dtype=torch.float32),
            viewmat,
            K,
            height,
            width,
            bg_color=torch.zeros(3, device=device),
        )

    return image.clamp(0.0, 1.0).detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Render trained 3D Gaussians to a PNG image.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--frame", type=int, default=0, help="Camera frame index to render from.")
    parser.add_argument("--yaw-deg", type=float, default=0.0,
                        help="Optional orbit yaw angle around the Gaussian median center.")
    parser.add_argument("--max-gaussians", type=int, default=None,
                        help="Randomly sample at most this many Gaussians before rendering.")
    parser.add_argument("--no-ba", dest="use_ba", action="store_false",
                        help="Use original VGGT camera instead of BA-optimized camera.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.set_defaults(use_ba=True)
    args = parser.parse_args()

    datasets = get_dataset_list()
    if args.dataset not in datasets:
        print(f"Dataset '{args.dataset}' not found. Available: {list(datasets.keys())}")
        sys.exit(1)

    output_dir = get_output_dir(args.dataset)
    render_dir = os.path.join(output_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    image = render_dataset_view(
        args.dataset,
        args.frame,
        yaw_deg=args.yaw_deg,
        use_ba=args.use_ba,
        max_gaussians=args.max_gaussians,
        device=args.device,
    )

    suffix = f"frame_{args.frame:04d}"
    if abs(args.yaw_deg) > 1e-8:
        suffix += f"_yaw_{args.yaw_deg:+.1f}".replace(".", "p")
    save_path = os.path.join(render_dir, f"render_{suffix}.png")
    cv2.imwrite(save_path, cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"[Render] Saved {save_path}")


if __name__ == "__main__":
    main()

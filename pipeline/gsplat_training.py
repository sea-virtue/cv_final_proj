"""
pipeline/gsplat_training.py -- 3DGS training using gsplat's official API.

Usage:
    python pipeline/gsplat_training.py --dataset 数据1-人体

This script keeps the project pipeline interface, but uses gsplat's official
rasterizer, losses, DefaultStrategy densification/pruning, and exporter.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_GSPLAT_ROOT = PROJECT_ROOT / "gsplat"

sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

from pipeline.utils import get_dataset_list, get_output_dir, load_dataset_images_and_masks


C0 = 0.28209479177387814


def configure_gsplat_import(use_local: bool):
    """Configure import path before importing gsplat modules."""
    if use_local:
        torch_ext_dir = PROJECT_ROOT / "output" / "torch_extensions"
        torch_ext_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_ext_dir))
        sys.path.insert(0, str(LOCAL_GSPLAT_ROOT))


def import_gsplat_modules():
    from gsplat import DefaultStrategy, export_splats, rasterization

    try:
        from gsplat.losses import l1_loss, ssim_loss
    except ImportError:
        l1_loss = lambda pred, target: F.l1_loss(pred, target, reduction="none")

        def ssim_loss(img1, img2, window_size=11):
            channel = img1.shape[1]
            pad = window_size // 2
            x = torch.arange(window_size, device=img1.device, dtype=img1.dtype)
            gauss = torch.exp(-((x - pad) ** 2) / (2 * 1.5**2))
            gauss = gauss / gauss.sum()
            window = (gauss[:, None] @ gauss[None, :]).view(1, 1, window_size, window_size)
            window = window.expand(channel, 1, window_size, window_size).contiguous()
            mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
            mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
            sigma1 = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1 * mu1
            sigma2 = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2 * mu2
            sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1 * mu2
            c1, c2 = 0.01**2, 0.03**2
            ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
                (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 + sigma2 + c2)
            )
            return 1.0 - ssim_map.mean()

    return DefaultStrategy, export_splats, rasterization, l1_loss, ssim_loss


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target)
    if float(mse) <= 1e-12:
        return 100.0
    return float(20.0 * torch.log10(1.0 / torch.sqrt(mse)))


def as_nhwc_float(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    images = images.astype(np.float32)
    if images.max() > 2.0:
        images = images / 255.0
    return np.clip(images, 0.0, 1.0)


def load_dataset_masks(dataset: str, image_shape):
    datasets = get_dataset_list()
    dataset_path = datasets.get(dataset)
    if not dataset_path or not os.path.isdir(dataset_path):
        return None

    _, mask_paths, _ = load_dataset_images_and_masks(dataset_path)
    if not mask_paths:
        return None

    s, h, w = image_shape[:3]
    masks = []
    for mask_path in mask_paths[:s]:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append((mask > 128).astype(np.float32))

    if len(masks) != s:
        print(f"[gsplat] Warning: found {len(masks)} usable masks for {s} frames; mask loss disabled.")
        return None
    return np.stack(masks, axis=0)[..., None]


def unproject_depth_with_extrinsics(depth, extrinsics, intrinsics):
    """Unproject depth maps with world-to-camera extrinsics."""
    depth = depth.squeeze(-1).astype(np.float32)
    s, h, w = depth.shape
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    world_points = np.empty((s, h, w, 3), dtype=np.float32)
    for i in range(s):
        z = depth[i]
        k = intrinsics[i]
        x = (xs - k[0, 2]) * z / (k[0, 0] + 1e-8)
        y = (ys - k[1, 2]) * z / (k[1, 1] + 1e-8)
        cam = np.stack([x, y, z], axis=-1)
        r = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        world_points[i] = (cam - t) @ r
    return world_points


def load_pipeline_data(dataset: str, use_ba: bool, use_masks: bool, reproject_depth: bool):
    datasets = get_dataset_list()
    if dataset not in datasets:
        raise FileNotFoundError(f"Dataset '{dataset}' not found. Available: {list(datasets.keys())}")

    output_dir = get_output_dir(dataset)
    pred_path = os.path.join(output_dir, "predictions.npz")
    ba_path = os.path.join(output_dir, "ba_result.npz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing predictions: {pred_path}")

    pred = np.load(pred_path, allow_pickle=True)
    images = as_nhwc_float(pred["images"])
    world_points = pred["world_points_from_depth"].astype(np.float32)
    depth = pred["depth"].astype(np.float32) if "depth" in pred else None
    intrinsics = pred["intrinsic"].astype(np.float32)
    extrinsics = pred["extrinsic"].astype(np.float32)
    depth_conf = pred["depth_conf"].astype(np.float32) if "depth_conf" in pred else None

    pose_source = "VGGT"
    if use_ba and os.path.exists(ba_path):
        ba = np.load(ba_path, allow_pickle=True)
        extrinsics = ba["extrinsic_opt"].astype(np.float32)
        if "intrinsic" in ba:
            intrinsics = ba["intrinsic"].astype(np.float32)
        ba.close()
        pose_source = "BA"

    if reproject_depth and depth is not None and pose_source == "BA":
        world_points = unproject_depth_with_extrinsics(depth, extrinsics, intrinsics)
        print("[gsplat] Reprojected VGGT depth with BA poses for dense initialization.")

    pred.close()
    masks = load_dataset_masks(dataset, images.shape) if use_masks else None
    if masks is not None:
        fg_ratio = float(masks.mean())
        print(f"[gsplat] Loaded masks: foreground ratio {fg_ratio * 100:.1f}%")
    return output_dir, images, world_points, depth, depth_conf, masks, extrinsics, intrinsics, pose_source


def chroma_key_foreground_mask(images, min_green=0.25, margin=0.08):
    """Estimate foreground by removing green-screen pixels from RGB channels."""
    r = images[..., 0]
    g = images[..., 1]
    b = images[..., 2]
    green_bg = (g > min_green) & (g > r + margin) & (g > b + margin)
    return (~green_bg).astype(np.float32)[..., None]


def resolve_bg_mode(bg_mode: str, dataset_masks):
    if bg_mode == "auto":
        if dataset_masks is not None:
            return "mask-black"
        return "original"
    if bg_mode in {"mask-black", "foreground-only"} and dataset_masks is None:
        print(f"[gsplat] No dataset masks found; falling back from '{bg_mode}' to 'original'.")
        return "original"
    return bg_mode


def prepare_training_targets(images, dataset_masks, args):
    """Return composited target images, foreground mask, and optional loss weights."""
    bg_mode = resolve_bg_mode(args.bg_mode, dataset_masks)
    print(f"[gsplat] Background mode: {bg_mode}")

    if bg_mode == "original":
        return images, None, None

    if bg_mode == "chroma-black":
        foreground = chroma_key_foreground_mask(
            images, min_green=args.chroma_min_green, margin=args.chroma_margin
        )
        print(f"[gsplat] Chroma-key foreground ratio {float(foreground.mean()) * 100:.1f}%")
    else:
        foreground = dataset_masks

    if bg_mode == "foreground-only":
        return images, foreground, foreground

    # mask-black / chroma-black: train on a full black-background composite.
    weights = 1.0 + (args.foreground_loss_weight - 1.0) * foreground
    return images * foreground, foreground, weights.astype(np.float32)


def sample_initial_points(world_points, images, depth_conf, masks, max_points, conf_threshold, seed):
    points = world_points.reshape(-1, 3)
    colors = images.reshape(-1, 3)
    if depth_conf is None:
        conf = np.ones((points.shape[0],), dtype=np.float32)
    else:
        conf = depth_conf.reshape(-1).astype(np.float32)

    valid = np.isfinite(points).all(axis=1)
    valid &= np.linalg.norm(points, axis=1) > 1e-6
    valid &= conf > conf_threshold
    if masks is not None:
        valid &= masks.reshape(-1) > 0.5
    points = points[valid]
    colors = colors[valid]
    conf = conf[valid]

    if len(points) == 0:
        raise ValueError("No valid points for Gaussian initialization.")

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        conf = conf[idx]

    return points.astype(np.float32), colors.astype(np.float32), conf.astype(np.float32)


def estimate_log_scales(points: torch.Tensor, init_scale: float, ref_count: int = 4096, chunk: int = 4096):
    """Approximate per-point nearest-neighbor scale without sklearn."""
    n = points.shape[0]
    ref_idx = torch.randperm(n, device=points.device)[: min(ref_count, n)]
    ref = points[ref_idx]
    scales = []
    with torch.no_grad():
        for start in tqdm(range(0, n, chunk), desc="Estimating scales"):
            pts = points[start : start + chunk]
            d = torch.cdist(pts, ref)
            k = 2 if ref.shape[0] > 1 else 1
            nearest = torch.topk(d, k=k, largest=False).values[:, -1]
            scales.append(nearest)
        scale = torch.cat(scales).clamp_min(1e-4) * init_scale
    return torch.log(scale[:, None].repeat(1, 3))


def create_splats(points_np, colors_np, args, device):
    points = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
    colors = torch.from_numpy(colors_np).to(device=device, dtype=torch.float32)
    n = points.shape[0]

    means = torch.nn.Parameter(points)
    scales = torch.nn.Parameter(estimate_log_scales(points, args.init_scale))
    quats = torch.nn.Parameter(
        F.normalize(torch.randn((n, 4), device=device, dtype=torch.float32), dim=-1)
    )
    opacities = torch.nn.Parameter(
        torch.logit(torch.full((n,), args.init_opacity, device=device), eps=1e-6)
    )

    sh_count = (args.sh_degree + 1) ** 2
    sh0 = torch.nn.Parameter(rgb_to_sh(colors).unsqueeze(1))
    shN = torch.nn.Parameter(torch.zeros((n, sh_count - 1, 3), device=device))

    splats = torch.nn.ParameterDict(
        {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh0": sh0,
            "shN": shN,
        }
    )

    lrs = {
        "means": args.means_lr,
        "scales": args.scales_lr,
        "quats": args.quats_lr,
        "opacities": args.opacities_lr,
        "sh0": args.sh0_lr,
        "shN": args.shN_lr,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}], eps=1e-15)
        for name, lr in lrs.items()
    }
    return splats, optimizers


def scene_scale_from_points(points_np):
    lo = np.percentile(points_np, 5, axis=0)
    hi = np.percentile(points_np, 95, axis=0)
    return float(max(np.linalg.norm(hi - lo), 1e-3))


def conf_threshold_from_percentile(depth_conf, percentile):
    """Return an absolute depth_conf threshold at the given percentile.

    Used both to drop low-confidence seed points and to mask the depth loss.
    Returns None when there is no usable confidence map or filtering is off.
    """
    if depth_conf is None or percentile <= 0:
        return None
    conf = depth_conf.reshape(-1).astype(np.float32)
    conf = conf[np.isfinite(conf)]
    if conf.size == 0:
        return None
    return float(np.percentile(conf, percentile))


def train(args):
    configure_gsplat_import(args.use_local_gsplat)
    DefaultStrategy, export_splats, rasterization, l1_loss, ssim_loss = import_gsplat_modules()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. gsplat training requires CUDA.")
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir, images, world_points, depth, depth_conf, masks, extrinsics, intrinsics, pose_source = load_pipeline_data(
        args.dataset, args.use_ba, args.mask_loss, args.reproject_depth_with_poses
    )
    train_images, foreground_masks, loss_weights = prepare_training_targets(images, masks, args)

    # Confidence-based filtering of the seed point cloud. VGGT depth on distant /
    # textureless regions (common in the 360 scene data) is noisy; seeding Gaussians
    # there produces floaters. Dropping the lowest-confidence points removes most of them.
    init_conf_thr = conf_threshold_from_percentile(depth_conf, args.init_conf_percentile)
    conf_threshold = args.conf_threshold
    if init_conf_thr is not None:
        conf_threshold = max(conf_threshold, init_conf_thr)
        print(f"[gsplat] Init conf filter: keep depth_conf > {conf_threshold:.4g} "
              f"(p{args.init_conf_percentile:g})")

    points_np, colors_np, conf_np = sample_initial_points(
        world_points, train_images, depth_conf, foreground_masks, args.max_points, conf_threshold, args.seed
    )
    print(f"[gsplat] Dataset: {args.dataset}")
    print(f"[gsplat] Pose source: {pose_source}")
    print(f"[gsplat] Frames: {images.shape[0]}, image size: {images.shape[2]}x{images.shape[1]}")
    print(f"[gsplat] Initial Gaussians: {len(points_np):,}")

    splats, optimizers = create_splats(points_np, colors_np, args, device)
    scene_scale = scene_scale_from_points(points_np)

    # Densification that runs all the way to the last step keeps minting Gaussians
    # (and floaters) without enough steps left to refine them -- the training loss
    # then drifts upward in the second half. Stop refining at ~half the schedule by
    # default so the back half only polishes existing Gaussians.
    refine_stop = args.refine_stop if args.refine_stop is not None else max(args.refine_start + 1, int(args.steps * 0.5))
    print(f"[gsplat] Densify window: [{args.refine_start}, {refine_stop}] of {args.steps} steps")
    strategy = DefaultStrategy(
        refine_start_iter=args.refine_start,
        refine_stop_iter=refine_stop,
        refine_every=args.refine_every,
        reset_every=args.reset_every,
        prune_opa=args.prune_opa,
        grow_grad2d=args.grow_grad2d,
        verbose=args.verbose_strategy,
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    images_t = torch.from_numpy(train_images).to(device=device, dtype=torch.float32)
    loss_weights_t = torch.from_numpy(loss_weights).to(device=device, dtype=torch.float32) if loss_weights is not None else None
    foreground_t = (
        torch.from_numpy(foreground_masks).to(device=device, dtype=torch.float32)
        if foreground_masks is not None else None
    )
    extr_t = torch.from_numpy(extrinsics).to(device=device, dtype=torch.float32)
    intr_t = torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32)

    # Depth supervision tensors. Rendered expected depth is anchored to the VGGT depth
    # (re-projected into the BA camera frame), masked to confident pixels. This is the
    # main floater fix for the surround scene.
    depth_t = None
    depth_conf_t = None
    depth_conf_thr = 0.0
    use_depth_loss = args.depth_loss_weight > 0 and depth is not None
    if use_depth_loss:
        depth_np = depth.squeeze(-1) if depth.ndim == 4 else depth  # (S, H, W)
        depth_t = torch.from_numpy(np.ascontiguousarray(depth_np)).to(device=device, dtype=torch.float32)
        if depth_conf is not None:
            depth_conf_t = torch.from_numpy(np.ascontiguousarray(depth_conf)).to(device=device, dtype=torch.float32)
            thr = conf_threshold_from_percentile(depth_conf, args.depth_conf_percentile)
            depth_conf_thr = thr if thr is not None else 0.0
        print(f"[gsplat] Depth supervision ON: weight {args.depth_loss_weight}, "
              f"conf>{depth_conf_thr:.4g} (p{args.depth_conf_percentile:g})")
    render_mode = "RGB+ED" if use_depth_loss else "RGB"

    steps = args.steps
    s, h, w = images.shape[:3]
    loss_history = []
    psnr_history = []
    best_psnr = 0.0
    start_time = time.time()

    pbar = tqdm(range(steps), desc="gsplat training")
    for step in pbar:
        cam_idx = int(torch.randint(0, s, (1,), device=device).item())
        viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        viewmat[0, :3, :4] = extr_t[cam_idx]
        k = intr_t[cam_idx].unsqueeze(0)
        gt = images_t[cam_idx : cam_idx + 1]
        weights = loss_weights_t[cam_idx : cam_idx + 1] if loss_weights_t is not None else None

        sh_degree_to_use = min(step // args.sh_degree_interval, args.sh_degree)
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

        # Random background (only meaningful when we have a foreground alpha to composite
        # the GT against). It stops floaters from "hiding" as low-opacity dark splats over a
        # constant black background by forcing opacities to explain a changing background.
        fg = foreground_t[cam_idx : cam_idx + 1] if foreground_t is not None else None
        if args.random_bg and fg is not None:
            bg = torch.rand((1, 3), device=device)
            gt = gt * fg + bg * (1.0 - fg)
        else:
            bg = torch.zeros((1, 3), device=device)

        renders, alphas, info = rasterization(
            means=splats["means"],
            quats=F.normalize(splats["quats"], dim=-1),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=k,
            width=w,
            height=h,
            packed=args.packed,
            sh_degree=sh_degree_to_use,
            absgrad=strategy.absgrad,
            render_mode=render_mode,
            backgrounds=bg,
        )
        rgb = renders[..., :3]
        depth_pred = renders[..., 3:4] if use_depth_loss else None

        strategy.step_pre_backward(splats, optimizers, strategy_state, step, info)
        if weights is not None:
            weights3 = weights.expand_as(gt)
            l1 = (l1_loss(rgb, gt) * weights3).sum() / weights3.sum().clamp_min(1.0)
            if args.bg_mode == "foreground-only":
                ssim = ssim_loss(
                    (rgb * weights3).permute(0, 3, 1, 2),
                    (gt * weights3).permute(0, 3, 1, 2),
                )
            else:
                ssim = ssim_loss(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
        else:
            l1 = l1_loss(rgb, gt).mean()
            ssim = ssim_loss(rgb.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))
        loss = torch.lerp(l1, ssim, args.ssim_lambda)

        # Depth supervision: pull rendered expected depth toward the (confident) VGGT depth.
        if use_depth_loss:
            dgt = depth_t[cam_idx]  # (H, W)
            dpred = depth_pred[0, ..., 0]  # (H, W)
            valid = torch.isfinite(dgt) & (dgt > 0)
            if depth_conf_t is not None:
                valid = valid & (depth_conf_t[cam_idx] > depth_conf_thr)
            if fg is not None:
                valid = valid & (fg[0, ..., 0] > 0.5)
            denom = valid.sum().clamp_min(1.0)
            depth_l1 = ((dpred - dgt).abs() * valid).sum() / denom
            loss = loss + args.depth_loss_weight * depth_l1

        # Lightweight regularizers that suppress floaters / smeared Gaussians.
        if args.opacity_reg > 0:
            loss = loss + args.opacity_reg * torch.sigmoid(splats["opacities"]).mean()
        if args.scale_reg > 0:
            loss = loss + args.scale_reg * (torch.exp(splats["scales"]) / scene_scale).mean()

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
            packed=args.packed,
        )

        loss_history.append(float(loss.detach().cpu()))
        if step % args.eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                psnr = compute_psnr(rgb[0].clamp(0.0, 1.0), gt[0])
                best_psnr = max(best_psnr, psnr)
                psnr_history.append((step, psnr))
            pbar.set_postfix(loss=f"{loss_history[-1]:.4f}", psnr=f"{psnr:.2f}", n=len(splats["means"]))

    elapsed = time.time() - start_time
    ply_path = os.path.join(output_dir, args.output_name)
    export_splats(
        means=splats["means"].detach(),
        scales=splats["scales"].detach(),
        quats=F.normalize(splats["quats"].detach(), dim=-1),
        opacities=splats["opacities"].detach(),
        sh0=splats["sh0"].detach(),
        shN=splats["shN"].detach(),
        format="ply",
        save_to=ply_path,
    )

    loss_path = os.path.join(output_dir, args.loss_plot)
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, linewidth=0.5, alpha=0.65, label="Loss")
    if len(loss_history) > 100:
        smooth = np.convolve(loss_history, np.ones(100) / 100, mode="valid")
        plt.plot(np.arange(99, len(loss_history)), smooth, linewidth=2, label="Smoothed")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"gsplat training loss, best PSNR {best_psnr:.2f} dB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=120)
    plt.close()

    preview_idx = min(args.preview_frame, s - 1)
    render_path = os.path.join(output_dir, "renders", f"gsplat_train_preview_{preview_idx:04d}.png")
    os.makedirs(os.path.dirname(render_path), exist_ok=True)
    with torch.no_grad():
        viewmat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        viewmat[0, :3, :4] = extr_t[preview_idx]
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        render, _, _ = rasterization(
            means=splats["means"],
            quats=F.normalize(splats["quats"], dim=-1),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=intr_t[preview_idx : preview_idx + 1],
            width=w,
            height=h,
            packed=args.packed,
            sh_degree=args.sh_degree,
            backgrounds=torch.zeros((1, 3), device=device),
        )
        image = (render[0].clamp(0.0, 1.0).detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(render_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print("\n--- gsplat training summary ---")
    print(f"  Gaussians: {len(splats['means']):,}")
    print(f"  Best PSNR: {best_psnr:.2f} dB")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  PLY: {ply_path}")
    print(f"  Loss plot: {loss_path}")
    print(f"  Preview render: {render_path}")


def main():
    parser = argparse.ArgumentParser(description="Train 3DGS using gsplat's official API")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--max-points", type=int, default=150000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-ba", action="store_true", default=True)
    parser.add_argument("--no-ba", dest="use_ba", action="store_false")
    parser.add_argument("--use-local-gsplat", action="store_true", default=False,
                        help="Import ./gsplat before site-packages. This may compile CUDA extensions.")
    parser.add_argument("--installed-gsplat", dest="use_local_gsplat", action="store_false",
                        help="Use the installed gsplat package instead of ./gsplat.")
    parser.add_argument("--output-name", default="gaussians.ply")
    parser.add_argument("--loss-plot", default="gsplat_train_loss.png")
    parser.add_argument("--preview-frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf-threshold", type=float, default=1e-6)
    parser.add_argument("--mask-loss", action="store_true", default=True,
                        help="Use dataset masks for background handling when available.")
    parser.add_argument("--no-mask-loss", dest="mask_loss", action="store_false",
                        help="Disable mask-guided foreground filtering.")
    parser.add_argument("--bg-mode", choices=["auto", "mask-black", "chroma-black", "foreground-only", "original"],
                        default="auto",
                        help="How to handle background pixels. Auto uses masks when present and original RGB otherwise.")
    parser.add_argument("--foreground-loss-weight", type=float, default=8.0,
                        help="Foreground L1 loss weight for black-background modes.")
    parser.add_argument("--chroma-min-green", type=float, default=0.25,
                        help="Minimum green value for --bg-mode chroma-black.")
    parser.add_argument("--chroma-margin", type=float, default=0.08,
                        help="Require G to exceed R and B by this margin for --bg-mode chroma-black.")
    parser.add_argument("--reproject-depth-with-poses", action="store_true", default=True,
                        help="When using BA poses, recompute dense depth points in the BA camera frame.")
    parser.add_argument("--no-reproject-depth-with-poses", dest="reproject_depth_with_poses", action="store_false",
                        help="Use the world_points_from_depth saved by VGGT inference without recomputing.")
    parser.add_argument("--packed", action="store_true", help="Use gsplat packed rasterization mode.")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--sh-degree-interval", type=int, default=1000)
    parser.add_argument("--ssim-lambda", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--init-opacity", type=float, default=0.1)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--means-lr", type=float, default=1.6e-4)
    parser.add_argument("--scales-lr", type=float, default=5e-3)
    parser.add_argument("--opacities-lr", type=float, default=5e-2)
    parser.add_argument("--quats-lr", type=float, default=1e-3)
    parser.add_argument("--sh0-lr", type=float, default=2.5e-3)
    parser.add_argument("--shN-lr", type=float, default=1.25e-4)
    parser.add_argument("--refine-start", type=int, default=500)
    parser.add_argument("--refine-stop", type=int, default=None,
                        help="Step to stop densification. Default: half the training schedule, "
                             "so the second half only refines (prevents the loss drifting up).")
    parser.add_argument("--refine-every", type=int, default=100)
    parser.add_argument("--reset-every", type=int, default=3000)
    parser.add_argument("--prune-opa", type=float, default=0.005)
    parser.add_argument("--grow-grad2d", type=float, default=0.0002)
    parser.add_argument("--verbose-strategy", action="store_true")

    # --- Quality / anti-floater improvements (default on, each can be disabled) ---
    parser.add_argument("--depth-loss-weight", type=float, default=0.5,
                        help="Weight of VGGT depth supervision (render_mode RGB+ED). 0 disables.")
    parser.add_argument("--depth-conf-percentile", type=float, default=30.0,
                        help="Only supervise depth on pixels above this depth_conf percentile.")
    parser.add_argument("--init-conf-percentile", type=float, default=30.0,
                        help="Drop seed points below this depth_conf percentile. 0 disables.")
    parser.add_argument("--random-bg", action="store_true", default=True,
                        help="Composite GT onto a random background each step (masked modes only).")
    parser.add_argument("--no-random-bg", dest="random_bg", action="store_false",
                        help="Keep a fixed black training background.")
    parser.add_argument("--opacity-reg", type=float, default=0.001,
                        help="L1 regularization on opacities to suppress low-opacity floaters. 0 disables.")
    parser.add_argument("--scale-reg", type=float, default=0.001,
                        help="Regularization on Gaussian scale (relative to scene scale). 0 disables.")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

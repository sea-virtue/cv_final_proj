"""
Generate experiment analysis for the report/PPT.

This script summarizes:
1. Whether BA improves the geometry used by 3DGS.
2. The implemented VGGT improvement: mask-guided confidence filtering.

Usage:
    python pipeline/analyze_experiments.py
"""

import os
import sys
import csv

import cv2
import numpy as np
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.utils import get_dataset_list, get_output_dir


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "output", "analysis")


def _plot_label(dataset):
    mapping = {
        "数据1-人体": "D1-Human",
        "数据2-人体": "D2-Human",
        "数据3-场景": "D3-Scene",
    }
    return mapping.get(dataset, dataset)


def _unique_dataset_names():
    names = []
    seen_outputs = set()
    for name in get_dataset_list().keys():
        output_dir = get_output_dir(name)
        if output_dir in seen_outputs:
            continue
        if os.path.exists(os.path.join(output_dir, "predictions.npz")):
            names.append(os.path.basename(output_dir))
            seen_outputs.add(output_dir)
    return names


def _read_gaussian_positions(ply_path, max_points=50000):
    with open(ply_path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY: {ply_path}")
            text = line.decode("utf-8", errors="ignore").strip()
            header.append(text)
            if text == "end_header":
                break
        raw = f.read()

    vertex_count = 0
    prop_count = 0
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            prop_count += 1

    if vertex_count <= 0 or prop_count <= 0:
        raise ValueError(f"Cannot parse PLY header: {ply_path}")

    floats_per_vertex = prop_count
    # Backward compatibility: older generated files declared 32 properties
    # but wrote 62 floats per vertex.
    if len(raw) // (floats_per_vertex * 4) != vertex_count and len(raw) // (62 * 4) == vertex_count:
        floats_per_vertex = 62

    values = np.frombuffer(raw[: vertex_count * floats_per_vertex * 4], dtype="<f4")
    values = values.reshape(vertex_count, floats_per_vertex)
    means = values[:, :3].copy()
    if len(means) > max_points:
        idx = np.linspace(0, len(means) - 1, max_points, dtype=np.int64)
        means = means[idx]
    return means, vertex_count


def _rotation_frobenius(before, after):
    return np.array([
        np.linalg.norm(after[i, :3, :3] - before[i, :3, :3])
        for i in range(before.shape[0])
    ])


def _projection_metrics(points, extrinsics, intrinsics, image_shape, max_frames=16):
    if points.size == 0:
        return 0.0, 0.0

    frame_ids = np.linspace(0, extrinsics.shape[0] - 1, min(max_frames, extrinsics.shape[0]), dtype=np.int64)
    height, width = image_shape
    visible_ratios = []
    coverage_ratios = []

    for frame_id in frame_ids:
        ext = extrinsics[frame_id]
        K = intrinsics[frame_id]
        cam = points @ ext[:3, :3].T + ext[:3, 3]
        z = cam[:, 2]
        u = K[0, 0] * cam[:, 0] / (z + 1e-8) + K[0, 2]
        v = K[1, 1] * cam[:, 1] / (z + 1e-8) + K[1, 2]
        visible = (z > 1e-4) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        visible_ratios.append(float(visible.mean()))

        if visible.any():
            grid_w, grid_h = 64, 64
            gx = np.clip((u[visible] / width * grid_w).astype(np.int32), 0, grid_w - 1)
            gy = np.clip((v[visible] / height * grid_h).astype(np.int32), 0, grid_h - 1)
            occupied = np.unique(gy * grid_w + gx).size
            coverage_ratios.append(float(occupied / (grid_w * grid_h)))
        else:
            coverage_ratios.append(0.0)

    return float(np.mean(visible_ratios)), float(np.mean(coverage_ratios))


def analyze_ba_effect(dataset_names):
    rows = []
    for dataset in dataset_names:
        output_dir = get_output_dir(dataset)
        pred_path = os.path.join(output_dir, "predictions.npz")
        ba_path = os.path.join(output_dir, "ba_result.npz")
        ply_path = os.path.join(output_dir, "gaussians.ply")
        if not (os.path.exists(pred_path) and os.path.exists(ba_path) and os.path.exists(ply_path)):
            continue

        pred = np.load(pred_path, allow_pickle=True)
        ba = np.load(ba_path, allow_pickle=True)
        images = pred["images"]
        if images.ndim == 4 and images.shape[1] == 3:
            image_shape = (images.shape[2], images.shape[3])
        else:
            image_shape = (images.shape[1], images.shape[2])

        ext_before = pred["extrinsic"]
        ext_after = ba["extrinsic_opt"]
        points, gaussian_count = _read_gaussian_positions(ply_path)

        reproj_before = float(ba["reproj_before"])
        reproj_after = float(ba["reproj_after"])
        improvement = (reproj_before - reproj_after) / max(reproj_before, 1e-8) * 100.0
        trans_change = np.linalg.norm(ext_after[:, :3, 3] - ext_before[:, :3, 3], axis=1)
        rot_change = _rotation_frobenius(ext_before, ext_after)
        vis_before, cov_before = _projection_metrics(points, ext_before, pred["intrinsic"], image_shape)
        vis_after, cov_after = _projection_metrics(points, ext_after, ba["intrinsic"], image_shape)

        rows.append({
            "dataset": dataset,
            "frames": int(ext_before.shape[0]),
            "ba_reproj_before": reproj_before,
            "ba_reproj_after": reproj_after,
            "ba_improvement_pct": improvement,
            "ba_landmarks": int(ba["points3d_opt"].shape[0]),
            "avg_translation_change": float(trans_change.mean()),
            "avg_rotation_change": float(rot_change.mean()),
            "gaussians": int(gaussian_count),
            "visible_before": vis_before,
            "visible_after": vis_after,
            "coverage_before": cov_before,
            "coverage_after": cov_after,
        })
        pred.close()
        ba.close()
    return rows


def analyze_mask_improvement(dataset_names):
    rows = []
    datasets = get_dataset_list()
    for dataset in dataset_names:
        if "人体" not in dataset:
            continue
        dataset_path = datasets.get(dataset)
        output_dir = get_output_dir(dataset)
        pred_path = os.path.join(output_dir, "predictions.npz")
        if not dataset_path or not os.path.isdir(dataset_path) or not os.path.exists(pred_path):
            continue

        mask_paths = sorted([
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.startswith("msk_") and f.endswith(".png")
        ])
        if not mask_paths:
            continue

        pred = np.load(pred_path, allow_pickle=True)
        depth_conf = pred["depth_conf"]
        _, height, width = depth_conf.shape

        masks = []
        for path in mask_paths[: depth_conf.shape[0]]:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height))
            masks.append(mask > 128)

        if not masks:
            pred.close()
            continue

        mask_stack = np.stack(masks, axis=0)
        conf = depth_conf[: mask_stack.shape[0]]
        foreground_ratio = float(mask_stack.mean())
        background_ratio = 1.0 - foreground_ratio
        background_zero_ratio = float((conf[~mask_stack] <= 1e-8).mean()) if background_ratio > 0 else 0.0
        foreground_conf_mean = float(conf[mask_stack].mean()) if foreground_ratio > 0 else 0.0

        rows.append({
            "dataset": dataset,
            "frames": int(mask_stack.shape[0]),
            "foreground_ratio": foreground_ratio,
            "background_ratio": background_ratio,
            "background_zero_conf_ratio": background_zero_ratio,
            "foreground_conf_mean": foreground_conf_mean,
        })
        pred.close()
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_ba(rows):
    if not rows:
        return
    datasets = [_plot_label(r["dataset"]) for r in rows]
    before = [r["ba_reproj_before"] for r in rows]
    after = [r["ba_reproj_after"] for r in rows]
    coverage_before = [r["coverage_before"] * 100 for r in rows]
    coverage_after = [r["coverage_after"] * 100 for r in rows]

    x = np.arange(len(datasets))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(x - width / 2, before, width, label="Before BA")
    axes[0].bar(x + width / 2, after, width, label="After BA")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=15)
    axes[0].set_ylabel("Reprojection Error (px)")
    axes[0].set_title("BA reduces reprojection error")
    axes[0].legend()

    axes[1].bar(x - width / 2, coverage_before, width, label="Original poses")
    axes[1].bar(x + width / 2, coverage_after, width, label="BA poses")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=15)
    axes[1].set_ylabel("Projected Gaussian Coverage (%)")
    axes[1].set_title("3DGS camera coverage proxy")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "ba_effect_on_3dgs.png")
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_mask(rows):
    if not rows:
        return
    datasets = [_plot_label(r["dataset"]) for r in rows]
    fg = [r["foreground_ratio"] * 100 for r in rows]
    bg_zero = [r["background_zero_conf_ratio"] * 100 for r in rows]

    x = np.arange(len(datasets))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, fg, width, label="Foreground pixels")
    plt.bar(x + width / 2, bg_zero, width, label="Background zero-conf")
    plt.xticks(x, datasets, rotation=15)
    plt.ylabel("Ratio (%)")
    plt.title("Mask-guided confidence filtering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "mask_filter_analysis.png"), dpi=140)
    plt.close()


def _write_markdown(ba_rows, mask_rows):
    path = os.path.join(ANALYSIS_DIR, "experiment_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 实验分析：BA 对 3DGS 的影响与 VGGT 改进方法\n\n")
        f.write("## 1. BA 是否改善 3DGS 所依赖的几何输入\n\n")
        f.write(
            "3DGS 训练需要相机外参作为多视角监督。如果相机位姿不准，"
            "同一空间点在不同视角下会投影不一致，容易造成重影、漂浮点或收敛变慢。"
            "因此这里用 BA 重投影误差和 Gaussian 投影覆盖率作为分析指标。\n\n"
        )
        f.write("| 数据集 | BA前误差(px) | BA后误差(px) | 改善 | BA地标 | 平均平移变化 | Gaussian覆盖率(原始→BA) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in ba_rows:
            f.write(
                f"| {r['dataset']} | {r['ba_reproj_before']:.4f} | {r['ba_reproj_after']:.4f} | "
                f"{r['ba_improvement_pct']:.1f}% | {r['ba_landmarks']} | "
                f"{r['avg_translation_change']:.4f} | "
                f"{r['coverage_before']*100:.1f}% → {r['coverage_after']*100:.1f}% |\n"
            )
        f.write("\n结论：BA 明显降低重投影误差，尤其在场景视频中改善最大。由于本项目 3DGS 默认使用 BA 后相机训练，BA 的作用是提供更一致的多视角几何监督。\n\n")

        f.write("## 2. 改进方法实验：Mask 引导的置信度过滤\n\n")
        f.write(
            "针对人体数据集，使用 `msk_*.png` 将背景区域的 `depth_conf` 置零。"
            "这样后续点云查看和 3DGS 初始化时更关注人体前景，减少背景噪声干扰。\n\n"
        )
        f.write("| 数据集 | 帧数 | 前景像素占比 | 背景像素占比 | 背景零置信度比例 | 前景平均置信度 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in mask_rows:
            f.write(
                f"| {r['dataset']} | {r['frames']} | {r['foreground_ratio']*100:.1f}% | "
                f"{r['background_ratio']*100:.1f}% | {r['background_zero_conf_ratio']*100:.1f}% | "
                f"{r['foreground_conf_mean']:.2f} |\n"
            )
        f.write("\n结论：Mask 过滤把人体数据中的背景区域从有效深度置信度中排除，属于低成本、可解释的 VGGT 后处理改进。\n\n")
        f.write("## 3. 可用于 PPT 的图片\n\n")
        f.write("- `output/analysis/ba_effect_on_3dgs.png`\n")
        f.write("- `output/analysis/mask_filter_analysis.png`\n")


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    dataset_names = _unique_dataset_names()
    ba_rows = analyze_ba_effect(dataset_names)
    mask_rows = analyze_mask_improvement(dataset_names)

    _write_csv(os.path.join(ANALYSIS_DIR, "ba_effect_on_3dgs.csv"), ba_rows)
    _write_csv(os.path.join(ANALYSIS_DIR, "mask_filter_analysis.csv"), mask_rows)
    _plot_ba(ba_rows)
    _plot_mask(mask_rows)
    _write_markdown(ba_rows, mask_rows)

    print(f"[Analysis] Wrote results to {ANALYSIS_DIR}")
    print("  - ba_effect_on_3dgs.csv")
    print("  - mask_filter_analysis.csv")
    print("  - ba_effect_on_3dgs.png")
    print("  - mask_filter_analysis.png")
    print("  - experiment_analysis.md")


if __name__ == "__main__":
    main()

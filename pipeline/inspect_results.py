"""
pipeline/inspect_results.py — 查看每一步结果（人类可读）
用法:
    python pipeline/inspect_results.py --dataset 数据1-人体 --step 1   # 看VGGT结果
    python pipeline/inspect_results.py --dataset 数据1-人体 --step 2   # 看BA结果
    python pipeline/inspect_results.py --dataset 数据1-人体 --step 3   # 看3DGS结果
    python pipeline/inspect_results.py --dataset 数据1-人体 --step all # 全部
"""

import os
import sys
import argparse
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 设置中文字体（避免 glyph missing 警告）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# 标题用英文避免 CJK 问题，正文也统一用英文

# 确保能 import pipeline 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.utils import get_output_dir, get_dataset_list


# ============================================================
# Step 1: 查看 VGGT predictions
# ============================================================

def inspect_predictions(dataset_name: str):
    """打印 predictions.npz 摘要，保存可视化图片。"""
    output_dir = get_output_dir(dataset_name)
    pred_path = os.path.join(output_dir, "predictions.npz")

    if not os.path.exists(pred_path):
        print(f"❌ 未找到 {pred_path}，请先运行 vggt_inference.py")
        return

    data = np.load(pred_path, allow_pickle=True)
    print("=" * 60)
    print(f"📦 VGGT Predictions — {dataset_name}")
    print("=" * 60)
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:<25s}  shape={str(v.shape):<20s}  dtype={v.dtype}  range=[{v.min():.4f}, {v.max():.4f}]")
        else:
            print(f"  {k:<25s}  type={type(v).__name__}")

    S = data["extrinsic"].shape[0]
    print(f"\n📷 帧数: {S}")
    print(f"📐 图像尺寸: {data['images'].shape[1]}×{data['images'].shape[2]}")

    # 打印每帧相机位置
    print("\n📍 相机位置 (world → cam, translation):")
    for i in range(min(S, 5)):
        t = data["extrinsic"][i, :3, 3]
        print(f"    帧{i}: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    if S > 5:
        print(f"    ... (共{S}帧，仅显示前5帧)")

    # 打印深度统计
    depth = data["depth"]
    depth_conf = data.get("depth_conf", np.ones_like(depth[..., 0]))
    valid = depth > 1e-6
    print(f"\n🌊 深度统计:")
    print(f"    有效像素比例: {valid.mean()*100:.1f}%")
    print(f"    深度范围: [{depth[valid].min():.3f}, {depth[valid].max():.3f}]")
    print(f"    深度均值: {depth[valid].mean():.3f}")

    # 点数统计
    wp = data["world_points_from_depth"]
    valid_pts = (np.linalg.norm(wp, axis=-1) > 1e-6).sum()
    print(f"\n☁️  世界坐标点云:")
    print(f"    总点数: {wp.shape[0]*wp.shape[1]*wp.shape[2]:,}")
    print(f"    有效点数: {valid_pts:,}")

    # 保存可视化
    _save_prediction_visualization(data, dataset_name, output_dir)
    print(f"\n✅ 可视化已保存到 {output_dir}/pred_vis.png")
    data.close()


def _save_prediction_visualization(data, dataset_name, output_dir):
    """保存 VGGT 预测的可视化图。"""
    images = data["images"]   # (S, 3, H, W) or (S, H, W, 3)
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))  # NCHW → NHWC
    depth = data["depth"]     # (S, H, W, 1)
    S = images.shape[0]

    # 选择展示的帧（均匀采样5帧）
    show_frames = np.linspace(0, S-1, min(5, S)).astype(int)

    fig, axes = plt.subplots(3, len(show_frames), figsize=(3*len(show_frames), 9))
    if len(show_frames) == 1:
        axes = axes[:, np.newaxis]

    for col, fi in enumerate(show_frames):
        # 原始图像
        axes[0, col].imshow(images[fi])
        axes[0, col].set_title(f"Frame {fi} (Image)")
        axes[0, col].axis("off")

        # 深度图
        d = depth[fi].squeeze()
        d_valid = np.where(d > 1e-6, d, np.nan)
        im = axes[1, col].imshow(d_valid, cmap="plasma")
        axes[1, col].set_title(f"Frame {fi} (Depth)")
        axes[1, col].axis("off")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)

        # 置信度
        conf = data.get("depth_conf", np.ones_like(d))
        c = conf[fi]
        im2 = axes[2, col].imshow(c, cmap="viridis", vmin=0, vmax=1)
        axes[2, col].set_title(f"Frame {fi} (Confidence)")
        axes[2, col].axis("off")
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046)

    plt.suptitle(f"VGGT Predictions — {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pred_vis.png"), dpi=120)
    plt.close()
    print(f"  → 保存: {os.path.join(output_dir, 'pred_vis.png')}")


# ============================================================
# Step 2: 查看 BA 结果
# ============================================================

def inspect_ba(dataset_name: str):
    """打印 BA 结果摘要。"""
    output_dir = get_output_dir(dataset_name)
    ba_path = os.path.join(output_dir, "ba_result.npz")
    pred_path = os.path.join(output_dir, "predictions.npz")

    if not os.path.exists(ba_path):
        print(f"❌ 未找到 {ba_path}，请先运行 bundle_adjustment.py")
        return

    ba_data = np.load(ba_path, allow_pickle=True)
    print("=" * 60)
    print(f"🔧 Bundle Adjustment Results — {dataset_name}")
    print("=" * 60)
    for k, v in ba_data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:<25s}  shape={str(v.shape):<20s}  dtype={v.dtype}")
        else:
            print(f"  {k:<25s}  = {v}")

    reproj_before = float(ba_data["reproj_before"])
    reproj_after = float(ba_data["reproj_after"])
    improvement = (reproj_before - reproj_after) / max(reproj_before, 1e-8) * 100

    print(f"\n📊 重投影误差:")
    print(f"    BA前: {reproj_before:.4f} px")
    print(f"    BA后: {reproj_after:.4f} px")
    print(f"    改善: {improvement:.1f}%")

    # 对比 BA 前后相机位姿变化
    if os.path.exists(pred_path):
        pred_data = np.load(pred_path, allow_pickle=True)
        ext_before = pred_data["extrinsic"]
        ext_after = ba_data["extrinsic_opt"]
        S = ext_before.shape[0]

        # 计算每帧位姿变化量
        trans_diff = np.linalg.norm(ext_after[:, :3, 3] - ext_before[:, :3, 3], axis=1)
        # 旋转差异 (用 rotation matrix Frobenius norm)
        rot_diff = np.array([
            np.linalg.norm(ext_after[i, :3, :3] - ext_before[i, :3, :3])
            for i in range(S)
        ])

        print(f"\n🔄 位姿变化 (BA前→BA后):")
        print(f"    平均平移变化: {trans_diff.mean():.4f}")
        print(f"    最大平移变化: {trans_diff.max():.4f}")
        print(f"    平均旋转变化(Frobenius): {rot_diff.mean():.4f}")
        print(f"    最大旋转变化(Frobenius): {rot_diff.max():.4f}")

        # 绘制位姿变化图
        _save_ba_comparison(ext_before, ext_after, reproj_before, reproj_after, dataset_name, output_dir)
        pred_data.close()

    ba_data.close()
    print(f"\n✅ 对比图已保存到 {output_dir}/ba_comparison.png")


def _save_ba_comparison(ext_before, ext_after, err_before, err_after, dataset_name, output_dir):
    """保存 BA 前后位姿对比图。"""
    S = ext_before.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 平移量对比
    trans_before = ext_before[:, :3, 3]
    trans_after = ext_after[:, :3, 3]
    axes[0].plot(trans_before[:, 0], trans_before[:, 2], "ro-", label="BA前", alpha=0.6, markersize=4)
    axes[0].plot(trans_after[:, 0], trans_after[:, 2], "go-", label="BA后", alpha=0.6, markersize=4)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Z")
    axes[0].set_title("Camera Trajectory (top view)")
    axes[0].legend()
    axes[0].axis("equal")

    # 平移变化逐帧
    trans_diff = np.linalg.norm(trans_after - trans_before, axis=1)
    axes[1].bar(range(S), trans_diff, color="steelblue")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Translation Change")
    axes[1].set_title("Per-frame Translation Change")

    # 重投影误差对比
    axes[2].bar(["Before BA", "After BA"], [err_before, err_after],
                color=["#ff6b6b", "#51cf66"])
    axes[2].set_ylabel("Reprojection Error (px)")
    axes[2].set_title(f"BA: {err_before:.2f} -> {err_after:.2f} px")
    for i, v in enumerate([err_before, err_after]):
        axes[2].text(i, v + 0.1, f"{v:.2f}", ha="center", fontweight="bold")

    plt.suptitle(f"Bundle Adjustment — {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ba_comparison.png"), dpi=120)
    plt.close()
    print(f"  → 保存: {os.path.join(output_dir, 'ba_comparison.png')}")


# ============================================================
# Step 3: 查看 3DGS 结果
# ============================================================

def inspect_gaussians(dataset_name: str):
    """打印 3DGS 训练结果摘要。"""
    output_dir = get_output_dir(dataset_name)
    ply_path = os.path.join(output_dir, "gaussians.ply")
    loss_path = os.path.join(output_dir, "gs_train_loss.png")

    if not os.path.exists(ply_path):
        print(f"❌ 未找到 {ply_path}，请先运行 gaussian_splatting.py")
        return

    # 解析 PLY 获取统计
    means, colors, opacities, scales = _quick_parse_ply(ply_path)
    N = len(means)

    print("=" * 60)
    print(f"✨ 3D Gaussians — {dataset_name}")
    print("=" * 60)
    print(f"  Gaussians 数量: {N:,}")
    print(f"  位置范围 X: [{means[:,0].min():.3f}, {means[:,0].max():.3f}]")
    print(f"  位置范围 Y: [{means[:,1].min():.3f}, {means[:,1].max():.3f}]")
    print(f"  位置范围 Z: [{means[:,2].min():.3f}, {means[:,2].max():.3f}]")
    if opacities is not None:
        print(f"  不透明度: mean={opacities.mean():.4f}, max={opacities.max():.4f}")
    if scales is not None:
        print(f"  缩放: mean={scales.mean():.4f}")
    print(f"  颜色 (RGB): range=[{colors.min():.3f}, {colors.max():.3f}]")

    if os.path.exists(loss_path):
        print(f"\n📈 训练损失曲线: {loss_path}")

    # 保存空间分布图
    _save_gaussian_distribution(means, colors, dataset_name, output_dir)
    print(f"\n✅ 分布图已保存到 {output_dir}/gs_distribution.png")


def _quick_parse_ply(filepath):
    """快速解析 PLY 文件的 xyz/rgb/opacity/scale。"""
    with open(filepath, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        header_str = header.decode()
        N = 0
        for line in header_str.split("\n"):
            if line.startswith("element vertex"):
                N = int(line.split()[-1])
        data = f.read()

    vertex_bytes = len(data) // max(N, 1)
    # 格式: x,y,z(12) + nx,ny,nz(12) + f_dc(12) + f_rest(60) + opacity(4) + scale(12) + rot(16) = 128
    # offset: xyz=0, nx=12, f_dc=24, f_rest=36, opacity=96, scale=100, rot=112

    means = np.zeros((N, 3), dtype=np.float32)
    colors = np.zeros((N, 3), dtype=np.float32)
    opacities = np.zeros(N, dtype=np.float32)
    scales_arr = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        offset = i * vertex_bytes
        if offset + vertex_bytes > len(data):
            break
        means[i] = struct.unpack_from("fff", data, offset)
        # f_dc at offset 24
        r, g, b = struct.unpack_from("fff", data, offset + 24)
        colors[i] = [
            max(0, min(1, r / 0.28209479177387814)),
            max(0, min(1, g / 0.28209479177387814)),
            max(0, min(1, b / 0.28209479177387814)),
        ]
        # opacity at offset 96
        if offset + 100 <= len(data):
            opacities[i] = struct.unpack_from("f", data, offset + 96)[0]
            scales_arr[i] = struct.unpack_from("fff", data, offset + 100)

    return means, colors, opacities, scales_arr


def _save_gaussian_distribution(means, colors, dataset_name, output_dir):
    """保存 Gaussians 空间分布图。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 随机采样以提高绘图速度
    sample = min(50000, len(means))
    idx = np.random.choice(len(means), sample, replace=False)
    pts = means[idx]
    cols = np.clip(colors[idx], 0, 1)

    titles = ["XY (Top)", "XZ (Front)", "YZ (Side)"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    xlabels = ["X", "X", "Y"]
    ylabels = ["Y", "Z", "Z"]

    for ax, (pi, pj), title, xl, yl in zip(axes, pairs, titles, xlabels, ylabels):
        ax.scatter(pts[:, pi], pts[:, pj], c=cols, s=0.5, alpha=0.6, rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.axis("equal")

    plt.suptitle(f"3D Gaussians Distribution — {dataset_name} ({sample:,} sampled)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gs_distribution.png"), dpi=120)
    plt.close()
    print(f"  → 保存: {os.path.join(output_dir, 'gs_distribution.png')}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="查看 Pipeline 各步骤结果")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称（如 数据1-人体）")
    parser.add_argument("--step", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="查看哪步结果: 1=VGGT, 2=BA, 3=3DGS, all=全部")
    args = parser.parse_args()

    datasets = get_dataset_list()
    if args.dataset not in datasets:
        print(f"未知数据集: {args.dataset}")
        print(f"可用数据集: {list(datasets.keys())}")
        sys.exit(1)

    if args.step in ("1", "all"):
        inspect_predictions(args.dataset)
        print()

    if args.step in ("2", "all"):
        inspect_ba(args.dataset)
        print()

    if args.step in ("3", "all"):
        inspect_gaussians(args.dataset)


if __name__ == "__main__":
    main()

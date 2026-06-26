"""
pipeline/gs_viewer.py — 分步 3D 结果查看器
用法:
    python pipeline/gs_viewer.py --dataset 数据1-人体 --step 1    # 只看 VGGT 点云
    python pipeline/gs_viewer.py --dataset 数据1-人体 --step 2    # VGGT + BA 对比
    python pipeline/gs_viewer.py --dataset 数据1-人体 --step 3    # 只看 3DGS 高斯
    python pipeline/gs_viewer.py --dataset 数据1-人体 --step all  # 全部叠在一起对比
"""

import os
import sys
import argparse
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.utils import get_output_dir, get_dataset_list


def load_pointcloud_from_npz(npz_path: str, max_points: int = 300000):
    """从 predictions.npz 加载 VGGT 点云 (world_points_from_depth)。"""
    data = np.load(npz_path, allow_pickle=True)
    wp = data["world_points_from_depth"]  # (S, H, W, 3)
    images = data["images"]               # (S, 3, H, W) or (S, H, W, 3)
    depth_conf = data.get("depth_conf", None)

    # 图像格式统一: (S, H, W, 3)
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))

    pts = wp.reshape(-1, 3)
    cols = images.reshape(-1, 3)

    # 过滤
    valid = np.linalg.norm(pts, axis=-1) > 1e-6
    if depth_conf is not None:
        conf = depth_conf.reshape(-1)
        valid = valid & (conf > 1e-6)
    pts, cols = pts[valid], cols[valid]

    # 采样
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    data.close()
    return pts, cols


def load_ply_pointcloud(filepath: str, max_points: int = 300000):
    """从 .ply 加载 3DGS Gaussians (只用位置和颜色)。"""
    with open(filepath, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        N = 0
        for line in header.decode().split("\n"):
            if line.startswith("element vertex"):
                N = int(line.split()[-1])
        raw = f.read()

    vertex_bytes = len(raw) // max(N, 1)
    means = np.zeros((N, 3), dtype=np.float32)
    colors = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        off = i * vertex_bytes
        if off + vertex_bytes > len(raw):
            break
        means[i] = struct.unpack_from("fff", raw, off)
        r, g, b = struct.unpack_from("fff", raw, off + 24)
        colors[i] = [
            max(0, min(1, r / 0.28209479177387814)),
            max(0, min(1, g / 0.28209479177387814)),
            max(0, min(1, b / 0.28209479177387814)),
        ]

    if len(means) > max_points:
        idx = np.random.choice(len(means), max_points, replace=False)
        means, colors = means[idx], colors[idx]
    return means, colors


def make_frustum_lines(extrinsic: np.ndarray, intrinsic: np.ndarray):
    """从相机外参生成 5 个关键点: 中心 + 4 个角 → 用于画线。"""
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    c2w_R, c2w_t = R.T, -R.T @ t   # world-to-cam → cam-to-world

    fy = intrinsic[1, 1]
    fx = intrinsic[0, 0]
    H = int(intrinsic[1, 2] * 2)
    W = int(intrinsic[0, 2] * 2)

    d = 0.3
    h = d * H / fy
    w = d * W / fx

    corners_cam = np.array([
        [0, 0, 0],
        [-w, -h, d],
        [w, -h, d],
        [w, h, d],
        [-w, h, d],
    ], dtype=np.float32)

    return corners_cam @ c2w_R.T + c2w_t


def add_frustum(server, frustum_pts, name_prefix, color_rgb, center):
    """向 viser 场景添加相机视锥体线条。"""
    pts = frustum_pts - center
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    segments = np.stack([np.stack([pts[a], pts[b]], axis=0) for a, b in edges], axis=0)
    server.scene.add_line_segments(
        name=name_prefix,
        points=segments.astype(np.float32),  # (N, 2, 3)
        colors=tuple(color_rgb),
        line_width=2.0,
    )


def main():
    parser = argparse.ArgumentParser(description="Step-by-step 3D Viewer")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--step", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="1=VGGT, 2=VGGT+BA, 3=3DGS, all=全部")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sample", type=int, default=300000)
    args = parser.parse_args()

    try:
        from viser import ViserServer
    except ImportError:
        print("viser not installed. Run: pip install viser")
        sys.exit(1)

    datasets = get_dataset_list()
    if args.dataset not in datasets:
        print(f"Unknown: {args.dataset}, available: {list(datasets.keys())}")
        sys.exit(1)

    output_dir = get_output_dir(args.dataset)
    pred_path = os.path.join(output_dir, "predictions.npz")
    ba_path = os.path.join(output_dir, "ba_result.npz")
    ply_path = os.path.join(output_dir, "gaussians.ply")

    # ---------- 加载各步数据 ----------
    vggt_pts = vggt_cols = None
    ba_ext = ba_intr = None
    gs_pts = gs_cols = None

    if args.step in ("1", "2", "all"):
        if os.path.exists(pred_path):
            vggt_pts, vggt_cols = load_pointcloud_from_npz(pred_path, args.sample)
            print(f"[Step 1] VGGT point cloud: {len(vggt_pts):,} points")
        else:
            print("[Step 1] predictions.npz not found, run vggt_inference.py first")

    if args.step in ("2", "all"):
        if os.path.exists(ba_path):
            ba_data = np.load(ba_path, allow_pickle=True)
            ba_ext = ba_data["extrinsic_opt"]
            ba_intr = ba_data["intrinsic"]
            print(f"[Step 2] BA optimized: {ba_data['reproj_before']:.2f} -> {ba_data['reproj_after']:.2f} px")
            ba_data.close()
        else:
            print("[Step 2] ba_result.npz not found")

    if args.step in ("3", "all"):
        if os.path.exists(ply_path):
            gs_pts, gs_cols = load_ply_pointcloud(ply_path, args.sample)
            print(f"[Step 3] 3DGS Gaussians: {len(gs_pts):,} points")
        else:
            print("[Step 3] gaussians.ply not found, run gaussian_splatting.py first")

    # 用 VGGT 点云计算场景中心
    if vggt_pts is not None:
        center = vggt_pts.mean(axis=0)
    elif gs_pts is not None:
        center = gs_pts.mean(axis=0)
    else:
        print("No data to display!")
        sys.exit(1)

    # ---------- 启动 viser ----------
    print(f"[Viewer] Starting http://localhost:{args.port}")
    server = ViserServer(host="0.0.0.0", port=args.port)

    # VGGT 点云：白色（仅在 step 1/2/all 显示）
    if vggt_pts is not None and args.step in ("1", "2", "all"):
        server.scene.add_point_cloud(
            name="/vggt_points",
            points=vggt_pts - center,
            colors=vggt_cols,
            point_size=0.003,
            point_shape="circle",
        )

    # BA 相机：绿色
    if ba_ext is not None and args.step in ("2", "all"):
        pred_data = np.load(pred_path, allow_pickle=True)
        vggt_ext = pred_data["extrinsic"]
        vggt_intr = pred_data["intrinsic"]
        S = vggt_ext.shape[0]

        for i in range(S):
            # VGGT 原始相机 (红色)
            ft_vggt = make_frustum_lines(vggt_ext[i], vggt_intr[i])
            add_frustum(server, ft_vggt, f"/vggt_cam/{i}", [255, 80, 80], center)
            # BA 优化后相机 (绿色)
            ft_ba = make_frustum_lines(ba_ext[i], ba_intr[i])
            add_frustum(server, ft_ba, f"/ba_cam/{i}", [80, 255, 80], center)
        pred_data.close()
        print("[Viewer] Red cameras = VGGT original, Green cameras = BA optimized")

    elif vggt_pts is not None and args.step == "1":
        # 仅 step 1: VGGT 相机（蓝色）
        pred_data = np.load(pred_path, allow_pickle=True)
        vggt_ext = pred_data["extrinsic"]
        vggt_intr = pred_data["intrinsic"]
        for i in range(vggt_ext.shape[0]):
            ft = make_frustum_lines(vggt_ext[i], vggt_intr[i])
            add_frustum(server, ft, f"/cam/{i}", [100, 180, 255], center)
        pred_data.close()
        print("[Viewer] Blue cameras = VGGT estimated poses")

    # 3DGS 高斯球（仅在 step 3/all）
    if gs_pts is not None:
        server.scene.add_point_cloud(
            name="/gs_gaussians",
            points=gs_pts - center,
            colors=gs_cols,
            point_size=0.004,
            point_shape="circle",
        )

    print(f"\n{'='*50}")
    print(f"  Viewer: http://localhost:{args.port}")
    print(f"  Step: {args.step}")
    print(f"  Controls: Left-drag=rotate, Right-drag=pan, Scroll=zoom")
    print(f"{'='*50}\n")

    try:
        import time as _time
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Viewer] Done.")


if __name__ == "__main__":
    main()

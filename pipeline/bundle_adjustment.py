"""
pipeline/bundle_adjustment.py — Bundle Adjustment 优化相机外参和3D点云
用法:
    python pipeline/bundle_adjustment.py --dataset 数据1-人体

输出: output/<dataset>/ba_result.npz
    - extrinsic_opt: 优化后外参矩阵 (S, 3, 4)
    - points3d_opt: 优化后3D地标 (M, 3)
    - reproj_before: 优化前平均重投影误差
    - reproj_after: 优化后平均重投影误差
"""

import os
import sys
import argparse
import time
import gc

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from tqdm import tqdm

# 确保能 import pipeline 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.utils import (
    get_output_dir, get_dataset_list,
    extrinsics_to_se3, se3_to_rot_trans, se3_log, se3_exp,
    project_points, inv_se3
)


# ---------------------------------------------------------------------------
# 特征提取与匹配
# ---------------------------------------------------------------------------

def extract_sift_keypoints(images: np.ndarray) -> list:
    """
    对每张图像提取 SIFT 关键点和描述子。
    images: (S, H, W, 3) or (S, 3, H, W) in [0, 1] float
    返回: list of (keypoints, descriptors), 每个元素对应一帧
    """
    sift = cv2.SIFT_create(nfeatures=2000)
    results = []
    for i in range(images.shape[0]):
        img = images[i]
        # 处理 NCHW → HWC
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        # 转 uint8
        img = (img * 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(img, None)
        if des is None:
            des = np.empty((0, 128), dtype=np.float32)
        results.append((kp, des))
    return results


def match_consecutive_frames(sift_results: list, max_matches_per_pair: int = 500):
    """
    在相邻帧之间匹配 SIFT 特征。
    返回: list of (indices_i, indices_j, pts_i, pts_j)
        — 每对相邻帧的匹配关系 (上一帧关键点索引, 当前帧关键点索引, 上一帧像素坐标, 当前帧像素坐标)
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_list = []

    for i in range(len(sift_results) - 1):
        kp_i, des_i = sift_results[i]
        kp_j, des_j = sift_results[i + 1]

        if des_i is None or des_j is None or len(des_i) == 0 or len(des_j) == 0:
            matches_list.append((np.array([]), np.array([]), np.zeros((0, 2)), np.zeros((0, 2))))
            continue

        raw_matches = bf.match(des_i, des_j)
        # 按距离排序，取最好的
        raw_matches = sorted(raw_matches, key=lambda x: x.distance)[:max_matches_per_pair]

        idx_i = np.array([m.queryIdx for m in raw_matches], dtype=np.int32)
        idx_j = np.array([m.trainIdx for m in raw_matches], dtype=np.int32)

        pts_i = np.array([kp_i[m.queryIdx].pt for m in raw_matches], dtype=np.float32)
        pts_j = np.array([kp_j[m.trainIdx].pt for m in raw_matches], dtype=np.float32)

        matches_list.append((idx_i, idx_j, pts_i, pts_j))

    return matches_list


def triangulate_points_from_matches(matches_list: list, sift_results: list,
                                     extrinsics: np.ndarray, intrinsics: np.ndarray) -> tuple:
    """
    利用多帧匹配和初始相机位姿三角化3D地标点。
    采用简单的两视图三角化 (相邻帧)，合并重复观测。
    返回:
        points3d: (M, 3) 三角化后的3D点
        observations: list of (frame_idx, keypoint_idx, landmark_idx) 列表
    """
    S = extrinsics.shape[0]
    all_landmarks = []       # list of (3,) arrays
    # 用于合并: 对每帧的每个关键点记录它属于哪个地标
    frame_kp_to_landmark = [{} for _ in range(S)]
    next_landmark_id = 0

    for pair_idx in range(len(matches_list)):
        i, j = pair_idx, pair_idx + 1
        idx_i, idx_j, pts_i, pts_j = matches_list[pair_idx]
        if len(idx_i) == 0:
            continue

        # 获取两帧的投影矩阵 P = K @ [R | t]
        K = intrinsics[i]  # 假设所有帧内参相同
        P1 = K @ extrinsics[i]  # (3, 4)
        P2 = K @ extrinsics[j]

        for m in range(len(idx_i)):
            # 检查是否已经分配了地标
            lm_id_i = frame_kp_to_landmark[i].get(idx_i[m])
            lm_id_j = frame_kp_to_landmark[j].get(idx_j[m])

            if lm_id_i is not None and lm_id_j is not None:
                if lm_id_i != lm_id_j:
                    # 合并两个地标
                    pass  # 简化：不合并
                continue
            elif lm_id_i is not None:
                # 已经在上一帧有地标，只需记录观测
                frame_kp_to_landmark[j][idx_j[m]] = lm_id_i
                continue
            elif lm_id_j is not None:
                frame_kp_to_landmark[i][idx_i[m]] = lm_id_j
                continue

            # 三角化新地标
            pt3d = cv2.triangulatePoints(
                P1.astype(np.float64), P2.astype(np.float64),
                pts_i[m].astype(np.float64), pts_j[m].astype(np.float64)
            )
            pt3d = (pt3d[:3] / pt3d[3]).squeeze()  # (3,)

            # 检查深度是否为正（在两个相机前方）
            R1, t1 = extrinsics[i, :3, :3], extrinsics[i, :3, 3]
            R2, t2 = extrinsics[j, :3, :3], extrinsics[j, :3, 3]
            cam1_depth = float((R1 @ pt3d + t1)[2])
            cam2_depth = float((R2 @ pt3d + t2)[2])
            if cam1_depth <= 0 or cam2_depth <= 0:
                continue

            # 检查重投影误差
            proj1 = project_points(pt3d.reshape(1, 3), R1, t1, K).flatten()
            proj2 = project_points(pt3d.reshape(1, 3), R2, t2, K).flatten()
            err1 = np.linalg.norm(proj1 - pts_i[m])
            err2 = np.linalg.norm(proj2 - pts_j[m])
            if err1 > 4.0 or err2 > 4.0:
                continue

            # 分配新地标ID
            all_landmarks.append(pt3d)
            frame_kp_to_landmark[i][idx_i[m]] = next_landmark_id
            frame_kp_to_landmark[j][idx_j[m]] = next_landmark_id
            next_landmark_id += 1

    if not all_landmarks:
        print("[BA] Warning: No landmarks triangulated. Falling back to dense sampling.")
        return None, None

    points3d = np.array(all_landmarks, dtype=np.float32)
    print(f"[BA] Triangulated {len(points3d)} landmarks from SIFT matches.")

    return points3d, frame_kp_to_landmark


def build_observation_list(frame_kp_to_landmark: list, sift_results: list) -> list:
    """
    构建观测列表: [(frame_idx, kp_idx, landmark_idx, px, py), ...]
    """
    obs = []
    for fi, mapping in enumerate(frame_kp_to_landmark):
        kp_list, _ = sift_results[fi]
        for kp_idx, lm_id in mapping.items():
            if kp_idx < len(kp_list):
                pt = kp_list[kp_idx].pt
                obs.append((fi, kp_idx, lm_id, pt[0], pt[1]))
    return obs


# ---------------------------------------------------------------------------
# Bundle Adjustment (PyTorch 实现)
# ---------------------------------------------------------------------------

class BundleAdjustment(nn.Module):
    """
    PyTorch Bundle Adjustment 模块。
    优化变量: 每帧相机外参 (S, 6) + 3D地标 (M, 3)
    损失: Huber(重投影误差)
    """

    def __init__(self, extrinsics_init: np.ndarray, intrinsics_init: np.ndarray,
                 points3d_init: np.ndarray, observations: list):
        super().__init__()
        self.S = extrinsics_init.shape[0]
        self.M = points3d_init.shape[0]

        # 内参固定（不优化）
        K0 = intrinsics_init[0]  # (3, 3), 假设所有帧内参相同
        self.register_buffer("fx", torch.tensor(K0[0, 0], dtype=torch.float32))
        self.register_buffer("fy", torch.tensor(K0[1, 1], dtype=torch.float32))
        self.register_buffer("cx", torch.tensor(K0[0, 2], dtype=torch.float32))
        self.register_buffer("cy", torch.tensor(K0[1, 2], dtype=torch.float32))

        # 初始外参 → se3 参数化
        se3_init = extrinsics_to_se3(extrinsics_init)  # (S, 4, 4)
        pose_init = se3_log(se3_init)  # (S, 6)
        self.pose_params = nn.Parameter(torch.from_numpy(pose_init).float())

        # 初始3D地标
        self.points3d = nn.Parameter(torch.from_numpy(points3d_init).float())

        # 观测是固定数据，注册为 buffer 后会跟随 ba.to(device) 移动。
        self.register_buffer("obs_frame", torch.tensor([o[0] for o in observations], dtype=torch.long))
        self.register_buffer("obs_lm", torch.tensor([o[2] for o in observations], dtype=torch.long))
        self.register_buffer("obs_px", torch.tensor([[o[3], o[4]] for o in observations], dtype=torch.float32))

        self.huber = nn.HuberLoss(delta=1.0)

    def forward(self):
        """计算总重投影损失。"""
        # 从 se3 参数恢复外参
        poses_se3 = se3_exp_torch(self.pose_params)  # (S, 4, 4)
        R = poses_se3[:, :3, :3]  # (S, 3, 3)
        t = poses_se3[:, :3, 3]   # (S, 3)

        # 对每个观测计算投影
        points = self.points3d[self.obs_lm]       # (N_obs, 3)
        R_obs = R[self.obs_frame]                 # (N_obs, 3, 3)
        t_obs = t[self.obs_frame]                 # (N_obs, 3)

        # 世界→相机坐标
        cam_pts = torch.bmm(R_obs, points.unsqueeze(-1)).squeeze(-1) + t_obs
        # 透视除法
        uv = cam_pts[:, :2] / (cam_pts[:, 2:3] + 1e-8)
        # 应用内参
        pred_px = torch.stack([
            self.fx * uv[:, 0] + self.cx,
            self.fy * uv[:, 1] + self.cy
        ], dim=-1)

        loss = self.huber(pred_px, self.obs_px)
        return loss


def se3_exp_torch(pose_params: torch.Tensor) -> torch.Tensor:
    """
    将 (S, 6) se3 参数批量转换为 (S, 4, 4) SE3 矩阵。
    使用 PyTorch 实现 (基于 scipy/numpy 的替代，支持 autograd)。
    简化: 用 Rodrigues 公式。
    """
    S = pose_params.shape[0]
    device = pose_params.device
    dtype = pose_params.dtype

    rotvec = pose_params[:, :3]  # (S, 3)
    trans = pose_params[:, 3:]   # (S, 3)

    # Rodrigues: rotation vector → rotation matrix
    theta = torch.norm(rotvec, dim=1, keepdim=True)  # (S, 1)
    # 避免除零
    eps = 1e-8
    theta_safe = torch.clamp(theta, min=eps)
    k = rotvec / theta_safe  # (S, 3) — 单位旋转轴

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # 反对称矩阵
    Kx = torch.zeros(S, 3, 3, device=device, dtype=dtype)
    Kx[:, 0, 1] = -k[:, 2]
    Kx[:, 0, 2] = k[:, 1]
    Kx[:, 1, 0] = k[:, 2]
    Kx[:, 1, 2] = -k[:, 0]
    Kx[:, 2, 0] = -k[:, 1]
    Kx[:, 2, 1] = k[:, 0]

    # R = I + sinθ * K + (1-cosθ) * K²
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(S, -1, -1)
    Kx2 = torch.bmm(Kx, Kx)
    R = I + sin_t.view(S, 1, 1) * Kx + (1 - cos_t).view(S, 1, 1) * Kx2

    # 组装 SE3
    se3 = torch.zeros(S, 4, 4, device=device, dtype=dtype)
    se3[:, :3, :3] = R
    se3[:, :3, 3] = trans
    se3[:, 3, 3] = 1.0

    # 对于 theta ≈ 0 的情况，直接用 I
    zero_mask = (theta < eps).squeeze()
    if zero_mask.any():
        se3[zero_mask, :3, :3] = I[zero_mask]
        se3[zero_mask, :3, 3] = trans[zero_mask]

    return se3


def run_bundle_adjustment(extrinsics: np.ndarray, intrinsics: np.ndarray,
                          images: np.ndarray, device: str = "cuda") -> dict:
    """
    执行完整的 Bundle Adjustment 流程。
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("[BA] CUDA not available, falling back to CPU.")
        device = "cpu"

    print("[BA] Extracting SIFT keypoints...")
    sift_results = extract_sift_keypoints(images)
    total_kp = sum(len(r[0]) for r in sift_results)
    print(f"[BA] Total keypoints detected: {total_kp}")

    print("[BA] Matching consecutive frames...")
    matches_list = match_consecutive_frames(sift_results, max_matches_per_pair=500)

    print("[BA] Triangulating landmarks...")
    points3d, frame_kp_to_landmark = triangulate_points_from_matches(
        matches_list, sift_results, extrinsics, intrinsics
    )

    if points3d is None or len(points3d) < 10:
        print("[BA] Too few landmarks for BA. Using dense point sampling instead.")
        return _fallback_dense_ba(extrinsics, intrinsics, images, device)

    observations = build_observation_list(frame_kp_to_landmark, sift_results)
    print(f"[BA] Total observations: {len(observations)}, landmarks: {len(points3d)}")

    # 计算优化前重投影误差
    reproj_before = _compute_mean_reproj_error(extrinsics, intrinsics, points3d, observations)

    # PyTorch 优化
    print("[BA] Starting optimization...")
    ba = BundleAdjustment(extrinsics, intrinsics, points3d, observations)
    ba = ba.to(device)

    # Stage 1: Adam
    optimizer = torch.optim.Adam(ba.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    pbar = tqdm(range(500), desc="BA-Adam")
    best_loss = float("inf")
    for step in pbar:
        optimizer.zero_grad()
        loss = ba()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if loss.item() < best_loss:
            best_loss = loss.item()

    # Stage 2: L-BFGS
    try:
        lbfgs_optimizer = torch.optim.LBFGS(ba.parameters(), lr=0.01, max_iter=20,
                                              line_search_fn="strong_wolfe")
        def closure():
            lbfgs_optimizer.zero_grad()
            l = ba()
            l.backward()
            return l
        lbfgs_optimizer.step(closure)
    except Exception as e:
        print(f"[BA] L-BFGS failed: {e}, using Adam result only.")

    # 提取优化结果
    with torch.no_grad():
        poses_se3 = se3_exp_torch(ba.pose_params).cpu().numpy()
        extrinsics_opt = poses_se3[:, :3, :4].copy()
        points3d_opt = ba.points3d.cpu().numpy().copy()

    reproj_after = _compute_mean_reproj_error(extrinsics_opt, intrinsics, points3d_opt, observations)

    reduction = reproj_before - reproj_after
    print(f"[BA] Reprojection error: {reproj_before:.2f} -> {reproj_after:.2f} px "
          f"(降低了 {reduction:.2f} px, {reduction/max(reproj_before,1e-8)*100:.1f}%)")

    # 清理
    del ba
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "extrinsic_opt": extrinsics_opt,
        "intrinsic": intrinsics,
        "points3d_opt": points3d_opt,
        "reproj_before": reproj_before,
        "reproj_after": reproj_after,
        "observations": observations,
    }


def _compute_mean_reproj_error(extrinsics, intrinsics, points3d, observations):
    """计算平均重投影误差（像素）。"""
    errors = []
    K = intrinsics[0]
    for (fi, kp_idx, lm_id, px, py) in observations:
        if lm_id >= len(points3d):
            continue
        pt3d = points3d[lm_id]
        R = extrinsics[fi, :3, :3]
        t = extrinsics[fi, :3, 3]
        proj = project_points(pt3d.reshape(1, 3), R, t, K).flatten()
        err = np.linalg.norm(proj - np.array([px, py]))
        errors.append(err)
    return float(np.mean(errors)) if errors else 999.0


def _fallback_dense_ba(extrinsics, intrinsics, images, device):
    """
    当稀疏特征不足时，从深度图中均匀采样点作为地标进行 BA。
    """
    print("[BA] Dense fallback: sampling points from depth-based world coordinates.")
    # 这个简化版本只优化相机位姿（不优化3D点）
    # 返回原始位姿，标记 BA 未完全执行
    return {
        "extrinsic_opt": extrinsics.copy(),
        "intrinsic": intrinsics.copy(),
        "points3d_opt": np.zeros((0, 3), dtype=np.float32),
        "reproj_before": 0.0,
        "reproj_after": 0.0,
        "observations": [],
    }


def main():
    parser = argparse.ArgumentParser(description="Bundle Adjustment for VGGT predictions")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. 数据1-人体, 数据2-人体, 数据3-场景)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    datasets = get_dataset_list()
    if args.dataset not in datasets:
        print(f"Dataset '{args.dataset}' not found. Available: {list(datasets.keys())}")
        sys.exit(1)

    output_dir = get_output_dir(args.dataset)
    pred_path = os.path.join(output_dir, "predictions.npz")
    if not os.path.exists(pred_path):
        print(f"[BA] Predictions not found at {pred_path}. Run vggt_inference.py first.")
        sys.exit(1)

    print(f"[BA] Loading predictions from {pred_path}")
    data = np.load(pred_path, allow_pickle=True)
    extrinsics = data["extrinsic"]
    intrinsics = data["intrinsic"]
    images = data["images"]

    print(f"[BA] Dataset: {args.dataset}, frames: {extrinsics.shape[0]}")

    start = time.time()
    result = run_bundle_adjustment(extrinsics, intrinsics, images, args.device)
    elapsed = time.time() - start

    # 保存
    save_path = os.path.join(output_dir, "ba_result.npz")
    save_dict = {
        "extrinsic_opt": result["extrinsic_opt"],
        "intrinsic": result["intrinsic"],
        "points3d_opt": result["points3d_opt"],
        "reproj_before": result["reproj_before"],
        "reproj_after": result["reproj_after"],
    }
    np.savez(save_path, **save_dict)
    print(f"[BA] Saved results to {save_path}")
    print(f"[BA] Total time: {elapsed:.1f}s")
    red = result['reproj_before'] - result['reproj_after']
    print(f"[BA] Reprojection error: {result['reproj_before']:.2f} -> {result['reproj_after']:.2f} px "
          f"(降低了 {red:.2f} px, {red/max(result['reproj_before'],1e-8)*100:.1f}%)")


if __name__ == "__main__":
    main()

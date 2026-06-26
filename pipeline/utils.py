"""
pipeline/utils.py — 公共工具函数
为 BA、3DGS 等模块提供旋转/位姿转换、文件IO等工具。
"""

import os
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# 确保能 import vggt 模块
_VGGT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vggt")
if _VGGT_ROOT not in sys.path:
    sys.path.append(_VGGT_ROOT)

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def get_dataset_output_name(dataset_name: str) -> str:
    """Return the output folder name for a dataset or video file."""
    root, ext = os.path.splitext(dataset_name)
    if ext.lower() in VIDEO_EXTENSIONS:
        return root
    return dataset_name


def get_output_dir(dataset_name: str) -> str:
    """返回 output/<dataset_name>/ 目录，自动创建。"""
    path = os.path.join(OUTPUT_ROOT, get_dataset_output_name(dataset_name))
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 位姿 / SE3 工具
# ---------------------------------------------------------------------------

def extrinsics_to_se3(extrinsics: np.ndarray) -> np.ndarray:
    """
    将 3x4 外参矩阵扩展为 4x4 SE3 矩阵。
    extrinsics: (S, 3, 4) → (S, 4, 4)
    """
    S = extrinsics.shape[0]
    se3 = np.tile(np.eye(4), (S, 1, 1))
    se3[:, :3, :4] = extrinsics
    return se3


def se3_to_rot_trans(se3: np.ndarray):
    """
    从 (S, 4, 4) 或 (4, 4) SE3 中提取旋转矩阵和平移向量。
    返回: R (..., 3, 3), T (..., 3)
    """
    R = se3[..., :3, :3].copy()
    T = se3[..., :3, 3].copy()
    return R, T


def rot_trans_to_se3(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    R (..., 3, 3), T (..., 3) → SE3 (..., 4, 4)
    """
    shape = R.shape[:-2]
    se3 = np.tile(np.eye(4), shape + (1, 1))
    se3[..., :3, :3] = R
    se3[..., :3, 3] = T
    return se3


def se3_log(se3: np.ndarray) -> np.ndarray:
    """
    SE3 → 李代数 se3 (6维向量: 前3维旋转, 后3维平移)。
    使用 scipy 的 rotvec 表示旋转部分。
    """
    if se3.ndim == 3:
        result = np.zeros((se3.shape[0], 6))
        for i in range(se3.shape[0]):
            result[i] = _single_se3_log(se3[i])
        return result
    else:
        return _single_se3_log(se3)


def _single_se3_log(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    rotvec = Rotation.from_matrix(R).as_rotvec()
    return np.concatenate([rotvec, t])


def se3_exp(se3_vec: np.ndarray) -> np.ndarray:
    """
    se3 李代数 (6维) → SE3 矩阵。
    """
    if se3_vec.ndim == 2:
        result = np.zeros((se3_vec.shape[0], 4, 4))
        for i in range(se3_vec.shape[0]):
            result[i] = _single_se3_exp(se3_vec[i])
        return result
    else:
        return _single_se3_exp(se3_vec)


def _single_se3_exp(vec: np.ndarray) -> np.ndarray:
    rotvec = vec[:3]
    t = vec[3:]
    R = Rotation.from_rotvec(rotvec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def project_points(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    将 3D 世界坐标点投影到像素坐标系。
    points_3d: (N, 3)
    R: (3, 3) — world-to-cam rotation
    t: (3,) — world-to-cam translation
    K: (3, 3) — camera intrinsics
    返回: (N, 2) pixel coords
    """
    # world → camera
    cam_pts = points_3d @ R.T + t  # (N, 3)
    # perspective division
    uv = cam_pts[:, :2] / (cam_pts[:, 2:3] + 1e-8)
    # intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * uv[:, 0] + cx
    v = fy * uv[:, 1] + cy
    return np.stack([u, v], axis=-1)


def inv_se3(se3: np.ndarray) -> np.ndarray:
    """
    快速求 SE3 逆矩阵。
    se3: (..., 4, 4)
    """
    R = se3[..., :3, :3]
    t = se3[..., :3, 3:4]
    R_inv = np.swapaxes(R, -1, -2) if R.ndim >= 3 else R.T
    t_inv = -R_inv @ t
    inv = np.tile(np.eye(4), se3.shape[:-2] + (1, 1))
    inv[..., :3, :3] = R_inv
    inv[..., :3, 3:4] = t_inv
    return inv


# ---------------------------------------------------------------------------
# 图像 / 掩码加载
# ---------------------------------------------------------------------------

def load_dataset_images_and_masks(dataset_path: str):
    """
    从数据集文件夹加载所有 rgb_*.png 和 msk_*.png。
    返回:
        rgb_paths: sorted list of rgb file paths
        msk_paths: sorted list of mask file paths (可能为空)
        image_names: sorted list of image basenames
    """
    rgb_paths = sorted([
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.startswith("rgb_") and f.endswith(".png")
    ])
    msk_paths = sorted([
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.startswith("msk_") and f.endswith(".png")
    ])
    image_names = [os.path.basename(p) for p in rgb_paths]
    return rgb_paths, msk_paths, image_names


def get_dataset_list():
    """
    返回数据集的字典映射 {name: path}。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "大作业数据")
    datasets = {}
    for name in os.listdir(data_root):
        full = os.path.join(data_root, name)
        if os.path.isdir(full):
            datasets[name] = full
        elif os.path.isfile(full) and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS:
            datasets[name] = full
            datasets[os.path.splitext(name)[0]] = full
    return datasets

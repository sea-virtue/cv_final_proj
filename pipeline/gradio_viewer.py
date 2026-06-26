"""
pipeline/gradio_viewer.py — Gradio Web 端 3D 查看器（三阶段对比）
用法: python pipeline/gradio_viewer.py

三个阶段分别渲染不同数据:
  Step 1 → VGGT 原始点云 + VGGT 相机
  Step 2 → VGGT 点云 + BA 优化相机（相机位置可见变化）
  Step 3 → 3DGS 高斯球 .ply 直接渲染（完全不同的点集）
"""

import os, sys, argparse
import json
from datetime import datetime
import cv2
import numpy as np
import gradio as gr
import trimesh
import torch
from scipy.spatial.transform import Rotation

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vggt"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.utils import get_output_dir, get_dataset_list, get_dataset_output_name


# ---------------------------------------------------------------------------
# 各步骤的 GLB 构建
# ---------------------------------------------------------------------------

def _build_glb_step1(predictions, output_dir, conf_thres, frame_filter,
                      show_cam, mask_black_bg, mask_white_bg, prediction_mode):
    """Step 1: VGGT 原始点云 + VGGT 相机"""
    from visual_util import predictions_to_glb
    return predictions_to_glb(
        predictions, conf_thres=conf_thres, filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg, mask_white_bg=mask_white_bg,
        show_cam=show_cam, mask_sky=False, target_dir=output_dir,
        prediction_mode=prediction_mode)


def _build_glb_step2(predictions, ba_data, output_dir, conf_thres, frame_filter,
                      show_cam, mask_black_bg, mask_white_bg, prediction_mode):
    """Step 2: VGGT 点云 + BA 优化相机"""
    from visual_util import predictions_to_glb
    pred_copy = dict(predictions)
    pred_copy["extrinsic"] = ba_data["extrinsic_opt"]
    return predictions_to_glb(
        pred_copy, conf_thres=conf_thres, filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg, mask_white_bg=mask_white_bg,
        show_cam=show_cam, mask_sky=False, target_dir=output_dir,
        prediction_mode=prediction_mode)


def _read_gaussian_ply(ply_path):
    """Read the Gaussian PLY format written by gaussian_splatting.py."""
    with open(ply_path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY file: {ply_path}")
            text = line.decode("utf-8").strip()
            header.append(text)
            if text == "end_header":
                break
        header_len = f.tell()
        raw = f.read()

    N = 0
    properties = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                N = int(parts[2])
            continue
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            properties.append(parts[2])

    if N <= 0 or not properties:
        raise ValueError(f"Could not read vertex layout from {ply_path}")

    floats_per_vertex = len(properties)
    if len(raw) // (floats_per_vertex * 4) != N and len(raw) // (62 * 4) == N:
        # Older generated files wrote 62 floats per vertex but declared only
        # 32 properties in the header. Keep them readable for existing results.
        floats_per_vertex = 62
        properties = (
            ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
            + [f"f_rest_{i}" for i in range(45)]
            + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        )
    vertex_bytes = floats_per_vertex * 4
    available_vertices = len(raw) // vertex_bytes
    N = min(N, available_vertices)
    values = np.frombuffer(raw[:N * vertex_bytes], dtype="<f4").reshape(N, floats_per_vertex)
    prop_idx = {name: i for i, name in enumerate(properties)}

    def cols(names, default=None):
        if all(name in prop_idx for name in names):
            return values[:, [prop_idx[name] for name in names]]
        if default is None:
            raise ValueError(f"Missing PLY properties: {names}")
        return np.tile(np.asarray(default, dtype=np.float32), (N, 1))

    means = cols(["x", "y", "z"])
    colors = cols(["f_dc_0", "f_dc_1", "f_dc_2"]) / 0.28209479177387814
    colors = np.clip(colors, 0.0, 1.0)
    opacities = values[:, prop_idx["opacity"]] if "opacity" in prop_idx else np.ones(N, dtype=np.float32)
    scales = cols(["scale_0", "scale_1", "scale_2"], default=[0.01, 0.01, 0.01])
    quats = cols(["rot_0", "rot_1", "rot_2", "rot_3"], default=[1.0, 0.0, 0.0, 0.0])
    return means, colors, opacities, scales, quats


def _add_gaussian_ellipsoid_preview(scene, means, colors_uint8, scales, quats,
                                    max_ellipsoids=1200, scale_multiplier=3.0):
    """Add a sampled ellipsoid mesh preview for Gaussian scale/rotation."""
    if len(means) == 0:
        return 0

    count = min(len(means), max_ellipsoids)
    idx = np.linspace(0, len(means) - 1, count, dtype=np.int64)
    base = trimesh.creation.uv_sphere(radius=1.0, count=[8, 8])
    vertices_all = []
    faces_all = []
    colors_all = []
    vertex_offset = 0

    safe_scales = np.clip(np.abs(scales[idx]), 1e-4, np.percentile(np.abs(scales), 95) * 2.0)
    safe_quats = quats[idx].copy()
    norms = np.linalg.norm(safe_quats, axis=1, keepdims=True)
    safe_quats = np.where(norms > 1e-8, safe_quats / norms, np.array([1.0, 0.0, 0.0, 0.0]))

    for mean, color, scale, quat in zip(means[idx], colors_uint8[idx], safe_scales, safe_quats):
        # Stored order is wxyz; scipy expects xyzw.
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        verts = (base.vertices * (scale * scale_multiplier)) @ rot.T + mean
        faces_all.append(base.faces + vertex_offset)
        vertices_all.append(verts)
        colors_all.append(np.tile(np.append(color[:3], 180), (len(base.vertices), 1)))
        vertex_offset += len(base.vertices)

    mesh = trimesh.Trimesh(
        vertices=np.vstack(vertices_all),
        faces=np.vstack(faces_all),
        vertex_colors=np.vstack(colors_all).astype(np.uint8),
        process=False,
    )
    scene.add_geometry(mesh, node_name="gaussian_ellipsoid_preview")
    return count


def _build_glb_step3(ply_path, predictions, ba_data, output_dir, conf_thres,
                     gaussian_display="Centers", ellipsoid_count=1200):
    """Step 3: 3DGS Gaussians from .ply."""
    means, colors, opacities, scales, quats = _read_gaussian_ply(ply_path)

    # 采样
    sample = min(len(means), 200000)
    if len(means) > sample:
        idx = np.random.choice(len(means), sample, replace=False)
        means = means[idx]
        colors = colors[idx]
        opacities = opacities[idx]
        scales = scales[idx]
        quats = quats[idx]

    colors_uint8 = (colors * 255).astype(np.uint8)

    # 构建 trimesh scene
    scene = trimesh.Scene()
    pc = trimesh.PointCloud(vertices=means, colors=colors_uint8)
    scene.add_geometry(pc)
    ellipsoid_added = 0
    if gaussian_display == "Centers + Ellipsoid Preview":
        ellipsoid_added = _add_gaussian_ellipsoid_preview(
            scene, means, colors_uint8, scales, quats, max_ellipsoids=ellipsoid_count
        )
    scene.metadata["ellipsoid_count"] = ellipsoid_added

    # 添加 BA 相机
    if ba_data is not None:
        from visual_util import integrate_camera_into_scene
        import matplotlib
        colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
        extrinsics = ba_data["extrinsic_opt"]
        S = extrinsics.shape[0]
        ext_4x4 = np.zeros((S, 4, 4))
        ext_4x4[:, :3, :4] = extrinsics
        ext_4x4[:, 3, 3] = 1

        lower = np.percentile(means, 5, axis=0)
        upper = np.percentile(means, 95, axis=0)
        scene_scale = np.linalg.norm(upper - lower)

        for i in range(S):
            c2w = np.linalg.inv(ext_4x4[i])
            rgba = colormap(i / S)
            color = tuple(int(255 * x) for x in rgba[:3])
            integrate_camera_into_scene(scene, c2w, color, scene_scale)

        from visual_util import get_opengl_conversion_matrix
        from scipy.spatial.transform import Rotation
        opengl = get_opengl_conversion_matrix()
        align = np.eye(4)
        align[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
        scene.apply_transform(np.linalg.inv(ext_4x4[0]) @ opengl @ align)

    return scene


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def build_glb_for_stage(dataset_name, stage, conf_thres, frame_filter,
                         show_cam, mask_black_bg, mask_white_bg, prediction_mode,
                         gaussian_display, ellipsoid_count):
    cache_version = "v2"
    output_dir = get_output_dir(dataset_name)
    pred_path = os.path.join(output_dir, "predictions.npz")
    ba_path = os.path.join(output_dir, "ba_result.npz")
    ply_path = os.path.join(output_dir, "gaussians.ply")

    if not os.path.exists(pred_path):
        return None, "❌ 未找到 predictions.npz", None, gr.Dropdown(choices=["All"], value="All")

    predictions = dict(np.load(pred_path, allow_pickle=True))
    S = predictions["extrinsic"].shape[0]
    frame_choices = ["All"] + [f"{i}: frame_{i}" for i in range(S)]
    if frame_filter is None or frame_filter not in frame_choices:
        frame_filter = "All"

    # 确定 GLB 缓存名
    safe_stage = stage.replace(" ", "_").replace(":", "").replace("+", "_")
    display_tag = gaussian_display.replace(" ", "_").replace("+", "plus")
    glb_name = f"viewer_{safe_stage}_{cache_version}_conf{conf_thres}_{display_tag}_ell{ellipsoid_count}.glb"
    glb_path = os.path.join(output_dir, glb_name)

    # 缓存命中直接返回
    if os.path.exists(glb_path):
        log = _make_log(stage, S, conf_thres, predictions, ba_path, ply_path, cached=True)
        return glb_path, log, _gallery(predictions), gr.Dropdown(choices=frame_choices, value=frame_filter)

    try:
        if stage == "Step 1: VGGT":
            scene = _build_glb_step1(predictions, output_dir, conf_thres, frame_filter,
                                      show_cam, mask_black_bg, mask_white_bg, prediction_mode)
        elif stage == "Step 2: VGGT + BA":
            if not os.path.exists(ba_path):
                # 回退
                scene = _build_glb_step1(predictions, output_dir, conf_thres, frame_filter,
                                          show_cam, mask_black_bg, mask_white_bg, prediction_mode)
            else:
                ba_data = np.load(ba_path, allow_pickle=True)
                scene = _build_glb_step2(predictions, ba_data, output_dir, conf_thres, frame_filter,
                                          show_cam, mask_black_bg, mask_white_bg, prediction_mode)
                ba_data.close()
        else:  # Step 3
            if not os.path.exists(ply_path):
                # 回退到 step 2
                if os.path.exists(ba_path):
                    ba_data = np.load(ba_path, allow_pickle=True)
                    scene = _build_glb_step2(predictions, ba_data, output_dir, conf_thres, frame_filter,
                                              show_cam, mask_black_bg, mask_white_bg, prediction_mode)
                    ba_data.close()
                else:
                    scene = _build_glb_step1(predictions, output_dir, conf_thres, frame_filter,
                                              show_cam, mask_black_bg, mask_white_bg, prediction_mode)
            else:
                ba_data = np.load(ba_path, allow_pickle=True) if os.path.exists(ba_path) else None
                scene = _build_glb_step3(
                    ply_path, predictions, ba_data, output_dir, conf_thres,
                    gaussian_display=gaussian_display, ellipsoid_count=ellipsoid_count
                )
                if ba_data is not None:
                    ba_data.close()

        scene.export(file_obj=glb_path)
        log = _make_log(
            stage, S, conf_thres, predictions, ba_path, ply_path, cached=False,
            gaussian_display=gaussian_display, ellipsoid_count=ellipsoid_count
        )
        return glb_path, log, _gallery(predictions), gr.Dropdown(choices=frame_choices, value=frame_filter)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ {e}", None, gr.Dropdown(choices=frame_choices, value=frame_filter)


def _make_log(stage, S, conf_thres, predictions, ba_path, ply_path, cached=False,
              gaussian_display="Centers", ellipsoid_count=1200):
    cache_tag = "(缓存)" if cached else "(新生成)"
    if "Step 3" in stage:
        log = f"✅ {stage} {cache_tag} | {S} 帧 | 3DGS 完整高斯采样"
    else:
        log = f"✅ {stage} {cache_tag} | {S} 帧 | 阈值 {conf_thres}%"

    if "Step 1" in stage:
        wp = predictions["world_points_from_depth"]
        valid = int((np.linalg.norm(wp, axis=-1) > 1e-6).sum())
        log += f"\n📦 VGGT 原始点云: {valid:,} 有效点"
        log += f"\n📷 相机: VGGT 直接预测"

    elif "Step 2" in stage:
        if os.path.exists(ba_path):
            ba = np.load(ba_path, allow_pickle=True)
            b, a = float(ba["reproj_before"]), float(ba["reproj_after"])
            log += f"\n🔧 BA 优化: 重投影误差 {b:.2f} → {a:.2f} px (降低 {b-a:.2f})"
            log += f"\n📷 相机: BA 优化后（绿色）vs VGGT 原始（红色有明显位移）"
            ba.close()
        else:
            log += "\n⚠️ BA 结果未找到"

    elif "Step 3" in stage:
        if os.path.exists(ply_path):
            N = len(_read_gaussian_ply(ply_path)[0])
            log += f"\n✨ 3DGS 高斯球: {N:,} 个"
            if gaussian_display == "Centers + Ellipsoid Preview":
                log += f"\n🎯 当前显示中心点 + 最多 {ellipsoid_count:,} 个采样椭球预览"
            else:
                log += f"\n🎯 当前显示 Gaussian 中心点；可切换椭球预览查看尺度/旋转"
        else:
            log += "\n⚠️ 3DGS PLY 未找到"

    return log


def _gallery(predictions):
    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    S = min(images.shape[0], 16)
    return [(images[i] * 255).astype(np.uint8) for i in range(S)]


def _frame_index_from_filter(frame_filter):
    if not frame_filter or frame_filter == "All":
        return 0
    try:
        return int(str(frame_filter).split(":", 1)[0])
    except ValueError:
        return 0


def _look_at_world_to_camera(eye, target):
    """Build an OpenCV-style world-to-camera matrix from eye/target."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = np.cross(world_up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    down = np.cross(forward, right)
    down = down / (np.linalg.norm(down) + 1e-8)

    R = np.stack([right, down, forward], axis=0)
    t = -R @ eye
    extrinsic = np.zeros((3, 4), dtype=np.float32)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic


def _orbit_view_to_extrinsic(view_state_json, means):
    if not view_state_json:
        return None
    try:
        state = json.loads(view_state_json)
    except json.JSONDecodeError:
        return None
    theta = state.get("theta")
    phi = state.get("phi")
    radius = state.get("radius")
    if theta is None or phi is None or radius is None or radius <= 0:
        return None

    target = np.median(means, axis=0)
    offset = np.array([
        radius * np.sin(phi) * np.sin(theta),
        radius * np.cos(phi),
        radius * np.sin(phi) * np.cos(theta),
    ], dtype=np.float32)
    eye = target + offset
    return _look_at_world_to_camera(eye, target)


def render_splatting_rgb(dataset_name, frame_filter):
    """Render a true gsplat RGB image from the selected dataset camera."""
    if not dataset_name:
        return None, "请选择数据集。"
    if not torch.cuda.is_available():
        return None, "CUDA 不可用，无法使用 gsplat 渲染 RGB 图。"

    from pipeline.gaussian_splatting import render_gaussians

    output_dir = get_output_dir(dataset_name)
    pred_path = os.path.join(output_dir, "predictions.npz")
    ba_path = os.path.join(output_dir, "ba_result.npz")
    ply_path = os.path.join(output_dir, "gaussians.ply")
    if not os.path.exists(pred_path):
        return None, f"未找到 {pred_path}"
    if not os.path.exists(ply_path):
        return None, f"未找到 {ply_path}"

    pred = np.load(pred_path, allow_pickle=True)
    images = pred["images"]
    intrinsics = pred["intrinsic"]
    extrinsics = pred["extrinsic"]
    if os.path.exists(ba_path):
        ba = np.load(ba_path, allow_pickle=True)
        extrinsics = ba["extrinsic_opt"]
        ba.close()

    frame_idx = _frame_index_from_filter(frame_filter)
    frame_idx = max(0, min(frame_idx, extrinsics.shape[0] - 1))
    if images.ndim == 4 and images.shape[1] == 3:
        height, width = images.shape[2], images.shape[3]
    else:
        height, width = images.shape[1], images.shape[2]
    pred.close()

    means, colors, opacities, scales, quats = _read_gaussian_ply(ply_path)
    valid = np.isfinite(means).all(axis=1) & np.isfinite(colors).all(axis=1)
    valid &= np.isfinite(scales).all(axis=1) & np.isfinite(quats).all(axis=1)
    valid &= opacities > 1e-4
    means, colors = means[valid], colors[valid]
    opacities, scales, quats = opacities[valid], scales[valid], quats[valid]
    scales = np.clip(scales, 1e-4, np.percentile(scales, 99) * 2.0)

    device = "cuda"
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[:3, :4] = torch.from_numpy(extrinsics[frame_idx]).to(device=device, dtype=torch.float32)
    K = torch.from_numpy(intrinsics[frame_idx]).to(device=device, dtype=torch.float32)

    try:
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
        image_np = image.clamp(0.0, 1.0).detach().cpu().numpy()
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        return None, f"渲染失败：{exc}"

    max_value = float(image_np.max()) if image_np.size else 0.0
    if max_value < 1e-4:
        return None, "渲染结果接近纯黑：当前相机视角下没有可见高斯，或相机/坐标系仍需检查。"

    render_dir = os.path.join(output_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(render_dir, f"web_splat_frame_{frame_idx:04d}_{stamp}.png")
    cv2.imwrite(save_path, cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return save_path, f"已按显示帧相机渲染 3DGS RGB 图：{save_path}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui():
    datasets = get_dataset_list()
    choices = []
    seen = set()
    for name in datasets.keys():
        output_name = get_dataset_output_name(name)
        if output_name not in seen:
            choices.append(output_name)
            seen.add(output_name)

    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(theme=theme, title="3D Viewer",
                   css=""".log * {font-size:16px!important; text-align:center!important;}
                          .log h2 {margin:4px 0;}""",
                   js="""
                   () => {
                     window.getModel3DOrbitState = () => {
                       const root = document.querySelector('#model3d-viewer');
                       const modelViewer = root ? root.querySelector('model-viewer') : document.querySelector('model-viewer');
                       if (!modelViewer || typeof modelViewer.getCameraOrbit !== 'function') {
                         return [''];
                       }
                       const orbit = modelViewer.getCameraOrbit();
                       const unitValue = (value, unit) => {
                         if (value && typeof value.to === 'function') {
                           return Number(value.to(unit).value);
                         }
                         if (value && typeof value.value === 'number') {
                           return Number(value.value);
                         }
                         return Number(value);
                       };
                       const theta = unitValue(orbit.theta, 'rad');
                       const phi = unitValue(orbit.phi, 'rad');
                       const radius = unitValue(orbit.radius, 'm');
                       if (!Number.isFinite(theta) || !Number.isFinite(phi) || !Number.isFinite(radius)) {
                         return [''];
                       }
                       return [JSON.stringify({theta, phi, radius})];
                     };
                     window.saveCurrentModel3DView = () => {
                       const root = document.querySelector('#model3d-viewer');
                       const canvas = root ? root.querySelector('canvas') : document.querySelector('gradio-model3d canvas, model-viewer canvas, canvas');
                       if (!canvas) {
                         alert('没有找到 3D 视图画布，请先加载模型。');
                         return ['未找到 3D 视图画布，请先加载模型。'];
                       }
                       try {
                         const dataUrl = canvas.toDataURL('image/png');
                         const a = document.createElement('a');
                         const stamp = new Date().toISOString().replace(/[:.]/g, '-');
                         a.href = dataUrl;
                         a.download = `gradio_3d_view_${stamp}.png`;
                         document.body.appendChild(a);
                         a.click();
                         a.remove();
                         return ['已保存当前浏览器 3D 视图 PNG。'];
                       } catch (err) {
                         console.error(err);
                         alert('保存失败：浏览器阻止读取画布，或 3D 组件使用了跨域资源。');
                         return ['保存失败：浏览器阻止读取画布。'];
                       }
                     };
                   }
                   """) as demo:

        gr.HTML("<h1>🏛️ 三维重建 — 三阶段对比</h1>")

        with gr.Row():
            with gr.Column(scale=2):
                dataset_dd = gr.Dropdown(choices=choices, label="📂 数据集",
                                          value=choices[0] if choices else None)

                stage_radio = gr.Radio(
                    ["Step 1: VGGT", "Step 2: VGGT + BA", "Step 3: 3DGS"],
                    label="🔍 阶段", value="Step 2: VGGT + BA")

                gallery = gr.Gallery(label="📷 输入图像", columns=4, height="220px",
                                      object_fit="contain", show_download_button=False)

                conf_thres = gr.Slider(0, 100, 50, step=0.1, label="置信度阈值 (%)")
                frame_filter = gr.Dropdown(["All"], "All", label="显示帧")
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="预测模式", value="Depthmap and Camera Branch")
                gaussian_display = gr.Radio(
                    ["Centers", "Centers + Ellipsoid Preview"],
                    label="3DGS 显示", value="Centers")
                ellipsoid_count = gr.Slider(
                    100, 5000, 1200, step=100, label="椭球预览数量")
                with gr.Row():
                    show_cam = gr.Checkbox(label="📷 显示相机", value=True)
                    mask_black_bg = gr.Checkbox(label="⬛ 过滤黑背景", value=False)
                    mask_white_bg = gr.Checkbox(label="⬜ 过滤白背景", value=False)

            with gr.Column(scale=4):
                log = gr.Markdown("👈 选择数据集", elem_classes=["log"])
                model_3d = gr.Model3D(height=550, label="3D 重建结果", elem_id="model3d-viewer")
                with gr.Row():
                    save_view_btn = gr.Button("保存当前视图 PNG", variant="secondary")
                    render_rgb_btn = gr.Button("渲染3DGS RGB图", variant="primary")
                    save_view_status = gr.Markdown("")
                rendered_rgb = gr.Image(label="3DGS RGB 渲染图", type="filepath", height=320)

        def reload(ds, stage, conf, ff, cam, mb, mw, pm, gd, ec):
            if not ds:
                return None, "请选择数据集", None, gr.Dropdown(choices=["All"], value="All")
            return build_glb_for_stage(ds, stage, conf, ff, cam, mb, mw, pm, gd, int(ec))

        ins = [dataset_dd, stage_radio, conf_thres, frame_filter,
               show_cam, mask_black_bg, mask_white_bg, prediction_mode,
               gaussian_display, ellipsoid_count]
        outs = [model_3d, log, gallery, frame_filter]

        for t in [dataset_dd.change, stage_radio.change, conf_thres.change,
                  frame_filter.change, show_cam.change, mask_black_bg.change,
                  mask_white_bg.change, prediction_mode.change,
                  gaussian_display.change, ellipsoid_count.change]:
            t(fn=reload, inputs=ins, outputs=outs)

        save_view_btn.click(
            fn=None,
            inputs=[],
            outputs=[save_view_status],
            js="() => window.saveCurrentModel3DView ? window.saveCurrentModel3DView() : ['保存函数尚未初始化。']",
        )
        render_rgb_btn.click(
            fn=render_splatting_rgb,
            inputs=[dataset_dd, frame_filter],
            outputs=[rendered_rgb, save_view_status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"🏛️  Gradio 3D Viewer → http://localhost:{args.port}")
    print("   切换 Step 1/2/3 看三个阶段的不同数据")
    demo = build_ui()
    demo.queue(max_size=5).launch(server_name="0.0.0.0", server_port=args.port,
                                   share=args.share, show_error=True)


if __name__ == "__main__":
    main()

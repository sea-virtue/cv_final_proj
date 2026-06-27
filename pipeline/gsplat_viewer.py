"""
Interactive 3DGS viewer using the installed gsplat wheel.

Usage:
    python pipeline/gsplat_viewer.py --dataset 数据1-人体 --port 8080
"""

import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.utils import get_dataset_list, get_output_dir


C0 = 0.28209479177387814


def _camera_center_from_w2c(extrinsic):
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :4] = extrinsic.astype(np.float32)
    return np.linalg.inv(w2c)


def _missing_modules():
    required = ["viser", "splines", "nerfview"]
    missing = []
    for name in required:
        try:
            __import__(name)
        except Exception as exc:
            missing.append(f"{name} ({exc})")
    return missing


def _resolve_ply(dataset, ply):
    if ply:
        ply_path = Path(ply).expanduser().resolve()
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY not found: {ply_path}")
        return ply_path, ply_path.parent

    datasets = get_dataset_list()
    if dataset not in datasets:
        raise FileNotFoundError(f"Dataset '{dataset}' not found. Available: {list(datasets.keys())}")

    output_dir = Path(get_output_dir(dataset))
    ply_path = output_dir / "gaussians.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}. Run gsplat_training.py first.")
    return ply_path, output_dir


def _read_project_or_gsplat_ply(ply_path):
    """Read both standard gsplat PLY and older project PLY files."""
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
        raw = f.read()

    n = 0
    properties = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n = int(parts[2])
            continue
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            properties.append(parts[2])

    floats_per_vertex = len(properties)
    if n <= 0 or floats_per_vertex <= 0:
        raise ValueError(f"Could not parse PLY header: {ply_path}")
    if len(raw) // (floats_per_vertex * 4) != n and len(raw) // (62 * 4) == n:
        floats_per_vertex = 62
        properties = (
            ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
            + [f"f_rest_{i}" for i in range(45)]
            + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        )

    values = np.frombuffer(raw[: n * floats_per_vertex * 4], dtype="<f4").reshape(n, floats_per_vertex)
    prop_idx = {name: i for i, name in enumerate(properties)}

    def cols(names, default=None):
        if all(name in prop_idx for name in names):
            return values[:, [prop_idx[name] for name in names]]
        if default is None:
            raise ValueError(f"Missing PLY properties: {names}")
        return np.tile(np.asarray(default, dtype=np.float32), (n, 1))

    means = cols(["x", "y", "z"])
    sh0 = cols(["f_dc_0", "f_dc_1", "f_dc_2"])
    standard_sh = float(np.nanmin(sh0)) < -0.01 or float(np.nanmax(sh0)) > 0.35
    if not standard_sh:
        rgb = np.clip(sh0 / C0, 0.0, 1.0)
        sh0 = (rgb - 0.5) / C0
    sh0 = sh0[:, None, :]

    f_rest_names = sorted(
        (name for name in properties if name.startswith("f_rest_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    if f_rest_names:
        f_rest = cols(f_rest_names)
        if f_rest.shape[1] % 3 == 0:
            k_minus_1 = f_rest.shape[1] // 3
            shn = f_rest.reshape(n, 3, k_minus_1).swapaxes(1, 2)
        else:
            shn = np.zeros((n, 0, 3), dtype=np.float32)
    else:
        shn = np.zeros((n, 0, 3), dtype=np.float32)

    opacities = values[:, prop_idx["opacity"]] if "opacity" in prop_idx else np.zeros(n, dtype=np.float32)
    if np.isfinite(opacities).any() and 0.0 <= float(np.nanmin(opacities)) and float(np.nanmax(opacities)) <= 1.0:
        opacities = np.log(np.clip(opacities, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - opacities, 1e-6, 1.0))

    scales = cols(["scale_0", "scale_1", "scale_2"], default=[0.01, 0.01, 0.01])
    if np.isfinite(scales).any() and float(np.nanmin(scales)) > 0.0:
        scales = np.log(np.clip(scales, 1e-8, None))

    quats = cols(["rot_0", "rot_1", "rot_2", "rot_3"], default=[1.0, 0.0, 0.0, 0.0])
    colors = np.concatenate([sh0, shn], axis=1).astype(np.float32)
    return {
        "means": torch.from_numpy(np.ascontiguousarray(means)).float(),
        "quats": torch.from_numpy(np.ascontiguousarray(quats)).float(),
        "scales": torch.from_numpy(np.ascontiguousarray(scales)).float(),
        "opacities": torch.from_numpy(np.ascontiguousarray(opacities)).float(),
        "colors": torch.from_numpy(np.ascontiguousarray(colors)).float(),
    }


def _load_inside_camera(dataset, output_dir, center_shift, frame=None):
    if not dataset:
        return None

    output_dir = Path(output_dir)
    ba_path = output_dir / "ba_result.npz"
    pred_path = output_dir / "predictions.npz"
    if ba_path.exists():
        data = np.load(ba_path, allow_pickle=True)
        extrinsics = data["extrinsic_opt"].astype(np.float32)
        intrinsics = data["intrinsic"].astype(np.float32) if "intrinsic" in data else None
        source = "BA"
    elif pred_path.exists():
        data = np.load(pred_path, allow_pickle=True)
        extrinsics = data["extrinsic"].astype(np.float32)
        intrinsics = data["intrinsic"].astype(np.float32) if "intrinsic" in data else None
        source = "VGGT"
    else:
        return None

    idx = len(extrinsics) // 2 if frame is None else int(np.clip(frame, 0, len(extrinsics) - 1))
    c2w = _camera_center_from_w2c(extrinsics[idx])
    c2w[:3, 3] -= center_shift

    fov = math.radians(60.0)
    if pred_path.exists() and intrinsics is not None:
        pred = np.load(pred_path, allow_pickle=True)
        if "images" in pred:
            images = pred["images"]
            h = images.shape[-2] if images.ndim == 4 and images.shape[1] == 3 else images.shape[1]
            fy = float(intrinsics[idx, 1, 1])
            if fy > 1e-6:
                fov = float(2.0 * math.atan(h / (2.0 * fy)))
        pred.close()
    data.close()

    return {
        "c2w": c2w.astype(np.float32),
        "fov": fov,
        "frame": idx,
        "source": source,
    }


def _apply_camera_pose(server, client, pose, look_distance):
    import viser.transforms as vt

    c2w = pose["c2w"]
    rotation = c2w[:3, :3]
    position = c2w[:3, 3]
    forward = rotation[:, 2]
    up = -rotation[:, 1]

    with server.atomic():
        client.camera.wxyz = vt.SO3.from_matrix(rotation).wxyz
        client.camera.position = position
        client.camera.look_at = position + forward * look_distance
        client.camera.up_direction = up
        client.camera.fov = pose["fov"]


def _launch_viewer(args, ply_path, output_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat interactive rendering.")

    import viser
    from nerfview import CameraState, RenderTabState, Viewer
    from gsplat.rendering import rasterization

    device = torch.device("cuda")
    splats = _read_project_or_gsplat_ply(ply_path)
    center_shift = np.zeros(3, dtype=np.float32)
    if args.center:
        means_np = splats["means"].numpy()
        lower = np.percentile(means_np, 5, axis=0)
        upper = np.percentile(means_np, 95, axis=0)
        center_shift = ((lower + upper) * 0.5).astype(np.float32)
        splats["means"] = splats["means"] - torch.from_numpy(center_shift)
        print(f"[gsplat viewer] Recentered scene by {center_shift.tolist()} (viewer only)")
    bounds_np = splats["means"].numpy()
    lower = np.percentile(bounds_np, 5, axis=0)
    upper = np.percentile(bounds_np, 95, axis=0)
    look_distance = float(max(np.linalg.norm(upper - lower) * 0.25, 0.25))
    initial_camera = _load_inside_camera(args.dataset, output_dir, center_shift, args.view_frame)
    view_mode = args.view_mode
    if view_mode == "auto":
        view_mode = "inside" if args.dataset and "场景" in args.dataset else "orbit"
    splats = {key: value.to(device) for key, value in splats.items()}
    splats["quats"] = F.normalize(splats["quats"], p=2, dim=-1)

    sh_count = splats["colors"].shape[1]
    sh_degree = int(math.sqrt(sh_count) - 1) if int(math.sqrt(sh_count)) ** 2 == sh_count else None
    print(f"[gsplat viewer] Loaded {len(splats['means']):,} Gaussians")
    print(f"[gsplat viewer] Using gsplat from: {importlib.util.find_spec('gsplat').origin}")

    def render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        k = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)
        viewmat = c2w.inverse().contiguous()

        with torch.inference_mode():
            render_colors, render_alphas, info = rasterization(
                splats["means"],
                splats["quats"],
                splats["scales"].exp(),
                splats["opacities"].sigmoid(),
                splats["colors"],
                viewmats=viewmat[None],
                Ks=k[None],
                width=width,
                height=height,
                sh_degree=sh_degree,
                near_plane=0.01,
                far_plane=1e10,
                backgrounds=torch.zeros((1, 3), device=device),
                render_mode="RGB",
                packed=False,
            )

        return render_colors[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()

    server = viser.ViserServer(port=args.port, verbose=False)
    Viewer(
        server=server,
        render_fn=render_fn,
        output_dir=Path(output_dir),
        mode="rendering",
    )

    if initial_camera is not None:
        with server.gui.add_folder("View Presets"):
            inside_button = server.gui.add_button("Inside view")

        @inside_button.on_click
        def _inside_click(event):
            _apply_camera_pose(server, event.client, initial_camera, look_distance)

        if view_mode == "inside":
            @server.on_client_connect
            def _set_initial_inside_view(client):
                time.sleep(0.2)
                _apply_camera_pose(server, client, initial_camera, look_distance)

            print(
                "[gsplat viewer] Initial view: inside "
                f"({initial_camera['source']} frame {initial_camera['frame']})"
            )
    elif view_mode == "inside":
        print("[gsplat viewer] Warning: no predictions/BA camera found; inside view is unavailable.")

    print(f"[gsplat viewer] Running on http://localhost:{args.port}")
    print("[gsplat viewer] Ctrl+C to exit.")
    while True:
        time.sleep(100000)


def main():
    parser = argparse.ArgumentParser(description="Launch a gsplat interactive viewer using the installed gsplat wheel.")
    parser.add_argument("--dataset", help="Project dataset name, e.g. 数据1-人体")
    parser.add_argument("--ply", help="Direct path to a 3DGS PLY. Overrides --dataset.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output-dir", default=None, help="Viewer output directory. Defaults to the dataset output dir.")
    parser.add_argument("--center", action="store_true", default=True,
                        help="Recenter Gaussians around a robust bounding-box center for easier viewing.")
    parser.add_argument("--no-center", dest="center", action="store_false",
                        help="Disable viewer-only recentering.")
    parser.add_argument("--view-mode", choices=["auto", "orbit", "inside"], default="auto",
                        help="Initial camera mode. Auto uses inside view for scene datasets and orbit view otherwise.")
    parser.add_argument("--view-frame", type=int, default=None,
                        help="Training camera frame used for the inside view. Defaults to the middle frame.")
    args = parser.parse_args()

    if not args.dataset and not args.ply:
        parser.error("Please provide --dataset or --ply.")

    missing = _missing_modules()
    if missing:
        print("[gsplat viewer] Missing Python modules:", ", ".join(missing))
        print("Install the viewer dependencies:")
        print("  python -m pip install viser")
        print("  python -m pip install splines")
        print("  python -m pip install 'git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787'")
        sys.exit(1)

    ply_path, default_output_dir = _resolve_ply(args.dataset, args.ply)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[gsplat viewer] Launching installed-wheel gsplat viewer")
    print(f"[gsplat viewer] PLY: {ply_path}")
    _launch_viewer(args, ply_path, output_dir)


if __name__ == "__main__":
    main()

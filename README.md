# 三维重建与高斯绘制 — 课程项目

> 多视角图像 → VGGT 求相机参数和初步点云 → Bundle Adjustment 优化 → 3D Gaussian Splatting 渲染

---

## 快速开始

```bash
# 0. 激活环境并设置 CUDA 路径
source setup_env.sh

# 1. 运行 Pipeline（以数据1-人体为例）
python pipeline/vggt_inference.py   --dataset 数据1-人体     # VGGT推理
python pipeline/bundle_adjustment.py --dataset 数据1-人体     # BA优化
python pipeline/gaussian_splatting.py --dataset 数据1-人体    # 3DGS训练

# 视频数据集会自动抽帧，输出到 output/数据3-场景/
python pipeline/vggt_inference.py   --dataset 数据3-场景.mp4
# 可选：抽全部帧，或每 N 帧抽一帧；VGGT 不建议一次输入几百帧
python pipeline/vggt_inference.py   --dataset 数据3-场景.mp4 --video-all-frames
python pipeline/vggt_inference.py   --dataset 数据3-场景.mp4 --video-frame-stride 10

# 3. 查看结果
python pipeline/inspect_results.py  --dataset 数据1-人体     # 文本摘要+可视化PNG
python pipeline/gradio_viewer.py                         # Web端3D交互查看，可切换 Step 1/2/3 并保存当前视图PNG
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `pipeline/README.md` | 完整使用文档（环境配置、参数说明、FAQ） |
| `pipeline/vggt_inference.py` | VGGT 推理，输出相机位姿和点云 |
| `pipeline/bundle_adjustment.py` | BA 联合优化相机外参和3D点 |
| `pipeline/gaussian_splatting.py` | 3DGS 训练，输出 .ply 高斯球 |
| `pipeline/gradio_viewer.py` | 🌐 Gradio Web 查看器（推荐查看方式，可在菜单中比较 Step 1/2/3） |
| `pipeline/render_gaussian_image.py` | 备用：从训练好的 3DGS 结果按指定相机视角渲染 PNG |
| `pipeline/gs_viewer.py` | 旧版 viser 查看器，当前不再作为主查看入口 |
| `pipeline/inspect_results.py` | 终端摘要 + PNG 可视化 |
| `pipeline/vggt_improvements.md` | VGGT 改进方法调研文档 |
| `vggt/` | VGGT 官方代码 |
| `大作业数据/` | 数据集（人体 ×2 + 场景视频） |
| `output/` | 所有输出文件 |

## 数据集

- **数据1-人体**：16 张人体 rgb 图 + 16 张 mask
- **数据2-人体**：16 张人体 rgb 图 + 16 张 mask
- **数据3-场景.mp4**：1 段场景视频，`vggt_inference.py` 会自动抽帧并写入 `output/数据3-场景/frames/`

## 实验结果摘要

| 数据集 | 帧数 | BA前误差 | BA后误差 | 改善 | Gaussians |
|--------|------|----------|----------|------|-----------|
| 数据1-人体 | 16 | 15.3296 px | 13.2044 px | 13.9% | 150,000 |
| 数据2-人体 | 16 | 19.0536 px | 14.4084 px | 24.4% | 150,000 |
| 数据3-场景 | 64 | 83.5282 px | 17.7731 px | 78.7% | 150,000 |

完整统计和可视化输出见 [`pipeline/README.md`](pipeline/README.md) 与 `output/<dataset>/`。

## 评分对应

| 评分项 | 分数 | 实现 |
|--------|------|------|
| VGGT 求相机参数和初步点云 | 3分 | `vggt_inference.py` |
| 编程实现 Bundle Adjustment | 4分 | `bundle_adjustment.py` |
| 3D 高斯滤波优化 + 实时渲染 | 4分 | `gaussian_splatting.py` + `gradio_viewer.py` |
| VGGT 改进方法调研 | 3分 | `vggt_improvements.md` |
| PPT + 答辩 | 6分 | `inspect_results.py` 输出对比图表 |

> 详细文档见 [`pipeline/README.md`](pipeline/README.md)

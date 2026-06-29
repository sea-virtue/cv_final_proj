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
python pipeline/gsplat_training.py --dataset 数据1-人体       # 推荐：基于 gsplat 官方策略的 3DGS训练

# 视频数据集会自动抽帧，输出到 output/数据3-场景/
python pipeline/vggt_inference.py   --dataset 数据3-场景.mp4
# 可选：抽全部帧，或每 N 帧抽一帧；VGGT 不建议一次输入几百帧
python pipeline/vggt_inference.py   --dataset 数据3-场景.mp4 --video-all-frames
python pipeline/vggt_inference.py   --dataset 数据3-场景.mp4 --video-frame-stride 10

# 场景数据第3步建议保留原始背景；默认 auto 也会自动这样处理
python pipeline/gsplat_training.py --dataset 数据3-场景 --steps 15000 --max-points 250000 --bg-mode original

# 3. 查看结果
python pipeline/inspect_results.py  --dataset 数据1-人体     # 文本摘要+可视化PNG
python pipeline/gradio_viewer.py                         # Web端3D交互查看，可切换 Step 1/2/3，并从当前视角渲染3DGS RGB图
python pipeline/gsplat_viewer.py --dataset 数据1-人体 --port 8080  # 独立3DGS实时RGB交互渲染
python pipeline/gsplat_viewer.py --dataset 数据3-场景 --port 8080 --view-mode inside  # 场景数据从内部训练相机视角查看
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `pipeline/README.md` | 完整使用文档（环境配置、参数说明、FAQ） |
| `pipeline/vggt_inference.py` | VGGT 推理，输出相机位姿和点云 |
| `pipeline/bundle_adjustment.py` | BA 联合优化相机外参和3D点 |
| `pipeline/gaussian_splatting.py` | 3DGS 训练，输出 .ply 高斯球 |
| `pipeline/gsplat_training.py` | 推荐 3DGS 训练入口，使用 gsplat 的 DefaultStrategy 和 exporter |
| `pipeline/gradio_viewer.py` | 🌐 Gradio Web 查看器（推荐查看方式，可在菜单中比较 Step 1/2/3） |
| `pipeline/gsplat_viewer.py` | 独立 gsplat 官方 viewer 启动器，直接查看 `gaussians.ply` |
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

### Bundle Adjustment

| 数据集 | 帧数 | VGGT重投影误差 | BA后误差 | 改善 | 地标点数 | 耗时 |
|--------|------|---------------|----------|------|----------|------|
| 数据1-人体 | 16 | 15.33 px | 13.20 px | **13.9%** | 413 | 3.8s |
| 数据2-人体 | 16 | 19.05 px | 14.41 px | **24.4%** | 382 | 3.2s |
| 数据3-场景 | 16 | 80.92 px | 30.97 px | **61.7%** | 1272 | 3.1s |

> 场景数据集 VGGT 初始位姿差、重投影误差高，BA 改善最显著（61.7%）；人体数据集 VGGT 初始精度较好，BA 仍有 14-24% 改善。

### 3DGS 训练

| 数据集 | 位姿来源 | Best PSNR | Gaussians | 训练耗时 |
|--------|---------|-----------|-----------|----------|
| 数据1-人体 | BA | 35.14 dB | 149,785 | 57.8s |
| 数据1-人体 | VGGT | 38.17 dB | 166,368 | 55.7s |
| 数据2-人体 | BA | 29.96 dB | 95,438 | 50.7s |
| 数据2-人体 | VGGT | 38.38 dB | 178,974 | 58.6s |
| 数据3-场景 | BA | 33.64 dB | 363,270 | 50.0s |
| 数据3-场景 | VGGT | 30.08 dB | 476,256 | 60.4s |

### 关键发现

1. **BA 对场景数据效果最好**：场景纹理丰富（SIFT 检测到 9660 关键点，1272 地标），VGGT 初始位姿误差大（80.92 px），BA 优化后降到 30.97 px（-61.7%），3DGS PSNR 从 30.08 提升到 **33.64 dB（+3.56）**
2. **人体数据 BA 对 PSNR 影响不显著**：VGGT 初始位姿已较准（15-19 px），BA 微调对训练视角 PSNR 帮助有限。但 BA 减少了 Gaussians 数量（149k vs 166k），表示几何更一致
3. **BA 减少 Gaussians**：三个数据集中，BA 位姿训练的 Gaussians 都更少，说明更准确的相机位姿有助于 3DGS 用更少的球拟合场景

## 评分对应

| 评分项 | 分数 | 实现 |
|--------|------|------|
| VGGT 求相机参数和初步点云 | 3分 | `vggt_inference.py` |
| 编程实现 Bundle Adjustment | 4分 | `bundle_adjustment.py` |
| 3D 高斯滤波优化 + 实时渲染 | 4分 | `gsplat_training.py` + `gsplat_viewer.py` + `gradio_viewer.py` |
| VGGT 改进方法调研 | 3分 | `vggt_improvements.md` |
| PPT + 答辩 | 6分 | `inspect_results.py` 输出对比图表 |

> 详细文档见 [`pipeline/README.md`](pipeline/README.md)

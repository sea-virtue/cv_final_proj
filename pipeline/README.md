# 三维重建与高斯绘制 — 课程项目

> **课程要求**：多视角图像 → VGGT 求相机参数和初步点云 → Bundle Adjustment 优化 → 3D Gaussian Splatting 优化与渲染

---

## 项目结构

```
.
├── pipeline/                         # 本项目核心代码
│   ├── utils.py                      # 工具函数（SE3数学、数据IO）
│   ├── vggt_inference.py             # VGGT推理 → 相机位姿 + 深度图 + 点云
│   ├── bundle_adjustment.py          # Bundle Adjustment 优化相机外参和3D点
│   ├── gaussian_splatting.py         # 3D Gaussian Splatting 训练
│   ├── gradio_viewer.py              # 推荐 Web 查看器：菜单切换 Step 1/2/3
│   ├── render_gaussian_image.py      # 指定相机视角渲染 3DGS PNG
│   ├── gs_viewer.py                  # 旧版 viser 查看器（当前不作为主入口）
│   ├── vggt_improvements.md          # VGGT改进方法调研文档
│   └── README.md                     # 本文件
├── output/                           # 所有输出文件
│   ├── 数据1-人体/
│   │   ├── predictions.npz           # VGGT原始输出
│   │   ├── ba_result.npz             # BA优化结果
│   │   ├── gaussians.ply             # 3DGS训练好的高斯球
│   │   └── gs_train_loss.png         # 训练损失曲线
│   ├── 数据2-人体/
│   │   └── ...（同上结构）
│   └── 数据3-场景/
│       └── ...（同上结构）
├── vggt/                             # VGGT官方代码（已有）
├── 大作业数据/                        # 数据集
│   ├── 数据1-人体/     (16张rgb + 16张mask)
│   ├── 数据2-人体/     (16张rgb + 16张mask)
│   └── 数据3-场景.mp4  (视频)
└── 大作业.pdf / 大作业.pptx           # 作业说明
```

---

## 环境配置

### 1. 基础环境（VGGT 依赖）

```bash
pip install -r vggt/requirements_demo.txt
pip install torch torchvision  # 根据CUDA版本安装
```

### 2. 额外依赖（本项目新增）

```bash
pip install -r pipeline/requirements_extra.txt
```

> **注意**：`gsplat` 训练需要可用的 NVIDIA 驱动和 CUDA 版 PyTorch；不需要 `nvcc` 才能运行已安装好的 CUDA wheel。
> 如果 `opencv-python` 的 SIFT 不可用，请安装 `opencv-contrib-python`：
> ```bash
> pip uninstall opencv-python
> pip install opencv-contrib-python
> ```

### 3. 验证

```bash
python -c "import torch; print(torch.cuda.is_available())"  # 应为 True
python -c "from vggt.models.vggt import VGGT; print('VGGT OK')"
```

---

## 查看结果

每一步运行后，可以用检查脚本查看结果摘要和可视化：

```bash
python pipeline/inspect_results.py --dataset 数据1-人体 --step 1   # VGGT推理结果
python pipeline/inspect_results.py --dataset 数据1-人体 --step 2   # BA优化结果
python pipeline/inspect_results.py --dataset 数据1-人体 --step 3   # 3DGS训练结果
python pipeline/inspect_results.py --dataset 数据1-人体 --step all # 全部
```

每个步骤会输出：
- **终端文本摘要**：帧数、点数、误差、位姿变化等关键指标
- **PNG 可视化图**：深度图、置信度热力图、BA前后轨迹对比、Gaussians空间分布等

---

## 运行流程

### 第1步：VGGT 推理（相机参数和初步点云）

对每个数据集运行：

```bash
python pipeline/vggt_inference.py --dataset 数据1-人体
python pipeline/vggt_inference.py --dataset 数据2-人体
python pipeline/vggt_inference.py --dataset 数据3-场景.mp4
```

视频数据集会自动均匀抽帧到 `output/数据3-场景/frames/`。`--dataset 数据3-场景` 也可以作为 `数据3-场景.mp4` 的别名使用。如果已有 `predictions.npz`，可跳过此步。

视频抽帧参数：
- 默认：`--video-max-frames 16`，从整段视频均匀抽 16 帧
- 抽全部帧：`--video-all-frames`
- 每 N 帧抽一帧：`--video-frame-stride N`

VGGT 会把所有帧作为一个序列共同推理，显存消耗会随帧数快速增加。`--video-all-frames` 适合先抽帧留档，但不建议直接把几百帧一次送入模型。脚本默认在超过 128 帧时提前停止；实际建议先用 16、32、64 帧，4090 24GB 可再尝试 96 或 128。

示例：

```bash
python pipeline/vggt_inference.py --dataset 数据3-场景.mp4 --video-max-frames 32
python pipeline/vggt_inference.py --dataset 数据3-场景.mp4 --video-frame-stride 10
python pipeline/vggt_inference.py --dataset 数据3-场景.mp4 --video-all-frames
```

**输出**：`output/<dataset>/predictions.npz`
- `extrinsic`: 相机外参 (S, 3, 4)
- `intrinsic`: 相机内参 (S, 3, 3)
- `depth`: 深度图 (S, H, W, 1)
- `depth_conf`: 深度置信度 (S, H, W)
- `world_points`: 直接预测的世界坐标点云 (S, H, W, 3)
- `world_points_from_depth`: 从深度反投影的世界坐标 (S, H, W, 3)
- `images`: 原始图像 (S, H, W, 3)

**预计时间**：每数据集 10-30 秒（取决于图像数量）

---

### 第2步：Bundle Adjustment（编程实现BA优化）

```bash
python pipeline/bundle_adjustment.py --dataset 数据1-人体
python pipeline/bundle_adjustment.py --dataset 数据2-人体
python pipeline/bundle_adjustment.py --dataset 数据3-场景.mp4
```

**BA 实现细节**：
1. 使用 SIFT 提取每帧约 2000 个关键点
2. 在相邻帧之间进行特征匹配 (BFMatcher + crossCheck)
3. 利用 VGGT 初始相机位姿进行三角化，得到 3D 地标点
4. 使用 PyTorch autograd 联合优化相机外参（SE3 参数化）和 3D 地标
5. 损失函数：Huber Loss on 重投影误差
6. 优化策略：Adam (500步) → L-BFGS (20步) 两级优化

**输出**：`output/<dataset>/ba_result.npz`
- `extrinsic_opt`: 优化后外参 (S, 3, 4)
- `points3d_opt`: 优化后3D地标 (M, 3)
- `reproj_before` / `reproj_after`: 优化前后平均重投影误差（像素）

**预计时间**：每数据集 1-3 分钟

**预期结果**：重投影误差下降 30-60%

---

### 第3步：3D Gaussian Splatting 训练（3DGS优化与渲染）

```bash
python pipeline/gaussian_splatting.py --dataset 数据1-人体 --steps 7000
python pipeline/gaussian_splatting.py --dataset 数据2-人体 --steps 7000
python pipeline/gaussian_splatting.py --dataset 数据3-场景.mp4 --steps 7000
```

**3DGS 实现细节**：
1. 从 BA 优化后的点云（或 VGGT 原始点云）初始化 Gaussians
   - 位置：点云坐标
   - 颜色：对应图像像素 RGB
   - 不透明度：基于置信度映射
   - 缩放：基于 KNN 最近邻距离估计
   - 旋转：方向对齐到相机视线
2. 使用 `gsplat` 库进行可微光栅化训练
3. 损失：0.8 × L1 + 0.2 × SSIM
4. 优化器：Adam (lr=1e-2)，按参数类型分组学习率
5. 周期性剪枝：移除不透明度 < 0.01 的 Gaussians
6. 梯度裁剪：max_norm=1.0

**可选参数**：
- `--steps`: 训练步数（默认 7000）
- `--max-points`: 最大初始 Gaussians 数量（默认 150k）
- `--no-ba`: 使用 VGGT 原始位姿而非 BA 优化后的

**输出**：
- `output/<dataset>/gaussians.ply`：训练好的 3D Gaussians
- `output/<dataset>/gs_train_loss.png`：训练损失曲线

**预计时间**：每数据集 10-30 分钟（取决于点数）

---

### 第4步：Web 端三阶段对比渲染

```bash
python pipeline/gradio_viewer.py
```

打开浏览器访问 `http://localhost:7860`。页面左侧可以选择数据集，并在菜单中切换三个阶段，方便横向比较：

- **Step 1: VGGT**：VGGT 原始点云 + VGGT 相机
- **Step 2: VGGT + BA**：VGGT 点云 + BA 优化后的相机
- **Step 3: 3DGS**：训练后的 Gaussian 点集；可切换中心点显示或采样椭球预览

这里的“实时交互渲染”指结果加载后可以在 Web 页面中实时旋转、缩放、平移、切换数据集和阶段，用于展示重建结果。Web 页面提供两个输出按钮：

- **保存当前视图 PNG**：保存浏览器当前 3D 预览画面，适合展示交互查看结果。
- **渲染3DGS RGB图**：使用 `gsplat` 从训练好的 `gaussians.ply` 按当前“显示帧”相机视角渲染 RGB 图，并保存到 `output/<dataset>/renders/`。

完整 Gaussian 数量通常很大，Web 端默认显示 Gaussian 中心点；如需展示高斯的尺度和旋转，可选择 “Centers + Ellipsoid Preview”，查看抽样椭球。

如需用脚本按数据相机视角离线渲染 PNG，可使用备用脚本：

```bash
python pipeline/render_gaussian_image.py --dataset 数据1-人体 --frame 0
python pipeline/render_gaussian_image.py --dataset 数据1-人体 --frame 0 --yaw-deg 20
```

输出位于 `output/<dataset>/renders/`。这个脚本使用训练得到的 `gaussians.ply`、相机内参和外参，通过 `gsplat` 渲染成 PNG，更接近传统意义上的 Gaussian splatting 图像渲染；`gradio_viewer.py` 则用于交互式观察三维结构和三阶段对比。

**可选参数**：
- `--port`: 端口号（默认 7860）
- `--share`: 生成 Gradio 临时公网链接

`pipeline/gs_viewer.py` 是旧版 viser 查看器，当前结果查看统一使用 `pipeline/gradio_viewer.py`。

---

## 实验结果对比（用于PPT）

运行所有步骤后，你可以获得以下对比数据：

也可以一键重新生成分析表格和图：

```bash
python pipeline/analyze_experiments.py
```

输出位于 `output/analysis/`，包括：
- `experiment_analysis.md`：可直接整理进 PPT 的文字结论
- `ba_effect_on_3dgs.csv/.png`：BA 对 3DGS 几何输入的影响分析
- `mask_filter_analysis.csv/.png`：VGGT 改进方法实验分析

### BA 效果分析
| 数据集 | BA前重投影误差 | BA后重投影误差 | 改善 |
|--------|---------------|---------------|------|
| 数据1-人体 | 15.3296 px | 13.2044 px | 13.9% |
| 数据2-人体 | 19.0536 px | 14.4084 px | 24.4% |
| 数据3-场景 | 83.5282 px | 17.7731 px | 78.7% |

### VGGT 与 3DGS 输出规模
| 数据集 | 输入帧数 | VGGT有效点数 | BA地标点数 | Gaussians数量 | 训练曲线 |
|--------|----------|--------------|------------|---------------|----------|
| 数据1-人体 | 16 | 4,293,184 | 413 | 150,000 | `output/数据1-人体/gs_train_loss.png` |
| 数据2-人体 | 16 | 4,293,184 | 382 | 150,000 | `output/数据2-人体/gs_train_loss.png` |
| 数据3-场景 | 64 | 9,746,688 | 6,727 | 150,000 | `output/数据3-场景/gs_train_loss.png` |

### VGGT深度统计
| 数据集 | 图像尺寸 | 深度范围 | 平均深度 |
|--------|----------|----------|----------|
| 数据1-人体 | 518×518 | [0.369, 1.526] | 0.840 |
| 数据2-人体 | 518×518 | [0.364, 1.573] | 0.815 |
| 数据3-场景 | 294×518 | [0.060, 1.365] | 0.513 |

> 当前 `inspect_results.py` 输出中没有记录 Best PSNR，因此这里使用 BA 重投影误差、点云规模、Gaussian 数量和训练损失曲线作为结果汇总。具体可视化文件包括 `pred_vis.png`、`ba_comparison.png`、`gs_distribution.png` 和 `gs_train_loss.png`。

### BA 是否对 3DGS 有帮助

3DGS 的训练依赖多视角相机外参。如果相机位姿不一致，同一个空间区域在不同视角下的监督会互相冲突，容易造成漂浮点、重影或训练收敛变慢。因此本项目用 BA 后的相机外参作为 3DGS 的训练相机。

可从两方面说明 BA 的作用：

1. **几何一致性提升**：BA 将重投影误差从 `15.3296→13.2044 px`、`19.0536→14.4084 px`、`83.5282→17.7731 px`，说明多视图观测和相机位姿更加一致。
2. **3DGS 输入更稳定**：3DGS 使用 BA 后相机进行训练，减少多视角监督之间的几何冲突。人体数据中 Gaussian 投影覆盖率变化不大，说明 BA 是小幅精修；场景视频中 BA 改善最大，对长序列更明显。

PPT 中建议展示：
- `output/analysis/ba_effect_on_3dgs.png`
- 每个数据集的 `output/<dataset>/ba_comparison.png`
- Gradio 中 Step 1 / Step 2 / Step 3 的对比

### 改进方法实验分析

本项目实际实现的 VGGT 改进是 **Mask 引导的置信度过滤**。人体数据集提供 `msk_*.png`，脚本将背景区域的 `depth_conf` 置零，使后续点云与 3DGS 初始化更关注人体前景。

实验结果：

| 数据集 | 前景像素占比 | 背景像素占比 | 背景零置信度比例 | 前景平均置信度 |
|--------|-------------|-------------|------------------|----------------|
| 数据1-人体 | 8.2% | 91.8% | 100.0% | 45.06 |
| 数据2-人体 | 6.8% | 93.2% | 100.0% | 30.30 |

结论：Mask 过滤以很低成本去除了人体数据中 90% 以上的背景置信度贡献，减少背景噪声对点云展示和高斯初始化的干扰。该方法不需要重新训练 VGGT，属于可解释的后处理改进。

PPT 中建议展示：
- `output/analysis/mask_filter_analysis.png`
- 数据1/数据2 的 `pred_vis.png`
- Step 3 中关闭/开启前景过滤思想的解释图

---

## 评分点对应

| 评分项 | 分数 | 对应文件 |
|--------|------|----------|
| 使用 VGGT 求相机参数和初步点云 | 3分 | `vggt_inference.py` |
| 编程实现 Bundle Adjustment | 4分 | `bundle_adjustment.py` |
| 3D 高斯滤波优化 + 实时交互渲染 | 4分 | `gaussian_splatting.py` + `gradio_viewer.py` + `render_gaussian_image.py` |
| VGGT 改进方法调研 | 3分 | `vggt_improvements.md` |
| PPT 制作与答辩 | 6分 | — |

---

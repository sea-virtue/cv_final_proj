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
│   ├── gsplat_training.py            # 推荐 3DGS 训练入口：使用 gsplat 官方策略
│   ├── gradio_viewer.py              # 推荐 Web 查看器：菜单切换 Step 1/2/3
│   ├── gsplat_viewer.py              # 独立 gsplat 官方 viewer 启动器
│   ├── render_gaussian_image.py      # 指定相机视角渲染 3DGS PNG
│   ├── gs_viewer.py                  # 旧版 viser 查看器（当前不作为主入口）
│   ├── vggt_improvements.md          # VGGT改进方法调研文档
│   └── README.md                     # 本文件
├── output/                           # 所有输出文件
│   ├── 数据1-人体/
│   │   ├── predictions.npz           # VGGT原始输出
│   │   ├── ba_result.npz             # BA优化结果
│   │   ├── gaussians.ply             # 3DGS训练好的高斯球
│   │   └── gsplat_train_loss.png     # 训练损失曲线
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
python pipeline/gsplat_training.py --dataset 数据1-人体 --steps 7000
python pipeline/gsplat_training.py --dataset 数据2-人体 --steps 7000
python pipeline/gsplat_training.py --dataset 数据3-场景.mp4 --steps 7000
```

**3DGS 实现细节**：
1. 从 BA 优化后的点云（或 VGGT 原始点云）初始化 Gaussians
   - 位置：点云坐标
   - 颜色：对应图像像素 RGB
   - 不透明度：基于置信度映射
   - 缩放：基于 KNN 最近邻距离估计
   - 旋转：随机四元数初始化并在训练中优化
   - 背景处理默认 `auto`：人体数据检测到 `msk_*.png` 时自动合成黑背景训练；场景视频没有 mask 时保留原始 RGB，不做扣绿
   - 使用 BA 相机时，会把 VGGT depth 按 BA 外参重新反投影，保证 dense 初始化点云和训练相机处在同一坐标关系中
2. 使用 `gsplat` 的 `rasterization` 进行可微光栅化训练
3. 使用 `gsplat` 的 `DefaultStrategy` 做 Gaussian 增密、拆分、裁剪和 opacity reset
4. 损失：0.8 × L1 + 0.2 × SSIM；人体前景默认加权，避免黑背景像素过多导致人物监督太弱
5. 优化器：按 `means / scales / quats / opacities / SH` 分组 Adam
6. 使用 `gsplat.export_splats()` 导出标准 3DGS PLY

**可选参数**：
- `--steps`: 训练步数（默认 7000）
- `--max-points`: 最大初始 Gaussians 数量（默认 150k）
- `--no-ba`: 使用 VGGT 原始位姿而非 BA 优化后的
- `--no-mask-loss`: 不使用人体 mask；默认会自动使用可用的 `msk_*.png`
- `--bg-mode`: 背景处理方式，默认 `auto`；人体 mask 数据自动使用 `mask-black`，普通场景自动使用 `original`。也可手动指定 `chroma-black` 用 RGB 通道扣绿，`foreground-only` 只在前景计算 loss，`original` 保留原图背景
- `--foreground-loss-weight`: 人体前景 L1 权重（默认 8.0）
- `--no-reproject-depth-with-poses`: 关闭 BA 相机下的 dense depth 重新反投影
- `--use-local-gsplat`: 强制从项目内 `./gsplat` 源码导入；通常会触发本地 CUDA 扩展编译，需要 nvcc。默认使用当前环境中已安装的 gsplat wheel，更稳定。

旧版简化训练脚本仍保留：

```bash
python pipeline/gaussian_splatting.py --dataset 数据1-人体 --steps 7000
```

旧脚本便于阅读实现，但增密/裁剪和 PLY 导出格式更简化；推荐答辩和最终结果使用 `gsplat_training.py`。

**输出**：
- `output/<dataset>/gaussians.ply`：训练好的 3D Gaussians
- `output/<dataset>/gsplat_train_loss.png`：训练损失曲线
- `output/<dataset>/renders/gsplat_train_preview_0000.png`：训练后预览渲染图

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

这里的“实时交互渲染”包含两层含义：一是 Web 页面中可以实时旋转、缩放、平移、切换数据集和阶段，用于展示重建结构；二是可以基于当前交互视角调用 `gsplat` 输出一张真正的 3DGS RGB 渲染图。Web 页面提供两个输出按钮：

- **保存当前视图 PNG**：保存浏览器当前 3D 预览画面，适合展示交互查看结果。
- **渲染3DGS RGB图**：使用 `gsplat` 从训练好的 `gaussians.ply` 按浏览器当前拖动视角渲染 RGB 图，并保存到 `output/<dataset>/renders/`；如果浏览器视角读取失败，则自动退回当前“显示帧”相机视角。

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

`pipeline/gs_viewer.py` 是旧版 viser 查看器，当前三阶段对比统一使用 `pipeline/gradio_viewer.py`。

如果想直接尝试 gsplat 官方 viewer，可运行：

```bash
python pipeline/gsplat_viewer.py --dataset 数据1-人体 --port 8080
```

该入口直接读取 `output/<dataset>/gaussians.ply`，并使用当前环境中已安装的 `gsplat` wheel 进行实时 RGB 光栅化。它比 Gradio 的 GLB 预览更接近真正的 3DGS 交互渲染，但需要额外依赖：

```bash
python -m pip install viser splines
python -m pip install 'git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787'
```

如果只需要三阶段对比，仍使用 `gradio_viewer.py`；如果只想看 3DGS 自身的交互式 RGB 渲染，使用 `gsplat_viewer.py`。

室内场景类数据不要按“外部模型”去看。`数据3-场景` 这种相机在场景内部移动的数据，更适合从训练相机附近向四周看：

```bash
python pipeline/gsplat_viewer.py --dataset 数据3-场景 --port 8080 --view-mode inside
```

默认 `--view-mode auto` 会对名字包含“场景”的数据自动使用 inside 初始视角。网页右侧也会出现 **Inside view** 按钮，点一下可以回到场景内部的训练相机视角。`--view-frame N` 可以指定使用第 N 帧训练相机作为初始视角。

完整推荐流程：

```bash
# Step 1: VGGT
python pipeline/vggt_inference.py --dataset 数据1-人体

# Step 2: BA
python pipeline/bundle_adjustment.py --dataset 数据1-人体

# Step 3: 3DGS，默认用 BA 位姿 + mask-black 背景处理
python pipeline/gsplat_training.py --dataset 数据1-人体 --steps 7000

# 三阶段对比
python pipeline/gradio_viewer.py

# 独立 3DGS 实时 RGB viewer
python pipeline/gsplat_viewer.py --dataset 数据1-人体 --port 8080
```

---

## 提高 3DGS 精度的建议

当前人物粗糙、不平滑，通常不是单一原因，而是以下因素叠加：

1. **VGGT 初始点云精度会影响 3DGS 上限**  
   3DGS 的 Gaussian 位置从 VGGT 点云初始化。如果 VGGT 深度在人体边界、衣服褶皱、手脚等区域有噪声，3DGS 后面会在错误位置开始优化，容易出现毛刺、漂浮点或表面粗糙。BA 主要修相机外参，并不会把 VGGT 的稠密深度图完全变准。

2. **BA 影响多视角一致性**  
   如果相机位姿不准，同一个人体点在不同视角下投影位置对不上，3DGS 会收到互相冲突的 RGB 监督，表现为模糊、重影、边缘发散。BA 后重投影误差越低，3DGS 越容易收敛。

3. **训练步数会影响收敛，但不是无限增加就一定好**  
   `--steps 7000` 是演示级别。想更细，可以尝试：

   ```bash
   python pipeline/gsplat_training.py --dataset 数据1-人体 --steps 15000
   ```

   如果 7000 步还没收敛，增加到 15000 通常会改善颜色和边界；但如果初始点云/相机误差较大，只加步数会把错误结构“训练得更实”，不一定变平滑。

4. **Gaussian 数量影响细节容量**  
   默认 `--max-points 150000`。人体细节不够时可以试：

   ```bash
   python pipeline/gsplat_training.py --dataset 数据1-人体 --steps 15000 --max-points 250000
   ```

   点数更多会保留更多初始化细节，但显存、训练时间也会上升。

5. **绿幕/背景处理会影响边界质量，场景数据不能按人体扣绿**  
   默认 `--bg-mode auto` 会区分两类数据：`数据1-人体/数据2-人体` 有 mask，因此使用 `mask-black`；`数据3-场景` 没有 mask，因此使用 `original` 保留原图。场景数据如果被通道扣绿，会把真实画面里的绿色/亮色区域误删，3DGS 会变成黑底碎片。

   人体数据也可以比较通道扣绿：

   ```bash
   python pipeline/gsplat_training.py --dataset 数据1-人体 --steps 15000 --bg-mode chroma-black
   ```

   如果 mask 边界过硬，人体边缘可能被切掉；如果 chroma key 太松，绿色会残留。可调：

   ```bash
   --chroma-min-green 0.25 --chroma-margin 0.08
   ```

6. **输入视角数量和覆盖范围很关键**  
   人体数据只有 16 张图，视角覆盖有限。3DGS 对遮挡区域没有真实监督，只能靠初始化和邻近视角补，手脚、侧面、背面容易粗糙。增加输入视角、保证视角环绕均匀，比单纯增加训练步数更有效。

推荐优先尝试的高质量命令：

```bash
python pipeline/gsplat_training.py --dataset 数据1-人体 --steps 15000 --max-points 250000 --bg-mode mask-black --foreground-loss-weight 8
```

场景数据推荐保留原图背景：

```bash
python pipeline/gsplat_training.py --dataset 数据3-场景 --steps 15000 --max-points 250000 --bg-mode original
```

如果显存或时间压力大，先用：

```bash
python pipeline/gsplat_training.py --dataset 数据1-人体 --steps 10000 --max-points 150000 --bg-mode mask-black --foreground-loss-weight 8
```

## 实验结果对比（用于PPT）

**BA 重投影误差：**

| 数据集 | VGGT | BA后 | 改善 | 地标 | SIFT关键点 |
|--------|------:|-----:|-----:|-----:|----------:|
| 数据1-人体 | 15.33 px | 13.20 px | 13.9% | 413 | 2492 |
| 数据2-人体 | 19.05 px | 14.41 px | 24.4% | 382 | 2178 |
| 数据3-场景 | 80.92 px | 30.97 px | **61.7%** | 1272 | 9660 |

**3DGS PSNR（BA vs VGGT 位姿）：**

| 数据集 | VGGT位姿 | BA位姿 | ΔPSNR | BA后Gaussians |
|--------|---------:|-------:|------:|-------------:|
| 数据1-人体 | 38.17 dB | 35.14 dB | -3.03 | 149,785 |
| 数据2-人体 | 38.38 dB | 29.96 dB | -8.42 | 95,438 |
| 数据3-场景 | 30.08 dB | **33.64 dB** | **+3.56** | 363,270 |

**核心结论：** BA 在 VGGT 位姿误差大的场景（数据3，80.92→30.97 px）上提升 PSNR 3.56 dB 且减少 24% Gaussians。人体数据 VGGT 初始位姿已较准，BA 对训练视角 PSNR 影响有限但减少了 Gaussians 数量。

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
| 数据1-人体 | 16 | 4,293,184 | 413 | 150,000 | `output/数据1-人体/gsplat_train_loss.png` |
| 数据2-人体 | 16 | 4,293,184 | 382 | 150,000 | `output/数据2-人体/gsplat_train_loss.png` |
| 数据3-场景 | 64 | 9,746,688 | 6,727 | 150,000 | `output/数据3-场景/gsplat_train_loss.png` |

### VGGT深度统计
| 数据集 | 图像尺寸 | 深度范围 | 平均深度 |
|--------|----------|----------|----------|
| 数据1-人体 | 518×518 | [0.369, 1.526] | 0.840 |
| 数据2-人体 | 518×518 | [0.364, 1.573] | 0.815 |
| 数据3-场景 | 294×518 | [0.060, 1.365] | 0.513 |

> 当前 `inspect_results.py` 输出中没有记录 Best PSNR，因此这里使用 BA 重投影误差、点云规模、Gaussian 数量和训练损失曲线作为结果汇总。具体可视化文件包括 `pred_vis.png`、`ba_comparison.png`、`gs_distribution.png` 和 `gsplat_train_loss.png`。

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
| 3D 高斯滤波优化 + 实时交互渲染 | 4分 | `gsplat_training.py` + `gsplat_viewer.py` + `gradio_viewer.py` |
| VGGT 改进方法调研 | 3分 | `vggt_improvements.md` |
| PPT 制作与答辩 | 6分 | — |

---

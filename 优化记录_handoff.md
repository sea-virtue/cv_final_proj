# 3DGS 渲染质量优化 — 调试记录与环境配置（Handoff）

> 目的：把本次诊断、对 `pipeline/gsplat_training.py` 的代码改动、环境问题、以及在 **WSL/Linux** 上重训验证的完整步骤记录下来，方便换环境后直接继续。
> 适用数据：`数据3-场景`（相机在场景内部环视的视频），现象是渲染有大量空间黑点 + 整体模糊。

---

## 1. 诊断结论（为什么效果差）

### 1.1 训练本身在退化（不是单纯质量问题）
- `output/数据3-场景/gsplat_train_loss.png`：平滑后 loss 在 **第 ~2000 步降到最低（~0.18），之后一路爬升回 ~0.22** 直到 15000 步。健康训练应单调下降或持平。
- `output/数据3-场景/gaussians.ply` 头部：**1,103,102 个高斯**，从 15 万初始点暴涨到 110 万。场景约束弱（环视、无 mask），多出来的 ~95 万高斯大量是悬浮 floater —— 就是空间中的黑点。
- 预览图 `renders/gsplat_train_preview_0000.png` 糊成一片灰，无结构。

### 1.2 黑点为什么是“黑”的（4 个叠加原因）
1. **没有深度监督**：原训练只用 RGB（L1+SSIM），depth 只在初始化反投影时用一次。错误深度上的 floater 不会被几何惩罚 → 环视场景经典 floater。**这就是用户听说的“depth 功能”。**
2. **初始化不做置信度过滤**：原 `conf_threshold=1e-6` ≈ 不过滤。VGGT 对远处墙面/弱纹理深度是噪声，垃圾点全被当种子。
3. **黑色训练背景**：`backgrounds=zeros`，低不透明度 floater 叠黑底被染黑。
4. **增密太凶且不早停**：`refine_stop=15000`=训练全程都在增密 → 110 万高斯，大量虚假；大高斯导致模糊。

### 1.3 硬件不是瓶颈
4060 笔记本 8GB 跑 294×518、百万级高斯没问题。模糊/黑点是算法层面的。

---

## 2. 已做的代码改动（全部在 `pipeline/gsplat_training.py`）

所有改进**默认开启，可单独关闭**。核心是深度监督。

| 改进 | 作用 | 关闭方式 | 默认值 |
|---|---|---|---|
| **深度监督** | `render_mode="RGB+ED"` 渲染期望深度，与 VGGT 深度做 L1、按 `depth_conf` 加权。把高斯钉在真实表面，消灭黑点 | `--depth-loss-weight 0` | `0.5` |
| 深度监督置信度阈值 | 只在高置信度像素上监督深度 | `--depth-conf-percentile` | `30` |
| **初始化置信度过滤** | 丢掉低置信度噪声种子点 | `--init-conf-percentile 0` | `30` |
| **随机背景**（仅 mask 模式） | 逼高斯学对 opacity，杀半透明黑 floater | `--no-random-bg` | 开 |
| **opacity 正则** | L1 压制低不透明度 floater | `--opacity-reg 0` | `0.001` |
| **scale 正则** | 压制糊成片的大高斯（相对 scene_scale） | `--scale-reg 0` | `0.001` |
| **增密提前停** | 默认在训练一半步数停止增密，后半段只精修，修复 loss 后半段上升 | `--refine-stop <step>` 手动指定 | `steps*0.5` |

实现要点（便于 review / 继续改）：
- `load_pipeline_data` 现在额外返回 `depth`（之前只用于反投影，没返回）。
- 新增 `conf_threshold_from_percentile()`：从 `depth_conf` 按分位数算绝对阈值，初始化过滤和深度 loss mask 共用。
- 训练 loop：`renders` 现在可能是 4 通道（RGB+ED），用 `rgb = renders[..., :3]`、`depth_pred = renders[..., 3:4]`；所有 loss / PSNR 都改用 `rgb`。
- 深度一致性说明：默认 `--reproject-depth-with-poses` 开启，会把 VGGT depth 按 BA 外参重新反投影，所以初始化点正好落在 VGGT 深度处、渲染期望深度与 VGGT 深度在同一坐标系，深度监督是自洽的。

> 注意：代码已用 `conda run -n cv python -m py_compile` 通过语法检查，但**尚未在 GPU 上实跑验证**（被下面的环境问题卡住）。

---

## 3. 环境问题（Windows 上跑不起来的根因）

1. **`conda cv` 是 CPU 版 torch**（`2.3.1+cpu`），`torch.cuda.is_available()=False`，用不了 4060。用户一直以为在用 cv，但 cv 不能 GPU 训练。
2. **`conda minist` 有 `torch 2.6.0+cu124`，CUDA 可用** —— 是当前唯一能用 GPU 的环境。
3. **当前没有任何环境装了 `gsplat`**。`import gsplat` 会命中项目里的 `./gsplat` 源码目录（namespace 包，无 `_C`）→ 报 `cannot import name 'DefaultStrategy'`。
4. 本地 gsplat 源码首次运行要 **JIT 编译 CUDA 扩展**，需要 `CUDA_HOME` + **MSVC `cl.exe`**。已确认：
   - CUDA 12.4 工具链已装（`nvcc` 在 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`）。
   - **MSVC C++ 工作负载未安装**（`vswhere ... VC.Tools.x86.x64` 返回空，全盘找不到 `cl.exe`）→ 本地源码无法编译。
5. **gsplat 预编译 wheel 情况**（来自 `https://docs.gsplat.studio/whl/<tag>/gsplat/`）：
   - `pt26*`（torch 2.6）：**404，没有任何 wheel** → 所以 minist 的 torch 2.6 没法用预编译 wheel（ABI 也对不上）。
   - `pt24cu124`：有 wheel，**Windows 只有 `cp310`（Python 3.10）**：`gsplat-1.5.3+pt24cu124-cp310-cp310-win_amd64.whl`。
   - 推断：Jun 26 能出结果，很可能当时装过某个预编译 wheel，后来环境变了。

---

## 4. 推荐方案：在 WSL/Linux 上重训（最省事）

WSL 上 gsplat 要么有更全的预编译 wheel，要么直接源码编译（gcc+CUDA，无 MSVC 烦恼）。**已有的 `output/数据3-场景/predictions.npz` 和 `ba_result.npz` 是跨平台的 `.npz`，WSL 能直接读**，所以**不用重跑 VGGT/BA，只跑第 3 步 gsplat 训练即可**。

### 4.1 前提
- Windows 已装新版 NVIDIA 驱动 → WSL2 里自动有 CUDA（`nvidia-smi` 在 WSL 里能看到 4060）。
- 项目在 `/mnt/e/study/26春 THU/计算机视觉/cv_main_proj/大作业/大作业/`（WSL 通过 `/mnt/e/...` 访问）。

### 4.2 建环境（二选一）

**A. 用预编译 wheel（不用编译器，推荐先试）**
```bash
conda create -n gs python=3.10 -y
conda activate gs
# 先装 CUDA 版 torch（匹配 wheel 的 pt24cu124）
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
# 再装 gsplat 预编译 wheel（Linux 版在同一 index 下）
pip install gsplat==1.5.3 --no-deps --index-url https://docs.gsplat.studio/whl/pt24cu124
pip install jaxtyping rich numpy opencv-python tqdm matplotlib
# 验证
python -c "import torch,gsplat; print(torch.cuda.is_available()); from gsplat import rasterization,DefaultStrategy; print('gsplat OK')"
```
> 如果 Linux 没有 cp310 的 pt24cu124 wheel，改用方案 B。

**B. Linux 源码编译（最稳）**
```bash
conda create -n gs python=3.10 -y
conda activate gs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
sudo apt install -y build-essential   # gcc
# 需要 nvcc：装 CUDA toolkit 或 conda install -c nvidia cuda-toolkit=12.4
pip install gsplat        # 首次 import 时编译，或 pip install 时编译
pip install numpy opencv-python tqdm matplotlib
```

### 4.3 跑训练（重点）
**装的是 wheel/pip 包时，不要加 `--use-local-gsplat`**（那个会强制用项目内源码、触发编译）。默认就是用已安装的 gsplat。

```bash
cd "/mnt/e/study/26春 THU/计算机视觉/cv_main_proj/大作业/大作业"

# 先 50 步冒烟测试，确认深度监督等改动能跑（不覆盖正式结果）
python pipeline/gsplat_training.py --dataset 数据3-场景 --steps 50 --eval-every 10 \
    --output-name _smoke.ply --loss-plot _smoke_loss.png

# 正式重训（场景数据，质量优先）
python pipeline/gsplat_training.py --dataset 数据3-场景 --steps 15000 --max-points 250000
```

### 4.4 验证效果
- 看 `output/数据3-场景/gsplat_train_loss.png`：平滑 loss 后半段应**不再上升**（增密提前停的效果）。
- 看 `output/数据3-场景/renders/gsplat_train_preview_0000.png`：黑点应明显减少、结构更清晰。
- 想交互看：`python pipeline/gsplat_viewer.py --dataset 数据3-场景 --view-mode inside`。

### 4.5 消融实验（答辩用：证明每项改进有用）
```bash
# 关深度监督（对照）
python pipeline/gsplat_training.py --dataset 数据3-场景 --steps 15000 --depth-loss-weight 0 \
    --output-name gaussians_no_depth.ply --loss-plot loss_no_depth.png
# 关初始化过滤
python pipeline/gsplat_training.py --dataset 数据3-场景 --steps 15000 --init-conf-percentile 0 \
    --output-name gaussians_no_conf.ply --loss-plot loss_no_conf.png
```
对比黑点数量 / loss 曲线 / 预览图，正好对应作业要求里的“对提出的改进方法进行实验分析”。

---

## 5. Windows 备选（如果坚持在 Windows 跑）

仓库里已生成 `run_gsplat_train.bat`（在项目根目录）：自动配置 MSVC（vcvars64）+ CUDA_HOME，然后在 `minist` 环境里用 `--use-local-gsplat` 编译并训练。
**但前提是先装 MSVC**：打开 “Visual Studio Installer” → 修改 → 勾选 **“使用 C++ 的桌面开发”(Desktop development with C++)** → 装完再双击/命令行运行该 bat。
用法：
```
run_gsplat_train.bat --dataset 数据3-场景 --steps 15000 --max-points 250000
```
> 注意：从 git bash 调用该 bat 会因中文路径 UTF-8↔GBK 编码错乱失败；要在原生 cmd 里运行。

---

## 6. 关键文件清单
- `pipeline/gsplat_training.py` —— 已改（深度监督等，见第 2 节）。
- `run_gsplat_train.bat` —— Windows 编译+训练启动器（需先装 MSVC）。
- `output/数据3-场景/predictions.npz` / `ba_result.npz` —— VGGT/BA 结果，跨平台可直接复用，**不用重跑前两步**。
- `gsplat/`（项目内）—— gsplat 源码；装了 pip 包后**不要**用 `--use-local-gsplat`。

## 7. 还没做 / 下一步
- [ ] 在能跑 GPU 的环境（WSL gs 环境）里实跑冒烟 + 正式重训，确认黑点消失、loss 后半段不再上升。
- [ ] 跑消融实验，整理进 PPT。
- [ ] 若深度监督权重 0.5 太强/太弱，调 `--depth-loss-weight`（0.2~1.0 之间试）。

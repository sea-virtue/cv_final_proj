#!/bin/bash
# setup_env.sh — 激活 cv 环境并设置 CUDA 路径
# 用法: source setup_env.sh

# 激活 conda 环境
source /home/yx/miniconda3/etc/profile.d/conda.sh
conda activate cv

# 设置 CUDA runtime 库路径
CONDA_ENV="/home/yx/miniconda3/envs/cv"
export LD_LIBRARY_PATH="$CONDA_ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"

# matplotlib 缓存路径（避免只读文件系统报错）
export MPLCONFIGDIR=/tmp/matplotlib_cache
mkdir -p "$MPLCONFIGDIR"

echo "✅ Environment: $(which python)"
echo "✅ Python: $(python --version)"
python -c "import torch; print('✅ torch', torch.__version__, 'CUDA:', torch.cuda.is_available())" 2>/dev/null
python -c "import numpy; print('✅ numpy', numpy.__version__)" 2>/dev/null
python -c "import cv2; print('✅ cv2', cv2.__version__)" 2>/dev/null
python -c "import trimesh; print('✅ trimesh', trimesh.__version__)" 2>/dev/null
python -c "import gradio; print('✅ gradio', gradio.__version__)" 2>/dev/null

echo ""
echo "Now run:"
echo "  python pipeline/vggt_inference.py --dataset 数据1-人体"
echo "  python pipeline/bundle_adjustment.py --dataset 数据1-人体"
echo "  python pipeline/gaussian_splatting.py --dataset 数据1-人体"
echo "  python pipeline/gradio_viewer.py"

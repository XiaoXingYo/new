#!/bin/bash
# CRNN OCR 训练启动脚本
# 用法：bash scripts/train.sh [checkpoint_path] [config_path]

set -e  # 遇到错误立即退出

# 项目根目录自动检测
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# 创建必要目录
mkdir -p logs checkpoints data

# 设置Python路径
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# 默认配置
CONFIG_PATH="${2:-configs/base.yaml}"
RESUME_PATH="${1:-}"

echo "🚀 启动训练任务"
echo "📋 配置: $CONFIG_PATH"
if [ -n "$RESUME_PATH" ]; then
    echo "🔄 从 $RESUME_PATH 恢复训练"
fi

# 执行训练
python src/main.py \
    --mode train \
    --config "$CONFIG_PATH" \
    --resume "$RESUME_PATH"

echo "✅ 训练任务完成！"
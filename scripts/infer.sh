#!/bin/bash
# CRNN OCR 推理启动脚本
# 用法：bash scripts/infer.sh [config_path]

set -e

# 项目根目录自动检测
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# 设置Python路径
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

# 默认配置
CONFIG_PATH="${1:-configs/base.yaml}"

echo "🎯 启动推理任务"
echo "📋 使用配置: $CONFIG_PATH"

# 执行推理
python src/main.py \
    --mode infer \
    --config "$CONFIG_PATH"

echo "✅ 推理任务完成！"
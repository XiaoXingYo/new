"""
辅助工具模块
提供模型统计、随机种子设置等功能
"""

import torch
import numpy as np
from pathlib import Path
import random  # ✅ 修复：添加缺失的导入


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型可训练参数量

    Args:
        model: PyTorch模型

    Returns:
        参数量（整数）
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_summary(model: torch.nn.Module, save_path: str):
    """
    保存模型结构摘要到文件

    Args:
        model: PyTorch模型
        save_path: 保存路径（如 'logs/model_summary.txt'）
    """
    try:
        from torchsummary import summary

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # 重定向输出到文件
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()

        try:
            summary(model, (1, 32, 128), device='cpu')
            with open(save_path, 'w') as f:
                f.write(buffer.getvalue())
        finally:
            sys.stdout = old_stdout

    except ImportError:
        print("⚠️  torchsummary 未安装，跳过模型摘要保存")


def set_random_seed(seed: int = 42):
    """
    设置全局随机种子以保证结果可复现

    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 设置CUDA确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
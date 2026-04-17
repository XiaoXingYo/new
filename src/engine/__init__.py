"""
引擎模块
提供训练、评估、推理三个核心引擎
"""

from .trainer import Trainer
from .evaluator import Evaluator
from .inferencer import Inferencer

__all__ = [
    'Trainer',      # 训练引擎：负责训练循环和Checkpoint管理
    'Evaluator',    # 评估引擎：负责离线评估和指标计算
    'Inferencer'    # 推理引擎：负责实时预测和交互式演示
]
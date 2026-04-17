"""
核心模块包
提供配置、日志、Checkpoint和指标计算功能
"""

from .config import Config, DataConfig, ModelConfig, TrainConfig, InferenceConfig
from .logger import get_logger               # ← 改这里
from .checkpoint import CheckpointManager
from .metrics import calculate_cer, calculate_wer, calculate_cer_wer

__all__ = [
    'Config', 'DataConfig', 'ModelConfig', 'TrainConfig', 'InferenceConfig',
    'get_logger',
    'CheckpointManager',
    'calculate_cer', 'calculate_wer', 'calculate_cer_wer'
]
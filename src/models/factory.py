"""
模型工厂模块
统一模型构建入口：自动分发 CTC 和 Attention 架构
"""
from typing import Any
from .crnn import CRNN
from .attention_ocr import Seq2SeqAttention

def build_model(config: Any, arch: str = "ctc") -> Any:
    """
    根据配置和架构选择构建模型
    """
    if arch == "ctc":
        # CTC 模式：类别数 = 字符集长度 + 1个空白符
        num_classes = len(config.data.chars) + 1
        return CRNN(
            img_channel=1,
            num_classes=num_classes,
            hidden_size=config.model.hidden_size,
            rnn_layers=config.model.rnn_layers,
            dropout=config.model.dropout,
            backbone=config.model.backbone
        )

    elif arch == "attention":
        # Attention 模式：白嫖 CRNN 的 CNN 骨干网络
        cnn_backbone = CRNN(
            img_channel=1,
            num_classes=10, # 这里的 num_classes 不重要，只为了初始化
            backbone=config.model.backbone
        ).cnn
        # 类别数 = 字符集 + 3个护法控制符 (PAD/EOS/SOS)
        num_classes = len(config.data.chars) + 3
        return Seq2SeqAttention(
            cnn_backbone=cnn_backbone,
            num_classes=num_classes,
            hidden_size=config.model.hidden_size
        )

    else:
        raise ValueError(f"❌ 不支持的架构类型: {arch}")
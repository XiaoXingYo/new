import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.crnn import CRNN
from core.config import Config


def test_model_forward():
    """测试模型前向传播"""
    cfg = Config()
    model = CRNN(
        img_channel=1,
        num_classes=cfg.data_num_classes,
        hidden_size=128,
        backbone='vgg'
    )

    dummy_input = torch.randn(4, 1, 32, 128)
    output = model(dummy_input)

    assert output.shape[0] == 32  # 时间步
    assert output.shape[1] == 4  # batch
    assert output.shape[2] == cfg.data_num_classes  # 类别


def test_model_backbone():
    """测试不同backbone"""
    for backbone in ['vgg', 'cnn6']:
        model = CRNN(backbone=backbone, num_classes=11)
        dummy = torch.randn(1, 1, 32, 128)
        out = model(dummy)
        assert out.shape[-1] == 11


if __name__ == "__main__":
    test_model_forward()
    test_model_backbone()
    print("✅ 模型测试通过")
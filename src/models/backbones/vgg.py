import torch.nn as nn


class VGGBackbone(nn.Module):
    """VGG风格的CNN主干网络，专为OCR任务优化

    输入: (B, C, 32, 128)
    输出: (B, 512, 1, 32)
    """

    def __init__(self, img_channel: int = 1):
        super().__init__()

        # 使用VGG风格但调整pooling策略以保留宽度信息
        self.features = nn.Sequential(
            # Block 1: 32x128 -> 16x64
            nn.Conv2d(img_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16x64 -> 8x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 8x32 -> 4x32 (高度减半，宽度不变)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 特殊pooling保留宽度信息

            # Block 4: 4x32 -> 2x32
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            # Block 5: 2x32 -> 1x32
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)  # (B, 512, 1, W)


def build_vgg_backbone(img_channel: int = 1):
    """工厂函数"""
    return VGGBackbone(img_channel)
import torch
import torch.nn as nn
from typing import Literal

# 确保你能正确导入对应文件里的 build 函数
from .backbones.vgg import build_vgg_backbone
from .backbones.resnet import build_resnet18_backbone


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn: int, nHidden: int, nOut: int, dropout: float = 0.0):
        super().__init__()
        # batch_first=False，输入维度应该是 [Seq_Len, Batch, Feature]
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=False)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.rnn(x)
        T, B, H = recurrent.size()
        t_rec = recurrent.view(T * B, H)
        output = self.embedding(t_rec)
        output = self.dropout(output)
        output = output.view(T, B, -1)
        return output


class CRNN(nn.Module):
    def __init__(
            self, img_channel: int = 1, num_classes: int = 11,
            hidden_size: int = 256, rnn_layers: int = 2,
            dropout: float = 0.2, backbone: Literal["vgg", "resnet18", "cnn6"] = "vgg"
    ):
        super().__init__()
        self.backbone_type = backbone

        # ⚡️ 底层特征提取模块 (CNN)
        if backbone == "vgg":
            self.cnn = build_vgg_backbone(img_channel)
            cnn_output_channels = 512
        elif backbone == "resnet18":
            self.cnn = build_resnet18_backbone(img_channel)
            cnn_output_channels = 512
        elif backbone == "cnn6":
            self.cnn = self._build_cnn6(img_channel)
            cnn_output_channels = 256
        else:
            raise ValueError(f"不支持的 backbone: {backbone}")

        # ⚡️ 序列预测模块 (RNN)
        self.rnn = nn.Sequential()
        if rnn_layers == 1:
            self.rnn.add_module("BiLSTM", BidirectionalLSTM(cnn_output_channels, hidden_size, num_classes, dropout))
        else:
            self.rnn.add_module("BiLSTM1", BidirectionalLSTM(cnn_output_channels, hidden_size, hidden_size, dropout))
            self.rnn.add_module("BiLSTM2", BidirectionalLSTM(hidden_size, hidden_size, num_classes, dropout))

    def _build_cnn6(self, img_channel: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(img_channel, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # 修复：添加自适应池化将高度强制压为1，宽度动态推导
            nn.AdaptiveAvgPool2d((1, None))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 经过 CNN 提取特征
        conv = self.cnn(x)
        b, c, h, w = conv.size()

        # 确保高度被压缩成了 1
        assert h == 1, f"CNN 输出高度必须为1，当前为{h}"

        # 2. 维度转换: [b, c, 1, w] -> [b, c, w] -> [w, b, c]
        # 变成 Time_Step, Batch_Size, Channels 的格式，交给 LSTM
        conv = conv.squeeze(2).permute(2, 0, 1)

        # 3. 序列分类
        output = self.rnn(conv)
        return output



# 文件：src/models/core.py
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock


class OCR_ResNet18(nn.Module):
    """完美兼容旧权重的 ResNet 骨干"""

    def __init__(self, img_channel=1):
        super().__init__()
        self.inplanes = 64
        # 这里的变量名必须和原来一模一样，不能放进 Sequential 里
        self.conv1 = nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=(2, 1))
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=(2, 1))
        self.out_pool = nn.AdaptiveAvgPool2d((1, None))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out_pool(x)
        return x


class BidirectionalLSTM(nn.Module):
    """完美兼容旧权重的 LSTM 包装器"""

    def __init__(self, nIn, nHidden, nOut, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=False)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, B, H = recurrent.size()
        t_rec = recurrent.view(T * B, H)
        output = self.embedding(t_rec)
        output = self.dropout(output)
        output = output.view(T, B, -1)
        return output


class CRNN(nn.Module):
    """完美兼容旧权重的端到端架构"""

    def __init__(self, img_channel=1, num_classes=11, hidden_size=256, rnn_layers=2, dropout=0.2):
        super().__init__()
        self.cnn = OCR_ResNet18(img_channel)

        # 必须使用 add_module 并保持命名一致，才能对上锁
        self.rnn = nn.Sequential()
        if rnn_layers == 1:
            self.rnn.add_module("BiLSTM", BidirectionalLSTM(512, hidden_size, num_classes, dropout))
        else:
            self.rnn.add_module("BiLSTM1", BidirectionalLSTM(512, hidden_size, hidden_size, dropout))
            self.rnn.add_module("BiLSTM2", BidirectionalLSTM(hidden_size, hidden_size, num_classes, dropout))

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2).permute(2, 0, 1)
        output = self.rnn(conv)
        return output
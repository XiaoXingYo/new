import torch.nn as nn
from torchvision.models.resnet import BasicBlock


class OCR_ResNet18(nn.Module):
    """专门为 OCR 魔改的 ResNet18 (保留宽度，压缩高度)"""

    def __init__(self, img_channel=1):
        super(OCR_ResNet18, self).__init__()
        self.inplanes = 64

        # 接收 img_channel 参数
        self.conv1 = nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=(2, 1))  # 高度/2，宽度不变
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=(2, 1))  # 高度/2，宽度不变
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


def build_resnet18_backbone(img_channel=1):
    return OCR_ResNet18(img_channel=img_channel)
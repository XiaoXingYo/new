import torch
from sympy import true
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple
import random


class EMNISTSource:
    """EMNIST数据源管理"""

    def __init__(self, data_dir: str, split: str = 'digits'):
        self.data_dir = data_dir
        self.split = split
        self.images, self.labels = self._load_data()
        self.indices_by_label = {
            i: np.where(self.labels == i)[0] for i in range(10)
        }

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载并预处理EMNIST数据"""
        dataset = datasets.EMNIST(
            root=self.data_dir,
            split=self.split,
            train=True,
            download=true,
            transform=transforms.ToTensor()
        )

        images = dataset.data.float().numpy() / 255.0
        labels = dataset.targets.numpy()
        # 修复旋转问题 (EMNIST默认旋转了90度)
        images = np.transpose(images, (0, 2, 1))

        return images, labels

    def get_random_digit(self) -> Tuple[np.ndarray, str]:
        """随机获取一个数字图像"""
        digit = random.randint(0, 9)
        indices = self.indices_by_label[digit]
        idx = random.choice(indices.tolist())
        return self.images[idx], str(digit)

    def get_digit_by_label(self, label: int) -> np.ndarray:
        """根据标签获取数字图像"""
        indices = self.indices_by_label[label]
        idx = random.choice(indices.tolist())
        return self.images[idx]
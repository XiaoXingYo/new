import numpy as np
import cv2
import random
from typing import Tuple
from .emnist_source import EMNISTSource


class OCRDataGenerator:
    """OCR数据生成器"""

    def __init__(self, config):
        self.config = config
        self.source = EMNISTSource(config.data.emnist_dir)
        from .augmentor import OCRAugmentor
        self.augmentor = OCRAugmentor(config) if config.data.augment else None

    def generate_sample(self, seq_len_range: Tuple[int, int] = None) -> Tuple[np.ndarray, str]:
        """生成单个样本"""
        if seq_len_range is None:
            seq_len_range = self.config.data.seq_len_range

        h, w = self.config.data.height, self.config.data.width
        canvas = np.zeros((h, w), dtype=np.float32)

        seq_len = random.randint(*seq_len_range)
        label = ""
        x_cursor = random.randint(5, 15)

        for i in range(seq_len):
            char_img, digit = self.source.get_random_digit()
            # 随机缩放
            scale = random.uniform(0.8, 1.2)
            char_h, char_w = char_img.shape
            new_h = min(int(char_h * scale), h - 2)
            new_w = int(char_w * scale)
            char_img = cv2.resize(char_img, (new_w, new_h))

            # 垂直抖动
            y_offset = (h - new_h) // 2 + random.randint(-3, 3)
            y_offset = np.clip(y_offset, 0, h - new_h)

            # 边界检查
            if x_cursor + new_w >= w-8:#确保最后一个字不被物理截断（画布宽度是 128 像素）
                break
            label += digit
            # 粘贴（使用最大值避免黑框）
            roi = canvas[y_offset:y_offset + new_h, x_cursor:x_cursor + new_w]
            canvas[y_offset:y_offset + new_h, x_cursor:x_cursor + new_w] = np.maximum(roi, char_img)

            # 更新光标
            # ⚡️ 修改：读取配置中的重叠范围，而不是写死
            min_ov, max_ov = self.config.data.overlap_range
            overlap = random.randint(min_ov, max_ov)
            x_cursor += (new_w - overlap)

        # 数据增强
        if self.augmentor:
            canvas = self.augmentor.apply(canvas)

        # 添加batch和channel维度
        return np.expand_dims(canvas, axis=0), label
import numpy as np
import cv2
import random
from pathlib import Path
from typing import Tuple
from .emnist_source import EMNISTSource


class OCRDataGenerator:
    """全面升级的 OCR 数据生成工厂 (支持数字+公式双模式)"""

    def __init__(self, config):
        self.config = config
        self.source = EMNISTSource(config.data.emnist_dir)

        from .augmentor import OCRAugmentor
        self.augmentor = OCRAugmentor(config) if config.data.augment else None

        # 挂载自定义的数学符号引擎
        self.symbols = self._load_math_symbols()

    def _load_math_symbols(self):
        """静默加载清洗好的符号库"""
        symbols_path = Path('../data/processed_symbols.npz')
        if not symbols_path.exists():
            print("⚠️ 警告: 未找到 ./data/processed_symbols.npz，模型将只能生成纯数字。")
            return None

        data = np.load(symbols_path)
        # 建立 字符 -> numpy 数组 的映射
        return {
            '+': data.get('plus'),
            '-': data.get('minus'),
            '*': data.get('times'),
            '/': data.get('div'),
            '=': data.get('eq')
        }

    def _get_char_image(self, char: str) -> np.ndarray:
        """统一的字符获取接口 (根据字符自动路由到底层图库)"""
        if char.isdigit():
            # 数字走 EMNIST 获取
            return self.source.get_digit_by_label(int(char))
        elif self.symbols and char in self.symbols:
            # 符号走自定义图库获取
            images = self.symbols[char]
            if images is not None and len(images) > 0:
                idx = random.randint(0, len(images) - 1)
                return images[idx]
        # 兜底：如果找不到，返回一张全黑的图
        return np.zeros((28, 28), dtype=np.float32)

    def _generate_equation_string(self) -> str:
        """随机生成符合数学逻辑的公式字符串"""
        op = random.choice(['+', '-', '*', '/'])
        num1_len = random.randint(1, 3)
        num2_len = random.randint(1, 2)

        num1 = "".join([str(random.randint(0, 9)) for _ in range(num1_len)])
        num2 = "".join([str(random.randint(0, 9)) for _ in range(num2_len)])

        return f"{num1}{op}{num2}="

    def generate_sample(self, seq_len_range: Tuple[int, int] = None) -> Tuple[np.ndarray, str]:
        """将字符串渲染成连续的图像张量"""
        h, w = self.config.data.height, self.config.data.width
        canvas = np.zeros((h, w), dtype=np.float32)

        # 50% 概率生成公式，50% 概率生成纯数字串
        if self.symbols and random.random() < 0.5:
            target_str = self._generate_equation_string()
        else:
            if seq_len_range is None:
                seq_len_range = self.config.data.seq_len_range
            seq_len = random.randint(*seq_len_range)
            target_str = "".join([str(random.randint(0, 9)) for _ in range(seq_len)])

        x_cursor = random.randint(5, 15)
        label_result = ""

        for char in target_str:
            char_img = self._get_char_image(char)

            # 随机缩放 (模拟字号大小变化)
            scale = random.uniform(0.8, 1.1)
            char_h, char_w = char_img.shape
            new_h = min(int(char_h * scale), h - 2)
            new_w = max(1, int(char_w * scale))
            char_img = cv2.resize(char_img, (new_w, new_h))

            # 垂直位置扰动
            y_offset = (h - new_h) // 2 + random.randint(-4, 4)
            y_offset = np.clip(y_offset, 0, h - new_h)

            # 越界保护
            if x_cursor + new_w >= w - 4:
                break

            label_result += char

            # 粘贴 (使用最大值融合，防止覆盖已有笔画)
            roi = canvas[y_offset:y_offset + new_h, x_cursor:x_cursor + new_w]
            canvas[y_offset:y_offset + new_h, x_cursor:x_cursor + new_w] = np.maximum(roi, char_img)

            # 计算下一个字符的光标起始位置 (符号通常占用的空间更大，重叠率要降低)
            min_ov, max_ov = self.config.data.overlap_range
            overlap = random.randint(min_ov, max_ov)
            if not char.isdigit():
                overlap = random.randint(-4, 0)  # 符号左右通常有间距，不重叠

            x_cursor += (new_w - overlap)

        if self.augmentor:
            canvas = self.augmentor.apply(canvas)

        return np.expand_dims(canvas, axis=0), label_result
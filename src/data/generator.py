import numpy as np
import cv2
import random
from pathlib import Path
from typing import Tuple


class OCRDataGenerator:
    def __init__(self, config):
        self.config = config
        from .augmentor import OCRAugmentor
        self.augmentor = OCRAugmentor(config) if config.data.augment else None
        self.digits = self._load_digits()
        self.symbols = self._load_math_symbols()
    def _load_digits(self):
        digits_path = Path(r'D:\work\new\data\processed_digits.npz')
        if not digits_path.exists():
            print(f"⚠️ 找不到清洗后的数字字库: {digits_path}")
            return None
        data = np.load(digits_path)
        return {str(i): data.get(str(i)) for i in range(10)}
    def _load_math_symbols(self):
        symbols_path = Path(r'D:\work\new\data\processed_symbols.npz')
        if not symbols_path.exists():
            print(f"⚠️ 找不到清洗后的符号字库: {symbols_path}")
            return None
        data = np.load(symbols_path)
        return {
            '+': data.get('plus'),
            '-': data.get('minus'),
            '*': data.get('times'),
            '/': data.get('div'),
            '=': data.get('eq')
        }
    def _get_char_image(self, char: str) -> np.ndarray:
        images = None
        if char.isdigit() and self.digits:
            images = self.digits.get(char)
        elif self.symbols and char in self.symbols:
            images = self.symbols.get(char)
        if images is not None and len(images) > 0:
            idx = random.randint(0, len(images) - 1)
            # 🌟 直接返回 28x28 原始阵列
            return images[idx]
        return np.zeros((28, 28), dtype=np.float32)

    def _generate_equation_string(self) -> str:
        mode = random.choice(['add_sub', 'mul', 'div', 'mixed'])
        if mode == 'div':
            divisor = random.randint(2, 99)
            result = random.randint(1, 99)
            dividend = divisor * result
            equation = f"{dividend}/{divisor}"
            ans_str = str(result)
        elif mode == 'mul':
            num1 = random.randint(2, 99)
            num2 = random.randint(2, 99)
            equation = f"{num1}*{num2}"
            ans_str = str(num1 * num2)
        else:
            ops = ['+', '-'] if mode == 'add_sub' else ['+', '-', '*', '/']
            while True:
                if random.random() < 0.5:
                    num_blocks = 2
                else:
                    num_blocks = random.randint(3, 4)
                equation_temp = ""
                for i in range(num_blocks):
                    num_len = random.randint(1, 3)
                    if num_len == 1:
                        num_str = str(random.randint(0, 9))
                    else:
                        first_digit = str(random.randint(1, 9))
                        rest_digits = "".join([str(random.randint(0, 9)) for _ in range(num_len - 1)])
                        num_str = first_digit + rest_digits
                    equation_temp += num_str
                    if i < num_blocks - 1:
                        equation_temp += random.choice(ops)
                try:
                    ans = eval(equation_temp)
                    if ans == int(ans) and abs(ans) < 10000:
                        ans_str = str(int(ans))
                        equation = equation_temp
                        break
                except (ZeroDivisionError, SyntaxError):
                    continue
        rand_val = random.random()
        if rand_val < 0.2:
            return equation
        elif rand_val < 0.3:
            return equation + "="
        else:
            return equation + "=" + ans_str

    def generate_sample(self, seq_len_range: Tuple[int, int] = None) -> Tuple[np.ndarray, str]:
        h, w = self.config.data.height, self.config.data.width
        canvas = np.zeros((h, w), dtype=np.float32)
        if self.symbols and random.random() < 0.9:
            target_str = self._generate_equation_string()
        else:
            if seq_len_range is None:
                seq_len_range = self.config.data.seq_len_range
            seq_len = random.randint(*seq_len_range)
            target_str = "".join([str(random.randint(0, 9)) for _ in range(seq_len)])
        x_cursor = random.randint(20, 30)
        label_result = ""
        for i, char in enumerate(target_str):
            char_img = self._get_char_image(char)
            if char == '*':
                scale = random.uniform(0.95, 1.1)
            else:
                scale = random.uniform(0.75, 1.0)
            char_h, char_w = char_img.shape
            new_h = int(char_h * scale)
            new_w = max(1, int(char_w * scale))
            char_img = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            char_img = np.clip(char_img * 1.5, 0.0, 1.0)
            margin = 1
            max_r = max(0, (h - new_h) // 2 - margin)
            if max_r > 0:
                y_offset = (h - new_h) // 2 + random.randint(-max_r, max_r)
            else:
                y_offset = (h - new_h) // 2
            y_offset = np.clip(y_offset, 0, h - new_h)
            if x_cursor + new_w >= w - 20:
                break
            label_result += char
            roi = canvas[y_offset:y_offset + new_h, x_cursor:x_cursor + new_w]
            canvas[y_offset:y_offset + new_h, x_cursor:x_cursor + new_w] = np.maximum(roi, char_img)
            min_ov, max_ov = self.config.data.overlap_range
            if i > 0 and char == target_str[i - 1]:
                overlap = random.randint(-4, -2)
            elif char in ['*', '/']:
                overlap = random.randint(-6, -3)
            elif char in ['+', '-', '=']:
                overlap = random.randint(-2, 1)
            else:
                overlap = random.randint(min_ov, max_ov)
            x_cursor += (new_w - overlap)
        if self.augmentor:
            canvas = self.augmentor.apply(canvas)
        return np.expand_dims(canvas, axis=0), label_result
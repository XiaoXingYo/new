import numpy as np
import cv2
import random
from pathlib import Path
from typing import Tuple


class OCRDataGenerator:
    """全面升级的 OCR 数据生成工厂 (终极形态：纯净数据直供版 + 均衡乘除法 + 视觉特征保护)"""

    def __init__(self, config):
        self.config = config

        from .augmentor import OCRAugmentor
        self.augmentor = OCRAugmentor(config) if config.data.augment else None

        # 1. 直接加载你清洗好的两座纯净数据金矿
        self.digits = self._load_digits()
        self.symbols = self._load_math_symbols()

    def _load_digits(self):
        """加载纯净版 0-9 数字字库"""
        digits_path = Path(r'D:\work\new\data\processed_digits.npz')
        if not digits_path.exists():
            print(f"⚠️ 找不到清洗后的数字字库: {digits_path}")
            return None
        data = np.load(digits_path)
        return {str(i): data.get(str(i)) for i in range(10)}

    def _load_math_symbols(self):
        """加载纯净版数学符号字库"""
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
        """统一字符获取：既然已经是干净数据，拿到手直接用，不做任何二次过滤"""
        images = None

        if char.isdigit() and self.digits:
            images = self.digits.get(char)
        elif self.symbols and char in self.symbols:
            images = self.symbols.get(char)

        if images is not None and len(images) > 0:
            idx = random.randint(0, len(images) - 1)
            # 🌟 直接返回 28x28 原始阵列
            return images[idx]

        # 兜底返回空白图
        return np.zeros((28, 28), dtype=np.float32)

    def _generate_equation_string(self) -> str:
        """随机生成符合数学逻辑的公式，强制均衡四则运算，并计算出真正的正确答案"""

        # 强制均衡生成不同类型的算式，确保乘除法占比
        mode = random.choice(['add_sub', 'mul', 'div', 'mixed'])

        if mode == 'div':
            # 逆向构造除法：先生成除数和结果，再算出被除数，保证绝对能整除
            divisor = random.randint(2, 99)
            result = random.randint(1, 99)
            dividend = divisor * result
            equation = f"{dividend}/{divisor}"
            ans_str = str(result)

        elif mode == 'mul':
            # 限制乘数大小，防止结果过大导致序列超长
            num1 = random.randint(2, 99)
            num2 = random.randint(2, 99)
            equation = f"{num1}*{num2}"
            ans_str = str(num1 * num2)

        else:
            # 纯加减法 或 包含多种运算的混合模式
            ops = ['+', '-'] if mode == 'add_sub' else ['+', '-', '*', '/']
            while True:
                # 随机决定运算块的数量
                if random.random() < 0.5:
                    num_blocks = 2
                else:
                    num_blocks = random.randint(3, 4)

                equation_temp = ""
                for i in range(num_blocks):
                    # 动态生成数字：防止出现 "05" 这种带前导零的数字
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
                    # 用 eval 算出结果
                    ans = eval(equation_temp)
                    # 过滤条件
                    if ans == int(ans) and abs(ans) < 10000:
                        ans_str = str(int(ans))
                        equation = equation_temp
                        break
                except (ZeroDivisionError, SyntaxError):
                    continue

        # 3. 大幅提高完整公式（带有计算结果）的生成比例，强化 CTC 上下文记忆
        rand_val = random.random()
        if rand_val < 0.2:
            return equation  # 20% 纯算式 (如: 12+34)
        elif rand_val < 0.3:
            return equation + "="  # 10% 悬念式，增加等号鲁棒性 (如: 12+34=)
        else:
            return equation + "=" + ans_str  # 70% 完整公式 (如: 12+34=46)

    def generate_sample(self, seq_len_range: Tuple[int, int] = None) -> Tuple[np.ndarray, str]:
        h, w = self.config.data.height, self.config.data.width
        canvas = np.zeros((h, w), dtype=np.float32)

        # 大幅降低纯数字序列的比例，90%的概率生成公式算式
        if self.symbols and random.random() < 0.9:
            target_str = self._generate_equation_string()
        else:
            if seq_len_range is None:
                seq_len_range = self.config.data.seq_len_range
            seq_len = random.randint(*seq_len_range)
            target_str = "".join([str(random.randint(0, 9)) for _ in range(seq_len)])

        # 左侧铺设安全跑道，解决首字母丢失
        x_cursor = random.randint(20, 30)
        label_result = ""

        for i, char in enumerate(target_str):
            char_img = self._get_char_image(char)

            # 【新增防护】专门针对乘号：绝对不缩小，保护其密集的交叉线条特征不被压糊
            if char == '*':
                scale = random.uniform(0.95, 1.1)
            else:
                scale = random.uniform(0.75, 1.0)

            char_h, char_w = char_img.shape
            new_h = int(char_h * scale)
            new_w = max(1, int(char_w * scale))
            char_img = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 暴力提亮：把因为缩小而变灰的像素重新拉到纯白
            char_img = np.clip(char_img * 1.5, 0.0, 1.0)

            # 动态计算纵向安全扰动区，确保上下至少有 1 像素绝对安全气囊
            margin = 1
            max_r = max(0, (h - new_h) // 2 - margin)
            if max_r > 0:
                y_offset = (h - new_h) // 2 + random.randint(-max_r, max_r)
            else:
                y_offset = (h - new_h) // 2

            y_offset = np.clip(y_offset, 0, h - new_h)

            # 右侧铺设缓冲防撞区，解决末尾被一刀切
            if x_cursor + new_w >= w - 20:
                break

            label_result += char

            # 粘贴图像 (最大值融合)
            roi = canvas[y_offset:y_offset + new_h, x_cursor:x_cursor + new_w]
            canvas[y_offset:y_offset + new_h, x_cursor:x_cursor + new_w] = np.maximum(roi, char_img)

            min_ov, max_ov = self.config.data.overlap_range

            # 精准控制字符间距 (Overlap 控制核心)
            if i > 0 and char == target_str[i - 1]:
                # 针对 "44"、"88" 这种连续相同字符：绝对不允许重叠
                overlap = random.randint(-4, -2)
            elif char in ['*', '/']:
                # 【新增防护】专门针对乘除号：强行留白，设置安全气囊防粘连，防止被前面的数字带成连笔
                overlap = random.randint(-6, -3)
            elif char in ['+', '-', '=']:
                # 针对其他符号：不允许高度重叠，允许轻微靠近
                overlap = random.randint(-2, 1)
            else:
                # 普通不同数字：走配置的默认挤压范围
                overlap = random.randint(min_ov, max_ov)

            x_cursor += (new_w - overlap)

        if self.augmentor:
            canvas = self.augmentor.apply(canvas)

        return np.expand_dims(canvas, axis=0), label_result
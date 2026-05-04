# 文件：web/app.py
import sys
import io
import base64
import re
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import torch
import numpy as np
import cv2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.core import CRNN
from src.core.config import Config

app = FastAPI()


# 🌟 1. 极致精简版翻译官 (彻底脱离对 src/data 文件夹的依赖)
class LabelConverter:
    def __init__(self, chars, blank_label):
        self.chars = chars
        self.blank_label = blank_label

    def decode(self, indices):
        result = []
        for i, idx in enumerate(indices):
            val = idx.item()
            if val != self.blank_label:
                # CTC 规则：忽略连续重复的字符
                if i == 0 or val != indices[i - 1].item():
                    if val < len(self.chars):
                        result.append(self.chars[val])
        return "".join(result)


# 2. 唤醒模型
cfg = Config(str(PROJECT_ROOT / "configs/base.yaml"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(num_classes=len(cfg.data.chars) + 1).to(device)

# 尝试加载权重 (即便没有权重，也能启动画板，只是识别结果是乱码)
ckpt_path = PROJECT_ROOT / cfg.inference.model_path
if ckpt_path.exists():
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device)['model'])
model.eval()

converter = LabelConverter(cfg.data.chars, cfg.data.blank_label)


class ImageData(BaseModel):
    image_base64: str


# 3. 底层数学解析器 (双栈结构，拒绝 eval 黑盒)
def stack_calculator(expression: str) -> str:
    expr = expression.replace('=', '').strip()
    expr = expr.replace('x', '*').replace('X', '*').replace('×', '*').replace('÷', '/')
    tokens = re.findall(r'\d+\.\d+|\d+|[+*/-]', expr)
    if not tokens: return ""

    def apply_op(ops, values):
        op = ops.pop()
        b, a = values.pop(), values.pop()
        if op == '+':
            values.append(a + b)
        elif op == '-':
            values.append(a - b)
        elif op == '*':
            values.append(a * b)
        elif op == '/':
            values.append(a / b if b != 0 else 0)

    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    values, ops = [], []

    try:
        for token in tokens:
            if token.replace('.', '', 1).isdigit():
                values.append(float(token))
            else:
                while ops and precedence.get(ops[-1], 0) >= precedence.get(token, 0):
                    apply_op(ops, values)
                ops.append(token)
        while ops:
            apply_op(ops, values)
        res = values[0]
        return str(int(res)) if res.is_integer() else f"{res:.2f}"
    except Exception:
        return "?"


# 4. 核心 API 端点
@app.post("/api/recognize")
async def recognize(data: ImageData):
    img_data = base64.b64decode(data.image_base64.split(",")[1])
    pil_image = Image.open(io.BytesIO(img_data)).convert('L')

    img_cv = np.array(pil_image, dtype=np.uint8)
    img_cv = cv2.bitwise_not(img_cv)

    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"result": "未检测到笔画", "calc": ""}

    x_min = min([cv2.boundingRect(c)[0] for c in contours])
    y_min = min([cv2.boundingRect(c)[1] for c in contours])
    x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
    y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])

    # 🌟 修复 1：裁剪前留出 5 像素安全边距，防止膨胀时边缘被截断！
    margin = 5
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(img_cv.shape[1], x_max + margin)
    y_max = min(img_cv.shape[0], y_max + margin)

    img_cropped = img_cv[y_min:y_max, x_min:x_max]

    # 🌟 修复 2：针对拥挤字符，改用 2x2 的微小内核，或者直接注释掉这行不膨胀！
    # kernel = np.ones((2, 2), np.uint8)
    # img_cropped = cv2.dilate(img_cropped, kernel, iterations=1)

    h, w = img_cropped.shape

    # 🌟 修复 3：底层防御，绝不允许除零错误干倒我们的服务！
    if h == 0 or w == 0:
        return {"result": "无效图像", "calc": ""}

    target_digit_h = 28
    scale = target_digit_h / float(h)
    new_w = max(1, int(w * scale*1.3))
    img_resized = cv2.resize(img_cropped, (new_w, target_digit_h), interpolation=cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    img_resized = cv2.dilate(img_resized, kernel, iterations=1)
    canvas_h = 32
    config_w = 192
    pad_x = 24  # 模拟训练集 x_cursor 的起始位置
    # 如果用户写得实在太长，超过了 192，那我们就动态扩张，否则保底 192
    final_w = max(config_w, new_w + pad_x + 16)
    pad_y = (canvas_h - target_digit_h) // 2

    final_img = np.zeros((canvas_h, final_w), dtype=np.float32)
    img_resized_norm = img_resized.astype(np.float32) / 255.0
    img_resized_norm = np.clip(img_resized_norm * 2.0, 0.0, 1.0)
    final_img[pad_y:pad_y + target_digit_h, pad_x:pad_x + new_w] = img_resized_norm
    debug_img = (final_img * 255).astype(np.uint8)
    cv2.imwrite("debug_model_input.png", debug_img)
    img_tensor = torch.from_numpy(final_img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)
        pred_indices = preds.argmax(2).permute(1, 0)
        pred_str = converter.decode(pred_indices[0])

    calc_res = stack_calculator(pred_str) if '=' in pred_str else ""
    return {"result": pred_str, "calc": calc_res}


# 5. 挂载画板
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "web/static"), html=True), name="static")
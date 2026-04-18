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
    image = Image.open(io.BytesIO(img_data)).convert('L')

    # 1. 智能寻边：精准切出写的字 (完全复刻你的 gui_demo 逻辑)
    bbox = image.getbbox()
    if not bbox: return {"result": "", "calc": ""}

    img_cropped = image.crop(bbox)

    # 2. 解开宽度封印：只固定高度为 32，宽度按真实比例无限延伸！
    target_h = 32
    scale = target_h / img_cropped.height
    new_w = int(img_cropped.width * scale)
    if new_w <= 0: new_w = 1

    img_resized = img_cropped.resize((new_w, target_h), Image.Resampling.LANCZOS)

    # 3. 加装“安全气囊”：左右各加 16 像素的黑边
    pad_x = 16
    final_w = new_w + pad_x * 2
    final_img = Image.new("L", (final_w, target_h), "black")
    final_img.paste(img_resized, (pad_x, 0))

    # 4. 二值化强化笔画 (绝对不搞什么减0.5的归一化，保持黑底0，白字1！)
    img_arr = np.array(final_img, dtype=np.float32) / 255.0
    img_arr[img_arr > 0.2] = 1.0

    # 转换为模型需要的张量 (1, 1, H, W)
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)
        pred_indices = preds.argmax(2).permute(1, 0)
        pred_str = converter.decode(pred_indices[0])

    calc_res = stack_calculator(pred_str) if '=' in pred_str else ""
    return {"result": pred_str, "calc": calc_res}


# 5. 挂载画板
app.mount("/", StaticFiles(directory=str(PROJECT_ROOT / "web/static"), html=True), name="static")
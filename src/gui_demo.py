# 文件路径: src/gui_demo.py
import tkinter as tk
from tkinter import font
from PIL import Image, ImageDraw
import torch
import numpy as np
import sys
from pathlib import Path

# 确保能找到 src 下的模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.models.factory import build_model
from src.data.loader import LabelConverter


class OCRDemoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 RTX 5060 OCR 终极检验器")
        self.root.geometry("600x400")
        self.root.configure(bg="#2b2b2b")

        # 画板尺寸（按照你模型 32x128 的比例，放大 4 倍变成 128x512 以方便手写）
        self.canvas_width = 512
        self.canvas_height = 128

        # --- UI 组件设置 ---
        title_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.lbl_title = tk.Label(root, text="请在下方黑板中写下 0-9 的数字", bg="#2b2b2b", fg="white", font=title_font)
        self.lbl_title.pack(pady=15)

        # 交互画板
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="black", cursor="pencil")
        self.canvas.pack(pady=10)

        # 隐藏的 PIL 图像，用于真实保存你的轨迹 (黑底白字，匹配你的数据生成器)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)

        # 按钮区域
        btn_frame = tk.Frame(root, bg="#2b2b2b")
        btn_frame.pack(pady=10)

        self.btn_recognize = tk.Button(btn_frame, text="⚡ 识别 (Recognize)", command=self.recognize,
                                       bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), width=15)
        self.btn_recognize.pack(side=tk.LEFT, padx=10)

        self.btn_clear = tk.Button(btn_frame, text="🗑️ 清空 (Clear)", command=self.clear_canvas,
                                   bg="#f44336", fg="white", font=("Helvetica", 12, "bold"), width=15)
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        # 结果显示标签
        self.lbl_result = tk.Label(root, text="识别结果: 等待输入...", bg="#2b2b2b", fg="#00FF00",
                                   font=("Helvetica", 20, "bold"))
        self.lbl_result.pack(pady=20)

        # --- 绑定鼠标事件 ---
        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x, self.last_y = None, None
        self.canvas.bind("<ButtonRelease-1>", self.reset_pen)

        # --- 加载你的模型 ---
        self.setup_model()

    def setup_model(self):
        self.lbl_result.config(text="正在唤醒 RTX 5060...", fg="yellow")
        self.root.update()

        try:
            # 1. 加载配置
            self.cfg = Config("configs/base.yaml")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 2. 召唤工厂构建模型
            self.model = build_model(self.cfg, arch="ctc").to(self.device)
            self.model.eval()

            # 3. 加载你刚刚炼成的绝世权重
            ckpt_path = self.cfg.inference.model_path  # 读取 configs 里的 inference 路径
            ckpt = torch.load(ckpt_path, map_location=self.device)

            # 兼容不同格式的 checkpoint
            if 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'])
            else:
                self.model.load_state_dict(ckpt)

            # 4. 召唤翻译官
            self.converter = LabelConverter(self.cfg.data.chars, self.cfg.data.blank_label)

            self.lbl_result.config(text=f"模型就绪! (设备: {self.device})", fg="#00FF00")
        except Exception as e:
            self.lbl_result.config(text=f"模型加载失败! 请检查权重路径", fg="red")
            print(f"Error: {e}")

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # 在 Tkinter 画板上画线 (供人类看)
            # 笔刷粗细设置为 8，模拟 EMNIST 的笔画粗细
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=12, fill="white", capstyle=tk.ROUND,
                                    smooth=tk.TRUE)
            # 在隐藏的 PIL 图像上画线 (供模型看)
            self.draw.line([self.last_x, self.last_y, x, y], fill="white", width=12, joint="curve")
        self.last_x, self.last_y = x, y

    def reset_pen(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, self.canvas_width, self.canvas_height), fill="black")
        self.lbl_result.config(text="识别结果: 等待输入...", fg="#00FF00")

    def recognize(self):
        # 1. 智能寻边：精准切出你写的字
        bbox = self.image.getbbox()
        if not bbox:
            self.lbl_result.config(text="识别结果: [画板是空的]", fg="#00FFFF")
            return

        img_cropped = self.image.crop(bbox)

        # 2. ⚡️ 解开宽度封印：只固定高度为 32，宽度按真实比例无限延伸！
        target_h = 32
        scale = target_h / img_cropped.height
        new_w = int(img_cropped.width * scale)

        # 哪怕你写了 20 个字，算出来宽度是 500，我们也保留它原始的胖瘦！
        img_resized = img_cropped.resize((new_w, target_h), Image.Resampling.LANCZOS)

        # 3. 加装“安全气囊”：左右各加 16 像素的黑边，防止边缘字被吞噬
        pad_x = 16
        final_w = new_w + pad_x * 2
        final_img = Image.new("L", (final_w, target_h), "black")
        final_img.paste(img_resized, (pad_x, 0))

        # 4. 二值化强化笔画
        img_arr = np.array(final_img, dtype=np.float32) / 255.0
        img_arr[img_arr > 0.2] = 1.0

        # 🚀 这里的宽度是自由的 final_w，不再是死板的 128！
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(self.device)

        # 5. 模型推断
        with torch.no_grad():
            preds = self.model(img_tensor)
            pred_indices = preds.argmax(2).permute(1, 0)

            pred_str = self.converter.decode(pred_indices[0])
            if pred_str == "":
                pred_str = "[未识别出字符]"

            self.lbl_result.config(text=f"识别结果: {pred_str}", fg="#00FFFF")


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRDemoGUI(root)
    root.mainloop()
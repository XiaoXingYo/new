import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

from ..core.logger import get_logger           # ✅ 修复导入


class Inferencer:
    """推理引擎"""

    def __init__(self, config):
        self.cfg = config
        raw_device = config.train.device
        if raw_device == "auto":
            # 如果配置是 auto，自动检测是否有显卡
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🤖 设备设为 auto，自动检测可用硬件为: {self.device}")
        else:
            # 否则严格按照配置来
            self.device = torch.device(raw_device)
            print(f"⚙️ 根据配置，使用指定硬件: {self.device}")
        # 模型
        from ..models.factory import build_model  # ✅ 修复导入
        self.model = build_model(config).to(self.device)

        # 加载权重
        model_path = Path(config.inference.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state['model'])
        self.model.eval()

        # 转换器
        from ..data.loader import LabelConverter  # ✅ 修复导入
        self.converter = LabelConverter(config.data.chars, config.data.blank_label)

        self.logger = get_logger("Inferencer", "logs/infer.log")
        self.logger.info(f"模型加载成功: {model_path}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 调整大小
        h, w = self.cfg.data.height, self.cfg.data.width
        image = cv2.resize(image, (w, h))

        # 归一化
        if image.max() > 1.0:
            image = image / 255.0

        # 添加维度 (1, 1, H, W)
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: torch.Tensor) -> Tuple[str, str]:
        """预测单张图像"""
        with torch.no_grad():
            preds = self.model(image)  # (T, 1, C)

            if self.cfg.inference.decode_type == "greedy":
                pred_indices = preds.argmax(2).squeeze(1)  # (T,)
                text = self.converter.decode(pred_indices)
                raw_text = self.converter.decode(pred_indices, raw=True)
                return text, raw_text
            else:
                # Beam search实现
                from utils.decoders import beam_search_decode
                return beam_search_decode(
                    preds,
                    self.converter,
                    self.cfg.inference.beam_width
                )

    def run_interactive(self):
        """交互式推理（使用生成器）"""
        from data.generator import OCRDataGenerator

        generator = OCRDataGenerator(self.cfg)

        print("\n🎹 按 'Enter' 生成新样本，按 'q' 退出...")

        cv2.namedWindow("OCR Inference", cv2.WINDOW_NORMAL)

        while True:
            # 生成样本
            img_np, label_true = generator.generate_sample()
            img_tensor = torch.from_numpy(img_np).to(self.device)

            # 预测
            pred, raw = self.predict(img_tensor)

            # 可视化
            img_vis = (img_np[0] * 255).astype(np.uint8)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)

            # 绘制结果
            text_line = f"GT: {label_true} | Pred: {pred} | Raw: {raw}"
            cv2.putText(
                img_vis, text_line, (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

            cv2.imshow("OCR Inference", img_vis)

            key = cv2.waitKey(0)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
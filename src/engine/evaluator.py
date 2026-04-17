import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
import numpy as np

from ..core.metrics import calculate_cer_wer  # ✅ 修复导入
from ..core.logger import get_logger  # ✅ 修复导入


class Evaluator:
    """评估引擎"""

    def __init__(self, config, model, val_loader, device):
        self.cfg = config
        self.device = device

        # ✅ 修复：延迟加载模型
        if model is None:
            from ..models.factory import build_model
            self.model = build_model(config).to(device)
        else:
            self.model = model.to(device)

        self.val_loader = val_loader
        self.logger = get_logger("Evaluator", "logs/eval.log")

        # ✅ 修复：相对导入
        from ..data.loader import LabelConverter
        self.converter = LabelConverter(config.data.chars, config.data.blank_label)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """完整评估"""
        self.model.eval()
        total_loss = 0
        total_cer, total_wer, count = 0.0, 0.0, 0

        criterion = torch.nn.CTCLoss(
            blank=self.cfg.data.blank_label,
            zero_infinity=True
        ).to(self.device)

        all_preds, all_gts = [], []

        for images, labels_str in tqdm(self.val_loader, desc="Evaluating"):
            images = images.to(self.device)
            batch_size = images.size(0)

            # 编码标签
            targets, target_lengths = self.converter.encode(labels_str)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # 前向
            preds = self.model(images)
            preds_log = preds.log_softmax(2)

            input_lengths = torch.full(
                (batch_size,), preds.size(0),
                dtype=torch.long, device=self.device
            )
            loss = criterion(preds_log, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # 解码
            pred_indices = preds.argmax(2).permute(1, 0)

            # 注意：如果你的 decode 需要传入长度，这里可能需要改成：
            # pred_strs = self.converter.decode(pred_indices, [pred_indices.size(1)] * batch_size)
            # 先按你原来的写法保留
            pred_strs = [self.converter.decode(indices) for indices in pred_indices]

            # 收集预测
            all_preds.extend(pred_strs)
            all_gts.extend(labels_str)

            # 计算指标
            for pred, gt in zip(pred_strs, labels_str):
                # ⚡️ 核心排错探针：只打印每个 Epoch 的前 3 个结果，防止刷屏！
                if count < 3:
                    probe_msg = (
                        f"\n{'=' * 40}\n"
                        f"🚨 [探针排错 - 样本 {count + 1}]\n"
                        f"真实答案 (GT) : '{gt}' | 长度: {len(gt)}\n"
                        f"模型预测 (Pred): '{pred}' | 长度: {len(pred)}\n"
                        f"{'=' * 40}"
                    )
                    print(probe_msg)  # 打印到终端给你看
                    self.logger.info(probe_msg)  # 同时也写进日志里

                cer, wer = calculate_cer_wer(pred, gt)
                total_cer += cer
                total_wer += wer
                count += 1

        avg_loss = total_loss / len(self.val_loader)
        avg_cer = (total_cer / count) * 100
        avg_wer = (total_wer / count) * 100

        self.logger.info(
            f"评估完成: loss={avg_loss:.4f}, "
            f"cer={avg_cer:.2f}%, wer={avg_wer:.2f}%"
        )

        return {
            'loss': avg_loss,
            'cer': avg_cer,
            'wer': avg_wer,
            'predictions': list(zip(all_preds, all_gts))
        }
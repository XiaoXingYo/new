import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Dict, Any
from pathlib import Path
import numpy as np

from ..core.checkpoint import CheckpointManager
from ..core.metrics import calculate_cer_wer
from ..core.logger import get_logger


class Trainer:
    """训练引擎"""

    def __init__(self, config, model, optimizer, scheduler, criterion,
                 train_loader, val_loader, device):
        self.cfg = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.logger = get_logger("Trainer", "logs/trainer.log")
        self.ckpt_mgr = CheckpointManager(config.train.output_dir)

        self.step = 0
        self.epoch = 0
        self.best_val_cer = float('inf')

        # Label converter
        from ..data.loader import LabelConverter
        self.converter = LabelConverter(config.data.chars, config.data.blank_label)

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (images, labels_str) in enumerate(pbar):
            images = images.to(self.device)
            batch_size = images.size(0)

            # 编码标签
            targets, target_lengths = self.converter.encode(labels_str)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            preds = self.model(images)  # (T, B, C)
            preds_log = preds.log_softmax(2)

            # CTC Loss
            input_lengths = torch.full(
                (batch_size,), preds.size(0),
                dtype=torch.long, device=self.device
            )
            loss = self.criterion(preds_log, targets, input_lengths, target_lengths)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.train.grad_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            # 日志
            total_loss += loss.item()

            current_lr = self.optimizer.param_groups[0]['lr']
            if self.step % self.cfg.train.log_every == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.6f}'
                })
                self.logger.debug(
                    f"Step {self.step}: loss={loss.item():.4f}, lr={current_lr:.6f}"
                )

            self.step += 1

        return {'loss': total_loss / num_batches}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        total_cer, total_wer, count = 0.0, 0.0, 0

        for images, labels_str in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            batch_size = images.size(0)

            # 编码
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
            loss = self.criterion(preds_log, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # 解码
            pred_indices = preds.argmax(2).permute(1, 0)  # (B, T)
            pred_strs = [self.converter.decode(indices) for indices in pred_indices]

            # 计算指标
            for pred, gt in zip(pred_strs, labels_str):
                # ==========================================
                # ⚡️ 核心排错探针：每次验证只打印前 3 个结果看看
                # ==========================================
                if count < 3:
                    probe_msg = (
                        f"\n{'=' * 40}\n"
                        f"🚨 [探针排错 - 验证集样本 {count + 1}]\n"
                        f"真实答案 (GT) : '{gt}' | 长度: {len(gt)}\n"
                        f"模型预测 (Pred): '{pred}' | 长度: {len(pred)}\n"
                        f"{'=' * 40}"
                    )
                    print(probe_msg)  # 直接打在终端屏幕上

                cer, wer = calculate_cer_wer(pred, gt)
                total_cer += cer
                total_wer += wer
                count += 1

        avg_cer = (total_cer / count) * 100
        avg_wer = (total_wer / count) * 100

        return {
            'loss': total_loss / len(self.val_loader),
            'cer': avg_cer,
            'wer': avg_wer
        }

    def run(self):
        """主训练循环"""
        self.logger.info(
            f"开始训练: {self.cfg.train.epochs} epochs, "
            f"device: {self.device}"
        )

        for epoch in range(self.cfg.train.epochs):
            self.epoch = epoch

            # 训练
            train_stats = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_stats['loss']:.4f}"
            )

            # 验证
            if epoch % self.cfg.train.val_every == 0:
                val_stats = self.validate()
                self.logger.info(
                    f"Epoch {epoch}: val_loss={val_stats['loss']:.4f}, "
                    f"cer={val_stats['cer']:.2f}%, wer={val_stats['wer']:.2f}%"
                )

                # 保存最佳模型
                if val_stats['cer'] < self.best_val_cer:
                    self.best_val_cer = val_stats['cer']
                    self.ckpt_mgr.save(
                        self.model, self.optimizer, epoch, self.step,
                        val_stats, is_best=True
                    )
                    self.logger.info(
                        f"最佳模型已保存 (CER: {val_stats['cer']:.2f}%)"
                    )

                # 定期保存
                if epoch % self.cfg.train.save_every == 0:
                    self.ckpt_mgr.save(
                        self.model, self.optimizer, epoch, self.step,
                        val_stats, is_best=False
                    )

        self.logger.info("✅ 训练完成！")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """加载checkpoint"""
        return self.ckpt_mgr.load(self.model, self.optimizer, path)
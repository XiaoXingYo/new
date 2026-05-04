import torch
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np


class OCREngine:
    """高度内聚的训练与推理引擎 (已加入阶段分离、动态解码、混合精度与梯度累积)"""

    def __init__(self, model, device, converter, config):
        self.model = model.to(device)
        self.device = device
        self.converter = converter
        self.cfg = config
        self.train_batch_saved = False
        self.eval_count = 0
        self.current_stage = "Default"  # 🌟 记录当前训练阶段

        # 🌟 初始化混合精度 Scaler (已修复：使用新版 torch.amp API)
        self.use_amp = self.cfg.mixed_precision.enabled
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # 🌟 初始化梯度累积步数
        self.accum_steps = self.cfg.gradient_accumulation.steps if self.cfg.gradient_accumulation.enabled else 1

    def train_loop(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc="Training")

        # 梯度清零，为梯度累积做准备
        optimizer.zero_grad()

        for step, (images, labels_str) in enumerate(pbar):
            if not self.train_batch_saved:
                try:
                    out_dir = Path("logs")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    from src.utils.viz import visualize_predictions
                    visualize_predictions(
                        images[:10].cpu(), labels_str[:10], labels_str[:10],
                        str(out_dir / f"00_{self.current_stage}_preview.png"),
                        num_samples=10, is_preview=True
                    )
                except Exception:
                    pass
                self.train_batch_saved = True

            images = images.to(self.device)
            targets, target_lengths = self.converter.encode(labels_str)

            # 🌟 开启混合精度上下文 (已修复：使用新版 torch.amp API)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                preds = self.model(images)
                preds_log = preds.log_softmax(2)

                input_lengths = torch.full((images.size(0),), preds.size(0), dtype=torch.long)
                loss = criterion(preds_log, targets.to(self.device), input_lengths, target_lengths.to(self.device))

                # 🌟 梯度累积：按比例缩放 Loss
                loss = loss / self.accum_steps

            # 🌟 混合精度 Backward
            self.scaler.scale(loss).backward()

            # 🌟 梯度累积：达到指定步数才更新权重
            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(train_loader):
                # 解除缩放，以便进行梯度裁剪
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.grad_clip)

                # Optimizer Step & Scaler Update
                self.scaler.step(optimizer)
                self.scaler.update()

                # 更新后清空梯度
                optimizer.zero_grad()

            # 恢复真实的 loss 用于日志打印
            real_loss = loss.item() * self.accum_steps
            total_loss += real_loss
            pbar.set_postfix({'loss': f"{real_loss:.4f}"})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        correct, total = 0, 0
        self.eval_count += 1

        epoch_error_dir = Path(f"logs/bad_cases/{self.current_stage}/epoch_{self.eval_count:02d}")

        # 🌟 修复：改为面向对象的属性调用
        decode_type = self.cfg.inference.decode_type
        beam_width = self.cfg.inference.beam_width

        for images, labels_str in tqdm(val_loader, desc=f"Evaluating ({decode_type})"):
            images = images.to(self.device)

            # 推理阶段同样可以使用 autocast 提速 (已修复：使用新版 torch.amp API)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                preds = self.model(images)  # 形状: [Sequence_Length, Batch_Size, Num_Classes]

            # 🌟 根据配置准备解码输入
            if decode_type == 'beam_search':
                probs = torch.nn.functional.softmax(preds, dim=2).permute(1, 0, 2)
            else:
                pred_indices = preds.argmax(2).permute(1, 0)

            for i, label in enumerate(labels_str):
                # 🌟 调用不同的解码策略
                if decode_type == 'beam_search':
                    pred_str = self.converter.decode(probs[i], decode_type='beam_search', beam_size=beam_width)
                else:
                    pred_str = self.converter.decode(pred_indices[i], decode_type='greedy')

                if pred_str == label:
                    correct += 1
                else:
                    try:
                        if not epoch_error_dir.exists():
                            epoch_error_dir.mkdir(parents=True, exist_ok=True)

                        img_tensor = images[i].cpu().numpy().squeeze()
                        if img_tensor.max() <= 1.0:
                            img_tensor = (img_tensor * 255).astype(np.uint8)
                        else:
                            img_tensor = img_tensor.astype(np.uint8)

                        safe_truth = label.replace('*', 'X').replace('/', 'D')
                        safe_pred = pred_str.replace('*', 'X').replace('/', 'D')
                        filename = f"T_{safe_truth}__P_{safe_pred}.png"
                        cv2.imwrite(str(epoch_error_dir / filename), img_tensor)
                    except Exception:
                        pass
                total += 1
        return correct / total

    @torch.no_grad()
    def infer(self, image_tensor):
        self.model.eval()

        # 推理阶段同样可以使用 autocast 提速 (已修复：使用新版 torch.amp API)
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            preds = self.model(image_tensor.to(self.device))  # 形状: [Sequence_Length, 1, Num_Classes]

        # 🌟 修复：改为面向对象的属性调用
        decode_type = self.cfg.inference.decode_type
        beam_width = self.cfg.inference.beam_width

        # 🌟 根据配置执行解码
        if decode_type == 'beam_search':
            probs = torch.nn.functional.softmax(preds, dim=2).permute(1, 0, 2)
            return self.converter.decode(probs[0], decode_type='beam_search', beam_size=beam_width)
        else:
            pred_indices = preds.argmax(2).permute(1, 0)
            return self.converter.decode(pred_indices[0], decode_type='greedy')
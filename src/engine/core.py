import time
import torch
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np

# 🌟 新增：导入 CER 计算函数
from src.core.metrics import calculate_cer


class OCREngine:
    """高度内聚的训练与推理引擎 (兼容 CTC 与 Attention，支持 CER 与 FPS 测算)"""
    def __init__(self, model, device, converter, config):
        self.model = model.to(device)
        self.device = device
        self.converter = converter
        self.cfg = config
        self.train_batch_saved = False
        self.eval_count = 0
        self.current_stage = "Default"
        self.use_amp = self.cfg.mixed_precision.enabled
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.accum_steps = self.cfg.gradient_accumulation.steps if self.cfg.gradient_accumulation.enabled else 1
        self.is_attention = hasattr(model, 'attention')

    def train_loop(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc="Training")
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
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if self.is_attention:
                    targets = self.converter.encode(labels_str).to(self.device)
                    preds = self.model(images, targets)
                    loss = criterion(preds.view(-1, preds.size(-1)), targets.view(-1))
                else:
                    targets, target_lengths = self.converter.encode(labels_str)
                    preds = self.model(images)
                    preds_log = preds.log_softmax(2)
                    input_lengths = torch.full((images.size(0),), preds.size(0), dtype=torch.long)
                    loss = criterion(preds_log, targets.to(self.device), input_lengths, target_lengths.to(self.device))
                loss = loss / self.accum_steps
            self.scaler.scale(loss).backward()
            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(train_loader):
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            real_loss = loss.item() * self.accum_steps
            total_loss += real_loss
            pbar.set_postfix({'loss': f"{real_loss:.4f}"})
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        correct, total = 0, 0
        total_cer = 0.0
        self.eval_count += 1
        start_time = time.time()
        epoch_error_dir = Path(f"logs/bad_cases/{self.current_stage}/epoch_{self.eval_count:02d}")
        decode_type = self.cfg.inference.decode_type
        beam_width = self.cfg.inference.beam_width
        for images, labels_str in tqdm(val_loader, desc=f"Evaluating ({decode_type})"):
            images = images.to(self.device)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                preds = self.model(images)
            if self.is_attention:
                pred_indices = preds.argmax(2)
                pred_strs = self.converter.decode(pred_indices)
            else:
                if decode_type == 'beam_search':
                    probs = torch.nn.functional.softmax(preds, dim=2).permute(1, 0, 2)
                    pred_strs = [self.converter.decode(probs[i], decode_type='beam_search', beam_size=beam_width) for i
                                 in range(len(labels_str))]
                else:
                    pred_indices = preds.argmax(2).permute(1, 0)
                    pred_strs = [self.converter.decode(pred_indices[i], decode_type='greedy') for i in
                                 range(len(labels_str))]
            for i, label in enumerate(labels_str):
                pred_str = pred_strs[i]
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
                total_cer += calculate_cer(pred_str, label)
                total += 1
        eval_time = time.time() - start_time
        fps = total / eval_time if eval_time > 0 else 0
        avg_acc = correct / total
        avg_cer = total_cer / total
        return avg_acc, avg_cer, fps
    @torch.no_grad()
    def infer(self, image_tensor):
        self.model.eval()
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            preds = self.model(image_tensor.to(self.device))
        if self.is_attention:
            pred_indices = preds.argmax(2)
            return self.converter.decode(pred_indices)[0]
        else:
            decode_type = self.cfg.inference.decode_type
            beam_width = self.cfg.inference.beam_width
            if decode_type == 'beam_search':
                probs = torch.nn.functional.softmax(preds, dim=2).permute(1, 0, 2)
                return self.converter.decode(probs[0], decode_type='beam_search', beam_size=beam_width)
            else:
                pred_indices = preds.argmax(2).permute(1, 0)
                return self.converter.decode(pred_indices[0], decode_type='greedy')
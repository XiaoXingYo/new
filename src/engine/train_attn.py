import torch
import torch.nn as nn
from tqdm import tqdm


# 导入你刚才写的模型和翻译官
# from models.attention_ocr import Seq2SeqAttention
# from data.loader import AttentionLabelConverter

def train_attention_epoch(model, dataloader, optimizer, converter, device, epoch):
    model.train()

    # ⚡️ 核心差异 1：不用 CTCLoss 了！
    # 改用 CrossEntropyLoss，并且告诉它：遇到 PAD_IDX 就无视，不要算分
    criterion = nn.CrossEntropyLoss(ignore_index=converter.pad_idx)

    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Attention]")
    for images, labels in pbar:
        if images.dim() == 5:
            images = images.squeeze(1)
        images = images.to(device)

        # 1. 翻译官把标签转成 Tensor，形状是 (B, T)
        targets = converter.encode(labels).to(device)

        optimizer.zero_grad()

        # 2. 模型前向传播
        # ⚡️ 核心差异 2：Teacher Forcing
        # 把标准答案(targets)传给模型，如果模型上一步猜错了，老师强行塞给它正确的答案让它继续做下一步
        outputs = model(images, target_tensor=targets, teacher_forcing_ratio=0.5)

        # 3. 计算 Loss
        # outputs 的形状是 (Batch, Time, Classes) -> 比如 (128, 12, 13)
        # targets 的形状是 (Batch, Time) -> 比如 (128, 12)
        # CrossEntropy 要求输入是二维的 (N, Classes)，所以我们要把它们展平！
        B, T, C = outputs.size()
        outputs_flat = outputs.view(B * T, C)
        targets_flat = targets.view(B * T)

        loss = criterion(outputs_flat, targets_flat)

        # 4. 反向传播更新权重
        loss.backward()
        # 防治梯度爆炸（Attention 模型比较脆弱，这个必须加）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)
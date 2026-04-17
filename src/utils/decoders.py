"""
CTC解码器实现
提供贪婪解码和束搜索解码
"""

import torch
import numpy as np
from typing import Tuple


def beam_search_decode(
        preds: torch.Tensor,
        converter,
        beam_width: int = 5
) -> Tuple[str, str]:
    """
    Beam Search解码器

    Args:
        preds: 模型输出，形状 (T, B, C)
        converter: LabelConverter实例
        beam_width: 束宽度，越大越精确但越慢

    Returns:
        (解码文本, 原始beam路径)
    """
    # 取log_softmax确保数值稳定
    log_probs = preds.log_softmax(2).squeeze(1).cpu().numpy()  # (T, C)

    # 初始化beam: (前缀, 得分)
    beams = [([], 0.0)]

    for t in range(log_probs.shape[0]):
        new_beams = []
        for prefix, score in beams:
            for c in range(log_probs.shape[1]):
                # CTC规则：跳过重复字符
                if prefix and c == prefix[-1]:
                    continue

                new_prefix = prefix + [c]
                new_score = score + log_probs[t, c]
                new_beams.append((new_prefix, new_score))

        # 保留Top-K beam
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    # 解码最佳序列
    best_seq, best_score = beams[0]

    # 转换为字符串
    text = converter.decode(torch.tensor(best_seq))
    raw_text = converter.decode(torch.tensor(best_seq), raw=True)

    return text, raw_text


def greedy_decode(
        preds: torch.Tensor,
        converter
) -> Tuple[str, str]:
    """
    贪婪解码（Greedy Decode）
    每个时间步取概率最高的字符

    Args:
        preds: 模型输出，形状 (T, B, C)
        converter: LabelConverter实例

    Returns:
        (解码文本, 原始路径)
    """
    pred_indices = preds.argmax(2).squeeze(1)  # (T,)
    text = converter.decode(pred_indices)
    raw_text = converter.decode(pred_indices, raw=True)
    return text, raw_text
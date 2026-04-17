import editdistance
from typing import Tuple

def calculate_cer(predicted: str, ground_truth: str) -> float:
    """字符错误率 (Character Error Rate)"""
    if not ground_truth:
        return 1.0
    return editdistance.eval(predicted, ground_truth) / len(ground_truth)

def calculate_wer(predicted: str, ground_truth: str) -> float:
    """词错误率 (Word Error Rate) - 对于数字串，整个串看作一个词"""
    if not ground_truth:
        return 1.0
    # 修复：将整串视为一个词，全对为0，否则为1
    return 0.0 if predicted == ground_truth else 1.0

def calculate_cer_wer(predicted: str, ground_truth: str) -> Tuple[float, float]:
    """返回CER和WER"""
    return calculate_cer(predicted, ground_truth), calculate_wer(predicted, ground_truth)
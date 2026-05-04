import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List


def plot_training_curves(metrics: dict, save_path: str):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    # Loss曲线
    axes[0, 0].plot(metrics['train_loss'], label='Train')
    axes[0, 0].plot(metrics['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()

    # CER曲线
    axes[0, 1].plot(metrics['val_cer'], label='CER')
    axes[0, 1].set_title('Character Error Rate')
    axes[0, 1].legend()

    # WER曲线
    axes[1, 0].plot(metrics['val_wer'], label='WER')
    axes[1, 0].set_title('Word Error Rate')
    axes[1, 0].legend()

    # 学习率
    if 'lr' in metrics:
        axes[1, 1].plot(metrics['lr'], label='LR')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_predictions(images: torch.Tensor,
                          predictions: List[str],
                          ground_truths: List[str],
                          save_path: str,
                          num_samples: int = 10,
                          is_preview: bool = False):
    """可视化预测结果 或 预览训练数据"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(min(num_samples, len(images))):
        img = images[i].cpu().numpy().squeeze()
        gt = ground_truths[i]

        axes[i].imshow(img, cmap='gray')

        # 🌟 核心修改：如果是预览模式，只显示蓝色 Label；否则显示 P 和 G
        if is_preview:
            axes[i].set_title(f'Label: {gt}', color='blue')
        else:
            pred = predictions[i]
            axes[i].set_title(f'P: {pred}\nG: {gt}',
                              color='green' if pred == gt else 'red')

        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
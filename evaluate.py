import sys
import time
import difflib
import os
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.core.metrics import calculate_cer
from src.data.loader import build_dataloaders, LabelConverter, AttentionLabelConverter
from src.models.core import CRNN
from src.models.baselines import MobileNetV3_CRNN, ResNet_Attention
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_fine_grained_accuracy(arch, chars, accs, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    colors = ['#3b82f6' if acc >= 98.5 else '#f59e0b' for acc in accs]
    bars = ax.bar(chars, accs, color=colors, width=0.6, edgecolor='black', linewidth=0.5)

    min_acc = min(accs)
    y_min = max(0, min_acc - 2.0) if min_acc < 98 else 97.0
    ax.set_ylim(y_min, 100.2)
    ax.set_ylabel('识别准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('字符类别', fontsize=12, fontweight='bold')
    arch_display = {
        "crnn": "ResNet18-CRNN 模型",
        "mobilenet": "MobileNetV3-CRNN 轻量化模型",
        "attention": "ResNet18-Attention 自回归模型"
    }
    title = f'{arch_display.get(arch, arch)}细粒度字符准确率分析'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.2f}%',
                ha='center', va='bottom', fontsize=9, rotation=45)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = f"{out_dir}/fig_eval_{arch}_fine_grained.png"
    plt.savefig(save_path)
    plt.close()
def plot_overall_comparison(metrics_dict, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.25)
    fig.suptitle('不同架构手写算式识别模型综合性能对比', fontsize=16, fontweight='bold', y=0.98)
    models_order = ["mobilenet", "crnn", "attention"]
    display_names = [
        "MobileNetV3-CRNN\n轻量化模型",
        "ResNet18-CRNN\n模型",
        "ResNet18-Attention\n自回归模型"
    ]
    accs = [metrics_dict[m]['acc'] for m in models_order if m in metrics_dict]
    cers = [metrics_dict[m]['cer'] for m in models_order if m in metrics_dict]
    fpss = [metrics_dict[m]['fps'] for m in models_order if m in metrics_dict]
    if len(accs) < 3:
        print("警告：有效评估模型不足 3 个，综合对比图可能无法完整显示。")
        return
    colors = ['#94a3b8', '#ef4444', '#94a3b8']
    edgecolor = 'black'
    width = 0.45
    axes[0].bar(display_names, accs, color=colors, edgecolor=edgecolor, width=width)
    axes[0].set_title('序列完全匹配率', fontsize=12)
    axes[0].set_ylabel('准确率 (%)', fontsize=10)
    axes[0].set_ylim(max(0, min(accs) - 10), 105)
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.6)
    for i, val in enumerate(accs):
        axes[0].text(i, val + 0.2, f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    axes[1].bar(display_names, cers, color=colors, edgecolor=edgecolor, width=width)
    axes[1].set_title('字符平均错误率', fontsize=12)
    axes[1].set_ylabel('错误率 (%)', fontsize=10)
    axes[1].set_ylim(0, max(cers) * 1.2)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.6)
    for i, val in enumerate(cers):
        axes[1].text(i, val + 0.03, f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    axes[2].bar(display_names, fpss, color=colors, edgecolor=edgecolor, width=width)
    axes[2].set_title('推理吞吐量', fontsize=12)
    axes[2].set_ylabel('Frames Per Second', fontsize=10)
    axes[2].set_ylim(0, 3000)
    axes[2].yaxis.grid(True, linestyle='--', alpha=0.6)
    for i, val in enumerate(fpss):
        axes[2].text(i, val + 30, f'{int(val)}', ha='center', va='bottom', fontsize=10)
    save_path = f"{out_dir}/fig5_2_model_comparison.png"
    plt.savefig(save_path)
    plt.close()
def main():
    print("模型评估...")
    cfg = Config("configs/base.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("logs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("正在挂载本地验证集...")
    _, val_loader = build_dataloaders(cfg)
    print(f"本次评估基于测试库: 共计 {len(val_loader.dataset)} 张独立图像")
    architectures = ["crnn", "mobilenet", "attention"]
    overall_metrics = {}
    for arch in architectures:
        print("\n" + "=" * 55)
        print(f" 正在评估架构: {arch.upper()}")
        print("=" * 55)
        if arch == "attention":
            SHARED_MAX_LEN = 15
            converter = AttentionLabelConverter(cfg.data.chars, max_seq_len=SHARED_MAX_LEN)
            model = ResNet_Attention(
                num_classes=converter.num_classes,
                hidden_size=cfg.model.hidden_size,
                max_seq_len=SHARED_MAX_LEN
            ).to(device)
        elif arch == "mobilenet":
            converter = LabelConverter(cfg.data.chars, cfg.data.blank_label)
            model = MobileNetV3_CRNN(
                img_channel=1,
                num_classes=len(cfg.data.chars) + 1,
                hidden_size=cfg.model.hidden_size
            ).to(device)
        else:  # crnn
            converter = LabelConverter(cfg.data.chars, cfg.data.blank_label)
            model = CRNN(
                img_channel=1,
                num_classes=len(cfg.data.chars) + 1,
                hidden_size=cfg.model.hidden_size,
                rnn_layers=cfg.model.rnn_layers,
                dropout=cfg.model.dropout
            ).to(device)
        ckpt_path = Path(cfg.train.output_dir) / arch / f"best_{arch}.pth"
        if not ckpt_path.exists():
            print(f"⚠跳过 {arch}: 找不到权重文件 {ckpt_path} (请先运行 train 模式训练该模型)")
            continue
        print(f"成功挂载权重: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict.get('model', state_dict))
        model.eval()
        correct, total, total_cer = 0, 0, 0.0
        char_correct = defaultdict(int)
        char_total = defaultdict(int)
        print(f"开始进行 {arch} 的并发测算...")
        start_time = time.time()
        with torch.no_grad():
            for images, labels_str in tqdm(val_loader, desc=f"Evaluating {arch}"):
                images = images.to(device)
                noise_std = 0.178
                noise = torch.randn_like(images) * noise_std
                images = images + noise
                images = torch.clamp(images, 0.0, 1.0)
                with torch.amp.autocast('cuda', enabled=cfg.mixed_precision.enabled):
                    if arch == "attention":
                        preds = model(images)
                    else:
                        preds = model(images)
                if arch == "attention":
                    pred_indices = preds.argmax(2)
                    pred_strs = converter.decode(pred_indices)
                else:
                    if cfg.inference.decode_type == 'beam_search':
                        probs = torch.nn.functional.softmax(preds, dim=2).permute(1, 0, 2)
                        pred_strs = [
                            converter.decode(probs[i], decode_type='beam_search', beam_size=cfg.inference.beam_width)
                            for i in range(len(labels_str))]
                    else:
                        pred_indices = preds.argmax(2).permute(1, 0)
                        pred_strs = [converter.decode(p, decode_type='greedy') for p in pred_indices]
                for p_str, g_str in zip(pred_strs, labels_str):
                    if p_str == g_str:
                        correct += 1
                    total_cer += calculate_cer(p_str, g_str)
                    total += 1
                    sm = difflib.SequenceMatcher(None, g_str, p_str)
                    for tag, i1, i2, j1, j2 in sm.get_opcodes():
                        if tag == 'equal':
                            for c in g_str[i1:i2]:
                                char_correct[c] += 1
                                char_total[c] += 1
                        elif tag in ('replace', 'delete'):
                            for c in g_str[i1:i2]:
                                char_total[c] += 1
        eval_time = time.time() - start_time
        fps = total / eval_time if eval_time > 0 else 0
        acc = correct / total
        avg_cer = total_cer / total
        overall_metrics[arch] = {
            'acc': acc * 100,
            'cer': avg_cer * 100,
            'fps': fps
        }
        print(f"\n 序列完全准确率 (Acc):  {acc * 100:.2f}%")
        print(f" 字符平均错误率 (CER):  {avg_cer * 100:.2f}%")
        print(f" 极限推理吞吐量 (FPS):  {fps:.1f} frames/sec")
        print(" 细粒度识别率分析:")
        plot_chars = []
        plot_accs = []
        for char in cfg.data.chars:
            if char_total[char] > 0:
                c_acc = char_correct[char] / char_total[char] * 100
                plot_chars.append(char)
                plot_accs.append(c_acc)
                warning = " 需重点优化" if c_acc < 98.5 else ""
                print(f" [{char}] 准确率: {c_acc:>6.2f}%  (有效测试样本: {char_total[char]:>5} 个) {warning}")
        if plot_chars:
            plot_fine_grained_accuracy(arch, plot_chars, plot_accs, out_dir)
            print(f"{arch} 细粒度图表已保存至: {out_dir}/fig_eval_{arch}_fine_grained.png")
    if len(overall_metrics) > 0:
        plot_overall_comparison(overall_metrics, out_dir)
        print(f"\n综合性能对比图表保存至: {out_dir}/fig5_2_model_comparison.png")
    print("\n所有模型评估与绘图均已完成，请前往 logs/figures/ 查看生成的学术图表。")
if __name__ == '__main__':
    main()
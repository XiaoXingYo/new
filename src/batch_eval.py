import sys
import argparse
import os
from pathlib import Path
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# 确保能找到 src 目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.models.factory import build_model
from src.data.generator import OCRDataGenerator
from src.data.loader import LabelConverter, AttentionLabelConverter
from src.models.attention_ocr import Seq2SeqAttention
from src.models.crnn import CRNN
from src.core.metrics import calculate_cer_wer  # 引入你的游标卡尺！


def infer_attention(model, img_tensor, converter, device, max_len=12):
    """Attention 专属自回归解码"""
    model.eval()
    with torch.no_grad():
        encoder_outputs = model.encoder(img_tensor)
        if encoder_outputs.dim() == 4:
            encoder_outputs = encoder_outputs.squeeze(2).permute(0, 2, 1)
        else:
            encoder_outputs = encoder_outputs.permute(1, 0, 2)

        hidden = torch.zeros(1, 1, model.hidden_size).to(device)
        decoder_input = torch.tensor([[converter.sos_idx]]).to(device)
        predicted_indices = []

        for _ in range(max_len):
            prediction, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
            top1 = prediction.argmax(1).item()
            if top1 == converter.eos_idx:
                break
            predicted_indices.append(top1)
            decoder_input = torch.tensor([[top1]]).to(device)

        text = "".join([converter.idx2char[idx] for idx in predicted_indices if idx < len(converter.chars)])
        return text


def main():
    parser = argparse.ArgumentParser(description="期末考试 - 海量错题挖掘机")
    parser.add_argument("-a", "--arch", choices=["ctc", "attention"], default="ctc", help="评估模型架构")
    parser.add_argument("-n", "--num", type=int, default=1000, help="要生成的测试图片数量")
    args = parser.parse_args()

    print(f"🚀 正在启动海量错题挖掘机 [目标: {args.num}张 | 选手: {args.arch.upper()}]")

    cfg = Config("configs/base.yaml")
    cfg.device = torch.device("cpu")  # 推理强制CPU即可，稳定不报错

    # 1. 建立错题本文件夹
    bad_case_dir = PROJECT_ROOT / "logs" / f"bad_cases_{args.arch}"
    bad_case_dir.mkdir(parents=True, exist_ok=True)

    # 清空上次测试的旧错题，防止混淆
    for file in bad_case_dir.glob("*.png"):
        file.unlink()

    # 2. 唤醒模型
    print("🧠 正在加载模型权重...")
    if args.arch == "ctc":
        model = build_model(cfg).to(cfg.device)
        ckpt_path = Path("../checkpoints/best_ctc.pth")
        if not ckpt_path.exists() and Path("../checkpoints/best.pth").exists():
            ckpt_path = Path("../checkpoints/best.pth")
        converter = LabelConverter(cfg.data.chars, cfg.data.blank_label)
    else:
        cnn_backbone = CRNN(backbone="cnn6").cnn
        num_classes = len(cfg.data.chars) + 3
        model = Seq2SeqAttention(cnn_backbone, num_classes).to(cfg.device)
        ckpt_path = Path("../checkpoints/best_attn.pth")
        converter = AttentionLabelConverter(cfg.data.chars)

    if not ckpt_path.exists():
        print(f"❌ 找不到权重文件 {ckpt_path}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device)['model'])
    model.eval()

    generator = OCRDataGenerator(cfg)

    # 3. 统计变量
    total_cer = 0.0
    total_wer = 0.0
    exact_matches = 0
    error_count = 0

    print("🕵️‍♂️ 开始批量测试与错题收录...")

    # 用 tqdm 加个酷炫的进度条
    with torch.no_grad():
        for i in tqdm(range(args.num), desc="Testing"):
            img_np, true_label = generator.generate_sample()
            img_tensor = torch.from_numpy(img_np).float()

            # 张量维度自适应
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            elif img_tensor.dim() == 3:
                if img_tensor.shape[2] == 1:
                    img_tensor = img_tensor.permute(2, 0, 1)
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(cfg.device)

            # 作答
            if args.arch == "ctc":
                preds = model(img_tensor)
                pred_idx = preds.argmax(2).squeeze(1)
                pred_texts = converter.decode(pred_idx, [pred_idx.size(0)])
                pred_text = pred_texts[0] if pred_texts else ""
            else:
                pred_text = infer_attention(model, img_tensor, converter, cfg.device)

            # 算分
            cer, wer = calculate_cer_wer(pred_text, true_label)
            total_cer += cer
            total_wer += wer

            if pred_text == true_label:
                exact_matches += 1
            else:
                # 💥 捕捉到错题！保存图片！
                error_count += 1
                # 命名格式: 编号_真理_错误预测.png
                filename = f"err_{error_count:04d}_T[{true_label}]_P[{pred_text}].png"
                save_path = bad_case_dir / filename

                # 去掉多余维度，保存为灰度图
                img_save = img_np.squeeze()
                plt.imsave(save_path, img_save, cmap='gray')

    # 4. 终极报告
    acc = (exact_matches / args.num) * 100
    avg_cer = (total_cer / args.num) * 100
    avg_wer = (total_wer / args.num) * 100

    print("\n" + "=" * 40)
    print(f"📊 批量测试报告 ({args.num} 张图) | 架构: {args.arch.upper()}")
    print("=" * 40)
    print(f"🎯 整串完全匹配率 (Accuracy) : {acc:.2f}%")
    print(f"📉 平均字符错误率 (CER)      : {avg_cer:.2f}%")
    print(f"📉 平均序列错误率 (WER)      : {avg_wer:.2f}%")
    print("-" * 40)
    if error_count > 0:
        print(f"📸 发现了 {error_count} 张错题，已全部保存至: ")
        print(f"📁 {bad_case_dir.resolve()}")
    else:
        print("🎉 恭喜！满分通过，没有错题！")
    print("=" * 40)


if __name__ == "__main__":
    main()
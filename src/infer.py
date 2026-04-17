import sys
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# 确保能找到 src 目录 (定位到项目根目录 D:\work\new)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.models.factory import build_model
from src.data.generator import OCRDataGenerator

# 引入 CTC 和 Attention 的专属组件
from src.data.loader import LabelConverter, AttentionLabelConverter
from src.models.attention_ocr import Seq2SeqAttention
from src.models.crnn import CRNN


def infer_attention(model, img_tensor, converter, device, max_len=12):
    """Attention 专属的 '自回归' 解码循环"""
    model.eval()
    with torch.no_grad():
        # 1. 眼睛看图提取特征
        encoder_outputs = model.encoder(img_tensor)
        if encoder_outputs.dim() == 4:
            encoder_outputs = encoder_outputs.squeeze(2).permute(0, 2, 1)
        else:
            encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # 2. 准备大脑记忆和初始触发符 <SOS>
        hidden = torch.zeros(1, 1, model.hidden_size).to(device)
        decoder_input = torch.tensor([[converter.sos_idx]]).to(device)

        predicted_indices = []

        # 3. 开始一个字一个字地往外蹦
        for _ in range(max_len):
            prediction, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
            top1 = prediction.argmax(1).item()

            if top1 == converter.eos_idx:  # 听到 <EOS> 闭嘴
                break

            predicted_indices.append(top1)
            # 把当前吐出来的字喂给下一步
            decoder_input = torch.tensor([[top1]]).to(device)

        # 4. 翻译回字符串
        text = "".join([converter.idx2char[idx] for idx in predicted_indices if idx < len(converter.chars)])
        return text


def main():
    # 增加架构选择开关
    parser = argparse.ArgumentParser(description="期末考试监考系统")
    parser.add_argument("-a", "--arch", choices=["ctc", "attention"], default="ctc",
                        help="选择上场考试的选手: ctc 或 attention")
    args = parser.parse_args()

    print(f"🔍 正在启动期末考试监考系统... [当前上场选手: {args.arch.upper()}]")

    # 1. 加载配置（老规矩，强行用CPU防报错）
    cfg = Config("configs/base.yaml")  # 注意：统一基于项目根目录读取配置
    cfg.device = torch.device("cpu")

    # 2. 把学成归来的模型请出来
    print("🧠 正在唤醒你训练好的模型...")

    if args.arch == "ctc":
        model = build_model(cfg).to(cfg.device)
        ckpt_path = Path("../checkpoints/best_ctc.pth")
        converter = LabelConverter(cfg.data.chars, cfg.data.blank_label)

        # 兼容旧的文件名
        if not ckpt_path.exists() and Path("../checkpoints/best.pth").exists():
            ckpt_path = Path("../checkpoints/best.pth")

    elif args.arch == "attention":
        cnn_backbone = CRNN(backbone="cnn6").cnn
        num_classes = len(cfg.data.chars) + 3
        model = Seq2SeqAttention(cnn_backbone, num_classes).to(cfg.device)
        ckpt_path = Path("../checkpoints/best_attn.pth")
        converter = AttentionLabelConverter(cfg.data.chars)

    if not ckpt_path.exists():
        print(f"❌ 糟糕！找不到 {ckpt_path} 文件。你是不是还没跑完训练？")
        return

    # 把记忆注入模型
    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state['model'])
    model.eval()

    # 3. 请出出卷老师
    generator = OCRDataGenerator(cfg)

    # 4. 开始随堂测验，画20道题
    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    axes = axes.flatten()

    print("📝 模型正在埋头作答...")

    with torch.no_grad():
        for i in range(20):
            img_np, label_true = generator.generate_sample()
            img_tensor = torch.from_numpy(img_np).float()

            # ⚡️ 张量维度自适应处理 (防维度爆炸报错)
            if img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif img_tensor.dim() == 3:
                if img_tensor.shape[2] == 1:
                    img_tensor = img_tensor.permute(2, 0, 1)
                img_tensor = img_tensor.unsqueeze(0)  # 加上 Batch 维度

            img_tensor = img_tensor.to(cfg.device)

            # --- 分流作答 ---
            if args.arch == "ctc":
                preds = model(img_tensor)
                pred_idx = preds.argmax(2).squeeze(1)
                # 调用 CTC 翻译官的 decode (需要传入长度)
                pred_texts = converter.decode(pred_idx, [pred_idx.size(0)])
                pred_text = pred_texts[0] if pred_texts else ""
            else:
                # 调用 Attention 的自回归翻译官
                pred_text = infer_attention(model, img_tensor, converter, cfg.device)

            # 画图批改
            img_show = img_np.squeeze()
            axes[i].imshow(img_show, cmap='gray')

            # 答对标绿，答错标红
            color = '#00CC00' if pred_text == label_true else '#FF0000'
            title = f"Pred: {pred_text}  |  True: {label_true}"
            axes[i].set_title(title, color=color, fontsize=15, fontweight='bold')
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    print(f"🎉 考试结束！快看看 {args.arch.upper()} 选手考了多少分！")


if __name__ == "__main__":
    main()
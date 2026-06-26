import argparse
import sys
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.core.config import Config
from src.core.logger import get_logger
from src.engine.core import OCREngine
from src.models.core import CRNN
from src.models.baselines import MobileNetV3_CRNN, ResNet_Attention
from src.data.loader import build_dataloaders, LabelConverter, AttentionLabelConverter

def main():
    parser = argparse.ArgumentParser(description="OCR 终极训练引擎")
    parser.add_argument("--config", "-c", default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--mode", "-m", choices=["train", "eval"], default="train", help="运行模式")
    parser.add_argument("--arch", choices=["crnn", "mobilenet", "attention"], default="crnn", help="选择要跑的模型架构")
    args = parser.parse_args()
    cfg = Config(args.config)
    logger = get_logger("Main", "logs/main.log")
    logger.info(f"🚀 项目启动: {cfg.project_name} | 模式: {args.mode} | 架构: {args.arch}")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")
    logger.info(f"⚙️ 运行硬件: {device}")
    if args.arch == "attention":
        SHARED_MAX_LEN = 15
        converter = AttentionLabelConverter(cfg.data.chars, max_seq_len=SHARED_MAX_LEN)
        model = ResNet_Attention(
            num_classes=converter.num_classes,
            hidden_size=cfg.model.hidden_size,
            max_seq_len=SHARED_MAX_LEN
        ).to(device)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=converter.pad_idx)

    elif args.arch == "mobilenet":
        converter = LabelConverter(cfg.data.chars, cfg.data.blank_label)
        model = MobileNetV3_CRNN(
            img_channel=1,
            num_classes=len(cfg.data.chars) + 1,
            hidden_size=cfg.model.hidden_size
        ).to(device)
        criterion = torch.nn.CTCLoss(blank=cfg.data.blank_label, zero_infinity=True)
    else:
        converter = LabelConverter(cfg.data.chars, cfg.data.blank_label)
        model = CRNN(
            img_channel=1,
            num_classes=len(cfg.data.chars) + 1,
            hidden_size=cfg.model.hidden_size,
            rnn_layers=cfg.model.rnn_layers,
            dropout=cfg.model.dropout
        ).to(device)
        criterion = torch.nn.CTCLoss(blank=cfg.data.blank_label, zero_infinity=True)
    engine = OCREngine(model, device, converter, cfg)
    if args.mode == "train":
        train_loader, val_loader = build_dataloaders(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        ckpt_dir = Path(cfg.train.output_dir) / args.arch
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"🔥 开始训练 ({args.arch})，共 {cfg.train.epochs} 个 Epoch...")
        best_acc = 0.0
        for epoch in range(cfg.train.epochs):
            loss = engine.train_loop(train_loader, optimizer, criterion)
            acc, cer, fps = engine.evaluate(val_loader)
            logger.info(f"[Epoch {epoch}] Loss: {loss:.4f} | Acc: {acc:.4f} | CER: {cer:.4f} | FPS: {fps:.1f}")
            if acc >= best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'epoch': epoch}, ckpt_dir / f"best_{args.arch}.pth")
                logger.info(f"发现新最佳模型 准确率: {best_acc:.4f}，权重已保存至 {ckpt_dir}。")
    elif args.mode == "eval":
        _, val_loader = build_dataloaders(cfg)
        ckpt_path = Path(cfg.train.output_dir) / args.arch / f"best_{args.arch}.pth"
        if not ckpt_path.exists():
            logger.error(f"找不到权重文件: {ckpt_path}")
            return
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
        acc, cer, fps = engine.evaluate(val_loader)
        logger.info(
            f"最终评估 ({args.arch}) -> 准确率: {acc:.4f} | 字符错误率(CER): {cer:.4f} | 推理速度: {fps:.1f} FPS")
if __name__ == "__main__":
    main()
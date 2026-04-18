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
from src.data.loader import build_dataloaders, LabelConverter


def main():
    parser = argparse.ArgumentParser(description="OCR 终极训练引擎")
    parser.add_argument("--config", "-c", default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--mode", "-m", choices=["train", "eval"], default="train", help="运行模式")
    args = parser.parse_args()

    cfg = Config(args.config)
    logger = get_logger("Main", "logs/main.log")
    logger.info(f"🚀 项目启动: {cfg.project_name} | 模式: {args.mode}")

    # 优先使用配置的设备，如果没有则自动检测
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")
    logger.info(f"⚙️ 运行硬件: {device}")

    # 1. 初始化核心组件
    converter = LabelConverter(cfg.data.chars, cfg.data.blank_label)
    model = CRNN(num_classes=len(cfg.data.chars) + 1).to(device)
    engine = OCREngine(model, device, converter, cfg)

    # 2. 分流执行
    if args.mode == "train":
        train_loader, val_loader = build_dataloaders(cfg)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        criterion = torch.nn.CTCLoss(blank=cfg.data.blank_label, zero_infinity=True)

        ckpt_dir = Path(cfg.train.output_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔥 开始训练，共 {cfg.train.epochs} 个 Epoch...")
        best_acc = 0.0

        for epoch in range(cfg.train.epochs):
            # 训练与评估
            loss = engine.train_loop(train_loader, optimizer, criterion)
            acc = engine.evaluate(val_loader)

            logger.info(f"[Epoch {epoch}] Loss: {loss:.4f} | Accuracy: {acc:.4f}")

            # 保存最佳权重
            if acc >= best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'epoch': epoch}, ckpt_dir / "best_ctc.pth")
                logger.info(f"🎉 发现新最佳模型！准确率: {best_acc:.4f}，权重已保存。")

    elif args.mode == "eval":
        _, val_loader = build_dataloaders(cfg)
        ckpt_path = PROJECT_ROOT / cfg.inference.model_path

        if not ckpt_path.exists():
            logger.error(f"❌ 找不到权重文件: {ckpt_path}")
            return

        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
        acc = engine.evaluate(val_loader)
        logger.info(f"📊 最终评估准确率: {acc:.4f}")


if __name__ == "__main__":
    main()
import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config
from src.core.logger import get_logger
from src.engine.trainer import Trainer
from src.engine.inferencer import Inferencer
from src.models.factory import build_model
from src.data.loader import build_dataloaders
from src.data.loader import AttentionLabelConverter
from src.engine.train_attn import train_attention_epoch


def main():
    parser = argparse.ArgumentParser(description="CRNN/Attention OCR")
    parser.add_argument("--config", "-c", default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--mode", "-m", choices=["train", "eval", "infer"], default="train", help="运行模式")
    parser.add_argument("--resume", "-r", help="恢复训练的checkpoint路径")
    parser.add_argument("--arch", "-a", choices=["ctc", "attention"], default="ctc", help="架构选择")

    args = parser.parse_args()

    cfg = Config(args.config)
    logger = get_logger("Main", "logs/main.log")
    logger.info(f"🚀 项目启动: {cfg.project_name} | 架构: [{args.arch.upper()}]")

    # logger.warning("⚠️ 触发强制硬件覆盖: 使用 CPU 运行！")
    # cfg.device = torch.device("cpu")

    if args.mode == "train":
        train_loader, val_loader = build_dataloaders(cfg)

        # ⚡️ 高度浓缩：直接让工厂产出正确的模型
        model = build_model(cfg, arch=args.arch).to(cfg.device)

        if args.arch == "ctc":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=cfg.train.lr, steps_per_epoch=len(train_loader),
                epochs=cfg.train.epochs, pct_start=cfg.train.scheduler.pct_start
            )
            criterion = torch.nn.CTCLoss(blank=cfg.data.blank_label, zero_infinity=True)

            trainer = Trainer(cfg, model, optimizer, scheduler, criterion, train_loader, val_loader, cfg.device)
            if args.resume or cfg.train.resume:
                ckpt_path = args.resume or cfg.train.resume
                try:
                    trainer.load_checkpoint(ckpt_path)
                except FileNotFoundError:
                    logger.warning(f"未找到: {ckpt_path}")
            trainer.run()

        elif args.arch == "attention":
            converter = AttentionLabelConverter(cfg.data.chars)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            logger.info("开始 Attention 专属训练循环...")
            save_path = Path("checkpoints")
            save_path.mkdir(exist_ok=True)

            for epoch in range(cfg.train.epochs):
                loss = train_attention_epoch(model, train_loader, optimizer, converter, cfg.device, epoch)
                logger.info(f"Epoch {epoch} 完成 | 平均 Loss: {loss:.4f}")
                torch.save({'model': model.state_dict(), 'epoch': epoch}, save_path / "best_attn.pth")

    elif args.mode == "eval":
        from src.engine.evaluator import Evaluator
        model = build_model(cfg, arch=args.arch)
        _, val_loader = build_dataloaders(cfg)

        ckpt_path = args.resume or cfg.inference.model_path
        model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device)['model'])

        evaluator = Evaluator(cfg, model, val_loader, cfg.device)
        evaluator.evaluate()

    elif args.mode == "infer":
        inferencer = Inferencer(cfg)
        inferencer.run_interactive()


if __name__ == "__main__":
    main()